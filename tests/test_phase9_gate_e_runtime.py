import unittest

from tflt.loopscope.phase9_gate_e_spectral_core import (
    direction_schedule,
    lag1_schedule,
)
from tflt.loopscope.phase9_gate_e_runtime import Phase9GateERuntime


class Phase9GateEDirectionPolicyTests(unittest.TestCase):
    def test_fixed_and_lag1_schedules_are_explicit_and_distinct(self):
        self.assertEqual(direction_schedule("fixed_t0", 2), [(0, None, 0), (1, 0, None)])
        self.assertEqual(direction_schedule("fixed_t0", 3), [
            (0, None, 0), (1, 0, None), (2, 0, None),
        ])
        self.assertEqual(direction_schedule("fixed_t0", 4), [
            (0, None, 0), (1, 0, None), (2, 0, None), (3, 0, None),
        ])
        self.assertEqual(lag1_schedule(3), [(0, None, 0), (1, 0, 1), (2, 1, None)])
        self.assertEqual(lag1_schedule(4), [
            (0, None, 0), (1, 0, 1), (2, 1, 2), (3, 2, None),
        ])
        with self.assertRaises(ValueError):
            direction_schedule("implicit", 3)

    def test_policy_and_intervention_mode_are_required_and_validated(self):
        with self.assertRaises(TypeError):
            Phase9GateERuntime(intervention_mode="spectral", k=3)
        with self.assertRaises(ValueError):
            Phase9GateERuntime("unknown", "spectral", k=3)
        with self.assertRaises(ValueError):
            Phase9GateERuntime("lag1", "unknown", k=3)

    def test_fixed_t0_and_lag1_match_through_t1_then_use_distinct_sources(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is needed for tensor-level Gate E checks")

        d0 = torch.tensor([[[4.0, 0.0], [3.0, 0.0]]])
        d1 = torch.tensor([[[0.0, 5.0], [3.0, 4.0]]])
        d2 = torch.tensor([[[0.0, 4.0], [0.0, 3.0]]])
        outputs = {}
        for policy in ("fixed_t0", "lag1"):
            runtime = Phase9GateERuntime(policy, "spectral", k=3)
            with runtime.context("sample", 1, (0, 1), token_ids=(10, 11)):
                self.assertIs(runtime(d0, 0), d0)
                at_t1 = runtime(d1, 1)
                at_t2 = runtime(d2, 2)
            outputs[policy] = (at_t1, at_t2, runtime)

        self.assertTrue(torch.equal(outputs["fixed_t0"][0][0, 1], outputs["lag1"][0][0, 1]))
        self.assertTrue(torch.allclose(outputs["fixed_t0"][0][0, 1], torch.tensor([1.5, 4.0])))
        self.assertFalse(torch.equal(outputs["fixed_t0"][1][0, 1], outputs["lag1"][1][0, 1]))
        fixed_runtime = outputs["fixed_t0"][2]
        lag1_runtime = outputs["lag1"][2]
        self.assertEqual(set(fixed_runtime.fit_metadata), {0})
        self.assertEqual(set(lag1_runtime.fit_metadata), {0, 1})
        self.assertEqual(
            [(call["applied_t"], call["direction_used_fit_t"], call["direction_fit_t"])
             for call in lag1_runtime.calls],
            [(None, None, 0), (1, 0, 1), (2, 1, None)],
        )
        self.assertEqual(
            [(call["direction_used_fit_t"], call["direction_fit_t"])
             for call in fixed_runtime.calls],
            [(None, 0), (0, None), (0, None)],
        )
        self.assertTrue(all(call["nonanswer_max_abs_change"] == 0.0 for call in lag1_runtime.calls))

    def test_matched_norm_uses_the_selected_direction_policy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is needed for tensor-level Gate E checks")

        runtime = Phase9GateERuntime("lag1", "matched_norm", k=3)
        d0 = torch.tensor([[[4.0, 0.0], [3.0, 0.0]]])
        d1 = torch.tensor([[[0.0, 5.0], [3.0, 4.0]]])
        with runtime.context("sample", 1, (0, 1), token_ids=(10, 11)):
            runtime(d0, 0)
            result = runtime(d1, 1)
        observed_norm = torch.linalg.vector_norm(result[0, 1].float())
        directional = torch.tensor([1.5, 4.0])
        self.assertTrue(torch.allclose(observed_norm, torch.linalg.vector_norm(directional)))
        self.assertTrue(torch.allclose(result[0, 1], torch.tensor([3.0, 4.0]) * (observed_norm / 5.0)))

    def test_prompt_change_releases_cached_direction(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is needed for tensor-level Gate E checks")

        runtime = Phase9GateERuntime("fixed_t0", "spectral", k=2)
        first = torch.tensor([[[4.0, 0.0], [3.0, 0.0]]])
        second = torch.tensor([[[0.0, 5.0], [0.0, 3.0]]])
        with runtime.context("first", 1, (0, 1), token_ids=(10, 11)):
            runtime(first, 0)
        with runtime.context("second", 1, (0, 1), token_ids=(20, 21)):
            runtime(second, 0)
        self.assertEqual(runtime.cache_reset_count, 1)
        self.assertEqual(set(runtime.fit_metadata), {0})
        self.assertTrue(torch.allclose(runtime.direction, torch.tensor([0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
