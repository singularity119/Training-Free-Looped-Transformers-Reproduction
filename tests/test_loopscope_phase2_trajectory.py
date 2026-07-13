import math
import unittest

from tflt.loopscope.phase2_analysis import choice_output
from tflt.loopscope.phase2_schema import DIRECT_PROBE_SCORE_SOURCE
from tflt.loopscope.phase2_trajectory import (
    InMemoryProbeSession,
    Phase2TrajectoryCollector,
    TrajectoryError,
    VectorMetadata,
    baseline_native_alignment,
    cosine_measurement,
    k1_choice_equivalent,
    prefix_consistent,
)


def tensor(vector):
    return [[list(vector)]]


def cells():
    result = []
    settings = [
        ("shared_k2_anchor", 2, 1.0),
        ("fixed_step", 3, 1.5),
        ("fixed_step", 4, 2.0),
        ("fixed_horizon", 3, 1.0),
        ("fixed_horizon", 4, 1.0),
    ]
    for window in ("11:14", "12:15", "13:16"):
        for protocol, k, alpha in settings:
            result.append({"protocol": protocol, "window": window, "k": k, "alpha": alpha})
    return result


class Phase2TrajectoryTests(unittest.TestCase):
    def _run_collector(self, k):
        collector = Phase2TrajectoryCollector(window_width=4, expected_k=k)
        collector.begin_forward(
            task="mmlu_x",
            doc_id=7,
            doc_hash="a" * 64,
            answer_position=0,
            native_continuation=[1.0, 0.0],
            protocol="fixed_step",
            window="12:15",
            alpha=k * 0.5,
        )
        collector.record("wrapper_forward", {"bypass": False})
        before = [1.0, 1.0]
        for t in range(k):
            collector.record("body_call", {})
            delta = [1.0, 0.25 * t]
            after = [before[index] + delta[index] for index in range(2)]
            collector.record_tensor_diff("g_minus_x", tensor(before), tensor(after))
            before = after
        collector.record_tensor_diff("looped_hidden_vs_input", tensor([0.0, 0.0]), tensor(before))
        collector.record("stash_pass", {})
        for _ in range(3):
            collector.record("identity_forward", {})
        identity = {"task": "mmlu_x", "doc_id": "7", "doc_hash": "a" * 64}
        result = collector.end_forward(
            final_output=choice_output(
                [4.0, 3.0, 2.0, 1.0],
                identity=identity,
                score_source=DIRECT_PROBE_SCORE_SOURCE,
            )
        )
        self.assertEqual(collector.pop_admission_vectors()["sample_identity"], identity)
        return result

    def test_zero_based_body_calls_and_repeated_nca(self):
        result = self._run_collector(4)
        self.assertTrue(result["valid"])
        self.assertEqual([step["body_call_t"] for step in result["steps"]], [0, 1, 2, 3])
        self.assertIsNone(result["steps"][0]["nca"])
        self.assertTrue(all(result["steps"][i]["nca"]["valid"] for i in (1, 2, 3)))
        self.assertEqual(result["steps"][1]["nca"]["provenance"], "actual_loop_repeated_step")

    def test_k1_has_no_repeated_nca_but_is_valid(self):
        result = self._run_collector(1)
        self.assertTrue(result["valid"])
        self.assertTrue(result["repeated_step_nca_all_valid"])
        self.assertIsNone(result["steps"][0]["nca"])

    def test_cosine_validity_and_metadata_fail_fast(self):
        self.assertAlmostEqual(cosine_measurement([1, 0], [1, 0])["value"], 1.0)
        self.assertFalse(cosine_measurement([0, 0], [1, 0])["valid"])
        with self.assertRaises(TrajectoryError):
            cosine_measurement([float("nan")], [1.0])
        with self.assertRaises(TrajectoryError):
            cosine_measurement(
                [1.0],
                [1.0],
                left_meta=VectorMetadata((1,), "float16", "cuda:0"),
                right_meta=VectorMetadata((1,), "float32", "cuda:0"),
            )
        with self.assertRaises(TrajectoryError):
            cosine_measurement(
                [1.0],
                [1.0],
                left_meta=VectorMetadata((1,), "float16", "cuda:0"),
                right_meta=VectorMetadata((1,), "float16", "cuda:1"),
            )

    def test_zero_current_residual_is_saturation_but_next_zero_denominator_fails(self):
        collector = Phase2TrajectoryCollector(window_width=4, expected_k=2)
        collector.begin_forward(
            task="mmlu_x", doc_id=1, doc_hash="b" * 64, answer_position=0,
            native_continuation=[1.0, 0.0], protocol="fixed_step", window="12:15", alpha=1.0,
        )
        collector.record("wrapper_forward", {"bypass": False})
        collector.record("body_call", {})
        collector.record_tensor_diff("g_minus_x", tensor([1.0, 1.0]), tensor([2.0, 1.0]))
        collector.record("body_call", {})
        collector.record_tensor_diff("g_minus_x", tensor([2.0, 1.0]), tensor([2.0, 1.0]))
        self.assertEqual(collector._active["steps"][1]["residual_ratio_to_previous"], 0.0)
        self.assertFalse(collector._active["steps"][1]["nca"]["valid"])

        collector2 = Phase2TrajectoryCollector(window_width=4, expected_k=2)
        collector2.begin_forward(
            task="mmlu_x", doc_id=2, doc_hash="c" * 64, answer_position=0,
            native_continuation=[1.0, 0.0], protocol="fixed_step", window="12:15", alpha=1.0,
        )
        collector2.record("wrapper_forward", {"bypass": False})
        collector2.record("body_call", {})
        collector2.record_tensor_diff("g_minus_x", tensor([1.0, 1.0]), tensor([1.0, 1.0]))
        collector2.record("body_call", {})
        with self.assertRaises(TrajectoryError):
            collector2.record_tensor_diff("g_minus_x", tensor([1.0, 1.0]), tensor([2.0, 1.0]))

    def test_baseline_and_loop_provenance_are_distinct(self):
        result = baseline_native_alignment([0, 0], [1, 0], [1, 0])
        self.assertEqual(result["provenance"], "baseline_single_forward_B_(b+1)-B_a")
        self.assertIsNone(result["body_call_t"])

    def test_tolerances(self):
        self.assertTrue(k1_choice_equivalent([1, 2, 3, 4], [1, 2, 3, 4.00005]))
        self.assertFalse(k1_choice_equivalent([1, 2, 3, 4], [1, 2, 3, 4.01]))
        self.assertTrue(prefix_consistent([[1, 2]], [[1.000001, 2]])["match"])
        self.assertFalse(prefix_consistent([[1, 2]], [[1.1, 2]])["match"])

    def test_same_session_one_baseline_and_fifteen_cells(self):
        session = InMemoryProbeSession(cells())
        calls = {"baseline": 0, "loop": 0}

        def baseline(store):
            calls["baseline"] += 1
            store[("sample", "12:15")] = [1.0, 0.0]

        def loop(cell, store):
            calls["loop"] += 1
            self.assertIn(("sample", "12:15"), store)
            return cell["window"]

        self.assertEqual(len(session.run(baseline, loop)), 15)
        self.assertEqual(calls, {"baseline": 1, "loop": 15})
        with self.assertRaises(TrajectoryError):
            session.run(baseline, loop)


if __name__ == "__main__":
    unittest.main()
