"""CPU lifecycle checks with explicit fake directions, not a tensor/SVD claim."""
import copy
import unittest

from tflt.loopscope.phase11_runtime import Phase11Runtime, direction_schedule, make_runtime, validate_calls


class Tensor:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.ndim = 3
        self.shape = (len(data), len(data[0]), len(data[0][0]))


class FakeRuntime(Phase11Runtime):
    def __init__(self, policy, k, strength):
        super().__init__(policy, k, strength)
        self.fits = []

    def _fit_native_direction(self, delta, t):
        rows = [copy.deepcopy(delta.data[0][i]) for i in self._active["valid_positions"]]
        self.fits.append((self._active["identity"], t, rows))
        # The marker on the first retained row stands for an own-trajectory fit.
        axis = int(rows[0][0]) % 3
        self._directions[t] = tuple(1.0 if i == axis else 0.0 for i in range(3))
        self._fit_metadata[t] = {"algorithm": "fake_not_svd", "row_count": len(rows)}

    def _damp(self, delta, position, direction):
        result = Tensor(delta.data)
        raw = delta.data[0][position]
        coefficient = sum(x * y for x, y in zip(raw, direction))
        result.data[0][position] = [x - self.strength * v * coefficient for x, v in zip(raw, direction)]
        return result

    def _mutation(self, before, after, position):
        differences = [[abs(x - y) for x, y in zip(a, b)]
                       for a, b in zip(before.data[0], after.data[0])]
        return {"same_object": before is after, "answer_max_abs_change": max(differences[position]),
                "nonanswer_max_abs_change": max([max(row) for i, row in enumerate(differences) if i != position] or [0]),
                "changed_element_count": sum(x != 0 for row in differences for x in row)}


def raw_prefill(t):
    # Rows0 and2 are excluded; row4 is a candidate/special suffix.
    return Tensor([[[90., 91., 92.], [float(t), 8., 9.], [70., 71., 72.],
                    [4., 6., 8.], [100., 101., 102.]]])


class RuntimeTests(unittest.TestCase):
    def execute(self, policy, k=3, strength=.5, identity="q1", runtime=None):
        runtime = runtime or FakeRuntime(policy, k, strength)
        outputs = []
        with runtime.question(identity, 3, (1, 3), token_ids=(0, 11, 99, 12, 99)):
            with runtime.forward_context("prefill", 5):
                for t in range(k):
                    raw = raw_prefill(t)
                    result = runtime(raw, t)
                    outputs.append(result.data[0][3])
                    for i in (0, 1, 2, 4):
                        self.assertEqual(result.data[0][i], raw.data[0][i])
                    if strength == 0 or (policy != "current_t" and t == 0):
                        self.assertIs(result, raw)
            bank = runtime.directions
            for step in range(2):
                with runtime.forward_context("decode", 1):
                    for t in range(k):
                        raw = Tensor([[[4., 6., 8.]]])
                        result = runtime(raw, t)
                        outputs.append(result.data[0][0])
                        if strength == 0:
                            self.assertIs(result, raw)
                self.assertEqual(runtime.directions, bank)
        self.assertEqual(runtime.directions, {})
        self.assertEqual(runtime.fit_metadata, {})
        self.assertIsNone(runtime._active)
        return runtime, outputs

    def test_two_decode_steps_each_k_calls_and_prefill_direction_timing(self):
        expectations = {
            "fixed_t0": ([0], [None, 0, 0]),
            "lag1": ([0, 1], [None, 0, 1]),
            "current_t": ([0, 1, 2], [0, 1, 2]),
        }
        for policy, (fits, used) in expectations.items():
            with self.subTest(policy=policy):
                runtime, outputs = self.execute(policy)
                self.assertEqual([t for _, t, _ in runtime.fits], fits)
                self.assertEqual(sorted(runtime.summary()["fit_metadata_by_t"]), [str(t) for t in fits])
                self.assertEqual(validate_calls(runtime, 3), 3)
                self.assertEqual([c["timesteps"] for c in runtime.context_summaries], [[0, 1, 2]] * 3)
                for step in range(3):
                    calls = runtime.calls[step * 3:(step + 1) * 3]
                    self.assertEqual([c["direction_used_fit_t"] for c in calls], used)
                    self.assertEqual([c["position"] for c in calls], [3 if step == 0 else 0] * 3)
                    if step:
                        self.assertTrue(all(c["direction_fit_t"] is None for c in calls))
                expected = [[4., 6., 8.] if t is None else
                            [2., 6., 8.] if t == 0 else
                            [4., 3., 8.] if t == 1 else [4., 6., 4.] for t in used]
                self.assertEqual(outputs[3:6], expected)
                self.assertEqual(outputs[6:9], expected)
                self.assertTrue(all(len(rows) == 2 and rows[-1] == [4., 6., 8.] for _, _, rows in runtime.fits))

    def test_k2_alias_equivalent_with_each_strength_and_zero(self):
        self.assertEqual(direction_schedule("fixed_t0", 2), direction_schedule("lag1", 2))
        for strength in (0, .1, .5, .9):
            fixed, fixed_outputs = self.execute("fixed_t0", 2, strength)
            lag, lag_outputs = self.execute("lag1", 2, strength)
            self.assertEqual(fixed_outputs, lag_outputs)
            self.assertEqual(fixed.fits, lag.fits)
            self.assertEqual(validate_calls(lag), 3)
        for policy in ("fixed_t0", "lag1", "current_t"):
            runtime, _ = self.execute(policy, strength=0)
            self.assertTrue(all(c["same_object"] and not c["applied"] for c in runtime.calls))

    def test_same_identity_next_question_refits_and_releases_bank(self):
        runtime, _ = self.execute("current_t")
        self.execute("current_t", runtime=runtime)
        self.assertEqual(len(runtime.fits), 6)
        self.assertEqual(runtime.cache_reset_count, 2)
        self.assertEqual(validate_calls(runtime), 6)

    def test_arc_candidates_restart_even_same_identity(self):
        runtime = FakeRuntime("lag1", 3, .5)
        for candidate in (20, 21):
            with runtime.context("same", 3, (1, 3), token_ids=(0, 11, 99, 12, candidate)):
                self.assertEqual(runtime.directions, {})
                for t in range(3):
                    raw = raw_prefill(t)
                    result = runtime(raw, t)
                    self.assertEqual(result.data[0][4], raw.data[0][4])
        self.assertEqual([t for _, t, _ in runtime.fits], [0, 1, 0, 1])
        self.assertEqual(validate_calls(runtime), 2)
        self.assertTrue(all(c["kind"] == "prefill" for c in runtime.context_summaries))

    def test_callback_order_and_prefill_replay_are_rejected(self):
        runtime = FakeRuntime("lag1", 3, .5)
        with self.assertRaisesRegex(RuntimeError, "ordered"):
            with runtime.question("q", 3, (1, 3)):
                with runtime.forward_context("prefill", 5):
                    runtime(raw_prefill(0), 1)
        self.assertEqual(runtime.directions, {})
        with self.assertRaisesRegex(RuntimeError, "replay"):
            with runtime.question("q", 3, (1, 3)):
                with runtime.forward_context("prefill", 5):
                    for t in range(3):
                        runtime(raw_prefill(t), t)
                with runtime.forward_context("prefill", 5):
                    pass

    def test_cell_builder(self):
        for arm in ("Native", "Loop"):
            self.assertIsNone(make_runtime({"arm": arm}))
        runtime = make_runtime({"arm": "Online", "direction_policy": "current_t", "k": 2, "strength": .9})
        self.assertEqual(runtime.strength, .9)
        for policy, k in (("matched", 2), ("current_t", 4)):
            with self.assertRaises(ValueError):
                Phase11Runtime(policy, k, .5)


if __name__ == "__main__":
    unittest.main()
