"""Synthetic generation/cache and ARC request routing; no real model imports."""
import inspect
from types import SimpleNamespace
import unittest

from tflt.loopscope.phase11_adapter import build_hflm_class, generation_forward_context, generation_token_metadata, token_metadata
from test_phase11_runtime import FakeRuntime, Tensor


class Ids:
    def __init__(self, rows):
        self.rows = rows
    def tolist(self):
        return self.rows


class Cache:
    def __init__(self):
        self.length = 0
    def get_seq_length(self):
        return self.length


class Model:
    def __init__(self, runtime=None):
        self.runtime = runtime
        self.seen = []
    def forward(self, input_ids, past_key_values=None, use_cache=True, attention_mask=None):
        length = len(input_ids.tolist()[0])
        self.seen.append(length)
        if self.runtime is not None:
            for t in range(self.runtime.k):
                rows = [[float(t), 8., 9.] for i in range(length)]
                rows[self.runtime._active["position"] if length > 1 else 0] = [4., 6., 8.]
                self.runtime(Tensor([rows]), t)
        cache = past_key_values or Cache()
        cache.length += length
        return SimpleNamespace(past_key_values=cache)
    def generate(self, input_ids):
        output = self.forward(input_ids, past_key_values=Cache(), use_cache=True)
        for token in (20, 21):
            output = self.forward(Ids([[token]]), past_key_values=output.past_key_values, use_cache=True)
        return (20, 21)


class Base:
    backend, batch_size, max_length = "causal", 1, 20
    tokenizer = SimpleNamespace(all_special_ids=(99,), pad_token_id=0)
    def __init__(self):
        self.outputs = []
    def _loglikelihood_tokens(self, requests, *args, **kwargs):
        return [self._model_call(Ids([(ctx + cont)[-(self.max_length + 1):][:-1]]))
                for _, ctx, cont in requests]
    def _model_call(self, inps, **kwargs):
        length = len(inps.tolist()[0])
        runtime = self.phase11_runtime
        for t in range(runtime.k):
            rows = [[float(t), 6., 8.] for i in range(length)]
            runtime(Tensor([rows]), t)
        self.outputs.append(inps.tolist())
        return inps.tolist()


class AdapterTests(unittest.TestCase):
    def test_generation_wraps_signature_two_real_incremental_steps_and_releases(self):
        runtime = FakeRuntime("current_t", 3, .5)
        model = Model(runtime)
        signature = inspect.signature(model.forward)
        with generation_forward_context(model, runtime, "q", (0, 11, 12, 99),
                                        attention_mask=(0, 1, 1, 1), special_ids=(99,), pad_token_id=0) as report:
            self.assertEqual(inspect.signature(model.forward), signature)
            self.assertEqual(model.generate(Ids([[0, 11, 12, 99]])), (20, 21))
        self.assertEqual(model.seen, [4, 1, 1])
        self.assertEqual([c["position"] for c in runtime.calls], [2] * 3 + [0] * 6)
        self.assertEqual(len(runtime.fits), 3)
        self.assertEqual(report["prefill_count"], 1)
        self.assertEqual(report["decode_count"], 2)
        self.assertEqual([(f["cache_length_before"], f["cache_length_after"]) for f in report["forwards"]],
                         [(0, 4), (4, 5), (5, 6)])
        self.assertEqual(runtime.directions, {})
        self.assertNotIn("forward", model.__dict__)
        self.assertEqual(inspect.signature(model.forward), signature)

    def test_full_sequence_replay_and_dropped_cache_rejected_and_forward_restored(self):
        for replay, cache in (([11, 12, 20], Cache()), ([20], None)):
            model = Model()
            original = model.forward
            with self.assertRaisesRegex(ValueError, "incremental cache"):
                with generation_forward_context(model, None, "q", (11, 12)):
                    output = model.forward(Ids([[11, 12]]))
                    if len(replay) > 1:
                        cache = output.past_key_values
                    model.forward(Ids([replay]), past_key_values=cache)
            self.assertEqual(model.forward, original)
            self.assertNotIn("forward", model.__dict__)

    def test_forward_instance_override_restored_on_exception(self):
        model = Model()
        model.forward = model.forward
        original = model.forward
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with generation_forward_context(model, None, "q", (11,)):
                model.forward(Ids([[11]]))
                raise RuntimeError("synthetic failure")
        self.assertIs(model.forward, original)

    def test_generation_metadata_uses_only_valid_prompt_rows(self):
        m = generation_token_metadata((0, 99, 11, 12, 99), (0, 1, 1, 1, 1), (99,), 0)
        self.assertEqual((m["position"], m["valid_positions"]), (3, (2, 3)))

    def test_arc_candidates_are_independent_prefill_and_fit_common_prompt_only(self):
        runtime = FakeRuntime("lag1", 3, .5)
        model = build_hflm_class(Base)(phase11_runtime=runtime)
        model.phase11_identity = "arc"
        model._loglikelihood_tokens([(("p", "a"), [0, 99, 11, 12, 13], [20, 21]),
                                     (("p", "b"), [0, 99, 11, 12, 13], [30, 31, 32])])
        self.assertEqual([m["position"] for m in model.phase11_positions], [4, 4])
        self.assertEqual([m["valid_positions"] for m in model.phase11_positions], [(2, 3, 4)] * 2)
        self.assertEqual([c["kind"] for c in runtime.context_summaries], ["prefill", "prefill"])
        self.assertEqual([t for _, t, _ in runtime.fits], [0, 1, 0, 1])
        self.assertTrue(all(len(rows) == 3 for _, _, rows in runtime.fits))
        self.assertTrue(all(c["nonanswer_max_abs_change"] == 0 for c in runtime.calls))
        self.assertIsNone(model._phase11_lookup)
        self.assertEqual(runtime.directions, {})

    def test_arc_truncation_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbids truncating"):
            token_metadata(list(range(1, 12)), [20, 21], 10)


if __name__ == "__main__":
    unittest.main()
