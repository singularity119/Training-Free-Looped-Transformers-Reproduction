import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from tflt.loopscope.phase10_adapter import (
    build_hflm_class, native_predictions, request_arguments, token_metadata,
)


class Tensor:
    def __init__(self, values):
        self.values = values
    def tolist(self):
        return self.values


class Tokenizer:
    all_special_ids = (99,)
    pad_token_id = 0


class Runtime:
    def __init__(self):
        self.active = None
        self.calls = []
    @contextmanager
    def context(self, identity, position, valid_positions, token_ids=None):
        self.active = (identity, position, tuple(valid_positions), tuple(token_ids))
        self.calls.append(self.active)
        try:
            yield
        finally:
            self.active = None


class Base:
    backend = "causal"
    batch_size = 1
    max_length = 10
    tokenizer = Tokenizer()
    def __init__(self):
        self.forwarded = None
    def _loglikelihood_tokens(self, requests, *args, **kwargs):
        self.forwarded = requests
        return [self._model_call(Tensor([(ctx + cont)[-(self.max_length + 1):][:-1]]))
                for _, ctx, cont in requests]
    def _model_call(self, inps, **kwargs):
        return inps.tolist()


class AdapterTests(unittest.TestCase):
    def make(self, runtime=None):
        model = build_hflm_class(Base)(phase10_runtime=runtime)
        model.phase10_identity = "sample"
        return model

    def test_multitoken_answer_never_enters_direction_prefix(self):
        runtime = Runtime()
        model = self.make(runtime)
        model._loglikelihood_tokens([(("p", "a"), [0, 99, 11, 12, 13], [200, 201]),
                                     (("p", "b"), [0, 99, 11, 12, 13], [202, 203, 204])])
        self.assertEqual([m["position"] for m in model.phase10_positions], [4, 4])
        self.assertEqual([m["valid_positions"] for m in model.phase10_positions], [(2, 3, 4)] * 2)
        self.assertEqual([c[1:3] for c in runtime.calls], [(4, (2, 3, 4))] * 2)
        self.assertEqual(model.phase10_positions[1]["actual_tokens"][-2:], (202, 203))
        self.assertIsNone(runtime.active)

    def test_unequal_lengths_after_truncation_require_decision(self):
        model = self.make()
        requests = [(("p", "a"), list(range(1, 12)), [21]),
                    (("p", "b"), list(range(1, 12)), [22, 23])]
        with self.assertRaisesRegex(ValueError, "different pre-answer contexts"):
            model._loglikelihood_tokens(requests)
        self.assertIsNone(model.forwarded)
        first = token_metadata(requests[0][1], requests[0][2], 10)
        second = token_metadata(requests[1][1], requests[1][2], 10)
        self.assertEqual(first["left_truncated_tokens"], 1)
        self.assertEqual(second["left_truncated_tokens"], 2)
        self.assertNotEqual(first["prefix_tokens"], second["prefix_tokens"])

    def test_single_choice_complete_continuation_scoring_arguments(self):
        captured = []
        class Instance:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        class RequestBase(Base):
            def loglikelihood(self, requests, disable_tqdm=False):
                captured.extend(requests)
                return [(-1.0, False)] * len(requests)
        module = types.ModuleType("lm_eval.api.instance")
        module.Instance = Instance
        with patch.dict(sys.modules, {"lm_eval.api.instance": module}):
            model = build_hflm_class(RequestBase)()
            results = model.score_choices("Question: synthetic\nAnswer:", "id", ["water vapor", "ice", "snow"])
        self.assertEqual(len(results), 3)
        self.assertEqual([r.arguments[1] for r in captured], [" water vapor", " ice", " snow"])
        self.assertTrue(all(r.doc == {} for r in captured))
        self.assertIsNone(model.phase10_identity)

    def test_native_character_normalization_excludes_space(self):
        # ll/raw chars: -2/1 vs -3/5. Raw argmax chooses0, normalized chooses1.
        self.assertEqual(native_predictions([-2, -3], ["a", "éabcd"]), {"acc": 0, "acc_norm": 1})
        self.assertEqual(native_predictions([-1, -1], ["a", "b"]), {"acc": 0, "acc_norm": 0})
        self.assertEqual(request_arguments("Answer:", ["a", "b"])[0], ("Answer:", " a"))

    def test_attention_mask_and_failure_restore(self):
        class MaskedBase(Base):
            def _model_call(self, inps, **kwargs):
                raise RuntimeError("forward failed")
            def _loglikelihood_tokens(self, requests, **kwargs):
                ctx, cont = requests[0][1:]
                return self._model_call(Tensor([(ctx + cont)[-11:][:-1]]), attn_mask=[[1, 1, 0, 1, 1]])
        runtime = Runtime()
        model = build_hflm_class(MaskedBase)(phase10_runtime=runtime)
        model.phase10_identity = "id"
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            model._loglikelihood_tokens([(("p", "a"), [0, 99, 11, 12, 13], [200])])
        self.assertEqual(runtime.calls[0][2], (3, 4))
        self.assertIsNone(model._phase10_lookup)
        self.assertIsNone(runtime.active)


if __name__ == "__main__":
    unittest.main()
