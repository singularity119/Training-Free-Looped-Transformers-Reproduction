import unittest
from contextlib import contextmanager

from tflt.loopscope.phase9_adapter import build_hflm_class


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
        self.forwarded = (requests, args, kwargs)
        return [self._model_call(Tensor([(ctx + cont)[-11:][:-1]]), marker=7)
                for _, ctx, cont in requests]

    def _model_call(self, inps, marker):
        return (inps.tolist(), marker)


class Phase9AdapterTests(unittest.TestCase):
    def make(self, runtime=None):
        model = build_hflm_class(Base)(phase9_runtime=runtime)
        model.phase9_identity = {"subject": "math", "doc_index": 3}
        return model

    def test_mask_excludes_special_padding_and_multitoken_continuation(self):
        runtime = Runtime()
        model = self.make(runtime)
        requests = [
            (("prompt", " candidate"), [0, 99, 11, 12, 13], [200, 201]),
            (("prompt", " candidate"), [0, 99, 11, 12, 13], [202, 203]),
        ]
        observed = model._loglikelihood_tokens(requests)
        self.assertEqual(len(observed), 2)
        self.assertEqual([row["position"] for row in model.phase9_positions], [4, 4])
        self.assertEqual([row["valid_positions"] for row in model.phase9_positions], [(2, 3, 4)] * 2)
        self.assertEqual([call[1] for call in runtime.calls], [4, 4])
        self.assertEqual([call[2] for call in runtime.calls], [(2, 3, 4)] * 2)
        self.assertEqual([call[3] for call in runtime.calls],
                         [(0, 99, 11, 12, 13, 200), (0, 99, 11, 12, 13, 202)])
        self.assertEqual(runtime.active, None)

    def test_different_retained_prefixes_fail_closed(self):
        model = self.make(Runtime())
        requests = [
            (("prompt", " a"), [1, 2, 3], [4]),
            (("prompt", " b"), [1, 2, 8], [9]),
        ]
        with self.assertRaisesRegex(ValueError, "different pre-answer contexts"):
            model._loglikelihood_tokens(requests)
        self.assertIsNone(model.forwarded)

    def test_runtime_context_is_restored_on_model_failure(self):
        class FailingBase(Base):
            def _model_call(self, *args, **kwargs):
                raise RuntimeError("forward failed")

        runtime = Runtime()
        model = build_hflm_class(FailingBase)(phase9_runtime=runtime)
        model.phase9_identity = "sample"
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            model._loglikelihood_tokens([(("p", "a"), [1, 2], [3])])
        self.assertIsNone(runtime.active)
        self.assertIsNone(model._phase9_lookup)

    def test_effective_attention_mask_is_forwarded_to_runtime(self):
        class MaskedBase(Base):
            def _model_call(self, inps, attention_mask=None):
                return super()._model_call(inps, marker=7)

            def _loglikelihood_tokens(self, requests, *args, **kwargs):
                self.forwarded = (requests, args, kwargs)
                return [
                    self._model_call(
                        Tensor([(ctx + cont)[-11:][:-1]]),
                        attention_mask=[[1, 1, 0, 1, 1]],
                    )
                    for _, ctx, cont in requests
                ]

        runtime = Runtime()
        model = build_hflm_class(MaskedBase)(phase9_runtime=runtime)
        model.phase9_identity = "sample"
        requests = [(('prompt', ' candidate'), [0, 99, 11, 12, 13], [200])]
        model._loglikelihood_tokens(requests)
        self.assertEqual(model.phase9_positions[0]["effective_valid_positions"], (3, 4))
        self.assertEqual(runtime.calls[0][2], (3, 4))


if __name__ == "__main__":
    unittest.main()
