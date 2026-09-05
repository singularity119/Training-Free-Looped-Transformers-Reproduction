import unittest
from contextlib import contextmanager

from tflt.loopscope.phase8_adapter import build_hflm_class


class Tensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Runtime:
    def __init__(self):
        self.active = None
        self.calls = []

    @contextmanager
    def context(self, identity, position):
        self.active = (identity, position)
        self.calls.append(self.active)
        try:
            yield
        finally:
            self.active = None


class Base:
    backend = "causal"
    batch_size = 1
    max_length = 5

    def __init__(self):
        self.forwarded = None

    def _loglikelihood_tokens(self, requests, *args, **kwargs):
        self.forwarded = (requests, args, kwargs)
        return [self._model_call(Tensor([(ctx + cont)[-6:][:-1]]), marker=7)
                for _, ctx, cont in requests]

    def _model_call(self, inps, marker):
        return (inps.tolist(), marker)


class AdapterTests(unittest.TestCase):
    def make(self, runtime=None):
        model = build_hflm_class(Base)(phase8_runtime=runtime)
        model.phase8_identity = "validation/subject/3"
        return model

    def test_unchanged_scoring_delegation_and_single_token_truncation(self):
        runtime = Runtime()
        model = self.make(runtime)
        requests = [(("prompt", " " + letter), list(range(9)), [token])
                    for token, letter in enumerate("ABCD", 20)]
        observed = model._loglikelihood_tokens(requests, disable_tqdm=True)
        expected = Base()._loglikelihood_tokens(requests, disable_tqdm=True)
        self.assertEqual(observed, expected)
        self.assertEqual(model.forwarded[2], {"disable_tqdm": True})
        self.assertEqual(runtime.calls, [(model.phase8_identity, 4)] * 4)
        self.assertIsNone(runtime.active)
        self.assertIsNone(model._phase8_lookup)
        self.assertEqual(model.phase8_positions[0]["left_truncated_tokens"], 4)

    def test_multitoken_continuations_use_context_boundary(self):
        runtime = Runtime()
        model = self.make(runtime)
        requests = [(("prompt", " one"), [1, 2], [3]),
                    (("prompt", " two tokens"), [1, 2], [4, 5])]
        model._loglikelihood_tokens(requests)
        self.assertEqual([pos for _, pos in runtime.calls], [1, 1])
        self.assertEqual([row["input_length"] for row in model.phase8_positions], [2, 3])

    def test_truncated_different_contexts_are_not_silently_deduplicated(self):
        model = self.make(Runtime())
        requests = [(("p", "a"), list(range(8)), [10]),
                    (("p", "bc"), list(range(8)), [11, 12])]
        with self.assertRaisesRegex(ValueError, "different pre-answer contexts"):
            model._loglikelihood_tokens(requests)
        self.assertIsNone(model.forwarded)

    def test_none_runtime_keeps_base_result(self):
        requests = [(("p", "a"), [1, 2], [3])]
        self.assertEqual(self.make()._loglikelihood_tokens(requests),
                         Base()._loglikelihood_tokens(requests))

    def test_context_restored_on_model_failure(self):
        class FailingBase(Base):
            def _model_call(self, *args, **kwargs):
                raise RuntimeError("forward failed")

        runtime = Runtime()
        model = build_hflm_class(FailingBase)(phase8_runtime=runtime)
        model.phase8_identity = "sample"
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            model._loglikelihood_tokens([(("p", "a"), [1], [2])])
        self.assertIsNone(runtime.active)
        self.assertIsNone(model._phase8_lookup)


if __name__ == "__main__":
    unittest.main()
