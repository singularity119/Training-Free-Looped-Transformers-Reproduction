"""Position-only adapter around the unchanged lm-eval 0.4.11 scorer.

Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/models/huggingface.py
No tokenization, logits selection, or likelihood implementation is replaced.
"""

from contextlib import nullcontext


def build_hflm_class(base_class):
    """Build lazily, allowing position tests without importing torch/lm-eval."""

    class Phase8HFLM(base_class):
        def __init__(self, *args, phase8_runtime=None, **kwargs):
            self.phase8_runtime = phase8_runtime
            self.phase8_identity = None
            self.phase8_positions = []
            self._phase8_lookup = None
            super().__init__(*args, **kwargs)

        def score_choices(self, prompt, identity):
            from lm_eval.api.instance import Instance

            requests = [
                Instance(
                    request_type="loglikelihood", doc={},
                    arguments=(prompt, " " + letter), idx=index,
                )
                for index, letter in enumerate("ABCD")
            ]
            previous = self.phase8_identity
            self.phase8_identity = identity
            try:
                return self.loglikelihood(requests, disable_tqdm=True)
            finally:
                self.phase8_identity = previous

        def _loglikelihood_tokens(self, requests, *args, **kwargs):
            if self.phase8_identity is None:
                raise ValueError("Phase8 scoring requires an explicit prompt identity")
            if self.backend != "causal" or self.batch_size != 1:
                raise ValueError("Phase8 adapter requires causal HFLM batch_size=1")
            requests = list(requests)
            lookup = {}
            contexts = set()
            positions = []
            for request, context, continuation in requests:
                if not context or not continuation or len(continuation) > self.max_length:
                    raise ValueError("invalid HFLM context/continuation lengths")
                actual = tuple((context + continuation)[-(self.max_length + 1):][:-1])
                position = len(actual) - len(continuation)
                if position < 0:
                    raise ValueError("pre-answer context is absent after truncation")
                contexts.add(actual[:position + 1])
                if actual in lookup and lookup[actual] != position:
                    raise ValueError("same evaluator input has ambiguous context boundary")
                lookup[actual] = position
                positions.append({
                    "continuation": request[1] if request is not None else None,
                    "continuation_length": len(continuation),
                    "input_length": len(actual), "position": position,
                    "left_truncated_tokens": max(0, len(context) + len(continuation) - self.max_length - 1),
                })
            if len(contexts) > 1:
                raise ValueError(
                    "candidate continuations retain different pre-answer contexts; "
                    "a single calibration trajectory requires a scientific decision"
                )
            old_lookup = self._phase8_lookup
            self._phase8_lookup = lookup
            self.phase8_positions = positions
            try:
                return super()._loglikelihood_tokens(requests, *args, **kwargs)
            finally:
                self._phase8_lookup = old_lookup

        def _model_call(self, inps, *args, **kwargs):
            rows = inps.tolist()
            if len(rows) != 1 or self._phase8_lookup is None:
                raise ValueError("Phase8 model call must be a bound batch of one")
            actual = tuple(rows[0])
            if actual not in self._phase8_lookup:
                raise ValueError("actual HFLM input does not match evaluator token metadata")
            position = self._phase8_lookup[actual]
            manager = (
                self.phase8_runtime.context(self.phase8_identity, position)
                if self.phase8_runtime is not None else nullcontext()
            )
            with manager:
                return super()._model_call(inps, *args, **kwargs)

    return Phase8HFLM


def score_choices(hflm, prompt, identity):
    """Return original A/B/C/D full-continuation (score, greedy) pairs."""
    return hflm.score_choices(prompt, identity)
