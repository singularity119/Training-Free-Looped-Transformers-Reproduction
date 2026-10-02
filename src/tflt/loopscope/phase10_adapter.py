"""ARC full-text request adapter; native lm-eval 0.4.11 computes scores.

MMLU continues to use the unchanged phase9_adapter. ARC has variable option
counts and multi-token continuations, but the same causal pre-answer contract.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

from .phase9_adapter import _attention_mask, _tokenizer_special_ids
from .phase9_spectral_core import prefix_positions


def token_metadata(context, continuation, max_length, special_ids=(), pad_token_id=None):
    """Describe precisely the input made by HFLM._loglikelihood_tokens."""
    context, continuation = list(context), list(continuation)
    if not context or not continuation or len(continuation) > max_length:
        raise ValueError("invalid HFLM context/continuation lengths")
    actual = tuple((context + continuation)[-(max_length + 1):][:-1])
    position = len(actual) - len(continuation)
    if position < 0:
        raise ValueError("pre-answer context is absent after truncation")
    valid = prefix_positions(actual, position, special_ids, pad_token_id)
    if position not in valid:
        raise ValueError("true pre-answer token must be a valid prompt position")
    return {
        "position": position, "valid_positions": valid,
        "effective_valid_positions": valid, "prefix_tokens": actual[:position + 1],
        "actual_tokens": actual, "input_length": len(actual),
        "context_length": len(context), "continuation_length": len(continuation),
        "left_truncated_tokens": max(0, len(context) + len(continuation) - max_length - 1),
    }


def request_arguments(prompt: str, choices: Sequence[str]):
    """Native ARC target delimiter is one space; choices are full text."""
    if not prompt or len(choices) < 2 or any(not isinstance(c, str) or not c for c in choices):
        raise ValueError("ARC needs a prompt and nonempty full-text choices")
    return [(prompt, " " + choice) for choice in choices]


def native_predictions(scores, choices):
    """Gold-free predictions using native character length normalization."""
    import math
    if len(scores) != len(choices) or not scores or any(not c for c in choices):
        raise ValueError("scores and nonempty choices must have identical lengths")
    values = [float(s[0]) if isinstance(s, (tuple, list)) else float(s) for s in scores]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("choice loglikelihoods must be finite")
    # max returns the first index on ties, matching numpy.argmax.
    return {"acc": max(range(len(values)), key=values.__getitem__),
            "acc_norm": max(range(len(values)), key=lambda i: values[i] / len(choices[i]))}


def build_hflm_class(base_class):
    class Phase10HFLM(base_class):
        def __init__(self, *args, phase10_runtime=None, **kwargs):
            self.phase10_runtime = phase10_runtime
            self.phase10_identity = None
            self.phase10_positions = []
            self._phase10_lookup = None
            super().__init__(*args, **kwargs)

        def score_choices(self, prompt, identity, choices):
            from lm_eval.api.instance import Instance
            requests = [Instance(request_type="loglikelihood", doc={}, arguments=pair, idx=i)
                        for i, pair in enumerate(request_arguments(prompt, choices))]
            old = self.phase10_identity
            self.phase10_identity = identity
            try:
                return self.loglikelihood(requests, disable_tqdm=True)
            finally:
                self.phase10_identity = old

        def _loglikelihood_tokens(self, requests, *args, **kwargs):
            if self.phase10_identity is None:
                raise ValueError("Phase10 scoring requires an explicit prompt identity")
            if self.backend != "causal" or self.batch_size != 1:
                raise ValueError("Phase10 adapter requires causal HFLM batch_size=1")
            requests = list(requests)
            special, pad = _tokenizer_special_ids(self)
            metadata = [token_metadata(c, a, self.max_length, special, pad) for _, c, a in requests]
            if len({m["prefix_tokens"] for m in metadata}) > 1:
                raise ValueError("candidate continuations retain different pre-answer contexts; "
                                 "a single prompt direction requires a scientific decision")
            lookup = {}
            for item in metadata:
                actual = item["actual_tokens"]
                if actual in lookup and lookup[actual] != item:
                    raise ValueError("same evaluator input has ambiguous Phase10 context metadata")
                lookup[actual] = item
            old = self._phase10_lookup
            self._phase10_lookup = lookup
            self.phase10_positions = metadata
            try:
                return super()._loglikelihood_tokens(requests, *args, **kwargs)
            finally:
                self._phase10_lookup = old

        def _model_call(self, inps, *args, **kwargs):
            rows = inps.tolist()
            if len(rows) != 1 or self._phase10_lookup is None:
                raise ValueError("Phase10 model call must be a bound batch of one")
            actual = tuple(rows[0])
            if actual not in self._phase10_lookup:
                raise ValueError("actual HFLM input does not match Phase10 token metadata")
            metadata = self._phase10_lookup[actual]
            positions = metadata["valid_positions"]
            mask = _attention_mask(args, kwargs, len(actual))
            if mask is None and "attn_mask" in kwargs:
                mask = _attention_mask((), {"attention_mask": kwargs["attn_mask"]}, len(actual))
            if mask is not None:
                positions = tuple(i for i in positions if mask[i])
                if metadata["position"] not in positions:
                    raise ValueError("attention mask removes the true pre-answer token")
            metadata["effective_valid_positions"] = positions
            manager = self.phase10_runtime.context(self.phase10_identity, metadata["position"],
                positions, token_ids=actual) if self.phase10_runtime is not None else nullcontext()
            with manager:
                return super()._model_call(inps, *args, **kwargs)
    return Phase10HFLM


def score_choices(hflm, prompt, identity, choices):
    return hflm.score_choices(prompt, identity, choices)
