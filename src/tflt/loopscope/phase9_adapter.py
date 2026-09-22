"""Position-only Phase 9 adapter around the unchanged lm-eval scorer.

The adapter owns only request identity and retained-prefix bookkeeping.  Score
calculation remains the lm-eval 0.4.11 implementation, as in Phase 8.  It
passes the valid prompt positions and the true answer boundary to
``Phase9Runtime``; continuation tokens, padding, and tokenizer special tokens
cannot enter the t0 SVD mask.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .phase9_spectral_core import prefix_positions


def _as_list(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _first_row(value: Any, length: int) -> Optional[Tuple[int, ...]]:
    """Read a one-row attention mask without assuming a tensor library."""

    value = _as_list(value)
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    if not isinstance(value, (tuple, list)) or len(value) != length:
        return None
    return tuple(int(item) for item in value)


def _attention_mask(args: Sequence[Any], kwargs: Dict[str, Any], length: int) -> Optional[Tuple[int, ...]]:
    candidate = kwargs.get("attention_mask")
    if candidate is None and args:
        candidate = args[0]
    return _first_row(candidate, length)


def _tokenizer_special_ids(model: Any) -> Tuple[Tuple[int, ...], Optional[int]]:
    tokenizer = getattr(model, "tokenizer", None)
    special = getattr(tokenizer, "all_special_ids", ()) if tokenizer is not None else ()
    special = tuple(int(item) for item in (special or ()))
    pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    return special, None if pad is None else int(pad)


def build_hflm_class(base_class):
    """Build a lazy Phase 9 HFLM subclass for local fake-model tests too."""

    class Phase9HFLM(base_class):
        def __init__(self, *args, phase9_runtime=None, **kwargs):
            self.phase9_runtime = phase9_runtime
            self.phase9_identity = None
            self.phase9_positions = []
            self._phase9_lookup = None
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
            previous = self.phase9_identity
            self.phase9_identity = identity
            try:
                return self.loglikelihood(requests, disable_tqdm=True)
            finally:
                self.phase9_identity = previous

        def _loglikelihood_tokens(self, requests, *args, **kwargs):
            if self.phase9_identity is None:
                raise ValueError("Phase9 scoring requires an explicit prompt identity")
            if self.backend != "causal" or self.batch_size != 1:
                raise ValueError("Phase9 adapter requires causal HFLM batch_size=1")
            requests = list(requests)
            special_ids, pad_token_id = _tokenizer_special_ids(self)
            lookup: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            contexts = set()
            positions = []
            for request, context, continuation in requests:
                if not context or not continuation or len(continuation) > self.max_length:
                    raise ValueError("invalid HFLM context/continuation lengths")
                actual = tuple((context + continuation)[-(self.max_length + 1):][:-1])
                position = len(actual) - len(continuation)
                if position < 0:
                    raise ValueError("pre-answer context is absent after truncation")
                prefix = actual[:position + 1]
                contexts.add(prefix)
                valid = prefix_positions(actual, position, special_ids, pad_token_id)
                metadata = {
                    "position": position,
                    "valid_positions": valid,
                    "prefix_tokens": prefix,
                    "input_length": len(actual),
                    "continuation_length": len(continuation),
                    "left_truncated_tokens": max(
                        0, len(context) + len(continuation) - self.max_length - 1
                    ),
                }
                if actual in lookup and lookup[actual] != metadata:
                    raise ValueError("same evaluator input has ambiguous Phase9 context metadata")
                lookup[actual] = metadata
                positions.append(metadata)
            if len(contexts) > 1:
                raise ValueError(
                    "candidate continuations retain different pre-answer contexts; "
                    "a single prompt direction requires a scientific decision"
                )
            old_lookup = self._phase9_lookup
            self._phase9_lookup = lookup
            self.phase9_positions = positions
            try:
                return super()._loglikelihood_tokens(requests, *args, **kwargs)
            finally:
                self._phase9_lookup = old_lookup

        def _model_call(self, inps, *args, **kwargs):
            rows = inps.tolist()
            if len(rows) != 1 or self._phase9_lookup is None:
                raise ValueError("Phase9 model call must be a bound batch of one")
            actual = tuple(rows[0])
            if actual not in self._phase9_lookup:
                raise ValueError("actual HFLM input does not match Phase9 token metadata")
            metadata = self._phase9_lookup[actual]
            valid_positions = metadata["valid_positions"]
            mask = _attention_mask(args, kwargs, len(actual))
            if mask is not None:
                valid_positions = tuple(
                    index for index in valid_positions if bool(mask[index])
                )
                if metadata["position"] not in valid_positions:
                    raise ValueError("attention mask removes the true answer-position context token")
            manager = (
                self.phase9_runtime.context(
                    self.phase9_identity,
                    metadata["position"],
                    valid_positions,
                    token_ids=actual,
                )
                if self.phase9_runtime is not None else nullcontext()
            )
            with manager:
                return super()._model_call(inps, *args, **kwargs)

    return Phase9HFLM


def score_choices(hflm, prompt, identity):
    """Return unchanged full-continuation A/B/C/D score pairs."""

    return hflm.score_choices(prompt, identity)
