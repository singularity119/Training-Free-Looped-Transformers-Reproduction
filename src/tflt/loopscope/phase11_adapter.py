"""Phase 11 generation lifecycle and native full-candidate ARC scoring."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import inspect

from .phase9_adapter import _attention_mask, _tokenizer_special_ids
from .phase10_adapter import native_predictions, request_arguments
from .phase10_adapter import token_metadata as _arc_token_metadata


def token_metadata(context, continuation, max_length, special_ids=(), pad_token_id=None):
    metadata = _arc_token_metadata(context, continuation, max_length, special_ids, pad_token_id)
    if metadata["left_truncated_tokens"]:
        raise ValueError("Phase 11 forbids truncating ARC prompts or candidates")
    return metadata


def _single_row(value, name):
    rows = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(rows, (list, tuple)) or len(rows) != 1:
        raise ValueError("Phase 11 %s requires batch one" % name)
    return tuple(rows[0])


def generation_token_metadata(token_ids, attention_mask=None, special_ids=(), pad_token_id=None):
    tokens = tuple(token_ids)
    mask = tuple(attention_mask) if attention_mask is not None else (1,) * len(tokens)
    if not tokens or len(tokens) != len(mask):
        raise ValueError("generation prompt and attention mask differ")
    excluded = set(special_ids)
    if pad_token_id is not None:
        excluded.add(pad_token_id)
    positions = tuple(i for i, token in enumerate(tokens) if mask[i] and token not in excluded)
    if not positions:
        raise ValueError("generation prompt has no valid non-special tokens")
    return {"position": positions[-1], "valid_positions": positions, "token_ids": tokens}


def _cache_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        return int(cache[0][0].shape[-2])
    raise ValueError("Phase 11 cannot establish model cache sequence length")


def _output_cache(output):
    if isinstance(output, dict):
        return output.get("past_key_values")
    return getattr(output, "past_key_values", None)


@contextmanager
def generation_forward_context(model, runtime, identity, token_ids, attention_mask=None,
                               special_ids=(), pad_token_id=None):
    """Wrap forward during one model.generate call; restore it even on failure.

    ``token_ids`` and ``attention_mask`` are flat prompt rows. The yielded
    label-free report proves one prefill followed only by growing cached,
    single-token forwards. ``runtime=None`` also monitors Native/Loop cache.
    """
    metadata = generation_token_metadata(token_ids, attention_mask, special_ids, pad_token_id)
    original = model.forward
    signature = inspect.signature(original)
    report = {"identity": identity, "prefill_count": 0, "decode_count": 0, "forwards": []}
    expected_cache_length = 0
    had_local_forward = "forward" in getattr(model, "__dict__", {})

    @wraps(original)
    def forward(*args, **kwargs):
        nonlocal expected_cache_length
        bound = signature.bind_partial(*args, **kwargs).arguments
        # Generic fake/model forward signatures may collect all kwargs here.
        values = dict(bound)
        for name, parameter in signature.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                values.update(bound.get(name, {}))
        values.update(kwargs)
        input_ids = values.get("input_ids")
        if input_ids is None:
            raise ValueError("Phase 11 generation requires explicit input_ids")
        actual_tokens = _single_row(input_ids, "generation input")
        length = len(actual_tokens)
        before = _cache_length(values.get("past_key_values"))
        if values.get("use_cache") is False:
            raise ValueError("Phase 11 generation requires use_cache=True")
        if report["prefill_count"] == 0:
            if before != 0 or actual_tokens != metadata["token_ids"]:
                raise ValueError("Phase 11 prefill must be the complete uncached prompt")
            kind = "prefill"
        else:
            if length != 1 or before != expected_cache_length or before < len(token_ids):
                raise ValueError("Phase 11 decode must consume one token with continuous incremental cache")
            kind = "decode"
        manager = runtime.forward_context(kind, length) if runtime is not None else nullcontext()
        with manager:
            output = original(*args, **kwargs)
        after = _cache_length(_output_cache(output))
        if after != before + length:
            raise ValueError("Phase 11 model cache did not advance by actual input length")
        expected_cache_length = after
        report[kind + "_count"] += 1
        report["forwards"].append({"kind": kind, "sequence_length": length,
                                    "cache_length_before": before, "cache_length_after": after})
        return output

    question = runtime.question(identity, metadata["position"], metadata["valid_positions"],
                                token_ids=metadata["token_ids"]) if runtime is not None else nullcontext()
    with question:
        model.forward = forward
        try:
            yield report
            if report["prefill_count"] != 1:
                raise RuntimeError("Phase 11 generation completed without exactly one prefill")
        finally:
            if had_local_forward:
                model.forward = original
            else:
                del model.forward


def build_hflm_class(base_class):
    class Phase11HFLM(base_class):
        def __init__(self, *args, phase11_runtime=None, **kwargs):
            self.phase11_runtime = phase11_runtime
            self.phase11_identity = None
            self.phase11_positions = []
            self._phase11_lookup = None
            super().__init__(*args, **kwargs)

        def score_choices(self, prompt, identity, choices):
            from lm_eval.api.instance import Instance
            requests = [Instance(request_type="loglikelihood", doc={}, arguments=pair, idx=i)
                        for i, pair in enumerate(request_arguments(prompt, choices))]
            old = self.phase11_identity
            self.phase11_identity = identity
            try:
                return self.loglikelihood(requests, disable_tqdm=True)
            finally:
                self.phase11_identity = old

        def _loglikelihood_tokens(self, requests, *args, **kwargs):
            if self.phase11_identity is None:
                raise ValueError("Phase11 scoring requires an explicit prompt identity")
            if self.backend != "causal" or self.batch_size != 1:
                raise ValueError("Phase11 ARC adapter requires causal HFLM batch_size=1")
            requests = list(requests)
            special, pad = _tokenizer_special_ids(self)
            metadata = [token_metadata(c, a, self.max_length, special, pad) for _, c, a in requests]
            if len({m["prefix_tokens"] for m in metadata}) > 1:
                raise ValueError("Phase11 candidates retain different common prompt contexts")
            lookup = {}
            for item in metadata:
                actual = item["actual_tokens"]
                if actual in lookup and lookup[actual] != item:
                    raise ValueError("same ARC evaluator input has ambiguous Phase11 context")
                lookup[actual] = item
            old = self._phase11_lookup
            self._phase11_lookup = lookup
            self.phase11_positions = metadata
            try:
                return super()._loglikelihood_tokens(requests, *args, **kwargs)
            finally:
                self._phase11_lookup = old

        def _model_call(self, inps, *args, **kwargs):
            rows = inps.tolist()
            if len(rows) != 1 or self._phase11_lookup is None:
                raise ValueError("Phase11 ARC model call requires a bound batch of one")
            actual = tuple(rows[0])
            if actual not in self._phase11_lookup:
                raise ValueError("actual HFLM input differs from Phase11 token metadata")
            metadata = self._phase11_lookup[actual]
            positions = metadata["valid_positions"]
            mask = _attention_mask(args, kwargs, len(actual))
            if mask is None and "attn_mask" in kwargs:
                mask = _attention_mask((), {"attention_mask": kwargs["attn_mask"]}, len(actual))
            if mask is not None:
                positions = tuple(i for i in positions if mask[i])
                if metadata["position"] not in positions:
                    raise ValueError("attention mask removes the true prompt intervention position")
            metadata["effective_valid_positions"] = positions
            manager = self.phase11_runtime.context(self.phase11_identity, metadata["position"],
                positions, token_ids=actual) if self.phase11_runtime is not None else nullcontext()
            with manager:
                return super()._model_call(inps, *args, **kwargs)
    return Phase11HFLM


def score_choices(hflm, prompt, identity, choices):
    return hflm.score_choices(prompt, identity, choices)
