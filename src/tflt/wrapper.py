"""Reversible Hugging Face decoder-layer loop wrappers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from tflt.cache import cache_arg, crop_cache, snapshot_cache
from tflt.config import LoopConfig
from tflt.strategies import run_loop


try:
    from torch import nn

    _ModuleBase = nn.Module
except Exception:

    class _ModuleBase(object):  # type: ignore[no-redef]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)


LayerPath = Tuple[Any, str]


@dataclass
class LoopHandle:
    """Handle returned by `apply_loop_wrapper`."""

    model: Any
    layer_owner: Any
    layer_attr: str
    originals: Dict[int, Any]
    original_forward: Any
    active: bool = True

    def restore(self) -> None:
        if not self.active:
            return
        layers = getattr(self.layer_owner, self.layer_attr)
        for idx, layer in self.originals.items():
            layers[idx] = layer
        if self.original_forward is not None:
            self.model.forward = self.original_forward
        self.active = False

    def __enter__(self) -> Any:
        return self.model

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.restore()


def apply_loop_wrapper(model: Any, config: LoopConfig) -> LoopHandle:
    """Patch a model's decoder layers in place and return a reversible handle.

    The implementation targets the common Hugging Face causal-LM shape where the
    decoder owns a list or ModuleList named `layers`, `h`, or `blocks`.
    """

    owner, attr = _find_layer_owner(model)
    layers = getattr(owner, attr)
    if config.end >= len(layers):
        raise ValueError("window %s exceeds layer count %d" % (config.window, len(layers)))

    originals: Dict[int, Any] = {}
    for idx in range(config.start, config.end + 1):
        originals[idx] = layers[idx]

    if config.iteration_mode == "layer":
        for idx in range(config.start, config.end + 1):
            layers[idx] = LoopLayerWrapper(originals[idx], config)
    elif config.iteration_mode == "block":
        block_layers = [originals[idx] for idx in range(config.start, config.end + 1)]
        layers[config.start] = LoopBlockEntryWrapper(block_layers, config)
        for idx in range(config.start + 1, config.end + 1):
            layers[idx] = LoopIdentityLayer(config, idx)
    else:
        raise ValueError("unsupported iteration_mode: %s" % config.iteration_mode)

    return LoopHandle(
        model=model,
        layer_owner=owner,
        layer_attr=attr,
        originals=originals,
        original_forward=getattr(model, "forward", None),
    )


@contextmanager
def looped_model(model: Any, config: LoopConfig) -> Iterator[Any]:
    handle = apply_loop_wrapper(model, config)
    try:
        yield model
    finally:
        handle.restore()


class LoopLayerWrapper(_ModuleBase):
    def __init__(self, layer: Any, config: LoopConfig) -> None:
        super().__init__()
        self.layer = layer
        self.config = config

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        bypass = _should_bypass_decode(hidden_states, kwargs, self.config)
        _audit(
            self.config,
            "wrapper_forward",
            wrapper_type="layer",
            bypass=bypass,
            **_forward_audit_metadata(kwargs),
        )
        if bypass:
            _audit(self.config, "bypass", wrapper_type="layer")
            return self.layer(hidden_states, *args, **kwargs)

        body_kwargs = _body_kwargs(kwargs, allow_cache_reads=_has_decode_cache(kwargs))

        def operator(x: Any) -> Any:
            snap = snapshot_cache(body_kwargs)
            try:
                out = _hidden(self.layer(x, *args, **body_kwargs))
                _audit(self.config, "body_call", wrapper_type="layer")
                _audit_tensor_diff(self.config, "g_minus_x", x, out)
                return out
            finally:
                crop_cache(snap)

        looped = run_loop(operator, hidden_states, self.config)
        _audit_tensor_diff(self.config, "looped_hidden_vs_input", hidden_states, looped)
        if self.config.cache_strategy != "none":
            _audit(self.config, "stash_pass", wrapper_type="layer", cache_strategy=self.config.cache_strategy)
        return _stash_or_return(self.layer, hidden_states, looped, args, kwargs, self.config)


class LoopBlockEntryWrapper(_ModuleBase):
    def __init__(self, layers: Sequence[Any], config: LoopConfig) -> None:
        super().__init__()
        self.layers = list(layers)
        self.config = config

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        bypass = _should_bypass_decode(hidden_states, kwargs, self.config)
        _audit(
            self.config,
            "wrapper_forward",
            wrapper_type="block",
            bypass=bypass,
            **_forward_audit_metadata(kwargs),
        )
        if bypass:
            _audit(self.config, "bypass", wrapper_type="block")
            return _run_layers(self.layers, hidden_states, args, kwargs)

        body_kwargs = _body_kwargs(kwargs, allow_cache_reads=_has_decode_cache(kwargs))

        def operator(x: Any) -> Any:
            snap = snapshot_cache(body_kwargs)
            try:
                out = _run_layers_hidden(self.layers, x, args, body_kwargs)
                _audit(self.config, "body_call", wrapper_type="block")
                _audit_tensor_diff(self.config, "g_minus_x", x, out)
                return out
            finally:
                crop_cache(snap)

        looped = run_loop(operator, hidden_states, self.config)
        _audit_tensor_diff(self.config, "looped_hidden_vs_input", hidden_states, looped)
        if self.config.cache_strategy == "none":
            return _as_layer_output(looped, kwargs)

        stash_input = hidden_states if self.config.cache_strategy == "first" else looped
        _audit(self.config, "stash_pass", wrapper_type="block", cache_strategy=self.config.cache_strategy)
        stash_result = _run_layers(self.layers, stash_input, args, kwargs)
        return _replace_hidden(stash_result, looped)


class LoopIdentityLayer(_ModuleBase):
    def __init__(self, config: Optional[LoopConfig] = None, index: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.index = index

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        if self.config is not None:
            _audit(self.config, "identity_forward", index=self.index, **_forward_audit_metadata(kwargs))
        return _as_layer_output(hidden_states, kwargs)


def _find_layer_owner(model: Any) -> LayerPath:
    candidates = [
        (model, "layers"),
        (getattr(model, "model", None), "layers"),
        (getattr(model, "transformer", None), "h"),
        (getattr(model, "gpt_neox", None), "layers"),
        (getattr(model, "decoder", None), "layers"),
        (getattr(getattr(model, "model", None), "decoder", None), "layers"),
        (getattr(getattr(model, "base_model", None), "model", None), "layers"),
    ]
    for owner, attr in candidates:
        if owner is not None and hasattr(owner, attr):
            layers = getattr(owner, attr)
            if hasattr(layers, "__len__") and hasattr(layers, "__getitem__"):
                return owner, attr
    raise TypeError("could not locate decoder layers on model")


def _run_layers(layers: Iterable[Any], hidden_states: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    result: Any = hidden_states
    for layer in layers:
        result = layer(_hidden(result), *args, **kwargs)
    return result


def _run_layers_hidden(
    layers: Iterable[Any], hidden_states: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Any:
    return _hidden(_run_layers(layers, hidden_states, args, kwargs))


def _hidden(result: Any) -> Any:
    if isinstance(result, tuple):
        return result[0]
    return result


def _replace_hidden(result: Any, hidden_states: Any) -> Any:
    if isinstance(result, tuple):
        return (hidden_states,) + result[1:]
    return hidden_states


def _as_layer_output(hidden_states: Any, kwargs: Dict[str, Any]) -> Any:
    output_attentions = bool(kwargs.get("output_attentions", False))
    use_cache = bool(kwargs.get("use_cache", False))
    if not output_attentions and not use_cache:
        return (hidden_states,)
    out: Tuple[Any, ...] = (hidden_states,)
    if output_attentions:
        out = out + (None,)
    if use_cache:
        out = out + (_cache_arg(kwargs),)
    return out


def _stash_or_return(
    layer: Any,
    original_hidden: Any,
    looped_hidden: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    config: LoopConfig,
) -> Any:
    if config.cache_strategy == "none":
        return _as_layer_output(looped_hidden, kwargs)
    stash_input = original_hidden if config.cache_strategy == "first" else looped_hidden
    stash_result = layer(stash_input, *args, **kwargs)
    return _replace_hidden(stash_result, looped_hidden)


def _body_kwargs(kwargs: Dict[str, Any], allow_cache_reads: bool) -> Dict[str, Any]:
    body = dict(kwargs)
    if allow_cache_reads:
        if "use_cache" in body:
            body["use_cache"] = True
        return body
    if "use_cache" in body:
        body["use_cache"] = False
    for key in ("past_key_value", "past_key_values"):
        if key in body:
            body[key] = None
    return body


def _cache_arg(kwargs: Dict[str, Any]) -> Any:
    return cache_arg(kwargs)


def _should_bypass_decode(hidden_states: Any, kwargs: Dict[str, Any], config: LoopConfig) -> bool:
    if config.decode_mode == "full":
        return False
    if config.decode_mode == "bypass":
        return _has_decode_cache(kwargs)
    if config.decode_mode == "first_n":
        pos = _cache_position(kwargs)
        return pos is not None and config.first_n is not None and pos >= config.first_n
    return False


def _has_decode_cache(kwargs: Dict[str, Any]) -> bool:
    cache = _cache_arg(kwargs)
    if cache is not None:
        return True
    pos = _cache_position(kwargs)
    return pos is not None and pos > 0


def _cache_position(kwargs: Dict[str, Any]) -> Optional[int]:
    pos = kwargs.get("cache_position")
    if pos is None:
        return None
    try:
        if hasattr(pos, "max"):
            value = pos.max()
            if hasattr(value, "item"):
                return int(value.item())
            return int(value)
        return int(pos)
    except Exception:
        return None


def _cache_seq_length(cache: Any) -> Optional[int]:
    if cache is None:
        return None
    getter = getattr(cache, "get_seq_length", None)
    if getter is not None:
        for args in ((), (0,)):
            try:
                value = getter(*args)
                if value is not None:
                    return int(value)
            except Exception:
                continue
    seen = getattr(cache, "seen_tokens", None)
    if seen is not None:
        try:
            return int(seen)
        except Exception:
            return None
    return None


def _forward_audit_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    cache = _cache_arg(kwargs)
    return {
        "use_cache": bool(kwargs.get("use_cache", False)),
        "past_key_value_non_null": kwargs.get("past_key_value") is not None,
        "past_key_values_non_null": kwargs.get("past_key_values") is not None,
        "has_decode_cache": _has_decode_cache(kwargs),
        "cache_position": _cache_position(kwargs),
        "cache_seq_length": _cache_seq_length(cache),
    }


def _audit(config: LoopConfig, event: str, **payload: Any) -> None:
    collector = getattr(config, "audit_collector", None)
    record = getattr(collector, "record", None)
    if callable(record):
        record(event, payload)


def _audit_tensor_diff(config: LoopConfig, name: str, before: Any, after: Any) -> None:
    collector = getattr(config, "audit_collector", None)
    record = getattr(collector, "record_tensor_diff", None)
    if callable(record):
        record(name, before, after)
