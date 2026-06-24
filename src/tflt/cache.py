"""Best-effort KV cache snapshot/crop helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class CacheSnapshot:
    cache: Any
    seq_length: Optional[int]
    key_lengths: Optional[List[Optional[int]]]
    value_lengths: Optional[List[Optional[int]]]


def cache_arg(kwargs: dict) -> Any:
    if "past_key_value" in kwargs:
        return kwargs["past_key_value"]
    if "past_key_values" in kwargs:
        return kwargs["past_key_values"]
    return None


def snapshot_cache(kwargs: dict) -> Optional[CacheSnapshot]:
    cache = cache_arg(kwargs)
    if cache is None:
        return None
    return CacheSnapshot(
        cache=cache,
        seq_length=_seq_length(cache),
        key_lengths=_list_lengths(getattr(cache, "key_cache", None)),
        value_lengths=_list_lengths(getattr(cache, "value_cache", None)),
    )


def crop_cache(snapshot: Optional[CacheSnapshot]) -> None:
    if snapshot is None:
        return
    cache = snapshot.cache
    if snapshot.seq_length is not None and hasattr(cache, "crop"):
        try:
            cache.crop(snapshot.seq_length)
            return
        except Exception:
            pass
    _restore_list_lengths(getattr(cache, "key_cache", None), snapshot.key_lengths)
    _restore_list_lengths(getattr(cache, "value_cache", None), snapshot.value_lengths)


def _seq_length(cache: Any) -> Optional[int]:
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


def _list_lengths(values: Any) -> Optional[List[Optional[int]]]:
    if values is None:
        return None
    try:
        items = list(values)
    except Exception:
        return None
    return [_tensor_seq_len(item) for item in items]


def _restore_list_lengths(values: Any, lengths: Optional[List[Optional[int]]]) -> None:
    if values is None or lengths is None:
        return
    for idx, length in enumerate(lengths):
        if length is None:
            continue
        try:
            values[idx] = _slice_seq(values[idx], length)
        except Exception:
            continue


def _tensor_seq_len(value: Any) -> Optional[int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    try:
        return int(shape[-2])
    except Exception:
        return None


def _slice_seq(value: Any, length: int) -> Any:
    # KV tensors are normally [batch, heads, seq, head_dim].
    return value[..., :length, :]
