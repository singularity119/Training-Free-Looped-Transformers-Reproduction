"""Artifact schemas and write-once helpers for LoopScope phase one."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


PROBE_SCHEMA_VERSION = "loopscope.probe.v1"
WINDOW_GRID_SCHEMA_VERSION = "loopscope.window-grid.v1"
SELECTION_SCHEMA_VERSION = "loopscope.selection.v1"
ANALYSIS_SCHEMA_VERSION = "loopscope.analysis.v1"


class SchemaError(ValueError):
    """Raised when a LoopScope artifact violates its public contract."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes and reject NaN/Inf."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(payload: Mapping[str, Any], hash_field: str = "manifest_sha256") -> str:
    """Hash a manifest while excluding its top-level self-hash field."""

    body = {key: value for key, value in payload.items() if key != hash_field}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def attach_manifest_sha256(
    payload: MutableMapping[str, Any], hash_field: str = "manifest_sha256"
) -> MutableMapping[str, Any]:
    payload[hash_field] = manifest_sha256(payload, hash_field=hash_field)
    return payload


def verify_manifest_sha256(
    payload: Mapping[str, Any], hash_field: str = "manifest_sha256"
) -> None:
    expected = payload.get(hash_field)
    actual = manifest_sha256(payload, hash_field=hash_field)
    if not isinstance(expected, str) or expected != actual:
        raise SchemaError(
            "%s mismatch: expected %r, computed %s" % (hash_field, expected, actual)
        )


def ensure_new_directory(path: Path) -> Path:
    """Create an output directory only when no object already exists there."""

    if path.exists():
        raise FileExistsError("refusing to reuse existing output directory: %s" % path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_new_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact without replacing an existing file."""

    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def validate_probe_report(payload: Mapping[str, Any]) -> None:
    """Validate the common layer/window probe envelope."""

    required = {
        "schema_version",
        "git",
        "model",
        "tokenizer",
        "runtime",
        "probe_pool",
        "position_rule",
        "layer_metrics",
        "window_metrics",
        "warnings",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise SchemaError("probe report missing fields: %s" % ", ".join(missing))
    if payload["schema_version"] != PROBE_SCHEMA_VERSION:
        raise SchemaError("unsupported probe schema_version: %r" % payload["schema_version"])
    for key in ("git", "model", "tokenizer", "runtime", "probe_pool"):
        if not isinstance(payload[key], Mapping):
            raise SchemaError("probe report %s must be an object" % key)
    for key in ("layer_metrics", "window_metrics", "warnings"):
        if not isinstance(payload[key], list):
            raise SchemaError("probe report %s must be a list" % key)
    _reject_non_finite(payload)


def probe_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a scalar-only summary compatible with the existing report CLI."""

    validate_probe_report(payload)
    probe_pool = payload["probe_pool"]
    model = payload["model"]
    windows = payload["window_metrics"]
    valid_windows = sum(
        1 for item in windows if isinstance(item, Mapping) and bool(item.get("valid"))
    )
    metrics: Dict[str, Any] = {
        "schema_version": payload["schema_version"],
        "model_alias": str(model.get("alias", "")),
        "model_revision": str(model.get("revision", "")),
        "layer_count": int(model.get("layer_count", 0) or 0),
        "probe_count": int(probe_pool.get("count", 0) or 0),
        "layer_metric_count": len(payload["layer_metrics"]),
        "window_metric_count": len(windows),
        "valid_window_count": valid_windows,
        "warning_count": len(payload["warnings"]),
    }
    if any(not isinstance(value, (str, int, float, bool)) for value in metrics.values()):
        raise SchemaError("probe summary contains a non-scalar metric")
    return {"results": {"loopscope_probe": metrics}}


def require_keys(payload: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys).difference(payload))
    if missing:
        raise SchemaError("%s missing fields: %s" % (context, ", ".join(missing)))


def _reject_non_finite(value: Any, path: Sequence[str] = ()) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaError("non-finite number at %s" % ".".join(path))
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, tuple(path) + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, tuple(path) + (str(index),))
