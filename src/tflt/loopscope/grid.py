"""Deterministic width-constrained window-grid generation."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope.schema import (
    WINDOW_GRID_SCHEMA_VERSION,
    SchemaError,
    attach_manifest_sha256,
    verify_manifest_sha256,
)


Window = Tuple[int, int]


def parse_window(text: str) -> Window:
    cleaned = str(text).strip().replace("[", "").replace("]", "").replace(" ", "")
    for separator in (":", ",", "-"):
        if separator in cleaned:
            left, right = cleaned.split(separator, 1)
            window = (int(left), int(right))
            _validate_window(window)
            return window
    layer = int(cleaned)
    return (layer, layer)


def format_window(window: Window) -> str:
    _validate_window(window)
    return "%d:%d" % window


def generate_window_grid(
    num_hidden_layers: int,
    width: int = 4,
    min_center_fraction: float = 0.20,
    max_center_fraction: float = 0.85,
    candidate_count: int = 9,
    anchor: Window = (12, 15),
    fixed_depth_fraction: float = 0.525,
    random_count: int = 5,
    random_seed: int = 20260710,
) -> Dict[str, Any]:
    """Generate candidates plus frozen fixed/random comparison windows."""

    _validate_generation_args(
        num_hidden_layers,
        width,
        min_center_fraction,
        max_center_fraction,
        candidate_count,
        anchor,
        random_count,
    )
    centers = _linspace(
        min_center_fraction * num_hidden_layers,
        max_center_fraction * num_hidden_layers,
        candidate_count,
    )
    candidates = {
        _window_for_center(center, num_hidden_layers, width) for center in centers
    }
    candidates.add(anchor)
    fixed_window = _window_for_center(
        fixed_depth_fraction * num_hidden_layers,
        num_hidden_layers,
        width,
    )
    candidates.add(fixed_window)

    roles = {window: ["candidate"] for window in candidates}
    roles[anchor].append("anchor")
    roles[fixed_window].append("fixed_depth")
    windows = [_window_record(window, roles[window]) for window in sorted(candidates)]

    eligible_random = _eligible_windows(
        num_hidden_layers,
        width,
        min_center_fraction,
        max_center_fraction,
    )
    if len(eligible_random) < random_count:
        raise ValueError(
            "only %d random-in-band windows are available, need %d"
            % (len(eligible_random), random_count)
        )
    generator = random.Random(random_seed)
    random_windows = sorted(generator.sample(eligible_random, random_count))

    manifest: Dict[str, Any] = {
        "schema_version": WINDOW_GRID_SCHEMA_VERSION,
        "layer_count": int(num_hidden_layers),
        "generation": {
            "width": int(width),
            "min_center_fraction": float(min_center_fraction),
            "max_center_fraction": float(max_center_fraction),
            "candidate_count_requested": int(candidate_count),
            "rounding": "half_up_start_then_boundary_clip",
            "fixed_depth_fraction": float(fixed_depth_fraction),
            "random_count": int(random_count),
            "random_seed": int(random_seed),
        },
        "anchors": {
            "required": format_window(anchor),
            "fixed_depth": format_window(fixed_window),
        },
        "windows": windows,
        "comparison_windows": {
            "fixed_depth": format_window(fixed_window),
            "random_in_band": [format_window(window) for window in random_windows],
        },
    }
    attach_manifest_sha256(manifest)
    validate_window_grid(manifest)
    return manifest


def validate_window_grid(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != WINDOW_GRID_SCHEMA_VERSION:
        raise SchemaError("unsupported window-grid schema_version")
    verify_manifest_sha256(payload)
    layer_count = int(payload.get("layer_count", 0))
    generation = payload.get("generation")
    windows = payload.get("windows")
    if layer_count < 1 or not isinstance(generation, Mapping) or not isinstance(windows, list):
        raise SchemaError("invalid window-grid envelope")
    width = int(generation.get("width", 0))
    seen = set()
    for item in windows:
        if not isinstance(item, Mapping):
            raise SchemaError("window entries must be objects")
        window = (int(item.get("start", -1)), int(item.get("end", -1)))
        _validate_window(window, layer_count=layer_count, width=width)
        if format_window(window) != item.get("window"):
            raise SchemaError("window text disagrees with start/end")
        if window in seen:
            raise SchemaError("window grid contains a duplicate: %s" % (window,))
        seen.add(window)
    anchor = parse_window(str(payload.get("anchors", {}).get("required", "")))
    fixed_window = parse_window(str(payload.get("anchors", {}).get("fixed_depth", "")))
    if anchor not in seen or fixed_window not in seen:
        raise SchemaError("required anchor/fixed-depth window is missing")


def candidate_windows(payload: Mapping[str, Any]) -> List[Window]:
    validate_window_grid(payload)
    return [(int(item["start"]), int(item["end"])) for item in payload["windows"]]


def _window_for_center(center: float, layer_count: int, width: int) -> Window:
    start_float = float(center) - (width - 1) / 2.0
    start = int(math.floor(start_float + 0.5))
    start = min(max(start, 0), layer_count - width)
    return (start, start + width - 1)


def _eligible_windows(
    layer_count: int,
    width: int,
    min_center_fraction: float,
    max_center_fraction: float,
) -> List[Window]:
    low = min_center_fraction * layer_count
    high = max_center_fraction * layer_count
    result = []
    for start in range(0, layer_count - width + 1):
        window = (start, start + width - 1)
        center = (window[0] + window[1]) / 2.0
        if low <= center <= high:
            result.append(window)
    return result


def _window_record(window: Window, roles: Sequence[str]) -> Dict[str, Any]:
    return {
        "window": format_window(window),
        "start": window[0],
        "end": window[1],
        "center": (window[0] + window[1]) / 2.0,
        "roles": sorted(set(roles)),
    }


def _linspace(start: float, end: float, count: int) -> List[float]:
    if count == 1:
        return [(start + end) / 2.0]
    step = (end - start) / float(count - 1)
    return [start + index * step for index in range(count)]


def _validate_generation_args(
    layer_count: int,
    width: int,
    min_fraction: float,
    max_fraction: float,
    candidate_count: int,
    anchor: Window,
    random_count: int,
) -> None:
    if layer_count < 1 or width < 1 or width > layer_count:
        raise ValueError("width must fit inside a positive layer count")
    if not 0.0 <= min_fraction < max_fraction <= 1.0:
        raise ValueError("center fractions must satisfy 0 <= min < max <= 1")
    if candidate_count < 1 or random_count < 0:
        raise ValueError("candidate_count must be positive and random_count non-negative")
    _validate_window(anchor, layer_count=layer_count, width=width)


def _validate_window(
    window: Window,
    layer_count: int = 0,
    width: int = 0,
) -> None:
    start, end = window
    if start < 0 or end < start:
        raise ValueError("window must be an inclusive non-negative range")
    if layer_count and end >= layer_count:
        raise ValueError("window %s exceeds layer count %d" % (window, layer_count))
    if width and end - start + 1 != width:
        raise ValueError("window %s does not have required width %d" % (window, width))
