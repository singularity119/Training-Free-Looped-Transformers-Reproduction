"""Phase 9 frozen panel and manifest semantics.

This module does not score a model or read labels.  It only expands the
contract's three model/window pairs into the 47 logical configurations:
27 historical non-Native cells, 18 new Online/Matched-norm cells, and two
model-level Native reference cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


SCHEMA = "loopscope.phase9.panel.v2"
CONFIG_SCHEMA = "loopscope.phase9.model_windows.v2"
ARMS = ("Loop", "Shared-t1", "Shared-t0", "Online-t0", "Matched-norm")
NEW_ARMS = ("Online-t0", "Matched-norm")
HISTORICAL_ARMS = ("Loop", "Shared-t1", "Shared-t0")
K_VALUES = (2, 3, 4)
MODEL_PREFIXES = ("q4", "q17")
NATIVE_ARM = "Native"
EXPECTED_MODEL_ORDER = ("Qwen/Qwen3-4B-Base", "Qwen/Qwen3-1.7B-Base")
EXPECTED_MODEL_WINDOWS = {
    "Qwen/Qwen3-4B-Base": {(13, 16), (15, 18)},
    "Qwen/Qwen3-1.7B-Base": {(12, 15)},
}
EXPECTED_WINDOW_COUNT = 3
EXPECTED_NEW_COUNT = 18
EXPECTED_HISTORICAL_NON_NATIVE_COUNT = 27
EXPECTED_NATIVE_COUNT = 2
EXPECTED_CELL_COUNT = 47


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "loopscope" / "phase9_model_windows.json"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    target = default_config_path() if path is None else Path(path)
    with target.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == CONFIG_SCHEMA, "unexpected Phase 9 config schema")
    _require(config.get("task") == "mmlu" and config.get("num_fewshot") == 5, "Phase 9 task recipe mismatch")
    _require(config.get("K") == list(K_VALUES), "Phase 9 K must be [2, 3, 4]")
    _require(config.get("alpha") == 1.0 and config.get("strength") == 0.5, "Phase 9 alpha/strength mismatch")
    _require(config.get("batch_size") == 1 and config.get("decode_mode") == "bypass", "Phase 9 runtime recipe mismatch")
    _require(config.get("arms") == list(ARMS) and config.get("new_arms") == list(NEW_ARMS), "Phase 9 arm set differs from contract v2")
    models = config.get("models")
    _require(isinstance(models, list) and len(models) == 2, "Phase 9 requires two model entries")
    _require(tuple(model.get("model") for model in models) == EXPECTED_MODEL_ORDER, "Phase 9 model order differs from contract v2")
    seen = set()
    observed_by_model = {}
    for model in models:
        _require(model.get("model") and model.get("revision"), "model revision is required")
        _require(model["model"] in EXPECTED_MODEL_WINDOWS, "unexpected Phase 9 model")
        _require(model.get("dtype") in ("bfloat16", "float16"), "unexpected Phase 9 dtype")
        _require(model.get("cache_strategy") in ("first", "last"), "unexpected Phase 9 cache strategy")
        windows = model.get("windows")
        _require(isinstance(windows, list), "model windows must be a list")
        observed_by_model.setdefault(model["model"], set())
        for window in windows:
            layers = window.get("layers")
            boundaries = window.get("boundaries")
            _require(isinstance(layers, list) and len(layers) == 2, "window must have two inclusive layers")
            _require(boundaries == [layers[0], layers[1] + 1], "window boundary does not close at end+1")
            _require(0 <= layers[0] <= layers[1] < model["num_hidden_layers"], "window is outside model depth")
            key = (model["model"], tuple(layers))
            _require(key not in seen, "duplicate model/window entry")
            seen.add(key)
            observed_by_model[model["model"]].add(tuple(layers))
    _require(observed_by_model == EXPECTED_MODEL_WINDOWS, "Phase 9 model/window selection differs from contract v2")
    _require(len(seen) == EXPECTED_WINDOW_COUNT, "Phase 9 contract requires three model/window pairs")


def _model_prefix(model_index: int) -> str:
    try:
        return MODEL_PREFIXES[model_index]
    except IndexError as exc:
        raise ValueError("unsupported Phase 9 model index") from exc


def _base_cell(model_index: int, model: Mapping[str, Any], window: Optional[Mapping[str, Any]], arm: str, k: Optional[int]) -> Dict[str, Any]:
    prefix = _model_prefix(model_index)
    if window is None:
        return {
            "model_index": model_index,
            "model": model["model"],
            "revision": model["revision"],
            "dtype": model["dtype"],
            "cache_strategy": model["cache_strategy"],
            "window": None,
            "boundaries": None,
            "k": None,
            "arm": NATIVE_ARM,
            "fit_t": None,
            "source": "phase8",
            "new_in_phase9": False,
            "cell_id": f"{prefix}-native",
        }
    layers = list(window["layers"])
    suffix = {
        "Loop": "loop",
        "Shared-t1": "shared-t1",
        "Shared-t0": "shared-t0",
        "Online-t0": "online-t0",
        "Matched-norm": "matched-norm",
    }[arm]
    return {
        "model_index": model_index,
        "model": model["model"],
        "revision": model["revision"],
        "dtype": model["dtype"],
        "cache_strategy": model["cache_strategy"],
        "window": layers,
        "boundaries": list(window["boundaries"]),
        "k": k,
        "arm": arm,
        "fit_t": {"Shared-t1": 1, "Shared-t0": 0}.get(arm),
        "source": "phase9" if arm in NEW_ARMS else "phase8",
        "new_in_phase9": arm in NEW_ARMS,
        "cell_id": f"{prefix}-w{layers[0]}-{layers[1]}-k{k}-{suffix}",
    }


def logical_panel(config: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return the ordered, complete 47-cell logical panel."""

    config = load_config() if config is None else config
    validate_config(config)
    cells: List[Dict[str, Any]] = []
    for model_index, model in enumerate(config["models"]):
        cells.append(_base_cell(model_index, model, None, NATIVE_ARM, None))
        for window in model["windows"]:
            for k in K_VALUES:
                for arm in ARMS:
                    cells.append(_base_cell(model_index, model, window, arm, k))
    validate_panel(cells, config)
    return cells


def validate_panel(cells: Iterable[Mapping[str, Any]], config: Optional[Mapping[str, Any]] = None) -> Dict[str, int]:
    """Validate panel membership without reading prompts, labels, or scores."""

    config = load_config() if config is None else config
    validate_config(config)
    cells = list(cells)
    expected_count = EXPECTED_CELL_COUNT
    _require(len(cells) == expected_count, f"Phase 9 requires exactly {expected_count} logical cells")
    _require(len({cell.get("cell_id") for cell in cells}) == expected_count, "Phase 9 cell IDs must be unique")
    _require(sum(bool(cell.get("new_in_phase9")) for cell in cells) == EXPECTED_NEW_COUNT, "Phase 9 requires 18 new cells")
    _require(sum(cell.get("arm") == NATIVE_ARM for cell in cells) == EXPECTED_NATIVE_COUNT, "Phase 9 requires two Native cells")
    _require(sum(cell.get("arm") in HISTORICAL_ARMS for cell in cells) == EXPECTED_HISTORICAL_NON_NATIVE_COUNT, "Phase 9 requires 27 historical non-Native cells")
    expected = {cell["cell_id"]: cell for cell in logical_panel_without_recursive_validation(config)}
    actual = {cell["cell_id"]: dict(cell) for cell in cells}
    _require(actual == expected, "Phase 9 panel differs from frozen model/window/K/arm membership")
    return {
        "cell_count": expected_count,
        "historical_count": EXPECTED_HISTORICAL_NON_NATIVE_COUNT + EXPECTED_NATIVE_COUNT,
        "new_count": EXPECTED_NEW_COUNT,
        "native_count": EXPECTED_NATIVE_COUNT,
    }


def logical_panel_without_recursive_validation(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    for model_index, model in enumerate(config["models"]):
        cells.append(_base_cell(model_index, model, None, NATIVE_ARM, None))
        for window in model["windows"]:
            for k in K_VALUES:
                for arm in ARMS:
                    cells.append(_base_cell(model_index, model, window, arm, k))
    return cells


def manifest(config: Optional[Mapping[str, Any]] = None, config_path: Optional[Path] = None) -> Dict[str, Any]:
    config = load_config(config_path) if config is None else config
    cells = logical_panel(config)
    counts = validate_panel(cells, config)
    return {
        "schema": SCHEMA,
        "phase": 9,
        "config": str(config_path or default_config_path()),
        "target_gold_loaded": False,
        "cells": cells,
        "cell_count": counts["cell_count"],
        "historical_cell_count": counts["historical_count"],
        "new_cell_count": counts["new_count"],
        "native_cell_count": counts["native_count"],
        "new_arms": list(NEW_ARMS),
        "sample_count_per_cell": 14042,
        "total_configuration_sample_records": counts["cell_count"] * 14042,
        "new_configuration_sample_records": counts["new_count"] * 14042,
        "closure": "logical membership only; no scores or outcomes",
    }


def write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    """Write a manifest once; never overwrite a previous attempt."""

    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
