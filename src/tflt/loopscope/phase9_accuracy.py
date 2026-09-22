"""Phase 9 frozen panel and manifest semantics.

This module does not score a model or read labels.  It only expands the
contract's three model/window pairs into the 47 logical configurations:
27 historical non-Native cells, 18 new Online/Matched-norm cells, and two
model-level Native reference cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


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
EXPECTED_TEST_COUNT = 14042
EXPECTED_NEW_SCORE_COUNT = 18
SCORE_SCHEMA = "loopscope.phase9.score_manifest.v1"


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


def new_cells(config: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return the 18 outcome-free Phase 9 score cells in panel order."""

    config = load_config() if config is None else config
    return [dict(cell) for cell in logical_panel(config) if cell["new_in_phase9"]]


def score_manifest(
    config: Optional[Mapping[str, Any]] = None,
    pool_path: Optional[Path] = None,
    scope: str = "FORMAL_TEST",
) -> Dict[str, Any]:
    """Build the gold-free manifest consumed by the new-arm score launcher."""

    if scope not in ("PREFLIGHT_ONLY", "FORMAL_TEST"):
        raise ValueError("Phase 9 score scope must be PREFLIGHT_ONLY or FORMAL_TEST")
    config = load_config() if config is None else config
    cells = new_cells(config)
    if len(cells) != EXPECTED_NEW_SCORE_COUNT:
        raise ValueError("Phase 9 score manifest requires exactly 18 new cells")
    dataset = {
        "repo": config["dataset_repo"],
        "revision": config["dataset_revision"],
        "split": "test",
    }
    return {
        "schema": SCORE_SCHEMA,
        "phase": 9,
        "scope": scope,
        "dataset": dataset,
        "pool": str(pool_path) if pool_path is not None else None,
        "target_gold_loaded": False,
        "cells": cells,
        "cell_count": len(cells),
        "sample_count_per_cell": EXPECTED_TEST_COUNT,
        "new_configuration_sample_records": len(cells) * EXPECTED_TEST_COUNT,
        "arms": list(NEW_ARMS),
        "closure": "new score identities and finite raw scores only; no gold or outcomes",
    }


def validate_score_manifest(
    value: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    pool_path: Optional[Path] = None,
    scope: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Validate new-cell membership without loading labels or score outcomes."""

    config = load_config() if config is None else config
    _require(value.get("schema") == SCORE_SCHEMA, "unexpected Phase 9 score manifest schema")
    _require(value.get("dataset") == {
        "repo": config["dataset_repo"],
        "revision": config["dataset_revision"],
        "split": "test",
    }, "Phase 9 score manifest dataset differs from contract")
    if pool_path is not None:
        _require(value.get("pool") == str(pool_path), "Phase 9 score manifest pool differs from requested pool")
    if scope is not None:
        _require(value.get("scope") == scope, "Phase 9 score manifest scope differs from requested scope")
    _require(value.get("target_gold_loaded") is False, "Phase 9 score manifest is not gold-free")
    expected = new_cells(config)
    actual = list(value.get("cells", ()))
    _require(actual == expected, "Phase 9 score manifest differs from frozen new-cell panel")
    _require(value.get("cell_count") == EXPECTED_NEW_SCORE_COUNT, "Phase 9 score cell count differs from contract")
    _require(value.get("sample_count_per_cell") == EXPECTED_TEST_COUNT, "Phase 9 sample count differs from contract")
    _require(value.get("new_configuration_sample_records") == EXPECTED_NEW_SCORE_COUNT * EXPECTED_TEST_COUNT,
             "Phase 9 score manifest sample total differs from contract")
    return {cell["cell_id"]: cell for cell in actual}


def representative_canary_indices(
    pool: Mapping[str, Any],
    model: str,
    count: int = 512,
) -> List[int]:
    """Select a fixed, label-free mixed canonical canary for one model.

    The first half retains ordinary canonical identities.  The second half
    takes the longest prompts, which gives the A800 measurement a real upper
    tail without using labels, predictions, or scores.
    """

    rows = list(pool.get("rows", ()))
    if len(rows) != EXPECTED_TEST_COUNT or count <= 0 or count > len(rows):
        raise ValueError("Phase 9 canary selection requires the complete test pool")
    for row in rows:
        lengths = row.get("prompt_token_lengths", {})
        if model not in lengths or type(lengths[model]) is not int or lengths[model] <= 0:
            raise ValueError("Phase 9 canary selection requires frozen prompt lengths")
    ordinary_count = count // 2
    selected = list(range(ordinary_count))
    tail = sorted(range(len(rows)), key=lambda index: (-rows[index]["prompt_token_lengths"][model], index))
    for index in tail:
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError("Phase 9 canary selection did not produce the requested unique count")
    return sorted(selected)


def canary_bundle(
    pool: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    count: int = 512,
) -> Dict[str, Any]:
    """Create a readable canary/complement identity bundle from the safe pool."""

    config = load_config() if config is None else config
    models = [model["model"] for model in config["models"]]
    by_model = {model: representative_canary_indices(pool, model, count) for model in models}
    return {
        "schema": "loopscope.phase9.canary_indices.v1",
        "dataset": {
            "repo": config["dataset_repo"],
            "revision": config["dataset_revision"],
            "split": "test",
        },
        "count": count,
        "canonical_count": EXPECTED_TEST_COUNT,
        "policy": "first canonical half plus longest prompt upper-tail half; labels and scores unused",
        "model_indices": by_model,
        "target_gold_loaded": False,
    }


def validate_canary_bundle(
    value: Mapping[str, Any],
    pool: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    count: int = 512,
) -> Dict[str, List[int]]:
    """Validate the write-once canary bundle and return model-index lists."""

    config = load_config() if config is None else config
    expected_dataset = {
        "repo": config["dataset_repo"],
        "revision": config["dataset_revision"],
        "split": "test",
    }
    _require(value.get("schema") == "loopscope.phase9.canary_indices.v1", "unexpected Phase 9 canary schema")
    _require(value.get("dataset") == expected_dataset and value.get("canonical_count") == EXPECTED_TEST_COUNT,
             "Phase 9 canary dataset differs from contract")
    _require(value.get("count") == count and value.get("target_gold_loaded") is False,
             "Phase 9 canary count or gold boundary differs from contract")
    rows = list(pool.get("rows", ()))
    _require(len(rows) == EXPECTED_TEST_COUNT, "Phase 9 canary requires the complete test pool")
    observed = value.get("model_indices")
    expected_models = [model["model"] for model in config["models"]]
    _require(isinstance(observed, Mapping) and set(observed) == set(expected_models),
             "Phase 9 canary model set differs from contract")
    result: Dict[str, List[int]] = {}
    for model in expected_models:
        indices = list(observed[model])
        _require(len(indices) == count and indices == sorted(indices) and len(set(indices)) == count,
                 "Phase 9 canary indices must be sorted and unique")
        _require(all(type(index) is int and 0 <= index < EXPECTED_TEST_COUNT for index in indices),
                 "Phase 9 canary index outside canonical test population")
        _require(indices == representative_canary_indices(pool, model, count),
                 "Phase 9 canary indices differ from the frozen label-free selection")
        result[model] = indices
    return result
