"""Frozen Gate E Lag1 panel and gold-free score-manifest semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .phase8_test_pool import validate_test_pool
from .phase9_accuracy import EXPECTED_TEST_COUNT, load_config as load_phase9_config
from .phase9_gate_e_spectral_core import DIRECTION_POLICIES, INTERVENTION_MODES, LAG1_ARMS


CONFIG_SCHEMA = "loopscope.phase9.gate_e_lag1.v2"
SCORE_SCHEMA = "loopscope.phase9.gate_e.score_manifest.v2"
CANARY_SCHEMA = "loopscope.phase9.gate_e.canary_indices.v1"
EXPECTED_NEW_COUNT = 12
K_VALUES = (3, 4)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "loopscope" / "phase9_gate_e_lag1.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    target = default_config_path() if path is None else Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == CONFIG_SCHEMA, "unexpected Gate E config schema")
    _require(config.get("phase") == 9 and config.get("gate") == "E", "Gate E phase/gate mismatch")
    _require(config.get("task") == "mmlu" and config.get("num_fewshot") == 5, "Gate E task recipe mismatch")
    _require(config.get("dataset_repo") == "cais/mmlu" and config.get("evaluation_split") == "test",
             "Gate E dataset recipe mismatch")
    _require(config.get("evaluation_count") == EXPECTED_TEST_COUNT, "Gate E test count differs from contract")
    _require(config.get("K") == list(K_VALUES), "Gate E K must be [3, 4]")
    _require(config.get("alpha") == 1.0 and config.get("beta") == 0.0 and config.get("strength") == 0.5,
             "Gate E Euler or damping recipe mismatch")
    _require(config.get("batch_size") == 1 and config.get("decode_mode") == "bypass",
             "Gate E batch/decode recipe mismatch")
    _require(config.get("new_arms") == list(LAG1_ARMS), "Gate E arm set differs from contract")
    _require(config.get("direction_policies") == list(DIRECTION_POLICIES),
             "Gate E direction policies must explicitly include fixed_t0 and lag1")
    _require(config.get("intervention_modes") == list(INTERVENTION_MODES),
             "Gate E intervention modes must explicitly include spectral and matched_norm")
    _require(config.get("formal_direction_policy") == "lag1",
             "Gate E formal score panel is restricted to direction_policy=lag1")
    expected_policy_rules = {
        "fixed_t0": {
            "source": "native residual D0 over all valid retained prompt-token rows",
            "used_at": "every t>=1 answer-position residual only",
            "fit_at": "t=0 only",
            "final_round_fit": False,
        },
        "lag1": {
            "source": "previous round native residual D_(t-1) over all valid retained prompt-token rows",
            "used_at": "t>=1 answer-position residual only",
            "fit_at": "t=0..K-2",
            "final_round_fit": False,
        },
    }
    _require(config.get("direction_policy_rules") == expected_policy_rules,
             "Gate E direction policy semantics differ from the frozen contract")
    _require(config.get("direction_svd") ==
             "uncentered, unnormalized, FP32 reduced SVD; CUDA gesvd; canonical sign and unit norm",
             "Gate E direction SVD differs from the frozen contract")
    _require(config.get("direction_strength") == 0.5,
             "Gate E direction strength differs from the frozen contract")
    expected_intervention_rule = {
        "spectral": "delta - strength * v * (v dot delta)",
        "matched_norm": "uniformly scale delta to the norm of its selected-direction spectral update",
    }
    _require(config.get("intervention_rule") == expected_intervention_rule,
             "Gate E intervention mode semantics differ from the frozen contract")
    models = config.get("models")
    base = load_phase9_config()
    _require(isinstance(models, list) and len(models) == len(base["models"]), "Gate E model set differs from Phase 9")
    for observed, frozen in zip(models, base["models"]):
        _require(observed == frozen, "Gate E model/window/revision differs from Phase 9 contract")


def _prefix(model_index: int) -> str:
    return ("q4", "q17")[model_index]


def new_cells(config: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    config = load_config() if config is None else config
    validate_config(config)
    cells: List[Dict[str, Any]] = []
    for model_index, model in enumerate(config["models"]):
        for window in model["windows"]:
            layers = list(window["layers"])
            for k in K_VALUES:
                for arm in LAG1_ARMS:
                    suffix = "lag1-online" if arm == "Lag1-Online" else "lag1-matched-norm"
                    cells.append({
                        "model_index": model_index,
                        "model": model["model"],
                        "revision": model["revision"],
                        "dtype": model["dtype"],
                        "cache_strategy": model["cache_strategy"],
                        "window": layers,
                        "boundaries": list(window["boundaries"]),
                        "k": k,
                        "arm": arm,
                        "direction_policy": config["formal_direction_policy"],
                        "intervention_mode": "spectral" if arm == "Lag1-Online" else "matched_norm",
                        "fit_semantics": "lag1_previous_round_native_full_prefix",
                        "source": "phase9_gate_e",
                        "new_in_phase9_gate_e": True,
                        "cell_id": f"{_prefix(model_index)}-w{layers[0]}-{layers[1]}-k{k}-{suffix}",
                    })
    _require(len(cells) == EXPECTED_NEW_COUNT, "Gate E requires exactly 12 new cells")
    return cells


def validate_new_cells(cells: Iterable[Mapping[str, Any]], config: Optional[Mapping[str, Any]] = None) -> Dict[str, int]:
    config = load_config() if config is None else config
    expected = new_cells(config)
    actual = list(cells)
    _require(actual == expected, "Gate E cell membership differs from frozen contract")
    _require(len({cell["cell_id"] for cell in actual}) == EXPECTED_NEW_COUNT, "Gate E cell IDs are not unique")
    return {"cell_count": EXPECTED_NEW_COUNT, "sample_count_per_cell": EXPECTED_TEST_COUNT}


def score_manifest(
    config: Optional[Mapping[str, Any]] = None,
    pool_path: Optional[Path] = None,
    scope: str = "FORMAL_TEST",
) -> Dict[str, Any]:
    if scope not in ("PREFLIGHT_ONLY", "FORMAL_TEST"):
        raise ValueError("Gate E score scope must be PREFLIGHT_ONLY or FORMAL_TEST")
    config = load_config() if config is None else config
    cells = new_cells(config)
    return {
        "schema": SCORE_SCHEMA,
        "phase": 9,
        "gate": "E",
        "scope": scope,
        "dataset": {"repo": config["dataset_repo"], "revision": config["dataset_revision"], "split": "test"},
        "pool": str(pool_path) if pool_path is not None else None,
        "target_gold_loaded": False,
        "cells": cells,
        "cell_count": len(cells),
        "sample_count_per_cell": EXPECTED_TEST_COUNT,
        "new_configuration_sample_records": len(cells) * EXPECTED_TEST_COUNT,
        "arms": list(LAG1_ARMS),
        "direction_policies": [config["formal_direction_policy"]],
        "intervention_modes": list(INTERVENTION_MODES),
        "closure": "Gate E new score identities and finite raw scores only; no gold or outcomes",
    }


def validate_score_manifest(
    value: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    pool_path: Optional[Path] = None,
    scope: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    config = load_config() if config is None else config
    _require(value.get("schema") == SCORE_SCHEMA, "unexpected Gate E score manifest schema")
    _require(value.get("gate") == "E" and value.get("phase") == 9, "Gate E score manifest phase mismatch")
    _require(value.get("dataset") == {
        "repo": config["dataset_repo"], "revision": config["dataset_revision"], "split": "test"
    }, "Gate E score manifest dataset differs from contract")
    if pool_path is not None:
        _require(value.get("pool") == str(pool_path), "Gate E score manifest pool differs from requested pool")
    if scope is not None:
        _require(value.get("scope") == scope, "Gate E score manifest scope differs from requested scope")
    _require(value.get("target_gold_loaded") is False, "Gate E score manifest crossed the gold boundary")
    _require(value.get("direction_policies") == [config["formal_direction_policy"]],
             "Gate E score manifest direction policy differs from the authorized formal panel")
    _require(value.get("intervention_modes") == list(INTERVENTION_MODES),
             "Gate E score manifest intervention modes differ from the frozen panel")
    cells = new_cells(config)
    _require(value.get("cells") == cells, "Gate E score manifest cell membership differs")
    _require(value.get("cell_count") == EXPECTED_NEW_COUNT, "Gate E score manifest cell count differs")
    _require(value.get("sample_count_per_cell") == EXPECTED_TEST_COUNT,
             "Gate E score manifest sample count differs")
    _require(value.get("new_configuration_sample_records") == EXPECTED_NEW_COUNT * EXPECTED_TEST_COUNT,
             "Gate E score manifest sample total differs")
    return {cell["cell_id"]: cell for cell in cells}


def representative_canary_indices(pool: Mapping[str, Any], model: str, count: int = 512) -> List[int]:
    rows = list(pool.get("rows", ()))
    if len(rows) != EXPECTED_TEST_COUNT or count <= 0 or count > len(rows):
        raise ValueError("Gate E canary selection requires the complete test pool")
    for row in rows:
        lengths = row.get("prompt_token_lengths", {})
        if model not in lengths or type(lengths[model]) is not int or lengths[model] <= 0:
            raise ValueError("Gate E canary selection requires frozen prompt lengths")
    ordinary_count = count // 2
    selected = list(range(ordinary_count))
    tail = sorted(range(len(rows)), key=lambda index: (-rows[index]["prompt_token_lengths"][model], index))
    for index in tail:
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError("Gate E canary selection is not unique")
    return sorted(selected)


def canary_bundle(pool: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None, count: int = 512) -> Dict[str, Any]:
    config = load_config() if config is None else config
    validate_test_pool(pool)
    return {
        "schema": CANARY_SCHEMA,
        "dataset": {"repo": config["dataset_repo"], "revision": config["dataset_revision"], "split": "test"},
        "count": count,
        "canonical_count": EXPECTED_TEST_COUNT,
        "policy": "first canonical half plus longest prompt upper-tail half; labels and scores unused",
        "model_indices": {
            model["model"]: representative_canary_indices(pool, model["model"], count)
            for model in config["models"]
        },
        "target_gold_loaded": False,
    }


def validate_canary_bundle(
    value: Mapping[str, Any],
    pool: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    count: int = 512,
) -> Dict[str, List[int]]:
    config = load_config() if config is None else config
    validate_test_pool(pool)
    _require(value.get("schema") == CANARY_SCHEMA, "unexpected Gate E canary schema")
    _require(value.get("dataset") == {
        "repo": config["dataset_repo"], "revision": config["dataset_revision"], "split": "test"
    }, "Gate E canary dataset differs")
    _require(value.get("count") == count and value.get("canonical_count") == EXPECTED_TEST_COUNT,
             "Gate E canary count differs")
    _require(value.get("target_gold_loaded") is False, "Gate E canary crossed the gold boundary")
    observed = value.get("model_indices")
    result: Dict[str, List[int]] = {}
    for model in config["models"]:
        name = model["model"]
        _require(isinstance(observed, Mapping) and name in observed, "Gate E canary model set differs")
        indices = list(observed[name])
        _require(indices == sorted(indices) and len(indices) == count and len(set(indices)) == count,
                 "Gate E canary indices must be sorted and unique")
        _require(indices == representative_canary_indices(pool, name, count),
                 "Gate E canary indices differ from frozen selection")
        result[name] = indices
    return result
