"""Frozen Phase 11 task panels and the three K2 display aliases.

This module reads no model, dataset, label, or outcome. Data binding remains a
separate Gate A artifact; a panel by itself never authorizes model execution.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

DATASETS = ("arc_challenge", "mmlu_pro", "gpqa_main")
POLICIES = ("fixed_t0", "lag1", "current_t")
STRENGTHS = (0.1, 0.5, 0.9)
COUNTS = {"arc_challenge": 1172, "mmlu_pro": 12032, "gpqa_main": 448}
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
CONFIG_SCHEMA = "loopscope.phase11.panel_config.v1"
PANEL_SCHEMA = "loopscope.phase11.panel.v1"
MANIFEST_SCHEMA = "loopscope.phase11.score_manifest.v1"
SCOPES = ("PREFLIGHT_ONLY", "FORMAL_TEST")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs/loopscope/phase11_panel.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    value = json.loads((Path(path) if path else default_config_path()).read_text(encoding="utf-8"))
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    require(config.get("schema") == CONFIG_SCHEMA and config.get("phase") == 11,
            "unexpected Phase 11 config")
    require(config.get("scientific_contract") == ".planning/phase11/loopscope_phase11_contract_v1.md",
            "Phase 11 requires contract v1")
    expected = {"model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
        "dtype": "bfloat16", "window": [15, 18], "boundaries": [15, 19],
        "k_values": [2, 3], "strengths": list(STRENGTHS), "direction_policies": list(POLICIES),
        "rank": 1, "batch_size": 1, "strategy": "damped_euler", "iteration_mode": "block",
        "alpha": 1.0, "beta": 0.0, "cache_strategy": "first", "decode_mode": "full",
        "direction_fit_scope": "own_prefill_prompt", "decode_direction_source": "frozen_prefill_bank",
        "max_new_tokens": 2048, "do_sample": False, "max_length": 262144}
    for key, value in expected.items():
        require(config.get(key) == value, f"Phase 11 {key} differs from frozen recipe")
    require(config.get("bootstrap") == {"replicates": 10000, "seed": 20261008, "ci_level": 0.95},
            "Phase 11 bootstrap differs from contract")
    require(config.get("family_sizes") == {"online": 30, "policy": 12, "loop": 2},
            "Phase 11 Holm families differ from contract")
    datasets = config.get("datasets", {})
    require(set(datasets) == set(DATASETS), "Phase 11 must contain exactly three tasks")
    expected_tasks = {
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", 25, 1172, "acc_norm", "candidate_loglikelihood"),
        "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test", 5, 12032, "acc", "generation"),
        "gpqa_main": ("Idavidrein/gpqa", "gpqa_main", "main_csv", 0, 448, "acc", "generation"),
    }
    for dataset, facts in expected_tasks.items():
        recipe = datasets[dataset]
        keys = ("repo", "name", "split", "num_fewshot", "evaluation_count", "primary_metric", "evaluation_mode")
        require(tuple(recipe.get(key) for key in keys) == facts, f"Phase 11 {dataset} recipe differs")
    require(datasets["arc_challenge"].get("revision") == "210d026faf9955653af8916fad021475a3f00453" and
            datasets["arc_challenge"].get("fewshot_seed") == 20261002,
            "ARC revision or demonstrations differ")
    require(datasets["mmlu_pro"].get("revision") == "b189ec765aa7ed75c8acfea42df31fdae71f97be" and
            datasets["mmlu_pro"].get("fewshot_rule") == "category_validation_first_n_5",
            "MMLU-Pro revision or demonstrations differ")
    gpqa = datasets["gpqa_main"]
    for key, value in {"revision": "56686c06f5e19865c153de0fdb11be3890014df7", "source_kind": "author_github_archive",
                       "github_repo": "idavidrein/gpqa", "github_commit": "56686c06f5e19865c153de0fdb11be3890014df7",
                       "archive_member": "dataset/gpqa_main.csv"}.items():
        require(gpqa.get(key) == value, "GPQA accepted author source binding differs")
    require(datasets["gpqa_main"].get("option_shuffle_seed") == 20261008,
            "GPQA option permutation seed differs")


def build_panel(dataset: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    require(dataset in DATASETS, "unknown Phase 11 task")
    config = load_config() if config is None else config
    validate_config(config)
    base = {key: copy.deepcopy(config[key]) for key in (
        "model", "revision", "tokenizer_revision", "dtype", "batch_size", "strategy", "iteration_mode",
        "alpha", "beta", "cache_strategy", "decode_mode", "rank", "direction_fit_scope", "decode_direction_source")}
    base.update(phase=11, dataset=dataset)
    cells, display = [], []

    def add(row: dict[str, Any], alias_of: str | None = None) -> None:
        if alias_of is None:
            cells.append(row)
        display.append({**row, "result_cell_id": alias_of or row["cell_id"], "alias_of": alias_of,
                        "independent_result": alias_of is None})

    add({**base, "cell_id": f"{dataset}-q4-native", "arm": "Native", "window": None,
         "boundaries": None, "k": None, "direction_policy": None, "strength": 0.0,
         "intervention_mode": None})
    for k in (2, 3):
        stem = f"{dataset}-q4-w15-18-k{k}"
        own = {**base, "window": [15, 18], "boundaries": [15, 19], "k": k}
        add({**own, "cell_id": stem + "-loop", "arm": "Loop", "direction_policy": None,
             "strength": 0.0, "intervention_mode": None})
        for policy in POLICIES:
            for strength in STRENGTHS:
                alias = f"{stem}-online-fixed-t0-lambda{strength:.1f}" if k == 2 and policy == "lag1" else None
                add({**own, "cell_id": f"{stem}-online-{policy.replace('_', '-')}-lambda{strength:.1f}",
                     "arm": "Online", "direction_policy": policy, "strength": strength,
                     "intervention_mode": "spectral"}, alias)
    require(len(cells) == 18 and len(display) == 21, "Phase 11 panel expansion differs")
    return {"schema": PANEL_SCHEMA, "phase": 11, "dataset": dataset,
        "scientific_contract": config["scientific_contract"], "dataset_recipe": copy.deepcopy(config["datasets"][dataset]),
        "target_gold_loaded": False, "cells": cells, "display_rows": display,
        "cell_count": 18, "independent_cell_count": 18, "display_row_count": 21, "alias_count": 3,
        "sample_count_per_cell": COUNTS[dataset], "configuration_sample_records": 18 * COUNTS[dataset],
        "k2_alias_admission": "local schedule check plus Gate B real prefill/decode equivalence",
        "closure": "frozen logical membership; not task input or outcome completeness"}


def build_all_panels(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    panels = [build_panel(dataset, config) for dataset in DATASETS]
    return {"schema": "loopscope.phase11.all_panels.v1", "panels": panels,
            "independent_cell_count": 54, "display_row_count": 63, "alias_count": 9,
            "configuration_sample_records": sum(p["configuration_sample_records"] for p in panels),
            "generation_sample_records": 18 * (COUNTS["mmlu_pro"] + COUNTS["gpqa_main"]),
            "target_gold_loaded": False}


def score_manifest(dataset: str, pool_path: Path | None = None, scope: str = "FORMAL_TEST",
                   config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    require(scope in SCOPES, "unknown Phase 11 acquisition scope")
    panel = build_panel(dataset, config)
    return {**panel, "schema": MANIFEST_SCHEMA, "scope": scope,
            "pool": str(pool_path) if pool_path is not None else None,
            "score_cells": copy.deepcopy(panel["cells"]), "score_cell_count": 18}


def validate_score_manifest(value: Mapping[str, Any], config: Mapping[str, Any] | None = None,
                            pool_path: Path | None = None, scope: str | None = None) -> dict[str, dict[str, Any]]:
    expected = score_manifest(value.get("dataset"), Path(value["pool"]) if value.get("pool") else None,
                              value.get("scope"), config)
    require(dict(value) == expected, "Phase 11 manifest differs from frozen 18-cell panel")
    if pool_path is not None:
        require(value.get("pool") == str(pool_path), "manifest input path differs")
    if scope is not None:
        require(value.get("scope") == scope, "manifest scope differs")
    return {row["cell_id"]: row for row in expected["cells"]}


def write_json_once(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
