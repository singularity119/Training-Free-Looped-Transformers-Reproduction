#!/usr/bin/env python3
"""Verify Gate E Lag1 score shards without reading labels or outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from tflt.loopscope.phase8_test_pool import validate_test_pool
from tflt.loopscope.phase9_gate_e_accuracy import (
    EXPECTED_NEW_COUNT,
    EXPECTED_TEST_COUNT,
    load_config,
    validate_canary_bundle,
    validate_score_manifest,
)
from tflt.loopscope.phase9_gate_e_spectral_core import (
    DIRECTION_POLICIES,
    INTERVENTION_MODES,
    direction_schedule,
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_indices(scope: str, selection: str, cell: Mapping[str, Any], pool: Mapping[str, Any], config: Mapping[str, Any], canary_path: Path | None) -> List[int]:
    if scope == "PREFLIGHT_ONLY":
        require(selection == "preflight" and canary_path is None, "invalid Gate E preflight selection")
        chosen = {0, 1}
        ranked = sorted(range(len(pool["rows"])), key=lambda index: (-pool["rows"][index]["prompt_token_lengths"][cell["model"]], index))
        for index in ranked:
            chosen.add(index)
            if len(chosen) >= 6:
                break
        return sorted(chosen)
    require(scope == "FORMAL_TEST" and selection in ("canary", "remaining"), "invalid Gate E formal selection")
    require(canary_path is not None, "Gate E formal verification requires canary indices")
    by_model = validate_canary_bundle(read_json(canary_path), pool, config)
    canary = set(by_model[cell["model"]])
    return sorted(canary) if selection == "canary" else [index for index in range(EXPECTED_TEST_COUNT) if index not in canary]


def verify(
    cell_roots: Sequence[Path], manifest_path: Path, pool_path: Path, scope: str,
    selection: str, canary_path: Path | None, full_panel: bool = False,
) -> Dict[str, Any]:
    config = load_config()
    manifest = read_json(manifest_path)
    cells = validate_score_manifest(manifest, config, pool_path, scope)
    pool = read_json(pool_path)
    validate_test_pool(pool)
    canonical = pool["rows"]
    covered: Dict[str, set[int]] = {}
    records = []
    commits = set()
    for root in cell_roots:
        root = Path(root)
        metadata = read_json(root / "command_args.json")
        summary = read_json(root / "summary.json")
        env = read_json(root / "env.json")
        require(summary.get("status") == "SCORES_COMPLETE", f"incomplete Gate E score shard: {root}")
        require(summary.get("metadata") == metadata, "Gate E summary/command metadata mismatch")
        require(summary.get("target_gold_loaded") is False and metadata.get("target_gold_loaded") is False,
                "Gate E score shard crossed the gold boundary")
        cell = metadata.get("cell")
        cell_id = cell.get("cell_id") if isinstance(cell, Mapping) else None
        require(cell_id in cells and cell == cells[cell_id], "Gate E shard configuration differs from manifest")
        direction_policy = metadata.get("direction_policy")
        intervention_mode = metadata.get("intervention_mode")
        require(direction_policy in DIRECTION_POLICIES and intervention_mode in INTERVENTION_MODES,
                "Gate E run metadata is missing an explicit policy or mode")
        require(intervention_mode == cell["intervention_mode"],
                "Gate E intervention mode differs from its frozen cell")
        if scope == "FORMAL_TEST":
            require(direction_policy == cell["direction_policy"] == "lag1",
                    "formal Gate E score shards must use direction_policy=lag1")
        else:
            require(direction_policy in ("fixed_t0", "lag1"),
                    "Gate E preflight direction policy is unsupported")
        require(summary.get("direction_policy") == direction_policy and
                summary.get("intervention_mode") == intervention_mode,
                "Gate E summary does not repeat the explicit policy and mode")
        root_selection = metadata.get("selection")
        require(metadata.get("scope") == scope and root_selection in ("preflight", "canary", "remaining"),
                "Gate E shard scope or selection differs from requested closure")
        if selection != "combined":
            require(root_selection == selection, "Gate E shard selection differs from requested closure")
        elif scope == "FORMAL_TEST":
            require(root_selection in ("canary", "remaining"), "combined Gate E closure requires canary and remaining shards")
        require(metadata.get("pool") == str(pool_path) and metadata.get("manifest") == str(manifest_path),
                "Gate E shard source paths differ from closure inputs")
        commit = metadata.get("source_commit")
        require(isinstance(commit, str) and bool(commit), "Gate E shard is missing source commit")
        commits.add(commit)
        require(env.get("source_commit") == commit and env.get("model_revision") == cell["revision"] and
                env.get("direction_policy") == direction_policy and
                env.get("intervention_mode") == intervention_mode,
                "Gate E runtime revision/source differs from score metadata")
        require(env.get("model_dtype") == "torch." + cell["dtype"], "Gate E runtime dtype differs from frozen cell")
        expected = expected_indices(scope, root_selection, cell, pool, config, canary_path)
        require(metadata.get("indices") == expected, "Gate E shard identity selection differs from frozen canary/complement")
        require(summary.get("count") == len(expected), "Gate E score count differs from selected identities")
        runtime = summary.get("positions_and_runtime", {})
        k = int(cell["k"])
        callback = runtime.get("callback_by_t")
        applied = runtime.get("applied_by_t")
        fit_by_t = runtime.get("direction_fit_by_t")
        used_by_t = runtime.get("direction_used_fit_by_t")
        require(isinstance(callback, Mapping) and isinstance(applied, Mapping) and isinstance(fit_by_t, Mapping),
                "missing Gate E callback/fit counts")
        require(isinstance(used_by_t, Mapping), "missing Gate E per-round direction-source counts")
        require(set(callback) == set(applied) == {str(t) for t in range(k)}, "Gate E callback timestep set differs")
        require(set(fit_by_t) == {str(t) for t in range(k - 1)}, "Gate E fit timestep set differs")
        require(set(used_by_t) == {str(t) for t in range(k)}, "Gate E direction-source timestep set differs")
        require(all(callback[str(t)] == len(expected) for t in range(k)), "Gate E callback count differs from K")
        require(applied["0"] == 0 and all(applied[str(t)] == len(expected) for t in range(1, k)),
                "Gate E intervention timing/count differs")
        require(used_by_t["0"] == 0 and
                all(used_by_t[str(t)] == len(expected) for t in range(1, k)),
                "Gate E per-round direction-use count differs")
        expected_fit_times = set(range(k - 1)) if direction_policy == "lag1" else {0}
        require(all(fit_by_t[str(t)] == (len(expected) if t in expected_fit_times else 0)
                    for t in range(k - 1)), "Gate E direction-fit round counts differ from policy")
        require(runtime.get("direction_fit_count") == len(expected) * len(expected_fit_times),
                "Gate E direction fit count differs from the explicit policy")
        schedule = summary.get("round_schedule")
        require(isinstance(schedule, list) and len(schedule) == k,
                "Gate E summary is missing bounded per-round policy logs")
        for t, (observed, expected_schedule) in enumerate(
                zip(schedule, direction_schedule(direction_policy, k))):
            expected_t, used_t, fit_t = expected_schedule
            require(expected_t == t and observed.get("t") == t and
                    observed.get("applied_t") == (t if t > 0 else None) and
                    observed.get("direction_used_fit_t") == used_t and
                    observed.get("direction_fit_t") == fit_t,
                    "Gate E bounded per-round policy log differs from the declared schedule")
            require(observed.get("applied_count") == applied[str(t)] and
                    observed.get("direction_used_fit_count") == used_by_t[str(t)] and
                    observed.get("direction_fit_count") == (fit_by_t.get(str(fit_t), 0) if fit_t is not None else 0),
                    "Gate E bounded per-round counts differ from runtime counters")
        require(runtime.get("nonanswer_mutation_count") == 0 and runtime.get("t0_mutation_count") == 0,
                "Gate E runtime recorded an invalid mutation")
        seen = covered.setdefault(cell_id, set())
        require(not seen.intersection(expected), "overlapping or duplicate Gate E shard identities")
        count = 0
        with (root / "scores.jsonl").open(encoding="utf-8") as handle:
            for count, line in enumerate(handle, 1):
                require(count <= len(expected), "extra Gate E score rows")
                row = json.loads(line)
                require(set(row) == {"identity", "subject", "doc_index", "scores"},
                        "Gate E score row contains fields outside gold-free raw scores")
                target = canonical[expected[count - 1]]
                require(all(row[key] == target[key] for key in ("identity", "subject", "doc_index")),
                        "Gate E score identity/order differs from canonical selection")
                scores = row["scores"]
                require(isinstance(scores, list) and len(scores) == 4 and
                        all(type(value) in (int, float) and math.isfinite(value) for value in scores),
                        "Gate E score row requires four finite raw choice scores")
        require(count == len(expected), "Gate E score shard is incomplete")
        seen.update(expected)
        records.append({
            "root": str(root), "cell_id": cell_id, "scope": scope, "selection": selection,
            "direction_policy": direction_policy, "intervention_mode": intervention_mode,
            "count": count, "first_index": expected[0], "last_index": expected[-1],
            "samples_per_second": summary.get("samples_per_second"),
            "peak_reserved_bytes": summary.get("peak_reserved_bytes"),
            "job_id": env.get("job_id"), "partition": env.get("partition"), "gpu": env.get("gpu"),
        })
    require(records, "no Gate E score shards supplied")
    if full_panel:
        require(scope == "FORMAL_TEST" and selection == "combined", "full Gate E closure requires combined formal shards")
        require(set(covered) == set(cells), "full Gate E closure requires all 12 cells")
        require(all(indices == set(range(EXPECTED_TEST_COUNT)) for indices in covered.values()),
                "full Gate E closure requires every cell's 14042 identities")
    return {
        "schema": "loopscope.phase9.gate_e.score_verification.v2",
        "status": "FULL_GATE_E_SCORE_PANEL_CLOSED" if full_panel else "GATE_E_SCORE_SHARDS_CLOSED",
        "target_gold_loaded": False, "manifest": str(manifest_path), "pool": str(pool_path),
        "direction_policies": manifest["direction_policies"],
        "intervention_modes": manifest["intervention_modes"],
        "scope": scope, "selection": selection if not full_panel else "canary_plus_remaining",
        "cell_roots": [record["root"] for record in records], "cell_count": len(covered),
        "shard_count": len(records), "sample_count": sum(len(indices) for indices in covered.values()),
        "counts_by_cell": {key: len(value) for key, value in covered.items()},
        "subject_count": len({row["subject"] for row in canonical}),
        "source_commits": sorted(commits), "shards": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--scope", choices=("PREFLIGHT_ONLY", "FORMAL_TEST"), required=True)
    parser.add_argument("--selection", choices=("preflight", "canary", "remaining", "combined"), required=True)
    parser.add_argument("--canary-indices", type=Path)
    parser.add_argument("--cell-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--full-panel", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.cell_roots, args.manifest, args.pool, args.scope, args.selection,
                   args.canary_indices, args.full_panel)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value[key] for key in (
        "status", "cell_count", "shard_count", "sample_count", "target_gold_loaded"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
