#!/usr/bin/env python3
"""Verify Phase 9 new-arm score shards without reading labels or outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from tflt.loopscope.phase8_test_pool import validate_test_pool
from tflt.loopscope.phase9_accuracy import (
    EXPECTED_TEST_COUNT,
    load_config,
    validate_canary_bundle,
    validate_score_manifest,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_indices(
    scope: str,
    selection: str,
    cell: Mapping[str, Any],
    pool: Mapping[str, Any],
    config: Mapping[str, Any],
    canary_path: Path | None,
) -> List[int]:
    if scope == "PREFLIGHT_ONLY":
        require(selection == "preflight" and canary_path is None, "invalid Phase 9 preflight selection")
        rows = pool["rows"]
        chosen = {0, 1}
        ranked = sorted(
            range(len(rows)),
            key=lambda index: (-rows[index]["prompt_token_lengths"][cell["model"]], index),
        )
        for index in ranked:
            chosen.add(index)
            if len(chosen) >= 6:
                break
        return sorted(chosen)
    require(scope == "FORMAL_TEST" and selection in ("canary", "remaining"),
            "invalid Phase 9 formal selection")
    require(canary_path is not None, "formal Phase 9 verification requires canary indices")
    bundle = read_json(canary_path)
    by_model = validate_canary_bundle(bundle, pool, config)
    canary = set(by_model[cell["model"]])
    if selection == "canary":
        return sorted(canary)
    return [index for index in range(EXPECTED_TEST_COUNT) if index not in canary]


def verify(
    cell_roots: Sequence[Path],
    manifest_path: Path,
    pool_path: Path,
    scope: str,
    selection: str,
    canary_path: Path | None,
    full_panel: bool = False,
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
        require(summary.get("status") == "SCORES_COMPLETE", f"incomplete Phase 9 score shard: {root}")
        require(summary.get("metadata") == metadata, "Phase 9 summary/command metadata mismatch")
        require(summary.get("target_gold_loaded") is False and metadata.get("target_gold_loaded") is False,
                "Phase 9 score shard crossed the gold boundary")
        cell = metadata.get("cell")
        cell_id = cell.get("cell_id") if isinstance(cell, Mapping) else None
        require(cell_id in cells and cell == cells[cell_id], "Phase 9 shard configuration differs from manifest")
        root_selection = metadata.get("selection") if selection == "combined" else selection
        require(metadata.get("scope") == scope and root_selection in ("preflight", "canary", "remaining"),
                "Phase 9 shard scope or selection differs from requested closure")
        if selection != "combined":
            require(metadata.get("selection") == selection,
                    "Phase 9 shard selection differs from requested closure")
        elif scope == "FORMAL_TEST":
            require(root_selection in ("canary", "remaining"),
                    "combined formal closure requires canary and remaining shards")
        require(metadata.get("pool") == str(pool_path) and metadata.get("manifest") == str(manifest_path),
                "Phase 9 shard source paths differ from closure inputs")
        commit = metadata.get("source_commit")
        require(isinstance(commit, str) and bool(commit), "Phase 9 shard is missing source commit")
        commits.add(commit)
        require(env.get("source_commit") == commit and env.get("model_revision") == cell["revision"],
                "Phase 9 runtime revision/source differs from frozen cell")
        require(env.get("model_dtype") == "torch." + cell["dtype"],
                "Phase 9 runtime dtype differs from frozen cell")
        expected = expected_indices(scope, root_selection, cell, pool, config, canary_path)
        observed_indices = metadata.get("indices")
        require(observed_indices == expected, "Phase 9 shard identity selection differs from frozen canary/complement")
        require(summary.get("count") == len(expected), "Phase 9 score count differs from selected identities")
        runtime = summary.get("positions_and_runtime", {})
        k = int(cell["k"])
        callback = runtime.get("callback_by_t")
        applied = runtime.get("applied_by_t")
        require(isinstance(callback, Mapping) and isinstance(applied, Mapping), "missing Phase 9 callback counts")
        require(set(callback) == set(applied) == {str(t) for t in range(k)},
                "Phase 9 callback timestep set differs from frozen K")
        require(len(set(callback.values())) == 1 and callback["0"] == len(expected),
                "Phase 9 callback count is not one K-step trajectory per identity")
        require(applied["0"] == 0 and all(applied[str(t)] == len(expected) for t in range(1, k)),
                "Phase 9 intervention timing/count differs from frozen t>=1 rule")
        require(runtime.get("direction_fit_count") == len(expected),
                "Phase 9 did not fit exactly one prompt-local t0 direction per identity")
        require(runtime.get("nonanswer_mutation_count") == 0 and runtime.get("t0_mutation_count") == 0,
                "Phase 9 score runtime recorded an invalid mutation")
        seen = covered.setdefault(cell_id, set())
        require(not seen.intersection(expected), "overlapping or duplicate Phase 9 shard identities")
        count = 0
        with (root / "scores.jsonl").open(encoding="utf-8") as handle:
            for count, line in enumerate(handle, 1):
                require(count <= len(expected), "extra Phase 9 score rows")
                row = json.loads(line)
                require(set(row) == {"identity", "subject", "doc_index", "scores"},
                        "Phase 9 score row contains fields outside gold-free raw scores")
                target = canonical[expected[count - 1]]
                require(all(row[key] == target[key] for key in ("identity", "subject", "doc_index")),
                        "Phase 9 score identity/order differs from canonical selection")
                scores = row["scores"]
                require(isinstance(scores, list) and len(scores) == 4 and
                        all(type(value) in (int, float) and math.isfinite(value) for value in scores),
                        "Phase 9 score row requires four finite raw choice scores")
        require(count == len(expected), "Phase 9 score shard is incomplete")
        seen.update(expected)
        records.append({
            "root": str(root),
            "cell_id": cell_id,
            "scope": scope,
            "selection": root_selection,
            "count": count,
            "first_index": expected[0],
            "last_index": expected[-1],
            "samples_per_second": summary.get("samples_per_second"),
            "peak_reserved_bytes": summary.get("peak_reserved_bytes"),
            "job_id": env.get("job_id"),
            "partition": env.get("partition"),
            "gpu": env.get("gpu"),
        })

    require(records, "no Phase 9 score shards supplied")
    if full_panel:
        require(scope == "FORMAL_TEST", "full Phase 9 closure requires FORMAL_TEST shards")
        require(selection == "combined", "full Phase 9 closure requires combined canary and remaining shards")
        require(set(covered) == set(cells), "full Phase 9 closure requires all 18 new cells")
        require(all(indices == set(range(EXPECTED_TEST_COUNT)) for indices in covered.values()),
                "full Phase 9 closure requires every new cell's 14042 canonical identities")
    return {
        "schema": "loopscope.phase9.score_verification.v1",
        "status": "FULL_NEW_SCORE_PANEL_CLOSED" if full_panel else "SCORE_SHARDS_CLOSED",
        "target_gold_loaded": False,
        "manifest": str(manifest_path),
        "pool": str(pool_path),
        "scope": scope,
        "selection": selection if not full_panel else "canary_plus_remaining",
        "cell_roots": [record["root"] for record in records],
        "cell_count": len(covered),
        "shard_count": len(records),
        "sample_count": sum(len(indices) for indices in covered.values()),
        "counts_by_cell": {key: len(value) for key, value in covered.items()},
        "subject_count": len({row["subject"] for row in canonical}),
        "source_commits": sorted(commits),
        "shards": records,
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
    value = verify(
        args.cell_roots,
        args.manifest,
        args.pool,
        args.scope,
        args.selection,
        args.canary_indices,
        args.full_panel,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": value["status"],
        "cell_count": value["cell_count"],
        "shard_count": value["shard_count"],
        "sample_count": value["sample_count"],
        "target_gold_loaded": value["target_gold_loaded"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
