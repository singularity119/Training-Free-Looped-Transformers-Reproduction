#!/usr/bin/env python3
"""Verify the 29 Phase 9 historical raw-score cells without reading gold."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from tflt.loopscope.phase8_test_pool import validate_test_pool
from tflt.loopscope.phase9_accuracy import (
    EXPECTED_TEST_COUNT,
    HISTORICAL_ARMS,
    NATIVE_ARM,
    load_config,
    logical_panel,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_old_cell(old: Mapping[str, Any], target: Mapping[str, Any], source_ids: Sequence[str]) -> None:
    require(old.get("cell_id") in set(source_ids), "historical source cell is not declared by the map")
    for key in ("model", "revision", "dtype", "cache_strategy", "window", "k"):
        require(old.get(key) == target.get(key), f"historical {key} differs from Phase 9 target")
    if target["arm"] == NATIVE_ARM:
        require(old.get("arm") == NATIVE_ARM, "historical Native root is not Native")
        return
    require(target["arm"] in HISTORICAL_ARMS, "unexpected historical target arm")
    if target["arm"] == "Loop":
        require(old.get("arm") == "Loop", "historical intervention arm differs from Phase 9 target")
        return
    require(old.get("arm") == "Spectral", "historical intervention arm differs from Phase 9 target")
    if target["arm"] == "Shared-t0":
        require(old.get("fit_t") == 0, "Shared-t0 root does not carry fit_t=0")
    else:
        require(old.get("fit_t") in (None, 1), "Shared-t1 root carries an unexpected fit_t")
    require(isinstance(old.get("basis_path"), str) and Path(old["basis_path"]).is_file(),
            "historical spectral root is missing its frozen basis file")


def verify_root(
    root: Path,
    target: Mapping[str, Any],
    source_ids: Sequence[str],
    pool_path: Path,
    canonical: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    metadata = read_json(root / "command_args.json")
    summary = read_json(root / "summary.json")
    env = read_json(root / "env.json")
    require(summary.get("status") == "SCORES_COMPLETE", f"incomplete historical root: {root}")
    require(summary.get("metadata") == metadata, "historical summary/command metadata mismatch")
    require(summary.get("target_gold_loaded") is False, "historical root crossed the gold boundary")
    require(metadata.get("scope") == "FORMAL_TEST", "historical root is not a formal score shard")
    require(metadata.get("pool") == str(pool_path), "historical root uses a different canonical pool")
    old = metadata.get("cell")
    require(isinstance(old, Mapping), "historical root has no cell metadata")
    validate_old_cell(old, target, source_ids)
    require(env.get("model_revision") == target["revision"], "historical model revision differs")
    require(env.get("model_dtype") == "torch." + target["dtype"], "historical model dtype differs")
    start = metadata.get("start")
    end = metadata.get("end")
    require(type(start) is int and type(end) is int and 0 <= start < end <= EXPECTED_TEST_COUNT,
            "historical shard range is outside the canonical test pool")
    expected_count = end - start
    require(summary.get("count") == expected_count, "historical shard count differs from its frozen range")
    covered = []
    count = 0
    with (root / "scores.jsonl").open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            require(count <= expected_count, "extra historical raw-score row")
            row = json.loads(line)
            require(set(row) == {"identity", "subject", "doc_index", "scores"},
                    "historical score row contains target/outcome fields")
            target_row = canonical[start + count - 1]
            require(all(row[key] == target_row[key] for key in ("identity", "subject", "doc_index")),
                    "historical raw-score identity/order differs from canonical pool")
            scores = row["scores"]
            require(isinstance(scores, list) and len(scores) == 4 and
                    all(type(value) in (int, float) and math.isfinite(value) for value in scores),
                    "historical score row requires four finite raw choice scores")
            covered.append(start + count - 1)
    require(count == expected_count, "historical score shard is incomplete")
    return {
        "root": str(root),
        "source_cell_id": old["cell_id"],
        "start": start,
        "end": end,
        "count": count,
        "subject_count": len({canonical[index]["subject"] for index in covered}),
        "job_id": env.get("job_id"),
        "partition": env.get("partition"),
        "gpu": env.get("gpu"),
    }


def verify(mapping_path: Path, pool_path: Path) -> Dict[str, Any]:
    config = load_config()
    mapping = read_json(mapping_path)
    require(mapping.get("schema") == "loopscope.phase9.history_map.v1", "unexpected Phase 9 history map schema")
    require(mapping.get("target_gold_loaded") is False, "history map crossed the gold boundary")
    pool = read_json(pool_path)
    validate_test_pool(pool)
    canonical = pool["rows"]
    panel = {cell["cell_id"]: cell for cell in logical_panel(config)}
    expected_ids = {cell_id for cell_id, cell in panel.items() if not cell["new_in_phase9"]}
    entries = mapping.get("cells", [])
    require(mapping.get("cell_count") == len(expected_ids) and len(entries) == len(expected_ids),
            "historical map does not contain exactly 29 cells")
    seen_targets = set()
    used_roots = set()
    verified_cells = []
    for entry in entries:
        cell_id = entry.get("cell_id")
        require(cell_id in expected_ids and cell_id not in seen_targets, "history map target cell set differs from contract")
        seen_targets.add(cell_id)
        target = panel[cell_id]
        source_ids = list(entry.get("source_cell_ids", []))
        roots = [Path(value) for value in entry.get("roots", [])]
        require(source_ids and roots, f"historical map entry is empty: {cell_id}")
        require(not used_roots.intersection(roots), "historical root is reused by multiple target cells")
        used_roots.update(roots)
        coverage = set()
        root_records = []
        for root in roots:
            record = verify_root(root, target, source_ids, pool_path, canonical)
            expected_range = set(range(record["start"], record["end"]))
            require(not coverage.intersection(expected_range), "historical shards overlap within a cell")
            coverage.update(expected_range)
            root_records.append(record)
        require(coverage == set(range(EXPECTED_TEST_COUNT)),
                f"historical cell does not cover all canonical identities: {cell_id}")
        verified_cells.append({
            "cell_id": cell_id,
            "source_cell_ids": sorted(source_ids),
            "root_count": len(root_records),
            "sample_count": len(coverage),
            "subject_count": len({canonical[index]["subject"] for index in coverage}),
            "roots": root_records,
        })
    require(seen_targets == expected_ids, "historical map omitted a Phase 9 historical cell")
    require(len({row["subject"] for row in canonical}) == 57, "canonical pool subject count differs from contract")
    return {
        "schema": "loopscope.phase9.history_verification.v1",
        "status": "HISTORICAL_29_CELLS_VERIFIED",
        "target_gold_loaded": False,
        "mapping": str(mapping_path),
        "pool": str(pool_path),
        "cell_count": len(verified_cells),
        "sample_count": len(verified_cells) * EXPECTED_TEST_COUNT,
        "subject_count": 57,
        "cells": sorted(verified_cells, key=lambda value: value["cell_id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.mapping, args.pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": value["status"],
        "cell_count": value["cell_count"],
        "sample_count": value["sample_count"],
        "target_gold_loaded": value["target_gold_loaded"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
