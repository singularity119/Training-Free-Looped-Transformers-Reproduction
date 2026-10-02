#!/usr/bin/env python3
"""Estimate remaining MMLU allocation from seven label-free A800 canaries."""
from __future__ import annotations
import argparse
import bisect
import json
import math
from pathlib import Path

from tflt.loopscope.phase10_accuracy import load_pool, require, write_json_once
from tflt.loopscope.phase10_panel import validate_score_manifest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def group_id(cell):
    return cell["cell_id"].rsplit("-lambda", 1)[0]


def length_envelope(measurements, length):
    """Use the larger adjacent measured cost, retaining repeated-length maxima."""
    maxima = {}
    for observed_length, seconds in measurements:
        require(seconds > 0 and math.isfinite(seconds), "invalid row timing")
        maxima[observed_length] = max(maxima.get(observed_length, 0), seconds)
    lengths = sorted(maxima)
    require(lengths and lengths[0] <= length <= lengths[-1], "canary misses length envelope")
    right = bisect.bisect_left(lengths, length)
    if lengths[right] == length:
        return maxima[length]
    return max(maxima[lengths[right - 1]], maxima[lengths[right]])


def estimate(manifest_path, pool_path, input_plan_path, roots, consumed, shard_size=1024):
    manifest = read_json(manifest_path)
    cells = validate_score_manifest(manifest, pool_path=pool_path, scope="FORMAL_TEST")
    require(manifest["dataset"] == "mmlu" and len(roots) == 7, "seven MMLU canaries required")
    require(consumed >= 0 and shard_size > 0, "invalid budget/shape")
    rows = load_pool(pool_path, "mmlu", "FORMAL_TEST")
    plan = read_json(input_plan_path)
    require(plan["pool"] == str(pool_path) and plan["target_gold_loaded"] is False, "input plan differs")
    lengths = {model: read_json(entry["lengths_path"]) for model, entry in plan["models"].items()}
    require(all([item["identity"] for item in items] == [row["identity"] for row in rows]
                for items in lengths.values()), "length identity order differs")
    observed, facts = {}, []
    for root in roots:
        root = Path(root)
        metadata, summary, env = (read_json(root / name) for name in
                                  ("command_args.json", "summary.json", "env.json"))
        cell = metadata["cell"]
        require(cell == cells[cell["cell_id"]] and cell["new_in_phase10"] and cell["strength"] == 0.1,
                "canary differs from frozen candidate")
        require(metadata["scope"] == "FORMAL_TEST" and metadata["target_gold_loaded"] is False and
                summary["status"] == "SCORES_COMPLETE" and summary["count"] == 64 and
                summary["target_gold_loaded"] is False and "A800" in env["gpu"], "canary scope/hardware differs")
        indices = metadata["indices"]
        ordered = sorted(range(len(rows)), key=lambda i: (lengths[cell["model"]][i]["context_length"], i))
        expected = sorted(ordered[round(j * (len(rows) - 1) / 63)] for j in range(64))
        require(indices == expected, "canary does not realize frozen quantile selection")
        telemetry = [json.loads(line) for line in (root / "telemetry.jsonl").read_text().splitlines()]
        require([item["canonical_index"] for item in telemetry] == indices, "timing identity order differs")
        require(all(item["identity"] == rows[item["canonical_index"]]["identity"] and
                    item["context_length"] == lengths[cell["model"]][item["canonical_index"]]["context_length"]
                    for item in telemetry), "measured context/identity differs from cost input")
        measurements = [(lengths[cell["model"]][item["canonical_index"]]["context_length"],
                         item["elapsed_seconds"]) for item in telemetry]
        gid = group_id(cell)
        require(gid not in observed, "duplicate canary strategy")
        observed[gid] = (cell, indices, measurements, summary["model_loading"]["elapsed_seconds"])
        facts.append({"cell_id": cell["cell_id"], "root": str(root), "source_commit": metadata["source_commit"],
                      "job_id": env["job_id"], "gpu": env["gpu"], "count": 64,
                      "loading_seconds": summary["model_loading"]["elapsed_seconds"],
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "peak_reserved_bytes": max(summary["peak_reserved_bytes"],
                                                  summary["model_loading"]["peak_reserved_bytes"])})
    require(len({fact["source_commit"] for fact in facts}) == 1, "canaries mix source revisions")
    estimates = []
    for cell in manifest["score_cells"]:
        reference, used, measurements, loading = observed[group_id(cell)]
        remaining = [i for i in range(len(rows)) if cell["cell_id"] != reference["cell_id"] or i not in used]
        count = len(remaining)
        seconds = sum(length_envelope(measurements, lengths[cell["model"]][i]["context_length"])
                      for i in remaining)
        jobs = math.ceil(count / shard_size)
        estimates.append({"cell_id": cell["cell_id"], "remaining_count": count,
                          "reference_cell_id": reference["cell_id"], "shard_count": jobs,
                          "estimated_seconds": seconds + jobs * loading,
                          "max_measured_row_seconds": max(t for _, t in measurements)})
    require(sum(item["remaining_count"] for item in estimates) == 828478 - 7 * 64,
            "remaining new record budget does not close")
    remaining_hours = sum(item["estimated_seconds"] for item in estimates) / 3600
    return {"schema": "loopscope.phase10.gate_c_cost_estimate.v1", "target_gold_loaded": False,
            "packing": 1, "batch_size": 1, "consumed_allocated_gpu_hours": consumed,
            "hard_budget_gpu_hours": 240, "remaining_estimated_gpu_hours": remaining_hours,
            "admission_gpu_hours": consumed + 1.2 * remaining_hours,
            "full_expansion_admitted": consumed + 1.2 * remaining_hours <= 240,
            "shard_size": shard_size, "canaries": facts, "cells": estimates,
            "method": "larger adjacent canary row-time by model context length plus per-shard measured loading; 1.2 reserve",
            "uncertainty": "one observation per quantile at lambda0.1; later strengths, system contention and repeated model loading may differ; update with label-free timing within hard budget"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, action="append", required=True)
    parser.add_argument("--consumed-gpu-hours", type=float, required=True)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = estimate(args.manifest, args.pool, args.input_plan, args.canary_root, args.consumed_gpu_hours, args.shard_size)
    write_json_once(args.output, value)
    print(json.dumps({key: value[key] for key in ("remaining_estimated_gpu_hours", "admission_gpu_hours",
                                                 "full_expansion_admitted", "target_gold_loaded")}))


if __name__ == "__main__":
    main()
