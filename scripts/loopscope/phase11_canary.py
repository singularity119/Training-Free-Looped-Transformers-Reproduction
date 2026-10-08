#!/usr/bin/env python3
"""Freeze length-only canary membership or verify its raw singleton shards.

This does not close a full task panel and never opens gold. Record validation
runs internally; outputs contain identities and resource metadata only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tflt.loopscope.phase11_accuracy import load_pool
from tflt.loopscope.phase11_analysis import _attempt_records
from tflt.loopscope.phase11_panel import (
    COUNTS, DATASETS, MODEL, require, validate_score_manifest, write_json_once,
)

QUANTILES = (0.25, 0.5, 0.75, 1.0)
MEMBERSHIP_SCHEMA = "loopscope.phase11.canary_membership.v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_length(dataset, row):
    if dataset == "arc_challenge":
        candidates = row["tokenization"][MODEL]["candidates"]
        require(len(candidates) == len(row["choices"]), "ARC candidate lengths are incomplete")
        lengths = [candidate["input_length"] for candidate in candidates]
    else:
        lengths = [len(row["input_ids"])]
    require(lengths and all(type(n) is int and n > 0 for n in lengths),
            "canary selection requires actual positive token lengths")
    return max(lengths)


def select_members(dataset, rows):
    """Keep original indices; ties use the canonical index, never outputs."""
    require(len(rows) == COUNTS[dataset], "canary requires the complete frozen population")
    ordered = sorted((input_length(dataset, row), index) for index, row in enumerate(rows))
    samples = []
    for quantile in QUANTILES:
        length, index = ordered[math.floor((len(rows) - 1) * quantile)]
        samples.append({"quantile": quantile, "canonical_index": index,
                        "identity": rows[index]["identity"], "input_token_length": length})
    require(len({sample["canonical_index"] for sample in samples}) == 4,
            "canary quantiles must yield four distinct identities")
    return samples


def build_membership(manifests):
    require(set(manifests) == set(DATASETS), "canary requires exactly three frozen manifests")
    tasks = {}
    for dataset in DATASETS:
        path = Path(manifests[dataset])
        panel = read_json(path)
        validate_score_manifest(panel, scope="FORMAL_TEST")
        require(panel["dataset"] == dataset, "manifest task differs from requested task")
        pool = load_pool(Path(panel["pool"]), dataset, "FORMAL_TEST")
        tasks[dataset] = {"manifest": str(path), "pool": panel["pool"],
                          "population_count": len(pool["rows"]),
                          "samples": select_members(dataset, pool["rows"])}
    return {"schema_version": MEMBERSHIP_SCHEMA, "scope": "FORMAL_TEST",
            "target_gold_loaded": False, "selection_rule": "floor((N-1)*q); sort(input_length,canonical_index)",
            "quantiles": list(QUANTILES), "record_count": 216, "tasks": tasks}


def verify_canary(membership, worklist):
    require(membership.get("schema_version") == MEMBERSHIP_SCHEMA,
            "unexpected canary membership schema")
    tasks = membership["tasks"]
    require(build_membership({dataset: item["manifest"] for dataset, item in tasks.items()}) == membership,
            "canary membership differs from frozen length-only selection")
    expected, inputs = {}, {}
    for dataset in DATASETS:
        task = tasks[dataset]
        panel = read_json(task["manifest"])
        pool = load_pool(Path(task["pool"]), dataset, "FORMAL_TEST")
        inputs[dataset] = panel, pool
        for cell in panel["cells"]:
            for sample in task["samples"]:
                expected[(dataset, cell["cell_id"], sample["canonical_index"])] = sample
    jobs = worklist["jobs"]
    require(len(jobs) == len(expected) == 216, "worklist must contain all 216 canary singleton shards")
    completed, seen, commits, checked_manifests = [], set(), set(), set()
    for job in jobs:
        key = (job["dataset"], job["cell_id"], job["canonical_index"])
        require(key in expected and key not in seen, "worklist contains an unknown or duplicate canary shard")
        seen.add(key)
        dataset, cell_id, index = key
        sample = expected[key]
        panel, pool = inputs[dataset]
        root = Path(job["run_root"])
        summary = read_json(root / "summary.json")
        metadata = summary["metadata"]
        require(metadata["shard"] == {"index": index, "count": COUNTS[dataset],
                "rule": "canonical_index_modulo_count"} and metadata["indices"] == [index],
                "canary must retain the full-population index/N singleton shard")
        require(metadata["engineering_check"] is None and
                metadata.get("engineering_generate_steps") is None and
                metadata["target_gold_loaded"] is False and metadata["max_new_tokens"] == 2048,
                "formal canary cannot use synthetic acquisition controls")
        facts, records = _attempt_records(panel, pool, root, "FORMAL_TEST", checked_manifests)
        require(facts["cell_id"] == cell_id and facts["indices"] == [index] and len(records) == 1,
                "raw singleton does not belong to its worklist cell/index")
        require(records[0]["identity"] == sample["identity"] and summary["count"] == 1,
                "canary record identity/count differs from membership")
        commits.add(facts["source_commit"])
        telemetry = [json.loads(line) for line in (root / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        require(len(telemetry) == 1 and telemetry[0]["identity"] == sample["identity"] and
                telemetry[0]["canonical_index"] == index, "telemetry must match the raw singleton")
        item = telemetry[0]
        length_key = "max_input_tokens" if dataset == "arc_challenge" else "prefill_input_tokens"
        require(item[length_key] == sample["input_token_length"], "actual input length differs from frozen membership")
        env = read_json(root / "env.json")
        safe = {"dataset": dataset, "cell_id": cell_id, "identity": sample["identity"],
                "canonical_index": index, "population_count": COUNTS[dataset],
                "run_root": facts["run_root"], "source_commit": facts["source_commit"],
                "input_token_length": sample["input_token_length"], "gpu": env["gpu"],
                "total_memory_bytes": env["total_memory"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "question_elapsed_seconds": item["elapsed_seconds"],
                "peak_allocated_bytes": summary["peak_allocated_bytes"],
                "peak_reserved_bytes": summary["peak_reserved_bytes"]}
        if dataset != "arc_challenge":
            safe.update(generated_token_count=records[0]["generated_token_count"],
                        truncated=records[0]["truncated"], prefill_count=item["prefill_count"],
                        decode_count=item["decode_count"])
        for name in ("prefill_elapsed_seconds", "decode_elapsed_seconds"):
            if name in item:
                safe[name] = item[name]
        completed.append(safe)
    require(len(commits) == 1, "canary raw roots must share one source commit")
    return {"schema_version": "loopscope.phase11.canary_closure.v1",
            "status": "CANARY_CLOSED_GOLD_UNREAD", "scope": "FORMAL_TEST",
            "target_gold_loaded": False, "full_panel_closed": False,
            "source_commit": next(iter(commits)), "record_count": len(completed),
            "completed": sorted(completed, key=lambda item: (item["dataset"], item["cell_id"], item["canonical_index"]))}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    membership = commands.add_parser("membership")
    membership.add_argument("--manifest", action="append", required=True, metavar="DATASET=PATH")
    membership.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--membership", type=Path, required=True)
    verify.add_argument("--worklist", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    return result


def main(args):
    if args.command == "membership":
        manifests = {}
        for declaration in args.manifest:
            dataset, path = declaration.split("=", 1)
            require(dataset not in manifests, "duplicate manifest task")
            manifests[dataset] = path
        result = build_membership(manifests)
    else:
        result = verify_canary(read_json(args.membership), read_json(args.worklist))
    write_json_once(args.out, result)
    print(json.dumps({"status": result.get("status", "CANARY_MEMBERSHIP_FROZEN"),
                      "record_count": result["record_count"], "out": str(args.out),
                      "target_gold_loaded": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
