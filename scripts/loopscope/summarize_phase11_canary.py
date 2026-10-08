#!/usr/bin/env python3
"""Summarize safe canary metadata without opening raw records or gold."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from tflt.loopscope.phase11_panel import COUNTS, DATASETS, require, write_json_once


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values):
    values = list(values)
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)} if values else None


def aggregate(rows):
    result = {"record_count": len(rows)}
    fields = ("input_token_length", "question_elapsed_seconds", "prefill_elapsed_seconds", "decode_elapsed_seconds",
              "scoring_elapsed_seconds", "generated_token_count", "peak_allocated_bytes", "peak_reserved_bytes")
    for field in fields:
        result[field] = stats(row[field] for row in rows if field in row)
    generated = [row for row in rows if "truncated" in row]
    result["truncated_rate"] = sum(row["truncated"] for row in generated) / len(generated) if generated else None
    return result


def summarize(closure, ledger):
    require(closure.get("status") == "CANARY_CLOSED_GOLD_UNREAD" and closure.get("record_count") == 216 and
            closure.get("full_panel_closed") is False and closure.get("target_gold_loaded") is False,
            "summary requires the 216-record gold-unread canary closure")
    completed, jobs = closure["completed"], ledger["jobs"]
    require(len(completed) == 216 and jobs, "canary closure or parent ledger is empty")
    by_job = {str(job["job_id"]): job for job in jobs}
    require(len(by_job) == len(jobs), "ledger must contain each parent allocation once")
    require(all(job.get("state") == "COMPLETED" and job.get("exit_code") == "0:0" and
                "elapsed_allocated_seconds" in job for job in jobs), "terminal successful parent ledger required")
    device_peaks, allocation, batches_ok = [], [], True
    for job in jobs:
        root = Path(job["batch_root"])
        batch = read_json(root / "batch_summary.json")
        batches_ok &= (batch["status"] == "BATCH_COMPLETE" and batch["completed_attempts"] == batch["expected_attempts"]
                       and all(a["exit_code"] == 0 for a in batch["attempts"]))
        samples, uuids, errors = [], set(), 0
        for sample in read_lines(root / "device_memory.jsonl"):
            if sample["exit_code"] != 0:
                errors += 1
                continue
            devices = list(csv.reader(sample["devices_csv"].splitlines()))
            require(len(devices) == 1, "device samples must identify one allocated GPU per parent job")
            uuid, name, used, total = [value.strip() for value in devices[0]]
            uuids.add(uuid)
            samples.append((float(used), float(total), name))
        require(samples and len(uuids) == 1, "parent job lacks a single allocated UUID memory trace")
        device_peaks.append({"job_id": str(job["job_id"]), "class": job["class"], "gpu_uuid": next(iter(uuids)),
                             "gpu": samples[0][2], "peak_used_mib": max(s[0] for s in samples),
                             "total_mib": samples[0][1], "sampling_errors": errors})
        allocation.append({"job_id": str(job["job_id"]), "class": job["class"], "packing": job["packing"],
                           "allocation_gpuh": job["elapsed_allocated_seconds"] * job["gpus"] / 3600})
    rows, groups, tasks = [], defaultdict(list), defaultdict(list)
    for item in completed:
        root = Path(item["run_root"])
        env, summary = read_json(root / "env.json"), read_json(root / "summary.json")
        telemetry = read_lines(root / "telemetry.jsonl")
        require(len(telemetry) == 1 and summary["status"] == "RAW_COMPLETE" and
                summary["target_gold_loaded"] is False, "completed root metadata differs from canary closure")
        timing = telemetry[0]
        require(timing["identity"] == item["identity"] and timing["canonical_index"] == item["canonical_index"] and
                summary["metadata"]["source_commit"] == closure["source_commit"], "metadata identity/source differs")
        job = by_job[str(env["job_id"])]
        row = {key: item[key] for key in ("dataset", "cell_id", "identity", "canonical_index", "population_count",
               "run_root", "source_commit", "input_token_length", "peak_allocated_bytes", "peak_reserved_bytes")}
        row.update(job_id=str(env["job_id"]), packing=job["packing"], question_elapsed_seconds=timing["elapsed_seconds"])
        for key in ("prefill_elapsed_seconds", "decode_elapsed_seconds", "scoring_elapsed_seconds"):
            if key in timing:
                row[key] = timing[key]
        if item["dataset"] != "arc_challenge":
            row.update({key: item[key] for key in ("generated_token_count", "truncated", "decode_count")})
        rows.append(row)
        groups[(row["dataset"], row["cell_id"])].append(row)
        tasks[row["dataset"]].append(row)
    require(set(tasks) == set(DATASETS) and len(groups) == 54 and all(len(group) == 4 for group in groups.values()),
            "canary summary requires 18 cells and four identities per cell for each task")
    cells, totals, scenarios = [], defaultdict(float), {str(t): defaultdict(float) for t in (256, 512, 2048)}
    for (dataset, cell_id), group in sorted(groups.items()):
        require(all(row["population_count"] == COUNTS[dataset] for row in group), "full population count differs")
        cost = stats(row["question_elapsed_seconds"] / row["packing"] * COUNTS[dataset] / 3600 for row in group)
        scenario_cost = {}
        for tokens in (256, 512, 2048):
            estimates = []
            for row in group:
                if dataset == "arc_challenge":
                    seconds = row["question_elapsed_seconds"]
                elif row.get("decode_count", 0) > 0 and "prefill_elapsed_seconds" in row and "decode_elapsed_seconds" in row:
                    prefill, decode = row["prefill_elapsed_seconds"], row["decode_elapsed_seconds"]
                    overhead = max(0.0, row["question_elapsed_seconds"] - prefill - decode)
                    seconds = prefill + decode / row["decode_count"] * (tokens - 1) + overhead
                else:
                    break
                estimates.append(seconds / row["packing"] * COUNTS[dataset] / 3600)
            scenario_cost[str(tokens)] = stats(estimates) if len(estimates) == 4 else None
            if scenario_cost[str(tokens)] is not None:
                scenarios[str(tokens)][dataset] += scenario_cost[str(tokens)]["mean"]
        cells.append({"dataset": dataset, "cell_id": cell_id, **aggregate(group),
                      "full_population_question_gpuh_proxy": cost, "output_length_gpuh_scenarios": scenario_cost})
        for key, value in cost.items():
            totals[key] += value
    scenario_totals = {t: {dataset: v.get(dataset) if all(c["output_length_gpuh_scenarios"][t] is not None
        for c in cells if c["dataset"] == dataset) else None for dataset in DATASETS} for t, v in scenarios.items()}
    return {"schema_version": "loopscope.phase11.canary_resource_summary.v1", "status": "CANARY_RESOURCES_SUMMARIZED",
            "record_count": 216, "source_commit": closure["source_commit"], "target_gold_loaded": False,
            "full_panel_closed": False, "datasets": {dataset: aggregate(group) for dataset, group in tasks.items()},
            "cells": cells, "allocation_jobs": allocation, "allocation_gpuh": sum(j["allocation_gpuh"] for j in allocation),
            "sampled_device_memory": device_peaks, "oom_count": 0 if batches_ok else None,
            "full_54_configuration_question_gpuh_proxy": dict(totals),
            "output_length_scenario_mean_gpuh_by_dataset": scenario_totals,
            "reusable_completed": rows,
            "limitations": ["Four length-selected identities per task are biased planning proxies, not population estimates or confidence intervals.",
                "Each measured attempt is divided by its own parent-job packing; this does not establish a packing speedup.",
                "Question-time costs exclude model startup, queue time and batch tail; allocation GPUh includes all ledger parent jobs.",
                "Length scenarios use observed decode seconds per step and nonnegative question overhead; ARC keeps complete candidate time.",
                "Scenario totals require every cell estimate; sampled device peaks are distinct from process allocator peaks."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("closure", "ledger", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(read_json(args.closure), read_json(args.ledger))
    write_json_once(args.out, result)
    print(json.dumps({"status": result["status"], "record_count": 216, "out": str(args.out), "target_gold_loaded": False}))


if __name__ == "__main__":
    main()
