#!/usr/bin/env python3
"""Freeze D7 input-only membership and first-batch-before-Loop scheduling."""
import argparse
import copy
import json
import math
from pathlib import Path

from phase11_d7 import selected_cells, SEED
from phase11_d7_batch import (FIRST_KEYS, LOOP_KEYS, read, save, validate_task,
                             validate_first_closure, validate_plan_barrier)

POPULATION = 12032


def canary_indices(pool):
    """Select only by prompt token length, with canonical-index tie breaking."""
    rows = pool["rows"]
    if len(rows) != POPULATION or pool.get("target_gold_loaded") is not False:
        raise ValueError("D7 canary requires the full label-free 12032-row pool")
    order = sorted(range(POPULATION), key=lambda i: (len(rows[i]["input_ids"]), i))
    return sorted(order[math.floor((POPULATION - 1) * q)] for q in (.25, .5, .75, 1))


def chunks(indices, size):
    if type(size) is not int or size < 64:
        raise ValueError("D7 formal children require a shard size of at least 64")
    parts = [indices[i:i + size] for i in range(0, len(indices), size)]
    if len(parts) > 1 and len(parts[-1]) < 64:
        parts[-2].extend(parts.pop())
    return parts


def cell_specs(source, commit, pool, keys):
    frozen = selected_cells()
    return [{"cell_key": key, "cell_id": frozen[key]["cell_id"], "cell": frozen[key],
             "prompt_mode": "cot", "damping_scope": frozen[key]["damping_scope"],
             "population_count": POPULATION, "seed": SEED, "input_seed": 20261008,
             "synthetic": False, "engineering": False, "source_commit": commit,
             "producer_source": source, "pool": str(pool), "raw_roots": []} for key in keys]


def add_job(cell, indices, root):
    job = {k: v for k, v in cell.items() if k not in ("cell", "raw_roots")}
    job.update(indices=indices, run_root=root)
    cell["raw_roots"].append({"run_root": root, "records_path": root + "/records.jsonl",
                             "telemetry_path": root + "/telemetry.jsonl", "indices": indices})
    return job


def validate_canary(plan, cells):
    """Read completion metadata only; test generations and labels stay unread."""
    if plan.get("stage") != "canary" or plan.get("record_count") != 12:
        raise ValueError("D7 first stage requires the frozen 12-record canary")
    previous = {c["cell_key"]: c for c in plan["cells"]}
    if set(previous) != set(FIRST_KEYS):
        raise ValueError("D7 canary membership differs")
    indices = plan["canary_indices"]
    for cell in cells:
        old = previous[cell["cell_key"]]
        for field in ("cell_id", "source_commit", "producer_source", "pool", "seed",
                      "prompt_mode", "damping_scope", "population_count"):
            if old.get(field) != cell.get(field):
                raise ValueError("D7 canary binding differs: " + field)
        if len(old["raw_roots"]) != 1 or old["raw_roots"][0]["indices"] != indices:
            raise ValueError("D7 canary raw membership differs")
        raw = old["raw_roots"][0]
        summary = read(Path(raw["run_root"]) / "summary.json")
        if (summary.get("status") != "RAW_COMPLETE" or summary.get("count") != 4
            or summary.get("target_gold_loaded") is not False):
            raise ValueError("D7 canary is not raw complete")
        metadata = summary["metadata"]
        for field in ("cell_key", "cell_id", "pool", "source_commit", "seed",
                      "prompt_mode", "damping_scope"):
            if metadata.get(field) != cell.get(field):
                raise ValueError("D7 canary completion binding differs: " + field)
        if metadata.get("indices") != indices or metadata.get("run_root") != raw["run_root"]:
            raise ValueError("D7 canary completion indices/root differ")
        cell["raw_roots"] = copy.deepcopy(old["raw_roots"])
    return indices


def verify_membership(cells, expected):
    for cell in cells:
        indices = [i for root in cell["raw_roots"] for i in root["indices"]]
        if sorted(indices) != list(expected) or len(set(indices)) != len(indices):
            raise ValueError("D7 cell membership missing or repeated: " + cell["cell_key"])


def build(kind, source, run_prefix, commit, pool, synthetic_pool=None,
          partition="emergency_gpu", packing=1, tasks=8, shard_size=256,
          canary_plan=None, first_plan=None, first_closure=None):
    if kind not in ("debug", "canary", "first", "last"):
        raise ValueError("D7 stage must be debug/canary/first/last")
    if type(tasks) is not int or not 1 <= tasks <= 8:
        raise ValueError("D7 GPU task count must be within 1..8")
    if type(packing) is not int or not 1 <= packing <= 5:
        raise ValueError("D7 packing must be measured and frozen within 1..5")
    if kind != "debug" and partition == "debug":
        raise ValueError("D7 formal stages cannot use debug")
    keys = FIRST_KEYS if kind in ("canary", "first") else LOOP_KEYS
    if kind == "debug":
        keys = FIRST_KEYS + LOOP_KEYS
    cells = cell_specs(source, commit, pool, keys)
    jobs, memberships, barrier = [], None, None
    if kind == "debug":
        if synthetic_pool is None:
            raise ValueError("D7 debug requires an explicit synthetic pool")
        partition, packing, tasks = "debug", 1, 1
        for cell in cells:
            cell.update(synthetic=True, engineering=True, pool=str(synthetic_pool))
            jobs.append(add_job(cell, [0], run_prefix + "/debug/" + cell["cell_key"]))
        expected, total = [0], 5
    elif kind == "canary":
        if tasks != 3:
            raise ValueError("D7 canary freezes exactly three GPU tasks")
        if packing != 1:
            raise ValueError("D7 initial canary packing must be one")
        memberships = canary_indices(read(pool))
        for cell in cells:
            jobs.append(add_job(cell, memberships, run_prefix + "/canary/" + cell["cell_key"]))
        expected, total = memberships, 12
    else:
        if kind == "first":
            if canary_plan is None:
                raise ValueError("D7 first stage requires successful canary roots")
            memberships = validate_canary(read(canary_plan), cells)
            remaining = sorted(set(range(POPULATION)) - set(memberships))
        else:
            if first_plan is None or first_closure is None:
                raise ValueError("D7 last stage requires first plan and raw closure")
            first = read(first_plan)
            if first.get("stage") != "first" or first.get("record_count") != 36096:
                raise ValueError("D7 first plan is incomplete")
            first_cells = first["cells"]
            validate_first_closure(read(first_closure), first_cells)
            for cell in first_cells:
                if (cell["pool"] != str(pool) or cell["source_commit"] != commit
                    or cell["producer_source"] != source or cell["seed"] != SEED):
                    raise ValueError("D7 last stage source/input/runtime binding differs")
            verify_membership(first_cells, range(POPULATION))
            barrier = {"receipt_path": str(first_closure), "first_cells": first_cells}
            remaining = list(range(POPULATION))
        for cell in cells:
            for n, indices in enumerate(chunks(remaining, shard_size)):
                jobs.append(add_job(cell, indices,
                    run_prefix + "/" + kind + "/" + cell["cell_key"] + f"/slice-{n:03d}"))
        expected, total = range(POPULATION), len(keys) * POPULATION
    verify_membership(cells, expected)
    assignments = [[] for _ in range(min(tasks, len(jobs)))]
    # Small independent children backfill as each process finishes. Membership and
    # within-child order remain canonical, independent of completion order.
    for n, job in enumerate(jobs):
        assignments[n % len(assignments)].append(job)
    task_specs = [{"number": n, "partition": partition, "packing": packing,
                   "batch_root": run_prefix + "/" + kind + f"/batches/task-{n:02d}",
                   "jobs": group} for n, group in enumerate(assignments)]
    for task in task_specs:
        validate_task(task)
    if kind == "last":
        cells = copy.deepcopy(barrier["first_cells"]) + cells
        total = 60160
    plan = {"schema": "loopscope.phase11.d7.plan.v1", "gate": "D-7", "stage": kind,
            "source_commit": commit, "seed": SEED, "input_seed": 20261008,
            "population_count": POPULATION, "record_count": total,
            "scheduled_record_count": sum(len(j["indices"]) for j in jobs),
            "cells": cells, "tasks": task_specs, "target_gold_loaded": False}
    if memberships is not None:
        plan["canary_indices"] = memberships
    if barrier is not None:
        plan["first_batch_barrier"] = barrier
    validate_plan_barrier(plan)
    recipe = {k: v for k, v in plan.items() if k != "tasks"}
    recipe.update(schema="loopscope.phase11.d7.recipe.v1", cell_count=len(cells),
        executor="01a142a7-562a-7e83-b085-70999fbbc780",
        planning="01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f",
        generation={"max_new_tokens": 3 if kind == "debug" else 8192,
                    "greedy": True, "native_eos": True},
        analysis={"main_holm_family_size": 4, "baseline_holm_family_size": 2,
                  "alpha": .05, "bootstrap_replicates": 10000, "bootstrap_seed": 20261008,
                  "gold_barrier": "all 60160 raw and S2 predictions and identity/config closed"})
    return recipe, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", "--kind", choices=("debug", "canary", "first", "last"), required=True)
    for key in ("source", "run-prefix", "commit", "pool", "output-dir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--synthetic-pool")
    parser.add_argument("--partition", default="emergency_gpu")
    parser.add_argument("--packing", type=int, default=1)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--canary-plan")
    parser.add_argument("--first-plan")
    parser.add_argument("--first-closure")
    args = parser.parse_args()
    recipe, plan = build(args.stage, args.source, args.run_prefix, args.commit, args.pool,
        args.synthetic_pool, args.partition, args.packing,
        args.tasks if args.tasks is not None else (3 if args.stage == "canary" else 8),
        args.shard_size, args.canary_plan, args.first_plan, args.first_closure)
    for name, value in (("recipe.json", recipe), (args.stage + "-plan.json", plan)):
        save(Path(args.output_dir) / name, value, once=True)
    print(json.dumps({"stage": args.stage, "cells": len(plan["cells"]),
        "records": plan["record_count"], "scheduled_records": plan["scheduled_record_count"],
        "tasks": len(plan["tasks"]), "target_gold_loaded": False}))


if __name__ == "__main__":
    main()
