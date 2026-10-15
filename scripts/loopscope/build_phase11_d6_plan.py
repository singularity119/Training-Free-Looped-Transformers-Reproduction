#!/usr/bin/env python3
"""Freeze D6's twelve fresh GPQA cells and disjoint canonical shards, without gold."""
import argparse
import json
from pathlib import Path

from phase11_d6 import selected_cells, SEED
from phase11_d6_batch import save, validate_task


def build(stage, run_prefix, commit, input_dir, synthetic_input_dir, partition="emergency_gpu", packing=5, shards=8):
    if type(shards) is not int or not 1 <= shards <= 8:
        raise ValueError("D6 shard/GPU count must be within 1..8")
    if partition == "debug":
        raise ValueError("D6 formal partition cannot be debug")
    cells = []
    for key, cell in selected_cells().items():
        cells.append({"cell_key": key, "cell_id": cell["cell_id"], "prompt_mode": cell["prompt_mode"],
            "damping_scope": cell["damping_scope"], "cell": cell, "population_count": 448,
            "seed": SEED, "input_seed": 20261008, "synthetic": False, "engineering": False,
            "source_commit": commit, "producer_source": stage,
            "pool": input_dir + "/" + cell["prompt_mode"] + "-pool.json", "raw_roots": []})
    tasks = []
    for shard in range(shards):
        indices = list(range(448 * shard // shards, 448 * (shard + 1) // shards))
        jobs = []
        for cell in cells:
            root = run_prefix + "/" + cell["cell_key"] + f"/slice-{shard:02d}"
            jobs.append({k: v for k, v in cell.items() if k not in ("raw_roots", "cell")} |
                        {"indices": indices, "run_root": root})
            cell["raw_roots"].append({"run_root": root, "records_path": root + "/records.jsonl",
                "telemetry_path": root + "/telemetry.jsonl", "indices": indices})
        task = {"number": shard, "partition": partition, "packing": packing,
                "batch_root": run_prefix + f"/batches/task-{shard:02d}", "jobs": jobs}
        validate_task(task)
        tasks.append(task)
    membership = [(job["cell_key"], i) for task in tasks for job in task["jobs"] for i in job["indices"]]
    if len(membership) != 5376 or len(set(membership)) != 5376:
        raise ValueError("D6 formal membership missing or repeated")
    debug_jobs = []
    for cell in cells:
        debug_jobs.append({k: v for k, v in cell.items() if k not in ("raw_roots", "cell")} |
            {"indices": [0], "synthetic": True, "engineering": True,
             "pool": synthetic_input_dir + "/" + cell["prompt_mode"] + "-pool.json",
             "run_root": run_prefix + "/debug/" + cell["cell_key"]})
    debug_task = {"number": 0, "partition": "debug", "packing": 1,
                  "batch_root": run_prefix + "/debug/batches/task-00", "jobs": debug_jobs}
    validate_task(debug_task)
    shared = {"schema": "loopscope.phase11.d6.plan.v1", "gate": "D-6", "population_count": 448,
        "record_count": 5376, "source_commit": commit, "seed": SEED, "input_seed": 20261008,
        "cells": cells, "target_gold_loaded": False}
    recipe = {**shared, "schema": "loopscope.phase11.d6.recipe.v1", "cell_count": 12,
        "executor": "01a13d93-a681-7d20-ab05-434dc91e1097", "planning": "01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f",
        "generation": {"max_new_tokens": 8192, "greedy": True, "native_eos": True},
        "analysis": {"main_holm_family_size": 4, "auxiliary_holm_family_size": 10,
            "alpha": 0.05, "bootstrap_replicates": 10000, "bootstrap_seed": 20261008,
            "randomization_draws": 100000, "randomization_seed": 20261008,
            "gold_barrier": "all 5376 raw and S2 predictions and identity/config closed"}}
    return recipe, {**shared, "tasks": tasks}, {"schema": "loopscope.phase11.d6.debug_plan.v1",
        "source_commit": commit, "tasks": [debug_task], "target_gold_loaded": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("stage", "run-prefix", "commit", "input-dir", "synthetic-input-dir", "output-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--partition", default="emergency_gpu")
    p.add_argument("--packing", type=int, default=5)
    p.add_argument("--shards", type=int, default=8)
    a = p.parse_args()
    recipe, formal, debug = build(a.stage, a.run_prefix, a.commit, a.input_dir,
        a.synthetic_input_dir, a.partition, a.packing, a.shards)
    for name, value in (("recipe.json", recipe), ("formal-plan.json", formal), ("debug-plan.json", debug)):
        save(Path(a.output_dir) / name, value, once=True)
    print(json.dumps({"cells": 12, "records": 5376, "formal_tasks": a.shards,
                      "debug_tasks": 1, "target_gold_loaded": False}))
