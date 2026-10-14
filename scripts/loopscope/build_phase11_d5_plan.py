#!/usr/bin/env python3
"""Freeze D5 source-specific repeats and independent canonical slices, without gold."""
import argparse
import json
from pathlib import Path
from phase11_d5_batch import save, validate_task

W = "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope"
P8 = "2dd3dfd59caf547a3abbc0a1516bc4a3207d9af0"
P8G = "cffc8d3902c92f0c0cc3107b060aefe18c28a9b6"
P10C = "2644034a9d9c806099693e810c1fada9d73cd750"
P10D = "0fa5ea760eae859fe069db7a84d9777a608e55f6"
NATIVE, LOOP = "q4-native", "q4-w15-18-k2-loop"
FAMILIES = {
    "p10-base-mmlu": (14042, [NATIVE, LOOP, "q4-w15-18-k2-online-current-t-lambda0.9"]),
    "p10-base-arc": (1172, [NATIVE, LOOP, "q4-w15-18-k2-online-fixed-t0-lambda0.4"]),
    "p11-instruct-arc": (1172, ["arc_challenge-" + i for i in (NATIVE, LOOP, "q4-w15-18-k2-online-fixed-t0-lambda0.9")]),
    "p11-instruct-gpqa": (448, ["gpqa_main-" + i for i in (NATIVE, LOOP, "q4-w15-18-k3-loop", "q4-w15-18-k2-online-current-t-lambda0.5", "q4-w15-18-k3-online-fixed-t0-lambda0.9")]),
}


def recipe(stage, commit):
    result = []
    for repeat in (1, 2):
        for family, (count, cells) in FAMILIES.items():
            for cid in cells:
                job = {"family": family, "repeat": repeat, "seed": {1: 20261014, 2: 20261015}[repeat],
                       "cell_id": cid, "population_count": count, "synthetic": False, "engineering": False,
                       "source_commit": commit}
                if family == "p10-base-mmlu":
                    producer = {NATIVE: P8, LOOP: P8G}.get(cid, P10C)
                    manifests = {NATIVE: W + "/inputs/phase8-gate-d-20260905T142900Z-a1/panel.json",
                                 LOOP: W + "/inputs/phase8-gate-g-20260909T163419Z-test-a1/panel.json"}
                    job.update(pool=W + "/inputs/phase8-gate-d-20260905T142900Z-a1/test_pool.json",
                               manifest=manifests.get(cid, W + "/inputs/phase10-gate-c-20261002T081604Z/prepared-attempt1/mmlu-manifest.json"),
                               producer_commit=producer, producer_source=stage + "/historical/" + producer)
                elif family == "p10-base-arc":
                    job.update(pool=W + "/inputs/phase10-gate-a-20261002T062100Z/prepared-attempt1/ARC-test-pool.json",
                               manifest=W + "/inputs/phase10-gate-d-20261006T161109Z/prepared-attempt2/arc-manifest.json",
                               producer_commit=P10D, producer_source=stage + "/historical/" + P10D)
                else:
                    dataset = "arc_challenge" if family.endswith("arc") else "gpqa_main"
                    job.update(pool=W + "/inputs/phase11-gate-a-20261008T094044Z/" + dataset + "-prepared/" + dataset + "-pool.json",
                               reference_producer_commit="f9845ea86e3deaae4145fd383d1f316a70d7dfe3" if dataset == "arc_challenge" else "ba218fc60cc7eb3bd0e45a7e30002549b4404357")
                result.append(job)
    return result


def build(stage, run_prefix, commit):
    cells = recipe(stage, commit)
    if len(cells) != 28 or sum(j["population_count"] for j in cells) != 102796:
        raise ValueError("D5 frozen panel count differs")
    tasks = []
    for shard in range(8):
        groups = []
        for family, packing in (("p10-base-mmlu", 1), ("p10-base-arc", 1), ("p11-instruct-arc", 4), ("p11-instruct-gpqa", 5)):
            jobs = []
            for cell in cells:
                if cell["family"] != family:
                    continue
                n = cell["population_count"]
                indices = list(range(n * shard // 8, n * (shard + 1) // 8))
                jobs.append({**cell, "indices": indices,
                    "run_root": run_prefix + f"/repeat{cell['repeat']}/{family}/{cell['cell_id']}/slice-{shard:02d}"})
            groups.append({"family": family, "packing": packing, "jobs": jobs})
        task = {"number": shard, "partition": "emergency_gpu", "packing": 5,
                "batch_root": run_prefix + f"/batches/task-{shard:02d}", "groups": groups,
                "jobs": [j for g in groups for j in g["jobs"]]}
        validate_task(task)
        tasks.append(task)
    all_pairs = [(j["repeat"], j["family"], j["cell_id"], i) for t in tasks for j in t["jobs"] for i in j["indices"]]
    if len(all_pairs) != 102796 or len(set(all_pairs)) != 102796:
        raise ValueError("formal task membership missing or repeated")
    debug_tasks = []
    for number, base in enumerate((True, False)):
        jobs = []
        for cell in cells:
            if cell["family"].startswith("p10-") != base:
                continue
            j = {**cell, "indices": [0], "engineering": True,
                 "run_root": run_prefix + f"/debug/repeat{cell['repeat']}/{cell['family']}/{cell['cell_id']}"}
            if base and cell.get("producer_commit") in (P10C, P10D):
                dataset = "mmlu" if cell["family"].endswith("mmlu") else "arc_challenge"
                j["manifest"] = W + "/inputs/phase10-gate-b-20261002T070214Z/prepared-attempt1/" + dataset + "-manifest.json"
                j["pool"] = W + ("/inputs/phase10-gate-b-20261002T070214Z/prepared-attempt1/MMLU-validation-pool.json" if dataset == "mmlu" else "/inputs/phase10-gate-a-20261002T062100Z/prepared-attempt1/ARC-validation-pool.json")
            if not base:
                dataset = "arc_challenge" if cell["family"].endswith("arc") else "gpqa_main"
                j.update(synthetic=True, pool=W + "/inputs/phase11-gate-a-20261008T094044Z/synthetic/" + dataset + "-pool.json")
            jobs.append(j)
        task = {"number": number, "partition": "debug", "packing": 1,
                "batch_root": run_prefix + f"/debug/batches/task-{number:02d}", "jobs": jobs}
        validate_task(task)
        debug_tasks.append(task)
    info = {"schema": "loopscope.phase11.d5.recipe.v1", "gate": "D-5",
        "executor": "01a1385a-fdfa-7d43-a6b8-1405a585b8b9", "planning": "01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f",
        "cells": cells, "repeat_count": 2, "cells_per_repeat": 14, "record_count": 102796,
        "input_binding": "original pools, prompts, demonstrations and candidate order unchanged",
        "numeric_comparison": {"atol": 0.0001, "rtol": 0.00001, "strict_equality_reported_separately": True,
                               "role": "descriptive raw-score comparison, not engineering pass threshold"},
        "analysis": {"main_holm_family_size": 20, "secondary_holm_family_size": 10, "mcnemar": "two-sided exact",
            "alpha": 0.05, "bootstrap_replicates": 10000, "base_bootstrap_seed": 20261002,
            "instruct_bootstrap_seed": 20261008, "ci_level": 0.95, "gold_barrier": "all 102796 raw records and predictions closed"},
        "packing": {"p10-base-mmlu": 1, "p10-base-arc": 1, "p11-instruct-arc": 4, "p11-instruct-gpqa": 5},
        "target_gold_loaded": False}
    return info, {"tasks": tasks, "record_count": 102796, "target_gold_loaded": False}, {"tasks": debug_tasks, "target_gold_loaded": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("stage", "run-prefix", "commit", "output-dir"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    info, formal, debug = build(a.stage, a.run_prefix, a.commit)
    for name, value in (("recipe.json", info), ("formal-plan.json", formal), ("debug-plan.json", debug)):
        save(Path(a.output_dir) / name, value, once=True)
    print(json.dumps({"cells": 28, "records": 102796, "formal_tasks": 8, "debug_tasks": 2, "target_gold_loaded": False}))
