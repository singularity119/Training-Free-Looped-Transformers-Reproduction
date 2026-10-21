#!/usr/bin/env python3
"""Verify real Lingjing synthetic thinking/cache/S2 paths without opening gold."""
import argparse
from pathlib import Path

from phase11_d8 import read, metadata, validate_bundle
from check_phase11_d8_gpu import validate_generation_path
from tflt.loopscope.phase11_d8 import CELLS, validate_job, validate_pool
from tflt.loopscope.phase11_panel import require, write_json_once


def run(plan_file, model_dir, output):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    plan = read(plan_file)
    require(plan["total_gpus"] in (1, 8, 16) and plan["packing"] == 1, "synthetic preflight allocation differs")
    found, positive = [], 0
    for task in plan["tasks"]:
        for job in task["jobs"]:
            require(job["engineering"] and job["synthetic"], "preflight cannot consume formal outcomes")
            cell, pool = validate_job(job), validate_pool(read(job["pool"]))
            meta, root = metadata(job, cell, pool), Path(job["run_root"])
            summary = read(root / "summary.json")
            require(summary["status"] == "RAW_COMPLETE" and summary["metadata"] == meta,
                    "preflight raw root incomplete")
            for i in job["indices"]:
                bundle = read(root / "complete" / f"{i:06d}.json")
                r, t = bundle["record"], bundle["telemetry"]
                validate_bundle(i, r, t, meta, pool, tokenizer)
                positive += validate_generation_path(r, t)
            found.append(job["cell_key"])
    require(sorted(found) == sorted(list(CELLS) * plan["total_gpus"]) and positive > 0,
            "preflight needs all worker/cell paths and real S2 positive coverage")
    write_json_once(Path(output), {"status": "D9_SYNTHETIC_PATH_VERIFIED", "cells": found,
        "source_commit": plan["source_commit"], "S2_positive_count": positive, "target_gold_loaded": False,
        "capacity_representative": False,
        "capacity_note": "short natural EOS does not prove the 32768 upper-tail memory budget"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "model-dir", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    run(a.plan, a.model_dir, a.output)
