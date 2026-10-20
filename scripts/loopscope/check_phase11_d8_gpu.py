#!/usr/bin/env python3
"""Verify the D8 debug producer, complete token spans, and loop/cache lifecycle."""
import argparse
import json
from pathlib import Path
from phase11_d8 import read, validate_bundle, validate_job, validate_pool, metadata
from tflt.loopscope.phase11_d8 import MODEL, REVISION, CELLS
from tflt.loopscope.phase11_panel import require


def validate_generation_path(record, telemetry):
    span = record["final_span"]
    require(not span["unclosed_thinking"] and span["thinking_token_count"] > 0 and
            not span["final_empty"] and telemetry["decode_count"] >= 2,
            "D8 smoke lacks actual thinking, final text, or incremental decode")
    # A correctly extracted no_match is an allowed scientific outcome, not a
    # producer failure. Positive parser coverage is checked across the smoke.
    return span["prediction_index"] is not None


def run(plan_file):
    from transformers import AutoTokenizer
    plan = read(plan_file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    found = []
    s2_positive_count = 0
    for task in plan["tasks"]:
        require(task["partition"] == "debug", "D8 smoke verifier requires debug plan")
        for job in task["jobs"]:
            cell, pool = validate_job(job), validate_pool(read(job["pool"]))
            meta = metadata(job, cell, pool)
            root = Path(job["run_root"])
            require(read(root / "summary.json")["status"] == "RAW_COMPLETE", "D8 smoke incomplete")
            for index in job["indices"]:
                bundle = read(root / "complete" / f"{index:06d}.json")
                r, t = bundle["record"], bundle["telemetry"]
                validate_bundle(index, r, t, meta, pool, tokenizer)
                s2_positive_count += validate_generation_path(r, t)
            found.append(job["cell_key"])
    require(sorted(found) == sorted(CELLS), "D8 smoke must cover all three cells once")
    require(s2_positive_count > 0, "D8 smoke lacks a real positive final-span S2 example")
    result = {"status": "D8_DEBUG_PATH_VERIFIED", "cells": found,
              "actual_thinking": True, "final_span_S2": True,
              "S2_positive_count": s2_positive_count,
              "no_match_is_valid_outcome": True, "target_gold_loaded": False}
    with Path(plan["verification_path"]).open("x") as h:
        json.dump(result, h, indent=2)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    raise SystemExit(run(p.parse_args().plan))
