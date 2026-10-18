#!/usr/bin/env python3
"""Verify the D8 debug producer, complete token spans, and loop/cache lifecycle."""
import argparse
import json
from pathlib import Path
from phase11_d8 import read, validate_bundle, validate_job, validate_pool, metadata
from tflt.loopscope.phase11_d8 import MODEL, REVISION, CELLS
from tflt.loopscope.phase11_panel import require


def run(plan_file):
    from transformers import AutoTokenizer
    plan = read(plan_file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    found = []
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
                s = r["final_span"]
                require(not s["unclosed_thinking"] and s["thinking_token_count"] > 0 and
                        s["prediction_index"] is not None and t["decode_count"] >= 2,
                        "D8 smoke lacks actual thinking, final S2, or incremental decode")
            found.append(job["cell_key"])
    require(sorted(found) == sorted(CELLS), "D8 smoke must cover all three cells once")
    result = {"status": "D8_DEBUG_PATH_VERIFIED", "cells": found,
              "actual_thinking": True, "final_span_S2": True, "target_gold_loaded": False}
    with Path(plan["verification_path"]).open("x") as h:
        json.dump(result, h, indent=2)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    raise SystemExit(run(p.parse_args().plan))
