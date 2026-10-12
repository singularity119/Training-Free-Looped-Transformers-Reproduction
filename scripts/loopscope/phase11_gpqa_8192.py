#!/usr/bin/env python3
"""D-4 full GPQA 8192 producer; frozen generation path reused from D-3."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from tflt.loopscope.phase11_accuracy import load_pool, validate_record, extract_final_answer
from tflt.loopscope.phase11_panel import build_panel, load_config, require, write_json_once
from tflt.loopscope.phase11_posthoc import extract_s1
from run_phase11_accuracy import acquire_row, load_runtime, source_provenance, ForwardTiming

LIMIT = 8192
DATASET = "gpqa_main"
SCHEMA = "loopscope.phase11.gpqa_8192.v1"


def membership():
    extension = read(Path(__file__).resolve().parents[2] / "configs/loopscope/phase11_gpqa_8192.json")
    require(extension == {"schema": SCHEMA, "gate": "D-4", "dataset": DATASET,
        "population_count": 448, "max_new_tokens": LIMIT, "stop": "native_eos",
        "base_config": "configs/loopscope/phase11_panel.json", "primary_parser": "S1", "auxiliary_parser": "S0"},
        "8192 extension differs from D-4 handoff")
    indices = list(range(448))
    return {"schema": SCHEMA, "population_count": 448, "indices": indices,
        "max_new_tokens": LIMIT, "target_gold_loaded": False, "record_count": 8064,
        "cells": [{"cell": c, "indices": indices} for c in build_panel(DATASET)["cells"]]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze(args):
    value = membership()
    pool = load_pool(args.pool, DATASET, "FORMAL_TEST")
    value["pool"] = str(args.pool)
    value["samples"] = [{"canonical_index": i, "identity": r["identity"],
        "prompt_token_length": len(r["input_ids"])} for i, r in enumerate(pool["rows"])]
    config = load_config()
    max_prompt = max(len(r["input_ids"]) for r in pool["rows"])
    require(max_prompt + LIMIT <= config["max_length"], "full GPQA prompt + 8192 exceeds native context")
    value["context_admission"] = {"all_population_inputs_checked": 448,
        "max_prompt_tokens": max_prompt, "required_capacity": max_prompt + LIMIT,
        "configured_native_capacity": config["max_length"]}
    write_json_once(args.out, value)
    print(json.dumps({"status": "D4_FULL_MEMBERSHIP_FROZEN", "record_count": 8064, "out": str(args.out)}))


def acquire(args):
    require(os.environ.get("SLURM_JOB_ID") and (os.environ.get("SLURM_JOB_PARTITION") == "debug" if args.synthetic else os.environ.get("SLURM_JOB_PARTITION") == "emergency_gpu"),
            "D-4 requires debug synthetic or authorized formal allocation")
    frozen = read(args.membership)
    expected = membership()
    require(all(frozen.get(k) == v for k, v in expected.items()), "membership differs from frozen 448 x 18 panel")
    cells = {item["cell"]["cell_id"]: item for item in frozen["cells"]}
    require(args.cell_id in cells, "cell outside D-4")
    cell = cells[args.cell_id]["cell"]
    scope = "PREFLIGHT_ONLY" if args.synthetic else "FORMAL_TEST"
    pool = load_pool(args.pool, DATASET, scope)
    if args.synthetic:
        indices = [0]
    else:
        require(frozen.get("pool") == str(args.pool), "formal pool differs from membership")
        indices = cells[args.cell_id]["indices"]
        if args.indices:
            indices = [int(i) for i in args.indices.split(",")]
        require(indices and len(indices) == len(set(indices)) and all(i in cells[args.cell_id]["indices"] for i in indices),
                "requested indices outside full panel membership")
    source = source_provenance(args.commit)
    args.run_root.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": SCHEMA, "cell": cell, "indices": indices, "scope": scope,
        "max_new_tokens": LIMIT, "synthetic_min_new_tokens": 3 if args.synthetic else None,
        "target_gold_loaded": False, "source_commit": args.commit, "argv": sys.argv,
        "pool": str(args.pool), "membership": str(args.membership), **source}
    write_json_once(args.run_root / "command_args.json", metadata)
    start = time.monotonic()
    count, total = 0, 0
    config = load_config()
    config = {**config, "max_new_tokens": LIMIT}
    try:
        torch, model, tokenizer, adapter, eos_ids, env = load_runtime(cell, config, DATASET)
        capacity = model.config.max_position_embeddings
        require(capacity == config["max_length"], "actual native model context differs")
        require(all(len(pool["rows"][i]["input_ids"]) + LIMIT <= capacity for i in indices),
                "complete prompt + 8192 exceeds actual native context")
        write_json_once(args.run_root / "env.json", {**env, **source, "source_commit": args.commit,
            "native_context_capacity": capacity, "target_gold_loaded": False})
        torch.set_grad_enabled(False)
        torch.manual_seed(20261008)
        torch.cuda.reset_peak_memory_stats()
        timing = ForwardTiming(torch, model, DATASET)
        with (args.run_root / "records.jsonl").open("x", encoding="utf-8") as records, \
             (args.run_root / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry:
            for index in indices:
                torch.cuda.synchronize()
                row_start = time.monotonic()
                record, fact = acquire_row(torch, model, tokenizer, adapter, eos_ids, pool["rows"][index],
                    index, cell, config, scope, LIMIT, 3 if args.synthetic else None)
                torch.cuda.synchronize()
                validate_record(record, DATASET)
                fact.update(timing.summary())
                fact.update(canonical_index=index, identity=record["identity"],
                    elapsed_seconds=time.monotonic() - row_start,
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                records.flush()
                telemetry.write(json.dumps(fact, allow_nan=False) + "\n")
                telemetry.flush()
                count += 1
                total += record["generated_token_count"]
                print(json.dumps({"cell_id": cell["cell_id"], "canonical_index": index,
                    "completed": count, "generated_token_count": record["generated_token_count"],
                    "eos_reached": record["eos_reached"], "elapsed_seconds": fact["elapsed_seconds"]}), flush=True)
        timing.close()
        write_json_once(args.run_root / "summary.json", {"schema": SCHEMA, "status": "GPQA_8192_ATTEMPT_COMPLETE",
            "metadata": metadata, "count": count, "generated_token_count": total,
            "elapsed_seconds": time.monotonic() - start,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "target_gold_loaded": False})
    except Exception as exc:
        write_json_once(args.run_root / "failure.json", {"status": "LENGTH_ATTEMPT_FAILED",
            "exception_type": type(exc).__name__, "completed": count,
            "elapsed_seconds": time.monotonic() - start, "target_gold_loaded": False})
        raise



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--pool", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.set_defaults(func=freeze)
    a = sub.add_parser("acquire")
    for name in ("membership", "pool", "run-root"):
        a.add_argument("--" + name, type=Path, required=True)
    a.add_argument("--cell-id", required=True)
    a.add_argument("--commit", required=True)
    a.add_argument("--indices")
    a.add_argument("--synthetic", action="store_true")
    a.set_defaults(func=acquire)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
