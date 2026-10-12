#!/usr/bin/env python3
"""D-3 label-free GPQA length extension; original 2048 producer stays frozen."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

from tflt.loopscope.phase11_accuracy import load_pool, validate_record, extract_final_answer
from tflt.loopscope.phase11_panel import build_panel, load_config, require, write_json_once
from tflt.loopscope.phase11_posthoc import extract_s1
from run_phase11_accuracy import acquire_row, load_runtime, source_provenance, ForwardTiming

SEED = 20261012
LIMIT = 8192
DATASET = "gpqa_main"
SCHEMA = "loopscope.phase11.length_pilot.v1"


def membership():
    indices = random.Random(SEED).sample(range(448), 64)
    cells = build_panel(DATASET)["cells"]
    return {"schema": SCHEMA, "seed": SEED, "population_count": 448,
            "indices": indices, "max_new_tokens": LIMIT, "target_gold_loaded": False,
            "record_count": 336, "cells": [{"cell": c, "indices": indices if c["arm"] == "Native"
                                              else indices[:16]} for c in cells]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze(args):
    value = membership()
    if args.pool:
        pool = load_pool(args.pool, DATASET, "FORMAL_TEST")
        value["pool"] = str(args.pool)
        value["samples"] = [{"canonical_index": i, "identity": pool["rows"][i]["identity"],
                             "prompt_token_length": len(pool["rows"][i]["input_ids"])} for i in value["indices"]]
        # Entire original input population is checked; no output or gold is opened.
        config = load_config()
        lengths = [len(r["input_ids"]) for r in pool["rows"]]
        require(max(lengths) + LIMIT <= config["max_length"], "full GPQA prompt + 8192 exceeds context")
        value["context_admission"] = {"all_population_inputs_checked": 448,
            "max_prompt_tokens": max(lengths), "required_capacity": max(lengths) + LIMIT,
            "configured_native_capacity": config["max_length"]}
    write_json_once(args.out, value)
    print(json.dumps({"status": "D3_MEMBERSHIP_FROZEN", "record_count": 336, "out": str(args.out)}))


def acquire(args):
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "D-3 requires an allocated debug job")
    frozen = read(args.membership)
    expected = membership()
    require(all(frozen.get(k) == v for k, v in expected.items()), "membership differs from frozen random pilot")
    cells = {item["cell"]["cell_id"]: item for item in frozen["cells"]}
    require(args.cell_id in cells, "cell outside D-3")
    cell = cells[args.cell_id]["cell"]
    scope = "PREFLIGHT_ONLY" if args.synthetic else "FORMAL_TEST"
    pool = load_pool(args.pool, DATASET, scope)
    if args.synthetic:
        indices = [0]
    else:
        require(frozen.get("pool") == str(args.pool), "pilot pool differs from membership")
        indices = cells[args.cell_id]["indices"]
        if args.indices:
            indices = [int(i) for i in args.indices.split(",")]
        require(indices and len(indices) == len(set(indices)) and all(i in cells[args.cell_id]["indices"] for i in indices),
                "requested indices outside cell pilot membership")
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
        write_json_once(args.run_root / "summary.json", {"schema": SCHEMA, "status": "LENGTH_PILOT_ATTEMPT_COMPLETE",
            "metadata": metadata, "count": count, "generated_token_count": total,
            "elapsed_seconds": time.monotonic() - start,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "target_gold_loaded": False})
    except Exception as exc:
        write_json_once(args.run_root / "failure.json", {"status": "LENGTH_ATTEMPT_FAILED",
            "exception_type": type(exc).__name__, "completed": count,
            "elapsed_seconds": time.monotonic() - start, "target_gold_loaded": False})
        raise


def quantiles(values):
    values = sorted(values)
    return {str(q): values[round((len(values) - 1) * q)] for q in (0, .25, .5, .75, .9, .95, 1)}


def summarize(args):
    frozen = read(args.membership)
    expected = membership()
    require(all(frozen.get(k) == v for k, v in expected.items()), "membership differs")
    pool = load_pool(Path(frozen["pool"]), DATASET, "FORMAL_TEST")
    expected_pairs = {(item["cell"]["cell_id"], i) for item in frozen["cells"] for i in item["indices"]}
    cells, seen, attempts, commits = {}, set(), [], set()
    from transformers import AutoTokenizer
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["tokenizer_revision"], local_files_only=True)
    for root in args.attempt:
        summary = read(root / "summary.json")
        meta = summary["metadata"]
        require(meta["scope"] == "FORMAL_TEST" and meta["target_gold_loaded"] is False and
                meta["max_new_tokens"] == LIMIT and summary["status"] == "LENGTH_PILOT_ATTEMPT_COMPLETE",
                "summary is not a complete D-3 pilot attempt")
        records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
        telemetry = [json.loads(line) for line in (root / "telemetry.jsonl").read_text().splitlines()]
        require([r["canonical_index"] for r in records] == meta["indices"] and len(records) == summary["count"] == len(telemetry),
                "attempt records/telemetry membership incomplete")
        cell_id = meta["cell"]["cell_id"]
        canonical_cell = next(item["cell"] for item in frozen["cells"] if item["cell"]["cell_id"] == cell_id)
        require(meta["cell"] == canonical_cell, "cell recipe drift")
        commits.add(meta["source_commit"])
        for record, fact in zip(records, telemetry):
            validate_record(record, DATASET)
            index = record["canonical_index"]
            pair = cell_id, index
            require(pair in expected_pairs and pair not in seen and record["identity"] == pool["rows"][index]["identity"],
                    "duplicate/unknown sample identity")
            require(fact["canonical_index"] == index and fact["identity"] == record["identity"] and
                    fact["prefill_count"] == 1 and fact["decode_count"] == record["generated_token_count"] - 1 and
                    fact["cache_growth_closed"] and record["max_new_tokens"] == LIMIT,
                    "length/cache mechanism differs")
            if "online" in fact:
                require(fact["online"]["decode_fit_count"] == 0 and fact["online"]["directions_released"], "direction lifecycle drift")
            seen.add(pair)
            cells.setdefault(cell_id, []).append((record, fact))
        attempts.append({"run_root": str(root), "count": len(records), "elapsed_seconds": summary["elapsed_seconds"],
            "peak_reserved_bytes": summary["peak_reserved_bytes"], "gpu": read(root / "env.json")["gpu"]})
    tables = []
    for cell_id, rows in cells.items():
        budgets = []
        for budget in (2048, 4096, LIMIT):
            eos = s0 = s1 = 0
            for record, fact in rows:
                eos += int(record["eos_reached"] and record["generated_token_count"] <= budget)
                text = tokenizer.decode(record["generated_token_ids"][:budget], skip_special_tokens=True)
                s0 += int(extract_final_answer(text, record["candidate_labels"])["extraction_status"] == "unique_valid")
                s1 += int(extract_s1(text)["extraction_status"] == "unique_valid")
            budgets.append({"budget": budget, "n": len(rows), "eos_count": eos, "unfinished_count": len(rows) - eos,
                "eos_rate": eos / len(rows), "s0_format_success_count": s0, "s1_format_success_count": s1,
                "s0_format_success_rate": s0 / len(rows), "s1_format_success_rate": s1 / len(rows)})
        tables.append({"cell_id": cell_id, "n": len(rows), "budgets": budgets,
            "generated_length_quantiles_including_right_censoring": quantiles([r["generated_token_count"] for r, f in rows]),
            "question_seconds": sum(f["elapsed_seconds"] for r, f in rows),
            "generated_tokens": sum(r["generated_token_count"] for r, f in rows),
            "max_peak_reserved_bytes": max(f["peak_reserved_bytes"] for r, f in rows)})
    closed = seen == expected_pairs
    require(not args.require_complete or closed, "336 pilot records not closed")
    result = {"schema": SCHEMA, "status": "D3_PILOT_CLOSED" if closed else "D3_PARTIAL_LABEL_FREE",
        "target_gold_loaded": False, "accuracy_computed": False, "completed_records": len(seen), "expected_records": 336,
        "source_commit": next(iter(commits)) if len(commits) == 1 else None,
        "source_commits": sorted(commits), "prefix_diagnostic_not_independent_rerun": True,
        "length_limit_is_right_censoring": True, "cells": tables, "attempts": attempts,
        "missing": [{"cell_id": c, "canonical_index": i} for c, i in sorted(expected_pairs - seen)]}
    write_json_once(args.out, result)
    print(json.dumps({"status": result["status"], "completed_records": len(seen), "target_gold_loaded": False}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--pool", type=Path)
    f.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--membership", type=Path, required=True)
    a.add_argument("--pool", type=Path, required=True)
    a.add_argument("--cell-id", required=True)
    a.add_argument("--indices")
    a.add_argument("--synthetic", action="store_true")
    a.add_argument("--run-root", type=Path, required=True)
    a.add_argument("--commit", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--membership", type=Path, required=True)
    s.add_argument("--attempt", type=Path, action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--require-complete", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"freeze": freeze, "acquire": acquire, "summarize": summarize}[args.command](args)
