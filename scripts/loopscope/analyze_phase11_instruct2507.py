#!/usr/bin/env python3
"""Close all Instruct2507 predictions, then separately score with the unchanged D8 statistics."""
import argparse
import datetime
import json
import os
from pathlib import Path

from instruct2507_contract import read, validate_bundle
from phase11_instruct2507_worker import metadata
from analyze_phase11_d8 import lines, flattened
from tflt.loopscope.phase11_instruct2507 import CELLS, COUNT, PARSER, validate_job, validate_pool
from tflt.loopscope.phase11_d8_analysis import analyze
from tflt.loopscope.phase11_analysis import gold_indices
from tflt.loopscope.phase11_panel import require, write_json_once


def load_all(registry, tokenizer):
    require(registry["schema"] == "loopscope.phase11.instruct2507.full_registry.v1", "registry schema differs")
    data, common, pools, roots, commits = {k: {} for k in CELLS}, None, {}, set(), set()
    for job in registry["jobs"]:
        cell = validate_job(job)
        require(not job["synthetic"] and not job["engineering"], "formal closure contains engineering work")
        if job["pool"] not in pools:
            pools[job["pool"]] = validate_pool(read(job["pool"]))
        pool = pools[job["pool"]]
        if common is None:
            common = pool
        require(pool["rows"] == common["rows"] and pool["test_identities"] == common["test_identities"],
                "cells use different canonical prompts or options")
        meta, root = metadata(job, cell, pool), Path(job["run_root"])
        require(str(root) not in roots, "duplicate run root")
        roots.add(str(root))
        summary = read(root / "summary.json")
        require(summary["status"] == "RAW_COMPLETE" and summary["metadata"] == meta and
                summary["count"] == len(job["indices"]), "incomplete raw root")
        commits.add(job["source_commit"])
        for i in job["indices"]:
            require(i not in data[job["cell_key"]], "duplicate cell/index")
            bundle = read(root / "complete" / f"{i:06d}.json")
            r, t = bundle["record"], bundle["telemetry"]
            validate_bundle(i, r, t, meta, pool, tokenizer)
            compact = {k: r[k] for k in ("identity", "canonical_index", "generated_token_count", "end_reason", "actual_seed")}
            compact.update({k: v for k, v in r["final_span"].items() if k != "final_text"})
            compact.update({k: t[k] for k in ("elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes")})
            compact.update({k: t[k] for k in ("prefill_elapsed_seconds", "decode_elapsed_seconds") if k in t})
            data[job["cell_key"]][i] = compact
    require(len(commits) == 1, "producer revisions differ")
    for key in CELLS:
        require(set(data[key]) == set(range(COUNT)), "full 448 identities missing in " + key)
        data[key] = [data[key][i] for i in range(COUNT)]
    return data, common


def run(args):
    require(bool(args.registry) != bool(args.closure), "choose gold-free registry closure or prior sealed closure")
    require(bool(args.closure) == bool(args.gold), "gold is allowed only after a prior complete closure")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if args.registry:
        registry = read(args.registry)
        data, pool = load_all(registry, tokenizer)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        pred = args.output_dir / "predictions.jsonl"
        with pred.open("x") as stream:
            for row in flattened(data):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_json_once(args.output_dir / "closure.json", {"schema": "loopscope.phase11.instruct2507.closure.v1",
            "status": "GOLD_UNREAD", "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "record_count": 1344, "prediction_closed": True, "target_gold_loaded": False,
            "registry": registry, "predictions_path": str(pred),
            "model_provenance": pool.get("instruct2507_model_provenance")})
        print(json.dumps({"status": "GOLD_UNREAD", "record_count": 1344}))
        return
    closure = read(args.closure)
    require(closure["schema"] == "loopscope.phase11.instruct2507.closure.v1" and closure["status"] == "GOLD_UNREAD" and
            closure["prediction_closed"] is True and closure["target_gold_loaded"] is False and
            closure["record_count"] == 1344, "prediction closure incomplete")
    data, pool = load_all(closure["registry"], tokenizer)
    require(flattened(data) == lines(closure["predictions_path"]), "saved predictions differ before gold access")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_once(args.output_dir / "analysis_spec.json", {"gold_read": False,
        "source_closure": str(args.closure), "family_size": 2, "bootstrap_replicates": 10000, "seed": 20261018})
    ids = pool["test_identities"]
    gold = gold_indices(lines(args.gold), ids, {r["identity"]: r["choice_labels"] for r in pool["rows"]},
                        {r["identity"]: r["subject"] for r in pool["rows"]})
    write_json_once(args.output_dir / "gold_read_receipt.json", {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gold_path": str(args.gold), "source_closure": str(args.closure), "prior_prediction_closed": True, "n": 448})
    result = analyze(data, gold)
    result.update(source_closure=str(args.closure), gold_read=True, gate="INSTRUCT2507_GREEDY8192_WHOLE_S2",
                  schema="loopscope.phase11.instruct2507.analysis.v1",
                  status="INSTRUCT2507_COMPLETE_EXPLORATORY_TRANSFER", primary_parser=PARSER,
                  model_provenance=pool.get("instruct2507_model_provenance"))
    result["interpretation"] = ("Configurations were selected on these same GPQA items. "
        "Greedy8192 exploratory transfer to the user-designated Instruct2507 OSS checkpoint; "
        "whole generated output is scored by S2. Not independent-item confirmation or a "
        "separate causal estimate of checkpoint/template/platform changes; OSS revision is unverified.")
    write_json_once(args.output_dir / "analysis.json", result)
    print(json.dumps({"status": result["status"], "record_count": 1344, "gold_read": True}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("registry", "closure", "gold", "model-dir", "output-dir"):
        p.add_argument("--" + key, type=Path, required=key in ("model-dir", "output-dir"))
    run(p.parse_args())
