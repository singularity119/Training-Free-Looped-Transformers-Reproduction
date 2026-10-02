#!/usr/bin/env python3
"""Acquire one Phase 10 Native/Loop/Online cell, keeping all target labels sealed."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from tflt.loopscope.phase10_accuracy import (
    SCOPES, check_position_metadata, choices_for_row, engineering_cell, load_pool, require,
    row_identity, score_record, select_indices, write_json_once,
)
from tflt.loopscope.phase10_runtime import direction_schedule, make_runtime, validate_calls


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("mmlu", "arc_challenge"), required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--cell-id", required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--scope", choices=SCOPES, required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--preflight-indices", type=Path)
    result.add_argument("--max-length", type=int)
    result.add_argument("--engineering-check", choices=("zero-strength", "k2-policy"))
    result.add_argument("--dry-run", action="store_true")
    return result


def load_inputs(args):
    from tflt.loopscope.phase10_panel import load_config, validate_score_manifest
    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = validate_score_manifest(manifest, config=config, pool_path=args.pool, scope=args.scope)
    require(manifest["dataset"] == args.dataset, "requested dataset differs from score manifest")
    require(args.cell_id in cells, "requested cell is absent from score manifest")
    cell = cells[args.cell_id]
    require(args.scope != "FORMAL_TEST" or cell["new_in_phase10"],
            "historical reuse cells cannot be reacquired without planning authorization")
    rows = load_pool(args.pool, args.dataset, args.scope)
    revision = manifest["dataset_recipe"].get("revision")
    if args.dataset == "arc_challenge" and revision:
        require(all(row_identity(row, args.dataset).startswith("allenai/ai2_arc@" + revision + ":ARC-Challenge:") for row in rows),
                "ARC pool dataset revision differs from frozen manifest")
    explicit = json.loads(args.preflight_indices.read_text()) if args.preflight_indices else None
    indices = select_indices(rows, cell, args.scope, explicit)
    return manifest, cell, rows, indices


def make_loop_config(cell, runtime):
    from tflt.config import LoopConfig
    return LoopConfig(model_alias=cell["model"], window=tuple(cell["window"]), k=int(cell["k"]),
        alpha=1.0, beta=0.0, strategy="damped_euler", iteration_mode="block",
        cache_strategy=cell["cache_strategy"], decode_mode="bypass", residual_transform=runtime)


def load_runtime(cell, max_length=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    require(importlib.metadata.version("lm_eval") == "0.4.11", "Phase 10 requires native lm_eval 0.4.11 scorer")
    tokenizer = AutoTokenizer.from_pretrained(cell["model"], revision=cell["tokenizer_revision"],
        local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cell["model"], revision=cell["revision"],
        local_files_only=True, trust_remote_code=True, torch_dtype=getattr(torch, cell["dtype"])).to("cuda").eval()
    observed = getattr(model.config, "_commit_hash", None)
    require(not observed or observed == cell["revision"], "model revision differs from frozen cell")
    runtime = make_runtime(cell)
    if cell["dataset"] == "mmlu":
        from tflt.loopscope.phase9_adapter import build_hflm_class
        runtime_kw = {"phase9_runtime": runtime}
    else:
        from tflt.loopscope.phase10_adapter import build_hflm_class
        runtime_kw = {"phase10_runtime": runtime}
    extra = {"max_length": max_length} if max_length is not None else {}
    adapter = build_hflm_class(HFLM)(pretrained=model, tokenizer=tokenizer, batch_size="1",
        trust_remote_code=True, **runtime_kw, **extra)
    env = {"python": sys.executable, "platform": platform.platform(),
        "model_revision": observed or cell["revision"], "tokenizer_revision": cell["tokenizer_revision"],
        "model_dtype": str(model.dtype), "max_length": int(adapter.max_length),
        "job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "qos": os.environ.get("SLURM_JOB_QOS"), "gpu": torch.cuda.get_device_name(),
        "total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "lm_eval", "datasets")}}
    return torch, model, tokenizer, adapter, runtime, env


def check_frozen_arc_tokens(row, cell, positions):
    metadata = row["tokenization"]
    model_key = cell["model"] if cell["model"] in metadata else ("q4" if "4B" in cell["model"] else "q17")
    frozen = metadata[model_key]
    require([item["continuation_length"] for item in positions] == frozen["continuation_lengths"],
        "runtime ARC continuation tokenization differs from frozen input")
    expected = frozen["candidates"]
    require(len(expected) == len(positions), "frozen ARC candidate metadata is incomplete")
    for actual, target in zip(positions, expected):
        for key in ("position", "input_length", "continuation_length", "left_truncated_tokens"):
            require(json.dumps(actual[key]) == json.dumps(target[key]), "runtime ARC retained-prefix/tokenization differs from frozen input: " + key)
        require(len(actual["valid_positions"]) == target["valid_prefix_count"], "ARC retained valid-prefix length differs")


def run(args):
    manifest, cell, rows, indices = load_inputs(args)
    acquisition_cell = engineering_cell(cell, args.engineering_check, args.scope)
    if args.dry_run:
        print(json.dumps({"status": "PHASE10_SCORE_DRY_RUN", "cell": cell, "acquisition_cell": acquisition_cell,
            "engineering_check": args.engineering_check, "scope": args.scope,
            "count": len(indices), "target_gold_loaded": False}, sort_keys=True))
        return 0
    require(bool(os.environ.get("SLURM_JOB_ID")), "model score acquisition requires an authorized Slurm job")
    require(manifest.get("dataset_recipe", {}).get("revision"), "dataset revision must be accepted before model execution")
    args.run_root.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": "loopscope.phase10.score_attempt.v1", "cell": cell, "dataset": args.dataset,
        "acquisition_cell": acquisition_cell, "engineering_check": args.engineering_check,
        "dataset_recipe": manifest["dataset_recipe"], "scope": args.scope, "indices": indices,
        "source_commit": args.commit, "pool": str(args.pool), "manifest": str(args.manifest),
        "max_length_requested": args.max_length, "target_gold_loaded": False, "argv": sys.argv}
    write_json_once(args.run_root / "command_args.json", metadata)
    torch, model, tokenizer, adapter, runtime, env = load_runtime(acquisition_cell, args.max_length)
    env.update(source_commit=args.commit, dataset=args.dataset, scope=args.scope, target_gold_loaded=False)
    write_json_once(args.run_root / "env.json", env)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261002)
    torch.cuda.reset_peak_memory_stats()
    from tflt.wrapper import looped_model
    if args.dataset == "mmlu":
        from tflt.loopscope.phase9_adapter import score_choices
    else:
        from tflt.loopscope.phase10_adapter import score_choices
    manager = nullcontext(model) if cell["arm"] == "Native" else looped_model(model, make_loop_config(acquisition_cell, runtime))
    k = int(cell["k"] or 0)
    stats = {"request_count": 0, "model_context_count": 0, "max_input_length": 0,
        "left_truncated_requests": 0, "multitoken_requests": 0,
        "callback_by_t": {str(t): 0 for t in range(k)}, "applied_by_t": {str(t): 0 for t in range(k)},
        "direction_fit_by_t": {str(t): 0 for t in range(k)},
        "direction_used_fit_by_t": {str(t): 0 for t in range(k)},
        "direction_schedule": direction_schedule(acquisition_cell["direction_policy"], k) if runtime else [],
        "t0_mutation_count": 0, "unexpected_t0_mutation_count": 0, "nonanswer_mutation_count": 0}
    started = time.monotonic()
    with manager, (args.run_root / "scores.jsonl").open("x", encoding="utf-8") as output:
        for completed, index in enumerate(indices, 1):
            row = rows[index]
            choices = choices_for_row(row, args.dataset)
            identity = row_identity(row, args.dataset)
            values = score_choices(adapter, row["prompt"], identity) if args.dataset == "mmlu" else score_choices(adapter, row["prompt"], identity, choices)
            positions = adapter.phase9_positions if args.dataset == "mmlu" else adapter.phase10_positions
            position_stats = check_position_metadata(positions, tokenizer)
            if args.dataset == "arc_challenge":
                check_frozen_arc_tokens(row, cell, positions)
            for key, value in position_stats.items():
                stats[key] = max(stats[key], value) if key == "max_input_length" else stats[key] + value
            if runtime is not None:
                stats["model_context_count"] += validate_calls(runtime, k)
                for call in runtime.calls:
                    stats["callback_by_t"][str(call["t"])] += 1
                    stats["applied_by_t"][str(call["t"])] += int(call["applied"])
                    fit_t = call.get("direction_fit_t", 0 if call.get("fitted") else None)
                    used_t = call.get("direction_used_fit_t", 0 if call["t"] >= 1 else None)
                    if fit_t is not None:
                        stats["direction_fit_by_t"][str(fit_t)] += 1
                    if used_t is not None:
                        stats["direction_used_fit_by_t"][str(call["t"])] += 1
                    if call["t"] == 0 and call["answer_max_abs_change"] != 0:
                        stats["t0_mutation_count"] += 1
                runtime.calls.clear()
                runtime.context_summaries.clear()
                if hasattr(runtime, "diagnostics"):
                    runtime.diagnostics.clear()
            output.write(json.dumps(score_record(row, args.dataset, values, positions), ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            if completed % 32 == 0:
                print(json.dumps({"completed": completed, "total": len(indices), "cell_id": cell["cell_id"],
                    "elapsed_seconds": time.monotonic() - started}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    summary = {"schema": "loopscope.phase10.score_summary.v1", "status": "SCORES_COMPLETE",
        "metadata": metadata, "count": len(indices), "target_gold_loaded": False,
        "elapsed_seconds": elapsed, "samples_per_second": len(indices) / elapsed if elapsed else None,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "oom_count": 0,
        "positions_and_runtime": stats}
    write_json_once(args.run_root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "cell_id": cell["cell_id"], "count": len(indices), "target_gold_loaded": False}))
    return 0


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
