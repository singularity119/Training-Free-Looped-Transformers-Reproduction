#!/usr/bin/env python3
"""D8 Qwen3 thinking input preparation and resumable three-cell acquisition."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import time
import uuid

from tflt.loopscope.phase11_accuracy import load_pool, select_indices, validate_record
from tflt.loopscope.phase11_checkpoint import RecordStore
from tflt.loopscope.phase11_panel import load_config, require, write_json_once
from tflt.loopscope.phase11_d8 import (MODEL, REVISION, SEED, MAX_LENGTH, MAX_NEW_TOKENS,
    SAMPLING, selected_cells, prepare_pool, validate_pool, actual_seed, answer_span, validate_job)
from run_phase11_accuracy import acquire_row, load_runtime, source_provenance, ForwardTiming


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_compute(torch, seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {k: seed for k in ("python", "numpy", "torch", "cuda_all_devices")}


def prepare(args):
    from transformers import AutoTokenizer, AutoConfig
    config = load_config()
    original = load_pool(args.original_pool, "gpqa_main", "PREFLIGHT_ONLY" if args.synthetic else "FORMAL_TEST", config)
    if args.synthetic:
        original = dict(original, rows=original["rows"][:1], test_identities=original["test_identities"][:1])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    require(model_config.max_position_embeddings == MAX_LENGTH and model_config.rope_scaling is None and
            model_config.num_hidden_layers == 36 and model_config.hidden_size == 2560,
            "D8 cached model capacity/architecture differs")
    pool = prepare_pool(original, tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_once(args.output_dir / "thinking-pool.json", pool)
    write_json_once(args.output_dir / "prompt-recipe.json", {"model": MODEL, "revision": REVISION,
        "tokenizer_revision": REVISION, "source_commit": args.commit, "original_pool": str(args.original_pool),
        "input_seed": 20261008, "runtime_seed": SEED, "seed_rule": "20261018+canonical_index",
        "count": len(pool["rows"]), "max_prompt_tokens": max(len(r["input_ids"]) for r in pool["rows"]),
        "max_length": MAX_LENGTH, "max_new_tokens": MAX_NEW_TOKENS, "sampling": SAMPLING,
        "enable_thinking": True, "add_generation_prompt": True, "roles": ["user"],
        "think_open_id": pool["think_open_id"], "think_close_id": pool["think_close_id"],
        "prompt_think_open_count": pool["rows"][0]["input_ids"].count(pool["think_open_id"]),
        "target_gold_loaded": False})
    print(json.dumps({"status": "D8_INPUTS_READY", "count": len(pool["rows"]),
        "max_prompt_tokens": max(len(r["input_ids"]) for r in pool["rows"]), "target_gold_loaded": False}))


def metadata(job, cell, pool):
    return {"schema": "loopscope.phase11.d8.job.v1", "cell": cell,
        **{k: job[k] for k in ("cell_key", "cell_id", "seed", "source_commit", "prompt_mode", "damping_scope",
                              "pool", "run_root", "indices", "synthetic", "engineering")},
        "model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
        "max_length": MAX_LENGTH, "max_new_tokens": MAX_NEW_TOKENS, "sampling": SAMPLING,
        "seed_rule": "20261018+canonical_index", "input_seed": 20261008,
        "think_open_id": pool["think_open_id"], "think_close_id": pool["think_close_id"],
        "target_gold_loaded": False}


def validate_bundle(index, record, telemetry, meta, pool, tokenizer):
    validate_record(record, "gpqa_main")
    row, cell = pool["rows"][index], meta["cell"]
    require(record["canonical_index"] == index and record["identity"] == row["identity"] and
            record["candidate_labels"] == row["choice_labels"] and record["category"] == row["subject"],
            "D8 raw identity/order differs")
    require(record.get("d8_binding") == meta and telemetry.get("d8_binding") == meta and
            record.get("actual_seed") == actual_seed(index) and telemetry.get("actual_seed") == actual_seed(index),
            "D8 record seed/source/cell differs")
    require(record["generated_text"] == tokenizer.decode(record["generated_token_ids"], skip_special_tokens=False),
            "D8 complete decoded response differs from token IDs")
    require(record["final_span"] == answer_span(record["generated_token_ids"], row["input_ids"], tokenizer,
            record["eos_token_ids"]), "D8 final-span extraction differs")
    n = record["generated_token_count"]
    require(record["max_new_tokens"] == MAX_NEW_TOKENS and telemetry.get("prefill_count") == 1 and
            telemetry.get("decode_count") == n - 1 and telemetry.get("prefill_input_tokens") == len(row["input_ids"])
            and telemetry.get("generated_token_count") == n and telemetry.get("cache_growth_closed") is True,
            "D8 generation/cache lifecycle differs")
    if cell["arm"] == "Online":
        online = telemetry.get("online", {})
        applied = {str(t): int(cell["direction_policy"] == "current_t" or t > 0) for t in range(cell["k"])}
        require(online.get("decode_fit_count") == 0 and online.get("directions_released") is True and
                online.get("forward_count") == n and online.get("callback_count") == n * cell["k"] and
                online.get("callbacks_by_t") == {str(t): n for t in range(cell["k"])} and
                online.get("applied_by_t") == {t: a * n for t, a in applied.items()} and
                online.get("prefill_applied_by_t") == applied and
                online.get("decode_applied_by_t") == {t: a * (n - 1) for t, a in applied.items()} and
                online.get("damping_scope") == "full", "D8 loop/direction lifecycle differs")
    else:
        require("online" not in telemetry and cell["window"] is None, "D8 Native contains loop runtime")


def run(args):
    job = read(args.job_file)
    base = load_config()
    cell = validate_job(job, base)
    pool = validate_pool(read(job["pool"]))
    scope = "PREFLIGHT_ONLY" if job["synthetic"] else "FORMAL_TEST"
    require(pool["engineering_synthetic"] == job["synthetic"], "D8 input scope differs")
    select_indices(pool["rows"], scope, shard=job["indices"])
    source = source_provenance(job["source_commit"])
    require(os.environ.get("SLURM_JOB_ID"), "D8 requires a Slurm allocation")
    require(not job["engineering"] or os.environ.get("SLURM_JOB_PARTITION") == "debug", "D8 engineering requires debug")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    meta = metadata(job, cell, pool)
    root = Path(job["run_root"])
    if root.exists():
        require(read(root / "command_args.json") == meta, "D8 resume binding differs")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json_once(root / "command_args.json", meta)
    store = RecordStore(root, job["indices"], lambda i, r, t: validate_bundle(i, r, t, meta, pool, tokenizer))
    if not store.pending:
        store.finalize_records()
        if not (root / "summary.json").exists():
            write_json_once(root / "summary.json", {"status": "RAW_COMPLETE", "count": len(job["indices"]),
                "metadata": meta, "recovered_from_atomic_records": True, "target_gold_loaded": False})
        return 0
    attempt = root / "attempts" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    write_json_once(attempt / "command_args.json", {"metadata": meta, "job_id": os.environ["SLURM_JOB_ID"],
        "pending_indices": store.pending, **source})
    began, timing, index = time.monotonic(), None, store.pending[0]
    try:
        import torch
        seed_compute(torch, SEED)
        config = dict(base, max_length=MAX_LENGTH)
        torch, model, tokenizer, adapter, eos_ids, env = load_runtime(cell, config, "gpqa_main")
        require(model.config.max_position_embeddings == MAX_LENGTH and model.config.rope_scaling is None,
                "D8 runtime native context differs")
        write_json_once(attempt / "env.json", {**env, **source, "metadata": meta,
            "effective_sampling": SAMPLING, "target_gold_loaded": False,
            "environment": {k: v for k, v in os.environ.items() if k.startswith("SLURM_") or k in
                ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH", "HF_HOME", "TRANSFORMERS_OFFLINE")}})
        torch.set_grad_enabled(False)
        timing = ForwardTiming(torch, model, "gpqa_main")
        for index in store.pending:
            store.mark_in_progress(index)
            seed = actual_seed(index)
            effective = seed_compute(torch, seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.monotonic()
            record, telemetry = acquire_row(torch, model, tokenizer, adapter, eos_ids, pool["rows"][index],
                index, cell, config, scope, MAX_NEW_TOKENS, None,
                generation_options=SAMPLING, preserve_special_tokens=True)
            torch.cuda.synchronize()
            record.update(d8_binding=meta, actual_seed=seed,
                final_span=answer_span(record["generated_token_ids"], pool["rows"][index]["input_ids"], tokenizer, eos_ids))
            telemetry.update(timing.summary())
            telemetry.update(d8_binding=meta, actual_seed=seed, effective_rng_seeds=effective,
                elapsed_seconds=time.monotonic()-start, peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
            store.commit(index, record, telemetry)
            print(json.dumps({"cell": job["cell_key"], "completed": len(store.done), "total": len(job["indices"]),
                "generated_tokens": record["generated_token_count"], "target_gold_loaded": False}), flush=True)
        store.finalize_records()
        summary = {"status": "RAW_COMPLETE", "count": len(job["indices"]), "metadata": meta,
            "attempt_root": str(attempt), "elapsed_seconds": time.monotonic()-began, "target_gold_loaded": False}
        write_json_once(root / "summary.json", summary)
        write_json_once(attempt / "summary.json", summary)
    except Exception as exc:
        write_json_once(attempt / "failure.json", {"status": "FAILED", "canonical_index": index,
            "exception_type": type(exc).__name__, "reason": str(exc), "done_count": len(store.done),
            "elapsed_seconds": time.monotonic()-began, "target_gold_loaded": False})
        raise
    finally:
        if timing is not None:
            timing.close()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-file", type=Path)
    p.add_argument("--prepare-inputs", action="store_true")
    p.add_argument("--original-pool", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--commit")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    raise SystemExit(prepare(args) if args.prepare_inputs else run(args))
