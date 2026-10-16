#!/usr/bin/env python3
"""D7 fresh MMLU-Pro acquisition and tokenizer-only frozen prompt preparation."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import random
import time
import uuid

from tflt.loopscope.phase11_accuracy import load_pool, select_indices, validate_record
from tflt.loopscope.phase11_checkpoint import RecordStore
from tflt.loopscope.phase11_panel import build_panel, load_config, require, write_json_once, MODEL, REVISION
from run_phase11_accuracy import acquire_row, load_runtime, source_provenance, ForwardTiming

SEED = 20261016
TAG_FIELDS = ("cell_key", "cell_id", "seed", "source_commit", "prompt_mode", "damping_scope")
SUFFIXES = {"native": "native", "loop_k2": "w15-18-k2-loop", "loop_k3": "w15-18-k3-loop",
    "current_t_k2_l05": "w15-18-k2-online-current-t-lambda0.5",
    "fixed_t0_k3_l09": "w15-18-k3-online-fixed-t0-lambda0.9"}


def selected_cells(config=None):
    originals = {c["cell_id"]: c for c in build_panel("mmlu_pro", config)["cells"]}
    result = {}
    for name, suffix in SUFFIXES.items():
        online = name in ("current_t_k2_l05", "fixed_t0_k3_l09")
        cell = copy.deepcopy(originals["mmlu_pro-q4-" + suffix])
        cell.update(cell_key=name, prompt_mode="cot", damping_scope="full" if online else "none")
        result[name] = cell
    return result


def validate_job(job, config=None):
    require(type(job.get("seed")) is int and job["seed"] == SEED, "D7 runtime seed differs")
    require(type(job.get("synthetic")) is bool and type(job.get("engineering")) is bool,
            "D7 requires explicit synthetic/engineering bools")
    require(not job["synthetic"] or job["engineering"], "D7 synthetic requires engineering scope")
    cells = selected_cells(config)
    require(job.get("cell_key") in cells, "D7 cell outside frozen 5-cell panel")
    cell = cells[job["cell_key"]]
    require(all(job.get(k) == cell[k] for k in ("cell_id", "prompt_mode", "damping_scope")),
            "D7 cell/prompt/scope binding differs")
    for key in ("pool", "run_root", "source_commit"):
        require(isinstance(job.get(key), str) and bool(job[key]), "D7 job lacks " + key)
    return cell


def load_job(job):
    config = load_config()
    cell = validate_job(job, config)
    scope = "PREFLIGHT_ONLY" if job["synthetic"] else "FORMAL_TEST"
    pool = load_pool(Path(job["pool"]), "mmlu_pro", scope, config)
    require(pool.get("prompt_mode") == job["prompt_mode"], "D7 prepared pool prompt binding differs")
    require(isinstance(job.get("indices"), list), "D7 explicit canonical indices required")
    indices = select_indices(pool["rows"], scope, shard=job["indices"])
    return scope, config, cell, pool, indices


def binding(job, scope, config, cell, indices):
    return {"schema": "loopscope.phase11.d7.job.v1", **{k: job[k] for k in TAG_FIELDS},
        "cell": cell, "pool": job["pool"], "run_root": job["run_root"], "indices": indices,
        "dataset": "mmlu_pro", "scope": scope, "synthetic": job["synthetic"],
        "engineering": job["engineering"], "max_length": config["max_length"],
        "input_seed": 20261008, "max_new_tokens": 3 if job["synthetic"] else 8192,
        "target_gold_loaded": False}


def validate_bundle(index, record, telemetry, metadata, pool):
    validate_record(record, "mmlu_pro")
    row, cell = pool["rows"][index], metadata["cell"]
    require(record["canonical_index"] == index and record["identity"] == row["identity"] and
        record["category"] == row["subject"] and record["candidate_labels"] == row["choice_labels"],
        "D7 record identity/order/category differs")
    tag = {key: metadata[key] for key in TAG_FIELDS}
    require(record.get("d7_binding") == tag and telemetry.get("d7_binding") == tag,
            "D7 checkpoint cell/source/prompt/scope binding differs")
    require(telemetry.get("identity") == row["identity"] and telemetry.get("canonical_index") == index,
            "D7 telemetry identity differs")
    require(record["max_new_tokens"] == metadata["max_new_tokens"] and
        telemetry.get("prefill_count") == 1 and telemetry.get("decode_count") == record["generated_token_count"] - 1 and
        telemetry.get("prefill_input_tokens") == len(row["input_ids"]) and
        telemetry.get("generated_token_count") == record["generated_token_count"] and
        telemetry.get("cache_growth_closed") is True and telemetry.get("incremental_generation") is True,
        "D7 bound/cache lifecycle differs")
    if cell["arm"] == "Online":
        online = telemetry.get("online", {})
        forwards = record["generated_token_count"]
        applied = {str(t): int(cell["direction_policy"] == "current_t" or t > 0) for t in range(cell["k"])}
        expected = {str(t): int(cell["direction_policy"] == "current_t" or t > 0) *
                    (1 if cell["damping_scope"] == "prefill_only" else forwards) for t in range(cell["k"])}
        require(online.get("decode_fit_count") == 0 and online.get("directions_released") is True and
            online.get("applied_by_t") == expected and online.get("forward_count") == forwards and
            online.get("callback_count") == forwards * cell["k"] and
            online.get("callbacks_by_t") == {str(t): forwards for t in range(cell["k"])} and
            online.get("damping_scope") == cell["damping_scope"] and
            online.get("prefill_applied_by_t") == applied and
            online.get("decode_applied_by_t") == {t: n * (0 if cell["damping_scope"] == "prefill_only"
                else forwards - 1) for t, n in applied.items()} and
            online.get("prefill_callback_count") == cell["k"] and
            online.get("decode_callback_count") == (forwards - 1) * cell["k"],
            "D7 damping scope or callback/direction lifecycle differs")
    else:
        require("online" not in telemetry, "D7 control unexpectedly has online runtime")


def seed_compute(torch, seed=SEED):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {key: seed for key in ("python", "numpy", "torch", "cuda_all_devices")}


def prepare_inputs(args):
    from transformers import AutoTokenizer
    from tflt.loopscope.phase11_data import SCHEMA, render_generation_prompt, chat_input
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, trust_remote_code=True)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    if args.synthetic:
        choices = ["synthetic option " + c for c in "ABCDEFGHIJ"]
        demos = [{"question": "Synthetic demonstration " + str(i), "options": choices,
            "category": "synthetic", "cot_content": "Answer: Let's think step by step. Synthetic reasoning.",
            "answer": "A"} for i in range(5)]
        target = {"question": "Synthetic engineering input only: choose the first option.",
            "options": choices, "category": "synthetic"}
        row = {"identity": "phase11-d7-synthetic:mmlu_pro:0", "sample_id": "phase11-d7-synthetic:mmlu_pro:0",
            "task": "mmlu_pro", "split": "synthetic", "source_index": 0, "source_id": 0,
            "subject": "synthetic", "question": target["question"], "choices": choices,
            "choice_labels": list("ABCDEFGHIJ"), "fewshot_sample_ids": ["synthetic-demo:" + str(i) for i in range(5)]}
        row.update(chat_input(render_generation_prompt(target, demos), tokenizer, config["max_length"]))
        pool = {"schema_version": SCHEMA, "task": "mmlu_pro", "model": MODEL,
            "model_revision": REVISION, "tokenizer_revision": REVISION, "dataset_revision": None,
            "scope": "PREFLIGHT_ONLY", "engineering_synthetic": True, "split": "synthetic",
            "target_gold_loaded": False, "rows": [row], "test_identities": [row["identity"]]}
    else:
        pool = load_pool(args.original_pool, "mmlu_pro", "FORMAL_TEST", config)
        for row in pool["rows"]:
            require(tokenizer.encode(row["prompt"], add_special_tokens=False) == row["input_ids"],
                    "D7 cached tokenization differs from original Gate A prompt")
            require(len(row["input_ids"]) + 8192 <= config["max_length"], "D7 complete 8192 context exceeds bound")
        require(len({r["subject"] for r in pool["rows"]}) == 14, "D7 category count differs")
    pool = dict(pool, prompt_mode="cot")
    write_json_once(root / "cot-pool.json", pool)
    lengths = sorted(len(r["input_ids"]) for r in pool["rows"])
    write_json_once(root / "prompt-recipe.json", {"schema": "loopscope.phase11.d7.prompts.v1",
        "source_commit": args.commit, "original_pool": str(args.original_pool), "synthetic": args.synthetic,
        "count": len(pool["rows"]), "max_prompt_tokens": lengths[-1], "min_prompt_tokens": lengths[0],
        "runtime_seed": SEED, "input_seed": 20261008, "max_new_tokens": 8192,
        "model": MODEL, "tokenizer_revision": REVISION, "prompt_and_ids_unchanged": not args.synthetic,
        "target_gold_loaded": False})
    print(json.dumps({"status": "D7_INPUTS_PREPARED", "count": len(pool["rows"]),
        "max_prompt_tokens": lengths[-1], "target_gold_loaded": False}))
    return 0


def run(args):
    job = json.loads(args.job_file.read_text(encoding="utf-8"))
    scope, config, cell, pool, indices = load_job(job)
    source = source_provenance(job["source_commit"])
    require(os.environ.get("SLURM_JOB_ID"), "D7 requires allocation")
    require(not job["engineering"] or os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "D7 engineering requires debug")
    metadata = binding(job, scope, config, cell, indices)
    root = Path(job["run_root"])
    if root.exists():
        require(json.loads((root / "command_args.json").read_text()) == metadata, "D7 resume binding differs")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json_once(root / "command_args.json", metadata)
    store = RecordStore(root, indices, lambda i, r, t: validate_bundle(i, r, t, metadata, pool))
    if not store.pending:
        store.finalize_records()
        summary = {"schema": "loopscope.phase11.d7.summary.v1", "status": "RAW_COMPLETE", "metadata": metadata,
                   "count": len(indices), "target_gold_loaded": False, "recovered_from_atomic_records": True}
        if not (root / "summary.json").exists():
            write_json_once(root / "summary.json", summary)
        saved = json.loads((root / "summary.json").read_text())
        require(saved["status"] == "RAW_COMPLETE" and saved["metadata"] == metadata and saved["count"] == len(indices),
                "D7 complete summary differs")
        return 0
    attempt = root / "attempts" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    write_json_once(attempt / "command_args.json", {"metadata": metadata, "job_id": os.environ["SLURM_JOB_ID"],
        "pending_indices": store.pending, **source})
    started, timing, index = time.monotonic(), None, store.pending[0]
    try:
        import torch
        seed_compute(torch)
        torch, model, tokenizer, adapter, eos_ids, env = load_runtime(cell, config, "mmlu_pro")
        effective = seed_compute(torch)
        captured = {key: value for key, value in os.environ.items() if key.startswith("SLURM_") or key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "TOKENIZERS_PARALLELISM", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "HF_HOME", "HF_DATASETS_CACHE",
            "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}
        write_json_once(attempt / "env.json", {**env, **source, "environment": captured, "metadata": metadata,
            "effective_rng_seeds": effective, "target_gold_loaded": False})
        torch.set_grad_enabled(False)
        torch.cuda.reset_peak_memory_stats()
        timing = ForwardTiming(torch, model, "mmlu_pro")
        for index in store.pending:
            store.mark_in_progress(index)
            torch.cuda.synchronize()
            began = time.monotonic()
            record, telemetry = acquire_row(torch, model, tokenizer, adapter, eos_ids, pool["rows"][index],
                index, cell, config, scope, metadata["max_new_tokens"], 3 if job["synthetic"] else None)
            torch.cuda.synchronize()
            tag = {key: metadata[key] for key in TAG_FIELDS}
            record["d7_binding"] = tag
            telemetry.update(timing.summary())
            telemetry.update(d7_binding=tag, identity=pool["rows"][index]["identity"], canonical_index=index,
                elapsed_seconds=time.monotonic() - began, peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
            store.commit(index, record, telemetry)
        store.finalize_records()
        summary = {"schema": "loopscope.phase11.d7.summary.v1", "status": "RAW_COMPLETE", "metadata": metadata,
            "count": len(indices), "target_gold_loaded": False, "attempt_root": str(attempt),
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        write_json_once(root / "summary.json", summary)
        write_json_once(attempt / "summary.json", summary)
    except Exception as exc:
        write_json_once(attempt / "failure.json", {"status": "FAILED", "canonical_index": index,
            "exception_type": type(exc).__name__, "reason": str(exc), "metadata": metadata,
            "done_count": len(store.done), "elapsed_seconds": time.monotonic() - started, "target_gold_loaded": False})
        raise
    finally:
        if timing is not None:
            timing.close()
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-file", type=Path)
    p.add_argument("--prepare-inputs", action="store_true")
    p.add_argument("--original-pool", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--commit")
    p.add_argument("--synthetic", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.prepare_inputs:
        require(args.original_pool and args.output_dir and args.commit and not args.job_file,
                "prepare-inputs requires original-pool/output-dir/commit and no job-file")
        raise SystemExit(prepare_inputs(args))
    require(args.job_file is not None, "job-file required")
    raise SystemExit(run(args))
