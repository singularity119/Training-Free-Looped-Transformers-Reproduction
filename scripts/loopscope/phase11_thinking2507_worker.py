#!/usr/bin/env python3
"""Thinking2507 greedy8192 acquisition; raw records remain gold-free."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

from thinking2507_contract import read, metadata, validate_bundle, seed_compute
from phase11_d9_lingjing_worker import allocation
from run_phase11_accuracy import acquire_row, ForwardTiming
from tflt.loopscope.phase11_checkpoint import RecordStore
from tflt.loopscope.phase11_panel import load_config, require, write_json_once
from tflt.loopscope.phase11_thinking2507 import (
    MODEL, REVISION, SEED, MAX_LENGTH, MAX_NEW_TOKENS, SAMPLING,
    validate_job, validate_pool, actual_seed, answer_span,
)


def load_runtime(model_dir, cell):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one CUDA device per worker")
    require("H20" in torch.cuda.get_device_name(0).upper(), "Thinking2507 requires an H20")
    require(importlib.metadata.version("lm_eval") == "0.4.11", "frozen lm_eval differs")
    require(importlib.metadata.version("transformers") == "4.51.3", "frozen transformers differs")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    require(model.dtype == torch.bfloat16 and model.config.model_type == "qwen3" and
            model.config.num_hidden_layers == 36 and model.config.hidden_size == 2560 and
            model.config.max_position_embeddings == MAX_LENGTH and model.config.rope_scaling is None,
            "model architecture/dtype/native context differs")
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or ())
    require(eos_ids == [151645, 151643], "native EOS differs")
    env = {"model": MODEL, "model_revision": "UNVERIFIED_OSS_COPY", "tokenizer_revision": "UNVERIFIED_OSS_COPY",
           "model_dir": str(model_dir), "python": sys.version, "platform": platform.platform(),
           "dtype": str(model.dtype), "gpu": torch.cuda.get_device_name(0),
           "total_memory": int(torch.cuda.get_device_properties(0).total_memory),
           "native_generation_eos_ids": eos_ids, "cuda": torch.version.cuda,
           "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "lm_eval")}}
    return torch, model, tokenizer, None, eos_ids, env


def run(job_file, model_dir, admission_file):
    alloc = allocation(os.environ)
    job, admission = read(job_file), read(admission_file)
    bundle = read(Path(__file__).resolve().parents[2] / "bundle-info.json")
    require(admission.get("runtime_authorized") is True, "Thinking2507 runtime not authorized")
    require(job["source_commit"] == bundle["source_commit"] == admission["source_commit"],
            "job/admission/bundle source differs")
    provenance = admission["model_provenance"]
    require(admission["model_repo"] == MODEL and
            provenance["mode"] == "USER_DESIGNATED_OSS_UNVERIFIED_REVISION" and
            provenance["model_mount"] == str(model_dir) and
            provenance["actual_revision"] == "UNVERIFIED_OSS_COPY" and
            provenance["actual_tokenizer_revision"] == "UNVERIFIED_OSS_COPY" and
            "/users/270228/" in provenance["model_oss"] and
            provenance["model_oss"].rstrip("/").endswith("Qwen3-4B-Thinking-2507"),
            "Thinking2507 actual OSS asset admission differs")
    cell = validate_job(job)
    pool = validate_pool(read(job["pool"]))
    require(pool.get("thinking2507_model_provenance") == provenance,
            "pool/admission model provenance differs")
    require(pool["engineering_synthetic"] == job["synthetic"], "input scope differs")
    require(job["synthetic"] or (admission.get("gpqa_alignment_receipt") and
            pool.get("thinking2507_source_alignment", {}).get("canonical_source_reused") is True),
            "formal inputs must reuse the original fixed author archive and option builder")
    require(all(0 <= i < len(pool["rows"]) for i in job["indices"]), "indices outside pool")
    root = Path(job["run_root"])
    meta = metadata(job, cell, pool)
    if root.exists():
        require(read(root / "command_args.json") == meta, "resume binding differs")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json_once(root / "command_args.json", meta)
    attempt = root / "attempts" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    write_json_once(attempt / "allocation.json", {**alloc, "metadata": meta,
        "bundle": bundle, "admission": admission, "target_gold_loaded": False})
    started, timing, index = time.monotonic(), None, None
    try:
        torch, model, tokenizer, adapter, eos_ids, env = load_runtime(model_dir, cell)
        env.update(model_revision="UNVERIFIED_OSS_COPY", tokenizer_revision="UNVERIFIED_OSS_COPY",
                   model_provenance=provenance)
        seed_compute(torch, SEED)
        write_json_once(attempt / "env.json", {**env, **alloc, "source_commit": bundle["source_commit"],
            "effective_sampling": SAMPLING, "target_gold_loaded": False})
        store = RecordStore(root, job["indices"], lambda i, r, t: validate_bundle(i, r, t, meta, pool, tokenizer))
        torch.set_grad_enabled(False)
        timing = ForwardTiming(torch, model, "gpqa_main")
        config = dict(load_config(), max_length=MAX_LENGTH)
        for index in store.pending:
            store.mark_in_progress(index)
            effective = seed_compute(torch, actual_seed(index))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            began = time.monotonic()
            record, telemetry = acquire_row(torch, model, tokenizer, adapter, eos_ids,
                pool["rows"][index], index, cell, config,
                "PREFLIGHT_ONLY" if job["synthetic"] else "FORMAL_TEST", MAX_NEW_TOKENS, None,
                generation_options=SAMPLING, preserve_special_tokens=True)
            torch.cuda.synchronize()
            record.update(thinking2507_binding=meta, actual_seed=actual_seed(index),
                final_span=answer_span(record["generated_token_ids"], pool["rows"][index]["input_ids"], tokenizer, eos_ids))
            telemetry.update(timing.summary())
            telemetry.update(thinking2507_binding=meta, actual_seed=actual_seed(index), effective_rng_seeds=effective,
                elapsed_seconds=time.monotonic()-began, peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
            store.commit(index, record, telemetry)
            print(json.dumps({"cell": job["cell_key"], "index": index, "completed": len(store.done),
                "total": len(job["indices"]), "target_gold_loaded": False}), flush=True)
        store.finalize_records()
        summary = {"status": "RAW_COMPLETE", "count": len(job["indices"]), "metadata": meta,
            "attempt_root": str(attempt), "elapsed_seconds": time.monotonic()-started, "target_gold_loaded": False}
        if not (root / "summary.json").exists():
            write_json_once(root / "summary.json", summary)
        write_json_once(attempt / "summary.json", summary)
    except Exception as exc:
        write_json_once(attempt / "failure.json", {"status": "FAILED", "canonical_index": index,
            "exception_type": type(exc).__name__, "reason": str(exc), "target_gold_loaded": False})
        raise
    finally:
        if timing is not None:
            timing.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-file", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--admission", type=Path, required=True)
    a = p.parse_args()
    run(a.job_file, a.model_dir, a.admission)
