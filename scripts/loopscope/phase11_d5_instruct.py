#!/usr/bin/env python3
"""D5 fresh Instruct repeats, retaining original per-question science."""
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
from tflt.loopscope.phase11_panel import build_panel, load_config, require, write_json_once
from run_phase11_accuracy import acquire_row, load_runtime, source_provenance, ForwardTiming

FAMILIES = {"p11-instruct-arc": "arc_challenge", "p11-instruct-gpqa": "gpqa_main"}
SUFFIXES = {
    "arc_challenge": ("native", "w15-18-k2-loop", "w15-18-k2-online-fixed-t0-lambda0.9"),
    "gpqa_main": ("native", "w15-18-k2-loop", "w15-18-k3-loop",
                  "w15-18-k2-online-current-t-lambda0.5", "w15-18-k3-online-fixed-t0-lambda0.9"),
}


def selected_cells(dataset, config=None):
    require(dataset in SUFFIXES, "D5 Instruct dataset differs")
    allowed = {dataset + "-q4-" + suffix for suffix in SUFFIXES[dataset]}
    return {cell["cell_id"]: cell for cell in build_panel(dataset, config)["cells"]
            if cell["cell_id"] in allowed}


def load_job(job):
    require(job.get("family") in FAMILIES, "D5 Instruct family differs")
    dataset = FAMILIES[job["family"]]
    require(type(job.get("repeat")) is int and job["repeat"] in (1, 2), "D5 repeat must be integer 1 or 2")
    require(type(job.get("synthetic")) is bool, "D5 job requires explicit synthetic bool")
    require(type(job.get("seed")) is int and job["seed"] == {1: 20261014, 2: 20261015}[job["repeat"]],
            "D5 job seed differs from frozen repeat seed")
    require(type(job.get("engineering", False)) is bool, "D5 engineering flag must be bool")
    scope = "PREFLIGHT_ONLY" if job["synthetic"] else "FORMAL_TEST"
    config = load_config()
    cells = selected_cells(dataset, config)
    require(job.get("cell_id") in cells, "D5 cell is outside frozen subset")
    for key in ("pool", "run_root", "source_commit"):
        require(isinstance(job.get(key), str) and bool(job[key]), "D5 job lacks " + key)
    pool = load_pool(Path(job["pool"]), dataset, scope, config)
    require(isinstance(job.get("indices"), list), "D5 requires explicit canonical indices")
    indices = select_indices(pool["rows"], scope, shard=job["indices"])
    return dataset, scope, config, cells[job["cell_id"]], pool, indices


def binding(job, dataset, scope, config, cell, indices):
    return {"schema": "loopscope.phase11.d5_instruct_job.v1", "family": job["family"],
            "repeat": job["repeat"], "cell_id": cell["cell_id"], "cell": cell,
            "pool": job["pool"], "run_root": job["run_root"], "source_commit": job["source_commit"],
            "indices": indices, "synthetic": job["synthetic"], "scope": scope, "dataset": dataset,
            "engineering": job.get("engineering", False),
            "seed": job["seed"], "max_length": config["max_length"],
            "max_new_tokens": 3 if job["synthetic"] and dataset == "gpqa_main" else
                              (8192 if dataset == "gpqa_main" else config["max_new_tokens"]),
            "target_gold_loaded": False}


def validate_bundle(index, record, telemetry, metadata, pool):
    dataset, cell = metadata["dataset"], metadata["cell"]
    validate_record(record, dataset)
    row = pool["rows"][index]
    require(record["canonical_index"] == index and record["identity"] == row["identity"] and
            record["category"] == row["subject"] and record["candidate_labels"] == row["choice_labels"],
            "D5 record identity/order/category differs")
    expected = {key: metadata[key] for key in ("family", "repeat", "cell_id", "source_commit", "seed")}
    require(record.get("d5_binding") == expected and telemetry.get("d5_binding") == expected,
            "D5 checkpoint family/repeat/cell/source binding differs")
    require(telemetry.get("identity") == row["identity"] and telemetry.get("canonical_index") == index,
            "D5 telemetry identity differs")
    if dataset == "arc_challenge":
        require(record["choice_text_lengths"] == [len(choice) for choice in row["choices"]] and
                telemetry.get("candidate_count") == len(row["choices"]) and
                telemetry.get("left_truncated_count") == 0 and telemetry.get("incremental_generation") is False,
                "D5 ARC original candidates/lifecycle differ")
        frozen = row.get("tokenization", {}).get(cell["model"])
        if frozen is not None:
            require(record["continuation_token_lengths"] == frozen["continuation_lengths"],
                    "D5 ARC continuation tokenization differs")
    else:
        require(record["max_new_tokens"] == metadata["max_new_tokens"] and
                telemetry.get("prefill_count") == 1 and
                telemetry.get("decode_count") == record["generated_token_count"] - 1 and
                telemetry.get("prefill_input_tokens") == len(row["input_ids"]) and
                telemetry.get("generated_token_count") == record["generated_token_count"] and
                telemetry.get("cache_growth_closed") is True and telemetry.get("incremental_generation") is True,
                "D5 GPQA frozen bound/cache lifecycle differs")
    if cell["arm"] == "Online":
        online = telemetry.get("online", {})
        require(online.get("decode_fit_count") == 0 and online.get("directions_released") is True,
                "D5 online direction lifecycle differs")
    else:
        require("online" not in telemetry, "D5 Native/Loop unexpectedly has online runtime")


def run(args):
    job = json.loads(args.job_file.read_text(encoding="utf-8"))
    dataset, scope, config, cell, pool, indices = load_job(job)
    source = source_provenance(job["source_commit"])
    require(os.environ.get("SLURM_JOB_ID"), "D5 requires allocation")
    require(not (job["synthetic"] or job.get("engineering", False)) or
            os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "D5 synthetic preflight requires debug")
    metadata = binding(job, dataset, scope, config, cell, indices)
    root = Path(job["run_root"])
    if root.exists():
        require(json.loads((root / "command_args.json").read_text()) == metadata,
                "D5 resume root binding differs")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json_once(root / "command_args.json", metadata)
    store = RecordStore(root, indices,
                        lambda i, r, t: validate_bundle(i, r, t, metadata, pool))
    if not store.pending:
        store.finalize_records()
        summary_path = root / "summary.json"
        if not summary_path.exists():
            write_json_once(summary_path, {"schema": "loopscope.phase11.d5_instruct_summary.v1",
                "status": "RAW_COMPLETE", "metadata": metadata, "count": len(indices),
                "target_gold_loaded": False, "recovered_from_atomic_records": True})
        summary = json.loads(summary_path.read_text())
        require(summary["status"] == "RAW_COMPLETE" and summary["metadata"] == metadata and
                summary["count"] == len(indices), "complete D5 root summary differs")
        return 0
    attempt = root / "attempts" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                                  + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    write_json_once(attempt / "command_args.json", {"metadata": metadata, "job_id": os.environ["SLURM_JOB_ID"],
                                                   "pending_indices": store.pending, **source})
    started = time.monotonic()
    timing = None
    index = store.pending[0]
    try:
        import numpy as np
        import torch
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        torch.cuda.manual_seed_all(job["seed"])
        effective_rng_seeds = {"python": job["seed"], "numpy": job["seed"],
                               "torch": job["seed"], "cuda_all_devices": job["seed"]}
        torch, model, tokenizer, adapter, eos_ids, env = load_runtime(cell, config, dataset)
        captured = {key: value for key, value in os.environ.items() if key.startswith("SLURM_") or key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
            "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}
        write_json_once(attempt / "env.json", {**env, **source, "environment": captured,
            "metadata": metadata, "effective_rng_seeds": effective_rng_seeds, "target_gold_loaded": False})
        torch.set_grad_enabled(False)
        torch.cuda.reset_peak_memory_stats()
        timing = ForwardTiming(torch, model, dataset)
        for index in store.pending:
            store.mark_in_progress(index)
            torch.cuda.synchronize()
            began = time.monotonic()
            record, telemetry = acquire_row(torch, model, tokenizer, adapter, eos_ids, pool["rows"][index],
                index, cell, config, scope, metadata["max_new_tokens"],
                3 if job["synthetic"] and dataset == "gpqa_main" else None)
            torch.cuda.synchronize()
            tag = {key: metadata[key] for key in ("family", "repeat", "cell_id", "source_commit", "seed")}
            record["d5_binding"] = tag
            telemetry.update(timing.summary())
            telemetry.update(d5_binding=tag, identity=pool["rows"][index]["identity"], canonical_index=index,
                elapsed_seconds=time.monotonic() - began,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
            store.commit(index, record, telemetry)
        store.finalize_records()
        summary = {"schema": "loopscope.phase11.d5_instruct_summary.v1", "status": "RAW_COMPLETE",
                   "metadata": metadata, "count": len(indices), "target_gold_loaded": False,
                   "attempt_root": str(attempt), "elapsed_seconds": time.monotonic() - started,
                   "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                   "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        write_json_once(root / "summary.json", summary)
        write_json_once(attempt / "summary.json", summary)
    except Exception as exc:
        write_json_once(attempt / "failure.json", {"status": "FAILED", "canonical_index": index,
            "exception_type": type(exc).__name__, "reason": str(exc), "metadata": metadata,
            "done_count": len(store.done), "elapsed_seconds": time.monotonic() - started,
            "target_gold_loaded": False})
        raise
    finally:
        if timing is not None:
            timing.close()
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-file", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
