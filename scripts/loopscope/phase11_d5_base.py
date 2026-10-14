#!/usr/bin/env python3
"""D5 question checkpoints around immutable Base likelihood producers."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import importlib.metadata
import json
import math
import os
import random
from pathlib import Path
import sys
import time


MODEL = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
NATIVE = "q4-native"
LOOP = "q4-w15-18-k2-loop"
CURRENT = "q4-w15-18-k2-online-current-t-lambda0.9"
FIXED = "q4-w15-18-k2-online-fixed-t0-lambda0.4"
P8 = "2dd3dfd59caf547a3abbc0a1516bc4a3207d9af0"
P8G = "cffc8d3902c92f0c0cc3107b060aefe18c28a9b6"
P10C = "2644034a9d9c806099693e810c1fada9d73cd750"
P10D = "0fa5ea760eae859fe069db7a84d9777a608e55f6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def producer_kind(job):
    family, cid = job["family"], job["cell_id"]
    if family == "p10-base-mmlu":
        require(cid in (NATIVE, LOOP, CURRENT), "cell outside D5 Base MMLU membership")
        return {NATIVE: ("p8", P8, 20260905), LOOP: ("p8g", P8G, 20260905),
                CURRENT: ("p10", P10C, 20261002)}[cid]
    require(family == "p10-base-arc" and cid in (NATIVE, LOOP, FIXED),
            "cell outside D5 Base ARC membership")
    return "p10", P10D, 20261002


def validate_job(job):
    required = {"family", "repeat", "cell_id", "pool", "manifest", "indices", "run_root",
                "source_commit", "producer_source", "producer_commit", "seed", "synthetic"}
    require(required.issubset(job), "D5 Base job is missing required fields")
    require(type(job["repeat"]) is int and job["repeat"] in (1, 2), "repeat must be integer 1/2")
    require(type(job["synthetic"]) is bool, "synthetic must be explicit boolean")
    kind, commit, _ = producer_kind(job)
    require(job["producer_commit"] == commit, "producer commit differs from historical source")
    require(type(job["seed"]) is int and job["seed"] == {1: 20261014, 2: 20261015}[job["repeat"]],
            "effective seed differs from frozen D5 repeat")
    indices = read_json(job["indices"]) if isinstance(job["indices"], str) else job["indices"]
    require(isinstance(indices, list) and indices and
            all(type(i) is int and i >= 0 for i in indices) and indices == sorted(set(indices)),
            "canonical indices must be nonempty sorted unique integers")
    return kind, indices


def load_inputs(job, kind, indices):
    dataset = "mmlu" if job["family"] == "p10-base-mmlu" else "arc_challenge"
    require(not job["synthetic"], "Base debug uses original gold-free pool and fixed indices; synthetic Base inputs are unsupported")
    if kind in ("p8", "p8g"):
        pool, manifest = read_json(job["pool"]), read_json(job["manifest"])
        if kind == "p8":
            from tflt.loopscope.phase8_accuracy import panel
            config = read_json(Path(job["producer_source"]) / "configs/loopscope/phase8_model_windows.json")
            require(manifest["cells"] == panel(config), "original Phase8 manifest differs")
            cell = next(c for c in manifest["cells"] if c["cell_id"] == job["cell_id"])
        else:
            from tflt.loopscope.phase8_gate_g_accuracy import validate_manifest
            validate_manifest(manifest, pool, Path(job["pool"]))
            cell = next(c for c in manifest["cells"] if c["cell_id"] == job["cell_id"])
        from tflt.loopscope.phase8_accuracy import select_rows
        rows = select_rows(pool, cell, "FORMAL_TEST", 0, 14042)
    else:
        from tflt.loopscope.phase10_panel import validate_score_manifest
        from tflt.loopscope.phase10_accuracy import load_pool
        manifest = read_json(job["manifest"])
        scope = "PREFLIGHT_ONLY" if job.get("engineering", False) else "FORMAL_TEST"
        cell = validate_score_manifest(manifest, pool_path=Path(job["pool"]), scope=scope)[job["cell_id"]]
        require(cell["new_in_phase10"], "D5 cannot route a reuse cell through Phase10 producer")
        rows = load_pool(Path(job["pool"]), dataset, scope)
    require(max(indices) < len(rows), "canonical index outside original input population")
    if not job["synthetic"]:
        require(cell["model"] == MODEL and cell["revision"] == REVISION and
                cell["dtype"] == "bfloat16" and cell["cache_strategy"] == "first",
                "historical Base model recipe differs")
        require(cell["arm"] == "Native" or (cell["window"] == [15, 18] and cell["k"] == 2),
                "historical Base window/K differs")
    return dataset, cell, rows


def identity(row, dataset):
    return row["identity"] if dataset == "mmlu" else row["sample_id"]


def binding(job):
    return {k: job[k] for k in ("family", "repeat", "cell_id", "source_commit", "producer_commit",
                               "producer_source", "seed", "synthetic")} | {"effective_seed": job["seed"], "engineering": bool(job.get("engineering", False))}


def record_validator(job, rows, dataset):
    expected = binding(job)
    def validate(index, record, telemetry):
        require(record.get("canonical_index") == index and telemetry.get("canonical_index") == index,
                "checkpoint canonical index differs")
        require(record.get("identity") == identity(rows[index], dataset) and
                telemetry.get("identity") == record["identity"], "checkpoint identity differs")
        require(all(record.get(k) == v and telemetry.get(k) == v for k, v in expected.items()),
                "checkpoint repeat/cell/producer binding differs")
        n = 4 if dataset == "mmlu" else len(rows[index]["choices"])
        require(len(record.get("scores", [])) == n and
                all(math.isfinite(float(v)) for v in record["scores"]),
                "checkpoint lacks all finite candidate scores")
        require(len(record.get("continuation_token_lengths", [])) == n and
                all(type(v) is int and v > 0 for v in record["continuation_token_lengths"]),
                "checkpoint continuation metadata differs")
    return validate


def once(path, value):
    path = Path(path)
    if path.exists():
        require(read_json(path) == value, "existing D5 metadata differs: " + str(path))
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def seed_compute(torch, seed):
    """Explicit D5 compute seed; original input sampling is never repeated."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HistoricalRuntime:
    def __init__(self, job, kind, dataset, cell):
        require(os.environ.get("SLURM_JOB_ID"), "real Base acquisition requires authorized Slurm")
        if job.get("engineering", False):
            require(os.environ.get("SLURM_JOB_PARTITION") == "debug", "D5 preflight requires debug")
        require(importlib.metadata.version("lm_eval") == "0.4.11", "original lm_eval version required")
        self.job, self.kind, self.dataset, self.cell = job, kind, dataset, cell
        load_started = time.monotonic()
        if kind == "p10":
            import torch
            # Seed model initialization too; the immutable loader has no hard seed.
            seed_compute(torch, job["seed"])
            legacy = load_file(Path(job["producer_source"]) / "scripts/loopscope/run_phase10_accuracy.py", "d5_original_p10")
            self.legacy = legacy
            maximum = 32768 if dataset == "arc_challenge" else None
            self.torch, model, tokenizer, self.adapter, self.transform, self.env = legacy.load_runtime(cell, maximum)
            self.torch.set_grad_enabled(False)
            from tflt.wrapper import looped_model
            self.manager = nullcontext(model) if cell["arm"] == "Native" else looped_model(model, legacy.make_loop_config(cell, self.transform))
            if dataset == "mmlu":
                from tflt.loopscope.phase9_adapter import score_choices
            else:
                from tflt.loopscope.phase10_adapter import score_choices
            self.scorer = score_choices
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from lm_eval.models.huggingface import HFLM
            from tflt.loopscope.phase8_adapter import build_hflm_class, score_choices
            from tflt.config import LoopConfig
            from tflt.wrapper import looped_model
            self.torch = torch
            # Match Phase8/8G: seed before tokenizer/model loading.
            torch.set_grad_enabled(False)
            seed_compute(torch, job["seed"])
            tokenizer = AutoTokenizer.from_pretrained(cell["model"], revision=cell["revision"], local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(cell["model"], revision=cell["revision"],
                local_files_only=True, torch_dtype=getattr(torch, cell["dtype"])).to("cuda").eval()
            require(model.config._commit_hash == cell["revision"], "original model revision differs")
            self.transform = None
            self.adapter = build_hflm_class(HFLM)(pretrained=model, tokenizer=tokenizer, batch_size="1",
                                                 trust_remote_code=True, phase8_runtime=None)
            self.manager = nullcontext(model)
            if cell["arm"] == "Loop":
                cfg = LoopConfig(model_alias=cell["model"], window=tuple(cell["window"]), k=cell["k"],
                    alpha=1., beta=0., strategy="damped_euler", iteration_mode="block",
                    cache_strategy=cell["cache_strategy"], decode_mode="bypass", residual_transform=None)
                self.manager = looped_model(model, cfg)
            self.scorer = score_choices
            self.env = {"python": sys.executable, "model_revision": model.config._commit_hash,
                "model_dtype": str(model.dtype), "max_length": int(self.adapter.max_length),
                "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "lm_eval", "datasets")},
                "gpu": torch.cuda.get_device_name(), "total_memory": torch.cuda.get_device_properties(0).total_memory}
        # Freeze computation RNG after model loading as well as before it.
        seed_compute(self.torch, job["seed"])
        require(self.torch.initial_seed() == job["seed"], "effective torch seed differs")
        self.tokenizer = tokenizer
        self.env.update(job_id=os.environ.get("SLURM_JOB_ID"), partition=os.environ.get("SLURM_JOB_PARTITION"),
                        qos=os.environ.get("SLURM_JOB_QOS"), **binding(job))
        self.env["effective_rng_seeds"] = {key: job["seed"] for key in ("python", "numpy", "torch", "cuda_all_devices")}
        self.env["torch_initial_seed"] = self.torch.initial_seed()
        self.torch.cuda.synchronize()
        self.env["model_loading"] = {"elapsed_seconds": time.monotonic() - load_started, **self.peak()}
        self.torch.cuda.reset_peak_memory_stats()

    def synchronize(self):
        self.torch.cuda.synchronize()

    def peak(self):
        return {"peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()), "oom_count": 0}

    def score(self, row, index, dataset):
        ident = identity(row, dataset)
        values = self.scorer(self.adapter, row["prompt"], ident) if dataset == "mmlu" else self.scorer(self.adapter, row["prompt"], ident, row["choices"])
        positions = (self.adapter.phase8_positions if self.kind != "p10" else
                     self.adapter.phase9_positions if dataset == "mmlu" else self.adapter.phase10_positions)
        facts = {}
        if self.kind == "p10":
            facts = self.legacy.check_position_metadata(positions, self.tokenizer)
            if dataset == "arc_challenge":
                self.legacy.check_frozen_arc_tokens(row, self.cell, positions)
            if self.transform is not None:
                facts["model_context_count"] = self.legacy.validate_calls(self.transform, self.cell["k"])
                facts["direction_calls"] = list(self.transform.calls)
                self.transform.calls.clear()
                self.transform.context_summaries.clear()
                if hasattr(self.transform, "diagnostics"):
                    self.transform.diagnostics.clear()
        return [float(v[0]) for v in values], positions, facts


def run(job, runtime_factory=None):
    kind, indices = validate_job(job)
    # The orchestrator sets PYTHONPATH to this historical source. No current tflt
    # package is inserted; only the checkpoint is loaded by its absolute file.
    import tflt.loopscope
    require(Path(tflt.loopscope.__file__).resolve().is_relative_to(Path(job["producer_source"]).resolve()),
            "tflt imports must resolve to immutable historical producer_source")
    dataset, cell, rows = load_inputs(job, kind, indices)
    root = Path(job["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    meta = {"schema": "loopscope.phase11.d5.base_attempt.v1", "job": job, "cell": cell,
            "dataset": dataset, "indices": indices, "target_gold_loaded": False}
    once(root / "command_args.json", meta)
    checkpoint_path = Path(__file__).resolve().parents[2] / "src/tflt/loopscope/phase11_checkpoint.py"
    store = load_file(checkpoint_path, "d5_controller_checkpoint").RecordStore(root, indices, record_validator(job, rows, dataset))
    started = time.monotonic()
    runtime = None
    if store.pending:
        runtime = (runtime_factory(job, kind, dataset, cell) if runtime_factory else HistoricalRuntime(job, kind, dataset, cell))
        # Retain environment per process; a resumed attempt can use a different GPU.
        env_path = root / ("env-%s.json" % time.time_ns())
        once(env_path, {**runtime.env, **binding(job)})
        complete_count = len(store.done)
        with runtime.manager:
            for index in store.pending:
                store.mark_in_progress(index)
                runtime.synchronize()
                tick = time.monotonic()
                scores, positions, facts = runtime.score(rows[index], index, dataset)
                runtime.synchronize()
                elapsed = time.monotonic() - tick
                row = rows[index]
                choices = list("ABCD") if dataset == "mmlu" else row["choices"]
                record = {**binding(job), "canonical_index": index, "identity": identity(row, dataset),
                    "scores": scores, "choice_text_lengths": [len(c) for c in choices],
                    "continuation_token_lengths": [int(p["continuation_length"]) for p in positions]}
                if dataset == "mmlu":
                    record.update(subject=row["subject"], doc_index=row["doc_index"])
                telemetry = {**binding(job), "canonical_index": index, "identity": record["identity"],
                    "elapsed_seconds": elapsed, "context_length": positions[0]["position"] + 1,
                    "max_input_length": max(p["input_length"] for p in positions),
                    "position_metadata": [{key: p[key] for key in ("position", "input_length", "continuation_length", "left_truncated_tokens")} for p in positions],
                    "mechanism": facts, "environment_file": str(env_path), **runtime.peak()}
                store.commit(index, record, telemetry)
                complete_count += 1
                if complete_count % 32 == 0:
                    print(json.dumps({"cell_id": job["cell_id"], "completed": complete_count, "total": len(indices)}), flush=True)
    records, telemetry = store.finalize_records()
    done = store.done
    store._materialize(root / "scores.jsonl", [done[i]["record"] for i in indices])
    peaks = {key: max(bundle["telemetry"].get(key, 0) for bundle in done.values())
             for key in ("peak_allocated_bytes", "peak_reserved_bytes")}
    summary = {"schema": "loopscope.phase11.d5.base_summary.v1", "status": "SCORES_COMPLETE",
               "metadata": meta, "count": len(indices), "target_gold_loaded": False,
               "records": str(records), "telemetry": str(telemetry), "oom_count": 0, **peaks}
    once(root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "count": len(indices), "cell_id": job["cell_id"],
                      "synthetic": job["synthetic"], "process_elapsed_seconds": time.monotonic() - started}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, required=True)
    return run(read_json(parser.parse_args(argv).job_file))


if __name__ == "__main__":
    raise SystemExit(main())
