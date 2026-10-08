#!/usr/bin/env python3
"""Acquire one frozen Phase 11 cell without opening target gold.

Generation uses the already rendered/tokenized native chat prompt in the
input pool. ARC retains native full-candidate HFLM scoring. Every question
gets a fresh runtime and releases its own prefill directions at completion.
"""
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

from tflt.loopscope.phase11_accuracy import (
    arc_record, engineering_cell, failure_record, generation_record, load_pool,
    select_indices,
)
from tflt.loopscope.phase11_panel import (
    DATASETS, SCOPES, load_config, require, validate_score_manifest, write_json_once,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--dataset", choices=DATASETS, required=True)
    result.add_argument("--cell-id", required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--scope", choices=SCOPES, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--shard", help="zero-based INDEX/COUNT; fixed canonical modulo partition")
    result.add_argument("--engineering-check", choices=("zero-strength", "k2-policy"))
    result.add_argument("--engineering-generate-steps", type=int, choices=(3,),
                        help="PREFLIGHT_ONLY synthetic min=max3; formal generation always uses 2048")
    result.add_argument("--dry-run", action="store_true")
    return result


def load_inputs(args):
    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = validate_score_manifest(manifest, config, args.pool, args.scope)
    require(manifest["dataset"] == args.dataset and args.cell_id in cells,
            "requested dataset or independent cell differs from manifest")
    cell = cells[args.cell_id]
    acquisition = engineering_cell(cell, args.engineering_check, args.scope)
    pool = load_pool(args.pool, args.dataset, args.scope, config)
    rows = pool["rows"]
    shard = None
    shard_metadata = None
    if args.shard is not None:
        parts = args.shard.split("/")
        require(len(parts) == 2 and all(part.isdecimal() for part in parts), "shard must be zero-based INDEX/COUNT")
        index, count = map(int, parts)
        require(0 <= index < count <= len(rows), "shard index/count is outside canonical population")
        shard = [i for i in range(len(rows)) if i % count == index]
        shard_metadata = {"index": index, "count": count, "rule": "canonical_index_modulo_count"}
    indices = select_indices(rows, args.scope, shard=shard)
    if args.engineering_generate_steps is not None:
        require(args.scope == "PREFLIGHT_ONLY" and pool["engineering_synthetic"] is True and
                args.dataset != "arc_challenge", "three-token generation control requires a synthetic generation preflight")
    return config, manifest, cell, acquisition, pool, indices, shard_metadata


def source_provenance(expected_commit):
    """Read Git HEAD/refs directly so compute jobs do not need a Git executable.

    Submitter admission owns clean-tree verification. This check prevents an
    allocated job from loading models from a different source revision.
    """
    root = Path(__file__).resolve().parents[2]
    git_dir = root / ".git"
    if git_dir.is_file():
        declaration = git_dir.read_text(encoding="utf-8").strip()
        require(declaration.startswith("gitdir: "), "source .git pointer is invalid")
        git_dir = (root / declaration.removeprefix("gitdir: ")).resolve()
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    require(head == "ref: refs/heads/loopscope", "Phase 11 execution source must be the loopscope branch")
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text(encoding="utf-8").strip()).resolve()
    ref = common / "refs/heads/loopscope"
    observed = ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
    if observed is None and (common / "packed-refs").is_file():
        for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[1] == "refs/heads/loopscope":
                observed = fields[0]
                break
    require(isinstance(expected_commit, str) and bool(expected_commit) and observed == expected_commit,
            "source Git HEAD differs from --commit; stop before model loading")
    return {"source_root": str(root), "source_branch": "loopscope", "source_head": observed}


def make_loop_config(cell, runtime):
    from tflt.config import LoopConfig
    require(cell["cache_strategy"] == "first" and cell["decode_mode"] == "full", "Phase 11 cache/decode mode differs")
    return LoopConfig(model_alias=cell["model"], window=tuple(cell["window"]), k=cell["k"],
                      iteration_mode="block", strategy="damped_euler", alpha=1.0, beta=0.0,
                      cache_strategy="first", decode_mode="full", residual_transform=runtime)


def load_runtime(cell, config, dataset):
    """Heavy libraries and cached weights are reached only by the allocated path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.cuda.is_available(), "Phase 11 model execution requires CUDA")
    require(importlib.metadata.version("lm_eval") == "0.4.11", "Phase 11 requires lm_eval 0.4.11")
    tokenizer = AutoTokenizer.from_pretrained(cell["model"], revision=cell["tokenizer_revision"],
                                             local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cell["model"], revision=cell["revision"],
                local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    require(getattr(model.config, "_commit_hash", None) in (None, cell["revision"]), "cached model revision differs")
    require(model.dtype == torch.bfloat16, "model dtype differs from frozen BF16")
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or ())
    require(eos_ids == [151645, 151643], "cached native EOS/EOT ids differ from frozen Qwen revision")
    adapter = None
    if dataset == "arc_challenge":
        from lm_eval.models.huggingface import HFLM
        from tflt.loopscope.phase11_adapter import build_hflm_class
        adapter = build_hflm_class(HFLM)(pretrained=model, tokenizer=tokenizer, batch_size="1",
                    trust_remote_code=True, max_length=config["max_length"])
    env = {"python": sys.executable, "platform": platform.platform(), "model_revision": cell["revision"],
           "tokenizer_revision": cell["tokenizer_revision"], "dtype": str(model.dtype),
           "native_generation_eos_ids": eos_ids, "max_length": config["max_length"],
           "gpu": torch.cuda.get_device_name(), "total_memory": int(torch.cuda.get_device_properties(0).total_memory),
           "job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
           "qos": os.environ.get("SLURM_JOB_QOS"),
           "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "lm_eval")}}
    return torch, model, tokenizer, adapter, eos_ids, env


def _check_arc_metadata(row, positions, cell, scope):
    require(len(positions) == len(row["choices"]), "ARC candidate metadata is incomplete")
    require(all(item["left_truncated_tokens"] == 0 for item in positions), "ARC truncation is forbidden")
    require(len({item["prefix_tokens"] for item in positions}) == 1, "ARC candidates do not retain a common prompt")
    tokenizations = row.get("tokenization", {})
    if not tokenizations:
        require(scope == "PREFLIGHT_ONLY", "formal ARC input lacks frozen candidate tokenization")
        return
    require(cell["model"] in tokenizations, "ARC frozen tokenization lacks the bound model")
    frozen = tokenizations[cell["model"]]
    require([item["continuation_length"] for item in positions] == frozen["continuation_lengths"],
            "ARC continuation tokenization differs from frozen inputs")
    require(len(frozen["candidates"]) == len(positions), "ARC frozen candidate metadata is incomplete")
    for actual, expected in zip(positions, frozen["candidates"]):
        require(all(actual[key] == expected[key] for key in (
            "position", "input_length", "context_length", "continuation_length", "left_truncated_tokens")),
            "ARC retained prompt/candidate lengths differ from frozen inputs")
        require(len(actual["effective_valid_positions"]) == expected["valid_prefix_count"],
                "ARC retained prompt direction rows differ from frozen inputs")


def runtime_telemetry(runtime):
    from tflt.loopscope.phase11_runtime import validate_calls
    forwards = validate_calls(runtime)
    require(runtime._active is None and runtime._forward is None and not runtime.directions,
            "question left an active runtime or direction tensor")
    fits = {str(t): 0 for t in range(runtime.k)}
    calls = {str(t): 0 for t in range(runtime.k)}
    applied = {str(t): 0 for t in range(runtime.k)}
    decode_fit_count = 0
    for call in runtime.calls:
        t = str(call["t"])
        calls[t] += 1
        applied[t] += int(call["applied"])
        if call["direction_fit_t"] is not None:
            fits[str(call["direction_fit_t"])] += 1
            decode_fit_count += int(call["forward_kind"] == "decode")
    require(decode_fit_count == 0, "incremental decode fitted a new direction")
    return {"forward_count": forwards, "callback_count": len(runtime.calls), "callbacks_by_t": calls,
            "applied_by_t": applied, "direction_fits_by_t": fits, "decode_fit_count": decode_fit_count,
            "directions_released": True,
            "last_prefill_fit_metadata": {str(t): dict(value) for t, value in runtime._last_fit_metadata.items()}}


def acquire_row(torch, model, tokenizer, adapter, eos_ids, row, index, cell, config, scope, max_new_tokens,
                engineering_steps):
    from tflt.wrapper import looped_model
    from tflt.loopscope.phase11_adapter import generation_forward_context
    from tflt.loopscope.phase11_runtime import make_runtime
    runtime = make_runtime(cell)
    manager = nullcontext() if cell["arm"] == "Native" else looped_model(model, make_loop_config(cell, runtime))
    if row["task"] == "arc_challenge":
        adapter.phase11_runtime = runtime
        try:
            with manager:
                scores = adapter.score_choices(row["prompt"], row["identity"], row["choices"])
            positions = adapter.phase11_positions
            _check_arc_metadata(row, positions, cell, scope)
            record = arc_record(row, scores, positions, index)
            telemetry = {"candidate_count": len(positions), "max_input_tokens": max(item["input_length"] for item in positions),
                         "left_truncated_count": 0, "incremental_generation": False}
        finally:
            adapter.phase11_runtime = None
    else:
        tokens = row["input_ids"]
        require(tokenizer.encode(row["prompt"], add_special_tokens=False) == tokens,
                "actual cached tokenizer differs from complete frozen chat input IDs")
        require(len(tokens) + max_new_tokens <= config["max_length"], "untruncated prompt exceeds bound context")
        inputs = torch.tensor([tokens], dtype=torch.long, device=model.device)
        attention = torch.ones_like(inputs)
        options = {"do_sample": False, "num_beams": 1, "use_cache": True, "max_new_tokens": max_new_tokens}
        if engineering_steps is not None:
            options["min_new_tokens"] = engineering_steps
        with manager, generation_forward_context(model, runtime, row["identity"], tokens,
                    attention_mask=[1] * len(tokens), special_ids=tokenizer.all_special_ids,
                    pad_token_id=tokenizer.pad_token_id) as lifecycle:
            output = model.generate(input_ids=inputs, attention_mask=attention, **options)
        output_ids = output[0].tolist()
        require(output_ids[:len(tokens)] == tokens, "generation output does not preserve the original full prompt")
        generated = output_ids[len(tokens):]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        record = generation_record(row, text, generated, eos_ids, max_new_tokens, index)
        if engineering_steps is not None:
            require(lifecycle["decode_count"] >= 2, "synthetic preflight lacks two real incremental decode steps")
        forwards = lifecycle["forwards"]
        telemetry = {"prefill_count": lifecycle["prefill_count"], "decode_count": lifecycle["decode_count"],
                     "prefill_input_tokens": len(tokens), "cache_growth_closed": True,
                     "last_cache_length": forwards[-1]["cache_length_after"],
                     "generated_token_count": len(generated), "incremental_generation": True}
    if runtime is not None:
        telemetry["online"] = runtime_telemetry(runtime)
    return record, telemetry


def run(args):
    config, manifest, cell, acquisition, pool, indices, shard = load_inputs(args)
    source = source_provenance(args.commit)
    max_new_tokens = args.engineering_generate_steps or config["max_new_tokens"]
    metadata = {"schema": "loopscope.phase11.raw_attempt.v1", "cell": cell, "acquisition_cell": acquisition,
                "engineering_check": args.engineering_check, "engineering_generate_steps": args.engineering_generate_steps,
                "indices": indices, "source_commit": args.commit, "dataset": args.dataset, "scope": args.scope,
                "pool": str(args.pool), "manifest": str(args.manifest), "shard": shard,
                "max_new_tokens": max_new_tokens, "target_gold_loaded": False, "argv": sys.argv, **source}
    if args.dry_run:
        print(json.dumps({"status": "PHASE11_RAW_DRY_RUN", "metadata": metadata, "count": len(indices),
                          "model_loading": False, "target_gold_loaded": False}, sort_keys=True))
        return 0
    require(bool(os.environ.get("SLURM_JOB_ID")), "Phase 11 acquisition requires an allocated Slurm job")
    require(args.scope != "PREFLIGHT_ONLY" or os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "Phase 11 synthetic preflight requires the debug partition")
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json_once(args.run_root / "command_args.json", metadata)
    started = time.monotonic()
    completed = 0
    failed_index = indices[0]
    stage = "runtime_load"
    summary = {"schema": "loopscope.phase11.raw_summary.v1", "metadata": metadata, "target_gold_loaded": False}
    with (args.run_root / "records.jsonl").open("x", encoding="utf-8") as records, \
         (args.run_root / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry_file:
        try:
            torch, model, tokenizer, adapter, eos_ids, env = load_runtime(acquisition, config, args.dataset)
            write_json_once(args.run_root / "env.json", {**env, **source, "source_commit": args.commit,
                                                       "scope": args.scope, "target_gold_loaded": False})
            torch.set_grad_enabled(False)
            torch.manual_seed(20261008)
            torch.cuda.reset_peak_memory_stats()
            total_generated = 0
            for index in indices:
                failed_index, stage = index, "question_acquisition"
                row = pool["rows"][index]
                torch.cuda.synchronize()
                question_started = time.monotonic()
                record, telemetry = acquire_row(torch, model, tokenizer, adapter, eos_ids, row, index,
                    acquisition, config, args.scope, max_new_tokens, args.engineering_generate_steps)
                torch.cuda.synchronize()
                telemetry.update(identity=row["identity"], canonical_index=index,
                                 elapsed_seconds=time.monotonic() - question_started)
                stage = "raw_record_write"
                records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                records.flush()
                telemetry_file.write(json.dumps(telemetry, ensure_ascii=False, allow_nan=False) + "\n")
                telemetry_file.flush()
                completed += 1
                total_generated += record.get("generated_token_count", 0)
                if completed % 32 == 0:
                    print(json.dumps({"cell_id": cell["cell_id"], "completed": completed,
                                      "total": len(indices), "elapsed_seconds": time.monotonic() - started}), flush=True)
            require(completed == len(indices), "raw producer did not finish all declared canonical indices")
            summary.update(status="RAW_COMPLETE", count=completed, generated_token_count=total_generated,
                           elapsed_seconds=time.monotonic() - started,
                           peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                           peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
            write_json_once(args.run_root / "summary.json", summary)
        except Exception as exc:
            records.write(json.dumps(failure_record(pool["rows"][failed_index], failed_index, exc),
                                     ensure_ascii=False, allow_nan=False) + "\n")
            records.flush()
            summary.update(status="FAILED", count=completed, failed_index=failed_index, failed_stage=stage,
                           exception_type=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
            write_json_once(args.run_root / "summary.json", summary)
            write_json_once(args.run_root / "failure.json", {"status": "RUNTIME_FAILED", "stage": stage,
                "canonical_index": failed_index, "exception_type": type(exc).__name__, "target_gold_loaded": False})
            raise
    print(json.dumps({"status": "RAW_COMPLETE", "cell_id": cell["cell_id"], "count": completed,
                      "target_gold_loaded": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
