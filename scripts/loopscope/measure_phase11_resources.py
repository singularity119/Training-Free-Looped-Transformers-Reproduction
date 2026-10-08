#!/usr/bin/env python3
"""One synthetic Phase 11 resource case per debug job, always PREFLIGHT_ONLY.

Lengths come from Gate A's maxima, never from loading target data. Generation
forces 2048 new tokens only for sizing. Its final KV contains prompt+2047
tokens: the last generated token has not been fed back into the model.
Only scalar timing, memory and cache evidence is written, never model tensors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

from run_phase11_accuracy import load_runtime, make_loop_config, runtime_telemetry, source_provenance
from tflt.loopscope.phase11_panel import build_panel, load_config, require, write_json_once


PROMPT_LENGTHS = {"mmlu_pro": 2860, "gpqa_main": 2819, "arc_challenge": 1242}
FAMILIES = ("Native", "LoopK3", "OnlineK3")
NEW_TOKENS = 2048
ARC_CONTINUATION_TOKENS = 46
ARC_CANDIDATES = 4


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=tuple(PROMPT_LENGTHS), required=True)
    result.add_argument("--family", choices=FAMILIES, required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--dry-run", action="store_true")
    return result


def resource_plan(dataset, family, config):
    suffix = {"Native": "-native", "LoopK3": "-k3-loop",
              "OnlineK3": "-k3-online-current-t-lambda0.9"}[family]
    cell = next(cell for cell in build_panel(dataset, config)["cells"] if cell["cell_id"].endswith(suffix))
    generation = dataset != "arc_challenge"
    return {"schema": "loopscope.phase11.resource_plan.v1", "scope": "PREFLIGHT_ONLY",
            "target_gold_loaded": False, "engineering_synthetic": True,
            "dataset": dataset, "family": family, "cell": cell, "batch_size": 1,
            "packing": "one independent process; submitter records co-resident processes",
            "synthetic_input": "ordinary token IDs repeated to Gate A length maxima",
            "prompt_tokens": PROMPT_LENGTHS[dataset],
            "max_new_tokens": NEW_TOKENS if generation else None,
            "min_new_tokens": NEW_TOKENS if generation else None,
            "expected_decode_forwards": NEW_TOKENS - 1 if generation else 0,
            "expected_final_cache_length": PROMPT_LENGTHS[dataset] + NEW_TOKENS - 1 if generation else None,
            "arc_candidate_count": ARC_CANDIDATES if not generation else None,
            "arc_continuation_tokens": ARC_CONTINUATION_TOKENS if not generation else None,
            "formal_greedy_eos_behavior": "not measured; forced length is PREFLIGHT_ONLY"}


def ordinary_tokens(tokenizer, length, offset=0):
    excluded = set(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is not None:
        excluded.add(tokenizer.pad_token_id)
    seed = [token for token in tokenizer.encode(" Synthetic engineering triangle circle blue red square.",
                                              add_special_tokens=False) if token not in excluded]
    require(len(set(seed)) >= ARC_CANDIDATES, "synthetic token seed lacks ordinary distinct tokens")
    return [seed[(i + offset) % len(seed)] for i in range(length)]


def cache_lengths(cache, layer_count):
    require(cache is not None, "generation output has no incremental cache")
    if hasattr(cache, "get_seq_length"):
        return [int(cache.get_seq_length(i)) for i in range(layer_count)]
    require(len(cache) == layer_count, "generation cache layer count differs")
    return [int(cache[i][0].shape[-2]) for i in range(layer_count)]


def memory(torch):
    return {"allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}


@contextmanager
def forward_measurement(torch, model, plan, stream, stages):
    """Observe actual model calls without retaining output tensors."""
    report = {"forward_count": 0, "decode_count": 0, "final_cache_lengths_by_layer": None}
    current = {}
    generation = plan["dataset"] != "arc_challenge"

    def before(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        require(ids is not None and ids.shape[0] == 1, "resource model forward requires batch one IDs")
        seq = int(ids.shape[1])
        stage = "prefill" if report["forward_count"] == 0 or not generation else "decode"
        if not generation:
            stage = "candidate_scoring"
        torch.cuda.synchronize()
        if stage not in stages:
            torch.cuda.reset_peak_memory_stats()
            stages[stage] = {"forward_seconds": 0.0, "forward_count": 0}
        current.update(stage=stage, sequence_length=seq, started=time.monotonic())

    def after(module, args, kwargs, output):
        torch.cuda.synchronize()
        seconds = time.monotonic() - current["started"]
        stage = current["stage"]
        stages[stage]["forward_seconds"] += seconds
        stages[stage]["forward_count"] += 1
        stages[stage].update(memory(torch))
        row = {"forward_index": report["forward_count"], "stage": stage,
               "sequence_length": current["sequence_length"], "forward_seconds": seconds, **memory(torch)}
        if generation:
            expected = plan["prompt_tokens"] + report["forward_count"]
            require(current["sequence_length"] == (plan["prompt_tokens"] if report["forward_count"] == 0 else 1),
                    "resource generation did not use one full prefill then single-token decode")
            cache = getattr(output, "past_key_values", None)
            lengths = cache_lengths(cache, model.config.num_hidden_layers)
            require(lengths == [expected] * model.config.num_hidden_layers,
                    "per-layer resource cache growth differs from actual forward")
            row["cache_lengths_by_layer"] = lengths
            report["final_cache_lengths_by_layer"] = lengths
            report["decode_count"] += int(stage == "decode")
        report["forward_count"] += 1
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        if report["forward_count"] == 1 or report["forward_count"] % 128 == 0:
            print(json.dumps({"status": "RESOURCE_PROGRESS", "forward_count": report["forward_count"],
                              "stage": stage, "elapsed_forward_seconds": stages[stage]["forward_seconds"]}), flush=True)

    pre = model.register_forward_pre_hook(before, with_kwargs=True)
    post = model.register_forward_hook(after, with_kwargs=True)
    try:
        yield report
    finally:
        pre.remove()
        post.remove()


def measure_case(torch, model, tokenizer, adapter, plan, config, stream, stages):
    from tflt.wrapper import looped_model
    from tflt.loopscope.phase11_adapter import generation_forward_context
    from tflt.loopscope.phase11_runtime import make_runtime
    cell = plan["cell"]
    runtime = make_runtime(cell)
    manager = nullcontext() if cell["arm"] == "Native" else looped_model(model, make_loop_config(cell, runtime))
    tokens = ordinary_tokens(tokenizer, plan["prompt_tokens"])
    torch.cuda.synchronize()
    started = time.monotonic()
    with forward_measurement(torch, model, plan, stream, stages) as evidence, manager:
        if plan["dataset"] == "arc_challenge":
            # This is HFLM's complete native teacher-forced scoring path, using
            # synthetic token requests of exact lengths rather than target text.
            requests = [(None, tokens, ordinary_tokens(tokenizer, ARC_CONTINUATION_TOKENS, i))
                        for i in range(ARC_CANDIDATES)]
            adapter.phase11_runtime = runtime
            adapter.phase11_identity = "synthetic-resource-arc"
            try:
                scores = adapter._loglikelihood_tokens(requests, disable_tqdm=True)
                positions = adapter.phase11_positions
                require(len(scores) == ARC_CANDIDATES and all(math.isfinite(float(score[0])) for score in scores),
                        "ARC resource scoring did not close finite full-candidate scores")
                require(evidence["forward_count"] == ARC_CANDIDATES,
                        "ARC resource case did not execute all four distinct batch-one candidates")
                require(len(positions) == ARC_CANDIDATES and all(
                    p["context_length"] == plan["prompt_tokens"] and
                    p["continuation_length"] == ARC_CONTINUATION_TOKENS and p["left_truncated_tokens"] == 0
                    for p in positions), "ARC resource candidate lengths differ")
                evidence.update(candidate_count=len(scores), scored_continuation_tokens=ARC_CANDIDATES * ARC_CONTINUATION_TOKENS,
                                candidate_input_tokens=[p["input_length"] for p in positions],
                                input_path="native_HFLM_full_tokenized_candidate_scoring", incremental_generation=False)
            finally:
                adapter.phase11_runtime = None
                adapter.phase11_identity = None
        else:
            ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
            with generation_forward_context(model, runtime, "synthetic-resource-generation", tokens,
                    attention_mask=[1] * len(tokens), special_ids=tokenizer.all_special_ids,
                    pad_token_id=tokenizer.pad_token_id) as lifecycle:
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        do_sample=False, num_beams=1, use_cache=True,
                                        min_new_tokens=NEW_TOKENS, max_new_tokens=NEW_TOKENS)
            count = int(output.shape[1]) - len(tokens)
            require(count == NEW_TOKENS and lifecycle["prefill_count"] == 1 and
                    lifecycle["decode_count"] == NEW_TOKENS - 1 and evidence["forward_count"] == NEW_TOKENS,
                    "resource generation did not actually reach the fixed 2048-token limit")
            require(evidence["final_cache_lengths_by_layer"] ==
                    [plan["expected_final_cache_length"]] * model.config.num_hidden_layers,
                    "resource generation final per-layer cache length differs")
            evidence.update(generated_token_count=count, fixed_generation_limit_reached=True,
                            incremental_generation=True, prefill_count=lifecycle["prefill_count"],
                            last_generated_token_not_fed_to_cache=True)
    torch.cuda.synchronize()
    evidence["question_seconds"] = time.monotonic() - started
    evidence["questions_per_second"] = 1.0 / evidence["question_seconds"]
    work = evidence.get("generated_token_count", evidence.get("scored_continuation_tokens"))
    evidence["tokens_per_second"] = work / evidence["question_seconds"]
    if "decode" in stages:
        evidence["decode_tokens_per_second"] = evidence["decode_count"] / stages["decode"]["forward_seconds"]
    if runtime is not None:
        evidence["online"] = runtime_telemetry(runtime)
    return evidence


def run(args):
    config = load_config(args.config)
    plan = resource_plan(args.dataset, args.family, config)
    source = source_provenance(args.commit)
    command = {**plan, **source, "source_commit": args.commit, "argv": sys.argv}
    if args.dry_run:
        print(json.dumps({"status": "PHASE11_RESOURCE_DRY_RUN", "model_loading": False, **command}, sort_keys=True))
        return 0
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "Phase 11 resource measurement requires a real Slurm debug allocation")
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json_once(args.run_root / "command_args.json", command)
    stages = {}
    started = time.monotonic()
    torch = None
    stage = "load"
    try:
        import torch
        require(torch.cuda.is_available(), "resource measurement requires CUDA")
        write_json_once(args.run_root / "allocation.json", {
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
            "visible_cuda_devices": int(torch.cuda.device_count()),
            "job_id": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
            "qos": os.environ.get("SLURM_JOB_QOS"), "node": os.environ.get("SLURMD_NODENAME"),
            "co_resident_processes": os.environ.get("PHASE11_PACKING", "1"),
            "python": sys.executable, "scope": "PREFLIGHT_ONLY", **source})
        torch.set_grad_enabled(False)
        torch.manual_seed(20261008)
        torch.cuda.reset_peak_memory_stats()
        load_started = time.monotonic()
        torch, model, tokenizer, adapter, eos_ids, env = load_runtime(plan["cell"], config, args.dataset)
        torch.cuda.synchronize()
        stages["load"] = {"seconds": time.monotonic() - load_started, **memory(torch)}
        write_json_once(args.run_root / "env.json", {**env, **source, "scope": "PREFLIGHT_ONLY",
                        "target_gold_loaded": False, "co_resident_processes": os.environ.get("PHASE11_PACKING", "1")})
        stage = "synthetic_measurement"
        with (args.run_root / "forward_resources.jsonl").open("x", encoding="utf-8") as stream:
            evidence = measure_case(torch, model, tokenizer, adapter, plan, config, stream, stages)
        summary = {"schema": "loopscope.phase11.resources.v1", "status": "RESOURCE_COMPLETE",
                   "metadata": command, "stages": stages, "evidence": evidence,
                   "elapsed_seconds": time.monotonic() - started, "oom": False,
                   "scope": "PREFLIGHT_ONLY", "target_gold_loaded": False}
        write_json_once(args.run_root / "summary.json", summary)
        print(json.dumps({"status": "RESOURCE_COMPLETE", "dataset": args.dataset, "family": args.family,
                          "question_seconds": evidence["question_seconds"], "tokens_per_second": evidence["tokens_per_second"]}), flush=True)
        return 0
    except Exception as exc:
        failure = {"schema": "loopscope.phase11.resources.v1", "status": "RESOURCE_FAILED", "stage": stage,
                   "metadata": command, "stages": stages, "exception_type": type(exc).__name__,
                   "message": str(exc), "oom": torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError),
                   "elapsed_seconds": time.monotonic() - started,
                   "scope": "PREFLIGHT_ONLY", "target_gold_loaded": False}
        if torch is not None and torch.cuda.is_available():
            failure["failure_memory"] = memory(torch)
        write_json_once(args.run_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
