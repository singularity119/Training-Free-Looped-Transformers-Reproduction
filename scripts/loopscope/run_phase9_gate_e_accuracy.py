#!/usr/bin/env python3
"""Acquire one gold-free Phase 9 Gate E score cell under an explicit policy.

The evaluator and full-continuation scoring path remain the frozen Phase 9
interfaces.  Only the residual callback and Gate E cell manifest are new.
Each process keeps batch size one and never writes residuals or directions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from tflt.loopscope.phase9_gate_e_spectral_core import (
    DIRECTION_POLICIES,
    INTERVENTION_MODES,
    direction_schedule,
)


CHOICES = "ABCD"
TEST_COUNT = 14042


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def finite_scores(values: Iterable[Any]) -> List[float]:
    result = [float(value[0] if isinstance(value, (tuple, list)) else value) for value in values]
    if len(result) != 4 or not all(math.isfinite(value) for value in result):
        raise ValueError("expected four finite raw choice scores")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--cell-id", required=True)
    result.add_argument("--direction-policy", choices=DIRECTION_POLICIES, required=True)
    result.add_argument("--intervention-mode", choices=INTERVENTION_MODES, required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--scope", choices=("PREFLIGHT_ONLY", "FORMAL_TEST"), required=True)
    result.add_argument("--selection", choices=("preflight", "canary", "remaining"), required=True)
    result.add_argument("--canary-indices", type=Path)
    result.add_argument("--config", type=Path)
    result.add_argument("--dry-run", action="store_true")
    return result


def load_inputs(args: argparse.Namespace):
    from tflt.loopscope.phase8_test_pool import validate_test_pool
    from tflt.loopscope.phase9_gate_e_accuracy import (
        load_config,
        validate_canary_bundle,
        validate_score_manifest,
    )

    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = validate_score_manifest(manifest, config, args.pool, args.scope)
    if args.cell_id not in cells:
        raise ValueError("requested cell is not one of the 12 Gate E score cells")
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    validate_test_pool(pool)
    cell = cells[args.cell_id]
    fixed_policy_preflight = args.scope == "PREFLIGHT_ONLY" and args.direction_policy == "fixed_t0"
    if ((cell["direction_policy"] != args.direction_policy and not fixed_policy_preflight) or
            cell["intervention_mode"] != args.intervention_mode):
        raise ValueError("explicit direction policy/intervention mode differs from the frozen score cell")
    if args.scope == "PREFLIGHT_ONLY":
        if args.selection != "preflight" or args.canary_indices is not None:
            raise ValueError("Gate E preflight scope must use the fixed label-free selection")
        indices = []
    else:
        if args.selection not in ("canary", "remaining") or args.canary_indices is None:
            raise ValueError("Gate E formal scope requires the canary index bundle")
        bundle = json.loads(args.canary_indices.read_text(encoding="utf-8"))
        by_model = validate_canary_bundle(bundle, pool, config)
        canary = set(by_model[cell["model"]])
        indices = sorted(canary) if args.selection == "canary" else [
            index for index in range(TEST_COUNT) if index not in canary
        ]
        if not indices:
            raise ValueError("Gate E formal selection produced no canonical identities")
    return config, manifest, cell, pool, indices


def selected_rows(args: argparse.Namespace, cell: Mapping[str, Any], pool: Mapping[str, Any], config: Mapping[str, Any]):
    from tflt.loopscope.phase9_gate_e_accuracy import validate_canary_bundle

    rows = list(pool["rows"])
    if args.scope == "PREFLIGHT_ONLY":
        selected = {0, 1}
        ranked = sorted(range(len(rows)), key=lambda index: (-rows[index]["prompt_token_lengths"][cell["model"]], index))
        for index in ranked:
            selected.add(index)
            if len(selected) >= 6:
                break
        indices = sorted(selected)
    else:
        bundle = json.loads(args.canary_indices.read_text(encoding="utf-8"))
        by_model = validate_canary_bundle(bundle, pool, config)
        canary = set(by_model[cell["model"]])
        indices = sorted(canary) if args.selection == "canary" else [
            index for index in range(TEST_COUNT) if index not in canary
        ]
    chosen = [rows[index] for index in indices]
    if not chosen or len(chosen) != len(indices):
        raise ValueError("Gate E selected rows are empty or incomplete")
    return chosen, indices


def make_loop_config(cell: Mapping[str, Any], runtime: Any) -> Any:
    from tflt.config import LoopConfig

    return LoopConfig(
        model_alias=cell["model"], window=tuple(cell["window"]), k=int(cell["k"]),
        alpha=1.0, beta=0.0, strategy="damped_euler", iteration_mode="block",
        cache_strategy=cell["cache_strategy"], decode_mode="bypass", residual_transform=runtime,
    )


def build_choice_requests(row: Mapping[str, Any], Instance: Any) -> List[Any]:
    return [
        Instance(
            request_type="loglikelihood", doc={}, arguments=(row["prompt"], " " + letter),
            idx=index, metadata=("phase9-gate-e", row["identity"], 1),
        )
        for index, letter in enumerate(CHOICES)
    ]


def check_position_metadata(adapter: Any, tokenizer: Any) -> Dict[str, int]:
    special = set(int(value) for value in (getattr(tokenizer, "all_special_ids", ()) or ()))
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        special.add(int(pad))
    if not adapter.phase9_positions:
        raise RuntimeError("Gate E scorer produced no position metadata")
    maximum = 0
    left_truncated = 0
    multitoken = 0
    for item in adapter.phase9_positions:
        position = int(item["position"])
        valid = tuple(int(value) for value in item["valid_positions"])
        effective = tuple(int(value) for value in item.get("effective_valid_positions", valid))
        actual = tuple(int(value) for value in item["actual_tokens"])
        if not valid or tuple(sorted(set(valid))) != valid or position not in valid:
            raise RuntimeError("Gate E valid position mask is not a retained prefix")
        if any(value > position or value in special for value in valid):
            raise RuntimeError("Gate E mask includes post-answer or special positions")
        if not effective or not set(effective).issubset(valid) or position not in effective:
            raise RuntimeError("Gate E effective mask is not a retained prefix")
        if len(item["prefix_tokens"]) != position + 1 or len(actual) != int(item["input_length"]):
            raise RuntimeError("Gate E position metadata has inconsistent lengths")
        if len(actual) != position + int(item["continuation_length"]):
            raise RuntimeError("Gate E answer boundary differs from continuation length")
        maximum = max(maximum, len(actual))
        left_truncated += int(int(item["left_truncated_tokens"]) > 0)
        multitoken += int(int(item["continuation_length"]) > 1)
    return {
        "max_input_length": maximum,
        "left_truncated_requests": left_truncated,
        "multitoken_requests": multitoken,
        "request_count": len(adapter.phase9_positions),
    }


def load_runtime(
    cell: Mapping[str, Any], direction_policy: str, intervention_mode: str
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from tflt.loopscope.phase9_adapter import build_hflm_class
    from tflt.loopscope.phase9_gate_e_runtime import Phase9GateERuntime

    if importlib.metadata.version("lm_eval") != "0.4.11":
        raise RuntimeError("frozen Phase 9 scorer requires lm_eval 0.4.11")
    revision = cell["revision"]
    tokenizer = AutoTokenizer.from_pretrained(cell["model"], revision=revision, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cell["model"], revision=revision, local_files_only=True, trust_remote_code=True,
        torch_dtype=getattr(torch, cell["dtype"]),
    ).to("cuda").eval()
    observed_revision = getattr(model.config, "_commit_hash", None)
    if observed_revision and observed_revision != revision:
        raise RuntimeError("model revision differs from the Gate E frozen contract")
    runtime = Phase9GateERuntime(
        direction_policy, intervention_mode, int(cell["k"]), strength=0.5, intervene=True
    )
    adapter_class = build_hflm_class(HFLM)
    adapter = adapter_class(
        pretrained=model, tokenizer=tokenizer, batch_size="1", trust_remote_code=True,
        phase9_runtime=runtime,
    )
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "lm_eval", "datasets")}
    env = {
        "python": sys.executable, "platform": platform.platform(),
        "model_revision": observed_revision or revision, "tokenizer_revision": revision,
        "model_dtype": str(model.dtype), "max_length": int(adapter.max_length),
        "job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "qos": os.environ.get("SLURM_JOB_QOS"), "gpu": torch.cuda.get_device_name(),
        "direction_policy": direction_policy, "intervention_mode": intervention_mode,
        "total_memory": int(torch.cuda.get_device_properties(0).total_memory), "versions": versions,
    }
    return torch, model, tokenizer, adapter, runtime, env


def run(args: argparse.Namespace) -> int:
    config, manifest, cell, pool, _ = load_inputs(args)
    rows, indices = selected_rows(args, cell, pool, config)
    if args.dry_run:
        print(json.dumps({
            "status": "PHASE9_GATE_E_SCORE_DRY_RUN", "cell_id": args.cell_id,
            "direction_policy": args.direction_policy,
            "intervention_mode": args.intervention_mode,
            "scope": args.scope, "selection": args.selection, "count": len(rows),
            "first_index": indices[0], "last_index": indices[-1], "target_gold_loaded": False,
        }, sort_keys=True))
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate E score acquisition requires a real Slurm job")
    if args.scope == "PREFLIGHT_ONLY" and os.environ.get("SLURM_JOB_PARTITION") != "debug":
        raise RuntimeError("Gate E score preflight requires the debug partition")

    args.run_root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": "loopscope.phase9.gate_e.score_attempt.v2", "cell": cell,
        "direction_policy": args.direction_policy,
        "intervention_mode": args.intervention_mode,
        "scope": args.scope, "selection": args.selection,
        "indices_file": str(args.canary_indices) if args.canary_indices else None,
        "indices": indices, "source_commit": args.commit, "pool": str(args.pool),
        "manifest": str(args.manifest), "target_gold_loaded": False, "argv": sys.argv,
    }
    write_json_once(args.run_root / "command_args.json", metadata)
    torch, model, tokenizer, adapter, runtime, env = load_runtime(
        cell, args.direction_policy, args.intervention_mode
    )
    env.update({"source_commit": args.commit, "scope": args.scope, "selection": args.selection,
                "target_gold_loaded": False, "test_pool_gold_free": True})
    write_json_once(args.run_root / "env.json", env)
    torch.set_grad_enabled(False)
    torch.manual_seed(20260923)
    torch.cuda.reset_peak_memory_stats()
    from lm_eval.api.instance import Instance
    from tflt.loopscope.phase9_adapter import score_choices
    from tflt.wrapper import looped_model

    k = int(cell["k"])
    stats = {
        "max_input_length": 0, "left_truncated_requests": 0, "multitoken_requests": 0,
        "request_count": 0, "callback_by_t": {str(t): 0 for t in range(k)},
        "applied_by_t": {str(t): 0 for t in range(k)},
        "direction_fit_by_t": {str(t): 0 for t in range(k - 1)},
        "direction_used_fit_by_t": {str(t): 0 for t in range(k)},
        "nonanswer_mutation_count": 0, "t0_mutation_count": 0,
        "direction_fit_count": 0, "svd_seconds_total": 0.0, "svd_seconds_max": 0.0,
        "svd_top1_energy_fraction_sum": 0.0, "svd_top1_energy_fraction_min": None,
        "svd_top1_energy_fraction_max": None,
    }
    started = time.monotonic()
    acquisition = time.monotonic()
    with looped_model(model, make_loop_config(cell, runtime)), (args.run_root / "scores.jsonl").open("x", encoding="utf-8") as output:
        for completed, row in enumerate(rows, 1):
            context_start = len(runtime.context_summaries)
            call_start = len(runtime.calls)
            scores = finite_scores(score_choices(adapter, row["prompt"], row["identity"]))
            position_stats = check_position_metadata(adapter, tokenizer)
            contexts = runtime.context_summaries[context_start:]
            if len(contexts) != 1 or contexts[0]["call_count"] != k or contexts[0]["timesteps"] != list(range(k)):
                raise RuntimeError("Gate E score did not produce one ordered K-step candidate context")
            calls = runtime.calls[call_start:]
            if len(calls) != k:
                raise RuntimeError("Gate E callback count differs from frozen K")
            expected_fit = {0} if args.direction_policy == "fixed_t0" else set(range(k - 1))
            observed_fit = {call["direction_fit_t"] for call in calls if call["direction_fit_t"] is not None}
            if observed_fit != expected_fit:
                raise RuntimeError("Gate E direction fits do not stop before the final round")
            for call in calls:
                t = int(call["t"])
                if call["nonanswer_max_abs_change"] != 0.0:
                    stats["nonanswer_mutation_count"] += 1
                    raise RuntimeError("Gate E intervention changed a non-answer position")
                if t == 0 and call["applied"]:
                    stats["t0_mutation_count"] += 1
                    raise RuntimeError("Gate E intervention was applied at t0")
                if t >= 1 and not call["applied"]:
                    raise RuntimeError("Gate E intervention was not applied at t>=1")
                stats["callback_by_t"][str(t)] += 1
                stats["applied_by_t"][str(t)] += int(bool(call["applied"]))
                source_t = call["direction_used_fit_t"]
                if source_t is not None:
                    stats["direction_used_fit_by_t"][str(t)] += 1
                if call["direction_fit_t"] is not None:
                    fit_t = int(call["direction_fit_t"])
                    stats["direction_fit_by_t"][str(fit_t)] += 1
            fit_metadata = runtime.fit_metadata
            if set(fit_metadata) != expected_fit:
                raise RuntimeError("Gate E did not retain exactly the required previous-round fit metadata")
            for fit in fit_metadata.values():
                stats["direction_fit_count"] += 1
                svd_seconds = float(fit.get("svd_seconds", 0.0))
                top1 = float(fit["top1_energy_fraction"])
                stats["svd_seconds_total"] += svd_seconds
                stats["svd_seconds_max"] = max(stats["svd_seconds_max"], svd_seconds)
                stats["svd_top1_energy_fraction_sum"] += top1
                stats["svd_top1_energy_fraction_min"] = top1 if stats["svd_top1_energy_fraction_min"] is None else min(stats["svd_top1_energy_fraction_min"], top1)
                stats["svd_top1_energy_fraction_max"] = top1 if stats["svd_top1_energy_fraction_max"] is None else max(stats["svd_top1_energy_fraction_max"], top1)
            stats["max_input_length"] = max(stats["max_input_length"], position_stats["max_input_length"])
            stats["left_truncated_requests"] += position_stats["left_truncated_requests"]
            stats["multitoken_requests"] += position_stats["multitoken_requests"]
            stats["request_count"] += position_stats["request_count"]
            output.write(json.dumps({
                "identity": row["identity"], "subject": row["subject"],
                "doc_index": row["doc_index"], "scores": scores,
            }, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            runtime.calls.clear()
            del runtime.context_summaries[context_start:]
            if completed % 32 == 0:
                print(json.dumps({"completed": completed, "total": len(rows),
                                  "elapsed_seconds": time.monotonic() - acquisition,
                                  "cell_id": args.cell_id}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - acquisition
    expected_fits = len(rows) * (1 if args.direction_policy == "fixed_t0" else k - 1)
    if stats["direction_fit_count"] != expected_fits:
        raise RuntimeError("Gate E direction fit count differs from K-1 per identity")
    stats["svd_seconds_mean_per_fit"] = stats["svd_seconds_total"] / expected_fits
    summary = {
        "schema": "loopscope.phase9.gate_e.score_summary.v2", "status": "SCORES_COMPLETE",
        "direction_policy": args.direction_policy,
        "intervention_mode": args.intervention_mode,
        "metadata": metadata, "count": len(rows), "first_index": indices[0],
        "last_index": indices[-1], "target_gold_loaded": False,
        "elapsed_seconds": time.monotonic() - started, "acquisition_seconds": elapsed,
        "samples_per_second": len(rows) / elapsed if elapsed else None,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "oom_count": 0,
        "positions_and_runtime": stats,
        "round_schedule": [
            {
                "t": t, "applied_t": t if t > 0 else None,
                "direction_used_fit_t": used, "direction_fit_t": fit,
                "applied_count": stats["applied_by_t"][str(t)],
                "direction_used_fit_count": stats["direction_used_fit_by_t"].get(str(t), 0),
                "direction_fit_count": stats["direction_fit_by_t"].get(str(fit), 0)
                if fit is not None else 0,
            }
            for t, used, fit in direction_schedule(args.direction_policy, k)
        ],
    }
    write_json_once(args.run_root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "cell_id": args.cell_id,
                      "direction_policy": args.direction_policy,
                      "intervention_mode": args.intervention_mode,
                      "selection": args.selection, "count": summary["count"],
                      "samples_per_second": summary["samples_per_second"],
                      "target_gold_loaded": summary["target_gold_loaded"]}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
