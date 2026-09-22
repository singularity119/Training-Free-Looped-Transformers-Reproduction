#!/usr/bin/env python3
"""Acquire one gold-free Phase 9 Online-t0 or Matched-norm score cell.

The runner keeps batch size one and the lm-eval 0.4.11 continuation scorer.
It writes only canonical identity metadata and four raw choice scores.  The
per-prompt direction and residual tensors are never persisted.  A process is
one independent model/cell worker; same-GPU packing is performed by the shell
launcher, not by changing the scientific batch shape.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CHOICES = "ABCD"
MODES = ("Online-t0", "Matched-norm")
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


def score_map(values: Iterable[Any]) -> List[float]:
    scores = finite_scores(values)
    return scores


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--cell-id", required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--scope", choices=("PREFLIGHT_ONLY", "FORMAL_TEST"), required=True)
    result.add_argument("--selection", choices=("preflight", "canary", "remaining"), required=True)
    result.add_argument("--canary-indices", type=Path)
    result.add_argument("--config", type=Path)
    result.add_argument("--dry-run", action="store_true")
    return result


def load_inputs(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    from tflt.loopscope.phase8_test_pool import validate_test_pool
    from tflt.loopscope.phase9_accuracy import (
        load_config,
        validate_canary_bundle,
        validate_score_manifest,
    )

    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = validate_score_manifest(manifest, config, args.pool, args.scope)
    if args.cell_id not in cells:
        raise ValueError("requested cell is not one of the 18 Phase 9 new score cells")
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    validate_test_pool(pool)
    if args.scope == "PREFLIGHT_ONLY":
        if args.selection != "preflight" or args.canary_indices is not None:
            raise ValueError("preflight scope must use the fixed label-free preflight selection")
        indices = []
    else:
        if args.selection not in ("canary", "remaining") or args.canary_indices is None:
            raise ValueError("formal scope requires the canary index bundle")
        bundle = json.loads(args.canary_indices.read_text(encoding="utf-8"))
        model_names = [entry["model"] for entry in config["models"]]
        by_model = validate_canary_bundle(bundle, pool, config)
        model = cells[args.cell_id]["model"]
        canary = set(by_model[model])
        if args.selection == "canary":
            indices = sorted(canary)
        else:
            indices = [index for index in range(TEST_COUNT) if index not in canary]
        if not indices:
            raise ValueError("formal selection produced no canonical identities")
    return config, manifest, cells[args.cell_id], [pool["rows"][index] for index in indices] if indices else pool["rows"]


def selected_rows(
    args: argparse.Namespace,
    cell: Mapping[str, Any],
    pool: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[int]]:
    from tflt.loopscope.phase9_accuracy import representative_canary_indices, validate_canary_bundle

    rows = list(pool["rows"])
    if args.scope == "PREFLIGHT_ONLY":
        model = cell["model"]
        selected = {0, 1}
        ranked = sorted(
            range(len(rows)),
            key=lambda index: (-rows[index]["prompt_token_lengths"][model], index),
        )
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
    selected = [rows[index] for index in indices]
    if not selected or len(selected) != len(indices):
        raise ValueError("Phase 9 selected rows are empty or incomplete")
    return selected, indices


def make_loop_config(cell: Mapping[str, Any], runtime: Any) -> Any:
    from tflt.config import LoopConfig

    return LoopConfig(
        model_alias=cell["model"],
        window=tuple(cell["window"]),
        k=int(cell["k"]),
        alpha=1.0,
        beta=0.0,
        strategy="damped_euler",
        iteration_mode="block",
        cache_strategy=cell["cache_strategy"],
        decode_mode="bypass",
        residual_transform=runtime,
    )


def build_choice_requests(row: Mapping[str, Any], Instance: Any) -> List[Any]:
    return [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(row["prompt"], " " + letter),
            idx=index,
            metadata=("phase9-full", row["identity"], 1),
        )
        for index, letter in enumerate(CHOICES)
    ]


def check_position_metadata(adapter: Any, tokenizer: Any) -> Dict[str, int]:
    special = set(int(value) for value in (getattr(tokenizer, "all_special_ids", ()) or ()))
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        special.add(int(pad))
    if not adapter.phase9_positions:
        raise RuntimeError("Phase 9 scorer produced no position metadata")
    max_length = 0
    left_truncated = 0
    multitoken = 0
    for item in adapter.phase9_positions:
        position = int(item["position"])
        valid = tuple(int(value) for value in item["valid_positions"])
        effective = tuple(int(value) for value in item.get("effective_valid_positions", valid))
        actual = tuple(int(value) for value in item["actual_tokens"])
        if not valid or tuple(sorted(set(valid))) != valid:
            raise RuntimeError("Phase 9 valid position mask is not sorted and unique")
        if position not in valid or any(value > position for value in valid):
            raise RuntimeError("Phase 9 mask includes a position after the pre-answer boundary")
        if any(value in special for value in valid):
            raise RuntimeError("Phase 9 mask includes a padding or special token")
        if not effective or not set(effective).issubset(valid) or position not in effective:
            raise RuntimeError("effective Phase 9 mask is not a retained prefix")
        if any(value in special for value in effective):
            raise RuntimeError("effective Phase 9 mask includes a special token")
        if len(item["prefix_tokens"]) != position + 1 or len(actual) != int(item["input_length"]):
            raise RuntimeError("Phase 9 position metadata has inconsistent lengths")
        if len(actual) != position + int(item["continuation_length"]):
            raise RuntimeError("Phase 9 answer boundary differs from continuation length")
        max_length = max(max_length, len(actual))
        left_truncated += int(int(item["left_truncated_tokens"]) > 0)
        multitoken += int(int(item["continuation_length"]) > 1)
    return {
        "max_input_length": max_length,
        "left_truncated_requests": left_truncated,
        "multitoken_requests": multitoken,
        "request_count": len(adapter.phase9_positions),
    }


def load_runtime(cell: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any, Dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from tflt.loopscope.phase9_adapter import build_hflm_class
    from tflt.loopscope.phase9_runtime import Phase9Runtime

    if importlib.metadata.version("lm_eval") != "0.4.11":
        raise RuntimeError("frozen Phase 9 scorer requires lm_eval 0.4.11")
    revision = cell["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        cell["model"], revision=revision, local_files_only=True, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cell["model"], revision=revision, local_files_only=True, trust_remote_code=True,
        torch_dtype=getattr(torch, cell["dtype"]),
    ).to("cuda").eval()
    observed_revision = getattr(model.config, "_commit_hash", None)
    if observed_revision and observed_revision != revision:
        raise RuntimeError("model revision differs from the Phase 9 frozen contract")
    runtime = Phase9Runtime(cell["arm"], strength=0.5, intervene=True)
    adapter_class = build_hflm_class(HFLM)
    adapter = adapter_class(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size="1",
        trust_remote_code=True,
        phase9_runtime=runtime,
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "lm_eval", "datasets")
    }
    env = {
        "python": sys.executable,
        "platform": platform.platform(),
        "model_revision": observed_revision or revision,
        "tokenizer_revision": revision,
        "model_dtype": str(model.dtype),
        "max_length": int(adapter.max_length),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "qos": os.environ.get("SLURM_JOB_QOS"),
        "gpu": torch.cuda.get_device_name(),
        "total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        "versions": versions,
    }
    return torch, model, tokenizer, adapter, runtime, env


def run(args: argparse.Namespace) -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 9 formal score acquisition requires a real Slurm job")
    if args.scope == "PREFLIGHT_ONLY" and os.environ.get("SLURM_JOB_PARTITION") != "debug":
        raise RuntimeError("Phase 9 score preflight requires the debug partition")
    config, manifest, cell, _ = load_inputs(args)
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    rows, indices = selected_rows(args, cell, pool, config)
    if args.dry_run:
        print(json.dumps({
            "status": "PHASE9_SCORE_DRY_RUN",
            "cell_id": args.cell_id,
            "scope": args.scope,
            "selection": args.selection,
            "count": len(rows),
            "first_index": indices[0],
            "last_index": indices[-1],
            "target_gold_loaded": False,
        }, sort_keys=True))
        return 0

    args.run_root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": "loopscope.phase9.score_attempt.v1",
        "cell": cell,
        "scope": args.scope,
        "selection": args.selection,
        "indices_file": str(args.canary_indices) if args.canary_indices else None,
        "indices": indices,
        "source_commit": args.commit,
        "pool": str(args.pool),
        "manifest": str(args.manifest),
        "target_gold_loaded": False,
        "argv": sys.argv,
    }
    write_json_once(args.run_root / "command_args.json", metadata)
    torch, model, tokenizer, adapter, runtime, env = load_runtime(cell)
    env.update({
        "source_commit": args.commit,
        "scope": args.scope,
        "selection": args.selection,
        "target_gold_loaded": False,
        "test_pool_gold_free": True,
    })
    write_json_once(args.run_root / "env.json", env)
    torch.set_grad_enabled(False)
    torch.manual_seed(20260922)
    torch.cuda.reset_peak_memory_stats()
    from lm_eval.api.instance import Instance
    from tflt.loopscope.phase9_adapter import score_choices
    from tflt.wrapper import looped_model

    stats = {
        "max_input_length": 0,
        "left_truncated_requests": 0,
        "multitoken_requests": 0,
        "request_count": 0,
        "callback_by_t": {str(t): 0 for t in range(int(cell["k"]))},
        "applied_by_t": {str(t): 0 for t in range(int(cell["k"]))},
        "nonanswer_mutation_count": 0,
        "t0_mutation_count": 0,
        "direction_fit_count": 0,
        "svd_seconds_total": 0.0,
        "svd_seconds_max": 0.0,
        "svd_top1_energy_fraction_sum": 0.0,
        "svd_top1_energy_fraction_min": None,
        "svd_top1_energy_fraction_max": None,
    }
    started = time.monotonic()
    acquisition = time.monotonic()
    with looped_model(model, make_loop_config(cell, runtime)), (args.run_root / "scores.jsonl").open("x", encoding="utf-8") as output:
        for completed, row in enumerate(rows, 1):
            context_start = len(runtime.context_summaries)
            call_start = len(runtime.calls)
            scores = score_map(score_choices(adapter, row["prompt"], row["identity"]))
            position_stats = check_position_metadata(adapter, tokenizer)
            contexts = runtime.context_summaries[context_start:]
            if len(contexts) != 1 or contexts[0]["call_count"] != int(cell["k"]):
                raise RuntimeError("Phase 9 score did not produce one K-step candidate context")
            if contexts[0]["timesteps"] != list(range(int(cell["k"]))):
                raise RuntimeError("Phase 9 score callback timesteps differ from frozen K")
            calls = runtime.calls[call_start:]
            if len(calls) != int(cell["k"]):
                raise RuntimeError("Phase 9 score callback count differs from frozen K")
            for call in calls:
                t = int(call["t"])
                if t not in range(int(cell["k"])):
                    raise RuntimeError("Phase 9 score callback timestep outside frozen K")
                if bool(call["nonanswer_max_abs_change"] != 0.0):
                    stats["nonanswer_mutation_count"] += 1
                    raise RuntimeError("Phase 9 intervention changed a non-answer position")
                if t == 0 and call["applied"]:
                    stats["t0_mutation_count"] += 1
                    raise RuntimeError("Phase 9 intervention was applied at t0")
                if t >= 1 and not call["applied"]:
                    raise RuntimeError("Phase 9 intervention was not applied at t>=1")
                stats["callback_by_t"][str(t)] += 1
                stats["applied_by_t"][str(t)] += int(bool(call["applied"]))
            fit = dict(runtime._fit_metadata or {})
            if not fit:
                raise RuntimeError("Phase 9 score did not fit a t0 direction")
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
                "identity": row["identity"],
                "subject": row["subject"],
                "doc_index": row["doc_index"],
                "scores": scores,
            }, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            # Retain only bounded aggregate evidence, not all prompt metadata.
            runtime.calls.clear()
            del runtime.context_summaries[context_start:]
            runtime.diagnostics.clear()
            if completed % 32 == 0:
                print(json.dumps({
                    "completed": completed,
                    "total": len(rows),
                    "elapsed_seconds": time.monotonic() - acquisition,
                    "cell_id": args.cell_id,
                }), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - acquisition
    if stats["direction_fit_count"] != len(rows):
        raise RuntimeError("Phase 9 did not fit exactly one t0 direction per identity")
    stats["svd_seconds_mean"] = stats["svd_seconds_total"] / len(rows)
    summary = {
        "schema": "loopscope.phase9.score_summary.v1",
        "status": "SCORES_COMPLETE",
        "metadata": metadata,
        "count": len(rows),
        "first_index": indices[0],
        "last_index": indices[-1],
        "target_gold_loaded": False,
        "elapsed_seconds": time.monotonic() - started,
        "acquisition_seconds": elapsed,
        "samples_per_second": len(rows) / elapsed if elapsed else None,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "oom_count": 0,
        "positions_and_runtime": stats,
    }
    write_json_once(args.run_root / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "cell_id": args.cell_id,
        "selection": args.selection,
        "count": summary["count"],
        "samples_per_second": summary["samples_per_second"],
        "peak_reserved_bytes": summary["peak_reserved_bytes"],
        "target_gold_loaded": summary["target_gold_loaded"],
    }, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
