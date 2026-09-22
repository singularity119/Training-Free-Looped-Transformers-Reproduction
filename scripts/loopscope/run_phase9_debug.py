#!/usr/bin/env python3
"""Gate B real Phase 9 debug and validation-512 diagnostic runner.

The runner keeps the lm-eval 0.4.11 full-continuation scorer and the existing
TFLT wrapper intact.  It only binds the Phase 9 position adapter and residual
callback around the frozen batch-one MMLU recipe.  ``DEBUG_PREFLIGHT`` covers a
small sanitized validation pool plus the two new arms; ``FORMAL_DIAGNOSTIC``
collects label-free scalar trajectories for all frozen validation identities.
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CHOICES = "ABCD"
FORBIDDEN_ROW_FIELDS = {
    "answer", "target", "gold", "label", "correct", "correctness",
    "accuracy", "gain", "baseline", "outcome", "score",
}


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def finite_scores(values: Iterable[Any]) -> List[float]:
    scores = [float(value[0] if isinstance(value, (tuple, list)) else value) for value in values]
    if len(scores) != 4 or not all(math.isfinite(value) for value in scores):
        raise ValueError("expected four finite raw choice scores")
    return scores


def score_map(values: Iterable[Any], letters: Sequence[str] = CHOICES) -> Dict[str, float]:
    scores = finite_scores(values)
    if tuple(letters) not in (tuple(CHOICES), tuple(reversed(CHOICES))):
        raise ValueError("debug score order must be ABCD or DCBA")
    return {letter: score for letter, score in zip(letters, scores)}


def max_score_delta(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in CHOICES)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, help="legacy logical manifest check")
    result.add_argument("--config", type=Path, help="frozen Phase 9 model/window config")
    result.add_argument(
        "--mode", choices=("manifest", "tensor-smoke", "debug", "diagnostic"),
        default="debug",
    )
    result.add_argument("--scope", choices=("DEBUG_PREFLIGHT", "FORMAL_DIAGNOSTIC"))
    result.add_argument("--model-index", type=int, choices=(0, 1))
    result.add_argument("--pool", type=Path)
    result.add_argument("--run-root", type=Path)
    result.add_argument("--commit", help="source revision recorded by the launcher")
    result.add_argument(
        "--cell-index", type=int, action="append",
        help="frozen model/window/K cell index; repeat to run more than one cell",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def validate_pool(pool: Mapping[str, Any], scope: str) -> List[Dict[str, Any]]:
    expected_status = "SMOKE_ONLY" if scope == "DEBUG_PREFLIGHT" else "FORMAL_CALIBRATION"
    if pool.get("status") != expected_status:
        raise ValueError("pool status differs from the requested Gate B scope")
    dataset = pool.get("dataset")
    if dataset != {"repo": "cais/mmlu", "revision": "c30699e8356da336a370243923dbaf21066bb9fe", "split": "validation"}:
        raise ValueError("pool is not the frozen sanitized MMLU validation projection")
    rows = list(pool.get("rows", ()))
    expected_count = 8 if scope == "DEBUG_PREFLIGHT" else 512
    if len(rows) != expected_count:
        raise ValueError("Gate B pool row count differs from the frozen scope")
    identities = [row.get("identity") for row in rows]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Gate B pool identities must be unique")
    for row in rows:
        if row.get("split") != "validation" or not row.get("prompt") or len(row.get("choices", ())) != 4:
            raise ValueError("sanitized validation row is incomplete")
        if FORBIDDEN_ROW_FIELDS.intersection(row):
            raise ValueError("sanitized validation row contains a forbidden target/outcome field")
    if scope == "DEBUG_PREFLIGHT":
        if sum(row.get("role") == "fit" for row in rows) != 4 or sum(row.get("role") == "verify" for row in rows) != 4:
            raise ValueError("debug pool must contain four fit and four verify rows")
    return rows


def model_cells(config: Mapping[str, Any], model_index: int) -> Tuple[Mapping[str, Any], List[Tuple[List[int], int]]]:
    models = config.get("models", ())
    if not 0 <= model_index < len(models):
        raise ValueError("unsupported Phase 9 model index")
    spec = models[model_index]
    cells = [
        (list(window["layers"]), int(k))
        for window in spec["windows"]
        for k in config["K"]
    ]
    return spec, cells


def selected_cells(cells: Sequence[Tuple[List[int], int]], indexes: Sequence[int] | None) -> List[Tuple[List[int], int]]:
    if not indexes:
        return list(cells)
    if any(index < 0 or index >= len(cells) for index in indexes):
        raise ValueError("cell index outside the selected model's frozen cells")
    if len(set(indexes)) != len(indexes):
        raise ValueError("duplicate cell index")
    return [cells[index] for index in indexes]


def make_loop_config(spec: Mapping[str, Any], window: Sequence[int], k: int, runtime: Any = None) -> Any:
    from tflt.config import LoopConfig

    return LoopConfig(
        model_alias=spec["model"],
        window=tuple(window),
        k=k,
        alpha=1.0,
        beta=0.0,
        strategy="damped_euler",
        iteration_mode="block",
        cache_strategy=spec["cache_strategy"],
        decode_mode="bypass",
        residual_transform=runtime,
    )


def check_position_metadata(adapter: Any, tokenizer: Any) -> List[Dict[str, Any]]:
    special = set(int(value) for value in (getattr(tokenizer, "all_special_ids", ()) or ()))
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        special.add(int(pad))
    observed = []
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
            raise RuntimeError("effective attention mask is not a valid retained prefix")
        if any(value in special for value in effective):
            raise RuntimeError("effective Phase 9 mask includes a padding or special token")
        if item["input_length"] != len(actual) or len(item["prefix_tokens"]) != position + 1:
            raise RuntimeError("Phase 9 position metadata is inconsistent with actual evaluator input")
        if item["input_length"] != position + int(item["continuation_length"]):
            raise RuntimeError("Phase 9 answer boundary does not match continuation length")
        observed.append({
            "position": position,
            "valid_token_count": len(valid),
            "effective_valid_token_count": len(effective),
            "input_length": len(actual),
            "continuation_length": int(item["continuation_length"]),
        })
    return observed


def validate_context_calls(
    runtime: Any,
    k: int,
    expected_contexts: int,
    start_index: int | None = None,
) -> List[Dict[str, Any]]:
    # With lm-eval 0.4.11's default logits cache, the four one-token choice
    # requests share one actual causal model input.  The contract therefore
    # checks one shared candidate context with K real loop callbacks, not four
    # duplicated forwards.  A start index makes this an exact per-score check.
    contexts = (
        runtime.context_summaries[start_index:]
        if start_index is not None
        else runtime.context_summaries[-expected_contexts:]
    )
    if len(contexts) != expected_contexts:
        raise RuntimeError("Phase 9 evaluator did not produce the expected candidate contexts")
    for context in contexts:
        if context["call_count"] != k or context["timesteps"] != list(range(k)):
            raise RuntimeError("Phase 9 residual callback count differs from K")
    return contexts


def build_choice_requests(row: Mapping[str, Any], Instance: Any, tag: str) -> List[Any]:
    return [
        Instance(
            request_type="loglikelihood", doc={},
            arguments=(row["prompt"], " " + letter), idx=index,
            metadata=(tag, row["identity"], 1),
        )
        for index, letter in enumerate(CHOICES)
    ]


def load_runtime(spec: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from tflt.loopscope.phase9_adapter import build_hflm_class

    if importlib.metadata.version("lm_eval") != "0.4.11":
        raise RuntimeError("frozen scorer requires lm_eval 0.4.11")
    revision = spec["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=revision, local_files_only=True, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"], revision=revision, local_files_only=True, trust_remote_code=True,
        torch_dtype=getattr(torch, spec["dtype"]),
    ).to("cuda").eval()
    observed_revision = getattr(model.config, "_commit_hash", None)
    if observed_revision and observed_revision != revision:
        raise RuntimeError("model revision differs from the frozen contract")
    adapter_class = build_hflm_class(HFLM)
    vanilla = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="1", trust_remote_code=True)
    adapter = adapter_class(pretrained=model, tokenizer=tokenizer, batch_size="1", trust_remote_code=True)
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
        "gpu": torch.cuda.get_device_name(),
        "total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        "versions": versions,
    }
    return torch, model, tokenizer, vanilla, adapter, env


def native_scores(vanilla: Any, row: Mapping[str, Any], Instance: Any) -> Dict[str, float]:
    return score_map(vanilla.loglikelihood(build_choice_requests(row, Instance, "phase9-native"), disable_tqdm=True))


def adapted_scores(adapter: Any, row: Mapping[str, Any], runtime: Any, letters: str = CHOICES) -> Dict[str, float]:
    from tflt.loopscope.phase9_adapter import score_choices

    previous = adapter.phase9_runtime
    adapter.phase9_runtime = runtime
    try:
        return score_map(score_choices(adapter, row["prompt"], row["identity"], letters=letters), letters)
    finally:
        adapter.phase9_runtime = previous


def run_diagnostic_cell(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    spec: Mapping[str, Any],
    window: Sequence[int],
    k: int,
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Any]:
    import torch
    from tflt.loopscope.phase9_runtime import Phase9DiagnosticCollector, Phase9Runtime
    from tflt.wrapper import looped_model

    identities = [row["identity"] for row in rows]
    collector = Phase9DiagnosticCollector(identities, k)
    runtime = Phase9Runtime("Online-t0", collector=collector, intervene=False)
    config = make_loop_config(spec, window, k, runtime)
    started = time.monotonic()
    with looped_model(model, config):
        for row in rows:
            context_start = len(runtime.context_summaries)
            adapted_scores(adapter, row, runtime)
            check_position_metadata(adapter, tokenizer)
            validate_context_calls(runtime, k, 1, context_start)
            collector.finish(row["identity"], runtime.direction, runtime._fit_metadata)
    torch.cuda.synchronize()
    result = {
        "status": "DIAGNOSTIC_COMPLETE",
        "window": list(window),
        "k": k,
        "arm": "no-intervention-diagnostic",
        "elapsed_seconds": time.monotonic() - started,
        "collector": collector.summary(),
        "runtime": runtime.summary(),
        "records": collector.cell_records(),
        "target_gold_loaded": False,
        "choice_scores_saved": False,
    }
    return result, runtime


def run_debug_cell(
    model: Any,
    tokenizer: Any,
    vanilla: Any,
    adapter: Any,
    spec: Mapping[str, Any],
    window: Sequence[int],
    k: int,
    fit_rows: Sequence[Mapping[str, Any]],
    verify_rows: Sequence[Mapping[str, Any]],
    Instance: Any,
) -> Dict[str, Any]:
    from tflt.loopscope.phase9_runtime import Phase9Runtime
    from tflt.wrapper import looped_model

    diagnostics, diagnostic_runtime = run_diagnostic_cell(
        model, tokenizer, adapter, spec, window, k, fit_rows,
    )
    native = {row["identity"]: native_scores(vanilla, row, Instance) for row in verify_rows}
    loop_scores: Dict[str, Dict[str, float]] = {}
    zero_scores: Dict[str, Dict[str, float]] = {}
    loop_config = make_loop_config(spec, window, k, None)
    with looped_model(model, loop_config):
        for row in verify_rows:
            loop_scores[row["identity"]] = adapted_scores(adapter, row, None)
            check_position_metadata(adapter, tokenizer)
    zero_runtime = Phase9Runtime("Online-t0", strength=0.0, intervene=True)
    zero_config = make_loop_config(spec, window, k, zero_runtime)
    with looped_model(model, zero_config):
        for row in verify_rows:
            context_start = len(zero_runtime.context_summaries)
            zero_scores[row["identity"]] = adapted_scores(adapter, row, zero_runtime)
            check_position_metadata(adapter, tokenizer)
            validate_context_calls(zero_runtime, k, 1, context_start)
    zero_deltas = {
        identity: max_score_delta(loop_scores[identity], zero_scores[identity])
        for identity in loop_scores
    }
    if any(delta != 0.0 for delta in zero_deltas.values()):
        raise RuntimeError("zero-strength Phase 9 path changed frozen Loop scores")

    arm_reports = {}
    for mode in ("Online-t0", "Matched-norm"):
        runtime = Phase9Runtime(mode, strength=0.5, intervene=True)
        config = make_loop_config(spec, window, k, runtime)
        rows_report = []
        with looped_model(model, config):
            for row in verify_rows:
                context_start = len(runtime.context_summaries)
                call_start = len(runtime.calls)
                scores = adapted_scores(adapter, row, runtime)
                positions = check_position_metadata(adapter, tokenizer)
                recent = validate_context_calls(runtime, k, 1, context_start)
                row_calls = runtime.calls[call_start:]
                if any(call["t"] < 1 and call["applied"] for call in row_calls):
                    raise RuntimeError("Phase 9 intervention was applied at t0")
                if any(call["t"] >= 1 and not call["applied"] for call in row_calls):
                    raise RuntimeError("Phase 9 intervention was not applied at t>=1")
                if any(call["nonanswer_max_abs_change"] != 0.0 for call in row_calls):
                    raise RuntimeError("Phase 9 intervention changed a non-answer position")
                rows_report.append({
                    "identity": row["identity"],
                    "scores": scores,
                    "max_abs_vs_loop": max_score_delta(scores, loop_scores[row["identity"]]),
                    "positions": positions,
                    "calls": row_calls,
                    "contexts": recent,
                })
        if not any(item["max_abs_vs_loop"] > 0.0 for item in rows_report):
            raise RuntimeError("Phase 9 arm produced no observable intervention on the debug verify rows")
        arm_reports[mode] = {
            "rows": rows_report,
            "runtime": runtime.summary(),
            "changed_score_row_count": sum(item["max_abs_vs_loop"] > 0.0 for item in rows_report),
        }

    # Reorder only the four score requests for one representative row.  The
    # returned map must be invariant to request order, while a fresh runtime
    # proves that the candidate-local t0 direction is not order dependent.
    representative = verify_rows[0]
    ordered_runtime = Phase9Runtime("Online-t0", intervene=False)
    with looped_model(model, make_loop_config(spec, window, k, ordered_runtime)):
        context_start = len(ordered_runtime.context_summaries)
        standard = adapted_scores(adapter, representative, ordered_runtime, letters="ABCD")
        standard_positions = check_position_metadata(adapter, tokenizer)
        validate_context_calls(ordered_runtime, k, 1, context_start)
        context_start = len(ordered_runtime.context_summaries)
        reversed_scores = adapted_scores(adapter, representative, ordered_runtime, letters="DCBA")
        reversed_positions = check_position_metadata(adapter, tokenizer)
        validate_context_calls(ordered_runtime, k, 1, context_start)
    order_delta = max_score_delta(standard, reversed_scores)
    if order_delta != 0.0 or standard_positions != reversed_positions:
        raise RuntimeError("candidate request order changed Phase 9 score or mask semantics")
    if len(ordered_runtime.diagnostics) != 1:
        raise RuntimeError("same prompt did not reuse exactly one t0 direction")

    return {
        "status": "DEBUG_CELL_COMPLETE",
        "window": list(window),
        "k": k,
        "diagnostic": diagnostics,
        "native_scores": native,
        "loop_scores": loop_scores,
        "zero_strength_scores": zero_scores,
        "zero_strength_max_abs_delta": zero_deltas,
        "arms": arm_reports,
        "candidate_order_checks": [{
            "identity": representative["identity"],
            "score_max_abs_order_delta": order_delta,
            "direction_fit_count": len(ordered_runtime.diagnostics),
            "context_count": len(ordered_runtime.context_summaries),
        }],
        "fit_identity_count": len(fit_rows),
        "verify_identity_count": len(verify_rows),
        "target_gold_loaded": False,
        "choice_scores_saved": True,
        "diagnostic_runtime_cache_resets": diagnostic_runtime.cache_reset_count,
    }


def base_parser_data(args: argparse.Namespace, spec: Mapping[str, Any], cells: Sequence[Tuple[List[int], int]]) -> Dict[str, Any]:
    return {
        "scope": args.scope,
        "model_index": args.model_index,
        "model": spec["model"],
        "revision": spec["revision"],
        "dtype": spec["dtype"],
        "cache_strategy": spec["cache_strategy"],
        "cells": [{"window": window, "k": k} for window, k in cells],
        "source_commit": args.commit,
        "pool": str(args.pool),
        "argv": sys.argv,
    }


def run_model_scope(args: argparse.Namespace) -> int:
    if args.scope not in ("DEBUG_PREFLIGHT", "FORMAL_DIAGNOSTIC"):
        raise ValueError("--scope is required for real model execution")
    if args.model_index is None or args.pool is None or args.run_root is None or not args.commit:
        raise ValueError("real model execution requires --model-index, --pool, --run-root and --commit")
    repo = Path(__file__).resolve().parents[2]
    config = json.loads((repo / "configs/loopscope/phase9_model_windows.json").read_text(encoding="utf-8"))
    spec, all_cells = model_cells(config, args.model_index)
    cells = selected_cells(all_cells, args.cell_index)
    if args.dry_run:
        print(json.dumps({
            "scope": args.scope,
            "model": spec["model"],
            "revision": spec["revision"],
            "cells": [{"window": window, "k": k} for window, k in cells],
            "pool": str(args.pool),
            "run_root": str(args.run_root),
            "partition": "debug" if args.scope == "DEBUG_PREFLIGHT" else "caller-selected",
            "target_gold_loaded": False,
        }, indent=2, ensure_ascii=False))
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate B model execution requires a real Slurm job")
    if args.scope == "DEBUG_PREFLIGHT" and os.environ.get("SLURM_JOB_PARTITION") != "debug":
        raise RuntimeError("Gate B debug preflight requires the debug partition")
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    rows = validate_pool(pool, args.scope)
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json_once(args.run_root / "command_args.json", base_parser_data(args, spec, cells))
    torch, model, tokenizer, vanilla, adapter, env = load_runtime(spec)
    env.update({"source_commit": args.commit, "scope": args.scope, "target_gold_loaded": False})
    write_json_once(args.run_root / "env.json", env)
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    from lm_eval.api.instance import Instance
    reports = []
    if args.scope == "DEBUG_PREFLIGHT":
        fit_rows = [row for row in rows if row.get("role") == "fit"]
        verify_rows = [row for row in rows if row.get("role") == "verify"]
        indexes = args.cell_index or list(range(len(all_cells)))
        for cell_index, (window, k) in zip(indexes, cells):
            cell_name = "w%d-%d-k%d" % (*window, k)
            cell_root = args.run_root / cell_name
            cell_root.mkdir()
            report = run_debug_cell(
                model, tokenizer, vanilla, adapter, spec, window, k,
                fit_rows, verify_rows, Instance,
            )
            report["cell_index"] = cell_index
            write_json_once(cell_root / "debug.json", report)
            reports.append(report)
    else:
        indexes = args.cell_index or list(range(len(all_cells)))
        for cell_index, (window, k) in zip(indexes, cells):
            cell_name = "w%d-%d-k%d" % (*window, k)
            cell_root = args.run_root / cell_name
            cell_root.mkdir()
            report, _ = run_diagnostic_cell(model, tokenizer, adapter, spec, window, k, rows)
            report["cell_index"] = cell_index
            write_json_once(cell_root / "diagnostics.json", report)
            reports.append(report)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    summary = {
        "status": "DEBUG_PREFLIGHT_COMPLETE" if args.scope == "DEBUG_PREFLIGHT" else "FORMAL_DIAGNOSTIC_COMPLETE",
        "scope": args.scope,
        "model": spec["model"],
        "revision": spec["revision"],
        "source_commit": args.commit,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cell_count": len(reports),
        "cell_names": ["w%d-%d-k%d" % (*report["window"], report["k"]) for report in reports],
        "row_count": len(rows),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "oom_count": 0,
        "target_gold_loaded": False,
        "test_split_loaded": False,
        "reports": reports if args.scope == "DEBUG_PREFLIGHT" else [
            {key: report[key] for key in ("status", "window", "k", "collector", "elapsed_seconds", "target_gold_loaded")}
            for report in reports
        ],
    }
    write_json_once(args.run_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode in ("manifest", "tensor-smoke"):
        if args.manifest is None:
            raise ValueError("--manifest is required for manifest/tensor-smoke mode")
        from tflt.loopscope.phase9_accuracy import load_config, validate_panel
        config = load_config(args.config)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        counts = validate_panel(manifest.get("cells", []), config)
        if args.mode == "manifest":
            print(json.dumps({"status": "DEBUG_MANIFEST_READY", **counts}, sort_keys=True))
            return 0
        import torch
        from tflt.loopscope.phase9_runtime import tensor_smoke_checks
        print(json.dumps({"status": "TENSOR_SMOKE_COMPLETE", "checks": tensor_smoke_checks(torch)}, sort_keys=True))
        return 0
    return run_model_scope(args)


if __name__ == "__main__":
    raise SystemExit(main())
