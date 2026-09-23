#!/usr/bin/env python3
"""Real-model debug of Gate E's direction-policy/intervention-mode switches."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Sequence

from tflt.loopscope.phase9_gate_e_accuracy import load_config, new_cells
from tflt.loopscope.phase9_gate_e_spectral_core import (
    DIRECTION_POLICIES,
    INTERVENTION_MODES,
    direction_schedule,
)


# Predeclared exact-equivalence thresholds. The frozen Phase 9 real-debug
# already required zero raw-score delta for zero-strength and candidate-order
# checks; K2 follows the same deterministic model/scoring path and shared SVD.
SCORE_ABS_ATOL = 0.0
RESIDUAL_ABS_ATOL = 0.0


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--model-index", type=int, choices=(0, 1), required=True)
    result.add_argument("--direction-policies", nargs="+", choices=DIRECTION_POLICIES, required=True)
    result.add_argument("--intervention-modes", nargs="+", choices=INTERVENTION_MODES, required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--dry-run", action="store_true")
    return result


def select_rows(pool: Mapping[str, Any], model: str) -> list[int]:
    rows = pool["rows"]
    ordinary = 0
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (-rows[index]["prompt_token_lengths"][model], index),
    )
    upper_tail = ranked[0] if ranked[0] != ordinary else ranked[1]
    return [ordinary, upper_tail]


def max_score_delta(left: Sequence[float], right: Sequence[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def max_residual_delta(left: Sequence[Any], right: Sequence[Any]) -> float:
    import torch

    if len(left) != len(right) or not left:
        raise RuntimeError("Gate E residual comparison has mismatched round counts")
    return max(float(torch.max(torch.abs(a - b))) for a, b in zip(left, right))


@contextmanager
def capture_answer_rows(runtime_module: Any):
    """Capture transient answer rows for a direct K2 residual-equivalence check."""

    original = runtime_module._tensor_change
    captured = []

    def recording_change(torch: Any, before: Any, after: Any, position: int):
        captured.append(after[0, position].detach().float().cpu().clone())
        return original(torch, before, after, position)

    runtime_module._tensor_change = recording_change
    try:
        yield captured
    finally:
        runtime_module._tensor_change = original


def validate_calls(runtime: Any, cell: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    k = int(cell["k"])
    schedule = direction_schedule(cell["direction_policy"], k)
    calls = runtime.calls
    if len(calls) != k or [call["t"] for call in calls] != list(range(k)):
        raise RuntimeError("Gate E debug callback count/order differs from frozen K")
    fit_times = {int(t) for t in runtime.fit_metadata}
    expected_fits = {fit for _, _, fit in schedule if fit is not None}
    if fit_times != expected_fits:
        raise RuntimeError("Gate E debug direction-fit rounds differ from selected policy")
    for call, (t, used_t, fit_t) in zip(calls, schedule):
        if call["nonanswer_max_abs_change"] != 0.0:
            raise RuntimeError("Gate E debug changed non-answer residual positions")
        if call["applied_t"] != (t if t > 0 else None):
            raise RuntimeError("Gate E debug intervention round differs from frozen schedule")
        if call["direction_used_fit_t"] != used_t or call["direction_fit_t"] != fit_t:
            raise RuntimeError("Gate E debug direction source/fit round differs from policy")
        if not math.isfinite(float(call["answer_max_abs_change"])):
            raise RuntimeError("Gate E debug produced a nonfinite residual update")
    return {
        "direction_policy": cell["direction_policy"],
        "intervention_mode": cell["intervention_mode"],
        "cell_id": cell["cell_id"],
        "identity": row["identity"],
        "k": k,
        "round_log": [
            {
                "t": call["t"], "applied_t": call["applied_t"],
                "direction_used_fit_t": call["direction_used_fit_t"],
                "direction_fit_t": call["direction_fit_t"],
                "answer_max_abs_change": call["answer_max_abs_change"],
                "norm_ratio": call["norm_ratio"], "coverage": call["coverage"],
            }
            for call in calls
        ],
        "fit_count": len(fit_times),
        "max_svd_seconds": max(float(value["svd_seconds"]) for value in runtime.fit_metadata.values()),
    }


def run(args: argparse.Namespace) -> int:
    from tflt.loopscope.phase8_test_pool import validate_test_pool

    if set(args.direction_policies) != set(DIRECTION_POLICIES) or len(args.direction_policies) != len(DIRECTION_POLICIES):
        raise ValueError("debug must explicitly request both fixed_t0 and lag1 policies once")
    if set(args.intervention_modes) != set(INTERVENTION_MODES) or len(args.intervention_modes) != len(INTERVENTION_MODES):
        raise ValueError("debug must explicitly request both spectral and matched_norm modes once")
    config = load_config(args.config)
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    validate_test_pool(pool)
    model_spec = config["models"][args.model_index]
    rows = pool["rows"]
    chosen_indices = select_rows(pool, model_spec["model"])
    cells = [cell for cell in new_cells(config) if cell["model_index"] == args.model_index]
    windows = list(dict.fromkeys(tuple(cell["window"]) for cell in cells))
    if args.dry_run:
        print(json.dumps({
            "status": "PHASE9_GATE_E_DEBUG_DRY_RUN", "model": model_spec["model"],
            "revision": model_spec["revision"], "windows": [list(value) for value in windows],
            "rows": [{"index": index, "identity": rows[index]["identity"],
                      "prompt_tokens": rows[index]["prompt_token_lengths"][model_spec["model"]]}
                     for index in chosen_indices],
            "direction_policies": args.direction_policies,
            "intervention_modes": args.intervention_modes,
            "k_values": [3, 4], "k2_equivalence": True, "target_gold_loaded": False,
        }, sort_keys=True))
        return 0
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "debug":
        raise RuntimeError("Gate E real debug requires a Slurm job on the debug partition")

    from tflt.loopscope import phase9_gate_e_runtime as gate_e_runtime_module
    from tflt.loopscope import phase9_runtime as phase9_runtime_module
    from tflt.loopscope.phase9_adapter import score_choices
    from tflt.loopscope.phase9_gate_e_runtime import Phase9GateERuntime
    from tflt.loopscope.phase9_runtime import Phase9Runtime
    from tflt.loopscope.phase9_spectral_core import MATCHED_NORM, ONLINE_T0
    from tflt.wrapper import looped_model
    from run_phase9_gate_e_accuracy import (
        finite_scores,
        load_runtime,
        make_loop_config,
    )

    representative_cell = next(cell for cell in cells if cell["k"] == 3)
    torch, model, tokenizer, adapter, _, env = load_runtime(
        representative_cell, args.direction_policies[0], args.intervention_modes[0]
    )
    torch.set_grad_enabled(False)
    torch.manual_seed(20260923)
    torch.cuda.reset_peak_memory_stats()
    env.update({
        "source_commit": args.commit, "target_gold_loaded": False,
        "test_pool_gold_free": True, "debug_partition": True,
        "debug_direction_policies": list(args.direction_policies),
        "debug_intervention_modes": list(args.intervention_modes),
    })
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json_once(args.run_root / "command_args.json", {
        "schema": "loopscope.phase9.gate_e.debug_attempt.v2",
        "model_index": args.model_index,
        "model": model_spec["model"],
        "revision": model_spec["revision"],
        "windows": [list(value) for value in windows],
        "prompt_indices": chosen_indices,
        "pool": str(args.pool),
        "source_commit": args.commit,
        "direction_policies": list(args.direction_policies),
        "intervention_modes": list(args.intervention_modes),
        "score_abs_atol": SCORE_ABS_ATOL,
        "residual_abs_atol": RESIDUAL_ABS_ATOL,
        "argv": sys.argv,
        "target_gold_loaded": False,
    })
    write_json_once(args.run_root / "env.json", env)

    def score_one(cell: Mapping[str, Any], row: Mapping[str, Any], runtime: Any) -> list[float]:
        adapter.phase9_runtime = runtime
        started_contexts = len(runtime.context_summaries)
        started_calls = len(runtime.calls)
        with looped_model(model, make_loop_config(cell, runtime)):
            values = finite_scores(score_choices(adapter, row["prompt"], row["identity"]))
        contexts = runtime.context_summaries[started_contexts:]
        calls = runtime.calls[started_calls:]
        k = int(cell["k"])
        if len(contexts) != 1 or contexts[0]["timesteps"] != list(range(k)):
            raise RuntimeError("Gate E debug did not produce one ordered K-step model context")
        if len(calls) != k:
            raise RuntimeError("Gate E debug callback count differs from K")
        return values

    started = time.monotonic()
    policy_checks = []
    shared_round_checks = []
    contamination_checks = []
    for base_cell in cells:
        if base_cell["k"] not in (3, 4):
            continue
        for intervention_mode in args.intervention_modes:
            runs_by_policy = {}
            for direction_policy in args.direction_policies:
                cell = dict(base_cell, direction_policy=direction_policy, intervention_mode=intervention_mode)
                runtime = Phase9GateERuntime(
                    direction_policy, intervention_mode, int(cell["k"]), strength=0.5, intervene=True
                )
                policy_rows = []
                for row_index in chosen_indices:
                    row = rows[row_index]
                    call_start = len(runtime.calls)
                    context_start = len(runtime.context_summaries)
                    with capture_answer_rows(gate_e_runtime_module) as answer_rows:
                        scores = score_one(cell, row, runtime)
                    row_calls = runtime.calls[call_start:]
                    details = validate_calls(runtime, cell, row)
                    if len(answer_rows) != int(cell["k"]):
                        raise RuntimeError("Gate E debug transient residual capture differs from K")
                    policy_rows.append({
                        "identity": row["identity"], "calls": list(row_calls),
                        "answer_rows": answer_rows,
                    })
                    policy_checks.append({
                        **details,
                        "finite_scores": all(math.isfinite(value) for value in scores),
                        "score_abs_max": max(abs(value) for value in scores),
                        "prompt_tokens": row["prompt_token_lengths"][model_spec["model"]],
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    })
                    runtime.calls.clear()
                    del runtime.context_summaries[context_start:]
                if runtime.cache_reset_count != 1:
                    raise RuntimeError("Gate E runtime did not release the prior direction when the prompt changed")
                contamination_checks.append({
                    "direction_policy": direction_policy,
                    "intervention_mode": intervention_mode,
                    "cell_id": base_cell["cell_id"],
                    "identity_pair": [rows[index]["identity"] for index in chosen_indices],
                    "cache_reset_count": runtime.cache_reset_count,
                    "status": "PASS",
                })
                if len(policy_rows) == 2:
                    first_calls, second_calls = policy_rows[0]["calls"], policy_rows[1]["calls"]
                    if len(first_calls) != len(second_calls):
                        raise RuntimeError("Gate E prompt switch retained prior-round call state")
                runs_by_policy[direction_policy] = policy_rows

            # The first two rounds are identical under both policies; the
            # direction source differs at t=2 and, for K4, at t=3. Compare the
            # transient rows collected in the same scorer pass above.
            fixed_runs = runs_by_policy["fixed_t0"]
            lag1_runs = runs_by_policy["lag1"]
            for fixed_run, lag1_run in zip(fixed_runs, lag1_runs):
                if fixed_run["identity"] != lag1_run["identity"]:
                    raise RuntimeError("Gate E policy comparison identities differ")
                fixed = fixed_run["calls"]
                lag1 = lag1_run["calls"]
                for t in (0, 1):
                    if max_residual_delta(
                            [fixed_run["answer_rows"][t]], [lag1_run["answer_rows"][t]]) > RESIDUAL_ABS_ATOL:
                        raise RuntimeError("Gate E fixed_t0 and lag1 differ before their direction schedules diverge")
                    for key in ("answer_max_abs_change", "norm_ratio", "coverage"):
                        left, right = fixed[t].get(key), lag1[t].get(key)
                        if left is None or right is None:
                            if left != right:
                                raise RuntimeError("Gate E shared-round diagnostic differs between policies")
                        elif not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=RESIDUAL_ABS_ATOL):
                            raise RuntimeError("Gate E fixed_t0 and lag1 shared-round effects differ")
                if fixed[2]["direction_used_fit_t"] == lag1[2]["direction_used_fit_t"]:
                    raise RuntimeError("Gate E direction policies did not diverge at t2")
                if int(base_cell["k"]) == 4 and lag1[3]["direction_used_fit_t"] != 2:
                    raise RuntimeError("Gate E K4 t3 did not use the direction fitted from D2")
                shared_round_checks.append({
                    "cell_id": base_cell["cell_id"],
                    "identity": fixed_run["identity"],
                    "intervention_mode": intervention_mode,
                    "k": int(base_cell["k"]),
                    "shared_rounds": [0, 1],
                    "shared_residual_max_abs_delta": max_residual_delta(
                        fixed_run["answer_rows"][:2], lag1_run["answer_rows"][:2]
                    ),
                    "policy_divergence_at_t2": {
                        "fixed_t0_uses_fit_t": fixed[2]["direction_used_fit_t"],
                        "lag1_uses_fit_t": lag1[2]["direction_used_fit_t"],
                    },
                    "k4_t3_lag1_uses_fit_t": lag1[3]["direction_used_fit_t"] if int(base_cell["k"]) == 4 else None,
                })

    # K2 collapses both direction policies to the original Phase 9 algorithms.
    # Compare scores and transient answer-residual rows, not merely predictions.
    k2_checks = []
    k2_row_index = chosen_indices[-1]
    for window in windows:
        for intervention_mode in args.intervention_modes:
            old_mode = ONLINE_T0 if intervention_mode == "spectral" else MATCHED_NORM
            for direction_policy in args.direction_policies:
                cell = dict(
                    representative_cell, window=list(window), k=2,
                    direction_policy=direction_policy,
                    intervention_mode=intervention_mode,
                )
                new_runtime = Phase9GateERuntime(direction_policy, intervention_mode, 2, intervene=True)
                with capture_answer_rows(gate_e_runtime_module) as new_rows:
                    new_scores = score_one(cell, rows[k2_row_index], new_runtime)
                old_runtime = Phase9Runtime(old_mode, strength=0.5, intervene=True)
                with capture_answer_rows(phase9_runtime_module) as old_rows:
                    old_scores = score_one(cell, rows[k2_row_index], old_runtime)
                score_delta = max_score_delta(new_scores, old_scores)
                residual_delta = max_residual_delta(new_rows, old_rows)
                if score_delta > SCORE_ABS_ATOL or residual_delta > RESIDUAL_ABS_ATOL:
                    raise RuntimeError("Gate E K2 differs from its corresponding frozen Phase 9 arm")
                k2_checks.append({
                    "window": list(window), "direction_policy": direction_policy,
                    "intervention_mode": intervention_mode, "old_phase9_mode": old_mode,
                    "identity": rows[k2_row_index]["identity"],
                    "max_abs_score_delta": score_delta,
                    "max_abs_residual_delta": residual_delta,
                    "score_abs_atol": SCORE_ABS_ATOL,
                    "residual_abs_atol": RESIDUAL_ABS_ATOL,
                    "score_abs_scale": max(max(abs(value) for value in new_scores), max(abs(value) for value in old_scores)),
                    "new_direction_fit_count": len(new_runtime.fit_metadata),
                    "old_direction_fit_count": len(old_runtime.diagnostics),
                })
                new_runtime.calls.clear()
                old_runtime.calls.clear()

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    summary = {
        "schema": "loopscope.phase9.gate_e.debug_summary.v2",
        "status": "DEBUG_PREFLIGHT_COMPLETE",
        "model": model_spec["model"], "revision": model_spec["revision"],
        "source_commit": args.commit, "job_id": env["job_id"],
        "partition": env["partition"], "gpu": env["gpu"],
        "direction_policies": list(args.direction_policies),
        "intervention_modes": list(args.intervention_modes),
        "score_abs_atol": SCORE_ABS_ATOL, "residual_abs_atol": RESIDUAL_ABS_ATOL,
        "prompt_indices": chosen_indices,
        "prompt_count": len(chosen_indices),
        "k3_k4_check_count": len(policy_checks), "k3_k4_checks": policy_checks,
        "shared_round_check_count": len(shared_round_checks), "shared_round_checks": shared_round_checks,
        "switch_isolation_check_count": len(contamination_checks),
        "switch_isolation_checks": contamination_checks,
        "k2_equivalence_count": len(k2_checks), "k2_equivalence_checks": k2_checks,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "oom_count": 0,
        "target_gold_loaded": False,
        "test_pool_gold_free": True,
    }
    write_json_once(args.run_root / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"], "model": summary["model"],
        "k3_k4_checks": summary["k3_k4_check_count"],
        "shared_round_checks": summary["shared_round_check_count"],
        "switch_isolation_checks": summary["switch_isolation_check_count"],
        "k2_equivalence_checks": summary["k2_equivalence_count"],
        "elapsed_seconds": elapsed, "peak_reserved_bytes": summary["peak_reserved_bytes"],
        "target_gold_loaded": False,
    }, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
