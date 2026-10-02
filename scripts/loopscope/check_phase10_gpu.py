#!/usr/bin/env python3
"""Bounded, gold-free Gate B checks on real model CUDA residuals and HFLM scores.

Run once per frozen model/task/K in an allocated debug GPU. Hidden tensors stay
in memory and are discarded after the first candidate's tensor checks.
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

from tflt.loopscope.phase10_accuracy import (
    choices_for_row, load_pool, require, row_identity, select_indices, write_json_once,
)
from tflt.loopscope.phase10_panel import load_config, validate_score_manifest
from tflt.loopscope.phase10_runtime import make_runtime, validate_calls
from tflt.loopscope.phase9_runtime import Phase9Runtime, _fit_direction
from run_phase10_accuracy import load_runtime, make_loop_config


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("mmlu", "arc_challenge"), required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--cell-id", required=True, help="Online fixed_t0 cell for the model/window/K")
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--preflight-indices", type=Path)
    result.add_argument("--max-length", type=int)
    return result


def numeric_comparison(left, right):
    """Report errors before applying the frozen exact-equivalence criterion."""
    require(len(left) == len(right) and len(left) > 0, "comparison lengths differ/empty")
    a, b = [float(v) for v in left], [float(v) for v in right]
    require(all(math.isfinite(v) for v in a + b), "nonfinite comparison")
    differences = [abs(x-y) for x, y in zip(a, b)]
    relative = [d / max(abs(x), abs(y), sys.float_info.min)
                for x, y, d in zip(a, b, differences)]
    return {"max_abs_error": max(differences), "max_relative_error": max(relative),
            "exact": a == b}


def tensor_comparison(torch, left, right):
    require(left.shape == right.shape, "tensor comparison shapes differ")
    require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), "nonfinite tensor")
    a, b = left.float(), right.float()
    difference = (a-b).abs()
    denominator = torch.maximum(a.abs(), b.abs()).clamp_min(torch.finfo(torch.float32).tiny)
    return {"max_abs_error": float(difference.max()),
            "max_relative_error": float((difference/denominator).max()),
            "exact": bool(torch.equal(left, right))}


def score_comparison(left, right):
    require(len(left) == len(right), "candidate score counts differ")
    result = numeric_comparison([v[0] for v in left], [v[0] for v in right])
    result["greedy_flags_exact"] = [bool(v[1]) for v in left] == [bool(v[1]) for v in right]
    result["exact"] = result["exact"] and result["greedy_flags_exact"]
    return result


def validation_selection(rows, cell, explicit=None):
    require(all(row.get("split") == "validation" for row in rows),
            "GPU checker permits validation targets only")
    selected = select_indices(rows, cell, "PREFLIGHT_ONLY", explicit)
    if explicit is None:
        # First plus longest (with canonical tie breaking) bounds model work.
        key = lambda i: (rows[i].get("prompt_token_lengths", {}).get(cell["model"], 0)
                         if cell["dataset"] == "mmlu" else max(
                             v["context_length"] for v in rows[i]["tokenization"].values()))
        selected = sorted({0, max(range(len(rows)), key=key)})
    require(1 <= len(selected) <= 16, "checker requires 1..16 fixed validation indices")
    return selected


class CaptureResidual:
    """Observe unchanged real Loop residuals for one full candidate forward."""
    def __init__(self):
        self.raw = []
        self.active = None
        self.metadata = None
        self.done = False

    @contextmanager
    def context(self, identity, position, valid_positions, token_ids=None):
        require(self.active is None, "nested capture context")
        self.active = (identity, position, tuple(valid_positions), tuple(token_ids))
        if not self.done:
            self.metadata = self.active
        try:
            yield self
        finally:
            self.active = None
            self.done = True

    def __call__(self, delta, t):
        require(self.active is not None, "capture outside context")
        if not self.done:
            require(t == len(self.raw), "capture callbacks out of order")
            self.raw.append(delta.detach().clone())
        return delta


def replay(torch, runtime, raw, metadata):
    identity, position, valid, tokens = metadata
    output, projections = [], []
    with runtime.context(identity, position, valid, token_ids=tokens):
        for t, delta in enumerate(raw):
            output.append(runtime(delta, t))
            vector = runtime.direction
            projections.append(vector * torch.dot(vector, delta[0, position].float()))
    require(runtime._active is None, "runtime context was not released")
    return output, projections


def residual_checks(torch, cell, capture):
    raw, metadata = capture.raw, capture.metadata
    require(len(raw) == cell["k"] and all(v.device.type == "cuda" for v in raw),
            "real CUDA trajectory was not captured")
    results = {"source": "real_model_wrapper_raw_loop_residual", "shape": list(raw[0].shape),
               "dtype": str(raw[0].dtype), "checks": []}
    def compare(label, a, b):
        results["checks"].append({"check": label, **tensor_comparison(torch, a, b)})
    for policy in ("fixed_t0", "lag1", "current_t"):
        runtime = make_runtime({**cell, "direction_policy": policy, "strength": 0.0})
        outputs, _ = replay(torch, runtime, raw, metadata)
        for t, output in enumerate(outputs):
            compare(f"lambda0-{policy}-t{t}", output, raw[t])
    historical = Phase9Runtime("Online-t0", strength=0.5)
    old, old_projection = replay(torch, historical, raw, metadata)
    current, current_projection = replay(torch, make_runtime({**cell, "strength": 0.5}), raw, metadata)
    for t in range(cell["k"]):
        compare(f"phase9-fixed0.5-result-t{t}", current[t], old[t])
        compare(f"phase9-fixed0.5-projection-t{t}", current_projection[t], old_projection[t])
    if cell["k"] == 2:
        for strength in (0.1, 0.5, 0.9):
            fixed, fp = replay(torch, make_runtime({**cell, "strength": strength}), raw, metadata)
            lag, lp = replay(torch, make_runtime({**cell, "direction_policy": "lag1", "strength": strength}), raw, metadata)
            for t in range(2):
                compare(f"k2-lambda{strength}-result-t{t}", fixed[t], lag[t])
                compare(f"k2-lambda{strength}-projection-t{t}", fp[t], lp[t])
    # Drop every hidden reference before other model work.
    capture.raw.clear()
    results["exact"] = all(item["exact"] for item in results["checks"])
    return results


class CurrentOracle:
    """Check same-round fitting on the actual intervened model trajectory."""
    def __init__(self, torch, cell):
        self.torch = torch
        self.runtime = make_runtime({**cell, "direction_policy": "current_t", "strength": 0.5})
        self.checks = []
        self.contexts = 0

    @contextmanager
    def context(self, *args, **kwargs):
        with self.runtime.context(*args, **kwargs):
            require(self.runtime.direction is None and not self.runtime.fit_metadata,
                    "current candidate inherited a preceding direction")
            yield self
        require(self.runtime._active is None and self.runtime._last_context_t is None,
                "current candidate context was not released")
        self.contexts += 1

    def __call__(self, delta, t):
        torch, runtime = self.torch, self.runtime
        position = runtime._active["position"]
        positions = runtime._active["valid_positions"]
        index = torch.tensor(positions, device=delta.device, dtype=torch.long)
        # Independent fit sees the raw tensor before the production callback.
        vector, metadata = _fit_direction(delta[0].index_select(0, index), torch)
        row = delta[0, position].float()
        expected = (row - runtime.strength * vector * torch.dot(vector, row)).to(delta.dtype)
        result = runtime(delta, t)
        check = tensor_comparison(torch, result[0, position], expected)
        projection = tensor_comparison(torch, vector * torch.dot(vector, row),
                                       runtime.direction * torch.dot(runtime.direction, row))
        call = runtime.calls[-1]
        check.update(t=t, raw_fit_metadata=metadata, projection=projection,
                     fit_from_unmodified_current_residual=True,
                     nonanswer_max_abs_change=call["nonanswer_max_abs_change"],
                     applied=call["applied"], direction_fit_t=call["direction_fit_t"])
        check["exact"] = (check["exact"] and projection["exact"]
                          and call["nonanswer_max_abs_change"] == 0
                          and call["applied"] and call["direction_fit_t"] == t)
        self.checks.append(check)
        return result


class LifecycleRuntime:
    """Expose candidate restart/release evidence without changing callbacks."""
    def __init__(self, cell):
        self.runtime = make_runtime(cell)
        self.policy = cell["direction_policy"]
        self.boundaries = []

    @contextmanager
    def context(self, *args, **kwargs):
        with self.runtime.context(*args, **kwargs):
            started_empty = self.runtime.direction is None
            if self.policy == "lag1":
                require(started_empty and not self.runtime.fit_metadata,
                        "lag1 candidate inherited a preceding direction")
            yield self
        released = self.runtime._active is None
        require(released, "candidate context was not released")
        self.boundaries.append({"direction_empty_at_start": started_empty,
                                "active_context_released": released})

    def __call__(self, delta, t):
        return self.runtime(delta, t)


def run(args):
    manifest = json.loads(args.manifest.read_text())
    cells = validate_score_manifest(manifest, load_config(args.config), args.pool, "PREFLIGHT_ONLY")
    require(manifest["dataset"] == args.dataset and args.cell_id in cells, "manifest/cell/dataset differs")
    cell = cells[args.cell_id]
    require(cell["arm"] == "Online" and cell["direction_policy"] == "fixed_t0", "select canonical fixed_t0 Online cell")
    rows = load_pool(args.pool, args.dataset, "PREFLIGHT_ONLY")
    explicit = json.loads(args.preflight_indices.read_text()) if args.preflight_indices else None
    indices = validation_selection(rows, cell, explicit)
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "GPU checks require an allocated debug Slurm job")
    args.run_root.mkdir(parents=True, exist_ok=False)
    command = {"schema": "loopscope.phase10.gpu_check_attempt.v1", "scope": "PREFLIGHT_ONLY",
               "source_commit": args.commit, "cell": cell, "pool": str(args.pool),
               "indices": indices, "identities": [row_identity(rows[i], args.dataset) for i in indices],
               "argv": sys.argv, "target_gold_loaded": False, "equivalence_criterion": "exact"}
    write_json_once(args.run_root/"command_args.json", command)
    started = time.monotonic()
    try:
        torch, model, tokenizer, adapter, _, env = load_runtime(cell, args.max_length)
        write_json_once(args.run_root/"env.json", {**env, "source_commit": args.commit, "target_gold_loaded": False})
        torch.set_grad_enabled(False)
        torch.manual_seed(20261002)
        from lm_eval.api.instance import Instance
        from lm_eval.models.huggingface import HFLM
        from tflt.wrapper import looped_model
        original = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="1", trust_remote_code=True,
                        max_length=adapter.max_length)
        attribute = "phase9_runtime" if args.dataset == "mmlu" else "phase10_runtime"
        results = []
        tensor_result = None
        for index in indices:
            row = rows[index]
            identity = row_identity(row, args.dataset)
            choices = choices_for_row(row, args.dataset)
            def score(runtime, arm="Loop", base=False):
                setattr(adapter, attribute, runtime)
                manager = nullcontext() if arm == "Native" else looped_model(model, make_loop_config(cell, runtime))
                with manager:
                    if base:
                        requests = [Instance(request_type="loglikelihood", doc={},
                            arguments=(row["prompt"], " "+choice), idx=i) for i, choice in enumerate(choices)]
                        return original.loglikelihood(requests, disable_tqdm=True)
                    return adapter.score_choices(row["prompt"], identity, choices)
            entry = {"index": index, "identity": identity, "comparisons": [], "scores": {},
                     "new_runtime_per_question_and_configuration": True}
            for arm in ("Native", "Loop"):
                reference, actual = score(None, arm, True), score(None, arm)
                entry["scores"][arm] = actual
                entry["comparisons"].append({"check": arm+"-adapter-vs-original-HFLM", **score_comparison(reference, actual)})
            if tensor_result is None:
                capture = CaptureResidual()
                observed = score(capture)
                entry["comparisons"].append({"check": "raw-capture-vs-Loop", **score_comparison(observed, entry["scores"]["Loop"])})
                tensor_result = residual_checks(torch, cell, capture)
            for policy in ("fixed_t0", "lag1", "current_t"):
                runtime = CurrentOracle(torch, cell) if policy == "current_t" else LifecycleRuntime(
                    {**cell, "direction_policy": policy, "strength": 0.5})
                entry["scores"][policy] = score(runtime)
                actual_runtime = runtime.runtime
                context_count = validate_calls(actual_runtime, cell["k"])
                require(actual_runtime._active is None, "question left an active context")
                entry[policy] = {"model_context_count": context_count, "calls": list(actual_runtime.calls),
                                "context_summaries": list(actual_runtime.context_summaries)}
                if policy == "current_t":
                    entry[policy]["raw_fit_checks"] = runtime.checks
                    entry[policy]["candidate_start_empty_and_release_count"] = runtime.contexts
                else:
                    entry[policy]["candidate_boundaries"] = runtime.boundaries
            if cell["k"] == 2:
                entry["comparisons"].append({"check": "K2-fixed-vs-lag1-scores", **score_comparison(
                    entry["scores"]["fixed_t0"], entry["scores"]["lag1"])})
            zero = make_runtime({**cell, "direction_policy": "current_t", "strength": 0.0})
            entry["comparisons"].append({"check": "lambda0-current-vs-Loop-scores", **score_comparison(
                score(zero), entry["scores"]["Loop"])})
            validate_calls(zero, cell["k"])
            results.append(entry)
        torch.cuda.synchronize()
        exact = (tensor_result["exact"] and all(c["exact"] for r in results for c in r["comparisons"])
                 and all(c["exact"] for r in results for c in r["current_t"]["raw_fit_checks"]))
        summary = {"schema": "loopscope.phase10.gpu_check.v1", "status": "GPU_CHECK_COMPLETE" if exact else "GPU_CHECK_FAILED",
                   "target_gold_loaded": False, "metadata": command, "tensor_checks": tensor_result,
                   "score_and_runtime_checks": results, "elapsed_seconds": time.monotonic()-started}
        write_json_once(args.run_root/"summary.json", summary)
        print(json.dumps({"status": summary["status"], "count": len(indices), "exact": exact}), flush=True)
        return 0 if exact else 1
    except Exception as exc:
        write_json_once(args.run_root/"failure.json", {"status": "GPU_CHECK_EXCEPTION", "exception": type(exc).__name__,
                        "message": str(exc), "elapsed_seconds": time.monotonic()-started, "target_gold_loaded": False})
        raise


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
