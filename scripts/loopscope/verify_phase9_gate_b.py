#!/usr/bin/env python3
"""Verify Gate B debug/diagnostic artifacts without labels or test outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


FORBIDDEN_ROW_FIELDS = {
    "answer", "target", "gold", "label", "correct", "correctness",
    "accuracy", "gain", "baseline", "outcome",
}


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def cells_for(config: Mapping[str, Any], model_index: int, indexes: Sequence[int] | None):
    spec = config["models"][model_index]
    cells = [
        (list(window["layers"]), int(k))
        for window in spec["windows"]
        for k in config["K"]
    ]
    selected = list(range(len(cells))) if indexes is None else list(indexes)
    if any(index < 0 or index >= len(cells) for index in selected):
        raise ValueError("cell index outside frozen model cells")
    return [(selected_index, *cells[selected_index]) for selected_index in selected]


def assert_no_labels(pool: Mapping[str, Any], expected_count: int) -> list[str]:
    if pool.get("status") not in ("SMOKE_ONLY", "FORMAL_CALIBRATION"):
        raise ValueError("unexpected Gate B pool status")
    rows = list(pool.get("rows", ()))
    if len(rows) != expected_count:
        raise ValueError("pool count differs from requested Gate B scope")
    identities = [row.get("identity") for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("pool contains duplicate identities")
    for row in rows:
        if FORBIDDEN_ROW_FIELDS.intersection(row):
            raise ValueError("pool contains a forbidden target/outcome field")
    return identities


def assert_flags(value: Mapping[str, Any]) -> None:
    if value.get("target_gold_loaded") is not False:
        raise ValueError("artifact reports target/gold access")
    if value.get("test_split_loaded") is True:
        raise ValueError("artifact reports test split access")


def assert_context_trajectory(contexts: Sequence[Mapping[str, Any]], k: int, expected_count: int) -> None:
    if len(contexts) != expected_count:
        raise ValueError("Phase 9 candidate-context count differs from the frozen scorer path")
    for context in contexts:
        if context.get("call_count") != k or context.get("timesteps") != list(range(k)):
            raise ValueError("Phase 9 candidate context does not contain exactly K callbacks")


def verify_debug_cell(value: Mapping[str, Any], identities: Sequence[str], k: int) -> Dict[str, Any]:
    if value.get("status") != "DEBUG_CELL_COMPLETE" or value.get("k") != k:
        raise ValueError("debug cell status or K differs from the requested cell")
    assert_flags(value)
    if value.get("fit_identity_count") != 4 or value.get("verify_identity_count") != 4:
        raise ValueError("debug cell identity counts are incomplete")
    deltas = value.get("zero_strength_max_abs_delta", {})
    if len(deltas) != 4 or any(float(delta) != 0.0 for delta in deltas.values()):
        raise ValueError("zero-strength path is not exactly equivalent to the Loop path")
    order = value.get("candidate_order_checks", ())
    if len(order) != 1 or any(float(item.get("score_max_abs_order_delta", 1.0)) != 0.0 for item in order):
        raise ValueError("candidate order changed score or mask semantics")
    arms = value.get("arms", {})
    for mode in ("Online-t0", "Matched-norm"):
        report = arms.get(mode)
        if not report or report.get("changed_score_row_count", 0) < 1:
            raise ValueError("debug did not observe a real intervention for %s" % mode)
        for row in report.get("rows", ()):
            for call in row.get("calls", ()):
                if call.get("t", -1) == 0 and call.get("applied"):
                    raise ValueError("t0 intervention was applied")
                if call.get("t", -1) >= 1 and not call.get("applied"):
                    raise ValueError("t>=1 intervention was missing")
                if float(call.get("nonanswer_max_abs_change", 1.0)) != 0.0:
                    raise ValueError("non-answer residual position changed")
    diagnostic = value.get("diagnostic", {})
    records = diagnostic.get("records", ())
    if len(records) != 4:
        raise ValueError("debug diagnostic fit pool is incomplete")
    assert_context_trajectory(diagnostic.get("runtime", {}).get("context_summaries", ()), k, 4)
    for mode in ("Online-t0", "Matched-norm"):
        for row in arms[mode].get("rows", ()):
            assert_context_trajectory(row.get("contexts", ()), k, 1)
            if len(row.get("calls", ())) != k:
                raise ValueError("debug arm contains duplicated or missing callback records")
    if order[0].get("context_count") != 2:
        raise ValueError("candidate-order check did not record one context per ordered score call")
    return {
        "status": "DEBUG_CELL_VALID",
        "identity_count": 4,
        "arm_modes": ["Online-t0", "Matched-norm"],
        "zero_strength_exact": True,
        "candidate_order_exact": True,
    }


def verify_formal_cell(value: Mapping[str, Any], identities: Sequence[str], k: int) -> Dict[str, Any]:
    if value.get("status") != "DIAGNOSTIC_COMPLETE" or value.get("k") != k:
        raise ValueError("formal diagnostic cell status or K differs from the requested cell")
    assert_flags(value)
    records = value.get("records", ())
    if len(records) != len(identities):
        raise ValueError("formal diagnostic cell identity count is incomplete")
    observed = [record.get("identity") for record in records]
    if observed != list(identities):
        raise ValueError("formal diagnostic identity order differs from the sanitized pool")
    error_values = []
    near_repeated = 0
    for record in records:
        rows = record.get("records", ())
        if len(rows) != k or [row.get("t") for row in rows] != list(range(k)):
            raise ValueError("formal diagnostic trajectory has an incomplete timestep sequence")
        for row in rows:
            for key in ("E0", "A_t", "C0t", "reference", "spectral", "fit_comparison"):
                if key not in row:
                    raise ValueError("formal diagnostic row is missing %s" % key)
            if row["spectral"].get("diagnostic_direction_used_for_intervention") is not False:
                raise ValueError("v_t diagnostic direction was marked as an intervention input")
            comparison = row["fit_comparison"]
            if comparison.get("near_repeated_root"):
                near_repeated += 1
            error = comparison.get("projection_relative_error")
            if error is not None:
                if not math.isfinite(float(error)):
                    raise ValueError("formal diagnostic projection error is nonfinite")
                error_values.append(float(error))
            if row["mutation"].get("nonanswer_max_abs_change") != 0.0:
                raise ValueError("formal no-intervention diagnostic changed a non-answer position")
    normal_errors = [
        error for record in records for row in record["records"]
        for error in [row["fit_comparison"].get("projection_relative_error")]
        if error is not None and not row["fit_comparison"].get("near_repeated_root")
    ]
    if normal_errors and max(float(error) for error in normal_errors) > 1e-3:
        raise ValueError("normal-gap FP32/CPU-FP64 projection error exceeds 1e-3")
    return {
        "status": "FORMAL_DIAGNOSTIC_CELL_VALID",
        "identity_count": len(records),
        "trajectory_row_count": len(records) * k,
        "projection_error_max": max(error_values) if error_values else None,
        "near_repeated_root_row_count": near_repeated,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--scope", choices=("DEBUG_PREFLIGHT", "FORMAL_DIAGNOSTIC"), required=True)
    parser.add_argument("--cell-index", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    config_path = args.config or (repo / "configs/loopscope/phase9_model_windows.json")
    config = read_json(config_path)
    expected_rows = 8 if args.scope == "DEBUG_PREFLIGHT" else 512
    identities = assert_no_labels(read_json(args.pool), expected_rows)
    cells = cells_for(config, args.model_index, args.cell_index)
    summaries = []
    for cell_index, window, k in cells:
        name = "w%d-%d-k%d" % (*window, k)
        cell_root = args.run_root / name
        filename = "debug.json" if args.scope == "DEBUG_PREFLIGHT" else "diagnostics.json"
        value = read_json(cell_root / filename)
        summaries.append(
            verify_debug_cell(value, identities, k) if args.scope == "DEBUG_PREFLIGHT"
            else verify_formal_cell(value, identities, k)
        )
        summaries[-1].update({"cell_index": cell_index, "window": window, "k": k})
    result = {
        "status": "GATE_B_DEBUG_VALID" if args.scope == "DEBUG_PREFLIGHT" else "GATE_B_DIAGNOSTIC_VALID",
        "scope": args.scope,
        "model_index": args.model_index,
        "cell_count": len(summaries),
        "identity_count": len(identities),
        "target_gold_loaded": False,
        "test_split_loaded": False,
        "cells": summaries,
    }
    write_once(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
