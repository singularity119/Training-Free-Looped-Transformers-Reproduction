#!/usr/bin/env python3
"""Run the one frozen Phase 9 paired analysis after both raw-score closures.

This script performs fresh gold-free new/history verification first, writes a
pre-outcome receipt, and only then loads the MMLU test labels.  It computes
the contract's K2/K3/K4 Holm families, the exploratory Shared-arm family,
and descriptive Native comparisons from raw four-choice scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from tflt.loopscope.phase8_accuracy_stats import (
    BOOTSTRAP_REPLICATES,
    cell_accuracy,
    exact_mcnemar_p,
    holm_adjust,
    paired_contrast,
    score_predictions,
)
from tflt.loopscope.phase9_accuracy import EXPECTED_TEST_COUNT, load_config, logical_panel

from verify_phase9_history import verify as verify_history
from verify_phase9_scores import verify as verify_new


BOOTSTRAP_SEED = 20260922


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_raw_cells(
    new_closure: Mapping[str, Any],
    history_receipt: Mapping[str, Any],
    pool_rows: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Read only raw scores from already verified roots and align by identity."""

    by_cell: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    roots_by_cell: Dict[str, set[str]] = defaultdict(set)

    def consume(root: Path, target_cell_id: str) -> None:
        metadata = read_json(root / "command_args.json")
        cell = metadata.get("cell")
        require(isinstance(cell, Mapping), f"score root has no cell metadata: {root}")
        actual_id = cell.get("cell_id")
        if target_cell_id.startswith(("q4-", "q17-")) and target_cell_id.endswith(("-online-t0", "-matched-norm")):
            require(actual_id == target_cell_id, "new score root cell identity differs from target panel")
        roots_by_cell[target_cell_id].add(str(root))
        with (root / "scores.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get("identity")
                require(identity not in by_cell[target_cell_id],
                        f"duplicate raw score identity for {target_cell_id}")
                scores = row.get("scores")
                require(isinstance(scores, list) and len(scores) == 4 and
                        all(type(value) in (int, float) and math.isfinite(value) for value in scores),
                        f"nonfinite raw score in {root}")
                by_cell[target_cell_id][identity] = [float(value) for value in scores]

    for root_text in new_closure["cell_roots"]:
        root = Path(root_text)
        metadata = read_json(root / "command_args.json")
        target_id = metadata["cell"]["cell_id"]
        consume(root, target_id)
    for entry in history_receipt["cells"]:
        target_id = entry["cell_id"]
        for root_record in entry["roots"]:
            consume(Path(root_record["root"]), target_id)

    aligned = []
    identities = [row["identity"] for row in pool_rows]
    require(len(identities) == EXPECTED_TEST_COUNT and len(set(identities)) == EXPECTED_TEST_COUNT,
            "canonical pool identities are not unique")
    for cell in panel:
        scores = by_cell.get(cell["cell_id"], {})
        require(len(scores) == EXPECTED_TEST_COUNT,
                f"raw score closure has the wrong sample count: {cell['cell_id']}")
        require(len(roots_by_cell[cell["cell_id"]]) > 0,
                f"raw score closure has no roots: {cell['cell_id']}")
        aligned.append(dict(cell, scores=[scores[identity] for identity in identities]))
    return aligned


def add_contrast(
    rows: List[Dict[str, Any]],
    correctness: Mapping[str, Sequence[int]],
    by_id: Mapping[str, Any],
    reference_id: str,
    treatment_id: str,
    family: str,
    comparison: str,
) -> None:
    reference = by_id[reference_id]
    treatment = by_id[treatment_id]
    value = paired_contrast(
        correctness[reference_id], correctness[treatment_id], by_id["__subjects__"],
        bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
    )
    rows.append({
        "reference": reference_id,
        "treatment": treatment_id,
        "family": family,
        "comparison": comparison,
        "model": treatment["model"],
        "window": treatment["window"],
        "k": treatment["k"],
        **value,
        "holm_adjusted_p": None,
        "holm_reject_alpha_0_05": None,
    })


def build_analysis(cells: Sequence[Mapping[str, Any]], gold: Sequence[int], subjects: Sequence[str]) -> Dict[str, Any]:
    accuracy = []
    correctness: Dict[str, List[int]] = {}
    by_key: Dict[tuple[Any, Any, Any, Any], str] = {}
    by_id: Dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        key = (cell["model"], tuple(cell["window"]) if cell["window"] else None,
               cell["k"], cell["arm"])
        require(key not in by_key, "duplicate scientific Phase 9 panel configuration")
        by_key[key] = cell["cell_id"]
        by_id[cell["cell_id"]] = cell
        stats = cell_accuracy(cell["scores"], gold, subjects)
        accuracy.append({
            "cell_id": cell["cell_id"],
            "model": cell["model"],
            "window": cell["window"],
            "k": cell["k"],
            "arm": cell["arm"],
            **stats,
        })
        correctness[cell["cell_id"]] = [
            int(prediction == answer)
            for prediction, answer in zip(score_predictions(cell["scores"]), gold)
        ]
    by_id["__subjects__"] = list(subjects)
    contrasts: List[Dict[str, Any]] = []

    def id_for(model: str, window: Sequence[int] | None, k: int | None, arm: str) -> str:
        key = (model, tuple(window) if window else None, k, arm)
        require(key in by_key, f"missing Phase 9 panel arm: {key}")
        return by_key[key]

    window_specs = []
    seen_window_specs = set()
    for cell in cells:
        if cell["window"] is None:
            continue
        key = (cell["model"], tuple(cell["window"]))
        if key not in seen_window_specs:
            seen_window_specs.add(key)
            window_specs.append((cell["model"], cell["window"]))
    for model, window in window_specs:
        for k in (2, 3, 4):
            online = id_for(model, window, k, "Online-t0")
            loop = id_for(model, window, k, "Loop")
            matched = id_for(model, window, k, "Matched-norm")
            family = {2: "K2_PRIMARY", 3: "K3_SECONDARY", 4: "K4_SECONDARY"}[k]
            add_contrast(contrasts, correctness, by_id, loop, online, family, "Online-t0-Loop")
            add_contrast(contrasts, correctness, by_id, matched, online, family, "Online-t0-Matched-norm")
            shared_t0 = id_for(model, window, k, "Shared-t0")
            shared_t1 = id_for(model, window, k, "Shared-t1")
            add_contrast(contrasts, correctness, by_id, shared_t0, online,
                         "SHARED_EXPLORATORY", "Online-t0-Shared-t0")
            add_contrast(contrasts, correctness, by_id, shared_t1, online,
                         "SHARED_EXPLORATORY", "Online-t0-Shared-t1")

    # Native is a descriptive context only; it receives no multiplicity decision.
    for cell in cells:
        if cell["window"] is None:
            continue
        native = id_for(cell["model"], None, None, "Native")
        add_contrast(contrasts, correctness, by_id, native, cell["cell_id"],
                     "NATIVE_DESCRIPTIVE", f"{cell['arm']}-Native")

    for family, expected_count in (
        ("K2_PRIMARY", 6), ("K3_SECONDARY", 6), ("K4_SECONDARY", 6),
        ("SHARED_EXPLORATORY", 18),
    ):
        selected = [row for row in contrasts if row["family"] == family]
        require(len(selected) == expected_count, f"{family} has the wrong frozen contrast count")
        adjusted = holm_adjust([row["mcnemar_exact_p"] for row in selected])
        for row, value in zip(selected, adjusted):
            row["holm_adjusted_p"] = value
            row["holm_reject_alpha_0_05"] = value <= 0.05
            row["holm_family_size"] = expected_count

    return {
        "schema": "loopscope.phase9.accuracy_analysis.v1",
        "status": "ANALYZED",
        "target_gold_loaded": True,
        "panel_cell_count": len(accuracy),
        "contrasts": contrasts,
        "cells": accuracy,
        "bootstrap": {
            "method": "fixed_subject_stratified_paired_bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci_level": 0.95,
            "ci_kind": "nominal_percentile_linear",
        },
        "families": {
            "K2_PRIMARY": "Online-t0 minus Loop and Matched-norm across the three frozen windows",
            "K3_SECONDARY": "Online-t0 minus Loop and Matched-norm across the three frozen windows",
            "K4_SECONDARY": "Online-t0 minus Loop and Matched-norm across the three frozen windows",
            "SHARED_EXPLORATORY": "Online-t0 minus Shared-t0 and Shared-t1 across all nine window/K cells",
            "NATIVE_DESCRIPTIVE": "Native paired context, descriptive only",
        },
        "interpretation": {
            "non_significant": "not confirmatory evidence of equivalence or no effect",
            "native": "descriptive context; no Holm decision",
        },
    }


def independent_count_check(result: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], gold: Sequence[int], subjects: Sequence[str]) -> Dict[str, Any]:
    by_cell = {cell["cell_id"]: cell for cell in cells}
    reported = {cell["cell_id"]: cell for cell in result["cells"]}
    correctness = {}
    for cell in cells:
        stats = cell_accuracy(cell["scores"], gold, subjects)
        saved = reported[cell["cell_id"]]
        require(stats["correct"] == saved["correct"], "fresh accuracy recomputation failed")
        require(stats["n"] == saved["n"] and stats["micro_accuracy"] == saved["micro_accuracy"],
                "fresh micro accuracy recomputation failed")
        require(math.isclose(stats["subject_macro_accuracy"], saved["subject_macro_accuracy"], abs_tol=1e-14),
                "fresh subject macro recomputation failed")
        correctness[cell["cell_id"]] = [
            int(prediction == answer)
            for prediction, answer in zip(score_predictions(cell["scores"]), gold)
        ]
    fresh_p = {}
    for row in result["contrasts"]:
        reference = correctness[row["reference"]]
        treatment = correctness[row["treatment"]]
        gains = sum(int(t and not r) for r, t in zip(reference, treatment))
        losses = sum(int(r and not t) for r, t in zip(reference, treatment))
        require((gains, losses) == (row["wrong_to_right"], row["right_to_wrong"]),
                "fresh discordant-pair recomputation failed")
        pvalue = exact_mcnemar_p(gains, losses)
        require(math.isclose(pvalue, row["mcnemar_exact_p"], rel_tol=1e-12, abs_tol=1e-14),
                "fresh exact McNemar recomputation failed")
        fresh_p[(row["reference"], row["treatment"])] = pvalue
    for family in ("K2_PRIMARY", "K3_SECONDARY", "K4_SECONDARY", "SHARED_EXPLORATORY"):
        selected = [row for row in result["contrasts"] if row["family"] == family]
        adjusted = holm_adjust([row["mcnemar_exact_p"] for row in selected])
        for row, value in zip(selected, adjusted):
            require(math.isclose(value, row["holm_adjusted_p"], rel_tol=1e-12, abs_tol=1e-14),
                    "fresh Holm recomputation failed")
            require((value <= 0.05) == row["holm_reject_alpha_0_05"],
                    "fresh Holm decision differs")
    return {
        "status": "VERIFIED",
        "cell_count": len(cells),
        "contrast_count": len(result["contrasts"]),
        "n_per_cell": len(gold),
        "checks": [
            "raw-score argmax correct/N",
            "wrong_to_right/right_to_wrong",
            "exact two-sided McNemar",
            "subject macro accuracy",
            "independent Holm recomputation",
        ],
    }


def write_tables(output_dir: Path, result: Mapping[str, Any]) -> None:
    cell_fields = ["cell_id", "model", "window", "k", "arm", "correct", "n",
                   "micro_accuracy", "subject_macro_accuracy"]
    contrast_fields = ["reference", "treatment", "family", "comparison", "model", "window", "k", "n",
                       "reference_correct", "treatment_correct", "wrong_to_right", "right_to_wrong",
                       "delta_pp", "subject_macro_delta_pp", "ci_low_pp", "ci_high_pp",
                       "mcnemar_exact_p", "holm_adjusted_p", "holm_reject_alpha_0_05"]
    with (output_dir / "cells.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["cells"])
    with (output_dir / "contrasts.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=contrast_fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["contrasts"]:
            writer.writerow({**row, **row["bootstrap"]})


def load_frozen_gold(pool_rows: Sequence[Mapping[str, Any]], cache_dir: str):
    """Delegate label loading to the existing frozen-revision loader."""

    from analyze_phase8_accuracy import load_frozen_gold as load_phase8_gold

    return load_phase8_gold(pool_rows, cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-closure", type=Path, required=True)
    parser.add_argument("--history-verification", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--canary-indices", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    new_closure = read_json(args.new_closure)
    history_receipt = read_json(args.history_verification)
    require(new_closure.get("status") == "FULL_NEW_SCORE_PANEL_CLOSED" and
            new_closure.get("target_gold_loaded") is False,
            "gold access requires a saved full new-score closure")
    require(history_receipt.get("status") == "HISTORICAL_29_CELLS_VERIFIED" and
            history_receipt.get("target_gold_loaded") is False,
            "gold access requires a saved historical raw-score verification")

    # Fresh checks are deliberately completed before the output directory is
    # created and before any dataset label loader is imported.
    fresh_new = verify_new(
        [Path(root) for root in new_closure["cell_roots"]],
        Path(new_closure["manifest"]), args.pool, new_closure["scope"],
        "combined", args.canary_indices, full_panel=True,
    )
    fresh_history = verify_history(args.mapping, args.pool)
    require(fresh_new["status"] == "FULL_NEW_SCORE_PANEL_CLOSED" and
            fresh_history["status"] == "HISTORICAL_29_CELLS_VERIFIED",
            "fresh pre-outcome closure did not verify")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_once(args.output_dir / "pre_outcome_verification.json", {
        "schema": "loopscope.phase9.pre_outcome_verification.v1",
        "target_gold_loaded": False,
        "verified_at": now(),
        "new_score": fresh_new,
        "historical": fresh_history,
    })

    pool = read_json(args.pool)
    pool_rows = pool["rows"]
    config = load_config()
    panel = logical_panel(config)
    cells = load_raw_cells(new_closure, history_receipt, pool_rows, panel)
    subjects = [row["subject"] for row in pool_rows]
    gold_started = now()
    gold, sources = load_frozen_gold(pool_rows, args.cache_dir)
    result = build_analysis(cells, gold, subjects)
    result.update({
        "closure": str(args.new_closure),
        "history_verification": str(args.history_verification),
        "mapping": str(args.mapping),
        "pool": str(args.pool),
        "dataset": {
            "repo": config["dataset_repo"],
            "revision": config["dataset_revision"],
            "split": "test",
        },
        "gold_load_started": gold_started,
        "completed_at": now(),
        "gold_source_evidence": sources,
        "fresh_count_verification": independent_count_check(result, cells, gold, subjects),
    })
    write_json_once(args.output_dir / "analysis.json", result)
    write_tables(args.output_dir, result)
    print(json.dumps({
        "status": result["status"],
        "cells": result["panel_cell_count"],
        "contrasts": len(result["contrasts"]),
        "target_gold_loaded": result["target_gold_loaded"],
        "output_dir": str(args.output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
