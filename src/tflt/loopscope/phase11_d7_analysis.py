"""Frozen, aggregate-only D7 MMLU-Pro statistics. No data access on import."""
from collections import Counter
import math

from .phase8_accuracy_stats import holm_adjust, paired_contrast
from .phase11_mmlupro_s2 import PARSER_VERSION

SEED = 20261008
BOOTSTRAPS = 10000
RUNTIME_SEED = 20261016
CELLS = ("native", "current_t_k2_l05", "fixed_t0_k3_l09", "loop_k2", "loop_k3")
FIRST_BATCH = CELLS[:3]
METHODS = (("current_t_k2_l05", "loop_k2"), ("fixed_t0_k3_l09", "loop_k3"))


def analyze(data, gold, categories, *, replicates=BOOTSTRAPS):
    """Analyze fully aligned predictions, preserving every item in the denominator."""
    import numpy as np
    n = len(gold)
    if set(data) != set(CELLS) or not n or len(categories) != n:
        raise ValueError("D7 requires exactly five cells and aligned categories/gold")
    counts = [p["option_count"] for p in data["native"]["auxiliary"]]
    if len(counts) != n or any(type(g) is not int or not 0 <= g < m for g, m in zip(gold, counts)):
        raise ValueError("D7 gold outside actual options")
    correctness, cells, category_rows = {}, [], []
    for cid in CELLS:
        value = data[cid]
        raw, predictions, auxiliary = (value[k] for k in ("raw", "predictions", "auxiliary"))
        if not len(raw) == len(predictions) == len(auxiliary) == n:
            raise ValueError("D7 full cell denominator differs")
        if [p["option_count"] for p in auxiliary] != counts:
            raise ValueError("D7 cell option counts differ")
        right = [int(p is not None and p == g) for p, g in zip(predictions, gold)]
        correctness[cid] = right
        per_category = []
        for category in sorted(set(categories)):
            values = [r for r, c in zip(right, categories) if c == category]
            row = {"cell": cid, "category": category, "n": len(values), "correct": sum(values),
                   "accuracy": sum(values) / len(values)}
            per_category.append(row)
            category_rows.append(row)
        lengths = [r["generated_token_count"] for r in raw]
        statuses = Counter(p["extraction_status"] for p in auxiliary)
        ends = Counter(r["end_reason"] for r in raw)
        cells.append({"cell": cid, "n": n, "correct": sum(right), "accuracy": sum(right) / n,
            "accuracy_percent": 100 * sum(right) / n, "runtime_seed": RUNTIME_SEED,
            "category_count": len(per_category),
            "category_macro_accuracy": math.fsum(r["accuracy"] for r in per_category) / len(per_category),
            "generated_tokens_mean": sum(lengths) / n,
            "generated_tokens_median": float(np.median(lengths)),
            "generated_tokens_p90": float(np.quantile(lengths, .9)),
            "eos_count": ends["eos"], "length_limit_count": ends["length_limit"],
            "S2_success_count": statuses["ok"], "S2_no_match_count": statuses["no_match"],
            "S2_conflict_count": statuses["conflict"], "S2_out_of_range_count": statuses["out_of_range"],
            "S2_status_counts": {s: statuses[s] for s in ("ok", "no_match", "conflict", "out_of_range")},
            "S2_consistent_multiple_count": sum(p["answer_mode"] == "consistent_multiple" for p in auxiliary),
            **value.get("resource_summary", {})})
    contrasts = []
    specs = [(method, reference, "PRIMARY_4") for method, loop in METHODS for reference in ("native", loop)]
    specs += [(loop, "native", "BASELINE_2") for _, loop in METHODS]
    for treatment, reference, family in specs:
        stats = paired_contrast(correctness[reference], correctness[treatment], categories,
                                bootstrap_replicates=replicates, seed=SEED)
        contrasts.append({"family": family, "reference": reference, "treatment": treatment,
                          **stats, "p_value": stats["mcnemar_exact_p"]})
    for family in ("PRIMARY_4", "BASELINE_2"):
        selected = [r for r in contrasts if r["family"] == family]
        for row, adjusted in zip(selected, holm_adjust([r["p_value"] for r in selected])):
            row.update(holm_family_size=len(selected), holm_adjusted_p=adjusted,
                       holm_reject_alpha_0_05=adjusted <= .05)
    for row in cells:
        cid = row["cell"]
        row["delta_native_pp"] = 100 * (sum(correctness[cid]) - sum(correctness["native"])) / n
        loop = next((loop for method, loop in METHODS if cid in (method, loop)), None)
        row["delta_loop_pp"] = None if loop is None else 100 * (sum(correctness[cid]) - sum(correctness[loop])) / n
    return {"schema": "loopscope.phase11.d7.analysis.v1", "status": "D7_COMPLETE_TRANSFER_EVALUATION",
        "primary_parser": PARSER_VERSION, "primary_metric": "full_population_micro_accuracy",
        "cells": cells, "categories": category_rows, "contrasts": contrasts,
        "primary_contrasts": [r for r in contrasts if r["family"] == "PRIMARY_4"],
        "baseline_contrasts": [r for r in contrasts if r["family"] == "BASELINE_2"],
        "families": {"PRIMARY_4": 4, "BASELINE_2": 2},
        "bootstrap_replicates": replicates, "bootstrap_seed": SEED,
        "interpretation": {
            "selection": "fixed configurations selected post hoc on GPQA; transfer evaluation, not MMLU-Pro optimality",
            "history": "project has previously used MMLU-Pro; not a wholly unseen independent benchmark",
            "ci": "nominal 95 percent category-stratified paired bootstrap; fixed category sizes and original micro weights",
            "categories": "category accuracy and unweighted category macro are descriptive only; no category significance selection",
            "denominator": "all 12032 questions per cell; extraction failures count as wrong",
            "specificity": "without Matched-norm controls this does not establish direction specificity"}}
