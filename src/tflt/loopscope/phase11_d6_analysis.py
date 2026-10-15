"""Frozen D6 paired GPQA statistics. No files or gold are read on import."""
from __future__ import annotations

from collections import Counter

from .phase8_accuracy_stats import holm_adjust, paired_contrast

SEED = 20261008
BOOTSTRAPS = 10000
SIGN_FLIPS = 100000
CELLS = (
    "cot_native", "cot_loop_k2", "cot_loop_k3", "cot_current_k2_full", "cot_fixed_k3_full",
    "direct_native", "direct_loop_k2", "direct_loop_k3", "direct_current_k2_full", "direct_fixed_k3_full",
    "cot_current_k2_prefill", "cot_fixed_k3_prefill",
)
METHODS = ((2, "current"), (3, "fixed"))


def paired_mean_bootstrap(differences, *, scale=100., unit="pp", replicates=BOOTSTRAPS, seed=SEED):
    """Empirical paired item bootstrap, including non-binary interaction/length deltas."""
    import numpy as np
    if not differences or replicates <= 0:
        raise ValueError("paired bootstrap needs samples and positive draws")
    values, counts = np.unique(np.asarray(differences, dtype=np.float64), return_counts=True)
    if not np.isfinite(values).all():
        raise ValueError("paired differences must be finite")
    draws = np.random.default_rng(seed).multinomial(len(differences), counts / len(differences), size=replicates)
    means = draws @ values * (scale / len(differences))
    low, high = np.quantile(means, [.025, .975])
    return {"method": "paired_item_bootstrap", "sampling": "empirical_multinomial_difference_counts",
            "replicates": replicates, "seed": seed, "rng": "numpy.default_rng.PCG64",
            "numpy_version": np.__version__, "ci_level": .95, "ci_kind": "nominal_percentile_linear",
            "ci_low_" + unit: float(low), "ci_high_" + unit: float(high)}


def sign_flip(differences, *, draws=SIGN_FLIPS, seed=SEED):
    """Two-sided paired sign randomization; integer sums avoid tail-rounding errors."""
    import numpy as np
    values = np.asarray(differences)
    if values.ndim != 1 or not len(values) or draws <= 0 or not np.isin(values, [-2, -1, 0, 1, 2]).all():
        raise ValueError("interaction differences must be paired integer values -2 through 2")
    values = values.astype(np.int64)
    observed = abs(int(values.sum()))
    rng, tail = np.random.default_rng(seed), 0
    for offset in range(0, draws, 1000):
        signs = 2 * rng.integers(0, 2, size=(min(1000, draws - offset), len(values)), dtype=np.int8) - 1
        tail += int((np.abs(signs @ values) >= observed).sum())
    return {"method": "two_sided_paired_sign_flip", "draws": draws, "seed": seed,
            "rng": "numpy.default_rng.PCG64", "tail_count": tail,
            "observed_absolute_sum": observed, "p_value": (1 + tail) / (draws + 1),
            "assumption": "exploratory exchangeability under independent item-level sign flips"}


def _adjust(rows):
    for row, adjusted in zip(rows, holm_adjust([r["p_value"] for r in rows])):
        row.update(holm_family_size=len(rows), holm_adjusted_p=adjusted,
                   holm_reject_alpha_0_05=adjusted <= .05)


def analyze(data, gold, *, replicates=BOOTSTRAPS, sign_flip_draws=SIGN_FLIPS):
    """Consume fully aligned raw/predictions, return aggregate-only D6 results."""
    import numpy as np
    if set(data) != set(CELLS) or not gold or any(type(g) is not int or not 0 <= g < 4 for g in gold):
        raise ValueError("D6 requires exactly 12 cells and valid aligned gold")
    n = len(gold)
    correctness, cells = {}, []
    for cid in CELLS:
        value = data[cid]
        raw, predictions, auxiliary = (value[k] for k in ("raw", "predictions", "auxiliary"))
        if not len(raw) == len(predictions) == len(auxiliary) == n:
            raise ValueError("D6 cell denominator differs")
        right = [int(p is not None and p == g) for p, g in zip(predictions, gold)]
        correctness[cid] = right
        lengths = [row["generated_token_count"] for row in raw]
        eos_lengths = [row["generated_token_count"] for row in raw if row["end_reason"] == "eos"]
        modes = Counter(p["answer_mode"] for p in auxiliary)
        ends = Counter(row["end_reason"] for row in raw)
        cells.append({"cell": cid, "n": n, "correct": sum(right), "accuracy": sum(right) / n,
            "accuracy_percent": 100 * sum(right) / n, "runtime_seed": 20261015,
            "generated_tokens_mean": sum(lengths) / n, "generated_tokens_median": float(np.median(lengths)),
            "generated_tokens_p90": float(np.quantile(lengths, .9)),
            "eos_count": ends["eos"], "length_limit_count": ends["length_limit"],
            "eos_only_tokens_mean": sum(eos_lengths) / len(eos_lengths) if eos_lengths else None,
            "S2_success_count": sum(p["prediction_index"] is not None for p in auxiliary),
            "S2_no_match_count": modes["no_match"], "S2_conflict_count": modes["conflicting_multiple"],
            "S2_mode_counts": dict(modes), "end_reason_counts": dict(ends)})
    primary, auxiliary_rows, length_rows, descriptive = [], [], [], []
    for k, method in METHODS:
        cot, direct = f"cot_{method}_k{k}_full", f"direct_{method}_k{k}_full"
        differences = [cm - cl - dm + dl for cm, cl, dm, dl in zip(
            correctness[cot], correctness[f"cot_loop_k{k}"], correctness[direct], correctness[f"direct_loop_k{k}"])]
        randomization = sign_flip(differences, draws=sign_flip_draws)
        primary.append({"contrast": f"prompt_interaction_k{k}", "family": "PRIMARY_4", "k": k,
            "definition": "(CoT method_full - CoT Loop) - (direct method_full - direct Loop)",
            "n": n, "delta_pp": 100 * sum(differences) / n,
            "bootstrap": paired_mean_bootstrap(differences, replicates=replicates),
            "p_value": randomization["p_value"], "randomization": randomization})
    for k, method in METHODS:
        reference, treatment = f"cot_{method}_k{k}_prefill", f"cot_{method}_k{k}_full"
        stats = paired_contrast(correctness[reference], correctness[treatment], ["items"] * n,
                                bootstrap_replicates=replicates, seed=SEED)
        primary.append({"contrast": f"full_minus_prefill_k{k}", "family": "PRIMARY_4", "k": k,
                        "reference": reference, "treatment": treatment, **stats, "p_value": stats["mcnemar_exact_p"]})
    for prompt in ("cot", "direct"):
        for k, method in METHODS:
            treatment = f"{prompt}_{method}_k{k}_full"
            for reference in (f"{prompt}_native", f"{prompt}_loop_k{k}"):
                stats = paired_contrast(correctness[reference], correctness[treatment], ["items"] * n,
                                        bootstrap_replicates=replicates, seed=SEED)
                auxiliary_rows.append({"family": "AUXILIARY_10", "reference": reference,
                    "treatment": treatment, **stats, "p_value": stats["mcnemar_exact_p"]})
    for k, method in METHODS:
        reference, treatment = f"cot_loop_k{k}", f"cot_{method}_k{k}_prefill"
        stats = paired_contrast(correctness[reference], correctness[treatment], ["items"] * n,
                                bootstrap_replicates=replicates, seed=SEED)
        auxiliary_rows.append({"family": "AUXILIARY_10", "reference": reference,
            "treatment": treatment, **stats, "p_value": stats["mcnemar_exact_p"]})
    _adjust(primary)
    _adjust(auxiliary_rows)
    for prompt in ("cot", "direct"):
        for k, method in METHODS:
            for scope in (("full", "prefill") if prompt == "cot" else ("full",)):
                reference, treatment = f"{prompt}_loop_k{k}", f"{prompt}_{method}_k{k}_{scope}"
                diffs = [t["generated_token_count"] - r["generated_token_count"]
                         for r, t in zip(data[reference]["raw"], data[treatment]["raw"])]
                length_rows.append({"reference": reference, "treatment": treatment, "n": n,
                    "mean_delta_tokens": sum(diffs) / n,
                    "bootstrap": paired_mean_bootstrap(diffs, scale=1., unit="tokens", replicates=replicates)})
    for suffix in ("native", "loop_k2", "loop_k3", "current_k2_full", "fixed_k3_full"):
        a, b = f"cot_{suffix}", f"direct_{suffix}"
        descriptive.append({"reference": a, "treatment": b, "n": n,
            "delta_pp": 100 * (sum(correctness[b]) - sum(correctness[a])) / n})
    for row in cells:
        prompt = row["cell"].split("_")[0]
        right = correctness[row["cell"]]
        row["delta_native_pp"] = 100 * (sum(right) - sum(correctness[f"{prompt}_native"])) / n
        k = 2 if "k2" in row["cell"] else 3 if "k3" in row["cell"] else None
        row["delta_loop_pp"] = None if k is None else 100 * (sum(right) - sum(correctness[f"{prompt}_loop_k{k}"])) / n
    return {"schema": "loopscope.phase11.d6.analysis.v1", "status": "D6_COMPLETE_EXPLORATORY",
        "primary_parser": "unchanged phase11_s2.extract_s2", "cells": cells, "primary_contrasts": primary,
        "auxiliary_contrasts": auxiliary_rows, "length_contrasts": length_rows,
        "cross_prompt_descriptive": descriptive, "families": {"PRIMARY_4": 4, "AUXILIARY_10": 10},
        "interpretation": {
            "selection": "same GPQA questions used to select these configurations; exploratory mechanism test, not independent confirmation",
            "interaction": "sign-flip exchangeability is an exploratory assumption; separate significance in one prompt is not interaction evidence",
            "ci": "10000 paired item bootstrap nominal 95 percent intervals; Holm decisions use separate frozen 4 and 10 families",
            "prefill_only": "ordinary Loop remains full during decode; first output-token logits can still reflect damped prefill",
            "direct": "direct-answer instruction does not establish absence of internal reasoning; noncompliance retained",
            "denominator": "all 448 items per cell; no format-success or EOS-only replacement of primary denominator",
            "eos_only_length": "conditional descriptive mean, not an overall efficiency result",
            "scope": "negative effects are valid outcomes; no pooled repetitions or independent-dataset claim"}}
