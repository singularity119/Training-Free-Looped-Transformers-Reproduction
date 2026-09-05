"""Frozen Phase 8 paired accuracy statistics; no data/model access on import."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260905


def score_predictions(scores: Sequence[Sequence[float]]) -> list[int]:
    """Raw continuation-sum argmax; ties select the first A/B/C/D choice."""
    result = []
    for row in scores:
        if len(row) != 4 or not all(math.isfinite(float(value)) for value in row):
            raise ValueError("each sample requires four finite raw choice scores")
        result.append(max(range(4), key=lambda index: row[index]))
    return result


def cell_accuracy(scores, gold, subjects) -> dict[str, Any]:
    if not len(scores) == len(gold) == len(subjects) or not len(gold):
        raise ValueError("scores, gold and subjects must have matching positive lengths")
    if any(label not in (0, 1, 2, 3) for label in gold):
        raise ValueError("gold must contain original A/B/C/D indices")
    predictions = score_predictions(scores)
    correct = [int(pred == label) for pred, label in zip(predictions, gold)]
    by_subject = defaultdict(list)
    for subject, value in zip(subjects, correct):
        by_subject[subject].append(value)
    subject_accuracy = {subject: {"correct": sum(values), "n": len(values),
                                  "accuracy": sum(values) / len(values)}
                        for subject, values in sorted(by_subject.items())}
    return {"n": len(gold), "correct": sum(correct),
            "micro_accuracy": sum(correct) / len(gold),
            "subject_macro_accuracy": math.fsum(x["accuracy"] for x in subject_accuracy.values()) / len(by_subject),
            "subject_accuracy": subject_accuracy}


def exact_mcnemar_p(wrong_to_right: int, right_to_wrong: int) -> float:
    """Two-sided exact binomial test conditional on discordance, p=1/2."""
    if any(not isinstance(n, int) or n < 0 for n in (wrong_to_right, right_to_wrong)):
        raise ValueError("discordant counts must be nonnegative integers")
    n = wrong_to_right + right_to_wrong
    tail = min(wrong_to_right, right_to_wrong)
    if n == 0 or 2 * tail >= n - 1:
        return 1.0
    # Sum a binomial lower tail relative to its largest term, avoiding 2**n
    # overflow and underflow of the first term for full-sized panels.
    log_mass = math.lgamma(n + 1) - math.lgamma(tail + 1) - math.lgamma(n - tail + 1) - n * math.log(2)
    relative_sum = 1.0
    term = 1.0
    for k in range(tail, 0, -1):
        term *= k / (n - k + 1)
        relative_sum += term
        if term < relative_sum * 1e-16:
            break
    return min(1.0, 2 * math.exp(log_mass) * relative_sum)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p values in the original contrast order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("p values must be finite probabilities")
    output = [0.0] * len(pvalues)
    maximum = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        maximum = max(maximum, (len(pvalues) - rank) * pvalues[index])
        output[index] = min(1.0, maximum)
    return output


def stratified_paired_bootstrap(differences, subjects, replicates=BOOTSTRAP_REPLICATES,
                               seed=BOOTSTRAP_SEED) -> dict[str, Any]:
    """Resample paired outcomes within each fixed subject, preserving its n.

    Multinomial draws over {-1,0,+1} reproduce exactly the distribution of
    empirical paired resampling for the micro-accuracy difference statistic.
    No subject is resampled. NumPy is imported only when this function runs.
    """
    if len(differences) != len(subjects) or not len(subjects) or replicates <= 0:
        raise ValueError("bootstrap requires aligned samples and positive replicates")
    import numpy as np
    grouped = defaultdict(lambda: [0, 0, 0])
    for difference, subject in zip(differences, subjects):
        if difference not in (-1, 0, 1):
            raise ValueError("paired correctness differences must be -1/0/1")
        grouped[subject][difference + 1] += 1
    rng = np.random.default_rng(seed)
    totals = np.zeros(replicates, dtype=np.int64)
    for subject in sorted(grouped):
        counts = grouped[subject]
        n = sum(counts)
        draws = rng.multinomial(n, np.asarray(counts, dtype=np.float64) / n, size=replicates)
        totals += draws[:, 2] - draws[:, 0]
    values = totals * (100.0 / len(subjects))
    low, high = np.quantile(values, [0.025, 0.975])
    return {"method": "fixed_subject_stratified_paired_bootstrap",
            "sampling": "subject_multinomial_paired_difference_counts",
            "rng": "numpy.default_rng.PCG64", "numpy_version": np.__version__,
            "replicates": replicates, "seed": seed, "subject_count": len(grouped),
            "ci_level": 0.95, "ci_kind": "nominal_percentile_linear",
            "ci_low_pp": float(low), "ci_high_pp": float(high)}


def paired_contrast(reference_correct, treatment_correct, subjects,
                    bootstrap_replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED) -> dict[str, Any]:
    if not len(reference_correct) == len(treatment_correct) == len(subjects) or not len(subjects):
        raise ValueError("paired outcomes and subjects must have matching positive lengths")
    if any(value not in (0, 1) for value in list(reference_correct) + list(treatment_correct)):
        raise ValueError("correctness must be binary")
    differences = [int(t) - int(r) for r, t in zip(reference_correct, treatment_correct)]
    gains, losses = differences.count(1), differences.count(-1)
    grouped = defaultdict(list)
    for subject, difference in zip(subjects, differences):
        grouped[subject].append(difference)
    return {"n": len(subjects), "reference_correct": sum(reference_correct),
            "treatment_correct": sum(treatment_correct), "wrong_to_right": gains,
            "right_to_wrong": losses,
            "both_correct": sum(int(r and t) for r, t in zip(reference_correct, treatment_correct)),
            "both_wrong": sum(int(not r and not t) for r, t in zip(reference_correct, treatment_correct)),
            "delta_pp": (gains - losses) / len(subjects) * 100,
            "subject_macro_delta_pp": math.fsum(sum(v) / len(v) for v in grouped.values()) / len(grouped) * 100,
            "mcnemar_exact_p": exact_mcnemar_p(gains, losses),
            "bootstrap": stratified_paired_bootstrap(differences, subjects, bootstrap_replicates, seed)}


def analyze_panel(cells: Sequence[Mapping[str, Any]], gold, subjects,
                  bootstrap_replicates=BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    """Analyze aligned cell dicts containing manifest metadata and `scores`."""
    by_key, accuracy, correctness = {}, [], {}
    for cell in cells:
        key = (cell["model"], tuple(cell["window"]) if cell["window"] else None,
               cell["k"], cell["arm"])
        if key in by_key:
            raise ValueError("duplicate scientific panel configuration")
        by_key[key] = cell["cell_id"]
        stats = cell_accuracy(cell["scores"], gold, subjects)
        accuracy.append({"cell_id": cell["cell_id"], "model": cell["model"],
                         "window": cell["window"], "k": cell["k"], "arm": cell["arm"], **stats})
        correctness[cell["cell_id"]] = [int(p == g) for p, g in zip(score_predictions(cell["scores"]), gold)]
    contrasts = []
    for cell in cells:
        if cell["arm"] == "Native":
            continue
        window = tuple(cell["window"])
        references = [("Native", "EXPLORATORY_NATIVE_CONTEXT")]
        if cell["arm"] == "Spectral":
            references.insert(0, ("Loop", "K2_PRIMARY" if cell["k"] == 2 else "K4_SECONDARY"))
        for arm, family in references:
            ref_key = (cell["model"], None, None, arm) if arm == "Native" else (cell["model"], window, cell["k"], arm)
            reference = by_key[ref_key]
            result = paired_contrast(correctness[reference], correctness[cell["cell_id"]], subjects,
                                     bootstrap_replicates)
            contrasts.append({"reference": reference, "treatment": cell["cell_id"],
                              "family": family, "comparison": f'{cell["arm"]}-{arm}', **result,
                              "holm_adjusted_p": None, "holm_reject_alpha_0_05": None})
    for family in ("K2_PRIMARY", "K4_SECONDARY"):
        selected = [row for row in contrasts if row["family"] == family]
        if len(selected) != 4:
            raise ValueError(f"{family} requires exactly four frozen contrasts")
        adjusted = holm_adjust([row["mcnemar_exact_p"] for row in selected])
        for row, pvalue in zip(selected, adjusted):
            row.update(holm_adjusted_p=pvalue, holm_reject_alpha_0_05=pvalue <= 0.05,
                       holm_family_size=4)
    return {"cells": accuracy, "contrasts": contrasts,
            "interpretation": {"domain": "historically_used_development_benchmark",
                               "ci": "nominal; not a multiplicity-adjusted significance decision",
                               "limitation": "No matched-norm control; does not isolate direction specificity",
                               "k": "fixed horizon 1 with separately fitted K-specific bases"}}
