"""D8 full-denominator aggregates and the frozen two-comparison family."""
from collections import Counter

from .phase11_d8 import CELLS, SEED, COUNT, PARSER
from .phase11_panel import require
from .phase8_accuracy_stats import exact_mcnemar_p, holm_adjust


def analyze(data, gold, replicates=10000):
    import numpy as np
    require(set(data) == set(CELLS) and len(gold) == COUNT and
            all(type(g) is int and 0 <= g < 4 for g in gold), "D8 full gold/cell population differs")
    right, cells = {}, []
    for key in CELLS:
        rows = data[key]
        require(len(rows) == COUNT, "D8 all 448 items remain in the denominator")
        correctness = [int(r["prediction_index"] is not None and r["prediction_index"] == g)
                       for r, g in zip(rows, gold)]
        right[key] = np.asarray(correctness, dtype=np.int64)
        modes, ends = Counter(r["answer_mode"] for r in rows), Counter(r["end_reason"] for r in rows)
        cell = {"cell": key, "n": COUNT, "correct": sum(correctness),
            "accuracy_percent": 100 * sum(correctness) / COUNT,
            "EOS_count": ends["eos"], "length_limit_count": ends["length_limit"],
            "unclosed_thinking_count": sum(r["unclosed_thinking"] for r in rows),
            "empty_final_count": sum(r["final_empty"] for r in rows),
            "anomalous_thinking_count": sum(r["anomalous_thinking_boundaries"] for r in rows),
            "S2_no_match": modes["no_match"], "S2_conflict": modes["conflicting_multiple"],
            "S2_consistent_multiple": modes["consistent_multiple"], "S2_mode_counts": dict(modes),
            "question_wall_seconds": sum(r["elapsed_seconds"] for r in rows),
            "prefill_seconds": sum(r.get("prefill_elapsed_seconds", 0) for r in rows),
            "decode_seconds": sum(r.get("decode_elapsed_seconds", 0) for r in rows),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows)}
        for field in ("generated_token_count", "thinking_token_count", "final_token_count"):
            values = [r[field] for r in rows]
            cell[field] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                           "p90": float(np.quantile(values, .9)), "min": min(values), "max": max(values)}
        cells.append(cell)
    # Actual item-index resampling; one common paired draw matrix for both contrasts.
    draws = np.random.default_rng(SEED).integers(0, COUNT, size=(replicates, COUNT))
    contrasts = []
    for key in CELLS[1:]:
        d = right[key] - right["native"]
        gains, losses = int(np.sum(d == 1)), int(np.sum(d == -1))
        boot = d[draws].mean(axis=1) * 100
        low, high = np.quantile(boot, [.025, .975])
        contrasts.append({"treatment": key, "reference": "native", "n": COUNT,
            "delta_pp": float(d.mean() * 100), "wrong_to_right": gains, "right_to_wrong": losses,
            "mcnemar_exact_p": exact_mcnemar_p(gains, losses),
            "bootstrap": {"replicates": replicates, "seed": SEED, "method": "paired_item_index_percentile",
                "ci_level": .95, "ci_low_pp": float(low), "ci_high_pp": float(high), "nominal": True}})
    for row, p in zip(contrasts, holm_adjust([c["mcnemar_exact_p"] for c in contrasts])):
        row.update(holm_family_size=2, holm_adjusted_p=p, holm_reject_alpha_0_05=p <= .05)
    for row in cells:
        row["delta_native_pp"] = 100 * (row["correct"] - cells[0]["correct"]) / COUNT
    return {"schema": "loopscope.phase11.d8.analysis.v1", "status": "D8_COMPLETE_EXPLORATORY_TRANSFER",
        "record_count": 1344, "cell_count": 3, "primary_parser": PARSER,
        "cells": cells, "contrasts": contrasts, "bootstrap_seed": SEED, "bootstrap_replicates": replicates,
        "interpretation": "Configurations selected on the same GPQA items. Single-sample exploratory transfer across checkpoint/thinking recipe; not independent-item confirmation, not a separate causal effect of thinking or checkpoint, and no sampling-variance estimate."}
