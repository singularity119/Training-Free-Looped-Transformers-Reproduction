"""S2 consistent-final-line extraction and aggregate-only snapshot statistics."""
from collections import Counter
import re

from .phase11_analysis import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FAMILY_COUNTS, frozen_contrasts
from .phase11_posthoc import S1_PATTERN, score_s1
from .phase8_accuracy_stats import holm_adjust, paired_contrast, stratified_paired_bootstrap


def extract_s2(text):
    """Apply the unchanged S1 line syntax; accept one normalized distinct letter."""
    if not isinstance(text, str):
        raise ValueError("S2 response must be text")
    letters = []
    for line in text.splitlines():
        match = re.fullmatch(S1_PATTERN, line.strip(), re.IGNORECASE)
        if match:
            letters.append((match[1] or match[2]).upper())
    distinct = set(letters)
    mode = ("no_match" if not letters else "single" if len(letters) == 1
            else "consistent_multiple" if len(distinct) == 1 else "conflicting_multiple")
    valid = len(distinct) == 1
    return {"extraction_status": "unique_valid" if valid else
            ("no_valid_final_line" if not letters else "multiple_valid_final_lines"),
            "answer_mode": mode, "valid_final_line_count": len(letters),
            "distinct_valid_label_count": len(distinct),
            "predicted_label": letters[0] if valid else None,
            "prediction_index": "ABCD".index(letters[0]) if valid else None}


def _score(cell, rows, gold):
    parsed = [{**extract_s2(row["generated_text"]), "end_reason": row["end_reason"]} for row in rows]
    if rows:
        stats, correct = score_s1(parsed, [gold[row["identity"]] for row in rows])
        ci = stratified_paired_bootstrap(correct, ["all_items"] * len(rows),
                                        replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        ci = {**ci, "ci_low": ci["ci_low_pp"] / 100, "ci_high": ci["ci_high_pp"] / 100}
        lengths = [row["generated_token_count"] for row in rows]
        length_stats = {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)}
    else:
        stats = {"n": 0, "correct": 0, "accuracy": None, "extraction_success_count": 0,
                 "extraction_success_rate": None, "extraction_failure_count": 0, "extraction_failure_rate": None,
                 "truncated_count": 0, "truncated_rate": None,
                 "groups": {reason: {"n": 0, "extraction_success_count": 0, "correct": 0}
                            for reason in ("eos", "length_limit")}}
        correct, ci, length_stats = [], None, None
    modes = Counter(row["answer_mode"] for row in parsed)
    consistent_correct = sum(value for row, value in zip(parsed, correct) if row["answer_mode"] == "consistent_multiple")
    return {**cell, **stats, "accuracy_ci": ci, "generation_length": length_stats,
            "extraction_status_counts": dict(Counter(row["extraction_status"] for row in parsed)),
            "consistent_multiple_count": modes["consistent_multiple"],
            "consistent_multiple_correct": consistent_correct,
            "consistent_multiple_incorrect": modes["consistent_multiple"] - consistent_correct,
            "conflicting_multiple_count": modes["conflicting_multiple"],
            "single_count": modes["single"], "no_match_count": modes["no_match"]}, correct


def analyze_s2_snapshot(panel, aligned, gold_by_identity):
    """Score own completion sets and paired all-cell intersection without file I/O.

    The caller freezes the exact snapshot and S2 predictions before supplying
    gold. The returned object contains only cell and contrast aggregates.
    """
    if panel.get("dataset") != "gpqa_main":
        raise ValueError("S2 snapshot analysis is restricted to GPQA")
    specs = frozen_contrasts(panel)
    if set(aligned) != {cell["cell_id"] for cell in panel["cells"]}:
        raise ValueError("S2 snapshot must declare all 18 independent cells, including empty ones")
    identity_indices, index_identities, mappings = {}, {}, {}
    for cell in panel["cells"]:
        cid = cell["cell_id"]
        by_identity, previous = {}, -1
        for row in aligned[cid]:
            identity = row["identity"]
            index = row.get("canonical_index", row.get("index"))
            if (not isinstance(identity, str) or not identity or identity in by_identity
                    or type(index) is not int or not 0 <= index < 448 or index <= previous):
                raise ValueError("S2 snapshot identity/index repeats or differs from canonical order")
            if identity in identity_indices and identity_indices[identity] != index:
                raise ValueError("S2 snapshot identity has inconsistent canonical index")
            if index in index_identities and index_identities[index] != identity:
                raise ValueError("S2 snapshot canonical index has inconsistent identity")
            identity_indices[identity], index_identities[index] = index, identity
            previous = index
            if (row["candidate_labels"] != list("ABCD") or not isinstance(row["generated_text"], str)
                    or type(row["generated_token_count"]) is not int or not 0 < row["generated_token_count"] <= 8192
                    or row["end_reason"] not in ("eos", "length_limit")):
                raise ValueError("S2 snapshot scoring inputs differ from GPQA generation")
            if identity not in gold_by_identity or type(gold_by_identity[identity]) is not int or not 0 <= gold_by_identity[identity] < 4:
                raise ValueError("S2 snapshot gold join is absent or outside legal indices")
            by_identity[identity] = row
        mappings[cid] = by_identity
    intersection = set.intersection(*(set(rows) for rows in mappings.values()))
    order = sorted(intersection, key=identity_indices.__getitem__)
    own, common, correctness = [], [], {}
    for cell in panel["cells"]:
        cid = cell["cell_id"]
        stats, _ = _score(cell, aligned[cid], gold_by_identity)
        own.append(stats)
        stats, correctness[cid] = _score(cell, [mappings[cid][identity] for identity in order], gold_by_identity)
        common.append(stats)
    n = len(order)
    if n:
        contrasts = [{**spec, **paired_contrast(correctness[spec["reference"]], correctness[spec["treatment"]],
                      ["all_items"] * n, bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)}
                     for spec in specs]
        for family, count in FAMILY_COUNTS.items():
            selected = [row for row in contrasts if row["family"] == family]
            for row, p in zip(selected, holm_adjust([row["mcnemar_exact_p"] for row in selected])):
                row.update(holm_adjusted_p=p, holm_family_size=count, holm_reject_alpha_0_05=p <= .05)
        native = next(cell for cell in common if cell["arm"] == "Native")
        loops = {cell["k"]: cell for cell in common if cell["arm"] == "Loop"}
        for cell in common:
            cell["delta_native_pp"] = 100 * (cell["accuracy"] - native["accuracy"])
            cell["delta_loop_pp"] = None if cell["k"] is None else 100 * (cell["accuracy"] - loops[cell["k"]]["accuracy"])
    else:
        contrasts = [{**spec, "n": 0, "delta_pp": None, "mcnemar_exact_p": None, "bootstrap": None,
                      "holm_adjusted_p": None, "holm_family_size": FAMILY_COUNTS[spec["family"]],
                      "holm_reject_alpha_0_05": None} for spec in specs]
        for cell in common:
            cell.update(delta_native_pp=None, delta_loop_pp=None)
    return {"schema": "loopscope.phase11.s2_snapshot_stats.v1", "status": "S2_SNAPSHOT_EXPLORATORY",
            "dataset": "gpqa_main", "primary_parser": "S2",
            "s2": {"parser": "S2", "own_completion": own,
                   "common_intersection": {"n": n, "cells": common, "contrasts": contrasts,
                                           "status": "COMMON_SAMPLE_ANALYZED" if n else "NO_COMMON_SAMPLES"}},
            "families": dict(FAMILY_COUNTS), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "auxiliary_artifacts": "original S1/S0 artifacts retained unchanged",
            "interpretation": {"scope": "one frozen 8192 snapshot; not a final full-panel result",
                "parser_selection": "S2 chosen after output diagnostics; post-hoc exploratory, not confirmatory",
                "denominator": "each cell's actual completed n; missing outputs excluded rather than marked wrong",
                "comparison": "all-cell completed identity intersection; own-completion tables contain no deltas",
                "completion_bias": "completed and common samples are completion-selected, potentially by generation length",
                "ci": "nominal 95 percent bootstrap; Holm within frozen 30/12/2 contrast families",
                "adaptation": "snapshot results do not authorize changing, stopping or selecting the experiment"}}
