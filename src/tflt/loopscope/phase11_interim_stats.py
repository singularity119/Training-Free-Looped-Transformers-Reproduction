"""Aggregate-only GPQA interim statistics on one externally frozen snapshot."""
from collections import Counter
import re

from .phase11_accuracy import extract_final_answer
from .phase11_analysis import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FAMILY_COUNTS, frozen_contrasts
from .phase11_posthoc import S1_PATTERN, extract_s1, score_s1
from .phase8_accuracy_stats import holm_adjust, paired_contrast, stratified_paired_bootstrap


def _multiple_s1(text):
    labels = []
    for line in text.splitlines():
        match = re.fullmatch(S1_PATTERN, line.strip(), re.IGNORECASE)
        if match:
            labels.append((match[1] or match[2]).upper())
    if len(labels) < 2:
        return None
    return "consistent_multiple" if len(set(labels)) == 1 else "conflicting_multiple"


def _score(cell, rows, gold_by_identity, parser):
    parsed = []
    duplicates = Counter()
    for row in rows:
        value = extract_s1(row["generated_text"]) if parser == "S1" else extract_final_answer(
            row["generated_text"], row["candidate_labels"])
        parsed.append({**value, "end_reason": row["end_reason"]})
        duplicate = _multiple_s1(row["generated_text"])
        if duplicate is not None:
            duplicates[duplicate] += 1
    if rows:
        stats, correct = score_s1(parsed, [gold_by_identity[r["identity"]] for r in rows])
        ci = stratified_paired_bootstrap(correct, ["all_items"] * len(rows),
                                        replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        ci = {**ci, "ci_low": ci["ci_low_pp"] / 100, "ci_high": ci["ci_high_pp"] / 100}
        lengths = [r["generated_token_count"] for r in rows]
        length_stats = {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)}
    else:
        stats = {"n": 0, "correct": 0, "accuracy": None, "extraction_success_count": 0,
                 "extraction_success_rate": None, "extraction_failure_count": 0, "extraction_failure_rate": None,
                 "truncated_count": 0, "truncated_rate": None,
                 "groups": {reason: {"n": 0, "extraction_success_count": 0, "correct": 0}
                            for reason in ("eos", "length_limit")}}
        correct, ci, length_stats = [], None, None
    return {**cell, **stats, "accuracy_ci": ci, "generation_length": length_stats,
            "extraction_status_counts": dict(Counter(r["extraction_status"] for r in parsed)),
            "s1_duplicate_line_counts": {key: duplicates[key] for key in
                                        ("consistent_multiple", "conflicting_multiple")}}, correct


def _parser_analysis(panel, aligned, common_rows, gold, parser, specs, common_n):
    own, common, correctness = [], [], {}
    for cell in panel["cells"]:
        cid = cell["cell_id"]
        stats, _ = _score(cell, aligned[cid], gold, parser)
        own.append(stats)
        stats, correctness[cid] = _score(cell, common_rows[cid], gold, parser)
        common.append(stats)
    if common_n:
        contrasts = [{**spec, **paired_contrast(correctness[spec["reference"]], correctness[spec["treatment"]],
                     ["all_items"] * common_n, bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)}
                     for spec in specs]
        for family, count in FAMILY_COUNTS.items():
            selected = [r for r in contrasts if r["family"] == family]
            for row, p in zip(selected, holm_adjust([r["mcnemar_exact_p"] for r in selected])):
                row.update(holm_adjusted_p=p, holm_family_size=count, holm_reject_alpha_0_05=p <= .05)
        native = next(c for c in common if c["arm"] == "Native")
        loops = {c["k"]: c for c in common if c["arm"] == "Loop"}
        for cell in common:
            cell["delta_native_pp"] = 100 * (cell["accuracy"] - native["accuracy"])
            cell["delta_loop_pp"] = None if cell["k"] is None else 100 * (cell["accuracy"] - loops[cell["k"]]["accuracy"])
    else:
        contrasts = [{**spec, "n": 0, "delta_pp": None, "mcnemar_exact_p": None, "bootstrap": None,
                      "holm_adjusted_p": None, "holm_family_size": FAMILY_COUNTS[spec["family"]],
                      "holm_reject_alpha_0_05": None} for spec in specs]
        for cell in common:
            cell.update(delta_native_pp=None, delta_loop_pp=None)
    return {"parser": parser, "own_completion": own,
            "common_intersection": {"n": common_n, "cells": common, "contrasts": contrasts,
                                    "status": "COMMON_SAMPLE_ANALYZED" if common_n else "NO_COMMON_SAMPLES"}}


def analyze_snapshot(panel, aligned, gold_by_identity):
    """Use own n for completion descriptions and all-cell intersection for comparisons.

    The caller validates and freezes raw membership before supplying gold. This
    pure CPU function accesses no files and returns no sample identifiers or text.
    """
    if panel.get("dataset") != "gpqa_main":
        raise ValueError("interim analysis is restricted to GPQA")
    specs = frozen_contrasts(panel)
    if set(aligned) != {c["cell_id"] for c in panel["cells"]}:
        raise ValueError("snapshot must declare all 18 independent cells, including empty ones")
    identity_indices, index_identities, mappings = {}, {}, {}
    for cell in panel["cells"]:
        cid = cell["cell_id"]
        records = aligned[cid]
        by_identity, previous = {}, -1
        for record in records:
            identity = record["identity"]
            index = record.get("canonical_index", record.get("index"))
            if (not isinstance(identity, str) or not identity or identity in by_identity
                    or type(index) is not int or not 0 <= index < 448 or index <= previous):
                raise ValueError("snapshot identities/indices repeat or differ from canonical order")
            if identity in identity_indices and identity_indices[identity] != index:
                raise ValueError("shared snapshot identity has inconsistent canonical index")
            if index in index_identities and index_identities[index] != identity:
                raise ValueError("shared snapshot index has inconsistent identity")
            identity_indices[identity], index_identities[index] = index, identity
            previous = index
            if (record["candidate_labels"] != list("ABCD") or not isinstance(record["generated_text"], str)
                    or type(record["generated_token_count"]) is not int or not 0 < record["generated_token_count"] <= 8192
                    or record["end_reason"] not in ("eos", "length_limit")):
                raise ValueError("snapshot scoring inputs differ from frozen GPQA generation")
            if identity not in gold_by_identity or type(gold_by_identity[identity]) is not int or not 0 <= gold_by_identity[identity] < 4:
                raise ValueError("snapshot gold join is absent or outside legal candidate indices")
            by_identity[identity] = record
        mappings[cid] = by_identity
    common = set.intersection(*(set(rows) for rows in mappings.values()))
    common_order = sorted(common, key=identity_indices.__getitem__)
    common_rows = {cid: [rows[identity] for identity in common_order] for cid, rows in mappings.items()}
    return {"schema": "loopscope.phase11.interim_stats.v1", "status": "PARTIAL_SNAPSHOT_EXPLORATORY",
            "dataset": "gpqa_main", "primary_parser": "S1", "auxiliary_parser": "S0",
            "s1": _parser_analysis(panel, aligned, common_rows, gold_by_identity, "S1", specs, len(common)),
            "s0": _parser_analysis(panel, aligned, common_rows, gold_by_identity, "S0", specs, len(common)),
            "families": dict(FAMILY_COUNTS), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interpretation": {"scope": "one frozen interim snapshot; not a final full-panel result",
                "denominator": "own completed n only; missing outputs are excluded rather than marked wrong",
                "comparison": "all 18 cells share the same completed-identity intersection; no own-completion deltas",
                "completion_bias": "completed and common samples are completion-selected; selection can depend on generation length",
                "ci": "nominal 95 percent bootstrap; Holm within frozen 30/12/2 families separately per parser",
                "duplicates": "consistent and conflicting multiple S1 lines remain extraction failures; no prediction rescue",
                "adaptation": "interim results do not authorize changing, stopping or selecting the experiment"}}
