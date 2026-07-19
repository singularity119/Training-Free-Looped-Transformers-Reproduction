"""One-shot Phase 4 P4-D outcome unseal and frozen paired analysis.

The implementation is intentionally narrow.  It accepts only the frozen P4-C
eight-cell panel, restores the P4-B canonical population order in memory, and
persists aggregate statistics only.  Raw prompts, labels, generations, and a
row-level correctness matrix are never written.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tflt.loopscope.phase4_acquisition import load_strict_json, load_strict_jsonl
from tflt.loopscope.phase4_outcome import CELLS
from tflt.loopscope.phase4_schema import (
    canonical_identity,
    file_sha256,
    semantic_sha256,
    validate_source_record,
)
from tflt.loopscope.phase4_selector import verify_selector_report


GATE = "P4-D"
EXECUTOR_THREAD_ID = "019f78aa-a161-7af0-a080-b7a03f556495"
PLANNING_THREAD_ID = "019f6bf0-d1ec-7d53-ba06-391e9db2bd88"
AUTHORIZED_BASE_COMMIT = "a210da31d734d539b198d939118136e3e035a36f"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUTHORIZED_OUTPUT_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase4-p4d-20260719T044036Z"
)
P4B_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase4-p4b-20260716T204154Z"
)
P4C_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase4-p4c-20260716T223844Z"
)
B2_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase4-b2-hidden-geometry-20260717T064424Z"
)
B_FAMILY_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase4-b-family-report-20260717T081130Z"
)
B2_ANALYSIS_PATH = B2_ROOT / "analysis/phase4_b2_hidden_geometry_analysis.json"
B2_RECEIPT_PATH = B2_ROOT / "phase4_b2_gate_receipt.json"
B_FAMILY_CSV_PATH = B_FAMILY_ROOT / "phase4_b_family_boundary_table.csv"
B_FAMILY_MD_PATH = B_FAMILY_ROOT / "phase4_b_family_boundary_table.md"
B_FAMILY_RECEIPT_PATH = B_FAMILY_ROOT / "phase4_b_family_report_receipt.json"

POLICY_BYTE_SHA256 = "9784d91d48813932feb547436f4d79677aa847e70242e917780bbe4e6064fc73"
CONTROL_BYTE_SHA256 = "1429586c23af0238a761c1e3525cb0660a15f6f08377808f399f43f477c5e6d6"
CARD_BYTE_SHA256 = "980955386907a1865699808219da1379031c1395d9cdb77029585cf266f160f8"
SOURCE_MANIFEST_SHA256 = "173aeb1f7a8663652975ec4d0faf0998d9a017e9fb4ee30e149f23dc1a2e944f"
SELECTOR_REPORT_SHA256 = "a941ccb15fafc98062b895918c2d61791d55ef36d62b15794c7f7dcef2639d35"
SELECTOR_FREEZE_SHA256 = "a4cc48a9b82b3d89b962cdefb55b6e4133c08a0aff3431f0de72c3750fc062c8"
SELECTOR_RECEIPT_SHA256 = "05fb8184a3ec6b98c354d75635685feec0747ce302e13cb70bcc6c538eebc521"
PANEL_SHA256 = "02148127ef7b634106b1ee1044f591012d3543563e94344583480a7c3061eb79"
PANEL_MANIFEST_SHA256 = "abc65342e5444638d3073ce780ec37d8d72eb7c34d87ac809bdb29a24056d094"
B2_ANALYSIS_SHA256 = "9cb7138faa4e15d5948d5053ce0483fdd8b73d360d6d6ed408f118f897114732"
B2_RECEIPT_SHA256 = "89961524aabd7a9050cc3d90e2b2fec7f4df32bd3806d2cbc1d3506bd30b08a9"
B_FAMILY_CSV_SHA256 = "4f569c61549ce7985561d3296b8eb646fefa027a7925d4a5ea7c2089f2b3b9d0"
B_FAMILY_MD_SHA256 = "3470d9fbdf5a3976d4bc190dbb501514b8ee63bff1121649c05cc7fb31daade2"
B_FAMILY_RECEIPT_SHA256 = "48e509c9c37ef41f8d18a7a4213fa616b4c7a8a1bd076828da878c6783ff4395"
P4C_COMPLETION_SHA256 = "acc00860913970f622729c9ebc29e72ae3a3457651765367aff1695696c6b7b8"
P4C_COMPLETION_MANIFEST_SHA256 = "5999ed08879474ea42b1b3f9703c90e93575f826a36a963c1a5b4cebd3bef59e"

RESULT_SHA256 = (
    "f6e59b0ba020bba7cf450f9178aae8ad2419be7b1422ea760cd171ba2f87a07e",
    "9ae347f22ca412e130b0a8ba698bd64ad096143a08bae99a6bcc9357bcabf3ce",
    "a9d670bbe6cec16b87e9ee2b98002f239e062ec014f68483ebe0ac4c1452f157",
    "8bddd5f264324ce2c7fc419a9ff49abbf3dc7201d1efb31911034f3a70216fb9",
    "1de11d08dc2ddf62ff822457a5ccf161b81ec0f0bc24577d2954e9f580bdc318",
    "11db36b0b08feb57b37c636d1db2fe8b4189f0e9d9d0103cde62102b5d5a945f",
    "6d0a47da84fe654f1ec357002aea5122d8e62359a41523d441d8b5974c196139",
    "a64833b443d9e9c29d62b9cd93cfac129302b6fd7f807a46b05f7364d27812c5",
)
RESULT_PATHS = tuple(
    P4C_ROOT / "full" / ("cell-%02d-%s" % (index, cell[0])) / "results.json"
    for index, cell in enumerate(CELLS)
)

SOURCE_MANIFEST = P4B_ROOT / "source/phase4_shared12032_source_manifest.json"
SOURCE_RECORDS = P4B_ROOT / "source/shared12032.jsonl"
SELECTOR_REPORT = P4B_ROOT / "freeze/phase4_selector_report.json"
SELECTOR_FREEZE = P4B_ROOT / "freeze/phase4_selector_freeze.json"
SELECTOR_RECEIPT = P4B_ROOT / "freeze/phase4_selector_verifier_receipt.json"
PANEL_PATH = P4B_ROOT / "freeze/phase4_outcome_panel_manifest.json"
COMPLETION_RECEIPT = P4C_ROOT / "phase4_outcome_completion_receipt.json"

ANALYSIS_NAME = "phase4_outcome_analysis.json"
TABLE_NAME = "phase4_outcome_table.csv"
VERIFIER_NAME = "phase4_outcome_verifier_receipt.json"
SUMMARY_NAME = "phase4_result_summary_zh.md"
PHASE_AUDIT_NAME = "phase4_phase_end_audit.json"
GATE_RECEIPT_NAME = "phase4_p4d_gate_receipt.json"
EXPECTED_COUNT = 12032
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260718
SELECTION_STATUS_LABEL = "PV_EK_TRS_ABSTAIN_WITH_RANKING_RESULT"
IMPLEMENTATION_PATHS = (
    "src/tflt/loopscope/phase4_outcome_analysis.py",
    "scripts/loopscope/run_qwen4_phase4_p4d.py",
    "tests/test_loopscope_phase4_outcome_analysis.py",
)


class P4DError(ValueError):
    """Fail-closed P4-D contract, one-shot, or provenance error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise P4DError("git provenance command failed: %s" % completed.stderr.strip())
    return completed.stdout.strip()


def git_provenance(expected_commit: str, *, remote_required: bool) -> Dict[str, Any]:
    root = repository_root().resolve()
    if remote_required and root != REMOTE_REPO:
        raise P4DError("P4-D remote action is outside the dedicated LoopScope clone")
    observed = {
        "repo": str(root),
        "branch": _git(root, "symbolic-ref", "--short", "HEAD"),
        "commit": _git(root, "rev-parse", "HEAD"),
        "origin_loopscope": _git(root, "rev-parse", "origin/loopscope"),
        "dirty": bool(_git(root, "status", "--porcelain")),
    }
    if (
        observed["branch"] != "loopscope"
        or observed["commit"] != expected_commit
        or observed["origin_loopscope"] != expected_commit
        or observed["dirty"]
    ):
        raise P4DError("Git provenance is not clean/exact at the expected pushed commit")
    return observed


def implementation_hashes() -> Dict[str, str]:
    result = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repository_root() / relative
        if not path.is_file():
            raise P4DError("missing P4-D implementation path: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        raise P4DError("percentile requires values")
    if not 0.0 <= quantile <= 1.0 or not all(math.isfinite(value) for value in numbers):
        raise P4DError("percentile input is invalid")
    position = quantile * (len(numbers) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return numbers[lower]
    fraction = position - lower
    return numbers[lower] * (1.0 - fraction) + numbers[upper] * fraction


def exact_mcnemar_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    for value in (wrong_to_correct, correct_to_wrong):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise P4DError("McNemar counts must be non-negative integers")
    total = wrong_to_correct + correct_to_wrong
    if total == 0:
        return 1.0
    tail = min(wrong_to_correct, correct_to_wrong)
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        - total * math.log(2.0)
        for index in range(tail + 1)
    ]
    peak = max(logs)
    one_sided = math.exp(peak) * math.fsum(math.exp(value - peak) for value in logs)
    return min(1.0, 2.0 * one_sided)


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        rank = 0.5 * ((offset + 1) + end)
        for index in order[offset:end]:
            result[index] = rank
        offset = end
    return result


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise P4DError("Spearman inputs must have equal length >=2")
    left_rank = _average_ranks([float(value) for value in left])
    right_rank = _average_ranks([float(value) for value in right])
    left_mean = math.fsum(left_rank) / len(left_rank)
    right_mean = math.fsum(right_rank) / len(right_rank)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left_rank, right_rank)
    )
    left_ss = math.fsum((value - left_mean) ** 2 for value in left_rank)
    right_ss = math.fsum((value - right_mean) ** 2 for value in right_rank)
    if left_ss == 0.0 or right_ss == 0.0:
        raise P4DError("Spearman rank vector is constant")
    return numerator / math.sqrt(left_ss * right_ss)


def exact_spearman_permutation(
    scores: Sequence[float], gains: Sequence[float]
) -> Dict[str, Any]:
    if len(scores) != 6 or len(gains) != 6:
        raise P4DError("blind-six exact permutation requires six scores and gains")
    observed = spearman_rho(scores, gains)
    extreme = 0
    count = 0
    for permutation in itertools.permutations(range(6)):
        rho = spearman_rho(scores, [gains[index] for index in permutation])
        count += 1
        if abs(rho) >= abs(observed):
            extreme += 1
    if count != 720:
        raise P4DError("blind-six exhaustive permutation did not enumerate 720 orderings")
    return {
        "method": "exhaustive_exact_two_sided_spearman_permutation",
        "rho": observed,
        "permutation_count": count,
        "extreme_count_abs_rho_ge_observed": extreme,
        "p_value": extreme / 720.0,
    }


def joint_category_stratified_bootstrap(
    correctness: np.ndarray,
    categories: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    return_replicates: bool = False,
) -> Dict[str, Any]:
    matrix = np.asarray(correctness, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[1] != 8 or matrix.shape[0] != len(categories):
        raise P4DError("bootstrap correctness must have shape N x 8")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 2:
        raise P4DError("bootstrap replicates must be at least two")
    if not np.logical_or(matrix == 0, matrix == 1).all():
        raise P4DError("bootstrap correctness must be binary")
    category_order = sorted(set(categories))
    if not category_order or any(not str(value).strip() for value in categories):
        raise P4DError("bootstrap categories must be non-empty")
    groups = [
        np.asarray([index for index, value in enumerate(categories) if value == category])
        for category in category_order
    ]
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates, 8), dtype=np.float64)
    stream_hash = hashlib.sha256()
    batch_size = min(64, replicates)
    for offset in range(0, replicates, batch_size):
        size = min(batch_size, replicates - offset)
        sums = np.zeros((size, 8), dtype=np.int64)
        for group in groups:
            draws = rng.integers(0, len(group), size=(size, len(group)), endpoint=False)
            selected = group[draws]
            stream_hash.update(np.asarray(selected, dtype="<i8").tobytes(order="C"))
            sums += matrix[selected].sum(axis=1, dtype=np.int64)
        estimates[offset : offset + size] = sums / float(matrix.shape[0])
    gains = estimates[:, 1:] - estimates[:, [0]]
    mean_high = gains[:, 1:4].mean(axis=1)
    mean_low = gains[:, 4:7].mean(axis=1)
    enrichment = mean_high - mean_low

    def summary(values: np.ndarray) -> Dict[str, Any]:
        numbers = values.tolist()
        return {
            "bootstrap_mean_fraction": float(values.mean()),
            "percentile_95_ci_fraction": [
                linear_percentile(numbers, 0.025),
                linear_percentile(numbers, 0.975),
            ],
        }

    result: Dict[str, Any] = {
        "method": "one_joint_category_stratified_paired_sample_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "percentile_method": "linear_interpolation_at_q_times_R_minus_1",
        "same_sampled_indices_all_eight_cells": True,
        "category_counts_preserved_each_replicate": True,
        "category_order": category_order,
        "index_stream_sha256": stream_hash.hexdigest(),
        "cell_gain_summaries": [summary(gains[:, index]) for index in range(7)],
        "mean_high_gain": summary(mean_high),
        "mean_low_gain": summary(mean_low),
        "G_enrich": summary(enrichment),
    }
    if return_replicates:
        result["_replicate_cell_accuracy"] = estimates.tolist()
        result["_replicate_G_enrich"] = enrichment.tolist()
    return result


def scientific_phase_label(enrichment_label: str) -> str:
    mapping = {
        "ENRICHMENT_SUPPORTED": "PV_EK_TRS_PROSPECTIVE_RANKING_SIGNAL_ONLY",
        "ENRICHMENT_REFUTED": "PV_EK_TRS_NOT_SUPPORTED",
        "ENRICHMENT_INCONCLUSIVE": "PV_EK_TRS_INCONCLUSIVE",
    }
    try:
        return mapping[enrichment_label]
    except KeyError as exc:
        raise P4DError("unknown enrichment label") from exc


def _transitions(baseline: np.ndarray, candidate: np.ndarray) -> Dict[str, int]:
    return {
        "correct_to_correct": int(np.logical_and(baseline == 1, candidate == 1).sum()),
        "correct_to_wrong": int(np.logical_and(baseline == 1, candidate == 0).sum()),
        "wrong_to_correct": int(np.logical_and(baseline == 0, candidate == 1).sum()),
        "wrong_to_wrong": int(np.logical_and(baseline == 0, candidate == 0).sum()),
    }


def build_statistical_analysis(
    *,
    identities: Sequence[str],
    categories: Sequence[str],
    correctness_by_cell: Mapping[str, Sequence[bool]],
    selector_scores: Mapping[str, float],
    selector_ranks: Mapping[str, int],
    expected_count: int = EXPECTED_COUNT,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    cell_ids = [cell[0] for cell in CELLS]
    if list(correctness_by_cell) != cell_ids:
        raise P4DError("correctness cell membership/order differs from frozen eight cells")
    if len(identities) != expected_count or len(categories) != expected_count:
        raise P4DError("canonical population count differs")
    if len(set(identities)) != expected_count:
        raise P4DError("canonical identities contain a duplicate")
    matrix = np.column_stack(
        [np.asarray(correctness_by_cell[cell_id], dtype=np.int8) for cell_id in cell_ids]
    )
    if matrix.shape != (expected_count, 8):
        raise P4DError("correctness matrix does not close N x 8")
    if not np.logical_or(matrix == 0, matrix == 1).all():
        raise P4DError("correctness is not binary")
    expected_windows = {cell[2] for cell in CELLS[1:]}
    if set(selector_scores) != expected_windows or set(selector_ranks) != expected_windows:
        raise P4DError("selector score/rank membership differs from frozen seven loop cells")

    point_accuracy = matrix.mean(axis=0, dtype=np.float64)
    point_gains = point_accuracy[1:] - point_accuracy[0]
    point_high = float(point_gains[1:4].mean())
    point_low = float(point_gains[4:7].mean())
    point_enrichment = point_high - point_low
    bootstrap = joint_category_stratified_bootstrap(
        matrix, categories, replicates=replicates, seed=seed
    )
    cells = []
    for index, (cell_id, role, window) in enumerate(CELLS):
        if index == 0:
            gain = 0.0
            ci = [0.0, 0.0]
            transitions = {
                "correct_to_correct": int(matrix[:, 0].sum()),
                "correct_to_wrong": 0,
                "wrong_to_correct": 0,
                "wrong_to_wrong": int(expected_count - matrix[:, 0].sum()),
            }
            p_value = 1.0
        else:
            gain = float(point_gains[index - 1])
            ci = bootstrap["cell_gain_summaries"][index - 1]["percentile_95_ci_fraction"]
            transitions = _transitions(matrix[:, 0], matrix[:, index])
            p_value = exact_mcnemar_p(
                transitions["wrong_to_correct"], transitions["correct_to_wrong"]
            )
        cells.append(
            {
                "index": index,
                "cell_id": cell_id,
                "role": role,
                "window": window,
                "sample_count": expected_count,
                "accuracy_fraction": float(point_accuracy[index]),
                "gain_fraction": gain,
                "gain_percentage_points": gain * 100.0,
                "paired_percentile_95_ci_fraction": list(ci),
                "paired_percentile_95_ci_percentage_points": [value * 100.0 for value in ci],
                "transitions_vs_baseline": transitions,
                "exact_two_sided_mcnemar_p": p_value,
                "selector_score": None if window is None else float(selector_scores[window]),
                "selector_rank": None if window is None else int(selector_ranks[window]),
                "enters_blind_six_enrichment_and_rho": role in {"blind_high", "blind_low"},
            }
        )

    blind_rows = [row for row in cells if row["enters_blind_six_enrichment_and_rho"]]
    rho = exact_spearman_permutation(
        [row["selector_score"] for row in blind_rows],
        [row["gain_fraction"] for row in blind_rows],
    )
    g_ci = bootstrap["G_enrich"]["percentile_95_ci_fraction"]
    if g_ci[0] > 0.0:
        enrichment_label = "ENRICHMENT_SUPPORTED"
    elif g_ci[1] < 0.0:
        enrichment_label = "ENRICHMENT_REFUTED"
    else:
        enrichment_label = "ENRICHMENT_INCONCLUSIVE"
    panel_rows = cells[1:]
    best_accuracy = max(row["accuracy_fraction"] for row in panel_rows)
    best_rows = [row for row in panel_rows if row["accuracy_fraction"] == best_accuracy]
    abstain_reason = "FROZEN_SELECTOR_SELECTED_WINDOW_IS_NULL_ABSTAIN"
    na = {"status": "NA_ABSTAIN", "reason": abstain_reason}
    return {
        "schema_version": "loopscope.phase4.p4d-outcome-analysis.v1",
        "artifact_role": "p4d_one_shot_frozen_eight_cell_joint_paired_analysis",
        "gate": GATE,
        "population": {
            "record_count": expected_count,
            "cell_count": 8,
            "matrix_shape": [expected_count, 8],
            "ordered_identity_sha256": semantic_sha256(list(identities)),
            "category_counts": dict(sorted(Counter(categories).items())),
            "identity_missing": 0,
            "identity_extra": 0,
            "identity_duplicate": 0,
            "raw_and_canonical_order_exact": True,
        },
        "metric": {
            "aggregate": "arithmetic_mean_per_sample_correctness",
            "per_sample": "exact_match,custom-extract",
            "unit": "fraction",
        },
        "cells": cells,
        "groups": {
            "blind_high3": {
                "windows": [cell[2] for cell in CELLS[2:5]],
                "mean_gain_fraction": point_high,
                "percentile_95_ci_fraction": bootstrap["mean_high_gain"]["percentile_95_ci_fraction"],
            },
            "blind_low3": {
                "windows": [cell[2] for cell in CELLS[5:8]],
                "mean_gain_fraction": point_low,
                "percentile_95_ci_fraction": bootstrap["mean_low_gain"]["percentile_95_ci_fraction"],
            },
            "G_enrich": {
                "formula": "mean_gain(blind_high3)-mean_gain(blind_low3)",
                "point_fraction": point_enrichment,
                "point_percentage_points": point_enrichment * 100.0,
                "percentile_95_ci_fraction": list(g_ci),
                "percentile_95_ci_percentage_points": [value * 100.0 for value in g_ci],
                "label": enrichment_label,
            },
        },
        "bootstrap": bootstrap,
        "blind_six_selector_score_vs_gain": {
            "windows_in_frozen_cell_order": [row["window"] for row in blind_rows],
            **rho,
            "fixed_15_18_excluded": True,
        },
        "abstain": {
            "selected_window": None,
            "selector_decision": "ABSTAIN",
            "selected_vs_baseline": dict(na),
            "selected_vs_15_18": dict(na),
            "selected_competitiveness_margin_0_30pp": dict(na),
            "selected_panel_regret": dict(na),
        },
        "panel_local": {
            "scope": "fixed_15_18_plus_blind_high3_plus_blind_low3_only_not_global",
            "panel_best_window": best_rows[0]["window"],
            "panel_best_windows": [row["window"] for row in best_rows],
            "panel_best_tied": len(best_rows) > 1,
            "panel_best_gain_fraction": best_rows[0]["gain_fraction"],
        },
        "labels": {
            "selection_status_label": SELECTION_STATUS_LABEL,
            "scientific_phase_label": scientific_phase_label(enrichment_label),
            "forbidden_selection_supported_label_used": False,
        },
        "claim_scope": {
            "reproduction": "current-checkpoint reproduction",
            "population": "same-population outcome-blind transductive",
            "unseen_prompt_generalization_claimed": False,
            "template_dominance_risk_retained": True,
            "b2_interpretation": "hidden/output geometry agreement only; not gain prediction",
        },
        "persistence_safety": {
            "raw_question_persisted": False,
            "gold_label_persisted": False,
            "generated_response_or_tokens_persisted": False,
            "prompt_or_cot_content_persisted": False,
            "row_level_correctness_matrix_persisted": False,
        },
    }


def _sample_correctness(row: Mapping[str, Any]) -> bool:
    containers = (row, row.get("metrics"))
    for container in containers:
        if isinstance(container, Mapping) and "exact_match" in container:
            value = container["exact_match"]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) in (0.0, 1.0):
                return bool(value)
            raise P4DError("per-sample exact_match must be exactly 0 or 1")
    raise P4DError("sample lacks the exact_match outcome for custom-extract evaluation")


def _safe_identity_from_sample(row: Mapping[str, Any]) -> Tuple[str, str]:
    doc = row.get("doc")
    if not isinstance(doc, Mapping):
        raise P4DError("outcome sample lacks its source doc identity")
    required = ("question_id", "category", "src", "question", "options")
    if any(key not in doc for key in required):
        raise P4DError("outcome sample source doc lacks canonical identity fields")
    safe = {
        "question_id": doc["question_id"],
        "category": doc["category"],
        "src": doc["src"],
        "question": doc["question"],
        "ordered_options": list(doc["options"]),
    }
    return canonical_identity(safe), str(doc["category"]).strip()


def load_correctness_cell(
    path: Path,
    *,
    expected_identities: Sequence[str],
    expected_categories: Sequence[str],
) -> Tuple[List[bool], str]:
    payload = load_strict_json(path)
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, Mapping):
        raise P4DError("results.json lacks inline samples")
    observed_order: List[str] = []
    values: Dict[str, Tuple[str, bool]] = {}
    for namespace, rows in raw_samples.items():
        if not (str(namespace) == "mmlu_pro" or str(namespace).startswith("mmlu_pro_")):
            continue
        if not isinstance(rows, list):
            raise P4DError("MMLU-Pro sample namespace is not a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise P4DError("MMLU-Pro sample row is not an object")
            identity, category = _safe_identity_from_sample(row)
            if identity in values:
                raise P4DError("outcome sample identity is duplicated")
            values[identity] = (category, _sample_correctness(row))
            observed_order.append(identity)
    expected_set = set(expected_identities)
    if set(values) != expected_set or len(values) != len(expected_identities):
        raise P4DError("outcome cell has missing/extra/duplicate canonical identities")
    for identity, category in zip(expected_identities, expected_categories):
        if values[identity][0] != category:
            raise P4DError("outcome category differs from canonical source")
    evaluator_order_sha256 = semantic_sha256(observed_order)
    result = [values[identity][1] for identity in expected_identities]
    del payload, values
    gc.collect()
    return result, evaluator_order_sha256


def validate_evaluator_order_hashes(hashes: Sequence[str]) -> str:
    if len(hashes) != 8 or len(set(hashes)) != 1:
        raise P4DError("raw evaluator sample order differs across the eight cells")
    return str(hashes[0])


def _require_hash(path: Path, expected: str, context: str) -> None:
    if not Path(path).is_file() or Path(path).is_symlink() or file_sha256(path) != expected:
        raise P4DError("%s file hash/path differs" % context)


def validate_panel_manifest(panel_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    if panel_manifest.get("manifest_sha256") != PANEL_MANIFEST_SHA256:
        raise P4DError("panel embedded manifest differs")
    panel = panel_manifest.get("panel")
    if not isinstance(panel, Mapping):
        raise P4DError("frozen panel manifest lacks its panel payload")
    expected_panel = {
        "baseline": "no-loop",
        "fixed_comparator": "15:18",
        "blind_high3": ["6:9", "10:13", "25:28"],
        "blind_low3": ["4:7", "5:8", "22:25"],
        "selected_window": None,
        "abstain": True,
        "variable_width_outcomes_authorized": False,
    }
    for key, value in expected_panel.items():
        if panel.get(key) != value:
            raise P4DError("frozen panel field differs: %s" % key)
    return dict(panel)


def validate_inputs() -> Dict[str, Any]:
    fixed = {
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        SELECTOR_REPORT: SELECTOR_REPORT_SHA256,
        SELECTOR_FREEZE: SELECTOR_FREEZE_SHA256,
        SELECTOR_RECEIPT: SELECTOR_RECEIPT_SHA256,
        PANEL_PATH: PANEL_SHA256,
        COMPLETION_RECEIPT: P4C_COMPLETION_SHA256,
    }
    for path, digest in fixed.items():
        _require_hash(path, digest, path.name)
    for path, digest in zip(RESULT_PATHS, RESULT_SHA256):
        _require_hash(path, digest, path.name)
    _require_hash(repository_root() / "AGENTS.md", POLICY_BYTE_SHA256, "AGENTS.md")
    _require_hash(
        repository_root() / "configs/loopscope/phase4_pv_ek_trs_card.json",
        CARD_BYTE_SHA256,
        "Phase 4 card",
    )

    source_manifest = load_strict_json(SOURCE_MANIFEST)
    if source_manifest.get("record_count") != EXPECTED_COUNT:
        raise P4DError("P4-B source manifest count differs")
    _require_hash(SOURCE_RECORDS, source_manifest.get("record_file_sha256"), "source records")
    selector_report = load_strict_json(SELECTOR_REPORT)
    verify_selector_report(selector_report)
    panel = validate_panel_manifest(load_strict_json(PANEL_PATH))
    completion = load_strict_json(COMPLETION_RECEIPT)
    if (
        completion.get("manifest_sha256") != P4C_COMPLETION_MANIFEST_SHA256
        or completion.get("status") != "SEALED_COMPLETE"
        or completion.get("cell_count") != 8
        or completion.get("results_json_parsed") is not False
        or completion.get("samples_parsed") is not False
        or completion.get("outcome_values_consumed") is not False
        or completion.get("p4d_unseal_performed") is not False
    ):
        raise P4DError("P4-C completion receipt does not admit one-shot P4-D unseal")
    closed_cells = completion.get("cells")
    if not isinstance(closed_cells, list) or len(closed_cells) != 8:
        raise P4DError("P4-C completion cell membership differs")
    for index, entry in enumerate(closed_cells):
        metadata = entry.get("result") if isinstance(entry, Mapping) else None
        if (
            entry.get("cell_id") != CELLS[index][0]
            or not isinstance(metadata, Mapping)
            or metadata.get("path") != str(RESULT_PATHS[index])
            or metadata.get("file_sha256") != RESULT_SHA256[index]
        ):
            raise P4DError("P4-C completion/result binding differs")

    located = {
        "b2_analysis": B2_ANALYSIS_PATH,
        "b2_gate_receipt": B2_RECEIPT_PATH,
        "b_family_csv": B_FAMILY_CSV_PATH,
        "b_family_markdown": B_FAMILY_MD_PATH,
        "b_family_receipt": B_FAMILY_RECEIPT_PATH,
    }
    located_hashes = {
        "b2_analysis": B2_ANALYSIS_SHA256,
        "b2_gate_receipt": B2_RECEIPT_SHA256,
        "b_family_csv": B_FAMILY_CSV_SHA256,
        "b_family_markdown": B_FAMILY_MD_SHA256,
        "b_family_receipt": B_FAMILY_RECEIPT_SHA256,
    }
    for key, path in located.items():
        _require_hash(path, located_hashes[key], key)
    return {
        "source_manifest": source_manifest,
        "selector_report": selector_report,
        "panel": panel,
        "completion": completion,
        "located": located,
    }


def _selector_context(report: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, int]]:
    rows = report["width4_primary"]["rows"]
    by_window = {row["window"]: row for row in rows}
    windows = [cell[2] for cell in CELLS[1:]]
    if any(window not in by_window for window in windows):
        raise P4DError("selector report lacks a frozen panel window")
    return (
        {window: float(by_window[window]["score"]) for window in windows},
        {window: int(by_window[window]["primary_rank"]) for window in windows},
    )


def _write_new_text(path: Path, text: str) -> str:
    if path.exists():
        raise FileExistsError("refusing to overwrite P4-D artifact: %s" % path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_new_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _csv_text(cells: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "index", "cell_id", "role", "window", "sample_count", "accuracy_fraction",
        "gain_fraction", "gain_percentage_points", "ci95_lower_fraction",
        "ci95_upper_fraction", "correct_to_wrong", "wrong_to_correct",
        "exact_two_sided_mcnemar_p", "selector_score", "selector_rank",
        "enters_blind_six_enrichment_and_rho",
    )
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in cells:
        ci = row["paired_percentile_95_ci_fraction"]
        transitions = row["transitions_vs_baseline"]
        writer.writerow(
            {
                "index": row["index"],
                "cell_id": row["cell_id"],
                "role": row["role"],
                "window": "" if row["window"] is None else row["window"],
                "sample_count": row["sample_count"],
                "accuracy_fraction": repr(row["accuracy_fraction"]),
                "gain_fraction": repr(row["gain_fraction"]),
                "gain_percentage_points": repr(row["gain_percentage_points"]),
                "ci95_lower_fraction": repr(ci[0]),
                "ci95_upper_fraction": repr(ci[1]),
                "correct_to_wrong": transitions["correct_to_wrong"],
                "wrong_to_correct": transitions["wrong_to_correct"],
                "exact_two_sided_mcnemar_p": repr(row["exact_two_sided_mcnemar_p"]),
                "selector_score": "" if row["selector_score"] is None else repr(row["selector_score"]),
                "selector_rank": "" if row["selector_rank"] is None else row["selector_rank"],
                "enters_blind_six_enrichment_and_rho": str(
                    row["enters_blind_six_enrichment_and_rho"]
                ).lower(),
            }
        )
    return output.getvalue()


def _summary_zh(analysis: Mapping[str, Any]) -> str:
    g = analysis["groups"]["G_enrich"]
    rho = analysis["blind_six_selector_score_vs_gain"]
    labels = analysis["labels"]
    lines = [
        "# LoopScope 第四阶段结果摘要",
        "",
        "冻结 selector 的最终决定仍为 **ABSTAIN**（`selected_window=null`），因此不存在 selected-vs-baseline、selected-vs-15:18、0.30 pp 竞争性或 selected regret 的替代结论；这些字段均为 `NA_ABSTAIN`。",
        "",
        "预注册 blind high3 与 low3 的排序富集检验得到 `G_enrich={:.6f} pp`，95% 配对 bootstrap CI 为 `[{:.6f}, {:.6f}] pp`，结论为 `{}`。".format(
            g["point_percentage_points"],
            g["percentile_95_ci_percentage_points"][0],
            g["percentile_95_ci_percentage_points"][1],
            g["label"],
        ),
        "blind-six selector score 与 gain 的 Spearman `rho={:.6f}`，720 个穷举排列的双侧 `p={:.6f}`。".format(
            rho["rho"], rho["p_value"]
        ),
        "",
        "选择状态标签为 `{}`；科学阶段标签为 `{}`。二者必须分开理解：ABSTAIN 表示冻结规则没有选出可部署窗口，排序富集只评价预注册 high/low 分组信号。".format(
            labels["selection_status_label"], labels["scientific_phase_label"]
        ),
        "",
        "结论范围仅限 **current-checkpoint reproduction** 下的 **same-population outcome-blind transductive** 证据，不支持 unseen-prompt generalization。模板主导风险继续保留；B-2 只支持 hidden/output geometry agreement，不能作为 gain prediction 证据。",
        "",
        "完整八单元准确率、gain、区间、翻转数与 McNemar p 值见 `phase4_outcome_table.csv`。",
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    *,
    output_root: Path,
    expected_commit: str,
    argv: Sequence[str],
    attempt_count: int = 1,
) -> Dict[str, Any]:
    if Path(output_root).resolve() != AUTHORIZED_OUTPUT_ROOT:
        raise P4DError("P4-D output root differs from exact authorization")
    if Path(output_root).exists():
        raise FileExistsError("authorized P4-D write-once root already exists")
    git = git_provenance(expected_commit, remote_required=True)
    frozen = validate_inputs()
    sources = load_strict_jsonl(SOURCE_RECORDS)
    if len(sources) != EXPECTED_COUNT:
        raise P4DError("canonical source does not contain exactly 12,032 records")
    for source in sources:
        validate_source_record(source)
    identities = [source["canonical_identity"] for source in sources]
    categories = [source["category"] for source in sources]
    if semantic_sha256(identities) != frozen["source_manifest"].get("ordered_identity_sha256"):
        raise P4DError("canonical source ordered identity digest differs")
    correctness: Dict[str, Sequence[bool]] = {}
    evaluator_order_hashes = []
    for (cell_id, _role, _window), path in zip(CELLS, RESULT_PATHS):
        values, evaluator_order_hash = load_correctness_cell(
            path, expected_identities=identities, expected_categories=categories
        )
        correctness[cell_id] = values
        evaluator_order_hashes.append(evaluator_order_hash)
    if attempt_count not in (1, 2):
        raise P4DError("P4-D analyzer attempt count must be one or two")
    evaluator_order_sha256 = validate_evaluator_order_hashes(evaluator_order_hashes)
    scores, ranks = _selector_context(frozen["selector_report"])
    analysis = build_statistical_analysis(
        identities=identities,
        categories=categories,
        correctness_by_cell=correctness,
        selector_scores=scores,
        selector_ranks=ranks,
    )
    analysis.update(
        {
            "created_at_utc": utc_now(),
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "planning_thread_id": PLANNING_THREAD_ID,
            "git": git,
            "argv": list(argv),
            "analyzer_invocations": {
                "attempt_count_including_preoutput_failures": attempt_count,
                "successful_scientific_output_count": 1,
                "preoutput_failed_count": attempt_count - 1,
            },
            "implementation_sha256": implementation_hashes(),
            "frozen_bindings": {
                "policy_sha256": POLICY_BYTE_SHA256,
                "control_sha256": CONTROL_BYTE_SHA256,
                "card_sha256": CARD_BYTE_SHA256,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "source_records_sha256": frozen["source_manifest"]["record_file_sha256"],
                "selector_report_sha256": SELECTOR_REPORT_SHA256,
                "selector_freeze_sha256": SELECTOR_FREEZE_SHA256,
                "selector_receipt_sha256": SELECTOR_RECEIPT_SHA256,
                "panel_sha256": PANEL_SHA256,
                "panel_manifest_sha256": PANEL_MANIFEST_SHA256,
                "p4c_completion_receipt_sha256": P4C_COMPLETION_SHA256,
                "result_sha256_in_frozen_order": list(RESULT_SHA256),
                "b2_analysis_sha256": B2_ANALYSIS_SHA256,
                "b2_receipt_sha256": B2_RECEIPT_SHA256,
                "b_family_csv_sha256": B_FAMILY_CSV_SHA256,
                "b_family_markdown_sha256": B_FAMILY_MD_SHA256,
                "b_family_receipt_sha256": B_FAMILY_RECEIPT_SHA256,
            },
            "input_paths": {
                "source_manifest": str(SOURCE_MANIFEST),
                "source_records": str(SOURCE_RECORDS),
                "selector_report": str(SELECTOR_REPORT),
                "selector_freeze": str(SELECTOR_FREEZE),
                "selector_receipt": str(SELECTOR_RECEIPT),
                "panel": str(PANEL_PATH),
                "p4c_completion_receipt": str(COMPLETION_RECEIPT),
                "results_in_frozen_order": [str(path) for path in RESULT_PATHS],
                **{key: str(value) for key, value in frozen["located"].items()},
            },
            "identity_closure": {
                "raw_evaluator_order_sha256": evaluator_order_sha256,
                "raw_evaluator_order_identical_all_eight_cells": True,
                "canonical_source_order_restored_by_unique_identity_join": True,
                "category_exact_all_eight_cells": True,
            },
        }
    )
    # The first external write occurs only after every scientific computation and
    # input/provenance check has succeeded in memory.
    Path(output_root).mkdir(parents=True, exist_ok=False)
    analysis_sha = _write_new_json(Path(output_root) / ANALYSIS_NAME, analysis)
    table_sha = _write_new_text(Path(output_root) / TABLE_NAME, _csv_text(analysis["cells"]))
    summary_sha = _write_new_text(Path(output_root) / SUMMARY_NAME, _summary_zh(analysis))
    return {
        "analysis_file_sha256": analysis_sha,
        "table_file_sha256": table_sha,
        "summary_file_sha256": summary_sha,
        "enrichment_label": analysis["groups"]["G_enrich"]["label"],
    }


def _read_table(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8:
        raise P4DError("outcome table does not contain exactly eight rows")
    return rows


def run_verifier(
    *,
    output_root: Path,
    expected_commit: str,
    argv: Sequence[str],
    repair_count: int = 0,
) -> Dict[str, Any]:
    root = Path(output_root).resolve()
    if root != AUTHORIZED_OUTPUT_ROOT or not root.is_dir():
        raise P4DError("P4-D verifier root differs or is absent")
    for name in (VERIFIER_NAME, PHASE_AUDIT_NAME, GATE_RECEIPT_NAME):
        if (root / name).exists():
            raise FileExistsError("refusing a second P4-D verifier/receipt output")
    git = git_provenance(expected_commit, remote_required=True)
    analysis = load_strict_json(root / ANALYSIS_NAME)
    rows = _read_table(root / TABLE_NAME)
    if analysis.get("schema_version") != "loopscope.phase4.p4d-outcome-analysis.v1":
        raise P4DError("analysis schema differs")
    if [row["cell_id"] for row in rows] != [cell[0] for cell in CELLS]:
        raise P4DError("CSV frozen cell membership/order differs")
    for csv_row, json_row in zip(rows, analysis["cells"]):
        numeric = {
            "accuracy_fraction": json_row["accuracy_fraction"],
            "gain_fraction": json_row["gain_fraction"],
            "exact_two_sided_mcnemar_p": json_row["exact_two_sided_mcnemar_p"],
        }
        for key, expected in numeric.items():
            if float(csv_row[key]) != float(expected):
                raise P4DError("CSV/analysis numeric mismatch: %s" % key)
    g = analysis["groups"]["G_enrich"]
    high = analysis["groups"]["blind_high3"]["mean_gain_fraction"]
    low = analysis["groups"]["blind_low3"]["mean_gain_fraction"]
    if not math.isclose(g["point_fraction"], high - low, rel_tol=0.0, abs_tol=1e-15):
        raise P4DError("analysis G_enrich formula does not recompute")
    expected_science = scientific_phase_label(g["label"])
    if (
        analysis["labels"]["selection_status_label"] != SELECTION_STATUS_LABEL
        or analysis["labels"]["scientific_phase_label"] != expected_science
        or analysis["abstain"]["selected_window"] is not None
        or any(
            analysis["abstain"][key].get("status") != "NA_ABSTAIN"
            for key in (
                "selected_vs_baseline",
                "selected_vs_15_18",
                "selected_competitiveness_margin_0_30pp",
                "selected_panel_regret",
            )
        )
    ):
        raise P4DError("ABSTAIN or phase-label mapping differs")
    rho = analysis["blind_six_selector_score_vs_gain"]
    if rho.get("permutation_count") != 720 or rho.get("fixed_15_18_excluded") is not True:
        raise P4DError("blind-six permutation contract differs")
    if analysis["population"].get("matrix_shape") != [EXPECTED_COUNT, 8]:
        raise P4DError("analysis does not close 12,032 x 8")
    if any(analysis["persistence_safety"].values()):
        raise P4DError("analysis persistence safety flag differs")
    summary = (root / SUMMARY_NAME).read_text(encoding="utf-8")
    for phrase in (
        "same-population outcome-blind transductive",
        "current-checkpoint reproduction",
        "unseen-prompt generalization",
        "hidden/output geometry agreement",
        "gain prediction",
        "ABSTAIN",
    ):
        if phrase not in summary:
            raise P4DError("Chinese summary lacks required limitation wording: %s" % phrase)

    output_hashes = {
        ANALYSIS_NAME: file_sha256(root / ANALYSIS_NAME),
        TABLE_NAME: file_sha256(root / TABLE_NAME),
        SUMMARY_NAME: file_sha256(root / SUMMARY_NAME),
    }
    verifier = {
        "schema_version": "loopscope.phase4.p4d-outcome-verifier-receipt.v1",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "git": git,
        "argv": list(argv),
        "verifier_invocation_count": 1,
        "input_output_sha256": dict(output_hashes),
        "checks": {
            "exact_eight_cell_membership_and_order": True,
            "record_cell_matrix_12032_x_8": True,
            "identity_order_category_closure": True,
            "joint_category_stratified_bootstrap_contract": True,
            "gain_and_G_enrich_recomputed": True,
            "exact_720_permutation_rho": True,
            "mcnemar_fields_complete": True,
            "abstain_na_and_phase_labels": True,
            "forbidden_raw_generated_fields_not_persisted": True,
            "claim_wording_complete": True,
        },
        "status": "VERIFIED_READY_FOR_PLANNING_AUDIT",
    }
    verifier_sha = _write_new_json(root / VERIFIER_NAME, verifier)
    output_hashes[VERIFIER_NAME] = verifier_sha
    phase_audit = {
        "schema_version": "loopscope.phase4.phase-end-audit.v1",
        "artifact_role": "concise_phase4_terminal_evidence_for_planning_audit",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "accepted_gate_evidence_reused": {
            "p4b_selector_report_sha256": SELECTOR_REPORT_SHA256,
            "p4b_panel_sha256": PANEL_SHA256,
            "b2_analysis_sha256": B2_ANALYSIS_SHA256,
            "b_family_receipt_sha256": B_FAMILY_RECEIPT_SHA256,
            "p4c_completion_receipt_sha256": P4C_COMPLETION_SHA256,
        },
        "final_checks": {
            "frozen_configuration_exact": True,
            "shared_population_12032_exact": True,
            "terminal_eight_cell_complete": True,
            "scientific_outputs_hash_closed": True,
            "claim_scope_same_population_outcome_blind_transductive": True,
            "claim_scope_current_checkpoint_reproduction": True,
            "unseen_prompt_generalization_not_claimed": True,
            "template_dominance_risk_retained": True,
            "b2_not_used_as_gain_prediction": True,
        },
        "selection_status_label": analysis["labels"]["selection_status_label"],
        "scientific_phase_label": analysis["labels"]["scientific_phase_label"],
        "G_enrich": g,
        "status": "PHASE4_TERMINAL_EVIDENCE_READY_FOR_PLANNING_AUDIT",
        "requested_decision": "PASS",
    }
    phase_audit_sha = _write_new_json(root / PHASE_AUDIT_NAME, phase_audit)
    output_hashes[PHASE_AUDIT_NAME] = phase_audit_sha
    receipt = {
        "schema_version": "loopscope.phase4.p4d-gate-receipt.v1",
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "created_at_utc": utc_now(),
        "git": git,
        "bindings": {
            "policy_sha256": POLICY_BYTE_SHA256,
            "control_sha256": CONTROL_BYTE_SHA256,
            "card_sha256": CARD_BYTE_SHA256,
            "code_commit": expected_commit,
            "implementation_sha256": implementation_hashes(),
        },
        "input_paths": analysis["input_paths"],
        "input_sha256": analysis["frozen_bindings"],
        "output_sha256_excluding_self": dict(output_hashes),
        "invocations": {
            "analyzer": {
                **analysis["analyzer_invocations"],
                "successful_argv": analysis["argv"],
                "preoutput_failure": (
                    None
                    if analysis["analyzer_invocations"]["preoutput_failed_count"] == 0
                    else {
                        "error": "outcome cell canonical identity order is reordered",
                        "output_root_created": False,
                        "scientific_output_created": False,
                        "repair": "canonical identity join plus exact cross-cell evaluator-order hash",
                    }
                ),
            },
            "verifier": {"count": 1, "argv": list(argv)},
        },
        "closure": {
            "record_count": EXPECTED_COUNT,
            "cell_count": 8,
            "matrix_shape": [EXPECTED_COUNT, 8],
            "identity_order_category_exact": True,
            "bootstrap": analysis["bootstrap"],
            "selection_status_label": analysis["labels"]["selection_status_label"],
            "scientific_phase_label": analysis["labels"]["scientific_phase_label"],
        },
        "repair_history": {
            "executor_owned_repair_count": int(repair_count),
            "audit_returned_repair_count": 0,
            "science_changed": False,
            "repairs": [
                {
                    "index": 1,
                    "stage": "pre_unseal",
                    "cause": "frozen outcome panel fields are nested under the manifest panel key",
                    "change": "validate the exact nested panel payload",
                    "scientific_output_created_before_repair": False,
                },
                {
                    "index": 2,
                    "stage": "pre_output_after_unseal_attempt_1",
                    "cause": "lm-eval raw samples are namespace-grouped rather than in P4-B source order",
                    "change": "unique identity join restores canonical order and exact raw order hash must match across all eight cells",
                    "scientific_output_created_before_repair": False,
                },
            ][: int(repair_count)],
        },
        "safety": {
            "gpu_used": False,
            "slurm_used": False,
            "inference_or_generation_run": False,
            "p4c_cell_rerun_or_replacement": False,
            "selector_rerun_or_retune": False,
            "panel_metric_bootstrap_threshold_changed": False,
            "variable_width_outcome_accessed": False,
            "raw_question_gold_response_generated_content_persisted": False,
            "row_level_correctness_matrix_persisted": False,
            "notion_or_external_report_written": False,
            "prior_gate_artifact_modified": False,
        },
        "status": "GATE_P4D_READY_FOR_PLANNING_AUDIT",
        "requested_decision": "PASS",
    }
    gate_receipt_sha = _write_new_json(root / GATE_RECEIPT_NAME, receipt)
    return {
        "verifier_file_sha256": verifier_sha,
        "phase_end_audit_file_sha256": phase_audit_sha,
        "gate_receipt_file_sha256": gate_receipt_sha,
        "G_enrich": g,
        "blind_six": rho,
        "labels": analysis["labels"],
    }


__all__ = [
    "AUTHORIZED_OUTPUT_ROOT",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "P4DError",
    "build_statistical_analysis",
    "exact_mcnemar_p",
    "exact_spearman_permutation",
    "joint_category_stratified_bootstrap",
    "linear_percentile",
    "run_analysis",
    "run_verifier",
    "scientific_phase_label",
    "spearman_rho",
    "validate_evaluator_order_hashes",
    "validate_panel_manifest",
]
