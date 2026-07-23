"""Gate H absolute-rate ranking over frozen Gate G admission.

This module deliberately imports the narrow Gate G scalar/bootstrap primitives.
It does not import model, dataset, evaluator, accelerator, or result machinery.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from tflt.loopscope import phase5_relative_biphasic as v1


METHOD = "RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE"
EXPECTED_RECORD_COUNT = 1531
EXPECTED_SUBJECT_COUNT = 57
EXPECTED_REPLICATES = 2000
TIE_REL_TOL = 1e-12
TIE_ABS_TOL = 1e-12
EPSILON = 1e-12
V1_NUMERIC_ATOL = 1e-10

QWEN17_JSON = "qwen17_v2_absolute_rate_candidates.json"
QWEN17_CSV = "qwen17_v2_absolute_rate_candidates.csv"
QWEN4_JSON = "qwen4_v2_absolute_rate_candidates.json"
QWEN4_CSV = "qwen4_v2_absolute_rate_candidates.csv"
CROSS_MODEL_SUMMARY_JSON = "relative_biphasic_v2_cross_model_summary.json"
SUMMARY_ZH = "relative_biphasic_v2_summary_zh.md"
VERIFIER_RECEIPT_JSON = "relative_biphasic_v2_verifier_receipt.json"
MANIFEST_RECEIPT_JSON = "relative_biphasic_v2_manifest_receipt.json"
PRIMARY_OUTPUTS = (
    QWEN17_JSON,
    QWEN17_CSV,
    QWEN4_JSON,
    QWEN4_CSV,
    CROSS_MODEL_SUMMARY_JSON,
    SUMMARY_ZH,
)
ALLOWED_INFORMATION_BARRIER_FLAGS = {
    "outcome_values_consumed",
}
ALLOWED_SCIENTIFIC_KEYS_WITH_TOKEN_SUBSTRINGS = {
    "ratestable",
}


class AbsoluteRateError(ValueError):
    """Fail-closed Gate H contract error."""


def validate_card(card: Mapping[str, Any]) -> None:
    if card.get("card") != METHOD:
        raise AbsoluteRateError("card identity differs")
    authorization = card.get("authorization")
    if not isinstance(authorization, Mapping):
        raise AbsoluteRateError("authorization is missing")
    expected_authority = {
        "gate": "H",
        "planning_thread": "019f8604-4717-7be2-8bf8-9d4a26a3d7f7",
        "executor_thread": "019f8fe5-0880-7173-9650-7c4c76ee475d",
        "authorized_local_base_commit": "118de08fe45844edf406e811c2ece91886d3f6ed",
        "runtime_reference_commit": "f33a37a4e5c0cfc461184a3471277897c77cf83b",
        "protected_reference_ancestor": "4f59bd93eca4da3cbf458a93508f91c5b23912bc",
    }
    for key, expected in expected_authority.items():
        if authorization.get(key) != expected:
            raise AbsoluteRateError("authorization %s differs" % key)
    for model_key, expected_count in (("qwen17", 26), ("qwen4", 42)):
        observed = card.get("models", {}).get(model_key)
        if not isinstance(observed, Mapping):
            raise AbsoluteRateError("%s model card is missing" % model_key)
        if observed.get("candidate_count") != expected_count:
            raise AbsoluteRateError("%s candidate count differs" % model_key)
        spec = v1.MODEL_SPECS[model_key]
        bootstrap = observed.get("bootstrap", {})
        if bootstrap.get("replicates") != EXPECTED_REPLICATES:
            raise AbsoluteRateError("%s bootstrap count differs" % model_key)
        if bootstrap.get("seed") != spec["seed"]:
            raise AbsoluteRateError("%s bootstrap seed differs" % model_key)
    selection = card.get("selection", {})
    if selection.get("frequency_threshold") != 0.8:
        raise AbsoluteRateError("selection frequency threshold differs")
    if selection.get("frequency_denominator") != EXPECTED_REPLICATES:
        raise AbsoluteRateError("selection frequency denominator differs")


def rate_score(g_h: Any, g_k: Any) -> float | None:
    h = _finite(g_h, "G_H")
    k = _finite(g_k, "G_K")
    if h <= 0.0 or k <= 0.0:
        return None
    score = math.sqrt(h * k)
    return _finite(score, "S_RATE")


def compute_rate_statistics(
    g_h: Any,
    g_k: Any,
    boot_h: Sequence[Any],
    boot_k: Sequence[Any],
) -> Dict[str, Any]:
    point_h = _finite(g_h, "G_H")
    point_k = _finite(g_k, "G_K")
    values_h = [_finite(value, "bootstrap G_H") for value in boot_h]
    values_k = [_finite(value, "bootstrap G_K") for value in boot_k]
    if len(values_h) != len(values_k) or len(values_h) < 2:
        raise AbsoluteRateError("rate bootstrap dimensions differ")
    se_h = _sample_standard_deviation(values_h)
    se_k = _sample_standard_deviation(values_k)
    max_t = [
        max(
            (point_h - value_h) / max(se_h, EPSILON),
            (point_k - value_k) / max(se_k, EPSILON),
        )
        for value_h, value_k in zip(values_h, values_k)
    ]
    c95 = max(0.0, _linear_quantile(max_t, 0.95))
    lcb_h = point_h - c95 * max(se_h, EPSILON)
    lcb_k = point_k - c95 * max(se_k, EPSILON)
    for label, value in (
        ("rate_SE_H", se_h),
        ("rate_SE_K", se_k),
        ("rate_c95", c95),
        ("rate_LCB_H", lcb_h),
        ("rate_LCB_K", lcb_k),
    ):
        _finite(value, label)
    score = rate_score(point_h, point_k)
    positive = score is not None
    return {
        "rate_SE_H": se_h,
        "rate_SE_K": se_k,
        "rate_c95": c95,
        "rate_LCB_H": lcb_h,
        "rate_LCB_K": lcb_k,
        "RateStable": bool(positive and lcb_h > 0.0 and lcb_k > 0.0),
        "S_RATE": score,
        "balance_ratio": (
            min(point_h, point_k) / max(point_h, point_k) if positive else None
        ),
        "_bootstrap_rate_H": values_h,
        "_bootstrap_rate_K": values_k,
    }


def select_rate_decision(
    rows: Sequence[MutableMapping[str, Any]],
    bootstrap_rates: Mapping[Tuple[int, int], Mapping[str, Sequence[Any]]],
    replicates: int,
) -> Dict[str, Any]:
    count = _strict_int(replicates, "replicates")
    candidates = [
        row for row in rows if bool(row["NewEligible"]) and bool(row["RateStable"])
    ]
    for row in rows:
        row["ranking_candidate"] = row in candidates
        row["display_rank"] = None
        row["point_top_tie"] = False
        row["selected"] = False
        row["model_rate_selection_frequency"] = None
    ranking = sorted(
        candidates,
        key=lambda row: (-float(row["S_RATE"]), int(row["width"]), int(row["start"])),
    )
    for display_rank, row in enumerate(ranking, start=1):
        row["display_rank"] = display_rank
    if not ranking:
        return {
            "decision": "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
            "point_top_key": None,
            "point_top_tied_keys": [],
            "selection_frequency": None,
            "selected_key": None,
            "replicate_winner_count": 0,
            "replicate_no_winner_count": count,
        }
    point_max = float(ranking[0]["S_RATE"])
    point_winners = [
        row
        for row in ranking
        if math.isclose(
            float(row["S_RATE"]),
            point_max,
            rel_tol=TIE_REL_TOL,
            abs_tol=TIE_ABS_TOL,
        )
    ]
    for row in point_winners:
        row["point_top_tie"] = len(point_winners) > 1
    tied_keys = [[int(row["width"]), int(row["start"])] for row in point_winners]
    if len(point_winners) != 1:
        return {
            "decision": "ABSTAIN_NO_UNIQUE_TOP1",
            "point_top_key": None,
            "point_top_tied_keys": tied_keys,
            "selection_frequency": None,
            "selected_key": None,
            "replicate_winner_count": 0,
            "replicate_no_winner_count": None,
        }
    point_top = point_winners[0]
    point_top_key = (int(point_top["width"]), int(point_top["start"]))
    wins = 0
    no_winner = 0
    for replicate in range(count):
        replicate_candidates: List[Tuple[float, Tuple[int, int]]] = []
        for row in ranking:
            key = (int(row["width"]), int(row["start"]))
            rates = bootstrap_rates[key]
            value_h = _finite(rates["H"][replicate], "replicate G_H")
            value_k = _finite(rates["K"][replicate], "replicate G_K")
            score = rate_score(value_h, value_k)
            if score is not None:
                replicate_candidates.append((score, key))
        if not replicate_candidates:
            no_winner += 1
            continue
        replicate_max = max(value[0] for value in replicate_candidates)
        replicate_winners = [
            key
            for score, key in replicate_candidates
            if math.isclose(
                score,
                replicate_max,
                rel_tol=TIE_REL_TOL,
                abs_tol=TIE_ABS_TOL,
            )
        ]
        if len(replicate_winners) != 1:
            no_winner += 1
            continue
        if replicate_winners[0] == point_top_key:
            wins += 1
    frequency = wins / count
    point_top["model_rate_selection_frequency"] = frequency
    if frequency < 0.80:
        decision = "ABSTAIN_RATE_RANK_UNSTABLE"
        selected_key = None
    else:
        decision = "SELECTED_WINDOW"
        selected_key = point_top_key
        point_top["selected"] = True
    return {
        "decision": decision,
        "point_top_key": list(point_top_key),
        "point_top_tied_keys": [],
        "selection_frequency": frequency,
        "selected_key": list(selected_key) if selected_key else None,
        "replicate_winner_count": wins,
        "replicate_no_winner_count": no_winner,
    }


def compare_v1_rows(
    observed_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = V1_NUMERIC_ATOL,
) -> Dict[str, Any]:
    observed_by_key = {
        (int(row["width"]), int(row["start"])): row for row in observed_rows
    }
    expected_by_key = {
        (int(row["width"]), int(row["start"])): row for row in expected_rows
    }
    if set(observed_by_key) != set(expected_by_key):
        raise AbsoluteRateError("BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH: candidates")
    maximum = 0.0
    numeric_count = 0

    def compare(observed: Any, expected: Any, path: str) -> None:
        nonlocal maximum, numeric_count
        if isinstance(expected, bool) or expected is None or isinstance(expected, str):
            if type(observed) is not type(expected) or observed != expected:
                raise AbsoluteRateError(
                    "BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH: %s" % path
                )
            return
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                raise AbsoluteRateError(
                    "BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH: %s" % path
                )
            difference = abs(_finite(observed, path) - _finite(expected, path))
            maximum = max(maximum, difference)
            numeric_count += 1
            if difference > tolerance:
                raise AbsoluteRateError(
                    "BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH: %s %.17g"
                    % (path, difference)
                )
            return
        if isinstance(expected, Mapping):
            if not isinstance(observed, Mapping) or set(observed) != set(expected):
                raise AbsoluteRateError(
                    "BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH: %s keys" % path
                )
            for key in expected:
                compare(observed[key], expected[key], "%s.%s" % (path, key))
            return
        if isinstance(expected, list):
            if not isinstance(observed, list) or len(observed) != len(expected):
                raise AbsoluteRateError(
                    "BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH: %s length" % path
                )
            for index, (left, right) in enumerate(zip(observed, expected)):
                compare(left, right, "%s.%d" % (path, index))
            return
        raise AbsoluteRateError("unsupported V1 compatibility type at %s" % path)

    for key in sorted(expected_by_key):
        compare(observed_by_key[key], expected_by_key[key], "%s" % (key,))
    return {
        "status": "PASS",
        "candidate_count": len(expected_by_key),
        "numeric_value_count": numeric_count,
        "absolute_tolerance": tolerance,
        "max_absolute_difference": maximum,
        "compatibility_digest": v1.semantic_sha256(
            {
                "candidate_keys": [list(key) for key in sorted(expected_by_key)],
                "numeric_value_count": numeric_count,
                "max_absolute_difference": maximum,
                "absolute_tolerance": tolerance,
            }
        ),
    }


def analyze_v2_model(
    samples: Sequence[Mapping[str, Any]],
    model_key: str,
    canonical_v1_payload: Mapping[str, Any],
    *,
    replicates: int = EXPECTED_REPLICATES,
    seed: int | None = None,
    enforce_population: bool = True,
) -> Dict[str, Any]:
    spec = v1.MODEL_SPECS[model_key]
    selected_seed = spec["seed"] if seed is None else _strict_int(seed, "seed")
    selected_replicates = _strict_int(replicates, "replicates")
    if enforce_population and (
        selected_replicates != EXPECTED_REPLICATES or selected_seed != spec["seed"]
    ):
        raise AbsoluteRateError("%s formal bootstrap contract differs" % model_key)
    v1_analysis = v1.analyze_model(
        samples,
        model_key,
        replicates=selected_replicates,
        seed=selected_seed,
        enforce_population=enforce_population,
    )
    canonical_rows = canonical_v1_payload.get("rows")
    if not isinstance(canonical_rows, list):
        raise AbsoluteRateError("canonical V1 candidate payload lacks rows")
    compatibility = compare_v1_rows(v1_analysis["rows"], canonical_rows)
    expected_count = spec["candidate_count"]
    if enforce_population and compatibility["candidate_count"] != expected_count:
        raise AbsoluteRateError(
            "BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH: candidate count"
        )

    normalized = v1._normalize_preprojected(samples, model_key, enforce_population)
    bootstrap = v1._bootstrap_boundary_means(
        normalized,
        spec["layers"],
        replicates=selected_replicates,
        seed=selected_seed,
        digest_style=spec["legacy_style"],
    )
    if bootstrap["draw_index_sha256"] != v1_analysis["bootstrap"]["draw_index_sha256"]:
        raise AbsoluteRateError("bootstrap draw stream differs from V1")
    rows: List[Dict[str, Any]] = copy.deepcopy(v1_analysis["rows"])
    bootstrap_rates: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for row in rows:
        width = int(row["width"])
        start = int(row["start"])
        boot_h = [
            (bootstrap["H"][start][index] - bootstrap["H"][start + width][index])
            / width
            for index in range(selected_replicates)
        ]
        boot_k = [
            (bootstrap["D"][start + width][index] - bootstrap["D"][start][index])
            / width
            for index in range(selected_replicates)
        ]
        stats = compute_rate_statistics(row["G_H"], row["G_K"], boot_h, boot_k)
        row["v1_selected"] = row.pop("selected")
        row["v1_model_window_selection_frequency"] = row.pop(
            "model_window_selection_frequency"
        )
        row["end"] = start + width - 1
        row["boundary_entry"] = "B_%d" % start
        row["boundary_exit"] = "B_%d" % (start + width)
        row["v1_failure_reasons"] = list(row["new_eligibility_failures"])
        best = row["best_shape_pair"]
        full_best = (
            None
            if best is None
            else next(
                pair
                for pair in row["shape_pairs"]
                if pair.get("q") == best["q"] and pair.get("tau") == best["tau"]
            )
        )
        row["best_q"] = None if best is None else best["q"]
        row["best_tau"] = None if best is None else best["tau"]
        row["boundary_supported"] = (
            None if full_best is None else full_best["boundary_supported"]
        )
        row["external_side"] = None if full_best is None else full_best["external_side"]
        row["shape_margin_count"] = (
            0 if full_best is None else len(full_best.get("margins", {}))
        )
        for field in (
            "S_SHAPE",
            "T_seg_H",
            "T_seg_K",
            "T_core_H",
            "T_core_K",
            "tau_pair_frequency",
            "pi",
        ):
            row[field] = None if full_best is None else full_best.get(field)
        row.update({key: value for key, value in stats.items() if not key.startswith("_")})
        row["total_entropy_change"] = width * float(row["G_H"])
        row["total_kl_change"] = width * float(row["G_K"])
        bootstrap_rates[(width, start)] = {
            "H": stats["_bootstrap_rate_H"],
            "K": stats["_bootstrap_rate_K"],
        }
    decision = select_rate_decision(rows, bootstrap_rates, selected_replicates)
    for row in rows:
        failures: List[str] = []
        if not row["NewEligible"]:
            failures.append("NOT_NEW_ELIGIBLE")
        if not row["RateStable"]:
            failures.append("RATE_NOT_STABLE")
        if row["ranking_candidate"] and not row["selected"]:
            if row["point_top_tie"]:
                failures.append("POINT_TOP_TIE")
            elif row["display_rank"] != 1:
                failures.append("NOT_POINT_TOP")
            elif decision["decision"] == "ABSTAIN_RATE_RANK_UNSTABLE":
                failures.append("RATE_RANK_UNSTABLE")
        row["v2_decision"] = decision["decision"]
        row["v2_failure_reasons"] = failures
    summary = _model_summary(model_key, rows, decision, compatibility)
    return {
        "model_key": model_key,
        "model": spec["model"],
        "record_count": v1_analysis["record_count"],
        "subject_count": v1_analysis["subject_count"],
        "boundary_count": v1_analysis["boundary_count"],
        "candidate_count": len(rows),
        "candidate_counts_by_width": dict(v1_analysis["candidate_counts_by_width"]),
        "bootstrap": {
            **dict(v1_analysis["bootstrap"]),
            "rate_draw_index_sha256": bootstrap["draw_index_sha256"],
        },
        "v1_compatibility": compatibility,
        "rows": rows,
        "summary": summary,
    }


def build_primary_payloads(
    qwen17: Mapping[str, Any],
    qwen4: Mapping[str, Any],
    *,
    card_sha256: str,
    input_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    common = {
        "method": METHOD,
        "card_sha256": card_sha256,
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "outcome_values_consumed": False,
        "model_or_dataset_loaded": False,
        "forward": False,
        "gpu_cuda_slurm": False,
    }
    q17_payload = _candidate_payload(qwen17, common)
    q4_payload = _candidate_payload(qwen4, common)
    cross = {
        "schema_version": "loopscope.relative-biphasic-v2-cross-model-summary.v1",
        **common,
        "qwen17": copy.deepcopy(qwen17["summary"]),
        "qwen4": copy.deepcopy(qwen4["summary"]),
        "cross_model_comparability_boundary": (
            "S_RATE is model-local; raw scores and bootstrap draws are not shared "
            "or numerically comparable across models."
        ),
        "claim_boundary": (
            "Retrospective dual-model method development only; selected is not a "
            "positive-benefit or prospective-validation conclusion."
        ),
    }
    payloads: Dict[str, Any] = {
        QWEN17_JSON: q17_payload,
        QWEN17_CSV: render_candidates_csv(q17_payload["rows"]),
        QWEN4_JSON: q4_payload,
        QWEN4_CSV: render_candidates_csv(q4_payload["rows"]),
        CROSS_MODEL_SUMMARY_JSON: cross,
        SUMMARY_ZH: render_summary_zh(cross),
    }
    for payload in payloads.values():
        if isinstance(payload, MutableMapping):
            payload["manifest_sha256"] = v1.semantic_sha256(payload)
            assert_no_forbidden_output_keys(payload)
    return payloads


def write_primary_outputs(output_root: Path, payloads: Mapping[str, Any]) -> Dict[str, str]:
    root = Path(output_root)
    if not root.is_dir():
        raise AbsoluteRateError("Gate H run root must already exist")
    hashes: Dict[str, str] = {}
    for name in PRIMARY_OUTPUTS:
        value = payloads[name]
        if isinstance(value, str):
            hashes[name] = _write_new_text(root / name, value)
        else:
            hashes[name] = _write_new_json(root / name, value)
    return hashes


def verify_primary_outputs(
    output_root: Path,
    recomputed_payloads: Mapping[str, Any],
    *,
    input_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    root = Path(output_root)
    checks: Dict[str, bool] = {}
    observed_hashes: Dict[str, str] = {}
    for name in PRIMARY_OUTPUTS:
        path = root / name
        if not path.is_file():
            raise AbsoluteRateError("required primary artifact missing: %s" % name)
        expected = recomputed_payloads[name]
        if isinstance(expected, str):
            observed = path.read_text(encoding="utf-8")
            checks["byte_exact_%s" % name] = observed == expected
        else:
            observed = v1.load_strict_json(path)
            assert_no_forbidden_output_keys(observed)
            checks["byte_exact_%s" % name] = path.read_bytes() == _json_bytes(expected)
        observed_hashes[name] = v1.file_sha256(path)
    q17 = v1.load_strict_json(root / QWEN17_JSON)
    q4 = v1.load_strict_json(root / QWEN4_JSON)
    cross = v1.load_strict_json(root / CROSS_MODEL_SUMMARY_JSON)
    checks.update(
        {
            "population_and_candidate_closure": (
                q17["record_count"] == q4["record_count"] == EXPECTED_RECORD_COUNT
                and q17["subject_count"] == q4["subject_count"] == EXPECTED_SUBJECT_COUNT
                and q17["candidate_count"] == len(q17["rows"]) == 26
                and q4["candidate_count"] == len(q4["rows"]) == 42
            ),
            "v1_backward_compatibility": (
                q17["v1_compatibility"]["status"] == "PASS"
                and q4["v1_compatibility"]["status"] == "PASS"
                and q17["v1_compatibility"]["max_absolute_difference"] <= V1_NUMERIC_ATOL
                and q4["v1_compatibility"]["max_absolute_difference"] <= V1_NUMERIC_ATOL
            ),
            "rate_formulas_and_bounds": _rows_close(q17["rows"])
            and _rows_close(q4["rows"]),
            "ranking_frequency_and_decisions": _summary_closes(
                q17["rows"], cross["qwen17"]
            )
            and _summary_closes(q4["rows"], cross["qwen4"]),
            "information_boundary_preserved": (
                cross["outcome_values_consumed"] is False
                and cross["model_or_dataset_loaded"] is False
                and cross["forward"] is False
                and cross["gpu_cuda_slurm"] is False
            ),
        }
    )
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise AbsoluteRateError("BLOCK_VERIFIER_MISMATCH: %s" % failed)
    verifier = {
        "schema_version": "loopscope.relative-biphasic-v2-verifier-receipt.v1",
        "result": "PASS",
        "verification_mode": "fresh_process_raw_trajectory_recomputation_and_byte_compare",
        "checks": checks,
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "verified_primary_sha256": observed_hashes,
        "outcome_values_consumed": False,
        "model_or_dataset_loaded": False,
        "forward": False,
        "gpu_cuda_slurm": False,
    }
    verifier["manifest_sha256"] = v1.semantic_sha256(verifier)
    assert_no_forbidden_output_keys(verifier)
    verifier_file_sha = _write_new_json(root / VERIFIER_RECEIPT_JSON, verifier)
    manifest = {
        "schema_version": "loopscope.relative-biphasic-v2-manifest-receipt.v1",
        "result": "READY_FOR_PLANNING_AUDIT",
        "method": METHOD,
        "formal_analyzer_count": 1,
        "fresh_process_verifier_count": 1,
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "artifact_sha256": {
            **observed_hashes,
            VERIFIER_RECEIPT_JSON: verifier_file_sha,
        },
        "verifier_manifest_sha256": verifier["manifest_sha256"],
        "candidate_counts": {"qwen17": 26, "qwen4": 42},
        "outcome_values_consumed": False,
        "model_or_dataset_loaded": False,
        "forward": False,
        "gpu_cuda_slurm": False,
    }
    manifest["manifest_sha256"] = v1.semantic_sha256(manifest)
    assert_no_forbidden_output_keys(manifest)
    _write_new_json(root / MANIFEST_RECEIPT_JSON, manifest)
    return manifest


def render_candidates_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "model",
        "width",
        "start",
        "end",
        "window",
        "boundary_entry",
        "boundary_exit",
        "G_H",
        "G_K",
        "O_H",
        "O_K",
        "F_H",
        "F_K",
        "z_OH",
        "z_OK",
        "z_FH",
        "z_FK",
        "S_REL",
        "legacy_strict",
        "Scorable",
        "NetPositive",
        "SoftRelativeStable",
        "best_q",
        "best_tau",
        "boundary_supported",
        "external_side",
        "best_shape_pair",
        "shape_pairs",
        "shape_margin_count",
        "S_SHAPE",
        "T_seg_H",
        "T_seg_K",
        "T_core_H",
        "T_core_K",
        "tau_pair_frequency",
        "pi",
        "BiphasicStable",
        "v1_failure_reasons",
        "NewEligible",
        "rate_SE_H",
        "rate_SE_K",
        "rate_c95",
        "rate_LCB_H",
        "rate_LCB_K",
        "RateStable",
        "S_RATE",
        "balance_ratio",
        "total_entropy_change",
        "total_kl_change",
        "ranking_candidate",
        "display_rank",
        "point_top_tie",
        "selected",
        "model_rate_selection_frequency",
        "v2_decision",
        "v2_failure_reasons",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return output.getvalue()


def render_summary_zh(summary: Mapping[str, Any]) -> str:
    def section(key: str, title: str) -> List[str]:
        item = summary[key]
        lines = [
            "## %s" % title,
            "",
            "- 候选数：%d；NewEligible / RateStable / ranking candidate：%d / %d / %d。"
            % (
                item["candidate_count"],
                item["new_eligible_count"],
                item["rate_stable_count"],
                item["ranking_candidate_count"],
            ),
            "- V1 backward compatibility：`%s`；最大绝对差：`%.3g`。"
            % (
                item["v1_compatibility"]["status"],
                item["v1_compatibility"]["max_absolute_difference"],
            ),
            "- 最终决策：`%s`。" % item["decision"],
        ]
        if item["point_top"] is None:
            lines.append("- rank-1：无。")
        else:
            lines.append(
                "- rank-1：`%s`，S_RATE=`%.12g`，exact-window bootstrap frequency=%s。"
                % (
                    item["point_top"]["window"],
                    item["point_top"]["S_RATE"],
                    (
                        "不适用"
                        if item["selection_frequency"] is None
                        else "%.4f" % item["selection_frequency"]
                    ),
                )
            )
        if item["selected_window"] is None:
            lines.append("- selected：无（ABSTAIN）。")
        else:
            lines.append(
                "- selected：`%s`；这只是本冻结合同内的回顾性稳定发布。"
                % item["selected_window"]["window"]
            )
        lines.extend(
            [
                "- 本 Gate 未读取 outcome，因此不作正收益判断。",
                "",
            ]
        )
        return lines

    lines = [
        "# LoopScope 第五阶段 Gate H：V2 绝对变化率双模型离线重选窗",
        "",
        "本次完整复现 V1 admission，只在 `NewEligible ∧ RateStable` 内按中心窗口自身的"
        " `S_RATE=sqrt(G_H*G_K)` 排名。两个模型 cell 都参与了规则形成，因此结果只属于"
        "回顾性方法开发；不构成前瞻验证或通用自动选窗器证据。",
        "",
        "概念边界：`NewEligible` 是 V1 admission；`RateStable` 是窗口内 H/K 同时下界；"
        "`ranking candidate` 是两者交集；`rank-1` 是点分数最高；`selected` 还需唯一 top"
        " 且 bootstrap frequency≥0.80；正收益需要 outcome，本 Gate 完全不读取。",
        "",
    ]
    lines.extend(section("qwen17", "Qwen3-1.7B-Base × MMLU 5-shot"))
    lines.extend(section("qwen4", "Qwen3-4B-Base × MMLU 5-shot"))
    lines.extend(
        [
            "## 跨模型边界",
            "",
            "- 两个模型分别使用冻结的 seed 与 model-local bootstrap draw；原始 `S_RATE`"
            " 不能跨模型直接比较。",
            "- `selected` 只表示 Gate H 合同内部的唯一稳定窗口，不表示 accuracy、gain"
            " 或 positive benefit。",
            "",
        ]
    )
    return "\n".join(lines)


def assert_no_forbidden_output_keys(
    value: Any, path: Tuple[str, ...] = ()
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            text = str(key)
            lowered = text.lower()
            matched = [
                token
                for token in v1.FORBIDDEN_OUTPUT_KEY_TOKENS
                if token in lowered
            ]
            if (
                matched
                and lowered not in ALLOWED_INFORMATION_BARRIER_FLAGS
                and lowered not in ALLOWED_SCIENTIFIC_KEYS_WITH_TOKEN_SUBSTRINGS
            ):
                raise AbsoluteRateError(
                    "BLOCK_OUTPUT_INFORMATION_BARRIER_VIOLATION: %s"
                    % ".".join(path + (text,))
                )
            if lowered in ALLOWED_INFORMATION_BARRIER_FLAGS and nested is not False:
                raise AbsoluteRateError(
                    "BLOCK_OUTPUT_INFORMATION_BARRIER_VIOLATION: %s must be false"
                    % ".".join(path + (text,))
                )
            assert_no_forbidden_output_keys(nested, path + (text,))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_no_forbidden_output_keys(nested, path + (str(index),))


def _candidate_payload(
    analysis: Mapping[str, Any], common: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "schema_version": "loopscope.relative-biphasic-v2-candidates.v1",
        **dict(common),
        "model_key": analysis["model_key"],
        "model": analysis["model"],
        "record_count": analysis["record_count"],
        "subject_count": analysis["subject_count"],
        "boundary_count": analysis["boundary_count"],
        "candidate_count": analysis["candidate_count"],
        "bootstrap": copy.deepcopy(analysis["bootstrap"]),
        "v1_compatibility": copy.deepcopy(analysis["v1_compatibility"]),
        "rows": copy.deepcopy(analysis["rows"]),
    }


def _model_summary(
    model_key: str,
    rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> Dict[str, Any]:
    ranking = sorted(
        (row for row in rows if row["ranking_candidate"]),
        key=lambda row: (-float(row["S_RATE"]), int(row["width"]), int(row["start"])),
    )
    point_top = ranking[0] if ranking else None
    selected = next((row for row in rows if row["selected"]), None)
    return {
        "model_key": model_key,
        "model": v1.MODEL_SPECS[model_key]["model"],
        "candidate_count": len(rows),
        "new_eligible_count": sum(bool(row["NewEligible"]) for row in rows),
        "rate_stable_count": sum(bool(row["RateStable"]) for row in rows),
        "ranking_candidate_count": len(ranking),
        "complete_rate_ranking": [
            {
                "display_rank": row["display_rank"],
                "width": row["width"],
                "start": row["start"],
                "window": row["window"],
                "G_H": row["G_H"],
                "G_K": row["G_K"],
                "S_RATE": row["S_RATE"],
                "RateStable": row["RateStable"],
                "NewEligible": row["NewEligible"],
            }
            for row in ranking
        ],
        "point_top": (
            None
            if point_top is None
            else {
                "width": point_top["width"],
                "start": point_top["start"],
                "window": point_top["window"],
                "S_RATE": point_top["S_RATE"],
                "point_top_tie": point_top["point_top_tie"],
            }
        ),
        "point_top_tied_keys": copy.deepcopy(decision["point_top_tied_keys"]),
        "selection_frequency": decision["selection_frequency"],
        "replicate_winner_count": decision["replicate_winner_count"],
        "replicate_no_winner_count": decision["replicate_no_winner_count"],
        "decision": decision["decision"],
        "selected_window": (
            None
            if selected is None
            else {
                "width": selected["width"],
                "start": selected["start"],
                "window": selected["window"],
                "S_RATE": selected["S_RATE"],
            }
        ),
        "v1_compatibility": copy.deepcopy(compatibility),
        "positive_benefit_assessed": False,
    }


def _rows_close(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        score = rate_score(row["G_H"], row["G_K"])
        if score is None:
            if row["S_RATE"] is not None or row["balance_ratio"] is not None:
                return False
        else:
            if not math.isclose(float(row["S_RATE"]), score, rel_tol=0.0, abs_tol=1e-15):
                return False
            expected_balance = min(float(row["G_H"]), float(row["G_K"])) / max(
                float(row["G_H"]), float(row["G_K"])
            )
            if not math.isclose(
                float(row["balance_ratio"]),
                expected_balance,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                return False
        if bool(row["RateStable"]) != (
            score is not None
            and float(row["rate_LCB_H"]) > 0.0
            and float(row["rate_LCB_K"]) > 0.0
        ):
            return False
        if bool(row["ranking_candidate"]) != (
            bool(row["NewEligible"]) and bool(row["RateStable"])
        ):
            return False
    return True


def _summary_closes(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> bool:
    candidates = [
        row for row in rows if bool(row["NewEligible"]) and bool(row["RateStable"])
    ]
    selected = [row for row in rows if row["selected"]]
    return (
        int(summary["candidate_count"]) == len(rows)
        and int(summary["new_eligible_count"])
        == sum(bool(row["NewEligible"]) for row in rows)
        and int(summary["rate_stable_count"])
        == sum(bool(row["RateStable"]) for row in rows)
        and int(summary["ranking_candidate_count"]) == len(candidates)
        and len(selected) <= 1
        and (summary["selected_window"] is None) == (not selected)
    )


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise AbsoluteRateError("sample standard deviation requires two values")
    mean = math.fsum(values) / len(values)
    result = math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )
    return _finite(result, "sample standard deviation")


def _linear_quantile(values: Sequence[Any], quantile: float) -> float:
    ordered = sorted(_finite(value, "quantile value") for value in values)
    position = float(quantile) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    data = _json_bytes(payload)
    if path.exists():
        raise FileExistsError("refusing to overwrite Gate H artifact: %s" % path)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return v1.file_sha256(path)


def _write_new_text(path: Path, text: str) -> str:
    if path.exists():
        raise FileExistsError("refusing to overwrite Gate H artifact: %s" % path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return v1.file_sha256(path)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AbsoluteRateError("%s must be an integer" % label)
    return int(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AbsoluteRateError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise AbsoluteRateError("BLOCK_NONFINITE_RATE_ANALYSIS: %s" % label)
    return result


__all__ = [
    "AbsoluteRateError",
    "CROSS_MODEL_SUMMARY_JSON",
    "MANIFEST_RECEIPT_JSON",
    "METHOD",
    "PRIMARY_OUTPUTS",
    "QWEN17_CSV",
    "QWEN17_JSON",
    "QWEN4_CSV",
    "QWEN4_JSON",
    "SUMMARY_ZH",
    "VERIFIER_RECEIPT_JSON",
    "analyze_v2_model",
    "assert_no_forbidden_output_keys",
    "build_primary_payloads",
    "compare_v1_rows",
    "compute_rate_statistics",
    "rate_score",
    "render_candidates_csv",
    "render_summary_zh",
    "select_rate_decision",
    "validate_card",
    "verify_primary_outputs",
    "write_primary_outputs",
]
