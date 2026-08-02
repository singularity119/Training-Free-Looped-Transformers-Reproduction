"""Independent fresh-process verifier for the Gate N V3 rescore.

This file intentionally repeats the V3 numerical path instead of importing the
producer's analyzer.  It reloads the immutable Gate M JSONL, rebuilds the
identity/category/H/D projection, category-macro bootstrap, 42 rows, ranking,
digests and terminal decision, then compares that result with the write-once
freeze produced by the analyzer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase6_mmlu5_v3_selector import (
    ABS_TOL,
    BOUNDARY_COUNT,
    EXPECTED_CATEGORY_COUNT,
    EXPECTED_COMMIT,
    EXPECTED_RECORD_COUNT,
    EXPECTED_SOURCE_ROOT_NAME,
    EXPECTED_TRAJECTORY_SHA256,
    FORMAL_REPLICATES,
    FORMAL_SEED,
    FREQUENCY_THRESHOLD,
    INPUT_CLOSURE_SHA256,
    LAYERS,
    METHOD_FILE_SHA256,
    METHOD_ID,
    METHOD_VERSION,
    PROJECTION_FIELDS,
    Q_THRESHOLD,
    REL_TOL,
    SOURCE_GATE_M_COMMIT,
    STARTS,
    WIDTHS,
    GateNV3Error,
    canonical_json_bytes,
    file_sha256,
    load_json,
    load_jsonl,
    validate_freeze_payload,
    write_new_json,
)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: %s is not numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: %s is non-finite" % label)
    return result


def _identity(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: identity is not an object")
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _project(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    forbidden = {"gold", "label", "correctness", "accuracy", "gain", "flip", "outcome", "prediction", "answer", "target"}
    for record in records:
        if forbidden.intersection(record):
            raise GateNV3Error("BLOCK_INFORMATION_BARRIER_VIOLATION: outcome key in record")
        if not {"identity", "subject", "split", "boundaries"}.issubset(record) or record.get("split") != "validation":
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: trajectory contract differs")
        identity = _identity(record["identity"])
        if identity in seen:
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: duplicate identity")
        seen.add(identity)
        subject = record["subject"]
        if not isinstance(subject, str) or not subject:
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: category is empty")
        boundaries = record["boundaries"]
        if isinstance(boundaries, (str, bytes)) or not isinstance(boundaries, Sequence) or len(boundaries) != BOUNDARY_COUNT:
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: boundary count differs")
        h_values: List[float] = []
        d_values: List[float] = []
        for index, boundary in enumerate(boundaries):
            if not isinstance(boundary, Mapping) or boundary.get("boundary_id") != "B_%d" % index:
                raise GateNV3Error("BLOCK_INVALID_V3_INPUT: boundary IDs differ")
            h_value = _finite(boundary.get("choice_entropy"), "H[%d]" % index)
            d_value = _finite(boundary.get("kl_to_final"), "D[%d]" % index)
            if h_value < 0.0 or d_value < 0.0:
                raise GateNV3Error("BLOCK_INVALID_V3_INPUT: negative H/D")
            h_values.append(h_value)
            d_values.append(d_value)
        result.append({"identity": identity, "category": subject, "H": h_values, "D": d_values})
    if len(result) != EXPECTED_RECORD_COUNT or len({row["category"] for row in result}) != EXPECTED_CATEGORY_COUNT:
        raise GateNV3Error("BLOCK_POPULATION_OR_HASH_MISMATCH: verifier population differs")
    return result


def _layout(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[str], Dict[str, List[Tuple[int, Mapping[str, Any]]]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["category"]), []).append(row)
    categories = sorted(groups)
    indexed: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {}
    index = 0
    for category in categories:
        indexed[category] = []
        for row in sorted(groups[category], key=lambda item: str(item["identity"])):
            indexed[category].append((index, row))
            index += 1
    return categories, indexed


def _macro(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[float], List[float]]:
    categories, groups = _layout(rows)
    h_bar: List[float] = []
    d_bar: List[float] = []
    for layer in range(BOUNDARY_COUNT):
        h_by_category = [math.fsum(row["H"][layer] for _index, row in groups[category]) / len(groups[category]) for category in categories]
        d_by_category = [math.fsum(row["D"][layer] for _index, row in groups[category]) / len(groups[category]) for category in categories]
        h_bar.append(math.fsum(h_by_category) / len(categories))
        d_bar.append(math.fsum(d_by_category) / len(categories))
    return categories, h_bar, d_bar


def _bootstrap(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int) -> Dict[str, Any]:
    categories, groups = _layout(rows)
    rng = random.Random(seed)
    draws: List[List[int]] = []
    h_curves: List[List[float]] = []
    d_curves: List[List[float]] = []
    for _replicate in range(replicates):
        draw: List[int] = []
        category_h: List[List[float]] = []
        category_d: List[List[float]] = []
        for category in categories:
            group = groups[category]
            choices = [rng.randrange(len(group)) for _ in group]
            draw.extend(group[index][0] for index in choices)
            category_h.append([math.fsum(group[index][1]["H"][layer] for index in choices) / len(group) for layer in range(BOUNDARY_COUNT)])
            category_d.append([math.fsum(group[index][1]["D"][layer] for index in choices) / len(group) for layer in range(BOUNDARY_COUNT)])
        draws.append(draw)
        h_curves.append([math.fsum(row[layer] for row in category_h) / len(categories) for layer in range(BOUNDARY_COUNT)])
        d_curves.append([math.fsum(row[layer] for row in category_d) / len(categories) for layer in range(BOUNDARY_COUNT)])
    return {
        "H": h_curves,
        "D": d_curves,
        "draw_index_sha256": hashlib.sha256(canonical_json_bytes(draws)).hexdigest(),
    }


def _rates(h_bar: Sequence[float], d_bar: Sequence[float]) -> Tuple[List[float], List[float]]:
    h_rates = [float(h_bar[index]) - float(h_bar[index + 1]) for index in range(LAYERS)]
    k_rates = [float(d_bar[index + 1]) - float(d_bar[index]) for index in range(LAYERS)]
    if not all(math.isfinite(value) for value in h_rates + k_rates):
        raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: non-finite rate")
    return h_rates, k_rates


def _fit_tau(series: Sequence[float], start: int, width: int, tau: int) -> Dict[str, Any]:
    values = [float(value) for value in series[start : start + width]]
    left = values[: tau - start]
    right = values[tau - start :]
    mu_left = math.fsum(left) / len(left)
    mu_right = math.fsum(right) / len(right)
    mu_all = math.fsum(values) / len(values)
    delta = mu_right - mu_left
    sst = math.fsum((value - mu_all) ** 2 for value in values)
    bss = (len(left) * len(right) / width) * (mu_left - mu_right) ** 2
    wss = math.fsum((value - mu_left) ** 2 for value in left) + math.fsum((value - mu_right) ** 2 for value in right)
    tolerance = 1e-10 * max(sst, bss, wss, 1e-24)
    if bss < -tolerance or wss < -tolerance or abs(sst - bss - wss) > tolerance:
        raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: decomposition mismatch")
    q: float | None = None if sst == 0.0 else bss / sst
    if q is not None:
        if q < 0.0 and abs(q) <= 1e-12:
            q = 0.0
        if q > 1.0 and abs(q - 1.0) <= 1e-12:
            q = 1.0
        if q < -1e-12 or q > 1.0 + 1e-12:
            raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: Q outside [0,1]")
    return {
        "tau": tau,
        "n_L": len(left),
        "n_R": len(right),
        "mu_L": mu_left,
        "mu_R": mu_right,
        "delta": delta,
        "T": abs(delta),
        "BSS": bss,
        "WSS": wss,
        "Q": q,
        "SST": sst,
    }


def _fit_turn(series: Sequence[float], start: int, width: int) -> Dict[str, Any]:
    values = [float(value) for value in series[start : start + width]]
    scale = max(1.0, max(abs(value) for value in values))
    rate_tol = 1e-12 * scale
    sst_tol = width * rate_tol**2
    entries = [_fit_tau(series, start, width, tau) for tau in range(start + 1, start + width)]
    mean = math.fsum(values) / width
    sst = math.fsum((value - mean) ** 2 for value in values)
    if sst <= sst_tol or not any(entry["T"] > rate_tol for entry in entries):
        return {"entries": entries, "tau_candidates": [entry["tau"] for entry in entries], "tau": None, "ties": [], "turn_strength_pass": False, "failure": "NO_MEASURABLE_TURN"}
    q_max = max(float(entry["Q"]) for entry in entries if entry["Q"] is not None)
    ties = [entry["tau"] for entry in entries if entry["Q"] is not None and math.isclose(float(entry["Q"]), q_max, rel_tol=REL_TOL, abs_tol=ABS_TOL)]
    if len(ties) != 1:
        return {"entries": entries, "tau_candidates": [entry["tau"] for entry in entries], "tau": None, "ties": ties, "turn_strength_pass": False, "failure": "TURN_TAU_TIE"}
    tau = ties[0]
    q = float(next(entry for entry in entries if entry["tau"] == tau)["Q"])
    dominant = q > Q_THRESHOLD and not math.isclose(q, Q_THRESHOLD, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return {"entries": entries, "tau_candidates": [entry["tau"] for entry in entries], "tau": tau, "ties": ties, "turn_strength_pass": dominant, "failure": None if dominant else "TURN_NOT_DOMINANT"}


def _sd(values: Sequence[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] + fraction * (ordered[right] - ordered[left])


def _rate_statistics(point_h: float, point_k: float, boot_h: Sequence[float], boot_k: Sequence[float]) -> Dict[str, Any]:
    se_h = _sd(list(boot_h))
    se_k = _sd(list(boot_k))
    max_t = [max((point_h - h_value) / max(se_h, 1e-12), (point_k - k_value) / max(se_k, 1e-12)) for h_value, k_value in zip(boot_h, boot_k)]
    c95 = max(0.0, _quantile(max_t, 0.95))
    lcb_h = point_h - c95 * max(se_h, 1e-12)
    lcb_k = point_k - c95 * max(se_k, 1e-12)
    if not all(math.isfinite(value) for value in (se_h, se_k, c95, lcb_h, lcb_k)):
        raise GateNV3Error("BLOCK_NONFINITE_RATE_ANALYSIS")
    return {"rate_SE_H": se_h, "rate_SE_K": se_k, "rate_c95": c95, "rate_LCB_H": lcb_h, "rate_LCB_K": lcb_k, "RateStable": lcb_h > 0.0 and lcb_k > 0.0}


def _key(row: Mapping[str, Any]) -> Tuple[int, int]:
    return int(row["width"]), int(row["start"])


def _candidates() -> List[Dict[str, Any]]:
    return [
        {"width": width, "start": start, "stop_exclusive": start + width, "end_inclusive": start + width - 1, "window_half_open": "%d:%d" % (start, start + width), "window_layers_inclusive": "%d:%d" % (start, start + width - 1), "boundary_entry": "B_%d" % start, "boundary_exit": "B_%d" % (start + width)}
        for width in WIDTHS
        for start in STARTS[width]
    ]


def _fixed_q(series: Sequence[float], start: int, width: int, tau: int) -> float | None:
    fit = _fit_tau(series, start, width, tau)
    scale = max(1.0, max(abs(value) for value in series[start : start + width]))
    if fit["SST"] <= width * (1e-12 * scale) ** 2 or fit["T"] <= 1e-12 * scale or fit["Q"] is None or fit["Q"] <= 0.0:
        return None
    return float(fit["Q"])


def _rank(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[List[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
    remaining = [row for row in rows if row["EligibleV3"]]
    groups: List[List[Tuple[int, int]]] = []
    ranks: Dict[Tuple[int, int], int] = {}
    rank = 1
    while remaining:
        anchor = max(float(row["S_RATE_TURN"]) for row in remaining)
        tied = sorted([row for row in remaining if math.isclose(float(row["S_RATE_TURN"]), anchor, rel_tol=REL_TOL, abs_tol=ABS_TOL)], key=_key)
        keys = [_key(row) for row in tied]
        groups.append(keys)
        for key in keys:
            ranks[key] = rank
        remaining = [row for row in remaining if _key(row) not in set(keys)]
        rank += 1
    return groups, ranks


def _build_expected_freeze(
    rows: Sequence[Mapping[str, Any]],
    source_root: Path,
    source_hashes: Mapping[str, str],
    projection_sha: str,
    replicates: int,
    seed: int,
    expected_commit: str,
) -> Dict[str, Any]:
    categories, h_bar, d_bar = _macro(rows)
    r_h, r_k = _rates(h_bar, d_bar)
    boot = _bootstrap(rows, replicates, seed)
    point_rows: List[Dict[str, Any]] = []
    turn_map: Dict[Tuple[int, int], Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for candidate in _candidates():
        width, start = int(candidate["width"]), int(candidate["start"])
        g_h = (h_bar[start] - h_bar[start + width]) / width
        g_k = (d_bar[start + width] - d_bar[start]) / width
        turn_h = _fit_turn(r_h, start, width)
        turn_k = _fit_turn(r_k, start, width)
        turn_map[(width, start)] = (turn_h, turn_k)
        common = turn_h["tau"] is not None and turn_k["tau"] is not None and turn_h["turn_strength_pass"] and turn_k["turn_strength_pass"] and turn_h["tau"] == turn_k["tau"]
        positive = g_h > 0.0 and g_k > 0.0
        s_rate = math.sqrt(g_h * g_k) if positive else None
        q_h = next((entry["Q"] for entry in turn_h["entries"] if entry["tau"] == turn_h["tau"]), None)
        q_k = next((entry["Q"] for entry in turn_k["entries"] if entry["tau"] == turn_k["tau"]), None)
        s_turn = math.sqrt(float(q_h) * float(q_k)) if common else None
        row: Dict[str, Any] = {**candidate, "Scorable": True, "NetPositive": positive, "RateStable": False, "AggregateCommonTurn": common, "EligibleV3": False, "G_H": g_h, "G_K": g_k, "S_RATE": s_rate, "S_TURN": s_turn, "S_RATE_TURN": s_rate * s_turn if s_rate is not None and s_turn is not None else None, "rate_SE_H": None, "rate_SE_K": None, "rate_c95": None, "rate_LCB_H": None, "rate_LCB_K": None, "rH_window": r_h[start : start + width], "rK_window": r_k[start : start + width], "tau_candidates": list(turn_h["tau_candidates"]), "turn_H_by_tau": turn_h["entries"], "turn_K_by_tau": turn_k["entries"], "tau_H": turn_h["tau"], "tau_K": turn_k["tau"], "common_tau": turn_h["tau"] if common else None, "turn_T_H": None, "turn_T_K": None, "turn_delta_H": None, "turn_delta_K": None, "turn_BSS_H": None, "turn_BSS_K": None, "turn_WSS_H": None, "turn_WSS_K": None, "turn_Q_H": None, "turn_Q_K": None, "turn_tau_tie_H": list(turn_h["ties"]), "turn_tau_tie_K": list(turn_k["ties"]), "turn_strength_pass_H": bool(turn_h["turn_strength_pass"]), "turn_strength_pass_K": bool(turn_k["turn_strength_pass"]), "point_rank": None, "point_top_tie": False, "tie_break_applied": False, "combined_rank_selection_frequency": None, "selected": False, "v3_failure_reasons": [], "legacy_diagnostics": {}, "hidden_diagnostics": {}}
        if turn_h["tau"] is not None:
            fit = turn_h["entries"][turn_h["tau"] - start]
            row.update({"turn_T_H": fit["T"], "turn_delta_H": fit["delta"], "turn_BSS_H": fit["BSS"], "turn_WSS_H": fit["WSS"], "turn_Q_H": fit["Q"]})
        if turn_k["tau"] is not None:
            fit = turn_k["entries"][turn_k["tau"] - start]
            row.update({"turn_T_K": fit["T"], "turn_delta_K": fit["delta"], "turn_BSS_K": fit["BSS"], "turn_WSS_K": fit["WSS"], "turn_Q_K": fit["Q"]})
        point_rows.append(row)
    for row in point_rows:
        width, start = _key(row)
        boot_h = [(curve[start] - curve[start + width]) / width for curve in boot["H"]]
        boot_k = [(curve[start + width] - curve[start]) / width for curve in boot["D"]]
        row.update(_rate_statistics(row["G_H"], row["G_K"], boot_h, boot_k))
        row["EligibleV3"] = bool(row["Scorable"] and row["NetPositive"] and row["RateStable"] and row["AggregateCommonTurn"])
    groups, ranks = _rank(point_rows)
    top = groups[0] if groups else []
    operational = top[0] if top else None
    for row in point_rows:
        key = _key(row)
        group = next((group for group in groups if key in group), [])
        row["point_rank"] = ranks.get(key)
        row["point_top_tie"] = key in top
        row["tie_break_applied"] = bool(group and len(group) > 1)
    winner_counts = {_key(row): 0 for row in point_rows if row["EligibleV3"]}
    winners: List[List[int] | None] = []
    for index in range(replicates):
        h_boot, k_boot = _rates(boot["H"][index], boot["D"][index])
        available: List[Tuple[float, Tuple[int, int]]] = []
        for row in point_rows:
            if not row["EligibleV3"]:
                continue
            width, start = _key(row)
            g_h = (boot["H"][index][start] - boot["H"][index][start + width]) / width
            g_k = (boot["D"][index][start + width] - boot["D"][index][start]) / width
            if g_h <= 0.0 or g_k <= 0.0:
                continue
            q_h = _fixed_q(h_boot, start, width, int(row["tau_H"]))
            q_k = _fixed_q(k_boot, start, width, int(row["tau_K"]))
            if q_h is None or q_k is None:
                continue
            score = math.sqrt(g_h * g_k * q_h * q_k)
            if not math.isfinite(score):
                raise GateNV3Error("BLOCK_INVALID_V3_BOOTSTRAP_TURN_ANALYSIS")
            available.append((score, (width, start)))
        if not available:
            winners.append(None)
            continue
        anchor = max(score for score, _key_value in available)
        tie_keys = sorted(key for score, key in available if math.isclose(score, anchor, rel_tol=REL_TOL, abs_tol=ABS_TOL))
        winner = tie_keys[0]
        winner_counts[winner] += 1
        winners.append([winner[0], winner[1]])
    for row in point_rows:
        key = _key(row)
        if row["EligibleV3"]:
            row["combined_rank_selection_frequency"] = winner_counts[key] / replicates
        turn_h, turn_k = turn_map[key]
        reasons: List[str] = []
        if not row["NetPositive"]:
            reasons.append("NET_NOT_POSITIVE")
        if not row["RateStable"]:
            reasons.append("RATE_NOT_STABLE")
        if turn_h["failure"]:
            reasons.append("%s_H" % turn_h["failure"])
        if turn_k["failure"]:
            reasons.append("%s_K" % turn_k["failure"])
        if turn_h["tau"] is not None and turn_k["tau"] is not None and turn_h["tau"] != turn_k["tau"]:
            reasons.append("TURN_LOCATION_MISMATCH")
        row["v3_failure_reasons"] = reasons
    point_frequency = winner_counts[operational] / replicates if operational else None
    if not winner_counts:
        decision = "ABSTAIN_NO_V3_ELIGIBLE"
        selected_key = None
    elif point_frequency < FREQUENCY_THRESHOLD:
        decision = "ABSTAIN_COMBINED_RANK_UNSTABLE"
        selected_key = None
    else:
        decision = "SELECTED_WINDOW"
        selected_key = list(operational)
    for row in point_rows:
        row["selected"] = bool(decision == "SELECTED_WINDOW" and _key(row) == operational)
        if row["EligibleV3"] and decision == "ABSTAIN_COMBINED_RANK_UNSTABLE":
            row["v3_failure_reasons"].append("COMBINED_RANK_UNSTABLE")
    point_rows.sort(key=_key)
    ranking_digest = hashlib.sha256(canonical_json_bytes(point_rows)).hexdigest()
    payload: Dict[str, Any] = {
        "schema_version": "loopscope.phase6.mmlu5-gate-n-v3-selector-freeze.v1", "artifact_role": "gate_n_post_hoc_v3_selector_freeze", "gate": "N", "planning_thread_id": "019fb3de-2298-75f2-a083-0dca453ea79c", "executor_thread_id": "019fc2f1-812c-73a3-b5e3-011222a2f539", "expected_commit": expected_commit, "source_gate_m_root": str(source_root.resolve()), "source_input_closure": dict(source_hashes), "method_id": METHOD_ID, "method_version": METHOD_VERSION, "method_file_sha256": METHOD_FILE_SHA256, "input_distribution_contract": "full-vocabulary entropy and KL-to-final from sanitized Gate M MMLU 5-shot trajectory", "probe_contract": "Gate M validation-1531 native no-loop scalar trajectory; post-hoc rescore", "trajectory_file_sha256": EXPECTED_TRAJECTORY_SHA256, "identity_manifest_sha256": hashlib.sha256(canonical_json_bytes([json.loads(row["identity"]) for row in rows])).hexdigest(), "record_count": len(rows), "category_count": len(categories), "ordered_category_sha256": hashlib.sha256(canonical_json_bytes(categories)).hexdigest(), "layer_count": LAYERS, "boundary_count": BOUNDARY_COUNT, "central_blocks": [11, 24], "widths": list(WIDTHS), "candidate_count": len(point_rows), "aggregation": "equal_category_macro", "selector_projection_fields": list(PROJECTION_FIELDS), "hidden_diagnostics_in_selector": False, "outcome_fields_consumed": False, "bootstrap_method": "within-category resample, equal-category-macro joint bootstrap", "replicates": replicates, "seed": seed, "prng": "random.Random", "draw_api": "randrange", "standard_deviation_ddof": 1, "epsilon": 1e-12, "quantile_method": "linear_q_times_R_minus_1", "bootstrap_draw_index_sha256": boot["draw_index_sha256"], "bootstrap_combined_rank_winner_sha256": hashlib.sha256(canonical_json_bytes(winners)).hexdigest(), "turn_strength_Q_threshold": "0.5_strict", "numeric_tolerances": {"rel_tol": REL_TOL, "abs_tol": ABS_TOL, "decomposition_rel": 1e-10}, "point_rank_score": "S_RATE_TURN", "combined_rank_frequency_threshold": FREQUENCY_THRESHOLD, "selector_decision": decision, "selected_key": selected_key, "point_operational_winner": list(operational) if operational else None, "point_top_tie_set": [list(key) for key in top], "point_tie_break_applied": len(top) > 1, "combined_rank_selection_frequency": point_frequency, "combined_rank_winner_count": sum(winner is not None for winner in winners), "combined_rank_usable_replicates": sum(winner is not None for winner in winners), "selected_window": None, "ranking_digest": ranking_digest, "selector_projection_file_sha256": projection_sha, "information_barrier": {"validation_target_gold_read": False, "test_split_read": False, "outcome_read": False, "model_weights_loaded": False, "model_forward_executed": False, "cuda_gpu_slurm": False, "loop_executed": False, "hidden_diagnostics_consumed": False}, "rows": point_rows,
    }
    if selected_key is not None:
        selected_row = next(row for row in point_rows if _key(row) == tuple(selected_key))
        payload["selected_window"] = {key: selected_row[key] for key in ("window_half_open", "window_layers_inclusive", "boundary_entry", "boundary_exit")}
    payload["analysis_semantic_digest"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    validate_freeze_payload(payload)
    return payload


def _validate_source(root: Path) -> Dict[str, str]:
    root = root.resolve()
    if root.name != EXPECTED_SOURCE_ROOT_NAME:
        raise GateNV3Error("BLOCK_INPUT_CLOSURE_HASH_MISMATCH: source root name differs")
    observed: Dict[str, str] = {}
    for relative, expected in INPUT_CLOSURE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise GateNV3Error("BLOCK_INPUT_CLOSURE_READ: missing %s" % relative)
        observed[relative] = file_sha256(path)
        if observed[relative] != expected:
            raise GateNV3Error("BLOCK_INPUT_CLOSURE_HASH_MISMATCH: %s" % relative)
    for relative in ("manifest/formal_manifest.json", "manifest/freeze_receipt.json", "merge/merge_receipt.json", "merge/verifier_receipt.json"):
        value = load_json(root / relative)
        if value.get("status") not in (None, "PASS"):
            raise GateNV3Error("BLOCK_INPUT_CLOSURE_STATUS: %s" % relative)
        for key in ("selector_executed", "loop_executed", "outcome_read", "validation_target_gold_read", "test_split_read", "model_weights_loaded", "model_forward_executed", "cuda_gpu_slurm"):
            if value.get(key) is True:
                raise GateNV3Error("BLOCK_INFORMATION_BARRIER_VIOLATION: %s" % key)
        if relative == "manifest/formal_manifest.json" and value.get("expected_commit") not in (None, SOURCE_GATE_M_COMMIT):
            raise GateNV3Error("BLOCK_INPUT_CLOSURE_HASH_MISMATCH: Gate M source commit differs")
    return observed


def verify_freeze(*, source_root: Path, run_root: Path, expected_commit: str = EXPECTED_COMMIT) -> Dict[str, Any]:
    if not isinstance(expected_commit, str) or len(expected_commit) != 40:
        raise GateNV3Error("BLOCK_REPOSITORY_ADMISSION_MISMATCH: commit is not a SHA-1")
    try:
        int(expected_commit, 16)
    except ValueError as exc:
        raise GateNV3Error("BLOCK_REPOSITORY_ADMISSION_MISMATCH: commit is not hexadecimal") from exc
    source_root = Path(source_root).resolve()
    run_root = Path(run_root).resolve()
    source_hashes = _validate_source(source_root)
    freeze_path = run_root / "analysis/selector_freeze.json"
    projection_path = run_root / "analysis/selector_projection.jsonl"
    if not freeze_path.is_file() or not projection_path.is_file():
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: analysis artifacts missing")
    frozen = load_json(freeze_path)
    validate_freeze_payload(frozen)
    if frozen["expected_commit"] != expected_commit or frozen["source_gate_m_root"] != str(source_root):
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: source identity differs")
    records = load_jsonl(source_root / "merge/merged_trajectory_records.jsonl")
    projected = _project(records)
    projection_bytes = b"".join(json.dumps(row, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n" for row in projected)
    projection_sha = hashlib.sha256(projection_bytes).hexdigest()
    if file_sha256(projection_path) != projection_sha or frozen["selector_projection_file_sha256"] != projection_sha:
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: selector projection differs")
    observed_projection = load_jsonl(projection_path)
    if observed_projection != projected:
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: selector projection content differs")
    expected = _build_expected_freeze(projected, source_root, source_hashes, projection_sha, FORMAL_REPLICATES, FORMAL_SEED, expected_commit)
    if expected != frozen:
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: fresh V3 recomputation differs")
    receipt = {
        "schema_version": "loopscope.phase6.mmlu5-gate-n-v3-verifier.v1",
        "artifact_role": "gate_n_fresh_process_v3_verifier_receipt",
        "gate": "N",
        "expected_commit": expected_commit,
        "source_gate_m_root": str(source_root),
        "analysis_freeze_file_sha256": file_sha256(freeze_path),
        "selector_projection_file_sha256": file_sha256(projection_path),
        "trajectory_file_sha256": EXPECTED_TRAJECTORY_SHA256,
        "record_count": len(projected),
        "category_count": len({row["category"] for row in projected}),
        "candidate_count": 42,
        "replicates": FORMAL_REPLICATES,
        "seed": FORMAL_SEED,
        "decision": frozen["selector_decision"],
        "selected_key": frozen["selected_key"],
        "point_top_tie_set": frozen["point_top_tie_set"],
        "combined_rank_selection_frequency": frozen["combined_rank_selection_frequency"],
        "bootstrap_draw_index_sha256": frozen["bootstrap_draw_index_sha256"],
        "bootstrap_combined_rank_winner_sha256": frozen["bootstrap_combined_rank_winner_sha256"],
        "ranking_digest": frozen["ranking_digest"],
        "analysis_semantic_digest": frozen["analysis_semantic_digest"],
        "semantic_agreement": True,
        "selector_projection_fields": list(PROJECTION_FIELDS),
        "information_barrier": frozen["information_barrier"],
        "status": "PASS",
    }
    receipt_sha = write_new_json(run_root / "verifier/verifier_receipt.json", receipt)
    return {**receipt, "verifier_receipt_sha256": receipt_sha}


__all__ = ["verify_freeze"]
