"""Depth-general, outcome-blind V3.1 core for LoopScope Phase 7.

The implementation consumes only the sanitized ``H`` and ``D`` trajectories.
No hidden diagnostic, target, prediction, or outcome field participates in the
selector.  Formal 2,000-replicate scoring is available to the later Gate D
launcher; Gate A tests use small synthetic replicate counts only.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase7_schema import (
    FORMAL_BOOTSTRAP_REPLICATES,
    FORMAL_BOOTSTRAP_SEED,
    METHOD_ID,
    METHOD_VERSION,
    RANK_FREQUENCY_THRESHOLD,
    V3_SCHEMA_VERSION,
    WIDTHS,
    Phase7ContractError,
    candidate_domain,
    scan_forbidden_fields,
    validate_sanitized_record,
)


REL_TOL = 1e-12
ABS_TOL = 1e-12
Q_THRESHOLD = 0.5
EPSILON = 1e-12
LEGAL_TERMINAL_STATES = (
    "SELECTED_WINDOW",
    "ABSTAIN_NO_V3_ELIGIBLE",
    "ABSTAIN_COMBINED_RANK_UNSTABLE",
)
BOUNDARY_SOURCE_METRICS = (
    "choice_entropy",
    "kl_to_final",
    "hidden_rms_l2_to_final",
    "hidden_cosine_to_final",
    "hidden_cosine_distance_to_final",
)
TRANSITION_SOURCE_METRICS = ("adjacent_angular_distance",)


class V3Error(Phase7ContractError):
    """Fail-closed V3 analysis error."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V3Error("%s is not numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise V3Error("%s is non-finite" % label)
    return result


def project_records(
    records: Sequence[Mapping[str, Any]], *, expected_layer_count: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Project records to the exact V3 selector fields ``identity/category/H/D``."""

    projected: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        validate_sanitized_record(record, expected_layer_count=expected_layer_count)
        identity = str(record["canonical_identity"])
        if identity in seen:
            raise V3Error("BLOCK_INVALID_V3_INPUT: duplicate identity")
        seen.add(identity)
        h_values = [float(row["choice_entropy"]) for row in record["boundaries"]]
        d_values = [float(row["kl_to_final"]) for row in record["boundaries"]]
        projected.append(
            {
                "identity": identity,
                "category": str(record["subject"]),
                "H": h_values,
                "D": d_values,
            }
        )
    if not projected:
        raise V3Error("BLOCK_INVALID_V3_INPUT: no records")
    return projected


def _source_metric_records(
    records: Sequence[Mapping[str, Any]], *, expected_layer_count: int
) -> List[Dict[str, Any]]:
    """Project sanitized scalar diagnostics for the plotting-only source data.

    This projection is deliberately separate from :func:`project_records`:
    V3 eligibility and ranking continue to consume only identity/category/H/D.
    """

    rows: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        validate_sanitized_record(record, expected_layer_count=expected_layer_count)
        identity = str(record["canonical_identity"])
        if identity in seen:
            raise V3Error("BLOCK_INVALID_V3_INPUT: duplicate identity")
        seen.add(identity)
        row: Dict[str, Any] = {
            "identity": identity,
            "category": str(record["subject"]),
        }
        for metric in BOUNDARY_SOURCE_METRICS:
            row[metric] = [_finite(boundary[metric], metric) for boundary in record["boundaries"]]
        for metric in TRANSITION_SOURCE_METRICS:
            row[metric] = [_finite(transition[metric], metric) for transition in record["transitions"]]
        rows.append(row)
    if not rows:
        raise V3Error("BLOCK_INVALID_V3_INPUT: no records")
    return rows


def _layout(
    projected: Sequence[Mapping[str, Any]],
) -> Tuple[List[str], Dict[str, List[Mapping[str, Any]]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in projected:
        groups.setdefault(str(row["category"]), []).append(row)
    categories = sorted(groups)
    ordered = {
        category: sorted(groups[category], key=lambda row: str(row["identity"]))
        for category in categories
    }
    if not categories or any(not ordered[category] for category in categories):
        raise V3Error("BLOCK_INVALID_V3_INPUT: empty category")
    return categories, ordered


def macro_curves(projected: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[float], List[float]]:
    categories, groups = _layout(projected)
    boundary_count = len(projected[0]["H"])
    if boundary_count < 2 or any(len(row["H"]) != boundary_count or len(row["D"]) != boundary_count for row in projected):
        raise V3Error("BLOCK_INVALID_V3_INPUT: boundary array lengths differ")
    h_bar: List[float] = []
    d_bar: List[float] = []
    for boundary in range(boundary_count):
        h_by_category = [
            math.fsum(float(row["H"][boundary]) for row in groups[category]) / len(groups[category])
            for category in categories
        ]
        d_by_category = [
            math.fsum(float(row["D"][boundary]) for row in groups[category]) / len(groups[category])
            for category in categories
        ]
        h_bar.append(math.fsum(h_by_category) / len(categories))
        d_bar.append(math.fsum(d_by_category) / len(categories))
    return categories, h_bar, d_bar


def _macro_curve_for_metric(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> Tuple[List[str], List[float]]:
    categories, groups = _layout(rows)
    length = len(rows[0][metric])
    if length < 1 or any(len(row[metric]) != length for row in rows):
        raise V3Error("BLOCK_INVALID_V3_INPUT: source metric lengths differ")
    curve: List[float] = []
    for index in range(length):
        by_category = [
            math.fsum(float(row[metric][index]) for row in groups[category]) / len(groups[category])
            for category in categories
        ]
        curve.append(math.fsum(by_category) / len(categories))
    return categories, curve


def _bootstrap_metric_curves(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], replicates: int, seed: int
) -> Dict[str, List[List[float]]]:
    """Bootstrap plotting metrics with the same canonical category draw map.

    The selector's H/D bootstrap remains unchanged.  This companion routine
    replays the identical seeded category/index draws for source-data CIs while
    keeping diagnostic metrics outside all selector calculations.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise V3Error("bootstrap replicates must be an integer >= 2")
    categories, groups = _layout(rows)
    lengths = {metric: len(rows[0][metric]) for metric in metrics}
    if any(length < 1 for length in lengths.values()) or any(
        len(row[metric]) != lengths[metric] for row in rows for metric in metrics
    ):
        raise V3Error("BLOCK_INVALID_V3_INPUT: source metric lengths differ")
    rng = random.Random(seed)
    output: Dict[str, List[List[float]]] = {metric: [] for metric in metrics}
    for _replicate in range(replicates):
        category_curves: Dict[str, List[List[float]]] = {metric: [] for metric in metrics}
        for category in categories:
            group = groups[category]
            choices = [rng.randrange(len(group)) for _ in group]
            for metric in metrics:
                category_curves[metric].append(
                    [
                        math.fsum(float(group[index][metric][position]) for index in choices) / len(group)
                        for position in range(lengths[metric])
                    ]
                )
        for metric in metrics:
            output[metric].append(
                [
                    math.fsum(curve[position] for curve in category_curves[metric]) / len(categories)
                    for position in range(lengths[metric])
                ]
            )
    return output


def _aggregate_curve(point: Sequence[float], bootstrap: Sequence[Sequence[float]]) -> Dict[str, List[Optional[float]]]:
    if not point or len(bootstrap) < 2 or any(len(curve) != len(point) for curve in bootstrap):
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: aggregate curve shape differs")
    lower = [_quantile([float(curve[index]) for curve in bootstrap], 0.025) for index in range(len(point))]
    upper = [_quantile([float(curve[index]) for curve in bootstrap], 0.975) for index in range(len(point))]
    return {
        "point_mean": [float(value) for value in point],
        "ci95_lower": lower,
        "ci95_upper": upper,
        "mean_change_from_previous": [None]
        + [float(point[index]) - float(point[index - 1]) for index in range(1, len(point))],
    }


def aggregate_source_data(
    records: Sequence[Mapping[str, Any]], *, layer_count: int, replicates: int, seed: int
) -> Dict[str, Any]:
    """Create plotting-safe category-macro aggregates from sealed scalar records."""

    rows = _source_metric_records(records, expected_layer_count=layer_count)
    categories, _groups = _layout(rows)
    boundary_bootstrap = _bootstrap_metric_curves(rows, BOUNDARY_SOURCE_METRICS, replicates, seed)
    transition_bootstrap = _bootstrap_metric_curves(rows, TRANSITION_SOURCE_METRICS, replicates, seed)
    boundary_metrics = {}
    for metric in BOUNDARY_SOURCE_METRICS:
        metric_categories, point = _macro_curve_for_metric(rows, metric)
        if metric_categories != categories or len(point) != layer_count + 1:
            raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: boundary aggregate differs")
        boundary_metrics[metric] = _aggregate_curve(point, boundary_bootstrap[metric])
    transition_metrics = {}
    for metric in TRANSITION_SOURCE_METRICS:
        metric_categories, point = _macro_curve_for_metric(rows, metric)
        if metric_categories != categories or len(point) != layer_count:
            raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: transition aggregate differs")
        transition_metrics[metric] = _aggregate_curve(point, transition_bootstrap[metric])
    return {
        "aggregation": "equal_category_macro",
        "ci_method": "within-category resample, equal-category-macro percentile bootstrap 95%",
        "replicates": replicates,
        "seed": seed,
        "record_count": len(rows),
        "subject_count": len(categories),
        "boundary_count": layer_count + 1,
        "transition_count": layer_count,
        "boundary_metrics": boundary_metrics,
        "transition_metrics": transition_metrics,
    }


def bootstrap_macro_curves(
    projected: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> Dict[str, List[List[float]]]:
    """Joint within-category resampling for H and D using one draw map."""

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise V3Error("bootstrap replicates must be an integer >= 2")
    categories, groups = _layout(projected)
    boundary_count = len(projected[0]["H"])
    rng = random.Random(seed)
    h_curves: List[List[float]] = []
    d_curves: List[List[float]] = []
    for _replicate in range(replicates):
        h_by_category: List[List[float]] = []
        d_by_category: List[List[float]] = []
        for category in categories:
            group = groups[category]
            choices = [rng.randrange(len(group)) for _ in group]
            h_by_category.append(
                [math.fsum(group[index]["H"][layer] for index in choices) / len(group) for layer in range(boundary_count)]
            )
            d_by_category.append(
                [math.fsum(group[index]["D"][layer] for index in choices) / len(group) for layer in range(boundary_count)]
            )
        h_curves.append(
            [math.fsum(row[layer] for row in h_by_category) / len(categories) for layer in range(boundary_count)]
        )
        d_curves.append(
            [math.fsum(row[layer] for row in d_by_category) / len(categories) for layer in range(boundary_count)]
        )
    return {"H": h_curves, "D": d_curves}


def rates(h_bar: Sequence[float], d_bar: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(h_bar) != len(d_bar) or len(h_bar) < 2:
        raise V3Error("macro curves must have equal length >= 2")
    r_h = [float(h_bar[index]) - float(h_bar[index + 1]) for index in range(len(h_bar) - 1)]
    r_k = [float(d_bar[index + 1]) - float(d_bar[index]) for index in range(len(d_bar) - 1)]
    if not all(math.isfinite(value) for value in r_h + r_k):
        raise V3Error("non-finite rate")
    return r_h, r_k


def _fit_at_tau(series: Sequence[float], start: int, width: int, tau: int) -> Dict[str, Any]:
    values = [float(value) for value in series[start : start + width]]
    split = tau - start
    if len(values) != width or split <= 0 or split >= width:
        raise V3Error("tau is not a strict internal boundary")
    left = values[:split]
    right = values[split:]
    mu_left = math.fsum(left) / len(left)
    mu_right = math.fsum(right) / len(right)
    mu_all = math.fsum(values) / width
    delta = mu_right - mu_left
    turn = abs(delta)
    sst = math.fsum((value - mu_all) ** 2 for value in values)
    bss = (len(left) * len(right) / width) * (mu_left - mu_right) ** 2
    wss = math.fsum((value - mu_left) ** 2 for value in left) + math.fsum(
        (value - mu_right) ** 2 for value in right
    )
    tolerance = 1e-10 * max(sst, bss, wss, 1e-24)
    if abs(sst - bss - wss) > tolerance:
        raise V3Error("SST/BSS/WSS decomposition mismatch")
    q = None if sst == 0.0 else bss / sst
    if q is not None:
        if q < 0.0 and abs(q) <= ABS_TOL:
            q = 0.0
        if q > 1.0 and abs(q - 1.0) <= ABS_TOL:
            q = 1.0
        if q < -ABS_TOL or q > 1.0 + ABS_TOL:
            raise V3Error("turn Q is outside [0,1]")
    return {
        "tau": tau,
        "n_L": len(left),
        "n_R": len(right),
        "mu_L": mu_left,
        "mu_R": mu_right,
        "delta": delta,
        "T": turn,
        "BSS": bss,
        "WSS": wss,
        "Q": q,
        "SST": sst,
    }


def fit_turn(series: Sequence[float], start: int, width: int) -> Dict[str, Any]:
    values = [float(value) for value in series[start : start + width]]
    if len(values) != width or not all(math.isfinite(value) for value in values):
        raise V3Error("candidate rate vector is incomplete/non-finite")
    scale = max(1.0, max(abs(value) for value in values))
    rate_tol = 1e-12 * scale
    sst_tol = width * rate_tol**2
    entries = [_fit_at_tau(series, start, width, tau) for tau in range(start + 1, start + width)]
    if sum(float(entry["T"]) > rate_tol for entry in entries) == 0 or math.fsum(
        (value - math.fsum(values) / width) ** 2 for value in values
    ) <= sst_tol:
        return {
            "entries": entries,
            "tau": None,
            "ties": [],
            "tau_candidates": [entry["tau"] for entry in entries],
            "turn_strength_pass": False,
            "failure": "NO_MEASURABLE_TURN",
        }
    q_max = max(float(entry["Q"]) for entry in entries if entry["Q"] is not None)
    ties = [
        entry["tau"]
        for entry in entries
        if entry["Q"] is not None
        and math.isclose(float(entry["Q"]), q_max, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    ]
    if len(ties) != 1:
        return {
            "entries": entries,
            "tau": None,
            "ties": ties,
            "tau_candidates": [entry["tau"] for entry in entries],
            "turn_strength_pass": False,
            "failure": "TURN_TAU_TIE",
        }
    tau = ties[0]
    strength_pass = q_max > Q_THRESHOLD and not math.isclose(
        q_max, Q_THRESHOLD, rel_tol=REL_TOL, abs_tol=ABS_TOL
    )
    return {
        "entries": entries,
        "tau": tau,
        "ties": ties,
        "tau_candidates": [entry["tau"] for entry in entries],
        "turn_strength_pass": strength_pass,
        "failure": None if strength_pass else "TURN_NOT_DOMINANT",
    }


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise V3Error("at least two bootstrap values are required")
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise V3Error("cannot compute an empty quantile")
    position = q * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] + fraction * (ordered[right] - ordered[left])


def rate_statistics(
    point_h: float, point_k: float, bootstrap_h: Sequence[float], bootstrap_k: Sequence[float]
) -> Dict[str, Any]:
    if len(bootstrap_h) != len(bootstrap_k) or len(bootstrap_h) < 2:
        raise V3Error("H/K bootstrap draw count differs")
    values_h = [_finite(value, "bootstrap G_H") for value in bootstrap_h]
    values_k = [_finite(value, "bootstrap G_K") for value in bootstrap_k]
    se_h = _sample_sd(values_h)
    se_k = _sample_sd(values_k)
    max_t = [
        max(
            (point_h - value_h) / max(se_h, EPSILON),
            (point_k - value_k) / max(se_k, EPSILON),
        )
        for value_h, value_k in zip(values_h, values_k)
    ]
    c95 = max(0.0, _quantile(max_t, 0.95))
    lcb_h = point_h - c95 * max(se_h, EPSILON)
    lcb_k = point_k - c95 * max(se_k, EPSILON)
    return {
        "rate_SE_H": se_h,
        "rate_SE_K": se_k,
        "rate_c95": c95,
        "rate_LCB_H": lcb_h,
        "rate_LCB_K": lcb_k,
        "RateStable": bool(lcb_h > 0.0 and lcb_k > 0.0),
    }


def _fixed_tau_q(series: Sequence[float], start: int, width: int, tau: int) -> Optional[float]:
    fit = _fit_at_tau(series, start, width, tau)
    scale = max(1.0, max(abs(float(value)) for value in series[start : start + width]))
    rate_tol = 1e-12 * scale
    sst_tol = width * rate_tol**2
    if fit["SST"] <= sst_tol or fit["T"] <= rate_tol or fit["Q"] is None:
        return None
    q = float(fit["Q"])
    return q if q > 0.0 else None


def _key(row: Mapping[str, Any]) -> Tuple[int, int]:
    return int(row["width"]), int(row["start"])


def _failure_reasons(row: Mapping[str, Any], turn_h: Mapping[str, Any], turn_k: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not row["NetPositive"]:
        reasons.append("NET_NOT_POSITIVE")
    if not row["RateStable"]:
        reasons.append("RATE_NOT_STABLE")
    if turn_h.get("failure"):
        reasons.append("%s_H" % turn_h["failure"])
    if turn_k.get("failure"):
        reasons.append("%s_K" % turn_k["failure"])
    if turn_h.get("tau") is not None and turn_k.get("tau") is not None and turn_h["tau"] != turn_k["tau"]:
        reasons.append("TURN_LOCATION_MISMATCH")
    if not row["AggregateCommonTurn"]:
        reasons.append("AGGREGATE_COMMON_TURN_NOT_CLOSED")
    return reasons


def analyze_projected(
    projected: Sequence[Mapping[str, Any]],
    *,
    layer_count: int,
    replicates: int = FORMAL_BOOTSTRAP_REPLICATES,
    seed: int = FORMAL_BOOTSTRAP_SEED,
    formal: bool = False,
) -> Dict[str, Any]:
    """Compute a model-local V3.1 result from the sanitized projection."""

    if formal and (replicates != FORMAL_BOOTSTRAP_REPLICATES or seed != FORMAL_BOOTSTRAP_SEED):
        raise V3Error("BLOCK_METHOD_CONTRACT_MISMATCH: formal bootstrap differs")
    domain = candidate_domain(layer_count)
    categories, h_bar, d_bar = macro_curves(projected)
    if len(h_bar) != layer_count + 1:
        raise V3Error("projected depth differs from candidate domain")
    r_h, r_k = rates(h_bar, d_bar)
    bootstrap = bootstrap_macro_curves(projected, replicates, seed)
    rows: List[Dict[str, Any]] = []
    turns: Dict[Tuple[int, int], Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for candidate in domain["candidates"]:
        width, start = int(candidate["width"]), int(candidate["start"])
        g_h = (h_bar[start] - h_bar[start + width]) / width
        g_k = (d_bar[start + width] - d_bar[start]) / width
        turn_h = fit_turn(r_h, start, width)
        turn_k = fit_turn(r_k, start, width)
        turns[(width, start)] = (turn_h, turn_k)
        net_positive = bool(math.isfinite(g_h) and math.isfinite(g_k) and g_h > 0.0 and g_k > 0.0)
        common_turn = bool(
            turn_h["tau"] is not None
            and turn_k["tau"] is not None
            and turn_h["turn_strength_pass"]
            and turn_k["turn_strength_pass"]
            and turn_h["tau"] == turn_k["tau"]
        )
        h_fit = next((entry for entry in turn_h["entries"] if entry["tau"] == turn_h["tau"]), None)
        k_fit = next((entry for entry in turn_k["entries"] if entry["tau"] == turn_k["tau"]), None)
        s_rate = math.sqrt(g_h * g_k) if net_positive else None
        s_turn = math.sqrt(float(h_fit["Q"]) * float(k_fit["Q"])) if common_turn and h_fit and k_fit else None
        row: Dict[str, Any] = {
            **candidate,
            "Scorable": True,
            "NetPositive": net_positive,
            "RateStable": False,
            "AggregateCommonTurn": common_turn,
            "EligibleV3": False,
            "G_H": g_h,
            "G_K": g_k,
            "S_RATE": s_rate,
            "S_TURN": s_turn,
            "S_RATE_TURN": s_rate * s_turn if s_rate is not None and s_turn is not None else None,
            "rate_SE_H": None,
            "rate_SE_K": None,
            "rate_c95": None,
            "rate_LCB_H": None,
            "rate_LCB_K": None,
            "rH_window": list(r_h[start : start + width]),
            "rK_window": list(r_k[start : start + width]),
            "tau_H": turn_h["tau"],
            "tau_K": turn_k["tau"],
            "common_tau": turn_h["tau"] if common_turn else None,
            "turn_H_by_tau": turn_h["entries"],
            "turn_K_by_tau": turn_k["entries"],
            "turn_tau_tie_H": list(turn_h["ties"]),
            "turn_tau_tie_K": list(turn_k["ties"]),
            "turn_strength_pass_H": bool(turn_h["turn_strength_pass"]),
            "turn_strength_pass_K": bool(turn_k["turn_strength_pass"]),
            "point_rank": None,
            "point_top_tie": False,
            "tie_break_applied": False,
            "combined_rank_selection_frequency": None,
            "selected": False,
            "v3_failure_reasons": [],
        }
        if h_fit is not None:
            row.update(
                {
                    "turn_T_H": h_fit["T"],
                    "turn_delta_H": h_fit["delta"],
                    "turn_BSS_H": h_fit["BSS"],
                    "turn_WSS_H": h_fit["WSS"],
                    "turn_Q_H": h_fit["Q"],
                }
            )
        else:
            row.update({"turn_T_H": None, "turn_delta_H": None, "turn_BSS_H": None, "turn_WSS_H": None, "turn_Q_H": None})
        if k_fit is not None:
            row.update(
                {
                    "turn_T_K": k_fit["T"],
                    "turn_delta_K": k_fit["delta"],
                    "turn_BSS_K": k_fit["BSS"],
                    "turn_WSS_K": k_fit["WSS"],
                    "turn_Q_K": k_fit["Q"],
                }
            )
        else:
            row.update({"turn_T_K": None, "turn_delta_K": None, "turn_BSS_K": None, "turn_WSS_K": None, "turn_Q_K": None})
        rows.append(row)

    for row in rows:
        width, start = _key(row)
        boot_h = [
            (curve[start] - curve[start + width]) / width for curve in bootstrap["H"]
        ]
        boot_k = [
            (curve[start + width] - curve[start]) / width for curve in bootstrap["D"]
        ]
        row.update(rate_statistics(float(row["G_H"]), float(row["G_K"]), boot_h, boot_k))
        row["EligibleV3"] = bool(
            row["Scorable"]
            and row["NetPositive"]
            and row["RateStable"]
            and row["AggregateCommonTurn"]
        )

    eligible = [row for row in rows if row["EligibleV3"]]
    ordered = sorted(
        eligible,
        key=lambda row: (-float(row["S_RATE_TURN"]), int(row["width"]), int(row["start"])),
    )
    top_score = float(ordered[0]["S_RATE_TURN"]) if ordered else None
    top_ties = [row for row in ordered if top_score is not None and math.isclose(float(row["S_RATE_TURN"]), top_score, rel_tol=REL_TOL, abs_tol=ABS_TOL)]
    point_winner = _key(top_ties[0]) if top_ties else None
    for rank, row in enumerate(ordered, start=1):
        row["point_rank"] = rank
        row["point_top_tie"] = row in top_ties
        row["tie_break_applied"] = row in top_ties and len(top_ties) > 1

    winner_counts: Dict[Tuple[int, int], int] = {_key(row): 0 for row in eligible}
    for replicate in range(replicates):
        candidates: List[Tuple[float, Tuple[int, int]]] = []
        boot_h_curve = bootstrap["H"][replicate]
        boot_d_curve = bootstrap["D"][replicate]
        boot_r_h, boot_r_k = rates(boot_h_curve, boot_d_curve)
        for row in eligible:
            width, start = _key(row)
            g_h = (boot_h_curve[start] - boot_h_curve[start + width]) / width
            g_k = (boot_d_curve[start + width] - boot_d_curve[start]) / width
            if g_h <= 0.0 or g_k <= 0.0:
                continue
            q_h = _fixed_tau_q(boot_r_h, start, width, int(row["tau_H"]))
            q_k = _fixed_tau_q(boot_r_k, start, width, int(row["tau_K"]))
            if q_h is None or q_k is None:
                continue
            score = math.sqrt(g_h * g_k * q_h * q_k)
            if math.isfinite(score):
                candidates.append((score, (width, start)))
        if candidates:
            anchor = max(score for score, _candidate in candidates)
            tied = sorted(
                key
                for score, key in candidates
                if math.isclose(score, anchor, rel_tol=REL_TOL, abs_tol=ABS_TOL)
            )
            winner_counts[tied[0]] += 1

    for row in rows:
        key = _key(row)
        if key in winner_counts:
            row["combined_rank_selection_frequency"] = winner_counts[key] / replicates
    point_frequency = winner_counts.get(point_winner, 0) / replicates if point_winner else None
    if not eligible:
        decision = "ABSTAIN_NO_V3_ELIGIBLE"
    elif point_frequency is None or point_frequency < RANK_FREQUENCY_THRESHOLD:
        decision = "ABSTAIN_COMBINED_RANK_UNSTABLE"
    else:
        decision = "SELECTED_WINDOW"
    selected_window = None
    if decision == "SELECTED_WINDOW" and point_winner is not None:
        selected = next(row for row in rows if _key(row) == point_winner)
        selected["selected"] = True
        selected_window = {
            "window_half_open": selected["window_half_open"],
            "window_layers_inclusive": selected["window_layers_inclusive"],
            "boundary_entry": selected["boundary_entry"],
            "boundary_exit": selected["boundary_exit"],
        }
    for row in rows:
        turn_h, turn_k = turns[_key(row)]
        row["v3_failure_reasons"] = _failure_reasons(row, turn_h, turn_k)
        if row["EligibleV3"] and decision == "ABSTAIN_COMBINED_RANK_UNSTABLE":
            row["v3_failure_reasons"].append("COMBINED_RANK_UNSTABLE")
    rows.sort(key=_key)
    payload = {
        "schema_version": V3_SCHEMA_VERSION,
        "artifact_role": "phase7_model_local_v3_analysis",
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "layer_count": layer_count,
        "boundary_count": layer_count + 1,
        "central_blocks": list(domain["central_blocks"]),
        "widths": list(WIDTHS),
        "candidate_count": len(rows),
        "record_count": len(projected),
        "category_count": len(categories),
        "aggregation": "equal_category_macro",
        "selector_projection_fields": ["identity", "category", "H", "D"],
        "hidden_diagnostics_in_selector": False,
        "validation_target_accessed": False,
        "test_split_accessed": False,
        "post_probe_evaluation_channel_used": False,
        "model_forward_executed": False,
        "loop_executed": False,
        "bootstrap_method": "within-category resample, equal-category-macro joint bootstrap",
        "replicates": replicates,
        "seed": seed,
        "point_rank_score": "S_RATE_TURN",
        "combined_rank_frequency_threshold": RANK_FREQUENCY_THRESHOLD,
        "selector_decision": decision,
        "point_operational_winner": list(point_winner) if point_winner else None,
        "point_top_tie_set": [list(_key(row)) for row in top_ties],
        "combined_rank_selection_frequency": point_frequency,
        "selected_window": selected_window,
        "candidates": rows,
    }
    validate_v3_payload(payload, expected_layer_count=layer_count)
    return payload


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    layer_count: int,
    replicates: int = FORMAL_BOOTSTRAP_REPLICATES,
    seed: int = FORMAL_BOOTSTRAP_SEED,
    formal: bool = False,
) -> Dict[str, Any]:
    payload = analyze_projected(
        project_records(records, expected_layer_count=layer_count),
        layer_count=layer_count,
        replicates=replicates,
        seed=seed,
        formal=formal,
    )
    payload["subject_count"] = int(payload["category_count"])
    payload["aggregate_source_data"] = aggregate_source_data(
        records,
        layer_count=layer_count,
        replicates=replicates,
        seed=seed,
    )
    payload["selector_summary"] = {
        "terminal_state": payload["selector_decision"],
        "candidate_count": payload["candidate_count"],
        "point_operational_winner": payload["point_operational_winner"],
        "combined_rank_selection_frequency": payload["combined_rank_selection_frequency"],
        "selected_window": payload["selected_window"],
    }
    payload["analysis_provenance"] = {
        "input_projection": ["identity", "subject", "H", "D"],
        "model_forward_executed": False,
        "loop_executed": False,
        "test_split_accessed": False,
        "validation_target_accessed": False,
    }
    validate_v3_payload(
        payload,
        expected_layer_count=layer_count,
        require_source_data=True,
    )
    return payload


def _validate_aggregate_curve(value: Any, expected_length: int) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "point_mean",
        "ci95_lower",
        "ci95_upper",
        "mean_change_from_previous",
    }:
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: aggregate curve schema differs")
    for field in ("point_mean", "ci95_lower", "ci95_upper"):
        series = value[field]
        if not isinstance(series, Sequence) or isinstance(series, (str, bytes)) or len(series) != expected_length:
            raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: aggregate curve length differs")
        for item in series:
            _finite(item, "aggregate source %s" % field)
    changes = value["mean_change_from_previous"]
    if not isinstance(changes, Sequence) or isinstance(changes, (str, bytes)) or len(changes) != expected_length:
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: aggregate change length differs")
    if changes[0] is not None:
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: first aggregate change must be null")
    for item in changes[1:]:
        _finite(item, "aggregate source mean change")


def _validate_aggregate_source_data(
    value: Any,
    *,
    layer_count: int,
    record_count: int,
    subject_count: int,
    replicates: int,
    seed: int,
) -> None:
    if not isinstance(value, Mapping):
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: source data is not an object")
    expected_keys = {
        "aggregation",
        "ci_method",
        "replicates",
        "seed",
        "record_count",
        "subject_count",
        "boundary_count",
        "transition_count",
        "boundary_metrics",
        "transition_metrics",
    }
    if set(value) != expected_keys:
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: source data schema differs")
    if (
        value.get("aggregation") != "equal_category_macro"
        or value.get("replicates") != replicates
        or value.get("seed") != seed
        or value.get("record_count") != record_count
        or value.get("subject_count") != subject_count
        or value.get("boundary_count") != layer_count + 1
        or value.get("transition_count") != layer_count
    ):
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: source data provenance differs")
    boundary_metrics = value.get("boundary_metrics")
    transition_metrics = value.get("transition_metrics")
    if not isinstance(boundary_metrics, Mapping) or set(boundary_metrics) != set(BOUNDARY_SOURCE_METRICS):
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: boundary metrics differ")
    if not isinstance(transition_metrics, Mapping) or set(transition_metrics) != set(TRANSITION_SOURCE_METRICS):
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: transition metrics differ")
    for aggregate in boundary_metrics.values():
        _validate_aggregate_curve(aggregate, layer_count + 1)
    for aggregate in transition_metrics.values():
        _validate_aggregate_curve(aggregate, layer_count)


def validate_v3_payload(
    payload: Mapping[str, Any],
    *,
    expected_layer_count: Optional[int] = None,
    require_source_data: bool = False,
) -> None:
    if not isinstance(payload, Mapping):
        raise V3Error("V3 payload must be an object")
    scan_forbidden_fields(payload)
    if payload.get("schema_version") != V3_SCHEMA_VERSION:
        raise V3Error("V3 schema_version differs")
    layer_count = payload.get("layer_count")
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 1:
        raise V3Error("V3 layer_count is invalid")
    if expected_layer_count is not None and layer_count != expected_layer_count:
        raise V3Error("V3 layer_count differs from expected")
    if payload.get("boundary_count") != layer_count + 1:
        raise V3Error("V3 boundary_count differs")
    domain = candidate_domain(layer_count)
    if payload.get("central_blocks") != domain["central_blocks"] or payload.get("candidate_count") != domain["candidate_count"]:
        raise V3Error("V3 candidate domain differs")
    if payload.get("selector_projection_fields") != ["identity", "category", "H", "D"]:
        raise V3Error("V3 selector projection differs")
    if payload.get("hidden_diagnostics_in_selector") is not False:
        raise V3Error("hidden diagnostics entered V3")
    if any(payload.get(key) is not False for key in ("validation_target_accessed", "test_split_accessed", "post_probe_evaluation_channel_used", "model_forward_executed", "loop_executed")):
        raise V3Error("V3 information barrier is not closed")
    if payload.get("selector_decision") not in LEGAL_TERMINAL_STATES:
        raise V3Error("V3 terminal state is invalid")
    rows = payload.get("candidates")
    if not isinstance(rows, Sequence) or len(rows) != domain["candidate_count"]:
        raise V3Error("V3 candidate table is incomplete")
    expected_pairs = [(row["width"], row["start"]) for row in domain["candidates"]]
    actual_pairs = [(row.get("width"), row.get("start")) for row in rows]
    if actual_pairs != expected_pairs:
        raise V3Error("V3 candidate membership/order differs")
    for row in rows:
        if not isinstance(row, Mapping):
            raise V3Error("V3 candidate row is not an object")
        if row.get("selected") not in (True, False):
            raise V3Error("V3 selected flag is invalid")
        for field in ("G_H", "G_K"):
            _finite(row.get(field), field)
        if row.get("EligibleV3") and row.get("S_RATE_TURN") is None:
            raise V3Error("eligible candidate has no S_RATE_TURN")
    selected_rows = [row for row in rows if row.get("selected")]
    if payload["selector_decision"] == "SELECTED_WINDOW" and len(selected_rows) != 1:
        raise V3Error("selected terminal must have exactly one selected row")
    if payload["selector_decision"] != "SELECTED_WINDOW" and selected_rows:
        raise V3Error("ABSTAIN terminal cannot mark a selected row")
    source_data = payload.get("aggregate_source_data")
    if require_source_data and source_data is None:
        raise V3Error("BLOCK_INVALID_V3_SOURCE_DATA: aggregate source data is missing")
    if source_data is not None:
        _validate_aggregate_source_data(
            source_data,
            layer_count=layer_count,
            record_count=int(payload["record_count"]),
            subject_count=int(payload.get("subject_count", payload["category_count"])),
            replicates=int(payload["replicates"]),
            seed=int(payload["seed"]),
        )


__all__ = [
    "ABS_TOL",
    "FORMAL_BOOTSTRAP_REPLICATES",
    "FORMAL_BOOTSTRAP_SEED",
    "LEGAL_TERMINAL_STATES",
    "METHOD_ID",
    "METHOD_VERSION",
    "Q_THRESHOLD",
    "V3Error",
    "aggregate_source_data",
    "analyze_projected",
    "analyze_records",
    "bootstrap_macro_curves",
    "fit_turn",
    "macro_curves",
    "project_records",
    "rate_statistics",
    "rates",
    "validate_v3_payload",
]
