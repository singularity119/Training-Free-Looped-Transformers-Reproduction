"""Width-4 primary and variable-width secondary Phase 4 selector."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tflt.loopscope.phase4_schema import (
    Phase4ContractError,
    WIDTH4_CONSENSUS_STARTS,
    WIDTHS,
    scan_forbidden_fields,
    validate_trajectory_record,
)


STRICT_KEYS: Tuple[Tuple[int, int], ...] = tuple(
    (width, start) for width in WIDTHS for start in range(width, 37 - 2 * width)
)
EDGE_KEYS: Tuple[Tuple[int, int], ...] = tuple(
    (width, start) for width in WIDTHS for start in range(37 - width)
)
FULL_KEYS: Tuple[Tuple[int, int], ...] = EDGE_KEYS
FULL_INDEX = {key: index for index, key in enumerate(FULL_KEYS)}
STRICT_INDEX = {key: index for index, key in enumerate(STRICT_KEYS)}


def analyze_selector(
    records: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 2000,
    seed: int = 20260717,
    selection_frequency_threshold: float = 0.80,
    d36_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Analyze scalar trajectories with one category-stratified joint bootstrap."""

    if not records:
        raise Phase4ContractError("selector records cannot be empty")
    if not isinstance(replicates, int) or replicates < 2:
        raise Phase4ContractError("replicates must be at least two")
    identities: List[str] = []
    categories: List[str] = []
    entropy_rows: List[List[float]] = []
    kl_rows: List[List[float]] = []
    for record in records:
        scan_forbidden_fields(record, allow_options=False)
        validate_trajectory_record(record, d36_tolerance=d36_tolerance)
        identities.append(record["canonical_identity"])
        categories.append(record["category"].strip())
        entropy_rows.append([row["full_vocabulary_entropy"] for row in record["boundaries"]])
        kl_rows.append([row["kl_to_final"] for row in record["boundaries"]])
    if len(set(identities)) != len(identities):
        raise Phase4ContractError("duplicate selector identity")
    entropy = np.asarray(entropy_rows, dtype=np.float64)
    kl = np.asarray(kl_rows, dtype=np.float64)
    if entropy.shape != (len(records), 37) or kl.shape != entropy.shape:
        raise Phase4ContractError("selector trajectories must have shape N x 37")
    if not np.isfinite(entropy).all() or not np.isfinite(kl).all():
        raise Phase4ContractError("selector trajectory contains non-finite values")

    sample_e = np.column_stack(
        [(entropy[:, start] - entropy[:, start + width]) / width for width, start in FULL_KEYS]
    )
    sample_k = np.column_stack(
        [(kl[:, start + width] - kl[:, start]) / width for width, start in FULL_KEYS]
    )
    point_e = np.mean(sample_e, axis=0)
    point_k = np.mean(sample_k, axis=0)
    point_strict = _strict_contrasts(point_e, point_k)
    point_edge = _edge_contrasts(point_e, point_k)

    strict_replicates, edge_replicates = _bootstrap_contrasts(
        sample_e, sample_k, categories, replicates=replicates, seed=seed
    )
    # Width four must reproduce the primary un-normalized E/K score even when the
    # frozen 1e-12 SE floor binds; other widths remain in E_rate/K_rate units.
    strict_score_scale = np.asarray(
        [4.0 if width == 4 else 1.0 for width, _start in STRICT_KEYS], dtype=np.float64
    )[:, None]
    strict_score_point = point_strict * strict_score_scale
    strict_score_replicates = strict_replicates * strict_score_scale[None, :, :]
    strict_se = np.std(strict_score_replicates, axis=0, ddof=1)
    edge_se = np.std(edge_replicates, axis=0, ddof=1)
    strict_denominator = np.maximum(strict_se, 1e-12)
    edge_denominator = np.maximum(edge_se, 1e-12)
    strict_scores = np.min(strict_score_point / strict_denominator, axis=1)
    edge_scores = np.min(point_edge / edge_denominator, axis=1)
    if not np.isfinite(strict_scores).all() or not np.isfinite(edge_scores).all():
        raise Phase4ContractError("selector score is non-finite")

    strict_rows = _rank_rows(
        [
            {
                "width": width,
                "start": start,
                "window": _window(width, start),
                "E_rate": float(point_e[FULL_INDEX[(width, start)]]),
                "K_rate": float(point_k[FULL_INDEX[(width, start)]]),
                "contrasts": _contrast_payload(
                    strict_score_point[index], strict_score_replicates[:, index, :]
                ),
                "score": float(strict_scores[index]),
            }
            for index, (width, start) in enumerate(STRICT_KEYS)
        ]
    )
    edge_rows = _rank_rows(
        [
            {
                "width": width,
                "start": start,
                "window": _window(width, start),
                "E_rate": float(point_e[FULL_INDEX[(width, start)]]),
                "K_rate": float(point_k[FULL_INDEX[(width, start)]]),
                "contrasts": _edge_payload(point_edge[index], edge_replicates[:, index, :]),
                "score": float(edge_scores[index]),
            }
            for index, (width, start) in enumerate(EDGE_KEYS)
        ]
    )

    width4_rows = [copy.deepcopy(row) for row in strict_rows if row["width"] == 4]
    width4_rows.sort(key=lambda row: row["start"])
    for row in width4_rows:
        row["E"] = row["E_rate"] * 4.0
        row["K"] = row["K_rate"] * 4.0
        start = row["start"]
        strict_index = STRICT_INDEX[(4, start)]
        comparisons = (start - 1, start + 1, start - 4, start + 4)
        center = FULL_INDEX[(4, start)]
        comparison_indices = [FULL_INDEX[(4, item)] for item in comparisons]
        lower = np.quantile(
            strict_replicates[:, strict_index, :], 0.025, axis=0, method="linear"
        )
        row["eligible"] = bool(
            point_e[center] > 0.0
            and point_k[center] > 0.0
            and all(point_e[index] < 0.0 and point_k[index] < 0.0 for index in comparison_indices)
            and bool(np.all(lower > 0.0))
        )
    primary_ranked = sorted(width4_rows, key=lambda row: (-row["score"], row["start"]))
    for rank, row in enumerate(primary_ranked, 1):
        row["primary_rank"] = rank
    eligible_ranked = [row for row in primary_ranked if row["eligible"]]
    selected: Optional[Dict[str, Any]] = None
    abstain_reason: Optional[str] = None
    if not eligible_ranked:
        abstain_reason = "NO_ELIGIBLE_WINDOW"
    elif len(eligible_ranked) > 1 and math.isclose(
        eligible_ranked[0]["score"], eligible_ranked[1]["score"], rel_tol=1e-12, abs_tol=1e-12
    ):
        abstain_reason = "TOP_SCORE_TIE"
    else:
        selected = eligible_ranked[0]
        eligible_by_start = sorted(eligible_ranked, key=lambda row: row["start"])
        eligible_indices = [STRICT_INDEX[(4, row["start"])] for row in eligible_by_start]
        replicate_scores = np.min(
            strict_score_replicates[:, eligible_indices, :]
            / strict_denominator[eligible_indices, :],
            axis=2,
        )
        winners = np.argmax(replicate_scores, axis=1)
        selected_index = next(
            index for index, row in enumerate(eligible_by_start) if row["start"] == selected["start"]
        )
        frequency = float(np.mean(winners == selected_index))
        selected["selection_frequency"] = frequency
        if frequency < selection_frequency_threshold:
            selected = None
            abstain_reason = "SELECTION_FREQUENCY_BELOW_0.80"

    width4_by_start = {row["start"]: row for row in width4_rows}
    if tuple(sorted(width4_by_start)) != WIDTH4_CONSENSUS_STARTS:
        raise Phase4ContractError("width-4 strict subset does not contain the frozen 25 starts")
    for row in strict_rows:
        if row["width"] == 4:
            primary = width4_by_start[row["start"]]
            row["eligible"] = primary["eligible"]
            row["width4_primary_rank"] = primary["primary_rank"]

    report = {
        "schema_version": "loopscope.phase4.selector-report.v1",
        "record_count": len(records),
        "bootstrap": {
            "method": "category_stratified_joint_sample_bootstrap",
            "replicates": replicates,
            "seed": seed,
            "percentile_method": "linear_q_times_R_minus_1",
            "standard_error": "sample_sd_ddof_1",
        },
        "width4_primary": {
            "candidate_start_count": 33,
            "consensus_start_count": len(width4_rows),
            "all_window_metrics": [
                {
                    "start": start,
                    "window": _window(4, start),
                    "E": float(point_e[FULL_INDEX[(4, start)]] * 4.0),
                    "K": float(point_k[FULL_INDEX[(4, start)]] * 4.0),
                    "consensus_scorable": start in WIDTH4_CONSENSUS_STARTS,
                }
                for start in range(33)
            ],
            "rows": primary_ranked,
            "selected_window": None if selected is None else selected["window"],
            "abstain": selected is None,
            "abstain_reason": abstain_reason,
        },
        "variable_width_strict": {"row_count": len(strict_rows), "rows": strict_rows},
        "variable_width_edge": {"row_count": len(edge_rows), "rows": edge_rows},
    }
    verify_selector_report(report)
    return report


def build_outcome_panel(
    selector_report: Mapping[str, Any], known_outcome_registry: Sequence[str]
) -> Dict[str, Any]:
    """Build the frozen width-4 panel without reading any outcome values."""

    verify_selector_report(selector_report)
    registry = set(known_outcome_registry)
    if not registry or "15:18" not in registry:
        raise Phase4ContractError("known_outcome_registry must include fixed comparator 15:18")
    rows = selector_report["width4_primary"]["rows"]
    blind = [row for row in rows if row["window"] not in registry]
    if len(blind) < 6:
        raise Phase4ContractError("known_outcome exclusions leave fewer than six blind windows")
    high = [row["window"] for row in blind[:3]]
    low = [row["window"] for row in sorted(blind, key=lambda row: (row["score"], row["start"]))[:3]]
    if set(high) & set(low):
        raise Phase4ContractError("blind high3 and low3 overlap")
    selected = selector_report["width4_primary"]["selected_window"]
    if selected is not None and selected not in set(high + low + ["15:18"]):
        raise Phase4ContractError("selected window is outside the frozen maximum panel")
    return {
        "schema_version": "loopscope.phase4.outcome-panel.v1",
        "baseline": "no-loop",
        "fixed_comparator": "15:18",
        "known_outcome_registry": sorted(registry),
        "blind_high3": high,
        "blind_low3": low,
        "selected_window": selected,
        "abstain": selected is None,
        "variable_width_outcomes_authorized": False,
    }


def verify_selector_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "loopscope.phase4.selector-report.v1":
        raise Phase4ContractError("unexpected selector report schema")
    primary = report.get("width4_primary")
    strict = report.get("variable_width_strict")
    edge = report.get("variable_width_edge")
    if not all(isinstance(value, Mapping) for value in (primary, strict, edge)):
        raise Phase4ContractError("selector report sections are missing")
    if primary.get("candidate_start_count") != 33 or primary.get("consensus_start_count") != 25:
        raise Phase4ContractError("width-4 33/25 contract mismatch")
    if strict.get("row_count") != 154 or edge.get("row_count") != 224:
        raise Phase4ContractError("variable-width 154/224 contract mismatch")
    primary_rows = primary.get("rows")
    all_window_metrics = primary.get("all_window_metrics")
    strict_rows = strict.get("rows")
    edge_rows = edge.get("rows")
    if not isinstance(primary_rows, list) or len(primary_rows) != 25:
        raise Phase4ContractError("primary width-4 rows are incomplete")
    if not isinstance(all_window_metrics, list) or len(all_window_metrics) != 33:
        raise Phase4ContractError("primary width-4 33-window metrics are incomplete")
    if [row.get("start") for row in all_window_metrics] != list(range(33)):
        raise Phase4ContractError("primary width-4 metrics start order differs")
    if [row.get("consensus_scorable") for row in all_window_metrics] != [
        start in WIDTH4_CONSENSUS_STARTS for start in range(33)
    ]:
        raise Phase4ContractError("primary width-4 scorability differs")
    _verify_ranked_table(strict_rows, STRICT_KEYS, "strict")
    _verify_ranked_table(edge_rows, EDGE_KEYS, "edge")
    expected_primary_order = sorted(primary_rows, key=lambda row: (-row["score"], row["start"]))
    if primary_rows != expected_primary_order:
        raise Phase4ContractError("primary width-4 rank/tie-break order differs")
    if [row.get("primary_rank") for row in primary_rows] != list(range(1, 26)):
        raise Phase4ContractError("primary width-4 rank fields differ")
    strict_width4 = {row["start"]: row for row in strict_rows if row.get("width") == 4}
    if len(strict_width4) != 25:
        raise Phase4ContractError("strict width-4 compatibility subset is incomplete")
    for primary_row in primary_rows:
        strict_row = strict_width4.get(primary_row["start"])
        if strict_row is None:
            raise Phase4ContractError("strict width-4 row is missing")
        if not math.isclose(strict_row["score"], primary_row["score"], rel_tol=0.0, abs_tol=1e-12):
            raise Phase4ContractError("strict width-4 score differs from primary")
        if strict_row.get("eligible") != primary_row.get("eligible"):
            raise Phase4ContractError("strict width-4 eligibility differs from primary")
        if strict_row.get("width4_primary_rank") != primary_row.get("primary_rank"):
            raise Phase4ContractError("strict width-4 rank differs from primary")


def _verify_ranked_table(
    rows: Any, expected_keys: Sequence[Tuple[int, int]], context: str
) -> None:
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise Phase4ContractError("%s table row membership is incomplete" % context)
    observed_keys = [(row.get("width"), row.get("start")) for row in rows if isinstance(row, Mapping)]
    if len(observed_keys) != len(rows) or set(observed_keys) != set(expected_keys):
        raise Phase4ContractError("%s table key set differs" % context)
    expected_order = sorted(rows, key=lambda row: (-row["score"], row["width"], row["start"]))
    if rows != expected_order:
        raise Phase4ContractError("%s cross-width tie-break order differs" % context)
    if [row.get("cross_width_rank") for row in rows] != list(range(1, len(rows) + 1)):
        raise Phase4ContractError("%s cross-width rank fields differ" % context)
    for width in WIDTHS:
        width_rows = sorted(
            (row for row in rows if row["width"] == width),
            key=lambda row: (-row["score"], row["start"]),
        )
        if [row.get("within_width_rank") for row in width_rows] != list(
            range(1, len(width_rows) + 1)
        ):
            raise Phase4ContractError("%s within-width rank fields differ" % context)


def _bootstrap_contrasts(
    sample_e: np.ndarray,
    sample_k: np.ndarray,
    categories: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = [np.asarray([i for i, value in enumerate(categories) if value == category]) for category in sorted(set(categories))]
    if any(group.size == 0 for group in groups):
        raise Phase4ContractError("empty bootstrap category")
    strict = np.empty((replicates, len(STRICT_KEYS), 4), dtype=np.float64)
    edge = np.empty((replicates, len(EDGE_KEYS), 2), dtype=np.float64)
    rng = np.random.default_rng(seed)
    batch_size = min(64, replicates)
    n = sample_e.shape[0]
    for offset in range(0, replicates, batch_size):
        size = min(batch_size, replicates - offset)
        weights = np.zeros((size, n), dtype=np.float64)
        for group in groups:
            probabilities = np.full(group.size, 1.0 / group.size, dtype=np.float64)
            weights[:, group] = rng.multinomial(group.size, probabilities, size=size)
        replicate_e = (weights @ sample_e) / n
        replicate_k = (weights @ sample_k) / n
        strict[offset : offset + size] = _strict_contrasts_batch(replicate_e, replicate_k)
        edge[offset : offset + size] = _edge_contrasts_batch(replicate_e, replicate_k)
    return strict, edge


def _strict_contrasts(point_e: np.ndarray, point_k: np.ndarray) -> np.ndarray:
    return _strict_contrasts_batch(point_e[None, :], point_k[None, :])[0]


def _strict_contrasts_batch(values_e: np.ndarray, values_k: np.ndarray) -> np.ndarray:
    rows = []
    for width, start in STRICT_KEYS:
        center = FULL_INDEX[(width, start)]
        left1, right1 = FULL_INDEX[(width, start - 1)], FULL_INDEX[(width, start + 1)]
        leftw, rightw = FULL_INDEX[(width, start - width)], FULL_INDEX[(width, start + width)]
        rows.append(
            np.column_stack(
                (
                    values_e[:, center] - 0.5 * (values_e[:, left1] + values_e[:, right1]),
                    values_k[:, center] - 0.5 * (values_k[:, left1] + values_k[:, right1]),
                    values_e[:, center] - 0.5 * (values_e[:, leftw] + values_e[:, rightw]),
                    values_k[:, center] - 0.5 * (values_k[:, leftw] + values_k[:, rightw]),
                )
            )
        )
    return np.stack(rows, axis=1)


def _edge_contrasts(point_e: np.ndarray, point_k: np.ndarray) -> np.ndarray:
    return _edge_contrasts_batch(point_e[None, :], point_k[None, :])[0]


def _edge_contrasts_batch(values_e: np.ndarray, values_k: np.ndarray) -> np.ndarray:
    rows = []
    for width, start in EDGE_KEYS:
        center = FULL_INDEX[(width, start)]
        neighbor_starts = sorted(
            {value for value in (start - 1, start + 1, start - width, start + width) if 0 <= value <= 36 - width}
        )
        neighbors = [FULL_INDEX[(width, value)] for value in neighbor_starts]
        rows.append(
            np.column_stack(
                (
                    values_e[:, center] - np.mean(values_e[:, neighbors], axis=1),
                    values_k[:, center] - np.mean(values_k[:, neighbors], axis=1),
                )
            )
        )
    return np.stack(rows, axis=1)


def _rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-row["score"], row["width"], row["start"]))
    per_width: Dict[int, int] = {}
    for rank, row in enumerate(ranked, 1):
        row["cross_width_rank"] = rank
    for width in WIDTHS:
        width_rows = sorted((row for row in rows if row["width"] == width), key=lambda row: (-row["score"], row["start"]))
        for rank, row in enumerate(width_rows, 1):
            row["within_width_rank"] = rank
    return ranked


def _contrast_payload(point: np.ndarray, replicates: np.ndarray) -> Dict[str, Any]:
    names = ("O_H", "O_K", "F_H", "F_K")
    return {
        name: {
            "point": float(point[index]),
            "se": float(np.std(replicates[:, index], ddof=1)),
            "lower95": float(np.quantile(replicates[:, index], 0.025, method="linear")),
            "upper95": float(np.quantile(replicates[:, index], 0.975, method="linear")),
        }
        for index, name in enumerate(names)
    }


def _edge_payload(point: np.ndarray, replicates: np.ndarray) -> Dict[str, Any]:
    names = ("A_H", "A_K")
    return {
        name: {
            "point": float(point[index]),
            "se": float(np.std(replicates[:, index], ddof=1)),
            "lower95": float(np.quantile(replicates[:, index], 0.025, method="linear")),
            "upper95": float(np.quantile(replicates[:, index], 0.975, method="linear")),
        }
        for index, name in enumerate(names)
    }


def _window(width: int, start: int) -> str:
    return "%d:%d" % (start, start + width - 1)


__all__ = [
    "EDGE_KEYS",
    "STRICT_KEYS",
    "analyze_selector",
    "build_outcome_panel",
    "verify_selector_report",
]
