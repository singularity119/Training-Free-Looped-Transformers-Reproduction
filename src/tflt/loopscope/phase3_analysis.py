"""Deterministic baseline-only analysis for LoopScope Phase 3.

The analyzer consumes validated no-loop choice trajectories only.  It never
loads labels or loop outcomes and it never chooses among SHIFT/FLANK/CONSENSUS;
it publishes each predeclared variant independently for later staged audit.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.schema import SchemaError, canonical_json_bytes
from tflt.loopscope.phase3_pool import partition_starts, validate_trajectory_records
from tflt.loopscope.phase3_schema import (
    BOUNDARY_IDS,
    PHASE3_ANALYSIS_SCHEMA_VERSION,
    PHASE3_CARD_BYTE_SHA256,
    PHASE3_TRAJECTORY_RECORD_SCHEMA_VERSION,
    canonical_identity,
    canonical_record_key,
    make_boundary_record,
    validate_phase3_card,
    validate_trajectory_record,
)


SIGNAL_RECORD_KEYS = frozenset(("identity", "subject", "windows"))
WINDOW_SIGNAL_KEYS = frozenset(
    ("start", "window", "entropy_drop", "kl_rise", "ce_drop")
)
VARIANT_CONTRASTS = {
    "SHIFT": ("O_H", "O_K"),
    "FLANK": ("F_H", "F_K"),
    "CONSENSUS": ("O_H", "O_K", "F_H", "F_K"),
}


def choice_probabilities_from_logits(logits: Sequence[Any]) -> List[float]:
    """Stable four-choice float64 softmax with fail-fast underflow semantics."""

    if isinstance(logits, (str, bytes)) or not isinstance(logits, Sequence) or len(logits) != 4:
        raise SchemaError("raw choice logits must contain exactly four numbers")
    values = [_finite_number(value, "raw choice logit") for value in logits]
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise SchemaError("choice softmax denominator is invalid")
    probabilities = [value / denominator for value in exponentials]
    if any(not math.isfinite(value) or value <= 0.0 for value in probabilities):
        raise SchemaError("choice softmax produced non-finite or zero probability")
    return probabilities


def trajectory_record_from_logits(
    metadata: Mapping[str, Any],
    logits_by_boundary: Mapping[str, Sequence[Any]],
    renderer_provenance: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct a selector-safe trajectory from raw producer-only logits."""

    validate_phase3_card(card)
    _exact_keys(metadata, ("identity", "subject", "split", "prompt_sha256"), "raw metadata")
    if not isinstance(logits_by_boundary, Mapping) or set(logits_by_boundary) != set(BOUNDARY_IDS):
        raise SchemaError("raw trajectory logits must have exactly B_0 through B_28")
    probabilities = {
        boundary_id: choice_probabilities_from_logits(logits_by_boundary[boundary_id])
        for boundary_id in BOUNDARY_IDS
    }
    final = probabilities["B_28"]
    record = {
        "schema_version": PHASE3_TRAJECTORY_RECORD_SCHEMA_VERSION,
        "identity": canonical_identity(metadata["identity"]),
        "subject": str(metadata["subject"]),
        "split": str(metadata["split"]),
        "prompt_sha256": str(metadata["prompt_sha256"]),
        "renderer_provenance": dict(renderer_provenance),
        "boundaries": [
            make_boundary_record(boundary_id, probabilities[boundary_id], final)
            for boundary_id in BOUNDARY_IDS
        ],
    }
    return validate_trajectory_record(record, card)


def window_signals_from_trajectory(
    record: Mapping[str, Any], card: Mapping[str, Any]
) -> Dict[str, Any]:
    """Recompute per-sample width-4 E/K/CEdrop from B_s and B_(s+4)."""

    normalized = validate_trajectory_record(record, card)
    boundaries = normalized["boundaries"]
    windows = []
    width = int(card["loop_outcome_recipe"]["window_width"])
    for start in card["loop_outcome_recipe"]["window_starts"]:
        entry = boundaries[start]
        exit_boundary = boundaries[start + width]
        entropy_drop = entry["choice_entropy"] - exit_boundary["choice_entropy"]
        kl_rise = exit_boundary["kl_to_final"] - entry["kl_to_final"]
        ce_drop = entry["cross_entropy_to_final"] - exit_boundary["cross_entropy_to_final"]
        _assert_close(ce_drop, entropy_drop - kl_rise, card, "sample CEdrop=E-K")
        windows.append(
            {
                "start": start,
                "window": _window_label(start, width),
                "entropy_drop": entropy_drop,
                "kl_rise": kl_rise,
                "ce_drop": ce_drop,
            }
        )
    return {
        "identity": normalized["identity"],
        "subject": normalized["subject"],
        "windows": windows,
    }


def analyze_phase3_trajectories(
    trajectory_records: Sequence[Mapping[str, Any]], card: Mapping[str, Any]
) -> Dict[str, Any]:
    """Analyze records already closed by the explicit manifested entry point.

    This lower-level entry point still enforces 1,531 records and 57 subjects.
    Production orchestration should call :func:`analyze_manifested_phase3_trajectories`
    so source and pool identity bindings are checked in the same call.
    """

    signals = [window_signals_from_trajectory(record, card) for record in trajectory_records]
    return analyze_window_signals(signals, card, enforce_population=True)


def analyze_manifested_phase3_trajectories(
    trajectory_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate signed source/pool closure, then run the frozen production analysis."""

    normalized = validate_trajectory_records(
        trajectory_records, source_manifest, pool_manifest, card
    )
    return analyze_phase3_trajectories(normalized, card)


def analyze_window_signals(
    signal_records: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    replicates: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_population: bool = True,
) -> Dict[str, Any]:
    """Analyze explicit per-sample signals with one joint bootstrap stream.

    ``replicates`` and ``seed`` overrides exist only for deterministic unit and
    verifier fixtures.  Production callers leave them unset and are bound to
    the values in the frozen card.
    """

    validate_phase3_card(card)
    normalized = [_normalize_signal_record(value, card) for value in signal_records]
    normalized.sort(key=canonical_record_key)
    if not normalized:
        raise SchemaError("Phase 3 analysis requires sample records")
    identities = [
        (
            value["identity"]["task"],
            value["identity"]["doc_id"],
            value["identity"]["doc_hash"],
        )
        for value in normalized
    ]
    if len(set(identities)) != len(identities):
        raise SchemaError("Phase 3 signal records contain duplicate identities")
    subject_count = len({value["subject"] for value in normalized})
    if enforce_population:
        if len(normalized) != int(card["task"]["trajectory_sample_count"]):
            raise SchemaError("production analysis requires exactly 1,531 records")
        if subject_count != int(card["task"]["trajectory_subject_count"]):
            raise SchemaError("production analysis requires exactly 57 subjects")
    bootstrap_contract = card["bootstrap"]["trajectory_primary"]
    production_replicates = int(bootstrap_contract["replicates"])
    production_seed = int(bootstrap_contract["seed"])
    if enforce_population and replicates not in (None, production_replicates):
        raise SchemaError("production analysis cannot override bootstrap replicates")
    if enforce_population and seed not in (None, production_seed):
        raise SchemaError("production analysis cannot override bootstrap seed")
    selected_replicates = production_replicates if replicates is None else _positive_int(
        replicates, "bootstrap replicates"
    )
    selected_seed = production_seed if seed is None else _strict_int(seed, "bootstrap seed")
    if selected_replicates < 2:
        raise SchemaError("bootstrap requires at least two replicates for ddof=1")

    e_rows = [[window["entropy_drop"] for window in record["windows"]] for record in normalized]
    k_rows = [[window["kl_rise"] for window in record["windows"]] for record in normalized]
    point_e = _column_means(e_rows)
    point_k = _column_means(k_rows)
    point_ce = [left - right for left, right in zip(point_e, point_k)]

    bootstrap_e = [[] for _ in range(25)]
    bootstrap_k = [[] for _ in range(25)]
    index_digest = hashlib.sha256()
    for indices in iter_subject_bootstrap_indices(
        normalized, replicates=selected_replicates, seed=selected_seed
    ):
        _update_index_digest(index_digest, indices)
        replicate_e, replicate_k = _resampled_window_means(e_rows, k_rows, indices)
        for start in range(25):
            bootstrap_e[start].append(replicate_e[start])
            bootstrap_k[start].append(replicate_k[start])

    window_rows = []
    for start in range(25):
        ce_values = [
            left - right for left, right in zip(bootstrap_e[start], bootstrap_k[start])
        ]
        window_rows.append(
            {
                "start": start,
                "window": _window_label(start, 4),
                "entropy_drop": bootstrap_summary(point_e[start], bootstrap_e[start]),
                "kl_rise": bootstrap_summary(point_k[start], bootstrap_k[start]),
                "ce_drop": bootstrap_summary(point_ce[start], ce_values),
            }
        )

    partitions = partition_starts(card)
    variants = {}
    for variant in ("SHIFT", "FLANK", "CONSENSUS"):
        rows, bootstrap_contrasts = _variant_rows(
            variant,
            point_e,
            point_k,
            bootstrap_e,
            bootstrap_k,
            card,
        )
        scopes = {
            "deployment": _rank_variant_scope(
                variant, rows, bootstrap_contrasts, tuple(row["start"] for row in rows), card
            ),
            "retrospective": _rank_variant_scope(
                variant,
                rows,
                bootstrap_contrasts,
                tuple(start for start in partitions["historical_13"] if start in {row["start"] for row in rows}),
                card,
            ),
            "blind_enrichment": _rank_variant_scope(
                variant,
                rows,
                bootstrap_contrasts,
                tuple(start for start in partitions["blind_12"] if start in {row["start"] for row in rows}),
                card,
            ),
        }
        variants[variant] = {
            "support_starts": [row["start"] for row in rows],
            "outside_support_starts": [
                start for start in range(25) if start not in {row["start"] for row in rows}
            ],
            "outside_support_result": card["variants"]["outside_support"],
            "windows": rows,
            "scopes": scopes,
        }

    return {
        "schema_version": PHASE3_ANALYSIS_SCHEMA_VERSION,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "analysis_role": "baseline_only_no_outcome_variant_independent",
        "record_count": len(normalized),
        "subject_count": subject_count,
        "bootstrap": {
            "method": bootstrap_contract["method"],
            "replicates": selected_replicates,
            "seed": selected_seed,
            "prng": bootstrap_contract["prng"],
            "random_api": bootstrap_contract["random_api"],
            "stream_count": 1,
            "iteration_order": bootstrap_contract["iteration_order"],
            "joint_index_reuse": bootstrap_contract["joint_index_reuse"],
            "index_stream_sha256": index_digest.hexdigest(),
            "fixture_override": not (
                selected_replicates == production_replicates and selected_seed == production_seed
            ),
        },
        "window_metrics": window_rows,
        "variants": variants,
        "variant_selection_performed": False,
        "outcome_fields_consumed": False,
    }


def verify_analysis_report(
    signal_records: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    replicates: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_population: bool = True,
) -> None:
    expected = analyze_window_signals(
        signal_records,
        card,
        replicates=replicates,
        seed=seed,
        enforce_population=enforce_population,
    )
    try:
        matches = canonical_json_bytes(report) == canonical_json_bytes(expected)
    except (TypeError, ValueError) as exc:
        raise SchemaError("Phase 3 analysis report is not canonical finite JSON") from exc
    if not matches:
        raise SchemaError("Phase 3 analysis report differs from raw deterministic recomputation")


def iter_subject_bootstrap_indices(
    records: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> Iterator[Tuple[int, ...]]:
    """Yield the frozen replicate-major/subject-major/draw-major index stream."""

    count = _positive_int(replicates, "bootstrap replicates")
    generator = random.Random(_strict_int(seed, "bootstrap seed"))
    by_subject: DefaultDict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        subject = str(record["subject"])
        by_subject[subject].append(index)
    for _ in range(count):
        indices: List[int] = []
        for subject in sorted(by_subject):
            subject_indices = by_subject[subject]
            subject_size = len(subject_indices)
            if subject_size < 1:
                raise SchemaError("bootstrap subject has no records")
            for _draw in range(subject_size):
                indices.append(subject_indices[generator.randrange(subject_size)])
        yield tuple(indices)


def linear_percentile(values: Sequence[Any], quantile: float) -> float:
    numbers = sorted(_finite_number(value, "percentile value") for value in values)
    if not numbers:
        raise SchemaError("percentile requires values")
    q = _finite_number(quantile, "percentile quantile")
    if q < 0.0 or q > 1.0:
        raise SchemaError("percentile quantile must lie in [0,1]")
    position = q * (len(numbers) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return numbers[lower]
    fraction = position - lower
    return numbers[lower] * (1.0 - fraction) + numbers[upper] * fraction


def sample_standard_deviation(values: Sequence[Any]) -> float:
    numbers = [_finite_number(value, "standard-deviation value") for value in values]
    if len(numbers) < 2:
        raise SchemaError("sample standard deviation requires at least two values")
    mean = sum(numbers) / len(numbers)
    return math.sqrt(sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1))


def bootstrap_summary(point: Any, replicate_values: Sequence[Any]) -> Dict[str, Any]:
    point_value = _finite_number(point, "bootstrap point")
    values = [_finite_number(value, "bootstrap replicate") for value in replicate_values]
    return {
        "point": point_value,
        "standard_error": sample_standard_deviation(values),
        "percentile_95_ci": [
            linear_percentile(values, 0.025),
            linear_percentile(values, 0.975),
        ],
    }


def variant_support(card: Mapping[str, Any], variant: str) -> Tuple[int, ...]:
    if variant not in VARIANT_CONTRASTS:
        raise SchemaError("unknown Phase 3 variant")
    return tuple(int(value) for value in card["variants"][variant]["support_starts"])


def variant_comparison_starts(variant: str, start: int, width: int = 4) -> Tuple[int, ...]:
    if variant == "SHIFT":
        result = (start - 1, start + 1)
    elif variant == "FLANK":
        result = (start - width, start + width)
    elif variant == "CONSENSUS":
        result = (start - 1, start + 1, start - width, start + width)
    else:
        raise SchemaError("unknown Phase 3 variant")
    if any(value < 0 or value > 24 for value in result):
        raise SchemaError("variant comparison lies outside the width-4 universe")
    return result


def rank_variant_scope(
    variant: str,
    rows: Sequence[Mapping[str, Any]],
    bootstrap_contrasts: Mapping[int, Mapping[str, Sequence[float]]],
    scope_starts: Sequence[int],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    """Public deterministic rank/selection helper for contract-level verification."""

    validate_phase3_card(card)
    return _rank_variant_scope(variant, rows, bootstrap_contrasts, scope_starts, card)


def _variant_rows(
    variant: str,
    point_e: Sequence[float],
    point_k: Sequence[float],
    bootstrap_e: Sequence[Sequence[float]],
    bootstrap_k: Sequence[Sequence[float]],
    card: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, List[float]]]]:
    rows = []
    all_bootstrap_contrasts: Dict[int, Dict[str, List[float]]] = {}
    for start in variant_support(card, variant):
        point_contrasts = _contrasts_for_start(start, point_e, point_k)
        bootstrap_contrasts = _bootstrap_contrasts_for_start(
            start, bootstrap_e, bootstrap_k
        )
        all_bootstrap_contrasts[start] = bootstrap_contrasts
        contrasts = {}
        for name in VARIANT_CONTRASTS[variant]:
            contrasts[name] = bootstrap_summary(
                point_contrasts[name], bootstrap_contrasts[name]
            )
        comparisons = variant_comparison_starts(variant, start, 4)
        failures = []
        if not point_e[start] > 0.0:
            failures.append("center_entropy_drop_not_strictly_positive")
        if not point_k[start] > 0.0:
            failures.append("center_kl_rise_not_strictly_positive")
        for comparison in comparisons:
            if not point_e[comparison] < 0.0:
                failures.append("comparison_%d_entropy_drop_not_strictly_negative" % comparison)
            if not point_k[comparison] < 0.0:
                failures.append("comparison_%d_kl_rise_not_strictly_negative" % comparison)
        for name, summary in contrasts.items():
            if not summary["percentile_95_ci"][0] > 0.0:
                failures.append("%s_ci_lower_not_strictly_positive" % name)
        score_components = {
            name: contrasts[name]["point"] / max(contrasts[name]["standard_error"], 1e-12)
            for name in VARIANT_CONTRASTS[variant]
        }
        rows.append(
            {
                "start": start,
                "window": _window_label(start, 4),
                "scorable": True,
                "comparison_starts": list(comparisons),
                "center": {
                    "entropy_drop": point_e[start],
                    "kl_rise": point_k[start],
                    "ce_drop": point_e[start] - point_k[start],
                },
                "comparisons": [
                    {
                        "start": comparison,
                        "entropy_drop": point_e[comparison],
                        "kl_rise": point_k[comparison],
                    }
                    for comparison in comparisons
                ],
                "contrasts": contrasts,
                "score_components": score_components,
                "score": min(score_components.values()),
                "point_eligible": not failures,
                "eligibility_failures": failures,
            }
        )
    return rows, all_bootstrap_contrasts


def _rank_variant_scope(
    variant: str,
    rows: Sequence[Mapping[str, Any]],
    bootstrap_contrasts: Mapping[int, Mapping[str, Sequence[float]]],
    scope_starts: Sequence[int],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    by_start = {int(row["start"]): row for row in rows}
    starts = tuple(int(value) for value in scope_starts)
    if len(set(starts)) != len(starts) or any(start not in by_start for start in starts):
        raise SchemaError("ranking scope contains duplicate or unsupported starts")
    ranked = sorted(starts, key=lambda start: (-float(by_start[start]["score"]), start))
    ranking = [
        {
            "rank": index + 1,
            "start": start,
            "window": _window_label(start, 4),
            "score": by_start[start]["score"],
            "point_eligible": by_start[start]["point_eligible"],
        }
        for index, start in enumerate(ranked)
    ]
    candidates = [start for start in ranked if bool(by_start[start]["point_eligible"])]
    base = {
        "scope_starts": list(starts),
        "scorable_count": len(starts),
        "published_ranking": ranking,
        "point_eligible_starts": sorted(candidates),
        "point_top1_start": None,
        "point_top1_window": None,
        "selection_frequency": None,
        "window_decision": "ABSTAIN_NO_POINT_ELIGIBLE",
    }
    if not candidates:
        return base
    point_ranked = sorted(candidates, key=lambda start: (-float(by_start[start]["score"]), start))
    point_top = point_ranked[0]
    if len(point_ranked) > 1 and math.isclose(
        float(by_start[point_top]["score"]),
        float(by_start[point_ranked[1]]["score"]),
        rel_tol=float(card["ranking"]["point_top_tie"]["rel_tolerance"]),
        abs_tol=float(card["ranking"]["point_top_tie"]["abs_tolerance"]),
    ):
        base["window_decision"] = card["ranking"]["point_top_tie"]["result"]
        return base
    replicate_count = len(next(iter(bootstrap_contrasts[point_top].values())))
    wins = 0
    for replicate_index in range(replicate_count):
        replicate_scores = {}
        for start in candidates:
            components = []
            for name in VARIANT_CONTRASTS[variant]:
                denominator = max(
                    float(by_start[start]["contrasts"][name]["standard_error"]), 1e-12
                )
                components.append(
                    float(bootstrap_contrasts[start][name][replicate_index]) / denominator
                )
            replicate_scores[start] = min(components)
        winner = sorted(candidates, key=lambda start: (-replicate_scores[start], start))[0]
        if winner == point_top:
            wins += 1
    frequency = wins / replicate_count
    threshold = float(card["ranking"]["selection_frequency"]["threshold"])
    base.update(
        {
            "point_top1_start": point_top,
            "point_top1_window": _window_label(point_top, 4),
            "selection_frequency": frequency,
            "window_decision": (
                "SELECTED" if frequency >= threshold else "ABSTAIN_LOW_SELECTION_FREQUENCY"
            ),
        }
    )
    return base


def _contrasts_for_start(
    start: int, entropy: Sequence[float], divergence: Sequence[float]
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if 1 <= start <= 23:
        result["O_H"] = entropy[start] - 0.5 * (entropy[start - 1] + entropy[start + 1])
        result["O_K"] = divergence[start] - 0.5 * (
            divergence[start - 1] + divergence[start + 1]
        )
    if 4 <= start <= 20:
        result["F_H"] = entropy[start] - 0.5 * (entropy[start - 4] + entropy[start + 4])
        result["F_K"] = divergence[start] - 0.5 * (
            divergence[start - 4] + divergence[start + 4]
        )
    return result


def _bootstrap_contrasts_for_start(
    start: int,
    bootstrap_e: Sequence[Sequence[float]],
    bootstrap_k: Sequence[Sequence[float]],
) -> Dict[str, List[float]]:
    count = len(bootstrap_e[0])
    result: Dict[str, List[float]] = {}
    if 1 <= start <= 23:
        result["O_H"] = [
            bootstrap_e[start][index]
            - 0.5 * (bootstrap_e[start - 1][index] + bootstrap_e[start + 1][index])
            for index in range(count)
        ]
        result["O_K"] = [
            bootstrap_k[start][index]
            - 0.5 * (bootstrap_k[start - 1][index] + bootstrap_k[start + 1][index])
            for index in range(count)
        ]
    if 4 <= start <= 20:
        result["F_H"] = [
            bootstrap_e[start][index]
            - 0.5 * (bootstrap_e[start - 4][index] + bootstrap_e[start + 4][index])
            for index in range(count)
        ]
        result["F_K"] = [
            bootstrap_k[start][index]
            - 0.5 * (bootstrap_k[start - 4][index] + bootstrap_k[start + 4][index])
            for index in range(count)
        ]
    return result


def _normalize_signal_record(value: Mapping[str, Any], card: Mapping[str, Any]) -> Dict[str, Any]:
    _exact_keys(value, SIGNAL_RECORD_KEYS, "Phase 3 signal record")
    identity = canonical_identity(value["identity"])
    subject = _nonempty_string(value["subject"], "signal subject")
    windows = value["windows"]
    if not isinstance(windows, list) or len(windows) != 25:
        raise SchemaError("signal record must contain all 25 width-4 starts")
    normalized_windows = []
    for expected_start, window in enumerate(windows):
        _exact_keys(window, WINDOW_SIGNAL_KEYS, "signal window")
        start = _strict_int(window["start"], "signal start")
        if start != expected_start or window["window"] != _window_label(start, 4):
            raise SchemaError("signal windows must be ordered starts 0..24")
        entropy = _finite_number(window["entropy_drop"], "signal entropy drop")
        divergence = _finite_number(window["kl_rise"], "signal KL rise")
        ce_drop = _finite_number(window["ce_drop"], "signal CE drop")
        _assert_close(ce_drop, entropy - divergence, card, "signal CEdrop=E-K")
        normalized_windows.append(
            {
                "start": start,
                "window": _window_label(start, 4),
                "entropy_drop": entropy,
                "kl_rise": divergence,
                "ce_drop": ce_drop,
            }
        )
    return {"identity": identity, "subject": subject, "windows": normalized_windows}


def _column_means(rows: Sequence[Sequence[float]]) -> List[float]:
    count = len(rows)
    return [sum(row[index] for row in rows) / count for index in range(25)]


def _resampled_window_means(
    e_rows: Sequence[Sequence[float]],
    k_rows: Sequence[Sequence[float]],
    indices: Sequence[int],
) -> Tuple[List[float], List[float]]:
    if not indices:
        raise SchemaError("bootstrap replicate contains no indices")
    e_sums = [0.0] * 25
    k_sums = [0.0] * 25
    for index in indices:
        e_row = e_rows[index]
        k_row = k_rows[index]
        for start in range(25):
            e_sums[start] += e_row[start]
            k_sums[start] += k_row[start]
    denominator = len(indices)
    return (
        [value / denominator for value in e_sums],
        [value / denominator for value in k_sums],
    )


def _update_index_digest(digest: Any, indices: Sequence[int]) -> None:
    digest.update(struct.pack(">I", len(indices)))
    for index in indices:
        digest.update(struct.pack(">I", int(index)))


def _window_label(start: int, width: int) -> str:
    return "%d:%d" % (start, start + width - 1)


def _assert_close(observed: float, expected: float, card: Mapping[str, Any], context: str) -> None:
    if not math.isclose(
        observed,
        expected,
        rel_tol=float(card["numeric_contract"]["recomputed_scalar_rel_tolerance"]),
        abs_tol=float(card["numeric_contract"]["recomputed_scalar_abs_tolerance"]),
    ):
        raise SchemaError("%s mismatch" % context)


def _exact_keys(value: Any, expected: Iterable[str], context: str) -> None:
    if not isinstance(value, Mapping):
        raise SchemaError("%s must be an object" % context)
    actual = set(value)
    frozen = set(expected)
    if actual != frozen:
        raise SchemaError(
            "%s fields differ; missing=%s extra=%s"
            % (context, sorted(frozen - actual), sorted(actual - frozen))
        )


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError("%s must be a finite number" % context)
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError("%s must be a finite number" % context)
    return result


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError("%s must be an integer" % context)
    return int(value)


def _positive_int(value: Any, context: str) -> int:
    result = _strict_int(value, context)
    if result < 1:
        raise SchemaError("%s must be positive" % context)
    return result


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("%s must be a non-empty string" % context)
    return value.strip()


__all__ = [
    "analyze_manifested_phase3_trajectories",
    "analyze_phase3_trajectories",
    "analyze_window_signals",
    "bootstrap_summary",
    "choice_probabilities_from_logits",
    "iter_subject_bootstrap_indices",
    "linear_percentile",
    "rank_variant_scope",
    "sample_standard_deviation",
    "trajectory_record_from_logits",
    "variant_comparison_starts",
    "variant_support",
    "verify_analysis_report",
    "window_signals_from_trajectory",
]
