"""Fail-closed Phase 5 formal selector and panel artifact builders.

The mathematical selector is frozen in :mod:`phase5_schema`.  This module is
the Gate B production adapter around those helpers: it closes trajectory
membership, removes the permissive synthetic-fixture surface, records the
exact bootstrap index stream, explains eligibility failures, and packages
hash-closed selector/panel artifacts.  It is deliberately standard-library
only and never reads model outputs, labels, correctness, or test outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase5_schema import (
    ABSTAIN_LABELS,
    CONSENSUS_STARTS,
    WINDOW_STARTS,
    Phase5ContractError,
    analyze_signal_samples,
    canonical_json_bytes,
    construct_panel,
    semantic_sha256,
    trajectory_window_signals,
    validate_card,
    validate_registry,
    validate_trajectory_record,
)


FORMAL_RECORD_COUNT = 1531
FORMAL_SUBJECT_COUNT = 57
FORMAL_BOOTSTRAP_REPLICATES = 2000
FORMAL_BOOTSTRAP_SEED = 20260722

BOOTSTRAP_DIGEST_SCHEMA_VERSION = "loopscope.phase5.bootstrap-index-digest.v1"
SELECTOR_REPORT_SCHEMA_VERSION = "loopscope.phase5.selector-report-artifact.v1"
OUTCOME_PANEL_SCHEMA_VERSION = "loopscope.phase5.outcome-panel.v1"
SELECTOR_FREEZE_SCHEMA_VERSION = "loopscope.phase5.selector-freeze.v1"

_IDENTITY_KEYS = frozenset(("task", "doc_id", "doc_hash"))
_MEMBERSHIP_KEYS = frozenset(("identity", "subject", "split", "prompt_sha256"))
_MEMBERSHIP_WITH_ORDINAL_KEYS = frozenset(_MEMBERSHIP_KEYS | {"ordinal"})
_SIGNAL_KEYS = frozenset(("identity", "subject", "E", "K"))
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTRAST_NAMES = ("O_H", "O_K", "F_H", "F_K")


def _exact_keys(value: Any, expected: Sequence[str] | frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise Phase5ContractError("%s must be an object" % label)
    actual = set(value)
    frozen = set(expected)
    if actual != frozen:
        raise Phase5ContractError(
            "%s keys differ; missing=%s extra=%s"
            % (label, sorted(frozen - actual), sorted(actual - frozen))
        )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase5ContractError("%s must be a non-empty string" % label)
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Phase5ContractError("%s must be a lowercase SHA256" % label)
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase5ContractError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise Phase5ContractError("%s must be finite" % label)
    return result


def _canonical_mapping(value: Mapping[str, Any], label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase5ContractError("%s must be an object" % label)
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase5ContractError("%s must be strict canonical JSON" % label) from exc


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _attach_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["manifest_sha256"] = semantic_sha256(payload)
    return payload


def _require_manifest(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("manifest_sha256") != semantic_sha256(payload):
        raise Phase5ContractError("%s manifest_sha256 mismatch" % label)


def canonical_identity(identity: Mapping[str, Any]) -> Dict[str, str]:
    """Return the exact non-empty Phase 5 identity tuple as a JSON object."""

    _exact_keys(identity, _IDENTITY_KEYS, "identity")
    return {
        "task": _nonempty_string(identity["task"], "identity.task"),
        "doc_id": _nonempty_string(identity["doc_id"], "identity.doc_id"),
        "doc_hash": _sha256(identity["doc_hash"], "identity.doc_hash"),
    }


def canonical_identity_string(identity: Mapping[str, Any]) -> str:
    """Stable string consumed by the frozen Gate A bootstrap helper."""

    return canonical_json_bytes(canonical_identity(identity)).decode("utf-8")


def _trajectory_projection(record: Mapping[str, Any]) -> Dict[str, Any]:
    validate_trajectory_record(record)
    return {
        "identity": canonical_identity(record["identity"]),
        "subject": _nonempty_string(record["subject"], "trajectory.subject"),
        "split": "validation",
        "prompt_sha256": _sha256(record["prompt_sha256"], "trajectory.prompt_sha256"),
    }


def _membership_projection(value: Mapping[str, Any], ordinal: int) -> Tuple[Dict[str, Any], bool]:
    """Normalize one of the two explicit membership row representations.

    An identity-only list is useful to independently close order.  A full row
    additionally closes subject, split, and prompt hash.  Formal manifests may
    add only the deterministic ``ordinal`` field; arbitrary extra metadata is
    rejected rather than silently ignored.
    """

    if not isinstance(value, Mapping):
        raise Phase5ContractError("membership row must be an object")
    if set(value) == _IDENTITY_KEYS:
        return {"identity": canonical_identity(value)}, False
    if set(value) == _MEMBERSHIP_WITH_ORDINAL_KEYS:
        if isinstance(value["ordinal"], bool) or not isinstance(value["ordinal"], int):
            raise Phase5ContractError("membership ordinal must be an integer")
        if int(value["ordinal"]) != ordinal:
            raise Phase5ContractError("membership ordinal/order differs")
    elif set(value) != _MEMBERSHIP_KEYS:
        _exact_keys(value, _MEMBERSHIP_KEYS, "membership row")
    projection = {
        "identity": canonical_identity(value["identity"]),
        "subject": _nonempty_string(value["subject"], "membership.subject"),
        "split": str(value["split"]),
        "prompt_sha256": _sha256(value["prompt_sha256"], "membership.prompt_sha256"),
    }
    if projection["split"] != "validation":
        raise Phase5ContractError("membership split must be validation")
    return projection, True


def selector_inputs_from_trajectories(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_membership: Optional[Sequence[Mapping[str, Any]]] = None,
    expected_count: Optional[int] = None,
    expected_subject_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Validate closed-world trajectories and derive exact 33-entry E/K rows."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise Phase5ContractError("selector trajectory records must be a non-empty sequence")
    if expected_count is not None and len(records) != expected_count:
        raise Phase5ContractError("trajectory record count differs from expectation")
    if expected_membership is not None and len(expected_membership) != len(records):
        raise Phase5ContractError("trajectory and membership counts differ")

    result: List[Dict[str, Any]] = []
    seen = set()
    subjects = set()
    for ordinal, record in enumerate(records):
        projection = _trajectory_projection(record)
        identity_string = canonical_identity_string(projection["identity"])
        if identity_string in seen:
            raise Phase5ContractError("duplicate trajectory identity")
        seen.add(identity_string)
        subjects.add(projection["subject"])
        if expected_membership is not None:
            expected, closes_full_row = _membership_projection(expected_membership[ordinal], ordinal)
            if expected["identity"] != projection["identity"]:
                raise Phase5ContractError("trajectory membership identity/order differs")
            if closes_full_row and expected != projection:
                raise Phase5ContractError("trajectory membership metadata/order differs")

        signals = trajectory_window_signals(record)
        windows = signals.get("windows")
        if not isinstance(windows, list) or len(windows) != len(WINDOW_STARTS):
            raise Phase5ContractError("trajectory signals must cover exactly 33 starts")
        e_values: List[float] = []
        k_values: List[float] = []
        for start, row in zip(WINDOW_STARTS, windows):
            if not isinstance(row, Mapping):
                raise Phase5ContractError("trajectory signal row must be an object")
            if row.get("start") != start or row.get("window") != "%d:%d" % (start, start + 3):
                raise Phase5ContractError("trajectory signal start/window order differs")
            e_values.append(_finite(row.get("E"), "trajectory signal E"))
            k_values.append(_finite(row.get("K"), "trajectory signal K"))
        result.append(
            {
                "identity": identity_string,
                "subject": projection["subject"],
                "E": e_values,
                "K": k_values,
            }
        )
    if expected_subject_count is not None and len(subjects) != expected_subject_count:
        raise Phase5ContractError("trajectory subject count differs from expectation")
    return result


def _normalize_signal_samples(samples: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
        raise Phase5ContractError("selector signals must be a non-empty sequence")
    normalized = []
    seen = set()
    for sample in samples:
        _exact_keys(sample, _SIGNAL_KEYS, "selector signal sample")
        identity = _nonempty_string(sample["identity"], "selector signal identity")
        subject = _nonempty_string(sample["subject"], "selector signal subject")
        if identity in seen:
            raise Phase5ContractError("selector signal identities must be unique")
        seen.add(identity)
        e_values = [_finite(value, "selector signal E") for value in sample["E"]]
        k_values = [_finite(value, "selector signal K") for value in sample["K"]]
        if len(e_values) != len(WINDOW_STARTS) or len(k_values) != len(WINDOW_STARTS):
            raise Phase5ContractError("selector signals must cover exactly 33 starts")
        normalized.append({"identity": identity, "subject": subject, "E": e_values, "K": k_values})
    return sorted(normalized, key=lambda row: (row["subject"], row["identity"]))


def bootstrap_index_digest(
    samples: Sequence[Mapping[str, Any]],
    *,
    replicates: int = FORMAL_BOOTSTRAP_REPLICATES,
    seed: int = FORMAL_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Hash the exact Gate A ``Random/randrange`` joint bootstrap draw stream.

    The digest is the SHA256 of a canonical JSON array containing, for every
    replicate, the selected global indices in sorted ``(subject, identity)``
    order.  It is streamed so the formal 2000 x 1531 draw matrix need not be
    retained in memory.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise Phase5ContractError("bootstrap replicates must be an integer >=2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Phase5ContractError("bootstrap seed must be an integer")
    normalized = _normalize_signal_samples(samples)
    groups: Dict[str, List[int]] = {}
    for index, row in enumerate(normalized):
        groups.setdefault(row["subject"], []).append(index)
    subject_order = sorted(groups)
    rng = random.Random(seed)
    digest = hashlib.sha256()
    digest.update(b"[")
    for replicate_index in range(replicates):
        if replicate_index:
            digest.update(b",")
        drawn: List[int] = []
        for subject in subject_order:
            indices = groups[subject]
            drawn.extend(indices[rng.randrange(len(indices))] for _ in indices)
        digest.update(canonical_json_bytes(drawn))
    digest.update(b"]")
    return {
        "schema_version": BOOTSTRAP_DIGEST_SCHEMA_VERSION,
        "replicates": replicates,
        "seed": seed,
        "prng": "python_stdlib_random.Random",
        "random_api": "randrange(n_s)",
        "normalized_order": "subject_ascending_identity_ascending",
        "normalized_identity_sha256": _payload_sha256(
            [row["identity"] for row in normalized]
        ),
        "subject_order_sha256": _payload_sha256(subject_order),
        "subject_sizes": {subject: len(groups[subject]) for subject in subject_order},
        "draw_representation": "canonical_json_array_of_global_indices_in_normalized_order",
        "draw_index_sha256": digest.hexdigest(),
    }


def _point_means(samples: Sequence[Mapping[str, Any]]) -> Tuple[List[float], List[float]]:
    normalized = _normalize_signal_samples(samples)
    count = len(normalized)
    point_e = [math.fsum(row["E"][start] for row in normalized) / count for start in WINDOW_STARTS]
    point_k = [math.fsum(row["K"][start] for row in normalized) / count for start in WINDOW_STARTS]
    return point_e, point_k


def _eligibility_rows(
    report: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    point_e, point_k = _point_means(samples)
    ranking = report.get("published_ranking")
    if not isinstance(ranking, list) or len(ranking) != len(CONSENSUS_STARTS):
        raise Phase5ContractError("selector report must contain exactly 25 CONSENSUS rows")
    observed_starts = [row.get("start") for row in ranking if isinstance(row, Mapping)]
    if set(observed_starts) != set(CONSENSUS_STARTS) or len(observed_starts) != len(set(observed_starts)):
        raise Phase5ContractError("selector ranking start closure differs")
    expected_order = sorted(ranking, key=lambda row: (-_finite(row.get("score"), "score"), row["start"]))
    if [row["start"] for row in ranking] != [row["start"] for row in expected_order]:
        raise Phase5ContractError("selector ranking order differs")

    enriched = []
    for raw in ranking:
        _exact_keys(
            raw,
            frozenset(("start", "window", "E", "K", "contrasts", "score", "point_eligible")),
            "selector ranking row",
        )
        start = raw["start"]
        if isinstance(start, bool) or not isinstance(start, int) or start not in CONSENSUS_STARTS:
            raise Phase5ContractError("selector ranking start differs")
        if raw["window"] != "%d:%d" % (start, start + 3):
            raise Phase5ContractError("selector ranking window differs")
        if not math.isclose(_finite(raw["E"], "ranking E"), point_e[start], rel_tol=1e-12, abs_tol=1e-12):
            raise Phase5ContractError("selector ranking E differs from input recomputation")
        if not math.isclose(_finite(raw["K"], "ranking K"), point_k[start], rel_tol=1e-12, abs_tol=1e-12):
            raise Phase5ContractError("selector ranking K differs from input recomputation")
        comparisons = (start - 1, start + 1, start - 4, start + 4)
        comparison_e = {str(index): point_e[index] < 0.0 for index in comparisons}
        comparison_k = {str(index): point_k[index] < 0.0 for index in comparisons}
        contrasts = raw["contrasts"]
        _exact_keys(contrasts, frozenset(_CONTRAST_NAMES), "selector contrasts")
        lower95 = {}
        for name in _CONTRAST_NAMES:
            payload = contrasts[name]
            _exact_keys(
                payload,
                frozenset(("point", "standard_error", "lower95", "upper95")),
                "selector contrast %s" % name,
            )
            for key in ("point", "standard_error", "lower95", "upper95"):
                _finite(payload[key], "selector contrast %s.%s" % (name, key))
            lower95[name] = float(payload["lower95"]) > 0.0
        checks = {
            "center_E_positive": point_e[start] > 0.0,
            "center_K_positive": point_k[start] > 0.0,
            "comparison_E_negative": comparison_e,
            "comparison_K_negative": comparison_k,
            "contrast_lower95_positive": lower95,
        }
        failures = []
        if not checks["center_E_positive"]:
            failures.append("CENTER_E_NOT_STRICTLY_POSITIVE")
        if not checks["center_K_positive"]:
            failures.append("CENTER_K_NOT_STRICTLY_POSITIVE")
        for index in comparisons:
            if not comparison_e[str(index)]:
                failures.append("COMPARISON_S%d_E_NOT_STRICTLY_NEGATIVE" % index)
            if not comparison_k[str(index)]:
                failures.append("COMPARISON_S%d_K_NOT_STRICTLY_NEGATIVE" % index)
        for name in _CONTRAST_NAMES:
            if not lower95[name]:
                failures.append("%s_LOWER95_NOT_STRICTLY_POSITIVE" % name)
        eligible = not failures
        if type(raw["point_eligible"]) is not bool or raw["point_eligible"] != eligible:
            raise Phase5ContractError("selector point eligibility differs from recomputation")
        item = dict(raw)
        item["scorable"] = True
        item["eligibility_checks"] = checks
        item["failure_reasons"] = failures
        enriched.append(item)
    return enriched


def _population_closure(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    projections = [_trajectory_projection(record) for record in records]
    return {
        "record_count": len(projections),
        "subject_count": len({row["subject"] for row in projections}),
        "ordered_identity_sha256": _payload_sha256([row["identity"] for row in projections]),
        "ordered_prompt_sha256": _payload_sha256([row["prompt_sha256"] for row in projections]),
        "ordered_trajectory_record_sha256": _payload_sha256(
            [_payload_sha256(record) for record in records]
        ),
    }


def _formal_parameters(formal: bool, replicates: int, seed: int) -> Tuple[int, int]:
    if formal and (replicates != FORMAL_BOOTSTRAP_REPLICATES or seed != FORMAL_BOOTSTRAP_SEED):
        raise Phase5ContractError("formal selector must use R=2000 and seed=20260722")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise Phase5ContractError("selector replicates must be an integer >=2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Phase5ContractError("selector seed must be an integer")
    return replicates, seed


def build_selector_artifacts(
    records: Sequence[Mapping[str, Any]],
    *,
    card: Mapping[str, Any],
    registry: Mapping[str, Any],
    expected_membership: Optional[Sequence[Mapping[str, Any]]],
    bindings: Mapping[str, Any],
    formal: bool = True,
    replicates: int = FORMAL_BOOTSTRAP_REPLICATES,
    seed: int = FORMAL_BOOTSTRAP_SEED,
) -> Dict[str, Dict[str, Any]]:
    """Build deterministic selector report, panel, and freeze payloads.

    ``formal=True`` is the production entry point and cannot override the
    population shape or bootstrap stream.  Smaller non-formal fixtures are
    supported only so pure-Python tests can exercise the artifact contract.
    """

    validate_card(card)
    validate_registry(registry)
    replicates, seed = _formal_parameters(formal, replicates, seed)
    normalized_bindings = _canonical_mapping(bindings, "selector bindings")
    inputs = selector_inputs_from_trajectories(
        records,
        expected_membership=expected_membership,
        expected_count=FORMAL_RECORD_COUNT if formal else None,
        expected_subject_count=FORMAL_SUBJECT_COUNT if formal else None,
    )
    raw_report = analyze_signal_samples(inputs, replicates=replicates, seed=seed)
    if raw_report.get("geometry_fields_consumed") is not False:
        raise Phase5ContractError("selector consumed geometry fields")
    if raw_report.get("outcome_fields_consumed") is not False:
        raise Phase5ContractError("selector consumed outcome fields")
    ranking = _eligibility_rows(raw_report, inputs)
    digest = bootstrap_index_digest(inputs, replicates=replicates, seed=seed)
    population = _population_closure(records)

    selector_report: Dict[str, Any] = {
        "schema_version": SELECTOR_REPORT_SCHEMA_VERSION,
        "selector": "ENTROPY_KL_CONSENSUS_WIDTH4_V1",
        "formal": formal,
        "population": population,
        "bootstrap": digest,
        "published_ranking": ranking,
        "eligible_count": raw_report["eligible_count"],
        "selected_window": raw_report["selected_window"],
        "point_top1_window": raw_report["point_top1_window"],
        "selection_frequency": raw_report["selection_frequency"],
        "window_decision": raw_report["window_decision"],
        "geometry_fields_consumed": False,
        "outcome_fields_consumed": False,
        "bindings": normalized_bindings,
    }
    _attach_manifest(selector_report)
    validate_selector_report_payload(selector_report)

    panel = construct_panel(
        ranking,
        selected_window=selector_report["selected_window"],
        known_windows=registry["blind_group_exclusion_windows"],
    )
    expected_panel = construct_panel(
        selector_report["published_ranking"],
        selected_window=selector_report["selected_window"],
        known_windows=registry["blind_group_exclusion_windows"],
    )
    if canonical_json_bytes(panel) != canonical_json_bytes(expected_panel):
        raise Phase5ContractError("panel construction is not deterministic")
    outcome_panel: Dict[str, Any] = {
        "schema_version": OUTCOME_PANEL_SCHEMA_VERSION,
        "selector_report_manifest_sha256": selector_report["manifest_sha256"],
        "known_outcome_registry_manifest_sha256": registry["manifest_sha256"],
        "baseline": "no-loop",
        "fixed_comparator": "15:18",
        "raw_high3": panel["raw_high3"],
        "panel_high3": panel["panel_high3"],
        "selected_inclusive_replacement": panel["selected_inclusive_replacement"],
        "selected_is_known_comparator": panel["selected_is_known_comparator"],
        "blind_low3": panel["blind_low3"],
        "unique_cells": panel["unique_cells"],
        "unique_cell_count": panel["unique_cell_count"],
        "bindings": normalized_bindings,
    }
    _attach_manifest(outcome_panel)
    validate_outcome_panel_payload(outcome_panel, selector_report, registry)

    selector_freeze: Dict[str, Any] = {
        "schema_version": SELECTOR_FREEZE_SCHEMA_VERSION,
        "selector": "ENTROPY_KL_CONSENSUS_WIDTH4_V1",
        "formal": formal,
        "replicates": replicates,
        "seed": seed,
        "selector_report_manifest_sha256": selector_report["manifest_sha256"],
        "outcome_panel_manifest_sha256": outcome_panel["manifest_sha256"],
        "window_decision": selector_report["window_decision"],
        "selected_window": selector_report["selected_window"],
        "point_top1_window": selector_report["point_top1_window"],
        "selection_frequency": selector_report["selection_frequency"],
        "eligible_count": selector_report["eligible_count"],
        "bootstrap_draw_index_sha256": digest["draw_index_sha256"],
        "population": population,
        "bindings": normalized_bindings,
    }
    _attach_manifest(selector_freeze)
    validate_selector_freeze_payload(selector_freeze, selector_report, outcome_panel)
    return {
        "selector_report": selector_report,
        "outcome_panel": outcome_panel,
        "selector_freeze": selector_freeze,
    }


def validate_selector_report_payload(payload: Mapping[str, Any]) -> None:
    expected = frozenset(
        (
            "schema_version",
            "selector",
            "formal",
            "population",
            "bootstrap",
            "published_ranking",
            "eligible_count",
            "selected_window",
            "point_top1_window",
            "selection_frequency",
            "window_decision",
            "geometry_fields_consumed",
            "outcome_fields_consumed",
            "bindings",
            "manifest_sha256",
        )
    )
    _exact_keys(payload, expected, "selector report artifact")
    _require_manifest(payload, "selector report artifact")
    if payload["schema_version"] != SELECTOR_REPORT_SCHEMA_VERSION:
        raise Phase5ContractError("selector report artifact schema differs")
    if payload["selector"] != "ENTROPY_KL_CONSENSUS_WIDTH4_V1":
        raise Phase5ContractError("selector report name differs")
    if type(payload["formal"]) is not bool:
        raise Phase5ContractError("selector report formal flag must be boolean")
    if payload["geometry_fields_consumed"] is not False or payload["outcome_fields_consumed"] is not False:
        raise Phase5ContractError("selector report field-consumption flags differ")
    population = payload["population"]
    _exact_keys(
        population,
        frozenset(
            (
                "record_count",
                "subject_count",
                "ordered_identity_sha256",
                "ordered_prompt_sha256",
                "ordered_trajectory_record_sha256",
            )
        ),
        "selector population",
    )
    for key in ("ordered_identity_sha256", "ordered_prompt_sha256", "ordered_trajectory_record_sha256"):
        _sha256(population[key], "selector population.%s" % key)
    for key in ("record_count", "subject_count"):
        if isinstance(population[key], bool) or not isinstance(population[key], int):
            raise Phase5ContractError("selector population.%s must be an integer" % key)
    bootstrap = payload["bootstrap"]
    _exact_keys(
        bootstrap,
        frozenset(
            (
                "schema_version",
                "replicates",
                "seed",
                "prng",
                "random_api",
                "normalized_order",
                "normalized_identity_sha256",
                "subject_order_sha256",
                "subject_sizes",
                "draw_representation",
                "draw_index_sha256",
            )
        ),
        "selector bootstrap digest",
    )
    if bootstrap["schema_version"] != BOOTSTRAP_DIGEST_SCHEMA_VERSION:
        raise Phase5ContractError("selector bootstrap digest schema differs")
    if (
        bootstrap["prng"] != "python_stdlib_random.Random"
        or bootstrap["random_api"] != "randrange(n_s)"
        or bootstrap["normalized_order"] != "subject_ascending_identity_ascending"
        or bootstrap["draw_representation"]
        != "canonical_json_array_of_global_indices_in_normalized_order"
    ):
        raise Phase5ContractError("selector bootstrap algorithm contract differs")
    if (
        isinstance(bootstrap["replicates"], bool)
        or not isinstance(bootstrap["replicates"], int)
        or bootstrap["replicates"] < 2
        or isinstance(bootstrap["seed"], bool)
        or not isinstance(bootstrap["seed"], int)
    ):
        raise Phase5ContractError("selector bootstrap R/seed types differ")
    _sha256(bootstrap["normalized_identity_sha256"], "normalized identity digest")
    _sha256(bootstrap["subject_order_sha256"], "subject order digest")
    _sha256(bootstrap["draw_index_sha256"], "bootstrap draw digest")
    if not isinstance(bootstrap["subject_sizes"], Mapping) or not bootstrap["subject_sizes"]:
        raise Phase5ContractError("selector bootstrap subject sizes are missing")
    if any(
        not isinstance(subject, str)
        or not subject
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        for subject, size in bootstrap["subject_sizes"].items()
    ):
        raise Phase5ContractError("selector bootstrap subject sizes differ")
    if sum(bootstrap["subject_sizes"].values()) != population["record_count"]:
        raise Phase5ContractError("selector bootstrap population size differs")
    if len(bootstrap["subject_sizes"]) != population["subject_count"]:
        raise Phase5ContractError("selector bootstrap subject count differs")
    if payload["formal"] and (
        population["record_count"] != FORMAL_RECORD_COUNT
        or population["subject_count"] != FORMAL_SUBJECT_COUNT
        or bootstrap["replicates"] != FORMAL_BOOTSTRAP_REPLICATES
        or bootstrap["seed"] != FORMAL_BOOTSTRAP_SEED
    ):
        raise Phase5ContractError("formal selector population/bootstrap contract differs")
    ranking = payload["published_ranking"]
    if not isinstance(ranking, list) or len(ranking) != len(CONSENSUS_STARTS):
        raise Phase5ContractError("selector report ranking must contain 25 rows")
    starts = []
    eligible_count = 0
    for row in ranking:
        _exact_keys(
            row,
            frozenset(
                (
                    "start",
                    "window",
                    "E",
                    "K",
                    "contrasts",
                    "score",
                    "point_eligible",
                    "scorable",
                    "eligibility_checks",
                    "failure_reasons",
                )
            ),
            "selector report ranking row",
        )
        starts.append(row["start"])
        _finite(row["score"], "selector score")
        if row["scorable"] is not True or type(row["point_eligible"]) is not bool:
            raise Phase5ContractError("selector ranking scorability/eligibility differs")
        if not isinstance(row["failure_reasons"], list) or any(
            not isinstance(value, str) or not value for value in row["failure_reasons"]
        ):
            raise Phase5ContractError("selector failure reasons must be strings")
        if row["point_eligible"] != (len(row["failure_reasons"]) == 0):
            raise Phase5ContractError("selector failure reasons disagree with eligibility")
        eligible_count += int(row["point_eligible"])
    if set(starts) != set(CONSENSUS_STARTS) or len(starts) != len(set(starts)):
        raise Phase5ContractError("selector ranking start closure differs")
    expected_order = sorted(ranking, key=lambda row: (-float(row["score"]), int(row["start"])))
    if starts != [row["start"] for row in expected_order]:
        raise Phase5ContractError("selector ranking order differs")
    if payload["eligible_count"] != eligible_count:
        raise Phase5ContractError("selector eligible count differs")
    decision = payload["window_decision"]
    if decision not in ("SELECTED",) + ABSTAIN_LABELS:
        raise Phase5ContractError("selector decision differs")
    selected = payload["selected_window"]
    frequency = payload["selection_frequency"]
    if decision == "SELECTED":
        if not isinstance(selected, str) or not selected:
            raise Phase5ContractError("SELECTED decision lacks selected window")
        if frequency is None or _finite(frequency, "selection frequency") < 0.80:
            raise Phase5ContractError("SELECTED decision lacks passing selection frequency")
    elif selected is not None:
        raise Phase5ContractError("ABSTAIN decision cannot expose selected_window")
    if frequency is not None and not 0.0 <= _finite(frequency, "selection frequency") <= 1.0:
        raise Phase5ContractError("selection frequency lies outside [0,1]")
    if decision == "ABSTAIN_LOW_SELECTION_FREQUENCY" and (
        frequency is None or float(frequency) >= 0.80 or not payload["point_top1_window"]
    ):
        raise Phase5ContractError("low-frequency ABSTAIN semantics differ")
    if decision in ("ABSTAIN_NO_POINT_ELIGIBLE", "ABSTAIN_NO_UNIQUE_TOP1") and frequency is not None:
        raise Phase5ContractError("non-frequency ABSTAIN cannot expose selection frequency")


def validate_outcome_panel_payload(
    payload: Mapping[str, Any],
    selector_report: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    validate_selector_report_payload(selector_report)
    validate_registry(registry)
    expected_keys = frozenset(
        (
            "schema_version",
            "selector_report_manifest_sha256",
            "known_outcome_registry_manifest_sha256",
            "baseline",
            "fixed_comparator",
            "raw_high3",
            "panel_high3",
            "selected_inclusive_replacement",
            "selected_is_known_comparator",
            "blind_low3",
            "unique_cells",
            "unique_cell_count",
            "bindings",
            "manifest_sha256",
        )
    )
    _exact_keys(payload, expected_keys, "outcome panel artifact")
    _require_manifest(payload, "outcome panel artifact")
    if payload["schema_version"] != OUTCOME_PANEL_SCHEMA_VERSION:
        raise Phase5ContractError("outcome panel schema differs")
    if payload["selector_report_manifest_sha256"] != selector_report["manifest_sha256"]:
        raise Phase5ContractError("outcome panel selector binding differs")
    if payload["known_outcome_registry_manifest_sha256"] != registry["manifest_sha256"]:
        raise Phase5ContractError("outcome panel registry binding differs")
    expected = construct_panel(
        selector_report["published_ranking"],
        selected_window=selector_report["selected_window"],
        known_windows=registry["blind_group_exclusion_windows"],
    )
    projected = {
        "raw_high3": payload["raw_high3"],
        "panel_high3": payload["panel_high3"],
        "selected_inclusive_replacement": payload["selected_inclusive_replacement"],
        "selected_is_known_comparator": payload["selected_is_known_comparator"],
        "blind_low3": payload["blind_low3"],
        "unique_cells": payload["unique_cells"],
        "unique_cell_count": payload["unique_cell_count"],
    }
    expected_projected = {key: expected[key] for key in projected}
    if canonical_json_bytes(projected) != canonical_json_bytes(expected_projected):
        raise Phase5ContractError("outcome panel differs from deterministic reconstruction")
    if payload["baseline"] != "no-loop" or payload["fixed_comparator"] != "15:18":
        raise Phase5ContractError("outcome panel baseline/comparator differs")


def validate_selector_freeze_payload(
    payload: Mapping[str, Any],
    selector_report: Mapping[str, Any],
    outcome_panel: Mapping[str, Any],
) -> None:
    validate_selector_report_payload(selector_report)
    expected_keys = frozenset(
        (
            "schema_version",
            "selector",
            "formal",
            "replicates",
            "seed",
            "selector_report_manifest_sha256",
            "outcome_panel_manifest_sha256",
            "window_decision",
            "selected_window",
            "point_top1_window",
            "selection_frequency",
            "eligible_count",
            "bootstrap_draw_index_sha256",
            "population",
            "bindings",
            "manifest_sha256",
        )
    )
    _exact_keys(payload, expected_keys, "selector freeze artifact")
    _require_manifest(payload, "selector freeze artifact")
    if payload["schema_version"] != SELECTOR_FREEZE_SCHEMA_VERSION:
        raise Phase5ContractError("selector freeze schema differs")
    exact = {
        "selector": selector_report["selector"],
        "formal": selector_report["formal"],
        "replicates": selector_report["bootstrap"]["replicates"],
        "seed": selector_report["bootstrap"]["seed"],
        "selector_report_manifest_sha256": selector_report["manifest_sha256"],
        "outcome_panel_manifest_sha256": outcome_panel["manifest_sha256"],
        "window_decision": selector_report["window_decision"],
        "selected_window": selector_report["selected_window"],
        "point_top1_window": selector_report["point_top1_window"],
        "selection_frequency": selector_report["selection_frequency"],
        "eligible_count": selector_report["eligible_count"],
        "bootstrap_draw_index_sha256": selector_report["bootstrap"]["draw_index_sha256"],
        "population": selector_report["population"],
        "bindings": selector_report["bindings"],
    }
    for key, expected in exact.items():
        if payload[key] != expected:
            raise Phase5ContractError("selector freeze %s differs" % key)


def selector_ranking_csv(selector_report: Mapping[str, Any]) -> str:
    """Render the frozen ranking to deterministic UTF-8 CSV text."""

    validate_selector_report_payload(selector_report)
    columns = ["start", "window", "score", "E", "K", "point_eligible", "failure_reasons"]
    for name in _CONTRAST_NAMES:
        columns.extend(
            (
                "%s_point" % name,
                "%s_standard_error" % name,
                "%s_lower95" % name,
                "%s_upper95" % name,
            )
        )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in selector_report["published_ranking"]:
        output: Dict[str, Any] = {
            "start": row["start"],
            "window": row["window"],
            "score": row["score"],
            "E": row["E"],
            "K": row["K"],
            "point_eligible": row["point_eligible"],
            "failure_reasons": "|".join(row["failure_reasons"]),
        }
        for name in _CONTRAST_NAMES:
            for key in ("point", "standard_error", "lower95", "upper95"):
                output["%s_%s" % (name, key)] = row["contrasts"][name][key]
        writer.writerow(output)
    return handle.getvalue()


def verify_selector_artifacts(
    records: Sequence[Mapping[str, Any]],
    *,
    card: Mapping[str, Any],
    registry: Mapping[str, Any],
    expected_membership: Optional[Sequence[Mapping[str, Any]]],
    bindings: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    formal: bool = True,
    replicates: int = FORMAL_BOOTSTRAP_REPLICATES,
    seed: int = FORMAL_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Recompute all formal artifacts from persisted scalar records and compare."""

    _exact_keys(
        artifacts,
        frozenset(("selector_report", "outcome_panel", "selector_freeze")),
        "selector artifact bundle",
    )
    expected = build_selector_artifacts(
        records,
        card=card,
        registry=registry,
        expected_membership=expected_membership,
        bindings=bindings,
        formal=formal,
        replicates=replicates,
        seed=seed,
    )
    for name in ("selector_report", "outcome_panel", "selector_freeze"):
        if canonical_json_bytes(artifacts[name]) != canonical_json_bytes(expected[name]):
            raise Phase5ContractError("%s differs from deterministic recomputation" % name)
    return {
        "schema_version": "loopscope.phase5.selector-verifier-receipt.v1",
        "status": "PASS",
        "selector_report_manifest_sha256": expected["selector_report"]["manifest_sha256"],
        "outcome_panel_manifest_sha256": expected["outcome_panel"]["manifest_sha256"],
        "selector_freeze_manifest_sha256": expected["selector_freeze"]["manifest_sha256"],
        "bootstrap_draw_index_sha256": expected["selector_report"]["bootstrap"][
            "draw_index_sha256"
        ],
        "record_count": expected["selector_report"]["population"]["record_count"],
        "subject_count": expected["selector_report"]["population"]["subject_count"],
        "outcome_values_consumed": False,
    }


__all__ = [
    "BOOTSTRAP_DIGEST_SCHEMA_VERSION",
    "FORMAL_BOOTSTRAP_REPLICATES",
    "FORMAL_BOOTSTRAP_SEED",
    "FORMAL_RECORD_COUNT",
    "FORMAL_SUBJECT_COUNT",
    "OUTCOME_PANEL_SCHEMA_VERSION",
    "SELECTOR_FREEZE_SCHEMA_VERSION",
    "SELECTOR_REPORT_SCHEMA_VERSION",
    "bootstrap_index_digest",
    "build_selector_artifacts",
    "canonical_identity",
    "canonical_identity_string",
    "selector_inputs_from_trajectories",
    "selector_ranking_csv",
    "validate_outcome_panel_payload",
    "validate_selector_freeze_payload",
    "validate_selector_report_payload",
    "verify_selector_artifacts",
]
