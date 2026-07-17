"""Outcome-blind hidden-geometry contracts and analysis for Phase 4 Gate B-2."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase4_schema import (
    BOUNDARY_RECORD_KEYS,
    Phase4ContractError,
    scan_forbidden_fields,
    validate_trajectory_record,
)


B2_RECORD_SCHEMA = "loopscope.phase4.b2-hidden-geometry-record.v1"
L2_KEY = "hidden_l2_rms_to_final"
COSINE_KEY = "hidden_cosine_to_final"
GEOMETRY_KEYS = {L2_KEY, COSINE_KEY}
B2_BOUNDARY_KEYS = set(BOUNDARY_RECORD_KEYS) | GEOMETRY_KEYS
ANALYZED_METRICS: Tuple[str, ...] = (
    "full_vocabulary_entropy",
    "kl_to_final",
    L2_KEY,
    COSINE_KEY,
    "hidden_cosine_distance_to_final",
)


class Phase4HiddenGeometryError(Phase4ContractError):
    """Raised when the frozen B-2 geometry contract is violated."""


def geometry_scalars_from_numpy(vectors: Any) -> List[Dict[str, float]]:
    """Reference float32 computation for exactly B_0..B_36 normalized vectors."""

    import numpy as np

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 37 or values.shape[1] < 1:
        raise Phase4HiddenGeometryError("geometry vectors must have shape (37, hidden_dim)")
    if not bool(np.isfinite(values).all()):
        raise Phase4HiddenGeometryError("geometry vectors contain a non-finite value")
    reference = values[36]
    reference_norm = np.sqrt(np.sum(reference * reference, dtype=np.float32), dtype=np.float32)
    if not math.isfinite(float(reference_norm)) or float(reference_norm) == 0.0:
        raise Phase4HiddenGeometryError("final normalized hidden vector has zero/non-finite norm")
    rows: List[Dict[str, float]] = []
    for current in values:
        current_norm = np.sqrt(np.sum(current * current, dtype=np.float32), dtype=np.float32)
        if not math.isfinite(float(current_norm)) or float(current_norm) == 0.0:
            raise Phase4HiddenGeometryError("normalized hidden vector has zero/non-finite norm")
        difference = current - reference
        l2_rms = np.sqrt(np.mean(difference * difference, dtype=np.float32), dtype=np.float32)
        dot = np.sum(current * reference, dtype=np.float32)
        cosine = dot / (current_norm * reference_norm)
        l2_value, cosine_value = float(l2_rms), float(cosine)
        if not math.isfinite(l2_value) or not math.isfinite(cosine_value):
            raise Phase4HiddenGeometryError("geometry reduction produced a non-finite value")
        rows.append({L2_KEY: l2_value, COSINE_KEY: cosine_value})
    return rows


def torch_hidden_geometry_scalars(
    torch_module: Any, current: Any, reference: Any
) -> Dict[str, float]:
    """Reduce two already final-normalized vectors in float32 without fallback."""

    current32 = current.float()
    reference32 = reference.float()
    if current32.shape != reference32.shape or current32.ndim != 2 or current32.shape[0] != 1:
        raise Phase4HiddenGeometryError("runtime geometry vectors have an unexpected shape")
    if not bool(torch_module.isfinite(current32).all().item()) or not bool(
        torch_module.isfinite(reference32).all().item()
    ):
        raise Phase4HiddenGeometryError("runtime geometry vector is non-finite")
    current_norm = torch_module.sqrt(torch_module.sum(current32 * current32, dim=-1))
    reference_norm = torch_module.sqrt(torch_module.sum(reference32 * reference32, dim=-1))
    if float(current_norm.item()) == 0.0 or float(reference_norm.item()) == 0.0:
        raise Phase4HiddenGeometryError("runtime geometry vector has zero norm")
    difference = current32 - reference32
    l2_rms = torch_module.sqrt(torch_module.mean(difference * difference, dim=-1))
    cosine = torch_module.sum(current32 * reference32, dim=-1) / (
        current_norm * reference_norm
    )
    l2_value = float(l2_rms.double().cpu().item())
    cosine_value = float(cosine.double().cpu().item())
    if not math.isfinite(l2_value) or not math.isfinite(cosine_value):
        raise Phase4HiddenGeometryError("runtime geometry reduction is non-finite")
    return {L2_KEY: l2_value, COSINE_KEY: cosine_value}


def strip_geometry_to_p4b(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a B-2 record to the exact P4-B scalar record contract."""

    projected = copy.deepcopy(dict(record))
    projected["schema_version"] = "loopscope.phase4.prefix-trajectory-record.v1"
    for boundary in projected.get("boundaries", []):
        if isinstance(boundary, dict):
            boundary.pop(L2_KEY, None)
            boundary.pop(COSINE_KEY, None)
    return projected


def validate_hidden_geometry_record(
    record: Mapping[str, Any],
    *,
    d36_tolerance: float = 1e-6,
    endpoint_l2_tolerance: float = 1e-7,
    endpoint_cosine_tolerance: float = 1e-5,
    cosine_range_tolerance: float = 1e-5,
) -> None:
    scan_forbidden_fields(record, allow_options=False)
    if record.get("schema_version") != B2_RECORD_SCHEMA:
        raise Phase4HiddenGeometryError("unexpected B-2 trajectory schema_version")
    boundaries = record.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != 37:
        raise Phase4HiddenGeometryError("B-2 trajectory must contain exactly 37 boundaries")
    if any(not isinstance(boundary, Mapping) or set(boundary) != B2_BOUNDARY_KEYS for boundary in boundaries):
        raise Phase4HiddenGeometryError("B-2 boundary record is not closed-world")
    validate_trajectory_record(
        strip_geometry_to_p4b(record), d36_tolerance=d36_tolerance
    )
    for expected_id, boundary in enumerate(boundaries):
        if boundary.get("boundary_id") != expected_id:
            raise Phase4HiddenGeometryError("B-2 boundary order/id mismatch")
        l2_value = boundary.get(L2_KEY)
        cosine_value = boundary.get(COSINE_KEY)
        for key, value in ((L2_KEY, l2_value), (COSINE_KEY, cosine_value)):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise Phase4HiddenGeometryError("B-2 scalar %s must be finite" % key)
        if float(l2_value) < 0.0:
            raise Phase4HiddenGeometryError("B-2 RMS-L2 distance must be non-negative")
        if not -1.0 - cosine_range_tolerance <= float(cosine_value) <= 1.0 + cosine_range_tolerance:
            raise Phase4HiddenGeometryError("B-2 cosine is outside its tolerated mathematical range")
        if "hidden_cosine_distance_to_final" in boundary:
            raise Phase4HiddenGeometryError("cosine distance must be analyzer-derived only")
    if abs(float(boundaries[36][L2_KEY])) > endpoint_l2_tolerance:
        raise Phase4HiddenGeometryError("B_36 hidden RMS-L2 endpoint exceeds tolerance")
    if abs(float(boundaries[36][COSINE_KEY]) - 1.0) > endpoint_cosine_tolerance:
        raise Phase4HiddenGeometryError("B_36 hidden cosine endpoint exceeds tolerance")


def analyze_hidden_geometry_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Produce the one diagnostic aggregate pass defined by Gate B-2."""

    import numpy as np

    if not records:
        raise Phase4HiddenGeometryError("B-2 analysis requires non-empty records")
    identities: List[str] = []
    matrices: Dict[str, List[List[float]]] = {key: [] for key in ANALYZED_METRICS[:-1]}
    for record in records:
        validate_hidden_geometry_record(record)
        identity = str(record["canonical_identity"])
        if identity in identities:
            raise Phase4HiddenGeometryError("duplicate B-2 analysis identity")
        identities.append(identity)
        for key in ANALYZED_METRICS[:-1]:
            matrices[key].append([float(row[key]) for row in record["boundaries"]])
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in matrices.items()}
    arrays["hidden_cosine_distance_to_final"] = 1.0 - arrays[COSINE_KEY]
    if not all(bool(np.isfinite(value).all()) for value in arrays.values()):
        raise Phase4HiddenGeometryError("B-2 analysis input contains a non-finite value")

    means = {key: value.mean(axis=0) for key, value in arrays.items()}
    layer_deltas = {key: np.diff(value, axis=1) for key, value in arrays.items()}
    boundary_rows = []
    for boundary_id in range(37):
        row: Dict[str, Any] = {"boundary_id": boundary_id}
        for key in ANALYZED_METRICS:
            row[key + "_mean"] = float(means[key][boundary_id])
            row["delta_" + key] = (
                None if boundary_id == 0 else float(means[key][boundary_id] - means[key][boundary_id - 1])
            )
        boundary_rows.append(row)

    window_rows = []
    sign_rows = []
    maximum_additivity_error = 0.0
    maximum_cosine_delta_sign_error = 0.0
    for width in range(2, 9):
        for start in range(37 - width):
            stop = start + width
            row = {"width": width, "start": start, "entry_boundary": start, "exit_boundary": stop}
            sample_deltas: Dict[str, Any] = {}
            for key in ANALYZED_METRICS:
                delta = arrays[key][:, stop] - arrays[key][:, start]
                sample_deltas[key] = delta
                row["delta_" + key + "_mean"] = float(delta.mean())
                internal = layer_deltas[key][:, start:stop].sum(axis=1)
                maximum_additivity_error = max(
                    maximum_additivity_error, float(np.max(np.abs(delta - internal)))
                )
            cosine_error = sample_deltas["hidden_cosine_distance_to_final"] + sample_deltas[COSINE_KEY]
            maximum_cosine_delta_sign_error = max(
                maximum_cosine_delta_sign_error, float(np.max(np.abs(cosine_error)))
            )
            window_rows.append(row)
            if width == 4:
                for left, right in (
                    ("kl_to_final", L2_KEY),
                    ("kl_to_final", "hidden_cosine_distance_to_final"),
                    (L2_KEY, "hidden_cosine_distance_to_final"),
                ):
                    left_sign = np.sign(sample_deltas[left])
                    right_sign = np.sign(sample_deltas[right])
                    agreement = left_sign == right_sign
                    sign_rows.append(
                        {
                            "width": 4,
                            "start": start,
                            "metric_pair": [left, right],
                            "agreement_count": int(agreement.sum()),
                            "record_count": len(records),
                            "agreement_fraction": float(agreement.mean()),
                            "both_negative_count": int(((left_sign < 0) & (right_sign < 0)).sum()),
                            "both_positive_count": int(((left_sign > 0) & (right_sign > 0)).sum()),
                            "zero_involved_count": int(((left_sign == 0) | (right_sign == 0)).sum()),
                        }
                    )

    correlations = []
    for left, right in (
        ("kl_to_final", L2_KEY),
        ("kl_to_final", "hidden_cosine_distance_to_final"),
        (L2_KEY, "hidden_cosine_distance_to_final"),
    ):
        correlations.append(
            {
                "metrics": [left, right],
                "pearson": _pearson(means[left], means[right]),
                "spearman": _spearman(means[left], means[right]),
                "point_count": 37,
            }
        )
    return {
        "schema_version": "loopscope.phase4.b2-hidden-geometry-analysis.v1",
        "analysis_scope": "diagnostic_only_no_outcome_no_gain_claim",
        "record_count": len(records),
        "ordered_identities": identities,
        "boundary_means_and_deltas": boundary_rows,
        "width4_exit_minus_entry": [row for row in window_rows if row["width"] == 4],
        "width2_8_exit_minus_entry": window_rows,
        "mean_curve_correlations": correlations,
        "width4_sample_delta_sign_agreement": sign_rows,
        "checks": {
            "maximum_additivity_abs_error": maximum_additivity_error,
            "maximum_delta_cosine_distance_plus_delta_cosine_abs_error": maximum_cosine_delta_sign_error,
            "window_count_width4": sum(row["width"] == 4 for row in window_rows),
            "window_count_width2_8": len(window_rows),
            "cosine_distance_producer_field_present": False,
        },
    }


def _pearson(left: Any, right: Any) -> float:
    import numpy as np

    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        raise Phase4HiddenGeometryError("correlation input curve is constant")
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        raise Phase4HiddenGeometryError("Pearson correlation is non-finite")
    return value


def _spearman(left: Any, right: Any) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _average_ranks(values: Any) -> Any:
    import numpy as np

    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=np.float64)
    cursor = 0
    while cursor < len(data):
        stop = cursor + 1
        while stop < len(data) and data[order[stop]] == data[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + stop - 1) + 1.0
        cursor = stop
    return ranks


__all__ = [
    "ANALYZED_METRICS",
    "B2_RECORD_SCHEMA",
    "COSINE_KEY",
    "L2_KEY",
    "Phase4HiddenGeometryError",
    "analyze_hidden_geometry_records",
    "geometry_scalars_from_numpy",
    "strip_geometry_to_p4b",
    "torch_hidden_geometry_scalars",
    "validate_hidden_geometry_record",
]
