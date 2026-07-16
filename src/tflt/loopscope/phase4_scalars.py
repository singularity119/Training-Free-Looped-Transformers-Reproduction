"""Stable full-vocabulary scalar projection for Phase 4 synthetic verification."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from tflt.loopscope.phase4_schema import (
    BOUNDARY_IDS,
    Phase4ContractError,
    validate_trajectory_record,
)


def stable_log_softmax_float32(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or values.size < 100:
        raise Phase4ContractError("full-vocabulary logits must be a one-dimensional vector")
    if not np.isfinite(values).all():
        raise Phase4ContractError("non-finite logits are forbidden")
    maximum = np.max(values)
    shifted = values - maximum
    # Exponentiation and reduction stay float32, matching the frozen producer contract.
    normalizer = np.log(np.sum(np.exp(shifted, dtype=np.float32), dtype=np.float32))
    log_probabilities = shifted - normalizer
    if not np.isfinite(log_probabilities).all():
        raise Phase4ContractError("stable float32 log_softmax produced a non-finite value")
    return log_probabilities.astype(np.float32, copy=False)


def trajectory_record_from_logits(
    *,
    canonical_identity: str,
    category: str,
    boundary_logits: Sequence[Sequence[float]],
    producer_provenance: Mapping[str, Any],
    d36_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Project B_0..B_36 logits to scalar-only records; never return tensors/logits."""

    if len(boundary_logits) != len(BOUNDARY_IDS):
        raise Phase4ContractError("boundary_logits must contain exactly B_0..B_36")
    log_ps = [stable_log_softmax_float32(values) for values in boundary_logits]
    vocabulary_size = int(log_ps[0].size)
    if any(int(values.size) != vocabulary_size for values in log_ps):
        raise Phase4ContractError("all boundary vocabularies must have the same size")
    final_log_p = log_ps[-1]
    boundaries = []
    for boundary_id, (raw_logits, log_p) in enumerate(zip(boundary_logits, log_ps)):
        probabilities = np.exp(log_p, dtype=np.float32)
        entropy = float(-np.sum(probabilities * log_p, dtype=np.float32))
        kl = float(np.sum(probabilities * (log_p - final_log_p), dtype=np.float32))
        logits = np.asarray(raw_logits, dtype=np.float32)
        rms = float(np.sqrt(np.mean(logits * logits, dtype=np.float32), dtype=np.float32))
        sorted_probabilities = np.sort(probabilities)[::-1]
        mass1 = float(np.sum(sorted_probabilities[:1], dtype=np.float32))
        mass10 = float(np.sum(sorted_probabilities[: min(10, vocabulary_size)], dtype=np.float32))
        mass100 = float(np.sum(sorted_probabilities[: min(100, vocabulary_size)], dtype=np.float32))
        scalars = [entropy, kl, rms, mass1, mass10, mass100]
        if not all(math.isfinite(value) for value in scalars):
            raise Phase4ContractError("non-finite boundary scalar")
        boundaries.append(
            {
                "boundary_id": boundary_id,
                "full_vocabulary_entropy": entropy,
                "kl_to_final": kl,
                "normalized_entropy": entropy / math.log(vocabulary_size),
                "logit_rms": rms,
                "top1_probability_mass": mass1,
                "top10_probability_mass": mass10,
                "top100_probability_mass": mass100,
                "finite": True,
            }
        )
    record = {
        "schema_version": "loopscope.phase4.prefix-trajectory-record.v1",
        "canonical_identity": canonical_identity,
        "category": category,
        "boundaries": boundaries,
        "producer_provenance": dict(producer_provenance),
    }
    validate_trajectory_record(record, d36_tolerance=d36_tolerance)
    return record


__all__ = ["stable_log_softmax_float32", "trajectory_record_from_logits"]
