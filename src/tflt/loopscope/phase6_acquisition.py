"""Pure NumPy helpers for Phase 6 pre-answer trajectory acquisition.

This module is deliberately producer-only.  It accepts generation/replay token
IDs, full-vocabulary logits, and hidden vectors in memory, but the returned
record contains only the closed, sanitized Phase 6 trajectory contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from tflt.loopscope.phase6_schema import (
    Phase6ContractError,
    validate_trajectory_record,
)


SCHEMA_VERSION = "loopscope.phase6.pre-answer-trajectory.v2"
BOUNDARY_COUNT = 37
TRANSITION_COUNT = 36
FINAL_TOLERANCE = 1e-6

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "model_repo",
        "model_revision",
        "dataset_repo",
        "dataset_revision",
        "split",
        "canonical_identity",
        "category",
        "prompt_sha256",
        "generated_completion_sha256",
        "generation_length",
        "replay_length",
        "anchor_token_index",
        "answer_span_start_offset",
        "answer_span_end_offset",
        "answer_match_count",
        "selected_match_ordinal",
        "answer_first_token_index",
        "answer_span_extractor_sha256",
        "generated_id_text_aligner_sha256",
        "generation_count",
        "replay_count",
        "loop_insertions",
        "H",
        "D",
        "hidden_rms_l2_to_final",
        "hidden_cosine_to_final",
        "hidden_cosine_distance_to_final",
        "adjacent_angular_distance",
        "provenance",
    }
)

PROVENANCE_KEYS = frozenset(
    {
        "producer_version",
        "card_sha256",
        "renderer_manifest_sha256",
        "tokenizer_manifest_sha256",
        "generation_ids_sha256",
        "replay_ids_sha256",
        "boundary_capture",
        "final_norm_path",
        "lm_head_path",
    }
)


def _fail(message: str) -> None:
    raise Phase6ContractError(message)


def _as_token_ids(values: Sequence[int], name: str) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("%s must be a token-ID sequence" % name)
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            _fail("%s must contain only integer token IDs" % name)
        integer = int(value)
        if integer < 0:
            _fail("%s contains a negative token ID" % name)
        result.append(integer)
    if not result:
        _fail("%s must not be empty" % name)
    return tuple(result)


def token_ids_sha256(values: Sequence[int]) -> str:
    """Hash token IDs without retaining token text or the IDs themselves."""

    token_ids = _as_token_ids(values, "token IDs")
    body = json.dumps(token_ids, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(body).hexdigest()


def validate_generation_replay_id_closure(
    generation_prefix_ids: Sequence[int],
    replay_ids: Sequence[int],
) -> Tuple[int, ...]:
    """Require byte-for-byte-equivalent integer ID sequences, fail closed."""

    generation = _as_token_ids(generation_prefix_ids, "generation prefix IDs")
    replay = _as_token_ids(replay_ids, "replay IDs")
    if generation != replay:
        _fail("generation-prefix/replay token IDs mismatch")
    return generation


# A descriptive alias for callers that treat the closure itself as the value.
exact_generation_replay_id_closure = validate_generation_replay_id_closure


def _float32_matrix(
    values: Sequence[Sequence[float]],
    name: str,
    *,
    rows: int = BOUNDARY_COUNT,
) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        _fail("%s must be a numeric matrix" % name)
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Phase6ContractError("%s must be a rectangular float32 matrix" % name) from exc
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        _fail("%s must have shape (%d, non-zero width)" % (name, rows))
    if not bool(np.all(np.isfinite(matrix))):
        _fail("%s contains non-finite values" % name)
    return matrix


def stable_float32_log_softmax(logits: Sequence[float]) -> np.ndarray:
    """Compute stable full-vocabulary log-softmax after float32 conversion."""

    if isinstance(logits, (str, bytes)):
        _fail("logits must be a non-empty vector")
    try:
        vector = np.asarray(logits, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Phase6ContractError("logits must be a float32 vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        _fail("logits must be a non-empty vector")
    if not bool(np.all(np.isfinite(vector))):
        _fail("logits contain non-finite values")
    maximum = np.max(vector)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        shifted = np.subtract(vector, maximum, dtype=np.float32)
        exponentials = np.exp(shifted, dtype=np.float32)
        denominator = np.sum(exponentials, dtype=np.float32)
        log_denominator = np.log(denominator, dtype=np.float32)
        result = np.subtract(shifted, log_denominator, dtype=np.float32)
    if (
        not bool(np.isfinite(denominator))
        or float(denominator) <= 0.0
        or not bool(np.all(np.isfinite(result)))
    ):
        _fail("float32 log-softmax produced non-finite values")
    return result


def full_vocabulary_entropy_and_kl(
    boundary_logits: Sequence[Sequence[float]],
) -> Tuple[list[float], list[float]]:
    """Return H and KL(p_l || p_36) for exactly B0..B36."""

    logits = _float32_matrix(boundary_logits, "boundary logits")
    log_probabilities = [stable_float32_log_softmax(row) for row in logits]
    final_log_probabilities = log_probabilities[-1]
    entropies = []
    divergences = []
    for log_probability in log_probabilities:
        with np.errstate(under="ignore", invalid="ignore", over="ignore"):
            probability = np.exp(log_probability, dtype=np.float32)
            entropy_terms = -probability.astype(np.float64) * log_probability.astype(np.float64)
            kl_terms = probability.astype(np.float64) * (
                log_probability.astype(np.float64)
                - final_log_probabilities.astype(np.float64)
            )
        entropy = float(np.sum(entropy_terms, dtype=np.float64))
        divergence = float(np.sum(kl_terms, dtype=np.float64))
        if not math.isfinite(entropy) or not math.isfinite(divergence):
            _fail("entropy/KL reduction produced a non-finite value")
        entropies.append(entropy)
        divergences.append(divergence)
    # Identical distributions have exact zero KL by construction.
    divergences[-1] = 0.0
    if divergences[-1] > FINAL_TOLERANCE:
        _fail("D36 exceeds tolerance")
    return entropies, divergences


def hidden_diagnostics(
    final_normalized_vectors: Sequence[Sequence[float]],
) -> Tuple[list[float], list[float], list[float]]:
    """Compute diagnostic geometry over provided FinalNorm outputs."""

    vectors = _float32_matrix(final_normalized_vectors, "final-normalized vectors")
    final = vectors[-1]
    final_norm = float(np.linalg.norm(final))
    if not math.isfinite(final_norm) or final_norm == 0.0:
        _fail("final-normalized vectors contain a zero/non-finite norm")

    rms_values = []
    cosine_values = []
    distance_values = []
    hidden_width = int(vectors.shape[1])
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0.0:
            _fail("final-normalized vectors contain a zero/non-finite norm")
        difference = np.subtract(vector, final, dtype=np.float32)
        squared_l2 = float(np.dot(difference, difference))
        dot = float(np.dot(vector, final))
        cosine = dot / (norm * final_norm)
        rms_l2 = math.sqrt(squared_l2 / hidden_width)
        cosine_distance = 1.0 - cosine
        if not all(math.isfinite(value) for value in (rms_l2, cosine, cosine_distance)):
            _fail("hidden diagnostics produced a non-finite value")
        if cosine < -1.0 - FINAL_TOLERANCE or cosine > 1.0 + FINAL_TOLERANCE:
            _fail("hidden cosine lies outside tolerance")
        rms_values.append(rms_l2)
        cosine_values.append(cosine)
        distance_values.append(cosine_distance)

    if (
        abs(rms_values[-1]) > FINAL_TOLERANCE
        or abs(cosine_values[-1] - 1.0) > FINAL_TOLERANCE
        or abs(distance_values[-1]) > FINAL_TOLERANCE
    ):
        _fail("final hidden diagnostics fail endpoint tolerances")
    rms_values[-1] = 0.0
    cosine_values[-1] = 1.0
    distance_values[-1] = 0.0
    return rms_values, cosine_values, distance_values


def raw_adjacent_angular_distance(
    raw_boundaries: Sequence[Sequence[float]],
) -> list[float]:
    """Compute acos(clamp(cos, -1, 1))/pi over raw pre-FinalNorm B0..B36."""

    vectors = _float32_matrix(raw_boundaries, "raw pre-FinalNorm boundaries")
    result = []
    for left, right in zip(vectors[:-1], vectors[1:]):
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if (
            not math.isfinite(left_norm)
            or not math.isfinite(right_norm)
            or left_norm == 0.0
            or right_norm == 0.0
        ):
            _fail("raw pre-FinalNorm boundaries contain a zero/non-finite norm")
        cosine = float(np.dot(left, right)) / (left_norm * right_norm)
        if not math.isfinite(cosine):
            _fail("raw adjacent cosine is non-finite")
        angle = math.acos(min(1.0, max(-1.0, cosine))) / math.pi
        if not math.isfinite(angle) or not 0.0 <= angle <= 1.0:
            _fail("raw adjacent angular distance is invalid")
        result.append(angle)
    if len(result) != TRANSITION_COUNT:
        _fail("adjacent angular trajectory must contain exactly 36 values")
    return result


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        _fail(
            "%s keys differ: missing=%s extra=%s"
            % (name, sorted(expected - actual), sorted(actual - expected))
        )


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail("%s must be a lowercase SHA-256" % name)
    return text


def _require_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _fail("%s must be an integer" % name)
    result = int(value)
    if result < 0:
        _fail("%s must be non-negative" % name)
    return result


def build_sanitized_trajectory_record(
    *,
    model_repo: str,
    model_revision: str,
    dataset_repo: str,
    dataset_revision: str,
    split: str,
    canonical_identity: str,
    category: str,
    prompt_sha256: str,
    generated_completion_sha256: str,
    generation_length: int,
    replay_length: int,
    anchor_token_index: int,
    answer_span_start_offset: int,
    answer_span_end_offset: int,
    answer_match_count: int,
    selected_match_ordinal: int,
    answer_first_token_index: int,
    answer_span_extractor_sha256: str,
    generated_id_text_aligner_sha256: str,
    generation_prefix_ids: Sequence[int],
    replay_ids: Sequence[int],
    boundary_logits: Sequence[Sequence[float]],
    final_normalized_vectors: Sequence[Sequence[float]],
    raw_boundaries: Sequence[Sequence[float]],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reduce in-memory producer values to the exact sanitized record schema."""

    generation_ids = validate_generation_replay_id_closure(
        generation_prefix_ids, replay_ids
    )
    H, D = full_vocabulary_entropy_and_kl(boundary_logits)
    hidden_rms, hidden_cosine, hidden_cosine_distance = hidden_diagnostics(
        final_normalized_vectors
    )
    adjacent_angular = raw_adjacent_angular_distance(raw_boundaries)

    generation_length_value = _require_nonnegative_integer(
        generation_length, "generation_length"
    )
    replay_length_value = _require_nonnegative_integer(replay_length, "replay_length")
    anchor_index = _require_nonnegative_integer(anchor_token_index, "anchor_token_index")
    span_start = _require_nonnegative_integer(
        answer_span_start_offset, "answer_span_start_offset"
    )
    span_end = _require_nonnegative_integer(
        answer_span_end_offset, "answer_span_end_offset"
    )
    match_count = _require_nonnegative_integer(
        answer_match_count, "answer_match_count"
    )
    selected_ordinal = _require_nonnegative_integer(
        selected_match_ordinal, "selected_match_ordinal"
    )
    answer_index = _require_nonnegative_integer(
        answer_first_token_index, "answer_first_token_index"
    )
    if generation_length_value == 0 or replay_length_value == 0:
        _fail("generation/replay lengths must be positive")
    if replay_length_value != len(generation_ids):
        _fail("replay_length must equal the exact replay-ID sequence length")
    if span_end <= span_start:
        _fail("answer span offsets must form a non-empty half-open interval")
    if match_count < 1 or selected_ordinal != 0 or selected_ordinal >= match_count:
        _fail("selected answer match ordinal must be zero within match count")
    if answer_index == 0 or anchor_index + 1 != answer_index:
        _fail("anchor must be the token immediately preceding the answer-first token")

    supplied_provenance = dict(provenance)
    base_keys = PROVENANCE_KEYS - {"generation_ids_sha256", "replay_ids_sha256"}
    _require_exact_keys(supplied_provenance, base_keys, "provenance input")
    reduced_provenance = dict(supplied_provenance)
    reduced_provenance["generation_ids_sha256"] = token_ids_sha256(generation_ids)
    reduced_provenance["replay_ids_sha256"] = token_ids_sha256(replay_ids)
    _require_exact_keys(reduced_provenance, PROVENANCE_KEYS, "trajectory provenance")

    for name in (
        "model_repo",
        "model_revision",
        "dataset_repo",
        "dataset_revision",
        "split",
        "canonical_identity",
        "category",
    ):
        if not str(locals()[name]).strip():
            _fail("%s must be a non-empty string" % name)
    for name, value in (
        ("prompt_sha256", prompt_sha256),
        ("generated_completion_sha256", generated_completion_sha256),
        ("answer_span_extractor_sha256", answer_span_extractor_sha256),
        ("generated_id_text_aligner_sha256", generated_id_text_aligner_sha256),
    ):
        _require_sha256(value, name)
    for name in (
        "card_sha256",
        "renderer_manifest_sha256",
        "tokenizer_manifest_sha256",
        "generation_ids_sha256",
        "replay_ids_sha256",
    ):
        _require_sha256(reduced_provenance[name], "provenance.%s" % name)

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_repo": str(model_repo),
        "model_revision": str(model_revision),
        "dataset_repo": str(dataset_repo),
        "dataset_revision": str(dataset_revision),
        "split": str(split),
        "canonical_identity": str(canonical_identity),
        "category": str(category),
        "prompt_sha256": str(prompt_sha256),
        "generated_completion_sha256": str(generated_completion_sha256),
        "generation_length": generation_length_value,
        "replay_length": replay_length_value,
        "anchor_token_index": anchor_index,
        "answer_span_start_offset": span_start,
        "answer_span_end_offset": span_end,
        "answer_match_count": match_count,
        "selected_match_ordinal": selected_ordinal,
        "answer_first_token_index": answer_index,
        "answer_span_extractor_sha256": str(answer_span_extractor_sha256),
        "generated_id_text_aligner_sha256": str(generated_id_text_aligner_sha256),
        "generation_count": 1,
        "replay_count": 1,
        "loop_insertions": 0,
        "H": H,
        "D": D,
        "hidden_rms_l2_to_final": hidden_rms,
        "hidden_cosine_to_final": hidden_cosine,
        "hidden_cosine_distance_to_final": hidden_cosine_distance,
        "adjacent_angular_distance": adjacent_angular,
        "provenance": reduced_provenance,
    }
    _require_exact_keys(record, TOP_LEVEL_KEYS, "trajectory record")
    validate_trajectory_record(record)
    return record


__all__ = [
    "BOUNDARY_COUNT",
    "FINAL_TOLERANCE",
    "PROVENANCE_KEYS",
    "SCHEMA_VERSION",
    "TOP_LEVEL_KEYS",
    "TRANSITION_COUNT",
    "build_sanitized_trajectory_record",
    "exact_generation_replay_id_closure",
    "full_vocabulary_entropy_and_kl",
    "hidden_diagnostics",
    "raw_adjacent_angular_distance",
    "stable_float32_log_softmax",
    "token_ids_sha256",
    "validate_generation_replay_id_closure",
]
