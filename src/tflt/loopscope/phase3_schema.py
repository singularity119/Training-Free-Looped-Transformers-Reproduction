"""Closed-world schemas for the LoopScope Phase 3 baseline selector.

This module is deliberately standard-library only.  Gate P3-A defines artifact
contracts and deterministic validators; it does not load MMLU, a tokenizer, or
the model.  Raw choice logits are accepted only by the synthetic/producer
constructor and are never part of a selector trajectory artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.schema import (
    SchemaError,
    attach_manifest_sha256,
    canonical_json_bytes,
    manifest_sha256,
    verify_manifest_sha256,
)


PHASE3_CARD_SCHEMA_VERSION = "loopscope.phase3.h3_card.v1"
PHASE3_CARD_ID = "H3_BASELINE_LOCAL_REVERSAL_WINDOW_RECOVERY_V1"
PHASE3_CARD_BYTE_SHA256 = "5b5cb333d0bb191ab8da1db23b7d4b36f9f4e9ccc97c16dc46c5e7aeb2ca4825"
PHASE3_CARD_SEMANTIC_SHA256 = "ca9fa5cf9b2be7ddb5d4ce598636dd5c5a5b2ec665770993e14e279214c586b6"

PHASE3_SOURCE_CONTRACT_SCHEMA_VERSION = "loopscope.phase3.source-contract.v1"
PHASE3_SOURCE_MANIFEST_SCHEMA_VERSION = "loopscope.phase3.source-manifest.v1"
PHASE3_POOL_RECORD_SCHEMA_VERSION = "loopscope.phase3.pool-record.v1"
PHASE3_POOL_MANIFEST_SCHEMA_VERSION = "loopscope.phase3.pool-manifest.v1"
PHASE3_TRAJECTORY_RECORD_SCHEMA_VERSION = "loopscope.phase3.trajectory-record.v1"
PHASE3_TRAJECTORY_MANIFEST_SCHEMA_VERSION = "loopscope.phase3.trajectory-manifest.v1"
PHASE3_ANALYSIS_SCHEMA_VERSION = "loopscope.phase3.baseline-analysis.v1"
PHASE3_VERIFIER_RECEIPT_SCHEMA_VERSION = "loopscope.phase3.card-verifier-receipt.v1"

CHOICE_ORDER = ("A", "B", "C", "D")
BOUNDARY_IDS = tuple("B_%d" % index for index in range(29))
WIDTH4_STARTS = tuple(range(25))

POOL_RECORD_KEYS = frozenset(
    (
        "schema_version",
        "identity",
        "subject",
        "split",
        "question",
        "ordered_choices",
        "rendered_prompt",
        "prompt_sha256",
        "sanitized_content_sha256",
        "renderer_provenance",
    )
)
POOL_RENDERER_KEYS = frozenset(
    (
        "dataset_repo",
        "dataset_revision",
        "task_group",
        "fewshot_split",
        "num_fewshot",
        "renderer_id",
        "renderer_source_sha256",
        "render_contract_sha256",
        "source_projection_sha256",
        "render_sha256",
    )
)
SOURCE_RECORD_KEYS = frozenset(
    (
        "identity",
        "subject",
        "split",
        "prompt_sha256",
        "sanitized_content_sha256",
        "renderer_provenance",
    )
)
TRAJECTORY_RECORD_KEYS = frozenset(
    (
        "schema_version",
        "identity",
        "subject",
        "split",
        "prompt_sha256",
        "renderer_provenance",
        "boundaries",
    )
)
TRAJECTORY_RENDERER_KEYS = frozenset(
    (
        "dataset_repo",
        "dataset_revision",
        "model_repo",
        "model_revision",
        "tokenizer_revision",
        "source_manifest_sha256",
        "pool_manifest_sha256",
        "renderer_manifest_sha256",
        "forward_type",
        "formal_forward_count_per_identity",
        "loop_insertions",
        "answer_position",
        "projection",
        "choice_order",
    )
)
BOUNDARY_RECORD_KEYS = frozenset(
    (
        "boundary_id",
        "choice_distribution",
        "choice_entropy",
        "kl_to_final",
        "cross_entropy_to_final",
        "top1_to_final_agreement",
    )
)
IDENTITY_KEYS = frozenset(("task", "doc_id", "doc_hash"))


def load_phase3_card(path: Path) -> Dict[str, Any]:
    """Load the byte-frozen card, rejecting formatting as well as semantic drift."""

    path = Path(path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != PHASE3_CARD_BYTE_SHA256:
        raise SchemaError(
            "Phase 3 card byte SHA256 mismatch: expected %s, computed %s"
            % (PHASE3_CARD_BYTE_SHA256, actual)
        )
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("Phase 3 card is not strict UTF-8 JSON") from exc
    validate_phase3_card(payload)
    return dict(payload)


def validate_phase3_card(card: Mapping[str, Any]) -> None:
    """Validate the complete planning-frozen semantic object."""

    if not isinstance(card, Mapping):
        raise SchemaError("Phase 3 card must be an object")
    try:
        digest = hashlib.sha256(canonical_json_bytes(card)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise SchemaError("Phase 3 card contains a non-JSON or non-finite value") from exc
    if digest != PHASE3_CARD_SEMANTIC_SHA256:
        raise SchemaError(
            "Phase 3 card semantic SHA256 mismatch: expected %s, computed %s"
            % (PHASE3_CARD_SEMANTIC_SHA256, digest)
        )
    if card.get("schema_version") != PHASE3_CARD_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 3 card schema")
    if card.get("card") != PHASE3_CARD_ID or card.get("status") != "PLANNING_FROZEN":
        raise SchemaError("Phase 3 card identity/status differs from the frozen contract")
    validate_historical_blind_partition(card)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_identity(value: Mapping[str, Any]) -> Dict[str, str]:
    _exact_keys(value, IDENTITY_KEYS, "sample identity")
    task = _nonempty_string(value["task"], "identity.task")
    doc_id = _nonempty_string(value["doc_id"], "identity.doc_id")
    doc_hash = str(value["doc_hash"]).strip().lower()
    _sha256(doc_hash, "identity.doc_hash")
    return {"task": task, "doc_id": doc_id, "doc_hash": doc_hash}


def identity_tuple(value: Mapping[str, Any]) -> Tuple[str, str, str]:
    item = canonical_identity(value)
    return item["task"], item["doc_id"], item["doc_hash"]


def canonical_record_key(value: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    subject = _nonempty_string(value.get("subject"), "record.subject")
    task, doc_id, doc_hash = identity_tuple(_mapping(value.get("identity"), "record.identity"))
    return subject, task, doc_id, doc_hash


def ordered_identity_sha256(values: Sequence[Mapping[str, Any]]) -> str:
    normalized = [canonical_identity(value) for value in values]
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def normalize_sanitized_string(value: Any) -> str:
    if not isinstance(value, str):
        raise SchemaError("sanitized-content inputs must be strings")
    normalized = unicodedata.normalize("NFC", value)
    return " ".join(normalized.split())


def sanitized_content_sha256(question: Any, ordered_choices: Sequence[Any]) -> str:
    if isinstance(ordered_choices, (str, bytes)) or not isinstance(ordered_choices, Sequence):
        raise SchemaError("ordered_choices must be a four-string sequence")
    if len(ordered_choices) != 4:
        raise SchemaError("ordered_choices must contain exactly A/B/C/D")
    payload = {
        "choices": [normalize_sanitized_string(value) for value in ordered_choices],
        "question": normalize_sanitized_string(question),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def reject_forbidden_selector_fields(
    payload: Any, card: Mapping[str, Any], path: Tuple[str, ...] = ()
) -> None:
    """Reject forbidden key tokens without unsafe substring matching.

    The frozen list includes single-letter ``r`` and ``q``.  Keys are therefore
    split into complete alphanumeric tokens; ``renderer_provenance`` remains
    valid while a field named exactly ``r`` or ``loop_residual`` is rejected.
    Values are not scanned: exact-key object validators define their meaning.
    """

    forbidden = card["trajectory"]["forbidden_selector_field_tokens"]
    phrases = {_normalize_field_key(value) for value in forbidden}
    singles = {value for value in phrases if "_" not in value}
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            normalized = _normalize_field_key(key)
            tokens = tuple(re.findall(r"[a-z0-9]+", key.lower()))
            if normalized in phrases or any(token in singles for token in tokens):
                raise SchemaError(
                    "forbidden selector field at %s"
                    % ".".join(path + (key,))
                )
            reject_forbidden_selector_fields(value, card, path + (key,))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            reject_forbidden_selector_fields(value, card, path + (str(index),))


def validate_pool_record(record: Mapping[str, Any], card: Mapping[str, Any]) -> Dict[str, Any]:
    validate_phase3_card(card)
    reject_forbidden_selector_fields(record, card)
    _exact_keys(record, POOL_RECORD_KEYS, "Phase 3 pool record")
    if record["schema_version"] != PHASE3_POOL_RECORD_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 3 pool record schema")
    identity = canonical_identity(_mapping(record["identity"], "pool.identity"))
    subject = _nonempty_string(record["subject"], "pool.subject")
    if record["split"] != card["task"]["trajectory_split"]:
        raise SchemaError("Phase 3 pool record must use the frozen validation split")
    question = record["question"]
    choices = record["ordered_choices"]
    expected_content = sanitized_content_sha256(question, choices)
    _sha256(record["sanitized_content_sha256"], "pool.sanitized_content_sha256")
    if record["sanitized_content_sha256"] != expected_content:
        raise SchemaError("pool sanitized-content SHA256 mismatch")
    prompt = _nonempty_string(record["rendered_prompt"], "pool.rendered_prompt")
    expected_prompt = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    _sha256(record["prompt_sha256"], "pool.prompt_sha256")
    if record["prompt_sha256"] != expected_prompt:
        raise SchemaError("pool rendered-prompt SHA256 mismatch")
    renderer = validate_pool_renderer(record["renderer_provenance"], card)
    if renderer["render_sha256"] != expected_prompt:
        raise SchemaError("renderer render hash differs from pool prompt hash")
    return {
        "schema_version": PHASE3_POOL_RECORD_SCHEMA_VERSION,
        "identity": identity,
        "subject": subject,
        "split": str(record["split"]),
        "question": str(question),
        "ordered_choices": [str(value) for value in choices],
        "rendered_prompt": prompt,
        "prompt_sha256": expected_prompt,
        "sanitized_content_sha256": expected_content,
        "renderer_provenance": renderer,
    }


def validate_pool_renderer(value: Any, card: Mapping[str, Any]) -> Dict[str, Any]:
    renderer = _mapping(value, "pool.renderer_provenance")
    _exact_keys(renderer, POOL_RENDERER_KEYS, "pool.renderer_provenance")
    if renderer["dataset_repo"] != card["task"]["dataset"]:
        raise SchemaError("pool renderer dataset repo differs from the card")
    if renderer["dataset_revision"] != card["task"]["dataset_revision"]:
        raise SchemaError("pool renderer dataset revision differs from the card")
    if renderer["task_group"] != card["task"]["task_group"]:
        raise SchemaError("pool renderer task group differs from the card")
    if renderer["fewshot_split"] != card["task"]["fewshot_split"]:
        raise SchemaError("pool renderer few-shot split differs from the card")
    if _strict_int(renderer["num_fewshot"], "renderer.num_fewshot") != card["task"][
        "num_fewshot"
    ]:
        raise SchemaError("pool renderer few-shot count differs from the card")
    _nonempty_string(renderer["renderer_id"], "renderer.renderer_id")
    for key in (
        "renderer_source_sha256",
        "render_contract_sha256",
        "source_projection_sha256",
        "render_sha256",
    ):
        _sha256(renderer[key], "renderer.%s" % key)
    return dict(renderer)


def source_record_from_pool_record(
    record: Mapping[str, Any], card: Mapping[str, Any]
) -> Dict[str, Any]:
    normalized = validate_pool_record(record, card)
    return {
        "identity": normalized["identity"],
        "subject": normalized["subject"],
        "split": normalized["split"],
        "prompt_sha256": normalized["prompt_sha256"],
        "sanitized_content_sha256": normalized["sanitized_content_sha256"],
        "renderer_provenance": normalized["renderer_provenance"],
    }


def validate_source_record(record: Mapping[str, Any], card: Mapping[str, Any]) -> Dict[str, Any]:
    reject_forbidden_selector_fields(record, card)
    _exact_keys(record, SOURCE_RECORD_KEYS, "Phase 3 source record")
    identity = canonical_identity(_mapping(record["identity"], "source.identity"))
    subject = _nonempty_string(record["subject"], "source.subject")
    if record["split"] != card["task"]["trajectory_split"]:
        raise SchemaError("Phase 3 source record must use validation")
    _sha256(record["prompt_sha256"], "source.prompt_sha256")
    _sha256(record["sanitized_content_sha256"], "source.sanitized_content_sha256")
    renderer = validate_pool_renderer(record["renderer_provenance"], card)
    if renderer["render_sha256"] != record["prompt_sha256"]:
        raise SchemaError("source renderer/prompt hashes disagree")
    return {
        "identity": identity,
        "subject": subject,
        "split": str(record["split"]),
        "prompt_sha256": str(record["prompt_sha256"]),
        "sanitized_content_sha256": str(record["sanitized_content_sha256"]),
        "renderer_provenance": renderer,
    }


def validate_probability_vector(value: Any, card: Mapping[str, Any]) -> List[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise SchemaError("choice_distribution must be a four-element list")
    result = [_finite_number(item, "choice probability") for item in value]
    if any(item <= 0.0 for item in result):
        raise SchemaError("every choice probability must be strictly greater than zero")
    tolerance = float(card["numeric_contract"]["choice_probability_sum_abs_tolerance"])
    if abs(sum(result) - 1.0) > tolerance:
        raise SchemaError("choice probabilities are not normalized within tolerance")
    return result


def boundary_metrics_from_probabilities(
    probabilities: Sequence[float], final_probabilities: Sequence[float]
) -> Dict[str, Any]:
    p = [float(value) for value in probabilities]
    final = [float(value) for value in final_probabilities]
    if len(p) != 4 or len(final) != 4:
        raise SchemaError("boundary metrics require four-choice distributions")
    if any(not math.isfinite(value) or value <= 0.0 for value in p + final):
        raise SchemaError("boundary metrics require finite positive probabilities")
    entropy = -sum(value * math.log(value) for value in p)
    divergence = sum(value * math.log(value / target) for value, target in zip(p, final))
    cross_entropy = -sum(value * math.log(target) for value, target in zip(p, final))
    top1 = max(range(4), key=lambda index: p[index])
    final_top1 = max(range(4), key=lambda index: final[index])
    return {
        "choice_entropy": entropy,
        "kl_to_final": divergence,
        "cross_entropy_to_final": cross_entropy,
        "top1_to_final_agreement": top1 == final_top1,
    }


def validate_trajectory_record(
    record: Mapping[str, Any],
    card: Mapping[str, Any],
    expected_source: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    validate_phase3_card(card)
    reject_forbidden_selector_fields(record, card)
    _exact_keys(record, TRAJECTORY_RECORD_KEYS, "Phase 3 trajectory record")
    if record["schema_version"] != PHASE3_TRAJECTORY_RECORD_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 3 trajectory record schema")
    identity = canonical_identity(_mapping(record["identity"], "trajectory.identity"))
    subject = _nonempty_string(record["subject"], "trajectory.subject")
    if record["split"] != card["task"]["trajectory_split"]:
        raise SchemaError("trajectory record must use validation")
    _sha256(record["prompt_sha256"], "trajectory.prompt_sha256")
    renderer = validate_trajectory_renderer(record["renderer_provenance"], card)
    boundaries = record["boundaries"]
    if not isinstance(boundaries, list) or len(boundaries) != len(BOUNDARY_IDS):
        raise SchemaError("trajectory must contain exactly B_0 through B_28")
    actual_ids = [item.get("boundary_id") if isinstance(item, Mapping) else None for item in boundaries]
    if actual_ids != list(BOUNDARY_IDS):
        raise SchemaError("trajectory boundary IDs/order must be exactly B_0 through B_28")
    final_boundary = _mapping(boundaries[-1], "trajectory.boundaries.B_28")
    _exact_keys(final_boundary, BOUNDARY_RECORD_KEYS, "trajectory boundary B_28")
    final_probabilities = validate_probability_vector(final_boundary["choice_distribution"], card)
    normalized_boundaries = []
    for expected_id, raw_boundary in zip(BOUNDARY_IDS, boundaries):
        boundary = _mapping(raw_boundary, "trajectory boundary %s" % expected_id)
        _exact_keys(boundary, BOUNDARY_RECORD_KEYS, "trajectory boundary %s" % expected_id)
        if boundary["boundary_id"] != expected_id:
            raise SchemaError("trajectory boundary order differs from B_0...B_28")
        probabilities = validate_probability_vector(boundary["choice_distribution"], card)
        computed = boundary_metrics_from_probabilities(probabilities, final_probabilities)
        for key in ("choice_entropy", "kl_to_final", "cross_entropy_to_final"):
            observed = _finite_number(boundary[key], "boundary.%s" % key)
            _assert_close(observed, float(computed[key]), card, "boundary.%s" % key)
        agreement = boundary["top1_to_final_agreement"]
        if type(agreement) is not bool or agreement != computed["top1_to_final_agreement"]:
            raise SchemaError("boundary top1-to-final agreement mismatch")
        _assert_close(
            float(boundary["cross_entropy_to_final"]),
            float(boundary["choice_entropy"]) + float(boundary["kl_to_final"]),
            card,
            "CE=H+D identity",
        )
        normalized_boundaries.append(
            {
                "boundary_id": expected_id,
                "choice_distribution": probabilities,
                "choice_entropy": float(boundary["choice_entropy"]),
                "kl_to_final": float(boundary["kl_to_final"]),
                "cross_entropy_to_final": float(boundary["cross_entropy_to_final"]),
                "top1_to_final_agreement": agreement,
            }
        )
    final = normalized_boundaries[-1]
    _assert_close(final["kl_to_final"], 0.0, card, "B_28 KL closure")
    _assert_close(
        final["cross_entropy_to_final"], final["choice_entropy"], card, "B_28 CE closure"
    )
    if not final["top1_to_final_agreement"]:
        raise SchemaError("B_28 must agree with its own top-1")
    normalized = {
        "schema_version": PHASE3_TRAJECTORY_RECORD_SCHEMA_VERSION,
        "identity": identity,
        "subject": subject,
        "split": str(record["split"]),
        "prompt_sha256": str(record["prompt_sha256"]),
        "renderer_provenance": renderer,
        "boundaries": normalized_boundaries,
    }
    if expected_source is not None:
        source = validate_source_record(expected_source, card)
        for key in ("identity", "subject", "split", "prompt_sha256"):
            if normalized[key] != source[key]:
                raise SchemaError("trajectory record differs from signed source at %s" % key)
        if renderer["source_manifest_sha256"] == renderer["pool_manifest_sha256"]:
            raise SchemaError("source and pool manifests must retain distinct artifact identities")
    return normalized


def validate_trajectory_renderer(value: Any, card: Mapping[str, Any]) -> Dict[str, Any]:
    renderer = _mapping(value, "trajectory.renderer_provenance")
    _exact_keys(renderer, TRAJECTORY_RENDERER_KEYS, "trajectory.renderer_provenance")
    exact = {
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "forward_type": card["trajectory"]["forward_type"],
        "formal_forward_count_per_identity": card["trajectory"][
            "formal_forward_count_per_identity"
        ],
        "loop_insertions": card["trajectory"]["loop_insertions"],
        "answer_position": card["trajectory"]["answer_position"],
        "projection": card["trajectory"]["projection"],
        "choice_order": card["trajectory"]["choice_order"],
    }
    for key, expected in exact.items():
        if renderer[key] != expected:
            raise SchemaError("trajectory renderer %s differs from the card" % key)
    revision = str(renderer["tokenizer_revision"]).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise SchemaError("tokenizer_revision must be an exact lowercase commit/hash")
    for key in ("source_manifest_sha256", "pool_manifest_sha256", "renderer_manifest_sha256"):
        _sha256(renderer[key], "trajectory.renderer.%s" % key)
    return dict(renderer)


def make_boundary_record(
    boundary_id: str,
    probabilities: Sequence[float],
    final_probabilities: Sequence[float],
) -> Dict[str, Any]:
    metrics = boundary_metrics_from_probabilities(probabilities, final_probabilities)
    return {
        "boundary_id": boundary_id,
        "choice_distribution": [float(value) for value in probabilities],
        "choice_entropy": metrics["choice_entropy"],
        "kl_to_final": metrics["kl_to_final"],
        "cross_entropy_to_final": metrics["cross_entropy_to_final"],
        "top1_to_final_agreement": metrics["top1_to_final_agreement"],
    }


def validate_historical_blind_partition(card: Mapping[str, Any]) -> None:
    historical = _parse_window_partition(card["partition"]["historical_13"], "historical_13")
    blind = _parse_window_partition(card["partition"]["blind_12"], "blind_12")
    if len(historical) != 13 or len(blind) != 12:
        raise SchemaError("historical/blind width-4 partition must contain 13 and 12 cells")
    if set(historical) & set(blind):
        raise SchemaError("historical and blind width-4 partitions overlap")
    if set(historical) | set(blind) != set(WIDTH4_STARTS):
        raise SchemaError("historical/blind partition does not close over all width-4 starts")


def parse_window(value: Any, width: int = 4) -> Tuple[int, int]:
    match = re.fullmatch(r"([0-9]+):([0-9]+)", str(value))
    if match is None:
        raise SchemaError("window must use inclusive start:end syntax")
    start, end = int(match.group(1)), int(match.group(2))
    if end - start + 1 != int(width):
        raise SchemaError("window width differs from the frozen width")
    if start < 0 or end > 27:
        raise SchemaError("window lies outside the 28-layer model")
    return start, end


def pool_record_json_schema() -> Dict[str, Any]:
    """Return the interoperable JSON Schema; Python validators remain authoritative."""

    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    identity = {
        "type": "object",
        "required": ["task", "doc_id", "doc_hash"],
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "doc_id": {"type": "string", "minLength": 1},
            "doc_hash": sha,
        },
        "additionalProperties": False,
    }
    renderer = {
        "type": "object",
        "required": sorted(POOL_RENDERER_KEYS),
        "properties": {
            "dataset_repo": {"const": "cais/mmlu"},
            "dataset_revision": {"const": "c30699e8356da336a370243923dbaf21066bb9fe"},
            "task_group": {"const": "mmlu"},
            "fewshot_split": {"const": "dev"},
            "num_fewshot": {"const": 5},
            "renderer_id": {"type": "string", "minLength": 1},
            "renderer_source_sha256": sha,
            "render_contract_sha256": sha,
            "source_projection_sha256": sha,
            "render_sha256": sha,
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PHASE3_POOL_RECORD_SCHEMA_VERSION,
        "type": "object",
        "required": sorted(POOL_RECORD_KEYS),
        "properties": {
            "schema_version": {"const": PHASE3_POOL_RECORD_SCHEMA_VERSION},
            "identity": identity,
            "subject": {"type": "string", "minLength": 1},
            "split": {"const": "validation"},
            "question": {"type": "string"},
            "ordered_choices": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "string"},
            },
            "rendered_prompt": {"type": "string", "minLength": 1},
            "prompt_sha256": sha,
            "sanitized_content_sha256": sha,
            "renderer_provenance": renderer,
        },
        "additionalProperties": False,
    }


def trajectory_record_json_schema() -> Dict[str, Any]:
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    renderer_properties: Dict[str, Any] = {
        "dataset_repo": {"const": "cais/mmlu"},
        "dataset_revision": {"const": "c30699e8356da336a370243923dbaf21066bb9fe"},
        "model_repo": {"const": "Qwen/Qwen3-1.7B-Base"},
        "model_revision": {"const": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"},
        "tokenizer_revision": {"type": "string", "pattern": "^[0-9a-f]{40,64}$"},
        "source_manifest_sha256": sha,
        "pool_manifest_sha256": sha,
        "renderer_manifest_sha256": sha,
        "forward_type": {"const": "native_no_loop"},
        "formal_forward_count_per_identity": {"const": 1},
        "loop_insertions": {"const": 0},
        "answer_position": {"const": "final_pre_answer_prompt_token"},
        "projection": {
            "const": "raw_direct_logit_lens_through_frozen_final_norm_and_lm_head"
        },
        "choice_order": {"const": list(CHOICE_ORDER)},
    }
    boundary = {
        "type": "object",
        "required": sorted(BOUNDARY_RECORD_KEYS),
        "properties": {
            "boundary_id": {"type": "string", "pattern": "^B_([0-9]|1[0-9]|2[0-8])$"},
            "choice_distribution": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number", "exclusiveMinimum": 0},
            },
            "choice_entropy": {"type": "number"},
            "kl_to_final": {"type": "number"},
            "cross_entropy_to_final": {"type": "number"},
            "top1_to_final_agreement": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PHASE3_TRAJECTORY_RECORD_SCHEMA_VERSION,
        "type": "object",
        "required": sorted(TRAJECTORY_RECORD_KEYS),
        "properties": {
            "schema_version": {"const": PHASE3_TRAJECTORY_RECORD_SCHEMA_VERSION},
            "identity": pool_record_json_schema()["properties"]["identity"],
            "subject": {"type": "string", "minLength": 1},
            "split": {"const": "validation"},
            "prompt_sha256": sha,
            "renderer_provenance": {
                "type": "object",
                "required": sorted(TRAJECTORY_RENDERER_KEYS),
                "properties": renderer_properties,
                "additionalProperties": False,
            },
            "boundaries": {
                "type": "array",
                "minItems": 29,
                "maxItems": 29,
                "items": boundary,
            },
        },
        "additionalProperties": False,
    }


def _parse_window_partition(values: Any, name: str) -> List[int]:
    if not isinstance(values, list):
        raise SchemaError("%s must be a list" % name)
    starts = [parse_window(value)[0] for value in values]
    if len(set(starts)) != len(starts):
        raise SchemaError("%s contains duplicate windows" % name)
    return starts


def _normalize_field_key(value: Any) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError("%s must be an object" % context)
    return value


def _exact_keys(value: Any, expected: Iterable[str], context: str) -> None:
    mapping = _mapping(value, context)
    actual = set(mapping)
    frozen = set(expected)
    if actual != frozen:
        raise SchemaError(
            "%s fields differ; missing=%s extra=%s"
            % (context, sorted(frozen - actual), sorted(actual - frozen))
        )


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("%s must be a non-empty string" % context)
    return value.strip()


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError("%s must be an integer" % context)
    return int(value)


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError("%s must be a finite number" % context)
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError("%s must be a finite number" % context)
    return result


def _sha256(value: Any, context: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise SchemaError("%s must be a lowercase SHA256" % context)
    return text


def _assert_close(observed: float, expected: float, card: Mapping[str, Any], context: str) -> None:
    absolute = float(card["numeric_contract"]["recomputed_scalar_abs_tolerance"])
    relative = float(card["numeric_contract"]["recomputed_scalar_rel_tolerance"])
    if not math.isclose(float(observed), float(expected), rel_tol=relative, abs_tol=absolute):
        raise SchemaError(
            "%s mismatch: observed %.17g expected %.17g" % (context, observed, expected)
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant %s" % value)


__all__ = [
    "BOUNDARY_IDS",
    "CHOICE_ORDER",
    "PHASE3_ANALYSIS_SCHEMA_VERSION",
    "PHASE3_CARD_BYTE_SHA256",
    "PHASE3_CARD_ID",
    "PHASE3_CARD_SCHEMA_VERSION",
    "PHASE3_POOL_MANIFEST_SCHEMA_VERSION",
    "PHASE3_POOL_RECORD_SCHEMA_VERSION",
    "PHASE3_SOURCE_CONTRACT_SCHEMA_VERSION",
    "PHASE3_SOURCE_MANIFEST_SCHEMA_VERSION",
    "PHASE3_TRAJECTORY_MANIFEST_SCHEMA_VERSION",
    "PHASE3_TRAJECTORY_RECORD_SCHEMA_VERSION",
    "PHASE3_VERIFIER_RECEIPT_SCHEMA_VERSION",
    "WIDTH4_STARTS",
    "attach_manifest_sha256",
    "boundary_metrics_from_probabilities",
    "canonical_identity",
    "canonical_json_bytes",
    "canonical_record_key",
    "file_sha256",
    "identity_tuple",
    "load_phase3_card",
    "make_boundary_record",
    "manifest_sha256",
    "normalize_sanitized_string",
    "ordered_identity_sha256",
    "parse_window",
    "pool_record_json_schema",
    "reject_forbidden_selector_fields",
    "sanitized_content_sha256",
    "source_record_from_pool_record",
    "trajectory_record_json_schema",
    "validate_historical_blind_partition",
    "validate_phase3_card",
    "validate_pool_record",
    "validate_probability_vector",
    "validate_source_record",
    "validate_trajectory_record",
    "verify_manifest_sha256",
]
