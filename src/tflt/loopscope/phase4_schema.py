"""Closed-world schemas and identity checks for LoopScope Phase 4."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class Phase4ContractError(ValueError):
    """Raised when a Phase 4 artifact violates the frozen contract."""


BOUNDARY_IDS: Tuple[int, ...] = tuple(range(37))
WIDTHS: Tuple[int, ...] = tuple(range(2, 9))
WIDTH4_STARTS: Tuple[int, ...] = tuple(range(33))
WIDTH4_CONSENSUS_STARTS: Tuple[int, ...] = tuple(range(4, 29))
EXPECTED_SHARED_COUNT = 12032
CARD_SCHEMA_VERSION = "loopscope.phase4.pv-ek-trs-card.v1"
RECEIPT_SCHEMA_VERSION = "loopscope.phase4.card-verifier-receipt.v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40_64 = re.compile(r"^[0-9a-f]{40,64}$")
_FORBIDDEN_KEYS = {
    "answer",
    "answer_index",
    "gold",
    "gold_label",
    "label",
    "cot_content",
    "correct",
    "correctness",
    "accuracy",
    "acc",
    "generated_token",
    "generated_tokens",
    "generated_answer",
    "generated_response",
    "teacher_forcing",
    "baseline_outcome",
    "loop_outcome",
    "answer_flip",
    "full_logits",
    "logits",
    "probabilities",
    "log_probabilities",
    "hidden_states",
    "hidden_tensors",
    "residual_tensors",
}
_FORBIDDEN_KEY_FRAGMENTS = (
    "cot_content",
    "logits",
    "log_probabilities",
    "probabilities",
    "hidden_state",
    "hidden_tensor",
    "residual_tensor",
    "tensor_payload",
)
_FORBIDDEN_TARGET_SUFFIXES = (
    "answer",
    "answers",
    "cot",
    "cot_content",
    "correct",
    "correctness",
    "gold",
    "label",
    "token",
    "tokens",
)

SOURCE_RECORD_KEYS = {
    "schema_version",
    "dataset_revision",
    "split",
    "question_id",
    "category",
    "src",
    "question",
    "ordered_options",
    "canonical_identity",
    "sanitized_content_sha256",
    "rendered_prefix_sha256",
    "rendered_token_ids_sha256",
    "attention_mask_sha256",
    "last_effective_prefix_token_index",
    "renderer_provenance",
}

TRAJECTORY_RECORD_KEYS = {
    "schema_version",
    "canonical_identity",
    "category",
    "boundaries",
    "producer_provenance",
}

BOUNDARY_RECORD_KEYS = {
    "boundary_id",
    "full_vocabulary_entropy",
    "kl_to_final",
    "normalized_entropy",
    "logit_rms",
    "top1_probability_mass",
    "top10_probability_mass",
    "top100_probability_mass",
    "finite",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_bad_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Phase4ContractError("cannot load strict JSON artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise Phase4ContractError("JSON artifact must contain an object: %s" % path)
    return value


def load_phase4_card(path: Path, *, require_provenance_closed: bool = True) -> Dict[str, Any]:
    card = load_json_object(path)
    validate_phase4_card(card, require_provenance_closed=require_provenance_closed)
    return card


def validate_phase4_card(
    card: Mapping[str, Any], *, require_provenance_closed: bool = True
) -> None:
    if card.get("schema_version") != CARD_SCHEMA_VERSION:
        raise Phase4ContractError("unexpected Phase 4 card schema_version")
    if card.get("card") != "H4_PV_EK_TRS_QWEN4B_MMLUPRO_V1":
        raise Phase4ContractError("unexpected Phase 4 card identity")
    model = _mapping(card, "model")
    task = _mapping(card, "task")
    selector = _mapping(card, "selector")
    outcome = _mapping(card, "formal_outcome_recipe")
    bootstrap = _mapping(card, "bootstrap")
    panel = _mapping(card, "panel")
    provenance = _mapping(card, "provenance_closure")

    _require_equal(model, "repo", "Qwen/Qwen3-4B-Instruct-2507")
    _require_equal(model, "decoder_layers", 36)
    _require_equal(model, "dtype", "bfloat16")
    _require_equal(task, "dataset", "TIGER-Lab/MMLU-Pro")
    _require_equal(task, "task_alias", "mmlu_pro")
    _require_equal(task, "num_fewshot", 5)
    _require_equal(task, "selector_split", "test")
    _require_equal(task, "shared_population_count", EXPECTED_SHARED_COUNT)
    _require_equal(selector, "boundaries", list(BOUNDARY_IDS))
    _require_equal(selector, "primary_width", 4)
    _require_equal(selector, "width4_start_count", 33)
    _require_equal(selector, "width4_consensus_count", 25)
    _require_equal(selector, "secondary_widths", list(WIDTHS))
    _require_equal(selector, "secondary_strict_count", 154)
    _require_equal(selector, "secondary_edge_count", 224)
    _require_equal(outcome, "window_width", 4)
    _require_equal(outcome, "k", 3)
    _require_equal(outcome, "iteration_mode", "block")
    _require_equal(outcome, "strategy", "euler")
    _require_close(outcome, "step_size", 1.0 / 3.0)
    _require_equal(outcome, "cache_strategy", "first")
    _require_equal(outcome, "decode_mode", "full")
    _require_equal(bootstrap, "selector_replicates", 2000)
    _require_equal(bootstrap, "selector_seed", 20260717)
    _require_equal(panel, "fixed_comparator", "15:18")
    _require_close(panel, "competitiveness_margin_percentage_points", 0.30)
    _require_equal(panel, "selection_frequency_threshold", 0.80)

    required_exact = provenance.get("required_exact_fields")
    if not isinstance(required_exact, list) or not required_exact:
        raise Phase4ContractError("provenance_closure.required_exact_fields is missing")
    values = provenance.get("values")
    if not isinstance(values, Mapping):
        raise Phase4ContractError("provenance_closure.values must be an object")
    missing = [name for name in required_exact if not _is_exact_provenance_value(values.get(name))]
    state = provenance.get("state")
    if missing and state != "UNRESOLVED_LOCAL_EVIDENCE":
        raise Phase4ContractError("unresolved provenance must be explicit")
    if not missing and state != "CLOSED":
        raise Phase4ContractError("closed provenance values require state=CLOSED")
    if require_provenance_closed and missing:
        raise Phase4ContractError(
            "exact immutable provenance is not closed: %s" % ", ".join(missing)
        )


def scan_forbidden_fields(value: Any, *, path: str = "$", allow_options: bool = True) -> None:
    """Recursively reject outcome/gold/tensor fields on the normal producer path."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise Phase4ContractError("non-string field name at %s" % path)
            key = raw_key.strip().lower()
            target_alias = key.startswith("target_") and any(
                key.endswith(suffix) for suffix in _FORBIDDEN_TARGET_SUFFIXES
            )
            generated_alias = key.startswith("generated_") and any(
                token in key for token in ("answer", "response", "token", "cot")
            )
            tensor_alias = any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS)
            if (
                key in _FORBIDDEN_KEYS
                or key.startswith("outcome_")
                or key.endswith("_outcome")
                or target_alias
                or generated_alias
                or tensor_alias
            ):
                raise Phase4ContractError("forbidden Phase 4 selector field %s.%s" % (path, raw_key))
            if not allow_options and key in {"ordered_options", "options"}:
                raise Phase4ContractError("options are not allowed in trajectory artifacts")
            scan_forbidden_fields(child, path="%s.%s" % (path, raw_key), allow_options=allow_options)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_forbidden_fields(child, path="%s[%d]" % (path, index), allow_options=allow_options)
    elif isinstance(value, float) and not math.isfinite(value):
        raise Phase4ContractError("non-finite value at %s" % path)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise Phase4ContractError("non-JSON payload is forbidden at %s" % path)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        raise Phase4ContractError("text field must be a string")
    normalized = unicodedata.normalize("NFC", value).strip()
    return " ".join(normalized.split())


def sanitized_content_sha256(question: str, ordered_options: Sequence[str]) -> str:
    if not isinstance(ordered_options, Sequence) or isinstance(ordered_options, (str, bytes)):
        raise Phase4ContractError("ordered_options must be a sequence")
    if len(ordered_options) < 2:
        raise Phase4ContractError("ordered_options must contain at least two values")
    payload = {
        "question": normalize_text(question),
        "ordered_options": [normalize_text(item) for item in ordered_options],
    }
    return semantic_sha256(payload)


def canonical_identity(record: Mapping[str, Any]) -> str:
    question_id = record.get("question_id")
    try:
        numeric_id = int(str(question_id).strip())
    except (TypeError, ValueError) as exc:
        raise Phase4ContractError("question_id must be numeric") from exc
    if numeric_id < 0:
        raise Phase4ContractError("question_id must be non-negative")
    category = normalize_text(record.get("category"))
    src = normalize_text(record.get("src"))
    content_hash = sanitized_content_sha256(record.get("question"), record.get("ordered_options"))
    return "%d:%s:%s:%s" % (numeric_id, category, src, content_hash)


def validate_source_record(record: Mapping[str, Any]) -> None:
    scan_forbidden_fields(record)
    _closed_world(record, SOURCE_RECORD_KEYS, "source record")
    if record.get("schema_version") != "loopscope.phase4.shared-source-record.v1":
        raise Phase4ContractError("unexpected source record schema_version")
    if record.get("split") != "test":
        raise Phase4ContractError("source record split must be test")
    revision = record.get("dataset_revision")
    if not isinstance(revision, str) or not _HEX40_64.fullmatch(revision):
        raise Phase4ContractError("dataset_revision must be an exact lowercase revision")
    options = record.get("ordered_options")
    expected_content = sanitized_content_sha256(record.get("question"), options)
    if record.get("sanitized_content_sha256") != expected_content:
        raise Phase4ContractError("sanitized_content_sha256 mismatch")
    if record.get("canonical_identity") != canonical_identity(record):
        raise Phase4ContractError("canonical_identity mismatch")
    for key in (
        "rendered_prefix_sha256",
        "rendered_token_ids_sha256",
        "attention_mask_sha256",
    ):
        if not isinstance(record.get(key), str) or not _HEX64.fullmatch(record[key]):
            raise Phase4ContractError("%s must be a SHA256" % key)
    index = record.get("last_effective_prefix_token_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise Phase4ContractError("last_effective_prefix_token_index must be non-negative")
    if not isinstance(record.get("renderer_provenance"), Mapping):
        raise Phase4ContractError("renderer_provenance must be an object")


def validate_shared_identity_contract(
    expected_rows: Sequence[Mapping[str, Any]],
    observed_rows: Sequence[Mapping[str, Any]],
    *,
    require_full_population: bool = True,
) -> None:
    """Fail fast on missing, extra, reordered, or duplicate shared identities."""

    if require_full_population and len(expected_rows) != EXPECTED_SHARED_COUNT:
        raise Phase4ContractError("canonical shared manifest must contain exactly 12032 rows")
    for row in expected_rows:
        validate_source_record(row)
    expected = [row["canonical_identity"] for row in expected_rows]
    canonical_rows = sorted(
        expected_rows,
        key=lambda row: (int(str(row["question_id"]).strip()), row["sanitized_content_sha256"]),
    )
    if [row["canonical_identity"] for row in canonical_rows] != expected:
        raise Phase4ContractError("canonical shared manifest order is not question_id/hash order")
    question_id_to_hash: Dict[int, str] = {}
    for row in expected_rows:
        question_id = int(str(row["question_id"]).strip())
        content_hash = row["sanitized_content_sha256"]
        prior = question_id_to_hash.setdefault(question_id, content_hash)
        if prior != content_hash:
            raise Phase4ContractError("repeated question_id has conflicting content")
    observed = []
    closure_fields = (
        "canonical_identity",
        "sanitized_content_sha256",
        "rendered_prefix_sha256",
        "rendered_token_ids_sha256",
        "attention_mask_sha256",
    )
    for index, row in enumerate(observed_rows):
        if not isinstance(row, Mapping):
            raise Phase4ContractError("observed identity row must be an object")
        scan_forbidden_fields(row)
        identity = row.get("canonical_identity")
        if not isinstance(identity, str) or not identity:
            raise Phase4ContractError("observed canonical_identity is missing")
        observed.append(identity)
        if index < len(expected_rows):
            for field in closure_fields:
                if row.get(field) != expected_rows[index].get(field):
                    raise Phase4ContractError("shared identity/hash closure mismatch at %s" % field)
    if len(set(expected)) != len(expected):
        raise Phase4ContractError("canonical shared manifest contains duplicate identity")
    if len(set(observed)) != len(observed):
        raise Phase4ContractError("observed rows contain duplicate identity")
    expected_set, observed_set = set(expected), set(observed)
    if expected_set != observed_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        raise Phase4ContractError(
            "shared identity membership mismatch: missing=%d extra=%d" % (len(missing), len(extra))
        )
    if observed != expected:
        raise Phase4ContractError("shared identity order mismatch")


def validate_trajectory_record(record: Mapping[str, Any], *, d36_tolerance: float) -> None:
    scan_forbidden_fields(record, allow_options=False)
    _closed_world(record, TRAJECTORY_RECORD_KEYS, "trajectory record")
    if record.get("schema_version") != "loopscope.phase4.prefix-trajectory-record.v1":
        raise Phase4ContractError("unexpected trajectory schema_version")
    if not isinstance(record.get("canonical_identity"), str) or not record["canonical_identity"]:
        raise Phase4ContractError("trajectory canonical_identity is missing")
    if not isinstance(record.get("category"), str) or not record["category"].strip():
        raise Phase4ContractError("trajectory category is missing")
    boundaries = record.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != 37:
        raise Phase4ContractError("trajectory must contain exactly B_0..B_36")
    for expected_id, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise Phase4ContractError("boundary record must be an object")
        _closed_world(boundary, BOUNDARY_RECORD_KEYS, "boundary record")
        if boundary.get("boundary_id") != expected_id:
            raise Phase4ContractError("boundary order/id mismatch")
        for key in BOUNDARY_RECORD_KEYS - {"boundary_id", "finite"}:
            value = boundary.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise Phase4ContractError("boundary scalar %s must be finite" % key)
        if boundary.get("finite") is not True:
            raise Phase4ContractError("boundary finite flag must be true")
        if not 0.0 <= boundary["normalized_entropy"] <= 1.0 + 1e-6:
            raise Phase4ContractError("normalized entropy is out of range")
        masses = [
            boundary["top1_probability_mass"],
            boundary["top10_probability_mass"],
            boundary["top100_probability_mass"],
        ]
        if not (0.0 <= masses[0] <= masses[1] <= masses[2] <= 1.0 + 1e-6):
            raise Phase4ContractError("top-k masses are invalid")
    if abs(boundaries[36]["kl_to_final"]) > d36_tolerance:
        raise Phase4ContractError("D_36 exceeds the frozen tolerance")


def source_record_json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "loopscope.phase4.shared-source-record.v1",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(SOURCE_RECORD_KEYS),
        "properties": {
            "schema_version": {"const": "loopscope.phase4.shared-source-record.v1"},
            "dataset_revision": {"type": "string", "pattern": "^[0-9a-f]{40,64}$"},
            "split": {"const": "test"},
            "question_id": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "string", "pattern": "^[0-9]+$"}]},
            "category": {"type": "string", "minLength": 1},
            "src": {"type": "string", "minLength": 1},
            "question": {"type": "string"},
            "ordered_options": {"type": "array", "minItems": 2, "items": {"type": "string"}},
            "canonical_identity": {"type": "string", "minLength": 1},
            "sanitized_content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "rendered_prefix_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "rendered_token_ids_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "attention_mask_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "last_effective_prefix_token_index": {"type": "integer", "minimum": 0},
            "renderer_provenance": {"type": "object"},
        },
    }


def trajectory_record_json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "loopscope.phase4.prefix-trajectory-record.v1",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(TRAJECTORY_RECORD_KEYS),
        "properties": {
            "schema_version": {"const": "loopscope.phase4.prefix-trajectory-record.v1"},
            "canonical_identity": {"type": "string", "minLength": 1},
            "category": {"type": "string", "minLength": 1},
            "boundaries": {
                "type": "array",
                "minItems": 37,
                "maxItems": 37,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(BOUNDARY_RECORD_KEYS),
                    "properties": {
                        "boundary_id": {"type": "integer", "minimum": 0, "maximum": 36},
                        "full_vocabulary_entropy": {"type": "number"},
                        "kl_to_final": {"type": "number"},
                        "normalized_entropy": {"type": "number"},
                        "logit_rms": {"type": "number"},
                        "top1_probability_mass": {"type": "number"},
                        "top10_probability_mass": {"type": "number"},
                        "top100_probability_mass": {"type": "number"},
                        "finite": {"const": True},
                    },
                },
            },
            "producer_provenance": {"type": "object"},
        },
    }


def _closed_world(value: Mapping[str, Any], allowed: Iterable[str], context: str) -> None:
    keys = set(value)
    allowed_set = set(allowed)
    missing, extra = allowed_set - keys, keys - allowed_set
    if missing or extra:
        raise Phase4ContractError(
            "%s is not closed-world: missing=%s extra=%s"
            % (context, sorted(missing), sorted(extra))
        )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise Phase4ContractError("card.%s must be an object" % key)
    return child


def _require_equal(value: Mapping[str, Any], key: str, expected: Any) -> None:
    if value.get(key) != expected:
        raise Phase4ContractError("frozen card field %s differs" % key)


def _require_close(value: Mapping[str, Any], key: str, expected: float) -> None:
    observed = value.get(key)
    if not isinstance(observed, (int, float)) or not math.isclose(
        float(observed), expected, rel_tol=0.0, abs_tol=1e-15
    ):
        raise Phase4ContractError("frozen card field %s differs" % key)


def _is_exact_provenance_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip()) and not value.startswith("UNRESOLVED")
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return value is not None


def _bad_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant %s" % value)


__all__ = [
    "BOUNDARY_IDS",
    "CARD_SCHEMA_VERSION",
    "EXPECTED_SHARED_COUNT",
    "Phase4ContractError",
    "RECEIPT_SCHEMA_VERSION",
    "WIDTH4_CONSENSUS_STARTS",
    "WIDTH4_STARTS",
    "WIDTHS",
    "canonical_identity",
    "canonical_json_bytes",
    "file_sha256",
    "load_phase4_card",
    "sanitized_content_sha256",
    "scan_forbidden_fields",
    "semantic_sha256",
    "source_record_json_schema",
    "trajectory_record_json_schema",
    "validate_phase4_card",
    "validate_shared_identity_contract",
    "validate_source_record",
    "validate_trajectory_record",
]
