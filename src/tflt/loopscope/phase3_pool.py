"""Phase 3 validation-pool/source closure and leakage checks.

All helpers operate on explicit in-memory artifacts.  They never glob paths or
discover datasets, which keeps P3-A outcome-blind and makes the P3-B producer
boundary auditable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.schema import SchemaError
from tflt.loopscope.phase3_schema import (
    PHASE3_CARD_BYTE_SHA256,
    PHASE3_POOL_MANIFEST_SCHEMA_VERSION,
    PHASE3_SOURCE_CONTRACT_SCHEMA_VERSION,
    PHASE3_SOURCE_MANIFEST_SCHEMA_VERSION,
    PHASE3_TRAJECTORY_MANIFEST_SCHEMA_VERSION,
    attach_manifest_sha256,
    canonical_identity,
    canonical_json_bytes,
    canonical_record_key,
    identity_tuple,
    ordered_identity_sha256,
    source_record_from_pool_record,
    validate_historical_blind_partition,
    validate_phase3_card,
    validate_pool_record,
    validate_source_record,
    validate_trajectory_record,
    verify_manifest_sha256,
)


SOURCE_CONTRACT_KEYS = frozenset(
    (
        "schema_version",
        "artifact_role",
        "status",
        "card_sha256",
        "dataset",
        "population",
        "identity_contract",
        "content_hash_contract",
        "selector_boundary",
        "live_source_bindings",
        "input_artifact_allowlist",
        "external_actions_performed",
        "manifest_sha256",
    )
)
SOURCE_MANIFEST_KEYS = frozenset(
    (
        "schema_version",
        "artifact_role",
        "card_sha256",
        "dataset",
        "record_count",
        "subject_count",
        "sampling",
        "canonical_record_order",
        "records",
        "ordered_identity_sha256",
        "ordered_source_record_sha256",
        "manifest_sha256",
    )
)
POOL_MANIFEST_KEYS = frozenset(
    (
        "schema_version",
        "artifact_role",
        "card_sha256",
        "source_manifest_sha256",
        "record_count",
        "subject_count",
        "sampling",
        "canonical_record_order",
        "records",
        "ordered_identity_sha256",
        "ordered_prompt_sha256",
        "ordered_content_sha256",
        "manifest_sha256",
    )
)
POOL_MANIFEST_RECORD_KEYS = frozenset(
    ("identity", "subject", "split", "prompt_sha256", "sanitized_content_sha256")
)
TRAJECTORY_MANIFEST_KEYS = frozenset(
    (
        "schema_version",
        "artifact_role",
        "card_sha256",
        "source_manifest_sha256",
        "pool_manifest_sha256",
        "forward_type",
        "formal_forward_count_per_identity",
        "loop_insertions",
        "record_count",
        "subject_count",
        "canonical_record_order",
        "records",
        "ordered_identity_sha256",
        "ordered_trajectory_record_sha256",
        "manifest_sha256",
    )
)
TRAJECTORY_MANIFEST_RECORD_KEYS = frozenset(
    ("identity", "subject", "split", "prompt_sha256", "record_sha256")
)

ALLOWED_INPUT_ARTIFACT_KINDS = frozenset(
    (
        "frozen_phase3_card",
        "p3a_synthetic_fixture",
        "planning_signed_validation_source_manifest",
        "validation1531_pool",
        "validation1531_no_loop_trajectory",
        "test14042_identity_content_metadata_only",
    )
)


def make_source_contract(card: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the P3-A declarative source contract (never a fake live manifest)."""

    validate_phase3_card(card)
    payload: Dict[str, Any] = {
        "schema_version": PHASE3_SOURCE_CONTRACT_SCHEMA_VERSION,
        "artifact_role": "p3a_validation1531_source_contract",
        "status": "CONTRACT_ONLY_NO_LIVE_RECORDS",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "dataset": {
            "repo": card["task"]["dataset"],
            "revision": card["task"]["dataset_revision"],
            "task_group": card["task"]["task_group"],
            "split": card["task"]["trajectory_split"],
            "fewshot_split": card["task"]["fewshot_split"],
            "num_fewshot": card["task"]["num_fewshot"],
        },
        "population": {
            "record_count": card["task"]["trajectory_sample_count"],
            "subject_count": card["task"]["trajectory_subject_count"],
            "sampling": card["task"]["trajectory_sampling"],
            "formal_records_per_identity": 1,
        },
        "identity_contract": dict(card["identity"]),
        "content_hash_contract": dict(card["sanitized_content_hash"]),
        "selector_boundary": {
            "allowed_fields": list(card["trajectory"]["allowed_selector_fields"]),
            "forbidden_field_tokens": list(
                card["trajectory"]["forbidden_selector_field_tokens"]
            ),
            "unknown_fields": card["trajectory"]["closed_world_unknown_fields"],
        },
        "live_source_bindings": {
            "status": "UNMATERIALIZED_UNTIL_SEPARATELY_AUTHORIZED_P3B",
            "renderer_manifest_sha256": None,
            "source_projection_sha256": None,
            "ordered_identity_sha256": None,
        },
        "input_artifact_allowlist": sorted(ALLOWED_INPUT_ARTIFACT_KINDS),
        "external_actions_performed": [],
    }
    attach_manifest_sha256(payload)
    validate_source_contract(payload, card)
    return payload


def validate_source_contract(value: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    validate_phase3_card(card)
    _exact_keys(value, SOURCE_CONTRACT_KEYS, "Phase 3 source contract")
    verify_manifest_sha256(value)
    expected = make_source_contract_without_validation(card)
    if value != expected:
        raise SchemaError("Phase 3 source contract differs from the frozen P3-A contract")


def make_source_contract_without_validation(card: Mapping[str, Any]) -> Dict[str, Any]:
    """Internal non-recursive constructor used by the validator."""

    payload: Dict[str, Any] = {
        "schema_version": PHASE3_SOURCE_CONTRACT_SCHEMA_VERSION,
        "artifact_role": "p3a_validation1531_source_contract",
        "status": "CONTRACT_ONLY_NO_LIVE_RECORDS",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "dataset": {
            "repo": card["task"]["dataset"],
            "revision": card["task"]["dataset_revision"],
            "task_group": card["task"]["task_group"],
            "split": card["task"]["trajectory_split"],
            "fewshot_split": card["task"]["fewshot_split"],
            "num_fewshot": card["task"]["num_fewshot"],
        },
        "population": {
            "record_count": card["task"]["trajectory_sample_count"],
            "subject_count": card["task"]["trajectory_subject_count"],
            "sampling": card["task"]["trajectory_sampling"],
            "formal_records_per_identity": 1,
        },
        "identity_contract": dict(card["identity"]),
        "content_hash_contract": dict(card["sanitized_content_hash"]),
        "selector_boundary": {
            "allowed_fields": list(card["trajectory"]["allowed_selector_fields"]),
            "forbidden_field_tokens": list(
                card["trajectory"]["forbidden_selector_field_tokens"]
            ),
            "unknown_fields": card["trajectory"]["closed_world_unknown_fields"],
        },
        "live_source_bindings": {
            "status": "UNMATERIALIZED_UNTIL_SEPARATELY_AUTHORIZED_P3B",
            "renderer_manifest_sha256": None,
            "source_projection_sha256": None,
            "ordered_identity_sha256": None,
        },
        "input_artifact_allowlist": sorted(ALLOWED_INPUT_ARTIFACT_KINDS),
        "external_actions_performed": [],
    }
    attach_manifest_sha256(payload)
    return payload


def validate_input_artifact_kind(value: Any) -> str:
    if not isinstance(value, str) or value not in ALLOWED_INPUT_ARTIFACT_KINDS:
        raise SchemaError("artifact kind is outside the Phase 3 explicit input allowlist")
    return value


def make_source_manifest(
    pool_records: Sequence[Mapping[str, Any]], card: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build the future planning-signed live source manifest from explicit records."""

    validate_phase3_card(card)
    normalized_pool = [validate_pool_record(value, card) for value in pool_records]
    normalized_pool.sort(key=canonical_record_key)
    source_records = [source_record_from_pool_record(value, card) for value in normalized_pool]
    payload: Dict[str, Any] = {
        "schema_version": PHASE3_SOURCE_MANIFEST_SCHEMA_VERSION,
        "artifact_role": "planning_signed_validation_source_manifest",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "dataset": {
            "repo": card["task"]["dataset"],
            "revision": card["task"]["dataset_revision"],
            "task_group": card["task"]["task_group"],
            "split": card["task"]["trajectory_split"],
            "fewshot_split": card["task"]["fewshot_split"],
            "num_fewshot": card["task"]["num_fewshot"],
        },
        "record_count": len(source_records),
        "subject_count": len({record["subject"] for record in source_records}),
        "sampling": card["task"]["trajectory_sampling"],
        "canonical_record_order": list(card["identity"]["canonical_record_order"]),
        "records": source_records,
        "ordered_identity_sha256": ordered_identity_sha256(
            [record["identity"] for record in source_records]
        ),
        "ordered_source_record_sha256": hashlib.sha256(
            canonical_json_bytes(source_records)
        ).hexdigest(),
    }
    attach_manifest_sha256(payload)
    validate_source_manifest(payload, card)
    return payload


def validate_source_manifest(value: Mapping[str, Any], card: Mapping[str, Any]) -> List[Dict[str, Any]]:
    validate_phase3_card(card)
    _exact_keys(value, SOURCE_MANIFEST_KEYS, "Phase 3 source manifest")
    verify_manifest_sha256(value)
    if value["schema_version"] != PHASE3_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 3 source manifest schema")
    if value["artifact_role"] != "planning_signed_validation_source_manifest":
        raise SchemaError("source manifest artifact role differs")
    if value["card_sha256"] != PHASE3_CARD_BYTE_SHA256:
        raise SchemaError("source manifest is bound to a different Phase 3 card")
    expected_dataset = {
        "repo": card["task"]["dataset"],
        "revision": card["task"]["dataset_revision"],
        "task_group": card["task"]["task_group"],
        "split": card["task"]["trajectory_split"],
        "fewshot_split": card["task"]["fewshot_split"],
        "num_fewshot": card["task"]["num_fewshot"],
    }
    if value["dataset"] != expected_dataset:
        raise SchemaError("source manifest dataset contract differs from the card")
    records = value["records"]
    if not isinstance(records, list):
        raise SchemaError("source manifest records must be a list")
    expected_count = int(card["task"]["trajectory_sample_count"])
    expected_subjects = int(card["task"]["trajectory_subject_count"])
    if _strict_int(value["record_count"], "source.record_count") != expected_count:
        raise SchemaError("source manifest must contain exactly 1,531 records")
    if len(records) != expected_count:
        raise SchemaError("source manifest record list length differs from 1,531")
    if _strict_int(value["subject_count"], "source.subject_count") != expected_subjects:
        raise SchemaError("source manifest must contain exactly 57 subjects")
    if value["sampling"] != "none":
        raise SchemaError("Phase 3 primary source manifest forbids sampling")
    if value["canonical_record_order"] != card["identity"]["canonical_record_order"]:
        raise SchemaError("source manifest canonical order contract differs")
    normalized = [validate_source_record(record, card) for record in records]
    _validate_population_order_and_uniqueness(normalized, expected_subjects)
    expected_identity_hash = ordered_identity_sha256([record["identity"] for record in normalized])
    if value["ordered_identity_sha256"] != expected_identity_hash:
        raise SchemaError("source ordered identity SHA256 mismatch")
    expected_records_hash = hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()
    if value["ordered_source_record_sha256"] != expected_records_hash:
        raise SchemaError("source ordered record SHA256 mismatch")
    return normalized


def validate_pool_records(
    pool_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    source_records = validate_source_manifest(source_manifest, card)
    if not isinstance(pool_records, list):
        raise SchemaError("Phase 3 pool must be a JSONL-equivalent list")
    expected_count = int(card["task"]["trajectory_sample_count"])
    if len(pool_records) != expected_count:
        raise SchemaError("Phase 3 pool must contain exactly 1,531 records")
    normalized = [validate_pool_record(value, card) for value in pool_records]
    _validate_population_order_and_uniqueness(
        normalized, int(card["task"]["trajectory_subject_count"])
    )
    projected = [source_record_from_pool_record(value, card) for value in normalized]
    if projected != source_records:
        raise SchemaError("pool records do not exactly close against the signed source manifest")
    return normalized


def make_pool_manifest(
    pool_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = validate_pool_records(pool_records, source_manifest, card)
    records = [
        {
            "identity": value["identity"],
            "subject": value["subject"],
            "split": value["split"],
            "prompt_sha256": value["prompt_sha256"],
            "sanitized_content_sha256": value["sanitized_content_sha256"],
        }
        for value in normalized
    ]
    payload: Dict[str, Any] = {
        "schema_version": PHASE3_POOL_MANIFEST_SCHEMA_VERSION,
        "artifact_role": "validation1531_pool",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "record_count": len(records),
        "subject_count": len({value["subject"] for value in records}),
        "sampling": "none",
        "canonical_record_order": list(card["identity"]["canonical_record_order"]),
        "records": records,
        "ordered_identity_sha256": ordered_identity_sha256(
            [value["identity"] for value in records]
        ),
        "ordered_prompt_sha256": hashlib.sha256(
            canonical_json_bytes([value["prompt_sha256"] for value in records])
        ).hexdigest(),
        "ordered_content_sha256": hashlib.sha256(
            canonical_json_bytes([value["sanitized_content_sha256"] for value in records])
        ).hexdigest(),
    }
    attach_manifest_sha256(payload)
    validate_pool_manifest(payload, source_manifest, card)
    return payload


def validate_pool_manifest(
    value: Mapping[str, Any], source_manifest: Mapping[str, Any], card: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    source_records = validate_source_manifest(source_manifest, card)
    _exact_keys(value, POOL_MANIFEST_KEYS, "Phase 3 pool manifest")
    verify_manifest_sha256(value)
    if value["schema_version"] != PHASE3_POOL_MANIFEST_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 3 pool manifest schema")
    if value["artifact_role"] != "validation1531_pool":
        raise SchemaError("pool manifest artifact role differs")
    if value["card_sha256"] != PHASE3_CARD_BYTE_SHA256:
        raise SchemaError("pool manifest is bound to a different card")
    if value["source_manifest_sha256"] != source_manifest["manifest_sha256"]:
        raise SchemaError("pool manifest is bound to a different source manifest")
    records = value["records"]
    if not isinstance(records, list) or len(records) != len(source_records):
        raise SchemaError("pool manifest record closure differs from source")
    normalized = []
    for record in records:
        _exact_keys(record, POOL_MANIFEST_RECORD_KEYS, "pool manifest record")
        normalized.append(
            {
                "identity": canonical_identity(record["identity"]),
                "subject": _nonempty_string(record["subject"], "pool manifest subject"),
                "split": str(record["split"]),
                "prompt_sha256": _sha256(record["prompt_sha256"], "pool manifest prompt"),
                "sanitized_content_sha256": _sha256(
                    record["sanitized_content_sha256"], "pool manifest content"
                ),
            }
        )
    projected_source = [
        {key: source[key] for key in POOL_MANIFEST_RECORD_KEYS} for source in source_records
    ]
    if normalized != projected_source:
        raise SchemaError("pool manifest metadata differs from signed source records")
    expected_count = int(card["task"]["trajectory_sample_count"])
    expected_subjects = int(card["task"]["trajectory_subject_count"])
    if value["record_count"] != expected_count or value["subject_count"] != expected_subjects:
        raise SchemaError("pool manifest population shape differs from the card")
    if value["sampling"] != "none":
        raise SchemaError("pool manifest sampling must be none")
    if value["canonical_record_order"] != card["identity"]["canonical_record_order"]:
        raise SchemaError("pool manifest canonical order differs from the card")
    expected_hashes = {
        "ordered_identity_sha256": ordered_identity_sha256(
            [record["identity"] for record in normalized]
        ),
        "ordered_prompt_sha256": hashlib.sha256(
            canonical_json_bytes([record["prompt_sha256"] for record in normalized])
        ).hexdigest(),
        "ordered_content_sha256": hashlib.sha256(
            canonical_json_bytes([record["sanitized_content_sha256"] for record in normalized])
        ).hexdigest(),
    }
    for key, expected in expected_hashes.items():
        if value[key] != expected:
            raise SchemaError("pool manifest %s mismatch" % key)
    return normalized


def make_trajectory_manifest(
    trajectory_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = validate_trajectory_records(
        trajectory_records, source_manifest, pool_manifest, card
    )
    records = [
        {
            "identity": value["identity"],
            "subject": value["subject"],
            "split": value["split"],
            "prompt_sha256": value["prompt_sha256"],
            "record_sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
        }
        for value in normalized
    ]
    payload: Dict[str, Any] = {
        "schema_version": PHASE3_TRAJECTORY_MANIFEST_SCHEMA_VERSION,
        "artifact_role": "validation1531_no_loop_trajectory",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "forward_type": card["trajectory"]["forward_type"],
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "record_count": len(records),
        "subject_count": len({value["subject"] for value in records}),
        "canonical_record_order": list(card["identity"]["canonical_record_order"]),
        "records": records,
        "ordered_identity_sha256": ordered_identity_sha256(
            [value["identity"] for value in records]
        ),
        "ordered_trajectory_record_sha256": hashlib.sha256(
            canonical_json_bytes([value["record_sha256"] for value in records])
        ).hexdigest(),
    }
    attach_manifest_sha256(payload)
    validate_trajectory_manifest(payload, source_manifest, pool_manifest, card)
    return payload


def validate_trajectory_records(
    trajectory_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    source_records = validate_source_manifest(source_manifest, card)
    pool_records = validate_pool_manifest(pool_manifest, source_manifest, card)
    if not isinstance(trajectory_records, list) or len(trajectory_records) != len(source_records):
        raise SchemaError("trajectory population must contain exactly the signed source set")
    normalized = []
    for index, record in enumerate(trajectory_records):
        item = validate_trajectory_record(record, card, expected_source=source_records[index])
        renderer = item["renderer_provenance"]
        if renderer["source_manifest_sha256"] != source_manifest["manifest_sha256"]:
            raise SchemaError("trajectory record source-manifest binding mismatch")
        if renderer["pool_manifest_sha256"] != pool_manifest["manifest_sha256"]:
            raise SchemaError("trajectory record pool-manifest binding mismatch")
        if any(item[key] != pool_records[index][key] for key in ("identity", "subject", "split", "prompt_sha256")):
            raise SchemaError("trajectory record differs from canonical pool metadata")
        normalized.append(item)
    _validate_population_order_and_uniqueness(
        normalized, int(card["task"]["trajectory_subject_count"])
    )
    return normalized


def validate_trajectory_manifest(
    value: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    pool_records = validate_pool_manifest(pool_manifest, source_manifest, card)
    _exact_keys(value, TRAJECTORY_MANIFEST_KEYS, "Phase 3 trajectory manifest")
    verify_manifest_sha256(value)
    if value["schema_version"] != PHASE3_TRAJECTORY_MANIFEST_SCHEMA_VERSION:
        raise SchemaError("unsupported trajectory manifest schema")
    exact = {
        "artifact_role": "validation1531_no_loop_trajectory",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "forward_type": card["trajectory"]["forward_type"],
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "record_count": card["task"]["trajectory_sample_count"],
        "subject_count": card["task"]["trajectory_subject_count"],
        "canonical_record_order": card["identity"]["canonical_record_order"],
    }
    for key, expected in exact.items():
        if value[key] != expected:
            raise SchemaError("trajectory manifest %s differs from the frozen contract" % key)
    records = value["records"]
    if not isinstance(records, list) or len(records) != len(pool_records):
        raise SchemaError("trajectory manifest record count differs from pool")
    normalized = []
    for index, record in enumerate(records):
        _exact_keys(record, TRAJECTORY_MANIFEST_RECORD_KEYS, "trajectory manifest record")
        item = {
            "identity": canonical_identity(record["identity"]),
            "subject": _nonempty_string(record["subject"], "trajectory manifest subject"),
            "split": str(record["split"]),
            "prompt_sha256": _sha256(record["prompt_sha256"], "trajectory manifest prompt"),
            "record_sha256": _sha256(record["record_sha256"], "trajectory record digest"),
        }
        for key in ("identity", "subject", "split", "prompt_sha256"):
            if item[key] != pool_records[index][key]:
                raise SchemaError("trajectory manifest differs from pool at %s" % key)
        normalized.append(item)
    if value["ordered_identity_sha256"] != ordered_identity_sha256(
        [record["identity"] for record in normalized]
    ):
        raise SchemaError("trajectory ordered identity SHA256 mismatch")
    digest = hashlib.sha256(
        canonical_json_bytes([record["record_sha256"] for record in normalized])
    ).hexdigest()
    if value["ordered_trajectory_record_sha256"] != digest:
        raise SchemaError("trajectory ordered record SHA256 mismatch")
    return normalized


def verify_population_disjointness(
    validation_records: Sequence[Mapping[str, Any]],
    test_metadata_records: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    enforce_frozen_counts: bool = True,
) -> Dict[str, Any]:
    """Verify both canonical identity and gold-free content disjointness."""

    validate_phase3_card(card)
    validation = [_normalize_disjointness_record(value, "validation") for value in validation_records]
    test = [_normalize_disjointness_record(value, "test") for value in test_metadata_records]
    if enforce_frozen_counts:
        if len(validation) != int(card["task"]["trajectory_sample_count"]):
            raise SchemaError("validation disjointness input must contain 1,531 records")
        if len(test) != int(card["task"]["outcome_sample_count"]):
            raise SchemaError("test disjointness input must contain 14,042 records")
    validation_identity = {identity_tuple(value["identity"]) for value in validation}
    test_identity = {identity_tuple(value["identity"]) for value in test}
    if len(validation_identity) != len(validation) or len(test_identity) != len(test):
        raise SchemaError("disjointness metadata contains duplicate identities")
    identity_overlap = validation_identity & test_identity
    validation_content = {value["sanitized_content_sha256"] for value in validation}
    test_content = {value["sanitized_content_sha256"] for value in test}
    content_overlap = validation_content & test_content
    if identity_overlap:
        raise SchemaError("validation/test canonical identity intersection is non-zero")
    if content_overlap:
        raise SchemaError("validation/test sanitized-content intersection is non-zero")
    return {
        "validation_count": len(validation),
        "test_count": len(test),
        "identity_intersection_count": 0,
        "sanitized_content_intersection_count": 0,
        "validation_identity_sha256": hashlib.sha256(
            canonical_json_bytes([value["identity"] for value in validation])
        ).hexdigest(),
        "test_identity_sha256": hashlib.sha256(
            canonical_json_bytes([value["identity"] for value in test])
        ).hexdigest(),
    }


def partition_starts(card: Mapping[str, Any]) -> Dict[str, Tuple[int, ...]]:
    validate_historical_blind_partition(card)
    historical = tuple(int(value.split(":", 1)[0]) for value in card["partition"]["historical_13"])
    blind = tuple(int(value.split(":", 1)[0]) for value in card["partition"]["blind_12"])
    return {"historical_13": historical, "blind_12": blind}


def _normalize_disjointness_record(value: Mapping[str, Any], expected_split: str) -> Dict[str, Any]:
    _exact_keys(
        value,
        ("identity", "split", "sanitized_content_sha256"),
        "%s disjointness record" % expected_split,
    )
    if value["split"] != expected_split:
        raise SchemaError("disjointness record uses the wrong split")
    return {
        "identity": canonical_identity(value["identity"]),
        "split": expected_split,
        "sanitized_content_sha256": _sha256(
            value["sanitized_content_sha256"], "disjointness content hash"
        ),
    }


def _validate_population_order_and_uniqueness(
    records: Sequence[Mapping[str, Any]], expected_subjects: int
) -> None:
    keys = [canonical_record_key(record) for record in records]
    if keys != sorted(keys):
        raise SchemaError("population records are not in frozen canonical lexicographic order")
    identities = [identity_tuple(record["identity"]) for record in records]
    if len(set(identities)) != len(identities):
        raise SchemaError("population contains duplicate canonical identities")
    subjects = {record["subject"] for record in records}
    if len(subjects) != expected_subjects:
        raise SchemaError("population subject count differs from the frozen 57 subjects")


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


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError("%s must be an integer" % context)
    return int(value)


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("%s must be a non-empty string" % context)
    return value.strip()


def _sha256(value: Any, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SchemaError("%s must be a lowercase SHA256" % context)
    return text


__all__ = [
    "ALLOWED_INPUT_ARTIFACT_KINDS",
    "make_pool_manifest",
    "make_source_contract",
    "make_source_manifest",
    "make_trajectory_manifest",
    "partition_starts",
    "validate_input_artifact_kind",
    "validate_pool_manifest",
    "validate_pool_records",
    "validate_source_contract",
    "validate_source_manifest",
    "validate_trajectory_manifest",
    "validate_trajectory_records",
    "verify_population_disjointness",
]
