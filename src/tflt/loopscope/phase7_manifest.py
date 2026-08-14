"""Safe Phase 7 validation membership and shard helpers.

Only the renderer's safe validation projection is used to build the manifest.
Question text, choices, demonstrations, and validation targets remain
transient inputs to the renderer and never cross this module's persistence
boundary.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from tflt.loopscope.phase7_schema import Phase7ContractError, scan_forbidden_fields


MANIFEST_KEYS = frozenset(
    {"canonical_identity", "subject", "task_name", "validation_index"}
)


def _validate_row(raw: Mapping[str, Any], *, seen: set[str]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise Phase7ContractError("validation manifest row must be an object")
    scan_forbidden_fields(raw)
    if set(raw) != set(MANIFEST_KEYS):
        raise Phase7ContractError("validation manifest row keys differ from the safe contract")
    subject = str(raw["subject"]).strip()
    task_name = str(raw["task_name"]).strip()
    identity = str(raw["canonical_identity"]).strip()
    index = raw["validation_index"]
    if not subject or task_name != "mmlu_" + subject:
        raise Phase7ContractError("validation manifest subject/task closure failed")
    if not identity or identity in seen:
        raise Phase7ContractError("validation manifest has a missing or duplicate identity")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise Phase7ContractError("validation manifest index is invalid")
    expected_identity = "mmlu_%s:validation:%d" % (subject, index)
    if identity != expected_identity:
        raise Phase7ContractError("validation manifest identity does not close to subject/index")
    seen.add(identity)
    return {
        "canonical_identity": identity,
        "subject": subject,
        "task_name": task_name,
        "validation_index": index,
    }


def validate_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_count: int | None = None,
    expected_subjects: int | None = None,
) -> List[Dict[str, Any]]:
    """Validate and normalize a safe, ordered validation manifest."""

    if isinstance(rows, (str, bytes)):
        raise Phase7ContractError("validation manifest must be a sequence of rows")
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        normalized.append(_validate_row(raw, seen=seen))
    if not normalized:
        raise Phase7ContractError("validation manifest is empty")
    if expected_count is not None and len(normalized) != expected_count:
        raise Phase7ContractError("validation manifest count differs from expected")
    subjects = {row["subject"] for row in normalized}
    if expected_subjects is not None and len(subjects) != expected_subjects:
        raise Phase7ContractError("validation manifest subject count differs from expected")
    return normalized


def build_canonical_validation_manifest(*, cache_dir: str | None = None) -> List[Dict[str, Any]]:
    """Build the common natural task/subject/index validation membership."""

    from tflt.loopscope.phase7_renderer import (
        SAFE_VALIDATION_COLUMNS,
        _safe_dataset_pair,
        _task_map,
        canonical_identity,
    )

    rows: List[Dict[str, Any]] = []
    for task_name, task in _task_map().items():
        subject = task_name[len("mmlu_") :]
        validation, _ = _safe_dataset_pair(task, cache_dir=cache_dir)
        for validation_index, raw in enumerate(validation):
            if set(raw) != set(SAFE_VALIDATION_COLUMNS):
                raise Phase7ContractError("validation row is outside the safe projection")
            if str(raw["subject"]).strip() != subject:
                raise Phase7ContractError("validation subject differs from task name")
            choices = raw["choices"]
            if not isinstance(raw["question"], str) or not isinstance(choices, Sequence) or len(choices) != 4:
                raise Phase7ContractError("validation safe row shape differs")
            rows.append(
                {
                    "canonical_identity": canonical_identity(subject, validation_index),
                    "subject": subject,
                    "task_name": task_name,
                    "validation_index": validation_index,
                }
            )
    return validate_manifest_rows(rows, expected_count=1531, expected_subjects=57)


def split_manifest_rows(rows: Sequence[Mapping[str, Any]], shard_count: int) -> List[List[Dict[str, Any]]]:
    """Split an ordered manifest into contiguous, balanced, non-empty shards."""

    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise Phase7ContractError("shard_count must be a positive integer")
    if shard_count > len(rows):
        raise Phase7ContractError("shard_count exceeds manifest size")
    normalized = validate_manifest_rows(rows)
    shards: List[List[Dict[str, Any]]] = []
    for shard_index in range(shard_count):
        start = (len(normalized) * shard_index) // shard_count
        stop = (len(normalized) * (shard_index + 1)) // shard_count
        shard = [dict(row) for row in normalized[start:stop]]
        if not shard:
            raise Phase7ContractError("manifest shard is empty")
        shards.append(shard)
    if sum(len(shard) for shard in shards) != len(normalized):
        raise Phase7ContractError("manifest shard partition is incomplete")
    return shards


__all__ = [
    "MANIFEST_KEYS",
    "build_canonical_validation_manifest",
    "split_manifest_rows",
    "validate_manifest_rows",
]
