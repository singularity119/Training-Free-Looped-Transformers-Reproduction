"""Write-once artifact helpers for LoopScope Phase 4 Gate P4-B.

This module deliberately contains no dataset, renderer, model, scheduler, or
selector execution code.  Heavy-runtime producers can use these helpers to
freeze membership, close completed scalar-only shards, and package the
prospective outcome panel without widening the P4-B information boundary.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tflt.loopscope.phase4_schema import (
    Phase4ContractError,
    canonical_json_bytes,
    file_sha256,
    scan_forbidden_fields,
    semantic_sha256,
    validate_shared_identity_contract,
    validate_trajectory_record,
)
from tflt.loopscope.phase4_selector import verify_selector_report


GATE = "P4-B"
EXECUTOR_THREAD_ID = "019f6ca9-ef16-7a53-8949-e47d89fd2dc2"
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/"
    "training_free_looped_transformers_loopscope/runs/"
    "phase4-p4b-20260716T204154Z"
)
MEMBERSHIP_KINDS = ("original", "option", "template")
SHARD_MANIFEST_SCHEMA = "loopscope.phase4.p4b-shard-manifest.v1"
SHARD_RECEIPT_SCHEMA = "loopscope.phase4.p4b-shard-receipt.v1"
MERGED_MANIFEST_SCHEMA = "loopscope.phase4.p4b-merged-trajectory-manifest.v1"
OUTCOME_PANEL_MANIFEST_SCHEMA = "loopscope.phase4.outcome-panel-manifest.v1"


class P4BAcquisitionError(ValueError):
    """Raised when a P4-B artifact violates its write-once contract."""


def assert_authorized_run_root(
    path: Path,
    *,
    must_exist: Optional[bool] = None,
    expected_root: Path = AUTHORIZED_RUN_ROOT,
) -> Path:
    """Require the exact bound run root, with an injectable root for unit tests."""

    resolved = Path(path).expanduser().resolve()
    expected = Path(expected_root).expanduser().resolve()
    if resolved != expected:
        raise P4BAcquisitionError("run root differs from the exact P4-B authorization")
    if must_exist is True and not resolved.is_dir():
        raise P4BAcquisitionError("authorized P4-B run root does not exist")
    if must_exist is False and resolved.exists():
        raise FileExistsError("authorized P4-B run root already exists")
    return resolved


def resolve_run_child(run_root: Path, relative: str | Path) -> Path:
    root = Path(run_root).expanduser().resolve()
    child_relative = Path(relative)
    if child_relative.is_absolute() or not child_relative.parts or ".." in child_relative.parts:
        raise P4BAcquisitionError("artifact path must be a non-empty run-relative path")
    child = (root / child_relative).resolve()
    if root not in child.parents:
        raise P4BAcquisitionError("artifact path escapes the authorized run root")
    return child


def load_strict_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_bad_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise P4BAcquisitionError("cannot load strict JSON artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise P4BAcquisitionError("JSON artifact must contain an object: %s" % path)
    return value


def load_strict_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise P4BAcquisitionError(
                        "blank JSONL line is forbidden at %s:%d" % (path, line_number)
                    )
                value = json.loads(line, parse_constant=_bad_constant)
                if not isinstance(value, dict):
                    raise P4BAcquisitionError(
                        "JSONL record must be an object at %s:%d" % (path, line_number)
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, P4BAcquisitionError):
            raise
        raise P4BAcquisitionError("cannot load strict JSONL artifact: %s" % path) from exc
    if not records:
        raise P4BAcquisitionError("JSONL artifact contains no records: %s" % path)
    return records


def write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def write_new_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        raise P4BAcquisitionError("refusing to write an empty JSONL artifact")
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def build_shard_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    shard_count: int,
    run_root: Path,
    bindings: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a deterministic canonical-ordinal round-robin membership manifest."""

    if kind not in MEMBERSHIP_KINDS:
        raise P4BAcquisitionError("membership kind must be original, option, or template")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise P4BAcquisitionError("shard_count must be an integer")
    if not records or shard_count < 1 or shard_count > len(records):
        raise P4BAcquisitionError("shard_count is outside the membership range")
    scan_forbidden_fields(bindings)
    members = [_membership_record(record, ordinal) for ordinal, record in enumerate(records)]
    identities = [member["canonical_identity"] for member in members]
    if len(set(identities)) != len(identities):
        raise P4BAcquisitionError("membership identities must be unique")
    shards = []
    for shard_id in range(shard_count):
        shard_members = members[shard_id::shard_count]
        shards.append(
            {
                "shard_id": shard_id,
                "record_count": len(shard_members),
                "records": shard_members,
                "membership_sha256": semantic_sha256(shard_members),
                "output_relative": (
                    "formal/original/shards/shard-%04d" % shard_id
                    if kind == "original"
                    else "controls/%s/shards/shard-%04d" % (kind, shard_id)
                ),
            }
        )
    payload: Dict[str, Any] = {
        "schema_version": SHARD_MANIFEST_SCHEMA,
        "artifact_role": "phase4_p4b_%s_shard_manifest" % kind,
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(Path(run_root).expanduser().resolve()),
        "kind": kind,
        "distribution": "canonical_ordinal_round_robin",
        "record_count": len(members),
        "shard_count": shard_count,
        "ordered_identity_sha256": semantic_sha256(identities),
        "bindings": dict(bindings),
        "shards": shards,
    }
    payload["manifest_sha256"] = semantic_sha256(payload)
    return payload


def validate_shard_manifest(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_run_root: Optional[Path] = None,
) -> None:
    if manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA:
        raise P4BAcquisitionError("unexpected P4-B shard manifest schema")
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise P4BAcquisitionError("shard manifest SHA256 mismatch")
    run_root = Path(str(manifest.get("run_root") or ""))
    if expected_run_root is not None:
        assert_authorized_run_root(
            run_root, must_exist=None, expected_root=expected_run_root
        )
    expected = build_shard_manifest(
        records,
        kind=str(manifest.get("kind") or ""),
        shard_count=manifest.get("shard_count"),
        run_root=run_root,
        bindings=manifest.get("bindings") if isinstance(manifest.get("bindings"), Mapping) else {},
    )
    if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
        raise P4BAcquisitionError("shard manifest differs from deterministic recomputation")


def merge_completed_shards(
    *,
    run_root: Path,
    shard_manifest: Mapping[str, Any],
    expected_records: Sequence[Mapping[str, Any]],
    merged_jsonl_relative: str | Path,
    merged_manifest_relative: str | Path,
    source_records: Optional[Sequence[Mapping[str, Any]]] = None,
    require_full_population: bool = False,
    d36_tolerance: float = 1e-6,
    expected_run_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Close complete shard records and restore exact expected canonical order."""

    root = Path(run_root).expanduser().resolve()
    if expected_run_root is not None:
        assert_authorized_run_root(root, must_exist=True, expected_root=expected_run_root)
    validate_shard_manifest(
        shard_manifest,
        expected_records,
        expected_run_root=expected_run_root,
    )
    output_path = resolve_run_child(root, merged_jsonl_relative)
    manifest_path = resolve_run_child(root, merged_manifest_relative)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite a completed merged artifact")

    expected_identities = [_identity(record) for record in expected_records]
    by_identity: Dict[str, Dict[str, Any]] = {}
    shard_evidence = []
    for shard in shard_manifest["shards"]:
        output_dir = resolve_run_child(root, shard["output_relative"])
        record_path = output_dir / "records.jsonl"
        receipt_path = output_dir / "receipt.json"
        if not record_path.is_file() or not receipt_path.is_file():
            raise P4BAcquisitionError("completed shard artifact is missing")
        receipt = load_strict_json(receipt_path)
        _validate_shard_receipt(receipt, shard, record_path)
        shard_records = load_strict_jsonl(record_path)
        if len(shard_records) != shard["record_count"]:
            raise P4BAcquisitionError("shard record count differs from membership")
        for record, member in zip(shard_records, shard["records"]):
            validate_trajectory_record(record, d36_tolerance=d36_tolerance)
            identity = record.get("canonical_identity")
            if identity != member["canonical_identity"]:
                raise P4BAcquisitionError("shard record order or identity differs")
            if identity in by_identity:
                raise P4BAcquisitionError("duplicate trajectory identity across shards")
            by_identity[str(identity)] = dict(record)
        shard_evidence.append(
            {
                "shard_id": shard["shard_id"],
                "record_count": len(shard_records),
                "record_file_sha256": file_sha256(record_path),
                "receipt_file_sha256": file_sha256(receipt_path),
            }
        )

    observed = set(by_identity)
    expected = set(expected_identities)
    if observed != expected or len(by_identity) != len(expected_identities):
        raise P4BAcquisitionError(
            "merged membership mismatch: missing=%d extra=%d"
            % (len(expected - observed), len(observed - expected))
        )
    ordered = [by_identity[identity] for identity in expected_identities]
    if source_records is not None:
        observed_source_closure = [
            _source_closure_from_trajectory(record) for record in ordered
        ]
        validate_shared_identity_contract(
            source_records,
            observed_source_closure,
            require_full_population=require_full_population,
        )

    record_file_sha256 = write_new_jsonl(output_path, ordered)
    merged_manifest: Dict[str, Any] = {
        "schema_version": MERGED_MANIFEST_SCHEMA,
        "artifact_role": "phase4_p4b_%s_merged_trajectory_manifest"
        % shard_manifest["kind"],
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(root),
        "kind": shard_manifest["kind"],
        "record_count": len(ordered),
        "shard_count": shard_manifest["shard_count"],
        "shard_manifest_sha256": shard_manifest["manifest_sha256"],
        "ordered_identity_sha256": semantic_sha256(expected_identities),
        "ordered_record_sha256": semantic_sha256(ordered),
        "record_file": str(output_path.relative_to(root)),
        "record_file_sha256": record_file_sha256,
        "source_identity_closure_verified": source_records is not None,
        "shards": shard_evidence,
    }
    merged_manifest["manifest_sha256"] = semantic_sha256(merged_manifest)
    write_new_json(manifest_path, merged_manifest)
    return merged_manifest


def require_non_degenerate_width4_scores(
    selector_report: Mapping[str, Any], *, tie_tolerance: float = 1e-12
) -> None:
    verify_selector_report(selector_report)
    scores = [float(row["score"]) for row in selector_report["width4_primary"]["rows"]]
    if not scores or not all(math.isfinite(value) for value in scores):
        raise P4BAcquisitionError("width-4 selector scores must be finite")
    if max(scores) - min(scores) <= tie_tolerance:
        raise P4BAcquisitionError("width-4 selector scores are all tied/degenerate")


def extract_known_outcome_windows(card: Mapping[str, Any]) -> List[str]:
    try:
        registry = card["provenance_closure"]["values"]["known_outcome_registry"]
    except (KeyError, TypeError) as exc:
        raise P4BAcquisitionError("card known-outcome registry is missing") from exc
    if not isinstance(registry, list) or not registry:
        raise P4BAcquisitionError("card known-outcome registry is empty")
    windows = []
    for entry in registry:
        if not isinstance(entry, Mapping):
            raise P4BAcquisitionError("known-outcome registry entry must be an object")
        window = entry.get("window")
        if not isinstance(window, str) or not window:
            raise P4BAcquisitionError("known-outcome registry window is missing")
        if entry.get("outcome_value_recorded") is not False:
            raise P4BAcquisitionError("known-outcome registry must not record values")
        windows.append(window)
    if len(set(windows)) != len(windows) or windows.count("15:18") != 1:
        raise P4BAcquisitionError("known-outcome registry must contain unique 15:18")
    return sorted(windows)


def wrap_outcome_panel(
    panel: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
    selector_freeze_sha256: str,
    bindings: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Package the prospective panel with immutable fresh-acquisition flags."""

    if panel.get("schema_version") != "loopscope.phase4.outcome-panel.v1":
        raise P4BAcquisitionError("unexpected outcome panel schema")
    if panel.get("baseline") != "no-loop" or panel.get("fixed_comparator") != "15:18":
        raise P4BAcquisitionError("outcome panel baseline/fixed comparator differs")
    known = extract_known_outcome_windows(card)
    if sorted(panel.get("known_outcome_registry", [])) != known:
        raise P4BAcquisitionError("outcome panel known registry differs from card")
    if panel.get("variable_width_outcomes_authorized") is not False:
        raise P4BAcquisitionError("variable-width outcomes must remain unauthorized")
    if not _is_sha256(selector_freeze_sha256):
        raise P4BAcquisitionError("selector freeze SHA256 is invalid")
    payload: Dict[str, Any] = {
        "schema_version": OUTCOME_PANEL_MANIFEST_SCHEMA,
        "artifact_role": "phase4_p4c_prospective_outcome_panel_manifest",
        "gate_frozen_by": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "selector_freeze_sha256": selector_freeze_sha256,
        "fresh_baseline": True,
        "fresh_15_18": True,
        "panel": dict(panel),
        "bindings": dict(bindings or {}),
    }
    payload["manifest_sha256"] = semantic_sha256(payload)
    return payload


def _membership_record(record: Mapping[str, Any], ordinal: int) -> Dict[str, Any]:
    scan_forbidden_fields(record)
    identity = _identity(record)
    return {
        "ordinal": ordinal,
        "canonical_identity": identity,
        "source_record_sha256": semantic_sha256(record),
    }


def _identity(record: Mapping[str, Any]) -> str:
    identity = record.get("canonical_identity")
    if not isinstance(identity, str) or not identity:
        raise P4BAcquisitionError("membership canonical_identity is missing")
    return identity


def _validate_shard_receipt(
    receipt: Mapping[str, Any], shard: Mapping[str, Any], record_path: Path
) -> None:
    required = {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "status": "COMPLETED",
        "shard_id": shard["shard_id"],
        "membership_sha256": shard["membership_sha256"],
        "record_count": shard["record_count"],
        "record_file_sha256": file_sha256(record_path),
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise P4BAcquisitionError("completed shard receipt %s mismatch" % key)
    if receipt.get("ordered_record_sha256") != semantic_sha256(load_strict_jsonl(record_path)):
        raise P4BAcquisitionError("completed shard ordered-record SHA256 mismatch")
    if "manifest_sha256" in receipt and receipt["manifest_sha256"] != _manifest_sha256(receipt):
        raise P4BAcquisitionError("completed shard receipt manifest SHA256 mismatch")


def _source_closure_from_trajectory(record: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = record.get("producer_provenance")
    closure = provenance.get("source_closure") if isinstance(provenance, Mapping) else None
    if not isinstance(closure, Mapping):
        raise P4BAcquisitionError("formal trajectory lacks source_closure provenance")
    keys = (
        "canonical_identity",
        "sanitized_content_sha256",
        "rendered_prefix_sha256",
        "rendered_token_ids_sha256",
        "attention_mask_sha256",
    )
    observed = {key: closure.get(key) for key in keys}
    observed["canonical_identity"] = record["canonical_identity"]
    return observed


def _manifest_sha256(payload: Mapping[str, Any]) -> str:
    return semantic_sha256({key: value for key, value in payload.items() if key != "manifest_sha256"})


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _bad_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant %s" % value)


__all__ = [
    "AUTHORIZED_RUN_ROOT",
    "MERGED_MANIFEST_SCHEMA",
    "OUTCOME_PANEL_MANIFEST_SCHEMA",
    "P4BAcquisitionError",
    "SHARD_MANIFEST_SCHEMA",
    "SHARD_RECEIPT_SCHEMA",
    "assert_authorized_run_root",
    "build_shard_manifest",
    "extract_known_outcome_windows",
    "load_strict_json",
    "load_strict_jsonl",
    "merge_completed_shards",
    "require_non_degenerate_width4_scores",
    "resolve_run_child",
    "validate_shard_manifest",
    "wrap_outcome_panel",
    "write_new_json",
    "write_new_jsonl",
]
