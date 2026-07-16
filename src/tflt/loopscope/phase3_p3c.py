"""Staged, fail-closed execution for LoopScope Phase 3 Gate P3-C.

The module deliberately separates the three one-way stages:

* C0 projects gold-free MMLU test identity/content metadata and hashes, but
  never parses outcome files.
* C1 consumes only P3-B no-loop trajectories and freezes every baseline-only
  score/rank before any historical outcome is visible.
* C2 first independently revalidates the C1 freeze, then opens only the
  manifest-bound baseline and historical-13 outcome files.

No function in this module launches a model, a loop experiment, or a GPU job.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.analysis import (
    AnalysisError,
    _validate_manifest_eval_job,
    extract_accuracy,
    extract_correctness_samples,
    paired_comparison,
)
from tflt.loopscope.phase3_analysis import analyze_manifested_phase3_trajectories
from tflt.loopscope.phase3_pool import (
    partition_starts,
    validate_pool_manifest,
    validate_source_manifest,
    validate_trajectory_manifest,
    verify_population_disjointness,
)
from tflt.loopscope.phase3_schema import (
    PHASE3_CARD_BYTE_SHA256,
    attach_manifest_sha256,
    canonical_identity,
    canonical_json_bytes,
    canonical_record_key,
    file_sha256,
    identity_tuple,
    load_phase3_card,
    ordered_identity_sha256,
    sanitized_content_sha256,
    verify_manifest_sha256,
)
from tflt.loopscope.schema import SchemaError, manifest_sha256


GATE = "P3-C"
EXECUTOR_THREAD_ID = "019f69ae-1ce3-7303-9221-c7779d57c8c5"
PLANNING_THREAD_ID = "019f670d-24ca-7ad0-b92e-64438aa04ef1"
AUTHORIZED_BASE_COMMIT = "ef783c313298145d809bf6adafc2a0108c5639f6"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"

REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3c-20260716T064601Z"
)
P3B_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3b-20260715T223304Z"
)
PHASE1_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/"
    "runs/loopscope-qwen17-mmlu-phase1-20260711-053022"
)
BLIND_SEARCH_ROOT = Path("/hpc2hdd/home/xhuang225/workspaces")

EXPECTED_PHASE1_MANIFEST_INTERNAL_SHA256 = (
    "e497983d6b47ba2ff8e296a5375838737ad666b5f7dc95b1f366a8bd21111750"
)
EXPECTED_TEST_ORDERED_IDENTITY_SHA256 = (
    "872c9f6bd7f40e3c6fa40b13cc862033379ac0a9cbc439a1593653945f3d521d"
)
EXPECTED_P3B_TRAJECTORY_FILE_SHA256 = (
    "fb7ae66a1a2696f46727c7ca5069ade2e2f9d6e181db03ed1d005cb2eed1525f"
)
EXPECTED_P3B_TRAJECTORY_MANIFEST_FILE_SHA256 = (
    "75a568c6f6595f74d34d1e86efc0f716934e3a2a1cf972daab7874e4afe89cf0"
)
EXPECTED_P3B_TRAJECTORY_MANIFEST_INTERNAL_SHA256 = (
    "f3928b964198f6252c567cdc3c5b452d2397b5135838aaf2e0741d69524beec1"
)
EXPECTED_P3B_VERIFIER_FILE_SHA256 = (
    "cd5fb473930a045e5875c5bbd1e2565282f1763da61951226c64775b4c40def0"
)
EXPECTED_P3B_VERIFIER_INTERNAL_SHA256 = (
    "3cb1f7529e1f3e362d6a6eb0df6ec4cf08ccd118bceb98dd8d3b1e5343cc9cd2"
)

HISTORICAL_WINDOWS = (
    "4:7",
    "6:9",
    "8:11",
    "9:12",
    "11:14",
    "12:15",
    "13:16",
    "14:17",
    "15:18",
    "17:20",
    "18:21",
    "20:23",
    "22:25",
)
BLIND_WINDOWS = (
    "0:3",
    "1:4",
    "2:5",
    "3:6",
    "5:8",
    "7:10",
    "10:13",
    "16:19",
    "19:22",
    "21:24",
    "23:26",
    "24:27",
)
VARIANT_PRIORITY = ("CONSENSUS", "FLANK", "SHIFT")

TEST_METADATA_NAME = "test14042_identity_content_metadata.jsonl"
TEST_MANIFEST_NAME = "test14042_identity_content_manifest.json"
DISJOINTNESS_NAME = "validation1531_vs_test14042_disjointness_receipt.json"
BLIND_EXISTENCE_NAME = "phase3_blind12_outcome_existence_receipt.json"
HISTORICAL_ALLOWLIST_NAME = "historical13_source_allowlist_receipt.json"
C0_RECEIPT_NAME = "c0_metadata_closure_receipt.json"

ATLAS_NAME = "baseline_width4_atlas.json"
SHIFT_NAME = "shift_overlap_report.json"
FLANK_NAME = "nonoverlap_flank_report.json"
CONSENSUS_NAME = "consensus_report.json"
SELECTOR_FREEZE_NAME = "selector_freeze.json"
SELECTOR_RECEIPT_NAME = "selector_freeze_receipt.json"

RETROSPECTIVE_NAME = "retrospective_adjudication.json"
MISSING12_PREDICTIONS_NAME = "missing12_frozen_predictions.json"
RETROSPECTIVE_VERIFIER_NAME = "retrospective_adjudication_verifier_receipt.json"

P3B_SOURCE_NAME = "source_manifest.json"
P3B_POOL_NAME = "validation1531_pool_manifest.json"
P3B_TRAJECTORY_NAME = "validation1531_no_loop_trajectories.jsonl"
P3B_TRAJECTORY_MANIFEST_NAME = "validation1531_no_loop_trajectory_manifest.json"
P3B_VERIFIER_NAME = "validation1531_no_loop_probe_verifier_receipt.json"

IMPLEMENTATION_RELATIVE_PATHS = (
    "src/tflt/loopscope/phase3_p3c.py",
    "scripts/loopscope/run_qwen17_phase3_p3c.py",
)

TEST_METADATA_SCHEMA = "loopscope.phase3.p3c-test-metadata-record.v1"
TEST_MANIFEST_SCHEMA = "loopscope.phase3.p3c-test-metadata-manifest.v1"
C0_RECEIPT_SCHEMA = "loopscope.phase3.p3c-c0-receipt.v1"
BLIND_EXISTENCE_SCHEMA = "loopscope.phase3.p3c-blind-existence-receipt.v1"
HISTORICAL_ALLOWLIST_SCHEMA = "loopscope.phase3.p3c-historical-allowlist.v1"
C1_ARTIFACT_SCHEMA = "loopscope.phase3.p3c-c1-artifact.v1"
SELECTOR_FREEZE_SCHEMA = "loopscope.phase3.p3c-selector-freeze.v1"
SELECTOR_RECEIPT_SCHEMA = "loopscope.phase3.p3c-selector-freeze-receipt.v1"
RETROSPECTIVE_SCHEMA = "loopscope.phase3.p3c-retrospective-adjudication.v1"
MISSING12_SCHEMA = "loopscope.phase3.p3c-missing12-predictions.v1"
RETROSPECTIVE_VERIFIER_SCHEMA = "loopscope.phase3.p3c-retrospective-verifier.v1"


class P3CError(ValueError):
    """Fail-closed P3-C contract or provenance violation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    result = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise P3CError("P3-C implementation path is missing: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def git_provenance(expected_commit: str, *, require_remote_root: bool) -> Dict[str, Any]:
    root = repository_root().resolve()
    if require_remote_root and root != REMOTE_REPO:
        raise P3CError("P3-C remote execution is outside the dedicated clone")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "rev-parse", "refs/remotes/origin/loopscope")
    dirty = bool(_git(root, "status", "--porcelain"))
    if branch != "loopscope":
        raise P3CError("P3-C requires branch loopscope")
    if head != str(expected_commit) or origin != str(expected_commit):
        raise P3CError("P3-C HEAD/origin does not match the expected implementation commit")
    if dirty:
        raise P3CError("P3-C requires a clean working tree")
    return {
        "root": str(root),
        "branch": branch,
        "commit": head,
        "origin_loopscope": origin,
        "dirty": False,
    }


def assert_authorized_paths(
    card_path: Path,
    p3b_root: Path,
    p3c_root: Path,
    phase1_root: Optional[Path] = None,
) -> None:
    if Path(card_path).resolve() != (REMOTE_REPO / "configs/loopscope/phase3_card.json"):
        raise P3CError("card path differs from the exact remote authorization")
    if Path(p3b_root).resolve() != P3B_ROOT:
        raise P3CError("P3-B root differs from the exact P3-C authorization")
    if Path(p3c_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise P3CError("P3-C root differs from the exact write-once authorization")
    if phase1_root is not None and Path(phase1_root).resolve() != PHASE1_ROOT:
        raise P3CError("Phase 1 root differs from the exact historical authorization")


def load_strict_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise P3CError("cannot load strict JSON artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise P3CError("JSON artifact must contain an object: %s" % path)
    return value


def load_strict_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise P3CError("blank JSONL line at %s:%d" % (path, line_number))
                value = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(value, dict):
                    raise P3CError("JSONL record must be an object at %s:%d" % (path, line_number))
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, P3CError):
            raise
        raise P3CError("cannot load strict JSONL artifact: %s" % path) from exc
    if not records:
        raise P3CError("JSONL artifact contains no records: %s" % path)
    return records


def write_new_json(path: Path, payload: Any) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
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
        raise P3CError("refusing to write an empty JSONL artifact")
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def safe_test_dataset_fields(row: Mapping[str, Any], expected_subject: str) -> Dict[str, Any]:
    """Access only question, choices, and subject from an already projected row."""

    try:
        question = row["question"]
        choices = row["choices"]
        subject = row["subject"]
    except (KeyError, TypeError) as exc:
        raise P3CError("MMLU safe test projection fields are missing") from exc
    if not isinstance(question, str):
        raise P3CError("MMLU question must be a string")
    if not isinstance(choices, (list, tuple)) or len(choices) != 4:
        raise P3CError("MMLU choices must contain exactly four strings")
    if any(not isinstance(value, str) for value in choices):
        raise P3CError("MMLU choices must all be strings")
    if str(subject) != str(expected_subject):
        raise P3CError("MMLU row subject differs from the frozen task subject")
    return {"question": question, "choices": list(choices), "subject": str(subject)}


def safe_test_doc_sha256(safe_row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "question": safe_row["question"],
                "subject": safe_row["subject"],
                "choices": list(safe_row["choices"]),
            }
        )
    ).hexdigest()


def make_test_metadata_record(
    task_name: str,
    subject: str,
    row_index: int,
    safe_row: Mapping[str, Any],
) -> Dict[str, Any]:
    doc_id = str(row_index)
    return {
        "schema_version": TEST_METADATA_SCHEMA,
        "identity": {
            "task": task_name,
            "doc_id": doc_id,
            "doc_hash": safe_test_doc_sha256(safe_row),
        },
        "subject": subject,
        "split": "test",
        "sanitized_content_sha256": sanitized_content_sha256(
            safe_row["question"], safe_row["choices"]
        ),
    }


def validate_test_metadata_records(
    records: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    enforce_frozen_counts: bool = True,
    expected_ordered_identity_sha256: Optional[str] = None,
) -> List[Dict[str, Any]]:
    normalized = []
    for raw in records:
        _exact_keys(
            raw,
            ("schema_version", "identity", "subject", "split", "sanitized_content_sha256"),
            "test metadata record",
        )
        if raw["schema_version"] != TEST_METADATA_SCHEMA or raw["split"] != "test":
            raise P3CError("test metadata record schema/split differs")
        normalized.append(
            {
                "schema_version": TEST_METADATA_SCHEMA,
                "identity": canonical_identity(raw["identity"]),
                "subject": _nonempty_string(raw["subject"], "test subject"),
                "split": "test",
                "sanitized_content_sha256": _sha256(
                    raw["sanitized_content_sha256"], "test content hash"
                ),
            }
        )
    keys = [canonical_record_key(record) for record in normalized]
    if keys != sorted(keys):
        raise P3CError("test metadata records are not in canonical order")
    identities = [identity_tuple(record["identity"]) for record in normalized]
    if len(set(identities)) != len(identities):
        raise P3CError("test metadata contains duplicate identities")
    if enforce_frozen_counts:
        if len(normalized) != int(card["task"]["outcome_sample_count"]):
            raise P3CError("test metadata must contain exactly 14,042 identities")
        if len({record["subject"] for record in normalized}) != 57:
            raise P3CError("test metadata must cover exactly 57 subjects")
    digest = ordered_identity_sha256([record["identity"] for record in normalized])
    if expected_ordered_identity_sha256 is not None and digest != expected_ordered_identity_sha256:
        raise P3CError("ordered test identity SHA256 differs from the planning freeze")
    return normalized


def assert_offline_metadata_environment() -> None:
    for key in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise P3CError("%s must equal 1" % key)
    if os.environ.get("HF_ENDPOINT"):
        raise P3CError("HF_ENDPOINT must be unset for P3-C")
    if os.environ.get("HF_DATASETS_CACHE") != "/hpc2hdd/home/xhuang225/shared/datasets":
        raise P3CError("HF_DATASETS_CACHE differs from the authorized read-only cache")


def _default_test_dataset_loader(
    task_name: str, subject: str, card: Mapping[str, Any]
) -> Tuple[Sequence[Mapping[str, Any]], Dict[str, Any]]:
    try:
        from datasets import DownloadMode, load_dataset
    except Exception as exc:  # pragma: no cover - audited remote dependency path.
        raise P3CError("C0 requires the audited datasets runtime") from exc
    rows = load_dataset(
        path=card["task"]["dataset"],
        name=subject,
        revision=card["task"]["dataset_revision"],
        split="test",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    if not hasattr(rows, "select_columns"):
        raise P3CError("cached MMLU test split cannot enforce column projection")
    projected = rows.select_columns(["question", "choices", "subject"])
    if set(getattr(projected, "column_names", ())) != {"question", "choices", "subject"}:
        raise P3CError("safe test projection exposed unexpected columns")
    return projected, {
        "task_name": task_name,
        "subject": subject,
        "split": "test",
        "row_count": len(projected),
        "projected_columns": ["choices", "question", "subject"],
        "dataset_fingerprint": str(getattr(projected, "_fingerprint", "")),
    }


def build_safe_test_metadata(
    source_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    dataset_loader: Optional[
        Callable[[str, str, Mapping[str, Any]], Tuple[Sequence[Mapping[str, Any]], Dict[str, Any]]]
    ] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_records = validate_source_manifest(source_manifest, card)
    task_subject: Dict[str, str] = {}
    for record in source_records:
        task = record["identity"]["task"]
        subject = record["subject"]
        if task != "mmlu_%s" % subject:
            raise P3CError("P3-B task/subject identity differs from frozen MMLU naming")
        prior = task_subject.setdefault(task, subject)
        if prior != subject:
            raise P3CError("one MMLU task maps to multiple subjects")
    if len(task_subject) != int(card["task"]["trajectory_subject_count"]):
        raise P3CError("P3-B source does not expose exactly 57 test configurations")
    loader = dataset_loader or _default_test_dataset_loader
    records: List[Dict[str, Any]] = []
    evidence = []
    for task_name, subject in sorted(task_subject.items()):
        rows, task_evidence = loader(task_name, subject, card)
        evidence.append(dict(task_evidence))
        for row_index in range(len(rows)):
            safe = safe_test_dataset_fields(rows[row_index], subject)
            records.append(make_test_metadata_record(task_name, subject, row_index, safe))
    records.sort(key=canonical_record_key)
    normalized = validate_test_metadata_records(records, card)
    return normalized, sorted(evidence, key=lambda value: value["task_name"])


def make_test_metadata_manifest(
    records: Sequence[Mapping[str, Any]],
    dataset_evidence: Sequence[Mapping[str, Any]],
    metadata_file_sha256: str,
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = validate_test_metadata_records(records, card)
    payload: Dict[str, Any] = {
        "schema_version": TEST_MANIFEST_SCHEMA,
        "artifact_role": "test14042_identity_content_metadata_only",
        "gate": GATE,
        "stage": "C0",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "dataset": {
            "repo": card["task"]["dataset"],
            "revision": card["task"]["dataset_revision"],
            "split": "test",
        },
        "record_count": len(normalized),
        "subject_count": len({record["subject"] for record in normalized}),
        "canonical_record_order": list(card["identity"]["canonical_record_order"]),
        "ordered_identity_sha256": ordered_identity_sha256(
            [record["identity"] for record in normalized]
        ),
        "ordered_content_sha256": hashlib.sha256(
            canonical_json_bytes(
                [record["sanitized_content_sha256"] for record in normalized]
            )
        ).hexdigest(),
        "metadata_file": TEST_METADATA_NAME,
        "metadata_file_sha256": _sha256(metadata_file_sha256, "test metadata file"),
        "dataset_evidence": [dict(value) for value in dataset_evidence],
        "projected_fields": ["identity", "subject", "split", "sanitized_content_sha256"],
        "forbidden_fields_accessed": [],
        "outcome_values_read": False,
    }
    attach_manifest_sha256(payload)
    return payload


def load_verified_phase1_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    expected_path = PHASE1_ROOT / "control/phase1_run_manifest.json"
    if path != expected_path:
        raise P3CError("Phase 1 manifest path differs from the canonical root")
    payload = load_strict_json(path)
    if payload.get("manifest_sha256") != EXPECTED_PHASE1_MANIFEST_INTERNAL_SHA256:
        raise P3CError("Phase 1 manifest internal SHA256 differs from the handoff")
    if manifest_sha256(payload) != EXPECTED_PHASE1_MANIFEST_INTERNAL_SHA256:
        raise P3CError("Phase 1 manifest self-hash is invalid")
    if Path(str(payload.get("run_root", ""))).resolve() != PHASE1_ROOT:
        raise P3CError("Phase 1 manifest run_root differs")
    recipe = payload.get("frozen_recipe")
    if not isinstance(recipe, Mapping):
        raise P3CError("Phase 1 manifest lacks frozen_recipe")
    exact_recipe = {
        "repo_id": "Qwen/Qwen3-1.7B-Base",
        "revision": MODEL_REVISION,
        "task": "mmlu",
        "num_fewshot": 5,
        "dtype": "float16",
        "k": 2,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "window_width": 4,
    }
    for key, expected in exact_recipe.items():
        if recipe.get(key) != expected:
            raise P3CError("Phase 1 frozen recipe mismatch at %s" % key)
    return payload


def build_historical_source_allowlist(
    phase1_manifest_path: Path,
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest = load_verified_phase1_manifest(phase1_manifest_path)
    stage = manifest.get("stages", {}).get("gate-e-full")
    if not isinstance(stage, Mapping) or not isinstance(stage.get("jobs"), list):
        raise P3CError("Phase 1 manifest lacks Gate-E full jobs")
    recipe = manifest["frozen_recipe"]
    jobs: Dict[str, Dict[str, Any]] = {}
    baseline: Optional[Dict[str, Any]] = None
    for raw in stage["jobs"]:
        try:
            normalized = _validate_manifest_eval_job(raw, PHASE1_ROOT, recipe)
        except AnalysisError as exc:
            raise P3CError("Phase 1 Gate-E job metadata failed validation") from exc
        window = normalized["window"]
        if window is None:
            if baseline is not None:
                raise P3CError("Phase 1 manifest contains multiple baselines")
            baseline = normalized
        else:
            if window in jobs:
                raise P3CError("Phase 1 manifest contains duplicate historical window")
            jobs[window] = normalized
    if baseline is None or set(jobs) != set(HISTORICAL_WINDOWS):
        raise P3CError("Phase 1 mapping is not exactly baseline plus historical-13")
    entries = [_historical_cell_metadata("baseline", baseline, card)]
    entries.extend(_historical_cell_metadata(window, jobs[window], card) for window in HISTORICAL_WINDOWS)
    payload: Dict[str, Any] = {
        "schema_version": HISTORICAL_ALLOWLIST_SCHEMA,
        "artifact_role": "phase1_baseline_historical13_exact_source_allowlist",
        "gate": GATE,
        "stage": "C0",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "phase1_manifest": {
            "path": str(Path(phase1_manifest_path).resolve()),
            "file_sha256": file_sha256(phase1_manifest_path),
            "internal_sha256": manifest["manifest_sha256"],
        },
        "cells": entries,
        "baseline_count": 1,
        "historical_window_order": list(HISTORICAL_WINDOWS),
        "historical_count": 13,
        "results_content_parsed": False,
        "results_values_read": False,
        "sample_content_parsed": False,
    }
    attach_manifest_sha256(payload)
    return payload


def _historical_cell_metadata(
    cell: str,
    normalized_job: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(normalized_job["output_dir"]).resolve()
    expected_dir = (
        PHASE1_ROOT / "gate-e-full/baseline-full"
        if cell == "baseline"
        else PHASE1_ROOT / ("gate-e-full/window-%s-full" % cell.replace(":", "-"))
    )
    if output_dir != expected_dir:
        raise P3CError("Phase 1 output_dir differs for %s" % cell)
    command_path = output_dir / "command_args.json"
    revision_path = output_dir / "model_revision.json"
    results_path = output_dir / "results.json"
    for path in (command_path, revision_path, results_path):
        if not path.is_file() or path.is_symlink():
            raise P3CError("historical source is missing or symlinked: %s" % path)
    command = load_strict_json(command_path)
    revision = load_strict_json(revision_path)
    _validate_cell_command(command, cell, output_dir, card)
    _validate_cell_revision(revision)
    sidecars = sorted(output_dir.glob("samples_*.jsonl"), key=lambda path: path.name)
    return {
        "cell": cell,
        "window": None if cell == "baseline" else cell,
        "output_dir": str(output_dir),
        "job_id": normalized_job["job_id"],
        "command_args": _file_metadata(command_path),
        "model_revision": _file_metadata(revision_path),
        "results": _file_metadata(results_path),
        "sample_sidecars": [_file_metadata(path) for path in sidecars],
        "recipe": {
            "loop": cell != "baseline",
            "k": 2,
            "fixed_horizon": True,
            "step_size": 0.5,
            "total_horizon": 1.0,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
        },
    }


def _validate_cell_command(
    command: Mapping[str, Any],
    cell: str,
    output_dir: Path,
    card: Mapping[str, Any],
) -> None:
    exact = {
        "model": "qwen3-1.7b-base",
        "revision": MODEL_REVISION,
        "tasks": "mmlu",
        "num_fewshot": 5,
        "dtype": "float16",
        "limit": None,
        "first_n": None,
        "batch_size": "auto",
        "k": 2,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "output_dir": str(output_dir),
        "loop": cell != "baseline",
    }
    for key, expected in exact.items():
        if command.get(key) != expected:
            raise P3CError("historical command mismatch at %s for %s" % (key, cell))
    if cell != "baseline" and command.get("window") != cell:
        raise P3CError("historical command window differs from manifest mapping")


def _validate_cell_revision(revision: Mapping[str, Any]) -> None:
    if revision.get("repo_id") != "Qwen/Qwen3-1.7B-Base" or revision.get("match") is not True:
        raise P3CError("historical model revision receipt differs")
    for key in ("commit_hash", "manifest_commit", "model_commit", "tokenizer_commit"):
        if revision.get(key) != MODEL_REVISION:
            raise P3CError("historical model/tokenizer revision mismatch at %s" % key)


def scan_blind_outcome_existence(search_root: Path) -> Dict[str, Any]:
    search_root = Path(search_root).resolve()
    if search_root != BLIND_SEARCH_ROOT:
        raise P3CError("blind metadata search root differs from the authorization")
    counters = {
        window: {"command_args_occurrences": 0, "path_results_occurrences": 0}
        for window in BLIND_WINDOWS
    }
    tokens = {window: "window-%s" % window.replace(":", "-") for window in BLIND_WINDOWS}
    command_files_scanned = 0
    results_paths_scanned = 0
    for directory, dirnames, filenames in os.walk(str(search_root)):
        dirnames[:] = [name for name in dirnames if name not in {".git", "__pycache__"}]
        current = Path(directory)
        if "results.json" in filenames:
            results_paths_scanned += 1
            parts = set(current.parts)
            for window, token in tokens.items():
                if token in parts:
                    counters[window]["path_results_occurrences"] += 1
        if "command_args.json" in filenames:
            command_files_scanned += 1
            command = load_strict_json(current / "command_args.json")
            raw_window = command.get("window")
            if raw_window in counters:
                counters[str(raw_window)]["command_args_occurrences"] += 1
    if any(
        value["command_args_occurrences"] or value["path_results_occurrences"]
        for value in counters.values()
    ):
        raise P3CError("blind-12 outcome existence metadata is non-zero")
    payload: Dict[str, Any] = {
        "schema_version": BLIND_EXISTENCE_SCHEMA,
        "artifact_role": "blind12_outcome_existence_metadata_refresh",
        "gate": GATE,
        "stage": "C0",
        "checked_at_utc": utc_now(),
        "check_mode": "read_only_non_numeric_metadata_only",
        "search_root": str(search_root),
        "command_files_scanned": command_files_scanned,
        "results_paths_scanned": results_paths_scanned,
        "blind_windows": counters,
        "outcome_values_read": False,
        "decision": "BLIND12_METADATA_CLOSURE_PASS",
    }
    attach_manifest_sha256(payload)
    return payload


def materialize_c0(
    *,
    card_path: Path,
    p3b_root: Path,
    p3c_root: Path,
    phase1_root: Path,
    blind_search_root: Path,
    expected_commit: str,
    argv: Sequence[str],
    dataset_loader: Optional[
        Callable[[str, str, Mapping[str, Any]], Tuple[Sequence[Mapping[str, Any]], Dict[str, Any]]]
    ] = None,
) -> Dict[str, Any]:
    assert_authorized_paths(card_path, p3b_root, p3c_root, phase1_root)
    if Path(blind_search_root).resolve() != BLIND_SEARCH_ROOT:
        raise P3CError("blind search root differs from the exact C0 authorization")
    if Path(p3c_root).exists():
        raise FileExistsError("authorized P3-C run root already exists")
    assert_offline_metadata_environment()
    git = git_provenance(expected_commit, require_remote_root=True)
    card = load_phase3_card(card_path)
    source_path = Path(p3b_root) / P3B_SOURCE_NAME
    source_manifest = load_strict_json(source_path)
    source_records = validate_source_manifest(source_manifest, card)
    verifier = _load_verified_p3b_verifier(Path(p3b_root))
    test_records, dataset_evidence = build_safe_test_metadata(
        source_manifest, card, dataset_loader=dataset_loader
    )
    disjointness = verify_population_disjointness(
        [
            {
                "identity": record["identity"],
                "split": "validation",
                "sanitized_content_sha256": record["sanitized_content_sha256"],
            }
            for record in source_records
        ],
        [
            {
                "identity": record["identity"],
                "split": "test",
                "sanitized_content_sha256": record["sanitized_content_sha256"],
            }
            for record in test_records
        ],
        card,
    )
    allowlist = build_historical_source_allowlist(
        Path(phase1_root) / "control/phase1_run_manifest.json", card
    )
    blind = scan_blind_outcome_existence(blind_search_root)

    Path(p3c_root).mkdir(parents=False, exist_ok=False)
    metadata_file_sha = write_new_jsonl(Path(p3c_root) / TEST_METADATA_NAME, test_records)
    test_manifest = make_test_metadata_manifest(
        test_records, dataset_evidence, metadata_file_sha, card
    )
    test_manifest_file_sha = write_new_json(Path(p3c_root) / TEST_MANIFEST_NAME, test_manifest)
    disjointness_payload: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3c-disjointness-receipt.v1",
        "artifact_role": "validation1531_vs_test14042_disjointness",
        "gate": GATE,
        "stage": "C0",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_file_sha256": file_sha256(source_path),
        "source_manifest_internal_sha256": source_manifest["manifest_sha256"],
        "test_manifest_file_sha256": test_manifest_file_sha,
        "test_manifest_internal_sha256": test_manifest["manifest_sha256"],
        "closure": disjointness,
        "outcome_values_read": False,
    }
    attach_manifest_sha256(disjointness_payload)
    disjointness_file_sha = write_new_json(
        Path(p3c_root) / DISJOINTNESS_NAME, disjointness_payload
    )
    blind_file_sha = write_new_json(Path(p3c_root) / BLIND_EXISTENCE_NAME, blind)
    allowlist_file_sha = write_new_json(
        Path(p3c_root) / HISTORICAL_ALLOWLIST_NAME, allowlist
    )
    c0: Dict[str, Any] = {
        "schema_version": C0_RECEIPT_SCHEMA,
        "artifact_role": "p3c_c0_canonical_source_outcome_metadata_closure",
        "gate": GATE,
        "stage": "C0",
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "argv": list(argv),
        "implementation_sha256": implementation_hashes(),
        "inputs": {
            "p3b_root": str(Path(p3b_root).resolve()),
            "p3b_verifier_file_sha256": EXPECTED_P3B_VERIFIER_FILE_SHA256,
            "p3b_verifier_internal_sha256": verifier["manifest_sha256"],
            "phase1_root": str(Path(phase1_root).resolve()),
            "phase1_manifest_internal_sha256": allowlist["phase1_manifest"]["internal_sha256"],
        },
        "outputs": {
            TEST_METADATA_NAME: {"file_sha256": metadata_file_sha},
            TEST_MANIFEST_NAME: {
                "file_sha256": test_manifest_file_sha,
                "internal_sha256": test_manifest["manifest_sha256"],
            },
            DISJOINTNESS_NAME: {
                "file_sha256": disjointness_file_sha,
                "internal_sha256": disjointness_payload["manifest_sha256"],
            },
            BLIND_EXISTENCE_NAME: {
                "file_sha256": blind_file_sha,
                "internal_sha256": blind["manifest_sha256"],
            },
            HISTORICAL_ALLOWLIST_NAME: {
                "file_sha256": allowlist_file_sha,
                "internal_sha256": allowlist["manifest_sha256"],
            },
        },
        "checks": {
            "test14042_exact": True,
            "gold_free_ordered_test_identity_sha256": test_manifest[
                "ordered_identity_sha256"
            ],
            "phase1_legacy_ordered_identity_sha256_deferred_to_c2": (
                EXPECTED_TEST_ORDERED_IDENTITY_SHA256
            ),
            "validation_test_identity_intersection_zero": True,
            "validation_test_content_intersection_zero": True,
            "phase1_mapping_exact_baseline_plus_historical13": True,
            "blind12_existence_zero": True,
            "results_json_content_parsed": False,
            "samples_content_parsed": False,
            "outcome_values_read": False,
        },
        "status": "C0_METADATA_CLOSURE_PASS",
    }
    attach_manifest_sha256(c0)
    write_new_json(Path(p3c_root) / C0_RECEIPT_NAME, c0)
    return c0


def materialize_c1(
    *,
    card_path: Path,
    p3b_root: Path,
    p3c_root: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    """Freeze the complete baseline-only atlas in an outcome-root-free process."""

    assert_authorized_paths(card_path, p3b_root, p3c_root)
    git = git_provenance(expected_commit, require_remote_root=True)
    card = load_phase3_card(card_path)
    c0 = load_verified_c0(p3c_root, card)
    bundle = _load_p3b_analysis_bundle(p3b_root, card, load_trajectories=True)
    analysis = analyze_manifested_phase3_trajectories(
        bundle["trajectories"],
        bundle["source_manifest"],
        bundle["pool_manifest"],
        card,
    )
    validate_frozen_analysis_contract(analysis, card)

    atlas = _c1_artifact("baseline_width4_atlas", analysis["window_metrics"], bundle)
    reports = {
        "SHIFT": _c1_artifact("shift_overlap_report", analysis["variants"]["SHIFT"], bundle),
        "FLANK": _c1_artifact(
            "nonoverlap_flank_report", analysis["variants"]["FLANK"], bundle
        ),
        "CONSENSUS": _c1_artifact(
            "consensus_report", analysis["variants"]["CONSENSUS"], bundle
        ),
    }
    output_paths = {
        ATLAS_NAME: atlas,
        SHIFT_NAME: reports["SHIFT"],
        FLANK_NAME: reports["FLANK"],
        CONSENSUS_NAME: reports["CONSENSUS"],
    }
    for name in tuple(output_paths) + (SELECTOR_FREEZE_NAME, SELECTOR_RECEIPT_NAME):
        if (Path(p3c_root) / name).exists():
            raise FileExistsError("refusing to replace frozen C1 artifact: %s" % name)
    written = {}
    for name, payload in output_paths.items():
        written[name] = {
            "file_sha256": write_new_json(Path(p3c_root) / name, payload),
            "internal_sha256": payload["manifest_sha256"],
        }
    selector: Dict[str, Any] = {
        "schema_version": SELECTOR_FREEZE_SCHEMA,
        "artifact_role": "p3c_outcome_blind_all25_selector_freeze",
        "gate": GATE,
        "stage": "C1",
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "argv": list(argv),
        "implementation_sha256": implementation_hashes(),
        "analysis_implementation_sha256": file_sha256(
            repository_root() / "src/tflt/loopscope/phase3_analysis.py"
        ),
        "inputs": _bundle_provenance(bundle),
        "c0_receipt_internal_sha256": c0["manifest_sha256"],
        "component_artifacts": written,
        "analysis": analysis,
        "process_isolation": {
            "input_roles": [
                "frozen_card",
                "p3b_source_manifest",
                "p3b_pool_manifest",
                "p3b_no_loop_trajectory",
                "c0_receipts_only",
            ],
            "phase1_historical_root_argument_present": False,
            "historical_source_loaded": False,
            "outcome_root_loaded": False,
            "outcome_fields_consumed": False,
            "variant_selection_performed": False,
        },
        "freeze_irreversible": True,
    }
    attach_manifest_sha256(selector)
    selector_file_sha = write_new_json(Path(p3c_root) / SELECTOR_FREEZE_NAME, selector)

    # Re-open and recompute from raw trajectories before sealing the receipt.
    verification = _verify_c1_core(p3b_root, p3c_root, card, selector)
    receipt: Dict[str, Any] = {
        "schema_version": SELECTOR_RECEIPT_SCHEMA,
        "artifact_role": "p3c_selector_freeze_independent_recompute_receipt",
        "gate": GATE,
        "stage": "C1",
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "selector_freeze_file_sha256": selector_file_sha,
        "selector_freeze_internal_sha256": selector["manifest_sha256"],
        "component_artifacts": written,
        "verification": verification,
        "implementation_sha256": implementation_hashes(),
        "outcome_fields_consumed": False,
        "variant_selection_performed": False,
        "historical_source_loaded": False,
        "status": "C1_SELECTOR_FREEZE_PASS",
    }
    attach_manifest_sha256(receipt)
    write_new_json(Path(p3c_root) / SELECTOR_RECEIPT_NAME, receipt)
    return receipt


def verify_c1_freeze(
    *,
    card_path: Path,
    p3b_root: Path,
    p3c_root: Path,
    expected_commit: str,
) -> Dict[str, Any]:
    assert_authorized_paths(card_path, p3b_root, p3c_root)
    git = git_provenance(expected_commit, require_remote_root=True)
    card = load_phase3_card(card_path)
    c0 = load_verified_c0(p3c_root, card)
    selector = load_strict_json(Path(p3c_root) / SELECTOR_FREEZE_NAME)
    receipt = load_strict_json(Path(p3c_root) / SELECTOR_RECEIPT_NAME)
    verify_manifest_sha256(selector)
    verify_manifest_sha256(receipt)
    if (
        selector.get("schema_version") != SELECTOR_FREEZE_SCHEMA
        or selector.get("gate") != GATE
        or selector.get("stage") != "C1"
        or selector.get("card_sha256") != PHASE3_CARD_BYTE_SHA256
        or selector.get("freeze_irreversible") is not True
    ):
        raise P3CError("C1 selector freeze identity differs")
    if selector.get("git") != git:
        raise P3CError("C1 selector freeze git provenance differs from the live clone")
    if selector.get("implementation_sha256") != implementation_hashes():
        raise P3CError("C1 selector implementation hashes differ from committed code")
    if selector.get("analysis_implementation_sha256") != file_sha256(
        repository_root() / "src/tflt/loopscope/phase3_analysis.py"
    ):
        raise P3CError("C1 analysis implementation hash differs")
    if selector.get("c0_receipt_internal_sha256") != c0.get("manifest_sha256"):
        raise P3CError("C1 selector freeze does not bind the verified C0 receipt")
    isolation = selector.get("process_isolation")
    if not isinstance(isolation, Mapping) or any(
        isolation.get(key) is not False
        for key in (
            "phase1_historical_root_argument_present",
            "historical_source_loaded",
            "outcome_root_loaded",
            "outcome_fields_consumed",
            "variant_selection_performed",
        )
    ):
        raise P3CError("C1 selector freeze isolation proof differs")
    if receipt.get("status") != "C1_SELECTOR_FREEZE_PASS":
        raise P3CError("C1 selector receipt is not PASS")
    if receipt.get("selector_freeze_file_sha256") != file_sha256(
        Path(p3c_root) / SELECTOR_FREEZE_NAME
    ):
        raise P3CError("selector freeze file hash differs from its receipt")
    if receipt.get("selector_freeze_internal_sha256") != selector.get("manifest_sha256"):
        raise P3CError("selector freeze internal hash differs from its receipt")
    if receipt.get("implementation_sha256") != implementation_hashes():
        raise P3CError("C1 receipt implementation hashes differ from current committed code")
    if receipt.get("git") != git:
        raise P3CError("C1 receipt git provenance differs from the live clone")
    if receipt.get("component_artifacts") != selector.get("component_artifacts"):
        raise P3CError("C1 receipt component hashes differ from the selector freeze")
    if any(
        receipt.get(key) is not False
        for key in (
            "outcome_fields_consumed",
            "variant_selection_performed",
            "historical_source_loaded",
        )
    ):
        raise P3CError("C1 selector receipt reports premature scientific access")
    verification = _verify_c1_core(p3b_root, p3c_root, card, selector)
    if canonical_json_bytes(verification) != canonical_json_bytes(receipt.get("verification")):
        raise P3CError("C1 independent recomputation differs from the frozen receipt")
    return {
        "status": "C1_SELECTOR_FREEZE_REVERIFIED",
        "git": git,
        "selector_freeze_file_sha256": file_sha256(
            Path(p3c_root) / SELECTOR_FREEZE_NAME
        ),
        "selector_freeze_internal_sha256": selector["manifest_sha256"],
        "receipt_file_sha256": file_sha256(Path(p3c_root) / SELECTOR_RECEIPT_NAME),
        "receipt_internal_sha256": receipt["manifest_sha256"],
        "verification": verification,
        "outcome_values_read": False,
    }


def load_verified_c0(p3c_root: Path, card: Mapping[str, Any]) -> Dict[str, Any]:
    root = Path(p3c_root)
    c0 = load_strict_json(root / C0_RECEIPT_NAME)
    verify_manifest_sha256(c0)
    if c0.get("status") != "C0_METADATA_CLOSURE_PASS":
        raise P3CError("C0 receipt is not PASS")
    if c0.get("card_sha256") != PHASE3_CARD_BYTE_SHA256:
        raise P3CError("C0 receipt is bound to another card")
    if c0.get("implementation_sha256") != implementation_hashes():
        raise P3CError("C0 implementation hashes differ from current committed code")
    outputs = c0.get("outputs")
    if not isinstance(outputs, Mapping):
        raise P3CError("C0 receipt outputs are missing")
    expected_output_names = {
        TEST_METADATA_NAME,
        TEST_MANIFEST_NAME,
        DISJOINTNESS_NAME,
        BLIND_EXISTENCE_NAME,
        HISTORICAL_ALLOWLIST_NAME,
    }
    if set(outputs) != expected_output_names:
        raise P3CError("C0 receipt output set is not the exact five-artifact closure")
    for name, expected in outputs.items():
        path = root / name
        if not path.is_file() or file_sha256(path) != expected.get("file_sha256"):
            raise P3CError("C0 output file hash mismatch: %s" % name)
        if name != TEST_METADATA_NAME:
            payload = load_strict_json(path)
            verify_manifest_sha256(payload)
            if payload["manifest_sha256"] != expected.get("internal_sha256"):
                raise P3CError("C0 output internal hash mismatch: %s" % name)
    test_records = load_strict_jsonl(root / TEST_METADATA_NAME)
    validate_test_metadata_records(test_records, card)
    test_manifest = load_strict_json(root / TEST_MANIFEST_NAME)
    if test_manifest.get("metadata_file_sha256") != file_sha256(root / TEST_METADATA_NAME):
        raise P3CError("test metadata manifest does not bind the JSONL file")
    _validate_c0_artifact_contract(root, c0, test_records, test_manifest, card)
    return c0


def _validate_c0_artifact_contract(
    root: Path,
    c0: Mapping[str, Any],
    test_records: Sequence[Mapping[str, Any]],
    test_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> None:
    checks = c0.get("checks")
    required_true = {
        "test14042_exact",
        "validation_test_identity_intersection_zero",
        "validation_test_content_intersection_zero",
        "phase1_mapping_exact_baseline_plus_historical13",
        "blind12_existence_zero",
    }
    required_false = {
        "results_json_content_parsed",
        "samples_content_parsed",
        "outcome_values_read",
    }
    if not isinstance(checks, Mapping):
        raise P3CError("C0 receipt checks are missing")
    if any(checks.get(key) is not True for key in required_true) or any(
        checks.get(key) is not False for key in required_false
    ):
        raise P3CError("C0 receipt isolation/closure checks differ")
    if checks.get("phase1_legacy_ordered_identity_sha256_deferred_to_c2") != (
        EXPECTED_TEST_ORDERED_IDENTITY_SHA256
    ):
        raise P3CError("C0 receipt does not preserve the deferred Phase 1 identity hash")

    if test_manifest.get("schema_version") != TEST_MANIFEST_SCHEMA:
        raise P3CError("test metadata manifest schema differs")
    if test_manifest.get("record_count") != len(test_records) or len(test_records) != 14042:
        raise P3CError("test metadata manifest does not bind exact 14,042 closure")
    if test_manifest.get("subject_count") != 57:
        raise P3CError("test metadata manifest does not bind exact 57-subject closure")
    computed_ordered = ordered_identity_sha256(
        [record["identity"] for record in test_records]
    )
    if test_manifest.get("ordered_identity_sha256") != computed_ordered or checks.get(
        "gold_free_ordered_test_identity_sha256"
    ) != computed_ordered:
        raise P3CError("test metadata manifest ordered identity hash differs")
    dataset = test_manifest.get("dataset")
    if dataset != {
        "repo": card["task"]["dataset"],
        "revision": card["task"]["dataset_revision"],
        "split": "test",
    }:
        raise P3CError("test metadata manifest dataset binding differs")
    if test_manifest.get("forbidden_fields_accessed") != []:
        raise P3CError("test metadata manifest reports forbidden-field access")
    if test_manifest.get("outcome_values_read") is not False:
        raise P3CError("test metadata manifest reports outcome-value access")
    if test_manifest.get("projected_fields") != [
        "identity",
        "subject",
        "split",
        "sanitized_content_sha256",
    ]:
        raise P3CError("test metadata manifest projected fields differ")
    evidence = test_manifest.get("dataset_evidence")
    if not isinstance(evidence, list) or len(evidence) != 57:
        raise P3CError("test metadata dataset evidence does not cover 57 tasks")
    task_names = []
    total_rows = 0
    for item in evidence:
        if not isinstance(item, Mapping):
            raise P3CError("test metadata dataset evidence row is malformed")
        task_names.append(_nonempty_string(item.get("task_name"), "test evidence task"))
        if item.get("split") != "test" or item.get("projected_columns") != [
            "choices",
            "question",
            "subject",
        ]:
            raise P3CError("test metadata dataset evidence projection differs")
        row_count = item.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
            raise P3CError("test metadata dataset evidence row count is invalid")
        total_rows += row_count
        _nonempty_string(item.get("dataset_fingerprint"), "test dataset fingerprint")
    if task_names != sorted(task_names) or len(set(task_names)) != 57 or total_rows != 14042:
        raise P3CError("test metadata dataset evidence task/order/count closure differs")

    disjointness = load_strict_json(root / DISJOINTNESS_NAME)
    closure = disjointness.get("closure")
    if not isinstance(closure, Mapping) or any(
        closure.get(key) != expected
        for key, expected in {
            "validation_count": 1531,
            "test_count": 14042,
            "identity_intersection_count": 0,
            "sanitized_content_intersection_count": 0,
        }.items()
    ):
        raise P3CError("validation/test disjointness receipt closure differs")
    if disjointness.get("outcome_values_read") is not False:
        raise P3CError("disjointness receipt reports outcome access")

    blind = load_strict_json(root / BLIND_EXISTENCE_NAME)
    counters = blind.get("blind_windows")
    if (
        blind.get("schema_version") != BLIND_EXISTENCE_SCHEMA
        or blind.get("decision") != "BLIND12_METADATA_CLOSURE_PASS"
        or blind.get("outcome_values_read") is not False
        or not isinstance(counters, Mapping)
        or set(counters) != set(BLIND_WINDOWS)
    ):
        raise P3CError("blind-12 existence receipt contract differs")
    for value in counters.values():
        if value != {"command_args_occurrences": 0, "path_results_occurrences": 0}:
            raise P3CError("blind-12 existence receipt contains a value root")

    allowlist = load_strict_json(root / HISTORICAL_ALLOWLIST_NAME)
    _validate_allowlist_for_c2(allowlist)
    if (
        allowlist.get("results_content_parsed") is not False
        or allowlist.get("sample_content_parsed") is not False
        or allowlist.get("phase1_manifest", {}).get("internal_sha256")
        != EXPECTED_PHASE1_MANIFEST_INTERNAL_SHA256
    ):
        raise P3CError("historical source allowlist reports premature value access/drift")


def validate_frozen_analysis_contract(
    analysis: Mapping[str, Any], card: Mapping[str, Any]
) -> None:
    if analysis.get("outcome_fields_consumed") is not False:
        raise P3CError("baseline analysis consumed outcome fields")
    if analysis.get("variant_selection_performed") is not False:
        raise P3CError("baseline analysis selected a variant before C2")
    windows = analysis.get("window_metrics")
    if not isinstance(windows, list) or [row.get("start") for row in windows] != list(range(25)):
        raise P3CError("baseline atlas does not contain ordered starts 0..24")
    partitions = partition_starts(card)
    expected_scopes = {
        "deployment": None,
        "retrospective": set(partitions["historical_13"]),
        "blind_enrichment": set(partitions["blind_12"]),
    }
    variants = analysis.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != {"SHIFT", "FLANK", "CONSENSUS"}:
        raise P3CError("selector freeze variants differ from the card")
    for variant, payload in variants.items():
        expected_support = list(card["variants"][variant]["support_starts"])
        if payload.get("support_starts") != expected_support:
            raise P3CError("%s support differs from the card" % variant)
        rows = payload.get("windows")
        if not isinstance(rows, list) or [row.get("start") for row in rows] != expected_support:
            raise P3CError("%s rows differ from frozen support" % variant)
        row_by_start = {row["start"]: row for row in rows}
        scopes = payload.get("scopes")
        if not isinstance(scopes, Mapping) or set(scopes) != set(expected_scopes):
            raise P3CError("%s scopes differ from the frozen contract" % variant)
        for scope_name, exact_partition in expected_scopes.items():
            scope = scopes[scope_name]
            starts = scope.get("scope_starts")
            expected_starts = (
                expected_support
                if exact_partition is None
                else [start for start in partitions["historical_13" if scope_name == "retrospective" else "blind_12"] if start in expected_support]
            )
            if starts != expected_starts:
                raise P3CError("%s %s scope starts differ" % (variant, scope_name))
            ranking = scope.get("published_ranking")
            if not isinstance(ranking, list) or [row.get("rank") for row in ranking] != list(
                range(1, len(ranking) + 1)
            ):
                raise P3CError("%s %s rank closure differs" % (variant, scope_name))
            ranked_starts = [row.get("start") for row in ranking]
            if set(ranked_starts) != set(expected_starts) or len(ranked_starts) != len(set(ranked_starts)):
                raise P3CError("%s %s ranking is not an exact support closure" % (variant, scope_name))
            expected_rank = sorted(
                expected_starts,
                key=lambda start: (-float(row_by_start[start]["score"]), start),
            )
            if ranked_starts != expected_rank:
                raise P3CError("%s %s ranking differs from frozen scores" % (variant, scope_name))
            eligible = sorted(
                start for start in expected_starts if bool(row_by_start[start]["point_eligible"])
            )
            if scope.get("point_eligible_starts") != eligible:
                raise P3CError("%s %s eligibility closure differs" % (variant, scope_name))


def _c1_artifact(
    role: str, payload: Any, bundle: Mapping[str, Any]
) -> Dict[str, Any]:
    artifact: Dict[str, Any] = {
        "schema_version": C1_ARTIFACT_SCHEMA,
        "artifact_role": role,
        "gate": GATE,
        "stage": "C1",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "inputs": _bundle_provenance(bundle),
        "payload": payload,
        "outcome_fields_consumed": False,
        "variant_selection_performed": False,
    }
    attach_manifest_sha256(artifact)
    return artifact


def _verify_c1_core(
    p3b_root: Path,
    p3c_root: Path,
    card: Mapping[str, Any],
    selector: Mapping[str, Any],
) -> Dict[str, Any]:
    verify_manifest_sha256(selector)
    validate_frozen_analysis_contract(selector.get("analysis"), card)
    component_names = (ATLAS_NAME, SHIFT_NAME, FLANK_NAME, CONSENSUS_NAME)
    components = {}
    for name in component_names:
        payload = load_strict_json(Path(p3c_root) / name)
        verify_manifest_sha256(payload)
        if (
            payload.get("schema_version") != C1_ARTIFACT_SCHEMA
            or payload.get("gate") != GATE
            or payload.get("stage") != "C1"
            or payload.get("card_sha256") != PHASE3_CARD_BYTE_SHA256
            or payload.get("outcome_fields_consumed") is not False
            or payload.get("variant_selection_performed") is not False
        ):
            raise P3CError("selector component isolation/identity differs: %s" % name)
        components[name] = payload
        expected = selector.get("component_artifacts", {}).get(name)
        if not isinstance(expected, Mapping):
            raise P3CError("selector freeze misses component %s" % name)
        if file_sha256(Path(p3c_root) / name) != expected.get("file_sha256"):
            raise P3CError("selector component file hash mismatch: %s" % name)
        if payload["manifest_sha256"] != expected.get("internal_sha256"):
            raise P3CError("selector component internal hash mismatch: %s" % name)
    analysis = selector["analysis"]
    if components[ATLAS_NAME]["payload"] != analysis["window_metrics"]:
        raise P3CError("atlas component differs from selector freeze")
    mapping = {
        SHIFT_NAME: "SHIFT",
        FLANK_NAME: "FLANK",
        CONSENSUS_NAME: "CONSENSUS",
    }
    for name, variant in mapping.items():
        if components[name]["payload"] != analysis["variants"][variant]:
            raise P3CError("variant component differs from selector freeze: %s" % variant)
    bundle = _load_p3b_analysis_bundle(p3b_root, card, load_trajectories=True)
    provenance = _bundle_provenance(bundle)
    if selector.get("inputs") != provenance:
        raise P3CError("selector freeze input provenance differs from live P3-B artifacts")
    if any(payload.get("inputs") != provenance for payload in components.values()):
        raise P3CError("selector component input provenance differs from live P3-B artifacts")
    recomputed = analyze_manifested_phase3_trajectories(
        bundle["trajectories"],
        bundle["source_manifest"],
        bundle["pool_manifest"],
        card,
    )
    validate_frozen_analysis_contract(recomputed, card)
    if canonical_json_bytes(recomputed) != canonical_json_bytes(analysis):
        raise P3CError("selector freeze differs from raw trajectory recomputation")
    return {
        "raw_trajectory_recomputed": True,
        "all25_window_metrics_exact": True,
        "shift_support_rank_exact": True,
        "flank_support_rank_exact": True,
        "consensus_support_rank_exact": True,
        "component_file_internal_hashes_exact": True,
        "card_source_pool_trajectory_bindings_exact": True,
        "analysis_implementation_sha256": file_sha256(
            repository_root() / "src/tflt/loopscope/phase3_analysis.py"
        ),
        "bootstrap_index_stream_sha256": recomputed["bootstrap"]["index_stream_sha256"],
        "outcome_fields_consumed": False,
        "variant_selection_performed": False,
        "historical_source_loaded": False,
    }


def _load_p3b_analysis_bundle(
    p3b_root: Path,
    card: Mapping[str, Any],
    *,
    load_trajectories: bool,
) -> Dict[str, Any]:
    root = Path(p3b_root)
    source_path = root / P3B_SOURCE_NAME
    pool_path = root / P3B_POOL_NAME
    trajectory_path = root / P3B_TRAJECTORY_NAME
    trajectory_manifest_path = root / P3B_TRAJECTORY_MANIFEST_NAME
    source = load_strict_json(source_path)
    pool = load_strict_json(pool_path)
    trajectory_manifest = load_strict_json(trajectory_manifest_path)
    verifier = _load_verified_p3b_verifier(root)
    validate_source_manifest(source, card)
    validate_pool_manifest(pool, source, card)
    validate_trajectory_manifest(trajectory_manifest, source, pool, card)
    if file_sha256(trajectory_path) != EXPECTED_P3B_TRAJECTORY_FILE_SHA256:
        raise P3CError("P3-B merged trajectory file SHA256 mismatch")
    if file_sha256(trajectory_manifest_path) != EXPECTED_P3B_TRAJECTORY_MANIFEST_FILE_SHA256:
        raise P3CError("P3-B trajectory manifest file SHA256 mismatch")
    if trajectory_manifest.get("manifest_sha256") != EXPECTED_P3B_TRAJECTORY_MANIFEST_INTERNAL_SHA256:
        raise P3CError("P3-B trajectory manifest internal SHA256 mismatch")
    return {
        "source_manifest": source,
        "pool_manifest": pool,
        "trajectory_manifest": trajectory_manifest,
        "verifier": verifier,
        "trajectories": load_strict_jsonl(trajectory_path) if load_trajectories else None,
        "paths": {
            "source": source_path,
            "pool": pool_path,
            "trajectory": trajectory_path,
            "trajectory_manifest": trajectory_manifest_path,
            "verifier": root / P3B_VERIFIER_NAME,
        },
    }


def _load_verified_p3b_verifier(root: Path) -> Dict[str, Any]:
    path = Path(root) / P3B_VERIFIER_NAME
    if file_sha256(path) != EXPECTED_P3B_VERIFIER_FILE_SHA256:
        raise P3CError("P3-B verifier file SHA256 mismatch")
    verifier = load_strict_json(path)
    verify_manifest_sha256(verifier)
    if verifier.get("manifest_sha256") != EXPECTED_P3B_VERIFIER_INTERNAL_SHA256:
        raise P3CError("P3-B verifier internal SHA256 mismatch")
    if verifier.get("status") != "PASS" or verifier.get("checks", {}).get("outcome_values_read") is not False:
        raise P3CError("P3-B verifier does not prove an outcome-unread trajectory")
    return verifier


def _bundle_provenance(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    paths = bundle["paths"]
    return {
        "source_manifest": {
            "path": str(paths["source"]),
            "file_sha256": file_sha256(paths["source"]),
            "internal_sha256": bundle["source_manifest"]["manifest_sha256"],
        },
        "pool_manifest": {
            "path": str(paths["pool"]),
            "file_sha256": file_sha256(paths["pool"]),
            "internal_sha256": bundle["pool_manifest"]["manifest_sha256"],
        },
        "trajectory": {
            "path": str(paths["trajectory"]),
            "file_sha256": file_sha256(paths["trajectory"]),
        },
        "trajectory_manifest": {
            "path": str(paths["trajectory_manifest"]),
            "file_sha256": file_sha256(paths["trajectory_manifest"]),
            "internal_sha256": bundle["trajectory_manifest"]["manifest_sha256"],
        },
        "p3b_verifier": {
            "path": str(paths["verifier"]),
            "file_sha256": file_sha256(paths["verifier"]),
            "internal_sha256": bundle["verifier"]["manifest_sha256"],
        },
    }


def retrospective_variant_decision(
    analysis: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    evidence_complete: bool = True,
) -> Dict[str, Any]:
    """Apply only the frozen card section 3.6 rules to C1 ranks."""

    if not evidence_complete:
        return {
            "science_label": "RETROSPECTIVE_INCONCLUSIVE",
            "selected_variant": None,
            "variant_evaluations": {},
            "stop_h3": True,
        }
    validate_frozen_analysis_contract(analysis, card)
    evaluations = {}
    passing = []
    target_start = 12
    false_positive_starts = {4, 6}
    for variant in VARIANT_PRIORITY:
        scope = analysis["variants"][variant]["scopes"]["retrospective"]
        frequency = scope["selection_frequency"]
        checks = {
            "unique_top1_is_12_15": scope["point_top1_start"] == target_start,
            "selection_frequency_at_least_0_80": (
                frequency is not None and float(frequency) >= 0.80
            ),
            "window_decision_selected": scope["window_decision"] == "SELECTED",
            "4_7_not_point_eligible": 4 not in scope["point_eligible_starts"],
            "6_9_not_point_eligible": 6 not in scope["point_eligible_starts"],
        }
        passed = all(checks.values())
        if passed:
            passing.append(variant)
        ranking = scope["published_ranking"]
        evaluations[variant] = {
            "checks": checks,
            "passes_retrospective_rule": passed,
            "point_top1_window": scope["point_top1_window"],
            "selection_frequency": frequency,
            "point_eligible_starts": list(scope["point_eligible_starts"]),
            "top3_windows": [row["window"] for row in ranking[:3]],
            "false_positive_starts": sorted(false_positive_starts),
        }
    if passing:
        selected = next(variant for variant in VARIANT_PRIORITY if variant in passing)
        label = "RETROSPECTIVE_TOP1_RECOVERY"
        stop = False
    else:
        selected = None
        shortlist = any(
            "12:15" in evaluations[variant]["top3_windows"] for variant in VARIANT_PRIORITY
        )
        label = (
            "RETROSPECTIVE_SHORTLIST_ONLY"
            if shortlist
            else "RETROSPECTIVE_NOT_SUPPORTED"
        )
        stop = True
    return {
        "science_label": label,
        "selected_variant": selected,
        "variant_priority": list(VARIANT_PRIORITY),
        "passing_variants": passing,
        "variant_evaluations": evaluations,
        "stop_h3": stop,
    }


def exact_paired_comparison(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """Reject any identity/order drift before calling the Phase 1 helper."""

    baseline_order = list(baseline)
    candidate_order = list(candidate)
    if baseline_order != candidate_order:
        raise P3CError("paired outcome identities/order are not an exact 14,042 closure")
    if len(baseline_order) != len(set(baseline_order)):
        raise P3CError("paired outcome identities contain duplicates")
    try:
        result = paired_comparison(
            baseline,
            candidate,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
    except AnalysisError as exc:
        raise P3CError("Phase 1 paired helper rejected the exact mapping") from exc
    alignment = result["sample_alignment"]
    if not alignment["exact_match"] or alignment["matched_count"] != len(baseline_order):
        raise P3CError("Phase 1 paired helper did not preserve exact closure")
    return result


def make_historical_alignment(
    analysis: Mapping[str, Any],
    historical_summaries: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Join frozen selector ranks to all historical outcomes without rescoring."""

    if set(historical_summaries) != set(HISTORICAL_WINDOWS) or set(paired) != set(
        HISTORICAL_WINDOWS
    ):
        raise P3CError("historical alignment requires the exact historical-13")
    outcome_order = sorted(
        HISTORICAL_WINDOWS,
        key=lambda window: (
            -float(historical_summaries[window]["accuracy_fraction"]),
            _window_start(window),
        ),
    )
    outcome_rank = {window: rank for rank, window in enumerate(outcome_order, start=1)}
    aligned: Dict[str, List[Dict[str, Any]]] = {}
    for variant in VARIANT_PRIORITY:
        variant_payload = analysis["variants"][variant]
        by_start = {row["start"]: row for row in variant_payload["windows"]}
        ranking = variant_payload["scopes"]["retrospective"]["published_ranking"]
        selector_rank = {row["start"]: row["rank"] for row in ranking}
        rows = []
        for window in HISTORICAL_WINDOWS:
            start = _window_start(window)
            selector_row = by_start.get(start)
            comparison = paired[window]
            rows.append(
                {
                    "window": window,
                    "start": start,
                    "selector_scorable": selector_row is not None,
                    "frozen_selector_rank": selector_rank.get(start),
                    "frozen_score": None if selector_row is None else selector_row["score"],
                    "frozen_point_eligible": (
                        False if selector_row is None else selector_row["point_eligible"]
                    ),
                    "historical_outcome_rank": outcome_rank[window],
                    "accuracy_fraction": historical_summaries[window]["accuracy_fraction"],
                    "delta_accuracy_pp": historical_summaries[window][
                        "delta_accuracy_pp_direct"
                    ],
                    "discordant_count": comparison["discordant_count"],
                    "exact_mcnemar_p": comparison["mcnemar"]["p_value"],
                }
            )
        aligned[variant] = rows
    return aligned


def materialize_c2(
    *,
    card_path: Path,
    p3b_root: Path,
    p3c_root: Path,
    phase1_root: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    """Unseal only baseline/historical-13 after an independent C1 verification."""

    assert_authorized_paths(card_path, p3b_root, p3c_root, phase1_root)
    git = git_provenance(expected_commit, require_remote_root=True)
    card = load_phase3_card(card_path)
    c1_verification = verify_c1_freeze(
        card_path=card_path,
        p3b_root=p3b_root,
        p3c_root=p3c_root,
        expected_commit=expected_commit,
    )
    c0 = load_verified_c0(p3c_root, card)
    selector = load_strict_json(Path(p3c_root) / SELECTOR_FREEZE_NAME)
    verify_manifest_sha256(selector)
    decision = retrospective_variant_decision(selector["analysis"], card)
    test_records = validate_test_metadata_records(
        load_strict_jsonl(Path(p3c_root) / TEST_METADATA_NAME), card
    )
    test_manifest = load_strict_json(Path(p3c_root) / TEST_MANIFEST_NAME)
    verify_manifest_sha256(test_manifest)
    allowlist = load_strict_json(Path(p3c_root) / HISTORICAL_ALLOWLIST_NAME)
    verify_manifest_sha256(allowlist)
    _validate_allowlist_for_c2(allowlist)

    for name in (RETROSPECTIVE_NAME, MISSING12_PREDICTIONS_NAME, RETROSPECTIVE_VERIFIER_NAME):
        if (Path(p3c_root) / name).exists():
            raise FileExistsError("refusing to replace C2 artifact: %s" % name)

    entries = {entry["cell"]: entry for entry in allowlist["cells"]}
    baseline_map, baseline_summary = _load_outcome_cell(
        entries["baseline"], test_records, test_manifest
    )
    historical_summaries = {}
    paired = {}
    bootstrap_replicates = int(card["bootstrap"]["outcome_primary"]["replicates"])
    bootstrap_seed = int(card["bootstrap"]["outcome_primary"]["seed"])
    for window in HISTORICAL_WINDOWS:
        candidate_map, summary = _load_outcome_cell(entries[window], test_records, test_manifest)
        comparison = exact_paired_comparison(
            baseline_map,
            candidate_map,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        summary["delta_accuracy_pp_direct"] = (
            summary["accuracy_fraction"] - baseline_summary["accuracy_fraction"]
        ) * 100.0
        historical_summaries[window] = summary
        paired[window] = comparison
        del candidate_map
        gc.collect()

    ordered_outcomes = sorted(
        HISTORICAL_WINDOWS,
        key=lambda window: (-historical_summaries[window]["accuracy_fraction"], _window_start(window)),
    )
    best_accuracy = historical_summaries[ordered_outcomes[0]]["accuracy_fraction"]
    tied_best = [
        window
        for window in HISTORICAL_WINDOWS
        if math.isclose(
            historical_summaries[window]["accuracy_fraction"],
            best_accuracy,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ]
    if tied_best != ["12:15"]:
        raise P3CError("live historical-13 point oracle is not uniquely 12:15")

    predictions = make_missing12_predictions(selector["analysis"], decision, card)
    historical_alignment = make_historical_alignment(
        selector["analysis"], historical_summaries, paired
    )
    adjudication: Dict[str, Any] = {
        "schema_version": RETROSPECTIVE_SCHEMA,
        "artifact_role": "p3c_historical13_retrospective_adjudication",
        "gate": GATE,
        "stage": "C2",
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "argv": list(argv),
        "implementation_sha256": implementation_hashes(),
        "c0_receipt_internal_sha256": c0["manifest_sha256"],
        "c1_freeze": {
            "selector_file_sha256": c1_verification["selector_freeze_file_sha256"],
            "selector_internal_sha256": c1_verification[
                "selector_freeze_internal_sha256"
            ],
            "receipt_file_sha256": c1_verification["receipt_file_sha256"],
            "receipt_internal_sha256": c1_verification["receipt_internal_sha256"],
            "independently_recomputed_before_unseal": True,
        },
        "phase1_source_allowlist": {
            "file_sha256": file_sha256(Path(p3c_root) / HISTORICAL_ALLOWLIST_NAME),
            "internal_sha256": allowlist["manifest_sha256"],
        },
        "test_identity": {
            "record_count": len(test_records),
            "ordered_identity_sha256": test_manifest["ordered_identity_sha256"],
            "exact_closure_required_per_cell": True,
            "intersection_fallback_used": False,
        },
        "baseline": baseline_summary,
        "historical_outcomes": historical_summaries,
        "paired_comparisons": paired,
        "historical_alignment": historical_alignment,
        "historical_point_oracle": {
            "window": "12:15",
            "unique": True,
            "accuracy_fraction": best_accuracy,
            "ranked_windows": ordered_outcomes,
        },
        "retrospective_selection": decision,
        "science_label": decision["science_label"],
        "selected_variant": decision["selected_variant"],
        "process_isolation": {
            "c1_verified_before_outcome_open": True,
            "baseline_and_historical13_only": True,
            "selection_report_loaded": False,
            "prior_paired_summary_loaded": False,
            "baseline_scores_recomputed_after_unseal": False,
            "blind12_value_roots_loaded": False,
        },
        "p3d_authorized": False,
        "stop_h3": decision["stop_h3"],
    }
    attach_manifest_sha256(adjudication)
    predictions["retrospective_adjudication_internal_sha256"] = adjudication[
        "manifest_sha256"
    ]
    attach_manifest_sha256(predictions)
    adjudication_file_sha = write_new_json(
        Path(p3c_root) / RETROSPECTIVE_NAME, adjudication
    )
    predictions_file_sha = write_new_json(
        Path(p3c_root) / MISSING12_PREDICTIONS_NAME, predictions
    )
    # Re-open the just-written files so the verifier operates on serialized bytes.
    serialized_adjudication = load_strict_json(Path(p3c_root) / RETROSPECTIVE_NAME)
    serialized_predictions = load_strict_json(Path(p3c_root) / MISSING12_PREDICTIONS_NAME)
    verification = verify_retrospective_outputs(
        serialized_adjudication, serialized_predictions, selector, card
    )
    verifier: Dict[str, Any] = {
        "schema_version": RETROSPECTIVE_VERIFIER_SCHEMA,
        "artifact_role": "p3c_retrospective_adjudication_verifier_receipt",
        "gate": GATE,
        "stage": "C2",
        "created_at_utc": utc_now(),
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "retrospective_file_sha256": adjudication_file_sha,
        "retrospective_internal_sha256": adjudication["manifest_sha256"],
        "missing12_predictions_file_sha256": predictions_file_sha,
        "missing12_predictions_internal_sha256": predictions["manifest_sha256"],
        "verification": verification,
        "implementation_sha256": implementation_hashes(),
        "blind12_outcome_values_consumed": False,
        "status": "C2_RETROSPECTIVE_ADJUDICATION_PASS",
    }
    attach_manifest_sha256(verifier)
    write_new_json(Path(p3c_root) / RETROSPECTIVE_VERIFIER_NAME, verifier)
    return verifier


def make_missing12_predictions(
    analysis: Mapping[str, Any],
    decision: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    selected = decision["selected_variant"]
    if selected is None:
        ranking: List[Dict[str, Any]] = []
        top1 = None
        top3: List[str] = []
    else:
        scope = analysis["variants"][selected]["scopes"]["blind_enrichment"]
        ranking = [dict(row) for row in scope["published_ranking"]]
        expected_blind = {
            start for start in partition_starts(card)["blind_12"] if start in analysis["variants"][selected]["support_starts"]
        }
        if {row["start"] for row in ranking} != expected_blind:
            raise P3CError("selected variant blind ranking is not an exact support closure")
        top1 = ranking[0]["window"] if ranking else None
        top3 = [row["window"] for row in ranking[:3]]
    payload: Dict[str, Any] = {
        "schema_version": MISSING12_SCHEMA,
        "artifact_role": "p3c_frozen_missing12_predictions",
        "gate": GATE,
        "stage": "C2",
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "science_label": decision["science_label"],
        "selected_variant": selected,
        "blind12_universe": list(BLIND_WINDOWS),
        "scorable_ranking": ranking,
        "frozen_top1": top1,
        "frozen_top3": top3,
        "prediction_status": "FROZEN" if selected is not None else "NO_VARIANT_STOP_H3",
        "blind12_outcome_values_consumed": False,
        "p3d_authorized": False,
        "retrospective_adjudication_internal_sha256": None,
    }
    return payload


def verify_retrospective_outputs(
    adjudication: Mapping[str, Any],
    predictions: Mapping[str, Any],
    selector: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    verify_manifest_sha256(adjudication)
    verify_manifest_sha256(predictions)
    verify_manifest_sha256(selector)
    expected_decision = retrospective_variant_decision(selector["analysis"], card)
    if adjudication.get("retrospective_selection") != expected_decision:
        raise P3CError("retrospective selection differs from the frozen C1 ranking")
    if (
        adjudication.get("science_label") != expected_decision["science_label"]
        or adjudication.get("selected_variant") != expected_decision["selected_variant"]
        or adjudication.get("stop_h3") != expected_decision["stop_h3"]
    ):
        raise P3CError("retrospective top-level decision fields differ")
    expected_predictions = make_missing12_predictions(
        selector["analysis"], expected_decision, card
    )
    expected_predictions["retrospective_adjudication_internal_sha256"] = adjudication[
        "manifest_sha256"
    ]
    attach_manifest_sha256(expected_predictions)
    if canonical_json_bytes(expected_predictions) != canonical_json_bytes(predictions):
        raise P3CError("missing-12 predictions differ from the frozen variant ranking")
    outcomes = adjudication.get("historical_outcomes")
    paired = adjudication.get("paired_comparisons")
    if not isinstance(outcomes, Mapping) or set(outcomes) != set(HISTORICAL_WINDOWS):
        raise P3CError("historical outcome set differs from exact historical-13")
    if not isinstance(paired, Mapping) or set(paired) != set(HISTORICAL_WINDOWS):
        raise P3CError("paired comparison set differs from exact historical-13")
    baseline = adjudication.get("baseline")
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("sample_count") != 14042
        or baseline.get("phase1_ordered_sample_identity_sha256")
        != EXPECTED_TEST_ORDERED_IDENTITY_SHA256
        or baseline.get("model_revision") != MODEL_REVISION
        or baseline.get("k") != 2
        or baseline.get("fixed_horizon") is not True
    ):
        raise P3CError("retrospective baseline exact closure differs")
    for window in HISTORICAL_WINDOWS:
        alignment = paired[window].get("sample_alignment", {})
        if alignment.get("exact_match") is not True or alignment.get("matched_count") != 14042:
            raise P3CError("historical paired outcome lacks exact 14,042 closure")
        summary = outcomes[window]
        if (
            summary.get("sample_count") != 14042
            or summary.get("phase1_ordered_sample_identity_sha256")
            != EXPECTED_TEST_ORDERED_IDENTITY_SHA256
            or summary.get("model_revision") != MODEL_REVISION
            or summary.get("k") != 2
            or summary.get("fixed_horizon") is not True
            or summary.get("identity_intersection_fallback_used") is not False
        ):
            raise P3CError("historical outcome cell exact closure differs")
        counts = paired[window].get("counts")
        if not isinstance(counts, Mapping) or sum(counts.values()) != 14042:
            raise P3CError("historical paired contingency table does not close 14,042")
        expected_delta = (
            int(counts["window_only"]) - int(counts["baseline_only"])
        ) * 100.0 / 14042.0
        if not math.isclose(
            float(summary.get("delta_accuracy_pp_direct")),
            expected_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise P3CError("historical direct delta differs from paired counts")
        bootstrap = paired[window].get("paired_bootstrap")
        if (
            not isinstance(bootstrap, Mapping)
            or bootstrap.get("replicates")
            != int(card["bootstrap"]["outcome_primary"]["replicates"])
            or bootstrap.get("seed") != int(card["bootstrap"]["outcome_primary"]["seed"])
        ):
            raise P3CError("historical paired bootstrap contract differs")
    expected_alignment = make_historical_alignment(selector["analysis"], outcomes, paired)
    if canonical_json_bytes(adjudication.get("historical_alignment")) != canonical_json_bytes(
        expected_alignment
    ):
        raise P3CError("historical outcome alignment differs from frozen selector ranks")
    ranked_outcomes = sorted(
        HISTORICAL_WINDOWS,
        key=lambda window: (-float(outcomes[window]["accuracy_fraction"]), _window_start(window)),
    )
    best_accuracy = float(outcomes[ranked_outcomes[0]]["accuracy_fraction"])
    tied_best = [
        window
        for window in HISTORICAL_WINDOWS
        if math.isclose(
            float(outcomes[window]["accuracy_fraction"]),
            best_accuracy,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ]
    oracle = adjudication.get("historical_point_oracle")
    if tied_best != ["12:15"] or oracle != {
        "window": "12:15",
        "unique": True,
        "accuracy_fraction": best_accuracy,
        "ranked_windows": ranked_outcomes,
    }:
        raise P3CError("historical point oracle closure differs")
    isolation = adjudication.get("process_isolation")
    if not isinstance(isolation, Mapping) or any(
        isolation.get(key) is not False
        for key in (
            "selection_report_loaded",
            "prior_paired_summary_loaded",
            "baseline_scores_recomputed_after_unseal",
            "blind12_value_roots_loaded",
        )
    ) or isolation.get("c1_verified_before_outcome_open") is not True or isolation.get(
        "baseline_and_historical13_only"
    ) is not True:
        raise P3CError("C2 process-isolation proof differs")
    return {
        "c1_rank_reused_without_recomputation": True,
        "variant_priority_exact": True,
        "selection_frequency_boundary_exact": True,
        "false_positive_rules_exact": True,
        "historical13_exact": True,
        "paired_identity_14042_exact": True,
        "missing12_predictions_exact": True,
        "blind12_outcome_values_consumed": False,
        "science_label": adjudication["science_label"],
        "selected_variant": adjudication["selected_variant"],
    }


def _load_outcome_cell(
    entry: Mapping[str, Any],
    test_records: Sequence[Mapping[str, Any]],
    test_manifest: Mapping[str, Any],
) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    cell = str(entry.get("cell"))
    results_meta = entry.get("results")
    if not isinstance(results_meta, Mapping):
        raise P3CError("historical allowlist lacks results metadata")
    results_path = Path(str(results_meta.get("path", ""))).resolve()
    expected_root = Path(str(entry.get("output_dir", ""))).resolve()
    if results_path != expected_root / "results.json":
        raise P3CError("historical results path differs from allowlist output_dir")
    if file_sha256(results_path) != results_meta.get("file_sha256"):
        raise P3CError("historical results file hash changed after C0")
    actual_sidecars = sorted(expected_root.glob("samples_*.jsonl"), key=lambda path: path.name)
    expected_sidecars = entry.get("sample_sidecars")
    if not isinstance(expected_sidecars, list):
        raise P3CError("historical sidecar allowlist is invalid")
    if [str(path.resolve()) for path in actual_sidecars] != [
        value.get("path") for value in expected_sidecars
    ]:
        raise P3CError("historical sample sidecar set changed after C0")
    for path, expected in zip(actual_sidecars, expected_sidecars):
        if file_sha256(path) != expected.get("file_sha256"):
            raise P3CError("historical sample sidecar hash changed after C0")

    payload = load_strict_json(results_path)
    legacy_identity = _phase1_ordered_sample_identity(payload)
    if legacy_identity["ordered_identity_sha256"] != EXPECTED_TEST_ORDERED_IDENTITY_SHA256:
        raise P3CError("historical Phase 1 ordered sample identity SHA256 differs")
    try:
        raw_correctness = extract_correctness_samples(payload, task="mmlu", metric="acc,none")
        aggregate_accuracy = extract_accuracy(payload, task="mmlu", metric="acc,none")
    except AnalysisError as exc:
        raise P3CError("historical result cannot be parsed by Phase 1 helpers") from exc
    expected_by_pair_id = _expected_pair_id_map(test_records)
    if set(raw_correctness) != set(expected_by_pair_id):
        raise P3CError("historical cell does not exactly contain the frozen test identities")
    _verify_result_sample_content(payload, expected_by_pair_id)
    correctness: Dict[str, bool] = {}
    for record in test_records:
        pair_id = _pair_id_from_test_record(record)
        identity_key = canonical_json_bytes(record["identity"]).decode("utf-8")
        correctness[identity_key] = bool(raw_correctness[pair_id])
    if len(correctness) != len(test_records):
        raise P3CError("historical canonical identity map contains duplicates")
    sample_accuracy = sum(correctness.values()) / float(len(correctness))
    if not math.isclose(sample_accuracy, aggregate_accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise P3CError("historical aggregate accuracy differs from exact sample correctness")
    summary = {
        "cell": cell,
        "window": entry.get("window"),
        "output_dir": str(expected_root),
        "results_file_sha256": results_meta["file_sha256"],
        "sample_sidecar_file_sha256": {
            Path(value["path"]).name: value["file_sha256"] for value in expected_sidecars
        },
        "sample_count": len(correctness),
        "ordered_identity_sha256": test_manifest["ordered_identity_sha256"],
        "phase1_ordered_sample_identity_sha256": legacy_identity[
            "ordered_identity_sha256"
        ],
        "accuracy_fraction": sample_accuracy,
        "identity_intersection_fallback_used": False,
        "model_revision": MODEL_REVISION,
        "k": 2,
        "fixed_horizon": True,
    }
    del payload
    gc.collect()
    return correctness, summary


def _verify_result_sample_content(
    payload: Mapping[str, Any],
    expected_by_pair_id: Mapping[str, Mapping[str, Any]],
) -> None:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, Mapping):
        raise P3CError("historical results lacks inline task samples")
    seen = set()
    for namespace, rows in raw_samples.items():
        task = str(namespace)
        if not (task == "mmlu" or task.startswith("mmlu_")):
            continue
        if not isinstance(rows, list):
            raise P3CError("historical sample task payload must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise P3CError("historical sample row must be an object")
            raw_id = _raw_sample_id(row)
            pair_id = "%s:%s" % (task, raw_id)
            expected = expected_by_pair_id.get(pair_id)
            if expected is None or pair_id in seen:
                raise P3CError("historical sample identity is missing/extra/duplicate")
            seen.add(pair_id)
            doc = row.get("doc")
            if not isinstance(doc, Mapping):
                raise P3CError("historical sample lacks its exact source doc")
            safe = safe_test_dataset_fields(doc, expected["subject"])
            if safe_test_doc_sha256(safe) != expected["identity"]["doc_hash"]:
                raise P3CError("historical sample source doc identity hash differs from C0")
            if sanitized_content_sha256(safe["question"], safe["choices"]) != expected[
                "sanitized_content_sha256"
            ]:
                raise P3CError("historical sample sanitized content hash differs from C0")
    if seen != set(expected_by_pair_id):
        raise P3CError("historical sample content scan did not close all test identities")


def _phase1_ordered_sample_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Reproduce the frozen Phase 1 legacy [(doc_id, doc_hash)] digest."""

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, Mapping):
        raise P3CError("historical results lacks inline task samples")
    task_names = sorted(
        str(name)
        for name in raw_samples
        if str(name) == "mmlu" or str(name).startswith("mmlu_")
    )
    identities = []
    for task in task_names:
        rows = raw_samples[task]
        if not isinstance(rows, list):
            raise P3CError("historical sample task payload must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise P3CError("historical sample row must be an object")
            doc_id = str(row.get("doc_id", ""))
            if not doc_id.isdigit() or str(int(doc_id)) != doc_id:
                raise P3CError("historical Phase 1 doc_id is not canonical decimal")
            doc_hash = _sha256(row.get("doc_hash"), "historical Phase 1 doc_hash")
            identities.append((doc_id, doc_hash))
    if len(task_names) != 57 or len(identities) != 14042:
        raise P3CError("historical Phase 1 identity does not close 57/14,042")
    digest = hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "task_count": len(task_names),
        "sample_count": len(identities),
        "ordered_identity_sha256": digest,
    }


def _expected_pair_id_map(
    test_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    result = {}
    for record in test_records:
        pair_id = _pair_id_from_test_record(record)
        if pair_id in result:
            raise P3CError("test metadata contains duplicate Phase 1 pair ids")
        result[pair_id] = record
    return result


def _pair_id_from_test_record(record: Mapping[str, Any]) -> str:
    identity = canonical_identity(record["identity"])
    raw_id = identity["doc_id"]
    if not raw_id.isdigit() or str(int(raw_id)) != raw_id:
        raise P3CError("test identity row index is not canonical decimal")
    return "%s:%s" % (identity["task"], raw_id)


def _validate_allowlist_for_c2(allowlist: Mapping[str, Any]) -> None:
    if allowlist.get("schema_version") != HISTORICAL_ALLOWLIST_SCHEMA:
        raise P3CError("historical allowlist schema differs")
    cells = allowlist.get("cells")
    if not isinstance(cells, list) or [cell.get("cell") for cell in cells] != [
        "baseline"
    ] + list(HISTORICAL_WINDOWS):
        raise P3CError("historical allowlist is not exact ordered baseline+13")
    if allowlist.get("results_values_read") is not False:
        raise P3CError("C0 allowlist reports premature outcome access")


def _raw_sample_id(item: Mapping[str, Any]) -> str:
    for key in ("sample_id", "id", "doc_id"):
        if item.get(key) is not None:
            return str(item[key])
    doc = item.get("doc")
    if isinstance(doc, Mapping):
        for key in ("id", "doc_id"):
            if doc.get(key) is not None:
                return str(doc[key])
    raise P3CError("historical sample has no stable id")


def _file_metadata(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "file_sha256": file_sha256(resolved),
    }


def _window_start(window: str) -> int:
    return int(str(window).split(":", 1)[0])


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root)] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise P3CError("git provenance command failed: %s" % " ".join(args))
    return completed.stdout.strip()


def _reject_json_constant(value: str) -> Any:
    raise ValueError("non-finite JSON constant is forbidden: %s" % value)


def _exact_keys(value: Any, expected: Iterable[str], context: str) -> None:
    if not isinstance(value, Mapping):
        raise P3CError("%s must be an object" % context)
    actual = set(value)
    frozen = set(expected)
    if actual != frozen:
        raise P3CError(
            "%s fields differ; missing=%s extra=%s"
            % (context, sorted(frozen - actual), sorted(actual - frozen))
        )


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P3CError("%s must be a non-empty string" % context)
    return value.strip()


def _sha256(value: Any, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise P3CError("%s must be a lowercase SHA256" % context)
    return text


__all__ = [
    "BLIND_WINDOWS",
    "HISTORICAL_WINDOWS",
    "P3CError",
    "build_historical_source_allowlist",
    "build_safe_test_metadata",
    "exact_paired_comparison",
    "make_historical_alignment",
    "make_missing12_predictions",
    "make_test_metadata_manifest",
    "make_test_metadata_record",
    "materialize_c0",
    "materialize_c1",
    "materialize_c2",
    "retrospective_variant_decision",
    "safe_test_dataset_fields",
    "safe_test_doc_sha256",
    "scan_blind_outcome_existence",
    "validate_frozen_analysis_contract",
    "validate_test_metadata_records",
    "verify_c1_freeze",
    "verify_retrospective_outputs",
]
