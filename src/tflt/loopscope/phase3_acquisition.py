"""Live, outcome-blind acquisition for LoopScope Phase 3 Gate P3-B.

This module is intentionally narrow.  It materializes the validation-1531
gold-free pool from the planning-signed renderer anchor, runs one ordinary
native no-loop forward per selected identity, and closes write-once shard and
trajectory artifacts with the P3-A validators.  Heavy dependencies are loaded
only inside the remote acquisition functions.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.mmlu_renderer import (
    LM_EVAL_VERSION,
    PROJECTION_RECORD_VERSION,
    canonical_json_bytes as renderer_canonical_json_bytes,
    validate_renderer_manifest_payload,
)
from tflt.loopscope.phase3_analysis import trajectory_record_from_logits
from tflt.loopscope.phase3_pool import (
    make_pool_manifest,
    make_source_manifest,
    make_trajectory_manifest,
    validate_pool_manifest,
    validate_pool_records,
    validate_source_manifest,
    validate_trajectory_manifest,
    validate_trajectory_records,
)
from tflt.loopscope.phase3_schema import (
    BOUNDARY_IDS,
    PHASE3_CARD_BYTE_SHA256,
    PHASE3_POOL_RECORD_SCHEMA_VERSION,
    attach_manifest_sha256,
    canonical_json_bytes,
    canonical_record_key,
    file_sha256,
    identity_tuple,
    load_phase3_card,
    reject_forbidden_selector_fields,
    sanitized_content_sha256,
    validate_pool_record,
    validate_trajectory_record,
    verify_manifest_sha256,
)
from tflt.loopscope.schema import SchemaError, ensure_new_directory


GATE = "P3-B"
EXECUTOR_THREAD_ID = "019f67e9-4e17-7b53-a1db-b0d673a15bff"
PLANNING_THREAD_ID = "019f670d-24ca-7ad0-b92e-64438aa04ef1"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3b-20260715T223304Z"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
MODEL_SNAPSHOT = Path(
    "/hpc2hdd/home/xhuang225/shared/hf_home/hub/"
    "models--Qwen--Qwen3-1.7B-Base/snapshots/"
    "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
)
ANCHOR_PROJECTION = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/inputs/"
    "loopscope-qwen17-mmlu-phase1-20260711-043615/lm_eval_mmlu_projection.jsonl"
)
ANCHOR_RENDERER_MANIFEST = ANCHOR_PROJECTION.with_name(
    "lm_eval_mmlu_renderer_manifest.json"
)

MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
ANCHOR_PROJECTION_SHA256 = (
    "439e41113458ccfbd51d211ab7fae4e91ecc02723c7b14ad6c1969803aa61a6a"
)
ANCHOR_MANIFEST_FILE_SHA256 = (
    "48ba67c204249b5f2f9a6cf711f8678711b5e606398d35ba4d919d4eee70e6e7"
)
ANCHOR_MANIFEST_INTERNAL_SHA256 = (
    "e1baaa92c8ab3804c75a5e415a324d00fed0277d17c26b16d12b8c99aafbe949"
)
RENDERER_SOURCE_SHA256 = (
    "105ff765d068bd246387249d3941a697c964070834d92d7a3268056b7a68a8a9"
)
TEMPLATE_SHA256 = (
    "15dea12df4d7dc69b4d9d425a51bc14976f560b3e5ed52a319895cde79f2dabb"
)
RENDER_CONTRACT_SHA256 = (
    "eff9612c2984c7f2668e5cfb11d9eec89a5468c6f227ff94e4a226843043d6f4"
)
DATASET_FINGERPRINT_SHA256 = (
    "c3f97b35254b7953d20cffc65414e1c78a792128f215805962c0db5d4d185e11"
)

SOURCE_MANIFEST_NAME = "source_manifest.json"
POOL_JSONL_NAME = "validation1531_pool.jsonl"
POOL_MANIFEST_NAME = "validation1531_pool_manifest.json"
SOURCE_RECEIPT_NAME = "validation1531_renderer_source_receipt.json"
FORBIDDEN_SCAN_NAME = "validation1531_forbidden_field_scan_receipt.json"
SMOKE_REPORT_NAME = "validation1531_no_loop_smoke_report.json"
SHARD_MANIFEST_NAME = "validation1531_shard_manifest.json"
TRAJECTORY_JSONL_NAME = "validation1531_no_loop_trajectories.jsonl"
TRAJECTORY_MANIFEST_NAME = "validation1531_no_loop_trajectory_manifest.json"
MERGE_RECEIPT_NAME = "validation1531_shard_merge_receipt.json"
PROBE_REPORT_NAME = "validation1531_no_loop_probe_report.json"
VERIFIER_RECEIPT_NAME = "validation1531_no_loop_probe_verifier_receipt.json"
RESOURCE_RECEIPT_NAME = "p3b_scheduler_throttle_resource_accounting.json"

SOURCE_RECEIPT_SCHEMA = "loopscope.phase3.p3b-renderer-source-receipt.v1"
FORBIDDEN_SCAN_SCHEMA = "loopscope.phase3.p3b-forbidden-scan-receipt.v1"
MEMBERSHIP_MANIFEST_SCHEMA = "loopscope.phase3.p3b-membership-manifest.v1"
ACQUISITION_RECEIPT_SCHEMA = "loopscope.phase3.p3b-acquisition-receipt.v1"
SMOKE_REPORT_SCHEMA = "loopscope.phase3.p3b-smoke-report.v1"
MERGE_RECEIPT_SCHEMA = "loopscope.phase3.p3b-shard-merge-receipt.v1"
PROBE_REPORT_SCHEMA = "loopscope.phase3.p3b-probe-report.v1"
VERIFIER_RECEIPT_SCHEMA = "loopscope.phase3.p3b-probe-verifier-receipt.v1"
RESOURCE_RECEIPT_SCHEMA = "loopscope.phase3.p3b-resource-accounting.v1"
MONITOR_RECEIPT_SCHEMA = "loopscope.phase3.p3b-monitor-lifecycle.v1"

IMPLEMENTATION_RELATIVE_PATHS = (
    "src/tflt/loopscope/phase3_acquisition.py",
    "scripts/loopscope/run_qwen17_phase3_p3b.py",
)


class P3BAcquisitionError(ValueError):
    """Fail-closed P3-B contract violation."""


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
            raise P3BAcquisitionError("P3-B implementation path is missing: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def _implementation_hashes_at_ancestor_commit(
    scientific_artifact_commit: str, resource_accounting_commit: str
) -> Dict[str, str]:
    """Hash exact implementation blobs from a closed ancestor commit."""

    artifact_commit = _exact_commit(
        scientific_artifact_commit, "scientific artifact commit"
    )
    accounting_commit = _exact_commit(
        resource_accounting_commit, "resource accounting commit"
    )
    _require_artifact_commit_ancestor(artifact_commit, accounting_commit)
    root = repository_root()
    result = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "cat-file",
                "blob",
                "%s:%s" % (artifact_commit, relative),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise P3BAcquisitionError(
                "scientific artifact implementation blob is unavailable: %s" % relative
            )
        result[relative] = hashlib.sha256(completed.stdout).hexdigest()
    return result


def _require_smoke_producer_implementation_hashes(
    receipt: Mapping[str, Any],
    *,
    scientific_artifact_commit: Optional[str] = None,
    resource_accounting_commit: Optional[str] = None,
) -> None:
    historical = scientific_artifact_commit is not None or resource_accounting_commit is not None
    if historical:
        if scientific_artifact_commit is None or resource_accounting_commit is None:
            raise P3BAcquisitionError(
                "historical producer validation requires both closed Git commits"
            )
        expected = _implementation_hashes_at_ancestor_commit(
            scientific_artifact_commit, resource_accounting_commit
        )
    else:
        expected = implementation_hashes()
    if receipt.get("producer", {}).get("implementation_sha256") != expected:
        raise P3BAcquisitionError("B2 smoke producer implementation hashes drifted")


def git_provenance(expected_commit: Optional[str] = None, *, remote: bool = False) -> Dict[str, Any]:
    root = repository_root().resolve()

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(args), text=True
        ).strip()

    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    commit = run("rev-parse", "HEAD")
    origin = run("rev-parse", "origin/loopscope")
    dirty_output = run("status", "--porcelain", "--untracked-files=all")
    payload = {
        "root": str(root),
        "branch": branch,
        "commit": commit,
        "origin_loopscope": origin,
        "dirty": bool(dirty_output),
    }
    if branch != "loopscope":
        raise P3BAcquisitionError("P3-B requires branch=loopscope")
    if dirty_output:
        raise P3BAcquisitionError("P3-B remote producer requires a clean checkout")
    if expected_commit is not None:
        expected = _exact_commit(expected_commit, "expected Git commit")
        if commit != expected or origin != expected:
            raise P3BAcquisitionError(
                "local/origin Git commit differs from the authorized implementation commit"
            )
    if remote and root != REMOTE_REPO:
        raise P3BAcquisitionError("P3-B producer is outside the dedicated remote clone")
    return payload


def assert_offline_environment() -> None:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise P3BAcquisitionError("%s must equal 1" % key)
    if os.environ.get("HF_ENDPOINT"):
        raise P3BAcquisitionError("HF_ENDPOINT must be unset for P3-B")
    expected = {
        "HF_HOME": "/hpc2hdd/home/xhuang225/shared/hf_home",
        "HF_DATASETS_CACHE": "/hpc2hdd/home/xhuang225/shared/datasets",
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise P3BAcquisitionError("%s differs from the authorized cache path" % key)


def assert_authorized_run_root(path: Path, *, must_exist: Optional[bool] = None) -> Path:
    resolved = Path(path).resolve()
    if resolved != AUTHORIZED_RUN_ROOT:
        raise P3BAcquisitionError("run root differs from the exact P3-B authorization")
    if must_exist is True and not resolved.is_dir():
        raise P3BAcquisitionError("authorized P3-B run root does not exist")
    if must_exist is False and resolved.exists():
        raise FileExistsError("authorized P3-B run root already exists")
    return resolved


def load_strict_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise P3BAcquisitionError("cannot load strict JSON artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise P3BAcquisitionError("JSON artifact must contain an object: %s" % path)
    return value


def load_strict_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise P3BAcquisitionError(
                        "blank JSONL line is forbidden at %s:%d" % (path, line_number)
                    )
                value = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(value, dict):
                    raise P3BAcquisitionError(
                        "JSONL record must be an object at %s:%d" % (path, line_number)
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, P3BAcquisitionError):
            raise
        raise P3BAcquisitionError("cannot load strict JSONL artifact: %s" % path) from exc
    if not records:
        raise P3BAcquisitionError("JSONL artifact contains no records: %s" % path)
    return records


def write_new_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        raise P3BAcquisitionError("refusing to write an empty JSONL artifact")
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


def write_new_json(path: Path, payload: Any) -> str:
    """Write one readable JSON object with an atomic exclusive-create open."""

    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def write_new_text(path: Path, text: str, *, executable: bool = False) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        path.chmod(0o755)
    return file_sha256(path)


def load_verified_renderer_anchor(
    projection_path: Path, manifest_path: Path, card: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    projection_path = Path(projection_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if projection_path != ANCHOR_PROJECTION or manifest_path != ANCHOR_RENDERER_MANIFEST:
        raise P3BAcquisitionError("renderer anchor path differs from the planning handoff")
    if file_sha256(projection_path) != ANCHOR_PROJECTION_SHA256:
        raise P3BAcquisitionError("renderer projection file SHA256 mismatch")
    if file_sha256(manifest_path) != ANCHOR_MANIFEST_FILE_SHA256:
        raise P3BAcquisitionError("renderer manifest file SHA256 mismatch")
    manifest = load_strict_json(manifest_path)
    validate_renderer_manifest_payload(manifest)
    if manifest.get("manifest_sha256") != ANCHOR_MANIFEST_INTERNAL_SHA256:
        raise P3BAcquisitionError("renderer internal manifest SHA256 mismatch")
    renderer = manifest.get("renderer")
    if not isinstance(renderer, Mapping):
        raise P3BAcquisitionError("renderer manifest lacks renderer evidence")
    exact = {
        "renderer_source_sha256": RENDERER_SOURCE_SHA256,
        "source_files_sha256": RENDERER_SOURCE_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "task_configs_sha256": TEMPLATE_SHA256,
        "render_contract_sha256": RENDER_CONTRACT_SHA256,
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT_SHA256,
        "source_projection_sha256": ANCHOR_PROJECTION_SHA256,
        "dataset_revision": DATASET_REVISION,
        "lm_eval_version": LM_EVAL_VERSION,
    }
    for key, expected in exact.items():
        if renderer.get(key) != expected:
            raise P3BAcquisitionError("renderer anchor %s mismatch" % key)
    if manifest.get("target_split") != card["task"]["trajectory_split"]:
        raise P3BAcquisitionError("renderer anchor split differs from validation")
    if manifest.get("fewshot_split") != card["task"]["fewshot_split"]:
        raise P3BAcquisitionError("renderer anchor few-shot split mismatch")
    if manifest.get("num_fewshot") != card["task"]["num_fewshot"]:
        raise P3BAcquisitionError("renderer anchor few-shot count mismatch")
    if manifest.get("dataset", {}).get("revision") != DATASET_REVISION:
        raise P3BAcquisitionError("renderer dataset revision mismatch")

    records = load_strict_jsonl(projection_path)
    projection = manifest.get("projection")
    if not isinstance(projection, Mapping):
        raise P3BAcquisitionError("renderer projection manifest is missing")
    expected_count = int(card["task"]["trajectory_sample_count"])
    if len(records) != expected_count or projection.get("count") != expected_count:
        raise P3BAcquisitionError("renderer anchor must contain exactly 1,531 records")
    ids = []
    hashes = []
    seen_join_keys = set()
    required = (
        "schema_version",
        "id",
        "task_name",
        "target_doc_id",
        "target_doc_index",
        "target_doc_sha256",
        "subject",
        "split",
        "text",
        "render_sha256",
        "projection_record_sha256",
        "uses_target_gold_labels",
    )
    for index, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            raise P3BAcquisitionError(
                "renderer anchor record %d misses %s" % (index, ",".join(missing))
            )
        if record["schema_version"] != PROJECTION_RECORD_VERSION:
            raise P3BAcquisitionError("renderer anchor record schema mismatch")
        if record["split"] != "validation" or record["uses_target_gold_labels"] is not False:
            raise P3BAcquisitionError("renderer anchor target boundary is invalid")
        prompt = str(record["text"])
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if not prompt.rstrip().endswith("Answer:") or record["render_sha256"] != prompt_hash:
            raise P3BAcquisitionError("renderer anchor prompt closure mismatch")
        body = {key: value for key, value in record.items() if key != "projection_record_sha256"}
        record_hash = hashlib.sha256(renderer_canonical_json_bytes(body)).hexdigest()
        if record["projection_record_sha256"] != record_hash:
            raise P3BAcquisitionError("renderer anchor record self-hash mismatch")
        join_key = (
            str(record["task_name"]),
            str(record["target_doc_id"]),
            str(record["target_doc_sha256"]),
        )
        if join_key in seen_join_keys:
            raise P3BAcquisitionError("renderer anchor contains duplicate join keys")
        seen_join_keys.add(join_key)
        ids.append(str(record["id"]))
        hashes.append(str(record["projection_record_sha256"]))
    if ids != projection.get("record_ids") or hashes != projection.get("record_hashes"):
        raise P3BAcquisitionError("renderer anchor record arrays differ from its manifest")
    return records, manifest


def safe_dataset_fields(row: Mapping[str, Any], expected_subject: str) -> Dict[str, Any]:
    """Read only question, choices, and subject from one materialized dataset row."""

    try:
        question = row["question"]
        choices = row["choices"]
        subject = row["subject"]
    except (KeyError, TypeError) as exc:
        raise P3BAcquisitionError("MMLU safe projection fields are missing") from exc
    if not isinstance(question, str):
        raise P3BAcquisitionError("MMLU question must be a string")
    if not isinstance(choices, (list, tuple)) or len(choices) != 4:
        raise P3BAcquisitionError("MMLU choices must contain exactly four strings")
    if any(not isinstance(value, str) for value in choices):
        raise P3BAcquisitionError("MMLU choices must all be strings")
    if str(subject) != str(expected_subject):
        raise P3BAcquisitionError("MMLU row subject differs from the frozen task subject")
    return {
        "question": question,
        "choices": list(choices),
        "subject": str(subject),
    }


def safe_target_doc_sha256(safe_row: Mapping[str, Any]) -> str:
    payload = {
        "question": safe_row["question"],
        "subject": safe_row["subject"],
        "choices": list(safe_row["choices"]),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def pool_record_from_anchor_and_safe_row(
    anchor: Mapping[str, Any], safe_row: Mapping[str, Any], card: Mapping[str, Any]
) -> Dict[str, Any]:
    question = str(safe_row["question"])
    choices = [str(value) for value in safe_row["choices"]]
    prompt = str(anchor["text"])
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    record = {
        "schema_version": PHASE3_POOL_RECORD_SCHEMA_VERSION,
        "identity": {
            "task": str(anchor["task_name"]),
            "doc_id": str(anchor["target_doc_id"]),
            "doc_hash": str(anchor["target_doc_sha256"]),
        },
        "subject": str(anchor["subject"]),
        "split": "validation",
        "question": question,
        "ordered_choices": choices,
        "rendered_prompt": prompt,
        "prompt_sha256": prompt_hash,
        "sanitized_content_sha256": sanitized_content_sha256(question, choices),
        "renderer_provenance": {
            "dataset_repo": card["task"]["dataset"],
            "dataset_revision": card["task"]["dataset_revision"],
            "task_group": card["task"]["task_group"],
            "fewshot_split": card["task"]["fewshot_split"],
            "num_fewshot": card["task"]["num_fewshot"],
            "renderer_id": (
                "lm_eval.api.task.ConfigurableTask.fewshot_context@lm_eval==%s"
                % LM_EVAL_VERSION
            ),
            "renderer_source_sha256": RENDERER_SOURCE_SHA256,
            "render_contract_sha256": RENDER_CONTRACT_SHA256,
            "source_projection_sha256": ANCHOR_PROJECTION_SHA256,
            "render_sha256": prompt_hash,
        },
    }
    return validate_pool_record(record, card)


def _default_dataset_loader(
    task_evidence: Mapping[str, Any], card: Mapping[str, Any]
) -> Tuple[Sequence[Mapping[str, Any]], Dict[str, Any]]:
    try:
        from datasets import DownloadMode, load_dataset
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise P3BAcquisitionError("B0 requires the audited datasets runtime") from exc
    split_name = card["task"]["trajectory_split"]
    rows = load_dataset(
        path=str(task_evidence["dataset_path"]),
        name=str(task_evidence["dataset_name"] or "") or None,
        revision=card["task"]["dataset_revision"],
        split=split_name,
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    fingerprint = str(getattr(rows, "_fingerprint", ""))
    expected_fingerprint = str(
        task_evidence.get("raw_split_fingerprints", {}).get(split_name, "")
    )
    if not fingerprint or fingerprint != expected_fingerprint:
        raise P3BAcquisitionError("cached MMLU validation fingerprint mismatch")
    return rows, {
        "task_name": str(task_evidence["task_name"]),
        "dataset_name": str(task_evidence["dataset_name"]),
        "validation_fingerprint": fingerprint,
        "row_count": len(rows),
    }


def build_safe_validation_pool(
    anchor_records: Sequence[Mapping[str, Any]],
    renderer_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    dataset_loader: Optional[
        Callable[[Mapping[str, Any], Mapping[str, Any]], Tuple[Sequence[Mapping[str, Any]], Dict[str, Any]]]
    ] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    loader = dataset_loader or _default_dataset_loader
    anchor_by_key = {}
    for record in anchor_records:
        key = (
            str(record["task_name"]),
            str(record["target_doc_id"]),
            str(record["target_doc_sha256"]),
        )
        if key in anchor_by_key:
            raise P3BAcquisitionError("duplicate renderer anchor join key")
        anchor_by_key[key] = record

    tasks = renderer_manifest.get("dataset", {}).get("tasks")
    if not isinstance(tasks, list) or len(tasks) != int(card["task"]["trajectory_subject_count"]):
        raise P3BAcquisitionError("renderer dataset evidence must cover exactly 57 tasks")
    matched = set()
    pool = []
    dataset_evidence = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise P3BAcquisitionError("renderer dataset task evidence must be objects")
        task_name = str(task.get("task_name") or "")
        if not task_name.startswith("mmlu_"):
            raise P3BAcquisitionError("renderer task name is outside frozen MMLU")
        expected_subject = task_name[len("mmlu_") :]
        rows, evidence = loader(task, card)
        dataset_evidence.append(dict(evidence))
        task_anchor_count = sum(
            1 for value in anchor_records if str(value["task_name"]) == task_name
        )
        if len(rows) != task_anchor_count:
            raise P3BAcquisitionError("cached validation row count differs from renderer anchor")
        for row_index in range(len(rows)):
            safe_row = safe_dataset_fields(rows[row_index], expected_subject)
            doc_id = "%s:validation:%d" % (task_name, row_index)
            doc_hash = safe_target_doc_sha256(safe_row)
            join_key = (task_name, doc_id, doc_hash)
            anchor = anchor_by_key.get(join_key)
            if anchor is None:
                raise P3BAcquisitionError("safe validation row does not join the renderer anchor")
            if int(anchor["target_doc_index"]) != row_index:
                raise P3BAcquisitionError("renderer target index differs from safe projection")
            if str(anchor["subject"]) != expected_subject:
                raise P3BAcquisitionError("renderer subject differs from safe projection")
            pool.append(pool_record_from_anchor_and_safe_row(anchor, safe_row, card))
            matched.add(join_key)
    if matched != set(anchor_by_key):
        raise P3BAcquisitionError("safe projection does not close all renderer anchor records")
    pool.sort(key=canonical_record_key)
    if len(pool) != int(card["task"]["trajectory_sample_count"]):
        raise P3BAcquisitionError("safe pool must contain exactly 1,531 records")
    if len({record["subject"] for record in pool}) != int(card["task"]["trajectory_subject_count"]):
        raise P3BAcquisitionError("safe pool must contain exactly 57 subjects")
    return pool, sorted(dataset_evidence, key=lambda value: value["task_name"])


def materialize_b0_b1(
    *,
    card_path: Path,
    projection_path: Path,
    renderer_manifest_path: Path,
    run_root: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    run_root = assert_authorized_run_root(run_root, must_exist=False)
    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    card = load_phase3_card(card_path)
    anchor_records, renderer_manifest = load_verified_renderer_anchor(
        projection_path, renderer_manifest_path, card
    )
    pool_records, dataset_evidence = build_safe_validation_pool(
        anchor_records, renderer_manifest, card
    )
    source_manifest = make_source_manifest(pool_records, card)
    pool_manifest = make_pool_manifest(pool_records, source_manifest, card)
    validate_pool_records(pool_records, source_manifest, card)
    for record in pool_records:
        reject_forbidden_selector_fields(record, card)

    ensure_new_directory(run_root)
    (run_root / "slurm").mkdir(exist_ok=False)
    write_new_json(run_root / SOURCE_MANIFEST_NAME, source_manifest)
    pool_file_hash = write_new_jsonl(run_root / POOL_JSONL_NAME, pool_records)
    write_new_json(run_root / POOL_MANIFEST_NAME, pool_manifest)
    artifact_hashes = {
        SOURCE_MANIFEST_NAME: file_sha256(run_root / SOURCE_MANIFEST_NAME),
        POOL_JSONL_NAME: pool_file_hash,
        POOL_MANIFEST_NAME: file_sha256(run_root / POOL_MANIFEST_NAME),
    }
    receipt: Dict[str, Any] = {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "artifact_role": "validation1531_renderer_source_receipt",
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "created_at_utc": utc_now(),
        "run_root": str(run_root),
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "renderer_anchor": {
            "projection_path": str(ANCHOR_PROJECTION),
            "projection_file_sha256": ANCHOR_PROJECTION_SHA256,
            "manifest_path": str(ANCHOR_RENDERER_MANIFEST),
            "manifest_file_sha256": ANCHOR_MANIFEST_FILE_SHA256,
            "manifest_internal_sha256": ANCHOR_MANIFEST_INTERNAL_SHA256,
            "renderer_source_sha256": RENDERER_SOURCE_SHA256,
            "template_sha256": TEMPLATE_SHA256,
            "render_contract_sha256": RENDER_CONTRACT_SHA256,
            "dataset_fingerprint_sha256": DATASET_FINGERPRINT_SHA256,
        },
        "safe_dataset_projection": {
            "repo": card["task"]["dataset"],
            "revision": card["task"]["dataset_revision"],
            "split": card["task"]["trajectory_split"],
            "fields_read": ["question", "choices", "subject"],
            "target_value_read": False,
            "record_count": len(pool_records),
            "subject_count": len({record["subject"] for record in pool_records}),
            "task_evidence": dataset_evidence,
        },
        "join": {
            "keys": ["task_name", "target_doc_id", "target_doc_sha256"],
            "matched_count": len(pool_records),
            "missing_count": 0,
            "duplicate_count": 0,
            "extra_count": 0,
        },
        "artifacts": artifact_hashes,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "checks": {
            "anchor_hashes_closed": True,
            "validation1531_exact": True,
            "subjects57_exact": True,
            "canonical_order": True,
            "closed_world_pool_validated": True,
            "forbidden_selector_fields_absent": True,
            "raw_dataset_forbidden_after_receipt": True,
            "historical_anchor_forbidden_after_receipt": True,
        },
        "producer": {
            "argv": list(argv),
            "implementation_sha256": implementation_hashes(),
            "python": platform.python_version(),
        },
    }
    attach_manifest_sha256(receipt)
    write_new_json(run_root / SOURCE_RECEIPT_NAME, receipt)
    scan: Dict[str, Any] = {
        "schema_version": FORBIDDEN_SCAN_SCHEMA,
        "artifact_role": "validation1531_forbidden_field_scan_receipt",
        "created_at_utc": utc_now(),
        "run_root": str(run_root),
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "pool_file_sha256": pool_file_hash,
        "record_count": len(pool_records),
        "closed_world_validator": "tflt.loopscope.phase3_schema.validate_pool_record",
        "forbidden_field_scanner": (
            "tflt.loopscope.phase3_schema.reject_forbidden_selector_fields"
        ),
        "unknown_fields_rejected": True,
        "forbidden_fields_absent": True,
        "status": "PASS",
    }
    attach_manifest_sha256(scan)
    write_new_json(run_root / FORBIDDEN_SCAN_NAME, scan)
    validate_b0_b1_receipts(run_root, source_manifest, pool_manifest)
    return {
        "run_root": str(run_root),
        "record_count": len(pool_records),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "source_receipt_sha256": receipt["manifest_sha256"],
        "forbidden_scan_sha256": scan["manifest_sha256"],
    }


def validate_b0_b1_receipts(
    run_root: Path,
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Require complete B0 provenance and the B1 forbidden-field scan."""

    run_root = assert_authorized_run_root(run_root, must_exist=True)
    source_receipt = load_strict_json(run_root / SOURCE_RECEIPT_NAME)
    scan = load_strict_json(run_root / FORBIDDEN_SCAN_NAME)
    verify_manifest_sha256(source_receipt)
    verify_manifest_sha256(scan)
    expected_source = {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
    }
    for key, expected in expected_source.items():
        if source_receipt.get(key) != expected:
            raise P3BAcquisitionError("B0 source receipt %s mismatch" % key)
    expected_artifacts = {
        SOURCE_MANIFEST_NAME: file_sha256(run_root / SOURCE_MANIFEST_NAME),
        POOL_JSONL_NAME: file_sha256(run_root / POOL_JSONL_NAME),
        POOL_MANIFEST_NAME: file_sha256(run_root / POOL_MANIFEST_NAME),
    }
    if source_receipt.get("artifacts") != expected_artifacts:
        raise P3BAcquisitionError("B0 source receipt artifact hashes do not close")
    checks = source_receipt.get("checks")
    required_checks = (
        "anchor_hashes_closed",
        "validation1531_exact",
        "subjects57_exact",
        "canonical_order",
        "closed_world_pool_validated",
        "forbidden_selector_fields_absent",
        "raw_dataset_forbidden_after_receipt",
        "historical_anchor_forbidden_after_receipt",
    )
    if not isinstance(checks, Mapping) or any(checks.get(key) is not True for key in required_checks):
        raise P3BAcquisitionError("B0 source receipt checks are incomplete")
    expected_scan = {
        "schema_version": FORBIDDEN_SCAN_SCHEMA,
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "pool_file_sha256": expected_artifacts[POOL_JSONL_NAME],
        "record_count": int(pool_manifest["record_count"]),
        "unknown_fields_rejected": True,
        "forbidden_fields_absent": True,
        "status": "PASS",
    }
    for key, expected in expected_scan.items():
        if scan.get(key) != expected:
            raise P3BAcquisitionError("B1 forbidden-field scan %s mismatch" % key)
    return source_receipt, scan


def load_live_b1_artifacts(
    run_root: Path, card_path: Path
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    run_root = assert_authorized_run_root(run_root, must_exist=True)
    card = load_phase3_card(card_path)
    source_manifest = load_strict_json(run_root / SOURCE_MANIFEST_NAME)
    pool_records = load_strict_jsonl(run_root / POOL_JSONL_NAME)
    pool_manifest = load_strict_json(run_root / POOL_MANIFEST_NAME)
    validate_source_manifest(source_manifest, card)
    normalized_pool = validate_pool_records(pool_records, source_manifest, card)
    validate_pool_manifest(pool_manifest, source_manifest, card)
    validate_b0_b1_receipts(run_root, source_manifest, pool_manifest)
    return card, normalized_pool, source_manifest, pool_manifest


def _membership_record(record: Mapping[str, Any], ordinal: int) -> Dict[str, Any]:
    return {
        "ordinal": int(ordinal),
        "identity": dict(record["identity"]),
        "subject": str(record["subject"]),
        "split": str(record["split"]),
        "prompt_sha256": str(record["prompt_sha256"]),
    }


def _membership_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(records))).hexdigest()


def _scientific_config(card: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["revision"],
        "dtype": card["model"]["dtype"],
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "split": card["task"]["trajectory_split"],
        "forward_type": card["trajectory"]["forward_type"],
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "boundaries": list(card["trajectory"]["boundaries"]),
        "answer_position": card["trajectory"]["answer_position"],
        "projection": card["trajectory"]["projection"],
        "choice_order": list(card["trajectory"]["choice_order"]),
        "batch_size": 1,
    }


def build_smoke_manifest(
    pool_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
    git: Mapping[str, Any],
    run_root: Path,
    argv: Sequence[str],
) -> Dict[str, Any]:
    records = [_membership_record(pool_records[index], index) for index in range(4)]
    payload: Dict[str, Any] = {
        "schema_version": MEMBERSHIP_MANIFEST_SCHEMA,
        "artifact_role": "validation1531_native_no_loop_smoke_manifest",
        "mode": "smoke",
        "created_at_utc": utc_now(),
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(Path(run_root).resolve()),
        "git": dict(git),
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "renderer_manifest_sha256": ANCHOR_MANIFEST_INTERNAL_SHA256,
        "scientific_config": _scientific_config(card),
        "record_count": 4,
        "records": records,
        "membership_sha256": _membership_sha256(records),
        "output_relative": "smoke/attempt-0001",
        "argv": list(argv),
        "implementation_sha256": implementation_hashes(),
    }
    attach_manifest_sha256(payload)
    return payload


def build_shard_manifest(
    pool_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
    git: Mapping[str, Any],
    run_root: Path,
    shard_count: int,
    smoke_report_sha256: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise P3BAcquisitionError("shard_count must be an integer")
    if shard_count < 1 or shard_count > len(pool_records):
        raise P3BAcquisitionError("shard_count is outside 1..1531")
    if not re.fullmatch(r"[0-9a-f]{64}", str(smoke_report_sha256)):
        raise P3BAcquisitionError("smoke report SHA256 must be exact")
    shards = []
    for shard_id in range(shard_count):
        records = [
            _membership_record(pool_records[index], index)
            for index in range(shard_id, len(pool_records), shard_count)
        ]
        shards.append(
            {
                "shard_id": shard_id,
                "record_count": len(records),
                "records": records,
                "membership_sha256": _membership_sha256(records),
                "output_relative": "shards/shard-%04d" % shard_id,
            }
        )
    payload: Dict[str, Any] = {
        "schema_version": MEMBERSHIP_MANIFEST_SCHEMA,
        "artifact_role": "validation1531_native_no_loop_shard_manifest",
        "mode": "primary",
        "created_at_utc": utc_now(),
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(Path(run_root).resolve()),
        "git": dict(git),
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "renderer_manifest_sha256": ANCHOR_MANIFEST_INTERNAL_SHA256,
        "smoke_report_sha256": str(smoke_report_sha256),
        "scientific_config": _scientific_config(card),
        "distribution": "canonical_ordinal_round_robin",
        "shard_count": shard_count,
        "record_count": len(pool_records),
        "shards": shards,
        "ordered_pool_identity_sha256": pool_manifest["ordered_identity_sha256"],
        "argv": list(argv),
        "acquire_argv_template": [
            str(REMOTE_REPO / "scripts/loopscope/run_qwen17_phase3_p3b.py"),
            "acquire-shard",
            "--card",
            str(REMOTE_REPO / "configs/loopscope/phase3_card.json"),
            "--run-root",
            str(AUTHORIZED_RUN_ROOT),
            "--expected-commit",
            str(git["commit"]),
            "--shard-id",
            "${SLURM_ARRAY_TASK_ID}",
        ],
        "implementation_sha256": implementation_hashes(),
    }
    attach_manifest_sha256(payload)
    return payload


def validate_membership_manifest(
    manifest: Mapping[str, Any],
    pool_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> None:
    verify_manifest_sha256(manifest)
    if manifest.get("schema_version") != MEMBERSHIP_MANIFEST_SCHEMA:
        raise P3BAcquisitionError("unsupported P3-B membership manifest")
    exact = {
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "renderer_manifest_sha256": ANCHOR_MANIFEST_INTERNAL_SHA256,
        "scientific_config": _scientific_config(card),
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise P3BAcquisitionError("membership manifest %s mismatch" % key)
    mode = manifest.get("mode")
    if mode == "smoke":
        if manifest.get("record_count") != 4:
            raise P3BAcquisitionError("smoke manifest must freeze four identities")
        groups = [
            {
                "record_count": 4,
                "records": manifest.get("records"),
                "membership_sha256": manifest.get("membership_sha256"),
                "output_relative": manifest.get("output_relative"),
            }
        ]
        expected_ordinals = list(range(4))
    elif mode == "primary":
        shard_count = manifest.get("shard_count")
        if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
            raise P3BAcquisitionError("primary shard count is invalid")
        groups = manifest.get("shards")
        if not isinstance(groups, list) or len(groups) != shard_count:
            raise P3BAcquisitionError("primary shard list does not match shard_count")
        expected_ordinals = list(range(len(pool_records)))
        if manifest.get("record_count") != len(pool_records):
            raise P3BAcquisitionError("primary manifest record_count differs from pool")
        if manifest.get("distribution") != "canonical_ordinal_round_robin":
            raise P3BAcquisitionError("primary shard distribution differs")
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("smoke_report_sha256") or "")):
            raise P3BAcquisitionError("primary manifest lacks B2 smoke closure")
    else:
        raise P3BAcquisitionError("membership manifest mode must be smoke or primary")

    observed_ordinals = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise P3BAcquisitionError("membership group must be an object")
        if mode == "primary" and group.get("shard_id") != group_index:
            raise P3BAcquisitionError("shard IDs must be contiguous and ordered")
        records = group.get("records")
        if not isinstance(records, list) or group.get("record_count") != len(records):
            raise P3BAcquisitionError("membership group count mismatch")
        if group.get("membership_sha256") != _membership_sha256(records):
            raise P3BAcquisitionError("membership group SHA256 mismatch")
        relative = Path(str(group.get("output_relative") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise P3BAcquisitionError("membership output escapes run root")
        for member in records:
            if not isinstance(member, Mapping):
                raise P3BAcquisitionError("membership record must be an object")
            ordinal = member.get("ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise P3BAcquisitionError("membership ordinal must be an integer")
            if ordinal < 0 or ordinal >= len(pool_records):
                raise P3BAcquisitionError("membership ordinal is outside the pool")
            if dict(member) != _membership_record(pool_records[ordinal], ordinal):
                raise P3BAcquisitionError("membership record differs from canonical pool")
            observed_ordinals.append(ordinal)
    if sorted(observed_ordinals) != expected_ordinals or len(set(observed_ordinals)) != len(
        observed_ordinals
    ):
        raise P3BAcquisitionError("membership groups are not an exact disjoint union")


def validate_completed_smoke(
    run_root: Path,
    card: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    expected_commit: str,
    *,
    resource_accounting_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Revalidate the isolated four-record B2 gate before any B3 action."""

    run_root = assert_authorized_run_root(run_root, must_exist=True)
    manifest = load_strict_json(run_root / "smoke/smoke_manifest.json")
    validate_membership_manifest(manifest, pool, source, pool_manifest, card)
    if manifest.get("mode") != "smoke":
        raise P3BAcquisitionError("B2 manifest is not the isolated smoke manifest")
    output_dir = (run_root / str(manifest["output_relative"])).resolve()
    if run_root not in output_dir.parents:
        raise P3BAcquisitionError("B2 output escapes the authorized run root")
    records_path = output_dir / "records.jsonl"
    receipt_path = output_dir / "receipt.json"
    report_path = run_root / SMOKE_REPORT_NAME
    records = load_strict_jsonl(records_path)
    receipt = load_strict_json(receipt_path)
    report = load_strict_json(report_path)
    verify_manifest_sha256(receipt)
    verify_manifest_sha256(report)
    expected_receipt = {
        "schema_version": ACQUISITION_RECEIPT_SCHEMA,
        "mode": "smoke",
        "shard_id": None,
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "membership_manifest_sha256": manifest["manifest_sha256"],
        "membership_sha256": manifest["membership_sha256"],
        "record_count": 4,
        "record_file_sha256": file_sha256(records_path),
        "boundaries_per_record": len(BOUNDARY_IDS),
        "status": "COMPLETED",
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise P3BAcquisitionError("B2 smoke receipt %s mismatch" % key)
    receipt_git = receipt.get("git")
    if not isinstance(receipt_git, Mapping) or receipt_git.get("commit") != expected_commit:
        raise P3BAcquisitionError("B2 smoke receipt Git commit mismatch")
    if receipt_git.get("origin_loopscope") != expected_commit or receipt_git.get("dirty") is not False:
        raise P3BAcquisitionError("B2 smoke receipt Git provenance is not closed")
    if len(records) != 4:
        raise P3BAcquisitionError("B2 smoke records must contain exactly four identities")
    for record, member in zip(records, manifest["records"]):
        ordinal = int(member["ordinal"])
        validate_trajectory_record(record, card, expected_source=source["records"][ordinal])
        if identity_tuple(record["identity"]) != identity_tuple(member["identity"]):
            raise P3BAcquisitionError("B2 smoke identity differs from frozen membership")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise P3BAcquisitionError("B2 smoke runtime evidence is missing")
    expected_closure = {
        "schema_version": "loopscope.revision-closure.v1",
        "manifest_commit": MODEL_REVISION,
        "model_commit": MODEL_REVISION,
        "tokenizer_commit": MODEL_REVISION,
        "match": True,
    }
    runtime_exact = {
        "model_revision_closure": expected_closure,
        "decoder_layers": int(card["model"]["decoder_layers"]),
        "dtype": "float16",
        "device": "cuda",
        "use_cache": False,
        "logits_to_keep": 1,
        "native_logits_position": "final_pre_answer_prompt_token_only",
        "formal_forward_calls": 4,
        "loop_insertions": 0,
        "new_loop_modules_loaded_by_producer": [],
        "active_loop_wrapper_modules": [],
        "lm_head_is_model_output_embeddings": True,
        "choice_projection_kernel": "exact_selected_lm_head_weight_rows_A_B_C_D",
    }
    for key, expected in runtime_exact.items():
        if runtime.get(key) != expected:
            raise P3BAcquisitionError("B2 smoke runtime %s mismatch" % key)
    for key in ("final_norm_class", "lm_head_class"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise P3BAcquisitionError("B2 smoke runtime lacks %s closure" % key)
    mapping = runtime.get("choice_token_mapping")
    ids = runtime.get("choice_token_ids")
    if not isinstance(mapping, Mapping) or list(mapping) != ["A", "B", "C", "D"]:
        raise P3BAcquisitionError("B2 smoke choice-token mapping is not A/B/C/D")
    if not isinstance(ids, list) or len(ids) != 4 or len(set(ids)) != 4:
        raise P3BAcquisitionError("B2 smoke choice-token IDs are not four unique tokens")
    if resource_accounting_commit is None:
        _require_smoke_producer_implementation_hashes(receipt)
    else:
        _require_smoke_producer_implementation_hashes(
            receipt,
            scientific_artifact_commit=expected_commit,
            resource_accounting_commit=resource_accounting_commit,
        )
    slurm = receipt.get("slurm")
    if not isinstance(slurm, Mapping) or not re.fullmatch(
        r"[0-9]+", str(slurm.get("job_id") or "")
    ):
        raise P3BAcquisitionError("B2 smoke receipt lacks a numeric Slurm job ID")
    if slurm.get("array_job_id") is not None or slurm.get("array_task_id") is not None:
        raise P3BAcquisitionError("B2 smoke must not be an array task")
    smoke_job_id = str(slurm["job_id"])
    expected_stdout = str(run_root / ("slurm/p3b-smoke-%s.out" % smoke_job_id))
    expected_stderr = str(run_root / ("slurm/p3b-smoke-%s.err" % smoke_job_id))
    if slurm.get("stdout_path") != expected_stdout or slurm.get("stderr_path") != expected_stderr:
        raise P3BAcquisitionError("B2 smoke Slurm log paths differ from the launcher")
    expected_report = {
        "schema_version": SMOKE_REPORT_SCHEMA,
        "gate": GATE,
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "git_commit": expected_commit,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "smoke_manifest_sha256": manifest["manifest_sha256"],
        "smoke_receipt_sha256": receipt["manifest_sha256"],
        "record_file_sha256": receipt["record_file_sha256"],
        "record_count": 4,
        "formal_forward_calls": 4,
        "loop_insertions": 0,
        "boundaries": list(BOUNDARY_IDS),
        "model_revision_closure": expected_closure,
        "choice_token_mapping": mapping,
        "decoder_layers": int(card["model"]["decoder_layers"]),
        "final_norm_class": runtime["final_norm_class"],
        "lm_head_class": runtime["lm_head_class"],
        "slurm_job_id": smoke_job_id,
        "status": "PASS_ENGINEERING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise P3BAcquisitionError("B2 smoke report %s mismatch" % key)
    difference = report.get("final_head_max_abs_difference")
    if isinstance(difference, bool) or not isinstance(difference, (int, float)):
        raise P3BAcquisitionError("B2 smoke final-head closure evidence is invalid")
    if not (float(difference) >= 0.0 and float(difference) < float("inf")):
        raise P3BAcquisitionError("B2 smoke final-head difference is non-finite")
    return report


def freeze_smoke(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    card, pool, source, pool_manifest = load_live_b1_artifacts(run_root, card_path)
    manifest = build_smoke_manifest(pool, source, pool_manifest, card, git, run_root, argv)
    validate_membership_manifest(manifest, pool, source, pool_manifest, card)
    path = Path(run_root) / "smoke/smoke_manifest.json"
    write_new_json(path, manifest)
    return {"path": str(path), "manifest_sha256": manifest["manifest_sha256"]}


def freeze_shards(
    *,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    shard_count: int,
    argv: Sequence[str],
) -> Dict[str, Any]:
    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    card, pool, source, pool_manifest = load_live_b1_artifacts(run_root, card_path)
    smoke_report = validate_completed_smoke(
        run_root, card, pool, source, pool_manifest, expected_commit
    )
    manifest = build_shard_manifest(
        pool,
        source,
        pool_manifest,
        card,
        git,
        run_root,
        shard_count,
        smoke_report["manifest_sha256"],
        argv,
    )
    validate_membership_manifest(manifest, pool, source, pool_manifest, card)
    path = Path(run_root) / SHARD_MANIFEST_NAME
    write_new_json(path, manifest)
    return {
        "path": str(path),
        "shard_count": shard_count,
        "manifest_sha256": manifest["manifest_sha256"],
    }


def write_slurm_launchers(
    *,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    stage: str,
    partition: str,
    qos: Optional[str],
    account: Optional[str],
    time_limit: str,
    cpus_per_task: int,
    memory: str,
    gres: str,
    array_throttle: Optional[int],
) -> Dict[str, Any]:
    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    card, pool, source, pool_manifest = load_live_b1_artifacts(run_root, card_path)
    if stage not in {"smoke", "primary"}:
        raise P3BAcquisitionError("launcher stage must be smoke or primary")
    for value, context in (
        (partition, "partition"),
        (time_limit, "time limit"),
        (memory, "memory"),
        (gres, "GRES"),
    ):
        if not str(value or "").strip() or re.search(r"[\r\n\x00]", str(value)):
            raise P3BAcquisitionError("Slurm %s is invalid" % context)
    if isinstance(cpus_per_task, bool) or cpus_per_task < 1:
        raise P3BAcquisitionError("Slurm cpus_per_task must be positive")
    if not re.fullmatch(r"gpu(?::[A-Za-z0-9_.-]+)?:1", str(gres)):
        raise P3BAcquisitionError("P3-B launchers require exactly one GPU per task")
    if qos is not None and re.search(r"[\r\n\x00]", qos):
        raise P3BAcquisitionError("Slurm QoS is invalid")
    if account is not None and re.search(r"[\r\n\x00]", account):
        raise P3BAcquisitionError("Slurm account is invalid")
    common = """#!/usr/bin/env bash
set -euo pipefail
if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +e
  source /etc/profile.d/modules.sh >/dev/null 2>&1
  set -e
fi
module load anaconda3 cuda/12.4
. {venv}/bin/activate
cd {repo}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={src}
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset HF_ENDPOINT
""".format(
        venv=shlex.quote(str(AUDITED_VENV)),
        repo=shlex.quote(str(REMOTE_REPO)),
        src=shlex.quote(str(REMOTE_REPO / "src")),
    )
    script = REMOTE_REPO / "scripts/loopscope/run_qwen17_phase3_p3b.py"
    common_args = "--card {card} --run-root {root} --expected-commit {commit}".format(
        card=shlex.quote(str(REMOTE_REPO / "configs/loopscope/phase3_card.json")),
        root=shlex.quote(str(AUTHORIZED_RUN_ROOT)),
        commit=shlex.quote(expected_commit),
    )
    if stage == "smoke":
        membership_manifest = load_strict_json(
            Path(run_root) / "smoke/smoke_manifest.json"
        )
        validate_membership_manifest(
            membership_manifest, pool, source, pool_manifest, card
        )
        launcher_text = common + (
            "exec python {script} acquire-smoke {args}\n"
        ).format(script=shlex.quote(str(script)), args=common_args)
        launcher_path = Path(run_root) / "slurm/p3b_smoke_runner.sh"
        submit_path = Path(run_root) / "slurm/p3b_smoke_submit.sh"
        job_name = "loopscope-p3b-smoke"
        output_pattern = Path(run_root) / "slurm/p3b-smoke-%j.out"
        error_pattern = Path(run_root) / "slurm/p3b-smoke-%j.err"
        array_spec = None
        if array_throttle is not None:
            raise P3BAcquisitionError("smoke launcher forbids array throttle")
    else:
        membership_manifest = load_strict_json(Path(run_root) / SHARD_MANIFEST_NAME)
        validate_membership_manifest(
            membership_manifest, pool, source, pool_manifest, card
        )
        smoke_report = validate_completed_smoke(
            Path(run_root), card, pool, source, pool_manifest, expected_commit
        )
        if membership_manifest.get("smoke_report_sha256") != smoke_report["manifest_sha256"]:
            raise P3BAcquisitionError("primary launcher lacks exact B2 smoke binding")
        launcher_text = common + (
            "exec python {script} acquire-shard {args} "
            "--shard-id \"${{SLURM_ARRAY_TASK_ID}}\"\n"
        ).format(script=shlex.quote(str(script)), args=common_args)
        launcher_path = Path(run_root) / "slurm/p3b_primary_runner.sh"
        submit_path = Path(run_root) / "slurm/p3b_primary_submit.sh"
        job_name = "loopscope-p3b-primary"
        output_pattern = Path(run_root) / "slurm/p3b-primary-%A_%a.out"
        error_pattern = Path(run_root) / "slurm/p3b-primary-%A_%a.err"
        shard_count = int(membership_manifest["shard_count"])
        if (
            isinstance(array_throttle, bool)
            or not isinstance(array_throttle, int)
            or array_throttle < 1
            or array_throttle > shard_count
        ):
            raise P3BAcquisitionError("primary array throttle is outside frozen shards")
        array_spec = "0-%d%%%d" % (shard_count - 1, array_throttle)

    launcher_hash = write_new_text(launcher_path, launcher_text, executable=True)
    sbatch_argv = [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--job-name=%s" % job_name,
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=%d" % cpus_per_task,
        "--mem=%s" % memory,
        "--time=%s" % time_limit,
        "--partition=%s" % partition,
        "--gres=%s" % gres,
        "--output=%s" % output_pattern,
        "--error=%s" % error_pattern,
    ]
    if qos:
        sbatch_argv.append("--qos=%s" % qos)
    if account:
        sbatch_argv.append("--account=%s" % account)
    if array_spec is not None:
        sbatch_argv.append("--array=%s" % array_spec)
    sbatch_argv.append(str(launcher_path))
    submit_text = "#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n" % shlex.join(
        sbatch_argv
    )
    submit_hash = write_new_text(submit_path, submit_text, executable=True)
    receipt: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3b-launcher-receipt.v2",
        "artifact_role": "p3b_%s_slurm_launcher_receipt" % stage,
        "created_at_utc": utc_now(),
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "stage": stage,
        "membership_manifest_sha256": membership_manifest["manifest_sha256"],
        "launchers": {
            "runner": {
                "path": str(launcher_path.relative_to(run_root)),
                "sha256": launcher_hash,
            },
            "submit": {
                "path": str(submit_path.relative_to(run_root)),
                "sha256": submit_hash,
            },
        },
        "sbatch_argv": sbatch_argv,
        "resource_config": {
            "partition": partition,
            "qos": qos,
            "account": account,
            "time_limit": time_limit,
            "cpus_per_task": cpus_per_task,
            "memory": memory,
            "gres": gres,
            "array_spec": array_spec,
            "output_pattern": str(output_pattern),
            "error_pattern": str(error_pattern),
        },
        "offline_environment": True,
        "implementation_sha256": implementation_hashes(),
    }
    attach_manifest_sha256(receipt)
    write_new_json(
        Path(run_root) / ("slurm/p3b_%s_launcher_receipt.json" % stage), receipt
    )
    return dict(receipt)


class _NativeNoLoopRuntime:
    def __init__(
        self,
        *,
        torch_module: Any,
        tokenizer: Any,
        model: Any,
        choice_mapping: Mapping[str, Any],
        choice_ids: Sequence[int],
        revision_closure: Mapping[str, Any],
        final_norm: Any,
        lm_head: Any,
        layer_count: int,
        boundary_states_fn: Callable[[Sequence[Any], int], Tuple[Any, ...]],
        lens_space_fn: Callable[[Any, Any, int, int], Any],
    ) -> None:
        self.torch = torch_module
        self.tokenizer = tokenizer
        self.model = model
        self.choice_mapping = dict(choice_mapping)
        self.choice_ids = list(choice_ids)
        self.revision_closure = dict(revision_closure)
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.layer_count = int(layer_count)
        self.boundary_states_fn = boundary_states_fn
        self.lens_space_fn = lens_space_fn


def _assert_no_loop_modules_loaded() -> None:
    forbidden = (
        "tflt.wrapper",
        "tflt.strategies",
        "tflt.cache",
        "tflt.loopscope.window_probe",
    )
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == item or name.startswith(item + ".") for item in forbidden)
    )
    if loaded:
        raise P3BAcquisitionError("forbidden loop modules were imported: %s" % loaded)


def _active_loop_wrapper_modules(model: Any) -> List[str]:
    active = []
    try:
        modules = model.named_modules()
    except Exception as exc:
        raise P3BAcquisitionError("cannot inspect model for loop wrappers") from exc
    for name, module in modules:
        class_module = str(getattr(module.__class__, "__module__", ""))
        if class_module == "tflt.wrapper" or class_module.startswith("tflt.wrapper."):
            active.append(str(name or "<root>"))
    forward_module = str(getattr(getattr(model, "forward", None), "__module__", ""))
    if forward_module == "tflt.wrapper" or forward_module.startswith("tflt.wrapper."):
        active.append("<model.forward>")
    return sorted(set(active))


def _assert_native_model_unwrapped(model: Any) -> None:
    active = _active_loop_wrapper_modules(model)
    if active:
        raise P3BAcquisitionError("active loop wrappers found on native model: %s" % active)


def load_native_no_loop_runtime(card: Mapping[str, Any]) -> _NativeNoLoopRuntime:
    _assert_no_loop_modules_loaded()
    if not MODEL_SNAPSHOT.is_dir():
        raise P3BAcquisitionError("exact frozen model snapshot is missing")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tflt.loopscope.probe import (
            choice_tokenization,
            decoder_boundary_states,
            find_final_norm,
            lens_space_hidden,
        )
        from tflt.loopscope.revisions import (
            load_tokenizer_with_resolved_commit,
            strict_revision_closure,
        )
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise P3BAcquisitionError("P3-B acquisition runtime imports failed") from exc
    if not torch.cuda.is_available():
        raise P3BAcquisitionError("P3-B acquisition requires a visible CUDA GPU")
    load_kwargs = {
        "revision": card["model"]["revision"],
        "local_files_only": True,
        "trust_remote_code": True,
    }
    tokenizer, tokenizer_commit = load_tokenizer_with_resolved_commit(
        AutoTokenizer, card["model"]["repo"], load_kwargs
    )
    choice_mapping, choice_ids, warnings = choice_tokenization(tokenizer, ("A", "B", "C", "D"))
    if warnings:
        raise P3BAcquisitionError("choice tokenization emitted warnings")
    if list(choice_mapping) != ["A", "B", "C", "D"] or len(set(choice_ids)) != 4:
        raise P3BAcquisitionError("choice token mapping is not exact A/B/C/D")
    for label, metadata in choice_mapping.items():
        if metadata.get("token_count") != 1 or len(metadata.get("token_ids", [])) != 1:
            raise P3BAcquisitionError("spaced choice surface is not one token: %s" % label)
        if len(metadata.get("plain_surface_token_ids", [])) != 1:
            raise P3BAcquisitionError("plain choice surface is not one token: %s" % label)
    model = AutoModelForCausalLM.from_pretrained(
        card["model"]["repo"],
        torch_dtype=torch.float16,
        **load_kwargs,
    )
    closure = strict_revision_closure(model, tokenizer_commit, card["model"]["revision"])
    layer_count = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if layer_count != int(card["model"]["decoder_layers"]):
        raise P3BAcquisitionError("loaded model does not expose exactly 28 decoder layers")
    model.eval()
    model.to("cuda")
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise P3BAcquisitionError(
            "loaded Qwen forward lacks final-position logits_to_keep support"
        )
    final_norm = find_final_norm(model)
    lm_head = model.get_output_embeddings() or getattr(model, "lm_head", None)
    if lm_head is None or not callable(lm_head):
        raise P3BAcquisitionError("loaded model lacks a callable output head")
    _assert_native_model_unwrapped(model)
    _assert_no_loop_modules_loaded()
    return _NativeNoLoopRuntime(
        torch_module=torch,
        tokenizer=tokenizer,
        model=model,
        choice_mapping=choice_mapping,
        choice_ids=choice_ids,
        revision_closure=closure,
        final_norm=final_norm,
        lm_head=lm_head,
        layer_count=layer_count,
        boundary_states_fn=decoder_boundary_states,
        lens_space_fn=lens_space_hidden,
    )


def trajectory_renderer_provenance(
    card: Mapping[str, Any], source_manifest: Mapping[str, Any], pool_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["revision"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "renderer_manifest_sha256": ANCHOR_MANIFEST_INTERNAL_SHA256,
        "forward_type": card["trajectory"]["forward_type"],
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "answer_position": card["trajectory"]["answer_position"],
        "projection": card["trajectory"]["projection"],
        "choice_order": list(card["trajectory"]["choice_order"]),
    }


def _project_choice_logits(
    runtime: _NativeNoLoopRuntime, lens_hidden: Any
) -> Any:
    """Apply exactly the four frozen LM-head rows in A/B/C/D order."""

    weight = getattr(runtime.lm_head, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        raise P3BAcquisitionError("LM head does not expose a two-dimensional weight")
    selected_weight = weight[runtime.choice_ids]
    bias = getattr(runtime.lm_head, "bias", None)
    selected_bias = None if bias is None else bias[runtime.choice_ids]
    projected = runtime.torch.nn.functional.linear(
        lens_hidden, selected_weight, selected_bias
    )
    return projected[0, 0]


def acquire_one_trajectory(
    runtime: _NativeNoLoopRuntime,
    pool_record: Mapping[str, Any],
    source_record: Mapping[str, Any],
    renderer_provenance: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _assert_no_loop_modules_loaded()
    _assert_native_model_unwrapped(runtime.model)
    prompt = str(pool_record["rendered_prompt"])
    if not prompt.rstrip().endswith("Answer:"):
        raise P3BAcquisitionError("formal prompt does not end at Answer:")
    encoded = runtime.tokenizer(prompt, return_tensors="pt")
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise P3BAcquisitionError("tokenizer output lacks input_ids/attention_mask")
    mask = encoded["attention_mask"]
    valid = mask[0].detach().cpu().nonzero(as_tuple=False).flatten().tolist()
    if not valid or int(valid[-1]) != int(mask.shape[1]) - 1:
        raise P3BAcquisitionError("formal prompt answer position is padded or ambiguous")
    answer_position = int(valid[-1])
    inputs = {key: value.to("cuda") for key, value in encoded.items()}
    torch = runtime.torch
    with torch.inference_mode():
        outputs = runtime.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        hidden_states = tuple(outputs.hidden_states or ())
        boundary_states = runtime.boundary_states_fn(hidden_states, runtime.layer_count)
        logits_by_boundary = {}
        final_head_max_abs_difference = 0.0
        for boundary_index, hidden in enumerate(boundary_states):
            answer_hidden = hidden[:, answer_position : answer_position + 1, :]
            lens_hidden = runtime.lens_space_fn(
                runtime.final_norm,
                answer_hidden,
                boundary_index,
                runtime.layer_count,
            )
            projected_choice_logits = _project_choice_logits(runtime, lens_hidden)
            if boundary_index == runtime.layer_count:
                if int(outputs.logits.shape[1]) != 1:
                    raise P3BAcquisitionError(
                        "native model did not restrict logits to the final prompt position"
                    )
                choice_logits = outputs.logits[0, 0, runtime.choice_ids]
                difference = (
                    choice_logits - projected_choice_logits
                ).detach().abs().max()
                final_head_max_abs_difference = float(difference.to(dtype=torch.float64).cpu())
                if not torch.allclose(
                    choice_logits,
                    projected_choice_logits,
                    rtol=1e-3,
                    atol=1e-3,
                ):
                    raise P3BAcquisitionError(
                        "native final logits do not close through the frozen LM head"
                    )
            else:
                choice_logits = projected_choice_logits
            if not bool(torch.isfinite(choice_logits).all().item()):
                raise P3BAcquisitionError("choice logits contain non-finite values")
            logits_by_boundary["B_%d" % boundary_index] = (
                choice_logits.detach().to(dtype=torch.float64).cpu().tolist()
            )
    record = trajectory_record_from_logits(
        {
            "identity": pool_record["identity"],
            "subject": pool_record["subject"],
            "split": pool_record["split"],
            "prompt_sha256": pool_record["prompt_sha256"],
        },
        logits_by_boundary,
        renderer_provenance,
        card,
    )
    normalized = validate_trajectory_record(record, card, expected_source=source_record)
    _assert_native_model_unwrapped(runtime.model)
    _assert_no_loop_modules_loaded()
    del outputs, hidden_states, boundary_states, inputs, encoded, logits_by_boundary
    return normalized, {
        "sequence_length": answer_position + 1,
        "answer_position": answer_position,
        "final_head_max_abs_difference": final_head_max_abs_difference,
    }


def _select_membership(
    manifest: Mapping[str, Any], mode: str, shard_id: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], str, str]:
    if mode == "smoke":
        return (
            [dict(value) for value in manifest["records"]],
            str(manifest["membership_sha256"]),
            str(manifest["output_relative"]),
        )
    if shard_id is None or shard_id < 0 or shard_id >= int(manifest["shard_count"]):
        raise P3BAcquisitionError("shard_id is outside the frozen manifest")
    shard = manifest["shards"][shard_id]
    return (
        [dict(value) for value in shard["records"]],
        str(shard["membership_sha256"]),
        str(shard["output_relative"]),
    )


def _runtime_evidence(runtime: _NativeNoLoopRuntime, forward_calls: int) -> Dict[str, Any]:
    device_index = int(runtime.torch.cuda.current_device())
    properties = runtime.torch.cuda.get_device_properties(device_index)
    capability = runtime.torch.cuda.get_device_capability(device_index)
    device_uuid = getattr(properties, "uuid", None)
    if isinstance(device_uuid, bytes):
        device_uuid = device_uuid.decode("utf-8", errors="replace")
    return {
        "model_revision_closure": dict(runtime.revision_closure),
        "model_class": runtime.model.__class__.__name__,
        "tokenizer_class": runtime.tokenizer.__class__.__name__,
        "final_norm_class": runtime.final_norm.__class__.__name__,
        "lm_head_class": runtime.lm_head.__class__.__name__,
        "lm_head_is_model_output_embeddings": (
            runtime.model.get_output_embeddings() is runtime.lm_head
        ),
        "choice_projection_kernel": "exact_selected_lm_head_weight_rows_A_B_C_D",
        "decoder_layers": runtime.layer_count,
        "choice_token_mapping": dict(runtime.choice_mapping),
        "choice_token_ids": list(runtime.choice_ids),
        "dtype": "float16",
        "device": "cuda",
        "cuda_device": {
            "runtime_index": device_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "name": str(getattr(properties, "name", "")),
            "uuid": str(device_uuid) if device_uuid is not None else None,
            "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
            "compute_capability": [int(capability[0]), int(capability[1])],
        },
        "use_cache": False,
        "logits_to_keep": 1,
        "native_logits_position": "final_pre_answer_prompt_token_only",
        "formal_forward_calls": int(forward_calls),
        "loop_insertions": 0,
        "new_loop_modules_loaded_by_producer": [],
        "active_loop_wrapper_modules": _active_loop_wrapper_modules(runtime.model),
        "model_snapshot": str(MODEL_SNAPSHOT),
        "versions": {
            "python": platform.python_version(),
            "torch": str(getattr(runtime.torch, "__version__", "")),
            "transformers": _module_version("transformers"),
        },
    }


def acquire_membership(
    *,
    mode: str,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    argv: Sequence[str],
    shard_id: Optional[int] = None,
) -> Dict[str, Any]:
    if mode not in ("smoke", "primary"):
        raise P3BAcquisitionError("acquisition mode must be smoke or primary")
    env_job_id = os.environ.get("SLURM_JOB_ID")
    env_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    env_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if mode == "smoke":
        if not re.fullmatch(r"[0-9]+", str(env_job_id or "")):
            raise P3BAcquisitionError("B2 smoke must run inside one Slurm job")
        if env_array_job_id is not None or env_array_task_id is not None:
            raise P3BAcquisitionError("B2 smoke must not run as a Slurm array task")
    else:
        if not re.fullmatch(r"[0-9]+", str(env_array_job_id or "")):
            raise P3BAcquisitionError("B3 primary must run inside the frozen Slurm array")
        if str(env_array_task_id or "") != str(shard_id):
            raise P3BAcquisitionError("B3 Slurm array task differs from shard_id")
    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    card, pool, source, pool_manifest = load_live_b1_artifacts(run_root, card_path)
    manifest_path = (
        Path(run_root) / "smoke/smoke_manifest.json"
        if mode == "smoke"
        else Path(run_root) / SHARD_MANIFEST_NAME
    )
    membership_manifest = load_strict_json(manifest_path)
    validate_membership_manifest(membership_manifest, pool, source, pool_manifest, card)
    if membership_manifest["mode"] != mode:
        raise P3BAcquisitionError("membership mode differs from acquisition mode")
    if mode == "primary":
        smoke_report = validate_completed_smoke(
            Path(run_root), card, pool, source, pool_manifest, expected_commit
        )
        if membership_manifest.get("smoke_report_sha256") != smoke_report["manifest_sha256"]:
            raise P3BAcquisitionError("primary acquisition lacks exact B2 smoke binding")
    members, membership_hash, output_relative = _select_membership(
        membership_manifest, mode, shard_id
    )
    output_dir = (Path(run_root) / output_relative).resolve()
    if Path(run_root).resolve() not in output_dir.parents:
        raise P3BAcquisitionError("acquisition output escapes run root")
    if output_dir.exists():
        raise FileExistsError("refusing to reuse acquisition output: %s" % output_dir)

    renderer = trajectory_renderer_provenance(card, source, pool_manifest)
    runtime = load_native_no_loop_runtime(card)
    start_time = utc_now()
    ensure_new_directory(output_dir)
    records_path = output_dir / "records.jsonl"
    processed: List[Dict[str, Any]] = []
    sequence_lengths: List[int] = []
    answer_positions: List[int] = []
    final_head_differences: List[float] = []
    source_records = source["records"]
    try:
        with records_path.open("x", encoding="utf-8") as handle:
            for member_index, member in enumerate(members):
                ordinal = int(member["ordinal"])
                record, position = acquire_one_trajectory(
                    runtime,
                    pool[ordinal],
                    source_records[ordinal],
                    renderer,
                    card,
                )
                handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                processed.append(record)
                sequence_lengths.append(position["sequence_length"])
                answer_positions.append(position["answer_position"])
                final_head_differences.append(position["final_head_max_abs_difference"])
                if (member_index + 1) % 10 == 0 or member_index + 1 == len(members):
                    print(
                        "p3b %s progress %d/%d"
                        % (mode, member_index + 1, len(members)),
                        flush=True,
                    )
    except Exception as exc:
        failure: Dict[str, Any] = {
            "schema_version": "loopscope.phase3.p3b-acquisition-failure.v1",
            "gate": GATE,
            "mode": mode,
            "created_at_utc": utc_now(),
            "run_root": str(AUTHORIZED_RUN_ROOT),
            "output_dir": str(output_dir),
            "membership_manifest_sha256": membership_manifest["manifest_sha256"],
            "membership_sha256": membership_hash,
            "processed_valid_record_count": len(processed),
            "processed_identities": [record["identity"] for record in processed],
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
        }
        attach_manifest_sha256(failure)
        write_new_json(output_dir / "failure.json", failure)
        raise

    reloaded = load_strict_jsonl(records_path)
    if reloaded != processed or len(reloaded) != len(members):
        raise P3BAcquisitionError("persisted acquisition records differ from in-memory records")
    for record, member in zip(reloaded, members):
        ordinal = int(member["ordinal"])
        validate_trajectory_record(record, card, expected_source=source_records[ordinal])
        if identity_tuple(record["identity"]) != identity_tuple(member["identity"]):
            raise P3BAcquisitionError("acquired record identity differs from membership")
    end_time = utc_now()
    runtime_evidence = _runtime_evidence(runtime, len(reloaded))
    runtime_evidence["final_head_max_abs_difference"] = max(final_head_differences)
    job_id = os.environ.get("SLURM_JOB_ID")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if mode == "smoke" and job_id:
        stdout_path = Path(run_root) / ("slurm/p3b-smoke-%s.out" % job_id)
        stderr_path = Path(run_root) / ("slurm/p3b-smoke-%s.err" % job_id)
    elif mode == "primary" and array_job_id and array_task_id:
        stdout_path = Path(run_root) / (
            "slurm/p3b-primary-%s_%s.out" % (array_job_id, array_task_id)
        )
        stderr_path = Path(run_root) / (
            "slurm/p3b-primary-%s_%s.err" % (array_job_id, array_task_id)
        )
    else:
        stdout_path = None
        stderr_path = None
    receipt: Dict[str, Any] = {
        "schema_version": ACQUISITION_RECEIPT_SCHEMA,
        "artifact_role": (
            "validation1531_native_no_loop_smoke_receipt"
            if mode == "smoke"
            else "validation1531_native_no_loop_shard_receipt"
        ),
        "mode": mode,
        "shard_id": shard_id,
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "started_at_utc": start_time,
        "completed_at_utc": end_time,
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "output_dir": str(output_dir),
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "membership_manifest_sha256": membership_manifest["manifest_sha256"],
        "membership_sha256": membership_hash,
        "record_count": len(reloaded),
        "record_file": str(records_path),
        "record_file_sha256": file_sha256(records_path),
        "ordered_record_sha256": hashlib.sha256(canonical_json_bytes(reloaded)).hexdigest(),
        "boundaries_per_record": len(BOUNDARY_IDS),
        "sequence_length_min": min(sequence_lengths),
        "sequence_length_max": max(sequence_lengths),
        "answer_position_rule": "last non-padding final pre-answer prompt token",
        "answer_position_min": min(answer_positions),
        "answer_position_max": max(answer_positions),
        "runtime": runtime_evidence,
        "slurm": {
            "job_id": job_id,
            "array_job_id": array_job_id,
            "array_task_id": array_task_id,
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "qos": os.environ.get("SLURM_JOB_QOS"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "node": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "stdout_path": str(stdout_path) if stdout_path is not None else None,
            "stderr_path": str(stderr_path) if stderr_path is not None else None,
        },
        "producer": {
            "argv": list(argv),
            "implementation_sha256": implementation_hashes(),
        },
        "checks": {
            "native_no_loop": True,
            "formal_forward_count_per_identity_one": True,
            "loop_modules_not_loaded": True,
            "raw_logits_not_persisted": True,
            "hidden_states_not_persisted": True,
            "b0_through_b28_exact": True,
            "metrics_recomputed_by_p3a_validator": True,
        },
        "status": "COMPLETED",
    }
    attach_manifest_sha256(receipt)
    write_new_json(output_dir / "receipt.json", receipt)
    result = {
        "receipt_manifest_sha256": receipt["manifest_sha256"],
        "record_count": len(reloaded),
        "record_file_sha256": receipt["record_file_sha256"],
    }
    if mode == "smoke":
        report: Dict[str, Any] = {
            "schema_version": SMOKE_REPORT_SCHEMA,
            "artifact_role": "validation1531_no_loop_smoke_report",
            "gate": GATE,
            "created_at_utc": utc_now(),
            "run_root": str(AUTHORIZED_RUN_ROOT),
            "git_commit": expected_commit,
            "card_sha256": PHASE3_CARD_BYTE_SHA256,
            "source_manifest_sha256": source["manifest_sha256"],
            "pool_manifest_sha256": pool_manifest["manifest_sha256"],
            "smoke_manifest_sha256": membership_manifest["manifest_sha256"],
            "smoke_receipt_sha256": receipt["manifest_sha256"],
            "record_file_sha256": receipt["record_file_sha256"],
            "record_count": 4,
            "subject_count": len({record["subject"] for record in reloaded}),
            "formal_forward_calls": 4,
            "loop_insertions": 0,
            "boundaries": list(BOUNDARY_IDS),
            "model_revision_closure": runtime.revision_closure,
            "choice_token_mapping": runtime.choice_mapping,
            "decoder_layers": runtime.layer_count,
            "final_norm_class": runtime.final_norm.__class__.__name__,
            "lm_head_class": runtime.lm_head.__class__.__name__,
            "final_head_max_abs_difference": max(final_head_differences),
            "slurm_job_id": str(job_id) if job_id is not None else None,
            "status": "PASS_ENGINEERING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        }
        attach_manifest_sha256(report)
        write_new_json(Path(run_root) / SMOKE_REPORT_NAME, report)
        result["smoke_report_sha256"] = report["manifest_sha256"]
    del runtime
    return result


def acquire_smoke(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    return acquire_membership(
        mode="smoke",
        card_path=card_path,
        run_root=run_root,
        expected_commit=expected_commit,
        argv=argv,
    )


def acquire_shard(
    *,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    shard_id: int,
    argv: Sequence[str],
) -> Dict[str, Any]:
    return acquire_membership(
        mode="primary",
        card_path=card_path,
        run_root=run_root,
        expected_commit=expected_commit,
        argv=argv,
        shard_id=shard_id,
    )


def _load_primary_shards(
    run_root: Path,
    shard_manifest: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_records = []
    evidence = []
    for shard in shard_manifest["shards"]:
        output_dir = Path(run_root) / shard["output_relative"]
        record_path = output_dir / "records.jsonl"
        receipt_path = output_dir / "receipt.json"
        if not record_path.is_file() or not receipt_path.is_file():
            raise P3BAcquisitionError("primary shard output is incomplete: %s" % output_dir)
        receipt = load_strict_json(receipt_path)
        verify_manifest_sha256(receipt)
        if receipt.get("status") != "COMPLETED" or receipt.get("mode") != "primary":
            raise P3BAcquisitionError("primary shard receipt is not completed")
        if receipt.get("shard_id") != shard["shard_id"]:
            raise P3BAcquisitionError("primary shard receipt ID mismatch")
        if receipt.get("membership_sha256") != shard["membership_sha256"]:
            raise P3BAcquisitionError("primary shard membership binding mismatch")
        records = load_strict_jsonl(record_path)
        if len(records) != shard["record_count"] or receipt.get("record_count") != len(records):
            raise P3BAcquisitionError("primary shard record count mismatch")
        if receipt.get("record_file_sha256") != file_sha256(record_path):
            raise P3BAcquisitionError("primary shard file SHA256 mismatch")
        for record, member in zip(records, shard["records"]):
            ordinal = int(member["ordinal"])
            validate_trajectory_record(record, card, expected_source=source["records"][ordinal])
            if identity_tuple(record["identity"]) != identity_tuple(pool[ordinal]["identity"]):
                raise P3BAcquisitionError("primary shard identity differs from pool")
        all_records.extend(records)
        evidence.append(
            {
                "shard_id": shard["shard_id"],
                "record_count": len(records),
                "record_file": str(record_path),
                "record_file_sha256": file_sha256(record_path),
                "receipt": str(receipt_path),
                "receipt_manifest_sha256": receipt["manifest_sha256"],
                "slurm": receipt["slurm"],
            }
        )
    all_records.sort(key=canonical_record_key)
    normalized = validate_trajectory_records(all_records, source, pool_manifest, card)
    return normalized, evidence


def merge_primary(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    card, pool, source, pool_manifest = load_live_b1_artifacts(run_root, card_path)
    shard_manifest = load_strict_json(Path(run_root) / SHARD_MANIFEST_NAME)
    validate_membership_manifest(shard_manifest, pool, source, pool_manifest, card)
    smoke_report = validate_completed_smoke(
        Path(run_root), card, pool, source, pool_manifest, expected_commit
    )
    if shard_manifest.get("smoke_report_sha256") != smoke_report["manifest_sha256"]:
        raise P3BAcquisitionError("primary merge lacks exact B2 smoke binding")
    records, shard_evidence = _load_primary_shards(
        run_root, shard_manifest, pool, source, pool_manifest, card
    )
    trajectory_manifest = make_trajectory_manifest(records, source, pool_manifest, card)
    trajectory_path = Path(run_root) / TRAJECTORY_JSONL_NAME
    trajectory_file_sha = write_new_jsonl(trajectory_path, records)
    trajectory_manifest_path = Path(run_root) / TRAJECTORY_MANIFEST_NAME
    write_new_json(trajectory_manifest_path, trajectory_manifest)
    merge_receipt: Dict[str, Any] = {
        "schema_version": MERGE_RECEIPT_SCHEMA,
        "artifact_role": "validation1531_shard_merge_receipt",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "shard_manifest_sha256": shard_manifest["manifest_sha256"],
        "shard_count": shard_manifest["shard_count"],
        "shards": shard_evidence,
        "record_count": len(records),
        "subject_count": len({record["subject"] for record in records}),
        "missing_count": 0,
        "duplicate_count": 0,
        "extra_count": 0,
        "trajectory_file": str(trajectory_path),
        "trajectory_file_sha256": trajectory_file_sha,
        "trajectory_manifest": str(trajectory_manifest_path),
        "trajectory_manifest_file_sha256": file_sha256(trajectory_manifest_path),
        "trajectory_manifest_sha256": trajectory_manifest["manifest_sha256"],
        "ordered_identity_sha256": trajectory_manifest["ordered_identity_sha256"],
        "ordered_trajectory_record_sha256": trajectory_manifest[
            "ordered_trajectory_record_sha256"
        ],
        "producer": {
            "argv": list(argv),
            "implementation_sha256": implementation_hashes(),
        },
        "status": "PASS",
    }
    attach_manifest_sha256(merge_receipt)
    write_new_json(Path(run_root) / MERGE_RECEIPT_NAME, merge_receipt)
    probe_report: Dict[str, Any] = {
        "schema_version": PROBE_REPORT_SCHEMA,
        "artifact_role": "validation1531_no_loop_probe_report",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "git_commit": expected_commit,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "shard_manifest_sha256": shard_manifest["manifest_sha256"],
        "trajectory_manifest_sha256": trajectory_manifest["manifest_sha256"],
        "merge_receipt_sha256": merge_receipt["manifest_sha256"],
        "record_count": len(records),
        "subject_count": len({record["subject"] for record in records}),
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "boundaries": list(BOUNDARY_IDS),
        "selector_safe_fields_only": True,
        "outcome_values_read": False,
        "baseline_analysis_performed": False,
        "status": "TRAJECTORY_ACQUISITION_COMPLETE",
    }
    attach_manifest_sha256(probe_report)
    write_new_json(Path(run_root) / PROBE_REPORT_NAME, probe_report)
    return {
        "trajectory_manifest_sha256": trajectory_manifest["manifest_sha256"],
        "merge_receipt_sha256": merge_receipt["manifest_sha256"],
        "probe_report_sha256": probe_report["manifest_sha256"],
        "record_count": len(records),
    }


def verify_primary(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    card, pool, source, pool_manifest = load_live_b1_artifacts(run_root, card_path)
    shard_manifest = load_strict_json(Path(run_root) / SHARD_MANIFEST_NAME)
    validate_membership_manifest(shard_manifest, pool, source, pool_manifest, card)
    smoke_report = validate_completed_smoke(
        Path(run_root), card, pool, source, pool_manifest, expected_commit
    )
    if shard_manifest.get("smoke_report_sha256") != smoke_report["manifest_sha256"]:
        raise P3BAcquisitionError("primary verifier lacks exact B2 smoke binding")
    shard_records, shard_evidence = _load_primary_shards(
        run_root, shard_manifest, pool, source, pool_manifest, card
    )
    trajectories = load_strict_jsonl(Path(run_root) / TRAJECTORY_JSONL_NAME)
    if trajectories != shard_records:
        raise P3BAcquisitionError("merged trajectory file differs from verified shard union")
    normalized = validate_trajectory_records(trajectories, source, pool_manifest, card)
    trajectory_manifest = load_strict_json(Path(run_root) / TRAJECTORY_MANIFEST_NAME)
    validate_trajectory_manifest(trajectory_manifest, source, pool_manifest, card)
    expected_manifest = make_trajectory_manifest(normalized, source, pool_manifest, card)
    if trajectory_manifest != expected_manifest:
        raise P3BAcquisitionError("trajectory manifest differs from independent recomputation")
    merge_receipt = load_strict_json(Path(run_root) / MERGE_RECEIPT_NAME)
    probe_report = load_strict_json(Path(run_root) / PROBE_REPORT_NAME)
    verify_manifest_sha256(merge_receipt)
    verify_manifest_sha256(probe_report)
    if merge_receipt.get("trajectory_manifest_sha256") != trajectory_manifest["manifest_sha256"]:
        raise P3BAcquisitionError("merge receipt trajectory binding mismatch")
    if probe_report.get("trajectory_manifest_sha256") != trajectory_manifest["manifest_sha256"]:
        raise P3BAcquisitionError("probe report trajectory binding mismatch")
    receipt: Dict[str, Any] = {
        "schema_version": VERIFIER_RECEIPT_SCHEMA,
        "artifact_role": "validation1531_no_loop_probe_verifier_receipt",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "shard_manifest_sha256": shard_manifest["manifest_sha256"],
        "trajectory_manifest_sha256": trajectory_manifest["manifest_sha256"],
        "trajectory_file_sha256": file_sha256(Path(run_root) / TRAJECTORY_JSONL_NAME),
        "record_count": len(normalized),
        "subject_count": len({record["subject"] for record in normalized}),
        "shard_count": len(shard_evidence),
        "checks": {
            "source_manifest_revalidated": True,
            "pool_closed_world_revalidated": True,
            "shard_membership_exact_disjoint_union": True,
            "record_metrics_recomputed": True,
            "b0_through_b28_exact": True,
            "h_kl_ce_identity_recomputed": True,
            "formal_forward_count_per_identity_one": True,
            "loop_insertions_zero": True,
            "canonical_1531_57_closure": True,
            "raw_logits_absent": True,
            "hidden_states_absent": True,
            "outcome_values_read": False,
            "selector_analysis_performed": False,
        },
        "verifier": {
            "argv": list(argv),
            "implementation_sha256": implementation_hashes(),
        },
        "status": "PASS",
    }
    attach_manifest_sha256(receipt)
    write_new_json(Path(run_root) / VERIFIER_RECEIPT_NAME, receipt)
    return dict(receipt)


SACCT_FIELDS = (
    "JobID",
    "JobIDRaw",
    "JobName",
    "Partition",
    "QOS",
    "Account",
    "State",
    "ExitCode",
    "NodeList",
    "AllocTRES",
    "ReqTRES",
    "ElapsedRaw",
    "Submit",
    "Start",
    "End",
)


def parse_sacct_rows(text: str) -> List[Dict[str, str]]:
    rows = []
    for line in str(text).splitlines():
        if not line.strip():
            continue
        values = line.rstrip("|").split("|")
        if len(values) != len(SACCT_FIELDS):
            raise P3BAcquisitionError("unexpected sacct row field count")
        rows.append(dict(zip(SACCT_FIELDS, values)))
    if not rows:
        raise P3BAcquisitionError("sacct returned no rows")
    return rows


def _query_sacct(job_id: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise P3BAcquisitionError("Slurm job ID must be numeric")
    command = [
        "/opt/slurm/bin/sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        str(job_id),
        "--format=" + ",".join(SACCT_FIELDS),
    ]
    output = subprocess.check_output(command, text=True)
    return command, parse_sacct_rows(output)


def _tres_gpu_count(value: str) -> int:
    counts = []
    for part in str(value).split(","):
        if part.startswith("gres/gpu") and "=" in part:
            try:
                counts.append(int(part.rsplit("=", 1)[1]))
            except ValueError:
                raise P3BAcquisitionError("invalid GPU TRES value")
    return max(counts, default=0)


def _queue_seconds(row: Mapping[str, str]) -> Optional[int]:
    try:
        submit = datetime.fromisoformat(row["Submit"])
        start = datetime.fromisoformat(row["Start"])
    except (KeyError, TypeError, ValueError):
        return None
    return max(0, int((start - submit).total_seconds()))


def _parse_utc_timestamp(value: str, context: str) -> datetime:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise P3BAcquisitionError("%s must be an ISO-8601 timestamp" % context) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise P3BAcquisitionError("%s must be UTC" % context)
    return parsed


def record_monitor_lifecycle(
    *,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    stage: str,
    job_id: str,
    automation_id: str,
    automation_name: str,
    cadence_minutes: int,
    created_at_utc: str,
    terminal_wakeup_at_utc: str,
    disabled_at_utc: str,
    terminal_state: str,
    disable_evidence: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    """Seal one executor-owned heartbeat after it has been deleted."""

    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    _, _, source, pool_manifest = load_live_b1_artifacts(run_root, card_path)
    if stage not in {"smoke", "primary"}:
        raise P3BAcquisitionError("monitor stage must be smoke or primary")
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise P3BAcquisitionError("monitor job ID must be numeric")
    automation_id = str(automation_id or "").strip()
    automation_name = str(automation_name or "").strip()
    if not automation_id:
        raise P3BAcquisitionError("monitor automation ID is required")
    prefix = "monitor-LoopScope-H3V1-P3-B-%s-" % stage
    if not automation_name.startswith(prefix) or str(job_id) not in automation_name:
        raise P3BAcquisitionError("monitor automation name is not stage/job specific")
    if cadence_minutes not in (10, 30, 60):
        raise P3BAcquisitionError("monitor cadence must be 10, 30, or 60 minutes")
    created = _parse_utc_timestamp(created_at_utc, "monitor creation time")
    terminal = _parse_utc_timestamp(terminal_wakeup_at_utc, "monitor terminal wakeup")
    disabled = _parse_utc_timestamp(disabled_at_utc, "monitor deletion time")
    if not created <= terminal <= disabled:
        raise P3BAcquisitionError("monitor lifecycle timestamps are not ordered")
    terminal_state = str(terminal_state or "").strip().upper()
    if not terminal_state:
        raise P3BAcquisitionError("monitor terminal state is required")
    disable_evidence = str(disable_evidence or "").strip()
    if not disable_evidence:
        raise P3BAcquisitionError("monitor deletion evidence is required")
    receipt: Dict[str, Any] = {
        "schema_version": MONITOR_RECEIPT_SCHEMA,
        "artifact_role": "p3b_executor_owned_monitor_lifecycle",
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "target_executor_thread_id": EXECUTOR_THREAD_ID,
        "created_at_utc": utc_now(),
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "git": git,
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "stage": stage,
        "host": "hpc2-hkustgz",
        "job_ids": [str(job_id)],
        "automation": {
            "id": automation_id,
            "name": automation_name,
            "cadence_minutes": int(cadence_minutes),
            "created_at_utc": str(created_at_utc),
            "terminal_wakeup_at_utc": str(terminal_wakeup_at_utc),
            "disabled_at_utc": str(disabled_at_utc),
            "lifecycle_action": "DELETED",
            "active_after_terminal": False,
            "disable_evidence": disable_evidence,
        },
        "terminal_state": terminal_state,
        "check_contract": {
            "one_bounded_read_only_check_per_wakeup": True,
            "batch_mode": True,
            "connect_timeout_seconds": 10,
            "clear_all_forwardings": True,
            "manual_polling_or_shell_sleep": False,
            "mutation_or_retry_authority": False,
            "science_or_gate_decision_authority": False,
        },
        "producer": {
            "argv": list(argv),
            "implementation_sha256": implementation_hashes(),
        },
        "status": "TERMINAL_MONITOR_DELETED",
    }
    attach_manifest_sha256(receipt)
    path = Path(run_root) / (
        "monitoring/%s-%s-monitor-lifecycle.json" % (stage, job_id)
    )
    write_new_json(path, receipt)
    return {
        "path": str(path),
        "manifest_sha256": receipt["manifest_sha256"],
        "automation_id": automation_id,
        "active_after_terminal": False,
    }


def _load_monitor_lifecycle_receipts(
    run_root: Path,
    source: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    smoke_job_id: str,
    primary_job_id: str,
) -> List[Dict[str, Any]]:
    directory = Path(run_root) / "monitoring"
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise P3BAcquisitionError("monitoring artifact path is not a directory")
    receipts = []
    seen_ids = set()
    seen_names = set()
    expected_jobs = {"smoke": str(smoke_job_id), "primary": str(primary_job_id)}
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
        if not path.is_file() or not path.name.endswith("-monitor-lifecycle.json"):
            raise P3BAcquisitionError("unexpected monitoring artifact: %s" % path)
        receipt = load_strict_json(path)
        verify_manifest_sha256(receipt)
        exact = {
            "schema_version": MONITOR_RECEIPT_SCHEMA,
            "gate": GATE,
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "target_executor_thread_id": EXECUTOR_THREAD_ID,
            "run_root": str(AUTHORIZED_RUN_ROOT),
            "card_sha256": PHASE3_CARD_BYTE_SHA256,
            "source_manifest_sha256": source["manifest_sha256"],
            "pool_manifest_sha256": pool_manifest["manifest_sha256"],
            "status": "TERMINAL_MONITOR_DELETED",
        }
        for key, expected in exact.items():
            if receipt.get(key) != expected:
                raise P3BAcquisitionError("monitor lifecycle %s mismatch" % key)
        stage = receipt.get("stage")
        if stage not in expected_jobs or receipt.get("job_ids") != [expected_jobs[stage]]:
            raise P3BAcquisitionError("monitor lifecycle job binding mismatch")
        automation = receipt.get("automation")
        if not isinstance(automation, Mapping):
            raise P3BAcquisitionError("monitor lifecycle automation evidence is missing")
        if automation.get("lifecycle_action") != "DELETED":
            raise P3BAcquisitionError("monitor lifecycle was not deleted")
        if automation.get("active_after_terminal") is not False:
            raise P3BAcquisitionError("monitor remains active after terminal state")
        automation_id = str(automation.get("id") or "")
        automation_name = str(automation.get("name") or "")
        if not automation_id or automation_id in seen_ids:
            raise P3BAcquisitionError("monitor automation ID is missing or duplicated")
        if not automation_name or automation_name in seen_names:
            raise P3BAcquisitionError("monitor automation name is missing or duplicated")
        seen_ids.add(automation_id)
        seen_names.add(automation_name)
        normalized = dict(receipt)
        normalized["artifact_path"] = str(path)
        normalized["artifact_file_sha256"] = file_sha256(path)
        receipts.append(normalized)
    return receipts


def _closed_artifact_git_commit(receipt: Mapping[str, Any], context: str) -> str:
    git = receipt.get("git")
    if not isinstance(git, Mapping):
        raise P3BAcquisitionError("%s Git provenance is missing" % context)
    commit = _exact_commit(git.get("commit"), "%s Git commit" % context)
    if git.get("origin_loopscope") != commit or git.get("dirty") is not False:
        raise P3BAcquisitionError("%s Git provenance is not closed" % context)
    return commit


def _require_artifact_commit_ancestor(artifact_commit: str, accounting_commit: str) -> None:
    command = [
        "git",
        "merge-base",
        "--is-ancestor",
        _exact_commit(artifact_commit, "scientific artifact commit"),
        _exact_commit(accounting_commit, "resource accounting commit"),
    ]
    result = subprocess.run(
        command,
        cwd=str(repository_root()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise P3BAcquisitionError(
            "scientific artifact commit is not an ancestor of resource accounting commit"
        )


def write_resource_accounting(
    *,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    smoke_job_id: str,
    primary_job_id: str,
    initial_throttle: int,
    argv: Sequence[str],
) -> Dict[str, Any]:
    assert_offline_environment()
    git = git_provenance(expected_commit, remote=True)
    card, pool, source, pool_manifest = load_live_b1_artifacts(run_root, card_path)
    shard_manifest = load_strict_json(Path(run_root) / SHARD_MANIFEST_NAME)
    smoke_manifest = load_strict_json(Path(run_root) / "smoke/smoke_manifest.json")
    validate_membership_manifest(shard_manifest, pool, source, pool_manifest, card)
    validate_membership_manifest(smoke_manifest, pool, source, pool_manifest, card)
    smoke_receipt = load_strict_json(
        Path(run_root) / str(smoke_manifest["output_relative"]) / "receipt.json"
    )
    verify_manifest_sha256(smoke_receipt)
    artifact_commit = _closed_artifact_git_commit(smoke_receipt, "B2 smoke receipt")
    _require_artifact_commit_ancestor(artifact_commit, git["commit"])
    smoke_report = validate_completed_smoke(
        Path(run_root),
        card,
        pool,
        source,
        pool_manifest,
        artifact_commit,
        resource_accounting_commit=git["commit"],
    )
    if shard_manifest.get("smoke_report_sha256") != smoke_report["manifest_sha256"]:
        raise P3BAcquisitionError("resource accounting lacks exact B2 smoke binding")
    monitor_receipts = _load_monitor_lifecycle_receipts(
        Path(run_root), source, pool_manifest, smoke_job_id, primary_job_id
    )
    if smoke_report.get("slurm_job_id") != str(smoke_job_id):
        raise P3BAcquisitionError("resource accounting smoke job ID differs from B2")
    _, primary_shard_evidence = _load_primary_shards(
        Path(run_root), shard_manifest, pool, source, pool_manifest, card
    )
    for shard in primary_shard_evidence:
        shard_receipt = load_strict_json(Path(shard["receipt"]))
        verify_manifest_sha256(shard_receipt)
        if (
            _closed_artifact_git_commit(shard_receipt, "primary shard receipt")
            != artifact_commit
        ):
            raise P3BAcquisitionError("scientific shard producer commit mismatch")
        slurm = shard.get("slurm")
        if not isinstance(slurm, Mapping):
            raise P3BAcquisitionError("primary shard lacks Slurm receipt evidence")
        if slurm.get("array_job_id") != str(primary_job_id):
            raise P3BAcquisitionError("primary shard array job ID mismatch")
        if slurm.get("array_task_id") != str(shard["shard_id"]):
            raise P3BAcquisitionError("primary shard array task ID mismatch")
    smoke_command, smoke_rows = _query_sacct(smoke_job_id)
    primary_command, primary_rows = _query_sacct(primary_job_id)
    smoke_selected = [row for row in smoke_rows if row["JobID"] == str(smoke_job_id)]
    primary_selected = [
        row
        for row in primary_rows
        if re.fullmatch(re.escape(str(primary_job_id)) + r"_[0-9]+", row["JobID"])
    ]
    if len(smoke_selected) != 1:
        raise P3BAcquisitionError("smoke sacct evidence lacks one parent job row")
    if len(primary_selected) != int(shard_manifest["shard_count"]):
        raise P3BAcquisitionError("primary sacct task count differs from frozen shards")
    selected = smoke_selected + sorted(
        primary_selected, key=lambda row: int(row["JobID"].split("_", 1)[1])
    )
    if any(row["State"] != "COMPLETED" or row["ExitCode"] != "0:0" for row in selected):
        raise P3BAcquisitionError("Slurm jobs are not all COMPLETED/0:0")
    normalized_rows = []
    total_gpu_seconds = 0
    for row in selected:
        elapsed = int(row["ElapsedRaw"] or 0)
        allocated_gpus = _tres_gpu_count(row["AllocTRES"])
        requested_gpus = _tres_gpu_count(row["ReqTRES"])
        total_gpu_seconds += elapsed * allocated_gpus
        normalized = dict(row)
        if row["JobID"] == str(smoke_job_id):
            log_stem = "p3b-smoke-%s" % smoke_job_id
        else:
            task_id = row["JobID"].split("_", 1)[1]
            log_stem = "p3b-primary-%s_%s" % (primary_job_id, task_id)
        stdout_path = Path(run_root) / ("slurm/%s.out" % log_stem)
        stderr_path = Path(run_root) / ("slurm/%s.err" % log_stem)
        if not stdout_path.is_file() or not stderr_path.is_file():
            raise P3BAcquisitionError("terminal Slurm stdout/stderr evidence is missing")
        normalized.update(
            {
                "elapsed_seconds": elapsed,
                "queue_seconds": _queue_seconds(row),
                "allocated_gpu_count": allocated_gpus,
                "requested_gpu_count": requested_gpus,
                "gpu_seconds": elapsed * allocated_gpus,
                "stdout": {
                    "path": str(stdout_path),
                    "sha256": file_sha256(stdout_path),
                    "size_bytes": stdout_path.stat().st_size,
                },
                "stderr": {
                    "path": str(stderr_path),
                    "sha256": file_sha256(stderr_path),
                    "size_bytes": stderr_path.stat().st_size,
                },
            }
        )
        normalized_rows.append(normalized)
    receipt: Dict[str, Any] = {
        "schema_version": RESOURCE_RECEIPT_SCHEMA,
        "artifact_role": "p3b_scheduler_throttle_resource_accounting",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(AUTHORIZED_RUN_ROOT),
        "git": git,
        "git_lineage": {
            "scientific_artifact_commit": artifact_commit,
            "resource_accounting_commit": git["commit"],
            "artifact_commit_is_ancestor": True,
        },
        "card_sha256": PHASE3_CARD_BYTE_SHA256,
        "source_manifest_sha256": source["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "smoke_manifest_sha256": smoke_manifest["manifest_sha256"],
        "prelaunch_shard_manifest_sha256": shard_manifest["manifest_sha256"],
        "scientific_config_sha256": hashlib.sha256(
            canonical_json_bytes(shard_manifest["scientific_config"])
        ).hexdigest(),
        "jobs": {
            "smoke_job_id": str(smoke_job_id),
            "primary_array_job_id": str(primary_job_id),
        },
        "sacct_commands": [smoke_command, primary_command],
        "terminal_rows": normalized_rows,
        "actual_partitions": sorted({row["Partition"] for row in selected}),
        "actual_qos": sorted({row["QOS"] for row in selected}),
        "actual_accounts": sorted({row["Account"] for row in selected}),
        "actual_nodes": sorted({row["NodeList"] for row in selected}),
        "initial_array_throttle": int(initial_throttle),
        "throttle_events": [],
        "science_or_membership_changed_after_freeze": False,
        "monitoring": {
            "receipt_count": len(monitor_receipts),
            "lifecycle_receipts": monitor_receipts,
            "all_recorded_monitors_deleted": True,
            "active_recorded_monitor_count": 0,
            "manual_polling_used_after_long_wait_boundary": False,
        },
        "total_gpu_seconds": total_gpu_seconds,
        "total_gpu_hours": total_gpu_seconds / 3600.0,
        "producer": {
            "argv": list(argv),
            "implementation_sha256": implementation_hashes(),
        },
        "status": "PASS",
    }
    attach_manifest_sha256(receipt)
    write_new_json(Path(run_root) / RESOURCE_RECEIPT_NAME, receipt)
    return dict(receipt)


def _exact_commit(value: Any, context: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{40}", text):
        raise P3BAcquisitionError("%s must be an exact 40-hex commit" % context)
    return text


def _module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "")) or None


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant %s" % value)


__all__ = [
    "AUTHORIZED_RUN_ROOT",
    "P3BAcquisitionError",
    "acquire_shard",
    "acquire_smoke",
    "build_safe_validation_pool",
    "build_shard_manifest",
    "build_smoke_manifest",
    "freeze_shards",
    "freeze_smoke",
    "load_verified_renderer_anchor",
    "materialize_b0_b1",
    "merge_primary",
    "parse_sacct_rows",
    "pool_record_from_anchor_and_safe_row",
    "record_monitor_lifecycle",
    "safe_dataset_fields",
    "safe_target_doc_sha256",
    "trajectory_renderer_provenance",
    "validate_membership_manifest",
    "verify_primary",
    "write_new_json",
    "write_new_jsonl",
    "write_resource_accounting",
    "write_slurm_launchers",
]
