"""Outcome-blind Phase 5 Gate B validation trajectory acquisition.

The module deliberately reuses the already audited Phase 3 gold-free MMLU
validation pool.  It never imports or requests the MMLU test split.  Each
trajectory record is reduced from one native, unwrapped model forward to four
choice probabilities and five scalar boundary trajectories before it is
persisted.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase5_schema import (
    BOUNDARY_IDS,
    CHOICE_ORDER,
    EXPECTED_PLAIN_TOKEN_IDS,
    EXPECTED_SPACED_TOKEN_IDS,
    Phase5ContractError,
    canonical_json_bytes,
    file_sha256,
    load_json,
    stable_choice_probabilities,
    validate_card,
    validate_token_ids,
    validate_trajectory_record,
)


GATE = "B"
EXECUTOR_THREAD_ID = "019f864e-9b57-73e1-b56c-e7e40e7ad28d"
PLANNING_THREAD_ID = "019f8604-4717-7be2-8bf8-9d4a26a3d7f7"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
WORKSPACE = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope"
)
AUTHORIZED_RUN_ROOT = WORKSPACE / (
    "runs/phase5-gate-b-repair-b36-native-20260722T035452Z"
)
INVALID_RUN_ROOT = WORKSPACE / "runs/phase5-gate-b-20260721T201330Z"
INVALID_FORMAL_JOB_ID = "10030284"
INVALID_IMPLEMENTATION_COMMIT = "9431688b8e1f704837b22e1370978dea59d528c0"
INVALID_FORMAL_MEMBERSHIP_SHA256 = (
    "e874e1a896eadc0819858bc4cc465ced6008b6945adba67a016c4dffa23a3f63"
)
REPAIR_SMOKE_IDENTITIES = (
    ("mmlu_marketing", "mmlu_marketing:validation:24"),
    ("mmlu_human_aging", "mmlu_human_aging:validation:18"),
)
PHASE3_RUN_ROOT = WORKSPACE / "runs/phase3-p3b-20260715T223304Z"
MODEL_SNAPSHOT = Path(
    "/hpc2hdd/home/xhuang225/shared/hf_home/hub/"
    "models--Qwen--Qwen3-4B-Base/snapshots/"
    "906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
CARD_RELATIVE = Path("configs/loopscope/phase5_card.json")
REGISTRY_RELATIVE = Path("configs/loopscope/phase5_known_outcome_registry.json")
PHASE3_CARD_RELATIVE = Path("configs/loopscope/phase3_card.json")
PHASE3_POOL_RELATIVE = Path("validation1531_pool.jsonl")
PHASE3_POOL_MANIFEST_RELATIVE = Path("validation1531_pool_manifest.json")
PHASE3_SOURCE_MANIFEST_RELATIVE = Path("source_manifest.json")

CARD_FILE_SHA256 = "8d25996f994d131ea0fd9ca78f83c90680a67c41c7c460f7e4e8b4ae5ea42a73"
ANCHOR_MANIFEST_FILE_SHA256 = "48ba67c204249b5f2f9a6cf711f8678711b5e606398d35ba4d919d4eee70e6e7"
ANCHOR_MANIFEST_INTERNAL_SHA256 = "e1baaa92c8ab3804c75a5e415a324d00fed0277d17c26b16d12b8c99aafbe949"
ANCHOR_PROJECTION_SHA256 = "439e41113458ccfbd51d211ab7fae4e91ecc02723c7b14ad6c1969803aa61a6a"

IMPLEMENTATION_RELATIVE_PATHS = (
    "src/tflt/loopscope/phase5_acquisition.py",
    "src/tflt/loopscope/phase5_selector.py",
    "scripts/loopscope/run_qwen4base_phase5_gate_b.py",
)

MEMBERSHIP_SCHEMA = "loopscope.phase5.gate-b-membership.v1"
SHARD_RECEIPT_SCHEMA = "loopscope.phase5.gate-b-shard-receipt.v1"
TRAJECTORY_MANIFEST_SCHEMA = "loopscope.phase5.trajectory-manifest.v1"


class Phase5AcquisitionError(Phase5ContractError):
    """Fail-closed Gate B acquisition error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _reject_constant(value: str) -> None:
    raise Phase5AcquisitionError("non-finite JSON constant is forbidden: %s" % value)


def load_strict_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise Phase5AcquisitionError("JSON artifact must be an object: %s" % path)
    return payload


def load_strict_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise Phase5AcquisitionError("blank JSONL line at %s:%d" % (path, line_number))
            value = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise Phase5AcquisitionError("JSONL record must be an object")
            records.append(value)
    return records


def _write_new_bytes(path: Path, body: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(body).hexdigest()


def write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_new_bytes(path, canonical_json_bytes(payload) + b"\n")


def write_new_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    body = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    return _write_new_bytes(path, body)


def write_new_text(path: Path, value: str, *, executable: bool = False) -> str:
    body = value.encode("utf-8")
    digest = _write_new_bytes(path, body)
    if executable:
        Path(path).chmod(0o755)
    return digest


def attach_manifest_sha256(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "manifest_sha256" in payload:
        raise Phase5AcquisitionError("manifest_sha256 must not be pre-populated")
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def verify_manifest_sha256(payload: Mapping[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    expected = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if payload.get("manifest_sha256") != expected:
        raise Phase5AcquisitionError("manifest_sha256 mismatch")


def assert_authorized_run_root(path: Path, *, must_exist: Optional[bool] = None) -> Path:
    resolved = Path(path).resolve()
    if resolved != AUTHORIZED_RUN_ROOT:
        raise Phase5AcquisitionError("run root differs from Gate B authorization")
    if must_exist is True and not resolved.is_dir():
        raise Phase5AcquisitionError("authorized run root does not exist")
    if must_exist is False and resolved.exists():
        raise FileExistsError("refusing to reuse Gate B run root: %s" % resolved)
    return resolved


def _invalid_attempt_evidence() -> Dict[str, Any]:
    """Bind the superseded partial formal attempt without consuming its records."""

    root = INVALID_RUN_ROOT.resolve()
    if not root.is_dir() or root == AUTHORIZED_RUN_ROOT.resolve():
        raise Phase5AcquisitionError("invalid formal root is missing or aliases repair root")
    manifest_path = root / "formal/formal_shard_manifest.json"
    manifest = load_strict_json(manifest_path)
    verify_manifest_sha256(manifest)
    if manifest.get("manifest_sha256") != INVALID_FORMAL_MEMBERSHIP_SHA256:
        raise Phase5AcquisitionError("invalid formal membership hash differs")
    expected_counts = (194, 383, 167, 382)
    shard_evidence = []
    total_records = 0
    for shard_id, expected_count in enumerate(expected_counts):
        attempt = root / ("formal/shards/shard-%04d/attempt-0001" % shard_id)
        records_path = attempt / "records.jsonl"
        receipt_paths = [
            path
            for path in (attempt / "receipt.json", attempt / "failure_receipt.json")
            if path.is_file()
        ]
        if len(receipt_paths) != 1 or not records_path.is_file():
            raise Phase5AcquisitionError("invalid formal shard evidence is incomplete")
        receipt_path = receipt_paths[0]
        receipt = load_strict_json(receipt_path)
        verify_manifest_sha256(receipt)
        with records_path.open("rb") as handle:
            record_count = sum(1 for line in handle if line.strip())
        if record_count != expected_count:
            raise Phase5AcquisitionError("invalid formal shard record count differs")
        if receipt_path.name == "receipt.json":
            forward_count = int(receipt.get("runtime", {}).get("forward_calls", -1))
            if receipt.get("result") != "COMPLETED" or int(receipt.get("record_count", -1)) != record_count:
                raise Phase5AcquisitionError("invalid formal completed receipt differs")
            if receipt.get("git", {}).get("commit") != INVALID_IMPLEMENTATION_COMMIT:
                raise Phase5AcquisitionError("invalid formal implementation commit differs")
        else:
            forward_count = int(receipt.get("forward_calls", -1))
            if (
                receipt.get("result") != "FAILED"
                or int(receipt.get("records_written", -1)) != record_count
                or receipt.get("formal_retry_eligible") is not False
            ):
                raise Phase5AcquisitionError("invalid formal failure receipt differs")
        total_records += record_count
        shard_evidence.append(
            {
                "shard_id": shard_id,
                "result": receipt["result"],
                "forward_calls": forward_count,
                "record_count": record_count,
                "receipt_relative_path": str(receipt_path.relative_to(root)),
                "receipt_file_sha256": file_sha256(receipt_path),
                "receipt_manifest_sha256": receipt["manifest_sha256"],
                "records_relative_path": str(records_path.relative_to(root)),
                "records_file_sha256": file_sha256(records_path),
            }
        )
    if total_records != 1126:
        raise Phase5AcquisitionError("invalid formal persisted record total differs")
    forbidden_outputs = (
        root / "formal/validation1531_phase5_trajectories.jsonl",
        root / "formal/validation1531_phase5_trajectory_manifest.json",
        root / "selector/selector_freeze.json",
        root / "selector/outcome_panel.json",
    )
    if any(path.exists() for path in forbidden_outputs):
        raise Phase5AcquisitionError("invalid formal root was merged or selected")
    return {
        "root": str(root),
        "formal_job_id": INVALID_FORMAL_JOB_ID,
        "implementation_commit": INVALID_IMPLEMENTATION_COMMIT,
        "formal_membership_sha256": manifest["manifest_sha256"],
        "formal_membership_file_sha256": file_sha256(manifest_path),
        "persisted_records": total_records,
        "missing_records": 1531 - total_records,
        "canonical_selector_input": False,
        "merge_or_salvage": False,
        "invalidation_reason": "sliced-four-row-bfloat16-B36-closure-false-negative",
        "shards": shard_evidence,
    }


def git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = repository_root()
    commands = {
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "commit": ["git", "rev-parse", "HEAD"],
        "origin_loopscope": ["git", "rev-parse", "origin/loopscope"],
        "status": ["git", "status", "--porcelain=v1"],
    }
    values: Dict[str, str] = {}
    for key, command in commands.items():
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise Phase5AcquisitionError("Git provenance failed: %s" % key)
        values[key] = completed.stdout.strip()
    if values["branch"] != "loopscope":
        raise Phase5AcquisitionError("Gate B requires branch loopscope")
    if values["commit"] != expected_commit or values["origin_loopscope"] != expected_commit:
        raise Phase5AcquisitionError("local/origin/expected commit mismatch")
    if values["status"]:
        raise Phase5AcquisitionError("Gate B repository must be clean")
    return {
        "root": str(root),
        "branch": values["branch"],
        "commit": values["commit"],
        "origin_loopscope": values["origin_loopscope"],
        "dirty": False,
    }


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    result = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise Phase5AcquisitionError("implementation file is missing: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def load_frozen_card(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path or repository_root() / CARD_RELATIVE)
    if file_sha256(path) != CARD_FILE_SHA256:
        raise Phase5AcquisitionError("Gate A card byte hash differs")
    card = load_json(path)
    validate_card(card)
    return card


def _canonical_identity(identity: Mapping[str, Any]) -> str:
    if set(identity) != {"task", "doc_id", "doc_hash"}:
        raise Phase5AcquisitionError("canonical identity keys differ")
    if not str(identity["task"]).strip() or not str(identity["doc_id"]).strip():
        raise Phase5AcquisitionError("canonical identity contains an empty component")
    doc_hash = str(identity["doc_hash"])
    if not re.fullmatch(r"[0-9a-f]{64}", doc_hash):
        raise Phase5AcquisitionError("canonical doc_hash is invalid")
    return "\x1f".join((str(identity["task"]), str(identity["doc_id"]), doc_hash))


def _membership_row(record: Mapping[str, Any], ordinal: int) -> Dict[str, Any]:
    return {
        "ordinal": int(ordinal),
        "identity": dict(record["identity"]),
        "subject": str(record["subject"]),
        "split": str(record["split"]),
        "prompt_sha256": str(record["prompt_sha256"]),
    }


def _membership_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(rows))).hexdigest()


def load_phase3_gold_free_pool() -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Revalidate and return the prior outcome-blind validation pool only."""

    from tflt.loopscope.phase3_acquisition import load_live_b1_artifacts

    _, pool, source, pool_manifest = load_live_b1_artifacts(
        PHASE3_RUN_ROOT, repository_root() / PHASE3_CARD_RELATIVE
    )
    if len(pool) != 1531 or len({str(row["subject"]) for row in pool}) != 57:
        raise Phase5AcquisitionError("reused validation pool is not exactly 1531/57")
    seen = set()
    for row in pool:
        if row.get("split") != "validation":
            raise Phase5AcquisitionError("reused pool contains a non-validation split")
        identity = _canonical_identity(row["identity"])
        if identity in seen:
            raise Phase5AcquisitionError("reused pool contains duplicate identity")
        seen.add(identity)
    return [dict(row) for row in pool], dict(source), dict(pool_manifest)


def _module_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _snapshot_evidence(card: Mapping[str, Any]) -> Dict[str, Any]:
    if not MODEL_SNAPSHOT.is_dir():
        raise Phase5AcquisitionError("pinned Qwen3-4B-Base snapshot is missing")
    observed = {}
    for name, expected in sorted(card["model"]["weight_shard_sha256"].items()):
        path = MODEL_SNAPSHOT / name
        if not path.is_file():
            raise Phase5AcquisitionError("model shard is missing: %s" % name)
        digest = file_sha256(path)
        if digest != expected:
            raise Phase5AcquisitionError("model shard hash differs: %s" % name)
        observed[name] = {"sha256": digest, "size_bytes": path.stat().st_size}
    total_size = sum(value["size_bytes"] for value in observed.values())
    index = load_strict_json(MODEL_SNAPSHOT / "model.safetensors.index.json")
    declared_total_size = index.get("metadata", {}).get("total_size")
    if declared_total_size != int(card["model"]["safetensors_total_size_bytes"]):
        raise Phase5AcquisitionError("model snapshot index total_size differs")
    return {
        "path": str(MODEL_SNAPSHOT),
        "revision": card["model"]["revision"],
        "shards": observed,
        "index_declared_total_size_bytes": declared_total_size,
        "physical_shard_file_size_bytes": total_size,
    }


def _cpu_runtime_preflight(card: Mapping[str, Any]) -> Dict[str, Any]:
    """Inspect config/tokenizer/import provenance without instantiating model weights."""

    try:
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    except Exception as exc:
        raise Phase5AcquisitionError("CPU runtime provenance imports failed") from exc
    config = AutoConfig.from_pretrained(
        str(MODEL_SNAPSHOT), local_files_only=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_SNAPSHOT), local_files_only=True, trust_remote_code=True
    )
    if config.__class__.__module__ + "." + config.__class__.__name__ != card["model"]["config_class"]:
        raise Phase5AcquisitionError("CPU config class differs")
    if int(config.num_hidden_layers) != 36 or int(config.hidden_size) != 2560:
        raise Phase5AcquisitionError("CPU config layer/hidden size differs")
    if int(config.vocab_size) != 151936 or not bool(config.tie_word_embeddings):
        raise Phase5AcquisitionError("CPU config vocabulary/tied embeddings differ")
    if Qwen3ForCausalLM.__module__ + "." + Qwen3ForCausalLM.__name__ != card["model"]["runtime_causal_lm_class"]:
        raise Phase5AcquisitionError("CPU causal LM class import differs")
    spaced = {label: tokenizer.encode(" " + label, add_special_tokens=False) for label in CHOICE_ORDER}
    plain = {label: tokenizer.encode(label, add_special_tokens=False) for label in CHOICE_ORDER}
    validate_token_ids(spaced, plain)
    return {
        "config_class": config.__class__.__module__ + "." + config.__class__.__name__,
        "causal_lm_class": Qwen3ForCausalLM.__module__ + "." + Qwen3ForCausalLM.__name__,
        "tokenizer_class": tokenizer.__class__.__module__ + "." + tokenizer.__class__.__name__,
        "decoder_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "tie_word_embeddings": bool(config.tie_word_embeddings),
        "spaced_choice_token_ids": {key: values[0] for key, values in spaced.items()},
        "plain_choice_token_ids": {key: values[0] for key, values in plain.items()},
        "model_weights_instantiated": False,
        "forward_calls": 0,
        "cuda_used": False,
    }


def materialize_admission(*, expected_commit: str, argv: Sequence[str]) -> Dict[str, Any]:
    run_root = assert_authorized_run_root(AUTHORIZED_RUN_ROOT, must_exist=False)
    git = git_provenance(expected_commit)
    card = load_frozen_card()
    pool, source, pool_manifest = load_phase3_gold_free_pool()
    versions = {
        "python": platform.python_version(),
        "torch": _module_version("torch"),
        "transformers": _module_version("transformers"),
        "tokenizers": _module_version("tokenizers"),
        "datasets": _module_version("datasets"),
        "lm_eval": _module_version("lm_eval"),
        "venv": os.environ.get("VIRTUAL_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    expected_versions = {
        "transformers": card["task"]["transformers_version"],
        "tokenizers": card["task"]["tokenizers_version"],
        "datasets": card["task"]["datasets_version"],
        "lm_eval": card["task"]["lm_eval_version"],
    }
    for name, expected in expected_versions.items():
        if versions[name] != expected:
            raise Phase5AcquisitionError("CPU environment %s version differs" % name)
    if Path(str(versions["venv"] or "")).resolve() != AUDITED_VENV:
        raise Phase5AcquisitionError("CPU admission venv differs from audited environment")
    if versions["cuda_visible_devices"] not in (None, "", "NoDevFiles"):
        raise Phase5AcquisitionError("CPU admission must not expose CUDA devices")
    snapshot = _snapshot_evidence(card)
    cpu_runtime = _cpu_runtime_preflight(card)
    invalid_attempt = _invalid_attempt_evidence()
    receipt = attach_manifest_sha256(
        {
            "schema_version": "loopscope.phase5.gate-b-admission.v1",
            "created_at_utc": utc_now(),
            "gate": GATE,
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "planning_thread_id": PLANNING_THREAD_ID,
            "run_root": str(run_root),
            "git": git,
            "argv": list(argv),
            "card": {
                "relative_path": str(CARD_RELATIVE),
                "file_sha256": file_sha256(repository_root() / CARD_RELATIVE),
                "manifest_sha256": card["manifest_sha256"],
            },
            "model_snapshot": snapshot,
            "cpu_runtime": cpu_runtime,
            "environment": versions,
            "validation_source": {
                "phase3_run_root": str(PHASE3_RUN_ROOT),
                "source_manifest_sha256": source["manifest_sha256"],
                "pool_manifest_sha256": pool_manifest["manifest_sha256"],
                "renderer_manifest_file_sha256": ANCHOR_MANIFEST_FILE_SHA256,
                "renderer_manifest_internal_sha256": ANCHOR_MANIFEST_INTERNAL_SHA256,
                "renderer_projection_sha256": ANCHOR_PROJECTION_SHA256,
                "record_count": len(pool),
                "subject_count": len({row["subject"] for row in pool}),
                "ordered_identity_sha256": pool_manifest["ordered_identity_sha256"],
                "split": "validation",
                "test_imported": False,
                "outcome_fields_read": False,
            },
            "superseded_invalid_attempt": invalid_attempt,
            "repair_authority": {
                "planning_decision": "PASS_WITH_FIXES",
                "repair_cycle": "1_OF_1_PLANNING_AUDIT_RETURN",
                "b36_authoritative_source": "outputs.logits[0,0,choice_ids]",
                "full_head_native_closure": "same-shape-full-vocabulary-fail-closed",
                "sliced_choice_projection_gating": False,
            },
            "implementation_sha256": implementation_hashes(),
            "result": "ADMITTED",
        }
    )
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "admission").mkdir()
    write_new_json(run_root / "admission/gate_b_admission_receipt.json", receipt)
    return receipt


def _scientific_config(card: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["tokenizer_revision"],
        "dtype": card["model"]["dtype"],
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "split": "validation",
        "forward_type": "native_no_loop",
        "formal_forward_count_per_identity": 1,
        "use_cache": False,
        "loop_insertions": 0,
        "boundaries": list(BOUNDARY_IDS),
        "position": "final_non_padding_prompt_token",
        "choice_order": list(CHOICE_ORDER),
        "spaced_choice_token_ids": EXPECTED_SPACED_TOKEN_IDS,
        "hidden_reduction_dtype": "float32",
        "choice_analysis_dtype": "float64",
    }


def _smoke_ordinals(pool: Sequence[Mapping[str, Any]]) -> List[int]:
    ordinals = []
    for task, doc_id in REPAIR_SMOKE_IDENTITIES:
        matches = [
            index
            for index, row in enumerate(pool)
            if row.get("identity", {}).get("task") == task
            and row.get("identity", {}).get("doc_id") == doc_id
        ]
        if len(matches) != 1:
            raise Phase5AcquisitionError("repair smoke identity is missing or ambiguous")
        ordinals.append(matches[0])
    return sorted(ordinals)


def _smoke_manifest_path(run_root: Path, manifest_revision: int) -> Path:
    if manifest_revision == 1:
        return run_root / "smoke/smoke_membership_manifest.json"
    if manifest_revision in (2, 3):
        return run_root / ("smoke/smoke_membership_manifest-repair-%04d.json" % manifest_revision)
    raise Phase5AcquisitionError("smoke manifest revision exceeds repair budget")


def freeze_smoke(
    *, expected_commit: str, argv: Sequence[str], manifest_revision: int = 1
) -> Dict[str, Any]:
    run_root = assert_authorized_run_root(AUTHORIZED_RUN_ROOT, must_exist=True)
    git = git_provenance(expected_commit)
    card = load_frozen_card()
    pool, source, pool_manifest = load_phase3_gold_free_pool()
    ordinals = _smoke_ordinals(pool)
    rows = [_membership_row(pool[index], index) for index in ordinals]
    if len({row["subject"] for row in rows}) < 2:
        raise Phase5AcquisitionError("smoke membership must span at least two subjects")
    payload = attach_manifest_sha256(
        {
            "schema_version": MEMBERSHIP_SCHEMA,
            "created_at_utc": utc_now(),
            "gate": GATE,
            "mode": "smoke",
            "manifest_revision": manifest_revision,
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "run_root": str(run_root),
            "git": git,
            "card_file_sha256": CARD_FILE_SHA256,
            "source_manifest_sha256": source["manifest_sha256"],
            "pool_manifest_sha256": pool_manifest["manifest_sha256"],
            "scientific_config": _scientific_config(card),
            "selection_rule": "exact_planning_authorized_b36_false_negative_trigger_identities",
            "record_count": len(rows),
            "records": rows,
            "membership_sha256": _membership_sha256(rows),
            "output_base_relative": "smoke",
            "argv": list(argv),
            "implementation_sha256": implementation_hashes(),
        }
    )
    write_new_json(_smoke_manifest_path(run_root, manifest_revision), payload)
    return payload


def freeze_formal(
    *, expected_commit: str, shard_count: int, smoke_receipt_sha256: str, argv: Sequence[str]
) -> Dict[str, Any]:
    run_root = assert_authorized_run_root(AUTHORIZED_RUN_ROOT, must_exist=True)
    if shard_count < 1 or shard_count > 1531:
        raise Phase5AcquisitionError("formal shard_count is outside 1..1531")
    git = git_provenance(expected_commit)
    card = load_frozen_card()
    pool, source, pool_manifest = load_phase3_gold_free_pool()
    smoke = load_strict_json(run_root / "smoke/smoke_receipt.json")
    verify_manifest_sha256(smoke)
    if smoke["manifest_sha256"] != smoke_receipt_sha256 or smoke.get("result") != "PASS":
        raise Phase5AcquisitionError("formal freeze lacks exact passing smoke receipt")
    shards = []
    for shard_id in range(shard_count):
        rows = [
            _membership_row(pool[index], index)
            for index in range(shard_id, len(pool), shard_count)
        ]
        shards.append(
            {
                "shard_id": shard_id,
                "record_count": len(rows),
                "records": rows,
                "membership_sha256": _membership_sha256(rows),
                "output_base_relative": "formal/shards/shard-%04d" % shard_id,
            }
        )
    payload = attach_manifest_sha256(
        {
            "schema_version": MEMBERSHIP_SCHEMA,
            "created_at_utc": utc_now(),
            "gate": GATE,
            "mode": "formal",
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "run_root": str(run_root),
            "git": git,
            "card_file_sha256": CARD_FILE_SHA256,
            "source_manifest_sha256": source["manifest_sha256"],
            "pool_manifest_sha256": pool_manifest["manifest_sha256"],
            "smoke_receipt_sha256": smoke_receipt_sha256,
            "scientific_config": _scientific_config(card),
            "distribution": "canonical_ordinal_round_robin",
            "record_count": len(pool),
            "subject_count": len({row["subject"] for row in pool}),
            "ordered_pool_identity_sha256": pool_manifest["ordered_identity_sha256"],
            "shard_count": shard_count,
            "shards": shards,
            "argv": list(argv),
            "implementation_sha256": implementation_hashes(),
        }
    )
    write_new_json(run_root / "formal/formal_membership_manifest.json", payload)
    write_new_json(run_root / "formal/formal_shard_manifest.json", payload)
    return payload


def validate_membership_manifest(
    manifest: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    verify_manifest_sha256(manifest)
    if manifest.get("schema_version") != MEMBERSHIP_SCHEMA:
        raise Phase5AcquisitionError("membership schema differs")
    if manifest.get("executor_thread_id") != EXECUTOR_THREAD_ID:
        raise Phase5AcquisitionError("membership executor differs")
    if manifest.get("run_root") != str(AUTHORIZED_RUN_ROOT):
        raise Phase5AcquisitionError("membership run root differs")
    mode = manifest.get("mode")
    groups: Sequence[Mapping[str, Any]]
    if mode == "smoke":
        groups = [manifest]
        expected_ordinals = _smoke_ordinals(pool)
    elif mode == "formal":
        groups = manifest.get("shards", [])
        if len(groups) != manifest.get("shard_count"):
            raise Phase5AcquisitionError("formal shard list/count differs")
        expected_ordinals = list(range(len(pool)))
    else:
        raise Phase5AcquisitionError("membership mode differs")
    observed: List[int] = []
    for group_index, group in enumerate(groups):
        if mode == "formal" and group.get("shard_id") != group_index:
            raise Phase5AcquisitionError("formal shard ids differ")
        rows = group.get("records")
        if not isinstance(rows, list) or len(rows) != group.get("record_count"):
            raise Phase5AcquisitionError("membership group count differs")
        if group.get("membership_sha256") != _membership_sha256(rows):
            raise Phase5AcquisitionError("membership group hash differs")
        for row in rows:
            ordinal = row.get("ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise Phase5AcquisitionError("membership ordinal is invalid")
            if ordinal < 0 or ordinal >= len(pool):
                raise Phase5AcquisitionError("membership ordinal is outside pool")
            if dict(row) != _membership_row(pool[ordinal], ordinal):
                raise Phase5AcquisitionError("membership row drifts from canonical pool")
            observed.append(ordinal)
    if observed != expected_ordinals and sorted(observed) != expected_ordinals:
        raise Phase5AcquisitionError("membership is not the exact expected union")
    if len(set(observed)) != len(observed):
        raise Phase5AcquisitionError("membership contains duplicate ordinals")
    return [dict(pool[index]) for index in observed]


def _assert_no_loop_modules_loaded() -> None:
    forbidden = ("tflt.wrapper", "tflt.strategies", "tflt.cache", "tflt.loopscope.window_probe")
    loaded = sorted(
        name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in forbidden)
    )
    if loaded:
        raise Phase5AcquisitionError("forbidden loop modules were imported: %s" % loaded)


def _active_loop_wrapper_modules(model: Any) -> List[str]:
    active = []
    for name, module in model.named_modules():
        class_module = str(getattr(module.__class__, "__module__", ""))
        if class_module == "tflt.wrapper" or class_module.startswith("tflt.wrapper."):
            active.append(str(name or "<root>"))
    forward_module = str(getattr(getattr(model, "forward", None), "__module__", ""))
    if forward_module == "tflt.wrapper" or forward_module.startswith("tflt.wrapper."):
        active.append("<model.forward>")
    return sorted(set(active))


class NativeRuntime:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)
        self.forward_calls = 0


def load_native_runtime(card: Mapping[str, Any]) -> NativeRuntime:
    _assert_no_loop_modules_loaded()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tflt.loopscope.probe import (
            choice_tokenization,
            decoder_boundary_states,
            find_final_norm,
            lens_space_hidden,
        )
        from tflt.loopscope.revisions import load_tokenizer_with_resolved_commit, strict_revision_closure
    except Exception as exc:
        raise Phase5AcquisitionError("Gate B runtime imports failed") from exc
    if not torch.cuda.is_available():
        raise Phase5AcquisitionError("Gate B acquisition requires a visible CUDA GPU")
    kwargs = {"revision": card["model"]["revision"], "local_files_only": True, "trust_remote_code": True}
    tokenizer, tokenizer_commit = load_tokenizer_with_resolved_commit(
        AutoTokenizer, card["model"]["repo"], kwargs
    )
    mapping, choice_ids, warnings = choice_tokenization(tokenizer, CHOICE_ORDER)
    if warnings:
        raise Phase5AcquisitionError("choice tokenization emitted warnings")
    spaced = {label: metadata["token_ids"] for label, metadata in mapping.items()}
    plain = {label: metadata["plain_surface_token_ids"] for label, metadata in mapping.items()}
    if spaced != {key: [value] for key, value in EXPECTED_SPACED_TOKEN_IDS.items()}:
        raise Phase5AcquisitionError("spaced choice token ids differ")
    if plain != {key: [value] for key, value in EXPECTED_PLAIN_TOKEN_IDS.items()}:
        raise Phase5AcquisitionError("plain choice token ids differ")
    model = AutoModelForCausalLM.from_pretrained(
        card["model"]["repo"], torch_dtype=torch.bfloat16, **kwargs
    )
    closure = strict_revision_closure(model, tokenizer_commit, card["model"]["revision"])
    if int(getattr(model.config, "num_hidden_layers", 0)) != 36:
        raise Phase5AcquisitionError("runtime layer count differs")
    if int(getattr(model.config, "hidden_size", 0)) != 2560:
        raise Phase5AcquisitionError("runtime hidden size differs")
    if not bool(getattr(model.config, "tie_word_embeddings", False)):
        raise Phase5AcquisitionError("runtime output embeddings are not declared tied")
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise Phase5AcquisitionError("runtime lacks logits_to_keep")
    final_norm = find_final_norm(model)
    lm_head = model.get_output_embeddings() or getattr(model, "lm_head", None)
    input_embeddings = model.get_input_embeddings()
    if lm_head is None or getattr(lm_head, "weight", None) is not getattr(input_embeddings, "weight", None):
        raise Phase5AcquisitionError("runtime LM head is not tied to input embeddings")
    model.eval()
    model.to("cuda")
    _assert_no_loop_modules_loaded()
    return NativeRuntime(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        final_norm=final_norm,
        lm_head=lm_head,
        choice_ids=list(choice_ids),
        choice_mapping=mapping,
        revision_closure=closure,
        boundary_states_fn=decoder_boundary_states,
        lens_space_fn=lens_space_hidden,
    )


def _renderer_provenance(card: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["tokenizer_revision"],
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "renderer_manifest_sha256": ANCHOR_MANIFEST_INTERNAL_SHA256,
        "forward_type": "native_no_loop",
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "position": "final_non_padding_prompt_token",
        "projection": "frozen_final_norm_then_tied_output_weight_exact_choice_rows",
        "choice_order": list(CHOICE_ORDER),
        "spaced_choice_token_ids": EXPECTED_SPACED_TOKEN_IDS,
        "final_norm_path": "model.norm",
        "output_head_path": "lm_head_tied_to_model.embed_tokens.weight",
    }


def _choice_logits(runtime: NativeRuntime, normalized_hidden: Any) -> Any:
    weight = runtime.lm_head.weight[runtime.choice_ids]
    bias = getattr(runtime.lm_head, "bias", None)
    selected_bias = None if bias is None else bias[runtime.choice_ids]
    return runtime.torch.nn.functional.linear(normalized_hidden, weight, selected_bias)[0]


def authoritative_b36_choice_logits(
    native_choice_logits: Any,
    sliced_choice_logits: Any,
    *,
    full_head_shape_matches: bool,
    full_head_closes: bool,
) -> Any:
    """Return native B36 choices after independent full-head closure.

    ``sliced_choice_logits`` is accepted only to make its diagnostic-only role
    explicit and testable.  It must never gate or supply the recorded B36
    distribution.
    """

    if not full_head_shape_matches:
        raise Phase5AcquisitionError("native B36 full-head logits shape does not close")
    if not full_head_closes:
        raise Phase5AcquisitionError("native B36 full-head logits do not close")
    _ = sliced_choice_logits
    return native_choice_logits


def normalized_boundary_views(
    states: Sequence[Any],
    *,
    position: int,
    final_norm: Any,
    lens_space_fn: Any,
    layer_count: int = 36,
    state_at_position: Optional[Any] = None,
) -> List[Any]:
    """Return B0..BL lens views while making the B_L no-double-norm rule testable."""

    if len(states) != layer_count + 1:
        raise Phase5AcquisitionError("boundary states must be exactly B_0..B_L")
    accessor = state_at_position or (lambda state, index: state[:, index, :])
    return [
        lens_space_fn(final_norm, accessor(hidden, position), boundary_index, layer_count)
        for boundary_index, hidden in enumerate(states)
    ]


def formal_retry_allowed(failure_receipt: Mapping[str, Any]) -> bool:
    """Formal retries are legal only after proven zero-forward, zero-record failure."""

    return (
        failure_receipt.get("result") == "FAILED"
        and failure_receipt.get("forward_calls") == 0
        and failure_receipt.get("records_written") == 0
        and failure_receipt.get("formal_retry_eligible") is True
    )


def acquire_one(
    runtime: NativeRuntime, pool_record: Mapping[str, Any], card: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _assert_no_loop_modules_loaded()
    if _active_loop_wrapper_modules(runtime.model):
        raise Phase5AcquisitionError("native model is wrapped by loop modules")
    prompt = str(pool_record["rendered_prompt"])
    if not prompt.rstrip().endswith("Answer:"):
        raise Phase5AcquisitionError("renderer prompt does not end at Answer:")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != pool_record["prompt_sha256"]:
        raise Phase5AcquisitionError("renderer prompt hash differs")
    encoded = runtime.tokenizer(prompt, return_tensors="pt")
    mask = encoded.get("attention_mask")
    if mask is None:
        raise Phase5AcquisitionError("tokenizer output lacks attention mask")
    valid = mask[0].detach().cpu().nonzero(as_tuple=False).flatten().tolist()
    if not valid or int(valid[-1]) != int(mask.shape[1]) - 1:
        raise Phase5AcquisitionError("answer position is padded or ambiguous")
    position = int(valid[-1])
    inputs = {key: value.to("cuda") for key, value in encoded.items()}
    torch = runtime.torch
    start = time.monotonic()
    with torch.inference_mode():
        runtime.forward_calls += 1
        outputs = runtime.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        states = runtime.boundary_states_fn(tuple(outputs.hidden_states or ()), 36)
        normalized = normalized_boundary_views(
            states,
            position=position,
            final_norm=runtime.final_norm,
            lens_space_fn=runtime.lens_space_fn,
        )
        choice_logits = [_choice_logits(runtime, lens_hidden) for lens_hidden in normalized[:-1]]
        native_full_logits = outputs.logits
        reprojected_full_logits = runtime.lm_head(normalized[-1].unsqueeze(1))
        full_head_shape_matches = tuple(native_full_logits.shape) == tuple(
            reprojected_full_logits.shape
        )
        if not full_head_shape_matches:
            authoritative_b36_choice_logits(
                None,
                None,
                full_head_shape_matches=False,
                full_head_closes=False,
            )
        native_full_float = native_full_logits.float()
        reprojected_full_float = reprojected_full_logits.float()
        if not bool(torch.isfinite(native_full_float).all().item()) or not bool(
            torch.isfinite(reprojected_full_float).all().item()
        ):
            raise Phase5AcquisitionError("native B36 full-head closure contains non-finite logits")
        full_head_native_max_abs_difference = float(
            (native_full_float - reprojected_full_float).detach().abs().max().double().cpu()
        )
        full_head_closes = bool(
            torch.allclose(
                native_full_float,
                reprojected_full_float,
                rtol=1e-3,
                atol=1e-3,
            )
        )
        native_final = native_full_logits[0, 0, runtime.choice_ids]
        sliced_final = _choice_logits(runtime, normalized[-1])
        sliced_native_max_abs_difference = float(
            (native_final - sliced_final).detach().abs().max().float().cpu()
        )
        choice_logits.append(
            authoritative_b36_choice_logits(
                native_final,
                sliced_final,
                full_head_shape_matches=full_head_shape_matches,
                full_head_closes=full_head_closes,
            )
        )
        final_hidden = normalized[-1].float()
        final_norm_sq = torch.sum(final_hidden * final_hidden, dim=-1)
        if not bool((final_norm_sq > 0).all().item()):
            raise Phase5AcquisitionError("final hidden has zero norm")
        probabilities = [
            stable_choice_probabilities(values.detach().to(dtype=torch.float64).cpu().tolist())
            for values in choice_logits
        ]
        final_probabilities = probabilities[-1]
        boundaries = []
        for boundary_index, (distribution, lens_hidden) in enumerate(zip(probabilities, normalized)):
            if boundary_index == 36:
                l2, cosine, distance = 0.0, 1.0, 0.0
                kl = 0.0
            else:
                current = lens_hidden.float()
                difference = current - final_hidden
                l2 = float(torch.sqrt(torch.mean(difference * difference, dim=-1)).cpu())
                dot = torch.sum(current * final_hidden, dim=-1)
                denom = torch.sqrt(torch.sum(current * current, dim=-1) * final_norm_sq)
                if not bool((denom > 0).all().item()):
                    raise Phase5AcquisitionError("hidden cosine denominator is zero")
                cosine = min(1.0, max(-1.0, float((dot / denom).cpu())))
                distance = 1.0 - cosine
                kl = math.fsum(
                    value * math.log(value / reference)
                    for value, reference in zip(distribution, final_probabilities)
                )
            entropy = -math.fsum(value * math.log(value) for value in distribution)
            values = (entropy, kl, l2, cosine, distance)
            if not all(math.isfinite(value) for value in values):
                raise Phase5AcquisitionError("trajectory reduction produced non-finite scalar")
            boundaries.append(
                {
                    "boundary_id": BOUNDARY_IDS[boundary_index],
                    "choice_probabilities": distribution,
                    "choice_entropy": entropy,
                    "kl_to_final": kl,
                    "hidden_l2_to_final": l2,
                    "hidden_cosine_to_final": cosine,
                    "hidden_cosine_distance_to_final": distance,
                }
            )
        record = {
            "schema_version": "loopscope.phase5.trajectory-record.v1",
            "identity": dict(pool_record["identity"]),
            "subject": str(pool_record["subject"]),
            "split": "validation",
            "prompt_sha256": str(pool_record["prompt_sha256"]),
            "renderer_provenance": _renderer_provenance(card),
            "boundaries": boundaries,
        }
        validate_trajectory_record(record)
    evidence = {
        "sequence_length": position + 1,
        "answer_position": position,
        "elapsed_seconds": time.monotonic() - start,
        "final_choice_logit_max_abs_difference": full_head_native_max_abs_difference,
        "b36_choice_logit_source": "outputs.logits[0,0,choice_ids]",
        "b36_full_head_native_shape": list(native_full_logits.shape),
        "b36_full_head_native_max_abs_difference": full_head_native_max_abs_difference,
        "b36_full_head_native_allclose": full_head_closes,
        "b36_sliced_choice_native_max_abs_difference": sliced_native_max_abs_difference,
        "b36_sliced_choice_mismatch_gating": False,
        "full_vocabulary_tensors_persisted": False,
        "boundary_count": len(boundaries),
        "same_forward_choice_and_geometry": True,
        "b36_double_norm_calls": 0,
    }
    del (
        outputs,
        states,
        normalized,
        choice_logits,
        probabilities,
        final_hidden,
        native_full_logits,
        reprojected_full_logits,
        native_full_float,
        reprojected_full_float,
        sliced_final,
        native_final,
        inputs,
        encoded,
    )
    _assert_no_loop_modules_loaded()
    return record, evidence


def _runtime_evidence(runtime: NativeRuntime, facts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    torch = runtime.torch
    device = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    return {
        "model_revision_closure": dict(runtime.revision_closure),
        "model_class": runtime.model.__class__.__name__,
        "tokenizer_class": runtime.tokenizer.__class__.__name__,
        "final_norm_class": runtime.final_norm.__class__.__name__,
        "lm_head_class": runtime.lm_head.__class__.__name__,
        "lm_head_tied_to_input_embeddings": runtime.lm_head.weight is runtime.model.get_input_embeddings().weight,
        "decoder_layers": 36,
        "hidden_size": 2560,
        "choice_token_ids": list(runtime.choice_ids),
        "dtype": "bfloat16",
        "device": "cuda",
        "use_cache": False,
        "output_hidden_states": True,
        "forward_calls": runtime.forward_calls,
        "loop_insertions": 0,
        "active_loop_wrapper_modules": _active_loop_wrapper_modules(runtime.model),
        "new_loop_modules_loaded_by_producer": [],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": math.fsum(float(row["elapsed_seconds"]) for row in facts),
        "b36_choice_logit_source": "outputs.logits[0,0,choice_ids]",
        "b36_full_head_native_max_abs_difference_max": max(
            float(row["b36_full_head_native_max_abs_difference"]) for row in facts
        ),
        "b36_sliced_choice_native_max_abs_difference_max": max(
            float(row["b36_sliced_choice_native_max_abs_difference"]) for row in facts
        ),
        "b36_sliced_choice_mismatch_gating": False,
        "full_vocabulary_tensors_persisted": False,
        "versions": {
            "python": platform.python_version(),
            "torch": str(getattr(torch, "__version__", "")),
            "transformers": _module_version("transformers"),
        },
        "cuda_device": {
            "index": device,
            "name": str(getattr(properties, "name", "")),
            "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


def _members_for_mode(
    manifest: Mapping[str, Any], pool: Sequence[Mapping[str, Any]], mode: str, shard_id: Optional[int]
) -> Tuple[List[Dict[str, Any]], Mapping[str, Any]]:
    validate_membership_manifest(manifest, pool)
    if manifest.get("mode") != mode:
        raise Phase5AcquisitionError("acquisition mode differs from manifest")
    if mode == "smoke":
        group = manifest
    else:
        if shard_id is None or shard_id < 0 or shard_id >= int(manifest["shard_count"]):
            raise Phase5AcquisitionError("formal shard_id is outside manifest")
        group = manifest["shards"][shard_id]
    return [dict(pool[row["ordinal"]]) for row in group["records"]], group


def acquire_membership(
    *,
    mode: str,
    expected_commit: str,
    argv: Sequence[str],
    shard_id: Optional[int] = None,
    attempt: int = 1,
    smoke_manifest_revision: int = 1,
) -> Dict[str, Any]:
    run_root = assert_authorized_run_root(AUTHORIZED_RUN_ROOT, must_exist=True)
    git = git_provenance(expected_commit)
    if not os.environ.get("SLURM_JOB_ID"):
        raise Phase5AcquisitionError("GPU acquisition must run inside Slurm")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise Phase5AcquisitionError("GPU acquisition must use offline cache mode")
    card = load_frozen_card()
    pool, _, _ = load_phase3_gold_free_pool()
    manifest_path = (
        _smoke_manifest_path(run_root, smoke_manifest_revision)
        if mode == "smoke"
        else run_root / "formal/formal_shard_manifest.json"
    )
    manifest = load_strict_json(manifest_path)
    members, group = _members_for_mode(manifest, pool, mode, shard_id)
    if attempt < 1:
        raise Phase5AcquisitionError("attempt must be positive")
    if (mode == "formal" and attempt > 2) or (mode == "smoke" and attempt > 3):
        raise Phase5AcquisitionError("attempt exceeds the authorized repair/retry budget")
    output_base = (run_root / str(group["output_base_relative"])).resolve()
    output_dir = output_base / ("attempt-%04d" % attempt)
    if run_root not in output_dir.parents or output_dir.exists():
        raise FileExistsError("refusing to reuse acquisition attempt: %s" % output_dir)
    if attempt > 1:
        previous = output_base / ("attempt-%04d" % (attempt - 1)) / "failure_receipt.json"
        failure = load_strict_json(previous)
        verify_manifest_sha256(failure)
        if mode == "formal" and not formal_retry_allowed(failure):
            raise Phase5AcquisitionError(
                "formal retry is forbidden after any forward or identity record"
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "records.jsonl"
    runtime: Optional[NativeRuntime] = None
    facts = []
    records = []
    try:
        runtime = load_native_runtime(card)
        with records_path.open("xb") as handle:
            for member in members:
                record, fact = acquire_one(runtime, member, card)
                body = canonical_json_bytes(record) + b"\n"
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
                records.append(record)
                facts.append(fact)
    except Exception as exc:
        failure = attach_manifest_sha256(
            {
                "schema_version": "loopscope.phase5.gate-b-failed-attempt.v1",
                "created_at_utc": utc_now(),
                "gate": GATE,
                "mode": mode,
                "shard_id": shard_id,
                "attempt": "attempt-%04d" % attempt,
                "result": "FAILED",
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "forward_calls": 0 if runtime is None else int(runtime.forward_calls),
                "records_written": len(records),
                "records_file_exists": records_path.exists(),
                "records_file_size_bytes": records_path.stat().st_size if records_path.exists() else 0,
                "formal_retry_eligible": (
                    (0 if runtime is None else int(runtime.forward_calls)) == 0
                    and len(records) == 0
                ),
                "outcome_accessed": False,
                "test_split_accessed": False,
            }
        )
        write_new_json(output_dir / "failure_receipt.json", failure)
        raise
    assert runtime is not None
    if runtime.forward_calls != len(members) or len(records) != len(members):
        raise Phase5AcquisitionError("forward/member/record count closure failed")
    receipt = attach_manifest_sha256(
        {
            "schema_version": SHARD_RECEIPT_SCHEMA,
            "created_at_utc": utc_now(),
            "gate": GATE,
            "mode": mode,
            "shard_id": shard_id,
            "attempt": "attempt-%04d" % attempt,
            "result": "COMPLETED",
            "git": git,
            "argv": list(argv),
            "membership_manifest_sha256": manifest["manifest_sha256"],
            "membership_sha256": group["membership_sha256"],
            "record_count": len(records),
            "records_file_sha256": file_sha256(records_path),
            "runtime": _runtime_evidence(runtime, facts),
            "facts": facts,
            "implementation_sha256": implementation_hashes(),
            "outcome_accessed": False,
            "test_split_accessed": False,
        }
    )
    write_new_json(output_dir / "receipt.json", receipt)
    if mode == "smoke":
        elapsed = float(receipt["runtime"]["elapsed_seconds"])
        per_sample_bytes = records_path.stat().st_size / float(len(records))
        extrapolated_hours = elapsed / len(records) * 1531 / 3600.0
        storage_bytes = per_sample_bytes * 1531
        smoke_receipt = attach_manifest_sha256(
            {
                "schema_version": "loopscope.phase5.gate-b-smoke-receipt.v1",
                "created_at_utc": utc_now(),
                "gate": GATE,
                "result": "PASS",
                "membership_manifest_sha256": manifest["manifest_sha256"],
                "attempt_receipt_sha256": receipt["manifest_sha256"],
                "records_file_sha256": receipt["records_file_sha256"],
                "record_count": len(records),
                "subject_count": len({record["subject"] for record in records}),
                "forward_count": runtime.forward_calls,
                "extrapolation": {
                    "formal_gpu_hours_serial_equivalent": extrapolated_hours,
                    "formal_storage_bytes": storage_bytes,
                    "reapproval_gpu_hours_trigger": 12.0,
                    "reapproval_storage_bytes_trigger": 500 * 1024 * 1024,
                    "within_reapproval_limits": extrapolated_hours <= 12.0 and storage_bytes <= 500 * 1024 * 1024,
                },
                "runtime": receipt["runtime"],
                "implementation_sha256": implementation_hashes(),
                "outcome_accessed": False,
                "test_split_accessed": False,
            }
        )
        if not smoke_receipt["extrapolation"]["within_reapproval_limits"]:
            raise Phase5AcquisitionError("smoke exceeds planning reapproval trigger")
        write_new_json(run_root / "smoke/smoke_receipt.json", smoke_receipt)
        write_new_jsonl(run_root / "smoke/smoke_trajectory_records.jsonl", records)
        return smoke_receipt
    return receipt


def merge_formal(*, expected_commit: str, argv: Sequence[str]) -> Dict[str, Any]:
    run_root = assert_authorized_run_root(AUTHORIZED_RUN_ROOT, must_exist=True)
    git = git_provenance(expected_commit)
    pool, _, pool_manifest = load_phase3_gold_free_pool()
    manifest = load_strict_json(run_root / "formal/formal_shard_manifest.json")
    validate_membership_manifest(manifest, pool)
    if manifest.get("mode") != "formal":
        raise Phase5AcquisitionError("formal shard manifest mode differs")
    by_ordinal: Dict[int, Dict[str, Any]] = {}
    shard_receipts = []
    for shard in manifest["shards"]:
        output_base = run_root / shard["output_base_relative"]
        attempts = sorted(output_base.glob("attempt-*/receipt.json"))
        if len(attempts) != 1:
            raise Phase5AcquisitionError("formal shard must have exactly one completed attempt")
        output_dir = attempts[0].parent
        receipt = load_strict_json(output_dir / "receipt.json")
        verify_manifest_sha256(receipt)
        if receipt.get("result") != "COMPLETED" or receipt.get("record_count") != shard["record_count"]:
            raise Phase5AcquisitionError("formal shard is not terminal complete")
        if receipt.get("membership_sha256") != shard["membership_sha256"]:
            raise Phase5AcquisitionError("formal shard membership hash differs")
        records_path = output_dir / "records.jsonl"
        if file_sha256(records_path) != receipt.get("records_file_sha256"):
            raise Phase5AcquisitionError("formal shard records hash differs")
        records = load_strict_jsonl(records_path)
        if len(records) != len(shard["records"]):
            raise Phase5AcquisitionError("formal shard record count differs")
        for member, record in zip(shard["records"], records):
            validate_trajectory_record(record)
            if record["identity"] != member["identity"] or record["subject"] != member["subject"]:
                raise Phase5AcquisitionError("formal shard record identity differs")
            ordinal = int(member["ordinal"])
            if ordinal in by_ordinal:
                raise Phase5AcquisitionError("duplicate formal ordinal")
            by_ordinal[ordinal] = record
        shard_receipts.append(receipt["manifest_sha256"])
    if sorted(by_ordinal) != list(range(1531)):
        raise Phase5AcquisitionError("formal merge is missing or has extra ordinals")
    records = [by_ordinal[index] for index in range(1531)]
    if len({record["subject"] for record in records}) != 57:
        raise Phase5AcquisitionError("formal merge subject count differs")
    output = run_root / "formal/validation1531_phase5_trajectories.jsonl"
    output_sha = write_new_jsonl(output, records)
    trajectory_manifest = attach_manifest_sha256(
        {
            "schema_version": TRAJECTORY_MANIFEST_SCHEMA,
            "created_at_utc": utc_now(),
            "gate": GATE,
            "result": "COMPLETE",
            "git": git,
            "argv": list(argv),
            "record_count": 1531,
            "subject_count": 57,
            "forward_count": sum(
                int(load_strict_json(sorted((run_root / shard["output_base_relative"]).glob("attempt-*/receipt.json"))[0])["runtime"]["forward_calls"])
                for shard in manifest["shards"]
            ),
            "trajectory_file": str(output.relative_to(run_root)),
            "trajectory_file_sha256": output_sha,
            "formal_membership_manifest_sha256": manifest["manifest_sha256"],
            "source_pool_ordered_identity_sha256": pool_manifest["ordered_identity_sha256"],
            "ordered_identity_sha256": hashlib.sha256(canonical_json_bytes([record["identity"] for record in records])).hexdigest(),
            "shard_receipt_manifest_sha256": shard_receipts,
            "implementation_sha256": implementation_hashes(),
            "same_forward_choice_and_geometry": True,
            "outcome_accessed": False,
            "test_split_accessed": False,
        }
    )
    if trajectory_manifest["forward_count"] != 1531:
        raise Phase5AcquisitionError("formal forward count does not close to 1531")
    write_new_json(run_root / "formal/validation1531_phase5_trajectory_manifest.json", trajectory_manifest)
    return trajectory_manifest


def verify_trajectory_population() -> Dict[str, Any]:
    run_root = assert_authorized_run_root(AUTHORIZED_RUN_ROOT, must_exist=True)
    manifest = load_strict_json(run_root / "formal/validation1531_phase5_trajectory_manifest.json")
    verify_manifest_sha256(manifest)
    records_path = run_root / manifest["trajectory_file"]
    if file_sha256(records_path) != manifest["trajectory_file_sha256"]:
        raise Phase5AcquisitionError("merged trajectory file hash differs")
    records = load_strict_jsonl(records_path)
    if len(records) != 1531 or len({record.get("subject") for record in records}) != 57:
        raise Phase5AcquisitionError("merged trajectory population differs")
    identities = []
    for record in records:
        validate_trajectory_record(record)
        identities.append(_canonical_identity(record["identity"]))
    if len(set(identities)) != 1531:
        raise Phase5AcquisitionError("merged trajectory identities are not unique")
    return attach_manifest_sha256(
        {
            "schema_version": "loopscope.phase5.trajectory-verifier-receipt.v1",
            "created_at_utc": utc_now(),
            "gate": GATE,
            "result": "PASS",
            "trajectory_manifest_sha256": manifest["manifest_sha256"],
            "trajectory_file_sha256": manifest["trajectory_file_sha256"],
            "record_count": 1531,
            "subject_count": 57,
            "identity_count": len(set(identities)),
            "all_records_finite_and_final_closed": True,
            "outcome_accessed": False,
            "test_split_accessed": False,
        }
    )


__all__ = [
    "AUTHORIZED_RUN_ROOT",
    "AUDITED_VENV",
    "EXECUTOR_THREAD_ID",
    "MEMBERSHIP_SCHEMA",
    "MODEL_SNAPSHOT",
    "NativeRuntime",
    "Phase5AcquisitionError",
    "REMOTE_REPO",
    "WORKSPACE",
    "acquire_membership",
    "acquire_one",
    "assert_authorized_run_root",
    "attach_manifest_sha256",
    "freeze_formal",
    "freeze_smoke",
    "formal_retry_allowed",
    "git_provenance",
    "implementation_hashes",
    "load_frozen_card",
    "load_phase3_gold_free_pool",
    "load_strict_json",
    "load_strict_jsonl",
    "materialize_admission",
    "merge_formal",
    "normalized_boundary_views",
    "validate_membership_manifest",
    "verify_manifest_sha256",
    "verify_trajectory_population",
    "write_new_json",
    "write_new_jsonl",
    "write_new_text",
]
