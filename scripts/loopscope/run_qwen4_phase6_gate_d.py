#!/usr/bin/env python3
"""Shard, acquire, merge, and verify LoopScope Phase 6 Gate D trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tflt.loopscope.phase4_runtime import (  # noqa: E402
    CATEGORIES,
    project_target_row,
    project_test_dataset,
    render_exact_prefix,
    tokenization_metadata,
    validation_demos_by_category,
)
from tflt.loopscope.phase4_schema import (  # noqa: E402
    canonical_identity,
    canonical_json_bytes,
    file_sha256,
    sanitized_content_sha256,
    semantic_sha256,
)
from tflt.loopscope.phase6_runtime import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    GENERATION_MAX_NEW_TOKENS,
    GENERATION_STOP_STRING,
    MODEL_REPO,
    MODEL_REVISION,
    PRODUCER_VERSION,
    acquire_two_pass_record_and_payload,
    load_gate_c_runtime,
)
from tflt.loopscope.phase6_schema import (  # noqa: E402
    load_json,
    scan_forbidden_fields,
    validate_card,
    validate_trajectory_record,
)


EXECUTOR_THREAD_ID = "019fb6d3-9485-7f33-90c1-ece73c3863bc"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
CARD_PATH = REPO_ROOT / "configs/loopscope/phase6_pre_answer_v2_card.json"
CARD_SHA256 = "feef7250b72fa8ecd154d58835ad42eacc1fc96509ed984acdafbfa4c20432d4"
GATE_B_MANIFEST_SHA256 = (
    "ccb148bd61971d24ad81b240cbe1b5f0810e06e620fb568473a61b2147b44a1d"
)
ORDERED_IDENTITY_SHA256 = (
    "ac52d6e43c693bd0b47b567c2955dae0c6be895ce3230e854d4e1538f05503a5"
)
ORDERED_SAFE_TOKEN_CLOSURE_SHA256 = (
    "0d7e9ef3139f687fd3f15d825e883e52d1f84a504cae134ad4b17ebd6ad82108"
)
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")

POPULATION_SCHEMA = "loopscope.phase6.gate-d-population.v1"
SHARD_MANIFEST_SCHEMA = "loopscope.phase6.gate-d-shard-manifest.v1"
SHARD_RECEIPT_SCHEMA = "loopscope.phase6.gate-d-shard-receipt.v1"
SHARD_VERIFIER_SCHEMA = "loopscope.phase6.gate-d-shard-verifier.v1"
MERGED_MANIFEST_SCHEMA = "loopscope.phase6.gate-d-merged-manifest.v1"
FINAL_VERIFIER_SCHEMA = "loopscope.phase6.gate-d-final-verifier.v1"
SCHEDULER_RECEIPT_SCHEMA = "loopscope.phase6.gate-d-scheduler.v1"

SANITIZED_NAME = "sanitized_records.jsonl"
CLOSURE_NAME = "runtime_closure.jsonl"
SEALED_MEMBERSHIP_NAME = "sealed_membership.json"
RECEIPT_NAME = "receipt.json"
VERIFIER_NAME = "verifier_receipt.json"


class GateDError(RuntimeError):
    """Gate D failed a frozen sharding, acquisition, or verification contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateDError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        list(command), cwd=str(cwd), check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def _strict_json(path: Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    GateDError("non-finite JSON constant: %s" % value)
                ),
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateDError("cannot load strict JSON artifact: %s" % path) from exc


def _strict_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                _require(bool(line.strip()), "blank JSONL line at %s:%d" % (path, line_number))
                value = json.loads(
                    line,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        GateDError("non-finite JSON constant: %s" % token)
                    ),
                )
                _require(isinstance(value, dict), "JSONL row is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateDError("cannot load strict JSONL artifact: %s" % path) from exc
    _require(bool(rows), "JSONL artifact is empty: %s" % path)
    return rows


def _write_new_bytes(path: Path, body: bytes, *, mode: int = 0o644) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite artifact: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    return hashlib.sha256(body).hexdigest()


def _write_new_json(path: Path, payload: Any) -> str:
    body = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _write_new_bytes(path, body)


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    _require(bool(rows), "refusing to write empty JSONL")
    body = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    return _write_new_bytes(path, body)


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    return semantic_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = Path(_run(("git", "rev-parse", "--show-toplevel"))).resolve()
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"))
    commit = _run(("git", "rev-parse", "HEAD"))
    origin = _run(("git", "rev-parse", "origin/loopscope"))
    dirty = _run(("git", "status", "--porcelain"))
    _require(root == REPO_ROOT, "not running in the dedicated LoopScope clone")
    _require(branch == "loopscope", "branch differs from loopscope")
    _require(commit == expected_commit, "HEAD differs from expected commit")
    _require(origin == expected_commit, "origin/loopscope differs from expected commit")
    _require(dirty == "", "HPC2 dedicated clone is dirty")
    return {
        "repo": str(root),
        "branch": branch,
        "commit": commit,
        "origin_loopscope": origin,
        "dirty": False,
    }


def _assert_offline_environment() -> None:
    required = {
        "HF_HOME": str(HF_HOME),
        "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
        "HF_DATASETS_CACHE": str(HF_DATASETS_CACHE),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    for key, expected in required.items():
        _require(os.environ.get(key) == expected, "offline environment differs: %s" % key)
    _require(not os.environ.get("HF_ENDPOINT"), "HF_ENDPOINT substitution is forbidden")


def _load_card() -> Dict[str, Any]:
    _require(file_sha256(CARD_PATH) == CARD_SHA256, "frozen Phase 6 card hash differs")
    card = load_json(CARD_PATH)
    validate_card(card)
    return card


def _load_gate_b_manifest(path: Path) -> Dict[str, Any]:
    _require(path.is_file(), "Gate B manifest is missing")
    _require(file_sha256(path) == GATE_B_MANIFEST_SHA256, "Gate B manifest hash differs")
    value = load_json(path)
    identity = value.get("identity")
    _require(isinstance(identity, Mapping), "Gate B identity closure is missing")
    _require(
        identity.get("record_count") == 12032
        and identity.get("unique_count") == 12032,
        "Gate B population count differs",
    )
    _require(
        identity.get("ordered_identity_sha256") == ORDERED_IDENTITY_SHA256,
        "Gate B ordered identity hash differs",
    )
    _require(
        identity.get("ordered_safe_token_closure_sha256")
        == ORDERED_SAFE_TOKEN_CLOSURE_SHA256,
        "Gate B safe token closure hash differs",
    )
    _require(
        not value.get("target_labels_accessed")
        and not value.get("outcomes_read")
        and not value.get("generation_or_replay_executed"),
        "Gate B information-barrier closure differs",
    )
    return value


def _load_dataset(split: str) -> Any:
    from datasets import DownloadMode, load_dataset

    dataset = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split=split,
        cache_dir=str(HF_DATASETS_CACHE),
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    _require(
        len(dataset) == (12032 if split == "test" else 70),
        "cached %s split count differs" % split,
    )
    return dataset


def _safe_rows(projected: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for raw in projected:
        target = project_target_row(raw)
        content_hash = sanitized_content_sha256(
            target["question"], target["ordered_options"]
        )
        rows.append(
            {
                **target,
                "safe_content_sha256": content_hash,
                "canonical_identity": canonical_identity(target),
            }
        )
    rows.sort(
        key=lambda row: (
            int(str(row["question_id"]).strip()),
            row["safe_content_sha256"],
        )
    )
    return rows


def _membership_from_live(tokenizer: Any) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    from lm_eval.tasks.mmlu_pro import utils

    validation = _load_dataset("validation")
    demos = validation_demos_by_category(validation)
    rows = _safe_rows(project_test_dataset(_load_dataset("test")))
    _require(len(rows) == 12032, "canonical population count differs")
    _require(
        len({row["canonical_identity"] for row in rows}) == 12032,
        "canonical identities are not unique",
    )
    members = []
    for ordinal, row in enumerate(rows):
        target = {
            key: row[key]
            for key in ("question_id", "category", "src", "question", "ordered_options")
        }
        prefix = render_exact_prefix(target, demos[row["category"]], utils)
        metadata = tokenization_metadata(tokenizer, prefix)
        members.append(
            {
                "ordinal": ordinal,
                "canonical_identity": row["canonical_identity"],
                "category": row["category"],
                "question_id": int(str(row["question_id"]).strip()),
                "safe_content_sha256": row["safe_content_sha256"],
                "rendered_prefix_sha256": metadata["rendered_prefix_sha256"],
                "rendered_token_ids_sha256": metadata["rendered_token_ids_sha256"],
                "sequence_length": int(metadata["sequence_length"]),
            }
        )
    identities = [row["canonical_identity"] for row in members]
    token_closures = [
        {
            key: row[key]
            for key in (
                "ordinal",
                "canonical_identity",
                "category",
                "safe_content_sha256",
                "rendered_prefix_sha256",
                "rendered_token_ids_sha256",
                "sequence_length",
            )
        }
        for row in members
    ]
    _require(
        semantic_sha256(identities) == ORDERED_IDENTITY_SHA256,
        "live ordered identity hash differs from Gate B",
    )
    _require(
        semantic_sha256(token_closures) == ORDERED_SAFE_TOKEN_CLOSURE_SHA256,
        "live safe token closure differs from Gate B",
    )
    return members, rows


def _load_tokenizer_only() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=True,
    )


def _build_population_manifest(
    members: Sequence[Mapping[str, Any]], *, run_root: Path, git: Mapping[str, Any]
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": POPULATION_SCHEMA,
        "gate": "D",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(run_root),
        "created_at_utc": _utc_now(),
        "git": dict(git),
        "card_sha256": CARD_SHA256,
        "gate_b_manifest_sha256": GATE_B_MANIFEST_SHA256,
        "distribution_source": "gate_b_canonical_order",
        "record_count": len(members),
        "unique_count": len({row["canonical_identity"] for row in members}),
        "ordered_identity_sha256": semantic_sha256(
            [row["canonical_identity"] for row in members]
        ),
        "ordered_safe_token_closure_sha256": semantic_sha256(
            [
                {
                    key: row[key]
                    for key in (
                        "ordinal",
                        "canonical_identity",
                        "category",
                        "safe_content_sha256",
                        "rendered_prefix_sha256",
                        "rendered_token_ids_sha256",
                        "sequence_length",
                    )
                }
                for row in members
            ]
        ),
        "category_counts": dict(
            sorted(Counter(str(row["category"]) for row in members).items())
        ),
        "target_labels_accessed": False,
        "outcomes_read": False,
        "selector_executed": False,
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def _build_shard_manifest(
    members: Sequence[Mapping[str, Any]],
    *,
    run_root: Path,
    shard_count: int,
    population_manifest_sha256: str,
    expected_commit: str,
) -> Dict[str, Any]:
    _require(
        isinstance(shard_count, int)
        and not isinstance(shard_count, bool)
        and 8 <= shard_count <= 32,
        "formal shard count must be in [8,32]",
    )
    shards = []
    for shard_id in range(shard_count):
        shard_members = [dict(row) for row in members[shard_id::shard_count]]
        shards.append(
            {
                "shard_id": shard_id,
                "record_count": len(shard_members),
                "membership_sha256": semantic_sha256(shard_members),
                "members": shard_members,
                "output_relative": "formal/shards/shard-%04d" % shard_id,
            }
        )
    payload: Dict[str, Any] = {
        "schema_version": SHARD_MANIFEST_SCHEMA,
        "gate": "D",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(run_root),
        "distribution": "canonical_ordinal_modulo_shard_count",
        "record_count": len(members),
        "shard_count": shard_count,
        "max_concurrent_gpus": min(8, shard_count),
        "ordered_identity_sha256": ORDERED_IDENTITY_SHA256,
        "population_manifest_sha256": population_manifest_sha256,
        "expected_commit": expected_commit,
        "shards": shards,
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def _launcher_text(
    *,
    mode: str,
    run_root: Path,
    gate_b_manifest_path: Path,
    expected_commit: str,
) -> str:
    runner = REPO_ROOT / "scripts/loopscope/run_qwen4_phase6_gate_d.py"
    exports = [
        "#!/bin/bash",
        "set -euo pipefail",
        "source %s/bin/activate" % AUDITED_VENV,
        "export HF_HOME=%s" % HF_HOME,
        "export TRANSFORMERS_CACHE=%s" % (HF_HOME / "hub"),
        "export HF_DATASETS_CACHE=%s" % HF_DATASETS_CACHE,
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export HF_DATASETS_OFFLINE=1",
        "unset HF_ENDPOINT",
        "cd %s" % REPO_ROOT,
    ]
    common = (
        "--run-root %s --gate-b-manifest %s --expected-commit %s"
        % (run_root, gate_b_manifest_path, expected_commit)
    )
    if mode == "debug":
        commands = [
            "python %s acquire-debug %s" % (runner, common),
            "python %s verify-debug %s" % (runner, common),
        ]
    else:
        commands = [
            'test -n "${SLURM_ARRAY_TASK_ID:-}"',
            "python %s acquire-shard %s --shard-id ${SLURM_ARRAY_TASK_ID} --attempt 1"
            % (runner, common),
            "python %s verify-shard %s --shard-id ${SLURM_ARRAY_TASK_ID} --attempt 1"
            % (runner, common),
        ]
    return "\n".join(exports + commands) + "\n"


def prepare_debug(
    *,
    run_root: Path,
    gate_b_manifest_path: Path,
    expected_commit: str,
) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    _load_card()
    gate_b = _load_gate_b_manifest(gate_b_manifest_path)
    _require(not run_root.exists(), "debug run root must be fresh")
    run_root.mkdir(parents=True, exist_ok=False)
    smoke = gate_b["identity"]["smoke_identities"]
    _require(
        isinstance(smoke, list)
        and len(smoke) == 4
        and len({row["canonical_identity"] for row in smoke}) == 4,
        "Gate B smoke membership differs",
    )
    membership = {
        "schema_version": "loopscope.phase6.gate-d-debug-membership.v1",
        "gate": "D",
        "run_root": str(run_root),
        "git": git,
        "card_sha256": CARD_SHA256,
        "gate_b_manifest_sha256": GATE_B_MANIFEST_SHA256,
        "record_count": 4,
        "members": smoke,
    }
    membership["manifest_sha256"] = _manifest_hash(membership)
    membership_sha = _write_new_json(run_root / "debug/membership.json", membership)
    launcher = _launcher_text(
        mode="debug",
        run_root=run_root,
        gate_b_manifest_path=gate_b_manifest_path,
        expected_commit=expected_commit,
    )
    launcher_path = run_root / "control/run_debug.sh"
    launcher_sha = _write_new_bytes(launcher_path, launcher.encode("utf-8"), mode=0o755)
    os.chmod(launcher_path, 0o755)
    return {
        "status": "DEBUG_PREPARED",
        "run_root": str(run_root),
        "membership_sha256": membership_sha,
        "launcher_sha256": launcher_sha,
    }


def freeze_formal(
    *,
    run_root: Path,
    gate_b_manifest_path: Path,
    expected_commit: str,
    shard_count: int,
) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    _load_card()
    _load_gate_b_manifest(gate_b_manifest_path)
    _require(not run_root.exists(), "formal run root must be fresh")
    tokenizer = _load_tokenizer_only()
    members, _rows = _membership_from_live(tokenizer)
    run_root.mkdir(parents=True, exist_ok=False)
    population_sha = _write_new_jsonl(run_root / "source/population.jsonl", members)
    population = _build_population_manifest(members, run_root=run_root, git=git)
    population["population_file_sha256"] = population_sha
    population["manifest_sha256"] = _manifest_hash(population)
    population_manifest_sha = _write_new_json(
        run_root / "source/population_manifest.json", population
    )
    shards = _build_shard_manifest(
        members,
        run_root=run_root,
        shard_count=shard_count,
        population_manifest_sha256=population_manifest_sha,
        expected_commit=expected_commit,
    )
    shard_manifest_sha = _write_new_json(
        run_root / "formal/shard_manifest.json", shards
    )
    launcher = _launcher_text(
        mode="formal",
        run_root=run_root,
        gate_b_manifest_path=gate_b_manifest_path,
        expected_commit=expected_commit,
    )
    launcher_path = run_root / "control/run_formal_array.sh"
    launcher_sha = _write_new_bytes(launcher_path, launcher.encode("utf-8"), mode=0o755)
    os.chmod(launcher_path, 0o755)
    freeze = {
        "schema_version": "loopscope.phase6.gate-d-freeze.v1",
        "status": "FROZEN_BEFORE_FORMAL_SUBMISSION",
        "created_at_utc": _utc_now(),
        "git": git,
        "record_count": 12032,
        "shard_count": shard_count,
        "max_concurrent_gpus": min(8, shard_count),
        "ordered_identity_sha256": ORDERED_IDENTITY_SHA256,
        "population_file_sha256": population_sha,
        "population_manifest_file_sha256": population_manifest_sha,
        "shard_manifest_file_sha256": shard_manifest_sha,
        "launcher_file_sha256": launcher_sha,
    }
    freeze["manifest_sha256"] = _manifest_hash(freeze)
    freeze_sha = _write_new_json(run_root / "formal/freeze_receipt.json", freeze)
    return {
        "status": "FORMAL_FROZEN",
        "run_root": str(run_root),
        "record_count": 12032,
        "shard_count": shard_count,
        "freeze_receipt_sha256": freeze_sha,
    }


def _live_prefixes_for_members(
    members: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[tuple[Mapping[str, Any], str]]:
    from lm_eval.tasks.mmlu_pro import utils

    validation = _load_dataset("validation")
    demos = validation_demos_by_category(validation)
    projected = _safe_rows(project_test_dataset(_load_dataset("test")))
    by_ordinal = {ordinal: row for ordinal, row in enumerate(projected)}
    result = []
    for member in members:
        ordinal = int(member["ordinal"])
        _require(ordinal in by_ordinal, "membership ordinal is outside population")
        row = by_ordinal[ordinal]
        _require(
            row["canonical_identity"] == member["canonical_identity"]
            and row["safe_content_sha256"] == member["safe_content_sha256"]
            and row["category"] == member["category"],
            "live safe identity differs at ordinal %d" % ordinal,
        )
        target = {
            key: row[key]
            for key in ("question_id", "category", "src", "question", "ordered_options")
        }
        prefix = render_exact_prefix(target, demos[row["category"]], utils)
        metadata = tokenization_metadata(tokenizer, prefix)
        _require(
            metadata["rendered_prefix_sha256"] == member["rendered_prefix_sha256"]
            and metadata["rendered_token_ids_sha256"]
            == member["rendered_token_ids_sha256"]
            and int(metadata["sequence_length"]) == int(member["sequence_length"]),
            "live rendered prefix closure differs at ordinal %d" % ordinal,
        )
        result.append((member, prefix))
    return result


def _resource_evidence(runtime: Any) -> Dict[str, Any]:
    torch = runtime.torch
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_total_memory_bytes": int(properties.total_memory),
        "cpu_count": os.cpu_count(),
    }


def _safe_runtime_closure(ordinal: int, evidence: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "canonical_identity",
        "prompt_sha256",
        "generated_completion_sha256",
        "generation_ids_sha256",
        "replay_ids_sha256",
        "generation_length",
        "replay_length",
        "anchor_token_index",
        "answer_span_start_offset",
        "answer_span_end_offset",
        "answer_match_count",
        "selected_match_ordinal",
        "answer_first_token_index",
        "generation_count",
        "replay_count",
        "loop_insertions",
        "anchor_resolved",
        "unique_token_mapping",
        "record_semantic_sha256",
        "final_norm_pre_hook_count",
        "raw_boundary_count",
        "final_norm_postnorm_allclose",
        "final_norm_postnorm_max_abs",
        "replay_next_token_closure",
        "replay_sequence_length",
    )
    closure = {"ordinal": ordinal}
    closure.update({key: evidence[key] for key in keys})
    return closure


def _acquire_members(
    *,
    output: Path,
    members: Sequence[Mapping[str, Any]],
    membership_sha256: str,
    expected_commit: str,
    gate_b_manifest_path: Path,
    mode: str,
    shard_id: Optional[int],
    attempt: int,
) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    _load_card()
    _load_gate_b_manifest(gate_b_manifest_path)
    _require(not output.exists(), "fresh acquisition attempt path already exists")
    output.mkdir(parents=True, exist_ok=False)
    sealed_dir = output / "sealed_baseline"
    sealed_dir.mkdir(mode=0o700)
    os.chmod(sealed_dir, 0o700)
    runtime = load_gate_c_runtime()
    prefixes = _live_prefixes_for_members(members, runtime.tokenizer)
    records = []
    closures = []
    sealed = []
    for member, prefix in prefixes:
        record, evidence, opaque_payload = acquire_two_pass_record_and_payload(
            runtime,
            prefix=prefix,
            identity=member,
            gate_b_manifest_sha256=GATE_B_MANIFEST_SHA256,
            card_sha256=CARD_SHA256,
        )
        validate_trajectory_record(record)
        scan_forbidden_fields(record)
        ordinal = int(member["ordinal"])
        payload_relative = "sealed_baseline/%06d.bin" % ordinal
        payload_path = output / payload_relative
        payload_sha = _write_new_bytes(payload_path, opaque_payload, mode=0o600)
        os.chmod(payload_path, 0o600)
        _require(
            payload_sha == record["generated_completion_sha256"],
            "opaque payload hash differs from sanitized record",
        )
        records.append(record)
        closures.append(_safe_runtime_closure(ordinal, evidence))
        sealed.append(
            {
                "ordinal": ordinal,
                "canonical_identity": member["canonical_identity"],
                "payload_relative": payload_relative,
                "byte_count": len(opaque_payload),
                "sha256": payload_sha,
            }
        )
        runtime.torch.cuda.empty_cache()
    records_sha = _write_new_jsonl(output / SANITIZED_NAME, records)
    closures_sha = _write_new_jsonl(output / CLOSURE_NAME, closures)
    sealed_payload = {
        "schema_version": "loopscope.phase6.gate-d-sealed-membership.v1",
        "payload_format": "opaque_utf8_completion_bytes_not_parsed_before_gate_g",
        "record_count": len(sealed),
        "membership_sha256": membership_sha256,
        "members": sealed,
    }
    sealed_payload["manifest_sha256"] = _manifest_hash(sealed_payload)
    sealed_sha = _write_new_json(output / SEALED_MEMBERSHIP_NAME, sealed_payload)
    receipt: Dict[str, Any] = {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "status": "COMPLETED",
        "gate": "D",
        "mode": mode,
        "shard_id": shard_id,
        "attempt": attempt,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "completed_at_utc": _utc_now(),
        "git": git,
        "card_sha256": CARD_SHA256,
        "gate_b_manifest_sha256": GATE_B_MANIFEST_SHA256,
        "membership_sha256": membership_sha256,
        "record_count": len(records),
        "ordered_identity_sha256": semantic_sha256(
            [record["canonical_identity"] for record in records]
        ),
        "ordered_record_sha256": semantic_sha256(records),
        "artifacts": {
            SANITIZED_NAME: records_sha,
            CLOSURE_NAME: closures_sha,
            SEALED_MEMBERSHIP_NAME: sealed_sha,
        },
        "runtime": _resource_evidence(runtime),
        "closure": {
            "generation_count_each": 1,
            "replay_count_each": 1,
            "loop_insertions_each": 0,
            "anchor_resolved_count": len(records),
            "selected_match_ordinal_zero_count": len(records),
            "replay_next_token_closure_count": len(records),
            "raw_boundary_count_each": 37,
            "angular_transition_count_each": 36,
            "final_norm_pre_hook_count_each": 1,
        },
        "information_barrier": {
            "protected_target_fields_accessed": False,
            "sealed_payload_content_inspected_after_write": False,
            "selector_executed": False,
            "outcomes_read": False,
            "later_gate_entered": False,
        },
    }
    receipt["manifest_sha256"] = _manifest_hash(receipt)
    receipt_sha = _write_new_json(output / RECEIPT_NAME, receipt)
    return {
        "status": "COMPLETED",
        "mode": mode,
        "shard_id": shard_id,
        "attempt": attempt,
        "record_count": len(records),
        "receipt_sha256": receipt_sha,
    }


def _validate_attempt(
    *,
    output: Path,
    members: Sequence[Mapping[str, Any]],
    membership_sha256: str,
    expected_commit: str,
    mode: str,
    shard_id: Optional[int],
    attempt: int,
) -> Dict[str, Any]:
    receipt = _strict_json(output / RECEIPT_NAME)
    _require(receipt.get("schema_version") == SHARD_RECEIPT_SCHEMA, "receipt schema differs")
    _require(receipt.get("status") == "COMPLETED", "attempt is not completed")
    _require(
        receipt.get("mode") == mode
        and receipt.get("shard_id") == shard_id
        and receipt.get("attempt") == attempt,
        "attempt identity differs",
    )
    _require(receipt.get("git", {}).get("commit") == expected_commit, "attempt commit differs")
    _require(receipt.get("membership_sha256") == membership_sha256, "membership hash differs")
    _require(receipt.get("manifest_sha256") == _manifest_hash(receipt), "receipt hash differs")
    records_path = output / SANITIZED_NAME
    closures_path = output / CLOSURE_NAME
    sealed_path = output / SEALED_MEMBERSHIP_NAME
    for name, path in (
        (SANITIZED_NAME, records_path),
        (CLOSURE_NAME, closures_path),
        (SEALED_MEMBERSHIP_NAME, sealed_path),
    ):
        _require(path.is_file(), "attempt artifact is missing: %s" % name)
        _require(
            receipt["artifacts"][name] == file_sha256(path),
            "attempt artifact byte hash differs: %s" % name,
        )
    records = _strict_jsonl(records_path)
    closures = _strict_jsonl(closures_path)
    sealed = _strict_json(sealed_path)
    _require(
        len(records) == len(closures) == len(members) == sealed.get("record_count"),
        "attempt artifact counts differ",
    )
    expected_identities = [row["canonical_identity"] for row in members]
    _require(
        [record.get("canonical_identity") for record in records] == expected_identities,
        "sanitized record order differs",
    )
    _require(
        [row.get("canonical_identity") for row in closures] == expected_identities,
        "runtime closure order differs",
    )
    sealed_members = sealed.get("members")
    _require(isinstance(sealed_members, list), "sealed membership rows are missing")
    _require(
        [row.get("canonical_identity") for row in sealed_members] == expected_identities,
        "sealed membership order differs",
    )
    _require(
        sealed.get("membership_sha256") == membership_sha256
        and sealed.get("manifest_sha256") == _manifest_hash(sealed),
        "sealed membership closure differs",
    )
    for member, record, closure, sealed_row in zip(
        members, records, closures, sealed_members
    ):
        validate_trajectory_record(record)
        scan_forbidden_fields(record)
        _require(
            record["canonical_identity"] == member["canonical_identity"]
            and int(closure["ordinal"]) == int(member["ordinal"])
            and int(sealed_row["ordinal"]) == int(member["ordinal"]),
            "attempt member identity/ordinal differs",
        )
        _require(
            record["generation_count"] == 1
            and record["replay_count"] == 1
            and record["loop_insertions"] == 0
            and record["answer_match_count"] >= 1
            and record["selected_match_ordinal"] == 0,
            "sanitized generation/anchor closure differs",
        )
        _require(
            closure["generation_count"] == 1
            and closure["replay_count"] == 1
            and closure["loop_insertions"] == 0
            and closure["anchor_resolved"] is True
            and closure["unique_token_mapping"] is True
            and closure["replay_next_token_closure"] is True
            and closure["final_norm_pre_hook_count"] == 1
            and closure["raw_boundary_count"] == 37,
            "runtime closure differs",
        )
        _require(
            closure["record_semantic_sha256"] == semantic_sha256(record),
            "record semantic hash differs",
        )
        payload_relative = Path(str(sealed_row["payload_relative"]))
        _require(
            not payload_relative.is_absolute()
            and ".." not in payload_relative.parts
            and payload_relative.parts[:1] == ("sealed_baseline",),
            "sealed payload path escapes attempt",
        )
        payload_path = output / payload_relative
        _require(payload_path.is_file(), "sealed payload is missing")
        _require(
            stat.S_IMODE(payload_path.stat().st_mode) == 0o600,
            "sealed payload mode differs from 0600",
        )
        payload_sha = file_sha256(payload_path)
        _require(
            payload_sha
            == sealed_row["sha256"]
            == record["generated_completion_sha256"],
            "opaque sealed payload hash differs",
        )
        _require(
            payload_path.stat().st_size == int(sealed_row["byte_count"]),
            "opaque sealed payload byte count differs",
        )
    _require(
        not any(bool(value) for value in receipt["information_barrier"].values()),
        "attempt information barrier differs",
    )
    return {
        "records": records,
        "closures": closures,
        "sealed_members": sealed_members,
        "receipt": receipt,
    }


def acquire_debug(
    *, run_root: Path, gate_b_manifest_path: Path, expected_commit: str
) -> Dict[str, Any]:
    membership = _strict_json(run_root / "debug/membership.json")
    _require(
        membership.get("manifest_sha256") == _manifest_hash(membership),
        "debug membership hash differs",
    )
    return _acquire_members(
        output=run_root / "debug/attempt-0001",
        members=membership["members"],
        membership_sha256=membership["manifest_sha256"],
        expected_commit=expected_commit,
        gate_b_manifest_path=gate_b_manifest_path,
        mode="debug",
        shard_id=None,
        attempt=1,
    )


def verify_debug(
    *, run_root: Path, gate_b_manifest_path: Path, expected_commit: str
) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    _load_card()
    _load_gate_b_manifest(gate_b_manifest_path)
    membership = _strict_json(run_root / "debug/membership.json")
    output = run_root / "debug/attempt-0001"
    verified = _validate_attempt(
        output=output,
        members=membership["members"],
        membership_sha256=membership["manifest_sha256"],
        expected_commit=expected_commit,
        mode="debug",
        shard_id=None,
        attempt=1,
    )
    receipt = {
        "schema_version": SHARD_VERIFIER_SCHEMA,
        "status": "PASS",
        "gate": "D",
        "mode": "debug",
        "verified_at_utc": _utc_now(),
        "git": git,
        "record_count": len(verified["records"]),
        "membership_sha256": membership["manifest_sha256"],
        "attempt_receipt_sha256": file_sha256(output / RECEIPT_NAME),
        "sanitized_records_sha256": file_sha256(output / SANITIZED_NAME),
        "runtime_closure_sha256": file_sha256(output / CLOSURE_NAME),
        "sealed_membership_sha256": file_sha256(output / SEALED_MEMBERSHIP_NAME),
        "opaque_payload_content_parsed": False,
        "selector_executed": False,
        "outcomes_read": False,
    }
    receipt["manifest_sha256"] = _manifest_hash(receipt)
    verifier_sha = _write_new_json(output / VERIFIER_NAME, receipt)
    return {
        "status": "PASS",
        "record_count": len(verified["records"]),
        "verifier_receipt_sha256": verifier_sha,
    }


def _load_frozen_formal(
    run_root: Path, expected_commit: str
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    population_path = run_root / "source/population.jsonl"
    population_manifest_path = run_root / "source/population_manifest.json"
    shard_manifest_path = run_root / "formal/shard_manifest.json"
    freeze_path = run_root / "formal/freeze_receipt.json"
    members = _strict_jsonl(population_path)
    population = _strict_json(population_manifest_path)
    shards = _strict_json(shard_manifest_path)
    freeze = _strict_json(freeze_path)
    _require(len(members) == 12032, "frozen population count differs")
    _require(
        semantic_sha256([row["canonical_identity"] for row in members])
        == ORDERED_IDENTITY_SHA256,
        "frozen population identity order differs",
    )
    _require(
        population.get("manifest_sha256") == _manifest_hash(population)
        and population.get("population_file_sha256") == file_sha256(population_path),
        "population manifest closure differs",
    )
    _require(
        shards.get("schema_version") == SHARD_MANIFEST_SCHEMA
        and shards.get("manifest_sha256") == _manifest_hash(shards)
        and shards.get("expected_commit") == expected_commit
        and shards.get("ordered_identity_sha256") == ORDERED_IDENTITY_SHA256,
        "formal shard manifest closure differs",
    )
    _require(
        freeze.get("status") == "FROZEN_BEFORE_FORMAL_SUBMISSION"
        and freeze.get("manifest_sha256") == _manifest_hash(freeze)
        and freeze.get("shard_manifest_file_sha256") == file_sha256(shard_manifest_path),
        "formal freeze receipt differs",
    )
    reconstructed = []
    for shard in shards["shards"]:
        _require(
            shard["members"] == members[int(shard["shard_id"]) :: int(shards["shard_count"])],
            "shard membership differs from ordinal modulo rule",
        )
        _require(
            shard["membership_sha256"] == semantic_sha256(shard["members"]),
            "shard membership hash differs",
        )
        reconstructed.extend(shard["members"])
    _require(len(reconstructed) == 12032, "shard membership total differs")
    _require(
        len({row["canonical_identity"] for row in reconstructed}) == 12032,
        "shard membership is not unique",
    )
    return members, shards


def acquire_shard(
    *,
    run_root: Path,
    gate_b_manifest_path: Path,
    expected_commit: str,
    shard_id: int,
    attempt: int,
) -> Dict[str, Any]:
    _require(attempt >= 1, "attempt must be positive")
    _members, shards = _load_frozen_formal(run_root, expected_commit)
    _require(0 <= shard_id < shards["shard_count"], "shard id is outside frozen range")
    shard = shards["shards"][shard_id]
    output = (
        run_root
        / shard["output_relative"]
        / ("attempt-%04d" % attempt)
    )
    return _acquire_members(
        output=output,
        members=shard["members"],
        membership_sha256=shard["membership_sha256"],
        expected_commit=expected_commit,
        gate_b_manifest_path=gate_b_manifest_path,
        mode="formal",
        shard_id=shard_id,
        attempt=attempt,
    )


def verify_shard(
    *,
    run_root: Path,
    gate_b_manifest_path: Path,
    expected_commit: str,
    shard_id: int,
    attempt: int,
) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    _load_card()
    _load_gate_b_manifest(gate_b_manifest_path)
    _members, shards = _load_frozen_formal(run_root, expected_commit)
    _require(0 <= shard_id < shards["shard_count"], "shard id is outside frozen range")
    shard = shards["shards"][shard_id]
    output = run_root / shard["output_relative"] / ("attempt-%04d" % attempt)
    verified = _validate_attempt(
        output=output,
        members=shard["members"],
        membership_sha256=shard["membership_sha256"],
        expected_commit=expected_commit,
        mode="formal",
        shard_id=shard_id,
        attempt=attempt,
    )
    receipt = {
        "schema_version": SHARD_VERIFIER_SCHEMA,
        "status": "PASS",
        "gate": "D",
        "mode": "formal",
        "shard_id": shard_id,
        "attempt": attempt,
        "verified_at_utc": _utc_now(),
        "git": git,
        "record_count": len(verified["records"]),
        "membership_sha256": shard["membership_sha256"],
        "attempt_receipt_sha256": file_sha256(output / RECEIPT_NAME),
        "sanitized_records_sha256": file_sha256(output / SANITIZED_NAME),
        "runtime_closure_sha256": file_sha256(output / CLOSURE_NAME),
        "sealed_membership_sha256": file_sha256(output / SEALED_MEMBERSHIP_NAME),
        "opaque_payload_content_parsed": False,
        "selector_executed": False,
        "outcomes_read": False,
    }
    receipt["manifest_sha256"] = _manifest_hash(receipt)
    verifier_sha = _write_new_json(output / VERIFIER_NAME, receipt)
    return {
        "status": "PASS",
        "shard_id": shard_id,
        "attempt": attempt,
        "record_count": len(verified["records"]),
        "verifier_receipt_sha256": verifier_sha,
    }


def record_scheduler(
    *, run_root: Path, expected_commit: str, job_id: str
) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    _members, shards = _load_frozen_formal(run_root, expected_commit)
    _require(re.fullmatch(r"[0-9]+", job_id) is not None, "Slurm job id is invalid")
    command = (
        "sacct",
        "-n",
        "-P",
        "-j",
        job_id,
        "--format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,NodeList,AllocTRES",
    )
    raw = _run(command)
    rows = []
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) < 8:
            continue
        rows.append(
            {
                "job_id_raw": fields[0],
                "job_name": fields[1],
                "partition": fields[2],
                "state": fields[3],
                "exit_code": fields[4],
                "elapsed": fields[5],
                "node_list": fields[6],
                "alloc_tres": fields[7],
            }
        )
    array_rows: Dict[int, Mapping[str, Any]] = {}
    pattern = re.compile(r"^%s_([0-9]+)$" % re.escape(job_id))
    for row in rows:
        match = pattern.fullmatch(str(row["job_id_raw"]))
        if match:
            array_rows[int(match.group(1))] = row
    expected = set(range(int(shards["shard_count"])))
    _require(set(array_rows) == expected, "scheduler array membership differs")
    for shard_id, row in array_rows.items():
        _require(
            str(row["state"]).startswith("COMPLETED")
            and row["exit_code"] == "0:0",
            "scheduler task %d is not successful" % shard_id,
        )
    payload = {
        "schema_version": SCHEDULER_RECEIPT_SCHEMA,
        "status": "PASS",
        "recorded_at_utc": _utc_now(),
        "git": git,
        "job_id": job_id,
        "shard_count": shards["shard_count"],
        "array_tasks": [array_rows[index] for index in sorted(array_rows)],
        "command": list(command),
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    receipt_sha = _write_new_json(
        run_root / "formal/scheduler_receipt.json", payload
    )
    return {"status": "PASS", "scheduler_receipt_sha256": receipt_sha}


def _chosen_valid_attempt(
    *,
    run_root: Path,
    shard: Mapping[str, Any],
    expected_commit: str,
) -> tuple[Path, Dict[str, Any]]:
    shard_root = run_root / shard["output_relative"]
    valid = []
    for output in sorted(shard_root.glob("attempt-*")):
        if not output.is_dir() or not (output / VERIFIER_NAME).is_file():
            continue
        match = re.fullmatch(r"attempt-([0-9]{4})", output.name)
        _require(match is not None, "attempt directory name differs")
        attempt = int(match.group(1))
        verifier = _strict_json(output / VERIFIER_NAME)
        _require(
            verifier.get("schema_version") == SHARD_VERIFIER_SCHEMA
            and verifier.get("status") == "PASS"
            and verifier.get("manifest_sha256") == _manifest_hash(verifier),
            "shard verifier receipt differs",
        )
        verified = _validate_attempt(
            output=output,
            members=shard["members"],
            membership_sha256=shard["membership_sha256"],
            expected_commit=expected_commit,
            mode="formal",
            shard_id=int(shard["shard_id"]),
            attempt=attempt,
        )
        valid.append((output, verified))
    _require(len(valid) == 1, "shard must have exactly one verified valid attempt")
    return valid[0]


def merge_formal(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    members, shards = _load_frozen_formal(run_root, expected_commit)
    merged_records: Dict[str, Dict[str, Any]] = {}
    merged_sealed: Dict[str, Dict[str, Any]] = {}
    shard_evidence = []
    for shard in shards["shards"]:
        output, verified = _chosen_valid_attempt(
            run_root=run_root, shard=shard, expected_commit=expected_commit
        )
        for record, sealed in zip(
            verified["records"], verified["sealed_members"]
        ):
            identity = str(record["canonical_identity"])
            _require(identity not in merged_records, "duplicate identity across shards")
            merged_records[identity] = record
            merged_sealed[identity] = {
                "ordinal": int(sealed["ordinal"]),
                "canonical_identity": identity,
                "payload_relative": str(
                    (output / sealed["payload_relative"]).relative_to(run_root)
                ),
                "byte_count": int(sealed["byte_count"]),
                "sha256": sealed["sha256"],
            }
        shard_evidence.append(
            {
                "shard_id": int(shard["shard_id"]),
                "attempt_relative": str(output.relative_to(run_root)),
                "record_count": len(verified["records"]),
                "receipt_sha256": file_sha256(output / RECEIPT_NAME),
                "verifier_sha256": file_sha256(output / VERIFIER_NAME),
            }
        )
    expected = [row["canonical_identity"] for row in members]
    _require(
        set(merged_records) == set(expected) and len(merged_records) == 12032,
        "merged population membership differs",
    )
    ordered_records = [merged_records[identity] for identity in expected]
    ordered_sealed = [merged_sealed[identity] for identity in expected]
    merged_path = run_root / "merged/sanitized_trajectories.jsonl"
    sealed_path = run_root / "merged/sealed_baseline_membership.json"
    manifest_path = run_root / "merged/merged_manifest.json"
    for path in (merged_path, sealed_path, manifest_path):
        _require(not path.exists(), "merged artifact already exists")
    records_sha = _write_new_jsonl(merged_path, ordered_records)
    sealed_payload = {
        "schema_version": "loopscope.phase6.gate-d-merged-sealed-membership.v1",
        "record_count": 12032,
        "ordered_identity_sha256": ORDERED_IDENTITY_SHA256,
        "members": ordered_sealed,
        "opaque_payload_content_parsed": False,
    }
    sealed_payload["manifest_sha256"] = _manifest_hash(sealed_payload)
    sealed_sha = _write_new_json(sealed_path, sealed_payload)
    manifest = {
        "schema_version": MERGED_MANIFEST_SCHEMA,
        "status": "READY_FOR_FRESH_PROCESS_VERIFICATION",
        "gate": "D",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "merged_at_utc": _utc_now(),
        "git": git,
        "record_count": 12032,
        "missing_count": 0,
        "extra_count": 0,
        "duplicate_count": 0,
        "anchor_unresolved_count": 0,
        "ordered_identity_sha256": ORDERED_IDENTITY_SHA256,
        "ordered_record_sha256": semantic_sha256(ordered_records),
        "sanitized_trajectories_sha256": records_sha,
        "sealed_baseline_membership_sha256": sealed_sha,
        "shard_manifest_sha256": shards["manifest_sha256"],
        "shards": shard_evidence,
        "information_barrier": {
            "sealed_payload_content_parsed": False,
            "selector_executed": False,
            "outcomes_read": False,
            "later_gate_entered": False,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    manifest_sha = _write_new_json(manifest_path, manifest)
    return {
        "status": "READY_FOR_FRESH_PROCESS_VERIFICATION",
        "record_count": 12032,
        "merged_manifest_sha256": manifest_sha,
    }


def verify_formal(
    *,
    run_root: Path,
    gate_b_manifest_path: Path,
    expected_commit: str,
) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    _load_card()
    _load_gate_b_manifest(gate_b_manifest_path)
    members, shards = _load_frozen_formal(run_root, expected_commit)
    scheduler_path = run_root / "formal/scheduler_receipt.json"
    scheduler = _strict_json(scheduler_path)
    _require(
        scheduler.get("schema_version") == SCHEDULER_RECEIPT_SCHEMA
        and scheduler.get("status") == "PASS"
        and scheduler.get("manifest_sha256") == _manifest_hash(scheduler)
        and scheduler.get("shard_count") == shards["shard_count"],
        "scheduler receipt differs",
    )
    merged_path = run_root / "merged/sanitized_trajectories.jsonl"
    sealed_path = run_root / "merged/sealed_baseline_membership.json"
    manifest_path = run_root / "merged/merged_manifest.json"
    manifest = _strict_json(manifest_path)
    records = _strict_jsonl(merged_path)
    sealed = _strict_json(sealed_path)
    _require(
        manifest.get("schema_version") == MERGED_MANIFEST_SCHEMA
        and manifest.get("manifest_sha256") == _manifest_hash(manifest)
        and manifest.get("git") == git,
        "merged manifest closure differs",
    )
    _require(
        manifest["sanitized_trajectories_sha256"] == file_sha256(merged_path)
        and manifest["sealed_baseline_membership_sha256"] == file_sha256(sealed_path),
        "merged artifact byte hashes differ",
    )
    manifest_shards = {
        int(row["shard_id"]): row for row in manifest.get("shards", [])
    }
    _require(
        set(manifest_shards) == set(range(int(shards["shard_count"]))),
        "merged shard evidence membership differs",
    )
    for shard in shards["shards"]:
        shard_id = int(shard["shard_id"])
        output, verified = _chosen_valid_attempt(
            run_root=run_root,
            shard=shard,
            expected_commit=expected_commit,
        )
        evidence = manifest_shards[shard_id]
        _require(
            evidence["attempt_relative"] == str(output.relative_to(run_root))
            and evidence["record_count"] == len(verified["records"])
            and evidence["receipt_sha256"] == file_sha256(output / RECEIPT_NAME)
            and evidence["verifier_sha256"] == file_sha256(output / VERIFIER_NAME),
            "merged shard receipt/verifier evidence differs",
        )
    expected_identities = [row["canonical_identity"] for row in members]
    _require(len(records) == 12032, "merged trajectory count differs")
    _require(
        [row.get("canonical_identity") for row in records] == expected_identities,
        "merged trajectory identity order differs",
    )
    sealed_members = sealed.get("members")
    _require(
        isinstance(sealed_members, list)
        and len(sealed_members) == 12032
        and [row.get("canonical_identity") for row in sealed_members]
        == expected_identities,
        "merged sealed membership differs",
    )
    category_counts: Counter[str] = Counter()
    for member, record, sealed_row in zip(members, records, sealed_members):
        validate_trajectory_record(record)
        scan_forbidden_fields(record)
        _require(
            record["canonical_identity"] == member["canonical_identity"]
            and record["category"] == member["category"],
            "merged trajectory source identity differs",
        )
        _require(
            record["generation_count"] == 1
            and record["replay_count"] == 1
            and record["loop_insertions"] == 0
            and record["answer_match_count"] >= 1
            and record["selected_match_ordinal"] == 0,
            "merged generation/anchor closure differs",
        )
        _require(
            len(record["H"]) == 37
            and len(record["D"]) == 37
            and len(record["hidden_rms_l2_to_final"]) == 37
            and len(record["hidden_cosine_to_final"]) == 37
            and len(record["hidden_cosine_distance_to_final"]) == 37
            and len(record["adjacent_angular_distance"]) == 36,
            "merged trajectory shape differs",
        )
        values = (
            list(record["H"])
            + list(record["D"])
            + list(record["hidden_rms_l2_to_final"])
            + list(record["hidden_cosine_to_final"])
            + list(record["hidden_cosine_distance_to_final"])
            + list(record["adjacent_angular_distance"])
        )
        _require(
            all(math.isfinite(float(value)) for value in values)
            and float(record["D"][-1]) <= 1e-6,
            "merged trajectory numeric/endpoint closure differs",
        )
        _require(
            all(
                left == 1.0 - right
                for left, right in zip(
                    record["hidden_cosine_distance_to_final"],
                    record["hidden_cosine_to_final"],
                )
            ),
            "merged cosine distance is not derived",
        )
        payload_relative = Path(str(sealed_row["payload_relative"]))
        _require(
            not payload_relative.is_absolute() and ".." not in payload_relative.parts,
            "merged sealed payload path escapes run root",
        )
        payload_path = run_root / payload_relative
        _require(payload_path.is_file(), "merged sealed payload is missing")
        _require(
            file_sha256(payload_path)
            == sealed_row["sha256"]
            == record["generated_completion_sha256"],
            "merged opaque payload hash differs",
        )
        _require(
            payload_path.stat().st_size == int(sealed_row["byte_count"]),
            "merged opaque payload byte count differs",
        )
        category_counts[str(record["category"])] += 1
    _require(
        dict(sorted(category_counts.items()))
        == dict(sorted(Counter(row["category"] for row in members).items())),
        "merged category counts differ",
    )
    receipt = {
        "schema_version": FINAL_VERIFIER_SCHEMA,
        "status": "PASS",
        "gate": "D",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "verified_at_utc": _utc_now(),
        "git": git,
        "run_root": str(run_root),
        "record_count": 12032,
        "unique_count": 12032,
        "missing_count": 0,
        "extra_count": 0,
        "duplicate_count": 0,
        "anchor_resolved_count": 12032,
        "selected_match_ordinal_zero_count": 12032,
        "generation_count_each": 1,
        "replay_count_each": 1,
        "loop_insertions_each": 0,
        "raw_boundary_count_each": 37,
        "angular_transition_count_each": 36,
        "final_norm_pre_hook_count_each": 1,
        "ordered_identity_sha256": ORDERED_IDENTITY_SHA256,
        "merged_manifest_sha256": file_sha256(manifest_path),
        "sanitized_trajectories_sha256": file_sha256(merged_path),
        "sealed_baseline_membership_sha256": file_sha256(sealed_path),
        "scheduler_receipt_sha256": file_sha256(scheduler_path),
        "opaque_payload_content_parsed": False,
        "forbidden_persisted_payloads": False,
        "selector_executed": False,
        "outcomes_read": False,
        "later_gate_entered": False,
    }
    receipt["manifest_sha256"] = _manifest_hash(receipt)
    receipt_sha = _write_new_json(
        run_root / "merged/final_verifier_receipt.json", receipt
    )
    return {
        "status": "PASS",
        "record_count": 12032,
        "final_verifier_receipt_sha256": receipt_sha,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run")
    for command in (
        "prepare-debug",
        "acquire-debug",
        "verify-debug",
        "freeze-formal",
        "acquire-shard",
        "verify-shard",
        "record-scheduler",
        "merge",
        "verify",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-root", type=Path, required=True)
        subparser.add_argument("--gate-b-manifest", type=Path, required=True)
        subparser.add_argument("--expected-commit", required=True)
        if command == "freeze-formal":
            subparser.add_argument("--shard-count", type=int, required=True)
        if command in ("acquire-shard", "verify-shard"):
            subparser.add_argument("--shard-id", type=int, required=True)
            subparser.add_argument("--attempt", type=int, required=True)
        if command == "record-scheduler":
            subparser.add_argument("--job-id", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    if args.command == "dry-run":
        print(
            json.dumps(
                {
                    "status": "GATE_D_DRY_RUN_PASS",
                    "executor_thread_id": EXECUTOR_THREAD_ID,
                    "planning_thread_id": PLANNING_THREAD_ID,
                    "population_count": 12032,
                    "shard_count_range": [8, 32],
                    "max_concurrent_gpus": 8,
                    "distribution": "canonical_ordinal_modulo_shard_count",
                    "generation_count_each": 1,
                    "replay_count_each": 1,
                    "loop_insertions_each": 0,
                    "sealed_payload_is_opaque": True,
                    "model_or_data_loaded": False,
                    "gpu_or_slurm_used": False,
                    "selector_executed": False,
                    "gate_e_entered": False,
                },
                sort_keys=True,
            )
        )
        return 0
    common = {
        "run_root": args.run_root.resolve(),
        "gate_b_manifest_path": args.gate_b_manifest.resolve(),
        "expected_commit": args.expected_commit,
    }
    if args.command == "prepare-debug":
        result = prepare_debug(**common)
    elif args.command == "acquire-debug":
        result = acquire_debug(**common)
    elif args.command == "verify-debug":
        result = verify_debug(**common)
    elif args.command == "freeze-formal":
        result = freeze_formal(**common, shard_count=args.shard_count)
    elif args.command == "acquire-shard":
        result = acquire_shard(
            **common, shard_id=args.shard_id, attempt=args.attempt
        )
    elif args.command == "verify-shard":
        result = verify_shard(
            **common, shard_id=args.shard_id, attempt=args.attempt
        )
    elif args.command == "record-scheduler":
        result = record_scheduler(
            run_root=common["run_root"],
            expected_commit=common["expected_commit"],
            job_id=args.job_id,
        )
    elif args.command == "merge":
        result = merge_formal(
            run_root=common["run_root"],
            expected_commit=common["expected_commit"],
        )
    else:
        result = verify_formal(**common)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
