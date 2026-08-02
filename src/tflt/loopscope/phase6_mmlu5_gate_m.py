"""LoopScope Phase 6 Gate M: formal MMLU five-shot trajectory and V2 selector.

This module is the Gate M-only orchestration layer.  It reuses the accepted
Gate K projection and Gate L native no-loop producer, but owns the formal
manifest, eight write-once shards, per-shard verification, merge, fresh full
verification, and one frozen V2 selector.  It never reads validation answers,
test data, correctness, or loop outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from tflt.loopscope import phase5_relative_biphasic as v1
from tflt.loopscope import phase6_mmlu5_gate_k as gate_k
from tflt.loopscope import phase6_mmlu5_gate_l as gate_l
from tflt.loopscope import phase6_selector as selector_contract
from tflt.loopscope.phase5_relative_biphasic_v2_absolute_rate import (
    AbsoluteRateError,
    compute_rate_statistics,
)
from tflt.loopscope.phase6_mmlu5_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    FORMAL_REPLICATES,
    FORMAL_SEED,
    LAYERS,
    MODEL_REVISION,
    PROMPT_PROJECTION_SCHEMA_VERSION,
    TASK_SOURCE_MANIFEST_SHA256,
    TRAJECTORY_SCHEMA_VERSION,
    VALIDATION_IDENTITY_SHA256,
    VALIDATION_RECORD_COUNT,
    VALIDATION_SUBJECT_COUNT,
    canonical_json_bytes,
    enumerate_candidates,
    file_sha256,
    load_json,
    semantic_sha256,
    selector_sample,
    validate_card,
    validate_prompt_projection_record,
    validate_schema_document,
    validate_selector_sample,
    validate_trajectory_record,
)


GATE = "M"
EXECUTOR_THREAD_ID = "019fc180-e590-74a2-9323-f626eaf139bf"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
AUTHORIZED_ADMISSION_BASE = "93f264440f4cc70bf44876f42d29c7b3ba2d8b86"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/"
    "training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
GATE_K_ROOT_DEFAULT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/"
    "training_free_looped_transformers_loopscope/runs/"
    "phase6-gate-k-mmlu5-20260802T083000Z"
)

CARD_RELATIVE = Path("configs/loopscope/phase6_mmlu5_prefix_card.json")
PROMPT_SCHEMA_RELATIVE = Path("configs/loopscope/phase6_mmlu5_prompt_projection_schema.json")
TRAJECTORY_SCHEMA_RELATIVE = Path("configs/loopscope/phase6_mmlu5_trajectory_schema.json")
CARD_SHA256 = "ea0c0c55d4b661872136e5204b58e363c646fea9c0f979fc3db88af9a919e265"
PROMPT_SCHEMA_SHA256 = "668620930e9d73109d4f4d2880bef72b6fa8e8737f517d4ef940ff35f46d19c7"
TRAJECTORY_SCHEMA_SHA256 = "6d64eafd296c8abb84c72d0d490fa2c90ea19fd878e889af89a07542fd7762ec"
GATE_K_PROJECTION_SHA256 = "5ed42d8b873579aba60cf65d23dbd97d760c01a342e55ffde8c6accaee6c4b9f"
GATE_K_PROJECTION_RECEIPT_SHA256 = "681c809acd6ca321898fde5903730f6d5285d086e499a28102823cf091c70030"
GATE_K_VERIFIER_RECEIPT_SHA256 = "35b257b9155d59ffc27472c0aeb36389c15bd0ecacf95c399f62ab22c6738d23"

DEBUG_PARTITION = "debug"
DEBUG_TIME_LIMIT = "00:29:00"
FORMAL_TIME_LIMIT = "06:00:00"
CPUS_PER_TASK = 8
MEMORY = "64G"
SHARD_COUNT = 8
ARRAY_CONCURRENCY = 8
SCIENTIFIC_STATES = (
    "SELECTED_WINDOW",
    "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
    "ABSTAIN_NO_UNIQUE_TOP1",
    "ABSTAIN_RATE_RANK_UNSTABLE",
)

SACCT_FIELDS = (
    "JobID",
    "JobIDRaw",
    "JobName",
    "Partition",
    "State",
    "ExitCode",
    "Elapsed",
    "NodeList",
    "AllocTRES",
    "ReqTRES",
)


class GateMMMLU5Error(RuntimeError):
    """Fail-closed Gate M contract, runtime, or evidence error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateMMMLU5Error(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=str(cwd or repository_root()),
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_new_bytes(path: Path, body: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(body)


def _write_new_json(path: Path, value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return _write_new_bytes(path, body)


def _write_new_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> str:
    body = b"".join(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for value in values
    )
    _require(bool(body), "JSONL artifact must not be empty")
    return _write_new_bytes(path, body)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateMMMLU5Error("invalid JSONL at %s:%d" % (path, line_number)) from exc
            _require(isinstance(value, dict), "JSONL rows must be objects")
            rows.append(value)
    return rows


def _validate_expected_commit(expected_commit: str) -> None:
    _require(
        isinstance(expected_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None,
        "expected commit is not a full Git SHA-1",
    )
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", AUTHORIZED_ADMISSION_BASE, expected_commit],
        cwd=str(repository_root()),
        text=True,
        capture_output=True,
    )
    _require(result.returncode == 0, "final commit is not a descendant of Gate M admission base")


def _load_contract() -> Tuple[Dict[str, Any], Path, Path, Path]:
    root = repository_root()
    card_path = root / CARD_RELATIVE
    prompt_schema_path = root / PROMPT_SCHEMA_RELATIVE
    trajectory_schema_path = root / TRAJECTORY_SCHEMA_RELATIVE
    _require(file_sha256(card_path) == CARD_SHA256, "MMLU5 card SHA-256 differs")
    _require(file_sha256(prompt_schema_path) == PROMPT_SCHEMA_SHA256, "MMLU5 prompt schema SHA-256 differs")
    _require(file_sha256(trajectory_schema_path) == TRAJECTORY_SCHEMA_SHA256, "MMLU5 trajectory schema SHA-256 differs")
    card = load_json(card_path)
    validate_card(card)
    validate_schema_document(load_json(prompt_schema_path))
    validate_schema_document(load_json(trajectory_schema_path))
    return card, card_path, prompt_schema_path, trajectory_schema_path


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = Path(_run(("git", "rev-parse", "--show-toplevel"))).resolve()
    _require(root == repository_root().resolve(), "not running in the dedicated LoopScope clone")
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"))
    commit = _run(("git", "rev-parse", "HEAD"))
    origin = _run(("git", "rev-parse", "origin/loopscope"))
    dirty = _run(("git", "status", "--porcelain"))
    _require(branch == "loopscope", "Gate M must run on loopscope")
    _require(commit == expected_commit, "Gate M expected commit differs")
    _require(origin == expected_commit, "origin/loopscope differs from expected commit")
    _require(dirty == "", "dedicated HPC2 clone must be clean")
    return {
        "repo": str(root),
        "branch": branch,
        "commit": commit,
        "origin_loopscope": origin,
        "dirty": False,
    }


def _preflight_subset(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    subjects: List[str] = []
    for row in rows:
        subject = str(row["subject"])
        if subject not in subjects:
            subjects.append(subject)
        if len(subjects) == 8:
            break
    selected: List[Dict[str, Any]] = []
    for subject in subjects:
        count = 0
        for row in rows:
            if str(row["subject"]) != subject:
                continue
            selected.append(dict(row))
            count += 1
            if count == 2:
                break
    selected.sort(key=lambda row: int(row["ordinal"]))
    _require(len(selected) == 16 and len({row["subject"] for row in selected}) == 8, "debug preflight subset is incomplete")
    return selected


def _shard_rows(rows: Sequence[Mapping[str, Any]], shard_count: int) -> List[List[Dict[str, Any]]]:
    _require(1 <= int(shard_count) <= SHARD_COUNT, "shard count must be between 1 and 8")
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(int(shard_count))]
    for row in rows:
        shards[int(row["ordinal"]) % int(shard_count)].append(dict(row))
    _require(sum(len(shard) for shard in shards) == len(rows), "shard membership count differs")
    return shards


def _gate_k_projection_closure(gate_k_root: Path) -> Dict[str, Any]:
    root = Path(gate_k_root).resolve()
    manifest_path = root / "cpu/manifest.json"
    projection_path = root / "cpu/validation_5shot_projection.jsonl"
    projection_receipt_path = root / "cpu/projection_receipt.json"
    verifier_path = root / "cpu/verifier_receipt.json"
    for path in (manifest_path, projection_path, projection_receipt_path, verifier_path):
        _require(path.is_file(), "accepted Gate K artifact is missing: %s" % path)
    _require(file_sha256(projection_path) == GATE_K_PROJECTION_SHA256, "Gate K projection SHA-256 differs")
    _require(file_sha256(projection_receipt_path) == GATE_K_PROJECTION_RECEIPT_SHA256, "Gate K projection receipt SHA-256 differs")
    _require(file_sha256(verifier_path) == GATE_K_VERIFIER_RECEIPT_SHA256, "Gate K verifier receipt SHA-256 differs")
    manifest = load_json(manifest_path)
    receipt = load_json(projection_receipt_path)
    verifier = load_json(verifier_path)
    _require(manifest.get("projection", {}).get("file_sha256") == GATE_K_PROJECTION_SHA256, "Gate K manifest projection hash differs")
    _require(receipt.get("status") == "PASS" and verifier.get("status") == "PASS", "Gate K closure is not PASS")
    for payload in (manifest, receipt, verifier):
        for key in (
            "selector_executed",
            "validation_target_gold_read",
            "test_split_read",
            "outcome_read",
            "model_weights_loaded",
            "model_forward_executed",
        ):
            if key in payload:
                _require(payload[key] is False, "Gate K information barrier differs: %s" % key)
    records = gate_k._load_jsonl(projection_path)
    _require(len(records) == VALIDATION_RECORD_COUNT, "Gate K projection population differs")
    for row in records:
        gate_k._assert_no_forbidden_keys(row)
        validate_prompt_projection_record(row)
    _require([int(row["ordinal"]) for row in records] == list(range(VALIDATION_RECORD_COUNT)), "Gate K ordinals are not canonical")
    _require(len({canonical_json_bytes(row["identity"]) for row in records}) == VALIDATION_RECORD_COUNT, "Gate K identities duplicate")
    _require(len({str(row["subject"]) for row in records}) == VALIDATION_SUBJECT_COUNT, "Gate K subject count differs")
    _require(semantic_sha256([row["identity"] for row in records]) == manifest["population"]["ordered_identity_sha256"], "Gate K identity digest differs")
    _require(semantic_sha256(records) == manifest["population"]["ordered_prompt_projection_sha256"], "Gate K projection digest differs")
    return {
        "root": str(root),
        "manifest": manifest,
        "records": records,
        "projection_path": str(projection_path),
        "projection_file_sha256": GATE_K_PROJECTION_SHA256,
        "projection_receipt_file_sha256": GATE_K_PROJECTION_RECEIPT_SHA256,
        "verifier_receipt_file_sha256": GATE_K_VERIFIER_RECEIPT_SHA256,
    }


def freeze_manifest(
    *,
    gate_k_root: Path,
    run_root: Path,
    expected_commit: str,
    preflight: bool = False,
    shard_count: int = SHARD_COUNT,
) -> Dict[str, Any]:
    """Freeze a fresh full-population or debug-shaped Gate M manifest."""

    _validate_expected_commit(expected_commit)
    root = Path(run_root).resolve()
    _require(not root.exists(), "Gate M run root must be fresh")
    card, card_path, prompt_schema_path, trajectory_schema_path = _load_contract()
    closure = _gate_k_projection_closure(gate_k_root)
    all_rows = [dict(row) for row in closure["records"]]
    rows = _preflight_subset(all_rows) if preflight else all_rows
    scope = "debug_preflight" if preflight else "formal"
    root.mkdir(parents=True, exist_ok=False)
    for name in ("cpu", "manifest", "shards", "slurm", "merge", "selector"):
        (root / name).mkdir(parents=True, exist_ok=False)
    projection_source = Path(closure["projection_path"])
    projection_sha = _write_new_bytes(root / "cpu/validation_5shot_projection.jsonl", projection_source.read_bytes())
    _require(projection_sha == closure["projection_file_sha256"], "copied Gate K projection hash differs")
    selected_projection_sha = _write_new_jsonl(root / "manifest/selected_projection.jsonl", rows)
    shards = _shard_rows(rows, shard_count)
    shard_files: List[Dict[str, Any]] = []
    for shard_index, members in enumerate(shards):
        payload = {
            "schema_version": "loopscope.phase6.mmlu5-gate-m-shard.v1",
            "gate": GATE,
            "scope": scope,
            "expected_commit": expected_commit,
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "assignment": "ordinal_modulo",
            "members": members,
            "record_count": len(members),
            "identity_order_sha256": semantic_sha256([row["identity"] for row in members]),
            "selector_executed": False,
            "loop_executed": False,
            "outcome_read": False,
            "test_split_read": False,
        }
        payload["manifest_sha256"] = semantic_sha256(payload)
        path = root / "shards" / ("shard-%03d.json" % shard_index)
        file_hash = _write_new_json(path, payload)
        shard_files.append(
            {
                "shard_index": int(shard_index),
                "record_count": len(members),
                "identity_order_sha256": payload["identity_order_sha256"],
                "path": str(path.relative_to(root)),
                "file_sha256": file_hash,
                "manifest_sha256": payload["manifest_sha256"],
            }
        )
    manifest = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-manifest.v1",
        "gate": GATE,
        "scope": scope,
        "created_at_utc": _utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "source_gate_k_root": closure["root"],
        "source_gate_k_projection_file_sha256": closure["projection_file_sha256"],
        "source_gate_k_projection_receipt_file_sha256": closure["projection_receipt_file_sha256"],
        "source_gate_k_verifier_receipt_file_sha256": closure["verifier_receipt_file_sha256"],
        "projection_file_sha256": projection_sha,
        "selected_projection_file_sha256": selected_projection_sha,
        "canonical_population_count": len(all_rows),
        "canonical_subject_count": len({str(row["subject"]) for row in all_rows}),
        "canonical_identity_sha256": semantic_sha256([row["identity"] for row in all_rows]),
        "canonical_projection_sha256": semantic_sha256(all_rows),
        "record_count": len(rows),
        "subject_count": len({str(row["subject"]) for row in rows}),
        "ordered_identity_sha256": semantic_sha256([row["identity"] for row in rows]),
        "ordered_prompt_projection_sha256": semantic_sha256(rows),
        "validation_identity_reuse_sha256": VALIDATION_IDENTITY_SHA256,
        "task_source_manifest_sha256": TASK_SOURCE_MANIFEST_SHA256,
        "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
        "model_revision": MODEL_REVISION,
        "shard_count": int(shard_count),
        "assignment": "ordinal_modulo",
        "shards": shard_files,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
        "test_split_read": False,
        "validation_target_gold_read": False,
        "raw_prompt_persisted": False,
        "token_ids_persisted": False,
        "full_vocab_persisted": False,
        "hidden_tensors_persisted": False,
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    manifest_file_sha = _write_new_json(root / "manifest/formal_manifest.json", manifest)
    freeze = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-freeze.v1",
        "gate": GATE,
        "scope": scope,
        "run_root": str(root),
        "expected_commit": expected_commit,
        "formal_manifest_path": "manifest/formal_manifest.json",
        "formal_manifest_file_sha256": manifest_file_sha,
        "formal_manifest_semantic_sha256": manifest["manifest_sha256"],
        "selected_projection_file_sha256": selected_projection_sha,
        "record_count": len(rows),
        "subject_count": len({str(row["subject"]) for row in rows}),
        "shard_count": int(shard_count),
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
        "information_barrier": {
            "validation_target_gold_read": False,
            "test_split_read": False,
            "outcome_read": False,
            "loop_executed": False,
            "raw_prompt_persisted": False,
            "token_ids_persisted": False,
            "full_vocab_persisted": False,
            "hidden_tensors_persisted": False,
        },
    }
    freeze_file_sha = _write_new_json(root / "manifest/freeze_receipt.json", freeze)
    return {
        "status": "PASS",
        "scope": scope,
        "run_root": str(root),
        "manifest_file_sha256": manifest_file_sha,
        "manifest_semantic_sha256": manifest["manifest_sha256"],
        "freeze_receipt_sha256": freeze_file_sha,
        "record_count": len(rows),
        "subject_count": len({str(row["subject"]) for row in rows}),
        "canonical_population_count": len(all_rows),
        "shard_count": int(shard_count),
        "selector_executed": False,
        "outcome_read": False,
    }


def _load_context(root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Path, Path, Path]:
    root = Path(root).resolve()
    manifest_path = root / "manifest/formal_manifest.json"
    freeze_path = root / "manifest/freeze_receipt.json"
    selected_path = root / "manifest/selected_projection.jsonl"
    _require(manifest_path.is_file() and freeze_path.is_file() and selected_path.is_file(), "Gate M manifest freeze is incomplete")
    manifest = load_json(manifest_path)
    manifest_hash = manifest.pop("manifest_sha256", None)
    _require(isinstance(manifest_hash, str) and semantic_sha256(manifest) == manifest_hash, "Gate M manifest semantic hash differs")
    manifest["manifest_sha256"] = manifest_hash
    freeze = load_json(freeze_path)
    _require(file_sha256(manifest_path) == freeze.get("formal_manifest_file_sha256"), "Gate M manifest file hash differs")
    rows = _load_jsonl(selected_path)
    _require(file_sha256(selected_path) == manifest["selected_projection_file_sha256"], "Gate M selected projection hash differs")
    _require(len(rows) == int(manifest["record_count"]), "Gate M selected projection count differs")
    _require(semantic_sha256(rows) == manifest["ordered_prompt_projection_sha256"], "Gate M selected projection digest differs")
    _require([int(row["ordinal"]) for row in rows] == sorted(int(row["ordinal"]) for row in rows), "Gate M selected projection order differs")
    for row in rows:
        validate_prompt_projection_record(row)
    card, card_path, prompt_schema_path, trajectory_schema_path = _load_contract()
    _require(manifest["card_sha256"] == file_sha256(card_path), "Gate M card hash differs")
    _require(manifest["prompt_projection_schema_sha256"] == file_sha256(prompt_schema_path), "Gate M prompt schema hash differs")
    _require(manifest["trajectory_schema_sha256"] == file_sha256(trajectory_schema_path), "Gate M trajectory schema hash differs")
    _require(manifest["selector_executed"] is False and manifest["outcome_read"] is False, "Gate M manifest barrier differs")
    if manifest["scope"] == "formal":
        _require(len(rows) == VALIDATION_RECORD_COUNT and len({str(row["subject"]) for row in rows}) == VALIDATION_SUBJECT_COUNT, "formal Gate M population differs")
    return manifest, rows, freeze, card, card_path, prompt_schema_path, trajectory_schema_path


def _reconstruct_prompts(
    selected: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    seed: int = FORMAL_SEED,
) -> Dict[str, str]:
    """Rebuild five-shot prompts from the gold-free Gate K projection."""

    try:
        import numpy as np
        from datasets import DatasetDict
        from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateMMMLU5Error("Gate M renderer dependencies failed") from exc
    task_manager, task_by_subject = gate_k._task_map()
    del task_manager
    wanted = {canonical_json_bytes(row["identity"]): row for row in selected}
    backend = LmEvalMMLURendererBackend()
    prompts: Dict[str, str] = {}
    for subject in sorted({str(row["subject"]) for row in selected}):
        task_name = "mmlu_%s" % subject
        _require(task_name in task_by_subject, "formal subject has no standard MMLU task")
        validation, dev, _evidence = gate_k._load_subject_datasets(subject)
        task = task_by_subject[task_name]
        task.dataset = DatasetDict({"validation": validation, "dev": dev})
        for target_index, row in enumerate(validation):
            safe = gate_k._safe_target_doc(row, subject)
            identity = {
                "task": "mmlu",
                "doc_id": "mmlu_%s:validation:%d" % (subject, target_index),
                "doc_hash": gate_k._safe_target_hash(safe),
            }
            key = canonical_json_bytes(identity)
            wanted_row = wanted.get(key)
            if wanted_row is None:
                continue
            prompt, demos = backend._render_with_captured_demos(
                task,
                gate_k._renderer_target_doc(safe),
                dev,
                np,
                int(seed),
            )
            _require(isinstance(prompt, str) and prompt.rstrip().endswith("Answer:"), "standard five-shot prompt terminal differs")
            demo_rows = gate_k._demo_provenance(task, demos, dev, subject)
            _require(demo_rows == wanted_row["demonstration_provenance"], "fresh demonstration provenance differs")
            token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            _require(token_ids, "formal prompt tokenizes empty")
            _require(_sha256_bytes(prompt.encode("utf-8")) == wanted_row["prompt_sha256"], "formal prompt hash differs")
            _require(_sha256_bytes(canonical_json_bytes(token_ids)) == wanted_row["prompt_tokenization_sha256"], "formal prompt tokenization hash differs")
            _require(len(token_ids) == int(wanted_row["sequence_length"]), "formal prompt sequence length differs")
            _require(len(token_ids) - 1 == int(wanted_row["probe_position"]), "formal probe position differs")
            prompts[key.decode("utf-8")] = prompt
    _require(len(prompts) == len(selected), "formal prompt reconstruction is incomplete")
    return prompts


RUNTIME_EVIDENCE_FIELDS = {
    "identity",
    "subject",
    "prompt_sha256",
    "prompt_tokenization_sha256",
    "sequence_length",
    "probe_position",
    "final_norm_prehook_calls",
    "final_norm_boundary_calls",
    "final_norm_closure_max_abs",
    "native_reprojected_vocab_max_abs",
    "boundary_count",
    "transition_count",
    "forward_count",
    "loop_insertions",
    "choice_surface_manifest_sha256",
    "elapsed_seconds",
    "full_vocab_persisted",
    "hidden_tensors_persisted",
    "raw_text_persisted",
    "token_ids_persisted",
}


def _validate_runtime_fact(fact: Mapping[str, Any], record: Mapping[str, Any], projection: Mapping[str, Any]) -> None:
    _require(set(fact) == RUNTIME_EVIDENCE_FIELDS, "runtime evidence fields are not closed")
    _require(fact["identity"] == record["identity"] and fact["subject"] == record["subject"], "runtime evidence identity differs")
    _require(fact["prompt_sha256"] == record["prompt_sha256"] == projection["prompt_sha256"], "runtime prompt hash differs")
    _require(fact["prompt_tokenization_sha256"] == projection["prompt_tokenization_sha256"], "runtime tokenization hash differs")
    _require(int(fact["sequence_length"]) == int(projection["sequence_length"]), "runtime sequence length differs")
    _require(int(fact["probe_position"]) == int(projection["probe_position"]), "runtime probe position differs")
    _require(int(fact["final_norm_prehook_calls"]) == 1, "FinalNorm pre-hook call count differs")
    _require(int(fact["final_norm_boundary_calls"]) == BOUNDARY_COUNT - 1, "FinalNorm boundary call count differs")
    _require(float(fact["final_norm_closure_max_abs"]) == 0.0, "FinalNorm closure is nonzero")
    _require(int(fact["boundary_count"]) == BOUNDARY_COUNT and int(fact["transition_count"]) == LAYERS, "trajectory dimensions differ")
    _require(int(fact["forward_count"]) == 1 and int(fact["loop_insertions"]) == 0, "per-record execution closure differs")
    _require(fact["choice_surface_manifest_sha256"] == CHOICE_SURFACE_MANIFEST_SHA256, "choice surface closure differs")
    _require(fact["full_vocab_persisted"] is False and fact["hidden_tensors_persisted"] is False, "forbidden tensors persisted")
    _require(fact["raw_text_persisted"] is False and fact["token_ids_persisted"] is False, "raw prompt payload persisted")
    _require(math.isfinite(float(fact["native_reprojected_vocab_max_abs"])) and math.isfinite(float(fact["elapsed_seconds"])), "runtime scalar is non-finite")


def _attempt_root(root: Path, shard_index: int, attempt: int) -> Path:
    _require(int(attempt) >= 1, "attempt must be positive")
    return Path(root).resolve() / "shards" / ("shard-%03d" % int(shard_index)) / ("attempt-%04d" % int(attempt))


def acquire_shard(*, run_root: Path, expected_commit: str, shard_index: int, attempt: int = 1) -> Dict[str, Any]:
    """Run one native no-loop shard; failed attempts remain write-once."""

    _validate_expected_commit(expected_commit)
    gate_l._assert_offline_environment()
    manifest, rows, _freeze, card, card_path, prompt_schema_path, trajectory_schema_path = _load_context(run_root)
    _require(0 <= int(shard_index) < int(manifest["shard_count"]), "shard index is out of range")
    shard_path = Path(run_root).resolve() / "shards" / ("shard-%03d.json" % int(shard_index))
    shard = load_json(shard_path)
    shard_hash = shard.pop("manifest_sha256", None)
    _require(isinstance(shard_hash, str) and semantic_sha256(shard) == shard_hash, "shard manifest hash differs")
    shard["manifest_sha256"] = shard_hash
    selected = [dict(row) for row in shard["members"]]
    by_key = {canonical_json_bytes(row["identity"]): row for row in rows}
    _require(all(canonical_json_bytes(row["identity"]) in by_key for row in selected), "shard member is not in selected projection")
    attempt_root = _attempt_root(run_root, shard_index, attempt)
    _require(not attempt_root.exists(), "shard attempt root must be fresh")
    attempt_root.mkdir(parents=True, exist_ok=False)
    actual_partition = os.environ.get("SLURM_JOB_PARTITION", "unknown")
    attempt_start = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-attempt-start.v1",
        "gate": GATE,
        "run_root": str(Path(run_root).resolve()),
        "shard_index": int(shard_index),
        "attempt": int(attempt),
        "expected_commit": expected_commit,
        "partition": actual_partition,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "generation_calls": 0,
        "loop_insertions": 0,
        "selector_executed": False,
        "outcome_read": False,
    }
    attempt_start_sha = _write_new_json(attempt_root / "attempt_start.json", attempt_start)
    torch_module, tokenizer, model, final_norm, lm_head, choice_ids = gate_l._load_native_runtime(card)
    prompts = _reconstruct_prompts(selected, tokenizer)
    records: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for member in selected:
        key = canonical_json_bytes(member["identity"])
        record, fact = gate_l._trajectory_from_forward(
            prompt=prompts[key.decode("utf-8")],
            projection_row=member,
            tokenizer=tokenizer,
            torch_module=torch_module,
            model=model,
            final_norm=final_norm,
            lm_head=lm_head,
            choice_ids=choice_ids,
            card=card,
        )
        records.append(record)
        evidence.append(fact)
    _require(len(records) == len(selected) and len(evidence) == len(selected), "shard record count differs")
    records_path = attempt_root / "trajectory_records.jsonl"
    evidence_path = attempt_root / "runtime_evidence.jsonl"
    records_sha = _write_new_jsonl(records_path, records)
    evidence_sha = _write_new_jsonl(evidence_path, evidence)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-shard-receipt.v1",
        "gate": GATE,
        "scope": manifest["scope"],
        "created_at_utc": _utc_now(),
        "run_root": str(Path(run_root).resolve()),
        "shard_index": int(shard_index),
        "attempt": int(attempt),
        "expected_commit": expected_commit,
        "partition": actual_partition,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "shard_manifest_sha256": shard["manifest_sha256"],
        "gate_k_projection_file_sha256": manifest["source_gate_k_projection_file_sha256"],
        "attempt_start_file_sha256": attempt_start_sha,
        "record_file_sha256": records_sha,
        "evidence_file_sha256": evidence_sha,
        "record_count": len(records),
        "identity_order_sha256": semantic_sha256([row["identity"] for row in records]),
        "forward_count": sum(int(row["forward_count"]) for row in evidence),
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "loop_insertions": sum(int(row["loop_insertions"]) for row in evidence),
        "generation_calls": 0,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
        "test_split_read": False,
        "validation_target_gold_read": False,
        "raw_prompt_persisted": False,
        "token_ids_persisted": False,
        "full_vocab_persisted": False,
        "hidden_tensors_persisted": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(attempt_root / "shard_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha, "attempt_root": str(attempt_root)}


def verify_shard(*, run_root: Path, expected_commit: str, shard_index: int, attempt: int) -> Dict[str, Any]:
    """Fresh-process verifier for one completed shard."""

    _validate_expected_commit(expected_commit)
    manifest, rows, _freeze, card, card_path, prompt_schema_path, trajectory_schema_path = _load_context(run_root)
    shard = load_json(Path(run_root).resolve() / "shards" / ("shard-%03d.json" % int(shard_index)))
    shard_hash = shard.pop("manifest_sha256", None)
    _require(isinstance(shard_hash, str) and semantic_sha256(shard) == shard_hash, "shard manifest hash differs")
    shard["manifest_sha256"] = shard_hash
    attempt_root = _attempt_root(run_root, shard_index, attempt)
    receipt_path = attempt_root / "shard_receipt.json"
    records_path = attempt_root / "trajectory_records.jsonl"
    evidence_path = attempt_root / "runtime_evidence.jsonl"
    _require(receipt_path.is_file() and records_path.is_file() and evidence_path.is_file(), "shard evidence is incomplete")
    receipt = load_json(receipt_path)
    _require(receipt.get("status") == "PASS", "shard receipt is not PASS")
    _require(receipt.get("expected_commit") == expected_commit, "shard receipt commit differs")
    _require(file_sha256(records_path) == receipt["record_file_sha256"], "shard trajectory hash differs")
    _require(file_sha256(evidence_path) == receipt["evidence_file_sha256"], "shard evidence hash differs")
    records = _load_jsonl(records_path)
    evidence = _load_jsonl(evidence_path)
    expected_members = {canonical_json_bytes(row["identity"]): row for row in shard["members"]}
    _require(len(records) == len(evidence) == len(expected_members), "shard record/evidence count differs")
    seen = set()
    for record, fact in zip(records, evidence):
        validate_trajectory_record(record, card)
        key = canonical_json_bytes(record["identity"])
        _require(key in expected_members and key not in seen, "shard identity membership differs")
        seen.add(key)
        _validate_runtime_fact(fact, record, expected_members[key])
    _require(seen == set(expected_members), "shard missing or extra identity")
    _require(semantic_sha256([row["identity"] for row in records]) == shard["identity_order_sha256"], "shard identity order differs")
    _require(receipt["identity_order_sha256"] == shard["identity_order_sha256"], "shard receipt identity order differs")
    verifier = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-shard-verifier.v1",
        "gate": GATE,
        "scope": manifest["scope"],
        "run_root": str(Path(run_root).resolve()),
        "shard_index": int(shard_index),
        "attempt": int(attempt),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "shard_manifest_sha256": shard["manifest_sha256"],
        "shard_receipt_file_sha256": file_sha256(receipt_path),
        "record_file_sha256": file_sha256(records_path),
        "evidence_file_sha256": file_sha256(evidence_path),
        "record_count": len(records),
        "subject_count": len({str(row["subject"]) for row in records}),
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "forward_count": sum(int(row["forward_count"]) for row in evidence),
        "loop_insertions": 0,
        "selector_executed": False,
        "outcome_read": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "forbidden_persisted_payloads": False,
        "status": "PASS",
    }
    verifier_sha = _write_new_json(attempt_root / "shard_verifier_receipt.json", verifier)
    return {**verifier, "receipt_sha256": verifier_sha}


def _valid_attempts(root: Path, shard_index: int) -> List[Tuple[int, Path, Dict[str, Any], Dict[str, Any]]]:
    shard_root = Path(root).resolve() / "shards" / ("shard-%03d" % int(shard_index))
    results: List[Tuple[int, Path, Dict[str, Any], Dict[str, Any]]] = []
    if not shard_root.is_dir():
        return results
    for path in sorted(shard_root.glob("attempt-*")):
        receipt_path = path / "shard_receipt.json"
        verifier_path = path / "shard_verifier_receipt.json"
        if not receipt_path.is_file() or not verifier_path.is_file():
            continue
        receipt = load_json(receipt_path)
        verifier = load_json(verifier_path)
        if receipt.get("status") != "PASS" or verifier.get("status") != "PASS":
            continue
        match = re.fullmatch(r"attempt-(\d{4})", path.name)
        _require(match is not None, "shard attempt directory name is invalid")
        results.append((int(match.group(1)), path, receipt, verifier))
    return results


def merge_shards(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    _validate_expected_commit(expected_commit)
    manifest, rows, _freeze, card, card_path, prompt_schema_path, trajectory_schema_path = _load_context(run_root)
    root = Path(run_root).resolve()
    all_records: Dict[bytes, Dict[str, Any]] = {}
    all_evidence: Dict[bytes, Dict[str, Any]] = {}
    attempt_receipts: Dict[str, Any] = {}
    for shard_index in range(int(manifest["shard_count"])):
        attempts = _valid_attempts(root, shard_index)
        _require(len(attempts) == 1, "each shard must have exactly one verified PASS attempt")
        attempt, attempt_root, receipt, verifier = attempts[0]
        records_path = attempt_root / "trajectory_records.jsonl"
        evidence_path = attempt_root / "runtime_evidence.jsonl"
        records = _load_jsonl(records_path)
        evidence = _load_jsonl(evidence_path)
        shard = load_json(root / "shards" / ("shard-%03d.json" % shard_index))
        expected_members = {canonical_json_bytes(row["identity"]): row for row in shard["members"]}
        _require(len(records) == len(evidence) == len(expected_members), "verified shard count differs")
        for record, fact in zip(records, evidence):
            validate_trajectory_record(record, card)
            key = canonical_json_bytes(record["identity"])
            _require(key in expected_members and key not in all_records, "duplicate identity across shards")
            _validate_runtime_fact(fact, record, expected_members[key])
            all_records[key] = record
            all_evidence[key] = fact
        attempt_receipts[str(shard_index)] = {
            "attempt": attempt,
            "shard_receipt_file_sha256": file_sha256(attempt_root / "shard_receipt.json"),
            "shard_verifier_file_sha256": file_sha256(attempt_root / "shard_verifier_receipt.json"),
            "record_file_sha256": receipt["record_file_sha256"],
            "evidence_file_sha256": receipt["evidence_file_sha256"],
            "record_count": len(records),
            "verified_status": verifier["status"],
        }
    expected = {canonical_json_bytes(row["identity"]): row for row in rows}
    _require(set(all_records) == set(expected), "merged identity membership differs")
    ordered_records = [all_records[canonical_json_bytes(row["identity"])] for row in rows]
    ordered_evidence = [all_evidence[canonical_json_bytes(row["identity"])] for row in rows]
    records_path = root / "merge/merged_trajectory_records.jsonl"
    evidence_path = root / "merge/merged_runtime_evidence.jsonl"
    records_sha = _write_new_jsonl(records_path, ordered_records)
    evidence_sha = _write_new_jsonl(evidence_path, ordered_evidence)
    payload = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-merge.v1",
        "gate": GATE,
        "scope": manifest["scope"],
        "run_root": str(root),
        "expected_commit": expected_commit,
        "manifest_sha256": manifest["manifest_sha256"],
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "record_file_sha256": records_sha,
        "evidence_file_sha256": evidence_sha,
        "record_count": len(ordered_records),
        "subject_count": len({str(row["subject"]) for row in ordered_records}),
        "identity_order_sha256": semantic_sha256([row["identity"] for row in ordered_records]),
        "forward_count": sum(int(row["forward_count"]) for row in ordered_evidence),
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "loop_insertions": sum(int(row["loop_insertions"]) for row in ordered_evidence),
        "shards": attempt_receipts,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "forbidden_persisted_payloads": False,
        "status": "PASS",
    }
    merge_sha = _write_new_json(root / "merge/merge_receipt.json", payload)
    return {**payload, "merge_receipt_sha256": merge_sha}


def query_scheduler(job_id: str) -> Tuple[List[str], List[Dict[str, str]]]:
    _require(re.fullmatch(r"[0-9]+", str(job_id)) is not None, "Slurm job ID is invalid")
    sacct = shutil.which("sacct") or "/opt/slurm/bin/sacct"
    command = [sacct, "-X", "-n", "-P", "-j", str(job_id), "--format=" + ",".join(SACCT_FIELDS)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows: List[Dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|")
        if len(fields) != len(SACCT_FIELDS):
            continue
        rows.append(dict(zip(SACCT_FIELDS, fields)))
    _require(rows, "sacct returned no rows")
    return command, rows


def _scheduler_acceptance(rows: Sequence[Mapping[str, str]], job_id: str, expected_shards: int, expected_partition: str) -> Dict[str, Any]:
    array_pattern = r"%s_[0-9]+" % re.escape(str(job_id))
    task_rows = [row for row in rows if re.fullmatch(array_pattern, str(row.get("JobID", "")))]
    if int(expected_shards) == 1 and not task_rows:
        task_rows = [row for row in rows if str(row.get("JobID", "")) == str(job_id) or str(row.get("JobIDRaw", "")) == str(job_id)]
    _require(len(task_rows) == int(expected_shards), "scheduler task count differs")
    expected_ids = {"%s_%d" % (job_id, index) for index in range(int(expected_shards))}
    observed_ids = {str(row.get("JobID", "")) for row in task_rows}
    _require(observed_ids == expected_ids, "scheduler task membership differs")
    for row in task_rows:
        _require(str(row["Partition"]) == expected_partition, "scheduler partition differs")
        _require(str(row["State"]).startswith("COMPLETED"), "scheduler state is not COMPLETED")
        _require(str(row["ExitCode"]) == "0:0", "scheduler ExitCode is not 0:0")
    return {
        "job_id": str(job_id),
        "expected_shards": int(expected_shards),
        "partition": expected_partition,
        "task_rows": [dict(row) for row in task_rows],
    }


def verify_formal_run(*, run_root: Path, expected_commit: str, job_id: str, expected_partition: str) -> Dict[str, Any]:
    """Fresh-process full verifier for formal or debug-preflight trajectory closure."""

    _validate_expected_commit(expected_commit)
    manifest, rows, _freeze, card, card_path, prompt_schema_path, trajectory_schema_path = _load_context(run_root)
    root = Path(run_root).resolve()
    _require(manifest["expected_commit"] == expected_commit, "manifest commit differs")
    merge_path = root / "merge/merge_receipt.json"
    records_path = root / "merge/merged_trajectory_records.jsonl"
    evidence_path = root / "merge/merged_runtime_evidence.jsonl"
    _require(merge_path.is_file() and records_path.is_file() and evidence_path.is_file(), "merge closure is missing")
    merge = load_json(merge_path)
    _require(merge.get("status") == "PASS", "merge receipt is not PASS")
    _require(file_sha256(records_path) == merge["record_file_sha256"], "merged trajectory hash differs")
    _require(file_sha256(evidence_path) == merge["evidence_file_sha256"], "merged evidence hash differs")
    records = _load_jsonl(records_path)
    evidence = _load_jsonl(evidence_path)
    _require(len(records) == len(evidence) == len(rows), "full verifier population differs")
    expected = {canonical_json_bytes(row["identity"]): row for row in rows}
    observed = set()
    for record, fact in zip(records, evidence):
        validate_trajectory_record(record, card)
        key = canonical_json_bytes(record["identity"])
        _require(key in expected and key not in observed, "full verifier identity membership differs")
        observed.add(key)
        _validate_runtime_fact(fact, record, expected[key])
    _require(observed == set(expected), "full verifier missing or extra identity")
    _require(semantic_sha256([row["identity"] for row in records]) == manifest["ordered_identity_sha256"], "full verifier identity order differs")
    _require(sum(int(row["forward_count"]) for row in evidence) == len(records), "full verifier forward count differs")
    _require(sum(int(row["loop_insertions"]) for row in evidence) == 0, "full verifier loop insertion count is nonzero")
    if manifest["scope"] == "formal":
        _require(len(records) == VALIDATION_RECORD_COUNT and len({str(row["subject"]) for row in records}) == VALIDATION_SUBJECT_COUNT, "formal verifier is not validation-1531")
    scheduler_command, scheduler_rows = query_scheduler(job_id)
    scheduler = _scheduler_acceptance(scheduler_rows, job_id, int(manifest["shard_count"]), expected_partition)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-full-verifier.v1",
        "gate": GATE,
        "scope": manifest["scope"],
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "merge_receipt_file_sha256": file_sha256(merge_path),
        "record_file_sha256": file_sha256(records_path),
        "evidence_file_sha256": file_sha256(evidence_path),
        "record_count": len(records),
        "subject_count": len({str(row["subject"]) for row in records}),
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "forward_count": len(records),
        "loop_insertions": 0,
        "scheduler": {"query": scheduler_command, **scheduler},
        "selector_executed": False,
        "loop_executed": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "outcome_read": False,
        "forbidden_persisted_payloads": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(root / "merge/verifier_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def _project_selector_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        sample = selector_sample(record)
        validate_selector_sample(sample)
        if sample["identity"] in seen:
            raise GateMMMLU5Error("selector identities must be unique")
        seen.add(sample["identity"])
        projected.append(
            {
                "identity": sample["identity"],
                "category": sample["category"],
                "H": list(sample["H"]),
                "D": list(sample["D"]),
            }
        )
    return projected


def analyze_selector(*, records: Sequence[Mapping[str, Any]], replicates: int = FORMAL_REPLICATES, seed: int = FORMAL_SEED, formal: bool = True) -> Dict[str, Any]:
    """Run the frozen V1-compatible/V2 absolute-rate math on MMLU5 records."""

    _require(isinstance(replicates, int) and not isinstance(replicates, bool) and replicates >= 2, "selector bootstrap requires at least two replicates")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "selector seed must be an integer")
    if formal:
        _require(replicates == FORMAL_REPLICATES and seed == FORMAL_SEED, "formal selector bootstrap contract differs")
        _require(len(records) == VALIDATION_RECORD_COUNT, "formal selector requires validation-1531")
        _require(len({str(row["subject"]) for row in records}) == VALIDATION_SUBJECT_COUNT, "formal selector requires 57 subjects")
    projected = _project_selector_records(records)
    legacy_samples = [
        {"identity": row["identity"], "subject": row["category"], "H": row["H"], "D": row["D"]}
        for row in projected
    ]
    try:
        v1_analysis = v1.analyze_model(legacy_samples, "qwen4", replicates=replicates, seed=seed, enforce_population=False)
        normalized = v1._normalize_preprojected(legacy_samples, "qwen4", False)
        bootstrap = v1._bootstrap_boundary_means(normalized, 36, replicates=replicates, seed=seed, digest_style=v1.MODEL_SPECS["qwen4"]["legacy_style"])
    except (v1.RelativeBiphasicError, KeyError, TypeError, ValueError) as exc:
        raise GateMMMLU5Error(str(exc)) from exc
    _require(bootstrap["draw_index_sha256"] == v1_analysis["bootstrap"]["draw_index_sha256"], "selector joint bootstrap draw stream differs")
    compatibility = selector_contract.v1_compatibility_view(v1_analysis["rows"])
    working_rows: List[Dict[str, Any]] = []
    bootstrap_rates: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for source in v1_analysis["rows"]:
        width = int(source["width"])
        start = int(source["start"])
        boot_h = [(bootstrap["H"][start][index] - bootstrap["H"][start + width][index]) / width for index in range(replicates)]
        boot_k = [(bootstrap["D"][start + width][index] - bootstrap["D"][start][index]) / width for index in range(replicates)]
        try:
            statistics = compute_rate_statistics(source["G_H"], source["G_K"], boot_h, boot_k)
        except (AbsoluteRateError, KeyError, TypeError, ValueError) as exc:
            raise GateMMMLU5Error(str(exc)) from exc
        best = source.get("best_shape_pair")
        row = {
            "width": width,
            "start": start,
            "end": start + width - 1,
            "window": "%d:%d" % (start, start + width),
            "boundary_entry": "B_%d" % start,
            "boundary_exit": "B_%d" % (start + width),
            "Scorable": bool(source["Scorable"]),
            "NetPositive": bool(source["NetPositive"]),
            "SoftRelativeStable": bool(source["SoftRelativeStable"]),
            "BiphasicStable": bool(source["BiphasicStable"]),
            "NewEligible": bool(source["NewEligible"]),
            "G_H": float(source["G_H"]),
            "G_K": float(source["G_K"]),
            "S_RATE": statistics["S_RATE"],
            "RateStable": bool(statistics["RateStable"]),
            "rate_SE_H": statistics["rate_SE_H"],
            "rate_SE_K": statistics["rate_SE_K"],
            "rate_c95": statistics["rate_c95"],
            "rate_LCB_H": statistics["rate_LCB_H"],
            "rate_LCB_K": statistics["rate_LCB_K"],
            "ranking_candidate": False,
            "point_top_tie": False,
            "selected": False,
            "display_rank": None,
            "selection_frequency": None,
            "best_q": None if best is None else best["q"],
            "best_tau": None if best is None else best["tau"],
            "tau_pair_frequency": None if best is None else best.get("tau_pair_frequency"),
            "v1_failure_reasons": list(source["new_eligibility_failures"]),
            "v2_failure_reasons": [],
        }
        working_rows.append(row)
        bootstrap_rates[(width, start)] = {"H": statistics["_bootstrap_rate_H"], "K": statistics["_bootstrap_rate_K"]}
    decision = selector_contract.select_scientific_state(working_rows, bootstrap_rates, replicates)
    _require(decision["decision"] in SCIENTIFIC_STATES, "selector produced an illegal terminal state")
    for row in working_rows:
        row["selection_frequency"] = row.pop("model_rate_selection_frequency")
        failures: List[str] = []
        if not row["NewEligible"]:
            failures.append("NOT_NEW_ELIGIBLE")
        if not row["RateStable"]:
            failures.append("RATE_NOT_STABLE")
        if row["ranking_candidate"] and not row["selected"]:
            if row["point_top_tie"]:
                failures.append("POINT_TOP_TIE")
            elif row["display_rank"] != 1:
                failures.append("NOT_POINT_TOP")
            elif decision["decision"] == "ABSTAIN_RATE_RANK_UNSTABLE":
                failures.append("RATE_RANK_UNSTABLE")
        row["v2_failure_reasons"] = failures
        _require(tuple(row) == selector_contract.PUBLIC_ROW_FIELDS, "selector row fields differ")
    canonical_rows = sorted(working_rows, key=lambda row: (int(row["width"]), int(row["start"])))
    return {
        "selector_decision": decision["decision"],
        "selected_key": decision["selected_key"],
        "selection_frequency": decision["selection_frequency"],
        "point_top_key": decision["point_top_key"],
        "point_top_tied_keys": decision["point_top_tied_keys"],
        "record_count": len(projected),
        "category_count": len({row["category"] for row in projected}),
        "candidate_count": len(canonical_rows),
        "candidate_counts_by_width": {"3": 12, "4": 11, "5": 10, "6": 9},
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "standard_deviation_ddof": 1,
            "draw_index_sha256": bootstrap["draw_index_sha256"],
            "category_stratified": True,
            "joint_reuse": True,
        },
        "v1_compatibility": {"status": "PASS", "view": compatibility},
        "rows": canonical_rows,
    }


def freeze_selector(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    _validate_expected_commit(expected_commit)
    manifest, _rows, _freeze, card, card_path, prompt_schema_path, trajectory_schema_path = _load_context(run_root)
    _require(manifest["scope"] == "formal", "official selector requires formal scope")
    verifier_path = Path(run_root).resolve() / "merge/verifier_receipt.json"
    records_path = Path(run_root).resolve() / "merge/merged_trajectory_records.jsonl"
    _require(verifier_path.is_file() and load_json(verifier_path).get("status") == "PASS", "fresh full verifier PASS is required before selector")
    records = _load_jsonl(records_path)
    analysis = analyze_selector(records=records, formal=True)
    projection = [_project for _project in (_project_selector_records(records))]
    projection_sha = _write_new_jsonl(Path(run_root).resolve() / "selector/selector_projection.jsonl", projection)
    payload = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-selector-freeze.v1",
        "gate": GATE,
        "run_root": str(Path(run_root).resolve()),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "formal_verifier_receipt_sha256": file_sha256(verifier_path),
        "trajectory_file_sha256": file_sha256(records_path),
        "selector_projection_file_sha256": projection_sha,
        "selector_projection_fields": ["identity", "category", "H", "D"],
        "hidden_diagnostics_in_selector": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "outcome_read": False,
        "loop_executed": False,
        "selector": analysis,
    }
    freeze_sha = _write_new_json(Path(run_root).resolve() / "selector/selector_freeze.json", payload)
    return {
        "status": "PASS",
        "selector_freeze_sha256": freeze_sha,
        "selector_projection_sha256": projection_sha,
        "selector_decision": analysis["selector_decision"],
        "selected_key": analysis["selected_key"],
        "selection_frequency": analysis["selection_frequency"],
        "candidate_count": analysis["candidate_count"],
        "record_count": analysis["record_count"],
        "subject_count": analysis["category_count"],
    }


def verify_selector(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    _validate_expected_commit(expected_commit)
    manifest, _rows, _freeze, _card, card_path, prompt_schema_path, trajectory_schema_path = _load_context(run_root)
    root = Path(run_root).resolve()
    selector_path = root / "selector/selector_freeze.json"
    projection_path = root / "selector/selector_projection.jsonl"
    verifier_path = root / "merge/verifier_receipt.json"
    records_path = root / "merge/merged_trajectory_records.jsonl"
    _require(selector_path.is_file() and projection_path.is_file() and verifier_path.is_file(), "selector closure is incomplete")
    frozen = load_json(selector_path)
    _require(frozen["expected_commit"] == expected_commit, "selector commit differs")
    _require(frozen["hidden_diagnostics_in_selector"] is False and frozen["outcome_read"] is False, "selector barrier differs")
    _require(file_sha256(projection_path) == frozen["selector_projection_file_sha256"], "selector projection hash differs")
    records = _load_jsonl(records_path)
    projected = _load_jsonl(projection_path)
    _require(projected == _project_selector_records(records), "selector projection differs from fresh projection")
    analysis = analyze_selector(records=records, formal=True)
    _require(analysis == frozen["selector"], "fresh selector decision/ranking differs")
    receipt = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-selector-verifier.v1",
        "gate": GATE,
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "formal_verifier_receipt_sha256": file_sha256(verifier_path),
        "selector_freeze_file_sha256": file_sha256(selector_path),
        "selector_projection_file_sha256": file_sha256(projection_path),
        "record_count": len(records),
        "subject_count": len({str(row["category"]) for row in projected}),
        "candidate_count": analysis["candidate_count"],
        "decision": analysis["selector_decision"],
        "selected_key": analysis["selected_key"],
        "selection_frequency": analysis["selection_frequency"],
        "ranking_digest": semantic_sha256(analysis["rows"]),
        "draw_index_sha256": analysis["bootstrap"]["draw_index_sha256"],
        "selector_projection_fields": ["identity", "category", "H", "D"],
        "hidden_diagnostics_in_selector": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "outcome_read": False,
        "loop_executed": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(root / "selector/verifier_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def build_sbatch_text(*, run_root: Path, expected_commit: str, partition: str, gpu_type: str, qos: str | None = None, time_limit: str | None = None, concurrency: int = ARRAY_CONCURRENCY, attempt: int = 1) -> str:
    _validate_expected_commit(expected_commit)
    _require(partition and gpu_type, "scheduler partition and GPU type are required")
    _require(1 <= int(concurrency) <= SHARD_COUNT, "array concurrency is invalid")
    _require(int(attempt) >= 1, "launcher attempt must be positive")
    manifest = load_json(Path(run_root).resolve() / "manifest/formal_manifest.json")
    scope = str(manifest.get("scope"))
    actual_time = time_limit or (DEBUG_TIME_LIMIT if scope == "debug_preflight" else FORMAL_TIME_LIMIT)
    runner = repository_root() / "scripts/loopscope/run_qwen4_phase6_mmlu5_gate_m.py"
    lines = [
        "#!/usr/bin/env bash",
        "# LoopScope Phase 6 Gate M native five-shot trajectory shard.",
        "#SBATCH --job-name=loopscope-p6-m-mmlu5-trajectory",
        "#SBATCH --partition=%s" % partition,
        "#SBATCH --array=0-%d%%%d" % (SHARD_COUNT - 1, int(concurrency)),
        "#SBATCH --time=%s" % actual_time,
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=%d" % CPUS_PER_TASK,
        "#SBATCH --mem=%s" % MEMORY,
        "#SBATCH --gres=gpu:%s:1" % gpu_type,
        "#SBATCH --output=%s/slurm/trajectory-%%A_%%a.out" % Path(run_root).resolve(),
        "#SBATCH --error=%s/slurm/trajectory-%%A_%%a.err" % Path(run_root).resolve(),
    ]
    if qos:
        lines.insert(5, "#SBATCH --qos=%s" % qos)
    lines.extend(
        [
            "set -euo pipefail",
            "source %s/bin/activate" % shlex.quote(str(AUDITED_VENV)),
            "cd %s" % shlex.quote(str(REMOTE_REPO)),
            "export PYTHONDONTWRITEBYTECODE=1",
            "export PYTHONPATH=%s" % shlex.quote(str(REMOTE_REPO / "src")),
            "export HF_HOME=%s" % shlex.quote(str(HF_HOME)),
            "export TRANSFORMERS_CACHE=%s" % shlex.quote(str(HF_HOME / "hub")),
            "export HF_DATASETS_CACHE=%s" % shlex.quote(str(HF_DATASETS_CACHE)),
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export HF_DATASETS_OFFLINE=1",
            "unset HF_ENDPOINT",
            "exec python %s shard --run-root %s --expected-commit %s --shard-index $SLURM_ARRAY_TASK_ID --attempt %d"
            % (shlex.quote(str(runner)), shlex.quote(str(run_root)), shlex.quote(expected_commit), int(attempt)),
            "",
        ]
    )
    text = "\n".join(lines)
    _require("#SBATCH --array=0-7%%%d" % int(concurrency) in text, "Gate M array layout drifted")
    _require("#SBATCH --cpus-per-task=8" in text and "#SBATCH --mem=64G" in text, "Gate M resources drifted")
    _require("#SBATCH --gres=gpu:%s:1" % gpu_type in text, "Gate M GPU request drifted")
    return text


def write_launcher(*, run_root: Path, expected_commit: str, partition: str, gpu_type: str, qos: str | None = None, time_limit: str | None = None, concurrency: int = ARRAY_CONCURRENCY, attempt: int = 1) -> Dict[str, Any]:
    manifest, _rows, _freeze, _card, _card_path, _prompt_schema_path, _trajectory_schema_path = _load_context(run_root)
    body = build_sbatch_text(run_root=run_root, expected_commit=expected_commit, partition=partition, gpu_type=gpu_type, qos=qos, time_limit=time_limit, concurrency=concurrency, attempt=attempt).encode("utf-8")
    root = Path(run_root).resolve()
    _require(int(attempt) >= 1, "launcher attempt must be positive")
    path = root / "slurm" / ("gate_m_trajectory_attempt-%04d.sbatch" % int(attempt))
    launcher_sha = _write_new_bytes(path, body)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu5-gate-m-launcher.v1",
        "gate": GATE,
        "scope": manifest["scope"],
        "run_root": str(root),
        "expected_commit": expected_commit,
        "path": str(path.relative_to(root)),
        "sha256": launcher_sha,
        "partition": partition,
        "qos": qos,
        "gpu_type": gpu_type,
        "shard_count": SHARD_COUNT,
        "concurrency": int(concurrency),
        "attempt": int(attempt),
        "time_limit": time_limit or (DEBUG_TIME_LIMIT if manifest["scope"] == "debug_preflight" else FORMAL_TIME_LIMIT),
        "cpus_per_task": CPUS_PER_TASK,
        "memory": MEMORY,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
    }
    receipt_sha = _write_new_json(root / "slurm/launcher_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def formal_runtime_dry_plan() -> Dict[str, Any]:
    return {
        "status": "GATE_M_FORMAL_RUNTIME_DRY_PLAN_IMPORTABLE",
        "gate": GATE,
        "formal_gpu": "A800",
        "formal_partition_policy": "highest_legal_normal_user_priority_A800",
        "validation_record_count": VALIDATION_RECORD_COUNT,
        "validation_subject_count": VALIDATION_SUBJECT_COUNT,
        "shard_count": SHARD_COUNT,
        "debug_preflight_required": True,
        "fresh_full_verifier_before_selector": True,
        "selector_execution_count": 0,
        "formal_acquisition_executed": False,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "cuda_gpu_slurm": False,
        "outcome_read": False,
    }


def dry_run() -> Dict[str, Any]:
    card, card_path, prompt_schema_path, trajectory_schema_path = _load_contract()
    candidates = enumerate_candidates()
    return {
        "status": "DRY_RUN_VALID",
        "gate": GATE,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "model_revision": card["model"]["revision"],
        "dataset_revision": card["task"]["revision"],
        "validation_record_count": VALIDATION_RECORD_COUNT,
        "validation_subject_count": VALIDATION_SUBJECT_COUNT,
        "candidate_count": len(candidates),
        "formal_bootstrap_replicates": FORMAL_REPLICATES,
        "formal_bootstrap_seed": FORMAL_SEED,
        "formal_acquisition": False,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
        "gate_m_formal_runtime_dry_plan": formal_runtime_dry_plan(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LoopScope Phase 6 Gate M MMLU five-shot runner")
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("dry-run")
    freeze = actions.add_parser("freeze-manifest")
    freeze.add_argument("--gate-k-root", type=Path, default=GATE_K_ROOT_DEFAULT)
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--expected-commit", required=True)
    freeze.add_argument("--shards", type=int, default=SHARD_COUNT)
    freeze.add_argument("--preflight", action="store_true")
    shard = actions.add_parser("shard")
    shard.add_argument("--run-root", type=Path, required=True)
    shard.add_argument("--expected-commit", required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--attempt", type=int, default=1)
    shard_verify = actions.add_parser("verify-shard")
    shard_verify.add_argument("--run-root", type=Path, required=True)
    shard_verify.add_argument("--expected-commit", required=True)
    shard_verify.add_argument("--shard-index", type=int, required=True)
    shard_verify.add_argument("--attempt", type=int, required=True)
    merge = actions.add_parser("merge")
    merge.add_argument("--run-root", type=Path, required=True)
    merge.add_argument("--expected-commit", required=True)
    verify = actions.add_parser("verify")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--job-id", required=True)
    verify.add_argument("--partition", required=True)
    launch = actions.add_parser("write-launcher")
    launch.add_argument("--run-root", type=Path, required=True)
    launch.add_argument("--expected-commit", required=True)
    launch.add_argument("--partition", required=True)
    launch.add_argument("--gpu-type", required=True)
    launch.add_argument("--qos")
    launch.add_argument("--time-limit")
    launch.add_argument("--concurrency", type=int, default=ARRAY_CONCURRENCY)
    launch.add_argument("--attempt", type=int, default=1)
    selector = actions.add_parser("selector")
    selector.add_argument("--run-root", type=Path, required=True)
    selector.add_argument("--expected-commit", required=True)
    selector_verify = actions.add_parser("verify-selector")
    selector_verify.add_argument("--run-root", type=Path, required=True)
    selector_verify.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        result = dry_run()
    elif args.command == "freeze-manifest":
        result = freeze_manifest(gate_k_root=args.gate_k_root, run_root=args.run_root, expected_commit=args.expected_commit, preflight=args.preflight, shard_count=args.shards)
    elif args.command == "shard":
        result = acquire_shard(run_root=args.run_root, expected_commit=args.expected_commit, shard_index=args.shard_index, attempt=args.attempt)
    elif args.command == "verify-shard":
        result = verify_shard(run_root=args.run_root, expected_commit=args.expected_commit, shard_index=args.shard_index, attempt=args.attempt)
    elif args.command == "merge":
        result = merge_shards(run_root=args.run_root, expected_commit=args.expected_commit)
    elif args.command == "verify":
        result = verify_formal_run(run_root=args.run_root, expected_commit=args.expected_commit, job_id=args.job_id, expected_partition=args.partition)
    elif args.command == "write-launcher":
        result = write_launcher(run_root=args.run_root, expected_commit=args.expected_commit, partition=args.partition, gpu_type=args.gpu_type, qos=args.qos, time_limit=args.time_limit, concurrency=args.concurrency, attempt=args.attempt)
    elif args.command == "selector":
        result = freeze_selector(run_root=args.run_root, expected_commit=args.expected_commit)
    else:
        result = verify_selector(run_root=args.run_root, expected_commit=args.expected_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AUTHORIZED_ADMISSION_BASE",
    "CARD_SHA256",
    "DEBUG_PARTITION",
    "DEBUG_TIME_LIMIT",
    "FORMAL_TIME_LIMIT",
    "GATE",
    "GateMMMLU5Error",
    "SHARD_COUNT",
    "analyze_selector",
    "build_sbatch_text",
    "dry_run",
    "formal_runtime_dry_plan",
    "freeze_manifest",
    "freeze_selector",
    "main",
    "merge_shards",
    "query_scheduler",
    "verify_formal_run",
    "verify_selector",
    "verify_shard",
    "write_launcher",
]
