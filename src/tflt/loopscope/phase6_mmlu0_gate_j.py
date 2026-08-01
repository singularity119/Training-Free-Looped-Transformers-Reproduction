"""LoopScope Phase 6 Gate J: formal MMLU-0 trajectory and V2 freeze.

This module is deliberately a thin Gate-J orchestration layer over the audited
Gate-I native producer.  It adds only the write-once formal manifest, deterministic
shards, merge/closure verification, and the one official outcome-blind selector
freeze.  It never reads labels, test data, outcomes, or runs a TFLT loop.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope import phase6_mmlu0_gate_i as gate_i
from tflt.loopscope.phase6_mmlu0_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    FORMAL_REPLICATES,
    FORMAL_SEED,
    LAYERS,
    VALIDATION_RECORD_COUNT,
    VALIDATION_SUBJECT_COUNT,
    canonical_json_bytes,
    file_sha256,
    load_json,
    semantic_sha256,
    selector_sample,
    validate_card,
    validate_schema_document,
    validate_trajectory_record,
)
from tflt.loopscope.phase6_mmlu0_selector import analyze_selector


GATE = "J"
EXECUTOR_THREAD_ID = "019fbf4b-5d69-7931-9724-16b03e2bc3f9"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
AUTHORIZED_BASE = "b2fc70167c7803645d442dad607945fdfacb8d0e"
CARD_RELATIVE = Path("configs/loopscope/phase6_mmlu0_prefix_card.json")
SCHEMA_RELATIVE = Path("configs/loopscope/phase6_mmlu0_trajectory_schema.json")
CARD_SHA256 = "a415307c6e33a5c90b2fc2072c1926af75bc6d00e2b36938c11b8dad263ff43a"
SCHEMA_SHA256 = "1d6d427f9bc1f34d8af19043661f2307989e72998c461df678feeade75c4da26"
EXPECTED_PROJECTION_SHA256 = "556cb1f02931da680088672c6b863398aaaebf71022d52b97bbb6cf4ad0f6280"
EXPECTED_ORDERED_IDENTITY_SHA256 = "20e224e90bb4db08ab18d447d017baa564b00b0a4c0080af2c28e7a029becfe2"
EXPECTED_VALIDATION_IDENTITY_SHA256 = "ced8dd2ac8277a30e4d4e101f8f76c2dd125659abe35af679f60e5c6a9f076eb"
EXPECTED_PROMPT_PROJECTION_SHA256 = "2be66cc4966f45135075d492cbb30ae67bd148431682e8588c69cee7b1de0278"
EXPECTED_TASK_SOURCE_SHA256 = "7e03a4bac9704839de842c099e068fb2ebba626a066adfc3499c69eb05336b68"
DEBUG_PARTITION = "debug"
DEBUG_TIME_LIMIT = "00:29:00"
DEBUG_CPUS = 8
DEBUG_MEMORY = "64G"
FORMAL_DEFAULT_TIME_LIMIT = "06:00:00"
MAX_SHARDS = 8
FORBIDDEN_PERSISTED_KEY_FRAGMENTS = (
    "target",
    "gold",
    "label",
    "correct",
    "accuracy",
    "gain",
    "flip",
    "outcome",
    "logit",
    "hidden_tensor",
    "input_id",
    "generated_text",
    "prompt_text",
)


class GateJMMLU0Error(RuntimeError):
    """Fail-closed Gate J contract or evidence error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateJMMLU0Error(message)


def _validate_expected_commit(expected_commit: str) -> None:
    _require(
        isinstance(expected_commit, str) and re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None,
        "Gate J expected commit is not a full Git SHA-1",
    )
    _require(expected_commit != AUTHORIZED_BASE, "Gate J producer commit cannot be the Gate I admission base")


def _write_new_bytes(path: Path, body: bytes) -> str:
    return gate_i._write_new_bytes(path, body)


def _write_new_json(path: Path, value: Any) -> str:
    return gate_i._write_new_json(path, value)


def _write_new_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> str:
    return gate_i._write_new_jsonl(path, values)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return gate_i._load_jsonl(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _repo_root() -> Path:
    return gate_i.repository_root()


def _load_card_schema() -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    card_path = _repo_root() / CARD_RELATIVE
    schema_path = _repo_root() / SCHEMA_RELATIVE
    _require(file_sha256(card_path) == CARD_SHA256, "Gate J card SHA-256 differs")
    _require(file_sha256(schema_path) == SCHEMA_SHA256, "Gate J schema SHA-256 differs")
    card = load_json(card_path)
    schema = load_json(schema_path)
    validate_card(card)
    validate_schema_document(schema)
    return card, schema, card_path, schema_path


def _projection_context(source_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read only the safe Gate-I projection and prove its semantic digests."""

    source_root = Path(source_root).resolve()
    cpu_receipt_path = source_root / "cpu/admission_receipt.json"
    projection_path = source_root / "cpu/validation_projection.jsonl"
    _require(cpu_receipt_path.is_file(), "Gate I CPU admission receipt is missing")
    _require(projection_path.is_file(), "Gate I validation projection is missing")
    cpu_receipt = load_json(cpu_receipt_path)
    _require(cpu_receipt.get("status") == "PASS", "Gate I CPU admission is not PASS")
    _require(cpu_receipt.get("git", {}).get("commit") == AUTHORIZED_BASE, "Gate I projection commit differs")
    _require(cpu_receipt.get("card_sha256") == CARD_SHA256, "Gate I CPU card hash differs")
    _require(cpu_receipt.get("schema_sha256") == SCHEMA_SHA256, "Gate I CPU schema hash differs")
    _require(file_sha256(projection_path) == EXPECTED_PROJECTION_SHA256, "Gate I projection file hash differs")
    rows = gate_i._projection_rows(projection_path)
    identities = [row["identity"] for row in rows]
    _require(
        semantic_sha256(identities) == EXPECTED_ORDERED_IDENTITY_SHA256,
        "Gate I ordered identity digest differs",
    )
    projection = cpu_receipt.get("projection", {})
    _require(
        projection.get("validation_identity_reuse_sha256") == EXPECTED_VALIDATION_IDENTITY_SHA256,
        "Gate I validation identity reuse digest differs",
    )
    _require(
        projection.get("ordered_prompt_projection_sha256") == EXPECTED_PROMPT_PROJECTION_SHA256,
        "Gate I zero-shot prompt projection digest differs",
    )
    _require(
        cpu_receipt.get("task_source", {}).get("manifest_sha256") == EXPECTED_TASK_SOURCE_SHA256,
        "Gate I task-source digest differs",
    )
    _require(len(rows) == VALIDATION_RECORD_COUNT, "Gate I projection population differs")
    _require(len({row["subject"] for row in rows}) == VALIDATION_SUBJECT_COUNT, "Gate I subject population differs")
    _require(all(int(row["ordinal"]) == index for index, row in enumerate(rows)), "Gate I ordinals are not canonical")
    return rows, {
        "source_root": str(source_root),
        "cpu_receipt_sha256": file_sha256(cpu_receipt_path),
        "projection_file_sha256": file_sha256(projection_path),
        "ordered_identity_sha256": EXPECTED_ORDERED_IDENTITY_SHA256,
        "validation_identity_reuse_sha256": EXPECTED_VALIDATION_IDENTITY_SHA256,
        "ordered_prompt_projection_sha256": EXPECTED_PROMPT_PROJECTION_SHA256,
        "task_source_manifest_sha256": EXPECTED_TASK_SOURCE_SHA256,
        "record_count": len(rows),
        "subject_count": len({row["subject"] for row in rows}),
    }


def _preflight_subset(rows: Sequence[Mapping[str, Any]], per_subject: int = 2, subject_limit: int = 8) -> List[Dict[str, Any]]:
    _require(per_subject > 0 and subject_limit > 0, "preflight subset limits must be positive")
    subjects = []
    for row in rows:
        subject = str(row["subject"])
        if subject not in subjects:
            subjects.append(subject)
        if len(subjects) == subject_limit:
            break
    selected = [
        dict(row)
        for subject in subjects
        for row in rows
        if str(row["subject"]) == subject
        and sum(1 for prior in rows if str(prior["subject"]) == subject and int(prior["ordinal"]) <= int(row["ordinal"])) <= per_subject
    ]
    selected.sort(key=lambda row: int(row["ordinal"]))
    _require(len(selected) == per_subject * subject_limit, "preflight subset is incomplete")
    return selected


def _shard_rows(rows: Sequence[Mapping[str, Any]], shard_count: int) -> List[List[Dict[str, Any]]]:
    _require(1 <= int(shard_count) <= MAX_SHARDS, "shard count must be between 1 and 8")
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(int(shard_count))]
    for row in rows:
        ordinal = int(row["ordinal"])
        shards[ordinal % int(shard_count)].append(dict(row))
    _require(sum(len(shard) for shard in shards) == len(rows), "shard membership count differs")
    return shards


def freeze_manifest(
    *,
    source_root: Path,
    run_root: Path,
    expected_commit: str,
    shard_count: int,
    preflight: bool = False,
) -> Dict[str, Any]:
    """Create a fresh formal-shaped manifest and deterministic shard files."""

    _validate_expected_commit(expected_commit)
    _require(not Path(run_root).exists(), "Gate J run root must be fresh")
    all_rows, source = _projection_context(source_root)
    rows = _preflight_subset(all_rows) if preflight else [dict(row) for row in all_rows]
    root = Path(run_root).resolve()
    for name in ("cpu", "manifest", "shards", "slurm", "merge", "selector"):
        (root / name).mkdir(parents=True, exist_ok=False)
    projection_source = Path(source["source_root"]) / "cpu/validation_projection.jsonl"
    projection_path = root / "cpu/validation_projection.jsonl"
    projection_body = projection_source.read_bytes()
    projection_sha = _write_new_bytes(projection_path, projection_body)
    _require(projection_sha == source["projection_file_sha256"], "copied Gate I projection hash differs")
    selected_projection_path = root / "manifest/selected_projection.jsonl"
    selected_projection_sha = _write_new_jsonl(selected_projection_path, rows)
    shards = _shard_rows(rows, shard_count)
    shard_files: List[Dict[str, Any]] = []
    for shard_index, members in enumerate(shards):
        shard_payload = {
            "schema_version": "loopscope.phase6.mmlu0-gate-j-shard.v1",
            "gate": GATE,
            "scope": "debug_preflight" if preflight else "formal",
            "expected_commit": expected_commit,
            "shard_index": shard_index,
            "shard_count": int(shard_count),
            "assignment": "ordinal_modulo",
            "members": members,
            "record_count": len(members),
            "identity_order_sha256": semantic_sha256([row["identity"] for row in members]),
            "selector_executed": False,
            "outcome_read": False,
            "test_split_read": False,
        }
        shard_payload["manifest_sha256"] = semantic_sha256(shard_payload)
        shard_path = root / "shards" / ("shard-%03d.json" % shard_index)
        shard_sha = _write_new_json(shard_path, shard_payload)
        shard_files.append(
            {
                "shard_index": shard_index,
                "record_count": len(members),
                "identity_order_sha256": shard_payload["identity_order_sha256"],
                "path": str(shard_path.relative_to(root)),
                "file_sha256": shard_sha,
                "manifest_sha256": shard_payload["manifest_sha256"],
            }
        )
    manifest_payload = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-manifest.v1",
        "gate": GATE,
        "scope": "debug_preflight" if preflight else "formal",
        "created_at_utc": _utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": CARD_SHA256,
        "schema_sha256": SCHEMA_SHA256,
        "source_cpu_root": source["source_root"],
        "source_cpu_receipt_sha256": source["cpu_receipt_sha256"],
        "source_projection_file_sha256": source["projection_file_sha256"],
        "projection_file_sha256": projection_sha,
        "selected_projection_file_sha256": selected_projection_sha,
        "ordered_identity_sha256": semantic_sha256([row["identity"] for row in rows]),
        "ordered_prompt_projection_sha256": semantic_sha256(rows),
        "validation_identity_reuse_sha256": source["validation_identity_reuse_sha256"],
        "task_source_manifest_sha256": source["task_source_manifest_sha256"],
        "record_count": len(rows),
        "subject_count": len({row["subject"] for row in rows}),
        "canonical_population_count": len(all_rows),
        "shard_count": int(shard_count),
        "assignment": "ordinal_modulo",
        "shards": shard_files,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
        "test_split_read": False,
        "label_read": False,
        "raw_text_persisted": False,
        "token_ids_persisted": False,
        "full_vocab_persisted": False,
        "hidden_tensors_persisted": False,
    }
    manifest_payload["manifest_sha256"] = semantic_sha256(manifest_payload)
    manifest_path = root / "manifest/formal_manifest.json"
    manifest_sha = _write_new_json(manifest_path, manifest_payload)
    freeze_payload = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-freeze.v1",
        "gate": GATE,
        "scope": manifest_payload["scope"],
        "run_root": str(root),
        "expected_commit": expected_commit,
        "formal_manifest_path": str(manifest_path.relative_to(root)),
        "formal_manifest_file_sha256": manifest_sha,
        "formal_manifest_semantic_sha256": manifest_payload["manifest_sha256"],
        "projection_file_sha256": projection_sha,
        "selected_projection_file_sha256": selected_projection_sha,
        "shard_count": int(shard_count),
        "record_count": len(rows),
        "subject_count": len({row["subject"] for row in rows}),
        "selector_executed": False,
        "outcome_read": False,
        "information_barrier": {
            "test_split_read": False,
            "label_read": False,
            "outcome_read": False,
            "loop_executed": False,
            "raw_text_persisted": False,
            "token_ids_persisted": False,
            "full_vocab_persisted": False,
            "hidden_tensors_persisted": False,
        },
    }
    freeze_sha = _write_new_json(root / "manifest/freeze_receipt.json", freeze_payload)
    return {
        "status": "PASS",
        "run_root": str(root),
        "scope": manifest_payload["scope"],
        "manifest_sha256": manifest_sha,
        "manifest_semantic_sha256": manifest_payload["manifest_sha256"],
        "freeze_receipt_sha256": freeze_sha,
        "projection_file_sha256": projection_sha,
        "selected_projection_file_sha256": selected_projection_sha,
        "record_count": len(rows),
        "subject_count": len({row["subject"] for row in rows}),
        "shard_count": int(shard_count),
        "selector_executed": False,
        "outcome_read": False,
    }


def _load_context(root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Path, Path]:
    root = Path(root).resolve()
    manifest_path = root / "manifest/formal_manifest.json"
    freeze_path = root / "manifest/freeze_receipt.json"
    projection_path = root / "manifest/selected_projection.jsonl"
    _require(manifest_path.is_file() and freeze_path.is_file(), "Gate J manifest freeze is incomplete")
    manifest = load_json(manifest_path)
    semantic = manifest.pop("manifest_sha256", None)
    _require(isinstance(semantic, str) and semantic_sha256(manifest) == semantic, "formal manifest semantic hash differs")
    manifest["manifest_sha256"] = semantic
    _validate_expected_commit(str(manifest["expected_commit"]))
    _require(manifest["card_sha256"] == CARD_SHA256 and manifest["schema_sha256"] == SCHEMA_SHA256, "formal manifest contract hash differs")
    _require(file_sha256(manifest_path) == load_json(freeze_path)["formal_manifest_file_sha256"], "formal manifest file hash differs")
    copied_projection = root / "cpu/validation_projection.jsonl"
    _require(copied_projection.is_file(), "copied Gate I projection is missing")
    _require(file_sha256(copied_projection) == manifest["projection_file_sha256"], "copied Gate I projection hash differs")
    rows = _load_jsonl(projection_path)
    _require(file_sha256(projection_path) == manifest["selected_projection_file_sha256"], "selected projection hash differs")
    _require(len(rows) == int(manifest["record_count"]), "selected projection count differs")
    _require(semantic_sha256(rows) == manifest["ordered_prompt_projection_sha256"], "selected projection semantic hash differs")
    _require(
        [row["identity"] for row in rows] == [row["identity"] for row in sorted(rows, key=lambda value: int(value["ordinal"]))],
        "selected projection order differs",
    )
    card, schema, card_path, schema_path = _load_card_schema()
    return manifest, rows, load_json(freeze_path), card_path, schema_path


def _reconstruct_prompts(
    task_by_subject: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> Dict[str, str]:
    try:
        from datasets import DownloadMode, load_dataset
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateJMMLU0Error("datasets import failed during Gate J") from exc
    wanted = {canonical_json_bytes(row["identity"]): row for row in selected}
    prompts: Dict[str, str] = {}
    counters: Counter[str] = Counter()
    for subject in sorted({str(row["subject"]) for row in selected}):
        task_name = "mmlu_%s" % subject
        _require(task_name in task_by_subject, "formal subject has no standard MMLU task")
        dataset = load_dataset(
            path=gate_i.DATASET_REPO,
            name=subject,
            revision=gate_i.DATASET_REVISION,
            split="validation",
            cache_dir=str(gate_i.HF_DATASETS_CACHE),
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        ).select_columns(["question", "choices", "subject"])
        for row in dataset:
            safe = gate_i._safe_doc(row, subject)
            index = counters[subject]
            counters[subject] += 1
            identity = {
                "task": "mmlu",
                "doc_id": "mmlu_%s:validation:%d" % (subject, index),
                "doc_hash": gate_i._safe_doc_hash(safe),
            }
            key = canonical_json_bytes(identity)
            wanted_row = wanted.get(key)
            if wanted_row is None:
                continue
            prompt = gate_i._render_zero_shot_prompt(task_by_subject[task_name], safe)
            token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            _require(token_ids, "formal prompt tokenizes empty")
            _require(gate_i._sha256_bytes(prompt.encode("utf-8")) == wanted_row["prompt_sha256"], "formal prompt hash differs")
            _require(gate_i._sha256_bytes(canonical_json_bytes(token_ids)) == wanted_row["prompt_token_ids_sha256"], "formal prompt token hash differs")
            _require(len(token_ids) == int(wanted_row["sequence_length"]), "formal prompt sequence length differs")
            prompts[key.decode("utf-8")] = prompt
    _require(len(prompts) == len(selected), "formal prompt reconstruction is incomplete")
    return prompts


def _attempt_root(root: Path, shard_index: int, attempt: int) -> Path:
    _require(attempt >= 1, "attempt must be positive")
    return Path(root).resolve() / "shards" / ("shard-%03d" % int(shard_index)) / ("attempt-%04d" % int(attempt))


RUNTIME_EVIDENCE_FIELDS = {
    "identity",
    "subject",
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


def _validate_runtime_fact(fact: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    _require(set(fact) == RUNTIME_EVIDENCE_FIELDS, "runtime evidence fields are not closed")
    _require(fact["identity"] == record["identity"] and fact["subject"] == record["subject"], "runtime evidence identity differs")
    _require(fact["choice_surface_manifest_sha256"] == CHOICE_SURFACE_MANIFEST_SHA256, "runtime choice surface hash differs")
    _require(fact["final_norm_prehook_calls"] == 1, "FinalNorm pre-hook count differs")
    _require(fact["final_norm_boundary_calls"] == BOUNDARY_COUNT - 1, "FinalNorm boundary count differs")
    _require(float(fact["final_norm_closure_max_abs"]) == 0.0, "FinalNorm closure is nonzero")
    _require(fact["boundary_count"] == BOUNDARY_COUNT and fact["transition_count"] == LAYERS, "trajectory dimensions differ")
    _require(fact["forward_count"] == 1 and fact["loop_insertions"] == 0, "native forward closure differs")
    _require(fact["full_vocab_persisted"] is False and fact["hidden_tensors_persisted"] is False, "forbidden tensor persistence flag differs")
    _require(fact["raw_text_persisted"] is False and fact["token_ids_persisted"] is False, "forbidden text/token persistence flag differs")


def acquire_shard(*, run_root: Path, expected_commit: str, shard_index: int, attempt: int = 1) -> Dict[str, Any]:
    _validate_expected_commit(expected_commit)
    manifest, _rows, _freeze, card_path, _schema_path = _load_context(run_root)
    shard_index = int(shard_index)
    _require(0 <= shard_index < int(manifest["shard_count"]), "shard index is outside frozen domain")
    attempt_root = _attempt_root(run_root, shard_index, attempt)
    _require(not attempt_root.exists(), "shard attempt root must be fresh")
    shard_path = Path(run_root).resolve() / "shards" / ("shard-%03d.json" % shard_index)
    shard = load_json(shard_path)
    shard_semantic = shard.pop("manifest_sha256", None)
    _require(isinstance(shard_semantic, str) and semantic_sha256(shard) == shard_semantic, "shard manifest semantic hash differs")
    shard["manifest_sha256"] = shard_semantic
    _require(file_sha256(shard_path) == next(item["file_sha256"] for item in manifest["shards"] if int(item["shard_index"]) == shard_index), "shard manifest file hash differs")
    attempt_root.mkdir(parents=True, exist_ok=False)
    git = gate_i._git_provenance(expected_commit)
    card, _schema, _card_path, _ = _load_card_schema()
    task_by_subject = gate_i._task_map()
    torch_module, tokenizer, model, final_norm, lm_head, choice_ids = gate_i._load_native_runtime(card)
    members = list(shard["members"])
    prompts = _reconstruct_prompts(task_by_subject, members, tokenizer)
    records: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    by_key = {canonical_json_bytes(row["identity"]): row for row in members}
    for member in members:
        key = canonical_json_bytes(member["identity"]).decode("utf-8")
        record, fact = gate_i._trajectory_from_forward(
            prompt=prompts[key],
            projection_row=by_key[canonical_json_bytes(member["identity"])],
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
    records_path = attempt_root / "trajectory_records.jsonl"
    evidence_path = attempt_root / "runtime_evidence.jsonl"
    records_sha = _write_new_jsonl(records_path, records)
    evidence_sha = _write_new_jsonl(evidence_path, evidence)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-shard-attempt.v1",
        "gate": GATE,
        "scope": manifest["scope"],
        "created_at_utc": _utc_now(),
        "run_root": str(Path(run_root).resolve()),
        "expected_commit": expected_commit,
        "shard_index": shard_index,
        "shard_count": int(manifest["shard_count"]),
        "attempt": int(attempt),
        "git": git,
        "card_sha256": file_sha256(card_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "shard_manifest_sha256": shard["manifest_sha256"],
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
        "outcome_read": False,
        "test_split_read": False,
        "label_read": False,
        "forbidden_persisted_payloads": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(attempt_root / "shard_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha, "attempt_root": str(attempt_root)}


def _valid_attempts(root: Path, shard_index: int) -> List[Tuple[int, Path, Dict[str, Any]]]:
    shard_root = Path(root).resolve() / "shards" / ("shard-%03d" % int(shard_index))
    results = []
    if not shard_root.is_dir():
        return results
    for path in sorted(shard_root.glob("attempt-*") ):
        receipt_path = path / "shard_receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = load_json(receipt_path)
        if receipt.get("status") != "PASS":
            continue
        match = re.fullmatch(r"attempt-(\d{4})", path.name)
        _require(match is not None, "shard attempt directory name is invalid")
        results.append((int(match.group(1)), path, receipt))
    return results


def merge_shards(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    _validate_expected_commit(expected_commit)
    manifest, rows, _freeze, card_path, schema_path = _load_context(run_root)
    card = load_json(_repo_root() / CARD_RELATIVE)
    root = Path(run_root).resolve()
    selected: Dict[int, Tuple[Path, Dict[str, Any]]] = {}
    all_records: Dict[bytes, Dict[str, Any]] = {}
    all_evidence: Dict[bytes, Dict[str, Any]] = {}
    attempt_receipts: Dict[str, Any] = {}
    for shard_index in range(int(manifest["shard_count"])):
        attempts = _valid_attempts(root, shard_index)
        _require(len(attempts) == 1, "each shard must have exactly one valid PASS attempt")
        attempt, attempt_root, receipt = attempts[0]
        selected[shard_index] = (attempt_root, receipt)
        attempt_receipts[str(shard_index)] = {
            "attempt": attempt,
            "receipt_sha256": file_sha256(attempt_root / "shard_receipt.json"),
            "record_file_sha256": receipt["record_file_sha256"],
            "evidence_file_sha256": receipt["evidence_file_sha256"],
        }
        records_path = attempt_root / "trajectory_records.jsonl"
        evidence_path = attempt_root / "runtime_evidence.jsonl"
        _require(file_sha256(records_path) == receipt["record_file_sha256"], "shard trajectory hash differs")
        _require(file_sha256(evidence_path) == receipt["evidence_file_sha256"], "shard evidence hash differs")
        records = _load_jsonl(records_path)
        evidence = _load_jsonl(evidence_path)
        _require(len(records) == len(evidence) == int(receipt["record_count"]), "shard record/evidence count differs")
        for record, fact in zip(records, evidence):
            validate_trajectory_record(record, card)
            key = canonical_json_bytes(record["identity"])
            _require(key not in all_records, "duplicate identity across shard attempts")
            _validate_runtime_fact(fact, record)
            all_records[key] = record
            all_evidence[key] = fact
    expected = {canonical_json_bytes(row["identity"]): row for row in rows}
    _require(set(all_records) == set(expected), "merged identity membership differs")
    ordered_records = [all_records[canonical_json_bytes(row["identity"])] for row in rows]
    ordered_evidence = [all_evidence[canonical_json_bytes(row["identity"])] for row in rows]
    merge_records_path = root / "merge/merged_trajectory_records.jsonl"
    merge_evidence_path = root / "merge/merged_runtime_evidence.jsonl"
    records_sha = _write_new_jsonl(merge_records_path, ordered_records)
    evidence_sha = _write_new_jsonl(merge_evidence_path, ordered_evidence)
    merge_payload = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-merge.v1",
        "gate": GATE,
        "scope": manifest["scope"],
        "run_root": str(root),
        "expected_commit": expected_commit,
        "manifest_sha256": manifest["manifest_sha256"],
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "record_file_sha256": records_sha,
        "evidence_file_sha256": evidence_sha,
        "record_count": len(ordered_records),
        "subject_count": len({row["subject"] for row in ordered_records}),
        "identity_order_sha256": semantic_sha256([row["identity"] for row in ordered_records]),
        "forward_count": sum(int(row["forward_count"]) for row in ordered_evidence),
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "loop_insertions": sum(int(row["loop_insertions"]) for row in ordered_evidence),
        "shards": attempt_receipts,
        "selector_executed": False,
        "outcome_read": False,
        "test_split_read": False,
        "label_read": False,
        "forbidden_persisted_payloads": False,
        "status": "PASS",
    }
    merge_sha = _write_new_json(root / "merge/merge_receipt.json", merge_payload)
    return {**merge_payload, "merge_receipt_sha256": merge_sha}


SACCT_FIELDS = (
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


def query_scheduler(job_id: str) -> Tuple[List[str], List[Dict[str, str]]]:
    _require(re.fullmatch(r"[0-9]+", str(job_id)) is not None, "Slurm job ID is invalid")
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
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|")
        if len(fields) != len(SACCT_FIELDS):
            continue
        rows.append(dict(zip(SACCT_FIELDS, fields)))
    _require(rows, "sacct returned no rows")
    return command, rows


def _scheduler_acceptance(
    rows: Sequence[Mapping[str, str]],
    job_id: str,
    expected_shards: int,
    expected_partition: str,
) -> Dict[str, Any]:
    task_rows = [row for row in rows if re.fullmatch(r"%s_[0-9]+" % re.escape(str(job_id)), str(row["JobIDRaw"]))]
    if expected_shards == 1 and not task_rows:
        task_rows = [row for row in rows if str(row["JobIDRaw"]) == str(job_id)]
    _require(len(task_rows) == int(expected_shards), "scheduler task count differs")
    expected_ids = {"%s_%d" % (job_id, index) for index in range(int(expected_shards))}
    observed_ids = {str(row["JobIDRaw"]) for row in task_rows}
    if expected_shards == 1 and observed_ids == {str(job_id)}:
        expected_ids = {str(job_id)}
    _require(observed_ids == expected_ids, "scheduler task membership differs")
    for row in task_rows:
        _require(str(row["Partition"]) == expected_partition, "scheduler partition differs")
        _require(str(row["State"]).startswith("COMPLETED"), "formal scheduler state is not COMPLETED")
        _require(str(row["ExitCode"]) == "0:0", "formal scheduler ExitCode is not 0:0")
    return {
        "job_id": str(job_id),
        "expected_shards": int(expected_shards),
        "partition": expected_partition,
        "task_rows": [dict(row) for row in task_rows],
    }


def verify_formal_run(
    *,
    run_root: Path,
    expected_commit: str,
    job_id: str,
    expected_partition: str,
) -> Dict[str, Any]:
    """Fresh-process verifier for formal or debug-shaped trajectory closure."""

    _validate_expected_commit(expected_commit)
    manifest, rows, _freeze, card_path, schema_path = _load_context(run_root)
    root = Path(run_root).resolve()
    _require(manifest["expected_commit"] == expected_commit, "manifest commit differs")
    _require(manifest["selector_executed"] is False and manifest["outcome_read"] is False, "manifest information barrier differs")
    merge_receipt_path = root / "merge/merge_receipt.json"
    records_path = root / "merge/merged_trajectory_records.jsonl"
    evidence_path = root / "merge/merged_runtime_evidence.jsonl"
    _require(merge_receipt_path.is_file() and records_path.is_file() and evidence_path.is_file(), "merge closure is missing")
    merge = load_json(merge_receipt_path)
    _require(merge["status"] == "PASS", "merge receipt is not PASS")
    _require(file_sha256(records_path) == merge["record_file_sha256"], "merged trajectory hash differs")
    _require(file_sha256(evidence_path) == merge["evidence_file_sha256"], "merged evidence hash differs")
    card = load_json(_repo_root() / CARD_RELATIVE)
    records = _load_jsonl(records_path)
    evidence = _load_jsonl(evidence_path)
    _require(len(records) == len(rows) == int(manifest["record_count"]), "formal population count differs")
    _require(len(evidence) == len(records), "merged evidence count differs")
    expected = {canonical_json_bytes(row["identity"]): row for row in rows}
    observed = set()
    for record, fact in zip(records, evidence):
        validate_trajectory_record(record, card)
        key = canonical_json_bytes(record["identity"])
        _require(key in expected and key not in observed, "formal identity membership or duplicate differs")
        observed.add(key)
        _validate_runtime_fact(fact, record)
    _require(observed == set(expected), "formal missing or extra identity")
    _require(semantic_sha256([row["identity"] for row in records]) == manifest["ordered_identity_sha256"], "formal identity order hash differs")
    _require(sum(int(row["forward_count"]) for row in evidence) == len(records), "formal forward count differs")
    _require(sum(int(row["loop_insertions"]) for row in evidence) == 0, "loop insertion count is nonzero")
    if manifest["scope"] == "formal":
        _require(len(records) == VALIDATION_RECORD_COUNT, "formal run is not validation-1531")
        _require(len({row["subject"] for row in records}) == VALIDATION_SUBJECT_COUNT, "formal subject count differs")
    scheduler_command, scheduler_rows = query_scheduler(job_id)
    scheduler = _scheduler_acceptance(scheduler_rows, job_id, int(manifest["shard_count"]), expected_partition)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-verifier.v1",
        "gate": GATE,
        "scope": manifest["scope"],
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "merge_receipt_sha256": file_sha256(merge_receipt_path),
        "record_file_sha256": file_sha256(records_path),
        "evidence_file_sha256": file_sha256(evidence_path),
        "record_count": len(records),
        "subject_count": len({row["subject"] for row in records}),
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "forward_count": len(records),
        "loop_insertions": 0,
        "scheduler": {"query": scheduler_command, **scheduler},
        "selector_executed": False,
        "test_split_read": False,
        "label_read": False,
        "outcome_read": False,
        "forbidden_persisted_payloads": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(root / "merge/verifier_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def build_sbatch_text(
    *,
    run_root: Path,
    expected_commit: str,
    shard_count: int,
    partition: str,
    gpu_type: str,
    qos: str | None = None,
    time_limit: str = FORMAL_DEFAULT_TIME_LIMIT,
    concurrency: int | None = None,
) -> str:
    _validate_expected_commit(expected_commit)
    _require(1 <= int(shard_count) <= MAX_SHARDS, "shard count must be between 1 and 8")
    _require(partition and gpu_type and time_limit, "scheduler resource fields must be explicit")
    concurrency = int(concurrency or shard_count)
    _require(1 <= concurrency <= int(shard_count), "concurrency must be within shard count")
    runner = _repo_root() / "scripts/loopscope/run_qwen4_phase6_mmlu0_gate_j.py"
    lines = [
        "#!/usr/bin/env bash",
        "# LoopScope Phase 6 Gate J trajectory shard; no loop/outcome/selector.",
        "#SBATCH --job-name=loopscope-p6-j-trajectory",
        "#SBATCH --partition=%s" % partition,
        "#SBATCH --array=0-%d%%%d" % (int(shard_count) - 1, concurrency),
        "#SBATCH --time=%s" % time_limit,
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=%d" % DEBUG_CPUS,
        "#SBATCH --mem=%s" % DEBUG_MEMORY,
        "#SBATCH --gres=gpu:%s:1" % gpu_type,
        "#SBATCH --output=%s/slurm/formal-%%A_%%a.out" % run_root,
        "#SBATCH --error=%s/slurm/formal-%%A_%%a.err" % run_root,
    ]
    if qos:
        lines.insert(4, "#SBATCH --qos=%s" % qos)
    lines.extend(
        [
            "set -euo pipefail",
            "source %s/bin/activate" % shlex.quote(str(gate_i.AUDITED_VENV)),
            "cd %s" % shlex.quote(str(gate_i.REMOTE_REPO)),
            "export PYTHONDONTWRITEBYTECODE=1",
            "export PYTHONPATH=%s" % shlex.quote(str(gate_i.REMOTE_REPO / "src")),
            "export HF_HOME=%s" % shlex.quote(str(gate_i.HF_HOME)),
            "export TRANSFORMERS_CACHE=%s" % shlex.quote(str(gate_i.HF_HOME / "hub")),
            "export HF_DATASETS_CACHE=%s" % shlex.quote(str(gate_i.HF_DATASETS_CACHE)),
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export HF_DATASETS_OFFLINE=1",
            "unset HF_ENDPOINT",
            "exec python %s shard --run-root %s --expected-commit %s --shard-index $SLURM_ARRAY_TASK_ID --attempt 1"
            % (shlex.quote(str(runner)), shlex.quote(str(run_root)), shlex.quote(expected_commit)),
            "",
        ]
    )
    text = "\n".join(lines)
    _require("--array=0-%d%%%d" % (int(shard_count) - 1, concurrency) in text, "formal array layout drifted")
    _require("--gres=gpu:%s:1" % gpu_type in text, "formal GPU type drifted")
    return text


def write_launcher(
    *,
    run_root: Path,
    expected_commit: str,
    partition: str,
    gpu_type: str,
    qos: str | None = None,
    time_limit: str = FORMAL_DEFAULT_TIME_LIMIT,
    concurrency: int | None = None,
) -> Dict[str, Any]:
    _validate_expected_commit(expected_commit)
    manifest, _rows, _freeze, _card_path, _schema_path = _load_context(run_root)
    _require(manifest["expected_commit"] == expected_commit, "launcher commit differs")
    body = build_sbatch_text(
        run_root=Path(run_root).resolve(),
        expected_commit=expected_commit,
        shard_count=int(manifest["shard_count"]),
        partition=partition,
        gpu_type=gpu_type,
        qos=qos,
        time_limit=time_limit,
        concurrency=concurrency,
    ).encode("utf-8")
    path = Path(run_root).resolve() / "slurm/gate_j_trajectory.sbatch"
    sha = _write_new_bytes(path, body)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-launcher.v1",
        "gate": GATE,
        "run_root": str(Path(run_root).resolve()),
        "expected_commit": expected_commit,
        "path": str(path.relative_to(Path(run_root).resolve())),
        "sha256": sha,
        "partition": partition,
        "qos": qos,
        "gpu_type": gpu_type,
        "shard_count": int(manifest["shard_count"]),
        "concurrency": int(concurrency or manifest["shard_count"]),
        "time_limit": time_limit,
        "cpus_per_task": DEBUG_CPUS,
        "memory": DEBUG_MEMORY,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
    }
    receipt_sha = _write_new_json(Path(run_root).resolve() / "slurm/launcher_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def freeze_selector(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    _validate_expected_commit(expected_commit)
    manifest, _rows, _freeze, card_path, schema_path = _load_context(run_root)
    _require(manifest["expected_commit"] == expected_commit, "selector commit differs")
    _require(manifest["scope"] == "formal", "official selector requires formal scope")
    merge = load_json(Path(run_root).resolve() / "merge/merge_receipt.json")
    verifier_path = Path(run_root).resolve() / "merge/verifier_receipt.json"
    _require(merge.get("status") == "PASS" and verifier_path.is_file(), "fresh full verifier PASS is required before selector")
    _require(load_json(verifier_path).get("status") == "PASS", "full verifier receipt is not PASS")
    records_path = Path(run_root).resolve() / "merge/merged_trajectory_records.jsonl"
    records = _load_jsonl(records_path)
    result = analyze_selector(records, replicates=FORMAL_REPLICATES, seed=FORMAL_SEED, formal=True)
    projection = [selector_sample(record) for record in records]
    projection_path = Path(run_root).resolve() / "selector/selector_projection.jsonl"
    projection_sha = _write_new_jsonl(projection_path, projection)
    payload = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-selector-freeze.v1",
        "gate": GATE,
        "run_root": str(Path(run_root).resolve()),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "formal_verifier_receipt_sha256": file_sha256(verifier_path),
        "trajectory_file_sha256": file_sha256(records_path),
        "selector_projection_file_sha256": projection_sha,
        "selector_projection_fields": ["identity", "category", "H", "D"],
        "hidden_diagnostics_in_selector": False,
        "outcomes_read": False,
        "test_split_read": False,
        "label_read": False,
        "loop_executed": False,
        "selector": result,
    }
    freeze_sha = _write_new_json(Path(run_root).resolve() / "selector/selector_freeze.json", payload)
    return {
        "status": "PASS",
        "selector_freeze_sha256": freeze_sha,
        "selector_projection_sha256": projection_sha,
        "selector_decision": result["selector_decision"],
        "selected_key": result["selected_key"],
        "selection_frequency": result["selection_frequency"],
        "point_top_key": result["point_top_key"],
        "candidate_count": result["candidate_count"],
        "record_count": result["record_count"],
        "subject_count": result["category_count"],
    }


def verify_selector(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    _validate_expected_commit(expected_commit)
    manifest, _rows, _freeze, card_path, schema_path = _load_context(run_root)
    _require(manifest["expected_commit"] == expected_commit, "selector commit differs")
    selector_path = Path(run_root).resolve() / "selector/selector_freeze.json"
    projection_path = Path(run_root).resolve() / "selector/selector_projection.jsonl"
    verifier_path = Path(run_root).resolve() / "merge/verifier_receipt.json"
    records_path = Path(run_root).resolve() / "merge/merged_trajectory_records.jsonl"
    _require(selector_path.is_file() and projection_path.is_file() and verifier_path.is_file(), "selector closure is missing")
    frozen = load_json(selector_path)
    _require(frozen["expected_commit"] == expected_commit, "selector commit differs")
    _require(frozen["outcomes_read"] is False and frozen["test_split_read"] is False and frozen["label_read"] is False, "selector information barrier differs")
    _require(file_sha256(projection_path) == frozen["selector_projection_file_sha256"], "selector projection hash differs")
    records = _load_jsonl(records_path)
    projected = _load_jsonl(projection_path)
    _require(projected == [selector_sample(record) for record in records], "selector projection differs from fresh record projection")
    result = analyze_selector(records, replicates=FORMAL_REPLICATES, seed=FORMAL_SEED, formal=True)
    _require(result == frozen["selector"], "fresh selector decision/ranking differs")
    payload_hash = file_sha256(selector_path)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-selector-verifier.v1",
        "gate": GATE,
        "run_root": str(Path(run_root).resolve()),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "formal_verifier_receipt_sha256": file_sha256(verifier_path),
        "selector_freeze_file_sha256": payload_hash,
        "selector_projection_file_sha256": file_sha256(projection_path),
        "record_count": len(records),
        "subject_count": len({row["category"] for row in projected}),
        "candidate_count": result["candidate_count"],
        "decision": result["selector_decision"],
        "selected_key": result["selected_key"],
        "selection_frequency": result["selection_frequency"],
        "ranking_digest": semantic_sha256(result["rows"]),
        "draw_index_sha256": result["bootstrap"]["draw_index_sha256"],
        "selector_projection_fields": ["identity", "category", "H", "D"],
        "hidden_diagnostics_in_selector": False,
        "outcomes_read": False,
        "test_split_read": False,
        "label_read": False,
        "loop_executed": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(Path(run_root).resolve() / "selector/verifier_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def aggregate_diagnostics(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    """Produce only layerwise scalar aggregates after selector closure."""

    _validate_expected_commit(expected_commit)
    manifest, _rows, _freeze, _card_path, _schema_path = _load_context(run_root)
    _require(manifest["expected_commit"] == expected_commit, "aggregate commit differs")
    selector_verifier = Path(run_root).resolve() / "selector/verifier_receipt.json"
    _require(selector_verifier.is_file() and load_json(selector_verifier).get("status") == "PASS", "selector verifier PASS is required")
    records = _load_jsonl(Path(run_root).resolve() / "merge/merged_trajectory_records.jsonl")
    aggregate = {
        "schema_version": "loopscope.phase6.mmlu0-gate-j-diagnostic-aggregate.v1",
        "gate": GATE,
        "run_root": str(Path(run_root).resolve()),
        "expected_commit": expected_commit,
        "record_count": len(records),
        "subject_count": len({row["subject"] for row in records}),
        "layerwise": [],
        "transitionwise": [],
        "information_barrier": {
            "outcomes_read": False,
            "test_split_read": False,
            "label_read": False,
            "loop_executed": False,
        },
    }
    for index in range(BOUNDARY_COUNT):
        boundaries = [row["boundaries"][index] for row in records]
        aggregate["layerwise"].append(
            {
                "boundary_id": "B_%d" % index,
                "mean_choice_entropy": sum(float(row["choice_entropy"]) for row in boundaries) / len(boundaries),
                "mean_kl_to_final": sum(float(row["kl_to_final"]) for row in boundaries) / len(boundaries),
                "mean_hidden_rms_l2_to_final": sum(float(row["hidden_rms_l2_to_final"]) for row in boundaries) / len(boundaries),
                "mean_hidden_cosine_to_final": sum(float(row["hidden_cosine_to_final"]) for row in boundaries) / len(boundaries),
                "mean_hidden_cosine_distance_to_final": sum(float(row["hidden_cosine_distance_to_final"]) for row in boundaries) / len(boundaries),
            }
        )
    for index in range(LAYERS):
        values = [row["adjacent_angular_distance"][index]["angle_over_pi"] for row in records]
        aggregate["transitionwise"].append(
            {
                "transition_id": "B_%d->B_%d" % (index, index + 1),
                "mean_angle_over_pi": sum(float(value) for value in values) / len(values),
            }
        )
    aggregate["aggregate_sha256"] = semantic_sha256(aggregate)
    aggregate_path = Path(run_root).resolve() / "selector/diagnostic_aggregate.json"
    aggregate_file_sha = _write_new_json(aggregate_path, aggregate)
    return {"status": "PASS", "path": str(aggregate_path), "file_sha256": aggregate_file_sha, "aggregate_sha256": aggregate["aggregate_sha256"]}


def dry_run() -> Dict[str, Any]:
    card, schema, card_path, schema_path = _load_card_schema()
    return {
        "status": "DRY_RUN_VALID",
        "gate": GATE,
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "model_revision": card["model"]["revision"],
        "dataset_revision": card["task"]["revision"],
        "validation_record_count": VALIDATION_RECORD_COUNT,
        "validation_subject_count": VALIDATION_SUBJECT_COUNT,
        "candidate_count": 42,
        "formal_bootstrap_replicates": FORMAL_REPLICATES,
        "formal_bootstrap_seed": FORMAL_SEED,
        "formal_loop_execution": False,
        "formal_outcome_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LoopScope Phase 6 Gate J MMLU 0-shot formal runner")
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("dry-run")
    freeze = actions.add_parser("freeze-manifest")
    freeze.add_argument("--source-root", type=Path, required=True)
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--expected-commit", required=True)
    freeze.add_argument("--shards", type=int, required=True)
    freeze.add_argument("--preflight", action="store_true")
    shard = actions.add_parser("shard")
    shard.add_argument("--run-root", type=Path, required=True)
    shard.add_argument("--expected-commit", required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--attempt", type=int, default=1)
    merge = actions.add_parser("merge")
    merge.add_argument("--run-root", type=Path, required=True)
    merge.add_argument("--expected-commit", required=True)
    verify = actions.add_parser("verify")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--job-id", required=True)
    verify.add_argument("--partition", required=True)
    launcher = actions.add_parser("write-launcher")
    launcher.add_argument("--run-root", type=Path, required=True)
    launcher.add_argument("--expected-commit", required=True)
    launcher.add_argument("--partition", required=True)
    launcher.add_argument("--gpu-type", required=True)
    launcher.add_argument("--qos")
    launcher.add_argument("--time-limit", default=FORMAL_DEFAULT_TIME_LIMIT)
    launcher.add_argument("--concurrency", type=int)
    select = actions.add_parser("freeze-selector")
    select.add_argument("--run-root", type=Path, required=True)
    select.add_argument("--expected-commit", required=True)
    verify_select = actions.add_parser("verify-selector")
    verify_select.add_argument("--run-root", type=Path, required=True)
    verify_select.add_argument("--expected-commit", required=True)
    aggregate = actions.add_parser("aggregate-diagnostics")
    aggregate.add_argument("--run-root", type=Path, required=True)
    aggregate.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        result = dry_run()
    elif args.command == "freeze-manifest":
        result = freeze_manifest(
            source_root=args.source_root,
            run_root=args.run_root,
            expected_commit=args.expected_commit,
            shard_count=args.shards,
            preflight=args.preflight,
        )
    elif args.command == "shard":
        result = acquire_shard(
            run_root=args.run_root,
            expected_commit=args.expected_commit,
            shard_index=args.shard_index,
            attempt=args.attempt,
        )
    elif args.command == "merge":
        result = merge_shards(run_root=args.run_root, expected_commit=args.expected_commit)
    elif args.command == "verify":
        result = verify_formal_run(
            run_root=args.run_root,
            expected_commit=args.expected_commit,
            job_id=args.job_id,
            expected_partition=args.partition,
        )
    elif args.command == "write-launcher":
        result = write_launcher(
            run_root=args.run_root,
            expected_commit=args.expected_commit,
            partition=args.partition,
            gpu_type=args.gpu_type,
            qos=args.qos,
            time_limit=args.time_limit,
            concurrency=args.concurrency,
        )
    elif args.command == "freeze-selector":
        result = freeze_selector(run_root=args.run_root, expected_commit=args.expected_commit)
    elif args.command == "verify-selector":
        result = verify_selector(run_root=args.run_root, expected_commit=args.expected_commit)
    else:
        result = aggregate_diagnostics(run_root=args.run_root, expected_commit=args.expected_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AUTHORIZED_BASE",
    "GateJMMLU0Error",
    "aggregate_diagnostics",
    "build_sbatch_text",
    "dry_run",
    "freeze_manifest",
    "freeze_selector",
    "main",
    "merge_shards",
    "verify_formal_run",
    "verify_selector",
    "write_launcher",
]
