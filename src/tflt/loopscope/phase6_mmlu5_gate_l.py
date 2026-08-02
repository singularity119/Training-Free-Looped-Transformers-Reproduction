"""LoopScope Phase 6 Gate L: MMLU five-shot CPU admission and GPU smoke.

Gate L revalidates the accepted Gate K outcome-free projection, materializes
the same four smoke identities used by the closed MMLU 0-shot Gate I, and
reconstructs only their fresh five-shot prompts in memory.  The GPU path runs
one native ``use_cache=False`` forward per identity and persists only the
closed scalar trajectory schema.  It never runs generation, a loop, the
selector, or a formal validation acquisition.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope import phase6_mmlu5_gate_k as gate_k
from tflt.loopscope.phase6_mmlu5_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    FORMAL_REPLICATES,
    FORMAL_SEED,
    HIDDEN_SIZE,
    LAYERS,
    MODEL_REVISION,
    PROMPT_PROJECTION_SCHEMA_VERSION,
    TASK_SOURCE_MANIFEST_SHA256,
    TRAJECTORY_SCHEMA_VERSION,
    VALIDATION_RECORD_COUNT,
    VALIDATION_SUBJECT_COUNT,
    build_trajectory_record,
    canonical_json_bytes,
    choice_logits_to_probabilities,
    file_sha256,
    freeze_choice_surface_manifest,
    load_json,
    semantic_sha256,
    validate_card,
    validate_choice_surface_manifest,
    validate_prompt_projection_record,
    validate_schema_document,
    validate_trajectory_record,
)


GATE = "L"
EXECUTOR_THREAD_ID = "019fc157-4d77-7390-b1c1-2b6d8dfcd7c4"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
AUTHORIZED_BASE = "3e0a4dbace1213c4af63472bf0ed18752474a77f"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/"
    "training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
MODEL_SNAPSHOT = (
    HF_HOME / "hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots" / MODEL_REVISION
)

CARD_RELATIVE = Path("configs/loopscope/phase6_mmlu5_prefix_card.json")
PROMPT_SCHEMA_RELATIVE = Path("configs/loopscope/phase6_mmlu5_prompt_projection_schema.json")
TRAJECTORY_SCHEMA_RELATIVE = Path("configs/loopscope/phase6_mmlu5_trajectory_schema.json")
CARD_SHA256 = "ea0c0c55d4b661872136e5204b58e363c646fea9c0f979fc3db88af9a919e265"
PROMPT_SCHEMA_SHA256 = "668620930e9d73109d4f4d2880bef72b6fa8e8737f517d4ef940ff35f46d19c7"
TRAJECTORY_SCHEMA_SHA256 = "6d64eafd296c8abb84c72d0d490fa2c90ea19fd878e889af89a07542fd7762ec"

GATE_K_ROOT_DEFAULT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/"
    "training_free_looped_transformers_loopscope/runs/"
    "phase6-gate-k-mmlu5-20260802T083000Z"
)
GATE_K_PROJECTION_SHA256 = "5ed42d8b873579aba60cf65d23dbd97d760c01a342e55ffde8c6accaee6c4b9f"
GATE_K_PROJECTION_RECEIPT_SHA256 = "681c809acd6ca321898fde5903730f6d5285d086e499a28102823cf091c70030"
GATE_K_VERIFIER_RECEIPT_SHA256 = "35b257b9155d59ffc27472c0aeb36389c15bd0ecacf95c399f62ab22c6738d23"

SMOKE_COUNT = 4
SMOKE_PARTITION = "debug"
SMOKE_TIME_LIMIT = "00:29:00"
SMOKE_CPUS = 8
SMOKE_MEMORY = "64G"
PREFERRED_SMOKE_IDENTITIES = (
    {
        "task": "mmlu",
        "doc_id": "mmlu_abstract_algebra:validation:0",
        "doc_hash": "3e758814027305e68aab8bf1e31ca874459ddd4abb84d1bc3d1a1c5149974e04",
    },
    {
        "task": "mmlu",
        "doc_id": "mmlu_anatomy:validation:0",
        "doc_hash": "7e8f0aa4fd1504cd33e42cd147f72bb805a20b457c18fafcd63f36f4c4d22fce",
    },
    {
        "task": "mmlu",
        "doc_id": "mmlu_astronomy:validation:0",
        "doc_hash": "7188638dca06b5f278f5a7baa3089f3fa8a6a5a9ede1bab660576f7c7b5a2012",
    },
    {
        "task": "mmlu",
        "doc_id": "mmlu_business_ethics:validation:0",
        "doc_hash": "e31d7e495e8c278a6c32e46ef5d155714553b2ad771ffcdfffa60872f206ffc4",
    },
)


class GateLMMLU5Error(RuntimeError):
    """Fail-closed Gate L contract or runtime error."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateLMMLU5Error(message)


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
                raise GateLMMLU5Error("invalid JSONL at %s:%d" % (path, line_number)) from exc
            _require(isinstance(value, dict), "JSONL rows must be objects")
            rows.append(value)
    return rows


def _assert_offline_environment(*, cpu: bool = False) -> None:
    expected = {
        "HF_HOME": str(HF_HOME),
        "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
        "HF_DATASETS_CACHE": str(HF_DATASETS_CACHE),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    for name, value in expected.items():
        _require(os.environ.get(name) == value, "offline environment differs: %s" % name)
    _require(not os.environ.get("HF_ENDPOINT"), "HF_ENDPOINT substitution is forbidden")
    if cpu:
        _require(
            os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "NoDevFiles"),
            "CPU admission must not expose CUDA devices",
        )


def _load_contract() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Path, Path, Path]:
    root = repository_root()
    card_path = root / CARD_RELATIVE
    prompt_schema_path = root / PROMPT_SCHEMA_RELATIVE
    trajectory_schema_path = root / TRAJECTORY_SCHEMA_RELATIVE
    _require(file_sha256(card_path) == CARD_SHA256, "Gate L card SHA-256 differs")
    _require(file_sha256(prompt_schema_path) == PROMPT_SCHEMA_SHA256, "Gate K prompt schema SHA-256 differs")
    _require(file_sha256(trajectory_schema_path) == TRAJECTORY_SCHEMA_SHA256, "Gate L trajectory schema SHA-256 differs")
    card = load_json(card_path)
    prompt_schema = load_json(prompt_schema_path)
    trajectory_schema = load_json(trajectory_schema_path)
    try:
        validate_card(card)
        validate_schema_document(prompt_schema)
        validate_schema_document(trajectory_schema)
    except ValueError as exc:
        raise GateLMMLU5Error(str(exc)) from exc
    return card, prompt_schema, trajectory_schema, card_path, prompt_schema_path, trajectory_schema_path


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = Path(_run(("git", "rev-parse", "--show-toplevel"))).resolve()
    _require(root == repository_root().resolve(), "not running in the dedicated LoopScope clone")
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"))
    commit = _run(("git", "rev-parse", "HEAD"))
    origin = _run(("git", "rev-parse", "origin/loopscope"))
    dirty = _run(("git", "status", "--porcelain"))
    _require(branch == "loopscope", "Gate L must run on loopscope")
    _require(commit == expected_commit, "Gate L expected commit differs")
    _require(origin == expected_commit, "remote origin/loopscope differs from expected commit")
    _require(dirty == "", "dedicated HPC2 clone must be clean")
    return {
        "repo": str(root),
        "branch": branch,
        "commit": commit,
        "origin_loopscope": origin,
        "dirty": False,
    }


def _require_authorized_descendant(expected_commit: str) -> None:
    _require(isinstance(expected_commit, str) and re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "expected commit is invalid")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", AUTHORIZED_BASE, expected_commit],
            cwd=str(repository_root()),
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GateLMMLU5Error("Gate L commit is not a descendant of the authorized base") from exc


def _package_versions() -> Dict[str, str]:
    try:
        values = gate_k._package_versions()
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateLMMLU5Error("Gate K package closure failed: %s" % exc) from exc
    return dict(values)


def _task_source_closure(card: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        value = gate_k._task_source_closure(card)
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateLMMLU5Error("Gate K task-source closure failed: %s" % exc) from exc
    _require(value.get("manifest_sha256") == TASK_SOURCE_MANIFEST_SHA256, "MMLU task source hash differs")
    _require(value.get("file_count") == 63, "MMLU task source file count differs")
    return dict(value)


def _model_cache_closure(card: Mapping[str, Any]) -> Tuple[Dict[str, Any], Any]:
    try:
        value, tokenizer = gate_k._model_tokenizer_closure(card)
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateLMMLU5Error("Gate K model/tokenizer closure failed: %s" % exc) from exc
    _require(value.get("weights_instantiated") is False, "CPU closure instantiated model weights")
    _require(value.get("weights_forwarded") is False, "CPU closure forwarded model weights")
    return dict(value), tokenizer


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
    manifest = gate_k.load_json(manifest_path)
    receipt = gate_k.load_json(projection_receipt_path)
    verifier = gate_k.load_json(verifier_path)
    _require(manifest.get("expected_commit") == AUTHORIZED_BASE, "Gate K expected commit differs")
    _require(manifest.get("projection", {}).get("file_sha256") == GATE_K_PROJECTION_SHA256, "Gate K manifest projection hash differs")
    _require(manifest.get("projection", {}).get("path") == "cpu/validation_5shot_projection.jsonl", "Gate K projection path differs")
    _require(receipt.get("status") == "PASS", "Gate K CPU receipt is not PASS")
    _require(verifier.get("status") == "PASS", "Gate K verifier is not PASS")
    for payload in (manifest, receipt, verifier):
        for key in ("selector_executed", "validation_target_gold_read", "test_split_read", "outcome_read", "model_weights_loaded", "model_forward_executed"):
            if key in payload:
                _require(payload[key] is False, "Gate K barrier differs: %s" % key)
    records = gate_k._load_jsonl(projection_path)
    _require(len(records) == VALIDATION_RECORD_COUNT, "Gate K projection record count differs")
    for record in records:
        try:
            gate_k._assert_no_forbidden_keys(record)
            validate_prompt_projection_record(record)
        except Exception as exc:
            raise GateLMMLU5Error("Gate K projection record is invalid") from exc
    _require([row["ordinal"] for row in records] == list(range(VALIDATION_RECORD_COUNT)), "Gate K ordinals are not canonical")
    _require(len({canonical_json_bytes(row["identity"]) for row in records}) == VALIDATION_RECORD_COUNT, "Gate K identities duplicate")
    _require(len({row["subject"] for row in records}) == VALIDATION_SUBJECT_COUNT, "Gate K subjects differ")
    _require(semantic_sha256([row["identity"] for row in records]) == manifest["population"]["ordered_identity_sha256"], "Gate K identity digest differs")
    _require(semantic_sha256(records) == manifest["population"]["ordered_prompt_projection_sha256"], "Gate K projection digest differs")
    cache_files = manifest.get("dataset_cache_files", [])
    _require(isinstance(cache_files, list) and bool(cache_files), "Gate K cache closure is empty")
    cache_hashes = []
    for item in cache_files:
        path = Path(str(item["path"]))
        _require(path.is_file(), "Gate K cache file is missing: %s" % path)
        digest = file_sha256(path)
        _require(digest == item["sha256"], "Gate K cache file hash differs: %s" % path)
        cache_hashes.append({"split": item.get("split"), "path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
    selected = []
    by_identity = {canonical_json_bytes(row["identity"]): row for row in records}
    for identity in PREFERRED_SMOKE_IDENTITIES:
        key = canonical_json_bytes(identity)
        _require(key in by_identity, "preferred Gate I smoke identity is absent from Gate K projection")
        selected.append(dict(by_identity[key]))
    _require(len(selected) == SMOKE_COUNT, "Gate L smoke membership count differs")
    _require(len({canonical_json_bytes(row["identity"]) for row in selected}) == SMOKE_COUNT, "Gate L smoke identities duplicate")
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "projection_path": str(projection_path),
        "projection_file_sha256": GATE_K_PROJECTION_SHA256,
        "projection_receipt_file_sha256": GATE_K_PROJECTION_RECEIPT_SHA256,
        "verifier_receipt_file_sha256": GATE_K_VERIFIER_RECEIPT_SHA256,
        "manifest": manifest,
        "receipt": receipt,
        "verifier": verifier,
        "records": records,
        "selected": selected,
        "cache_files": cache_hashes,
    }


def _membership_payload(*, expected_commit: str, gate_k_closure: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "loopscope.phase6.mmlu5-gate-l-smoke-membership.v1",
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "expected_commit": expected_commit,
        "gate_k_root": gate_k_closure["root"],
        "gate_k_projection_file_sha256": gate_k_closure["projection_file_sha256"],
        "record_count": SMOKE_COUNT,
        "subject_count": SMOKE_COUNT,
        "members": [dict(row) for row in gate_k_closure["selected"]],
        "partition": SMOKE_PARTITION,
        "time_limit": SMOKE_TIME_LIMIT,
        "cpus_per_task": SMOKE_CPUS,
        "memory": SMOKE_MEMORY,
        "selection_rule": "same four canonical identities as accepted Gate I; fresh Gate K five-shot prompts",
        "selector_executed": False,
        "test_split_read": False,
        "validation_target_gold_read": False,
        "outcome_read": False,
        "loop_executed": False,
    }


def _load_projection_and_membership(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    admission_path = Path(root) / "cpu/admission_receipt.json"
    membership_path = Path(root) / "cpu/smoke_membership.json"
    _require(admission_path.is_file() and membership_path.is_file(), "Gate L CPU admission artifacts are incomplete")
    admission = load_json(admission_path)
    membership = load_json(membership_path)
    membership_hash = membership.pop("manifest_sha256", None)
    _require(isinstance(membership_hash, str) and semantic_sha256(membership) == membership_hash, "Gate L membership hash differs")
    membership["manifest_sha256"] = membership_hash
    _require(file_sha256(membership_path) == admission.get("membership_file_sha256"), "Gate L membership file hash differs")
    _require(admission.get("status") == "PASS", "Gate L CPU admission is not PASS")
    admission_commit = admission.get("expected_commit")
    _require(admission.get("gate") == GATE, "Gate L admission gate differs")
    _require_authorized_descendant(admission_commit)
    _require(membership.get("expected_commit") == admission_commit, "Gate L membership commit differs")
    gate_k_closure = _gate_k_projection_closure(Path(admission["gate_k_root"]))
    _require(admission.get("gate_k_projection_file_sha256") == gate_k_closure["projection_file_sha256"], "Gate K projection reference differs")
    expected_members = [dict(row) for row in gate_k_closure["selected"]]
    members = membership.get("members")
    _require(isinstance(members, list) and len(members) == SMOKE_COUNT, "Gate L membership count differs")
    _require(members == expected_members, "Gate L membership differs from Gate K projection")
    _require(membership.get("partition") == SMOKE_PARTITION and membership.get("time_limit") == SMOKE_TIME_LIMIT, "Gate L resource contract differs")
    _require(membership.get("cpus_per_task") == SMOKE_CPUS and membership.get("memory") == SMOKE_MEMORY, "Gate L resource size differs")
    _require(membership.get("selector_executed") is False and membership.get("outcome_read") is False and membership.get("loop_executed") is False, "Gate L membership barrier differs")
    return gate_k_closure["records"], members, admission, gate_k_closure


def cpu_admission(*, run_root: Path, expected_commit: str, gate_k_root: Path = GATE_K_ROOT_DEFAULT) -> Dict[str, Any]:
    _assert_offline_environment(cpu=True)
    _require_authorized_descendant(expected_commit)
    root = Path(run_root).resolve()
    _require(not root.exists(), "Gate L run root must be fresh")
    card, _prompt_schema, _trajectory_schema, card_path, prompt_schema_path, trajectory_schema_path = _load_contract()
    git = _git_provenance(expected_commit)
    packages = _package_versions()
    source = _task_source_closure(card)
    model, _tokenizer = _model_cache_closure(card)
    gate_k_closure = _gate_k_projection_closure(Path(gate_k_root))
    membership = _membership_payload(expected_commit=expected_commit, gate_k_closure=gate_k_closure)
    membership_semantic_sha = semantic_sha256(membership)
    membership["manifest_sha256"] = membership_semantic_sha
    root.mkdir(parents=True, exist_ok=False)
    membership_file_sha = _write_new_json(root / "cpu/smoke_membership.json", membership)
    cache_digest = semantic_sha256(gate_k_closure["cache_files"])
    admission = {
        "schema_version": "loopscope.phase6.mmlu5-gate-l-cpu-admission.v1",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(root),
        "expected_commit": expected_commit,
        "git": git,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "gate_k_root": gate_k_closure["root"],
        "gate_k_projection_file_sha256": gate_k_closure["projection_file_sha256"],
        "gate_k_projection_receipt_file_sha256": gate_k_closure["projection_receipt_file_sha256"],
        "gate_k_verifier_receipt_file_sha256": gate_k_closure["verifier_receipt_file_sha256"],
        "gate_k_record_count": len(gate_k_closure["records"]),
        "gate_k_subject_count": len({row["subject"] for row in gate_k_closure["records"]}),
        "gate_k_cache_files_semantic_sha256": cache_digest,
        "gate_k_cache_file_count": len(gate_k_closure["cache_files"]),
        "task_source_manifest_sha256": source["manifest_sha256"],
        "task_source_file_count": source["file_count"],
        "packages": packages,
        "model": model,
        "membership_file_sha256": membership_file_sha,
        "membership_semantic_sha256": membership_semantic_sha,
        "smoke_count": SMOKE_COUNT,
        "partition": SMOKE_PARTITION,
        "time_limit": SMOKE_TIME_LIMIT,
        "cpus_per_task": SMOKE_CPUS,
        "memory": SMOKE_MEMORY,
        "native_forward_per_record": 1,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "cuda_gpu_slurm": False,
        "generation_calls": 0,
        "loop_insertions": 0,
        "selector_executed": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "outcome_read": False,
        "raw_prompt_persisted": False,
        "token_ids_persisted": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(root / "cpu/admission_receipt.json", admission)
    return {**admission, "receipt_sha256": receipt_sha}


def _active_loop_wrapper_modules(model: Any) -> List[str]:
    active: List[str] = []
    for name, module in model.named_modules():
        module_name = str(getattr(module.__class__, "__module__", ""))
        if module_name == "tflt.wrapper" or module_name.startswith("tflt.wrapper."):
            active.append(str(name or "<root>"))
    forward_module = str(getattr(getattr(model, "forward", None), "__module__", ""))
    if forward_module == "tflt.wrapper" or forward_module.startswith("tflt.wrapper."):
        active.append("<model.forward>")
    return sorted(set(active))


def _find_final_norm(model: Any) -> Any:
    candidates = (
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(model, "base_model", None), "norm", None),
        getattr(model, "norm", None),
    )
    for candidate in candidates:
        if callable(candidate):
            return candidate
    raise GateLMMLU5Error("Qwen3 final norm module is missing")


def _load_native_runtime(card: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any, Any, List[int]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateLMMLU5Error("GPU runtime imports failed") from exc
    _require(bool(torch.cuda.is_available()), "Gate L requires a visible CUDA GPU")
    try:
        model_meta, tokenizer = _model_cache_closure(card)
    except GateLMMLU5Error:
        raise
    choice_manifest = freeze_choice_surface_manifest(tokenizer)
    validate_choice_surface_manifest(choice_manifest)
    _require(model_meta["choice_surface_manifest_sha256"] == CHOICE_SURFACE_MANIFEST_SHA256, "choice surface manifest hash differs")
    choice_ids = [int(choice_manifest[key]["token_ids"][0]) for key in ("A", "B", "C", "D")]
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_SNAPSHOT),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    _require(int(model.config.num_hidden_layers) == LAYERS, "GPU runtime layer count differs")
    _require(int(model.config.hidden_size) == HIDDEN_SIZE, "GPU runtime hidden size differs")
    _require(str(getattr(model.config, "torch_dtype", "")) in {"torch.bfloat16", "bfloat16"}, "GPU runtime dtype differs")
    _require(not _active_loop_wrapper_modules(model), "native model has active loop wrapper modules")
    _require(hasattr(model, "get_output_embeddings"), "GPU runtime lacks output embeddings")
    lm_head = model.get_output_embeddings()
    _require(lm_head is not None, "GPU runtime output head is missing")
    final_norm = _find_final_norm(model)
    _require("logits_to_keep" in inspect.signature(model.forward).parameters, "GPU runtime lacks logits_to_keep")
    model.eval().to("cuda")
    _require(not _active_loop_wrapper_modules(model), "native model became wrapped after placement")
    return torch, tokenizer, model, final_norm, lm_head, choice_ids


def _capture_raw_final_boundary(final_norm: Any, forward: Any, torch_module: Any) -> Tuple[Any, Any, int, float]:
    captured: List[Any] = []

    def prehook(_module: Any, args: Tuple[Any, ...]) -> None:
        _require(bool(args), "FinalNorm pre-hook did not receive input")
        captured.append(args[0])

    handle = final_norm.register_forward_pre_hook(prehook)
    try:
        outputs = forward()
    finally:
        handle.remove()
    _require(len(captured) == 1, "FinalNorm pre-hook call count differs")
    raw_final = captured[0]
    with torch_module.inference_mode():
        recomputed = final_norm(raw_final)
    post = tuple(outputs.hidden_states or ())[-1]
    _require(tuple(recomputed.shape) == tuple(post.shape), "FinalNorm closure shape differs")
    max_abs = float((recomputed - post).detach().abs().max().double().cpu())
    _require(bool(torch_module.allclose(recomputed, post, rtol=0.0, atol=0.0)), "FinalNorm closure is not exact")
    return outputs, raw_final, len(captured), max_abs


def _max_abs(left: Any, right: Any, torch_module: Any) -> float:
    return float((left.float() - right.float()).detach().abs().max().double().cpu())


def _trajectory_from_forward(
    *,
    prompt: str,
    projection_row: Mapping[str, Any],
    tokenizer: Any,
    torch_module: Any,
    model: Any,
    final_norm: Any,
    lm_head: Any,
    choice_ids: Sequence[int],
    card: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    _require(_sha256_bytes(prompt.encode("utf-8")) == projection_row["prompt_sha256"], "prompt hash differs from Gate K projection")
    _require(_sha256_bytes(canonical_json_bytes(token_ids)) == projection_row["prompt_tokenization_sha256"], "prompt tokenization hash differs from Gate K projection")
    _require(len(token_ids) == int(projection_row["sequence_length"]), "prompt sequence length differs from Gate K projection")
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    _require("input_ids" in encoded and "attention_mask" in encoded, "tokenizer output lacks ids/mask")
    valid = encoded["attention_mask"][0].detach().cpu().nonzero(as_tuple=False).flatten().tolist()
    _require(valid and int(valid[-1]) == int(encoded["attention_mask"].shape[1]) - 1, "probe position is not the last non-padding token")
    position = int(valid[-1])
    _require(position == int(projection_row["probe_position"]), "probe position differs from Gate K projection")
    inputs = {key: value.to("cuda") for key, value in encoded.items()}
    start = time.monotonic()
    with torch_module.inference_mode():
        outputs, raw_final, hook_calls, closure = _capture_raw_final_boundary(
            final_norm,
            lambda: model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
            ),
            torch_module,
        )
        hidden_states = tuple(outputs.hidden_states or ())
        _require(len(hidden_states) == BOUNDARY_COUNT, "hidden-state boundary count differs")
        raw_boundaries = list(hidden_states[:-1]) + [raw_final]
        raw_vectors = [state[0, position, :].detach() for state in raw_boundaries]
        normalized_vectors = [
            final_norm(value.unsqueeze(0)).squeeze(0) for value in raw_vectors[:-1]
        ] + [hidden_states[-1][0, position, :].detach()]
        native_final_logits = outputs.logits.reshape(-1, outputs.logits.shape[-1])[0]
        projected_final_logits = lm_head(normalized_vectors[-1].unsqueeze(0)).reshape(-1)
        _require(tuple(native_final_logits.shape) == tuple(projected_final_logits.shape), "native/reprojected vocabulary shape differs")
        native_reprojected_max_abs = _max_abs(native_final_logits, projected_final_logits, torch_module)
        _require(bool(torch_module.allclose(native_final_logits.float(), projected_final_logits.float(), rtol=1e-3, atol=1e-3)), "native/reprojected final logits do not close")
        choice_logits = []
        for vector in normalized_vectors[:-1]:
            values = lm_head(vector.unsqueeze(0)).reshape(-1)
            choice_logits.append(values[list(choice_ids)])
        choice_logits.append(native_final_logits[list(choice_ids)])
        probabilities = []
        for values in choice_logits:
            cpu_values = values.detach().to(dtype=torch_module.float64).cpu().tolist()
            probabilities.append(choice_logits_to_probabilities(cpu_values))
        normalized_cpu = [value.detach().to(dtype=torch_module.float32).cpu().tolist() for value in normalized_vectors]
        raw_cpu = [value.detach().to(dtype=torch_module.float32).cpu().tolist() for value in raw_vectors]
        record = build_trajectory_record(
            projection_row["identity"],
            subject=str(projection_row["subject"]),
            prompt_sha256=str(projection_row["prompt_sha256"]),
            renderer_provenance={
                "contract_version": card["renderer"]["contract_version"],
                "semantic_contract_sha256": card["renderer"]["semantic_contract_sha256"],
                "task_source_manifest_sha256": card["task"]["lm_eval_task_source"]["default_source_manifest_sha256"],
                "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
            },
            choice_probabilities=probabilities,
            hidden_states=normalized_cpu,
            card=card,
            angular_hidden_states=raw_cpu,
        )
    _require(not _active_loop_wrapper_modules(model), "native model wrapper state changed")
    evidence = {
        "identity": dict(projection_row["identity"]),
        "subject": str(projection_row["subject"]),
        "prompt_sha256": str(projection_row["prompt_sha256"]),
        "prompt_tokenization_sha256": str(projection_row["prompt_tokenization_sha256"]),
        "sequence_length": int(encoded["attention_mask"].shape[1]),
        "probe_position": position,
        "final_norm_prehook_calls": hook_calls,
        "final_norm_boundary_calls": BOUNDARY_COUNT - 1,
        "final_norm_closure_max_abs": closure,
        "native_reprojected_vocab_max_abs": native_reprojected_max_abs,
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "forward_count": 1,
        "loop_insertions": 0,
        "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
        "elapsed_seconds": time.monotonic() - start,
        "full_vocab_persisted": False,
        "hidden_tensors_persisted": False,
        "raw_text_persisted": False,
        "token_ids_persisted": False,
    }
    del outputs, raw_final, hidden_states, raw_boundaries, raw_vectors, normalized_vectors
    del native_final_logits, projected_final_logits, choice_logits, probabilities
    del normalized_cpu, raw_cpu, inputs, encoded, token_ids
    return record, evidence


def _safe_dataset_prompts(task_by_subject: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], tokenizer: Any) -> Dict[str, str]:
    try:
        import numpy as np
        from datasets import DatasetDict
        from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateLMMLU5Error("Gate L renderer dependencies failed") from exc
    wanted = {canonical_json_bytes(row["identity"]): row for row in selected}
    selected_subjects = [str(row["subject"]) for row in selected]
    _require(len(selected_subjects) == SMOKE_COUNT and len(set(selected_subjects)) == SMOKE_COUNT, "smoke subject membership differs")
    backend = LmEvalMMLURendererBackend()
    prompts: Dict[str, str] = {}
    for subject in selected_subjects:
        task_name = "mmlu_%s" % subject
        _require(task_name in task_by_subject, "smoke subject has no standard MMLU task")
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
            if key not in wanted:
                continue
            prompt, demos = backend._render_with_captured_demos(
                task,
                gate_k._renderer_target_doc(safe),
                dev,
                np,
                int(FORMAL_SEED),
            )
            _require(isinstance(prompt, str) and prompt.rstrip().endswith("Answer:"), "standard five-shot prompt terminal differs")
            demo_rows = gate_k._demo_provenance(task, demos, dev, subject)
            _require([row["id"] for row in demo_rows] == wanted[key]["demonstration_ids"], "fresh demonstration IDs differ from Gate K projection")
            _require(demo_rows == wanted[key]["demonstration_provenance"], "fresh demonstration provenance differs from Gate K projection")
            token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            _require(token_ids, "smoke prompt tokenizes empty")
            _require(_sha256_bytes(prompt.encode("utf-8")) == wanted[key]["prompt_sha256"], "fresh five-shot prompt hash differs")
            _require(_sha256_bytes(canonical_json_bytes(token_ids)) == wanted[key]["prompt_tokenization_sha256"], "fresh five-shot tokenization hash differs")
            _require(len(token_ids) == int(wanted[key]["sequence_length"]), "fresh five-shot sequence length differs")
            _require(len(token_ids) - 1 == int(wanted[key]["probe_position"]), "fresh five-shot probe position differs")
            prompts[key.decode("utf-8")] = prompt
    _require(len(prompts) == len(selected), "smoke prompt reconstruction is incomplete")
    return prompts


def acquire_smoke(*, run_root: Path, expected_commit: str, attempt: int = 1) -> Dict[str, Any]:
    _assert_offline_environment()
    root = Path(run_root).resolve()
    rows, selected, admission, gate_k_closure = _load_projection_and_membership(root)
    git = _git_provenance(expected_commit)
    card, _prompt_schema, _trajectory_schema, card_path, _prompt_schema_path, _trajectory_schema_path = _load_contract()
    source = _task_source_closure(card)
    _require(attempt >= 1, "smoke attempt must be positive")
    output = root / "smoke" / ("attempt-%04d" % int(attempt))
    _require(not output.exists(), "smoke attempt root must be fresh")
    output.mkdir(parents=True, exist_ok=False)
    attempt_start = {
        "schema_version": "loopscope.phase6.mmlu5-gate-l-attempt-start.v1",
        "gate": GATE,
        "run_root": str(root),
        "attempt": int(attempt),
        "git": git,
        "partition": SMOKE_PARTITION,
        "expected_commit": expected_commit,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "generation_calls": 0,
        "loop_insertions": 0,
        "selector_executed": False,
        "outcome_read": False,
    }
    attempt_start_sha = _write_new_json(output / "attempt_start.json", attempt_start)
    torch_module, tokenizer, model, final_norm, lm_head, choice_ids = _load_native_runtime(card)
    task_manager, task_by_subject = gate_k._task_map()
    del task_manager
    prompts = _safe_dataset_prompts(task_by_subject, selected, tokenizer)
    by_key = {canonical_json_bytes(row["identity"]): row for row in rows}
    records: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for member in selected:
        key = canonical_json_bytes(member["identity"])
        _require(key in by_key, "smoke member is absent from Gate K projection")
        record, fact = _trajectory_from_forward(
            prompt=prompts[key.decode("utf-8")],
            projection_row=by_key[key],
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
    _require(len(records) == SMOKE_COUNT and len(evidence) == SMOKE_COUNT, "four-record smoke closure differs")
    records_path = output / "trajectory_records.jsonl"
    evidence_path = output / "runtime_evidence.jsonl"
    records_sha = _write_new_jsonl(records_path, records)
    evidence_sha = _write_new_jsonl(evidence_path, evidence)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu5-gate-l-gpu-smoke.v1",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(root),
        "attempt": int(attempt),
        "git": git,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": admission["prompt_projection_schema_sha256"],
        "trajectory_schema_sha256": admission["trajectory_schema_sha256"],
        "task_source_manifest_sha256": source["manifest_sha256"],
        "gate_k_projection_file_sha256": gate_k_closure["projection_file_sha256"],
        "membership_sha256": admission["membership_semantic_sha256"],
        "attempt_start_file_sha256": attempt_start_sha,
        "record_file_sha256": records_sha,
        "evidence_file_sha256": evidence_sha,
        "record_count": len(records),
        "identity_order_sha256": semantic_sha256([row["identity"] for row in records]),
        "forward_count": sum(int(row["forward_count"]) for row in evidence),
        "loop_insertions": sum(int(row["loop_insertions"]) for row in evidence),
        "generation_calls": 0,
        "selector_executed": False,
        "outcome_read": False,
        "test_split_read": False,
        "validation_target_gold_read": False,
        "full_vocab_persisted": False,
        "hidden_tensors_persisted": False,
        "raw_text_persisted": False,
        "token_ids_persisted": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(output / "acquisition_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha, "output": str(output)}


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


def query_sacct(job_id: str) -> Tuple[List[str], List[Dict[str, str]]]:
    _require(re.fullmatch(r"[0-9]+", str(job_id)) is not None, "Slurm job ID is invalid")
    sacct = shutil.which("sacct") or "/opt/slurm/bin/sacct"
    command = [sacct, "-X", "-n", "-P", "-j", str(job_id), "--format=" + ",".join(SACCT_FIELDS)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|")
        if len(fields) != len(SACCT_FIELDS):
            continue
        rows.append(dict(zip(SACCT_FIELDS, fields)))
    _require(rows, "sacct returned no rows")
    return command, rows


def _validate_scheduler(rows: Sequence[Mapping[str, str]], job_id: str) -> Dict[str, Any]:
    _require(len(rows) == 1, "Gate L smoke must have exactly one scheduler row")
    row = dict(rows[0])
    _require(row["JobIDRaw"] == str(job_id), "scheduler job identity differs")
    _require(row["Partition"] == SMOKE_PARTITION, "Gate L smoke did not use debug partition")
    _require(row["State"].startswith("COMPLETED"), "Gate L smoke scheduler state is not COMPLETED")
    _require(row["ExitCode"] == "0:0", "Gate L smoke scheduler ExitCode is not 0:0")
    return row


def formal_runtime_dry_plan() -> Dict[str, Any]:
    """Import-only Gate M admission plan; it does not enter Gate M."""
    return {
        "status": "GATE_M_FORMAL_RUNTIME_DRY_PLAN_IMPORTABLE",
        "gate": "M",
        "formal_gpu": "A800",
        "formal_partition_policy": "highest_legal_normal_user_priority_A800",
        "validation_record_count": VALIDATION_RECORD_COUNT,
        "validation_subject_count": VALIDATION_SUBJECT_COUNT,
        "debug_preflight_required": True,
        "selector_deferred": True,
        "formal_acquisition_executed": False,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "cuda_gpu_slurm": False,
        "outcome_read": False,
    }


def verify_smoke(*, run_root: Path, expected_commit: str, job_id: str, attempt: int = 1) -> Dict[str, Any]:
    _assert_offline_environment()
    root = Path(run_root).resolve()
    card, _prompt_schema, _trajectory_schema, card_path, prompt_schema_path, trajectory_schema_path = _load_contract()
    rows, members, admission, gate_k_closure = _load_projection_and_membership(root)
    _require(attempt >= 1, "smoke attempt must be positive")
    output = root / "smoke" / ("attempt-%04d" % int(attempt))
    acquisition = load_json(output / "acquisition_receipt.json")
    records_path = output / "trajectory_records.jsonl"
    evidence_path = output / "runtime_evidence.jsonl"
    _require(file_sha256(records_path) == acquisition["record_file_sha256"], "trajectory file hash differs")
    _require(file_sha256(evidence_path) == acquisition["evidence_file_sha256"], "evidence file hash differs")
    records = _load_jsonl(records_path)
    evidence = _load_jsonl(evidence_path)
    _require(len(records) == SMOKE_COUNT and len(evidence) == SMOKE_COUNT, "four-record smoke closure differs")
    _require(acquisition["git"]["commit"] == expected_commit and acquisition["git"]["branch"] == "loopscope" and acquisition["git"]["dirty"] is False, "acquisition Git provenance differs")
    _require(acquisition["attempt"] == int(attempt), "acquisition attempt differs")
    _require(acquisition["membership_sha256"] == admission["membership_semantic_sha256"], "acquisition membership hash differs")
    _require(acquisition["gate_k_projection_file_sha256"] == gate_k_closure["projection_file_sha256"], "acquisition Gate K projection hash differs")
    _require(acquisition["forward_count"] == SMOKE_COUNT, "forward count differs")
    _require(acquisition["loop_insertions"] == 0, "loop insertion count differs")
    _require(acquisition["generation_calls"] == 0, "generation calls are nonzero")
    _require(acquisition["selector_executed"] is False and acquisition["outcome_read"] is False, "Gate L information barrier differs")
    expected_members = {canonical_json_bytes(row["identity"]): row for row in members}
    seen = set()
    expected_evidence_keys = {
        "identity", "subject", "prompt_sha256", "prompt_tokenization_sha256", "sequence_length", "probe_position",
        "final_norm_prehook_calls", "final_norm_boundary_calls", "final_norm_closure_max_abs", "native_reprojected_vocab_max_abs",
        "boundary_count", "transition_count", "forward_count", "loop_insertions", "choice_surface_manifest_sha256",
        "elapsed_seconds", "full_vocab_persisted", "hidden_tensors_persisted", "raw_text_persisted", "token_ids_persisted",
    }
    for record, fact in zip(records, evidence):
        validate_trajectory_record(record, card)
        key = canonical_json_bytes(record["identity"])
        _require(key in expected_members and key not in seen, "trajectory identity membership/duplication differs")
        seen.add(key)
        _require(set(fact) == expected_evidence_keys, "runtime evidence fields are not closed")
        _require(fact["identity"] == record["identity"] and fact["subject"] == record["subject"], "runtime evidence identity differs")
        _require(fact["prompt_sha256"] == record["prompt_sha256"], "runtime evidence prompt hash differs")
        _require(fact["sequence_length"] == expected_members[key]["sequence_length"], "runtime sequence length differs")
        _require(fact["probe_position"] == expected_members[key]["probe_position"], "runtime probe position differs")
        _require(fact["final_norm_prehook_calls"] == 1, "FinalNorm pre-hook count differs")
        _require(fact["final_norm_boundary_calls"] == BOUNDARY_COUNT - 1, "FinalNorm boundary call count differs")
        _require(fact["final_norm_closure_max_abs"] == 0.0, "FinalNorm closure is nonzero")
        _require(fact["boundary_count"] == BOUNDARY_COUNT and fact["transition_count"] == LAYERS, "trajectory dimensions differ")
        _require(fact["forward_count"] == 1 and fact["loop_insertions"] == 0, "per-record execution closure differs")
        _require(fact["full_vocab_persisted"] is False and fact["hidden_tensors_persisted"] is False, "forbidden tensors persisted")
        _require(fact["raw_text_persisted"] is False and fact["token_ids_persisted"] is False, "raw prompt payload persisted")
        _require(math.isfinite(float(fact["native_reprojected_vocab_max_abs"])) and math.isfinite(float(fact["elapsed_seconds"])), "runtime scalar is non-finite")
    _require(seen == set(expected_members), "smoke membership is incomplete")
    scheduler_command, scheduler_rows = query_sacct(job_id)
    scheduler = _validate_scheduler(scheduler_rows, job_id)
    dry_plan = formal_runtime_dry_plan()
    verifier = {
        "schema_version": "loopscope.phase6.mmlu5-gate-l-verifier.v1",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "cpu_admission_receipt_sha256": file_sha256(root / "cpu/admission_receipt.json"),
        "smoke_acquisition_receipt_sha256": file_sha256(output / "acquisition_receipt.json"),
        "scheduler": {"job_id": str(job_id), "query": scheduler_command, "row": scheduler},
        "identity_count": len(records),
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "forward_count": SMOKE_COUNT,
        "loop_insertions": 0,
        "generation_calls": 0,
        "selector_executed": False,
        "test_split_read": False,
        "validation_target_gold_read": False,
        "outcome_read": False,
        "forbidden_persisted_payloads": False,
        "gate_m_formal_runtime_dry_plan": dry_plan,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(root / "verifier_receipt.json", verifier)
    return {**verifier, "receipt_sha256": receipt_sha}


def build_sbatch_text(*, run_root: Path, expected_commit: str, attempt: int = 1) -> str:
    _require(attempt >= 1, "smoke attempt must be positive")
    runner = repository_root() / "scripts/loopscope/run_qwen4_phase6_mmlu5_gate_l.py"
    lines = [
        "#!/usr/bin/env bash",
        "# Gate L only: four-identity five-shot debug smoke; Gate M remains locked.",
        "#SBATCH --job-name=loopscope-p6-l-mmlu5-smoke",
        "#SBATCH --partition=debug",
        "#SBATCH --time=00:29:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --output=%s/slurm/smoke-%%j.out" % run_root,
        "#SBATCH --error=%s/slurm/smoke-%%j.err" % run_root,
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
        "exec python %s smoke --run-root %s --expected-commit %s --attempt %d"
        % (
            shlex.quote(str(runner)),
            shlex.quote(str(run_root)),
            shlex.quote(str(expected_commit)),
            int(attempt),
        ),
        "",
    ]
    text = "\n".join(lines)
    _require("#SBATCH --partition=debug" in text, "smoke launcher partition drifted")
    _require("#SBATCH --time=00:29:00" in text, "smoke launcher time limit drifted")
    _require("#SBATCH --cpus-per-task=8" in text and "#SBATCH --mem=64G" in text, "smoke launcher resources drifted")
    _require("#SBATCH --gres=gpu:1" in text, "smoke launcher GPU request drifted")
    _require("selector" not in text.lower() and "outcome" not in text.lower(), "smoke launcher contains forbidden follow-up action")
    return text


def write_launcher(*, run_root: Path, expected_commit: str, attempt: int = 1) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    _require(root.is_dir(), "Gate L run root is absent")
    path = root / "slurm/gate_l_smoke.sbatch"
    body = build_sbatch_text(run_root=root, expected_commit=expected_commit, attempt=attempt).encode("utf-8")
    sha = _write_new_bytes(path, body)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu5-gate-l-launcher.v1",
        "gate": GATE,
        "run_root": str(root),
        "expected_commit": expected_commit,
        "path": str(path.relative_to(root)),
        "sha256": sha,
        "partition": SMOKE_PARTITION,
        "time_limit": SMOKE_TIME_LIMIT,
        "cpus_per_task": SMOKE_CPUS,
        "memory": SMOKE_MEMORY,
        "gpu_count": 1,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
    }
    receipt_sha = _write_new_json(root / "slurm/launcher_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def import_only_dry_plan(card_path: Path, trajectory_schema_path: Path) -> Dict[str, Any]:
    card = load_json(card_path)
    schema = load_json(trajectory_schema_path)
    validate_card(card)
    validate_schema_document(schema)
    return {
        "status": "IMPORT_ONLY_DRY_PLAN",
        "gate": GATE,
        "card_schema_version": card["schema_version"],
        "trajectory_schema_version": schema["$id"],
        "card_sha256": file_sha256(card_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "partition": SMOKE_PARTITION,
        "records": SMOKE_COUNT,
        "native_forward_per_record": 1,
        "model_weights_loaded": False,
        "forward_executed": False,
        "generation_executed": False,
        "loop_executed": False,
        "selector_executed": False,
        "outcome_read": False,
    }


def dry_run() -> Dict[str, Any]:
    card, _prompt_schema, trajectory_schema, card_path, prompt_schema_path, trajectory_schema_path = _load_contract()
    return {
        "status": "DRY_RUN_VALID",
        "gate": GATE,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(prompt_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "model_revision": card["model"]["revision"],
        "dataset_revision": card["task"]["revision"],
        "validation_record_count": card["task"]["validation_record_count"],
        "validation_subject_count": card["task"]["validation_subject_count"],
        "smoke_count": SMOKE_COUNT,
        "partition": SMOKE_PARTITION,
        "time_limit": SMOKE_TIME_LIMIT,
        "formal_bootstrap_replicates": FORMAL_REPLICATES,
        "formal_bootstrap_seed": FORMAL_SEED,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
        "formal_acquisition": False,
        "gate_m_formal_runtime_dry_plan": formal_runtime_dry_plan(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LoopScope Phase 6 Gate L MMLU five-shot runner")
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("dry-run")
    cpu = actions.add_parser("cpu-admission")
    cpu.add_argument("--run-root", type=Path, required=True)
    cpu.add_argument("--gate-k-root", type=Path, default=GATE_K_ROOT_DEFAULT)
    cpu.add_argument("--expected-commit", required=True)
    smoke = actions.add_parser("smoke")
    smoke.add_argument("--run-root", type=Path, required=True)
    smoke.add_argument("--expected-commit", required=True)
    smoke.add_argument("--attempt", type=int, default=1)
    verify = actions.add_parser("verify")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--job-id", required=True)
    verify.add_argument("--attempt", type=int, default=1)
    launch = actions.add_parser("write-launcher")
    launch.add_argument("--run-root", type=Path, required=True)
    launch.add_argument("--expected-commit", required=True)
    launch.add_argument("--attempt", type=int, default=1)
    actions.add_parser("gate-m-dry-plan")
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        result = dry_run()
    elif args.command == "cpu-admission":
        result = cpu_admission(run_root=args.run_root, expected_commit=args.expected_commit, gate_k_root=args.gate_k_root)
    elif args.command == "smoke":
        result = acquire_smoke(run_root=args.run_root, expected_commit=args.expected_commit, attempt=args.attempt)
    elif args.command == "verify":
        result = verify_smoke(run_root=args.run_root, expected_commit=args.expected_commit, job_id=args.job_id, attempt=args.attempt)
    elif args.command == "write-launcher":
        result = write_launcher(run_root=args.run_root, expected_commit=args.expected_commit, attempt=args.attempt)
    else:
        result = formal_runtime_dry_plan()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AUTHORIZED_BASE",
    "CARD_SHA256",
    "GATE",
    "GateLMMLU5Error",
    "GATE_K_ROOT_DEFAULT",
    "PREFERRED_SMOKE_IDENTITIES",
    "SMOKE_COUNT",
    "SMOKE_PARTITION",
    "SMOKE_TIME_LIMIT",
    "build_sbatch_text",
    "cpu_admission",
    "dry_run",
    "formal_runtime_dry_plan",
    "import_only_dry_plan",
    "main",
    "verify_smoke",
    "write_launcher",
]
