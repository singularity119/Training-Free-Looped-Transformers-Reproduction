"""Phase 6 Gate I: MMLU 0-shot CPU admission and four-identity GPU smoke.

The CPU stage closes the cached model/tokenizer/task/runtime provenance and
creates a fresh, gold-free zero-shot validation projection.  The GPU stage
rebuilds only the four frozen prompts, runs one native ``use_cache=False``
forward per identity, and persists the closed scalar trajectory contract.
Gate I never runs the V2 selector, a loop, or a formal validation acquisition.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase6_mmlu0_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    HIDDEN_SIZE,
    LAYERS,
    build_trajectory_record,
    canonical_json_bytes,
    choice_logits_to_probabilities,
    file_sha256,
    freeze_choice_surface_manifest,
    load_json,
    semantic_sha256,
    validate_card,
    validate_schema_document,
    validate_trajectory_record,
)


GATE = "I"
EXECUTOR_THREAD_ID = "019fbeff-d331-78d2-b1aa-9741937b93fb"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
AUTHORIZED_BASE = "0fda7bb191b73343d7555e44dbcd3dd67d24cb36"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/"
    "training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_REPO = "cais/mmlu"
MODEL_SNAPSHOT = (
    HF_HOME / "hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots" / MODEL_REVISION
)
CARD_RELATIVE = Path("configs/loopscope/phase6_mmlu0_prefix_card.json")
SCHEMA_RELATIVE = Path("configs/loopscope/phase6_mmlu0_trajectory_schema.json")
CARD_SHA256 = "a415307c6e33a5c90b2fc2072c1926af75bc6d00e2b36938c11b8dad263ff43a"
SCHEMA_SHA256 = "1d6d427f9bc1f34d8af19043661f2307989e72998c461df678feeade75c4da26"
EXPECTED_CONFIG_SHA256 = "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"
EXPECTED_TOKENIZER_CONFIG_SHA256 = (
    "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
)
EXPECTED_TOKENIZER_JSON_SHA256 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)
EXPECTED_TASK_SOURCE_MANIFEST_SHA256 = (
    "7e03a4bac9704839de842c099e068fb2ebba626a066adfc3499c69eb05336b68"
)
EXPECTED_TASK_SOURCE_FILE_COUNT = 63
EXPECTED_PACKAGE_VERSIONS = {
    "lm_eval": "0.4.11",
    "transformers": "4.51.3",
    "tokenizers": "0.21.4",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
}
EXPECTED_TORCH_VERSION = "2.3.1+cu121"
EXPECTED_VALIDATION_COUNT = 1531
EXPECTED_SUBJECT_COUNT = 57
EXPECTED_VALIDATION_IDENTITY_SHA256 = (
    "ced8dd2ac8277a30e4d4e101f8f76c2dd125659abe35af679f60e5c6a9f076eb"
)
SMOKE_COUNT = 4
SMOKE_PARTITION = "debug"
SMOKE_TIME_LIMIT = "00:29:00"
SMOKE_CPUS = 8
SMOKE_MEMORY = "64G"
CHOICE_KEYS = ("A", "B", "C", "D")
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
    "generation",
)


class GateIMMLU0Error(RuntimeError):
    """Fail-closed Gate I contract or execution error."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateIMMLU0Error(message)


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
    _require(body, "JSONL artifact must not be empty")
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
                raise GateIMMLU0Error(
                    "invalid JSONL at %s:%d" % (path, line_number)
                ) from exc
            _require(isinstance(value, dict), "JSONL rows must be objects")
            rows.append(value)
    return rows


def _file_records(root: Path) -> List[Dict[str, Any]]:
    root = Path(root).resolve()
    _require(root.is_dir(), "provenance root is missing: %s" % root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": file_sha256(path),
            }
        )
    _require(rows, "provenance root has no regular files: %s" % root)
    return rows


def _binary_source_manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Match the frozen Gate H task-source manifest encoding exactly."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(row["sha256"])))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = Path(_run(("git", "rev-parse", "--show-toplevel"))).resolve()
    _require(root == repository_root().resolve(), "not running in the dedicated clone")
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"))
    commit = _run(("git", "rev-parse", "HEAD"))
    origin = _run(("git", "rev-parse", "origin/loopscope"))
    dirty = _run(("git", "status", "--porcelain"))
    _require(branch == "loopscope", "remote branch differs from loopscope")
    _require(commit == expected_commit, "remote HEAD differs from expected commit")
    _require(origin == expected_commit, "remote origin/loopscope differs from expected commit")
    _require(dirty == "", "dedicated HPC2 clone must be clean")
    return {
        "repo": str(root),
        "branch": branch,
        "commit": commit,
        "origin_loopscope": origin,
        "dirty": False,
    }


def _assert_offline_environment() -> None:
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


def _package_versions() -> Dict[str, str]:
    observed = {
        name: importlib.metadata.version(name.replace("_", "-"))
        for name in EXPECTED_PACKAGE_VERSIONS
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateIMMLU0Error("torch import failed during CPU admission") from exc
    observed["torch"] = str(torch.__version__)
    expected = dict(EXPECTED_PACKAGE_VERSIONS)
    expected["torch"] = EXPECTED_TORCH_VERSION
    _require(observed == expected, "audited package versions differ: %r" % observed)
    _require(platform.python_version() == "3.11.9", "Python version differs")
    return observed


def _load_card_and_schema() -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    root = repository_root()
    card_path = root / CARD_RELATIVE
    schema_path = root / SCHEMA_RELATIVE
    _require(file_sha256(card_path) == CARD_SHA256, "Gate H card SHA-256 differs")
    _require(file_sha256(schema_path) == SCHEMA_SHA256, "trajectory schema SHA-256 differs")
    card = load_json(card_path)
    schema = load_json(schema_path)
    validate_card(card)
    validate_schema_document(schema)
    return card, schema, card_path, schema_path


def _model_cache_closure(card: Mapping[str, Any]) -> Tuple[Dict[str, Any], Any]:
    _require(MODEL_SNAPSHOT.is_dir(), "cached model snapshot is missing")
    config_path = MODEL_SNAPSHOT / "config.json"
    tokenizer_config_path = MODEL_SNAPSHOT / "tokenizer_config.json"
    tokenizer_json_path = MODEL_SNAPSHOT / "tokenizer.json"
    for path in (config_path, tokenizer_config_path, tokenizer_json_path):
        _require(path.is_file(), "cached model file is missing: %s" % path.name)
    hashes = {
        "config.json": file_sha256(config_path),
        "tokenizer_config.json": file_sha256(tokenizer_config_path),
        "tokenizer.json": file_sha256(tokenizer_json_path),
    }
    expected = {
        "config.json": EXPECTED_CONFIG_SHA256,
        "tokenizer_config.json": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "tokenizer.json": EXPECTED_TOKENIZER_JSON_SHA256,
    }
    _require(hashes == expected, "model/tokenizer cache hashes differ")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _require(config.get("model_type") == "qwen3", "model type differs")
    _require(config.get("num_hidden_layers") == LAYERS, "decoder layer count differs")
    _require(config.get("hidden_size") == HIDDEN_SIZE, "hidden size differs")
    _require(config.get("torch_dtype") == "bfloat16", "model dtype differs")
    _require(config.get("vocab_size") == 151936, "model vocabulary size differs")
    _require(bool(config.get("tie_word_embeddings")), "tied output embedding contract differs")
    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateIMMLU0Error("transformers CPU imports failed") from exc
    loaded_config = AutoConfig.from_pretrained(
        str(MODEL_SNAPSHOT), local_files_only=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_SNAPSHOT), local_files_only=True, trust_remote_code=True
    )
    config_class = loaded_config.__class__.__module__ + "." + loaded_config.__class__.__name__
    tokenizer_class = tokenizer.__class__.__module__ + "." + tokenizer.__class__.__name__
    _require(
        tokenizer_class == card["model"]["tokenizer_class"],
        "tokenizer class differs: %s" % tokenizer_class,
    )
    _require(int(loaded_config.num_hidden_layers) == LAYERS, "loaded config layer count differs")
    _require(int(loaded_config.hidden_size) == HIDDEN_SIZE, "loaded config hidden size differs")
    choice_manifest = freeze_choice_surface_manifest(tokenizer)
    return (
        {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "snapshot": str(MODEL_SNAPSHOT),
            "config_class": config_class,
            "tokenizer_class": tokenizer_class,
            "file_sha256": hashes,
            "config_fields": {
                "model_type": config["model_type"],
                "decoder_layers": config["num_hidden_layers"],
                "hidden_size": config["hidden_size"],
                "vocab_size": config["vocab_size"],
                "dtype": config["torch_dtype"],
                "tie_word_embeddings": bool(config["tie_word_embeddings"]),
            },
            "weights_instantiated": False,
            "weights_forwarded": False,
            "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
            "choice_surface_manifest": choice_manifest,
        },
        tokenizer,
    )


def _task_source_closure() -> Dict[str, Any]:
    package = importlib.import_module("lm_eval")
    package_root = Path(package.__file__).resolve().parent
    default_root = package_root / "tasks/mmlu/default"
    rows = _file_records(default_root)
    _require(len(rows) == EXPECTED_TASK_SOURCE_FILE_COUNT, "MMLU task source file count differs")
    manifest_hash = _binary_source_manifest_sha256(rows)
    _require(
        manifest_hash == EXPECTED_TASK_SOURCE_MANIFEST_SHA256,
        "MMLU task source manifest hash differs",
    )
    by_name = {row["path"]: row["sha256"] for row in rows}
    expected_named = {
        "_mmlu.yaml": "yaml_sha256",
        "_default_template_yaml": "default_template_sha256",
    }
    for filename, card_key in expected_named.items():
        _require(filename in by_name, "MMLU task source is missing %s" % filename)
    card = load_json(repository_root() / CARD_RELATIVE)
    source = card["task"]["lm_eval_task_source"]
    _require(by_name["_mmlu.yaml"] == source["yaml_sha256"], "MMLU group YAML hash differs")
    _require(
        by_name["_default_template_yaml"] == source["default_template_sha256"],
        "MMLU default template hash differs",
    )
    group_aliases = {
        "stem_group_yaml_sha256": ("_mmlu_stem.yaml", "_stem.yaml"),
        "other_group_yaml_sha256": ("_mmlu_other.yaml", "_other.yaml"),
        "social_sciences_group_yaml_sha256": (
            "_mmlu_social_sciences.yaml",
            "_social_sciences.yaml",
        ),
        "humanities_group_yaml_sha256": (
            "_mmlu_humanities.yaml",
            "_humanities.yaml",
        ),
    }
    named_hashes: Dict[str, str] = {
        "_mmlu.yaml": by_name["_mmlu.yaml"],
        "_default_template_yaml": by_name["_default_template_yaml"],
    }
    for card_key, aliases in group_aliases.items():
        matches = [name for name in aliases if name in by_name]
        _require(len(matches) == 1, "MMLU group YAML is not uniquely resolved: %s" % card_key)
        _require(by_name[matches[0]] == source[card_key], "MMLU group YAML hash differs: %s" % card_key)
        named_hashes[matches[0]] = by_name[matches[0]]
    return {
        "package_root": str(package_root),
        "default_root": str(default_root),
        "file_count": len(rows),
        "file_manifest": rows,
        "manifest_sha256": manifest_hash,
        "named_hashes": named_hashes,
        "lm_eval_version": EXPECTED_PACKAGE_VERSIONS["lm_eval"],
    }


def _task_map() -> Dict[str, Any]:
    try:
        from lm_eval.tasks import TaskManager
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateIMMLU0Error("lm-eval TaskManager import failed") from exc
    manager = TaskManager()
    loaded = manager.load_task_or_group("mmlu")
    tasks: Dict[str, Any] = {}

    def flatten(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).startswith("mmlu_") and hasattr(child, "doc_to_text"):
                    tasks[str(key)] = child
                else:
                    flatten(child)

    flatten(loaded)
    _require(len(tasks) == EXPECTED_SUBJECT_COUNT, "MMLU task count differs")
    return tasks


def _safe_doc(row: Mapping[str, Any], subject: str) -> Dict[str, Any]:
    allowed = {"question", "choices", "subject"}
    _require(set(row) >= allowed, "MMLU validation row lacks safe renderer fields")
    safe = {
        "question": row["question"],
        "choices": list(row["choices"]),
        "subject": str(row["subject"]),
    }
    _require(safe["subject"] == subject, "MMLU subject differs from task name")
    _require(isinstance(safe["question"], str), "MMLU question is not text")
    _require(len(safe["choices"]) == 4, "MMLU choice count differs")
    _require(all(isinstance(value, str) for value in safe["choices"]), "MMLU choice is not text")
    return safe


def _safe_doc_hash(safe: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "question": safe["question"],
                "subject": safe["subject"],
                "choices": list(safe["choices"]),
            }
        )
    )


def _render_zero_shot_prompt(task: Any, safe: Mapping[str, Any]) -> str:
    try:
        prompt = task.doc_to_text(dict(safe))
    except Exception as exc:
        raise GateIMMLU0Error("standard MMLU doc_to_text failed") from exc
    _require(isinstance(prompt, str), "MMLU renderer did not return text")
    _require(prompt.rstrip().endswith("Answer:"), "MMLU zero-shot prompt terminal differs")
    return prompt


def _dataset_projection(
    tokenizer: Any,
    task_by_subject: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        from datasets import DownloadMode, load_dataset
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateIMMLU0Error("datasets import failed") from exc
    dataset = load_dataset(
        path=DATASET_REPO,
        revision=DATASET_REVISION,
        split="validation",
        cache_dir=str(HF_DATASETS_CACHE),
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    columns = set(getattr(dataset, "column_names", ()))
    _require(
        {"question", "choices", "subject"}.issubset(columns),
        "MMLU validation dataset lacks the safe column set",
    )
    safe_dataset = dataset.select_columns(["question", "choices", "subject"])
    _require(
        set(getattr(safe_dataset, "column_names", ())) == {"question", "choices", "subject"},
        "safe MMLU projection retained non-safe columns",
    )
    by_subject_index: Counter[str] = Counter()
    rows: List[Dict[str, Any]] = []
    for row in safe_dataset:
        subject = str(row["subject"])
        _require(subject in task_by_subject, "validation subject has no standard MMLU task")
        safe = _safe_doc(row, subject)
        index = by_subject_index[subject]
        by_subject_index[subject] += 1
        prompt = _render_zero_shot_prompt(task_by_subject[subject], safe)
        token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        _require(token_ids, "rendered MMLU prompt tokenizes to empty")
        identity = {
            "task": "mmlu",
            "doc_id": "mmlu_%s:validation:%d" % (subject, index),
            "doc_hash": _safe_doc_hash(safe),
        }
        rows.append(
            {
                "ordinal_source": len(rows),
                "identity": identity,
                "subject": subject,
                "split": "validation",
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                "prompt_token_ids_sha256": _sha256_bytes(canonical_json_bytes(token_ids)),
                "sequence_length": len(token_ids),
            }
        )
    rows.sort(key=lambda row: row["identity"]["doc_id"])
    for ordinal, row in enumerate(rows):
        row["ordinal"] = ordinal
        row.pop("ordinal_source", None)
    _require(len(rows) == EXPECTED_VALIDATION_COUNT, "validation record count differs")
    _require(len({row["subject"] for row in rows}) == EXPECTED_SUBJECT_COUNT, "validation subject count differs")
    identities = [row["identity"] for row in rows]
    _require(len({canonical_json_bytes(value) for value in identities}) == len(rows), "validation identities duplicate")
    _require(
        semantic_sha256(identities) == EXPECTED_VALIDATION_IDENTITY_SHA256,
        "validation identity order hash differs from the frozen Phase 5 closure",
    )
    category_counts = Counter(row["subject"] for row in rows)
    smoke: List[Dict[str, Any]] = []
    seen_subjects = set()
    for row in rows:
        if row["subject"] not in seen_subjects:
            smoke.append(dict(row))
            seen_subjects.add(row["subject"])
        if len(smoke) == SMOKE_COUNT:
            break
    _require(len(smoke) == SMOKE_COUNT, "four-subject smoke membership could not be frozen")
    cache_files = []
    for item in getattr(dataset, "cache_files", []) or []:
        filename = Path(str(item.get("filename", "")))
        if filename.is_file():
            cache_files.append(
                {
                    "path": str(filename),
                    "sha256": file_sha256(filename),
                    "size_bytes": filename.stat().st_size,
                }
            )
    _require(cache_files, "validation cache file closure is empty")
    closure = {
        "record_count": len(rows),
        "subject_count": len(category_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "ordered_identity_sha256": semantic_sha256(identities),
        "ordered_prompt_projection_sha256": semantic_sha256(rows),
        "smoke_identities": [dict(row) for row in smoke],
        "dataset_fingerprint": str(getattr(dataset, "_fingerprint", "")),
        "dataset_cache_files": cache_files,
        "safe_columns": ["question", "choices", "subject"],
        "forbidden_columns_selected": False,
        "test_split_read": False,
        "label_read": False,
        "outcome_read": False,
    }
    _require(closure["dataset_fingerprint"], "validation dataset fingerprint is missing")
    return rows, closure


def _projection_rows(path: Path) -> List[Dict[str, Any]]:
    rows = _load_jsonl(path)
    _require(len(rows) == EXPECTED_VALIDATION_COUNT, "projection row count differs")
    required = {
        "ordinal",
        "identity",
        "subject",
        "split",
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "sequence_length",
    }
    for row in rows:
        _require(set(row) == required, "projection fields are not closed")
        _require(row["split"] == "validation", "projection split differs")
        _require(isinstance(row["sequence_length"], int) and row["sequence_length"] > 0, "projection sequence length invalid")
    return rows


def _projection_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return semantic_sha256(list(rows))


def cpu_admission(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    _assert_offline_environment()
    _require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "expected commit is invalid")
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "NoDevFiles"),
        "CPU admission must not expose CUDA devices",
    )
    _require(not Path(run_root).exists(), "Gate I run root must be fresh")
    git = _git_provenance(expected_commit)
    card, schema, card_path, schema_path = _load_card_and_schema()
    packages = _package_versions()
    model, tokenizer = _model_cache_closure(card)
    task_source = _task_source_closure()
    task_by_subject = _task_map()
    rows, projection = _dataset_projection(tokenizer, task_by_subject)
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "cpu").mkdir(exist_ok=False)
    (root / "slurm").mkdir(exist_ok=False)
    projection_path = root / "cpu/validation_projection.jsonl"
    projection_sha = _write_new_jsonl(projection_path, rows)
    smoke_path = root / "cpu/smoke_membership.json"
    smoke_members = projection["smoke_identities"]
    smoke_manifest = {
        "schema_version": "loopscope.phase6.mmlu0-gate-i-smoke-membership.v1",
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "projection_file_sha256": projection_sha,
        "record_count": SMOKE_COUNT,
        "subject_count": len({row["subject"] for row in smoke_members}),
        "members": smoke_members,
        "partition": SMOKE_PARTITION,
        "time_limit": SMOKE_TIME_LIMIT,
        "cpus_per_task": SMOKE_CPUS,
        "memory": SMOKE_MEMORY,
        "selector_executed": False,
        "loop_executed": False,
        "test_split_read": False,
        "label_read": False,
        "outcome_read": False,
    }
    smoke_manifest["manifest_sha256"] = semantic_sha256(smoke_manifest)
    smoke_sha = _write_new_json(smoke_path, smoke_manifest)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu0-gate-i-cpu-admission.v1",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(root),
        "git": git,
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "model": model,
        "task_source": task_source,
        "packages": packages,
        "projection": {
            "path": str(projection_path.relative_to(root)),
            "file_sha256": projection_sha,
            "record_count": len(rows),
            "subject_count": len({row["subject"] for row in rows}),
            "ordered_identity_sha256": projection["ordered_identity_sha256"],
            "ordered_prompt_projection_sha256": projection["ordered_prompt_projection_sha256"],
            "smoke_manifest_sha256": smoke_sha,
        },
        "information_barrier": {
            "model_weights_forwarded": False,
            "test_split_read": False,
            "label_read": False,
            "outcome_read": False,
            "selector_executed": False,
            "loop_executed": False,
            "raw_text_persisted": False,
            "token_ids_persisted": False,
        },
        "status": "PASS",
    }
    receipt_sha = _write_new_json(root / "cpu/admission_receipt.json", receipt)
    result = dict(receipt)
    result["receipt_sha256"] = receipt_sha
    return result


def _load_projection_and_membership(root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    cpu_receipt = load_json(root / "cpu/admission_receipt.json")
    projection_path = root / str(cpu_receipt["projection"]["path"])
    rows = _projection_rows(projection_path)
    _require(file_sha256(projection_path) == cpu_receipt["projection"]["file_sha256"], "projection file hash differs")
    membership = load_json(root / "cpu/smoke_membership.json")
    membership_payload = dict(membership)
    membership_hash = membership_payload.pop("manifest_sha256", None)
    _require(
        isinstance(membership_hash, str)
        and semantic_sha256(membership_payload) == membership_hash,
        "smoke membership internal hash differs",
    )
    _require(
        membership.get("expected_commit") == cpu_receipt["git"]["commit"],
        "smoke membership commit differs",
    )
    _require(
        membership.get("card_sha256") == cpu_receipt["card_sha256"],
        "smoke membership card hash differs",
    )
    _require(
        membership.get("schema_sha256") == cpu_receipt["schema_sha256"],
        "smoke membership schema hash differs",
    )
    _require(
        membership.get("projection_file_sha256") == cpu_receipt["projection"]["file_sha256"],
        "smoke membership projection hash differs",
    )
    _require(membership.get("record_count") == SMOKE_COUNT, "smoke membership count differs")
    _require(membership.get("partition") == SMOKE_PARTITION, "smoke partition differs")
    _require(membership.get("time_limit") == SMOKE_TIME_LIMIT, "smoke time limit differs")
    members = membership.get("members")
    _require(isinstance(members, list) and len(members) == SMOKE_COUNT, "smoke members differ")
    by_identity = {canonical_json_bytes(row["identity"]): row for row in rows}
    for row in members:
        _require(canonical_json_bytes(row["identity"]) in by_identity, "smoke member absent from projection")
    _require(len({canonical_json_bytes(row["identity"]) for row in members}) == SMOKE_COUNT, "smoke identities duplicate")
    return rows, membership, cpu_receipt


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
    raise GateIMMLU0Error("Qwen3 final norm module is missing")


def _load_native_runtime(card: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any, Any, List[int]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateIMMLU0Error("GPU runtime imports failed") from exc
    _require(bool(torch.cuda.is_available()), "Gate I requires a visible CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_SNAPSHOT), local_files_only=True, trust_remote_code=True
    )
    choice_manifest = freeze_choice_surface_manifest(tokenizer)
    choice_ids = [int(choice_manifest[key]["token_ids"][0]) for key in CHOICE_KEYS]
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
    _require(_sha256_bytes(prompt.encode("utf-8")) == projection_row["prompt_sha256"], "prompt hash differs from CPU projection")
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    _require("input_ids" in encoded and "attention_mask" in encoded, "tokenizer output lacks ids/mask")
    valid = encoded["attention_mask"][0].detach().cpu().nonzero(as_tuple=False).flatten().tolist()
    _require(valid and int(valid[-1]) == int(encoded["attention_mask"].shape[1]) - 1, "probe position is not the last non-padding token")
    position = int(valid[-1])
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
    del normalized_cpu, raw_cpu, inputs, encoded
    return record, evidence


def _safe_dataset_prompts(task_by_subject: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], tokenizer: Any) -> Dict[str, str]:
    try:
        from datasets import DownloadMode, load_dataset
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateIMMLU0Error("datasets import failed during smoke") from exc
    dataset = load_dataset(
        path=DATASET_REPO,
        revision=DATASET_REVISION,
        split="validation",
        cache_dir=str(HF_DATASETS_CACHE),
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    ).select_columns(["question", "choices", "subject"])
    wanted = {canonical_json_bytes(row["identity"]): row for row in selected}
    counters: Counter[str] = Counter()
    prompts: Dict[str, str] = {}
    for row in dataset:
        subject = str(row["subject"])
        safe = _safe_doc(row, subject)
        index = counters[subject]
        counters[subject] += 1
        identity = {
            "task": "mmlu",
            "doc_id": "mmlu_%s:validation:%d" % (subject, index),
            "doc_hash": _safe_doc_hash(safe),
        }
        key = canonical_json_bytes(identity)
        if key not in wanted:
            continue
        prompt = _render_zero_shot_prompt(task_by_subject[subject], safe)
        _require(list(tokenizer.encode(prompt, add_special_tokens=False)), "smoke prompt tokenizes empty")
        _require(_sha256_bytes(prompt.encode("utf-8")) == wanted[key]["prompt_sha256"], "smoke prompt hash differs")
        prompts[key.decode("utf-8")] = prompt
    _require(len(prompts) == len(selected), "smoke prompt reconstruction is incomplete")
    return prompts


def acquire_smoke(*, run_root: Path, expected_commit: str, attempt: int = 1) -> Dict[str, Any]:
    _assert_offline_environment()
    root = Path(run_root).resolve()
    rows, membership, cpu_receipt = _load_projection_and_membership(root)
    git = _git_provenance(expected_commit)
    card, _schema, card_path, _schema_path = _load_card_and_schema()
    task_source = _task_source_closure()
    task_by_subject = _task_map()
    _require(attempt >= 1, "smoke attempt must be positive")
    output = root / "smoke" / ("attempt-%04d" % int(attempt))
    _require(not output.exists(), "smoke attempt root must be fresh")
    output.mkdir(parents=True, exist_ok=False)
    torch_module, tokenizer, model, final_norm, lm_head, choice_ids = _load_native_runtime(card)
    selected = list(membership["members"])
    prompts = _safe_dataset_prompts(task_by_subject, selected, tokenizer)
    by_key = {canonical_json_bytes(row["identity"]): row for row in rows}
    records: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for member in selected:
        key = canonical_json_bytes(member["identity"])
        _require(key in by_key, "smoke member is absent from full projection")
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
    records_path = output / "trajectory_records.jsonl"
    evidence_path = output / "runtime_evidence.jsonl"
    records_sha = _write_new_jsonl(records_path, records)
    evidence_sha = _write_new_jsonl(evidence_path, evidence)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu0-gate-i-gpu-smoke.v1",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(root),
        "attempt": int(attempt),
        "git": git,
        "card_sha256": file_sha256(card_path),
        "task_source_manifest_sha256": task_source["manifest_sha256"],
        "membership_sha256": membership["manifest_sha256"],
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
        "label_read": False,
        "full_vocab_persisted": False,
        "hidden_tensors_persisted": False,
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


def _validate_scheduler(rows: Sequence[Mapping[str, str]], job_id: str) -> Dict[str, Any]:
    _require(len(rows) == 1, "Gate I smoke must have exactly one scheduler row")
    row = dict(rows[0])
    _require(row["JobIDRaw"] == str(job_id), "scheduler job identity differs")
    _require(row["Partition"] == SMOKE_PARTITION, "Gate I smoke did not use debug partition")
    _require(row["State"].startswith("COMPLETED"), "Gate I smoke scheduler state is not COMPLETED")
    _require(row["ExitCode"] == "0:0", "Gate I smoke scheduler ExitCode is not 0:0")
    return row


def verify_smoke(
    *, run_root: Path, expected_commit: str, job_id: str, attempt: int = 1
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = Path(run_root).resolve()
    card, schema, card_path, schema_path = _load_card_and_schema()
    _require(file_sha256(card_path) == CARD_SHA256, "card hash differs during fresh verify")
    _require(file_sha256(schema_path) == SCHEMA_SHA256, "schema hash differs during fresh verify")
    rows, membership, cpu_receipt = _load_projection_and_membership(root)
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
    _require(acquisition["git"]["commit"] == expected_commit, "acquisition commit differs")
    _require(acquisition["git"]["branch"] == "loopscope", "acquisition branch differs")
    _require(acquisition["git"]["dirty"] is False, "acquisition clone was dirty")
    _require(acquisition["attempt"] == int(attempt), "acquisition attempt differs")
    _require(
        acquisition["membership_sha256"] == membership["manifest_sha256"],
        "acquisition membership hash differs",
    )
    _require(
        acquisition["task_source_manifest_sha256"] == cpu_receipt["task_source"]["manifest_sha256"],
        "acquisition task-source hash differs",
    )
    _require(acquisition["forward_count"] == SMOKE_COUNT, "forward count differs")
    _require(acquisition["loop_insertions"] == 0, "loop insertion count differs")
    _require(acquisition["generation_calls"] == 0, "generation calls are nonzero")
    _require(acquisition["selector_executed"] is False, "selector executed in Gate I")
    _require(acquisition["outcome_read"] is False, "outcome was read in Gate I")
    expected_members = {canonical_json_bytes(row["identity"]): row for row in membership["members"]}
    seen = set()
    for record, fact in zip(records, evidence):
        validate_trajectory_record(record, card)
        key = canonical_json_bytes(record["identity"])
        _require(key in expected_members, "trajectory identity differs from smoke membership")
        _require(key not in seen, "trajectory identity is duplicated")
        seen.add(key)
        _require(set(fact) == {
            "identity", "subject", "sequence_length", "probe_position", "final_norm_prehook_calls",
            "final_norm_boundary_calls", "final_norm_closure_max_abs", "native_reprojected_vocab_max_abs", "boundary_count",
            "transition_count", "forward_count", "loop_insertions", "choice_surface_manifest_sha256",
            "elapsed_seconds", "full_vocab_persisted", "hidden_tensors_persisted", "raw_text_persisted",
            "token_ids_persisted",
        }, "runtime evidence fields are not closed")
        _require(fact["final_norm_prehook_calls"] == 1, "FinalNorm pre-hook count differs")
        _require(fact["final_norm_boundary_calls"] == BOUNDARY_COUNT - 1, "FinalNorm boundary call count differs")
        _require(fact["final_norm_closure_max_abs"] == 0.0, "FinalNorm closure is nonzero")
        _require(fact["boundary_count"] == BOUNDARY_COUNT and fact["transition_count"] == LAYERS, "boundary dimensions differ")
        _require(fact["forward_count"] == 1 and fact["loop_insertions"] == 0, "per-record forward closure differs")
        _require(fact["full_vocab_persisted"] is False and fact["hidden_tensors_persisted"] is False, "forbidden tensors persisted")
        _require(fact["raw_text_persisted"] is False and fact["token_ids_persisted"] is False, "raw prompt payload persisted")
        _require(fact["identity"] == record["identity"], "runtime evidence identity differs")
        _require(fact["subject"] == record["subject"], "runtime evidence subject differs")
        _require(
            expected_members[key]["prompt_sha256"] == record["prompt_sha256"],
            "trajectory prompt hash differs",
        )
    _require(seen == set(expected_members), "smoke membership is incomplete")
    scheduler_command, scheduler_rows = query_sacct(job_id)
    scheduler = _validate_scheduler(scheduler_rows, job_id)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu0-gate-i-verifier.v1",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "cpu_admission_receipt_sha256": file_sha256(root / "cpu/admission_receipt.json"),
        "smoke_acquisition_receipt_sha256": file_sha256(output / "acquisition_receipt.json"),
        "scheduler": {
            "job_id": str(job_id),
            "query": scheduler_command,
            "row": scheduler,
        },
        "identity_count": len(records),
        "boundary_count": BOUNDARY_COUNT,
        "transition_count": LAYERS,
        "forward_count": SMOKE_COUNT,
        "loop_insertions": 0,
        "selector_executed": False,
        "test_split_read": False,
        "label_read": False,
        "outcome_read": False,
        "forbidden_persisted_payloads": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(root / "verifier_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def build_sbatch_text(*, run_root: Path, expected_commit: str, attempt: int = 1) -> str:
    _require(attempt >= 1, "smoke attempt must be positive")
    runner = repository_root() / "scripts/loopscope/run_qwen4_phase6_mmlu0_gate_i.py"
    lines = [
        "#!/usr/bin/env bash",
        "# Gate I only: four identity debug smoke; Gate J remains locked.",
        "#SBATCH --job-name=loopscope-p6-i-smoke",
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
    _require("#SBATCH --gres=gpu:1" in text, "smoke launcher GPU request drifted")
    _require("formal" not in text.lower(), "smoke launcher contains formal execution")
    return text


def write_launcher(*, run_root: Path, expected_commit: str, attempt: int = 1) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    _require(root.is_dir(), "Gate I run root is absent")
    path = root / "slurm/gate_i_smoke.sbatch"
    body = build_sbatch_text(run_root=root, expected_commit=expected_commit, attempt=attempt).encode("utf-8")
    sha = _write_new_bytes(path, body)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu0-gate-i-launcher.v1",
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
    }
    receipt_sha = _write_new_json(root / "slurm/launcher_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def dry_run() -> Dict[str, Any]:
    card, schema, card_path, schema_path = _load_card_and_schema()
    return {
        "status": "DRY_RUN_VALID",
        "gate": GATE,
        "card_sha256": file_sha256(card_path),
        "schema_sha256": file_sha256(schema_path),
        "model_revision": card["model"]["revision"],
        "dataset_revision": card["task"]["revision"],
        "validation_record_count": card["task"]["validation_record_count"],
        "validation_subject_count": card["task"]["validation_subject_count"],
        "smoke_count": SMOKE_COUNT,
        "partition": SMOKE_PARTITION,
        "time_limit": SMOKE_TIME_LIMIT,
        "selector_executed": False,
        "loop_executed": False,
        "outcome_read": False,
        "formal_acquisition": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LoopScope Phase 6 Gate I MMLU 0-shot runner")
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("dry-run")
    cpu = actions.add_parser("cpu-admission")
    cpu.add_argument("--run-root", type=Path, required=True)
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
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        result = dry_run()
    elif args.command == "cpu-admission":
        result = cpu_admission(run_root=args.run_root, expected_commit=args.expected_commit)
    elif args.command == "smoke":
        result = acquire_smoke(run_root=args.run_root, expected_commit=args.expected_commit, attempt=args.attempt)
    elif args.command == "verify":
        result = verify_smoke(
            run_root=args.run_root,
            expected_commit=args.expected_commit,
            job_id=args.job_id,
            attempt=args.attempt,
        )
    else:
        result = write_launcher(run_root=args.run_root, expected_commit=args.expected_commit, attempt=args.attempt)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AUTHORIZED_BASE",
    "GateIMMLU0Error",
    "SMOKE_PARTITION",
    "SMOKE_TIME_LIMIT",
    "build_sbatch_text",
    "dry_run",
    "main",
]
