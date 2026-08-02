"""Gate K contract, CPU projection, and independent verifier for MMLU 5-shot.

The remote projection path imports lm-eval, datasets, numpy, and the cached
tokenizer only.  It selects validation ``question/choices/subject`` columns,
uses dev demonstrations (including their example answers) through the real
standard task renderer, hashes rendered prompts/tokenization in memory, and
persists only closed scalar/provenance records.  It never instantiates model
weights, calls a model, reads validation targets, touches test data, or runs a
selector/loop/scheduler action.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase6_mmlu5_schema import (
    CARD_SCHEMA_VERSION,
    CHOICE_SURFACE_MANIFEST_SHA256,
    DATASET_REPO,
    DATASET_REVISION,
    DEMONSTRATION_COUNT,
    FORMAL_REPLICATES,
    FORMAL_SEED,
    MODEL_CONFIG_SHA256,
    MODEL_REPO,
    MODEL_REVISION,
    MMLU5ContractError,
    PROMPT_PROJECTION_SCHEMA_VERSION,
    RENDER_CONTRACT_VERSION,
    TASK_SOURCE_FILE_COUNT,
    TASK_SOURCE_MANIFEST_SHA256,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_JSON_SHA256,
    TRAJECTORY_SCHEMA_VERSION,
    VALIDATION_IDENTITY_SHA256,
    VALIDATION_RECORD_COUNT,
    VALIDATION_SUBJECT_COUNT,
    ZERO_SHOT_PROMPT_PROJECTION_SHA256,
    canonical_json_bytes,
    enumerate_candidates,
    file_sha256,
    freeze_choice_surface_manifest,
    load_json,
    semantic_sha256,
    validate_card,
    validate_choice_surface_manifest,
    validate_prompt_projection_record,
    validate_schema_document,
)


GATE = "K"
EXECUTOR_THREAD_ID = "019fc119-30b4-7dd0-92bf-e6f2f994c8c8"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
AUTHORIZED_BASE = "37b454bca209b97512325475587cc919c13a4040"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/"
    "training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
MODEL_SNAPSHOT = HF_HOME / "hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots" / MODEL_REVISION
CARD_RELATIVE = Path("configs/loopscope/phase6_mmlu5_prefix_card.json")
PROMPT_SCHEMA_RELATIVE = Path("configs/loopscope/phase6_mmlu5_prompt_projection_schema.json")
TRAJECTORY_SCHEMA_RELATIVE = Path("configs/loopscope/phase6_mmlu5_trajectory_schema.json")
EXPECTED_PACKAGE_VERSIONS = {
    "lm_eval": "0.4.11",
    "transformers": "4.51.3",
    "tokenizers": "0.21.4",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
}
TARGET_COLUMNS = ("question", "choices", "subject")
DEV_COLUMNS = ("question", "choices", "subject", "answer")
FORBIDDEN_RECORD_KEYS = {
    "prompt",
    "prompt_text",
    "text",
    "token_ids",
    "input_ids",
    "target",
    "gold",
    "label",
    "correctness",
    "accuracy",
    "gain",
    "flip",
    "outcome",
    "logits",
    "probabilities",
    "hidden",
}


class GateKMMLU5Error(RuntimeError):
    """Fail-closed Gate K contract, projection, or evidence error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateKMMLU5Error(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    return _write_new_bytes(path, body)


def _write_new_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> str:
    body = b"".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
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
                raise GateKMMLU5Error("invalid JSONL at %s:%d" % (path, line_number)) from exc
            _require(isinstance(value, dict), "projection JSONL rows must be objects")
            rows.append(value)
    return rows


def _load_contract() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Path, Path, Path]:
    root = repository_root()
    card_path = root / CARD_RELATIVE
    projection_schema_path = root / PROMPT_SCHEMA_RELATIVE
    trajectory_schema_path = root / TRAJECTORY_SCHEMA_RELATIVE
    card = load_json(card_path)
    projection_schema = load_json(projection_schema_path)
    trajectory_schema = load_json(trajectory_schema_path)
    try:
        validate_card(card)
        validate_schema_document(projection_schema)
        validate_schema_document(trajectory_schema)
    except MMLU5ContractError as exc:
        raise GateKMMLU5Error(str(exc)) from exc
    return card, projection_schema, trajectory_schema, card_path, projection_schema_path, trajectory_schema_path


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = Path(_run(("git", "rev-parse", "--show-toplevel"))).resolve()
    _require(root == repository_root().resolve(), "not running in the dedicated LoopScope clone")
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"))
    commit = _run(("git", "rev-parse", "HEAD"))
    origin = _run(("git", "rev-parse", "origin/loopscope"))
    dirty = _run(("git", "status", "--porcelain"))
    _require(branch == "loopscope", "Gate K must run on loopscope")
    _require(commit == expected_commit, "Gate K expected commit differs")
    _require(origin == expected_commit, "remote origin/loopscope differs from expected commit")
    _require(dirty == "", "dedicated HPC2 clone must be clean")
    return {"repo": str(root), "branch": branch, "commit": commit, "origin_loopscope": origin, "dirty": False}


def _package_versions() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for name in EXPECTED_PACKAGE_VERSIONS:
        try:
            values[name] = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError as exc:
            raise GateKMMLU5Error("required package is not installed: %s" % name) from exc
    try:
        import torch
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateKMMLU5Error("torch CPU import failed") from exc
    values["torch"] = str(torch.__version__)
    expected = dict(EXPECTED_PACKAGE_VERSIONS)
    expected["torch"] = "2.3.1+cu121"
    _require(values == expected, "Gate K package versions differ")
    _require(platform.python_version() == "3.11.9", "Gate K Python version differs")
    return values


def _file_records(root: Path) -> List[Dict[str, Any]]:
    root = Path(root).resolve()
    _require(root.is_dir(), "source root is missing: %s" % root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rows.append({"path": str(path.relative_to(root)), "sha256": file_sha256(path)})
    _require(bool(rows), "source root has no regular files")
    return rows


def _binary_source_manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(row["sha256"])))
        digest.update(b"\n")
    return digest.hexdigest()


def _task_source_closure(card: Mapping[str, Any]) -> Dict[str, Any]:
    package = importlib.import_module("lm_eval")
    package_root = Path(package.__file__).resolve().parent
    default_root = package_root / "tasks/mmlu/default"
    rows = _file_records(default_root)
    _require(len(rows) == TASK_SOURCE_FILE_COUNT, "MMLU task source file count differs")
    manifest_hash = _binary_source_manifest_sha256(rows)
    _require(manifest_hash == TASK_SOURCE_MANIFEST_SHA256, "MMLU task source manifest differs")
    named = {row["path"]: row["sha256"] for row in rows}
    frozen = card["task"]["lm_eval_task_source"]
    for filename, key in (("_mmlu.yaml", "yaml_sha256"), ("_default_template_yaml", "default_template_sha256")):
        _require(named.get(filename) == frozen[key], "MMLU source hash differs: %s" % filename)
    return {
        "package_root": str(package_root),
        "default_root": str(default_root),
        "file_count": len(rows),
        "file_manifest": rows,
        "manifest_sha256": manifest_hash,
        "lm_eval_version": EXPECTED_PACKAGE_VERSIONS["lm_eval"],
    }


def _model_tokenizer_closure(card: Mapping[str, Any]) -> Tuple[Dict[str, Any], Any]:
    _require(MODEL_SNAPSHOT.is_dir(), "cached model snapshot is missing")
    paths = {
        "config.json": MODEL_SNAPSHOT / "config.json",
        "tokenizer_config.json": MODEL_SNAPSHOT / "tokenizer_config.json",
        "tokenizer.json": MODEL_SNAPSHOT / "tokenizer.json",
    }
    for path in paths.values():
        _require(path.is_file(), "cached tokenizer/model metadata is missing: %s" % path.name)
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    expected = {
        "config.json": MODEL_CONFIG_SHA256,
        "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
        "tokenizer.json": TOKENIZER_JSON_SHA256,
    }
    _require(hashes == expected, "cached model/tokenizer metadata hashes differ")
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    _require(config.get("model_type") == "qwen3", "model type differs")
    _require(config.get("num_hidden_layers") == 36 and config.get("hidden_size") == 2560, "model dimensions differ")
    _require(config.get("torch_dtype") == "bfloat16", "model dtype differs")
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateKMMLU5Error("transformers CPU tokenizer import failed") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SNAPSHOT), local_files_only=True, trust_remote_code=True)
    tokenizer_class = tokenizer.__class__.__module__ + "." + tokenizer.__class__.__name__
    _require(tokenizer_class == card["model"]["tokenizer_class"], "tokenizer class differs")
    choice_manifest = freeze_choice_surface_manifest(tokenizer)
    return (
        {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "snapshot": str(MODEL_SNAPSHOT),
            "config_sha256": hashes["config.json"],
            "tokenizer_config_sha256": hashes["tokenizer_config.json"],
            "tokenizer_json_sha256": hashes["tokenizer.json"],
            "tokenizer_class": tokenizer_class,
            "config_fields": {
                "model_type": config.get("model_type"),
                "decoder_layers": config.get("num_hidden_layers"),
                "hidden_size": config.get("hidden_size"),
                "dtype": config.get("torch_dtype"),
                "vocab_size": config.get("vocab_size"),
                "tie_word_embeddings": bool(config.get("tie_word_embeddings")),
            },
            "weights_instantiated": False,
            "weights_forwarded": False,
            "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
            "choice_surface_manifest": choice_manifest,
        },
        tokenizer,
    )


def _task_map() -> Tuple[Any, Dict[str, Any]]:
    try:
        from lm_eval.tasks import TaskManager
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateKMMLU5Error("lm-eval TaskManager import failed") from exc
    manager = TaskManager()
    loaded = manager.load_task_or_group("mmlu")
    tasks: Dict[str, Any] = {}

    def flatten(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).startswith("mmlu_") and hasattr(child, "fewshot_context"):
                    tasks[str(key)] = child
                else:
                    flatten(child)

    flatten(loaded)
    _require(len(tasks) == VALIDATION_SUBJECT_COUNT, "MMLU task count differs")
    return manager, tasks


def _safe_target_doc(row: Mapping[str, Any], subject: str) -> Dict[str, Any]:
    if set(row) != set(TARGET_COLUMNS):
        raise GateKMMLU5Error("validation safe projection columns differ")
    choices = list(row["choices"])
    _require(isinstance(row["question"], str) and len(choices) == 4 and all(isinstance(value, str) for value in choices), "validation safe document shape differs")
    _require(str(row["subject"]) == subject, "validation subject differs")
    return {"question": row["question"], "choices": choices, "subject": subject}


def _safe_target_hash(safe: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes({"question": safe["question"], "subject": safe["subject"], "choices": list(safe["choices"])}))


def _renderer_target_doc(safe: Mapping[str, Any]) -> Dict[str, Any]:
    """Add only lm-eval's empty target sentinel for final-turn rendering."""
    _require(set(safe) == set(TARGET_COLUMNS), "renderer target must be gold-free")
    rendered = dict(safe)
    rendered["answer"] = ""
    return rendered


def _demo_hash(doc: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(dict(doc)))


def _demo_provenance(task: Any, demos: Sequence[Mapping[str, Any]], fewshot_docs: Any, subject: str) -> List[Dict[str, Any]]:
    by_hash: Dict[str, List[int]] = {}
    for index, source_doc in enumerate(fewshot_docs):
        normalized = dict(source_doc)
        by_hash.setdefault(_demo_hash({key: normalized[key] for key in DEV_COLUMNS}), []).append(index)
    result = []
    for demo in demos:
        normalized = dict(demo)
        _require(set(normalized) >= set(DEV_COLUMNS), "dev demonstration lacks frozen fields")
        demo_key = _demo_hash({key: normalized[key] for key in DEV_COLUMNS})
        available = by_hash.get(demo_key, [])
        _require(bool(available), "selected dev demonstration is absent from the frozen dev split")
        demo_index = available.pop(0)
        rendered = str(task.doc_to_text(normalized)) + str(task.doc_to_target(normalized))
        example_answer = str(task.doc_to_target(normalized))
        result.append(
            {
                "id": "mmlu_%s:dev:%d" % (subject, demo_index),
                "doc_index": demo_index,
                "subject": subject,
                "split": "dev",
                "doc_sha256": _demo_hash({key: normalized[key] for key in DEV_COLUMNS}),
                "rendered_sha256": _sha256_bytes(rendered.encode("utf-8")),
                "example_answer_sha256": _sha256_bytes(example_answer.encode("utf-8")),
            }
        )
    _require(len(result) == DEMONSTRATION_COUNT, "renderer did not provide five demos")
    return result


def _load_subject_datasets(subject: str) -> Tuple[Any, Any, Dict[str, Any]]:
    try:
        from datasets import DownloadMode, load_dataset
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateKMMLU5Error("datasets CPU import failed") from exc
    validation = load_dataset(path=DATASET_REPO, name=subject, revision=DATASET_REVISION, split="validation", cache_dir=str(HF_DATASETS_CACHE), download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    dev = load_dataset(path=DATASET_REPO, name=subject, revision=DATASET_REVISION, split="dev", cache_dir=str(HF_DATASETS_CACHE), download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    validation_columns = set(getattr(validation, "column_names", ()))
    dev_columns = set(getattr(dev, "column_names", ()))
    _require(set(TARGET_COLUMNS).issubset(validation_columns), "MMLU validation dataset lacks the safe column set")
    _require(set(DEV_COLUMNS).issubset(dev_columns), "MMLU dev dataset lacks the demonstration column set")
    source_fingerprints = {
        "validation": str(getattr(validation, "_fingerprint", "")),
        "dev": str(getattr(dev, "_fingerprint", "")),
    }
    _require(all(source_fingerprints.values()), "MMLU split fingerprint is missing")
    validation = validation.select_columns(list(TARGET_COLUMNS))
    dev = dev.select_columns(list(DEV_COLUMNS))
    cache_files = []
    for split, dataset in (("validation", validation), ("dev", dev)):
        for item in getattr(dataset, "cache_files", []) or []:
            filename = Path(str(item.get("filename", "")))
            if filename.is_file():
                cache_files.append({"split": split, "path": str(filename), "sha256": file_sha256(filename), "size_bytes": filename.stat().st_size})
    fingerprints = {
        "validation": source_fingerprints["validation"],
        "dev": source_fingerprints["dev"],
    }
    _require(all(fingerprints.values()), "MMLU split fingerprint is missing")
    return validation, dev, {"subject": subject, "revision": DATASET_REVISION, "fingerprints": fingerprints, "cache_files": sorted(cache_files, key=lambda value: (value["split"], value["path"]))}


def _render_projection(
    *,
    card: Mapping[str, Any],
    task_manager: Any,
    tasks: Mapping[str, Any],
    tokenizer: Any,
    source: Mapping[str, Any],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    try:
        import numpy as np
        from datasets import DatasetDict
        from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend
    except Exception as exc:  # pragma: no cover - remote dependency path.
        raise GateKMMLU5Error("Gate K renderer dependencies failed to import") from exc
    backend = LmEvalMMLURendererBackend()
    records: List[Dict[str, Any]] = []
    task_configs: List[Dict[str, Any]] = []
    dataset_tasks: List[Dict[str, Any]] = []
    cache_files: List[Dict[str, Any]] = []
    source_files = {row["path"]: row["sha256"] for row in source["file_manifest"]}
    for task_name in sorted(tasks):
        subject = task_name[len("mmlu_") :]
        task = tasks[task_name]
        config = getattr(task, "config", None)
        _require(str(backend._config_value(config, "fewshot_split")) == "dev", "standard task fewshot_split differs")
        _require(backend._config_value(config, "process_docs") in (None, ""), "unexpected MMLU process_docs path")
        validation, dev, dataset_evidence = _load_subject_datasets(subject)
        task.dataset = DatasetDict({"validation": validation, "dev": dev})
        task_yaml = backend._registered_task_yaml(task_manager, task_name, Path(source["package_root"]))
        task_config, task_paths = backend._task_config_evidence(task, task_name, Path(source["package_root"]), task_yaml)
        task_configs.append(task_config)
        for path in task_paths:
            relative = str(Path(path).resolve().relative_to(Path(source["package_root"]).resolve()))
            source_files[relative] = file_sha256(Path(path))
        dataset_tasks.append({"task_name": task_name, "dataset_path": DATASET_REPO, "dataset_name": subject, "revision": DATASET_REVISION, "split_fingerprints": dataset_evidence["fingerprints"]})
        cache_files.extend(dataset_evidence["cache_files"])
        fixed_demo_contract = None
        for target_index, row in enumerate(validation):
            safe_target = _safe_target_doc(row, subject)
            identity = {"task": "mmlu", "doc_id": "mmlu_%s:validation:%d" % (subject, target_index), "doc_hash": _safe_target_hash(safe_target)}
            prompt, demos = backend._render_with_captured_demos(task, _renderer_target_doc(safe_target), dev, np, int(seed))
            _require(isinstance(prompt, str) and prompt.rstrip().endswith("Answer:"), "standard five-shot prompt terminal differs")
            demo_rows = _demo_provenance(task, demos, dev, subject)
            demo_contract = canonical_json_bytes(demo_rows)
            if fixed_demo_contract is None:
                fixed_demo_contract = demo_contract
            else:
                _require(fixed_demo_contract == demo_contract, "standard dev demonstration order changed within subject")
            token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            _require(token_ids, "rendered five-shot prompt tokenizes to empty")
            records.append(
                {
                    "schema_version": PROMPT_PROJECTION_SCHEMA_VERSION,
                    "identity": identity,
                    "subject": subject,
                    "split": "validation",
                    "num_fewshot": DEMONSTRATION_COUNT,
                    "fewshot_split": "dev",
                    "demonstration_count": DEMONSTRATION_COUNT,
                    "demonstration_ids": [row["id"] for row in demo_rows],
                    "demonstration_provenance": demo_rows,
                    "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    "prompt_tokenization_sha256": _sha256_bytes(canonical_json_bytes(token_ids)),
                    "sequence_length": len(token_ids),
                    "probe_position": len(token_ids) - 1,
                    "renderer_source_manifest_sha256": source["manifest_sha256"],
                    "render_contract_sha256": "PLACEHOLDER_RENDER_CONTRACT_SHA256",
                    "dataset_revision": DATASET_REVISION,
                }
            )
    records.sort(key=lambda row: row["identity"]["doc_id"])
    for ordinal, row in enumerate(records):
        row["ordinal"] = ordinal
    source_file_rows = [{"path": path, "sha256": digest} for path, digest in sorted(source_files.items())]
    source_hash = semantic_sha256(source_file_rows)
    task_configs.sort(key=lambda value: value["task_name"])
    task_config_hash = semantic_sha256(task_configs)
    dataset_tasks.sort(key=lambda value: value["task_name"])
    dataset_hash = semantic_sha256(dataset_tasks)
    render_contract = {
        "version": RENDER_CONTRACT_VERSION,
        "task_group": "mmlu",
        "task_names": sorted(tasks),
        "num_fewshot": DEMONSTRATION_COUNT,
        "fewshot_split": "dev",
        "target_split": "validation",
        "plain_prompt": True,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "generation": False,
        "seed": int(seed),
        "dataset_revision": DATASET_REVISION,
        "dataset_fingerprint_sha256": dataset_hash,
        "renderer_source_sha256": source_hash,
        "template_sha256": task_config_hash,
    }
    render_contract_hash = semantic_sha256(render_contract)
    for row in records:
        row["render_contract_sha256"] = render_contract_hash
        validate_prompt_projection_record(row)
    renderer_evidence = {
        "entrypoint": "lm_eval.api.task.ConfigurableTask.fewshot_context",
        "lm_eval_version": EXPECTED_PACKAGE_VERSIONS["lm_eval"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_file_count": len(source_file_rows),
        "source_file_rows": source_file_rows,
        "task_configs_sha256": task_config_hash,
        "dataset_fingerprint_sha256": dataset_hash,
        "render_contract_sha256": render_contract_hash,
        "render_contract": render_contract,
    }
    return records, dataset_tasks, {"renderer": renderer_evidence, "cache_files": sorted(cache_files, key=lambda value: (value["split"], value["path"]))}


def _manifest_body(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def build_cpu_projection(*, run_root: Path, expected_commit: str, seed: int = FORMAL_SEED) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    _require(not root.exists(), "Gate K run root must be fresh")
    card, projection_schema, trajectory_schema, card_path, projection_schema_path, trajectory_schema_path = _load_contract()
    git = _git_provenance(expected_commit)
    packages = _package_versions()
    source = _task_source_closure(card)
    model, tokenizer = _model_tokenizer_closure(card)
    task_manager, tasks = _task_map()
    records, dataset_tasks, rendering = _render_projection(card=card, task_manager=task_manager, tasks=tasks, tokenizer=tokenizer, source=source, seed=seed)
    _require(len(records) == VALIDATION_RECORD_COUNT, "validation projection count differs")
    _require(len({row["subject"] for row in records}) == VALIDATION_SUBJECT_COUNT, "validation projection subject count differs")
    identities = [row["identity"] for row in records]
    _require(semantic_sha256(identities) == VALIDATION_IDENTITY_SHA256, "validation identity order differs from Gate J")
    projection_path = root / "cpu/validation_5shot_projection.jsonl"
    manifest_path = root / "cpu/manifest.json"
    receipt_path = root / "cpu/projection_receipt.json"
    root.mkdir(parents=True, exist_ok=False)
    projection_sha = _write_new_jsonl(projection_path, records)
    manifest = {
        "schema_version": "loopscope.phase6.mmlu5-gate-k-cpu-manifest.v1",
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(root),
        "expected_commit": expected_commit,
        "git": git,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(projection_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "model": model,
        "packages": packages,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "renderer": rendering["renderer"],
        "dataset_tasks": dataset_tasks,
        "dataset_cache_files": rendering["cache_files"],
        "population": {
            "record_count": len(records),
            "subject_count": len({row["subject"] for row in records}),
            "ordered_identity_sha256": semantic_sha256(identities),
            "validation_identity_sha256": VALIDATION_IDENTITY_SHA256,
            "ordered_prompt_projection_sha256": semantic_sha256(records),
            "zero_shot_reference_prompt_projection_sha256": ZERO_SHOT_PROMPT_PROJECTION_SHA256,
            "five_shot_differs_from_zero_shot": semantic_sha256(records) != ZERO_SHOT_PROMPT_PROJECTION_SHA256,
            "demonstration_count_min": min(row["demonstration_count"] for row in records),
            "demonstration_count_max": max(row["demonstration_count"] for row in records),
        },
        "projection": {
            "path": str(projection_path.relative_to(root)),
            "schema_version": PROMPT_PROJECTION_SCHEMA_VERSION,
            "file_sha256": projection_sha,
        },
        "selector_compatibility": {
            "projection_fields": ["identity", "category", "H", "D"],
            "candidate_count": len(enumerate_candidates()),
            "bootstrap_replicates": FORMAL_REPLICATES,
            "bootstrap_seed": FORMAL_SEED,
            "selector_executed": False,
        },
        "information_barrier": {
            "validation_target_gold_read": False,
            "test_split_read": False,
            "outcome_read": False,
            "model_weights_loaded": False,
            "model_forward_executed": False,
            "cuda_gpu_slurm": False,
            "raw_prompt_persisted": False,
            "token_ids_persisted": False,
        },
    }
    manifest["manifest_sha256"] = semantic_sha256(_manifest_body(manifest))
    manifest_sha = _write_new_json(manifest_path, manifest)
    receipt = {
        "schema_version": "loopscope.phase6.mmlu5-gate-k-cpu-receipt.v1",
        "gate": GATE,
        "status": "PASS",
        "run_root": str(root),
        "expected_commit": expected_commit,
        "manifest_file_sha256": manifest_sha,
        "manifest_semantic_sha256": manifest["manifest_sha256"],
        "projection_file_sha256": projection_sha,
        "record_count": len(records),
        "subject_count": len({row["subject"] for row in records}),
        "ordered_identity_sha256": semantic_sha256(identities),
        "ordered_prompt_projection_sha256": semantic_sha256(records),
        "demonstration_count": DEMONSTRATION_COUNT,
        "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
        "selector_executed": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "outcome_read": False,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "cuda_gpu_slurm": False,
        "raw_prompt_persisted": False,
        "token_ids_persisted": False,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "status_note": "Gate K CPU renderer/projection only; Gate L/M remain locked",
    }
    receipt_sha = _write_new_json(receipt_path, receipt)
    return {**receipt, "manifest_sha256": manifest["manifest_sha256"], "receipt_sha256": receipt_sha}


def _assert_no_forbidden_keys(value: Any, path: Tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_RECORD_KEYS:
                raise GateKMMLU5Error("forbidden persisted field: %s" % ".".join(path + (str(key),)))
            _assert_no_forbidden_keys(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, path + (str(index),))


def verify_cpu_projection(*, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    card, projection_schema, trajectory_schema, card_path, projection_schema_path, trajectory_schema_path = _load_contract()
    manifest_path = root / "cpu/manifest.json"
    receipt_path = root / "cpu/projection_receipt.json"
    projection_path = root / "cpu/validation_5shot_projection.jsonl"
    _require(manifest_path.is_file() and receipt_path.is_file() and projection_path.is_file(), "Gate K CPU artifacts are incomplete")
    manifest = load_json(manifest_path)
    semantic = manifest.get("manifest_sha256")
    _require(isinstance(semantic, str) and semantic_sha256(_manifest_body(manifest)) == semantic, "Gate K manifest semantic hash differs")
    _require(manifest.get("expected_commit") == expected_commit, "Gate K manifest commit differs")
    _require(file_sha256(manifest_path) == load_json(receipt_path).get("manifest_file_sha256"), "Gate K manifest file hash differs")
    _require(manifest.get("card_sha256") == file_sha256(card_path), "Gate K card hash differs")
    _require(manifest.get("prompt_projection_schema_sha256") == file_sha256(projection_schema_path), "Gate K projection schema hash differs")
    _require(manifest.get("trajectory_schema_sha256") == file_sha256(trajectory_schema_path), "Gate K trajectory schema hash differs")
    _require(file_sha256(projection_path) == manifest["projection"]["file_sha256"], "Gate K projection file hash differs")
    records = _load_jsonl(projection_path)
    _require(len(records) == VALIDATION_RECORD_COUNT, "Gate K projection record count differs")
    for record in records:
        _assert_no_forbidden_keys(record)
        validate_prompt_projection_record(record, card)
    _require([row["ordinal"] for row in records] == list(range(len(records))), "Gate K projection ordinals are not canonical")
    identities = [row["identity"] for row in records]
    ordered_identity_sha = semantic_sha256(identities)
    ordered_projection_sha = semantic_sha256(records)
    _require(ordered_identity_sha == VALIDATION_IDENTITY_SHA256, "Gate K identity order differs")
    _require(len({row["subject"] for row in records}) == VALIDATION_SUBJECT_COUNT, "Gate K subject count differs")
    _require(all(row["demonstration_count"] == DEMONSTRATION_COUNT for row in records), "Gate K demonstration count differs")
    _require(ordered_projection_sha != ZERO_SHOT_PROMPT_PROJECTION_SHA256, "five-shot projection equals zero-shot reference")
    population = manifest.get("population", {})
    _require(population.get("ordered_identity_sha256") == ordered_identity_sha and population.get("ordered_prompt_projection_sha256") == ordered_projection_sha, "Gate K population digest differs")
    _require(population.get("five_shot_differs_from_zero_shot") is True, "Gate K zero-shot separation flag differs")
    _require(all(row["render_contract_sha256"] == manifest["renderer"]["render_contract_sha256"] for row in records), "Gate K render contract hash differs")
    barrier = manifest.get("information_barrier", {})
    for key in ("validation_target_gold_read", "test_split_read", "outcome_read", "model_weights_loaded", "model_forward_executed", "cuda_gpu_slurm", "raw_prompt_persisted", "token_ids_persisted"):
        _require(barrier.get(key) is False, "Gate K information barrier differs: %s" % key)
    receipt = load_json(receipt_path)
    for key in ("selector_executed", "validation_target_gold_read", "test_split_read", "outcome_read", "model_weights_loaded", "model_forward_executed", "cuda_gpu_slurm", "raw_prompt_persisted", "token_ids_persisted"):
        _require(receipt.get(key) is False, "Gate K receipt barrier differs: %s" % key)
    verifier = {
        "schema_version": "loopscope.phase6.mmlu5-gate-k-verifier.v1",
        "gate": GATE,
        "status": "PASS",
        "run_root": str(root),
        "expected_commit": expected_commit,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(projection_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_semantic_sha256": semantic,
        "projection_file_sha256": file_sha256(projection_path),
        "record_count": len(records),
        "subject_count": len({row["subject"] for row in records}),
        "ordered_identity_sha256": ordered_identity_sha,
        "ordered_prompt_projection_sha256": ordered_projection_sha,
        "demonstration_count": DEMONSTRATION_COUNT,
        "candidate_count": len(enumerate_candidates()),
        "selector_projection_fields": ["identity", "category", "H", "D"],
        "forbidden_persisted_fields": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "outcome_read": False,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "cuda_gpu_slurm": False,
        "selector_executed": False,
    }
    verifier_sha = _write_new_json(root / "cpu/verifier_receipt.json", verifier)
    return {**verifier, "receipt_sha256": verifier_sha}


def _synthetic_projection_record() -> Dict[str, Any]:
    demos = []
    for index in range(DEMONSTRATION_COUNT):
        demos.append({
            "id": "mmlu_math:dev:%d" % index,
            "doc_index": index,
            "subject": "math",
            "split": "dev",
            "doc_sha256": "%064x" % (index + 1),
            "rendered_sha256": "%064x" % (index + 2),
            "example_answer_sha256": "%064x" % (index + 3),
        })
    return {
        "schema_version": PROMPT_PROJECTION_SCHEMA_VERSION,
        "ordinal": 0,
        "identity": {"task": "mmlu", "doc_id": "mmlu_math:validation:0", "doc_hash": "a" * 64},
        "subject": "math",
        "split": "validation",
        "num_fewshot": DEMONSTRATION_COUNT,
        "fewshot_split": "dev",
        "demonstration_count": DEMONSTRATION_COUNT,
        "demonstration_ids": [row["id"] for row in demos],
        "demonstration_provenance": demos,
        "prompt_sha256": "b" * 64,
        "prompt_tokenization_sha256": "c" * 64,
        "sequence_length": 10,
        "probe_position": 9,
        "renderer_source_manifest_sha256": TASK_SOURCE_MANIFEST_SHA256,
        "render_contract_sha256": "d" * 64,
        "dataset_revision": DATASET_REVISION,
    }


def _expect_failure(label: str, operation: Any, failures: List[str]) -> None:
    try:
        operation()
    except (GateKMMLU5Error, MMLU5ContractError, KeyError, TypeError, ValueError):
        failures.append(label)
        return
    raise GateKMMLU5Error("self-test negative case unexpectedly passed: %s" % label)


def run_self_test() -> Dict[str, Any]:
    card, projection_schema, trajectory_schema, card_path, projection_schema_path, trajectory_schema_path = _load_contract()
    record = _synthetic_projection_record()
    validate_prompt_projection_record(record, card)
    _require(record["probe_position"] == record["sequence_length"] - 1, "synthetic probe position differs")
    _require(len(enumerate_candidates()) == 42, "candidate domain is not exactly 42")
    failures: List[str] = []
    invalid = json.loads(json.dumps(record))
    invalid["num_fewshot"] = 0
    _expect_failure("zero_shot_not_admitted", lambda: validate_prompt_projection_record(invalid, card), failures)
    invalid = json.loads(json.dumps(record))
    invalid["fewshot_split"] = None
    _expect_failure("missing_dev_split", lambda: validate_prompt_projection_record(invalid, card), failures)
    invalid = json.loads(json.dumps(record))
    invalid["demonstration_ids"] = invalid["demonstration_ids"][:4]
    _expect_failure("four_demos_rejected", lambda: validate_prompt_projection_record(invalid, card), failures)
    invalid = json.loads(json.dumps(record))
    invalid["demonstration_ids"][0], invalid["demonstration_ids"][1] = invalid["demonstration_ids"][1], invalid["demonstration_ids"][0]
    _expect_failure("demo_order_rejected", lambda: validate_prompt_projection_record(invalid, card), failures)
    invalid = json.loads(json.dumps(record))
    invalid["demonstration_ids"][0] = "mmlu_math:validation:3"
    _expect_failure("target_demo_overlap_rejected", lambda: validate_prompt_projection_record(invalid, card), failures)
    invalid = json.loads(json.dumps(record))
    invalid["prompt_text"] = "raw prompt"
    _expect_failure("raw_prompt_rejected", lambda: validate_prompt_projection_record(invalid, card), failures)
    invalid = json.loads(json.dumps(record))
    invalid["token_ids"] = [1, 2, 3]
    _expect_failure("token_ids_rejected", lambda: validate_prompt_projection_record(invalid, card), failures)
    invalid = json.loads(json.dumps(record))
    invalid["identity"]["gold"] = 0
    _expect_failure("validation_gold_rejected", lambda: validate_prompt_projection_record(invalid, card), failures)
    invalid_card = json.loads(json.dumps(card))
    invalid_card["evaluator"]["num_fewshot"] = 0
    _expect_failure("card_zero_shot_rejected", lambda: validate_card(invalid_card), failures)
    invalid_card = json.loads(json.dumps(card))
    invalid_card["renderer"]["semantic_contract"]["fewshot_split"] = None
    _expect_failure("card_renderer_split_rejected", lambda: validate_card(invalid_card), failures)
    return {
        "status": "SELF_TEST_PASS",
        "gate": GATE,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(projection_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "synthetic_record_valid": True,
        "candidate_count": len(enumerate_candidates()),
        "selector_projection_fields": ["identity", "category", "H", "D"],
        "negative_cases_passed": failures,
        "negative_case_count": len(failures),
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "outcome_read": False,
        "selector_executed": False,
    }


def dry_run() -> Dict[str, Any]:
    card, projection_schema, trajectory_schema, card_path, projection_schema_path, trajectory_schema_path = _load_contract()
    from tflt.loopscope.phase6_mmlu5_gate_l import import_only_dry_plan

    gate_l_plan = import_only_dry_plan(card_path, trajectory_schema_path)
    return {
        "status": "DRY_RUN_VALID",
        "gate": GATE,
        "card_sha256": file_sha256(card_path),
        "prompt_projection_schema_sha256": file_sha256(projection_schema_path),
        "trajectory_schema_sha256": file_sha256(trajectory_schema_path),
        "model_revision": card["model"]["revision"],
        "dataset_revision": card["task"]["revision"],
        "validation_record_count": VALIDATION_RECORD_COUNT,
        "validation_subject_count": VALIDATION_SUBJECT_COUNT,
        "demonstration_count": DEMONSTRATION_COUNT,
        "candidate_count": len(enumerate_candidates()),
        "formal_bootstrap_replicates": FORMAL_REPLICATES,
        "formal_bootstrap_seed": FORMAL_SEED,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "validation_target_gold_read": False,
        "test_split_read": False,
        "outcome_read": False,
        "selector_executed": False,
        "gate_l_import_only_plan": gate_l_plan,
    }


__all__ = [
    "AUTHORIZED_BASE", "CARD_RELATIVE", "GateKMMLU5Error", "GATE", "PLANNING_THREAD_ID", "PROMPT_SCHEMA_RELATIVE",
    "TRAJECTORY_SCHEMA_RELATIVE", "build_cpu_projection", "dry_run", "enumerate_candidates", "repository_root",
    "run_self_test", "verify_cpu_projection",
]
