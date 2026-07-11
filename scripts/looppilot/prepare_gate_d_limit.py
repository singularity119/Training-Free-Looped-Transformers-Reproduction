#!/usr/bin/env python3
"""Freeze the authoritative 20-doc Gate D renderer, doc, and mask manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from tflt.looppilot.limit import GATE_D_TASKS, validate_gate_d_config
from tflt.looppilot.probe import (
    assert_no_forbidden_fields,
    build_doc_identity,
    build_question_mask,
    sha256_bytes,
)
from tflt.looppilot.schema import write_json_once


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/looppilot/gate_d_limit.json")
    parser.add_argument("--join-schema", default="configs/looppilot/gate_d_join_schema.json")
    parser.add_argument(
        "--mask-schema", default="configs/looppilot/gate_c_question_mask_schema.json"
    )
    parser.add_argument("--criterion", default="configs/looppilot/criterion_v0.json")
    parser.add_argument("--gate-c-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    join_schema_path = Path(args.join_schema).resolve()
    mask_schema_path = Path(args.mask_schema).resolve()
    criterion_path = Path(args.criterion).resolve()
    config = _load_json(config_path)
    validate_gate_d_config(config)
    _validate_join_schema(_load_json(join_schema_path))
    _validate_criterion(_load_json(criterion_path))

    gate_c_run = Path(args.gate_c_run).resolve()
    gate_c_manifest = gate_c_run / "control/job_results_sha256.txt"
    if _sha256_file(gate_c_manifest) != config["gate_c_results_manifest_sha256"]:
        raise RuntimeError("accepted Gate C results manifest hash mismatch")
    _verify_sha256_manifest(gate_c_manifest, gate_c_run / "control")
    gate_c_documents = _read_jsonl(gate_c_run / "results/documents.jsonl")
    gate_c_by_task = {row["task"]: row for row in gate_c_documents}
    gate_c_renderer = _load_json(gate_c_run / "results/renderer_manifest.json")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from lm_eval import tasks as lm_tasks
        from lm_eval.tasks import TaskManager
    except Exception as exc:
        raise RuntimeError("Gate D preflight requires cached transformers and lm-eval") from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=config["model"],
            revision=config["model_revision"],
            local_files_only=True,
        )
    ).resolve()
    if snapshot_path.name != config["model_revision"]:
        raise RuntimeError("cached model snapshot does not match frozen Gate D revision")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        revision=config["tokenizer_revision"],
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("Gate D requires a fast tokenizer with offsets")

    lm_eval_version = _distribution_version("lm_eval", "lm-eval")
    if lm_eval_version != config["lm_eval_version"]:
        raise RuntimeError("Gate D lm-eval version mismatch")
    task_manager = TaskManager()
    renderer_rows: List[Dict[str, Any]] = []
    document_rows: List[Dict[str, Any]] = []

    for task_name in GATE_D_TASKS:
        entry = task_manager.task_index.get(task_name)
        if not isinstance(entry, Mapping) or "yaml_path" not in entry:
            raise RuntimeError("missing authoritative task YAML for %s" % task_name)
        yaml_path = Path(str(entry["yaml_path"])).resolve()
        yaml_hash = sha256_bytes(yaml_path.read_bytes())
        renderer_row = {
            "task": task_name,
            "task_yaml_path": str(yaml_path),
            "task_yaml_hash": yaml_hash,
            "lm_eval_version": lm_eval_version,
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "fewshot_seed": config["fewshot_seed"],
        }
        renderer_rows.append(renderer_row)

        gate_c_renderer_row = {
            row["task"]: row for row in gate_c_renderer.get("renderers", [])
        }.get(task_name)
        if gate_c_renderer_row != renderer_row:
            raise RuntimeError("Gate D renderer differs from accepted Gate C for %s" % task_name)

        task = lm_tasks.get_task_dict([task_name], task_manager=task_manager)[task_name]
        task.set_fewshot_seed(int(config["fewshot_seed"]))
        if not task.has_test_docs():
            raise RuntimeError("task %s has no test split" % task_name)
        docs = task.test_docs()
        task_version = str(getattr(task, "VERSION", ""))
        for doc_index in config["doc_indices"][task_name]:
            if doc_index >= len(docs):
                raise RuntimeError("task %s is missing frozen doc index %d" % (task_name, doc_index))
            doc = docs[doc_index]
            current_text = task.doc_to_text(doc)
            context = task.fewshot_context(doc, int(config["num_fewshot"]))
            if not isinstance(current_text, str) or not isinstance(context, str):
                raise TypeError("Gate D renderer must return plain-text context")
            encoded = tokenizer(
                context,
                add_special_tokens=True,
                return_attention_mask=True,
                return_offsets_mapping=True,
            )
            mask = build_question_mask(
                context,
                current_text,
                encoded["offset_mapping"],
                encoded["attention_mask"],
            )
            identity = build_doc_identity(
                task=task_name,
                task_version=task_version,
                renderer_hash=yaml_hash,
                occurrence_id="test:%d" % doc_index,
                doc=doc,
            )
            row = {
                "schema_version": 1,
                "task": task_name,
                "task_version": task_version,
                "split": "test",
                "doc_index": doc_index,
                "occurrence_id": "test:%d" % doc_index,
                "doc_key": identity["doc_key"],
                "doc_hash": identity["doc_hash"],
                "renderer_hash": yaml_hash,
                "context_hash": sha256_bytes(context.encode("utf-8")),
                "current_text_hash": sha256_bytes(current_text.encode("utf-8")),
                "manifest_doc": identity["manifest_doc"],
                "mask_hash": sha256_bytes(bytes(int(value) for value in mask)),
                "mask_schema_hash": sha256_bytes(mask_schema_path.read_bytes()),
                "total_token_count": len(encoded["input_ids"]),
                "question_token_count": sum(int(value) for value in mask),
                "choice_count": len(identity["manifest_doc"]["choices"]),
                "choice_continuations_evaluated": True,
                "choice_continuations_available_to_controller": False,
            }
            assert_no_forbidden_fields(row)
            document_rows.append(row)
            if doc_index == 0:
                _assert_gate_c_doc_link(row, gate_c_by_task.get(task_name))

    renderer_manifest = {
        "schema_version": 1,
        "lm_eval_version": lm_eval_version,
        "renderers": renderer_rows,
        "renderer_contract": "plain-text lm-eval fewshot_context; current doc is unlabeled suffix",
    }
    manifest = {
        "schema_version": 1,
        "gate": "D",
        "config_path": str(config_path),
        "config_hash": _sha256_file(config_path),
        "join_schema_path": str(join_schema_path),
        "join_schema_hash": _sha256_file(join_schema_path),
        "mask_schema_path": str(mask_schema_path),
        "mask_schema_hash": _sha256_file(mask_schema_path),
        "criterion_path": str(criterion_path),
        "criterion_hash": _sha256_file(criterion_path),
        "model": config["model"],
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "snapshot_path": str(snapshot_path),
        "lm_eval_version": lm_eval_version,
        "renderer_manifest": renderer_manifest,
        "documents": document_rows,
        "document_count": len(document_rows),
        "gate_c_link": {
            "accepted_run": str(gate_c_run),
            "results_manifest_path": str(gate_c_manifest),
            "results_manifest_sha256": _sha256_file(gate_c_manifest),
            "doc_zero_linked_for_all_tasks": True,
            "always_loop_legacy_equivalence_accepted": True,
        },
        "threshold_selected": False,
        "ranking_computed": False,
        "limit_for_scientific_claim": False,
    }
    if len(document_rows) != 20 or len({row["doc_key"] for row in document_rows}) != 20:
        raise RuntimeError("Gate D preflight did not freeze exactly 20 unique docs")
    assert_no_forbidden_fields(manifest)
    write_json_once(output_dir / "preflight_manifest.json", manifest)
    print(str(output_dir / "preflight_manifest.json"))
    return 0


def _assert_gate_c_doc_link(current: Mapping[str, Any], accepted: Any) -> None:
    if not isinstance(accepted, Mapping):
        raise RuntimeError("accepted Gate C document is missing")
    fields = (
        "task",
        "task_version",
        "doc_key",
        "doc_hash",
        "renderer_hash",
        "context_hash",
        "current_text_hash",
        "manifest_doc",
        "mask_hash",
        "mask_schema_hash",
        "total_token_count",
        "question_token_count",
        "choice_count",
    )
    mismatches = [field for field in fields if current.get(field) != accepted.get(field)]
    if mismatches:
        raise RuntimeError("Gate D doc zero differs from accepted Gate C: %s" % mismatches)


def _validate_join_schema(payload: Mapping[str, Any]) -> None:
    if payload.get("expected_doc_count") != 20:
        raise ValueError("Gate D join schema doc count mismatch")
    if payload.get("expected_signal_request_count") != 80:
        raise ValueError("Gate D join schema signal count mismatch")
    if payload.get("join_by_line_number") is not False:
        raise ValueError("Gate D join schema must prohibit line-number joins")


def _validate_criterion(payload: Mapping[str, Any]) -> None:
    if payload.get("threshold") is not None or payload.get("label_fitted") is not False:
        raise ValueError("Gate D criterion must remain engineering-only and unfitted")
    if payload.get("allowed_engineering_controllers") != ["always_loop", "never_loop"]:
        raise ValueError("Gate D criterion controller set mismatch")


def _verify_sha256_manifest(path: Path, cwd: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        target = Path(name.strip())
        if not target.is_absolute():
            target = cwd / target
        if _sha256_file(target) != digest:
            raise RuntimeError("SHA-256 manifest entry mismatch: %s" % target)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise RuntimeError("required distribution is missing: %s" % (names,))


if __name__ == "__main__":
    raise SystemExit(main())
