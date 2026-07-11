#!/usr/bin/env python3
"""Prepare and freeze label-free Gate C renderer/doc/mask manifests on CPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from tflt.looppilot.probe import (
    GATE_C_DOC_INDEX,
    GATE_C_SPLIT,
    GATE_C_TASKS,
    assert_no_forbidden_fields,
    build_doc_identity,
    build_question_mask,
    sha256_bytes,
    validate_gate_c_config,
)
from tflt.looppilot.schema import write_json_once


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/looppilot/gate_c_probe.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_gate_c_config(config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from lm_eval import tasks as lm_tasks
        from lm_eval.tasks import TaskManager
    except Exception as exc:
        raise RuntimeError("Gate C CPU preflight requires cached transformers and lm-eval") from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=config["model"],
            revision=config["model_revision"],
            local_files_only=True,
        )
    ).resolve()
    if snapshot_path.name != config["model_revision"]:
        raise RuntimeError("cached snapshot does not match frozen revision")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        revision=config["tokenizer_revision"],
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("Gate C preflight requires a fast tokenizer")

    mask_schema_path = Path("configs/looppilot/gate_c_question_mask_schema.json").resolve()
    action_schema_path = Path("configs/looppilot/gate_c_action_schema.json").resolve()
    lm_eval_version = _distribution_version("lm_eval", "lm-eval")
    task_manager = TaskManager()
    renderer_rows: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []

    for task_name in GATE_C_TASKS:
        entry = task_manager.task_index.get(task_name)
        if not isinstance(entry, Mapping) or "yaml_path" not in entry:
            raise RuntimeError("missing authoritative task YAML for %s" % task_name)
        yaml_path = Path(str(entry["yaml_path"])).resolve()
        yaml_hash = sha256_bytes(yaml_path.read_bytes())
        renderer_rows.append(
            {
                "task": task_name,
                "task_yaml_path": str(yaml_path),
                "task_yaml_hash": yaml_hash,
                "lm_eval_version": lm_eval_version,
                "apply_chat_template": False,
                "fewshot_as_multiturn": False,
                "fewshot_seed": config["fewshot_seed"],
            }
        )
        task = lm_tasks.get_task_dict([task_name], task_manager=task_manager)[task_name]
        task.set_fewshot_seed(int(config["fewshot_seed"]))
        if not task.has_test_docs():
            raise RuntimeError("task %s has no test split" % task_name)
        docs = task.test_docs()
        if len(docs) <= GATE_C_DOC_INDEX:
            raise RuntimeError("task %s is missing doc index 0" % task_name)
        doc = docs[GATE_C_DOC_INDEX]
        current_text = task.doc_to_text(doc)
        context = task.fewshot_context(doc, int(config["num_fewshot"]))
        if not isinstance(current_text, str) or not isinstance(context, str):
            raise TypeError("Gate C renderer must return one plain-text context")
        encoded = tokenizer(
            context,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        mask = build_question_mask(
            context, current_text, encoded["offset_mapping"], encoded["attention_mask"]
        )
        identity = build_doc_identity(
            task=task_name,
            task_version=str(getattr(task, "VERSION", "")),
            renderer_hash=yaml_hash,
            occurrence_id="%s:%d" % (GATE_C_SPLIT, GATE_C_DOC_INDEX),
            doc=doc,
        )
        row = {
            "schema_version": 1,
            "task": task_name,
            "task_version": str(getattr(task, "VERSION", "")),
            "split": GATE_C_SPLIT,
            "doc_index": GATE_C_DOC_INDEX,
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
            "choice_continuations_evaluated": False,
        }
        assert_no_forbidden_fields(row)
        documents.append(row)

    renderer_manifest = {
        "schema_version": 1,
        "lm_eval_version": lm_eval_version,
        "renderers": renderer_rows,
        "renderer_contract": "plain-text lm-eval fewshot_context; current doc is unlabeled suffix",
    }
    manifest = {
        "schema_version": 1,
        "config_path": str(config_path),
        "config_hash": sha256_bytes(config_path.read_bytes()),
        "model": config["model"],
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "snapshot_path": str(snapshot_path),
        "lm_eval_version": lm_eval_version,
        "renderer_manifest": renderer_manifest,
        "documents": documents,
        "mask_schema_path": str(mask_schema_path),
        "mask_schema_hash": sha256_bytes(mask_schema_path.read_bytes()),
        "action_schema_path": str(action_schema_path),
        "action_schema_hash": sha256_bytes(action_schema_path.read_bytes()),
    }
    assert_no_forbidden_fields(manifest)
    write_json_once(output_dir / "preflight_manifest.json", manifest)
    print(str(output_dir / "preflight_manifest.json"))
    return 0


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise RuntimeError("required distribution is missing: %s" % (names,))


if __name__ == "__main__":
    raise SystemExit(main())
