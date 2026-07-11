#!/usr/bin/env python3
"""Freeze authoritative Gate E MMLU tasks, all docs, split, and eight shards."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from tflt.looppilot.full import assign_task_shards, build_split_rows, validate_gate_e_config, validate_leaf_tasks
from tflt.looppilot.probe import build_doc_identity, build_question_mask, sha256_bytes
from tflt.looppilot.schema import write_json_once, write_jsonl_once


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/looppilot/gate_e_full.json")
    parser.add_argument("--analysis-contract", default="configs/looppilot/gate_e_analysis_contract.json")
    parser.add_argument("--artifact-schema", default="configs/looppilot/gate_e_artifact_schema.json")
    parser.add_argument("--mask-schema", default="configs/looppilot/gate_c_question_mask_schema.json")
    parser.add_argument("--gate-c-run", required=True)
    parser.add_argument("--gate-d-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    paths = {name: Path(value).resolve() for name, value in {
        "config": args.config,
        "analysis_contract": args.analysis_contract,
        "artifact_schema": args.artifact_schema,
        "mask_schema": args.mask_schema,
    }.items()}
    config = _json(paths["config"])
    validate_gate_e_config(config)
    gate_c = _accepted_run(Path(args.gate_c_run), config["gate_c_results_manifest_sha256"])
    gate_d = _accepted_run(Path(args.gate_d_run), config["gate_d_results_manifest_sha256"])
    job_id = (gate_d / "control/job_id.txt").read_text().strip()
    if job_id != config["gate_d_canonical_job"] or job_id == config["gate_d_rejected_duplicate_job"]:
        raise RuntimeError("Gate E linkage is not the accepted Gate D canonical job")

    try:
        from huggingface_hub import snapshot_download
        from lm_eval import tasks as lm_tasks
        from lm_eval.tasks import TaskManager
        from transformers import AutoTokenizer
        import yaml
    except Exception as exc:
        raise RuntimeError("Gate E preflight requires cached transformers, yaml, and lm-eval") from exc

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Path(snapshot_download(repo_id=config["model"], revision=config["model_revision"], local_files_only=True)).resolve()
    if snapshot.name != config["model_revision"]:
        raise RuntimeError("cached model snapshot differs from frozen revision")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["tokenizer_revision"], local_files_only=True,
        trust_remote_code=True, use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("Gate E requires a fast tokenizer")
    lm_eval_version = _version("lm_eval", "lm-eval")
    if lm_eval_version != config["lm_eval_version"]:
        raise RuntimeError("Gate E lm-eval version mismatch")

    manager = TaskManager()
    leaf_names = validate_leaf_tasks(_expand_group("mmlu", manager.task_index, yaml, set()))
    task_objects = lm_tasks.get_task_dict(list(leaf_names), task_manager=manager)
    task_rows: List[Dict[str, Any]] = []
    document_rows: List[Dict[str, Any]] = []
    renderer_rows: List[Dict[str, Any]] = []
    for task_name in leaf_names:
        entry = manager.task_index.get(task_name)
        if not isinstance(entry, Mapping) or entry.get("type") != "task" or "yaml_path" not in entry:
            raise RuntimeError("missing authoritative leaf YAML for %s" % task_name)
        yaml_path = Path(str(entry["yaml_path"])).resolve()
        yaml_hash = sha256_bytes(yaml_path.read_bytes())
        task = task_objects[task_name]
        task.set_fewshot_seed(int(config["fewshot_seed"]))
        if not task.has_test_docs():
            raise RuntimeError("task %s has no authoritative test docs" % task_name)
        docs = task.test_docs()
        task_version = str(getattr(task, "VERSION", ""))
        renderer_rows.append({
            "task": task_name, "task_yaml_path": str(yaml_path), "task_yaml_hash": yaml_hash,
            "task_version": task_version, "lm_eval_version": lm_eval_version,
            "num_fewshot": config["num_fewshot"], "fewshot_seed": config["fewshot_seed"],
            "apply_chat_template": False, "fewshot_as_multiturn": False,
        })
        doc_hashes = []
        for doc_index, doc in enumerate(docs):
            current_text = task.doc_to_text(doc)
            context = task.fewshot_context(doc, int(config["num_fewshot"]))
            if not isinstance(current_text, str) or not isinstance(context, str):
                raise RuntimeError("Gate E renderer must return plain text")
            encoded = tokenizer(context, add_special_tokens=True, return_attention_mask=True, return_offsets_mapping=True)
            mask = build_question_mask(context, current_text, encoded["offset_mapping"], encoded["attention_mask"])
            identity = build_doc_identity(task_name, task_version, yaml_hash, "test:%d" % doc_index, doc)
            row = {
                "schema_version": 1, "task": task_name, "task_version": task_version,
                "split": "test", "doc_index": doc_index, "occurrence_id": "test:%d" % doc_index,
                "doc_key": identity["doc_key"], "doc_hash": identity["doc_hash"], "renderer_hash": yaml_hash,
                "context_hash": sha256_bytes(context.encode()), "current_text_hash": sha256_bytes(current_text.encode()),
                "mask_hash": sha256_bytes(bytes(int(value) for value in mask)),
                "mask_schema_hash": _sha(paths["mask_schema"]), "total_token_count": len(encoded["input_ids"]),
                "question_token_count": sum(int(value) for value in mask),
                "choice_count": len(identity["manifest_doc"]["choices"]),
                "choice_continuations_available_to_controller": False,
            }
            document_rows.append(row)
            doc_hashes.append(identity["doc_hash"])
        task_rows.append({
            "schema_version": 1, "task": task_name, "task_version": task_version,
            "task_yaml_path": str(yaml_path), "task_yaml_hash": yaml_hash, "doc_count": len(docs),
            "test_doc_keys_sha256": sha256_bytes(("\n".join(sorted(doc_hashes)) + "\n").encode()),
        })

    if len({row["doc_key"] for row in document_rows}) != len(document_rows):
        raise RuntimeError("Gate E authoritative docs contain duplicate keys")
    split_rows = build_split_rows(document_rows)
    shards = assign_task_shards({row["task"]: row["doc_count"] for row in task_rows}, 8)
    renderer_manifest = {"schema_version": 1, "lm_eval_version": lm_eval_version, "renderers": renderer_rows}
    manifest = {
        "schema_version": 1, "gate": "E", "config_hash": _sha(paths["config"]),
        "analysis_contract_hash": _sha(paths["analysis_contract"]), "artifact_schema_hash": _sha(paths["artifact_schema"]),
        "mask_schema_hash": _sha(paths["mask_schema"]), "model": config["model"],
        "model_revision": config["model_revision"], "tokenizer_revision": config["tokenizer_revision"],
        "snapshot_path": str(snapshot), "lm_eval_version": lm_eval_version,
        "leaf_task_count": len(task_rows), "document_count": len(document_rows), "shard_count": 8,
        "gate_c_link": {"run": str(gate_c), "results_manifest_sha256": config["gate_c_results_manifest_sha256"]},
        "gate_d_link": {"run": str(gate_d), "job_id": job_id, "results_manifest_sha256": config["gate_d_results_manifest_sha256"], "rejected_duplicate_excluded": True},
        "renderer_manifest": renderer_manifest,
    }
    write_json_once(output / "preflight_manifest.json", manifest)
    write_json_once(output / "task_manifest.json", {"schema_version": 1, "tasks": task_rows})
    write_jsonl_once(output / "document_manifest.jsonl", document_rows)
    write_jsonl_once(output / "split_manifest.jsonl", split_rows)
    write_json_once(output / "shard_manifest.json", {"schema_version": 1, "algorithm": "lpt(-doc_count,task_name); min(load,shard_id)", "shards": shards})
    write_json_once(output / "renderer_manifest.json", renderer_manifest)
    print(json.dumps({"leaf_tasks": len(task_rows), "documents": len(document_rows), "shards": [row["doc_count"] for row in shards]}, sort_keys=True))
    return 0


def _expand_group(name: str, index: Mapping[str, Any], yaml: Any, active: Set[str]) -> List[str]:
    if name in active:
        raise RuntimeError("recursive lm-eval task group: %s" % name)
    entry = index.get(name)
    if not isinstance(entry, Mapping):
        raise RuntimeError("missing lm-eval task index entry: %s" % name)
    entry_type = entry.get("type")
    if entry_type == "task":
        return [name]
    if entry_type == "tag":
        children = entry.get("task")
    elif entry_type == "group" and "yaml_path" in entry:
        payload = yaml.safe_load(Path(str(entry["yaml_path"])).read_text())
        children = payload.get("task") if isinstance(payload, Mapping) else None
    else:
        raise RuntimeError("invalid lm-eval group entry: %s" % name)
    if not isinstance(children, list) or not children:
        raise RuntimeError("lm-eval group/tag has no children: %s" % name)
    result: List[str] = []
    for child in children:
        if not isinstance(child, str):
            raise RuntimeError("Gate E does not allow inline dynamic task definitions")
        result.extend(_expand_group(child, index, yaml, active | {name}))
    return result


def _accepted_run(path: Path, expected_hash: str) -> Path:
    root = path.resolve()
    manifest = root / "control/job_results_sha256.txt"
    if _sha(manifest) != expected_hash:
        raise RuntimeError("accepted results manifest hash mismatch: %s" % root)
    _verify_manifest(manifest, root / "control")
    preflight = root / "control/preflight_sha256.txt"
    _verify_manifest(preflight, root / "control")
    return root


def _verify_manifest(path: Path, cwd: Path) -> None:
    for line in path.read_text().splitlines():
        digest, name = line.split(None, 1)
        target = Path(name.strip())
        if not target.is_absolute():
            target = cwd / target
        if _sha(target) != digest:
            raise RuntimeError("manifest mismatch: %s" % target)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise RuntimeError("required distribution is missing")


if __name__ == "__main__":
    raise SystemExit(main())
