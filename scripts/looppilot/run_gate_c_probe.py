#!/usr/bin/env python3
"""Run the frozen four-document LoopPilot Gate C GPU probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tflt.audit import AuditCollector
from tflt.config import LoopConfig
from tflt.looppilot.collector import TensorSignalCollector
from tflt.looppilot.controller import AlwaysLoop, NeverLoop
from tflt.looppilot.probe import (
    GATE_C_DOC_INDEX,
    GATE_C_SPLIT,
    GATE_C_TASKS,
    assert_no_forbidden_fields,
    body_call_closed,
    build_doc_identity,
    build_question_mask,
    sha256_bytes,
    sha256_json,
    validate_action_record,
    validate_gate_c_config,
)
from tflt.looppilot.revisions import build_revision_closure, closure_dict
from tflt.looppilot.schema import write_json_once, write_jsonl_once
from tflt.wrapper import _find_layer_owner, apply_loop_wrapper


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/looppilot/gate_c_probe.json")
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_gate_c_config(config)
    preflight_path = Path(args.preflight_manifest).resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("config_hash") != sha256_bytes(config_path.read_bytes()):
        raise RuntimeError("preflight manifest config hash does not match checked-in Gate C config")
    if preflight.get("model_revision") != config["model_revision"]:
        raise RuntimeError("preflight manifest model revision mismatch")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    torch, AutoModelForCausalLM, AutoTokenizer, snapshot_download = _remote_imports()
    if not torch.cuda.is_available():
        raise RuntimeError("Gate C requires one visible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Gate C job must expose exactly one GPU")

    revision = config["model_revision"]
    snapshot_path = Path(
        snapshot_download(
            repo_id=config["model"],
            revision=revision,
            local_files_only=True,
        )
    ).resolve()
    if snapshot_path.name != revision:
        raise RuntimeError("resolved model snapshot does not match frozen revision")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        revision=config["tokenizer_revision"],
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("Gate C question mask requires a fast tokenizer with offsets")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        revision=revision,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.eval()
    model.to("cuda")
    _assert_model_revision(model, revision)

    model_artifact = {
        "schema_version": 1,
        "repo_id": config["model"],
        "commit_hash": revision,
        "snapshot_path": str(snapshot_path),
        "config_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_type": getattr(model.config, "model_type", None),
    }
    tokenizer_artifact = {
        "schema_version": 1,
        "repo_id": config["model"],
        "commit_hash": config["tokenizer_revision"],
        "snapshot_path": str(snapshot_path),
        "tokenizer_class": tokenizer.__class__.__module__ + "." + tokenizer.__class__.__name__,
        "is_fast": bool(tokenizer.is_fast),
    }
    write_json_once(output_dir / "resolved_model.json", model_artifact)
    write_json_once(output_dir / "resolved_tokenizer.json", tokenizer_artifact)

    mask_schema_path = Path("configs/looppilot/gate_c_question_mask_schema.json").resolve()
    action_schema_path = Path("configs/looppilot/gate_c_action_schema.json").resolve()
    lm_eval_version = _distribution_version("lm_eval", "lm-eval")
    renderer_rows: List[Dict[str, Any]] = []
    document_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    from lm_eval import tasks as lm_tasks
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager()
    owner, layer_attr = _find_layer_owner(model)
    layers = getattr(owner, layer_attr)
    original_classes = _window_classes(layers, tuple(config["window"]))
    original_forward = _forward_identity(model)

    for task_name in GATE_C_TASKS:
        index_entry = task_manager.task_index.get(task_name)
        if not isinstance(index_entry, Mapping) or "yaml_path" not in index_entry:
            raise RuntimeError("missing authoritative task YAML for %s" % task_name)
        yaml_path = Path(str(index_entry["yaml_path"])).resolve()
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
            raise RuntimeError("task %s is missing frozen doc index 0" % task_name)
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
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        attention = encoded["attention_mask"][0].tolist()
        mask_list = build_question_mask(context, current_text, offsets, attention)
        question_mask = torch.tensor([mask_list], dtype=torch.bool, device="cuda")
        inputs = {name: value.to("cuda") for name, value in encoded.items()}

        task_version = str(getattr(task, "VERSION", ""))
        identity = build_doc_identity(
            task=task_name,
            task_version=task_version,
            renderer_hash=yaml_hash,
            occurrence_id="%s:%d" % (GATE_C_SPLIT, GATE_C_DOC_INDEX),
            doc=doc,
        )
        mask_hash = sha256_bytes(bytes(int(value) for value in mask_list))
        document_row = {
            "schema_version": 1,
            "task": task_name,
            "task_version": task_version,
            "split": GATE_C_SPLIT,
            "doc_index": GATE_C_DOC_INDEX,
            "doc_key": identity["doc_key"],
            "doc_hash": identity["doc_hash"],
            "renderer_hash": yaml_hash,
            "context_hash": sha256_bytes(context.encode("utf-8")),
            "current_text_hash": sha256_bytes(current_text.encode("utf-8")),
            "manifest_doc": identity["manifest_doc"],
            "mask_hash": mask_hash,
            "mask_schema_hash": sha256_bytes(mask_schema_path.read_bytes()),
            "total_token_count": int(inputs["input_ids"].shape[-1]),
            "question_token_count": int(question_mask.sum().item()),
            "choice_count": len(identity["manifest_doc"]["choices"]),
            "choice_continuations_evaluated": False,
        }
        assert_no_forbidden_fields(document_row)
        document_rows.append(document_row)
        expected_document = {
            row["task"]: row for row in preflight.get("documents", [])
        }.get(task_name)
        if expected_document != document_row:
            raise RuntimeError("GPU renderer/doc/mask differs from frozen preflight for %s" % task_name)

        with torch.inference_mode():
            baseline = _endpoint_logits(model, inputs, bool(config["use_cache"]))
            baseline_duplicate = _endpoint_logits(model, inputs, bool(config["use_cache"]))
            legacy, legacy_record = _run_action(
                torch, model, inputs, config, identity["doc_key"], task_name, "legacy_k2", None, None
            )
            always_signal = TensorSignalCollector(question_mask, epsilon=float(config["epsilon"]))
            always, always_record = _run_action(
                torch,
                model,
                inputs,
                config,
                identity["doc_key"],
                task_name,
                "always_loop",
                AlwaysLoop(),
                always_signal,
            )
            never_signal = TensorSignalCollector(question_mask, epsilon=float(config["epsilon"]))
            never, never_record = _run_action(
                torch,
                model,
                inputs,
                config,
                identity["doc_key"],
                task_name,
                "never_loop",
                NeverLoop(),
                never_signal,
            )
            restored = _endpoint_logits(model, inputs, bool(config["use_cache"]))

        if _window_classes(getattr(owner, layer_attr), tuple(config["window"])) != original_classes:
            raise RuntimeError("wrapper classes were not restored")
        if _forward_identity(model) != original_forward:
            raise RuntimeError("model forward was not restored")

        endpoints = {
            "unpatched_baseline": _endpoint_summary(torch, baseline),
            "baseline_duplicate": _endpoint_summary(torch, baseline_duplicate),
            "legacy_k2": legacy_record["endpoint"],
            "always_loop": always_record["endpoint"],
            "never_loop": never_record["endpoint"],
            "restored_baseline": _endpoint_summary(torch, restored),
        }
        comparisons = {
            "baseline_duplicate_noise": _compare(
                torch, baseline, baseline_duplicate, config["endpoint_allclose"]
            ),
            "always_vs_legacy": _compare(torch, always, legacy, config["endpoint_allclose"]),
            "never_vs_baseline": _compare(torch, never, baseline, config["endpoint_allclose"]),
            "restore_vs_baseline": _compare(torch, restored, baseline, config["restore_allclose"]),
        }
        if not all(bool(row["allclose"]) for row in comparisons.values()):
            raise RuntimeError("Gate C semantic endpoint comparison failed for %s" % task_name)

        action_rows.extend([legacy_record, always_record, never_record])
        summary = {
            "schema_version": 1,
            "task": task_name,
            "doc_key": identity["doc_key"],
            "mask_hash": mask_hash,
            "endpoints": endpoints,
            "comparisons": comparisons,
            "signals": {
                "always_loop": always_record["signals"],
                "never_loop": never_record["signals"],
            },
            "restore": {
                "window_classes_match": True,
                "forward_identity_match": True,
                "cache_strategy": config["cache_strategy"],
            },
        }
        assert_no_forbidden_fields(summary)
        summaries.append(summary)

        del baseline, baseline_duplicate, legacy, always, never, restored, inputs, question_mask
        gc.collect()
        torch.cuda.empty_cache()

    renderer_manifest = {
        "schema_version": 1,
        "lm_eval_version": lm_eval_version,
        "renderers": renderer_rows,
        "renderer_contract": "plain-text lm-eval fewshot_context; current doc is unlabeled suffix",
    }
    if renderer_manifest != preflight.get("renderer_manifest"):
        raise RuntimeError("GPU renderer manifest differs from frozen preflight")
    assert_no_forbidden_fields(renderer_manifest)
    write_json_once(output_dir / "renderer_manifest.json", renderer_manifest)
    write_jsonl_once(output_dir / "documents.jsonl", document_rows)
    for row in action_rows:
        validate_action_record(row)
    write_jsonl_once(output_dir / "action_records.jsonl", action_rows)

    action_schema_hash = sha256_bytes(action_schema_path.read_bytes())
    probe_summary = {
        "schema_version": 1,
        "scope": "Gate C four-document tensor/schema/restore probe only",
        "config_hash": sha256_bytes(config_path.read_bytes()),
        "action_schema_hash": action_schema_hash,
        "mask_schema_hash": sha256_bytes(mask_schema_path.read_bytes()),
        "documents": summaries,
        "document_count": len(summaries),
        "action_record_count": len(action_rows),
        "body_call_closed": body_call_closed(action_rows),
        "all_semantic_comparisons_passed": all(
            comparison["allclose"]
            for summary in summaries
            for comparison in summary["comparisons"].values()
        ),
        "raw_hidden_persisted": False,
        "full_logits_persisted": False,
        "choice_continuations_evaluated": False,
        "threshold_selected": False,
        "accuracy_computed": False,
        "compute_saving_claimed": False,
    }
    if not probe_summary["body_call_closed"]:
        raise RuntimeError("aggregate operator body-call closure failed")
    assert_no_forbidden_fields(probe_summary)
    write_json_once(output_dir / "probe_summary.json", probe_summary)

    closure = build_revision_closure(
        output_dir / "resolved_model.json",
        output_dir / "resolved_tokenizer.json",
        output_dir / "renderer_manifest.json",
        config_path,
        lm_eval_version,
    )
    write_json_once(output_dir / "revision_closure.json", closure_dict(closure))
    runtime = {
        "schema_version": 1,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _distribution_version("transformers"),
        "lm_eval": lm_eval_version,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json_once(output_dir / "runtime.json", runtime)
    print(json.dumps({"output_dir": str(output_dir), "probe_summary_hash": sha256_json(probe_summary)}))
    return 0


def _remote_imports() -> Tuple[Any, Any, Any, Any]:
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Gate C requires torch, transformers, and huggingface_hub") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, snapshot_download


def _assert_model_revision(model: Any, revision: str) -> None:
    observed = getattr(getattr(model, "config", None), "_commit_hash", None)
    if observed is not None and str(observed) != revision:
        raise RuntimeError("loaded model config commit does not match frozen revision")


def _run_action(
    torch: Any,
    model: Any,
    inputs: Dict[str, Any],
    frozen: Mapping[str, Any],
    doc_key: str,
    task: str,
    action: str,
    controller: Any,
    signal_collector: Any,
) -> Tuple[Any, Dict[str, Any]]:
    audit = AuditCollector()
    config = LoopConfig(
        model_alias=frozen["model"],
        window=tuple(frozen["window"]),
        k=int(frozen["k"]),
        iteration_mode=frozen["iteration_mode"],
        strategy=frozen["strategy"],
        alpha=float(frozen["alpha"]),
        beta=float(frozen["beta"]),
        cache_strategy=frozen["cache_strategy"],
        decode_mode=frozen["decode_mode"],
        audit_collector=audit,
        controller=controller,
        controller_probe=(signal_collector.controller_probe if signal_collector else None),
        signal_collector=signal_collector,
    )
    handle = apply_loop_wrapper(model, config)
    try:
        endpoint = _endpoint_logits(model, inputs, bool(frozen["use_cache"]))
    finally:
        handle.restore()
    expected_calls = 1 if action == "never_loop" else 2
    observed_calls = int(audit.counts.get("operator_body_calls", 0))
    if observed_calls != expected_calls:
        raise RuntimeError("%s body-call mismatch" % action)
    signals = signal_collector.finalize(expected_calls) if signal_collector else None
    decision_reason = {
        "legacy_k2": "controller_none_legacy_k2",
        "always_loop": "always_loop_engineering_control",
        "never_loop": "never_loop_engineering_control",
    }[action]
    record = {
        "schema_version": 1,
        "doc_key": doc_key,
        "task": task,
        "action": action,
        "decision_reason": decision_reason,
        "k_used": expected_calls,
        "operator_body_calls": observed_calls,
        "endpoint": _endpoint_summary(torch, endpoint),
        "signals": signals,
        "audit": {
            "counts": dict(audit.counts),
            "controller_health_closed": audit.summary().get("controller_health_closed"),
            "cache_strategy": frozen["cache_strategy"],
            "wrapper_restored": True,
        },
    }
    validate_action_record(record)
    return endpoint, record


def _endpoint_logits(model: Any, inputs: Dict[str, Any], use_cache: bool) -> Any:
    outputs = model(**inputs, use_cache=use_cache)
    logits = outputs.logits[:, -1, :].detach()
    if not bool(logits.isfinite().all()):
        raise ValueError("endpoint logits contain NaN or Inf")
    return logits


def _endpoint_summary(torch: Any, endpoint: Any) -> Dict[str, Any]:
    cpu = endpoint.detach().contiguous().cpu()
    values = cpu.float()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(values).item()),
    }


def _compare(torch: Any, left: Any, right: Any, tolerance: Mapping[str, Any]) -> Dict[str, Any]:
    lhs = left.detach().float()
    rhs = right.detach().float()
    diff = (rhs - lhs).abs()
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])
    return {
        "rtol": rtol,
        "atol": atol,
        "max_abs": float(diff.max().item()),
        "max_rel": float((diff / lhs.abs().clamp_min(atol)).max().item()),
        "allclose": bool(torch.allclose(lhs, rhs, rtol=rtol, atol=atol)),
    }


def _window_classes(layers: Any, window: Tuple[int, int]) -> Dict[str, str]:
    return {
        str(index): layers[index].__class__.__module__ + "." + layers[index].__class__.__name__
        for index in range(window[0], window[1] + 1)
    }


def _forward_identity(model: Any) -> str:
    forward = getattr(model, "forward")
    function = getattr(forward, "__func__", forward)
    return getattr(function, "__module__", "") + "." + getattr(function, "__qualname__", repr(function))


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise RuntimeError("required distribution is missing: %s" % (names,))


if __name__ == "__main__":
    raise SystemExit(main())
