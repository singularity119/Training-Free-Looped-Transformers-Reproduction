#!/usr/bin/env python3
"""Run the frozen baseline and AlwaysLoop Gate D lm-eval limit arms."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tflt.audit import AuditCollector
from tflt.config import LoopConfig
from tflt.looppilot.collector import TensorSignalCollector
from tflt.looppilot.controller import AlwaysLoop
from tflt.looppilot.limit import (
    GATE_D_CONTROLLER_INPUT_FIELDS,
    GATE_D_TASKS,
    build_decision_id,
    validate_gate_d_config,
)
from tflt.looppilot.probe import build_doc_identity, build_question_mask, sha256_bytes
from tflt.looppilot.revisions import build_revision_closure, closure_dict
from tflt.looppilot.schema import write_json_once, write_jsonl_once
from tflt.wrapper import _find_layer_owner, apply_loop_wrapper


class _SignalRouter:
    def __init__(self) -> None:
        self.active: Optional[TensorSignalCollector] = None

    def bind(self, collector: TensorSignalCollector) -> None:
        if self.active is not None:
            raise RuntimeError("Gate D signal router is already bound")
        self.active = collector

    def clear(self) -> None:
        self.active = None

    def observe_operator_call(self, x: Any, y: Any, layer_updates: Any) -> None:
        if self.active is None:
            raise RuntimeError("Gate D signal router has no active request")
        self.active.observe_operator_call(x, y, layer_updates)

    def controller_probe(self, x0: Any, y0: Any) -> Dict[str, float]:
        if self.active is None:
            raise RuntimeError("Gate D controller probe has no active request")
        full = self.active.controller_probe(x0, y0)
        return {name: float(full[name]) for name in GATE_D_CONTROLLER_INPUT_FIELDS}


class _AuditRouter:
    def __init__(self) -> None:
        self.active: Optional[AuditCollector] = None

    def bind(self, collector: AuditCollector) -> None:
        if self.active is not None:
            raise RuntimeError("Gate D audit router is already bound")
        self.active = collector

    def clear(self) -> None:
        self.active = None

    def record(self, event: str, payload: Dict[str, Any]) -> None:
        if self.active is None:
            raise RuntimeError("Gate D audit router has no active request")
        self.active.record(event, payload)

    def record_tensor_diff(self, name: str, before: Any, after: Any) -> None:
        if self.active is None:
            raise RuntimeError("Gate D audit router has no active request")
        self.active.record_tensor_diff(name, before, after)


class _MetadataAdapter:
    """Bind one lm-eval Instance to one frozen doc without exposing its continuation."""

    def __init__(
        self,
        config: Mapping[str, Any],
        preflight: Mapping[str, Any],
        task_objects: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.task_objects = task_objects
        self.documents = {
            (row["task"], int(row["doc_index"])): row
            for row in preflight.get("documents", [])
        }
        if len(self.documents) != 20:
            raise RuntimeError("Gate D metadata adapter requires 20 frozen documents")

    def prepare(self, request: Any, lm: Any) -> Dict[str, Any]:
        task_name = getattr(request, "task_name", None)
        doc_id = getattr(request, "doc_id", None)
        choice_index = getattr(request, "idx", None)
        if task_name not in GATE_D_TASKS or not isinstance(doc_id, int):
            raise RuntimeError("lm-eval request lacks authorized Gate D task/doc metadata")
        if not isinstance(choice_index, int) or choice_index not in range(4):
            raise RuntimeError("lm-eval request lacks authoritative choice index 0..3")
        frozen = self.documents.get((task_name, doc_id))
        if frozen is None:
            raise RuntimeError("lm-eval produced a request outside the frozen Gate D docs")
        if not isinstance(request.doc, Mapping):
            raise RuntimeError("lm-eval Gate D request doc must be a mapping")

        task = self.task_objects[task_name]
        current_text = task.doc_to_text(request.doc)
        context, continuation = request.args
        if not isinstance(context, str) or not isinstance(continuation, str):
            raise RuntimeError("Gate D requires plain-text loglikelihood requests")
        if sha256_bytes(context.encode("utf-8")) != frozen["context_hash"]:
            raise RuntimeError("lm-eval request context differs from frozen authoritative renderer")
        if sha256_bytes(current_text.encode("utf-8")) != frozen["current_text_hash"]:
            raise RuntimeError("lm-eval current document renderer differs from preflight")

        identity = build_doc_identity(
            task=task_name,
            task_version=frozen["task_version"],
            renderer_hash=frozen["renderer_hash"],
            occurrence_id=frozen["occurrence_id"] if "occurrence_id" in frozen else "test:%d" % doc_id,
            doc=request.doc,
        )
        if identity["doc_key"] != frozen["doc_key"]:
            raise RuntimeError("lm-eval request doc identity differs from Gate D preflight")

        context_enc, continuation_enc = lm._encode_pair(context, continuation)
        if len(continuation_enc) != 1:
            raise RuntimeError("Gate D requires each MMLU choice continuation to be one token")
        if len(context_enc) > int(lm.max_length):
            raise RuntimeError("Gate D context would be left-truncated; frozen mask is invalid")
        encoded = lm.tokenizer(
            context,
            add_special_tokens=bool(lm.add_bos_token),
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        if list(encoded["input_ids"]) != list(context_enc):
            raise RuntimeError("Gate D offset tokenizer differs from lm-eval context encoding")
        mask = build_question_mask(
            context,
            current_text,
            encoded["offset_mapping"],
            encoded["attention_mask"],
        )
        mask_hash = sha256_bytes(bytes(int(value) for value in mask))
        if mask_hash != frozen["mask_hash"]:
            raise RuntimeError("Gate D runtime question mask differs from frozen preflight")
        if len(mask) != len(context_enc):
            raise RuntimeError("Gate D question mask does not match model input length")
        return {
            "frozen": frozen,
            "choice_index": choice_index,
            "question_mask": mask,
            "mask_hash": mask_hash,
            "context_token_count": len(context_enc),
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/looppilot/gate_d_limit.json")
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    preflight_path = Path(args.preflight_manifest).resolve()
    config = _load_json(config_path)
    validate_gate_d_config(config)
    preflight = _load_json(preflight_path)
    if preflight.get("config_hash") != _sha256_file(config_path):
        raise RuntimeError("Gate D preflight config hash mismatch")
    if preflight.get("document_count") != 20:
        raise RuntimeError("Gate D preflight must contain exactly 20 documents")
    if preflight.get("gate_c_link", {}).get("results_manifest_sha256") != config[
        "gate_c_results_manifest_sha256"
    ]:
        raise RuntimeError("Gate D preflight Gate C linkage mismatch")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch, HFLM, evaluator, TaskManager, lm_tasks, snapshot_download = _remote_imports()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Gate D requires exactly one visible CUDA GPU")

    revision = config["model_revision"]
    snapshot_path = Path(
        snapshot_download(
            repo_id=config["model"], revision=revision, local_files_only=True
        )
    ).resolve()
    if snapshot_path.name != revision:
        raise RuntimeError("Gate D resolved model snapshot mismatch")

    class GateDHFLM(HFLM):
        def __init__(self, *model_args: Any, **model_kwargs: Any) -> None:
            super().__init__(*model_args, **model_kwargs)
            self.gate_d_adapter: Optional[_MetadataAdapter] = None
            self.gate_d_signal_router: Optional[_SignalRouter] = None
            self.gate_d_audit_router: Optional[_AuditRouter] = None
            self.gate_d_records: List[Dict[str, Any]] = []

        def loglikelihood(self, requests: List[Any], disable_tqdm: bool = False) -> List[Any]:
            if self.gate_d_adapter is None:
                return super().loglikelihood(requests, disable_tqdm=disable_tqdm)
            if self.gate_d_signal_router is None or self.gate_d_audit_router is None:
                raise RuntimeError("Gate D lm adapter routers are not configured")
            answers: List[Any] = []
            for request in requests:
                binding = self.gate_d_adapter.prepare(request, self)
                signal_collector = TensorSignalCollector(
                    binding["question_mask"], epsilon=float(config["epsilon"])
                )
                audit = AuditCollector()
                self.gate_d_signal_router.bind(signal_collector)
                self.gate_d_audit_router.bind(audit)
                try:
                    answer = super().loglikelihood([request], disable_tqdm=True)
                    if len(answer) != 1:
                        raise RuntimeError("Gate D per-request lm-eval call returned wrong count")
                    signals = signal_collector.finalize(expected_calls=2)
                    audit_summary = audit.summary()
                finally:
                    self.gate_d_signal_router.clear()
                    self.gate_d_audit_router.clear()
                if int(audit.counts.get("operator_body_calls", 0)) != 2:
                    raise RuntimeError("Gate D request operator body-call closure failed")
                if audit_summary.get("controller_health_closed") is not True:
                    raise RuntimeError("Gate D request controller health did not close")
                frozen = binding["frozen"]
                decision_id = build_decision_id(
                    frozen["doc_key"], "LOOP_K2", "always_loop_engineering_control"
                )
                record = {
                    "schema_version": 1,
                    "doc_key": frozen["doc_key"],
                    "task": frozen["task"],
                    "task_version": frozen["task_version"],
                    "renderer_hash": frozen["renderer_hash"],
                    "doc_hash": frozen["doc_hash"],
                    "occurrence_id": "test:%d" % int(frozen["doc_index"]),
                    "choice_index": binding["choice_index"],
                    "decision_id": decision_id,
                    "action": "LOOP_K2",
                    "decision_reason": "always_loop_engineering_control",
                    "controller_input_fields": list(GATE_D_CONTROLLER_INPUT_FIELDS),
                    "question_token_count": int(signals["valid_token_count"]),
                    "mask_hash": binding["mask_hash"],
                    "context_token_count": binding["context_token_count"],
                    "r0_median": signals["r0_median"],
                    "r0_p90": signals["r0_p90"],
                    "r0_max": signals["r0_max"],
                    "c0": signals["c0"],
                    "n0_median": signals["n0_median"],
                    "n0_p90": signals["n0_p90"],
                    "n0_max": signals["n0_max"],
                    "q1": signals["q1"],
                    "residual_cosine": signals["residual_cosine"],
                    "operator_body_calls": int(audit.counts["operator_body_calls"]),
                    "k_used": int(audit.counts["k_used_total"]),
                    "controller_health_closed": True,
                    "choice_continuation_read_by_controller": False,
                    "labeled_output_read_by_controller": False,
                    "wrapper_restored": False,
                }
                self.gate_d_records.append(record)
                answers.extend(answer)
            return answers

    lm = GateDHFLM(
        pretrained=config["model"],
        revision=revision,
        tokenizer=config["model"],
        device="cuda",
        dtype=torch.float16,
        batch_size=1,
        logits_cache=False,
        trust_remote_code=True,
        use_fast_tokenizer=True,
    )
    model = _underlying_hf_model(lm)
    _assert_model_revision(model, revision)
    if not bool(getattr(lm.tokenizer, "is_fast", False)):
        raise RuntimeError("Gate D HFLM tokenizer must be fast")

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
        "tokenizer_class": lm.tokenizer.__class__.__module__
        + "."
        + lm.tokenizer.__class__.__name__,
        "is_fast": True,
    }
    write_json_once(output_dir / "resolved_model.json", model_artifact)
    write_json_once(output_dir / "resolved_tokenizer.json", tokenizer_artifact)
    write_json_once(output_dir / "renderer_manifest.json", preflight["renderer_manifest"])
    closure = build_revision_closure(
        output_dir / "resolved_model.json",
        output_dir / "resolved_tokenizer.json",
        output_dir / "renderer_manifest.json",
        config_path,
        config["lm_eval_version"],
    )
    write_json_once(output_dir / "revision_closure.json", closure_dict(closure))

    task_manager = TaskManager()
    task_objects = lm_tasks.get_task_dict(list(GATE_D_TASKS), task_manager=task_manager)
    for task in task_objects.values():
        task.set_fewshot_seed(int(config["fewshot_seed"]))
    adapter = _MetadataAdapter(config, preflight, task_objects)

    baseline_result = _simple_evaluate(evaluator, lm, task_manager, config)
    write_json_once(output_dir / "baseline/lm_eval_results.json", _json_safe(baseline_result))
    baseline_samples = _normalize_samples(
        baseline_result, preflight, "baseline", closure.closure_id
    )
    write_jsonl_once(output_dir / "baseline_samples.jsonl", baseline_samples)

    signal_router = _SignalRouter()
    audit_router = _AuditRouter()
    loop_config = LoopConfig(
        model_alias=config["model"],
        window=tuple(config["window"]),
        k=config["k"],
        iteration_mode=config["iteration_mode"],
        strategy=config["strategy"],
        alpha=config["alpha"],
        beta=config["beta"],
        cache_strategy=config["cache_strategy"],
        decode_mode=config["decode_mode"],
        audit_collector=audit_router,
        controller=AlwaysLoop(),
        controller_probe=signal_router.controller_probe,
        signal_collector=signal_router,
    )
    owner, layer_attr = _find_layer_owner(model)
    original_classes = _window_classes(getattr(owner, layer_attr), tuple(config["window"]))
    original_forward = _forward_identity(model)
    handle = apply_loop_wrapper(model, loop_config)
    lm.gate_d_adapter = adapter
    lm.gate_d_signal_router = signal_router
    lm.gate_d_audit_router = audit_router
    try:
        loop_result = _simple_evaluate(evaluator, lm, TaskManager(), config)
    finally:
        lm.gate_d_adapter = None
        lm.gate_d_signal_router = None
        lm.gate_d_audit_router = None
        handle.restore()
    if _window_classes(getattr(owner, layer_attr), tuple(config["window"])) != original_classes:
        raise RuntimeError("Gate D wrapper classes were not restored")
    if _forward_identity(model) != original_forward or handle.active:
        raise RuntimeError("Gate D model forward/handle was not restored")
    for row in lm.gate_d_records:
        row["wrapper_restored"] = True
    if len(lm.gate_d_records) != 80:
        raise RuntimeError("Gate D did not collect exactly 80 request signals")

    write_json_once(output_dir / "loop/lm_eval_results.json", _json_safe(loop_result))
    loop_samples = _normalize_samples(loop_result, preflight, "always_loop", closure.closure_id)
    write_jsonl_once(output_dir / "loop_samples.jsonl", loop_samples)
    write_jsonl_once(output_dir / "signal_records.jsonl", lm.gate_d_records)
    decisions = _aggregate_decisions(lm.gate_d_records)
    write_jsonl_once(output_dir / "decisions.jsonl", decisions)

    runtime = {
        "schema_version": 1,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _distribution_version("transformers"),
        "lm_eval": _distribution_version("lm_eval", "lm-eval"),
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "baseline_then_loop_same_allocation": True,
        "model_loaded_once": True,
        "threshold_selected": False,
        "ranking_computed": False,
        "limit_for_scientific_claim": False,
    }
    write_json_once(output_dir / "runtime.json", runtime)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "baseline_docs": len(baseline_samples),
                "loop_docs": len(loop_samples),
                "signals": len(lm.gate_d_records),
                "decisions": len(decisions),
            },
            sort_keys=True,
        )
    )
    del lm, model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _simple_evaluate(evaluator: Any, lm: Any, task_manager: Any, config: Mapping[str, Any]) -> Any:
    result = evaluator.simple_evaluate(
        model=lm,
        tasks=list(config["tasks"]),
        num_fewshot=int(config["num_fewshot"]),
        batch_size=1,
        limit=int(config["limit"]),
        bootstrap_iters=0,
        log_samples=True,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        task_manager=task_manager,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=int(config["fewshot_seed"]),
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("Gate D lm-eval returned no rank-zero result")
    return result


def _normalize_samples(
    result: Mapping[str, Any],
    preflight: Mapping[str, Any],
    arm: str,
    closure_id: str,
) -> List[Dict[str, Any]]:
    frozen = {
        (row["task"], int(row["doc_index"])): row for row in preflight["documents"]
    }
    rows: List[Dict[str, Any]] = []
    for task_name in GATE_D_TASKS:
        task_samples = result.get("samples", {}).get(task_name)
        if not isinstance(task_samples, list) or len(task_samples) != 5:
            raise RuntimeError("Gate D lm-eval sample count mismatch for %s" % task_name)
        for sample in task_samples:
            doc_index = int(sample["doc_id"])
            meta = frozen.get((task_name, doc_index))
            if meta is None:
                raise RuntimeError("lm-eval emitted a sample outside frozen Gate D docs")
            doc = sample.get("doc")
            if not isinstance(doc, Mapping):
                raise RuntimeError("Gate D lm-eval sample is missing its doc")
            identity = build_doc_identity(
                task_name,
                meta["task_version"],
                meta["renderer_hash"],
                "test:%d" % doc_index,
                doc,
            )
            if identity["doc_key"] != meta["doc_key"]:
                raise RuntimeError("Gate D lm-eval sample identity differs from preflight")
            scores = [_response_score(value) for value in sample.get("filtered_resps", [])]
            if len(scores) != 4 or any(not _finite(value) for value in scores):
                raise RuntimeError("Gate D MMLU sample must contain four finite scores")
            gold = _gold_index(doc.get("answer"))
            prediction = max(range(4), key=lambda index: scores[index])
            rows.append(
                {
                    "schema_version": 1,
                    "arm": arm,
                    "doc_key": meta["doc_key"],
                    "task": task_name,
                    "task_version": meta["task_version"],
                    "renderer_hash": meta["renderer_hash"],
                    "doc_hash": meta["doc_hash"],
                    "occurrence_id": "test:%d" % doc_index,
                    "revision_closure_id": closure_id,
                    "subject": str(doc.get("subject") or task_name[len("mmlu_") :]),
                    "choices": list(doc["choices"]),
                    "gold": gold,
                    "prediction": prediction,
                    "choice_loglikelihoods": scores,
                    "is_correct": prediction == gold,
                }
            )
    if len(rows) != 20 or len({row["doc_key"] for row in rows}) != 20:
        raise RuntimeError("Gate D normalized sample set is not 20 unique docs")
    return sorted(rows, key=lambda row: row["doc_key"])


def _aggregate_decisions(records: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["doc_key"]].append(row)
    decisions = []
    for doc_key in sorted(grouped):
        group = sorted(grouped[doc_key], key=lambda row: int(row["choice_index"]))
        if [row["choice_index"] for row in group] != [0, 1, 2, 3]:
            raise RuntimeError("Gate D decision aggregation is missing choice requests")
        reference = group[0]
        for field in (
            "decision_id",
            "action",
            "decision_reason",
            "r0_median",
            "c0",
            "n0_median",
        ):
            if any(row[field] != reference[field] for row in group[1:]):
                raise RuntimeError("Gate D choice requests disagree on %s" % field)
        decisions.append(
            {
                "schema_version": 1,
                "doc_key": doc_key,
                "task": reference["task"],
                "task_version": reference["task_version"],
                "renderer_hash": reference["renderer_hash"],
                "doc_hash": reference["doc_hash"],
                "occurrence_id": reference["occurrence_id"],
                "decision_id": reference["decision_id"],
                "action": reference["action"],
                "decision_reason": reference["decision_reason"],
                "controller_input_fields": list(GATE_D_CONTROLLER_INPUT_FIELDS),
                "r0_median": reference["r0_median"],
                "c0": reference["c0"],
                "n0_median": reference["n0_median"],
                "choice_indices": [0, 1, 2, 3],
            }
        )
    if len(decisions) != 20:
        raise RuntimeError("Gate D did not aggregate exactly 20 doc decisions")
    return decisions


def _remote_imports() -> Tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from huggingface_hub import snapshot_download
        from lm_eval import evaluator, tasks as lm_tasks
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
    except Exception as exc:
        raise RuntimeError("Gate D requires cached torch, transformers, and lm-eval") from exc
    return torch, HFLM, evaluator, TaskManager, lm_tasks, snapshot_download


def _underlying_hf_model(lm: Any) -> Any:
    for attr in ("model", "_model"):
        model = getattr(lm, attr, None)
        if model is not None:
            return model
    raise TypeError("could not locate underlying Hugging Face model")


def _assert_model_revision(model: Any, revision: str) -> None:
    observed = getattr(getattr(model, "config", None), "_commit_hash", None)
    if observed is not None and str(observed) != revision:
        raise RuntimeError("loaded model config commit does not match frozen revision")


def _window_classes(layers: Any, window: Tuple[int, int]) -> Dict[str, str]:
    return {
        str(index): layers[index].__class__.__module__ + "." + layers[index].__class__.__name__
        for index in range(window[0], window[1] + 1)
    }


def _forward_identity(model: Any) -> str:
    forward = getattr(model, "forward")
    function = getattr(forward, "__func__", forward)
    return getattr(function, "__module__", "") + "." + getattr(
        function, "__qualname__", repr(function)
    )


def _response_score(value: Any) -> float:
    while isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError("empty lm-eval response")
        value = value[0]
    return float(value)


def _gold_index(value: Any) -> int:
    if isinstance(value, bool):
        raise RuntimeError("boolean MMLU gold is invalid")
    if isinstance(value, int) and value in range(4):
        return value
    if isinstance(value, str) and value.strip().upper() in ("A", "B", "C", "D"):
        return ord(value.strip().upper()) - ord("A")
    raise RuntimeError("MMLU gold answer is not a choice index")


def _finite(value: float) -> bool:
    import math

    return math.isfinite(float(value))


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
