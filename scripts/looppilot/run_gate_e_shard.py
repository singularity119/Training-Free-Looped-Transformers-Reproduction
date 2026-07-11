#!/usr/bin/env python3
"""Run one frozen Gate E shard: baseline A, baseline B, then AlwaysLoop signals."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tflt.audit import AuditCollector
from tflt.config import LoopConfig
from tflt.looppilot.collector import TensorSignalCollector
from tflt.looppilot.controller import AlwaysLoop
from tflt.looppilot.full import GATE_E_CANDIDATE_FIELDS, validate_gate_e_config, verify_three_arm_rows
from tflt.looppilot.limit import build_decision_id
from tflt.looppilot.probe import build_doc_identity, build_question_mask, sha256_bytes
from tflt.looppilot.revisions import build_revision_closure, closure_dict
from tflt.looppilot.schema import write_json_once, write_jsonl_once
from tflt.wrapper import _find_layer_owner, apply_loop_wrapper


class _SignalRouter:
    def __init__(self) -> None:
        self.active: Optional[TensorSignalCollector] = None

    def bind(self, value: TensorSignalCollector) -> None:
        if self.active is not None:
            raise RuntimeError("Gate E signal router already bound")
        self.active = value

    def clear(self) -> None:
        self.active = None

    def observe_operator_call(self, x: Any, y: Any, layer_updates: Any) -> None:
        if self.active is None:
            raise RuntimeError("Gate E signal router has no request")
        self.active.observe_operator_call(x, y, layer_updates)

    def controller_probe(self, x0: Any, y0: Any) -> Dict[str, float]:
        if self.active is None:
            raise RuntimeError("Gate E controller probe has no request")
        full = self.active.controller_probe(x0, y0)
        return {name: float(full[name]) for name in GATE_E_CANDIDATE_FIELDS}


class _AuditRouter:
    def __init__(self) -> None:
        self.active: Optional[AuditCollector] = None

    def bind(self, value: AuditCollector) -> None:
        if self.active is not None:
            raise RuntimeError("Gate E audit router already bound")
        self.active = value

    def clear(self) -> None:
        self.active = None

    def record(self, event: str, payload: Dict[str, Any]) -> None:
        if self.active is None:
            raise RuntimeError("Gate E audit router has no request")
        self.active.record(event, payload)

    def record_tensor_diff(self, name: str, before: Any, after: Any) -> None:
        if self.active is None:
            raise RuntimeError("Gate E audit router has no request")
        self.active.record_tensor_diff(name, before, after)


class _MetadataAdapter:
    def __init__(self, documents: List[Mapping[str, Any]], task_objects: Mapping[str, Any]) -> None:
        self.task_objects = task_objects
        self.documents = {(row["task"], int(row["doc_index"])): row for row in documents}
        if len(self.documents) != len(documents) or not documents:
            raise RuntimeError("Gate E shard metadata must contain unique documents")

    def prepare(self, request: Any, lm: Any) -> Dict[str, Any]:
        task_name = getattr(request, "task_name", None)
        doc_id = getattr(request, "doc_id", None)
        choice_index = getattr(request, "idx", None)
        if not isinstance(task_name, str) or not isinstance(doc_id, int) or choice_index not in range(4):
            raise RuntimeError("Gate E lm-eval request metadata is incomplete")
        frozen = self.documents.get((task_name, doc_id))
        if frozen is None or not isinstance(request.doc, Mapping):
            raise RuntimeError("Gate E request is outside frozen shard")
        task = self.task_objects[task_name]
        current_text = task.doc_to_text(request.doc)
        context, continuation = request.args
        if not isinstance(context, str) or not isinstance(continuation, str):
            raise RuntimeError("Gate E requires plain-text loglikelihood")
        if sha256_bytes(context.encode()) != frozen["context_hash"] or sha256_bytes(current_text.encode()) != frozen["current_text_hash"]:
            raise RuntimeError("Gate E runtime renderer differs from preflight")
        identity = build_doc_identity(task_name, frozen["task_version"], frozen["renderer_hash"], frozen["occurrence_id"], request.doc)
        if identity["doc_key"] != frozen["doc_key"]:
            raise RuntimeError("Gate E runtime document identity differs from preflight")
        context_enc, continuation_enc = lm._encode_pair(context, continuation)
        if len(continuation_enc) != 1 or len(context_enc) > int(lm.max_length):
            raise RuntimeError("Gate E choice token or context length contract failed")
        encoded = lm.tokenizer(context, add_special_tokens=bool(lm.add_bos_token), return_attention_mask=True, return_offsets_mapping=True)
        if list(encoded["input_ids"]) != list(context_enc):
            raise RuntimeError("Gate E offset tokenizer differs from HFLM encoding")
        mask = build_question_mask(context, current_text, encoded["offset_mapping"], encoded["attention_mask"])
        if sha256_bytes(bytes(int(value) for value in mask)) != frozen["mask_hash"]:
            raise RuntimeError("Gate E runtime question mask differs from preflight")
        return {"frozen": frozen, "choice_index": choice_index, "mask": mask, "prompt_length": len(context_enc)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/looppilot/gate_e_full.json")
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.shard_id not in range(8):
        raise ValueError("Gate E shard id must be 0..7")
    config_path = Path(args.config).resolve()
    config = _json(config_path)
    validate_gate_e_config(config)
    preflight = Path(args.preflight_dir).resolve()
    manifest = _json(preflight / "preflight_manifest.json")
    if manifest.get("config_hash") != _sha(config_path) or manifest.get("leaf_task_count") != 57:
        raise RuntimeError("Gate E preflight/config closure failed")
    shard_rows = _json(preflight / "shard_manifest.json")["shards"]
    shard = next((row for row in shard_rows if int(row["shard_id"]) == args.shard_id), None)
    if not isinstance(shard, Mapping):
        raise RuntimeError("Gate E shard is absent from manifest")
    all_documents = _jsonl(preflight / "document_manifest.jsonl")
    documents = [row for row in all_documents if row["task"] in shard["tasks"]]
    if len(documents) != int(shard["doc_count"]):
        raise RuntimeError("Gate E shard document count differs from manifest")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    torch, AutoModelForCausalLM, HFLM, evaluator, TaskManager, lm_tasks, snapshot_download = _imports()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Gate E shard requires exactly one visible CUDA GPU")
    revision = config["model_revision"]
    snapshot = Path(snapshot_download(repo_id=config["model"], revision=revision, local_files_only=True)).resolve()
    if snapshot.name != revision:
        raise RuntimeError("Gate E resolved model revision mismatch")

    class GateEHFLM(HFLM):
        def __init__(self, *values: Any, **kwargs: Any) -> None:
            super().__init__(*values, **kwargs)
            self.adapter: Optional[_MetadataAdapter] = None
            self.signal_router: Optional[_SignalRouter] = None
            self.audit_router: Optional[_AuditRouter] = None
            self.signal_records: List[Dict[str, Any]] = []

        def loglikelihood(self, requests: List[Any], disable_tqdm: bool = False) -> List[Any]:
            if self.adapter is None:
                return super().loglikelihood(requests, disable_tqdm=disable_tqdm)
            if self.signal_router is None or self.audit_router is None:
                raise RuntimeError("Gate E signal routers are missing")
            answers: List[Any] = []
            for request in requests:
                binding = self.adapter.prepare(request, self)
                collector = TensorSignalCollector(binding["mask"], epsilon=1e-12)
                audit = AuditCollector()
                self.signal_router.bind(collector)
                self.audit_router.bind(audit)
                try:
                    answer = super().loglikelihood([request], disable_tqdm=True)
                    signals = collector.finalize(expected_calls=2)
                    audit_summary = audit.summary()
                finally:
                    self.signal_router.clear()
                    self.audit_router.clear()
                if len(answer) != 1 or int(audit.counts.get("operator_body_calls", 0)) != 2 or audit_summary.get("controller_health_closed") is not True:
                    raise RuntimeError("Gate E request body-call/controller closure failed")
                frozen = binding["frozen"]
                record = {
                    "schema_version": 1, "doc_key": frozen["doc_key"], "task": frozen["task"],
                    "task_version": frozen["task_version"], "renderer_hash": frozen["renderer_hash"],
                    "doc_hash": frozen["doc_hash"], "occurrence_id": frozen["occurrence_id"],
                    "choice_index": binding["choice_index"],
                    "decision_id": build_decision_id(frozen["doc_key"], "LOOP_K2", "always_loop_engineering_control"),
                    "action": "LOOP_K2", "decision_reason": "always_loop_engineering_control",
                    "controller_input_fields": list(GATE_E_CANDIDATE_FIELDS),
                    "question_token_count": int(signals["valid_token_count"]), "mask_hash": frozen["mask_hash"],
                    "context_token_count": binding["prompt_length"],
                    "r0_median": signals["r0_median"], "r0_p90": signals["r0_p90"], "r0_max": signals["r0_max"],
                    "c0": signals["c0"], "n0_median": signals["n0_median"], "n0_p90": signals["n0_p90"],
                    "n0_max": signals["n0_max"], "q1": signals["q1"], "residual_cosine": signals["residual_cosine"],
                    "operator_body_calls": int(audit.counts["operator_body_calls"]), "k_used": int(audit.counts["k_used_total"]),
                    "choice_continuation_read_by_controller": False, "labeled_output_read_by_controller": False,
                    "wrapper_restored": False,
                }
                self.signal_records.append(record)
                answers.extend(answer)
            return answers

    preloaded = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=revision, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, attn_implementation="eager",
    )
    preloaded.eval()
    preloaded.to("cuda")
    lm = GateEHFLM(pretrained=preloaded, revision=revision, tokenizer=config["model"], device="cuda", batch_size=1, logits_cache=False, trust_remote_code=True, use_fast_tokenizer=True)
    model = _model(lm)
    if not bool(getattr(lm.tokenizer, "is_fast", False)):
        raise RuntimeError("Gate E HFLM tokenizer must be fast")
    observed_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if observed_revision is not None and str(observed_revision) != revision:
        raise RuntimeError("Gate E loaded model revision mismatch")

    write_json_once(output / "resolved_model.json", {"schema_version": 1, "repo_id": config["model"], "commit_hash": revision, "snapshot_path": str(snapshot), "config_commit_hash": observed_revision})
    write_json_once(output / "resolved_tokenizer.json", {"schema_version": 1, "repo_id": config["model"], "commit_hash": revision, "snapshot_path": str(snapshot), "tokenizer_class": lm.tokenizer.__class__.__module__ + "." + lm.tokenizer.__class__.__name__, "is_fast": True})
    renderer_path = preflight / "renderer_manifest.json"
    closure = build_revision_closure(output / "resolved_model.json", output / "resolved_tokenizer.json", renderer_path, config_path, config["lm_eval_version"])
    write_json_once(output / "revision_closure.json", closure_dict(closure))

    task_names = list(shard["tasks"])
    task_manager = TaskManager()
    task_objects = lm_tasks.get_task_dict(task_names, task_manager=task_manager)
    for task in task_objects.values():
        task.set_fewshot_seed(int(config["fewshot_seed"]))
    adapter = _MetadataAdapter(documents, task_objects)

    baseline_a_result = _evaluate(evaluator, lm, TaskManager(), config, task_names)
    baseline_a = _normalize(baseline_a_result, documents, "baseline_a", closure.closure_id, task_names)
    write_json_once(output / "baseline_a/lm_eval_results.json", _safe(baseline_a_result))
    write_jsonl_once(output / "baseline_a_samples.jsonl", baseline_a)

    baseline_b_result = _evaluate(evaluator, lm, TaskManager(), config, task_names)
    baseline_b = _normalize(baseline_b_result, documents, "baseline_b", closure.closure_id, task_names)
    write_json_once(output / "baseline_b/lm_eval_results.json", _safe(baseline_b_result))
    write_jsonl_once(output / "baseline_b_samples.jsonl", baseline_b)

    signal_router, audit_router = _SignalRouter(), _AuditRouter()
    loop_config = LoopConfig(model_alias=config["model"], window=tuple(config["window"]), k=config["k"], iteration_mode=config["iteration_mode"], strategy=config["strategy"], alpha=config["alpha"], beta=config["beta"], cache_strategy=config["cache_strategy"], decode_mode=config["decode_mode"], audit_collector=audit_router, controller=AlwaysLoop(), controller_probe=signal_router.controller_probe, signal_collector=signal_router)
    owner, layer_attr = _find_layer_owner(model)
    original_classes = _classes(getattr(owner, layer_attr), tuple(config["window"]))
    original_forward = _forward(model)
    handle = apply_loop_wrapper(model, loop_config)
    lm.adapter, lm.signal_router, lm.audit_router = adapter, signal_router, audit_router
    try:
        loop_result = _evaluate(evaluator, lm, TaskManager(), config, task_names)
    finally:
        lm.adapter = lm.signal_router = lm.audit_router = None
        handle.restore()
    if _classes(getattr(owner, layer_attr), tuple(config["window"])) != original_classes or _forward(model) != original_forward or handle.active:
        raise RuntimeError("Gate E wrapper restore closure failed")
    for row in lm.signal_records:
        row["wrapper_restored"] = True
    loop = _normalize(loop_result, documents, "always_loop", closure.closure_id, task_names)
    decisions = _decisions(lm.signal_records, len(documents))
    report = verify_three_arm_rows(baseline_a, baseline_b, loop, lm.signal_records, decisions)
    write_json_once(output / "always_loop/lm_eval_results.json", _safe(loop_result))
    write_jsonl_once(output / "always_loop_samples.jsonl", loop)
    write_jsonl_once(output / "signal_records.jsonl", lm.signal_records)
    write_jsonl_once(output / "decisions.jsonl", decisions)
    write_json_once(output / "join_report.json", report)
    write_json_once(output / "runtime.json", {
        "schema_version": 1, "shard_id": args.shard_id, "python": platform.python_version(),
        "torch": torch.__version__, "transformers": _version("transformers"), "lm_eval": _version("lm_eval", "lm-eval"),
        "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "arm_order": ["baseline_a", "baseline_b", "always_loop"], "same_model_object": True, "batch_size": 1,
    })
    print(json.dumps({"shard": args.shard_id, "documents": len(documents), "signals": len(lm.signal_records)}, sort_keys=True))
    del lm, model, preloaded
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _evaluate(evaluator: Any, lm: Any, manager: Any, config: Mapping[str, Any], tasks: List[str]) -> Mapping[str, Any]:
    result = evaluator.simple_evaluate(model=lm, tasks=tasks, num_fewshot=5, batch_size=1, limit=None, bootstrap_iters=0, log_samples=True, apply_chat_template=False, fewshot_as_multiturn=False, task_manager=manager, random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)
    if not isinstance(result, Mapping):
        raise RuntimeError("Gate E lm-eval returned no rank-zero result")
    return result


def _normalize(result: Mapping[str, Any], documents: List[Mapping[str, Any]], arm: str, closure: str, tasks: List[str]) -> List[Dict[str, Any]]:
    frozen = {(row["task"], int(row["doc_index"])): row for row in documents}
    rows: List[Dict[str, Any]] = []
    for task_name in tasks:
        samples = result.get("samples", {}).get(task_name)
        expected = sum(1 for row in documents if row["task"] == task_name)
        if not isinstance(samples, list) or len(samples) != expected:
            raise RuntimeError("Gate E lm-eval sample count mismatch for %s" % task_name)
        for sample in samples:
            doc_index = int(sample["doc_id"])
            meta = frozen.get((task_name, doc_index))
            doc = sample.get("doc")
            if meta is None or not isinstance(doc, Mapping):
                raise RuntimeError("Gate E sample lies outside frozen documents")
            identity = build_doc_identity(task_name, meta["task_version"], meta["renderer_hash"], meta["occurrence_id"], doc)
            if identity["doc_key"] != meta["doc_key"]:
                raise RuntimeError("Gate E normalized sample identity mismatch")
            scores = [_score(value) for value in sample.get("filtered_resps", [])]
            if len(scores) != 4 or any(not math.isfinite(value) for value in scores):
                raise RuntimeError("Gate E MMLU sample requires four finite scores")
            gold = _gold(doc.get("answer"))
            prediction = max(range(4), key=lambda index: scores[index])
            rows.append({
                "schema_version": 1, "arm": arm, "doc_key": meta["doc_key"], "task": task_name,
                "task_version": meta["task_version"], "renderer_hash": meta["renderer_hash"], "doc_hash": meta["doc_hash"],
                "occurrence_id": meta["occurrence_id"], "revision_closure_id": closure,
                "subject": str(doc.get("subject") or task_name.removeprefix("mmlu_")), "gold": gold,
                "prediction": prediction, "choice_loglikelihoods": scores, "is_correct": prediction == gold,
                "prompt_length": int(meta["total_token_count"]),
            })
    if len(rows) != len(documents) or len({row["doc_key"] for row in rows}) != len(rows):
        raise RuntimeError("Gate E normalized sample set is not closed")
    return sorted(rows, key=lambda row: row["doc_key"])


def _decisions(records: List[Mapping[str, Any]], expected: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["doc_key"]].append(row)
    rows = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["choice_index"])
        if [row["choice_index"] for row in group] != [0, 1, 2, 3]:
            raise RuntimeError("Gate E decision lacks four choices")
        reference = group[0]
        for field in ("decision_id", "action", "decision_reason") + GATE_E_CANDIDATE_FIELDS:
            if any(row[field] != reference[field] for row in group[1:]):
                raise RuntimeError("Gate E choice requests disagree on %s" % field)
        rows.append({key_: reference[key_] for key_ in ("doc_key", "task", "task_version", "renderer_hash", "doc_hash", "occurrence_id", "decision_id", "action", "decision_reason", "controller_input_fields") + GATE_E_CANDIDATE_FIELDS} | {"schema_version": 1, "choice_indices": [0, 1, 2, 3]})
    if len(rows) != expected:
        raise RuntimeError("Gate E decision count mismatch")
    return rows


def _imports() -> Tuple[Any, ...]:
    try:
        import torch
        from huggingface_hub import snapshot_download
        from lm_eval import evaluator, tasks as lm_tasks
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError("Gate E requires cached torch, transformers, and lm-eval") from exc
    return torch, AutoModelForCausalLM, HFLM, evaluator, TaskManager, lm_tasks, snapshot_download


def _model(lm: Any) -> Any:
    for name in ("model", "_model"):
        value = getattr(lm, name, None)
        if value is not None:
            return value
    raise RuntimeError("HFLM model is unavailable")


def _classes(layers: Any, window: Tuple[int, int]) -> Dict[str, str]:
    return {str(i): layers[i].__class__.__module__ + "." + layers[i].__class__.__name__ for i in range(window[0], window[1] + 1)}


def _forward(model: Any) -> str:
    value = getattr(model, "forward")
    fn = getattr(value, "__func__", value)
    return getattr(fn, "__module__", "") + "." + getattr(fn, "__qualname__", repr(fn))


def _score(value: Any) -> float:
    while isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError("empty lm-eval response")
        value = value[0]
    return float(value)


def _gold(value: Any) -> int:
    if isinstance(value, bool):
        raise RuntimeError("boolean MMLU gold is invalid")
    if isinstance(value, int) and value in range(4):
        return value
    if isinstance(value, str) and value.strip().upper() in "ABCD":
        return "ABCD".index(value.strip().upper())
    raise RuntimeError("invalid MMLU gold")


def _safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> List[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise RuntimeError("required distribution missing")


if __name__ == "__main__":
    raise SystemExit(main())
