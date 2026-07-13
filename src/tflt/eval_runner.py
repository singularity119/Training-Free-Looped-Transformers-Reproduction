"""Remote lm-eval entrypoint that can patch the HF model before evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Optional

from tflt.config import LoopConfig
from tflt.models import resolve_model
from tflt.wrapper import apply_loop_wrapper
from tflt.loopscope.revisions import (
    load_tokenizer_with_resolved_commit,
    resolved_model_commit,
    resolved_tokenizer_commit,
    strict_revision_closure,
)
from tflt.loopscope.phase2_analysis import choice_output, required_full_cell_ids
from tflt.loopscope.phase2_schema import (
    FULL_IDENTITY_NAMESPACE,
    FULL_LM_EVAL_SCORE_SOURCE,
    atomic_write_new_json,
    file_sha256,
    make_full_final_output_envelope,
    stable_sample_identity,
    validate_identity_manifest,
    validate_source_provenance,
    verify_live_source_provenance,
    validate_phase2_card,
    validate_phase2_workspace_output_path,
    validate_producer_provenance,
    validate_protocol_cell,
)
from tflt.loopscope.schema import verify_manifest_sha256


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tflt.eval_runner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--window", default="12:15")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--iteration-mode", choices=["block", "layer"], default="block")
    parser.add_argument("--strategy", default="damped_euler")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--cache-strategy", choices=["first", "last", "none"], default="last")
    parser.add_argument("--decode-mode", choices=["bypass", "full", "first_n"], default="bypass")
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--phase2-final-output-manifest", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    phase2_final_output = None
    if hasattr(args, "phase2_final_output_manifest"):
        phase2_final_output = _load_phase2_final_output_manifest(
            Path(args.phase2_final_output_manifest), args
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_new_json(output_dir / "command_args.json", vars(args))
        atomic_write_new_json(output_dir / "env.json", _env_snapshot())
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "command_args.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "env.json").write_text(
            json.dumps(_env_snapshot(), indent=2, sort_keys=True), encoding="utf-8"
        )

    model_info = resolve_model(args.model)
    loop_config = _loop_config(args) if args.loop else None
    result = run_lm_eval(
        model_repo=model_info["repo_id"],
        tasks=args.tasks,
        output_dir=output_dir,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        dtype=args.dtype,
        loop_config=loop_config,
        revision=args.revision,
        exclusive_writes=phase2_final_output is not None,
    )
    if phase2_final_output is not None:
        atomic_write_new_json(
            output_dir / "results.json",
            json.loads(json.dumps(result, sort_keys=True, default=str)),
        )
        sidecar = _build_phase2_final_output_sidecar(
            result, phase2_final_output, output_dir=output_dir, args=args
        )
        atomic_write_new_json(output_dir / "phase2_final_outputs.json", sidecar)
    else:
        (output_dir / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    return 0


def run_lm_eval(
    model_repo: str,
    tasks: str,
    output_dir: Path,
    limit: Optional[int],
    num_fewshot: Optional[int],
    batch_size: str,
    dtype: str,
    loop_config: Optional[LoopConfig],
    revision: Optional[str] = None,
    exclusive_writes: bool = False,
) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
    except Exception as exc:
        raise RuntimeError(
            "lm-eval-harness is required on the remote environment. "
            "Install with: python -m pip install -e '.[eval]'"
        ) from exc

    load_kwargs = {"trust_remote_code": True}
    if revision:
        load_kwargs["revision"] = revision
    tokenizer, resolved_tokenizer_revision = load_tokenizer_with_resolved_commit(
        AutoTokenizer, model_repo, load_kwargs
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        torch_dtype=_torch_dtype(torch, dtype),
        **load_kwargs,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        batch_size=batch_size,
    )
    target = _underlying_hf_model(lm)
    if loop_config is not None:
        apply_loop_wrapper(target, loop_config)
    _write_model_revision(
        output_dir,
        target,
        tokenizer,
        model_repo,
        revision,
        resolved_tokenizer_revision=resolved_tokenizer_revision,
        exclusive=exclusive_writes,
    )

    task_manager = TaskManager()
    return evaluator.simple_evaluate(
        model=lm,
        tasks=[task.strip() for task in tasks.split(",") if task.strip()],
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        log_samples=True,
        task_manager=task_manager,
    )


def _loop_config(args: argparse.Namespace) -> LoopConfig:
    return LoopConfig.from_window_string(
        model_alias=args.model,
        window=args.window,
        k=args.k,
        iteration_mode=args.iteration_mode,
        strategy=args.strategy,
        alpha=args.alpha,
        beta=args.beta,
        cache_strategy=args.cache_strategy,
        decode_mode=args.decode_mode,
        first_n=args.first_n,
    )


def _underlying_hf_model(lm: Any) -> Any:
    for attr in ("model", "_model"):
        model = getattr(lm, attr, None)
        if model is not None:
            return model
    raise TypeError("could not locate underlying Hugging Face model inside lm-eval HFLM")


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    mapping = {
        "auto": "auto",
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    return mapping.get(str(dtype).lower(), dtype)


def _write_model_revision(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    repo_id: str,
    manifest_revision: Optional[str] = None,
    resolved_tokenizer_revision: Optional[str] = None,
    exclusive: bool = False,
) -> None:
    cfg = getattr(model, "config", None)
    closure = (
        strict_revision_closure(model, resolved_tokenizer_revision, manifest_revision)
        if manifest_revision
        else None
    )
    model_commit = (
        closure["model_commit"] if closure is not None else resolved_model_commit(model)
    )
    tokenizer_commit = (
        closure["tokenizer_commit"]
        if closure is not None
        else resolved_tokenizer_commit(tokenizer)
    )
    payload = {
        "schema_version": "loopscope.model-tokenizer-revision.v1",
        "repo_id": repo_id,
        "architectures": getattr(cfg, "architectures", None),
        "model_type": getattr(cfg, "model_type", None),
        "commit_hash": model_commit,
        "model_commit": model_commit,
        "tokenizer_commit": tokenizer_commit,
        "manifest_commit": closure["manifest_commit"] if closure is not None else None,
        "match": closure["match"] if closure is not None else None,
        "transformers_version": _module_version("transformers"),
        "lm_eval_version": _module_version("lm_eval"),
    }
    if exclusive:
        atomic_write_new_json(
            output_dir / "model_revision.json",
            json.loads(json.dumps(payload, sort_keys=True, default=str)),
        )
    else:
        (output_dir / "model_revision.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )


def _module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _env_snapshot() -> dict:
    keys = [
        "HF_ENDPOINT",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_DISABLE_XET",
        "UV_CACHE_DIR",
        "VIRTUAL_ENV",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _load_phase2_final_output_manifest(path: Path, args: argparse.Namespace) -> dict:
    """Load the explicit final-output-only adapter request.

    Auto batching is intentionally irrelevant: the adapter consumes completed
    evaluator logged samples and performs an unordered identity join after the
    evaluator returns.
    """

    request = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "artifact_kind", "card", "identity_manifest", "cell",
        "attempt_manifest", "receipt_manifest", "identity_join", "score_source",
        "output_filename", "manifest_sha256",
    }
    if not isinstance(request, dict) or set(request) != expected_keys:
        raise ValueError("Phase 2 final-output adapter manifest has an exact-key mismatch")
    verify_manifest_sha256(request)
    if request["schema_version"] != "loopscope.phase2-final-output-adapter-manifest.v2":
        raise ValueError("unsupported Phase 2 final-output adapter manifest")
    if request["artifact_kind"] != "full_final_output_adapter_request":
        raise ValueError("adapter manifest artifact kind differs")
    if request["identity_join"] != "unordered_exact_task_doc_id_doc_hash_to_canonical_manifest":
        raise ValueError("adapter manifest lacks the frozen unordered exact join")
    if request["score_source"] != FULL_LM_EVAL_SCORE_SOURCE:
        raise ValueError("adapter score source differs from lm-eval acc,none")
    if request["output_filename"] != "phase2_final_outputs.json":
        raise ValueError("adapter output filename differs from the write-once contract")
    base = path.parent
    card, card_path = _load_hashed_manifest_ref(request["card"], base, "adapter card")
    identity_manifest, identity_path = _load_hashed_manifest_ref(
        request["identity_manifest"], base, "adapter identity manifest"
    )
    attempt, attempt_path = _load_hashed_manifest_ref(
        request["attempt_manifest"], base, "adapter attempt manifest"
    )
    receipt, receipt_path = _load_hashed_manifest_ref(
        request["receipt_manifest"], base, "adapter receipt manifest"
    )
    validate_phase2_card(card)
    validate_phase2_workspace_output_path(
        args.output_dir, card, context="Phase 2 full output directory"
    )
    validate_identity_manifest(identity_manifest, card)
    if identity_manifest["identity_namespace"] != FULL_IDENTITY_NAMESPACE:
        raise ValueError("final-output adapter requires the full 14,042 identity namespace")
    validate_source_provenance(
        identity_manifest["source_provenance"],
        FULL_IDENTITY_NAMESPACE,
        require_live=True,
    )
    verify_live_source_provenance(
        identity_manifest["source_provenance"],
        FULL_IDENTITY_NAMESPACE,
        identity_manifest["ordered_sample_identity"],
    )
    validate_protocol_cell(request["cell"], allow_baseline=True)
    if request["cell"]["cell_id"] not in required_full_cell_ids(card):
        raise ValueError("adapter cell is outside the frozen 12-cell full evidence set")
    _validate_phase2_adapter_argv(args, card, request["cell"])
    _validate_adapter_control_manifests(
        attempt, receipt, card=card, cell=request["cell"], args=args
    )
    return {
        "request": request,
        "request_path": path.resolve(),
        "card": card,
        "card_path": card_path,
        "identity_manifest": identity_manifest,
        "identity_manifest_path": identity_path,
        "attempt": attempt,
        "attempt_path": attempt_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


def _build_phase2_final_output_sidecar(
    result: Any,
    context: dict,
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    request = context["request"]
    card = context["card"]
    identity_manifest = context["identity_manifest"]
    _validate_phase2_adapter_argv(args, card, request["cell"])
    actual = _verify_phase2_adapter_actual_files(
        output_dir=Path(output_dir), args=args, result=result, context=context
    )
    rows = _join_phase2_logged_samples(result, identity_manifest["ordered_sample_identity"])
    producer = {
        "producer_kind": "lm_eval_logged_samples_adapter",
        "attempt_manifest_sha256": context["attempt"]["manifest_sha256"],
        "receipt_manifest_sha256": context["receipt"]["manifest_sha256"],
        "command_sha256": actual["command_sha256"],
        "environment_sha256": actual["environment_sha256"],
        "revision_report_sha256": actual["revision_report_sha256"],
        "results_sha256": actual["results_sha256"],
    }
    validate_producer_provenance(
        producer, allowed_kinds={"lm_eval_logged_samples_adapter"}
    )
    return make_full_final_output_envelope(
        card,
        identity_manifest,
        cell=request["cell"],
        samples=rows,
        producer=producer,
    )


def _validate_phase2_adapter_argv(
    args: argparse.Namespace, card: dict, cell: dict
) -> None:
    """Bind the request cell to active scientific argv before model loading."""

    validate_phase2_card(card)
    validate_protocol_cell(cell, allow_baseline=True)
    required = {
        "model": "qwen3-1.7b-base",
        "revision": card["science"]["revision"],
        "tasks": card["science"]["task"],
        "limit": None,
        "num_fewshot": card["science"]["num_fewshot"],
        "dtype": card["science"]["dtype"],
    }
    for key, expected in required.items():
        if getattr(args, key, None) != expected:
            raise ValueError("Phase 2 adapter argv %s differs from the card" % key)
    if cell["protocol"] == "baseline_no_loop":
        if getattr(args, "loop", None) is not False:
            raise ValueError("baseline adapter cell requires --loop to be absent")
        return
    if getattr(args, "loop", None) is not True:
        raise ValueError("loop adapter cell requires --loop")
    active = {
        "window": cell["window"],
        "k": cell["k"],
        "alpha": cell["alpha"],
        "iteration_mode": card["science"]["iteration_mode"],
        "strategy": card["science"]["strategy"],
        "beta": card["science"]["beta"],
        "cache_strategy": card["science"]["cache_strategy"],
        "decode_mode": card["science"]["decode_mode"],
        "first_n": None,
    }
    for key, expected in active.items():
        if getattr(args, key, None) != expected:
            raise ValueError("Phase 2 loop argv %s differs from the requested cell" % key)


def _load_hashed_manifest_ref(ref: Any, base: Path, context: str) -> tuple:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("%s must be an exact path+hash reference" % context)
    if not isinstance(ref["path"], str) or not ref["path"].strip():
        raise ValueError("%s path must be non-empty" % context)
    if not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64:
        raise ValueError("%s hash must be SHA256" % context)
    resolved = _resolve_manifest_path(base, ref["path"]).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("%s must contain a JSON object" % context)
    verify_manifest_sha256(payload)
    if payload.get("manifest_sha256") != ref["sha256"]:
        raise ValueError("%s hash differs from the loaded manifest" % context)
    return payload, resolved


def _validate_adapter_control_manifests(
    attempt: dict, receipt: dict, *, card: dict, cell: dict, args: argparse.Namespace
) -> None:
    attempt_keys = {
        "schema_version", "artifact_kind", "card_manifest_sha256", "cell_id",
        "revision", "executor_thread_id", "created_at_utc", "manifest_sha256",
        "output_root",
    }
    receipt_keys = {
        "schema_version", "artifact_kind", "card_manifest_sha256", "cell_id",
        "revision", "attempt_manifest_sha256", "executor_thread_id",
        "created_at_utc", "manifest_sha256", "output_root",
    }
    if set(attempt) != attempt_keys or set(receipt) != receipt_keys:
        raise ValueError("adapter attempt/receipt exact-key contract differs")
    if (
        attempt["schema_version"] != "loopscope.phase2-full-attempt.v1"
        or attempt["artifact_kind"] != "full_final_output_attempt"
        or receipt["schema_version"] != "loopscope.phase2-full-receipt.v1"
        or receipt["artifact_kind"] != "full_final_output_execution_receipt"
    ):
        raise ValueError("unsupported adapter attempt/receipt schema")
    expected = (card["manifest_sha256"], cell["cell_id"], card["science"]["revision"])
    if (
        attempt["card_manifest_sha256"], attempt["cell_id"], attempt["revision"]
    ) != expected or (
        receipt["card_manifest_sha256"], receipt["cell_id"], receipt["revision"]
    ) != expected:
        raise ValueError("adapter attempt/receipt differs from card/cell/revision")
    if receipt["attempt_manifest_sha256"] != attempt["manifest_sha256"]:
        raise ValueError("adapter receipt is bound to a different attempt")
    if attempt["executor_thread_id"] != receipt["executor_thread_id"]:
        raise ValueError("adapter attempt/receipt executor differs")
    expected_root = str(
        validate_phase2_workspace_output_path(
            args.output_dir, card, context="Phase 2 full output directory"
        )
    )
    if attempt["output_root"] != expected_root or receipt["output_root"] != expected_root:
        raise ValueError("adapter attempt/receipt output_root differs from actual argv")
    for packet in (attempt, receipt):
        if not isinstance(packet["executor_thread_id"], str) or not packet["executor_thread_id"].strip():
            raise ValueError("adapter executor thread must be non-empty")
        _parse_utc(packet["created_at_utc"])
    if _parse_utc(receipt["created_at_utc"]) < _parse_utc(attempt["created_at_utc"]):
        raise ValueError("adapter receipt predates its attempt")


def _verify_phase2_adapter_actual_files(
    *, output_dir: Path, args: argparse.Namespace, result: Any, context: dict
) -> dict:
    resolved_output = validate_phase2_workspace_output_path(
        output_dir, context["card"], context="Phase 2 full artifact root"
    )
    if resolved_output != Path(output_dir).resolve() or Path(args.output_dir).resolve() != resolved_output:
        raise ValueError("Phase 2 full artifact root differs from actual --output-dir")
    command_path = output_dir / "command_args.json"
    environment_path = output_dir / "env.json"
    revision_path = output_dir / "model_revision.json"
    results_path = output_dir / "results.json"
    command = json.loads(command_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    stored_results = json.loads(results_path.read_text(encoding="utf-8"))
    if command != vars(args):
        raise ValueError("write-once command_args.json differs from actual parsed argv")
    if environment != _env_snapshot():
        raise ValueError("write-once env.json differs from the actual producer environment")
    expected_results = json.loads(json.dumps(result, sort_keys=True, default=str))
    if stored_results != expected_results:
        raise ValueError("write-once results.json differs from evaluator return value")
    expected_revision = context["card"]["science"]["revision"]
    if (
        revision.get("repo_id") != context["card"]["science"]["model"]
        or revision.get("model_commit") != expected_revision
        or revision.get("tokenizer_commit") != expected_revision
        or revision.get("manifest_commit") != expected_revision
        or revision.get("match") is not True
    ):
        raise ValueError("actual model/tokenizer revision closure differs from the card")
    for name in ("attempt", "receipt"):
        loaded, resolved = _load_hashed_manifest_ref(
            context["request"]["%s_manifest" % name],
            context["request_path"].parent,
            "adapter %s manifest" % name,
        )
        if loaded != context[name] or resolved != context["%s_path" % name]:
            raise ValueError("adapter %s provenance changed during execution" % name)
    return {
        "command_sha256": file_sha256(command_path),
        "environment_sha256": file_sha256(environment_path),
        "revision_report_sha256": file_sha256(revision_path),
        "results_sha256": file_sha256(results_path),
    }


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("adapter timestamp must be real UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("adapter timestamp must be real UTC") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("adapter timestamp must be UTC")
    return parsed


def _join_phase2_logged_samples(
    result: Any, expected_identities: list
) -> list:
    """Exact-join arbitrary-order evaluator samples and restore canonical order."""

    samples_by_task = result.get("samples") if isinstance(result, dict) else None
    if not isinstance(samples_by_task, dict):
        raise ValueError("lm-eval result does not expose logged samples for Phase 2 final output")
    expected = [
        stable_sample_identity(item.get("task"), item.get("doc_id"), item.get("doc_hash"))
        for item in expected_identities
    ]
    expected_keys = {
        (item["task"], item["doc_id"], item["doc_hash"]) for item in expected
    }
    if len(expected_keys) != len(expected):
        raise ValueError("canonical final-output identities contain duplicates")
    observed = {}
    for task, task_samples in samples_by_task.items():
        if not isinstance(task, str) or not isinstance(task_samples, list):
            raise ValueError("lm-eval samples mapping is malformed")
        for sample in task_samples:
            if not isinstance(sample, dict):
                raise ValueError("lm-eval logged sample must be an object")
            if "task" in sample and str(sample["task"]) != task:
                raise ValueError("logged sample task disagrees with its evaluator task bucket")
            try:
                sample_identity = stable_sample_identity(task, sample["doc_id"], sample["doc_hash"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("lm-eval sample lacks stable task/doc_id/doc_hash identity") from exc
            key = (
                sample_identity["task"], sample_identity["doc_id"], sample_identity["doc_hash"]
            )
            if key in observed:
                raise ValueError("lm-eval logged samples contain duplicate stable identity")
            scores = _extract_four_raw_choice_scores(sample)
            gold_index = sample.get("target")
            if isinstance(gold_index, str) and gold_index in ("A", "B", "C", "D"):
                gold_index = ("A", "B", "C", "D").index(gold_index)
            if isinstance(gold_index, bool) or not isinstance(gold_index, int) or not 0 <= gold_index < 4:
                raise ValueError("lm-eval sample lacks a frozen A-D gold index")
            observed[key] = choice_output(
                scores,
                identity=sample_identity,
                score_source=FULL_LM_EVAL_SCORE_SOURCE,
                gold_index=gold_index,
                evaluator_acc=_extract_sample_acc_none(sample),
            )
    if set(observed) != expected_keys:
        missing = len(expected_keys - set(observed))
        extra = len(set(observed) - expected_keys)
        raise ValueError("lm-eval exact identity join is incomplete; missing=%d extra=%d" % (missing, extra))
    return [observed[(item["task"], item["doc_id"], item["doc_hash"])] for item in expected]


def _resolve_manifest_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _extract_four_raw_choice_scores(sample: dict) -> list:
    values = sample.get("filtered_resps")
    if isinstance(values, list) and len(values) == 4 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in values
    ):
        return [float(item) for item in values]
    values = sample.get("resps")
    scores = []
    if isinstance(values, list) and len(values) == 4:
        for response in values:
            current = response
            while isinstance(current, (list, tuple)) and len(current) == 1:
                current = current[0]
            if isinstance(current, (list, tuple)) and current:
                current = current[0]
            if not isinstance(current, (int, float)) or isinstance(current, bool):
                break
            scores.append(float(current))
    if len(scores) != 4:
        raise ValueError("lm-eval sample lacks exactly four raw choice log-likelihood scores")
    return scores


def _extract_sample_acc_none(sample: dict) -> bool:
    metrics = sample.get("metrics")
    if isinstance(metrics, dict) and "acc,none" in metrics:
        value = metrics["acc,none"]
    else:
        value = sample.get("acc,none")
    if value not in (0, 1, False, True):
        raise ValueError("lm-eval sample lacks binary acc,none for correctness closure")
    return bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
