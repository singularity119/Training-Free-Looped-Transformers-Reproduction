"""Remote lm-eval entrypoint that can patch the HF model before evaluation."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
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
    )
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


if __name__ == "__main__":
    raise SystemExit(main())
