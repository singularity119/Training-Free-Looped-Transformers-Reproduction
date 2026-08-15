#!/usr/bin/env python3
"""Run the Phase 7 native zero-loop scalar producer.

The renderer is injected as a module function so MMLU prompt construction
stays in the audited lm-eval environment.  It returns ephemeral model inputs;
the producer never persists those inputs or their token ids.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TextIO

from tflt.loopscope.phase7_producer import admit_choice_surfaces, produce_record
from tflt.loopscope.phase7_schema import (
    DATASET_REPO,
    DATASET_SPLIT,
    FEWSHOT_SPLIT,
    FORMAL_BOOTSTRAP_SEED,
    METHOD_ID,
    METHOD_VERSION,
    NUM_FEWSHOT,
    RUNTIME_DTYPE,
    Phase7ContractError,
    candidate_domain,
    model_spec,
    scan_forbidden_fields,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce sanitized Phase 7 Base-model scalar trajectories from one native zero-loop forward."
    )
    parser.add_argument("--model-key", choices=("qwen25_3b", "llama32_3b", "gemma2_2b"), required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--revision", required=True, help="Resolved model revision from the audited local cache.")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--identity-manifest", help="Safe identity/subject JSON or JSONL; no prompt or target fields.")
    parser.add_argument("--renderer-entrypoint", help="module:function that returns ephemeral inputs for one identity.")
    parser.add_argument("--output", help="Write-once sanitized JSONL output path.")
    parser.add_argument("--max-identities", type=int, default=None)
    parser.add_argument("--cache-dir", default=None, help="Optional audited local HF cache root.")
    parser.add_argument("--dry-run", action="store_true", help="Print the future Gate B/C producer contract without loading a model.")
    parser.add_argument("--layer-count", type=int, default=None, help="Only used with --dry-run to print a candidate-domain table.")
    return parser


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise Phase7ContractError("identity manifest must be a non-empty list")
    result: List[Dict[str, Any]] = []
    seen = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise Phase7ContractError("identity manifest row must be an object")
        scan_forbidden_fields(value)
        if set(value) != {"canonical_identity", "subject", "task_name", "validation_index"}:
            raise Phase7ContractError("identity manifest row keys differ from the safe renderer contract")
        row = dict(value)
        identity = str(row.get("canonical_identity", ""))
        subject = str(row.get("subject", ""))
        task_name = str(row.get("task_name", ""))
        validation_index = row.get("validation_index")
        if not identity or not subject or identity in seen:
            raise Phase7ContractError("identity manifest has missing/duplicate identity")
        if not task_name.startswith("mmlu_") or isinstance(validation_index, bool) or not isinstance(validation_index, int) or validation_index < 0:
            raise Phase7ContractError("identity manifest has invalid renderer locator")
        seen.add(identity)
        result.append(row)
    return result


def _load_entrypoint(spec: str):
    if ":" not in spec:
        raise Phase7ContractError("renderer entrypoint must be module:function")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise Phase7ContractError("renderer entrypoint is not callable")
    return function


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        raise Phase7ContractError("BLOCK_WRITE_ONCE_VIOLATION: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _emit_progress(
    completed_records: int, total_records: int, *, stream: Optional[TextIO] = None
) -> None:
    if completed_records < 1 or total_records < 1 or completed_records > total_records:
        raise Phase7ContractError("progress counter is outside the active manifest")
    stream = sys.stderr if stream is None else stream
    stream.write(
        "event=phase7_progress completed_records=%d total_records=%d\n"
        % (completed_records, total_records)
    )
    stream.flush()


def _pretrained_load_kwargs(cache_dir: Optional[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return kwargs


def _require_cuda_device(torch: Any) -> Any:
    if not bool(torch.cuda.is_available()):
        raise Phase7ContractError("BLOCK_CUDA_UNAVAILABLE: runtime mode requires CUDA")
    device = torch.device("cuda:0")
    if str(getattr(device, "type", device)) != "cuda":
        raise Phase7ContractError("BLOCK_CUDA_BINDING: resolved device is not CUDA")
    return device


def _bind_model_to_device(model: Any, device: Any) -> Any:
    moved = model.to(device)
    model = moved if moved is not None else model
    parameters = tuple(model.parameters())
    if not parameters or any(str(parameter.device) != str(device) for parameter in parameters):
        raise Phase7ContractError("BLOCK_CUDA_BINDING: model parameters are not all on one CUDA device")
    return model


def _bind_tensor_inputs(torch: Any, inputs: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    bound: Dict[str, Any] = {}
    tensor_count = 0
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            tensor_count += 1
            if str(value.device) != str(device):
                raise Phase7ContractError("BLOCK_CUDA_BINDING: tensor input is not on the model device")
        bound[key] = value
    if tensor_count == 0:
        raise Phase7ContractError("BLOCK_CUDA_BINDING: renderer returned no tensor inputs")
    return bound


def dry_run(args: argparse.Namespace) -> int:
    payload: Dict[str, Any] = {
        "producer": "phase7_native_zero_loop_scalar_producer",
        "model_key": args.model_key,
        "model_repo": args.model_repo,
        "dataset_repo": DATASET_REPO,
        "split": DATASET_SPLIT,
        "num_fewshot": NUM_FEWSHOT,
        "fewshot_split": FEWSHOT_SPLIT,
        "runtime_dtype": RUNTIME_DTYPE,
        "batch_size": 1,
        "forward": "one_native_zero_loop_use_cache_false_per_identity",
        "choice_surfaces": [" A", " B", " C", " D"],
        "selector_method": "%s@%s" % (METHOD_ID, METHOD_VERSION),
        "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
        "forbidden_persistence": [
            "prompt_text",
            "target_gold_label_correctness",
            "test_split",
            "outcome",
            "raw_full_logits_or_probabilities",
            "hidden_tensors",
            "token_ids_or_input_ids",
            "loop_residual",
        ],
    }
    if args.layer_count is not None:
        payload["candidate_domain"] = candidate_domain(args.layer_count)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    if not args.identity_manifest or not args.renderer_entrypoint or not args.output:
        raise Phase7ContractError("runtime mode requires --identity-manifest, --renderer-entrypoint, and --output")
    spec = model_spec(args.model_key)
    if args.model_repo != spec["model_repo"]:
        raise Phase7ContractError("--model-repo differs from the frozen model_key")
    records = _load_manifest(Path(args.identity_manifest))
    if args.max_identities is not None:
        if args.max_identities < 1:
            raise Phase7ContractError("--max-identities must be positive")
        records = records[: args.max_identities]
    render = _load_entrypoint(args.renderer_entrypoint)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - remote-only dependency path.
        raise RuntimeError("Phase 7 runtime mode requires audited torch/transformers") from exc
    device = _require_cuda_device(torch)
    load_kwargs = _pretrained_load_kwargs(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_repo, revision=args.tokenizer_revision, **load_kwargs
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_repo,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        **load_kwargs,
    )
    model = _bind_model_to_device(model, device)
    model.eval()
    choice_admission = admit_choice_surfaces(tokenizer)
    output_rows: List[Dict[str, Any]] = []
    total_records = len(records)
    for completed_records, row in enumerate(records, start=1):
        rendered = render(row, tokenizer)
        if not isinstance(rendered, Mapping):
            raise Phase7ContractError("renderer must return a mapping")
        inputs = rendered.get("inputs")
        if not isinstance(inputs, Mapping):
            raise Phase7ContractError("renderer result lacks ephemeral inputs")
        inputs = _bind_tensor_inputs(torch, inputs, device)
        output_rows.append(
            produce_record(
                model,
                tokenizer,
                inputs,
                model_key=args.model_key,
                model_revision=args.revision,
                tokenizer_revision=args.tokenizer_revision,
                canonical_identity=str(row["canonical_identity"]),
                subject=str(row["subject"]),
                sequence_length=int(rendered["sequence_length"]),
                probe_index=int(rendered["probe_index"]),
                choice_admission=choice_admission,
            )
        )
        if completed_records % 25 == 0 or completed_records == total_records:
            _emit_progress(completed_records, total_records)
    _write_once(
        Path(args.output),
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output_rows),
    )
    print(json.dumps({"status": "PASS", "record_count": len(output_rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
