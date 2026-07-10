"""Baseline all-layer raw-logit-lens probe for LoopScope."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.metrics import (
    choice_entropy,
    effective_rank_from_singular_values,
    kl_to_reference,
    summarize,
    top1_agreement,
)
from tflt.loopscope.schema import (
    PROBE_SCHEMA_VERSION,
    attach_manifest_sha256,
    ensure_new_directory,
    probe_summary,
    validate_probe_report,
    verify_manifest_sha256,
    write_new_json,
)
from tflt.models import resolve_model


class ProbeInputError(ValueError):
    """Input error with structured details suitable for failure artifacts."""

    def __init__(self, message: str, details: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


def add_probe_layers_args(parser: Any) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--choice-labels", default="A,B,C,D")
    parser.add_argument("--erank-max-vectors", type=int, default=1024)


def cmd_probe_layers(args: Any) -> int:
    output_dir = ensure_new_directory(Path(args.output_dir))
    try:
        write_new_json(output_dir / "command_args.json", serializable_args(args))
        write_new_json(output_dir / "env.json", environment_snapshot())
        report = run_layer_probe(args)
        validate_probe_report(report)
        write_new_json(output_dir / "probe_report.json", report)
        write_new_json(output_dir / "probe_summary.json", probe_summary(report))
    except Exception as exc:
        _write_failure(output_dir, exc)
        raise
    print(str(output_dir / "probe_report.json"))
    print(str(output_dir / "probe_summary.json"))
    return 0


def run_layer_probe(args: Any) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - remote-only dependency path.
        raise RuntimeError("probe-layers requires torch and transformers on HPC2") from exc

    if args.erank_max_vectors < 2:
        raise ProbeInputError("--erank-max-vectors must be at least 2")
    records = load_probe_records(Path(args.input_jsonl), args.max_examples)
    pool_metadata, pool_warnings = probe_pool_metadata(records, args.input_manifest)
    model_info = resolve_model(args.model)
    repo_id = model_info["repo_id"]
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.revision:
        load_kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(repo_id, **load_kwargs)
    choice_metadata, choice_ids, token_warnings = choice_tokenization(
        tokenizer, parse_choice_labels(args.choice_labels)
    )

    device = select_device(torch, args.device)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype(torch, args.dtype),
        **load_kwargs,
    )
    model.eval()
    model.to(device)
    final_norm = find_final_norm(model)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise TypeError("could not locate the model output embedding/lm_head")

    layer_count = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if layer_count < 1:
        raise ProbeInputError("model config does not expose a positive num_hidden_layers")

    aggregate: DefaultDict[int, DefaultDict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    erank_vectors: DefaultDict[int, List[Any]] = defaultdict(list)
    erank_sample_ids: DefaultDict[int, List[str]] = defaultdict(list)
    examples = []
    warnings = list(pool_warnings) + list(token_warnings)
    excluded_special = 0

    for record in records:
        encoded = tokenizer(record["text"], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in encoded.items()}
        answer_position = resolve_answer_position(inputs["attention_mask"], record)
        answer_token_id = int(inputs["input_ids"][0, answer_position].item())
        answer_is_special = answer_token_id in set(getattr(tokenizer, "all_special_ids", []))

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = tuple(outputs.hidden_states or ())
            layer_states = decoder_layer_states(hidden_states, layer_count)
            final_choice_logits = outputs.logits[0, answer_position, choice_ids].float()
            final_distribution = torch.softmax(final_choice_logits, dim=-1).cpu().tolist()
            example_layers = []
            for layer_index, hidden in enumerate(layer_states):
                answer_hidden = hidden[:, answer_position : answer_position + 1, :]
                lens_hidden = lens_space_hidden(
                    final_norm,
                    answer_hidden,
                    layer_index,
                    layer_count,
                )
                if layer_index == layer_count - 1:
                    choice_logits = outputs.logits[0, answer_position, choice_ids].float()
                else:
                    choice_logits = lm_head(lens_hidden)[0, 0, choice_ids].float()
                distribution = torch.softmax(choice_logits, dim=-1).cpu().tolist()
                entropy = choice_entropy(distribution)
                kl_value = kl_to_reference(distribution, final_distribution)
                agreement = top1_agreement(distribution, final_distribution)
                aggregate[layer_index]["choice_entropy"].append(entropy)
                aggregate[layer_index]["kl_to_final"].append(kl_value)
                aggregate[layer_index]["top1_to_final_agreement"].append(agreement)
                example_layers.append(
                    {
                        "layer_index": layer_index,
                        "choice_distribution": dict(zip(choice_metadata.keys(), distribution)),
                        "choice_entropy": entropy,
                        "kl_to_final": kl_value,
                        "top1_to_final_agreement": agreement,
                    }
                )
                has_erank_capacity = len(erank_vectors[layer_index]) < args.erank_max_vectors
                if not answer_is_special and has_erank_capacity:
                    vector = lens_hidden[0, 0].detach().float().cpu()
                    erank_vectors[layer_index].append(vector)
                    erank_sample_ids[layer_index].append(record["id"])
                del choice_logits, lens_hidden
                del answer_hidden

        if answer_is_special:
            excluded_special += 1
        examples.append(
            {
                "id": record["id"],
                "subject": record["subject"],
                "prompt_sha256": record["prompt_sha256"],
                "answer_position": answer_position,
                "answer_token_id": answer_token_id,
                "answer_token_is_special": answer_is_special,
                "final_choice_distribution": dict(zip(choice_metadata.keys(), final_distribution)),
                "layer_metrics": example_layers,
            }
        )
        del outputs, hidden_states, layer_states, hidden, final_choice_logits, inputs, encoded
        if device == "cuda":
            torch.cuda.empty_cache()

    layer_metrics = []
    for layer_index in range(layer_count):
        effective_rank = compute_effective_rank(torch, erank_vectors[layer_index])
        layer_metrics.append(
            {
                "layer_index": layer_index,
                "choice_entropy": summarize(aggregate[layer_index]["choice_entropy"]),
                "kl_to_final": summarize(aggregate[layer_index]["kl_to_final"]),
                "top1_to_final_agreement": summarize(
                    aggregate[layer_index]["top1_to_final_agreement"]
                ),
                "effective_rank": effective_rank,
                "effective_rank_sampling": {
                    "object": "answer-position hidden vectors across probe examples",
                    "representation_space": "final_norm raw-logit-lens space",
                    "special_answer_positions_excluded": excluded_special,
                    "unit_normalized": True,
                    "centered_across_vectors": True,
                    "max_vectors": args.erank_max_vectors,
                    "count": len(erank_vectors[layer_index]),
                    "sample_ids": erank_sample_ids[layer_index],
                },
            }
        )

    actual_model_revision = getattr(model.config, "_commit_hash", None) or args.revision
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "git": git_provenance(),
        "model": model_metadata(model, model_info, layer_count, actual_model_revision),
        "tokenizer": {
            "revision": tokenizer_revision(tokenizer, actual_model_revision),
            "choice_token_ids": choice_metadata,
        },
        "runtime": runtime_metadata(torch, device, args.dtype),
        "probe_pool": pool_metadata,
        "position_rule": "explicit answer_position or last non-padding token",
        "layer_metrics": layer_metrics,
        "window_metrics": [],
        "examples": examples,
        "warnings": warnings,
    }


def load_probe_records(path: Path, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    if max_examples is not None and max_examples < 1:
        raise ProbeInputError("--max-examples must be positive")
    records = []
    seen_ids = set()
    forbidden_gold_keys = {"answer", "label", "target", "gold", "gold_label"}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProbeInputError("invalid JSONL at line %d" % line_number) from exc
            if not isinstance(item, dict):
                raise ProbeInputError("probe record line %d must be an object" % line_number)
            required = ("id", "text", "source", "split", "subject")
            missing = [key for key in required if not item.get(key)]
            if missing:
                raise ProbeInputError(
                    "probe record line %d missing fields: %s" % (line_number, ", ".join(missing))
                )
            if "test" in str(item["split"]).lower():
                raise ProbeInputError("test split is forbidden for LoopScope probes")
            leaked = sorted(forbidden_gold_keys.intersection(item))
            if leaked:
                raise ProbeInputError("probe record contains forbidden gold fields: %s" % leaked)
            record_id = str(item["id"])
            if record_id in seen_ids:
                raise ProbeInputError("duplicate probe record id: %s" % record_id)
            seen_ids.add(record_id)
            text = str(item["text"])
            prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if item.get("prompt_sha256") not in (None, prompt_hash):
                raise ProbeInputError("prompt_sha256 mismatch for record %s" % record_id)
            record = dict(item)
            record["id"] = record_id
            record["text"] = text
            record["prompt_sha256"] = prompt_hash
            records.append(record)
            if max_examples is not None and len(records) >= max_examples:
                break
    if not records:
        raise ProbeInputError("probe input contains no records")
    return records


def probe_pool_metadata(
    records: Sequence[Mapping[str, Any]], manifest_path: Optional[str]
) -> Tuple[Dict[str, Any], List[str]]:
    warnings = []
    sources = sorted({str(item["source"]) for item in records})
    splits = sorted({str(item["split"]) for item in records})
    source_manifest_hash = None
    source_manifest_count = None
    seed = None
    if manifest_path:
        supplied = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        _validate_source_pool_manifest(supplied)
        verify_manifest_sha256(supplied)
        selected_records = [
            {"id": str(item["id"]), "prompt_sha256": str(item["prompt_sha256"])}
            for item in records
        ]
        supplied_records = [
            {"id": str(item["id"]), "prompt_sha256": str(item["prompt_sha256"])}
            for item in supplied["records"]
        ]
        if supplied_records[: len(selected_records)] != selected_records:
            raise ProbeInputError(
                "selected JSONL records do not match the supplied manifest id/hash prefix"
            )
        if sources != [str(supplied["source"])] or splits != [str(supplied["split"])]:
            raise ProbeInputError("input source/split disagrees with the supplied pool manifest")
        source_manifest_hash = supplied["manifest_sha256"]
        source_manifest_count = int(supplied["count"])
        seed = supplied.get("seed")
    else:
        warnings.append(
            "No external pool manifest supplied; manifest hash was derived from input records."
        )
    selected_records = [
        {"id": item["id"], "prompt_sha256": item["prompt_sha256"]} for item in records
    ]
    selected_manifest: Dict[str, Any] = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": ",".join(sources),
        "split": ",".join(splits),
        "count": len(records),
        "seed": seed,
        "sample_ids": [item["id"] for item in records],
        "records": selected_records,
        "source_manifest_sha256": source_manifest_hash,
    }
    attach_manifest_sha256(selected_manifest)
    selection_hash = selected_manifest["manifest_sha256"]
    return (
        {
            "source": ",".join(sources),
            "split": ",".join(splits),
            "count": len(records),
            "seed": seed,
            "manifest_sha256": selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "selected_subset_sha256": selection_hash,
            "source_manifest_count": source_manifest_count,
            "sample_ids": [item["id"] for item in records],
            "records": selected_records,
        },
        warnings,
    )


def _validate_source_pool_manifest(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "source",
        "split",
        "count",
        "seed",
        "sample_ids",
        "records",
        "manifest_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ProbeInputError("pool manifest missing fields: %s" % ", ".join(missing))
    if payload["schema_version"] != "loopscope.probe-pool-manifest.v1":
        raise ProbeInputError("unsupported pool manifest schema_version")
    if "test" in str(payload["split"]).lower():
        raise ProbeInputError("test split is forbidden in a pool manifest")
    if not str(payload["source"]).strip() or not str(payload["split"]).strip():
        raise ProbeInputError("pool manifest source and split must be non-empty")
    try:
        int(payload["seed"])
    except Exception as exc:
        raise ProbeInputError("pool manifest seed must be an integer") from exc
    if not isinstance(payload["sample_ids"], list) or not isinstance(payload["records"], list):
        raise ProbeInputError("pool manifest sample_ids and records must be lists")
    count = int(payload["count"])
    if count < 1 or len(payload["sample_ids"]) != count or len(payload["records"]) != count:
        raise ProbeInputError("pool manifest count/sample_ids/records are inconsistent")
    record_ids = []
    for item in payload["records"]:
        if not isinstance(item, Mapping) or set(("id", "prompt_sha256")).difference(item):
            raise ProbeInputError("pool manifest records require id and prompt_sha256")
        record_ids.append(str(item["id"]))
        digest = str(item["prompt_sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ProbeInputError("pool manifest contains an invalid prompt_sha256")
    if [str(value) for value in payload["sample_ids"]] != record_ids:
        raise ProbeInputError("pool manifest sample_ids do not match records order")


def parse_choice_labels(text: str) -> List[str]:
    labels = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ProbeInputError("--choice-labels must contain at least two unique labels")
    return labels


def choice_tokenization(
    tokenizer: Any, labels: Iterable[str]
) -> Tuple[Dict[str, Any], List[int], List[str]]:
    metadata: Dict[str, Any] = {}
    ids = []
    warnings = []
    for label in labels:
        surface = " " + label
        token_ids = [int(value) for value in tokenizer.encode(surface, add_special_tokens=False)]
        plain_ids = [int(value) for value in tokenizer.encode(label, add_special_tokens=False)]
        metadata[label] = {
            "surface": surface,
            "token_ids": token_ids,
            "plain_surface_token_ids": plain_ids,
            "token_count": len(token_ids),
            "is_single_token": len(token_ids) == 1,
        }
        if len(token_ids) != 1:
            warning = (
                "Choice %s tokenizes to %d tokens for surface %r; "
                "no first-token approximation is allowed."
            )
            warnings.append(warning % (label, len(token_ids), surface))
        else:
            ids.append(token_ids[0])
    if warnings:
        raise ProbeInputError(
            "choice tokenization is not single-token; probe stopped without truncation",
            details={"choice_token_ids": metadata, "warnings": warnings},
        )
    return metadata, ids, warnings


def resolve_answer_position(attention_mask: Any, record: Mapping[str, Any]) -> int:
    valid = attention_mask[0].detach().cpu().nonzero(as_tuple=False).flatten().tolist()
    if not valid:
        raise ProbeInputError("record %s has no non-padding token" % record["id"])
    if record.get("answer_position") is None:
        return int(valid[-1])
    position = int(record["answer_position"])
    if position not in valid:
        raise ProbeInputError(
            "answer_position is padding/out of range for record %s" % record["id"]
        )
    return position


def decoder_layer_states(hidden_states: Sequence[Any], layer_count: int) -> Tuple[Any, ...]:
    if len(hidden_states) == layer_count + 1:
        return tuple(hidden_states[1:])
    if len(hidden_states) == layer_count:
        return tuple(hidden_states)
    raise RuntimeError(
        "expected %d or %d hidden-state tensors, got %d"
        % (layer_count, layer_count + 1, len(hidden_states))
    )


def lens_space_hidden(
    final_norm: Any,
    answer_hidden: Any,
    layer_index: int,
    layer_count: int,
) -> Any:
    """Put all answer vectors in final-norm lens space exactly once."""

    if layer_index < 0 or layer_index >= layer_count:
        raise ValueError("layer_index is outside the decoder layer range")
    if layer_index == layer_count - 1:
        return answer_hidden
    return final_norm(answer_hidden)


def find_final_norm(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(getattr(model, "gpt_neox", None), "final_layer_norm", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "final_layer_norm", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "norm", None),
        getattr(model, "norm", None),
    ]
    for candidate in candidates:
        if callable(candidate):
            return candidate
    raise TypeError("could not locate a Qwen/Llama-style final normalization module")


def compute_effective_rank(torch_module: Any, vectors: Sequence[Any]) -> float:
    if len(vectors) < 2:
        raise ProbeInputError("effective rank requires at least two non-special answer vectors")
    matrix = torch_module.stack(list(vectors), dim=0).float()
    norms = torch_module.linalg.vector_norm(matrix, dim=-1)
    if bool((norms <= 1e-12).any().item()):
        raise ProbeInputError("effective-rank input contains a zero-norm answer vector")
    normalized = matrix / norms.unsqueeze(-1)
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    singular_values = torch_module.linalg.svdvals(centered).detach().cpu().tolist()
    return effective_rank_from_singular_values(singular_values)


def select_device(torch_module: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("requested cuda but torch.cuda.is_available() is False")
    return requested


def torch_dtype(torch_module: Any, dtype: str) -> Any:
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


def git_provenance() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run(*args: str) -> str:
        command = ["git", "-C", str(root)] + list(args)
        return subprocess.check_output(command, text=True).strip()

    return {
        "root": run("rev-parse", "--show-toplevel"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def model_metadata(
    model: Any,
    model_info: Mapping[str, Any],
    layer_count: int,
    requested_revision: Optional[str] = None,
) -> Dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "alias": model_info.get("alias"),
        "repo_id": model_info.get("repo_id"),
        "revision": getattr(config, "_commit_hash", None) or requested_revision,
        "model_type": getattr(config, "model_type", None),
        "layer_count": layer_count,
    }


def tokenizer_revision(tokenizer: Any, requested_revision: Optional[str] = None) -> Any:
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    return (
        getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "commit_hash", None)
        or init_kwargs.get("_commit_hash")
        or init_kwargs.get("revision")
        or requested_revision
    )


def runtime_metadata(torch_module: Any, device: str, dtype: str) -> Dict[str, Any]:
    return {
        "device": device,
        "dtype": dtype,
        "use_cache": False,
        "env": environment_snapshot(),
        "versions": {
            "python": os.sys.version.split()[0],
            "torch": getattr(torch_module, "__version__", None),
            "transformers": module_version("transformers"),
            "lm_eval": module_version("lm_eval"),
            "tflt": module_version("tflt") or "source-tree",
        },
    }


def environment_snapshot() -> Dict[str, str]:
    keys = [
        "HF_ENDPOINT",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_DISABLE_XET",
        "UV_CACHE_DIR",
        "VIRTUAL_ENV",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "PYTHONPATH",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def serializable_args(args: Any) -> Dict[str, Any]:
    """Drop argparse dispatch callables before writing command provenance."""

    return {
        str(key): value
        for key, value in vars(args).items()
        if key != "func" and not callable(value)
    }


def module_version(name: str) -> Any:
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _write_failure(output_dir: Path, exc: Exception) -> None:
    payload: Dict[str, Any] = {
        "error_type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    details = getattr(exc, "details", None)
    if details:
        payload["details"] = details
    write_new_json(output_dir / "failure.json", payload)
