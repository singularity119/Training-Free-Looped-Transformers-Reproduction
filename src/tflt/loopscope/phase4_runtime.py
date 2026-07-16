"""Narrow renderer, controls, and native prefix runtime for Phase 4 P4-B."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase4_schema import (
    BOUNDARY_IDS,
    Phase4ContractError,
    canonical_json_bytes,
    scan_forbidden_fields,
    validate_trajectory_record,
)


CATEGORIES: Tuple[str, ...] = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)
EXPECTED_RAW_COLUMNS = {
    "question_id",
    "question",
    "options",
    "answer",
    "answer_index",
    "cot_content",
    "category",
    "src",
}
SAFE_DATASET_COLUMNS: Tuple[str, ...] = (
    "question_id",
    "category",
    "src",
    "question",
    "options",
)
SAFE_TARGET_KEYS = set(SAFE_DATASET_COLUMNS)
TOKENIZATION_CLOSURE_KEYS: Tuple[str, ...] = (
    "rendered_prefix_sha256",
    "rendered_token_ids_sha256",
    "attention_mask_sha256",
    "last_effective_prefix_token_index",
    "last_effective_prefix_token_sha256",
)
OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEFGHIJ")
TEMPLATE_QUESTION = "Select the most appropriate neutral option for this placeholder question."
TEMPLATE_OPTIONS: Tuple[str, ...] = tuple(
    "Neutral placeholder option %s" % label for label in OPTION_LABELS
)


class Phase4RuntimeError(Phase4ContractError):
    """Raised when the exact P4-B runtime cannot satisfy the frozen contract."""


def project_test_dataset(dataset: Any) -> Any:
    """Project cached target rows before Python iteration or record construction."""

    columns = getattr(dataset, "column_names", None)
    if not isinstance(columns, list) or set(columns) != EXPECTED_RAW_COLUMNS:
        raise Phase4RuntimeError("cached MMLU-Pro target column schema differs")
    projected = dataset.select_columns(list(SAFE_DATASET_COLUMNS))
    if set(getattr(projected, "column_names", [])) != set(SAFE_DATASET_COLUMNS):
        raise Phase4RuntimeError("target source projection did not close to the allowlist")
    return projected


def project_target_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an already projected dataset row to the frozen safe shape."""

    if set(row) != SAFE_TARGET_KEYS:
        raise Phase4RuntimeError("projected target row keys differ from the allowlist")
    value = {
        "question_id": row["question_id"],
        "category": row["category"],
        "src": row["src"],
        "question": row["question"],
        "ordered_options": list(row["options"]),
    }
    scan_forbidden_fields(value)
    if value["category"] not in CATEGORIES:
        raise Phase4RuntimeError("target category is outside the frozen 14 categories")
    if not 2 <= len(value["ordered_options"]) <= 10:
        raise Phase4RuntimeError("MMLU-Pro target must contain 2..10 ordered options")
    if not all(isinstance(item, str) for item in value["ordered_options"]):
        raise Phase4RuntimeError("MMLU-Pro target options must be strings")
    return value


def validation_demos_by_category(validation_rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Hold validation CoT only inside the renderer path and keep first_n=5."""

    grouped: Dict[str, List[Dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for raw in validation_rows:
        category = raw.get("category")
        if category in grouped and len(grouped[category]) < 5:
            # The renderer needs these exact fields. The returned object must remain
            # process-local and must never be inserted into a target/source record.
            if not all(key in raw for key in ("question", "options", "cot_content")):
                raise Phase4RuntimeError("validation renderer row lacks required fields")
            grouped[category].append(
                {
                    "question": raw["question"],
                    "options": list(raw["options"]),
                    "cot_content": raw["cot_content"],
                }
            )
    if any(len(rows) != 5 for rows in grouped.values()):
        raise Phase4RuntimeError("validation first_n renderer source is not exactly 5 x 14")
    return grouped


def render_exact_prefix(
    safe_target: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]],
    renderer_utils: Any,
) -> str:
    """Reproduce lm-eval 0.4.11 MMLU-Pro fewshot_context without target labels."""

    if set(safe_target) != {
        "question_id",
        "category",
        "src",
        "question",
        "ordered_options",
    }:
        raise Phase4RuntimeError("renderer target is not the frozen safe projection")
    scan_forbidden_fields(safe_target)
    if len(demos) != 5:
        raise Phase4RuntimeError("renderer requires exact first_n=5 demonstrations")
    target_doc = {
        "question": safe_target["question"],
        "options": list(safe_target["ordered_options"]),
    }
    # lm-eval's fewshot_context concatenates five user messages produced by
    # fewshot_to_text (each already ends in two newlines), then doc_to_text.
    prefix = "".join(str(renderer_utils.fewshot_to_text(dict(row))) for row in demos)
    prefix += str(renderer_utils.doc_to_text(target_doc))
    if not prefix.endswith("Answer: Let's think step by step."):
        raise Phase4RuntimeError("rendered prefix does not end at the frozen generation cue")
    return prefix


def proportional_category_quotas(category_counts: Mapping[str, int], total: int = 512) -> Dict[str, int]:
    """Freeze proportional largest-remainder category allocation."""

    if set(category_counts) != set(CATEGORIES):
        raise Phase4RuntimeError("category counts must cover the frozen 14 categories")
    if not isinstance(total, int) or isinstance(total, bool) or total < len(CATEGORIES):
        raise Phase4RuntimeError("control subset total is invalid")
    if any(
        not isinstance(category_counts[key], int)
        or isinstance(category_counts[key], bool)
        or category_counts[key] <= 0
        for key in CATEGORIES
    ):
        raise Phase4RuntimeError("category counts must be positive")
    counts = {key: category_counts[key] for key in CATEGORIES}
    population = sum(counts.values())
    if total > population:
        raise Phase4RuntimeError("control subset total exceeds the shared population")
    numerators = {key: total * counts[key] for key in CATEGORIES}
    quotas = {key: numerators[key] // population for key in CATEGORIES}
    remainder = total - sum(quotas.values())
    order = sorted(CATEGORIES, key=lambda key: (-(numerators[key] % population), key))
    for key in order[:remainder]:
        quotas[key] += 1
    if sum(quotas.values()) != total or any(quotas[key] > counts[key] for key in CATEGORIES):
        raise Phase4RuntimeError("category quota allocation is invalid")
    return quotas


def identity_hash(identity: str) -> str:
    if not isinstance(identity, str) or not identity:
        raise Phase4RuntimeError("canonical identity must be a non-empty string")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_option_control_identities(
    source_rows: Sequence[Mapping[str, Any]], quotas: Mapping[str, int]
) -> List[str]:
    """Select the lowest identity SHA256 values independently in each category."""

    if set(quotas) != set(CATEGORIES):
        raise Phase4RuntimeError("option-control quotas differ from frozen categories")
    grouped: Dict[str, List[str]] = {category: [] for category in CATEGORIES}
    for row in source_rows:
        category = row.get("category")
        identity = row.get("canonical_identity")
        if category not in grouped or not isinstance(identity, str):
            raise Phase4RuntimeError("source row lacks option-control identity/category")
        grouped[category].append(identity)
    selected: List[str] = []
    for category in CATEGORIES:
        ordered = sorted(grouped[category], key=lambda value: (identity_hash(value), value))
        quota = int(quotas[category])
        if quota < 0 or quota > len(ordered):
            raise Phase4RuntimeError("option-control quota exceeds category membership")
        selected.extend(ordered[:quota])
    if len(selected) != sum(int(value) for value in quotas.values()) or len(set(selected)) != len(selected):
        raise Phase4RuntimeError("option-control selection is not unique/complete")
    return selected


def deterministic_option_permutation(identity: str, option_count: int = 10) -> List[int]:
    """Derive a stable non-identity permutation solely from identity SHA256."""

    if option_count < 2:
        raise Phase4RuntimeError("option permutation requires at least two options")
    order = sorted(
        range(option_count),
        key=lambda index: hashlib.sha256(
            (identity_hash(identity) + ":%d" % index).encode("ascii")
        ).hexdigest(),
    )
    if order == list(range(option_count)):
        order = order[1:] + order[:1]
    return order


def permute_safe_target(safe_target: Mapping[str, Any], permutation: Sequence[int]) -> Dict[str, Any]:
    scan_forbidden_fields(safe_target)
    options = list(safe_target["ordered_options"])
    if sorted(permutation) != list(range(len(options))):
        raise Phase4RuntimeError("option permutation is not a complete index permutation")
    result = dict(safe_target)
    result["ordered_options"] = [options[index] for index in permutation]
    scan_forbidden_fields(result)
    return result


def template_safe_targets() -> List[Dict[str, Any]]:
    return [
        {
            "question_id": index,
            "category": category,
            "src": "loopscope_phase4_template_control",
            "question": TEMPLATE_QUESTION,
            "ordered_options": list(TEMPLATE_OPTIONS),
        }
        for index, category in enumerate(CATEGORIES)
    ]


def tokenization_metadata(tokenizer: Any, prefix: str) -> Dict[str, Any]:
    token_ids = tokenizer.encode(prefix)
    if not isinstance(token_ids, list) or not token_ids or not all(isinstance(value, int) for value in token_ids):
        raise Phase4RuntimeError("tokenizer did not return a non-empty integer sequence")
    encoded = tokenizer(prefix, return_tensors="pt")
    if not {"input_ids", "attention_mask"}.issubset(set(encoded)):
        raise Phase4RuntimeError("tokenizer output lacks input_ids/attention_mask")
    encoded_ids = encoded["input_ids"][0].detach().cpu().tolist()
    mask = encoded["attention_mask"][0].detach().cpu().tolist()
    if encoded_ids != token_ids:
        raise Phase4RuntimeError("tokenizer encode and batch-call token IDs differ")
    valid = [index for index, value in enumerate(mask) if int(value) == 1]
    if valid != list(range(len(token_ids))):
        raise Phase4RuntimeError("unbatched prefix attention mask is not all-valid")
    return {
        "rendered_prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "rendered_token_ids_sha256": hashlib.sha256(canonical_json_bytes(token_ids)).hexdigest(),
        "attention_mask_sha256": hashlib.sha256(canonical_json_bytes(mask)).hexdigest(),
        "last_effective_prefix_token_index": valid[-1],
        "last_effective_prefix_token_sha256": hashlib.sha256(
            canonical_json_bytes(token_ids[-1])
        ).hexdigest(),
        "sequence_length": len(token_ids),
    }


def tokenization_closure(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the five fields that must close again immediately before forward."""

    missing = [key for key in TOKENIZATION_CLOSURE_KEYS if key not in metadata]
    if missing:
        raise Phase4RuntimeError("tokenization closure lacks required fields")
    closure = {key: metadata[key] for key in TOKENIZATION_CLOSURE_KEYS}
    scan_forbidden_fields(closure)
    return closure


def _assert_no_loop_modules_loaded() -> None:
    import sys

    forbidden = [
        name
        for name in sys.modules
        if name == "tflt.wrapper"
        or name.startswith("tflt.wrapper.")
        or name == "tflt.strategies"
        or name.startswith("tflt.strategies.")
        or name == "tflt.cache"
        or name.startswith("tflt.cache.")
    ]
    if forbidden:
        raise Phase4RuntimeError("loop implementation modules are loaded in native producer")


@dataclass
class NativePrefixRuntime:
    torch: Any
    tokenizer: Any
    model: Any
    final_norm: Any
    lm_head: Any
    layer_count: int
    revision_closure: Mapping[str, Any]
    decoder_boundary_states: Any
    lens_space_hidden: Any


def load_native_prefix_runtime(card: Mapping[str, Any]) -> NativePrefixRuntime:
    _assert_no_loop_modules_loaded()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tflt.loopscope.probe import decoder_boundary_states, find_final_norm, lens_space_hidden
        from tflt.loopscope.revisions import load_tokenizer_with_resolved_commit, strict_revision_closure
    except Exception as exc:  # pragma: no cover - remote dependency path
        raise Phase4RuntimeError("P4-B audited runtime imports failed") from exc
    if not torch.cuda.is_available():
        raise Phase4RuntimeError("P4-B native prefix producer requires a visible CUDA GPU")
    model_cfg = card["model"]
    load_kwargs = {
        "revision": model_cfg["revision"],
        "local_files_only": True,
        "trust_remote_code": True,
    }
    tokenizer, tokenizer_commit = load_tokenizer_with_resolved_commit(
        AutoTokenizer, model_cfg["repo"], load_kwargs
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["repo"], torch_dtype=torch.bfloat16, **load_kwargs
    )
    closure = strict_revision_closure(model, tokenizer_commit, model_cfg["revision"])
    layer_count = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if layer_count != 36:
        raise Phase4RuntimeError("loaded model does not expose exactly 36 decoder layers")
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise Phase4RuntimeError("loaded model lacks final-position logits_to_keep support")
    final_norm = find_final_norm(model)
    lm_head = model.get_output_embeddings() or getattr(model, "lm_head", None)
    if lm_head is None or not callable(lm_head):
        raise Phase4RuntimeError("loaded model lacks a callable output head")
    model.eval()
    model.to("cuda")
    _assert_no_loop_modules_loaded()
    return NativePrefixRuntime(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        final_norm=final_norm,
        lm_head=lm_head,
        layer_count=layer_count,
        revision_closure=closure,
        decoder_boundary_states=decoder_boundary_states,
        lens_space_hidden=lens_space_hidden,
    )


def acquire_prefix_scalar_record(
    runtime: NativePrefixRuntime,
    *,
    prefix: str,
    canonical_identity: str,
    category: str,
    expected_tokenization: Mapping[str, Any],
    producer_provenance: Mapping[str, Any],
    d36_tolerance: float = 1e-6,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one native prefix forward and reduce B_0..B_36 to scalar-only JSON."""

    _assert_no_loop_modules_loaded()
    metadata = tokenization_metadata(runtime.tokenizer, prefix)
    for key in TOKENIZATION_CLOSURE_KEYS:
        if metadata.get(key) != expected_tokenization.get(key):
            raise Phase4RuntimeError("formal renderer/tokenization drift at %s" % key)
    encoded = runtime.tokenizer(prefix, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in encoded.items()}
    torch = runtime.torch
    with torch.inference_mode():
        outputs = runtime.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        states = runtime.decoder_boundary_states(tuple(outputs.hidden_states or ()), 36)
        position = int(expected_tokenization["last_effective_prefix_token_index"])
        final_hidden = states[-1][:, position, :]
        final_projection = runtime.lm_head(final_hidden).float()
        if tuple(outputs.logits.shape[:2]) != (1, 1):
            raise Phase4RuntimeError("native model did not restrict output projection to one position")
        closure_difference = float(
            (outputs.logits[:, 0, :].float() - final_projection).abs().max().double().cpu()
        )
        if not torch.allclose(outputs.logits[:, 0, :].float(), final_projection, rtol=1e-3, atol=1e-3):
            raise Phase4RuntimeError("native final output does not close through final projection")
        final_log_p = torch.log_softmax(final_projection, dim=-1)
        vocabulary_size = int(final_projection.shape[-1])
        if vocabulary_size != int(getattr(runtime.model.config, "vocab_size", 0) or 0):
            raise Phase4RuntimeError("runtime vocabulary size differs from model config")
        boundaries: List[Dict[str, Any]] = []
        for boundary_id, hidden in enumerate(states):
            current = hidden[:, position, :]
            lens_hidden = runtime.lens_space_hidden(runtime.final_norm, current, boundary_id, 36)
            projection = final_projection if boundary_id == 36 else runtime.lm_head(lens_hidden).float()
            log_p = final_log_p if boundary_id == 36 else torch.log_softmax(projection, dim=-1)
            probabilities = torch.exp(log_p)
            entropy = float((-(probabilities * log_p).sum(dim=-1)).double().cpu())
            kl = 0.0 if boundary_id == 36 else float(
                (probabilities * (log_p - final_log_p)).sum(dim=-1).double().cpu()
            )
            rms = float(torch.sqrt(torch.mean(projection * projection, dim=-1)).double().cpu())
            top = torch.topk(probabilities, k=min(100, vocabulary_size), dim=-1).values
            mass1 = float(top[:, :1].sum(dim=-1).double().cpu())
            mass10 = float(top[:, : min(10, vocabulary_size)].sum(dim=-1).double().cpu())
            mass100 = float(top.sum(dim=-1).double().cpu())
            values = (entropy, kl, rms, mass1, mass10, mass100)
            if not all(math.isfinite(value) for value in values):
                raise Phase4RuntimeError("prefix scalar reduction produced a non-finite value")
            if entropy < -1e-6 or entropy > math.log(vocabulary_size) + 1e-5:
                raise Phase4RuntimeError("full-vocabulary entropy is outside its mathematical range")
            if kl < -1e-5 or rms < 0.0:
                raise Phase4RuntimeError("KL or projection RMS violates its mathematical range")
            if not (0.0 <= mass1 <= mass10 <= mass100 <= 1.0 + 1e-6):
                raise Phase4RuntimeError("top-k probability masses violate the frozen invariant")
            boundaries.append(
                {
                    "boundary_id": boundary_id,
                    "full_vocabulary_entropy": entropy,
                    "kl_to_final": kl,
                    "normalized_entropy": entropy / math.log(vocabulary_size),
                    "logit_rms": rms,
                    "top1_probability_mass": mass1,
                    "top10_probability_mass": mass10,
                    "top100_probability_mass": mass100,
                    "finite": True,
                }
            )
            del probabilities, top
            if boundary_id != 36:
                del projection, log_p
        record = {
            "schema_version": "loopscope.phase4.prefix-trajectory-record.v1",
            "canonical_identity": canonical_identity,
            "category": category,
            "boundaries": boundaries,
            "producer_provenance": dict(producer_provenance),
        }
        validate_trajectory_record(record, d36_tolerance=d36_tolerance)
    del outputs, states, inputs, encoded, final_projection, final_log_p
    _assert_no_loop_modules_loaded()
    return record, {
        "sequence_length": metadata["sequence_length"],
        "last_effective_prefix_token_index": metadata["last_effective_prefix_token_index"],
        "final_projection_max_abs_difference": closure_difference,
        "vocabulary_size": vocabulary_size,
        "boundary_count": len(BOUNDARY_IDS),
        "generation_performed": False,
        "target_generation_count": 0,
    }


__all__ = [
    "CATEGORIES",
    "EXPECTED_RAW_COLUMNS",
    "NativePrefixRuntime",
    "Phase4RuntimeError",
    "SAFE_DATASET_COLUMNS",
    "TOKENIZATION_CLOSURE_KEYS",
    "acquire_prefix_scalar_record",
    "deterministic_option_permutation",
    "identity_hash",
    "load_native_prefix_runtime",
    "permute_safe_target",
    "project_target_row",
    "project_test_dataset",
    "proportional_category_quotas",
    "render_exact_prefix",
    "select_option_control_identities",
    "template_safe_targets",
    "tokenization_metadata",
    "tokenization_closure",
    "validation_demos_by_category",
]
