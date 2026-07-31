"""Minimal real-model runtime for LoopScope Phase 6 Gate C.

The runtime keeps generated text and token IDs in memory only.  Its public
result contains one closed sanitized trajectory record plus scalar/hash
closures needed by the independent Gate C verifier.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from tflt.loopscope.phase4_runtime import tokenization_metadata
from tflt.loopscope.phase6_acquisition import build_sanitized_trajectory_record
from tflt.loopscope.phase6_anchor import (
    ANSWER_SPAN_EXTRACTOR_SHA256,
    DEFAULT_DECODE_KWARGS,
    GENERATED_ID_TEXT_ALIGNER_SHA256,
    answer_span_extractor,
    generated_id_text_aligner,
    replay_prefix_ids,
)


MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REPO = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
LAYER_COUNT = 36
HIDDEN_SIZE = 2560
GENERATION_MAX_NEW_TOKENS = 2048
GENERATION_STOP_STRING = "Question:"
PRODUCER_VERSION = "loopscope.phase6.gate-c-runtime.v4"
FINAL_NORM_RTOL = 1e-3
FINAL_NORM_ATOL = 1e-3
LOOP_WRAPPER_CLASS_NAMES = frozenset(
    {"LoopLayerWrapper", "LoopBlockEntryWrapper", "LoopIdentityLayer"}
)


class GateCRuntimeError(RuntimeError):
    """The Gate C normal path failed a frozen runtime contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateCRuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be str")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def module_path(root: Any, target: Any) -> str:
    for name, module in root.named_modules():
        if module is target:
            return name or "<root>"
    raise GateCRuntimeError("runtime module is not registered under the model")


def assert_native_no_loop_runtime(model: Any) -> None:
    """Fail closed only when a loop wrapper is registered on the model itself."""

    named_modules = getattr(model, "named_modules", None)
    _require(callable(named_modules), "loaded model lacks a named_modules registry")
    wrapped_paths = []
    for path, module in named_modules():
        module_type = type(module)
        implementation_module = str(getattr(module_type, "__module__", ""))
        implementation_name = str(getattr(module_type, "__name__", ""))
        if (
            implementation_module == "tflt.wrapper"
            or implementation_module.startswith("tflt.wrapper.")
            or implementation_name in LOOP_WRAPPER_CLASS_NAMES
        ):
            wrapped_paths.append(str(path) or "<root>")
    _require(
        not wrapped_paths,
        "loop wrapper is applied to registered model modules: %s"
        % ",".join(sorted(wrapped_paths)),
    )


def assemble_raw_boundaries(
    hidden_states: Sequence[Any], captured_raw_b36: Any, pre_hook_count: int
) -> Tuple[Any, ...]:
    """Replace the post-FinalNorm endpoint with the one pre-hook raw endpoint."""

    _require(pre_hook_count == 1, "FinalNorm pre-hook count must equal one")
    _require(
        len(hidden_states) == LAYER_COUNT + 1,
        "replay hidden states must contain post-norm B0...B36 closure",
    )
    _require(captured_raw_b36 is not None, "FinalNorm pre-hook did not capture raw B36")
    return tuple(hidden_states[:-1]) + (captured_raw_b36,)


def normalize_raw_boundary_vectors(
    raw_state_tensors: Sequence[Any],
) -> Tuple[Any, ...]:
    """Select one `[1,H]` vector for each strict B0...B36 tensor shape."""

    states = tuple(raw_state_tensors)
    _require(
        len(states) == LAYER_COUNT + 1,
        "raw boundary states must contain B0...B36",
    )
    vectors = []
    hidden_width = None
    for boundary_index, state in enumerate(states):
        shape = tuple(int(value) for value in getattr(state, "shape", ()))
        if boundary_index < LAYER_COUNT:
            _require(
                getattr(state, "ndim", None) == 3
                and len(shape) == 3
                and shape[0] == 1
                and shape[1] >= 1
                and shape[2] >= 1,
                "model hidden boundary B%d must have shape [1,S,H]"
                % boundary_index,
            )
            vector = state[:, -1, :]
        else:
            _require(
                getattr(state, "ndim", None) == 2
                and len(shape) == 2
                and shape[0] == 1
                and shape[1] >= 1,
                "raw B36 pre-hook boundary must have shape [1,H]",
            )
            vector = state
        vector_shape = tuple(int(value) for value in getattr(vector, "shape", ()))
        _require(
            getattr(vector, "ndim", None) == 2
            and len(vector_shape) == 2
            and vector_shape[0] == 1,
            "normalized boundary B%d must have shape [1,H]" % boundary_index,
        )
        if hidden_width is None:
            hidden_width = vector_shape[1]
        _require(
            vector_shape[1] == hidden_width,
            "raw boundary hidden widths differ",
        )
        vectors.append(vector)
    return tuple(vectors)


def project_boundaries_in_model_dtype(
    final_norm: Any, lm_head: Any, raw_vectors: Any
) -> Tuple[Any, Any]:
    """Run FinalNorm/LM head in model dtype, then return float32 outputs."""

    normalized_native = final_norm(raw_vectors)
    _require(
        tuple(getattr(normalized_native, "shape", ()))
        == tuple(getattr(raw_vectors, "shape", ())),
        "FinalNorm boundary matrix shape differs",
    )
    boundary_logits_native = lm_head(normalized_native)
    logits_shape = tuple(getattr(boundary_logits_native, "shape", ()))
    _require(
        getattr(boundary_logits_native, "ndim", None) == 2
        and len(logits_shape) == 2
        and logits_shape[0] == LAYER_COUNT + 1
        and logits_shape[1] >= 1,
        "LM-head boundary logits shape differs",
    )
    return normalized_native.float(), boundary_logits_native.float()


def duplicate_result_sha256(record: Mapping[str, Any]) -> str:
    return semantic_sha256(record)


@dataclass
class GateCRuntime:
    torch: Any
    tokenizer: Any
    model: Any
    final_norm: Any
    lm_head: Any
    final_norm_path: str
    lm_head_path: str
    revision_closure: Mapping[str, Any]


def load_gate_c_runtime() -> GateCRuntime:
    """Load the pinned Qwen runtime from audited local cache only."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tflt.loopscope.probe import find_final_norm
        from tflt.loopscope.revisions import (
            load_tokenizer_with_resolved_commit,
            strict_revision_closure,
        )
    except Exception as exc:  # pragma: no cover - remote dependency path
        raise GateCRuntimeError("Gate C audited runtime imports failed") from exc

    _require(torch.cuda.is_available(), "Gate C requires one visible CUDA GPU")
    load_kwargs = {
        "revision": MODEL_REVISION,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    tokenizer, tokenizer_commit = load_tokenizer_with_resolved_commit(
        AutoTokenizer, MODEL_REPO, load_kwargs
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16,
        **load_kwargs,
    )
    assert_native_no_loop_runtime(model)
    closure = strict_revision_closure(model, tokenizer_commit, MODEL_REVISION)
    _require(
        int(getattr(model.config, "num_hidden_layers", 0) or 0) == LAYER_COUNT,
        "loaded model layer count differs",
    )
    _require(
        int(getattr(model.config, "hidden_size", 0) or 0) == HIDDEN_SIZE,
        "loaded model hidden size differs",
    )
    final_norm = find_final_norm(model)
    lm_head = model.get_output_embeddings() or getattr(model, "lm_head", None)
    _require(callable(lm_head), "loaded model lacks callable LM head")
    _require(
        "output_hidden_states" in inspect.signature(model.forward).parameters,
        "loaded model forward lacks output_hidden_states",
    )
    model.eval()
    model.to("cuda")
    assert_native_no_loop_runtime(model)
    return GateCRuntime(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        final_norm=final_norm,
        lm_head=lm_head,
        final_norm_path=module_path(model, final_norm),
        lm_head_path=module_path(model, lm_head),
        revision_closure=closure,
    )


def _token_ids(tokenizer: Any, prefix: str) -> Tuple[int, ...]:
    values = tokenizer.encode(prefix)
    _require(
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in values),
        "tokenizer did not return exact non-empty integer prompt IDs",
    )
    return tuple(values)


def _generate_once(runtime: GateCRuntime, prompt_ids: Sequence[int]) -> Tuple[int, ...]:
    assert_native_no_loop_runtime(runtime.model)
    torch = runtime.torch
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    eos_token_id = getattr(runtime.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(runtime.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    with torch.inference_mode():
        sequences = runtime.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
            stop_strings=[GENERATION_STOP_STRING],
            tokenizer=runtime.tokenizer,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    _require(
        getattr(sequences, "ndim", None) == 2 and int(sequences.shape[0]) == 1,
        "generation did not return one sequence",
    )
    complete = tuple(int(value) for value in sequences[0].detach().cpu().tolist())
    prompt = tuple(int(value) for value in prompt_ids)
    _require(complete[: len(prompt)] == prompt, "generation prompt-ID prefix differs")
    generated = complete[len(prompt) :]
    _require(bool(generated), "generation returned no new tokens")
    return generated


def _replay_once(
    runtime: GateCRuntime, replay_ids: Sequence[int], expected_next_token_id: int
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Run exactly one replay and return CPU float32 matrices plus closures."""

    assert_native_no_loop_runtime(runtime.model)
    torch = runtime.torch
    captured: Dict[str, Any] = {"count": 0, "last": None}

    def capture_raw_b36(_module: Any, inputs: Sequence[Any]) -> None:
        captured["count"] += 1
        _require(bool(inputs), "FinalNorm pre-hook received no positional input")
        tensor = inputs[0]
        _require(
            getattr(tensor, "ndim", None) == 3 and int(tensor.shape[0]) == 1,
            "FinalNorm pre-hook input shape differs",
        )
        captured["last"] = tensor[:, -1, :].detach().clone()

    handle = runtime.final_norm.register_forward_pre_hook(capture_raw_b36)
    input_ids = torch.tensor([list(replay_ids)], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.inference_mode():
            outputs = runtime.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        handle.remove()

    hidden_states = tuple(outputs.hidden_states or ())
    raw_state_tensors = assemble_raw_boundaries(
        hidden_states, captured["last"], int(captured["count"])
    )
    raw_vectors_native = torch.cat(
        [
            vector.detach()
            for vector in normalize_raw_boundary_vectors(raw_state_tensors)
        ],
        dim=0,
    )
    _require(
        tuple(raw_vectors_native.shape) == (LAYER_COUNT + 1, HIDDEN_SIZE),
        "raw boundary vector matrix shape differs",
    )
    with torch.inference_mode():
        normalized, boundary_logits = project_boundaries_in_model_dtype(
            runtime.final_norm, runtime.lm_head, raw_vectors_native
        )
    raw_vectors = raw_vectors_native.float()
    post_norm_final = hidden_states[-1][:, -1, :].float()
    final_norm_max_abs = float(
        (normalized[-1:, :] - post_norm_final).abs().max().double().cpu()
    )
    _require(
        bool(
            torch.allclose(
                normalized[-1:, :],
                post_norm_final,
                rtol=FINAL_NORM_RTOL,
                atol=FINAL_NORM_ATOL,
            )
        ),
        "FinalNorm(raw B36) does not close post-norm hidden state",
    )
    observed_next_token_id = int(torch.argmax(boundary_logits[-1]).item())
    _require(
        observed_next_token_id == int(expected_next_token_id),
        "replay final next-token argmax differs from generated answer-first token",
    )
    _require(
        bool(torch.isfinite(raw_vectors).all().item())
        and bool(torch.isfinite(normalized).all().item())
        and bool(torch.isfinite(boundary_logits).all().item()),
        "replay boundary/logit tensors contain non-finite values",
    )
    evidence = {
        "final_norm_pre_hook_count": int(captured["count"]),
        "raw_boundary_count": int(raw_vectors.shape[0]),
        "final_norm_postnorm_allclose": True,
        "final_norm_postnorm_max_abs": final_norm_max_abs,
        "replay_next_token_closure": True,
        "replay_sequence_length": len(replay_ids),
    }
    assert_native_no_loop_runtime(runtime.model)
    return (
        boundary_logits.detach().cpu().numpy(),
        normalized.detach().cpu().numpy(),
        raw_vectors.detach().cpu().numpy(),
        evidence,
    )


def acquire_two_pass_record(
    runtime: GateCRuntime,
    *,
    prefix: str,
    identity: Mapping[str, Any],
    gate_b_manifest_sha256: str,
    card_sha256: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Acquire one exact generation/replay replicate without persisting raw data."""

    expected_prompt_hash = str(identity["rendered_prefix_sha256"])
    prompt_metadata = tokenization_metadata(runtime.tokenizer, prefix)
    _require(
        prompt_metadata["rendered_prefix_sha256"] == expected_prompt_hash,
        "rendered prompt hash differs from Gate B",
    )
    _require(
        prompt_metadata["rendered_token_ids_sha256"]
        == identity["rendered_token_ids_sha256"],
        "rendered prompt token-ID hash differs from Gate B",
    )
    _require(
        int(prompt_metadata["sequence_length"]) == int(identity["sequence_length"]),
        "rendered prompt sequence length differs from Gate B",
    )
    prompt_ids = _token_ids(runtime.tokenizer, prefix)
    generated_ids = _generate_once(runtime, prompt_ids)
    generated_text = runtime.tokenizer.decode(
        list(generated_ids), **dict(DEFAULT_DECODE_KWARGS)
    )
    span = answer_span_extractor(generated_text)
    alignment = generated_id_text_aligner(
        generated_ids, runtime.tokenizer, generated_text, span
    )
    replay_ids = replay_prefix_ids(
        prompt_ids, generated_ids, alignment.answer_first_token_index
    )
    expected_next = int(generated_ids[alignment.answer_first_token_index])
    boundary_logits, normalized, raw_boundaries, replay_evidence = _replay_once(
        runtime, replay_ids, expected_next
    )
    provenance = {
        "producer_version": PRODUCER_VERSION,
        "card_sha256": card_sha256,
        "renderer_manifest_sha256": gate_b_manifest_sha256,
        "tokenizer_manifest_sha256": gate_b_manifest_sha256,
        "boundary_capture": "raw_B0_through_B36_final_norm_pre_hook",
        "final_norm_path": runtime.final_norm_path,
        "lm_head_path": runtime.lm_head_path,
    }
    record = build_sanitized_trajectory_record(
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        dataset_repo=DATASET_REPO,
        dataset_revision=DATASET_REVISION,
        split="test",
        canonical_identity=str(identity["canonical_identity"]),
        category=str(identity["category"]),
        prompt_sha256=expected_prompt_hash,
        generated_completion_sha256=text_sha256(generated_text),
        generation_length=len(generated_ids),
        replay_length=len(replay_ids),
        anchor_token_index=alignment.probe_token_index,
        answer_span_start_offset=span.byte_start,
        answer_span_end_offset=span.byte_end,
        answer_match_count=span.answer_match_count,
        selected_match_ordinal=span.selected_match_ordinal,
        answer_first_token_index=alignment.answer_first_token_index,
        answer_span_extractor_sha256=ANSWER_SPAN_EXTRACTOR_SHA256,
        generated_id_text_aligner_sha256=GENERATED_ID_TEXT_ALIGNER_SHA256,
        generation_prefix_ids=replay_ids,
        replay_ids=replay_ids,
        boundary_logits=boundary_logits,
        final_normalized_vectors=normalized,
        raw_boundaries=raw_boundaries,
        provenance=provenance,
    )
    diagnostic_values = (
        list(record["H"])
        + list(record["D"])
        + list(record["hidden_rms_l2_to_final"])
        + list(record["hidden_cosine_to_final"])
        + list(record["hidden_cosine_distance_to_final"])
        + list(record["adjacent_angular_distance"])
    )
    _require(
        all(math.isfinite(float(value)) for value in diagnostic_values),
        "sanitized trajectory contains non-finite values",
    )
    evidence = {
        "canonical_identity": str(identity["canonical_identity"]),
        "prompt_sha256": expected_prompt_hash,
        "generated_completion_sha256": record["generated_completion_sha256"],
        "generation_ids_sha256": record["provenance"]["generation_ids_sha256"],
        "replay_ids_sha256": record["provenance"]["replay_ids_sha256"],
        "generation_length": len(generated_ids),
        "replay_length": len(replay_ids),
        "anchor_token_index": alignment.probe_token_index,
        "answer_span_start_offset": span.byte_start,
        "answer_span_end_offset": span.byte_end,
        "answer_match_count": span.answer_match_count,
        "selected_match_ordinal": span.selected_match_ordinal,
        "answer_first_token_index": alignment.answer_first_token_index,
        "generation_count": 1,
        "replay_count": 1,
        "loop_insertions": 0,
        "anchor_resolved": True,
        "unique_token_mapping": True,
        "record_semantic_sha256": duplicate_result_sha256(record),
        **replay_evidence,
    }
    return record, evidence


__all__ = [
    "DATASET_REPO",
    "DATASET_REVISION",
    "GENERATION_MAX_NEW_TOKENS",
    "GENERATION_STOP_STRING",
    "GateCRuntime",
    "GateCRuntimeError",
    "LAYER_COUNT",
    "MODEL_REPO",
    "MODEL_REVISION",
    "PRODUCER_VERSION",
    "acquire_two_pass_record",
    "assert_native_no_loop_runtime",
    "assemble_raw_boundaries",
    "canonical_json_bytes",
    "duplicate_result_sha256",
    "load_gate_c_runtime",
    "normalize_raw_boundary_vectors",
    "project_boundaries_in_model_dtype",
    "semantic_sha256",
    "text_sha256",
]
