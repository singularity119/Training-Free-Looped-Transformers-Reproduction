"""Native zero-loop scalar producer for the Phase 7 Gate B contract.

The module keeps torch/transformers lazy.  Gate A can therefore test the
boundary protocol with fake modules and run the schema/V3 paths without
loading a model or executing a forward pass.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase7_runtime import (
    Phase7RuntimeError,
    assemble_raw_boundaries,
    assert_no_loop_wrapper,
    get_path,
    hidden_states_from_output,
    register_raw_final_norm_hook,
    resolve_architecture,
    resolve_lm_head,
    select_position_vector,
)
from tflt.loopscope.phase7_schema import (
    CHOICE_SURFACES,
    DATASET_REPO,
    DATASET_SPLIT,
    MODEL_SPEC_BY_KEY,
    RUNTIME_DTYPE,
    TRAJECTORY_SCHEMA_VERSION,
    Phase7ContractError,
    choice_surface_summary,
    validate_sanitized_record,
)


def _encode_surface(tokenizer: Any, surface: str) -> Sequence[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(surface, add_special_tokens=False)
    else:
        encoded = tokenizer(surface, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise Phase7ContractError("tokenizer did not return a token sequence")
    return encoded


def admit_choice_surfaces(tokenizer: Any) -> Tuple[Dict[str, Any], Tuple[int, ...]]:
    """Return a sanitized summary and ephemeral choice ids.

    The ids are intentionally returned only to the caller's memory and are not
    accepted by any record-building function.
    """

    summary = choice_surface_summary(tokenizer, CHOICE_SURFACES)
    encoded = [_encode_surface(tokenizer, surface) for surface in CHOICE_SURFACES]
    if not summary["all_single_token"] or not summary["all_distinct"]:
        raise Phase7ContractError("BLOCK_CHOICE_SURFACE_NOT_SINGLE_TOKEN")
    ids = tuple(int(value[0]) for value in encoded)
    if len(set(ids)) != len(ids):
        raise Phase7ContractError("BLOCK_CHOICE_SURFACE_NOT_SINGLE_TOKEN")
    return summary, ids


def _as_float(value: Any) -> float:
    item = value.item() if hasattr(value, "item") else value
    result = float(item)
    if not math.isfinite(result):
        raise Phase7RuntimeError("non-finite scalar produced")
    return result


def _torch_choice_distribution(torch: Any, logits: Any, choice_ids: Sequence[int]) -> Any:
    if getattr(logits, "ndim", None) == 2:
        logits = logits[0]
    if getattr(logits, "ndim", None) != 1:
        raise Phase7RuntimeError("choice logits must be a one-dimensional vector")
    selected = logits[list(choice_ids)].to(dtype=torch.float64)
    shifted = selected - torch.max(selected)
    probabilities = torch.exp(shifted)
    probabilities = probabilities / torch.sum(probabilities)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise Phase7RuntimeError("choice softmax is non-finite")
    return probabilities


def _torch_entropy(torch: Any, probabilities: Any) -> Any:
    return -torch.sum(probabilities * torch.log(probabilities))


def _torch_kl(torch: Any, probabilities: Any, reference: Any) -> Any:
    return torch.sum(probabilities * (torch.log(probabilities) - torch.log(reference)))


def _torch_cosine(torch: Any, left: Any, right: Any) -> Any:
    left = left.reshape(-1).to(dtype=torch.float64)
    right = right.reshape(-1).to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not bool(torch.isfinite(denominator).item()) or float(denominator.item()) <= 0.0:
        raise Phase7RuntimeError("cosine metric received a zero/non-finite vector")
    return torch.sum(left * right) / denominator


def _torch_hidden_metrics(torch: Any, hidden: Any, final_hidden: Any) -> Tuple[float, float, float]:
    left = hidden.reshape(-1).to(dtype=torch.float64)
    right = final_hidden.reshape(-1).to(dtype=torch.float64)
    if left.shape != right.shape:
        raise Phase7RuntimeError("hidden metric vector shapes differ")
    rms = torch.sqrt(torch.mean((left - right) ** 2))
    cosine = _torch_cosine(torch, left, right)
    distance = 1.0 - cosine
    return _as_float(rms), _as_float(cosine), _as_float(distance)


def _torch_angle(torch: Any, left: Any, right: Any) -> float:
    cosine = _torch_cosine(torch, left, right)
    cosine = torch.clamp(cosine, min=-1.0, max=1.0)
    return _as_float(torch.acos(cosine) / math.pi)


def _native_logits_at_position(outputs: Any, probe_index: int) -> Any:
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    if logits is None or getattr(logits, "ndim", None) != 3:
        raise Phase7RuntimeError("causal-LM output must expose native logits with shape [1,S,V]")
    return logits[:, probe_index, :]


def _project_causal_lm_logits(torch: Any, model: Any, lm_head: Any, hidden: Any) -> Any:
    """Apply the frozen model's native post-FinalNorm logits path."""

    logits = lm_head(hidden)
    softcap = getattr(getattr(model, "config", None), "final_logit_softcapping", None)
    if softcap is not None:
        softcap = float(softcap)
        if not math.isfinite(softcap) or softcap <= 0.0:
            raise Phase7RuntimeError("final_logit_softcapping must be finite and positive")
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def _native_final_hidden_and_logits(
    torch: Any,
    model: Any,
    lm_head: Any,
    outputs: Any,
    capture: Mapping[str, Any],
    probe_index: int,
) -> Tuple[Any, Any]:
    """Close a causal-LM output without requiring ``last_hidden_state``."""

    native_norm = capture.get("normalized")
    if native_norm is None:
        raise Phase7RuntimeError("native FinalNorm output was not captured")
    final_hidden = select_position_vector(native_norm, probe_index)
    native_logits = _native_logits_at_position(outputs, probe_index)
    projected_logits = _project_causal_lm_logits(torch, model, lm_head, final_hidden)
    if tuple(getattr(projected_logits, "shape", ())) != tuple(getattr(native_logits, "shape", ())):
        raise Phase7RuntimeError("native and projected final logits shapes differ")
    if not bool(torch.allclose(projected_logits, native_logits, rtol=1e-4, atol=1e-5)):
        raise Phase7RuntimeError("native causal-LM logits do not close lm_head(native final hidden)")
    return final_hidden, native_logits


def _check_single_batch(inputs: Mapping[str, Any], sequence_length: int) -> None:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise Phase7RuntimeError("renderer inputs must contain ephemeral input_ids")
    shape = tuple(getattr(input_ids, "shape", ()))
    if len(shape) != 2 or shape[0] != 1:
        raise Phase7RuntimeError("producer requires batch_size=1")
    if shape[1] != sequence_length:
        raise Phase7RuntimeError("renderer sequence length differs from producer metadata")


def produce_record(
    model: Any,
    tokenizer: Any,
    inputs: Mapping[str, Any],
    *,
    model_key: str,
    model_revision: str,
    tokenizer_revision: str,
    canonical_identity: str,
    subject: str,
    sequence_length: int,
    probe_index: int,
    choice_admission: Optional[Tuple[Mapping[str, Any], Sequence[int]]] = None,
    runtime_dtype: str = RUNTIME_DTYPE,
) -> Dict[str, Any]:
    """Run exactly one native zero-loop forward and return scalar-only data."""

    if model_key not in MODEL_SPEC_BY_KEY:
        raise Phase7RuntimeError("unknown Phase 7 model_key")
    if runtime_dtype != RUNTIME_DTYPE:
        raise Phase7RuntimeError("Phase 7 runtime dtype must be bfloat16")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise Phase7RuntimeError("sequence_length must be a positive integer")
    if isinstance(probe_index, bool) or not isinstance(probe_index, int):
        raise Phase7RuntimeError("probe_index must be an integer")
    if probe_index != sequence_length - 1:
        raise Phase7RuntimeError("probe_index must be the final non-padding position")
    _check_single_batch(inputs, sequence_length)
    adapter = resolve_architecture(model, model_key=model_key)
    assert_no_loop_wrapper(model)
    final_norm = get_path(model, adapter.final_norm_path)
    lm_head = resolve_lm_head(model, adapter)
    if choice_admission is None:
        choice_summary, choice_ids = admit_choice_surfaces(tokenizer)
    else:
        choice_summary, choice_ids = choice_admission
        if not choice_summary.get("all_single_token") or not choice_summary.get("all_distinct"):
            raise Phase7RuntimeError("BLOCK_CHOICE_SURFACE_NOT_SINGLE_TOKEN")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only on HPC2.
        raise Phase7RuntimeError("trajectory producer requires torch on the execution host") from exc

    capture, hook_handle = register_raw_final_norm_hook(final_norm)
    try:
        with torch.inference_mode():
            outputs = model(
                **dict(inputs),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        hook_handle.remove()
    hidden_states = hidden_states_from_output(outputs)
    config_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    if config_layers < 1:
        raise Phase7RuntimeError("model config lacks num_hidden_layers")
    raw_boundaries = assemble_raw_boundaries(
        hidden_states,
        capture.get("raw"),
        config_layers,
        int(capture.get("count", 0)),
    )
    raw_vectors = tuple(select_position_vector(state, probe_index) for state in raw_boundaries)
    # The native model forward above intentionally runs in inference mode.  Its
    # hidden-state outputs are therefore inference tensors, and every native
    # FinalNorm/lm_head projection below must remain in inference mode as well;
    # otherwise PyTorch's autograd wrapper rejects an inference tensor.
    with torch.inference_mode():
        final_hidden, final_logits = _native_final_hidden_and_logits(
            torch,
            model,
            lm_head,
            outputs,
            capture,
            probe_index,
        )
        final_distribution = _torch_choice_distribution(torch, final_logits, choice_ids)
        boundaries = []
        for boundary_index, raw_vector in enumerate(raw_vectors):
            if boundary_index == config_layers:
                normalized = final_hidden
                choice_logits = final_logits
            else:
                normalized = final_norm(raw_vector)
                choice_logits = _project_causal_lm_logits(torch, model, lm_head, normalized)
            distribution = _torch_choice_distribution(torch, choice_logits, choice_ids)
            entropy = _torch_entropy(torch, distribution)
            divergence = _torch_kl(torch, distribution, final_distribution)
            rms, cosine, cosine_distance = _torch_hidden_metrics(torch, normalized, final_hidden)
            boundaries.append(
                {
                    "boundary_id": "B_%d" % boundary_index,
                    "choice_entropy": _as_float(entropy),
                    "kl_to_final": _as_float(divergence),
                    "hidden_rms_l2_to_final": rms,
                    "hidden_cosine_to_final": cosine,
                    "hidden_cosine_distance_to_final": cosine_distance,
                }
            )
        transitions = [
            {
                "transition_id": "T_%d" % index,
                "adjacent_angular_distance": _torch_angle(
                    torch, raw_vectors[index], raw_vectors[index + 1]
                ),
            }
            for index in range(config_layers)
        ]
    record = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "model_key": model_key,
        "model_repo": MODEL_SPEC_BY_KEY[model_key]["model_repo"],
        "model_revision": str(model_revision),
        "tokenizer_revision": str(tokenizer_revision),
        "dataset_repo": DATASET_REPO,
        "split": DATASET_SPLIT,
        "canonical_identity": str(canonical_identity),
        "subject": str(subject),
        "sequence_length": int(sequence_length),
        "probe_index": int(probe_index),
        "boundary_count": config_layers + 1,
        "transition_count": config_layers,
        "forward_count": 1,
        "loop_insertions": 0,
        "batch_size": 1,
        "use_cache": False,
        "generation": False,
        "probe_rule": "final_non_padding_token_of_rendered_Answer_prefix",
        "boundary_capture": "raw_B0_through_BL",
        "raw_final_boundary_capture": "temporary_final_norm_forward_pre_hook",
        "final_norm_path": adapter.final_norm_path,
        "lm_head_path": adapter.lm_head_path,
        "final_norm_prehook_calls": int(capture["count"]),
        "final_norm_closure": True,
        "double_norm_applied": False,
        "choice_surface_count": int(choice_summary["surface_count"]),
        "choice_surface_single_token": bool(choice_summary["all_single_token"]),
        "choice_surface_distinct": bool(choice_summary["all_distinct"]),
        "runtime_dtype": runtime_dtype,
        "boundaries": boundaries,
        "transitions": transitions,
    }
    validate_sanitized_record(
        record,
        expected_model_key=model_key,
        expected_layer_count=config_layers,
    )
    return record


__all__ = [
    "admit_choice_surfaces",
    "produce_record",
]
