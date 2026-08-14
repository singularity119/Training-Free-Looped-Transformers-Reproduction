"""Closed-world contracts for the LoopScope Phase 7 sidecar.

The module is intentionally dependency free.  It is imported by local tests and
CLI help on machines that do not have torch, transformers, or lm-eval installed.
The only persisted trajectory values are sanitized scalar fields; model inputs,
token ids, prompts, logits, probabilities, hidden tensors, and outcome fields
are rejected recursively.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TRAJECTORY_SCHEMA_VERSION = "loopscope.phase7.base-trajectory.v1"
V3_SCHEMA_VERSION = "loopscope.phase7.v3-analysis.v1"
ADMISSION_SCHEMA_VERSION = "loopscope.phase7.model-admission.v1"
METHOD_ID = "AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE"
METHOD_VERSION = "3.1.0"
DATASET_REPO = "cais/mmlu"
DATASET_SPLIT = "validation"
POPULATION_SIZE = 1531
CATEGORY_COUNT = 57
NUM_FEWSHOT = 5
FEWSHOT_SPLIT = "dev"
RUNTIME_DTYPE = "bfloat16"
CHOICE_SURFACES = (" A", " B", " C", " D")
WIDTHS = (3, 4, 5, 6)
FORMAL_BOOTSTRAP_REPLICATES = 2000
FORMAL_BOOTSTRAP_SEED = 20260801
RANK_FREQUENCY_THRESHOLD = 0.80

MODEL_SPECS: Tuple[Dict[str, str], ...] = (
    {
        "model_key": "qwen25_3b",
        "model_repo": "Qwen/Qwen2.5-3B",
        "architecture_family": "qwen2",
        "decoder_container": "model",
        "decoder_blocks": "model.layers",
        "embedding": "model.embed_tokens",
        "final_norm": "model.norm",
        "lm_head": "lm_head",
    },
    {
        "model_key": "llama32_3b",
        "model_repo": "meta-llama/Llama-3.2-3B",
        "architecture_family": "llama",
        "decoder_container": "model",
        "decoder_blocks": "model.layers",
        "embedding": "model.embed_tokens",
        "final_norm": "model.norm",
        "lm_head": "lm_head",
    },
    {
        "model_key": "gemma2_2b",
        "model_repo": "google/gemma-2-2b",
        "architecture_family": "gemma2",
        "decoder_container": "model",
        "decoder_blocks": "model.layers",
        "embedding": "model.embed_tokens",
        "final_norm": "model.norm",
        "lm_head": "lm_head",
    },
)
MODEL_SPEC_BY_KEY = {row["model_key"]: row for row in MODEL_SPECS}
MODEL_REPO_TO_KEY = {row["model_repo"]: row["model_key"] for row in MODEL_SPECS}


# These are field-name fragments, not value filters.  In particular, a model
# revision string is allowed because it identifies the loaded model snapshot;
# token ids and content digests are never allowed to become record fields.
FORBIDDEN_KEY_FRAGMENTS = (
    "gold",
    "label",
    "correct",
    "accuracy",
    "gain",
    "flip",
    "outcome",
    "prediction",
    "answer",
    "prompt",
    "question",
    "choice_text",
    "token_id",
    "input_id",
    "logit",
    "probs",
    "probabilit",
    "hidden_states",
    "hidden_vector",
    "hidden_tensor",
    "hidden_state_tensor",
    "tensor",
    "residual",
    "residual_tensor",
    "loop_residual",
    "generated",
)

RECORD_KEYS = frozenset(
    {
        "schema_version",
        "model_key",
        "model_repo",
        "model_revision",
        "tokenizer_revision",
        "dataset_repo",
        "split",
        "canonical_identity",
        "subject",
        "sequence_length",
        "probe_index",
        "boundary_count",
        "transition_count",
        "forward_count",
        "loop_insertions",
        "batch_size",
        "use_cache",
        "generation",
        "probe_rule",
        "boundary_capture",
        "raw_final_boundary_capture",
        "final_norm_path",
        "lm_head_path",
        "final_norm_prehook_calls",
        "final_norm_closure",
        "double_norm_applied",
        "choice_surface_count",
        "choice_surface_single_token",
        "choice_surface_distinct",
        "runtime_dtype",
        "boundaries",
        "transitions",
    }
)
BOUNDARY_KEYS = frozenset(
    {
        "boundary_id",
        "choice_entropy",
        "kl_to_final",
        "hidden_rms_l2_to_final",
        "hidden_cosine_to_final",
        "hidden_cosine_distance_to_final",
    }
)
TRANSITION_KEYS = frozenset({"transition_id", "adjacent_angular_distance"})


class Phase7ContractError(ValueError):
    """Fail-closed Phase 7 contract error."""


def _normal_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_")


def scan_forbidden_fields(value: Any, path: str = "$") -> None:
    """Reject information-barrier fields anywhere in a persisted payload."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normal_key(raw_key)
            if any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise Phase7ContractError(
                    "BLOCK_INFORMATION_BARRIER_VIOLATION: forbidden field %s.%s"
                    % (path, raw_key)
                )
            scan_forbidden_fields(child, "%s.%s" % (path, raw_key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            scan_forbidden_fields(child, "%s[%d]" % (path, index))


def require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase7ContractError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise Phase7ContractError("%s must be finite" % label)
    return result


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase7ContractError("%s must be a non-empty string" % label)
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    actual = set(value)
    wanted = set(expected)
    missing = sorted(wanted - actual)
    extra = sorted(actual - wanted)
    if missing or extra:
        raise Phase7ContractError(
            "%s keys differ: missing=%s extra=%s" % (label, missing, extra)
        )


def validate_sanitized_record(
    record: Mapping[str, Any],
    *,
    expected_model_key: Optional[str] = None,
    expected_layer_count: Optional[int] = None,
) -> None:
    """Validate one scalar-only trajectory record."""

    if not isinstance(record, Mapping):
        raise Phase7ContractError("trajectory record must be an object")
    scan_forbidden_fields(record)
    _require_exact_keys(record, RECORD_KEYS, "trajectory record")
    if record["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise Phase7ContractError("trajectory schema_version differs")
    model_key = require_nonempty_string(record["model_key"], "model_key")
    if model_key not in MODEL_SPEC_BY_KEY:
        raise Phase7ContractError("unknown Phase 7 model_key")
    if expected_model_key is not None and model_key != expected_model_key:
        raise Phase7ContractError("record model_key differs from expected model")
    expected_repo = MODEL_SPEC_BY_KEY[model_key]["model_repo"]
    if record["model_repo"] != expected_repo:
        raise Phase7ContractError("record model_repo is not the frozen exact repo")
    for field in ("model_revision", "tokenizer_revision", "dataset_repo", "split"):
        require_nonempty_string(record[field], field)
    if record["dataset_repo"] != DATASET_REPO or record["split"] != DATASET_SPLIT:
        raise Phase7ContractError("record dataset/split differs from Phase 7 contract")
    require_nonempty_string(record["canonical_identity"], "canonical_identity")
    require_nonempty_string(record["subject"], "subject")
    sequence_length = record["sequence_length"]
    probe_index = record["probe_index"]
    for field in (
        "sequence_length",
        "probe_index",
        "boundary_count",
        "transition_count",
        "forward_count",
        "loop_insertions",
        "batch_size",
        "final_norm_prehook_calls",
        "choice_surface_count",
    ):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise Phase7ContractError("%s must be an integer" % field)
    if sequence_length < 1 or probe_index != sequence_length - 1:
        raise Phase7ContractError("probe is not the final non-padding token")
    if record["forward_count"] != 1 or record["loop_insertions"] != 0:
        raise Phase7ContractError("record is not a one-forward zero-loop record")
    if record["batch_size"] != 1 or record["use_cache"] is not False:
        raise Phase7ContractError("batch_size/use_cache differs from contract")
    if record["generation"] is not False:
        raise Phase7ContractError("generation is forbidden for the probe")
    if record["choice_surface_count"] != 4:
        raise Phase7ContractError("choice surface count differs")
    if record["choice_surface_single_token"] is not True or record["choice_surface_distinct"] is not True:
        raise Phase7ContractError("choice surface admission is not closed")
    if record["final_norm_prehook_calls"] != 1:
        raise Phase7ContractError("FinalNorm pre-hook count must equal one")
    if record["final_norm_closure"] is not True or record["double_norm_applied"] is not False:
        raise Phase7ContractError("FinalNorm closure/double-norm contract differs")
    for field in (
        "probe_rule",
        "boundary_capture",
        "raw_final_boundary_capture",
        "final_norm_path",
        "lm_head_path",
        "runtime_dtype",
    ):
        require_nonempty_string(record[field], field)
    if record["runtime_dtype"] != RUNTIME_DTYPE:
        raise Phase7ContractError("runtime dtype differs from contract")
    boundary_count = record["boundary_count"]
    transition_count = record["transition_count"]
    if expected_layer_count is not None:
        if isinstance(expected_layer_count, bool) or expected_layer_count < 1:
            raise Phase7ContractError("expected layer count is invalid")
        if boundary_count != expected_layer_count + 1 or transition_count != expected_layer_count:
            raise Phase7ContractError("boundary/transition count differs from expected L")
    elif boundary_count != transition_count + 1:
        raise Phase7ContractError("boundary/transition counts are not closed")
    boundaries = record["boundaries"]
    transitions = record["transitions"]
    if not isinstance(boundaries, Sequence) or isinstance(boundaries, (str, bytes)):
        raise Phase7ContractError("boundaries must be a sequence")
    if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)):
        raise Phase7ContractError("transitions must be a sequence")
    if len(boundaries) != boundary_count or len(transitions) != transition_count:
        raise Phase7ContractError("boundary/transition array lengths differ")
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise Phase7ContractError("boundary %d is not an object" % index)
        _require_exact_keys(boundary, BOUNDARY_KEYS, "boundary %d" % index)
        if boundary["boundary_id"] != "B_%d" % index:
            raise Phase7ContractError("boundary ids are not canonical")
        for field in (
            "choice_entropy",
            "kl_to_final",
            "hidden_rms_l2_to_final",
            "hidden_cosine_to_final",
            "hidden_cosine_distance_to_final",
        ):
            require_finite(boundary[field], "boundaries[%d].%s" % (index, field))
        if boundary["choice_entropy"] < 0.0 or boundary["kl_to_final"] < 0.0:
            raise Phase7ContractError("choice entropy/KL cannot be negative")
    if abs(float(boundaries[-1]["kl_to_final"])) > 1e-9:
        raise Phase7ContractError("D_L does not close to zero")
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            raise Phase7ContractError("transition %d is not an object" % index)
        _require_exact_keys(transition, TRANSITION_KEYS, "transition %d" % index)
        if transition["transition_id"] != "T_%d" % index:
            raise Phase7ContractError("transition ids are not canonical")
        require_finite(
            transition["adjacent_angular_distance"],
            "transitions[%d].adjacent_angular_distance" % index,
        )
        if not 0.0 <= float(transition["adjacent_angular_distance"]) <= 1.0:
            raise Phase7ContractError("adjacent angular distance is outside [0,1]")


def model_spec(model_key: str) -> Dict[str, str]:
    try:
        return dict(MODEL_SPEC_BY_KEY[model_key])
    except KeyError as exc:
        raise Phase7ContractError("unknown Phase 7 model_key: %s" % model_key) from exc


def candidate_domain(layer_count: int) -> Dict[str, Any]:
    """Generate the frozen central-domain candidate table for an arbitrary L."""

    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 1:
        raise Phase7ContractError("layer_count must be a positive integer")
    trim = (3 * layer_count + 9) // 10  # exact ceil(0.30 * L)
    central_start = trim
    central_end = layer_count - trim - 1
    if central_end < central_start:
        raise Phase7ContractError("central domain is empty for layer_count=%d" % layer_count)
    candidates: List[Dict[str, Any]] = []
    starts_by_width: Dict[str, List[int]] = {}
    for width in WIDTHS:
        starts = list(range(central_start, central_end - width + 2))
        starts_by_width[str(width)] = starts
        for start in starts:
            candidates.append(
                {
                    "width": width,
                    "start": start,
                    "stop_exclusive": start + width,
                    "end_inclusive": start + width - 1,
                    "window_half_open": "%d:%d" % (start, start + width),
                    "window_layers_inclusive": "%d:%d" % (start, start + width - 1),
                    "boundary_entry": "B_%d" % start,
                    "boundary_exit": "B_%d" % (start + width),
                }
            )
    expected_pairs = [
        (width, start)
        for width in WIDTHS
        for start in starts_by_width[str(width)]
    ]
    actual_pairs = [(row["width"], row["start"]) for row in candidates]
    if actual_pairs != expected_pairs or len(actual_pairs) != len(set(actual_pairs)):
        raise Phase7ContractError("BLOCK_CANDIDATE_DOMAIN_MISMATCH")
    return {
        "layer_count": layer_count,
        "trim_per_side": trim,
        "central_blocks": [central_start, central_end],
        "widths": list(WIDTHS),
        "starts_by_width": starts_by_width,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def validate_candidate_domain(domain: Mapping[str, Any]) -> None:
    if not isinstance(domain, Mapping):
        raise Phase7ContractError("candidate domain must be an object")
    generated = candidate_domain(int(domain["layer_count"]))
    for key in (
        "layer_count",
        "trim_per_side",
        "central_blocks",
        "widths",
        "starts_by_width",
        "candidate_count",
        "candidates",
    ):
        if domain.get(key) != generated[key]:
            raise Phase7ContractError("candidate domain differs at %s" % key)


def choice_surface_summary(tokenizer: Any, surfaces: Sequence[str] = CHOICE_SURFACES) -> Dict[str, Any]:
    """Check surface lengths and uniqueness without returning token ids."""

    lengths: List[int] = []
    one_token_values: List[Any] = []
    for surface in surfaces:
        if not isinstance(surface, str):
            raise Phase7ContractError("choice surface must be a string")
        if hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(surface, add_special_tokens=False)
        else:
            encoded_payload = tokenizer(surface, add_special_tokens=False)
            encoded = encoded_payload["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
            raise Phase7ContractError("tokenizer did not return a token sequence")
        lengths.append(len(encoded))
        if len(encoded) == 1:
            # Keep the token value in local memory only.  The result below
            # exposes the boolean closure, never the id itself.
            one_token_values.append(encoded[0])
    return {
        "surface_count": len(surfaces),
        "single_token_lengths": list(lengths),
        "all_single_token": all(length == 1 for length in lengths),
        # Equality is checked in-memory by the producer; only the boolean is
        # allowed to cross the persistence boundary.
        "all_distinct": bool(
            all(length == 1 for length in lengths)
            and len(set(one_token_values)) == len(one_token_values)
        ),
    }


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "BOUNDARY_KEYS",
    "CATEGORY_COUNT",
    "CHOICE_SURFACES",
    "DATASET_REPO",
    "DATASET_SPLIT",
    "FORMAL_BOOTSTRAP_REPLICATES",
    "FORMAL_BOOTSTRAP_SEED",
    "METHOD_ID",
    "METHOD_VERSION",
    "MODEL_REPO_TO_KEY",
    "MODEL_SPECS",
    "MODEL_SPEC_BY_KEY",
    "NUM_FEWSHOT",
    "Phase7ContractError",
    "POPULATION_SIZE",
    "RANK_FREQUENCY_THRESHOLD",
    "RECORD_KEYS",
    "RUNTIME_DTYPE",
    "TRAJECTORY_SCHEMA_VERSION",
    "TRANSITION_KEYS",
    "V3_SCHEMA_VERSION",
    "WIDTHS",
    "candidate_domain",
    "choice_surface_summary",
    "model_spec",
    "scan_forbidden_fields",
    "validate_candidate_domain",
    "validate_sanitized_record",
]
