"""Closed, outcome-blind local contracts for the Phase 6 MMLU 0-shot branch.

The module is intentionally standard-library only.  It contains pure reducers for
four-choice probabilities and hidden-state geometry, but never loads a model,
dataset, tokenizer implementation, or evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CARD_SCHEMA_VERSION = "loopscope.phase6.mmlu0-prefix-card.v1"
TRAJECTORY_SCHEMA_VERSION = "loopscope.phase6.mmlu0-prefix-trajectory.v1"
METHOD = "MMLU_ZERO_SHOT_PREFIX_TRAJECTORY_RBR_V2"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_CONFIG_SHA256 = "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
LAYERS = 36
BOUNDARY_COUNT = LAYERS + 1
HIDDEN_SIZE = 2560
VALIDATION_RECORD_COUNT = 1531
VALIDATION_SUBJECT_COUNT = 57
FORMAL_REPLICATES = 2000
FORMAL_SEED = 20260801
FREQUENCY_THRESHOLD = 0.80
CHOICE_KEYS = ("A", "B", "C", "D")
CHOICE_SURFACES = {key: " %s" % key for key in CHOICE_KEYS}
CHOICE_TOKEN_IDS = {"A": 362, "B": 425, "C": 356, "D": 422}
BOUNDARY_IDS = tuple("B_%d" % index for index in range(BOUNDARY_COUNT))
WIDTHS = (3, 4, 5, 6)
STARTS = {
    3: tuple(range(11, 23)),
    4: tuple(range(11, 22)),
    5: tuple(range(11, 21)),
    6: tuple(range(11, 20)),
}
LEGAL_SELECTOR_STATES = (
    "SELECTED_WINDOW",
    "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
    "ABSTAIN_NO_UNIQUE_TOP1",
    "ABSTAIN_RATE_RANK_UNSTABLE",
)


class MMLU0ContractError(ValueError):
    """Fail-closed contract error for Gate H's local MMLU 0-shot layer."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MMLU0ContractError("JSON root must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise MMLU0ContractError(
            "%s keys differ: expected %s, observed %s"
            % (label, sorted(expected_set), sorted(observed))
        )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MMLU0ContractError("%s must be a finite number" % label)
    result = float(value)
    if not math.isfinite(result):
        raise MMLU0ContractError("%s must be finite" % label)
    return result


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MMLU0ContractError("%s must be an integer" % label)
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise MMLU0ContractError("%s must be a SHA-256 hex string" % label)
    try:
        int(value, 16)
    except ValueError as exc:
        raise MMLU0ContractError("%s must be a SHA-256 hex string" % label) from exc
    if value != value.lower():
        raise MMLU0ContractError("%s must use lowercase hexadecimal" % label)
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MMLU0ContractError("%s must be a non-empty string" % label)
    return value


def _sequence(value: Any, label: str, length: int | None = None) -> List[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MMLU0ContractError("%s must be a sequence" % label)
    result = list(value)
    if length is not None and len(result) != length:
        raise MMLU0ContractError("%s must contain %d values" % (label, length))
    return result


def _close(left: Any, right: Any, label: str, tolerance: float = 1e-10) -> None:
    a = _finite(left, label)
    b = _finite(right, label)
    if abs(a - b) > tolerance:
        raise MMLU0ContractError("%s is not recomputed exactly" % label)


def expected_candidate_pairs() -> Tuple[Tuple[int, int], ...]:
    return tuple((width, start) for width in WIDTHS for start in STARTS[width])


def choice_surface_manifest() -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "surface": CHOICE_SURFACES[key],
            "token_ids": [CHOICE_TOKEN_IDS[key]],
            "decoded": CHOICE_SURFACES[key],
        }
        for key in CHOICE_KEYS
    }


CHOICE_SURFACE_MANIFEST_SHA256 = semantic_sha256(choice_surface_manifest())


def validate_choice_surface_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise MMLU0ContractError("choice surface manifest must be an object")
    _keys(manifest, CHOICE_KEYS, "choice surface manifest")
    ids: List[int] = []
    surfaces: List[str] = []
    for key in CHOICE_KEYS:
        entry = manifest[key]
        if not isinstance(entry, Mapping):
            raise MMLU0ContractError("choice surface %s must be an object" % key)
        _keys(entry, ("surface", "token_ids", "decoded"), "choice surface %s" % key)
        surface = entry["surface"]
        decoded = entry["decoded"]
        if surface != CHOICE_SURFACES[key] or decoded != surface:
            raise MMLU0ContractError("choice surface %s text differs" % key)
        token_ids = _sequence(entry["token_ids"], "choice surface %s token_ids" % key, 1)
        token_id = _strict_int(token_ids[0], "choice surface %s token id" % key)
        if token_id != CHOICE_TOKEN_IDS[key]:
            raise MMLU0ContractError("choice surface %s token id differs" % key)
        ids.append(token_id)
        surfaces.append(surface)
    if len(set(ids)) != len(ids) or len(set(surfaces)) != len(surfaces):
        raise MMLU0ContractError("choice continuation surfaces and token IDs must be unique")
    if semantic_sha256(manifest) != CHOICE_SURFACE_MANIFEST_SHA256:
        raise MMLU0ContractError("choice surface manifest hash differs")


def freeze_choice_surface_manifest(tokenizer: Any) -> Dict[str, Dict[str, Any]]:
    """Close the four one-token continuations using an already supplied tokenizer.

    The caller owns tokenizer construction.  This helper only calls encode/decode and
    never imports or downloads a tokenizer itself.
    """

    manifest: Dict[str, Dict[str, Any]] = {}
    for key in CHOICE_KEYS:
        surface = CHOICE_SURFACES[key]
        token_ids = list(tokenizer.encode(surface, add_special_tokens=False))
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        manifest[key] = {"surface": surface, "token_ids": token_ids, "decoded": decoded}
    validate_choice_surface_manifest(manifest)
    return manifest


def validate_card(card: Mapping[str, Any]) -> None:
    if not isinstance(card, Mapping):
        raise MMLU0ContractError("card must be an object")
    _keys(
        card,
        (
            "schema_version",
            "card",
            "status",
            "phase",
            "model",
            "task",
            "evaluator",
            "renderer",
            "choice_surfaces",
            "trajectory",
            "metrics",
            "selector",
            "loop_identity_suggestion",
            "information_barrier",
        ),
        "card",
    )
    if card["schema_version"] != CARD_SCHEMA_VERSION or card["card"] != METHOD:
        raise MMLU0ContractError("card identity differs")
    if card["status"] != "GATE_H_LOCAL_CONTRACT_FROZEN":
        raise MMLU0ContractError("card status differs")

    phase = card["phase"]
    if not isinstance(phase, Mapping):
        raise MMLU0ContractError("phase must be an object")
    _keys(phase, ("name", "hypothesis", "gate", "planning_thread", "executor_thread", "terminal_scope"), "phase")
    if phase["name"] != "LoopScope Phase 6" or phase["hypothesis"] != "H6_MMLU_ZERO_SHOT_PREFIX_CHOICE_TRAJECTORY_RBR_V2":
        raise MMLU0ContractError("phase identity differs")
    if phase["gate"] != "H":
        raise MMLU0ContractError("phase gate differs")
    if phase["planning_thread"] != "019fb3de-2298-75f2-a083-0dca453ea79c":
        raise MMLU0ContractError("planning thread differs")
    if phase["executor_thread"] != "019fbecf-be9a-71e3-94f0-a82356b79d29":
        raise MMLU0ContractError("executor thread differs")

    model = card["model"]
    if not isinstance(model, Mapping):
        raise MMLU0ContractError("model must be an object")
    _keys(model, ("repo_id", "revision", "tokenizer_revision", "decoder_layers", "hidden_size", "dtype", "config_sha256", "tokenizer_config_sha256", "tokenizer_json_sha256", "tokenizer_class"), "model")
    if model["repo_id"] != "Qwen/Qwen3-4B-Instruct-2507" or model["revision"] != MODEL_REVISION or model["tokenizer_revision"] != MODEL_REVISION:
        raise MMLU0ContractError("model revision differs")
    if model["decoder_layers"] != LAYERS or model["hidden_size"] != HIDDEN_SIZE or model["dtype"] != "bfloat16":
        raise MMLU0ContractError("model shape or dtype differs")
    for key in ("config_sha256", "tokenizer_config_sha256", "tokenizer_json_sha256"):
        _sha256(model[key], "model.%s" % key)
    if model["config_sha256"] != MODEL_CONFIG_SHA256:
        raise MMLU0ContractError("model config SHA-256 differs")

    task = card["task"]
    if not isinstance(task, Mapping):
        raise MMLU0ContractError("task must be an object")
    _keys(task, ("dataset", "revision", "task_name", "validation_split", "validation_record_count", "validation_subject_count", "lm_eval_task_source", "validation_identity_reuse"), "task")
    if task["dataset"] != "cais/mmlu" or task["revision"] != DATASET_REVISION or task["task_name"] != "mmlu":
        raise MMLU0ContractError("task identity differs")
    if task["validation_split"] != "validation" or task["validation_record_count"] != VALIDATION_RECORD_COUNT or task["validation_subject_count"] != VALIDATION_SUBJECT_COUNT:
        raise MMLU0ContractError("validation population differs")
    source = task["lm_eval_task_source"]
    if not isinstance(source, Mapping):
        raise MMLU0ContractError("lm-eval task source must be an object")
    _keys(source, ("version", "yaml_path", "yaml_sha256", "default_template_sha256", "stem_group_yaml_sha256", "other_group_yaml_sha256", "social_sciences_group_yaml_sha256", "humanities_group_yaml_sha256", "default_source_file_count", "default_source_manifest_sha256"), "lm-eval task source")
    if source["version"] != "0.4.11" or source["yaml_path"] != "lm_eval/tasks/mmlu/default/_mmlu.yaml" or source["default_source_file_count"] != 63:
        raise MMLU0ContractError("lm-eval task source identity differs")
    for key in ("yaml_sha256", "default_template_sha256", "stem_group_yaml_sha256", "other_group_yaml_sha256", "social_sciences_group_yaml_sha256", "humanities_group_yaml_sha256", "default_source_manifest_sha256"):
        _sha256(source[key], "lm-eval task source.%s" % key)
    reuse = task["validation_identity_reuse"]
    if not isinstance(reuse, Mapping):
        raise MMLU0ContractError("identity-only reuse must be an object")
    _keys(reuse, ("source", "validation_pool_manifest_file_sha256", "validation_pool_manifest_internal_sha256", "validation_ordered_identity_sha256", "prompt_hash_reuse", "fresh_zero_shot_prompt_projection_required"), "validation identity reuse")
    if reuse["prompt_hash_reuse"] is not False or reuse["fresh_zero_shot_prompt_projection_required"] is not True:
        raise MMLU0ContractError("prompt reuse boundary differs")
    for key in ("validation_pool_manifest_file_sha256", "validation_pool_manifest_internal_sha256", "validation_ordered_identity_sha256"):
        _sha256(reuse[key], "validation identity reuse.%s" % key)

    evaluator = card["evaluator"]
    if not isinstance(evaluator, Mapping):
        raise MMLU0ContractError("evaluator must be an object")
    _keys(evaluator, ("lm_eval_version", "transformers_version", "tokenizers_version", "datasets_version", "torch_version", "accelerate_version", "request_type", "num_fewshot", "plain_prompt", "apply_chat_template", "fewshot_as_multiturn", "generation", "direct_letter_fallback"), "evaluator")
    expected_eval = {"lm_eval_version": "0.4.11", "transformers_version": "4.51.3", "tokenizers_version": "0.21.4", "datasets_version": "5.0.0", "torch_version": "2.3.1+cu121", "accelerate_version": "1.14.0", "request_type": "loglikelihood", "num_fewshot": 0, "plain_prompt": True, "apply_chat_template": False, "fewshot_as_multiturn": False, "generation": False, "direct_letter_fallback": False}
    for key, expected in expected_eval.items():
        if evaluator[key] != expected:
            raise MMLU0ContractError("evaluator.%s differs" % key)

    renderer = card["renderer"]
    if not isinstance(renderer, Mapping):
        raise MMLU0ContractError("renderer must be an object")
    _keys(renderer, ("contract_version", "terminal", "semantic_contract", "semantic_contract_sha256", "zero_shot_prompt_projection_sha256", "zero_shot_prompt_projection_required_before_gate_i"), "renderer")
    if renderer["contract_version"] != "loopscope.mmlu-0shot-prefix-render.v1" or renderer["terminal"] != "standard task renderer terminal":
        raise MMLU0ContractError("renderer identity differs")
    semantic_contract = renderer["semantic_contract"]
    if not isinstance(semantic_contract, Mapping):
        raise MMLU0ContractError("renderer semantic contract must be an object")
    _keys(semantic_contract, ("task_group", "task_renderer", "num_fewshot", "fewshot_split", "plain_prompt", "apply_chat_template", "fewshot_as_multiturn", "generation", "direct_letter_fallback", "choice_surfaces", "terminal", "probe_position"), "renderer semantic contract")
    expected_semantics = {"task_group": "mmlu", "task_renderer": "standard lm-eval task renderer", "num_fewshot": 0, "fewshot_split": None, "plain_prompt": True, "apply_chat_template": False, "fewshot_as_multiturn": False, "generation": False, "direct_letter_fallback": False, "choice_surfaces": [" A", " B", " C", " D"], "terminal": "standard task renderer terminal", "probe_position": "last_non_padding_rendered_prefix_token"}
    for key, expected in expected_semantics.items():
        if semantic_contract[key] != expected:
            raise MMLU0ContractError("renderer semantic contract.%s differs" % key)
    if semantic_sha256(semantic_contract) != renderer["semantic_contract_sha256"]:
        raise MMLU0ContractError("renderer semantic contract hash differs")
    if renderer["zero_shot_prompt_projection_sha256"] is not None or renderer["zero_shot_prompt_projection_required_before_gate_i"] is not True:
        raise MMLU0ContractError("zero-shot prompt projection boundary differs")

    validate_choice_surface_manifest(card["choice_surfaces"])

    trajectory = card["trajectory"]
    if not isinstance(trajectory, Mapping):
        raise MMLU0ContractError("trajectory must be an object")
    _keys(trajectory, ("boundary_ids", "boundary_count", "raw_boundary_definition", "final_norm_definition", "probe_position", "forward_passes_per_record", "native_no_loop", "use_cache", "hidden_diagnostics_only", "persist_raw_logits", "persist_full_vocab_probabilities", "persist_hidden_tensors"), "trajectory")
    if tuple(trajectory["boundary_ids"]) != BOUNDARY_IDS or trajectory["boundary_count"] != BOUNDARY_COUNT:
        raise MMLU0ContractError("boundary domain differs")
    if trajectory["probe_position"] != "last_non_padding_rendered_prefix_token" or trajectory["forward_passes_per_record"] != 1 or trajectory["native_no_loop"] is not True or trajectory["use_cache"] is not False:
        raise MMLU0ContractError("trajectory execution semantics differ")
    if any(trajectory[key] is not False for key in ("persist_raw_logits", "persist_full_vocab_probabilities", "persist_hidden_tensors")):
        raise MMLU0ContractError("trajectory information boundary differs")

    metrics = card["metrics"]
    if not isinstance(metrics, Mapping):
        raise MMLU0ContractError("metrics must be an object")
    _keys(metrics, ("choice_entropy", "kl_to_final", "hidden_diagnostics", "adjacent_diagnostic", "selector_fields", "diagnostic_fields_not_in_selector"), "metrics")
    if metrics["selector_fields"] != ["H", "D"]:
        raise MMLU0ContractError("selector metrics differ")
    if set(metrics["selector_fields"]) & set(metrics["diagnostic_fields_not_in_selector"]):
        raise MMLU0ContractError("diagnostic fields entered selector")

    selector = card["selector"]
    if not isinstance(selector, Mapping):
        raise MMLU0ContractError("selector must be an object")
    _keys(selector, ("method", "central_blocks_inclusive", "widths", "candidate_starts", "candidate_count", "bootstrap", "projection_fields", "projection_excludes", "legal_terminal_states", "formal_execution_deferred_to"), "selector")
    if selector["method"] != "RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE" or selector["central_blocks_inclusive"] != [11, 24] or selector["widths"] != [3, 4, 5, 6] or selector["candidate_count"] != 42:
        raise MMLU0ContractError("selector domain differs")
    observed_starts = {int(key): tuple(value) for key, value in selector["candidate_starts"].items()}
    if observed_starts != STARTS:
        raise MMLU0ContractError("selector candidate starts differ")
    bootstrap = selector["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise MMLU0ContractError("selector bootstrap must be an object")
    _keys(bootstrap, ("replicates", "seed", "population_stratification", "standard_error_ddof", "frequency_threshold"), "selector bootstrap")
    if bootstrap["replicates"] != FORMAL_REPLICATES or bootstrap["seed"] != FORMAL_SEED or bootstrap["population_stratification"] != "subject-stratified joint bootstrap" or bootstrap["standard_error_ddof"] != 1 or bootstrap["frequency_threshold"] != FREQUENCY_THRESHOLD:
        raise MMLU0ContractError("selector bootstrap differs")
    if selector["projection_fields"] != ["identity", "category", "H", "D"] or selector["formal_execution_deferred_to"] != "Gate J":
        raise MMLU0ContractError("selector projection or phase boundary differs")
    if set(("hidden_rms_l2_to_final", "hidden_cosine_to_final", "hidden_cosine_distance_to_final", "adjacent_angular_distance")) & set(selector["projection_fields"]):
        raise MMLU0ContractError("hidden diagnostics entered selector")
    if selector["legal_terminal_states"] != list(LEGAL_SELECTOR_STATES):
        raise MMLU0ContractError("selector terminal states differ")

    loop = card["loop_identity_suggestion"]
    if not isinstance(loop, Mapping):
        raise MMLU0ContractError("loop identity suggestion must be an object")
    _keys(loop, ("authorized", "k", "block_euler_step", "horizon", "cache_strategy", "note"), "loop identity suggestion")
    if loop["authorized"] is not False:
        raise MMLU0ContractError("loop is not authorized in Gate H")
    barrier = card["information_barrier"]
    if not isinstance(barrier, Mapping):
        raise MMLU0ContractError("information barrier must be an object")
    _keys(barrier, ("gate_h_may_read", "gate_h_must_not_read", "no_model_forward", "no_generation", "no_cuda_gpu_slurm", "no_panel", "no_full_loop_outcome"), "information barrier")
    for key in ("no_model_forward", "no_generation", "no_cuda_gpu_slurm", "no_panel", "no_full_loop_outcome"):
        if barrier[key] is not True:
            raise MMLU0ContractError("information barrier.%s differs" % key)


def validate_schema_document(schema: Mapping[str, Any]) -> None:
    if not isinstance(schema, Mapping):
        raise MMLU0ContractError("trajectory schema must be an object")
    if schema.get("$id") != "loopscope.phase6.mmlu0-prefix-trajectory.v1" or schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise MMLU0ContractError("trajectory schema identity differs")
    required = schema.get("required")
    expected_required = ["schema_version", "identity", "subject", "split", "prompt_sha256", "renderer_provenance", "boundaries", "adjacent_angular_distance"]
    if required != expected_required:
        raise MMLU0ContractError("trajectory schema required fields differ")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != set(expected_required):
        raise MMLU0ContractError("trajectory schema properties differ")
    if properties["schema_version"].get("const") != TRAJECTORY_SCHEMA_VERSION or properties["split"].get("const") != "validation":
        raise MMLU0ContractError("trajectory schema constants differ")
    boundaries = properties["boundaries"]
    angular = properties["adjacent_angular_distance"]
    if boundaries.get("minItems") != BOUNDARY_COUNT or boundaries.get("maxItems") != BOUNDARY_COUNT or angular.get("minItems") != LAYERS or angular.get("maxItems") != LAYERS:
        raise MMLU0ContractError("trajectory schema dimensions differ")
    for node in (properties["identity"], properties["renderer_provenance"], boundaries["items"], angular["items"]):
        if node.get("additionalProperties") is not False:
            raise MMLU0ContractError("trajectory schema is not closed")


def choice_logits_to_probabilities(logits: Sequence[Any]) -> List[float]:
    values = [_finite(value, "choice logit") for value in _sequence(logits, "choice logits", 4)]
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = math.fsum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise MMLU0ContractError("choice softmax normalization failed")
    return [value / total for value in exponentials]


def _validate_probability_vector(values: Any, label: str) -> List[float]:
    result = [_finite(value, label) for value in _sequence(values, label, 4)]
    if any(value <= 0.0 for value in result):
        raise MMLU0ContractError("%s must be strictly positive" % label)
    if abs(math.fsum(result) - 1.0) > 1e-9:
        raise MMLU0ContractError("%s must sum to one" % label)
    return result


def choice_entropy(probabilities: Sequence[Any]) -> float:
    values = _validate_probability_vector(probabilities, "choice probabilities")
    return -math.fsum(value * math.log(value) for value in values)


def kl_to_final(probabilities: Sequence[Any], final_probabilities: Sequence[Any]) -> float:
    values = _validate_probability_vector(probabilities, "choice probabilities")
    final = _validate_probability_vector(final_probabilities, "final choice probabilities")
    value = math.fsum(p * math.log(p / q) for p, q in zip(values, final))
    if value < -1e-12:
        raise MMLU0ContractError("KL divergence cannot be negative")
    return max(0.0, value)


def choice_trajectory_metrics(choice_probabilities: Sequence[Sequence[Any]]) -> Dict[str, List[float]]:
    values = [_validate_probability_vector(row, "choice probability boundary") for row in _sequence(choice_probabilities, "choice probability trajectory", BOUNDARY_COUNT)]
    final = values[-1]
    return {
        "H": [choice_entropy(row) for row in values],
        "D": [kl_to_final(row, final) for row in values],
    }


def _vector(value: Any, label: str) -> List[float]:
    result = [_finite(item, label) for item in _sequence(value, label)]
    if not result:
        raise MMLU0ContractError("%s cannot be empty" % label)
    return result


def hidden_geometry_to_final(hidden_states: Sequence[Sequence[Any]]) -> Dict[str, List[float]]:
    states = [_vector(value, "hidden state") for value in _sequence(hidden_states, "hidden trajectory", BOUNDARY_COUNT)]
    dimension = len(states[0])
    if any(len(state) != dimension for state in states):
        raise MMLU0ContractError("hidden state dimensions differ")
    final = states[-1]
    final_norm = math.sqrt(math.fsum(value * value for value in final))
    if final_norm <= 0.0:
        raise MMLU0ContractError("final hidden state must have non-zero norm")
    rms_l2: List[float] = []
    cosine: List[float] = []
    distance: List[float] = []
    for state in states:
        delta = [left - right for left, right in zip(state, final)]
        rms_l2.append(math.sqrt(math.fsum(value * value for value in delta) / dimension))
        norm = math.sqrt(math.fsum(value * value for value in state))
        if norm <= 0.0:
            raise MMLU0ContractError("hidden state must have non-zero norm")
        cos = math.fsum(left * right for left, right in zip(state, final)) / (norm * final_norm)
        cos = min(1.0, max(-1.0, cos))
        cosine.append(cos)
        distance.append(1.0 - cos)
    return {
        "hidden_rms_l2_to_final": rms_l2,
        "hidden_cosine_to_final": cosine,
        "hidden_cosine_distance_to_final": distance,
    }


def adjacent_angular_distance(hidden_states: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    states = [_vector(value, "hidden state") for value in _sequence(hidden_states, "hidden trajectory", BOUNDARY_COUNT)]
    dimension = len(states[0])
    if any(len(state) != dimension for state in states):
        raise MMLU0ContractError("hidden state dimensions differ")
    result: List[Dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(states[:-1], states[1:])):
        left_norm = math.sqrt(math.fsum(value * value for value in left))
        right_norm = math.sqrt(math.fsum(value * value for value in right))
        if left_norm <= 0.0 or right_norm <= 0.0:
            raise MMLU0ContractError("adjacent hidden states must have non-zero norms")
        cosine = math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        cosine = min(1.0, max(-1.0, cosine))
        result.append({"transition_id": "B_%d->B_%d" % (index, index + 1), "angle_over_pi": math.acos(cosine) / math.pi})
    return result


def compute_trajectory_metrics(
    choice_probabilities: Sequence[Sequence[Any]],
    hidden_states: Sequence[Sequence[Any]],
    *,
    angular_hidden_states: Sequence[Sequence[Any]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reduce one trajectory while keeping raw-angle provenance explicit.

    The default keeps the historical Gate H behavior.  Gate I passes the
    final-normalized states for hidden-to-final diagnostics and the raw
    residual states for adjacent angular distance, matching the frozen card.
    """

    choice = choice_trajectory_metrics(choice_probabilities)
    hidden = hidden_geometry_to_final(hidden_states)
    angular = adjacent_angular_distance(
        hidden_states if angular_hidden_states is None else angular_hidden_states
    )
    boundaries: List[Dict[str, Any]] = []
    for index, boundary_id in enumerate(BOUNDARY_IDS):
        boundaries.append({
            "boundary_id": boundary_id,
            "choice_probabilities": list(_validate_probability_vector(choice_probabilities[index], "choice probability boundary")),
            "choice_entropy": choice["H"][index],
            "kl_to_final": choice["D"][index],
            "hidden_rms_l2_to_final": hidden["hidden_rms_l2_to_final"][index],
            "hidden_cosine_to_final": hidden["hidden_cosine_to_final"][index],
            "hidden_cosine_distance_to_final": hidden["hidden_cosine_distance_to_final"][index],
        })
    return boundaries, angular


def _validate_renderer_provenance(provenance: Mapping[str, Any]) -> None:
    _keys(provenance, ("contract_version", "semantic_contract_sha256", "task_source_manifest_sha256", "choice_surface_manifest_sha256"), "renderer provenance")
    if provenance["contract_version"] != "loopscope.mmlu-0shot-prefix-render.v1":
        raise MMLU0ContractError("renderer provenance contract differs")
    for key in ("semantic_contract_sha256", "task_source_manifest_sha256", "choice_surface_manifest_sha256"):
        _sha256(provenance[key], "renderer provenance.%s" % key)
    if provenance["task_source_manifest_sha256"] != "7e03a4bac9704839de842c099e068fb2ebba626a066adfc3499c69eb05336b68":
        raise MMLU0ContractError("renderer task source hash differs")
    if provenance["choice_surface_manifest_sha256"] != CHOICE_SURFACE_MANIFEST_SHA256:
        raise MMLU0ContractError("renderer choice surface hash differs")


def validate_trajectory_record(record: Mapping[str, Any], card: Mapping[str, Any] | None = None) -> None:
    if not isinstance(record, Mapping):
        raise MMLU0ContractError("trajectory record must be an object")
    _keys(record, ("schema_version", "identity", "subject", "split", "prompt_sha256", "renderer_provenance", "boundaries", "adjacent_angular_distance"), "trajectory record")
    if record["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise MMLU0ContractError("trajectory schema version differs")
    identity = record["identity"]
    if not isinstance(identity, Mapping):
        raise MMLU0ContractError("trajectory identity must be an object")
    _keys(identity, ("task", "doc_id", "doc_hash"), "trajectory identity")
    if identity["task"] != "mmlu":
        raise MMLU0ContractError("trajectory task differs")
    _nonempty_string(identity["doc_id"], "identity.doc_id")
    _sha256(identity["doc_hash"], "identity.doc_hash")
    _nonempty_string(record["subject"], "trajectory subject")
    if record["split"] != "validation":
        raise MMLU0ContractError("trajectory split differs")
    _sha256(record["prompt_sha256"], "prompt_sha256")
    provenance = record["renderer_provenance"]
    if not isinstance(provenance, Mapping):
        raise MMLU0ContractError("renderer provenance must be an object")
    _validate_renderer_provenance(provenance)
    if card is not None:
        validate_card(card)
        if provenance["semantic_contract_sha256"] != card["renderer"]["semantic_contract_sha256"]:
            raise MMLU0ContractError("record renderer semantic hash differs from card")
    boundaries = _sequence(record["boundaries"], "trajectory boundaries", BOUNDARY_COUNT)
    expected_h: List[float] = []
    expected_d: List[float] = []
    probabilities: List[List[float]] = []
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise MMLU0ContractError("trajectory boundary must be an object")
        _keys(boundary, ("boundary_id", "choice_probabilities", "choice_entropy", "kl_to_final", "hidden_rms_l2_to_final", "hidden_cosine_to_final", "hidden_cosine_distance_to_final"), "trajectory boundary")
        if boundary["boundary_id"] != BOUNDARY_IDS[index]:
            raise MMLU0ContractError("trajectory boundary IDs must close B_0...B_36")
        probability = _validate_probability_vector(boundary["choice_probabilities"], "choice probabilities")
        probabilities.append(probability)
        expected_h.append(choice_entropy(probability))
        _finite(boundary["choice_entropy"], "choice_entropy")
        _finite(boundary["kl_to_final"], "kl_to_final")
        _finite(boundary["hidden_rms_l2_to_final"], "hidden_rms_l2_to_final")
        cosine = _finite(boundary["hidden_cosine_to_final"], "hidden_cosine_to_final")
        distance = _finite(boundary["hidden_cosine_distance_to_final"], "hidden_cosine_distance_to_final")
        if cosine < -1.0 or cosine > 1.0 or distance < 0.0 or distance > 2.0:
            raise MMLU0ContractError("hidden cosine diagnostics out of range")
        if boundary["hidden_rms_l2_to_final"] < 0.0:
            raise MMLU0ContractError("hidden RMS-L2 diagnostic cannot be negative")
    expected_d = choice_trajectory_metrics(probabilities)["D"]
    for index, boundary in enumerate(boundaries):
        _close(boundary["choice_entropy"], expected_h[index], "choice_entropy")
        _close(boundary["kl_to_final"], expected_d[index], "kl_to_final")
    if boundaries[-1]["kl_to_final"] != 0.0:
        raise MMLU0ContractError("final KL endpoint must be zero")
    _close(boundaries[-1]["hidden_rms_l2_to_final"], 0.0, "final hidden RMS-L2 endpoint")
    _close(boundaries[-1]["hidden_cosine_to_final"], 1.0, "final hidden cosine endpoint")
    _close(boundaries[-1]["hidden_cosine_distance_to_final"], 0.0, "final hidden cosine distance endpoint")
    angular = _sequence(record["adjacent_angular_distance"], "adjacent angular distance", LAYERS)
    for index, item in enumerate(angular):
        if not isinstance(item, Mapping):
            raise MMLU0ContractError("adjacent angular item must be an object")
        _keys(item, ("transition_id", "angle_over_pi"), "adjacent angular item")
        if item["transition_id"] != "B_%d->B_%d" % (index, index + 1):
            raise MMLU0ContractError("adjacent angular transition IDs differ")
        angle = _finite(item["angle_over_pi"], "angle_over_pi")
        if angle < 0.0 or angle > 1.0:
            raise MMLU0ContractError("adjacent angular distance out of range")


def build_trajectory_record(
    identity: Mapping[str, Any],
    subject: str,
    prompt_sha256: str,
    renderer_provenance: Mapping[str, Any],
    choice_probabilities: Sequence[Sequence[Any]],
    hidden_states: Sequence[Sequence[Any]],
    card: Mapping[str, Any] | None = None,
    *,
    angular_hidden_states: Sequence[Sequence[Any]] | None = None,
) -> Dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise MMLU0ContractError("identity must be an object")
    _keys(identity, ("task", "doc_id", "doc_hash"), "identity")
    boundaries, angular = compute_trajectory_metrics(
        choice_probabilities,
        hidden_states,
        angular_hidden_states=angular_hidden_states,
    )
    record = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "identity": dict(identity),
        "subject": subject,
        "split": "validation",
        "prompt_sha256": prompt_sha256,
        "renderer_provenance": dict(renderer_provenance),
        "boundaries": boundaries,
        "adjacent_angular_distance": angular,
    }
    validate_trajectory_record(record, card)
    return record


def selector_sample(record: Mapping[str, Any]) -> Dict[str, Any]:
    validate_trajectory_record(record)
    identity = canonical_json_bytes(record["identity"]).decode("utf-8")
    values = {
        "identity": identity,
        "category": record["subject"],
        "H": [float(boundary["choice_entropy"]) for boundary in record["boundaries"]],
        "D": [float(boundary["kl_to_final"]) for boundary in record["boundaries"]],
    }
    validate_selector_sample(values)
    return values


def validate_selector_sample(sample: Mapping[str, Any]) -> None:
    if not isinstance(sample, Mapping):
        raise MMLU0ContractError("selector sample must be an object")
    _keys(sample, ("identity", "category", "H", "D"), "selector sample")
    _nonempty_string(sample["identity"], "selector identity")
    _nonempty_string(sample["category"], "selector category")
    for key in ("H", "D"):
        values = [_finite(value, "selector %s" % key) for value in _sequence(sample[key], "selector %s" % key, BOUNDARY_COUNT)]
        if key == "D" and any(value < 0.0 for value in values):
            raise MMLU0ContractError("selector KL values cannot be negative")


__all__ = [
    "BOUNDARY_COUNT",
    "BOUNDARY_IDS",
    "CARD_SCHEMA_VERSION",
    "CHOICE_KEYS",
    "CHOICE_SURFACES",
    "CHOICE_SURFACE_MANIFEST_SHA256",
    "CHOICE_TOKEN_IDS",
    "DATASET_REVISION",
    "FORMAL_REPLICATES",
    "FORMAL_SEED",
    "FREQUENCY_THRESHOLD",
    "HIDDEN_SIZE",
    "LAYERS",
    "LEGAL_SELECTOR_STATES",
    "METHOD",
    "MODEL_CONFIG_SHA256",
    "MMLU0ContractError",
    "MODEL_REVISION",
    "STARTS",
    "TRAJECTORY_SCHEMA_VERSION",
    "VALIDATION_RECORD_COUNT",
    "VALIDATION_SUBJECT_COUNT",
    "WIDTHS",
    "adjacent_angular_distance",
    "build_trajectory_record",
    "canonical_json_bytes",
    "choice_entropy",
    "choice_logits_to_probabilities",
    "choice_surface_manifest",
    "choice_trajectory_metrics",
    "compute_trajectory_metrics",
    "file_sha256",
    "freeze_choice_surface_manifest",
    "hidden_geometry_to_final",
    "kl_to_final",
    "load_json",
    "selector_sample",
    "semantic_sha256",
    "validate_card",
    "validate_choice_surface_manifest",
    "validate_schema_document",
    "validate_selector_sample",
    "validate_trajectory_record",
]
