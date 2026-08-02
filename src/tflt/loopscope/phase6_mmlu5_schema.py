"""Pure, outcome-blind contracts for the Phase 6 MMLU five-shot follow-up.

The Gate K projection contract is deliberately separate from the accepted
zero-shot card.  The trajectory math is reused from the audited pure helpers,
but every persisted five-shot record carries a new schema/renderer identity so
zero-shot and five-shot populations cannot be mixed accidentally.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase6_mmlu0_schema import (
    BOUNDARY_COUNT,
    BOUNDARY_IDS,
    CHOICE_KEYS,
    CHOICE_SURFACES,
    CHOICE_SURFACE_MANIFEST_SHA256,
    CHOICE_TOKEN_IDS,
    HIDDEN_SIZE,
    LAYERS,
    WIDTHS,
    STARTS,
    adjacent_angular_distance as _adjacent_angular_distance,
    choice_entropy,
    choice_logits_to_probabilities,
    choice_trajectory_metrics,
    choice_surface_manifest as _choice_surface_manifest,
    compute_trajectory_metrics as _compute_trajectory_metrics,
    freeze_choice_surface_manifest as _freeze_choice_surface_manifest,
    hidden_geometry_to_final,
    kl_to_final,
)


CARD_SCHEMA_VERSION = "loopscope.phase6.mmlu5-prefix-card.v1"
TRAJECTORY_SCHEMA_VERSION = "loopscope.phase6.mmlu5-prefix-trajectory.v1"
PROMPT_PROJECTION_SCHEMA_VERSION = "loopscope.phase6.mmlu5-prompt-projection.v1"
METHOD = "MMLU_FIVE_SHOT_PREFIX_TRAJECTORY_RBR_V2"
RENDER_CONTRACT_VERSION = "loopscope.mmlu-5shot-prefix-render.v1"
MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_CONFIG_SHA256 = "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"
TOKENIZER_CONFIG_SHA256 = "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
TOKENIZER_JSON_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
DATASET_REPO = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
VALIDATION_RECORD_COUNT = 1531
VALIDATION_SUBJECT_COUNT = 57
VALIDATION_IDENTITY_SHA256 = "ced8dd2ac8277a30e4d4e101f8f76c2dd125659abe35af679f60e5c6a9f076eb"
ZERO_SHOT_PROMPT_PROJECTION_SHA256 = "2be66cc4966f45135075d492cbb30ae67bd148431682e8588c69cee7b1de0278"
TASK_SOURCE_MANIFEST_SHA256 = "7e03a4bac9704839de842c099e068fb2ebba626a066adfc3499c69eb05336b68"
TASK_SOURCE_FILE_COUNT = 63
FORMAL_REPLICATES = 2000
FORMAL_SEED = 20260801
FREQUENCY_THRESHOLD = 0.80
DEMONSTRATION_COUNT = 5
PROJECTION_FIELDS = ("identity", "category", "H", "D")
LEGAL_SELECTOR_STATES = (
    "SELECTED_WINDOW",
    "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
    "ABSTAIN_NO_UNIQUE_TOP1",
    "ABSTAIN_RATE_RANK_UNSTABLE",
)
CANDIDATE_STARTS = {width: tuple(starts) for width, starts in STARTS.items()}


class MMLU5ContractError(ValueError):
    """Fail-closed error for the Gate K five-shot contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MMLU5ContractError("invalid JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise MMLU5ContractError("JSON root must be an object: %s" % path)
    return value


def _keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise MMLU5ContractError(
            "%s keys differ: expected %s, observed %s"
            % (label, sorted(expected_set), sorted(observed))
        )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MMLU5ContractError("%s must be a finite number" % label)
    result = float(value)
    if not math.isfinite(result):
        raise MMLU5ContractError("%s must be finite" % label)
    return result


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MMLU5ContractError("%s must be an integer" % label)
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise MMLU5ContractError("%s must be a SHA-256 hex string" % label)
    try:
        int(value, 16)
    except ValueError as exc:
        raise MMLU5ContractError("%s must be a SHA-256 hex string" % label) from exc
    if value != value.lower():
        raise MMLU5ContractError("%s must use lowercase hexadecimal" % label)
    return value


def _revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise MMLU5ContractError("%s must be a revision SHA-1" % label)
    try:
        int(value, 16)
    except ValueError as exc:
        raise MMLU5ContractError("%s must be a revision SHA-1" % label) from exc
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MMLU5ContractError("%s must be a non-empty string" % label)
    return value


def _sequence(value: Any, label: str, length: int | None = None) -> List[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MMLU5ContractError("%s must be a sequence" % label)
    result = list(value)
    if length is not None and len(result) != length:
        raise MMLU5ContractError("%s must contain %d values" % (label, length))
    return result


def _close(left: Any, right: Any, label: str, tolerance: float = 1e-10) -> None:
    if abs(_finite(left, label) - _finite(right, label)) > tolerance:
        raise MMLU5ContractError("%s is not recomputed exactly" % label)


def choice_surface_manifest() -> Dict[str, Dict[str, Any]]:
    return _choice_surface_manifest()


def freeze_choice_surface_manifest(tokenizer: Any) -> Dict[str, Dict[str, Any]]:
    try:
        return _freeze_choice_surface_manifest(tokenizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise MMLU5ContractError(str(exc)) from exc


def validate_choice_surface_manifest(manifest: Mapping[str, Any]) -> None:
    try:
        from tflt.loopscope.phase6_mmlu0_schema import validate_choice_surface_manifest as validate_zero

        validate_zero(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise MMLU5ContractError(str(exc)) from exc


def expected_candidate_pairs() -> Tuple[Tuple[int, int], ...]:
    return tuple((width, start) for width in WIDTHS for start in CANDIDATE_STARTS[width])


def enumerate_candidates() -> List[Dict[str, Any]]:
    candidates = [
        {
            "width": width,
            "start": start,
            "end": start + width - 1,
            "window": "%d:%d" % (start, start + width),
            "boundary_entry": "B_%d" % start,
            "boundary_exit": "B_%d" % (start + width),
        }
        for width in WIDTHS
        for start in CANDIDATE_STARTS[width]
    ]
    if len(candidates) != 42 or len({(row["width"], row["start"]) for row in candidates}) != 42:
        raise MMLU5ContractError("selector candidate domain is not exactly 42 cells")
    return candidates


def _validate_card_phase(card: Mapping[str, Any]) -> None:
    phase = card["phase"]
    if not isinstance(phase, Mapping):
        raise MMLU5ContractError("phase must be an object")
    _keys(phase, ("name", "hypothesis", "gate", "planning_thread", "executor_thread", "terminal_scope"), "phase")
    expected = {
        "name": "LoopScope Phase 6",
        "hypothesis": "H6_MMLU_FIVE_SHOT_PREFIX_CHOICE_TRAJECTORY_RBR_V2",
        "gate": "K",
        "planning_thread": "019fb3de-2298-75f2-a083-0dca453ea79c",
        "executor_thread": "019fc119-30b4-7dd0-92bf-e6f2f994c8c8",
        "terminal_scope": "Gate M validation-1531 selected window or legal ABSTAIN",
    }
    if dict(phase) != expected:
        raise MMLU5ContractError("phase identity differs")


def _validate_model(card: Mapping[str, Any]) -> None:
    model = card["model"]
    if not isinstance(model, Mapping):
        raise MMLU5ContractError("model must be an object")
    _keys(model, ("repo_id", "revision", "tokenizer_revision", "decoder_layers", "hidden_size", "dtype", "config_sha256", "tokenizer_config_sha256", "tokenizer_json_sha256", "tokenizer_class"), "model")
    expected = {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "decoder_layers": 36,
        "hidden_size": HIDDEN_SIZE,
        "dtype": "bfloat16",
        "config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "tokenizer_class": "transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast",
    }
    if dict(model) != expected:
        raise MMLU5ContractError("model contract differs")


def _validate_task(card: Mapping[str, Any]) -> None:
    task = card["task"]
    if not isinstance(task, Mapping):
        raise MMLU5ContractError("task must be an object")
    _keys(task, ("dataset", "revision", "task_name", "validation_split", "validation_record_count", "validation_subject_count", "validation_identity_sha256", "validation_identity_source", "zero_shot_reference_prompt_projection_sha256", "lm_eval_task_source"), "task")
    if task["dataset"] != DATASET_REPO or task["revision"] != DATASET_REVISION or task["task_name"] != "mmlu" or task["validation_split"] != "validation":
        raise MMLU5ContractError("dataset contract differs")
    if task["validation_record_count"] != VALIDATION_RECORD_COUNT or task["validation_subject_count"] != VALIDATION_SUBJECT_COUNT:
        raise MMLU5ContractError("validation population differs")
    if task["validation_identity_sha256"] != VALIDATION_IDENTITY_SHA256:
        raise MMLU5ContractError("validation identity closure differs")
    _sha256(task["zero_shot_reference_prompt_projection_sha256"], "zero-shot reference projection")
    source = task["lm_eval_task_source"]
    if not isinstance(source, Mapping):
        raise MMLU5ContractError("lm_eval_task_source must be an object")
    _keys(source, ("version", "yaml_path", "yaml_sha256", "default_template_sha256", "stem_group_yaml_sha256", "other_group_yaml_sha256", "social_sciences_group_yaml_sha256", "humanities_group_yaml_sha256", "default_source_file_count", "default_source_manifest_sha256"), "lm_eval_task_source")
    if source["version"] != "0.4.11" or source["default_source_file_count"] != TASK_SOURCE_FILE_COUNT or source["default_source_manifest_sha256"] != TASK_SOURCE_MANIFEST_SHA256:
        raise MMLU5ContractError("lm_eval task source closure differs")
    for key in ("yaml_sha256", "default_template_sha256", "stem_group_yaml_sha256", "other_group_yaml_sha256", "social_sciences_group_yaml_sha256", "humanities_group_yaml_sha256", "default_source_manifest_sha256"):
        _sha256(source[key], "lm_eval_task_source.%s" % key)


def _validate_evaluator(card: Mapping[str, Any]) -> None:
    evaluator = card["evaluator"]
    if not isinstance(evaluator, Mapping):
        raise MMLU5ContractError("evaluator must be an object")
    _keys(evaluator, ("lm_eval_version", "transformers_version", "tokenizers_version", "datasets_version", "torch_version", "accelerate_version", "request_type", "num_fewshot", "fewshot_split", "plain_prompt", "apply_chat_template", "fewshot_as_multiturn", "generation", "direct_letter_fallback"), "evaluator")
    expected = {
        "lm_eval_version": "0.4.11",
        "transformers_version": "4.51.3",
        "tokenizers_version": "0.21.4",
        "datasets_version": "5.0.0",
        "torch_version": "2.3.1+cu121",
        "accelerate_version": "1.14.0",
        "request_type": "loglikelihood",
        "num_fewshot": 5,
        "fewshot_split": "dev",
        "plain_prompt": True,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "generation": False,
        "direct_letter_fallback": False,
    }
    if dict(evaluator) != expected:
        raise MMLU5ContractError("evaluator contract differs")


def _validate_renderer(card: Mapping[str, Any]) -> None:
    renderer = card["renderer"]
    if not isinstance(renderer, Mapping):
        raise MMLU5ContractError("renderer must be an object")
    _keys(renderer, ("contract_version", "semantic_contract", "semantic_contract_sha256", "prompt_projection_required", "raw_prompt_persisted", "token_ids_persisted"), "renderer")
    if renderer["contract_version"] != RENDER_CONTRACT_VERSION:
        raise MMLU5ContractError("renderer contract version differs")
    semantic = renderer["semantic_contract"]
    if not isinstance(semantic, Mapping):
        raise MMLU5ContractError("renderer semantic contract must be an object")
    expected = {
        "task_group": "mmlu",
        "target_split": "validation",
        "fewshot_split": "dev",
        "num_fewshot": 5,
        "plain_prompt": True,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "generation": False,
        "direct_letter_fallback": False,
        "demonstration_count": 5,
        "demonstration_answers_allowed": True,
        "target_columns": ["question", "choices", "subject"],
        "probe_position": "last_non_padding_rendered_prefix_token",
        "raw_prompt_persisted": False,
        "token_ids_persisted": False,
    }
    if dict(semantic) != expected:
        raise MMLU5ContractError("renderer semantic contract differs")
    if renderer["semantic_contract_sha256"] != semantic_sha256(semantic):
        raise MMLU5ContractError("renderer semantic contract hash differs")
    if renderer["prompt_projection_required"] is not True or renderer["raw_prompt_persisted"] is not False or renderer["token_ids_persisted"] is not False:
        raise MMLU5ContractError("renderer persistence boundary differs")


def _validate_trajectory_contract(card: Mapping[str, Any]) -> None:
    trajectory = card["trajectory"]
    if not isinstance(trajectory, Mapping):
        raise MMLU5ContractError("trajectory must be an object")
    _keys(trajectory, ("schema_version", "schema_path", "boundary_ids", "boundary_count", "raw_boundary_definition", "final_norm_definition", "probe_position", "forward_passes_per_record", "native_no_loop", "use_cache", "hidden_diagnostics_only", "persist_raw_logits", "persist_full_vocab_probabilities", "persist_hidden_tensors"), "trajectory")
    if trajectory["schema_version"] != TRAJECTORY_SCHEMA_VERSION or trajectory["schema_path"] != "configs/loopscope/phase6_mmlu5_trajectory_schema.json":
        raise MMLU5ContractError("trajectory schema identity differs")
    if trajectory["boundary_ids"] != list(BOUNDARY_IDS) or trajectory["boundary_count"] != BOUNDARY_COUNT:
        raise MMLU5ContractError("trajectory boundary closure differs")
    if trajectory["probe_position"] != "last_non_padding_rendered_prefix_token" or trajectory["forward_passes_per_record"] != 1 or trajectory["native_no_loop"] is not True or trajectory["use_cache"] is not False:
        raise MMLU5ContractError("trajectory forward contract differs")
    for key in ("hidden_diagnostics_only", "persist_raw_logits", "persist_full_vocab_probabilities", "persist_hidden_tensors"):
        if trajectory[key] is not (True if key == "hidden_diagnostics_only" else False):
            raise MMLU5ContractError("trajectory persistence contract differs")


def _validate_selector(card: Mapping[str, Any]) -> None:
    selector = card["selector"]
    if not isinstance(selector, Mapping):
        raise MMLU5ContractError("selector must be an object")
    _keys(selector, ("method", "central_blocks_inclusive", "widths", "candidate_starts", "candidate_count", "bootstrap", "projection_fields", "projection_excludes", "legal_terminal_states", "formal_execution_deferred_to"), "selector")
    if selector["method"] != "RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE" or selector["central_blocks_inclusive"] != [11, 24] or selector["widths"] != [3, 4, 5, 6] or selector["candidate_count"] != 42:
        raise MMLU5ContractError("selector domain differs")
    starts = {int(key): tuple(value) for key, value in selector["candidate_starts"].items()}
    if starts != CANDIDATE_STARTS:
        raise MMLU5ContractError("selector candidate starts differ")
    bootstrap = selector["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise MMLU5ContractError("selector bootstrap must be an object")
    _keys(bootstrap, ("replicates", "seed", "population_stratification", "standard_error_ddof", "frequency_threshold"), "selector bootstrap")
    if bootstrap["replicates"] != FORMAL_REPLICATES or bootstrap["seed"] != FORMAL_SEED or bootstrap["population_stratification"] != "subject-stratified joint bootstrap" or bootstrap["standard_error_ddof"] != 1 or bootstrap["frequency_threshold"] != FREQUENCY_THRESHOLD:
        raise MMLU5ContractError("selector bootstrap differs")
    if selector["projection_fields"] != list(PROJECTION_FIELDS) or selector["formal_execution_deferred_to"] != "Gate M":
        raise MMLU5ContractError("selector projection boundary differs")
    if set(selector["projection_fields"]) & set(selector["projection_excludes"]):
        raise MMLU5ContractError("selector projection contains a diagnostic field")
    if selector["legal_terminal_states"] != list(LEGAL_SELECTOR_STATES):
        raise MMLU5ContractError("selector terminal states differ")


def validate_card(card: Mapping[str, Any]) -> None:
    if not isinstance(card, Mapping):
        raise MMLU5ContractError("card must be an object")
    _keys(card, ("schema_version", "card", "status", "phase", "model", "task", "evaluator", "renderer", "choice_surfaces", "trajectory", "metrics", "selector", "loop_identity_suggestion", "information_barrier"), "card")
    if card["schema_version"] != CARD_SCHEMA_VERSION or card["card"] != METHOD or card["status"] != "GATE_K_CONTRACT_FROZEN":
        raise MMLU5ContractError("card identity differs")
    _validate_card_phase(card)
    _validate_model(card)
    _validate_task(card)
    _validate_evaluator(card)
    _validate_renderer(card)
    try:
        validate_choice_surface_manifest(card["choice_surfaces"])
    except MMLU5ContractError:
        raise
    _validate_trajectory_contract(card)
    metrics = card["metrics"]
    if not isinstance(metrics, Mapping):
        raise MMLU5ContractError("metrics must be an object")
    _keys(metrics, ("choice_entropy", "kl_to_final", "hidden_diagnostics", "adjacent_diagnostic", "selector_fields", "diagnostic_fields_not_in_selector"), "metrics")
    if metrics["selector_fields"] != ["H", "D"] or set(metrics["selector_fields"]) & set(metrics["diagnostic_fields_not_in_selector"]):
        raise MMLU5ContractError("metric selector boundary differs")
    _validate_selector(card)
    loop = card["loop_identity_suggestion"]
    if not isinstance(loop, Mapping):
        raise MMLU5ContractError("loop identity suggestion must be an object")
    _keys(loop, ("authorized", "dtype", "k", "block_euler_step", "horizon", "cache_strategy", "decode_mode", "note"), "loop identity suggestion")
    if loop["authorized"] is not False:
        raise MMLU5ContractError("loop execution is not forbidden in Gate K")
    barrier = card["information_barrier"]
    if not isinstance(barrier, Mapping):
        raise MMLU5ContractError("information barrier must be an object")
    _keys(barrier, ("gate_k_may_read", "gate_k_must_not_read", "validation_target_gold_read", "test_split_read", "outcome_read", "model_weights_loaded", "model_forward_executed", "cuda_gpu_slurm", "selector_executed", "raw_prompt_persisted", "token_ids_persisted"), "information barrier")
    for key in ("validation_target_gold_read", "test_split_read", "outcome_read", "model_weights_loaded", "model_forward_executed", "cuda_gpu_slurm", "selector_executed", "raw_prompt_persisted", "token_ids_persisted"):
        if barrier[key] is not False:
            raise MMLU5ContractError("information barrier.%s differs" % key)


def validate_schema_document(schema: Mapping[str, Any]) -> None:
    if not isinstance(schema, Mapping) or schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise MMLU5ContractError("schema root is not closed")
    if schema.get("$id") == "loopscope.phase6.mmlu5-prompt-projection.v1":
        required = ["schema_version", "ordinal", "identity", "subject", "split", "num_fewshot", "fewshot_split", "demonstration_count", "demonstration_ids", "demonstration_provenance", "prompt_sha256", "prompt_tokenization_sha256", "sequence_length", "probe_position", "renderer_source_manifest_sha256", "render_contract_sha256", "dataset_revision"]
        if schema.get("required") != required or set(schema.get("properties", {})) != set(required):
            raise MMLU5ContractError("prompt projection schema fields differ")
        properties = schema["properties"]
        for node in (properties["identity"], properties["demonstration_provenance"]["items"], properties["demonstration_provenance"]["items"]["properties"]):
            if isinstance(node, Mapping) and node.get("additionalProperties") not in (False, None):
                raise MMLU5ContractError("prompt projection nested schema is open")
        if properties["schema_version"].get("const") != PROMPT_PROJECTION_SCHEMA_VERSION:
            raise MMLU5ContractError("prompt projection schema version differs")
        return
    if schema.get("$id") == "loopscope.phase6.mmlu5-prefix-trajectory.v1":
        required = ["schema_version", "identity", "subject", "split", "prompt_sha256", "renderer_provenance", "boundaries", "adjacent_angular_distance"]
        if schema.get("required") != required or set(schema.get("properties", {})) != set(required):
            raise MMLU5ContractError("trajectory schema fields differ")
        properties = schema["properties"]
        if properties["schema_version"].get("const") != TRAJECTORY_SCHEMA_VERSION or properties["split"].get("const") != "validation":
            raise MMLU5ContractError("trajectory schema constants differ")
        for node in (properties["identity"], properties["renderer_provenance"], properties["boundaries"]["items"], properties["adjacent_angular_distance"]["items"]):
            if node.get("additionalProperties") is not False:
                raise MMLU5ContractError("trajectory nested schema is open")
        return
    raise MMLU5ContractError("unsupported Gate K schema id")


PROJECTION_RECORD_FIELDS = (
    "schema_version", "ordinal", "identity", "subject", "split", "num_fewshot", "fewshot_split",
    "demonstration_count", "demonstration_ids", "demonstration_provenance", "prompt_sha256",
    "prompt_tokenization_sha256", "sequence_length", "probe_position", "renderer_source_manifest_sha256",
    "render_contract_sha256", "dataset_revision",
)
DEMO_FIELDS = ("id", "doc_index", "subject", "split", "doc_sha256", "rendered_sha256", "example_answer_sha256")


def validate_prompt_projection_record(record: Mapping[str, Any], card: Mapping[str, Any] | None = None) -> None:
    if not isinstance(record, Mapping):
        raise MMLU5ContractError("prompt projection record must be an object")
    _keys(record, PROJECTION_RECORD_FIELDS, "prompt projection record")
    if record["schema_version"] != PROMPT_PROJECTION_SCHEMA_VERSION:
        raise MMLU5ContractError("prompt projection schema version differs")
    ordinal = _strict_int(record["ordinal"], "projection ordinal")
    if ordinal < 0:
        raise MMLU5ContractError("projection ordinal cannot be negative")
    identity = record["identity"]
    if not isinstance(identity, Mapping):
        raise MMLU5ContractError("projection identity must be an object")
    _keys(identity, ("task", "doc_id", "doc_hash"), "projection identity")
    if identity["task"] != "mmlu":
        raise MMLU5ContractError("projection task differs")
    _nonempty(identity["doc_id"], "projection doc_id")
    _sha256(identity["doc_hash"], "projection doc_hash")
    _nonempty(record["subject"], "projection subject")
    if record["split"] != "validation" or record["num_fewshot"] != DEMONSTRATION_COUNT or record["fewshot_split"] != "dev" or record["demonstration_count"] != DEMONSTRATION_COUNT:
        raise MMLU5ContractError("five-shot split/count contract differs")
    ids = _sequence(record["demonstration_ids"], "demonstration IDs", DEMONSTRATION_COUNT)
    demos = _sequence(record["demonstration_provenance"], "demonstration provenance", DEMONSTRATION_COUNT)
    if len(set(ids)) != DEMONSTRATION_COUNT:
        raise MMLU5ContractError("demonstration IDs are not unique")
    normalized_ids = []
    for index, demo in enumerate(demos):
        if not isinstance(demo, Mapping):
            raise MMLU5ContractError("demonstration provenance must be an object")
        _keys(demo, DEMO_FIELDS, "demonstration provenance")
        if demo["id"] != ids[index] or demo["split"] != "dev" or demo["subject"] != record["subject"]:
            raise MMLU5ContractError("demonstration order/subject differs")
        if not isinstance(demo["doc_index"], int) or isinstance(demo["doc_index"], bool) or demo["doc_index"] < 0:
            raise MMLU5ContractError("demonstration index is invalid")
        for key in ("doc_sha256", "rendered_sha256", "example_answer_sha256"):
            _sha256(demo[key], "demonstration.%s" % key)
        normalized_ids.append(str(demo["id"]))
    if str(identity["doc_id"]) in normalized_ids or any(":validation:" in value for value in normalized_ids):
        raise MMLU5ContractError("target/demo split identity overlap")
    _sha256(record["prompt_sha256"], "prompt_sha256")
    _sha256(record["prompt_tokenization_sha256"], "prompt_tokenization_sha256")
    sequence_length = _strict_int(record["sequence_length"], "sequence_length")
    probe_position = _strict_int(record["probe_position"], "probe_position")
    if sequence_length <= 0 or probe_position != sequence_length - 1:
        raise MMLU5ContractError("probe position is not the last non-padding token")
    _sha256(record["renderer_source_manifest_sha256"], "renderer source manifest")
    _sha256(record["render_contract_sha256"], "render contract")
    if record["renderer_source_manifest_sha256"] != TASK_SOURCE_MANIFEST_SHA256 or record["dataset_revision"] != DATASET_REVISION:
        raise MMLU5ContractError("projection provenance differs")
    _revision(record["dataset_revision"], "dataset revision")
    if card is not None:
        validate_card(card)
        if record["renderer_source_manifest_sha256"] != TASK_SOURCE_MANIFEST_SHA256:
            raise MMLU5ContractError("projection renderer source does not match card")


def _validate_probability_vector(values: Any, label: str) -> List[float]:
    result = [_finite(value, label) for value in _sequence(values, label, 4)]
    if any(value <= 0.0 for value in result) or abs(math.fsum(result) - 1.0) > 1e-9:
        raise MMLU5ContractError("%s is not a positive probability vector" % label)
    return result


def validate_trajectory_record(record: Mapping[str, Any], card: Mapping[str, Any] | None = None) -> None:
    if not isinstance(record, Mapping):
        raise MMLU5ContractError("trajectory record must be an object")
    fields = ("schema_version", "identity", "subject", "split", "prompt_sha256", "renderer_provenance", "boundaries", "adjacent_angular_distance")
    _keys(record, fields, "trajectory record")
    if record["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise MMLU5ContractError("trajectory schema version differs")
    identity = record["identity"]
    if not isinstance(identity, Mapping):
        raise MMLU5ContractError("trajectory identity must be an object")
    _keys(identity, ("task", "doc_id", "doc_hash"), "trajectory identity")
    if identity["task"] != "mmlu":
        raise MMLU5ContractError("trajectory task differs")
    _nonempty(identity["doc_id"], "trajectory doc_id")
    _sha256(identity["doc_hash"], "trajectory doc_hash")
    _nonempty(record["subject"], "trajectory subject")
    if record["split"] != "validation":
        raise MMLU5ContractError("trajectory split differs")
    _sha256(record["prompt_sha256"], "trajectory prompt hash")
    provenance = record["renderer_provenance"]
    if not isinstance(provenance, Mapping):
        raise MMLU5ContractError("trajectory provenance must be an object")
    _keys(provenance, ("contract_version", "semantic_contract_sha256", "task_source_manifest_sha256", "choice_surface_manifest_sha256"), "trajectory provenance")
    if provenance["contract_version"] != RENDER_CONTRACT_VERSION or provenance["task_source_manifest_sha256"] != TASK_SOURCE_MANIFEST_SHA256:
        raise MMLU5ContractError("trajectory renderer provenance differs")
    for key in ("semantic_contract_sha256", "task_source_manifest_sha256", "choice_surface_manifest_sha256"):
        _sha256(provenance[key], "trajectory provenance.%s" % key)
    if card is not None:
        validate_card(card)
        if provenance["semantic_contract_sha256"] != card["renderer"]["semantic_contract_sha256"]:
            raise MMLU5ContractError("trajectory semantic contract hash differs")
    boundaries = _sequence(record["boundaries"], "trajectory boundaries", BOUNDARY_COUNT)
    probabilities: List[List[float]] = []
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise MMLU5ContractError("trajectory boundary must be an object")
        _keys(boundary, ("boundary_id", "choice_probabilities", "choice_entropy", "kl_to_final", "hidden_rms_l2_to_final", "hidden_cosine_to_final", "hidden_cosine_distance_to_final"), "trajectory boundary")
        if boundary["boundary_id"] != BOUNDARY_IDS[index]:
            raise MMLU5ContractError("trajectory boundary IDs differ")
        probabilities.append(_validate_probability_vector(boundary["choice_probabilities"], "choice probabilities"))
        for key in ("choice_entropy", "kl_to_final", "hidden_rms_l2_to_final", "hidden_cosine_to_final", "hidden_cosine_distance_to_final"):
            _finite(boundary[key], "trajectory.%s" % key)
        if boundary["hidden_rms_l2_to_final"] < 0.0 or not -1.0 <= boundary["hidden_cosine_to_final"] <= 1.0 or not 0.0 <= boundary["hidden_cosine_distance_to_final"] <= 2.0:
            raise MMLU5ContractError("hidden diagnostic range differs")
    expected = choice_trajectory_metrics(probabilities)
    for index, boundary in enumerate(boundaries):
        _close(boundary["choice_entropy"], expected["H"][index], "choice_entropy")
        _close(boundary["kl_to_final"], expected["D"][index], "kl_to_final")
    if boundaries[-1]["kl_to_final"] != 0.0:
        raise MMLU5ContractError("final KL endpoint must be zero")
    angular = _sequence(record["adjacent_angular_distance"], "adjacent angular distance", LAYERS)
    for index, item in enumerate(angular):
        if not isinstance(item, Mapping):
            raise MMLU5ContractError("angular item must be an object")
        _keys(item, ("transition_id", "angle_over_pi"), "angular item")
        if item["transition_id"] != "B_%d->B_%d" % (index, index + 1):
            raise MMLU5ContractError("angular transition ID differs")
        angle = _finite(item["angle_over_pi"], "angle_over_pi")
        if not 0.0 <= angle <= 1.0:
            raise MMLU5ContractError("angular distance is out of range")


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
        raise MMLU5ContractError("identity must be an object")
    _keys(identity, ("task", "doc_id", "doc_hash"), "trajectory identity")
    boundaries, angular = _compute_trajectory_metrics(
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
    values = {
        "identity": canonical_json_bytes(record["identity"]).decode("utf-8"),
        "category": record["subject"],
        "H": [float(row["choice_entropy"]) for row in record["boundaries"]],
        "D": [float(row["kl_to_final"]) for row in record["boundaries"]],
    }
    validate_selector_sample(values)
    return values


def validate_selector_sample(sample: Mapping[str, Any]) -> None:
    if not isinstance(sample, Mapping):
        raise MMLU5ContractError("selector sample must be an object")
    _keys(sample, PROJECTION_FIELDS, "selector sample")
    _nonempty(sample["identity"], "selector identity")
    _nonempty(sample["category"], "selector category")
    for key in ("H", "D"):
        values = [_finite(value, "selector %s" % key) for value in _sequence(sample[key], "selector %s" % key, BOUNDARY_COUNT)]
        if key == "D" and any(value < 0.0 for value in values):
            raise MMLU5ContractError("selector KL values cannot be negative")


__all__ = [
    "BOUNDARY_COUNT", "BOUNDARY_IDS", "CARD_SCHEMA_VERSION", "CHOICE_KEYS", "CHOICE_SURFACES", "CHOICE_SURFACE_MANIFEST_SHA256", "CHOICE_TOKEN_IDS",
    "CANDIDATE_STARTS", "DATASET_REPO", "DATASET_REVISION", "DEMONSTRATION_COUNT", "FORMAL_REPLICATES", "FORMAL_SEED",
    "FREQUENCY_THRESHOLD", "HIDDEN_SIZE", "LAYERS", "LEGAL_SELECTOR_STATES", "METHOD", "MODEL_CONFIG_SHA256",
    "MODEL_REPO", "MODEL_REVISION", "MMLU5ContractError", "PROJECTION_FIELDS", "PROMPT_PROJECTION_SCHEMA_VERSION",
    "RENDER_CONTRACT_VERSION", "STARTS", "TASK_SOURCE_FILE_COUNT", "TASK_SOURCE_MANIFEST_SHA256", "TOKENIZER_CONFIG_SHA256",
    "TOKENIZER_JSON_SHA256", "TRAJECTORY_SCHEMA_VERSION", "VALIDATION_IDENTITY_SHA256", "VALIDATION_RECORD_COUNT",
    "VALIDATION_SUBJECT_COUNT", "WIDTHS", "ZERO_SHOT_PROMPT_PROJECTION_SHA256", "adjacent_angular_distance",
    "build_trajectory_record", "canonical_json_bytes", "choice_entropy", "choice_logits_to_probabilities", "choice_surface_manifest",
    "enumerate_candidates", "expected_candidate_pairs", "file_sha256", "freeze_choice_surface_manifest", "hidden_geometry_to_final",
    "kl_to_final", "load_json", "semantic_sha256", "selector_sample", "validate_card", "validate_choice_surface_manifest",
    "validate_prompt_projection_record", "validate_schema_document", "validate_selector_sample", "validate_trajectory_record",
]

adjacent_angular_distance = _adjacent_angular_distance
