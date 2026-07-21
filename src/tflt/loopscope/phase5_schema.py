"""Phase 5 Gate A contract, synthetic trajectory, selector, and panel helpers.

This module is intentionally standard-library only.  It closes the data-free
Gate A path while leaving real model execution and formal population work to
later, separately authorized Gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CARD_SCHEMA_VERSION = "loopscope.phase5.card.v1"
REGISTRY_SCHEMA_VERSION = "loopscope.phase5.known-outcome-registry.v1"
TRAJECTORY_SCHEMA_VERSION = "loopscope.phase5.trajectory-record.v1"
RECEIPT_SCHEMA_VERSION = "loopscope.phase5.gate-a-receipt.v1"

LAYER_COUNT = 36
WINDOW_WIDTH = 4
BOUNDARY_IDS = tuple("B_%d" % index for index in range(LAYER_COUNT + 1))
WINDOW_STARTS = tuple(range(0, LAYER_COUNT - WINDOW_WIDTH + 1))
SHIFT_STARTS = tuple(range(1, LAYER_COUNT - WINDOW_WIDTH))
CONSENSUS_STARTS = tuple(range(WINDOW_WIDTH, LAYER_COUNT - 2 * WINDOW_WIDTH + 1))
CHOICE_ORDER = ("A", "B", "C", "D")
EXPECTED_SPACED_TOKEN_IDS = {"A": 362, "B": 425, "C": 356, "D": 422}
EXPECTED_PLAIN_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
ABSTAIN_LABELS = (
    "ABSTAIN_NO_POINT_ELIGIBLE",
    "ABSTAIN_NO_UNIQUE_TOP1",
    "ABSTAIN_LOW_SELECTION_FREQUENCY",
)

_TOP_LEVEL_TRAJECTORY_KEYS = frozenset(
    (
        "schema_version",
        "identity",
        "subject",
        "split",
        "prompt_sha256",
        "renderer_provenance",
        "boundaries",
    )
)
_IDENTITY_KEYS = frozenset(("task", "doc_id", "doc_hash"))
_RENDERER_KEYS = frozenset(
    (
        "model_repo",
        "model_revision",
        "tokenizer_revision",
        "dataset_repo",
        "dataset_revision",
        "renderer_manifest_sha256",
        "forward_type",
        "formal_forward_count_per_identity",
        "loop_insertions",
        "position",
        "projection",
        "choice_order",
        "spaced_choice_token_ids",
        "final_norm_path",
        "output_head_path",
    )
)
_BOUNDARY_KEYS = frozenset(
    (
        "boundary_id",
        "choice_probabilities",
        "choice_entropy",
        "kl_to_final",
        "hidden_l2_to_final",
        "hidden_cosine_to_final",
        "hidden_cosine_distance_to_final",
    )
)
_FORBIDDEN_KEY_TOKENS = (
    "target",
    "gold",
    "label",
    "correct",
    "accuracy",
    "answer",
    "outcome",
    "prediction",
    "generated",
    "logits",
    "hidden_state",
    "hidden_tensor",
    "residual_tensor",
    "full_probabilities",
)


class Phase5ContractError(ValueError):
    """Raised when a Phase 5 contract boundary fails closed."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise Phase5ContractError("JSON artifact must be an object: %s" % path)
    return payload


def _require_manifest_hash(payload: Mapping[str, Any], label: str) -> None:
    observed = payload.get("manifest_sha256")
    expected = semantic_sha256(payload)
    if observed != expected:
        raise Phase5ContractError("%s manifest_sha256 mismatch" % label)


def _require_exact_keys(payload: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    if set(payload) != expected_set:
        raise Phase5ContractError(
            "%s keys differ; missing=%s extra=%s"
            % (label, sorted(expected_set - set(payload)), sorted(set(payload) - expected_set))
        )


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise Phase5ContractError("%s must be a lowercase SHA256" % label)
    return text


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase5ContractError("%s must be numeric" % label)
    number = float(value)
    if not math.isfinite(number):
        raise Phase5ContractError("%s must be finite" % label)
    return number


def boundary_contract(layer_count: int = LAYER_COUNT, width: int = WINDOW_WIDTH) -> Dict[str, Any]:
    if layer_count < 2 * width or width < 1:
        raise Phase5ContractError("layer_count/width cannot support CONSENSUS")
    return {
        "boundaries": ["B_%d" % index for index in range(layer_count + 1)],
        "window_starts": list(range(0, layer_count - width + 1)),
        "shift_starts": list(range(1, layer_count - width)),
        "consensus_starts": list(range(width, layer_count - 2 * width + 1)),
        "inclusive_15_18": {"entry": "B_15", "exit": "B_19"},
    }


def validate_token_ids(
    spaced: Mapping[str, Sequence[int]], plain: Mapping[str, Sequence[int]]
) -> None:
    for name, observed, expected in (
        ("spaced", spaced, EXPECTED_SPACED_TOKEN_IDS),
        ("plain", plain, EXPECTED_PLAIN_TOKEN_IDS),
    ):
        if set(observed) != set(CHOICE_ORDER):
            raise Phase5ContractError("%s token map choices differ" % name)
        flattened = []
        for choice in CHOICE_ORDER:
            ids = observed[choice]
            if isinstance(ids, (str, bytes)) or len(ids) != 1:
                raise Phase5ContractError("%s %s must be exactly one token" % (name, choice))
            token_id = int(ids[0])
            if token_id != expected[choice]:
                raise Phase5ContractError("%s %s token id differs" % (name, choice))
            flattened.append(token_id)
        if len(set(flattened)) != 4:
            raise Phase5ContractError("%s choice token ids must be unique" % name)


def validate_card(card: Mapping[str, Any]) -> None:
    _require_manifest_hash(card, "card")
    required = {
        "schema_version",
        "card",
        "status",
        "hypothesis",
        "claim_scope",
        "model",
        "task",
        "identity",
        "boundary_contract",
        "trajectory",
        "loop_outcome_recipe",
        "selector",
        "panel",
        "outcome_statistics",
        "fail_closed",
        "manifest_sha256",
    }
    _require_exact_keys(card, required, "card")
    if card["schema_version"] != CARD_SCHEMA_VERSION:
        raise Phase5ContractError("unsupported Phase 5 card schema")
    if card["hypothesis"] != "H5_QWEN4B_BASE_MMLU_TRAJECTORY_SELECTION_V1":
        raise Phase5ContractError("hypothesis differs")
    model = card["model"]
    frozen_model = {
        "repo": "Qwen/Qwen3-4B-Base",
        "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "tokenizer_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "architecture": "Qwen3ForCausalLM",
        "decoder_layers": 36,
        "hidden_size": 2560,
        "vocab_size": 151936,
        "dtype": "bfloat16",
        "tie_word_embeddings": True,
    }
    for key, expected in frozen_model.items():
        if model.get(key) != expected:
            raise Phase5ContractError("model.%s differs" % key)
    if model.get("spaced_choice_token_ids") != EXPECTED_SPACED_TOKEN_IDS:
        raise Phase5ContractError("spaced choice token ids differ")
    if model.get("plain_choice_token_ids") != EXPECTED_PLAIN_TOKEN_IDS:
        raise Phase5ContractError("plain choice token ids differ")
    for key in (
        "config_sha256",
        "tokenizer_config_sha256",
        "tokenizer_json_sha256",
        "qwen3_model_init_sha256",
        "causal_lm_init_sha256",
        "causal_lm_forward_sha256",
    ):
        _require_sha256(model.get(key), "model.%s" % key)
    task = card["task"]
    expected_task = {
        "dataset": "cais/mmlu",
        "dataset_revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "task_group": "mmlu",
        "num_fewshot": 5,
        "fewshot_split": "dev",
        "trajectory_split": "validation",
        "trajectory_sample_count": 1531,
        "trajectory_subject_count": 57,
        "outcome_split": "test",
        "outcome_sample_count": 14042,
        "lm_eval_version": "0.4.11",
        "chat_template": False,
        "multiturn": False,
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            raise Phase5ContractError("task.%s differs" % key)
    for key in (
        "validation_renderer_manifest_file_sha256",
        "validation_renderer_manifest_internal_sha256",
        "validation_projection_sha256",
        "dataset_fingerprint_sha256",
        "renderer_source_sha256",
        "template_sha256",
        "render_contract_sha256",
        "test_identity_manifest_file_sha256",
        "test_identity_manifest_internal_sha256",
        "test_ordered_identity_sha256",
        "validation_test_disjointness_receipt_sha256",
    ):
        _require_sha256(task.get(key), "task.%s" % key)
    expected_boundary = boundary_contract()
    if card["boundary_contract"] != expected_boundary:
        raise Phase5ContractError("boundary contract differs")
    trajectory = card["trajectory"]
    if trajectory.get("choice_probability_rule") != "stable_float64_softmax_exact_spaced_A_B_C_D":
        raise Phase5ContractError("choice probability rule differs")
    if trajectory.get("generic_delta_direction") != "exit_minus_entry":
        raise Phase5ContractError("generic delta direction differs")
    if trajectory.get("geometry_role") != "diagnostic_only_not_selector":
        raise Phase5ContractError("geometry role differs")
    if trajectory.get("numeric_contract", {}).get("hidden_reduction_dtype") != "float32":
        raise Phase5ContractError("hidden reduction dtype differs")
    selector = card["selector"]
    if selector.get("name") != "ENTROPY_KL_CONSENSUS_WIDTH4_V1":
        raise Phase5ContractError("selector name differs")
    if selector.get("support_starts") != list(CONSENSUS_STARTS):
        raise Phase5ContractError("selector support differs")
    if selector.get("score") != "min(z_OH,z_OK,z_FH,z_FK)":
        raise Phase5ContractError("selector score differs")
    if selector.get("tie_abs_rel_tolerance") != 1e-12:
        raise Phase5ContractError("selector tie tolerance differs")
    if selector.get("selection_frequency_threshold") != 0.8:
        raise Phase5ContractError("selection frequency threshold differs")
    if selector.get("abstain_labels") != list(ABSTAIN_LABELS):
        raise Phase5ContractError("selector ABSTAIN labels differ")
    bootstrap = selector.get("bootstrap")
    expected_bootstrap = {
        "method": "subject_stratified_joint_sample_bootstrap",
        "replicates": 2000,
        "seed": 20260722,
        "prng": "python_stdlib_random.Random",
        "random_api": "randrange(n_s)",
        "standard_error": "sample_sd_ddof_1",
        "percentile_interval": "linear_interpolation_at_q_times_R_minus_1",
    }
    if bootstrap != expected_bootstrap:
        raise Phase5ContractError("trajectory bootstrap differs")
    loop = card["loop_outcome_recipe"]
    expected_loop = {
        "window_width": 4,
        "k": 3,
        "iteration_mode": "block",
        "strategy_cli": "euler",
        "strategy_kernel": "damped_euler_alias",
        "alpha": 1.0,
        "beta": 0.0,
        "step_size": 1.0 / 3.0,
        "total_horizon": 1.0,
        "cache_strategy": "first",
        "decode_mode": "bypass",
        "decode_scientific_semantics": "N/A_no_autoregressive_decode",
        "only_window_may_vary": True,
    }
    if loop != expected_loop:
        raise Phase5ContractError("loop outcome recipe differs")
    panel = card["panel"]
    if panel.get("max_unique_cells") != 8 or panel.get("fixed_comparator") != "15:18":
        raise Phase5ContractError("panel size/comparator differs")
    if panel.get("high3_rule") != "registry_excluded_selected_inclusive_preserve_raw_high3":
        raise Phase5ContractError("panel high3 rule differs")
    statistics = card["outcome_statistics"]
    if statistics.get("bootstrap") != {
        "method": "subject_stratified_joint_paired_bootstrap",
        "replicates": 2000,
        "seed": 20260723,
    }:
        raise Phase5ContractError("outcome bootstrap differs")
    if statistics.get("comparator_margin_percentage_points") != 0.30:
        raise Phase5ContractError("comparator margin differs")
    if statistics.get("selected_baseline_success") != "point_gain_strictly_greater_than_zero_with_paired_CI_reported":
        raise Phase5ContractError("selected-baseline success differs")
    forbidden = card["fail_closed"].get("forbidden_selector_or_gate_a_fields", [])
    for token in ("gold", "correctness", "test_outcome", "window_outcome"):
        if token not in forbidden:
            raise Phase5ContractError("fail_closed field list misses %s" % token)


def validate_registry(registry: Mapping[str, Any]) -> None:
    _require_manifest_hash(registry, "known-outcome registry")
    _require_exact_keys(
        registry,
        {
            "schema_version",
            "status",
            "source",
            "entries",
            "default_reuse_verdict",
            "blind_group_exclusion_windows",
            "formal_phase5_result",
            "manifest_sha256",
        },
        "known-outcome registry",
    )
    if registry["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise Phase5ContractError("known-outcome registry schema differs")
    if registry["status"] != "KNOWN_BACKGROUND_NOT_FORMAL_RESULT":
        raise Phase5ContractError("known-outcome registry status differs")
    if registry["default_reuse_verdict"] != "FRESH_ACQUISITION_REQUIRED":
        raise Phase5ContractError("known-outcome registry reuse verdict differs")
    if registry["formal_phase5_result"] is not False:
        raise Phase5ContractError("known background cannot be a formal Phase 5 result")
    entries = registry["entries"]
    if not isinstance(entries, list) or len(entries) != 2:
        raise Phase5ContractError("registry must contain baseline and 15:18")
    by_id = {entry.get("id"): entry for entry in entries if isinstance(entry, Mapping)}
    if set(by_id) != {"baseline", "window_15_18"}:
        raise Phase5ContractError("registry entry identities differ")
    baseline = by_id["baseline"]
    comparator = by_id["window_15_18"]
    if baseline.get("accuracy_percent") != 73.27 or baseline.get("window") is not None:
        raise Phase5ContractError("registered baseline background differs")
    if (
        comparator.get("window") != "15:18"
        or comparator.get("accuracy_percent") != 73.61
        or comparator.get("gain_percentage_points") != 0.34
        or comparator.get("excluded_from_blind_groups") is not True
    ):
        raise Phase5ContractError("registered 15:18 background differs")
    for entry in entries:
        if entry.get("reuse_verdict") != "FRESH_ACQUISITION_REQUIRED":
            raise Phase5ContractError("every registered entry must remain fresh")
        if entry.get("exact_per_sample_closure") is not False:
            raise Phase5ContractError("registry incorrectly claims per-sample closure")
    if registry["blind_group_exclusion_windows"] != ["15:18"]:
        raise Phase5ContractError("known-window exclusion differs")


def validate_card_schema(schema: Mapping[str, Any]) -> None:
    _require_manifest_hash(schema, "card schema")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise Phase5ContractError("card schema draft differs")
    if schema.get("$id") != CARD_SCHEMA_VERSION:
        raise Phase5ContractError("card schema id differs")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise Phase5ContractError("card schema must be closed-world at the root")
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not isinstance(properties, Mapping):
        raise Phase5ContractError("card schema required/properties are missing")
    if set(required) != set(properties):
        raise Phase5ContractError("card schema required/properties differ")
    if properties.get("schema_version", {}).get("const") != CARD_SCHEMA_VERSION:
        raise Phase5ContractError("card schema version const differs")
    if properties.get("manifest_sha256", {}).get("pattern") != "^[0-9a-f]{64}$":
        raise Phase5ContractError("card schema manifest pattern differs")


def stable_choice_probabilities(logits: Sequence[float]) -> List[float]:
    if isinstance(logits, (str, bytes)) or len(logits) != 4:
        raise Phase5ContractError("choice logits must contain exactly four values")
    values = [_require_finite(value, "choice logit") for value in logits]
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise Phase5ContractError("stable softmax denominator is invalid")
    probabilities = [value / denominator for value in exponentials]
    if any(not math.isfinite(value) or value <= 0.0 for value in probabilities):
        raise Phase5ContractError("choice softmax produced zero/non-finite probability")
    return probabilities


def _entropy(probabilities: Sequence[float]) -> float:
    return -math.fsum(value * math.log(value) for value in probabilities)


def _kl(probabilities: Sequence[float], final: Sequence[float]) -> float:
    return math.fsum(value * math.log(value / reference) for value, reference in zip(probabilities, final))


def hidden_geometry(vector: Sequence[float], final: Sequence[float]) -> Tuple[float, float, float]:
    if isinstance(vector, (str, bytes)) or isinstance(final, (str, bytes)):
        raise Phase5ContractError("hidden geometry inputs must be vectors")
    if len(vector) == 0 or len(vector) != len(final):
        raise Phase5ContractError("hidden geometry vectors must have equal non-zero shape")
    left = [_require_finite(value, "hidden vector") for value in vector]
    right = [_require_finite(value, "final hidden vector") for value in final]
    left_norm_sq = math.fsum(value * value for value in left)
    right_norm_sq = math.fsum(value * value for value in right)
    if left_norm_sq == 0.0 or right_norm_sq == 0.0:
        raise Phase5ContractError("zero-norm cosine is forbidden")
    rms_l2 = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)) / len(left))
    cosine = math.fsum(a * b for a, b in zip(left, right)) / math.sqrt(left_norm_sq * right_norm_sq)
    distance = 1.0 - cosine
    if not all(math.isfinite(value) for value in (rms_l2, cosine, distance)):
        raise Phase5ContractError("hidden geometry produced non-finite scalar")
    if cosine < -1.0 - 1e-12 or cosine > 1.0 + 1e-12:
        raise Phase5ContractError("hidden cosine lies outside [-1,1]")
    return rms_l2, cosine, distance


def trajectory_record_from_inputs(
    *,
    identity: Mapping[str, str],
    subject: str,
    prompt_sha256: str,
    boundary_choice_logits: Sequence[Sequence[float]],
    final_normalized_hidden: Sequence[Sequence[float]],
    renderer_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    if len(boundary_choice_logits) != len(BOUNDARY_IDS):
        raise Phase5ContractError("choice logits must cover B_0..B_36")
    if len(final_normalized_hidden) != len(BOUNDARY_IDS):
        raise Phase5ContractError("hidden vectors must cover B_0..B_36")
    probabilities = [stable_choice_probabilities(values) for values in boundary_choice_logits]
    final_probabilities = probabilities[-1]
    final_hidden = final_normalized_hidden[-1]
    boundaries = []
    for boundary_id, distribution, hidden in zip(
        BOUNDARY_IDS, probabilities, final_normalized_hidden
    ):
        l2, cosine, distance = hidden_geometry(hidden, final_hidden)
        boundaries.append(
            {
                "boundary_id": boundary_id,
                "choice_probabilities": distribution,
                "choice_entropy": _entropy(distribution),
                "kl_to_final": _kl(distribution, final_probabilities),
                "hidden_l2_to_final": l2,
                "hidden_cosine_to_final": cosine,
                "hidden_cosine_distance_to_final": distance,
            }
        )
    record = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "identity": dict(identity),
        "subject": subject,
        "split": "validation",
        "prompt_sha256": prompt_sha256,
        "renderer_provenance": dict(renderer_provenance),
        "boundaries": boundaries,
    }
    validate_trajectory_record(record)
    return record


def _scan_forbidden_keys(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN_KEY_TOKENS):
                raise Phase5ContractError("forbidden selector field at %s.%s" % (path, key))
            _scan_forbidden_keys(nested, "%s.%s" % (path, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_forbidden_keys(nested, "%s[%d]" % (path, index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise Phase5ContractError("non-finite selector value at %s" % path)


def validate_trajectory_record(record: Mapping[str, Any]) -> None:
    _require_exact_keys(record, _TOP_LEVEL_TRAJECTORY_KEYS, "trajectory record")
    if record["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise Phase5ContractError("trajectory schema differs")
    _require_exact_keys(record["identity"], _IDENTITY_KEYS, "trajectory identity")
    _require_sha256(record["identity"]["doc_hash"], "identity.doc_hash")
    _require_sha256(record["prompt_sha256"], "prompt_sha256")
    if record["split"] != "validation" or not str(record["subject"]).strip():
        raise Phase5ContractError("trajectory split/subject differs")
    renderer = record["renderer_provenance"]
    _require_exact_keys(renderer, _RENDERER_KEYS, "renderer provenance")
    expected_renderer = {
        "model_repo": "Qwen/Qwen3-4B-Base",
        "model_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "tokenizer_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "dataset_repo": "cais/mmlu",
        "dataset_revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "forward_type": "native_no_loop",
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "position": "final_non_padding_prompt_token",
        "projection": "frozen_final_norm_then_tied_output_weight_exact_choice_rows",
        "choice_order": list(CHOICE_ORDER),
        "spaced_choice_token_ids": EXPECTED_SPACED_TOKEN_IDS,
        "final_norm_path": "model.norm",
        "output_head_path": "lm_head_tied_to_model.embed_tokens.weight",
    }
    for key, expected in expected_renderer.items():
        if renderer.get(key) != expected:
            raise Phase5ContractError("renderer_provenance.%s differs" % key)
    _require_sha256(renderer["renderer_manifest_sha256"], "renderer manifest")
    boundaries = record["boundaries"]
    if not isinstance(boundaries, list) or len(boundaries) != len(BOUNDARY_IDS):
        raise Phase5ContractError("trajectory boundaries must be B_0..B_36")
    distributions = []
    for expected_id, row in zip(BOUNDARY_IDS, boundaries):
        if not isinstance(row, Mapping):
            raise Phase5ContractError("trajectory boundary must be an object")
        _require_exact_keys(row, _BOUNDARY_KEYS, "trajectory boundary")
        if row["boundary_id"] != expected_id:
            raise Phase5ContractError("trajectory boundary order differs")
        distribution = row["choice_probabilities"]
        if not isinstance(distribution, list) or len(distribution) != 4:
            raise Phase5ContractError("choice probabilities must contain four values")
        values = [_require_finite(value, "choice probability") for value in distribution]
        if any(value <= 0.0 for value in values):
            raise Phase5ContractError("choice probabilities must be strictly positive")
        if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise Phase5ContractError("choice probabilities are not normalized")
        distributions.append(values)
    final_distribution = distributions[-1]
    for row, distribution in zip(boundaries, distributions):
        entropy = _require_finite(row["choice_entropy"], "choice_entropy")
        kl = _require_finite(row["kl_to_final"], "kl_to_final")
        l2 = _require_finite(row["hidden_l2_to_final"], "hidden_l2_to_final")
        cosine = _require_finite(row["hidden_cosine_to_final"], "hidden_cosine_to_final")
        distance = _require_finite(
            row["hidden_cosine_distance_to_final"], "hidden_cosine_distance_to_final"
        )
        if not math.isclose(entropy, _entropy(distribution), rel_tol=1e-10, abs_tol=1e-10):
            raise Phase5ContractError("choice entropy recomputation failed")
        if not math.isclose(kl, _kl(distribution, final_distribution), rel_tol=1e-10, abs_tol=1e-10):
            raise Phase5ContractError("KL recomputation failed")
        if kl < -1e-10 or l2 < 0.0 or cosine < -1.0 - 1e-12 or cosine > 1.0 + 1e-12:
            raise Phase5ContractError("trajectory scalar range failed")
        if not math.isclose(distance, 1.0 - cosine, rel_tol=1e-10, abs_tol=1e-10):
            raise Phase5ContractError("cosine distance must be derived exactly once")
    final = boundaries[-1]
    if not math.isclose(final["kl_to_final"], 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise Phase5ContractError("final KL must close to zero")
    if not math.isclose(final["hidden_l2_to_final"], 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise Phase5ContractError("final RMS-L2 must close to zero")
    if not math.isclose(final["hidden_cosine_to_final"], 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise Phase5ContractError("final cosine must close to one")
    if not math.isclose(
        final["hidden_cosine_distance_to_final"], 0.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise Phase5ContractError("final cosine distance must close to zero")
    _scan_forbidden_keys(record)


def trajectory_window_signals(record: Mapping[str, Any]) -> Dict[str, Any]:
    validate_trajectory_record(record)
    boundaries = record["boundaries"]
    rows = []
    for start in WINDOW_STARTS:
        entry = boundaries[start]
        exit_ = boundaries[start + WINDOW_WIDTH]
        delta_entropy = exit_["choice_entropy"] - entry["choice_entropy"]
        delta_kl = exit_["kl_to_final"] - entry["kl_to_final"]
        rows.append(
            {
                "start": start,
                "window": "%d:%d" % (start, start + WINDOW_WIDTH - 1),
                "delta_choice_entropy": delta_entropy,
                "delta_kl_to_final": delta_kl,
                "delta_hidden_l2_to_final": exit_["hidden_l2_to_final"]
                - entry["hidden_l2_to_final"],
                "delta_hidden_cosine_to_final": exit_["hidden_cosine_to_final"]
                - entry["hidden_cosine_to_final"],
                "delta_hidden_cosine_distance_to_final": exit_[
                    "hidden_cosine_distance_to_final"
                ]
                - entry["hidden_cosine_distance_to_final"],
                "E": -delta_entropy,
                "K": delta_kl,
            }
        )
    return {"subject": record["subject"], "windows": rows}


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise Phase5ContractError("bootstrap standard deviation needs at least two values")
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _contrast(values: Sequence[float], start: int) -> Tuple[float, float]:
    local = values[start] - 0.5 * (values[start - 1] + values[start + 1])
    flank = values[start] - 0.5 * (
        values[start - WINDOW_WIDTH] + values[start + WINDOW_WIDTH]
    )
    return local, flank


def analyze_signal_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 2000,
    seed: int = 20260722,
) -> Dict[str, Any]:
    """Run the frozen CONSENSUS selector from per-identity E/K window signals."""

    if not samples or replicates < 2:
        raise Phase5ContractError("selector needs samples and at least two replicates")
    normalized = []
    identities = set()
    for index, sample in enumerate(samples):
        subject = str(sample.get("subject") or "").strip()
        identity = str(sample.get("identity") or "sample-%d" % index)
        if not subject or identity in identities:
            raise Phase5ContractError("selector subjects/identities must be non-empty and unique")
        identities.add(identity)
        e_values = [_require_finite(value, "sample E") for value in sample.get("E", [])]
        k_values = [_require_finite(value, "sample K") for value in sample.get("K", [])]
        if len(e_values) != len(WINDOW_STARTS) or len(k_values) != len(WINDOW_STARTS):
            raise Phase5ContractError("selector E/K samples must cover 33 starts")
        normalized.append({"subject": subject, "identity": identity, "E": e_values, "K": k_values})
    normalized.sort(key=lambda row: (row["subject"], row["identity"]))
    point_e = [math.fsum(row["E"][start] for row in normalized) / len(normalized) for start in WINDOW_STARTS]
    point_k = [math.fsum(row["K"][start] for row in normalized) / len(normalized) for start in WINDOW_STARTS]
    groups: Dict[str, List[int]] = {}
    for index, row in enumerate(normalized):
        groups.setdefault(row["subject"], []).append(index)
    rng = random.Random(seed)
    replicate_e: List[List[float]] = []
    replicate_k: List[List[float]] = []
    for _ in range(replicates):
        drawn = []
        for subject in sorted(groups):
            indices = groups[subject]
            drawn.extend(indices[rng.randrange(len(indices))] for _ in indices)
        replicate_e.append(
            [math.fsum(normalized[index]["E"][start] for index in drawn) / len(drawn) for start in WINDOW_STARTS]
        )
        replicate_k.append(
            [math.fsum(normalized[index]["K"][start] for index in drawn) / len(drawn) for start in WINDOW_STARTS]
        )
    rows = []
    replicate_contrasts: Dict[int, Dict[str, List[float]]] = {}
    for start in CONSENSUS_STARTS:
        oh, fh = _contrast(point_e, start)
        ok, fk = _contrast(point_k, start)
        values = {"O_H": [], "O_K": [], "F_H": [], "F_K": []}
        for e_values, k_values in zip(replicate_e, replicate_k):
            replicate_oh, replicate_fh = _contrast(e_values, start)
            replicate_ok, replicate_fk = _contrast(k_values, start)
            values["O_H"].append(replicate_oh)
            values["O_K"].append(replicate_ok)
            values["F_H"].append(replicate_fh)
            values["F_K"].append(replicate_fk)
        points = {"O_H": oh, "O_K": ok, "F_H": fh, "F_K": fk}
        contrast_payload = {}
        for name in ("O_H", "O_K", "F_H", "F_K"):
            standard_error = _sample_standard_deviation(values[name])
            contrast_payload[name] = {
                "point": points[name],
                "standard_error": standard_error,
                "lower95": _linear_percentile(values[name], 0.025),
                "upper95": _linear_percentile(values[name], 0.975),
            }
        comparisons = (start - 1, start + 1, start - WINDOW_WIDTH, start + WINDOW_WIDTH)
        point_eligible = (
            point_e[start] > 0.0
            and point_k[start] > 0.0
            and all(point_e[index] < 0.0 and point_k[index] < 0.0 for index in comparisons)
            and all(contrast_payload[name]["lower95"] > 0.0 for name in contrast_payload)
        )
        score = min(
            points[name] / max(contrast_payload[name]["standard_error"], 1e-12)
            for name in ("O_H", "O_K", "F_H", "F_K")
        )
        rows.append(
            {
                "start": start,
                "window": "%d:%d" % (start, start + WINDOW_WIDTH - 1),
                "E": point_e[start],
                "K": point_k[start],
                "contrasts": contrast_payload,
                "score": score,
                "point_eligible": point_eligible,
            }
        )
        replicate_contrasts[start] = values
    ranking = sorted(rows, key=lambda row: (-row["score"], row["start"]))
    candidates = [row for row in ranking if row["point_eligible"]]
    decision = "SELECTED"
    selected: Optional[Mapping[str, Any]] = None
    frequency: Optional[float] = None
    if not candidates:
        decision = ABSTAIN_LABELS[0]
    elif len(candidates) > 1 and math.isclose(
        candidates[0]["score"], candidates[1]["score"], rel_tol=1e-12, abs_tol=1e-12
    ):
        decision = ABSTAIN_LABELS[1]
    else:
        selected = candidates[0]
        winner_count = 0
        for replicate_index in range(replicates):
            replicate_rows = []
            for candidate in candidates:
                start = candidate["start"]
                score = min(
                    replicate_contrasts[start][name][replicate_index]
                    / max(candidate["contrasts"][name]["standard_error"], 1e-12)
                    for name in ("O_H", "O_K", "F_H", "F_K")
                )
                replicate_rows.append((score, start))
            replicate_winner = sorted(replicate_rows, key=lambda row: (-row[0], row[1]))[0][1]
            if replicate_winner == selected["start"]:
                winner_count += 1
        frequency = winner_count / float(replicates)
        if frequency < 0.80:
            decision = ABSTAIN_LABELS[2]
    return {
        "schema_version": "loopscope.phase5.selector-report.v1",
        "selector": "ENTROPY_KL_CONSENSUS_WIDTH4_V1",
        "replicates": replicates,
        "seed": seed,
        "published_ranking": ranking,
        "eligible_count": len(candidates),
        "selected_window": selected["window"] if selected is not None and decision == "SELECTED" else None,
        "point_top1_window": selected["window"] if selected is not None else None,
        "selection_frequency": frequency,
        "window_decision": decision,
        "geometry_fields_consumed": False,
        "outcome_fields_consumed": False,
    }


def construct_panel(
    published_ranking: Sequence[Mapping[str, Any]],
    *,
    selected_window: Optional[str],
    known_windows: Sequence[str],
) -> Dict[str, Any]:
    known = set(str(value) for value in known_windows)
    if "15:18" not in known:
        raise Phase5ContractError("known registry must include 15:18")
    seen = set()
    blind_rows = []
    for row in published_ranking:
        window = str(row.get("window") or "")
        if not window or window in seen:
            raise Phase5ContractError("published ranking windows must be non-empty and unique")
        seen.add(window)
        if window not in known:
            blind_rows.append(dict(row))
    if len(blind_rows) < 6:
        raise Phase5ContractError("known-window exclusion leaves fewer than six blind windows")
    raw_high3 = [row["window"] for row in blind_rows[:3]]
    panel_high3 = list(raw_high3)
    replacement = None
    selected_known = selected_window in known if selected_window is not None else False
    if selected_window is not None and not selected_known:
        if selected_window not in {row["window"] for row in blind_rows}:
            raise Phase5ContractError("selected window is absent from registry-excluded ranking")
        if selected_window not in panel_high3:
            replacement = {"removed": panel_high3[-1], "inserted": selected_window}
            panel_high3[-1] = selected_window
    low3 = []
    for row in reversed(blind_rows):
        window = row["window"]
        if window not in panel_high3 and window not in low3:
            low3.append(window)
        if len(low3) == 3:
            break
    cells = ["baseline", "15:18"] + panel_high3 + low3
    deduplicated = []
    for cell in cells:
        if cell not in deduplicated:
            deduplicated.append(cell)
    if len(deduplicated) > 8 or len(low3) != 3 or set(panel_high3) & set(low3):
        raise Phase5ContractError("panel unique-cell/disjointness closure failed")
    return {
        "schema_version": "loopscope.phase5.panel-construction.v1",
        "known_outcome_registry": sorted(known),
        "raw_high3": raw_high3,
        "panel_high3": panel_high3,
        "selected_inclusive_replacement": replacement,
        "selected_is_known_comparator": selected_known,
        "blind_low3": low3,
        "unique_cells": deduplicated,
        "unique_cell_count": len(deduplicated),
    }


def artifact_hashes(paths: Mapping[str, Path]) -> Dict[str, Dict[str, str]]:
    result = {}
    for name, path in paths.items():
        payload = load_json(path)
        _require_manifest_hash(payload, name)
        result[name] = {
            "relative_path": str(path),
            "file_sha256": file_sha256(path),
            "manifest_sha256": payload["manifest_sha256"],
        }
    return result


def validate_gate_a_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _require_manifest_hash(receipt, "Gate A receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise Phase5ContractError("Gate A receipt schema differs")
    if receipt.get("gate") != "A" or receipt.get("result") != "READY_FOR_PLANNING_AUDIT":
        raise Phase5ContractError("Gate A receipt state differs")
    if receipt.get("reuse_verdict") != {
        "baseline": "FRESH_ACQUISITION_REQUIRED",
        "15:18": "FRESH_ACQUISITION_REQUIRED",
    }:
        raise Phase5ContractError("Gate A receipt reuse verdict differs")
    git = receipt.get("git")
    if not isinstance(git, Mapping) or git.get("base_commit") != (
        "ee676a8d13d1bed4aa783da0606f5dd60ec8a11f"
    ):
        raise Phase5ContractError("Gate A receipt Git base differs")
    if git.get("final_commit_binding") != "GATE_A_FINAL_AUDIT_AND_PUSHED_BRANCH_HEAD":
        raise Phase5ContractError("Gate A receipt final commit binding differs")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Phase5ContractError("Gate A receipt artifacts are missing")
    for name, relative in {
        "card": "configs/loopscope/phase5_card.json",
        "schema": "configs/loopscope/phase5_card_schema.json",
        "registry": "configs/loopscope/phase5_known_outcome_registry.json",
    }.items():
        expected = artifacts.get(name)
        if not isinstance(expected, Mapping) or expected.get("relative_path") != relative:
            raise Phase5ContractError("Gate A receipt %s path differs" % name)
        path = Path(root) / relative
        payload = load_json(path)
        if expected.get("file_sha256") != file_sha256(path):
            raise Phase5ContractError("Gate A receipt %s file hash differs" % name)
        if expected.get("manifest_sha256") != payload.get("manifest_sha256"):
            raise Phase5ContractError("Gate A receipt %s semantic hash differs" % name)
    remote = receipt.get("remote_provenance")
    if not isinstance(remote, Mapping):
        raise Phase5ContractError("Gate A receipt remote provenance is missing")
    for key in (
        "model_config_sha256",
        "tokenizer_json_sha256",
        "validation_renderer_manifest_file_sha256",
        "test_identity_manifest_file_sha256",
        "validation_test_disjointness_receipt_sha256",
    ):
        _require_sha256(remote.get(key), "receipt.remote_provenance.%s" % key)
    if remote.get("outcome_values_read") is not False:
        raise Phase5ContractError("Gate A receipt reports outcome access")
    forbidden = receipt.get("actions_not_taken")
    required_not_taken = {
        "formal_model_forward",
        "formal_validation_trajectory",
        "test_outcome_access",
        "GPU_or_Slurm",
        "Gate_B_C_or_D_work",
    }
    if not isinstance(forbidden, list) or not required_not_taken.issubset(forbidden):
        raise Phase5ContractError("Gate A receipt forbidden-action closure differs")


__all__ = [
    "ABSTAIN_LABELS",
    "BOUNDARY_IDS",
    "CARD_SCHEMA_VERSION",
    "CHOICE_ORDER",
    "CONSENSUS_STARTS",
    "EXPECTED_PLAIN_TOKEN_IDS",
    "EXPECTED_SPACED_TOKEN_IDS",
    "LAYER_COUNT",
    "Phase5ContractError",
    "SHIFT_STARTS",
    "TRAJECTORY_SCHEMA_VERSION",
    "WINDOW_STARTS",
    "analyze_signal_samples",
    "artifact_hashes",
    "boundary_contract",
    "canonical_json_bytes",
    "construct_panel",
    "file_sha256",
    "hidden_geometry",
    "load_json",
    "semantic_sha256",
    "stable_choice_probabilities",
    "trajectory_record_from_inputs",
    "trajectory_window_signals",
    "validate_card",
    "validate_card_schema",
    "validate_gate_a_receipt",
    "validate_registry",
    "validate_token_ids",
    "validate_trajectory_record",
]
