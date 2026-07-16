"""Closed-world schemas and identity checks for LoopScope Phase 4."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class Phase4ContractError(ValueError):
    """Raised when a Phase 4 artifact violates the frozen contract."""


BOUNDARY_IDS: Tuple[int, ...] = tuple(range(37))
WIDTHS: Tuple[int, ...] = tuple(range(2, 9))
WIDTH4_STARTS: Tuple[int, ...] = tuple(range(33))
WIDTH4_CONSENSUS_STARTS: Tuple[int, ...] = tuple(range(4, 29))
EXPECTED_SHARED_COUNT = 12032
CARD_SCHEMA_VERSION = "loopscope.phase4.pv-ek-trs-card.v1"
RECEIPT_SCHEMA_VERSION = "loopscope.phase4.card-verifier-receipt.v1"
PROVENANCE_SCHEMA_VERSION = "loopscope.phase4.provenance-closure.v1"
CLOSURE_MODES = ("REUSE_CLOSED", "NOT_REUSABLE_FRESH_REQUIRED")
REUSE_REQUIREMENT_KEYS = {
    "current_checkpoint_model_tokenizer_revision",
    "dataset_revision_and_fingerprint",
    "task_renderer_chat_prefix_contract",
    "generation_metric_dtype_recipe",
    "baseline_and_15_18_artifact_identity",
    "shared_12032_identity_order_prompt_hashes",
    "per_sample_availability",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40_64 = re.compile(r"^[0-9a-f]{40,64}$")
_FORBIDDEN_KEYS = {
    "answer",
    "answer_index",
    "gold",
    "gold_label",
    "label",
    "cot_content",
    "correct",
    "correctness",
    "accuracy",
    "acc",
    "generated_token",
    "generated_tokens",
    "generated_answer",
    "generated_response",
    "teacher_forcing",
    "baseline_outcome",
    "loop_outcome",
    "answer_flip",
    "full_logits",
    "logits",
    "probabilities",
    "log_probabilities",
    "hidden_states",
    "hidden_tensors",
    "residual_tensors",
}
_FORBIDDEN_KEY_FRAGMENTS = (
    "cot_content",
    "logits",
    "log_probabilities",
    "probabilities",
    "hidden_state",
    "hidden_tensor",
    "residual_tensor",
    "tensor_payload",
)
_FORBIDDEN_TARGET_SUFFIXES = (
    "answer",
    "answers",
    "cot",
    "cot_content",
    "correct",
    "correctness",
    "gold",
    "label",
    "token",
    "tokens",
)

SOURCE_RECORD_KEYS = {
    "schema_version",
    "dataset_revision",
    "split",
    "question_id",
    "category",
    "src",
    "question",
    "ordered_options",
    "canonical_identity",
    "sanitized_content_sha256",
    "rendered_prefix_sha256",
    "rendered_token_ids_sha256",
    "attention_mask_sha256",
    "last_effective_prefix_token_index",
    "renderer_provenance",
}

TRAJECTORY_RECORD_KEYS = {
    "schema_version",
    "canonical_identity",
    "category",
    "boundaries",
    "producer_provenance",
}

BOUNDARY_RECORD_KEYS = {
    "boundary_id",
    "full_vocabulary_entropy",
    "kl_to_final",
    "normalized_entropy",
    "logit_rms",
    "top1_probability_mass",
    "top10_probability_mass",
    "top100_probability_mass",
    "finite",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_bad_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Phase4ContractError("cannot load strict JSON artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise Phase4ContractError("JSON artifact must contain an object: %s" % path)
    return value


def load_phase4_card(path: Path, *, require_provenance_closed: bool = True) -> Dict[str, Any]:
    card = load_json_object(path)
    validate_phase4_card(card, require_provenance_closed=require_provenance_closed)
    return card


def validate_phase4_card(
    card: Mapping[str, Any], *, require_provenance_closed: bool = True
) -> None:
    if card.get("schema_version") != CARD_SCHEMA_VERSION:
        raise Phase4ContractError("unexpected Phase 4 card schema_version")
    if card.get("card") != "H4_PV_EK_TRS_QWEN4B_MMLUPRO_V1":
        raise Phase4ContractError("unexpected Phase 4 card identity")
    model = _mapping(card, "model")
    task = _mapping(card, "task")
    selector = _mapping(card, "selector")
    outcome = _mapping(card, "formal_outcome_recipe")
    bootstrap = _mapping(card, "bootstrap")
    panel = _mapping(card, "panel")
    provenance = _mapping(card, "provenance_closure")

    _require_equal(model, "repo", "Qwen/Qwen3-4B-Instruct-2507")
    _require_equal(model, "decoder_layers", 36)
    _require_equal(model, "dtype", "bfloat16")
    _require_equal(task, "dataset", "TIGER-Lab/MMLU-Pro")
    _require_equal(task, "task_alias", "mmlu_pro")
    _require_equal(task, "num_fewshot", 5)
    _require_equal(task, "selector_split", "test")
    _require_equal(task, "shared_population_count", EXPECTED_SHARED_COUNT)
    _require_equal(selector, "boundaries", list(BOUNDARY_IDS))
    _require_equal(selector, "primary_width", 4)
    _require_equal(selector, "width4_start_count", 33)
    _require_equal(selector, "width4_consensus_count", 25)
    _require_equal(selector, "secondary_widths", list(WIDTHS))
    _require_equal(selector, "secondary_strict_count", 154)
    _require_equal(selector, "secondary_edge_count", 224)
    _require_equal(outcome, "window_width", 4)
    _require_equal(outcome, "k", 3)
    _require_equal(outcome, "iteration_mode", "block")
    _require_equal(outcome, "strategy", "euler")
    _require_close(outcome, "step_size", 1.0 / 3.0)
    _require_equal(outcome, "cache_strategy", "first")
    _require_equal(outcome, "decode_mode", "full")
    _require_equal(bootstrap, "selector_replicates", 2000)
    _require_equal(bootstrap, "selector_seed", 20260717)
    _require_equal(panel, "fixed_comparator", "15:18")
    _require_close(panel, "competitiveness_margin_percentage_points", 0.30)
    _require_equal(panel, "selection_frequency_threshold", 0.80)

    required_exact = provenance.get("required_exact_fields")
    if not isinstance(required_exact, list) or not required_exact:
        raise Phase4ContractError("provenance_closure.required_exact_fields is missing")
    values = provenance.get("values")
    if not isinstance(values, Mapping):
        raise Phase4ContractError("provenance_closure.values must be an object")
    missing = [name for name in required_exact if not _is_exact_provenance_value(values.get(name))]
    state = provenance.get("state")
    if missing and state != "UNRESOLVED_LOCAL_EVIDENCE":
        raise Phase4ContractError("unresolved provenance must be explicit")
    if not missing and state != "CLOSED":
        raise Phase4ContractError("closed provenance values require state=CLOSED")
    if require_provenance_closed and missing:
        raise Phase4ContractError(
            "exact immutable provenance is not closed: %s" % ", ".join(missing)
        )
    if not missing:
        if card.get("status") != "P4A_PROVENANCE_CLOSED":
            raise Phase4ContractError("closed provenance requires P4A_PROVENANCE_CLOSED status")
        if provenance.get("closure_mode") not in CLOSURE_MODES:
            raise Phase4ContractError("closed provenance requires an exact closure_mode")
        evidence_artifact = provenance.get("evidence_artifact")
        if not isinstance(evidence_artifact, Mapping):
            raise Phase4ContractError("closed provenance requires an evidence_artifact")
        relative_path = evidence_artifact.get("relative_path")
        evidence_path = Path(relative_path) if isinstance(relative_path, str) else None
        if (
            evidence_path is None
            or evidence_path.parent != Path("configs/loopscope")
            or not evidence_path.name.startswith("phase4_")
            or evidence_path.suffix != ".json"
        ):
            raise Phase4ContractError("provenance evidence path is not a Phase 4 config")
        if not isinstance(evidence_artifact.get("file_sha256"), str) or not _HEX64.fullmatch(
            evidence_artifact["file_sha256"]
        ):
            raise Phase4ContractError("provenance evidence file_sha256 is invalid")
        if model.get("revision") != values.get("model_revision"):
            raise Phase4ContractError("card model revision differs from provenance")
        if model.get("tokenizer_revision") != values.get("tokenizer_revision"):
            raise Phase4ContractError("card tokenizer revision differs from provenance")
        if task.get("dataset_revision") != values.get("dataset_revision"):
            raise Phase4ContractError("card dataset revision differs from provenance")
        if task.get("dataset_fingerprint") != values.get("dataset_fingerprint"):
            raise Phase4ContractError("card dataset fingerprint differs from provenance")


def validate_provenance_evidence(
    card: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    """Validate the exact REUSE or pre-registered fresh-acquisition closure branch."""

    validate_phase4_card(card, require_provenance_closed=True)
    if evidence.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise Phase4ContractError("unexpected Phase 4 provenance evidence schema_version")
    if evidence.get("card") != card.get("card"):
        raise Phase4ContractError("provenance evidence card identity mismatch")
    if evidence.get("provenance_closed") is not True:
        raise Phase4ContractError("provenance evidence must be explicitly closed")

    provenance = _mapping(card, "provenance_closure")
    mode = provenance.get("closure_mode")
    if evidence.get("closure_mode") != mode:
        raise Phase4ContractError("card/evidence closure_mode mismatch")

    values = _mapping(provenance, "values")
    required_exact = provenance.get("required_exact_fields")
    field_evidence = evidence.get("field_evidence")
    if not isinstance(field_evidence, Mapping) or set(field_evidence) != set(required_exact):
        raise Phase4ContractError("field_evidence must exactly cover the 17 provenance fields")
    for name in required_exact:
        entry = field_evidence.get(name)
        if not isinstance(entry, Mapping) or entry.get("value") != values.get(name):
            raise Phase4ContractError("field evidence value mismatch for %s" % name)
        if not isinstance(entry.get("role"), str) or not entry["role"].strip():
            raise Phase4ContractError("field evidence role is missing for %s" % name)
        sources = entry.get("sources")
        if not isinstance(sources, list) or not sources:
            raise Phase4ContractError("field evidence sources are missing for %s" % name)
        for source in sources:
            _validate_provenance_source(source, field=name)

    prospective = evidence.get("prospective_fresh_contract")
    if not isinstance(prospective, Mapping):
        raise Phase4ContractError("prospective_fresh_contract is missing")
    if prospective.get("contract_role") != "prospective_fresh_contract":
        raise Phase4ContractError("prospective contract role mismatch")
    if prospective.get("p4b_renderer_smoke_required") is not True:
        raise Phase4ContractError("P4-B renderer smoke must remain required")
    dataset_components = prospective.get("dataset_fingerprint_components")
    if not isinstance(dataset_components, Mapping) or prospective.get(
        "dataset_fingerprint"
    ) != semantic_sha256(dataset_components):
        raise Phase4ContractError("dataset fingerprint components do not recompute")
    if prospective.get("dataset_fingerprint") != values.get("dataset_fingerprint"):
        raise Phase4ContractError("prospective dataset fingerprint mismatch")
    category_yamls = prospective.get("category_yaml_sha256")
    if not isinstance(category_yamls, Mapping) or len(category_yamls) != 14:
        raise Phase4ContractError("all 14 MMLU-Pro category YAML hashes are required")
    prefix_contract = prospective.get("rendered_prefix_contract")
    if not isinstance(prefix_contract, Mapping):
        raise Phase4ContractError("rendered-prefix contract is missing")
    prefix_hash = prospective.get("rendered_prefix_contract_sha256")
    if prefix_hash != semantic_sha256(prefix_contract):
        raise Phase4ContractError("rendered-prefix contract SHA256 mismatch")
    if prefix_contract.get("category_yaml_bundle_sha256") != semantic_sha256(category_yamls):
        raise Phase4ContractError("MMLU-Pro category YAML bundle SHA256 mismatch")
    prospective_checks = {
        "model_revision": prefix_contract.get("model_revision"),
        "tokenizer_revision": prefix_contract.get("tokenizer_revision"),
        "mmlu_pro_task_yaml_sha256": prefix_contract.get("group_yaml_sha256"),
        "renderer_source_sha256": prefix_contract.get("renderer_source_sha256"),
        "chat_template_sha256": prefix_contract.get("chat_template_sha256"),
        "rendered_prefix_contract_sha256": prefix_hash,
        "generation_kwargs": prospective.get("generation_kwargs"),
        "primary_metric_name": prospective.get("primary_metric_name"),
    }
    for name, observed in prospective_checks.items():
        if values.get(name) != observed:
            raise Phase4ContractError("prospective contract mismatch for %s" % name)
    if prefix_contract.get("contract_role") != "prospective_fresh_contract":
        raise Phase4ContractError("rendered-prefix contract must be prospective")
    if prefix_contract.get("apply_chat_template") is not False:
        raise Phase4ContractError("current HFLM contract requires apply_chat_template=false")
    if prefix_contract.get("target_generated_token_count_before_selector") != 0:
        raise Phase4ContractError("selector prefix contract must generate zero target tokens")

    historical = evidence.get("historical_run")
    if not isinstance(historical, Mapping):
        raise Phase4ContractError("historical_run evidence is missing")
    baseline = _exact_historical_identity(historical, "historical_baseline_identity")
    fixed = _exact_historical_identity(historical, "historical_15_18_identity")
    if fixed.get("window") != "15:18":
        raise Phase4ContractError("historical fixed comparator must remain 15:18")
    baseline_hash = historical.get("historical_baseline_identity_sha256")
    fixed_hash = historical.get("historical_15_18_identity_sha256")
    if baseline_hash != semantic_sha256(baseline) or fixed_hash != semantic_sha256(fixed):
        raise Phase4ContractError("historical artifact identity SHA256 mismatch")
    if values["historical_baseline_identity"].get("identity_sha256") != baseline_hash:
        raise Phase4ContractError("card baseline identity hash mismatch")
    if values["historical_15_18_identity"].get("identity_sha256") != fixed_hash:
        raise Phase4ContractError("card 15:18 identity hash mismatch")
    for name, identity in (
        ("historical_baseline_identity", baseline),
        ("historical_15_18_identity", fixed),
    ):
        card_identity = values[name]
        if card_identity.get("artifact_path") != identity.get("artifact_path"):
            raise Phase4ContractError("card artifact path mismatch for %s" % name)
        if card_identity.get("artifact_sha256") != identity.get("artifact_sha256"):
            raise Phase4ContractError("card artifact SHA256 mismatch for %s" % name)

    registry = historical.get("known_outcome_registry")
    if not isinstance(registry, list) or not registry:
        raise Phase4ContractError("known_outcome_registry must identify 15:18")
    if registry != values.get("known_outcome_registry"):
        raise Phase4ContractError("card/evidence known_outcome_registry mismatch")
    fixed_entries = [entry for entry in registry if isinstance(entry, Mapping) and entry.get("window") == "15:18"]
    if len(fixed_entries) != 1:
        raise Phase4ContractError("known_outcome_registry must contain exactly one 15:18 identity")
    fixed_entry = fixed_entries[0]
    if fixed_entry.get("artifact_identity_sha256") != fixed_hash:
        raise Phase4ContractError("known 15:18 registry identity mismatch")
    if fixed_entry.get("outcome_value_recorded") is not False:
        raise Phase4ContractError("known-outcome registry must not record an outcome value")

    requirements = historical.get("reuse_requirements")
    if not isinstance(requirements, Mapping) or set(requirements) != REUSE_REQUIREMENT_KEYS:
        raise Phase4ContractError("reuse_requirements must cover every reuse admission condition")
    if any(not isinstance(value, bool) for value in requirements.values()):
        raise Phase4ContractError("reuse requirement values must be boolean")
    fallback = historical.get("p4c_fallback")
    if not isinstance(fallback, Mapping):
        raise Phase4ContractError("P4-C fallback state is missing")

    if mode == "REUSE_CLOSED":
        if historical.get("reuse_status") != "REUSE_CLOSED" or not all(requirements.values()):
            raise Phase4ContractError("REUSE_CLOSED requires complete reuse evidence")
        if historical.get("non_reuse_reasons") not in (None, []):
            raise Phase4ContractError("REUSE_CLOSED cannot carry non-reuse reasons")
        if fallback.get("baseline") != "REUSE_AUTHORIZED" or fallback.get(
            "fixed_15_18"
        ) != "REUSE_AUTHORIZED":
            raise Phase4ContractError("REUSE_CLOSED requires reuse-authorized P4-C cells")
        if values["historical_baseline_identity"].get("reuse_status") != "REUSE_CLOSED":
            raise Phase4ContractError("reused baseline identity must be marked REUSE_CLOSED")
        if values["historical_15_18_identity"].get("reuse_status") != "REUSE_CLOSED":
            raise Phase4ContractError("reused 15:18 identity must be marked REUSE_CLOSED")
    elif mode == "NOT_REUSABLE_FRESH_REQUIRED":
        reasons = historical.get("non_reuse_reasons")
        if historical.get("reuse_status") != "NOT_REUSABLE":
            raise Phase4ContractError("non-reuse branch requires reuse_status=NOT_REUSABLE")
        if not isinstance(reasons, list) or not reasons or any(
            not isinstance(reason, str) or not reason.strip() for reason in reasons
        ):
            raise Phase4ContractError("non-reuse branch requires exact non_reuse_reasons")
        if all(requirements.values()):
            raise Phase4ContractError("non-reuse branch must identify a failed reuse requirement")
        if fallback.get("baseline") != "FRESH_ACQUISITION_REQUIRED" or fallback.get(
            "fixed_15_18"
        ) != "FRESH_ACQUISITION_REQUIRED":
            raise Phase4ContractError("non-reuse branch requires fresh baseline and 15:18")
        if values["historical_baseline_identity"].get("reuse_status") != "NOT_REUSABLE":
            raise Phase4ContractError("baseline identity must be marked NOT_REUSABLE")
        if values["historical_15_18_identity"].get("reuse_status") != "NOT_REUSABLE":
            raise Phase4ContractError("15:18 identity must be marked NOT_REUSABLE")
    else:  # validate_phase4_card already rejects this; keep the branch explicit here.
        raise Phase4ContractError("unsupported provenance closure mode")


def _validate_provenance_source(source: Any, *, field: str) -> None:
    if not isinstance(source, Mapping):
        raise Phase4ContractError("provenance source must be an object for %s" % field)
    if not isinstance(source.get("path"), str) or not source["path"].strip():
        raise Phase4ContractError("provenance source path is missing for %s" % field)
    sha256 = source.get("sha256")
    git_object = source.get("git_object")
    if not (
        isinstance(sha256, str)
        and _HEX64.fullmatch(sha256)
        or isinstance(git_object, str)
        and _HEX40_64.fullmatch(git_object)
    ):
        raise Phase4ContractError("provenance source hash/object is missing for %s" % field)
    if not isinstance(source.get("access"), str) or not source["access"].strip():
        raise Phase4ContractError("provenance source access mode is missing for %s" % field)
    if source.get("outcome_content_displayed") is not False:
        raise Phase4ContractError("provenance source displayed outcome content for %s" % field)


def _exact_historical_identity(historical: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    identity = historical.get(name)
    if not isinstance(identity, Mapping):
        raise Phase4ContractError("%s must be an exact artifact identity" % name)
    path = identity.get("artifact_path")
    sha256 = identity.get("artifact_sha256")
    if not isinstance(path, str) or not path.strip():
        raise Phase4ContractError("%s artifact_path is missing" % name)
    if not isinstance(sha256, str) or not _HEX64.fullmatch(sha256):
        raise Phase4ContractError("%s artifact_sha256 is invalid" % name)
    return identity


def scan_forbidden_fields(value: Any, *, path: str = "$", allow_options: bool = True) -> None:
    """Recursively reject outcome/gold/tensor fields on the normal producer path."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise Phase4ContractError("non-string field name at %s" % path)
            key = raw_key.strip().lower()
            target_alias = key.startswith("target_") and any(
                key.endswith(suffix) for suffix in _FORBIDDEN_TARGET_SUFFIXES
            )
            generated_alias = key.startswith("generated_") and any(
                token in key for token in ("answer", "response", "token", "cot")
            )
            tensor_alias = any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS)
            if (
                key in _FORBIDDEN_KEYS
                or key.startswith("outcome_")
                or key.endswith("_outcome")
                or target_alias
                or generated_alias
                or tensor_alias
            ):
                raise Phase4ContractError("forbidden Phase 4 selector field %s.%s" % (path, raw_key))
            if not allow_options and key in {"ordered_options", "options"}:
                raise Phase4ContractError("options are not allowed in trajectory artifacts")
            scan_forbidden_fields(child, path="%s.%s" % (path, raw_key), allow_options=allow_options)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_forbidden_fields(child, path="%s[%d]" % (path, index), allow_options=allow_options)
    elif isinstance(value, float) and not math.isfinite(value):
        raise Phase4ContractError("non-finite value at %s" % path)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise Phase4ContractError("non-JSON payload is forbidden at %s" % path)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        raise Phase4ContractError("text field must be a string")
    normalized = unicodedata.normalize("NFC", value).strip()
    return " ".join(normalized.split())


def sanitized_content_sha256(question: str, ordered_options: Sequence[str]) -> str:
    if not isinstance(ordered_options, Sequence) or isinstance(ordered_options, (str, bytes)):
        raise Phase4ContractError("ordered_options must be a sequence")
    if len(ordered_options) < 2:
        raise Phase4ContractError("ordered_options must contain at least two values")
    payload = {
        "question": normalize_text(question),
        "ordered_options": [normalize_text(item) for item in ordered_options],
    }
    return semantic_sha256(payload)


def canonical_identity(record: Mapping[str, Any]) -> str:
    question_id = record.get("question_id")
    try:
        numeric_id = int(str(question_id).strip())
    except (TypeError, ValueError) as exc:
        raise Phase4ContractError("question_id must be numeric") from exc
    if numeric_id < 0:
        raise Phase4ContractError("question_id must be non-negative")
    category = normalize_text(record.get("category"))
    src = normalize_text(record.get("src"))
    content_hash = sanitized_content_sha256(record.get("question"), record.get("ordered_options"))
    return "%d:%s:%s:%s" % (numeric_id, category, src, content_hash)


def validate_source_record(record: Mapping[str, Any]) -> None:
    scan_forbidden_fields(record)
    _closed_world(record, SOURCE_RECORD_KEYS, "source record")
    if record.get("schema_version") != "loopscope.phase4.shared-source-record.v1":
        raise Phase4ContractError("unexpected source record schema_version")
    if record.get("split") != "test":
        raise Phase4ContractError("source record split must be test")
    revision = record.get("dataset_revision")
    if not isinstance(revision, str) or not _HEX40_64.fullmatch(revision):
        raise Phase4ContractError("dataset_revision must be an exact lowercase revision")
    options = record.get("ordered_options")
    expected_content = sanitized_content_sha256(record.get("question"), options)
    if record.get("sanitized_content_sha256") != expected_content:
        raise Phase4ContractError("sanitized_content_sha256 mismatch")
    if record.get("canonical_identity") != canonical_identity(record):
        raise Phase4ContractError("canonical_identity mismatch")
    for key in (
        "rendered_prefix_sha256",
        "rendered_token_ids_sha256",
        "attention_mask_sha256",
    ):
        if not isinstance(record.get(key), str) or not _HEX64.fullmatch(record[key]):
            raise Phase4ContractError("%s must be a SHA256" % key)
    index = record.get("last_effective_prefix_token_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise Phase4ContractError("last_effective_prefix_token_index must be non-negative")
    if not isinstance(record.get("renderer_provenance"), Mapping):
        raise Phase4ContractError("renderer_provenance must be an object")


def validate_shared_identity_contract(
    expected_rows: Sequence[Mapping[str, Any]],
    observed_rows: Sequence[Mapping[str, Any]],
    *,
    require_full_population: bool = True,
) -> None:
    """Fail fast on missing, extra, reordered, or duplicate shared identities."""

    if require_full_population and len(expected_rows) != EXPECTED_SHARED_COUNT:
        raise Phase4ContractError("canonical shared manifest must contain exactly 12032 rows")
    for row in expected_rows:
        validate_source_record(row)
    expected = [row["canonical_identity"] for row in expected_rows]
    canonical_rows = sorted(
        expected_rows,
        key=lambda row: (int(str(row["question_id"]).strip()), row["sanitized_content_sha256"]),
    )
    if [row["canonical_identity"] for row in canonical_rows] != expected:
        raise Phase4ContractError("canonical shared manifest order is not question_id/hash order")
    question_id_to_hash: Dict[int, str] = {}
    for row in expected_rows:
        question_id = int(str(row["question_id"]).strip())
        content_hash = row["sanitized_content_sha256"]
        prior = question_id_to_hash.setdefault(question_id, content_hash)
        if prior != content_hash:
            raise Phase4ContractError("repeated question_id has conflicting content")
    observed = []
    closure_fields = (
        "canonical_identity",
        "sanitized_content_sha256",
        "rendered_prefix_sha256",
        "rendered_token_ids_sha256",
        "attention_mask_sha256",
    )
    for index, row in enumerate(observed_rows):
        if not isinstance(row, Mapping):
            raise Phase4ContractError("observed identity row must be an object")
        scan_forbidden_fields(row)
        identity = row.get("canonical_identity")
        if not isinstance(identity, str) or not identity:
            raise Phase4ContractError("observed canonical_identity is missing")
        observed.append(identity)
        if index < len(expected_rows):
            for field in closure_fields:
                if row.get(field) != expected_rows[index].get(field):
                    raise Phase4ContractError("shared identity/hash closure mismatch at %s" % field)
    if len(set(expected)) != len(expected):
        raise Phase4ContractError("canonical shared manifest contains duplicate identity")
    if len(set(observed)) != len(observed):
        raise Phase4ContractError("observed rows contain duplicate identity")
    expected_set, observed_set = set(expected), set(observed)
    if expected_set != observed_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        raise Phase4ContractError(
            "shared identity membership mismatch: missing=%d extra=%d" % (len(missing), len(extra))
        )
    if observed != expected:
        raise Phase4ContractError("shared identity order mismatch")


def validate_trajectory_record(record: Mapping[str, Any], *, d36_tolerance: float) -> None:
    scan_forbidden_fields(record, allow_options=False)
    _closed_world(record, TRAJECTORY_RECORD_KEYS, "trajectory record")
    if record.get("schema_version") != "loopscope.phase4.prefix-trajectory-record.v1":
        raise Phase4ContractError("unexpected trajectory schema_version")
    if not isinstance(record.get("canonical_identity"), str) or not record["canonical_identity"]:
        raise Phase4ContractError("trajectory canonical_identity is missing")
    if not isinstance(record.get("category"), str) or not record["category"].strip():
        raise Phase4ContractError("trajectory category is missing")
    boundaries = record.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != 37:
        raise Phase4ContractError("trajectory must contain exactly B_0..B_36")
    for expected_id, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise Phase4ContractError("boundary record must be an object")
        _closed_world(boundary, BOUNDARY_RECORD_KEYS, "boundary record")
        if boundary.get("boundary_id") != expected_id:
            raise Phase4ContractError("boundary order/id mismatch")
        for key in BOUNDARY_RECORD_KEYS - {"boundary_id", "finite"}:
            value = boundary.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise Phase4ContractError("boundary scalar %s must be finite" % key)
        if boundary.get("finite") is not True:
            raise Phase4ContractError("boundary finite flag must be true")
        if not 0.0 <= boundary["normalized_entropy"] <= 1.0 + 1e-6:
            raise Phase4ContractError("normalized entropy is out of range")
        masses = [
            boundary["top1_probability_mass"],
            boundary["top10_probability_mass"],
            boundary["top100_probability_mass"],
        ]
        if not (0.0 <= masses[0] <= masses[1] <= masses[2] <= 1.0 + 1e-6):
            raise Phase4ContractError("top-k masses are invalid")
    if abs(boundaries[36]["kl_to_final"]) > d36_tolerance:
        raise Phase4ContractError("D_36 exceeds the frozen tolerance")


def source_record_json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "loopscope.phase4.shared-source-record.v1",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(SOURCE_RECORD_KEYS),
        "properties": {
            "schema_version": {"const": "loopscope.phase4.shared-source-record.v1"},
            "dataset_revision": {"type": "string", "pattern": "^[0-9a-f]{40,64}$"},
            "split": {"const": "test"},
            "question_id": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "string", "pattern": "^[0-9]+$"}]},
            "category": {"type": "string", "minLength": 1},
            "src": {"type": "string", "minLength": 1},
            "question": {"type": "string"},
            "ordered_options": {"type": "array", "minItems": 2, "items": {"type": "string"}},
            "canonical_identity": {"type": "string", "minLength": 1},
            "sanitized_content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "rendered_prefix_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "rendered_token_ids_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "attention_mask_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "last_effective_prefix_token_index": {"type": "integer", "minimum": 0},
            "renderer_provenance": {"type": "object"},
        },
    }


def trajectory_record_json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "loopscope.phase4.prefix-trajectory-record.v1",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(TRAJECTORY_RECORD_KEYS),
        "properties": {
            "schema_version": {"const": "loopscope.phase4.prefix-trajectory-record.v1"},
            "canonical_identity": {"type": "string", "minLength": 1},
            "category": {"type": "string", "minLength": 1},
            "boundaries": {
                "type": "array",
                "minItems": 37,
                "maxItems": 37,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(BOUNDARY_RECORD_KEYS),
                    "properties": {
                        "boundary_id": {"type": "integer", "minimum": 0, "maximum": 36},
                        "full_vocabulary_entropy": {"type": "number"},
                        "kl_to_final": {"type": "number"},
                        "normalized_entropy": {"type": "number"},
                        "logit_rms": {"type": "number"},
                        "top1_probability_mass": {"type": "number"},
                        "top10_probability_mass": {"type": "number"},
                        "top100_probability_mass": {"type": "number"},
                        "finite": {"const": True},
                    },
                },
            },
            "producer_provenance": {"type": "object"},
        },
    }


def _closed_world(value: Mapping[str, Any], allowed: Iterable[str], context: str) -> None:
    keys = set(value)
    allowed_set = set(allowed)
    missing, extra = allowed_set - keys, keys - allowed_set
    if missing or extra:
        raise Phase4ContractError(
            "%s is not closed-world: missing=%s extra=%s"
            % (context, sorted(missing), sorted(extra))
        )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise Phase4ContractError("card.%s must be an object" % key)
    return child


def _require_equal(value: Mapping[str, Any], key: str, expected: Any) -> None:
    if value.get(key) != expected:
        raise Phase4ContractError("frozen card field %s differs" % key)


def _require_close(value: Mapping[str, Any], key: str, expected: float) -> None:
    observed = value.get(key)
    if not isinstance(observed, (int, float)) or not math.isclose(
        float(observed), expected, rel_tol=0.0, abs_tol=1e-15
    ):
        raise Phase4ContractError("frozen card field %s differs" % key)


def _is_exact_provenance_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip()) and not value.startswith("UNRESOLVED")
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return value is not None


def _bad_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant %s" % value)


__all__ = [
    "BOUNDARY_IDS",
    "CARD_SCHEMA_VERSION",
    "EXPECTED_SHARED_COUNT",
    "Phase4ContractError",
    "PROVENANCE_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "WIDTH4_CONSENSUS_STARTS",
    "WIDTH4_STARTS",
    "WIDTHS",
    "canonical_identity",
    "canonical_json_bytes",
    "file_sha256",
    "load_phase4_card",
    "sanitized_content_sha256",
    "scan_forbidden_fields",
    "semantic_sha256",
    "source_record_json_schema",
    "trajectory_record_json_schema",
    "validate_phase4_card",
    "validate_provenance_evidence",
    "validate_shared_identity_contract",
    "validate_source_record",
    "validate_trajectory_record",
]
