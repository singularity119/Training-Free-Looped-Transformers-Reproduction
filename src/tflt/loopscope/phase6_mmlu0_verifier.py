"""Independent Gate H verifier and no-execution self-test for MMLU 0-shot."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from tflt.loopscope.phase6_mmlu0_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    FORMAL_REPLICATES,
    FORMAL_SEED,
    LAYERS,
    MMLU0ContractError,
    TRAJECTORY_SCHEMA_VERSION,
    build_trajectory_record,
    choice_logits_to_probabilities,
    file_sha256,
    load_json,
    selector_sample,
    semantic_sha256,
    validate_card,
    validate_choice_surface_manifest,
    validate_schema_document,
    validate_selector_sample,
    validate_trajectory_record,
)


def _expect_failure(label: str, operation: Callable[[], Any]) -> str:
    try:
        operation()
    except (MMLU0ContractError, KeyError, TypeError, ValueError):
        return label
    raise MMLU0ContractError("self-test negative case unexpectedly passed: %s" % label)


def verify_gate_h_contracts(card_path: Path, schema_path: Path) -> Dict[str, Any]:
    card = load_json(card_path)
    schema = load_json(schema_path)
    validate_card(card)
    validate_schema_document(schema)
    candidates = sum(len(value) for value in card["selector"]["candidate_starts"].values())
    if candidates != 42:
        raise MMLU0ContractError("candidate count is not exactly 42")
    if card["selector"]["bootstrap"]["replicates"] != FORMAL_REPLICATES or card["selector"]["bootstrap"]["seed"] != FORMAL_SEED:
        raise MMLU0ContractError("formal bootstrap closure differs")
    return {
        "status": "PASS",
        "card_sha256": file_sha256(card_path),
        "trajectory_schema_sha256": file_sha256(schema_path),
        "model_revision": card["model"]["revision"],
        "task_revision": card["task"]["revision"],
        "validation_record_count": card["task"]["validation_record_count"],
        "validation_subject_count": card["task"]["validation_subject_count"],
        "boundary_count": BOUNDARY_COUNT,
        "candidate_count": candidates,
        "formal_bootstrap_replicates": FORMAL_REPLICATES,
        "formal_bootstrap_seed": FORMAL_SEED,
        "model_or_data_forward_executed": False,
        "formal_selector_executed": False,
        "outcomes_read": False,
    }


def dry_run_payload(card_path: Path, schema_path: Path) -> Dict[str, Any]:
    result = verify_gate_h_contracts(card_path, schema_path)
    result.update(
        {
            "status": "DRY_RUN_VALID",
            "contract": "MMLU validation-1531 zero-shot prefix trajectory; Gate H local only",
            "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
            "selector_contract_deferred_to": "Gate J",
            "model_weights_loaded": False,
            "data_records_loaded": False,
            "model_or_data_forward_executed": False,
            "formal_selector_executed": False,
            "outcomes_read": False,
            "loop_executed": False,
        }
    )
    return result


def _synthetic_record(card: Mapping[str, Any]) -> Dict[str, Any]:
    probabilities = [
        choice_logits_to_probabilities(
            [1.0 + 0.01 * index, 0.2, -0.1, -0.4]
        )
        for index in range(BOUNDARY_COUNT)
    ]
    hidden = [[1.0 + 0.01 * index, 0.2 + 0.005 * index, 0.5] for index in range(BOUNDARY_COUNT)]
    provenance = {
        "contract_version": card["renderer"]["contract_version"],
        "semantic_contract_sha256": card["renderer"]["semantic_contract_sha256"],
        "task_source_manifest_sha256": card["task"]["lm_eval_task_source"]["default_source_manifest_sha256"],
        "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
    }
    return build_trajectory_record(
        {"task": "mmlu", "doc_id": "synthetic-0", "doc_hash": "a" * 64},
        "synthetic_subject",
        "b" * 64,
        provenance,
        probabilities,
        hidden,
        card,
    )


def run_self_test(card_path: Path, schema_path: Path) -> Dict[str, Any]:
    card = load_json(card_path)
    schema = load_json(schema_path)
    validate_card(card)
    validate_schema_document(schema)

    synthetic = _synthetic_record(card)
    validate_trajectory_record(synthetic, card)
    projected = selector_sample(synthetic)
    validate_selector_sample(projected)
    if set(projected) != {"identity", "category", "H", "D"}:
        raise MMLU0ContractError("selector projection retained a diagnostic field")

    invalid_cases = []

    invalid = copy.deepcopy(card)
    invalid["evaluator"]["num_fewshot"] = 1
    invalid_cases.append(_expect_failure("fewshot_nonzero", lambda: validate_card(invalid)))

    invalid = copy.deepcopy(card)
    invalid["evaluator"]["apply_chat_template"] = True
    invalid_cases.append(_expect_failure("chat_template_enabled", lambda: validate_card(invalid)))

    invalid = copy.deepcopy(card)
    invalid["evaluator"]["generation"] = True
    invalid_cases.append(_expect_failure("generation_enabled", lambda: validate_card(invalid)))

    invalid = copy.deepcopy(card)
    invalid["model"]["revision"] = "0" * 40
    invalid_cases.append(_expect_failure("wrong_model_revision", lambda: validate_card(invalid)))

    invalid = copy.deepcopy(card)
    invalid["model"]["config_sha256"] = "0" * 64
    invalid_cases.append(_expect_failure("wrong_model_config_sha256", lambda: validate_card(invalid)))

    invalid = copy.deepcopy(card)
    invalid["choice_surfaces"]["A"]["token_ids"] = [362, 999]
    invalid_cases.append(_expect_failure("multi_token_surface", lambda: validate_card(invalid)))

    invalid = copy.deepcopy(card)
    invalid["choice_surfaces"]["B"]["surface"] = " A"
    invalid["choice_surfaces"]["B"]["decoded"] = " A"
    invalid_cases.append(_expect_failure("duplicate_surface", lambda: validate_card(invalid)))

    invalid = copy.deepcopy(card)
    invalid["trajectory"]["probe_position"] = "last_generated_token"
    invalid_cases.append(_expect_failure("wrong_probe_position", lambda: validate_card(invalid)))

    invalid = copy.deepcopy(card)
    invalid["selector"]["candidate_count"] = 41
    invalid_cases.append(_expect_failure("wrong_candidate_count", lambda: validate_card(invalid)))

    invalid_selector = dict(projected)
    invalid_selector["hidden_cosine_to_final"] = [0.0] * BOUNDARY_COUNT
    invalid_cases.append(_expect_failure("hidden_diagnostic_in_selector", lambda: validate_selector_sample(invalid_selector)))

    invalid_record = copy.deepcopy(synthetic)
    invalid_record["gold"] = 0
    invalid_cases.append(_expect_failure("gold_field", lambda: validate_trajectory_record(invalid_record, card)))

    invalid_record = copy.deepcopy(synthetic)
    invalid_record["outcome"] = {"accuracy": 1.0}
    invalid_cases.append(_expect_failure("outcome_field", lambda: validate_trajectory_record(invalid_record, card)))

    if len(invalid_cases) != 12:
        raise MMLU0ContractError("self-test case count differs")
    return {
        "status": "SELF_TEST_PASS",
        "card_sha256": file_sha256(card_path),
        "trajectory_schema_sha256": file_sha256(schema_path),
        "synthetic_schema_version": synthetic["schema_version"],
        "synthetic_boundary_count": len(synthetic["boundaries"]),
        "synthetic_angular_count": len(synthetic["adjacent_angular_distance"]),
        "synthetic_final_kl": synthetic["boundaries"][-1]["kl_to_final"],
        "synthetic_final_hidden_rms_l2": synthetic["boundaries"][-1]["hidden_rms_l2_to_final"],
        "invalid_cases_passed": invalid_cases,
        "model_or_data_forward_executed": False,
        "formal_selector_executed": False,
        "outcomes_read": False,
        "selector_projection_fields": ["identity", "category", "H", "D"],
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "semantic_contract_digest_recomputed": semantic_sha256(card["renderer"]["semantic_contract"]),
    }


__all__ = ["dry_run_payload", "run_self_test", "verify_gate_h_contracts"]
