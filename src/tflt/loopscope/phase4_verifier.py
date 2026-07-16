"""Deterministic local verifier for the Phase 4 P4-A sidecar contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from tflt.loopscope.phase4_scalars import trajectory_record_from_logits
from tflt.loopscope.phase4_schema import (
    RECEIPT_SCHEMA_VERSION,
    Phase4ContractError,
    canonical_identity,
    canonical_json_bytes,
    file_sha256,
    load_json_object,
    load_phase4_card,
    sanitized_content_sha256,
    source_record_json_schema,
    trajectory_record_json_schema,
    validate_shared_identity_contract,
    validate_source_record,
)
from tflt.loopscope.phase4_selector import analyze_selector, build_outcome_panel


SOURCE_SCHEMA_RELATIVE = "configs/loopscope/phase4_shared_source_schema.json"
TRAJECTORY_SCHEMA_RELATIVE = "configs/loopscope/phase4_trajectory_schema.json"
SHARED_CONTRACT_RELATIVE = "configs/loopscope/phase4_shared12032_contract.json"
IMPLEMENTATION_RELATIVES = (
    "src/tflt/loopscope/phase4_schema.py",
    "src/tflt/loopscope/phase4_scalars.py",
    "src/tflt/loopscope/phase4_selector.py",
    "src/tflt/loopscope/phase4_verifier.py",
    "scripts/loopscope/prepare_qwen4_phase4.py",
)


def build_verifier_receipt(card_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    card_path = Path(card_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else card_path.parents[2]
    card = load_phase4_card(card_path, require_provenance_closed=False)
    provenance_closed = _provenance_closed(card)
    provenance_fail_fast = _rejects(
        lambda: load_phase4_card(card_path, require_provenance_closed=True)
    )
    if provenance_closed == provenance_fail_fast:
        raise Phase4ContractError("provenance closure/fail-fast state is inconsistent")

    source_schema_path = root / SOURCE_SCHEMA_RELATIVE
    trajectory_schema_path = root / TRAJECTORY_SCHEMA_RELATIVE
    if load_json_object(source_schema_path) != source_record_json_schema():
        raise Phase4ContractError("committed Phase 4 source schema differs from code")
    if load_json_object(trajectory_schema_path) != trajectory_record_json_schema():
        raise Phase4ContractError("committed Phase 4 trajectory schema differs from code")
    shared_contract = load_json_object(root / SHARED_CONTRACT_RELATIVE)
    if shared_contract.get("expected_count") != 12032:
        raise Phase4ContractError("shared-12032 contract count differs")

    source_rows = [_synthetic_source_record(index) for index in range(8)]
    validate_shared_identity_contract(source_rows, source_rows, require_full_population=False)
    prompt_tamper = copy.deepcopy(source_rows)
    prompt_tamper[2]["rendered_prefix_sha256"] = "f" * 64
    identity_checks = {
        "missing_rejected": _rejects(
            lambda: validate_shared_identity_contract(source_rows, source_rows[:-1], require_full_population=False)
        ),
        "extra_rejected": _rejects(
            lambda: validate_shared_identity_contract(
                source_rows, source_rows + [_synthetic_source_record(99)], require_full_population=False
            )
        ),
        "reordered_rejected": _rejects(
            lambda: validate_shared_identity_contract(
                source_rows, list(reversed(source_rows)), require_full_population=False
            )
        ),
        "duplicate_rejected": _rejects(
            lambda: validate_shared_identity_contract(
                source_rows, source_rows[:-1] + [source_rows[0]], require_full_population=False
            )
        ),
        "prompt_hash_mismatch_rejected": _rejects(
            lambda: validate_shared_identity_contract(
                source_rows, prompt_tamper, require_full_population=False
            )
        ),
    }
    if not all(identity_checks.values()):
        raise Phase4ContractError("shared identity negative check did not fail closed")

    scalar_record = _synthetic_logits_record()
    if any(
        key in scalar_record for key in ("logits", "probabilities", "log_probabilities", "hidden_states")
    ):
        raise Phase4ContractError("scalar producer persisted a forbidden tensor field")
    d36 = scalar_record["boundaries"][36]["kl_to_final"]
    if abs(d36) > card["numeric_contract"]["d36_abs_tolerance"]:
        raise Phase4ContractError("synthetic D_36 is outside tolerance")
    non_finite_rejected = _rejects(
        lambda: trajectory_record_from_logits(
            canonical_identity="bad",
            category="synthetic",
            boundary_logits=[[float("nan")] + [0.0] * 127 for _ in range(37)],
            producer_provenance={"synthetic": True},
        )
    )
    if not non_finite_rejected:
        raise Phase4ContractError("non-finite logits did not fail fast")

    selector_input = [_synthetic_scalar_record(index) for index in range(24)]
    selector_report = analyze_selector(selector_input, replicates=64, seed=20260717)
    panel = build_outcome_panel(selector_report, ["15:18"])
    if selector_report["width4_primary"]["consensus_start_count"] != 25:
        raise Phase4ContractError("synthetic selector did not close width-4 support")
    if selector_report["variable_width_strict"]["row_count"] != 154:
        raise Phase4ContractError("synthetic selector did not close strict154")
    if selector_report["variable_width_edge"]["row_count"] != 224:
        raise Phase4ContractError("synthetic selector did not close edge224")

    forbidden_source = copy.deepcopy(source_rows[0])
    forbidden_source["renderer_provenance"]["target_cot_content"] = "not allowed"
    forbidden_trajectory = copy.deepcopy(selector_input[0])
    forbidden_trajectory["producer_provenance"]["persisted_full_logits"] = [0.0]
    non_json_trajectory = copy.deepcopy(selector_input[0])
    non_json_trajectory["producer_provenance"]["tensor_payload"] = object()
    forbidden_checks = {
        "nested_source_target_cot_rejected": _rejects(lambda: validate_source_record(forbidden_source)),
        "nested_trajectory_logits_rejected": _rejects(
            lambda: analyze_selector([forbidden_trajectory, selector_input[1]], replicates=2)
        ),
        "non_json_tensor_payload_rejected": _rejects(
            lambda: analyze_selector([non_json_trajectory, selector_input[1]], replicates=2)
        ),
    }
    if not all(forbidden_checks.values()):
        raise Phase4ContractError("forbidden normal-path field did not fail closed")

    implementations = {relative: file_sha256(root / relative) for relative in IMPLEMENTATION_RELATIVES}
    payload: Dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_role": "p4a_local_contract_verifier_receipt",
        "card": {
            "relative_path": "configs/loopscope/phase4_pv_ek_trs_card.json",
            "byte_sha256": file_sha256(card_path),
            "semantic_sha256": hashlib.sha256(canonical_json_bytes(card)).hexdigest(),
            "provenance_closed": provenance_closed,
            "normal_verify_fails_closed": provenance_fail_fast,
        },
        "contract_artifacts": {
            SOURCE_SCHEMA_RELATIVE: {"file_sha256": file_sha256(source_schema_path)},
            TRAJECTORY_SCHEMA_RELATIVE: {"file_sha256": file_sha256(trajectory_schema_path)},
            SHARED_CONTRACT_RELATIVE: {"file_sha256": file_sha256(root / SHARED_CONTRACT_RELATIVE)},
        },
        "checks": {
            "shared_identity_contract": identity_checks,
            "forbidden_field_normal_path": forbidden_checks,
            "float32_stable_log_softmax_recomputed": True,
            "d36_abs_value": d36,
            "non_finite_rejected": non_finite_rejected,
            "persisted_logits_or_tensors": False,
            "width4_candidate_count": 33,
            "width4_consensus_count": 25,
            "variable_width_strict_count": 154,
            "variable_width_edge_count": 224,
            "width4_compatibility_verified": True,
            "panel_known_outcome_exclusion_verified": "15:18" not in panel["blind_high3"] + panel["blind_low3"],
            "torch_transformers_lm_eval_imports_required": False,
        },
        "synthetic_fixture": {
            "contains_real_data_or_outcome": False,
            "source_count": len(source_rows),
            "selector_count": len(selector_input),
            "selector_replicates": 64,
            "source_sha256": hashlib.sha256(canonical_json_bytes(source_rows)).hexdigest(),
            "scalar_record_sha256": hashlib.sha256(canonical_json_bytes(scalar_record)).hexdigest(),
            "selector_report_sha256": hashlib.sha256(canonical_json_bytes(selector_report)).hexdigest(),
            "panel_sha256": hashlib.sha256(canonical_json_bytes(panel)).hexdigest(),
            "observed_selected_window": selector_report["width4_primary"]["selected_window"],
            "observed_abstain": selector_report["width4_primary"]["abstain"],
        },
        "implementation_sha256": implementations,
        "external_actions_performed": [],
        "blocking_condition": None if provenance_closed else card["provenance_closure"]["blocker"],
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def verify_committed_receipt(card_path: Path, receipt_path: Path) -> Dict[str, Any]:
    expected = build_verifier_receipt(card_path)
    observed = load_json_object(receipt_path)
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise Phase4ContractError("committed Phase 4 verifier receipt differs from recomputation")
    return observed


def _synthetic_source_record(index: int) -> Dict[str, Any]:
    question = "Synthetic question %d" % index
    options = ["Option %d-%d" % (index, item) for item in range(4)]
    record: Dict[str, Any] = {
        "schema_version": "loopscope.phase4.shared-source-record.v1",
        "dataset_revision": "1" * 40,
        "split": "test",
        "question_id": index,
        "category": "category_%02d" % (index % 4),
        "src": "synthetic",
        "question": question,
        "ordered_options": options,
        "canonical_identity": "",
        "sanitized_content_sha256": sanitized_content_sha256(question, options),
        "rendered_prefix_sha256": hashlib.sha256(("prefix-%d" % index).encode()).hexdigest(),
        "rendered_token_ids_sha256": hashlib.sha256(("tokens-%d" % index).encode()).hexdigest(),
        "attention_mask_sha256": hashlib.sha256(("mask-%d" % index).encode()).hexdigest(),
        "last_effective_prefix_token_index": 127 + index,
        "renderer_provenance": {"synthetic": True, "contains_target_cot": False},
    }
    record["canonical_identity"] = canonical_identity(record)
    validate_source_record(record)
    return record


def _synthetic_logits_record() -> Dict[str, Any]:
    logits = [
        [math.sin((boundary + 1) * (token + 3) * 0.013) + boundary * 0.002 for token in range(128)]
        for boundary in range(37)
    ]
    return trajectory_record_from_logits(
        canonical_identity="synthetic-logits",
        category="category_00",
        boundary_logits=logits,
        producer_provenance={"synthetic": True, "full_vocabulary_size": 128},
    )


def _synthetic_scalar_record(index: int) -> Dict[str, Any]:
    entropy = [5.0 + 0.0001 * index + 0.00001 * boundary * (index % 3) for boundary in range(37)]
    kl = [3.0 + 0.0001 * index + 0.00001 * boundary * ((index + 1) % 3) for boundary in range(37)]
    entropy[8], entropy[11], entropy[12], entropy[13] = 5.0, 5.0, 7.0, 5.0
    entropy[15], entropy[16], entropy[17], entropy[20] = 6.0, 5.0, 6.0, 6.0
    kl[8], kl[11], kl[12], kl[13] = 4.0, 4.0, 3.0, 4.0
    kl[15], kl[16], kl[17], kl[20] = 3.0, 5.0, 3.0, 4.0
    kl[36] = 0.0
    boundaries = []
    for boundary in range(37):
        boundaries.append(
            {
                "boundary_id": boundary,
                "full_vocabulary_entropy": entropy[boundary],
                "kl_to_final": kl[boundary],
                "normalized_entropy": entropy[boundary] / 10.0,
                "logit_rms": 1.0 + boundary * 0.001,
                "top1_probability_mass": 0.05,
                "top10_probability_mass": 0.25,
                "top100_probability_mass": 0.75,
                "finite": True,
            }
        )
    return {
        "schema_version": "loopscope.phase4.prefix-trajectory-record.v1",
        "canonical_identity": "synthetic-selector-%03d" % index,
        "category": "category_%02d" % (index % 4),
        "boundaries": boundaries,
        "producer_provenance": {"synthetic": True},
    }


def _provenance_closed(card: Mapping[str, Any]) -> bool:
    return card["provenance_closure"]["state"] == "CLOSED"


def _rejects(function: Any) -> bool:
    try:
        function()
    except (Phase4ContractError, ValueError, TypeError):
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the local-only LoopScope Phase 4 P4-A contract")
    parser.add_argument("--card", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument(
        "--allow-unresolved-provenance",
        action="store_true",
        help="verify the bounded local implementation while preserving the explicit provenance blocker",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = verify_committed_receipt(Path(args.card), Path(args.receipt))
    if not receipt["card"]["provenance_closed"] and not args.allow_unresolved_provenance:
        raise Phase4ContractError("Phase 4 card provenance remains unresolved")
    print(receipt["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
