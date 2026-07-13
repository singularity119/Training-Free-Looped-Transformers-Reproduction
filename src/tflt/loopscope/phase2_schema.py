"""Closed-world contracts for LoopScope Phase 2 H1 V2 evidence.

The schemas in this module deliberately validate exact key sets.  A valid
self-hash proves byte-canonical content, but it never grants permission to
change a frozen field.  Every validator therefore checks the self-hash *and*
the complete frozen contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from tflt.loopscope.schema import (
    SchemaError,
    attach_manifest_sha256,
    canonical_json_bytes,
    manifest_sha256,
    verify_manifest_sha256,
)


PHASE2_CARD_ID = "H1_ITERATIVE_REFINEMENT_ZONE_V2"
PHASE2_CARD_SCHEMA_VERSION = "loopscope.phase2-h1-card.v4"
PHASE2_IDENTITY_MANIFEST_SCHEMA_VERSION = "loopscope.phase2-identity-manifest.v2"
PHASE2_TRACE_SCHEMA_VERSION = "loopscope.phase2-calibration-trajectory-sample.v2"
PHASE2_CALIBRATION_CELL_SCHEMA_VERSION = "loopscope.phase2-calibration-cell.v2"
PHASE2_CALIBRATION_BASELINE_SCHEMA_VERSION = "loopscope.phase2-calibration-baseline.v2"
PHASE2_FINAL_OUTPUT_SCHEMA_VERSION = "loopscope.phase2-final-output-sample.v2"
PHASE2_FULL_FINAL_OUTPUT_SCHEMA_VERSION = "loopscope.phase2-full-final-output-cell.v3"
PHASE2_CALIBRATION_LABEL_SCHEMA_VERSION = "loopscope.phase2-calibration-labels.v1"
PHASE2_UNSEAL_AUTHORIZATION_SCHEMA_VERSION = "loopscope.phase2-unseal-authorization.v1"
PHASE2_UNSEAL_RECEIPT_SCHEMA_VERSION = "loopscope.phase2-unseal-receipt.v1"
PHASE2_ANALYSIS_INPUT_SCHEMA_VERSION = "loopscope.phase2-h1-analysis-input.v4"
PHASE2_ANALYSIS_SCHEMA_VERSION = "loopscope.phase2-h1-analysis-report.v4"
PHASE2_REUSE_SCHEMA_VERSION = "loopscope.phase2-reuse-matrix.v1"

CALIBRATION_IDENTITY_NAMESPACE = "loopscope.phase2.calibration.validation512.v1"
FULL_IDENTITY_NAMESPACE = "loopscope.phase2.full.mmlu14042.v1"
DIRECT_PROBE_SCORE_SOURCE = (
    "direct_probe_next_token_log_probability_over_frozen_choice_token_ids"
)
FULL_LM_EVAL_SCORE_SOURCE = "lm_eval_acc_none_raw_per_choice_loglikelihood"
MMLU_DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
PHASE1_INPUT_ROOT = (
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/inputs/"
    "loopscope-qwen17-mmlu-phase1-20260711-043615"
)
PHASE1_RUN_ROOT = (
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/"
    "loopscope-qwen17-mmlu-phase1-20260711-053022"
)
PHASE2_WORKSPACE_ROOT = (
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope"
)

CHOICE_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
PRIMARY_CONTRAST_IDS: Tuple[str, ...] = (
    "fs_11_14_k2_k3",
    "fs_11_14_k3_k4",
    "fs_12_15_k2_k3",
    "fs_12_15_k3_k4",
    "fs_13_16_k2_k3",
    "fs_13_16_k3_k4",
)
FIXED_HORIZON_CONTRAST_IDS: Tuple[str, ...] = (
    "fh_12_15_k2_k3",
    "fh_12_15_k3_k4",
)
REUSE_STATUSES = frozenset(
    ("local_schema_confirmed", "historical_only", "requires_gate_b_live_check")
)
IMPLEMENTATION_FILES: Tuple[str, ...] = (
    "src/tflt/loopscope/__init__.py",
    "src/tflt/loopscope/phase2_schema.py",
    "src/tflt/loopscope/phase2_trajectory.py",
    "src/tflt/loopscope/phase2_analysis.py",
    "src/tflt/loopscope/phase2_reuse.py",
    "src/tflt/eval_runner.py",
    "src/tflt/cli.py",
    "scripts/loopscope/prepare_qwen17_phase2.py",
)


def frozen_decision_rules() -> Dict[str, Any]:
    """Return the complete versioned H1/NCA decision contract.

    A fresh mapping is returned so callers cannot mutate module-level state.
    The validator compares this object exactly; classifiers then read the
    registered targets and thresholds from the validated mapping.
    """

    return {
        "multiplicity": {
            "primary_family": list(PRIMARY_CONTRAST_IDS),
            "fixed_horizon_family": list(FIXED_HORIZON_CONTRAST_IDS),
            "primary_method": "two_sided_exact_mcnemar_then_holm_step_down",
            "fixed_horizon_method": "two_sided_exact_mcnemar_then_holm_step_down",
            "familywise_alpha": 0.05,
            "k1_to_k2_role": "equivalence_and_cumulative_context_not_in_holm_family",
            "nca_interval_role": "pointwise_unadjusted_secondary_diagnostic",
        },
        "h1": {
            "priority": [
                "PERTURBATION",
                "REFINEMENT_SUPPORTED",
                "TRANSIENT_ONLY",
                "SUGGESTIVE",
                "INCONCLUSIVE",
            ],
            "target_window": "12:15",
            "significance": {
                "field": "holm_adjusted_p",
                "operator": "strictly_less_than",
                "threshold": 0.05,
            },
            "perturbation": {
                "target_contrast_ids": ["fs_12_15_k2_k3", "fs_12_15_k3_k4"],
                "delta_operator": "strictly_less_than",
                "delta_threshold_pp": 0.0,
                "transition_comparison": "right_to_wrong_strictly_greater_than_wrong_to_right",
            },
            "refinement_supported": {
                "target_contrast_id": "fs_12_15_k2_k3",
                "delta_operator": "strictly_greater_than",
                "delta_threshold_pp": 0.0,
                "d24_operator": "greater_than_or_equal",
                "d24_threshold_pp": 0.0,
            },
            "transient_only": {
                "phase1_k2_vs_baseline_operator": "strictly_greater_than",
                "phase1_k2_vs_baseline_threshold_pp": 0.0,
                "d24_interval_endpoint": "upper",
                "d24_endpoint_operator": "strictly_less_than",
                "d24_endpoint_threshold_pp": 0.0,
            },
            "suggestive": {
                "target_contrast_id": "fs_12_15_k2_k3",
                "delta_operator": "strictly_greater_than",
                "delta_threshold_pp": 0.0,
                "d24_operator": "greater_than_or_equal",
                "d24_threshold_pp": 0.0,
                "significance_operator": "greater_than_or_equal",
            },
            "inconclusive": "otherwise_with_complete_evidence",
            "saturation": {
                "target_contrast_id": "fs_12_15_k3_k4",
                "delta_operator": "strictly_greater_than",
                "delta_threshold_pp": 0.0,
                "significance_operator": "strictly_less_than",
                "positive_label": "still_improving_at_k4",
                "otherwise_label": "saturation_not_established",
                "equivalence_claim_allowed": False,
                "k_greater_than_4_authorized": False,
            },
            "diagnostic_only": [
                "q",
                "js",
                "entropy",
                "margin",
                "wrong_overconfidence",
                "fixed_horizon_family",
                "cumulative_contrasts",
            ],
            "neighbor_window_claim": "controls_and_falsifiers_not_between_window_significance",
        },
        "nca": {
            "priority": [
                "NCA_INCONCLUSIVE",
                "NCA_NATIVE_FIDELITY_ONLY",
                "NCA_DIRECTION_SUPPORTED",
                "NCA_NONDISCRIMINATIVE",
            ],
            "primary_windows": ["11:14", "12:15", "13:16"],
            "primary_k": [2, 3, 4],
            "primary_cell_count": 9,
            "paired_keys": ["k2", "k3", "k4"],
            "paired_cell_count": 3,
            "sample_cell_validity": "all_declared_repeated_steps_valid",
            "denominator": 512,
            "valid_fraction_operator": "greater_than_or_equal",
            "valid_fraction_threshold": 0.95,
            "missing_or_nonfinite_primary_interval_label": "NCA_INCONCLUSIVE",
            "target_window": "12:15",
            "harmful_control_window": "13:16",
            "target_interval_endpoint": "lower",
            "target_positive_operator": "strictly_greater_than",
            "target_positive_threshold": 0.0,
            "target_positive_required_count": 3,
            "paired_difference": "12:15_minus_13:16",
            "paired_interval_endpoint": "lower",
            "paired_positive_operator": "strictly_greater_than",
            "paired_positive_threshold": 0.0,
            "direction_positive_required_count": 2,
            "native_fidelity": {
                "required_subgroups": ["right_to_right", "wrong_to_right", "wrong_to_wrong"],
                "min_valid_per_subgroup": 10,
                "eligible_cell_required_count": 2,
                "right_to_right_interval_endpoint": "lower",
                "right_to_right_operator": "strictly_greater_than",
                "right_to_right_threshold": 0.0,
                "wrong_to_right_minus_wrong_to_wrong_interval_endpoint": "upper",
                "wrong_to_right_minus_wrong_to_wrong_operator": "less_than_or_equal",
                "wrong_to_right_minus_wrong_to_wrong_threshold": 0.0,
                "not_evaluable_does_not_force_inconclusive": True,
            },
            "native_supported_required_count": 2,
            "interval_role": "pointwise_unadjusted_secondary_diagnostic_no_fwer_claim",
        },
        "wrong_overconfidence": {
            "condition": "entropy_decreases_and_raw_top_margin_increases",
            "denominator": "final_wrong_pairs=wrong_to_wrong+right_to_wrong",
            "subgroups_reported": ["wrong_to_wrong", "right_to_wrong"],
            "role": "mechanism_diagnostic_not_H1_decision_rule",
        },
    }

_CARD_KEYS = frozenset(
    (
        "schema_version",
        "card_id",
        "status",
        "science",
        "evidence_scales",
        "trajectories",
        "measurement",
        "statistics",
        "decision_rules",
        "tolerances",
        "primary_contrast_ids",
        "fixed_horizon_contrast_ids",
        "phase1_reuse_policy",
        "write_once_contract",
        "implementation_sha256",
        "manifest_sha256",
    )
)
_IDENTITY_KEYS = frozenset(("task", "doc_id", "doc_hash"))
_CELL_KEYS = frozenset(("cell_id", "protocol", "window", "k", "alpha", "step_size"))
_PRODUCER_KEYS = frozenset(
    (
        "producer_kind",
        "attempt_manifest_sha256",
        "receipt_manifest_sha256",
        "command_sha256",
        "environment_sha256",
        "revision_report_sha256",
    )
)
_FULL_ADAPTER_PRODUCER_KEYS = _PRODUCER_KEYS | frozenset(("results_sha256",))
_FULL_REUSE_PRODUCER_KEYS = _FULL_ADAPTER_PRODUCER_KEYS | frozenset(
    ("source_manifest_sha256",)
)
_FINAL_BASE_KEYS = frozenset(
    (
        "schema_version",
        "sample_identity",
        "choice_labels",
        "choice_score_source",
        "raw_choice_scores",
        "choice_probabilities",
        "probability_dtype",
        "entropy_nats",
        "top1_index",
        "top1_label",
        "top_margin_raw",
        "top1_tie_break",
    )
)
_FINAL_GOLD_KEYS = frozenset(
    ("gold_index", "correct_margin_raw", "correctness", "evaluator_acc_none")
)
_TRAJECTORY_SAMPLE_KEYS = frozenset(
    (
        "sample_identity",
        "answer_position",
        "position_rule",
        "boundary_identity",
        "body_call_indexing",
        "steps",
        "repeated_step_nca_all_valid",
        "valid",
        "errors",
        "event_counts",
        "final_output",
    )
)
_STEP_KEYS = frozenset(
    (
        "body_call_t",
        "residual_norm",
        "state_norm",
        "relative_activity",
        "residual_ratio_to_previous",
        "adjacent_residual_cosine",
        "adjacent_residual_cosine_valid",
        "nca",
        "valid",
    )
)
_NCA_KEYS = frozenset(
    (
        "value",
        "residual_norm",
        "native_continuation_norm",
        "valid",
        "invalid_reason",
        "dtype",
        "provenance",
        "body_call_t",
    )
)
_ENVELOPE_KEYS = frozenset(
    (
        "schema_version",
        "artifact_kind",
        "evidence_scale",
        "identity_namespace",
        "card_manifest_sha256",
        "identity_manifest_sha256",
        "ordered_identity_sha256",
        "revision",
        "cell",
        "label_state",
        "score_source",
        "sample_count",
        "samples",
        "producer",
        "manifest_sha256",
    )
)
_FULL_ENVELOPE_KEYS = _ENVELOPE_KEYS | frozenset(("artifact_root", "producer_evidence"))
_FILE_REF_KEYS = frozenset(("path", "sha256"))
_NEW_FULL_EVIDENCE_KEYS = frozenset(
    (
        "adapter_request",
        "attempt_manifest",
        "receipt_manifest",
        "command_args",
        "environment",
        "revision_report",
        "results",
        "sample_source",
    )
)
_REUSE_FULL_EVIDENCE_KEYS = frozenset(
    (
        "reuse_request",
        "attempt_manifest",
        "receipt_manifest",
        "command_args",
        "environment",
        "source_run_manifest",
        "source_command_args",
        "source_environment",
        "source_revision_report",
        "source_results",
        "sample_source",
    )
)
_BASELINE_SAMPLE_KEYS = frozenset(
    ("sample_identity", "answer_position", "final_output", "window_boundaries")
)
_BASELINE_BOUNDARY_KEYS = frozenset(
    (
        "window",
        "inclusive_boundary",
        "native_boundary_formula",
        "baseline_update_norm",
        "native_continuation_norm",
        "baseline_nca",
    )
)


def protocol_cell_id(protocol: str, window: str, k: int, alpha: float) -> str:
    """Return a protocol/window/K/alpha identity with no alias collision."""

    normalized = str(window).replace(":", "_")
    return "%s_%s_k%d_a%s" % (
        str(protocol),
        normalized,
        int(k),
        _format_float(alpha),
    )


def phase1_reuse_full_cell_ids() -> Tuple[str, ...]:
    """Return the only full cells whose immutable Phase 1 outputs may be reused."""

    return (
        protocol_cell_id("baseline_no_loop", "none", 1, 1.0),
        protocol_cell_id("shared_k2_anchor", "11:14", 2, 1.0),
        protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0),
        protocol_cell_id("shared_k2_anchor", "13:16", 2, 1.0),
    )


def phase2_new_full_cell_ids() -> Tuple[str, ...]:
    """Return the eight cells that require a new Phase 2 lm-eval execution."""

    result: List[str] = []
    for window in ("11:14", "12:15", "13:16"):
        result.extend(
            (
                protocol_cell_id("fixed_step", window, 3, 1.5),
                protocol_cell_id("fixed_step", window, 4, 2.0),
            )
        )
    result.extend(
        (
            protocol_cell_id("fixed_horizon", "12:15", 3, 1.0),
            protocol_cell_id("fixed_horizon", "12:15", 4, 1.0),
        )
    )
    if len(result) != 8 or len(set(result)) != 8:
        raise SchemaError("Phase 2 new-run full cell partition is not eight unique cells")
    return tuple(result)


def expected_full_producer_kind(cell: Mapping[str, Any]) -> str:
    """Bind every frozen full cell to exactly one producer namespace."""

    _validate_cell(cell, allow_baseline=True)
    cell_id = cell["cell_id"]
    reuse = phase1_reuse_full_cell_ids()
    new = phase2_new_full_cell_ids()
    if set(reuse) & set(new) or len(reuse) + len(new) != 12:
        raise SchemaError("full producer partitions overlap or are incomplete")
    if cell_id in reuse:
        return "phase1_immutable_reuse_adapter"
    if cell_id in new:
        return "lm_eval_logged_samples_adapter"
    raise SchemaError("full cell is outside the frozen producer partition")


def stable_sample_identity(task: str, doc_id: Any, doc_hash: str) -> Dict[str, str]:
    task_text = str(task).strip()
    doc_id_text = str(doc_id).strip()
    doc_hash_text = str(doc_hash).strip().lower()
    if not task_text or not doc_id_text or len(doc_hash_text) != 64:
        raise SchemaError("sample identity requires task, doc_id, and 64-char doc_hash")
    if any(character not in "0123456789abcdef" for character in doc_hash_text):
        raise SchemaError("doc_hash must be lowercase hexadecimal SHA256")
    return {"task": task_text, "doc_id": doc_id_text, "doc_hash": doc_hash_text}


def ordered_identity_sha256(identities: Sequence[Mapping[str, Any]]) -> str:
    normalized = [_validate_identity(item) for item in identities]
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def make_identity_manifest(
    card: Mapping[str, Any],
    *,
    identity_namespace: str,
    split: str,
    identities: Sequence[Mapping[str, Any]],
    source_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_phase2_card(card)
    normalized = [_validate_identity(item) for item in identities]
    payload: Dict[str, Any] = {
        "schema_version": PHASE2_IDENTITY_MANIFEST_SCHEMA_VERSION,
        "identity_namespace": str(identity_namespace),
        "evidence_scale": _scale_for_namespace(identity_namespace),
        "task_group": "mmlu",
        "split": str(split),
        "sample_count": len(normalized),
        "natural_order": "canonical_manifest_list_order_zero_based",
        "source_provenance": dict(source_provenance),
        "ordered_sample_identity": normalized,
        "ordered_identity_sha256": ordered_identity_sha256(normalized),
        "card_manifest_sha256": card["manifest_sha256"],
        "revision": card["science"]["revision"],
    }
    attach_manifest_sha256(payload)
    validate_identity_manifest(payload, card)
    return payload


def validate_identity_manifest(manifest: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    validate_phase2_card(card)
    _exact_keys(manifest, {
        "schema_version", "identity_namespace", "evidence_scale", "task_group", "split",
        "sample_count", "natural_order", "ordered_sample_identity", "ordered_identity_sha256",
        "source_provenance", "card_manifest_sha256", "revision", "manifest_sha256",
    }, "identity manifest")
    verify_manifest_sha256(manifest)
    if manifest["schema_version"] != PHASE2_IDENTITY_MANIFEST_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 2 identity manifest schema")
    namespace = _nonempty_string(manifest["identity_namespace"], "identity_namespace")
    scale = _scale_for_namespace(namespace)
    expected_scale = card["evidence_scales"][scale]
    if manifest["evidence_scale"] != scale:
        raise SchemaError("identity namespace and evidence scale disagree")
    if manifest["task_group"] != "mmlu" or manifest["split"] != expected_scale["split"]:
        raise SchemaError("identity manifest task/split differs from the card")
    if manifest["natural_order"] != "canonical_manifest_list_order_zero_based":
        raise SchemaError("identity manifest natural-order rule differs")
    validate_source_provenance(manifest["source_provenance"], namespace)
    if manifest["card_manifest_sha256"] != card["manifest_sha256"]:
        raise SchemaError("identity manifest is bound to a different card")
    if manifest["revision"] != card["science"]["revision"]:
        raise SchemaError("identity manifest revision differs from the card")
    identities = manifest["ordered_sample_identity"]
    if not isinstance(identities, list):
        raise SchemaError("ordered_sample_identity must be a list")
    count = _strict_int(manifest["sample_count"], "sample_count", minimum=1)
    if count != expected_scale["sample_count"] or len(identities) != count:
        raise SchemaError("identity manifest count differs from the frozen scale")
    normalized = [_validate_identity(item) for item in identities]
    tuples = [(item["task"], item["doc_id"], item["doc_hash"]) for item in normalized]
    if len(set(tuples)) != len(tuples):
        raise SchemaError("canonical sample identities must be unique")
    expected_hash = ordered_identity_sha256(normalized)
    _sha256(manifest["ordered_identity_sha256"], "ordered_identity_sha256")
    if manifest["ordered_identity_sha256"] != expected_hash:
        raise SchemaError("ordered identity hash mismatch")


def make_source_provenance(
    *,
    identity_namespace: str,
    verification_status: str,
    source_manifest_path: Optional[str],
    source_manifest_sha256: Optional[str],
    renderer_subset_sha256: Optional[str],
    identity_artifacts: Optional[Sequence[Mapping[str, str]]] = None,
    phase1_input_root: str = PHASE1_INPUT_ROOT,
    phase1_run_root: str = PHASE1_RUN_ROOT,
) -> Dict[str, Any]:
    """Build the canonical Phase 1 source binding for one identity namespace."""

    if identity_namespace == CALIBRATION_IDENTITY_NAMESPACE:
        source_kind = "phase1_frozen_validation_probe_pool"
        natural_order_producer = "phase1_frozen_probe_pool_manifest_records_order"
        identity_derivation = (
            "probe_pool_rendering_records_task_name_target_doc_id_target_doc_sha256"
        )
    elif identity_namespace == FULL_IDENTITY_NAMESPACE:
        source_kind = "phase1_full_mmlu_evaluator_identity"
        natural_order_producer = (
            "phase1_baseline_result_loader_sorted_task_then_logged_sample_order"
        )
        identity_derivation = (
            "results_samples_else_exact_sorted_sample_sidecars_task_doc_id_doc_hash"
        )
    else:
        raise SchemaError("unknown Phase 2 identity namespace")
    artifacts = [dict(value) for value in (identity_artifacts or [])]
    if (
        verification_status == "live_verified"
        and identity_namespace == CALIBRATION_IDENTITY_NAMESPACE
        and not artifacts
        and source_manifest_path is not None
        and source_manifest_sha256 is not None
    ):
        artifacts = [
            {"path": str(source_manifest_path), "sha256": str(source_manifest_sha256)}
        ]
    payload: Dict[str, Any] = {
        "schema_version": "loopscope.phase2-source-provenance.v2",
        "verification_status": verification_status,
        "source_kind": source_kind,
        "phase1_input_root": str(phase1_input_root),
        "phase1_run_root": str(phase1_run_root),
        "source_manifest_path": source_manifest_path,
        "source_manifest_sha256": source_manifest_sha256,
        "renderer_subset_sha256": renderer_subset_sha256,
        "dataset_source": "cais/mmlu",
        "dataset_revision": MMLU_DATASET_REVISION,
        "num_fewshot": 5,
        "natural_order_producer": natural_order_producer,
        "identity_derivation": identity_derivation,
        "identity_artifacts": artifacts,
    }
    attach_manifest_sha256(payload)
    validate_source_provenance(payload, identity_namespace)
    return payload


def validate_source_provenance(
    payload: Mapping[str, Any], identity_namespace: str, *, require_live: bool = False
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version", "verification_status", "source_kind",
            "phase1_input_root", "phase1_run_root", "source_manifest_path",
            "source_manifest_sha256", "renderer_subset_sha256", "dataset_source",
            "dataset_revision", "num_fewshot", "natural_order_producer",
            "identity_derivation", "identity_artifacts",
            "manifest_sha256",
        },
        "identity source provenance",
    )
    verify_manifest_sha256(payload)
    if payload["schema_version"] != "loopscope.phase2-source-provenance.v2":
        raise SchemaError("unsupported identity source-provenance schema")
    if identity_namespace == CALIBRATION_IDENTITY_NAMESPACE:
        expected_kind = "phase1_frozen_validation_probe_pool"
        expected_order = "phase1_frozen_probe_pool_manifest_records_order"
        expected_derivation = (
            "probe_pool_rendering_records_task_name_target_doc_id_target_doc_sha256"
        )
    elif identity_namespace == FULL_IDENTITY_NAMESPACE:
        expected_kind = "phase1_full_mmlu_evaluator_identity"
        expected_order = (
            "phase1_baseline_result_loader_sorted_task_then_logged_sample_order"
        )
        expected_derivation = (
            "results_samples_else_exact_sorted_sample_sidecars_task_doc_id_doc_hash"
        )
    else:
        raise SchemaError("unknown Phase 2 identity namespace")
    if (
        payload["source_kind"] != expected_kind
        or payload["natural_order_producer"] != expected_order
        or payload["identity_derivation"] != expected_derivation
    ):
        raise SchemaError("identity source kind/natural-order producer differs")
    if payload["phase1_input_root"] != PHASE1_INPUT_ROOT or payload["phase1_run_root"] != PHASE1_RUN_ROOT:
        raise SchemaError("identity source roots differ from immutable Phase 1 provenance")
    if (
        payload["dataset_source"] != "cais/mmlu"
        or payload["dataset_revision"] != MMLU_DATASET_REVISION
        or payload["num_fewshot"] != 5
    ):
        raise SchemaError("identity dataset/few-shot provenance differs")
    status = payload["verification_status"]
    if status not in {"live_verified", "requires_gate_b_live_check"}:
        raise SchemaError("identity source verification status differs")
    if require_live and status != "live_verified":
        raise SchemaError("analysis requires live-verified canonical source provenance")
    if status == "live_verified":
        path = _nonempty_string(payload["source_manifest_path"], "source_manifest_path")
        if not Path(path).is_absolute():
            raise SchemaError("source manifest path must be absolute")
        roots = (Path(PHASE1_INPUT_ROOT), Path(PHASE1_RUN_ROOT))
        if not any(Path(path).is_relative_to(root) for root in roots):
            raise SchemaError("source manifest path is outside immutable Phase 1 roots")
        expected_path = (
            Path(PHASE1_INPUT_ROOT) / "probe_pool_manifest.json"
            if identity_namespace == CALIBRATION_IDENTITY_NAMESPACE
            else Path(PHASE1_RUN_ROOT) / "control" / "phase1_run_manifest.json"
        )
        if Path(path) != expected_path:
            raise SchemaError("source manifest path differs from the canonical Phase 1 artifact")
        _sha256(payload["source_manifest_sha256"], "source_manifest_sha256")
        _sha256(payload["renderer_subset_sha256"], "renderer_subset_sha256")
        artifacts = payload["identity_artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise SchemaError("live source provenance requires identity artifacts")
        normalized_paths = []
        for ref in artifacts:
            _exact_keys(ref, {"path", "sha256"}, "identity source artifact")
            raw_artifact_path = _nonempty_string(
                ref["path"], "identity artifact path"
            )
            artifact_path = Path(raw_artifact_path)
            if not artifact_path.is_absolute():
                raise SchemaError("identity artifact path must be absolute")
            if ".." in artifact_path.parts or raw_artifact_path != str(artifact_path):
                raise SchemaError("identity artifact path must be canonical")
            _sha256(ref["sha256"], "identity artifact sha256")
            normalized_paths.append(artifact_path)
        if len(set(normalized_paths)) != len(normalized_paths):
            raise SchemaError("identity source artifacts must be unique")
        if identity_namespace == CALIBRATION_IDENTITY_NAMESPACE:
            if artifacts != [{"path": path, "sha256": payload["source_manifest_sha256"]}]:
                raise SchemaError("calibration identity must derive from the frozen pool manifest")
        else:
            if (
                not normalized_paths
                or not normalized_paths[0].is_relative_to(Path(PHASE1_RUN_ROOT))
                or normalized_paths[0].name != "results.json"
                or any(
                    path.parent != normalized_paths[0].parent
                    or not path.name.startswith("samples_")
                    or path.suffix != ".jsonl"
                    for path in normalized_paths[1:]
                )
                or normalized_paths[1:] != sorted(normalized_paths[1:])
            ):
                raise SchemaError(
                    "full identity artifacts must be baseline results.json followed by sorted sample sidecars"
                )
    elif any(
        payload[key] is not None
        for key in ("source_manifest_path", "source_manifest_sha256", "renderer_subset_sha256")
    ):
        raise SchemaError("unverified source digests must remain null pending Gate B")
    elif payload["identity_artifacts"] != []:
        raise SchemaError("unverified identity artifacts must remain empty pending Gate B")


def verify_live_source_provenance(
    payload: Mapping[str, Any],
    identity_namespace: str,
    expected_identities: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Mapping[str, Any]:
    """Read the declared immutable Phase 1 manifest and verify its live binding."""

    validate_source_provenance(payload, identity_namespace, require_live=True)
    path = Path(payload["source_manifest_path"])
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SchemaError("live Phase 1 source manifest is unreadable") from exc
    if not isinstance(source, Mapping):
        raise SchemaError("live Phase 1 source manifest must contain an object")
    verify_manifest_sha256(source)
    if source.get("manifest_sha256") != payload["source_manifest_sha256"]:
        raise SchemaError("live Phase 1 source manifest hash differs from identity provenance")
    if identity_namespace == CALIBRATION_IDENTITY_NAMESPACE:
        if (
            source.get("source") != "cais/mmlu@%s" % MMLU_DATASET_REVISION
            or source.get("split") != "validation"
            or source.get("count") != 512
            or source.get("seed") != 20260710
            or source.get("task_group") != "mmlu"
            or source.get("num_fewshot") != 5
            or source.get("uses_target_gold_labels") is not False
            or not isinstance(source.get("renderer"), Mapping)
            or source["renderer"].get("dataset_revision") != MMLU_DATASET_REVISION
            or source.get("render_contract_subset_sha256")
            != payload["renderer_subset_sha256"]
        ):
            raise SchemaError("live Phase 1 calibration pool provenance differs")
    else:
        recipe = source.get("frozen_recipe")
        pool = source.get("inputs", {}).get("probe_pool") if isinstance(source.get("inputs"), Mapping) else None
        if (
            not isinstance(recipe, Mapping)
            or not isinstance(pool, Mapping)
            or recipe.get("repo_id") != "Qwen/Qwen3-1.7B-Base"
            or recipe.get("revision") != "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
            or recipe.get("task") != "mmlu"
            or recipe.get("num_fewshot") != 5
            or recipe.get("dtype") != "float16"
            or source.get("run_root") != PHASE1_RUN_ROOT
            or pool.get("count") != 512
            or pool.get("render_contract_subset_sha256")
            != payload["renderer_subset_sha256"]
        ):
            raise SchemaError("live Phase 1 full run provenance differs")
        stages = source.get("stages")
        full_stage = stages.get("gate-e-full") if isinstance(stages, Mapping) else None
        jobs = full_stage.get("jobs") if isinstance(full_stage, Mapping) else None
        if not isinstance(jobs, list):
            raise SchemaError("live Phase 1 manifest lacks Gate E full jobs")
        baseline_jobs = [
            job for job in jobs
            if isinstance(job, Mapping) and job.get("job_id") == "baseline-full"
        ]
        if len(baseline_jobs) != 1:
            raise SchemaError("live Phase 1 manifest must contain one baseline-full job")
        baseline = baseline_jobs[0]
        expected_output = Path(str(baseline.get("output_dir", "")))
        expected_results = expected_output / "results.json"
        artifact_paths = [Path(ref["path"]) for ref in payload["identity_artifacts"]]
        if (
            not expected_results.is_absolute()
            or not expected_results.is_relative_to(Path(PHASE1_RUN_ROOT))
            or artifact_paths[0] != expected_results
            or any(path.parent != expected_output for path in artifact_paths[1:])
        ):
            raise SchemaError(
                "full identity artifact differs from the frozen baseline-full results"
            )
    derived = _derive_live_source_identities(payload, identity_namespace, source)
    expected_count = 512 if identity_namespace == CALIBRATION_IDENTITY_NAMESPACE else 14042
    if len(derived) != expected_count:
        raise SchemaError("live Phase 1 identity artifact count differs from the frozen scale")
    if len({(row["task"], row["doc_id"], row["doc_hash"]) for row in derived}) != len(derived):
        raise SchemaError("live Phase 1 identity artifacts contain duplicate identities")
    if expected_identities is not None:
        expected = [_validate_identity(value) for value in expected_identities]
        if derived != expected:
            raise SchemaError("canonical identities do not derive from the live Phase 1 source")
    return source


def _derive_live_source_identities(
    payload: Mapping[str, Any], identity_namespace: str, source: Mapping[str, Any]
) -> List[Dict[str, str]]:
    if identity_namespace == CALIBRATION_IDENTITY_NAMESPACE:
        records = source.get("rendering_records")
        if not isinstance(records, list):
            raise SchemaError("Phase 1 pool manifest lacks rendering_records identity source")
        result = []
        for record in records:
            if not isinstance(record, Mapping):
                raise SchemaError("Phase 1 rendering identity record must be an object")
            result.append(
                stable_sample_identity(
                    record.get("task_name"),
                    record.get("target_doc_id"),
                    record.get("target_doc_sha256"),
                )
            )
        return result
    refs = payload["identity_artifacts"]
    ref = refs[0]
    artifact_path = Path(ref["path"])
    resolved_run_root = Path(PHASE1_RUN_ROOT).resolve()
    resolved_artifact = artifact_path.resolve()
    if not resolved_artifact.is_relative_to(resolved_run_root):
        raise SchemaError("Phase 1 identity artifact resolves outside the immutable run root")
    try:
        raw = resolved_artifact.read_bytes()
    except OSError as exc:
        raise SchemaError("Phase 1 baseline results artifact is unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise SchemaError("Phase 1 baseline results file hash differs")
    try:
        results = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SchemaError("Phase 1 baseline results JSON is malformed") from exc
    if not isinstance(results, Mapping):
        raise SchemaError("Phase 1 baseline results must contain an object")
    if "samples" in results:
        samples = results["samples"]
        if not isinstance(samples, Mapping) or not samples:
            raise SchemaError(
                "present Phase 1 results.samples must be a non-empty mapping"
            )
        if len(refs) != 1:
            raise SchemaError(
                "embedded logged samples must use results.json as the sole identity artifact"
            )
    else:
        task_results = results.get("results")
        task_names = [str(name) for name in task_results] if isinstance(task_results, Mapping) else []
        actual_sidecars = sorted(artifact_path.parent.glob("samples_*.jsonl"))
        declared_sidecars = [Path(value["path"]) for value in refs[1:]]
        if not actual_sidecars or declared_sidecars != actual_sidecars:
            raise SchemaError(
                "declared Phase 1 sample sidecars differ from the exact baseline directory set"
            )
        samples = {}
        for sidecar_ref, sidecar_path in zip(refs[1:], actual_sidecars):
            resolved_sidecar = sidecar_path.resolve()
            if not resolved_sidecar.is_relative_to(resolved_run_root):
                raise SchemaError("Phase 1 sample sidecar resolves outside the run root")
            try:
                sidecar_raw = resolved_sidecar.read_bytes()
            except OSError as exc:
                raise SchemaError("Phase 1 sample sidecar is unreadable") from exc
            if hashlib.sha256(sidecar_raw).hexdigest() != sidecar_ref["sha256"]:
                raise SchemaError("Phase 1 sample sidecar file hash differs")
            raw_namespace = sidecar_path.stem[len("samples_") :]
            matching_names = [
                name for name in task_names
                if raw_namespace == name or raw_namespace.startswith(name + "_")
            ]
            namespace = max(matching_names, key=len) if matching_names else raw_namespace
            if namespace in samples:
                raise SchemaError("Phase 1 sample sidecars map to a duplicate task namespace")
            rows = []
            try:
                sidecar_lines = sidecar_raw.decode("utf-8").splitlines()
            except UnicodeDecodeError as exc:
                raise SchemaError("Phase 1 sample sidecar is not UTF-8") from exc
            for line in sidecar_lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise SchemaError("Phase 1 sample sidecar JSONL is malformed") from exc
                if not isinstance(row, Mapping):
                    raise SchemaError("Phase 1 sample sidecar row must be an object")
                rows.append(row)
            samples[namespace] = rows
    result = []
    for task in sorted(samples):
        rows = samples[task]
        if not isinstance(task, str) or not task or not isinstance(rows, list):
            raise SchemaError("Phase 1 baseline samples mapping is malformed")
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise SchemaError("Phase 1 baseline logged sample must be an object")
            if "task" in row and row["task"] != task:
                raise SchemaError("Phase 1 logged sample task differs from its task bucket")
            try:
                result.append(
                    stable_sample_identity(task, row.get("doc_id"), row.get("doc_hash"))
                )
            except SchemaError as exc:
                raise SchemaError(
                    "Phase 1 baseline task %s row %d lacks doc_id/doc_hash"
                    % (task, row_index)
                ) from exc
    return result


def validate_identity_against_manifest(
    observed: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], card: Mapping[str, Any]
) -> None:
    """Close observed rows against an independent canonical manifest."""

    validate_identity_manifest(manifest, card)
    normalized = [_validate_identity(item) for item in observed]
    expected = manifest["ordered_sample_identity"]
    if normalized != expected:
        raise SchemaError("observed identity order/content differs from canonical manifest")
    if ordered_identity_sha256(normalized) != manifest["ordered_identity_sha256"]:
        raise SchemaError("observed ordered identity hash differs from canonical manifest")


def validate_phase2_workspace_output_path(
    value: Any, card: Mapping[str, Any], *, context: str
) -> Path:
    """Reject legacy roots and symlink escapes for every new Phase 2 artifact."""

    validate_phase2_card(card)
    declared = card["write_once_contract"]["workspace_root"]
    if declared != PHASE2_WORKSPACE_ROOT:
        raise SchemaError("Phase 2 card workspace root differs from project governance")
    raw = Path(_nonempty_string(str(value), context))
    if not raw.is_absolute():
        raise SchemaError("%s must be absolute" % context)
    resolved_root = Path(declared).resolve(strict=False)
    resolved = raw.resolve(strict=False)
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise SchemaError("%s is outside the independent LoopScope workspace" % context)
    return resolved


def validate_contained_path(
    value: Any,
    root: Any,
    *,
    context: str,
    exact_relative: Optional[str] = None,
    allow_root: bool = False,
) -> Path:
    """Resolve a path without accepting traversal, prefix tricks, or symlink escape."""

    raw = Path(_nonempty_string(str(value), context))
    raw_root = Path(_nonempty_string(str(root), "%s root" % context))
    if not raw.is_absolute() or not raw_root.is_absolute():
        raise SchemaError("%s and its root must be absolute" % context)
    if ".." in raw.parts or ".." in raw_root.parts:
        raise SchemaError("%s cannot contain parent traversal" % context)
    resolved_root = raw_root.resolve(strict=False)
    resolved = raw.resolve(strict=False)
    if (resolved == resolved_root and not allow_root) or not resolved.is_relative_to(resolved_root):
        raise SchemaError("%s escapes its declared root" % context)
    if exact_relative is not None:
        relative = Path(exact_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise SchemaError("%s exact relative path is invalid" % context)
        expected = (resolved_root / relative).resolve(strict=False)
        if resolved != expected:
            raise SchemaError("%s differs from its exact governed location" % context)
    return resolved


def validate_ordered_identity(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> None:
    """Compatibility helper for pairwise checks; also rejects duplicates."""

    lhs = [_validate_identity(item) for item in left]
    rhs = [_validate_identity(item) for item in right]
    for name, values in (("left", lhs), ("right", rhs)):
        tuples = [(item["task"], item["doc_id"], item["doc_hash"]) for item in values]
        if len(set(tuples)) != len(tuples):
            raise SchemaError("%s ordered identities contain duplicates" % name)
    if lhs != rhs:
        raise SchemaError("ordered [task,doc_id,doc_hash] identity mismatch")


def validate_phase2_card(card: Mapping[str, Any]) -> None:
    _exact_keys(card, _CARD_KEYS, "Phase 2 card")
    verify_manifest_sha256(card)
    if card["schema_version"] != PHASE2_CARD_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 2 card schema")
    if card["card_id"] != PHASE2_CARD_ID or card["status"] != "freeze_candidate_not_preregistered":
        raise SchemaError("Phase 2 card must remain an unpreregistered H1 V2 freeze candidate")
    if card["primary_contrast_ids"] != list(PRIMARY_CONTRAST_IDS):
        raise SchemaError("primary contrast family/order differs from the frozen contract")
    if card["fixed_horizon_contrast_ids"] != list(FIXED_HORIZON_CONTRAST_IDS):
        raise SchemaError("fixed-horizon family/order differs from the frozen contract")

    _exact_mapping(card, "science", {
        "model": "Qwen/Qwen3-1.7B-Base",
        "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "task": "mmlu",
        "num_fewshot": 5,
        "dtype": "float16",
        "windows": ["11:14", "12:15", "13:16"],
        "window_width_inclusive": 4,
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "primary_metric": "acc,none",
        "calibration_split": "phase1_frozen_validation",
        "calibration_count": 512,
        "calibration_gold_sealed": True,
        "full_split": "mmlu_test_full",
        "full_ordered_sample_count": 14042,
    })
    _exact_mapping(card, "evidence_scales", {
        "calibration_512": {
            "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
            "sample_count": 512,
            "split": "phase1_frozen_validation",
            "trajectory_scope": "residual_q_adjacent_cosine_nca_and_unlabeled_final_choice",
            "residual_nca": "mandatory",
            "gold_state": "sealed_until_authorized_unseal",
            "score_source": DIRECT_PROBE_SCORE_SOURCE,
        },
        "full_14042": {
            "identity_namespace": FULL_IDENTITY_NAMESPACE,
            "sample_count": 14042,
            "split": "mmlu_test_full",
            "trajectory_scope": "evaluator_final_choice_entropy_margin_js_answer_and_gold_paired_outcome_only",
            "residual_nca": "forbidden",
            "gold_state": "full_evaluator_gold_authorized",
            "score_source": FULL_LM_EVAL_SCORE_SOURCE,
        },
    })
    _exact_mapping(card, "trajectories", {
        "fixed_step": [
            {"k": 2, "alpha": 1.0},
            {"k": 3, "alpha": 1.5},
            {"k": 4, "alpha": 2.0},
        ],
        "fixed_horizon": [
            {"k": 1, "alpha": 1.0},
            {"k": 2, "alpha": 1.0},
            {"k": 3, "alpha": 1.0},
            {"k": 4, "alpha": 1.0},
        ],
        "fixed_step_h": 0.5,
        "shared_execution": {"name": "shared_k2_anchor", "k": 2, "alpha": 1.0},
    })
    _exact_mapping(card, "measurement", {
        "nca_position": "final_pre_answer_prompt_token",
        "nca_continuation": "B_N-B_(b+1)",
        "nca_dtype": "float32",
        "nca_min_norm": 1e-8,
        "actual_loop_nca_steps": "t=1..K-1_only",
        "calibration_choice_score_source": DIRECT_PROBE_SCORE_SOURCE,
        "full_choice_score_source": FULL_LM_EVAL_SCORE_SOURCE,
        "choice_probability": "float32_softmax",
        "entropy_log_base": "e",
        "js_log_base": "e",
        "top1_tie_break": "first_maximum_A_B_C_D",
        "top_margin": "raw_top1_score_minus_raw_top2_score",
        "correct_margin": "raw_gold_score_minus_max_raw_non_gold_score",
        "final_output_forbids_intermediate_lens_fields": True,
    })
    _exact_mapping(card, "statistics", {
        "nca_bootstrap": {"replicates": 10000, "seed": 0, "interval": [2.5, 97.5], "denominator": 512},
        "full_paired_bootstrap": {"replicates": 2000, "seed": 20260710, "interval": [2.5, 97.5], "denominator": 14042},
        "mcnemar": "two_sided_exact_binomial",
        "holm_familywise_alpha": 0.05,
        "primary_family_order": list(PRIMARY_CONTRAST_IDS),
        "fixed_horizon_family_order": list(FIXED_HORIZON_CONTRAST_IDS),
        "nca_interval_role": "pointwise_unadjusted_secondary_diagnostic",
        "paired_nca_resampling": "same_sample_index_for_both_medians",
        "native_fidelity_resampling": "stratified_within_frozen_subgroup",
        "quantile_method": "linear_interpolation_n_minus_1",
        "unit": "percentage_points",
    })
    _exact_mapping(card, "decision_rules", frozen_decision_rules())
    _exact_mapping(card, "tolerances", {
        "k1_scores_atol": 1e-4,
        "k1_scores_rtol": 1e-5,
        "prefix_atol": 1e-5,
        "prefix_rtol": 1e-3,
    })
    _exact_mapping(card, "phase1_reuse_policy", {
        "allowed_statuses": [
            "local_schema_confirmed", "historical_only", "requires_gate_b_live_check"
        ],
        "gate_a_remote_reads": 0,
        "k2_full_rerun_authorized": False,
        "phase1_run_manifest_path": PHASE1_RUN_ROOT + "/control/phase1_run_manifest.json",
        "reuse_full_cell_ids": list(phase1_reuse_full_cell_ids()),
        "new_full_cell_ids": list(phase2_new_full_cell_ids()),
        "reuse_cell_to_job": {
            phase1_reuse_full_cell_ids()[0]: "baseline-full",
            phase1_reuse_full_cell_ids()[1]: "window-11-14-full",
            phase1_reuse_full_cell_ids()[2]: "window-12-15-full",
            phase1_reuse_full_cell_ids()[3]: "window-13-16-full",
        },
        "live_digest_status": "requires_gate_b_live_check",
        "unknown_digest_policy": "fail_closed_no_synthesis_no_k2_rerun",
    })
    _exact_mapping(card, "write_once_contract", {
        "workspace_root": PHASE2_WORKSPACE_ROOT,
        "sidecar_only": True,
        "analysis_output_create": "same_directory_atomic_exclusive_create_fsync",
        "native_continuation_lifetime": "same_process_memory_only",
        "full_vectors_persisted": False,
        "full_residual_nca_fields_allowed": False,
        "b2_no_loop_boundary_cells": 1,
        "b2_logical_loop_cells": 15,
    })
    hashes = _mapping(card, "implementation_sha256")
    if set(hashes) != set(IMPLEMENTATION_FILES):
        raise SchemaError("implementation hash closure differs from the frozen file set")
    for path, digest in hashes.items():
        _sha256(digest, "implementation_sha256.%s" % path)
    _reject_non_finite(card)


def make_calibration_cell_envelope(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    *,
    cell: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = _make_envelope(
        card,
        identity_manifest,
        schema_version=PHASE2_CALIBRATION_CELL_SCHEMA_VERSION,
        artifact_kind="calibration_trajectory_cell",
        evidence_scale="calibration_512",
        label_state="sealed",
        score_source=DIRECT_PROBE_SCORE_SOURCE,
        cell=cell,
        samples=samples,
        producer=producer,
    )
    validate_calibration_cell_envelope(payload, card, identity_manifest)
    return payload


def validate_calibration_cell_envelope(
    payload: Mapping[str, Any], card: Mapping[str, Any], identity_manifest: Mapping[str, Any]
) -> None:
    _validate_envelope_common(
        payload,
        card,
        identity_manifest,
        schema_version=PHASE2_CALIBRATION_CELL_SCHEMA_VERSION,
        artifact_kind="calibration_trajectory_cell",
        evidence_scale="calibration_512",
        label_state="sealed",
        score_source=DIRECT_PROBE_SCORE_SOURCE,
    )
    samples = payload["samples"]
    for observed, expected in zip(samples, identity_manifest["ordered_sample_identity"]):
        validate_trajectory_sample(observed, payload["cell"], expected_identity=expected)
    _reject_calibration_gold_fields(payload)


def make_full_final_output_envelope(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    *,
    cell: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    artifact_root: Any,
    producer_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = _make_envelope(
        card,
        identity_manifest,
        schema_version=PHASE2_FULL_FINAL_OUTPUT_SCHEMA_VERSION,
        artifact_kind="full_final_output_cell",
        evidence_scale="full_14042",
        label_state="full_evaluator_gold_authorized",
        score_source=FULL_LM_EVAL_SCORE_SOURCE,
        cell=cell,
        samples=samples,
        producer=producer,
        artifact_root=str(artifact_root),
        producer_evidence=producer_evidence,
    )
    validate_full_final_output_envelope(payload, card, identity_manifest)
    return payload


def make_calibration_baseline_envelope(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    *,
    samples: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    payload: Dict[str, Any] = {
        "schema_version": PHASE2_CALIBRATION_BASELINE_SCHEMA_VERSION,
        "artifact_kind": "calibration_no_loop_baseline",
        "evidence_scale": "calibration_512",
        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "revision": card["science"]["revision"],
        "label_state": "sealed",
        "score_source": DIRECT_PROBE_SCORE_SOURCE,
        "sample_count": len(samples),
        "samples": [dict(sample) for sample in samples],
        "producer": dict(producer),
    }
    attach_manifest_sha256(payload)
    validate_calibration_baseline_envelope(payload, card, identity_manifest)
    return payload


def validate_calibration_baseline_envelope(
    payload: Mapping[str, Any], card: Mapping[str, Any], identity_manifest: Mapping[str, Any]
) -> None:
    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    _exact_keys(payload, {
        "schema_version", "artifact_kind", "evidence_scale", "identity_namespace",
        "card_manifest_sha256", "identity_manifest_sha256", "ordered_identity_sha256",
        "revision", "label_state", "score_source", "sample_count", "samples",
        "producer", "manifest_sha256",
    }, "calibration baseline envelope")
    verify_manifest_sha256(payload)
    expected = {
        "schema_version": PHASE2_CALIBRATION_BASELINE_SCHEMA_VERSION,
        "artifact_kind": "calibration_no_loop_baseline",
        "evidence_scale": "calibration_512",
        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "revision": card["science"]["revision"],
        "label_state": "sealed",
        "score_source": DIRECT_PROBE_SCORE_SOURCE,
        "sample_count": 512,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise SchemaError("calibration baseline %s differs from frozen contract" % key)
    _validate_producer(
        _mapping(payload, "producer"), allowed_kinds={"gate_b_remote_calibration_probe"}
    )
    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) != 512:
        raise SchemaError("calibration baseline must retain 512 samples")
    validate_identity_against_manifest(
        [sample.get("sample_identity", {}) if isinstance(sample, Mapping) else {} for sample in samples],
        identity_manifest,
        card,
    )
    for sample, expected_identity in zip(samples, identity_manifest["ordered_sample_identity"]):
        validate_calibration_baseline_sample(
            sample, card, expected_identity=expected_identity
        )
    _reject_calibration_gold_fields(payload)
    _reject_non_finite(payload)


def validate_calibration_baseline_sample(
    sample: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
) -> None:
    """Validate one sealed direct-probe baseline row at either B1 or B2 scale."""

    validate_phase2_card(card)
    _exact_keys(sample, _BASELINE_SAMPLE_KEYS, "calibration baseline sample")
    if _validate_identity(sample["sample_identity"]) != _validate_identity(expected_identity):
        raise SchemaError("calibration baseline identity differs from canonical row")
    _strict_int(sample["answer_position"], "answer_position", minimum=0)
    validate_final_output_sample(
        sample["final_output"],
        score_source=DIRECT_PROBE_SCORE_SOURCE,
        gold_state="forbidden",
        expected_identity=expected_identity,
    )
    boundaries = sample["window_boundaries"]
    if (
        not isinstance(boundaries, list)
        or [row.get("window") for row in boundaries if isinstance(row, Mapping)]
        != card["science"]["windows"]
    ):
        raise SchemaError("calibration baseline must contain three frozen windows in order")
    for row in boundaries:
        _exact_keys(row, _BASELINE_BOUNDARY_KEYS, "calibration baseline boundary")
        window = row["window"]
        start, end = (int(value) for value in window.split(":"))
        if row["inclusive_boundary"] != "B_%d_to_B_%d" % (start, end + 1):
            raise SchemaError("baseline inclusive boundary identity differs")
        if row["native_boundary_formula"] != "B_N-B_%d" % (end + 1):
            raise SchemaError("baseline native boundary formula differs")
        _nonnegative_float(row["baseline_update_norm"], "baseline_update_norm")
        _nonnegative_float(row["native_continuation_norm"], "native_continuation_norm")
        _validate_baseline_nca(row["baseline_nca"])
    _reject_calibration_gold_fields(sample)
    _reject_non_finite(sample)


def make_unseal_authorization(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    *,
    sealed_artifact_sha256: Sequence[str],
    authorization_id: str,
    authorized_by_thread: str,
    issued_at_utc: str,
) -> Dict[str, Any]:
    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    sealed = sorted(_unique_sha256_list(sealed_artifact_sha256, "sealed_artifact_sha256"))
    payload: Dict[str, Any] = {
        "schema_version": PHASE2_UNSEAL_AUTHORIZATION_SCHEMA_VERSION,
        "artifact_kind": "calibration_label_unseal_authorization",
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "sealed_artifact_sha256": sealed,
        "purpose": "frozen_native_fidelity_conditional_analysis_only",
        "label_access": "gold_index_only_no_card_or_threshold_change",
        "authorization_id": _nonempty_string(authorization_id, "authorization_id"),
        "authorized_by_thread": _nonempty_string(authorized_by_thread, "authorized_by_thread"),
        "issued_at_utc": _utc_timestamp(issued_at_utc, "issued_at_utc"),
    }
    attach_manifest_sha256(payload)
    return payload


def validate_unseal_authorization(
    payload: Mapping[str, Any],
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    sealed_artifact_sha256: Sequence[str],
) -> None:
    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    _exact_keys(payload, {
        "schema_version", "artifact_kind", "card_manifest_sha256", "identity_namespace",
        "identity_manifest_sha256", "ordered_identity_sha256", "sealed_artifact_sha256",
        "purpose", "label_access", "authorization_id", "authorized_by_thread",
        "issued_at_utc", "manifest_sha256",
    }, "unseal authorization")
    verify_manifest_sha256(payload)
    expected = {
        "schema_version": PHASE2_UNSEAL_AUTHORIZATION_SCHEMA_VERSION,
        "artifact_kind": "calibration_label_unseal_authorization",
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "sealed_artifact_sha256": sorted(_unique_sha256_list(sealed_artifact_sha256, "sealed_artifact_sha256")),
        "purpose": "frozen_native_fidelity_conditional_analysis_only",
        "label_access": "gold_index_only_no_card_or_threshold_change",
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise SchemaError("unseal authorization %s differs from bound evidence" % key)
    _nonempty_string(payload["authorization_id"], "authorization_id")
    _nonempty_string(payload["authorized_by_thread"], "authorized_by_thread")
    _utc_timestamp(payload["issued_at_utc"], "issued_at_utc")


def make_calibration_label_sidecar(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    gold_indices: Sequence[int],
    producer: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    if len(gold_indices) != 512:
        raise SchemaError("calibration label sidecar must contain 512 gold indices")
    samples = []
    for expected_identity, gold in zip(identity_manifest["ordered_sample_identity"], gold_indices):
        value = _strict_int(gold, "gold_index", minimum=0)
        if value > 3:
            raise SchemaError("gold_index must be in [0,3]")
        samples.append({"sample_identity": dict(expected_identity), "gold_index": value})
    payload: Dict[str, Any] = {
        "schema_version": PHASE2_CALIBRATION_LABEL_SCHEMA_VERSION,
        "artifact_kind": "authorized_calibration_gold_indices",
        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "revision": card["science"]["revision"],
        "authorization_manifest_sha256": authorization.get("manifest_sha256"),
        "label_state": "authorized_unsealed_gold_index_only",
        "sample_count": 512,
        "samples": samples,
        "producer": dict(producer),
    }
    attach_manifest_sha256(payload)
    return payload


def validate_calibration_label_sidecar(
    payload: Mapping[str, Any],
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    _exact_keys(payload, {
        "schema_version", "artifact_kind", "identity_namespace", "card_manifest_sha256",
        "identity_manifest_sha256", "ordered_identity_sha256", "revision",
        "authorization_manifest_sha256", "label_state", "sample_count", "samples",
        "producer", "manifest_sha256",
    }, "calibration label sidecar")
    verify_manifest_sha256(payload)
    expected = {
        "schema_version": PHASE2_CALIBRATION_LABEL_SCHEMA_VERSION,
        "artifact_kind": "authorized_calibration_gold_indices",
        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "revision": card["science"]["revision"],
        "authorization_manifest_sha256": authorization.get("manifest_sha256"),
        "label_state": "authorized_unsealed_gold_index_only",
        "sample_count": 512,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise SchemaError("calibration label sidecar %s differs from authorization" % key)
    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) != 512:
        raise SchemaError("calibration label sidecar must contain 512 samples")
    validate_identity_against_manifest(
        [sample.get("sample_identity", {}) if isinstance(sample, Mapping) else {} for sample in samples],
        identity_manifest,
        card,
    )
    for sample in samples:
        _exact_keys(sample, {"sample_identity", "gold_index"}, "calibration label sample")
        gold = _strict_int(sample["gold_index"], "gold_index", minimum=0)
        if gold > 3:
            raise SchemaError("gold_index must be in [0,3]")
    _validate_producer(
        _mapping(payload, "producer"), allowed_kinds={"authorized_calibration_label_unseal"}
    )


def make_unseal_receipt(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    label_sidecar: Mapping[str, Any],
    *,
    sealed_artifact_sha256: Sequence[str],
    accessed_by_thread: str,
    accessed_at_utc: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": PHASE2_UNSEAL_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": "calibration_label_unseal_receipt",
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "authorization_manifest_sha256": authorization.get("manifest_sha256"),
        "label_sidecar_sha256": label_sidecar.get("manifest_sha256"),
        "sealed_artifact_sha256": sorted(_unique_sha256_list(sealed_artifact_sha256, "sealed_artifact_sha256")),
        "purpose": "frozen_native_fidelity_conditional_analysis_only",
        "accessed_by_thread": _nonempty_string(accessed_by_thread, "accessed_by_thread"),
        "accessed_at_utc": _utc_timestamp(accessed_at_utc, "accessed_at_utc"),
    }
    attach_manifest_sha256(payload)
    return payload


def validate_unseal_receipt(
    payload: Mapping[str, Any],
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    label_sidecar: Mapping[str, Any],
    sealed_artifact_sha256: Sequence[str],
) -> None:
    _exact_keys(payload, {
        "schema_version", "artifact_kind", "card_manifest_sha256", "identity_namespace",
        "identity_manifest_sha256", "ordered_identity_sha256", "authorization_manifest_sha256",
        "label_sidecar_sha256", "sealed_artifact_sha256", "purpose", "accessed_by_thread",
        "accessed_at_utc", "manifest_sha256",
    }, "unseal receipt")
    verify_manifest_sha256(payload)
    expected = {
        "schema_version": PHASE2_UNSEAL_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": "calibration_label_unseal_receipt",
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "authorization_manifest_sha256": authorization.get("manifest_sha256"),
        "label_sidecar_sha256": label_sidecar.get("manifest_sha256"),
        "sealed_artifact_sha256": sorted(_unique_sha256_list(sealed_artifact_sha256, "sealed_artifact_sha256")),
        "purpose": "frozen_native_fidelity_conditional_analysis_only",
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise SchemaError("unseal receipt %s differs from bound evidence" % key)
    _nonempty_string(payload["accessed_by_thread"], "accessed_by_thread")
    _utc_timestamp(payload["accessed_at_utc"], "accessed_at_utc")


def validate_full_final_output_envelope(
    payload: Mapping[str, Any], card: Mapping[str, Any], identity_manifest: Mapping[str, Any]
) -> None:
    _validate_envelope_common(
        payload,
        card,
        identity_manifest,
        schema_version=PHASE2_FULL_FINAL_OUTPUT_SCHEMA_VERSION,
        artifact_kind="full_final_output_cell",
        evidence_scale="full_14042",
        label_state="full_evaluator_gold_authorized",
        score_source=FULL_LM_EVAL_SCORE_SOURCE,
    )
    for observed, expected in zip(payload["samples"], identity_manifest["ordered_sample_identity"]):
        validate_final_output_sample(
            observed,
            score_source=FULL_LM_EVAL_SCORE_SOURCE,
            gold_state="required",
            expected_identity=expected,
        )


def validate_trajectory_sample(
    sample: Mapping[str, Any], cell: Mapping[str, Any], *, expected_identity: Mapping[str, Any]
) -> None:
    _exact_keys(sample, _TRAJECTORY_SAMPLE_KEYS, "trajectory sample")
    if _validate_identity(sample["sample_identity"]) != _validate_identity(expected_identity):
        raise SchemaError("trajectory sample identity differs from canonical row")
    k = _strict_int(cell["k"], "cell.k", minimum=1)
    if _strict_int(sample["answer_position"], "answer_position", minimum=0) < 0:
        raise SchemaError("answer_position must be non-negative")
    if sample["position_rule"] != "final_pre_answer_prompt_token":
        raise SchemaError("trajectory position rule differs")
    if sample["boundary_identity"] != "native_continuation=B_N-B_(b+1)":
        raise SchemaError("trajectory boundary identity differs")
    if sample["body_call_indexing"] != "zero_based_t=0..K-1":
        raise SchemaError("trajectory body-call indexing differs")
    steps = sample["steps"]
    if not isinstance(steps, list) or len(steps) != k:
        raise SchemaError("trajectory sample must contain exactly K steps")
    for index, step in enumerate(steps):
        _validate_trajectory_step(step, index)
    repeated_valid = all(step["nca"] is not None and step["nca"]["valid"] for step in steps[1:])
    if k == 1:
        repeated_valid = True
    if sample["repeated_step_nca_all_valid"] is not repeated_valid:
        raise SchemaError("repeated-step NCA validity summary disagrees with steps")
    if not isinstance(sample["valid"], bool):
        raise SchemaError("trajectory valid must be boolean")
    if not isinstance(sample["errors"], list) or any(
        not isinstance(value, str) or not value for value in sample["errors"]
    ):
        raise SchemaError("trajectory errors must be a list of non-empty strings")
    if sample["valid"] is not (not sample["errors"] and repeated_valid):
        raise SchemaError("trajectory valid flag disagrees with errors/NCA validity")
    counts = sample["event_counts"]
    _exact_keys(counts, {
        "wrapper_forward", "bypass_false", "body_call", "operator_body_calls",
        "g_minus_x", "looped_hidden_vs_input", "stash_pass", "identity_forward",
    }, "trajectory event_counts")
    expected_counts = {
        "wrapper_forward": 1,
        "bypass_false": 1,
        "body_call": k,
        "operator_body_calls": k,
        "g_minus_x": k,
        "looped_hidden_vs_input": 1,
        "stash_pass": 1,
        "identity_forward": 3,
    }
    if not strict_frozen_equal(dict(counts), expected_counts):
        raise SchemaError("trajectory event counts differ from the frozen wrapper timeline")
    validate_final_output_sample(
        sample["final_output"],
        score_source=DIRECT_PROBE_SCORE_SOURCE,
        gold_state="forbidden",
        expected_identity=expected_identity,
    )


def validate_final_output_sample(
    sample: Mapping[str, Any],
    *,
    score_source: str,
    gold_state: str,
    expected_identity: Optional[Mapping[str, Any]] = None,
) -> None:
    keys = _FINAL_BASE_KEYS | (_FINAL_GOLD_KEYS if gold_state == "required" else frozenset())
    if gold_state not in ("required", "forbidden"):
        raise SchemaError("gold_state must be required or forbidden")
    _exact_keys(sample, keys, "final-output sample")
    if sample["schema_version"] != PHASE2_FINAL_OUTPUT_SCHEMA_VERSION:
        raise SchemaError("unsupported final-output sample schema")
    identity = _validate_identity(sample["sample_identity"])
    if expected_identity is not None and identity != _validate_identity(expected_identity):
        raise SchemaError("final-output sample identity differs from canonical row")
    if sample["choice_labels"] != list(CHOICE_LABELS):
        raise SchemaError("choice labels/order must be A,B,C,D")
    if sample["choice_score_source"] != score_source:
        raise SchemaError("choice score source differs from the artifact contract")
    scores = _finite_float_list(sample["raw_choice_scores"], 4, "raw_choice_scores")
    probabilities = _finite_float_list(sample["choice_probabilities"], 4, "choice_probabilities")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise SchemaError("choice probabilities must be in [0,1]")
    expected_probabilities = _float32_softmax(scores)
    if any(not math.isclose(left, right, abs_tol=2e-7, rel_tol=2e-6) for left, right in zip(probabilities, expected_probabilities)):
        raise SchemaError("choice probabilities do not equal frozen float32 softmax")
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=2e-6, rel_tol=0.0):
        raise SchemaError("choice probabilities do not sum to one")
    if sample["probability_dtype"] != "float32":
        raise SchemaError("choice probability dtype must be float32")
    top1 = max(range(4), key=lambda index: scores[index])
    if _strict_int(sample["top1_index"], "top1_index", minimum=0) != top1 or top1 > 3:
        raise SchemaError("top1 index differs from first maximum")
    if sample["top1_label"] != CHOICE_LABELS[top1]:
        raise SchemaError("top1 label differs from top1 index")
    if sample["top1_tie_break"] != "first_maximum_A_B_C_D":
        raise SchemaError("top1 tie-break differs")
    ordered = sorted(scores, reverse=True)
    _close_float(sample["top_margin_raw"], ordered[0] - ordered[1], "top_margin_raw")
    entropy = -math.fsum(value * math.log(value) for value in probabilities if value > 0.0)
    _close_float(sample["entropy_nats"], entropy, "entropy_nats", abs_tol=2e-6)
    if gold_state == "required":
        gold = _strict_int(sample["gold_index"], "gold_index", minimum=0)
        if gold > 3:
            raise SchemaError("gold_index must be in [0,3]")
        correctness = top1 == gold
        if sample["correctness"] is not correctness or sample["evaluator_acc_none"] is not correctness:
            raise SchemaError("top1/correctness disagrees with evaluator acc,none")
        correct_margin = scores[gold] - max(score for index, score in enumerate(scores) if index != gold)
        _close_float(sample["correct_margin_raw"], correct_margin, "correct_margin_raw")


def validate_scalar_sidecar(payload: Mapping[str, Any]) -> None:
    """Validate one standalone sample without claiming canonical identity closure."""

    schema = payload.get("schema_version")
    if schema == PHASE2_FINAL_OUTPUT_SCHEMA_VERSION:
        source = payload.get("choice_score_source")
        gold_state = "required" if any(key in payload for key in _FINAL_GOLD_KEYS) else "forbidden"
        validate_final_output_sample(payload, score_source=source, gold_state=gold_state)
        return
    raise SchemaError("standalone trajectory samples require cell and canonical manifest context")


def validate_no_persisted_vectors(payload: Any) -> None:
    """Reject explicit full vector/tensor payloads in scalar-only artifacts."""

    forbidden_exact = {
        "hidden", "hidden_state", "hidden_states", "residual", "delta",
        "native_continuation", "native_continuation_vector", "tensor", "full_vectors",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                lower = str(key).lower()
                if lower in forbidden_exact or lower.endswith("_tensor") or lower.endswith("_vector"):
                    raise SchemaError("persisted full vector/tensor field is forbidden: %s.%s" % (path, key))
                visit(nested, "%s.%s" % (path, key))
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, "%s[%d]" % (path, index))

    visit(payload, "root")
    _reject_non_finite(payload)


def validate_protocol_cell(cell: Mapping[str, Any], *, allow_baseline: bool = False) -> None:
    _validate_cell(cell, allow_baseline=allow_baseline)


def validate_producer_provenance(
    producer: Mapping[str, Any], *, allowed_kinds: Optional[Iterable[str]] = None
) -> None:
    _validate_producer(producer, allowed_kinds=allowed_kinds)


def verify_phase2_implementation_hashes(card: Mapping[str, Any], repo_root: Path) -> None:
    validate_phase2_card(card)
    for relative_path in IMPLEMENTATION_FILES:
        actual = hashlib.sha256((Path(repo_root) / relative_path).read_bytes()).hexdigest()
        if actual != card["implementation_sha256"][relative_path]:
            raise SchemaError("implementation SHA256 mismatch for %s" % relative_path)


def make_hashed_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    attach_manifest_sha256(result)
    return result


def b2_logical_cells(card: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
    validate_phase2_card(card)
    settings = (
        ("shared_k2_anchor", 2, 1.0),
        ("fixed_step", 3, 1.5),
        ("fixed_step", 4, 2.0),
        ("fixed_horizon", 3, 1.0),
        ("fixed_horizon", 4, 1.0),
    )
    cells: List[Dict[str, Any]] = []
    for window in card["science"]["windows"]:
        for protocol, k, alpha in settings:
            cell = {
                "cell_id": protocol_cell_id(protocol, window, k, alpha),
                "protocol": protocol,
                "window": window,
                "k": k,
                "alpha": alpha,
                "step_size": alpha / k,
            }
            _validate_cell(cell, allow_baseline=False)
            cells.append(cell)
    if len(cells) != 15 or len({cell["cell_id"] for cell in cells}) != 15:
        raise SchemaError("B2 logical-cell plan is not exactly 15 unique cells")
    return cells


def atomic_write_new_json(path: Path, payload: Any) -> None:
    """Create a JSON file atomically and exclusively, then fsync file+directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = target.with_name(".%s.%s.tmp" % (target.name, uuid.uuid4().hex))
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary), str(target))
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        raise FileExistsError("refusing to overwrite existing artifact: %s" % target)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_envelope(
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    *,
    schema_version: str,
    artifact_kind: str,
    evidence_scale: str,
    label_state: str,
    score_source: str,
    cell: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    artifact_root: Optional[str] = None,
    producer_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "artifact_kind": artifact_kind,
        "evidence_scale": evidence_scale,
        "identity_namespace": identity_manifest["identity_namespace"],
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "revision": card["science"]["revision"],
        "cell": dict(cell),
        "label_state": label_state,
        "score_source": score_source,
        "sample_count": len(samples),
        "samples": [dict(sample) for sample in samples],
        "producer": dict(producer),
    }
    if artifact_root is not None or producer_evidence is not None:
        if artifact_root is None or producer_evidence is None:
            raise SchemaError("full envelope root and producer evidence must be supplied together")
        payload["artifact_root"] = artifact_root
        payload["producer_evidence"] = dict(producer_evidence)
    attach_manifest_sha256(payload)
    return payload


def _validate_envelope_common(
    payload: Mapping[str, Any],
    card: Mapping[str, Any],
    identity_manifest: Mapping[str, Any],
    *,
    schema_version: str,
    artifact_kind: str,
    evidence_scale: str,
    label_state: str,
    score_source: str,
) -> None:
    validate_phase2_card(card)
    validate_identity_manifest(identity_manifest, card)
    envelope_keys = _FULL_ENVELOPE_KEYS if evidence_scale == "full_14042" else _ENVELOPE_KEYS
    _exact_keys(payload, envelope_keys, "%s envelope" % artifact_kind)
    verify_manifest_sha256(payload)
    expected = {
        "schema_version": schema_version,
        "artifact_kind": artifact_kind,
        "evidence_scale": evidence_scale,
        "identity_namespace": card["evidence_scales"][evidence_scale]["identity_namespace"],
        "card_manifest_sha256": card["manifest_sha256"],
        "identity_manifest_sha256": identity_manifest["manifest_sha256"],
        "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
        "revision": card["science"]["revision"],
        "label_state": label_state,
        "score_source": score_source,
        "sample_count": card["evidence_scales"][evidence_scale]["sample_count"],
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise SchemaError("%s.%s differs from the frozen envelope" % (artifact_kind, key))
    if identity_manifest["identity_namespace"] != expected["identity_namespace"]:
        raise SchemaError("artifact and canonical manifest use different identity namespaces")
    cell = _mapping(payload, "cell")
    _validate_cell(cell, allow_baseline=evidence_scale == "full_14042")
    allowed_kinds = (
        {"gate_b_remote_calibration_probe"}
        if evidence_scale == "calibration_512"
        else {"lm_eval_logged_samples_adapter", "phase1_immutable_reuse_adapter"}
    )
    producer = _mapping(payload, "producer")
    _validate_producer(producer, allowed_kinds=allowed_kinds)
    if evidence_scale == "full_14042":
        expected_kind = expected_full_producer_kind(cell)
        if producer["producer_kind"] != expected_kind:
            raise SchemaError(
                "full cell %s requires producer_kind=%s"
                % (cell["cell_id"], expected_kind)
            )
        artifact_root = Path(_nonempty_string(payload["artifact_root"], "artifact_root"))
        if not artifact_root.is_absolute() or ".." in artifact_root.parts:
            raise SchemaError("full artifact_root must be an absolute canonical path")
        _validate_full_producer_evidence_structure(
            _mapping(payload, "producer_evidence"), producer
        )
    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) != expected["sample_count"]:
        raise SchemaError("artifact sample count differs from frozen scale")
    validate_identity_against_manifest(
        [sample.get("sample_identity", {}) if isinstance(sample, Mapping) else {} for sample in samples],
        identity_manifest,
        card,
    )
    validate_no_persisted_vectors({"samples": samples} if evidence_scale == "full_14042" else {})
    _reject_non_finite(payload)


def _validate_trajectory_step(step: Mapping[str, Any], expected_t: int) -> None:
    _exact_keys(step, _STEP_KEYS, "trajectory step")
    if _strict_int(step["body_call_t"], "body_call_t", minimum=0) != expected_t:
        raise SchemaError("trajectory body calls must be zero-based and contiguous")
    residual_norm = _nonnegative_float(step["residual_norm"], "residual_norm")
    state_norm = _positive_float(step["state_norm"], "state_norm")
    _close_float(step["relative_activity"], residual_norm / state_norm, "relative_activity")
    if step["valid"] is not True:
        raise SchemaError("scalar trajectory step must be valid; NCA validity is separate")
    if expected_t == 0:
        if any(step[key] is not None for key in (
            "residual_ratio_to_previous", "adjacent_residual_cosine",
            "adjacent_residual_cosine_valid", "nca"
        )):
            raise SchemaError("t=0 cannot contain adjacent or repeated-step NCA fields")
        return
    _nonnegative_float(step["residual_ratio_to_previous"], "residual_ratio_to_previous")
    if not isinstance(step["adjacent_residual_cosine_valid"], bool):
        raise SchemaError("adjacent cosine validity must be boolean")
    adjacent = step["adjacent_residual_cosine"]
    if step["adjacent_residual_cosine_valid"]:
        _bounded_cosine(adjacent, "adjacent_residual_cosine")
    elif adjacent is not None:
        raise SchemaError("invalid adjacent cosine must use null value")
    nca = step["nca"]
    if not isinstance(nca, Mapping):
        raise SchemaError("repeated steps require an explicit NCA validity object")
    _exact_keys(nca, _NCA_KEYS, "NCA measurement")
    if nca["dtype"] != "float32" or nca["provenance"] != "actual_loop_repeated_step":
        raise SchemaError("NCA dtype/provenance differs")
    if _strict_int(nca["body_call_t"], "NCA body_call_t", minimum=1) != expected_t:
        raise SchemaError("NCA body-call identity differs from its step")
    _close_float(nca["residual_norm"], residual_norm, "NCA residual_norm")
    _nonnegative_float(nca["native_continuation_norm"], "native_continuation_norm")
    if not isinstance(nca["valid"], bool):
        raise SchemaError("NCA valid must be boolean")
    if nca["valid"]:
        _bounded_cosine(nca["value"], "NCA value")
        if nca["invalid_reason"] is not None:
            raise SchemaError("valid NCA cannot carry invalid_reason")
    else:
        if nca["value"] is not None or nca["invalid_reason"] not in {
            "norm_below_1e-8", "non_finite", "shape_dtype_device_mismatch"
        }:
            raise SchemaError("invalid NCA requires null value and frozen reason")


def _validate_baseline_nca(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise SchemaError("baseline NCA must be an explicit validity object")
    _exact_keys(value, _NCA_KEYS, "baseline NCA measurement")
    if value["dtype"] != "float32" or value["provenance"] != "baseline_single_forward_B_(b+1)-B_a":
        raise SchemaError("baseline NCA dtype/provenance differs")
    if value["body_call_t"] is not None:
        raise SchemaError("baseline NCA cannot use a loop body-call index")
    _nonnegative_float(value["residual_norm"], "baseline NCA residual_norm")
    _nonnegative_float(value["native_continuation_norm"], "baseline NCA native_continuation_norm")
    if not isinstance(value["valid"], bool):
        raise SchemaError("baseline NCA valid must be boolean")
    if value["valid"]:
        _bounded_cosine(value["value"], "baseline NCA value")
        if value["invalid_reason"] is not None:
            raise SchemaError("valid baseline NCA cannot carry invalid_reason")
    elif value["value"] is not None or value["invalid_reason"] not in {
        "norm_below_1e-8", "non_finite", "shape_dtype_device_mismatch"
    }:
        raise SchemaError("invalid baseline NCA requires null value and frozen reason")


def _validate_cell(cell: Mapping[str, Any], *, allow_baseline: bool) -> None:
    _exact_keys(cell, _CELL_KEYS, "protocol cell")
    protocol = _nonempty_string(cell["protocol"], "cell.protocol")
    window = _nonempty_string(cell["window"], "cell.window")
    k = _strict_int(cell["k"], "cell.k", minimum=1)
    alpha = _positive_float(cell["alpha"], "cell.alpha")
    _close_float(cell["step_size"], alpha / k, "cell.step_size")
    if cell["cell_id"] != protocol_cell_id(protocol, window, k, alpha):
        raise SchemaError("cell_id does not encode protocol/window/K/alpha")
    allowed = {
        ("shared_k2_anchor", "11:14", 2, 1.0),
        ("shared_k2_anchor", "12:15", 2, 1.0),
        ("shared_k2_anchor", "13:16", 2, 1.0),
        ("fixed_step", "11:14", 3, 1.5),
        ("fixed_step", "11:14", 4, 2.0),
        ("fixed_step", "12:15", 3, 1.5),
        ("fixed_step", "12:15", 4, 2.0),
        ("fixed_step", "13:16", 3, 1.5),
        ("fixed_step", "13:16", 4, 2.0),
        ("fixed_horizon", "11:14", 3, 1.0),
        ("fixed_horizon", "11:14", 4, 1.0),
        ("fixed_horizon", "12:15", 3, 1.0),
        ("fixed_horizon", "12:15", 4, 1.0),
        ("fixed_horizon", "13:16", 3, 1.0),
        ("fixed_horizon", "13:16", 4, 1.0),
    }
    if allow_baseline:
        allowed |= {
            ("baseline_no_loop", "none", 1, 1.0),
            ("fixed_horizon", "12:15", 1, 1.0),
            ("fixed_horizon", "12:15", 2, 1.0),
        }
    if (protocol, window, k, alpha) not in allowed:
        raise SchemaError("protocol cell is not part of the frozen H1 matrix")


def _validate_producer(
    producer: Mapping[str, Any], *, allowed_kinds: Optional[Iterable[str]] = None
) -> None:
    if not isinstance(producer, Mapping):
        raise SchemaError("producer provenance must be an object")
    if producer.get("producer_kind") == "lm_eval_logged_samples_adapter":
        keys = _FULL_ADAPTER_PRODUCER_KEYS
    elif producer.get("producer_kind") == "phase1_immutable_reuse_adapter":
        keys = _FULL_REUSE_PRODUCER_KEYS
    else:
        keys = _PRODUCER_KEYS
    _exact_keys(producer, keys, "producer provenance")
    kind = _nonempty_string(producer["producer_kind"], "producer_kind")
    if allowed_kinds is not None and kind not in set(allowed_kinds):
        raise SchemaError("producer_kind is not allowed for this artifact")
    for key in keys - {"producer_kind"}:
        _sha256(producer[key], "producer.%s" % key)


def _validate_full_producer_evidence_structure(
    evidence: Mapping[str, Any], producer: Mapping[str, Any]
) -> None:
    kind = producer["producer_kind"]
    expected = (
        _NEW_FULL_EVIDENCE_KEYS
        if kind == "lm_eval_logged_samples_adapter"
        else _REUSE_FULL_EVIDENCE_KEYS
    )
    _exact_keys(evidence, expected, "full producer evidence")
    for key, value in evidence.items():
        if key == "sample_source":
            continue
        _validate_file_ref(value, "full producer evidence.%s" % key)
    sample_source = evidence["sample_source"]
    _exact_keys(sample_source, {"kind", "artifacts"}, "full sample source")
    if sample_source["kind"] not in {
        "results_inline_samples",
        "exact_sorted_logged_sample_sidecars",
    }:
        raise SchemaError("unsupported full sample-source kind")
    artifacts = sample_source["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise SchemaError("full sample source requires at least one artifact")
    for index, ref in enumerate(artifacts):
        _validate_file_ref(ref, "full sample source artifact %d" % index)
    if sample_source["kind"] == "results_inline_samples" and len(artifacts) != 1:
        raise SchemaError("inline full sample source must bind exactly results.json")
    hash_bindings = {
        "attempt_manifest": "attempt_manifest_sha256",
        "receipt_manifest": "receipt_manifest_sha256",
        "command_args": "command_sha256",
        "environment": "environment_sha256",
    }
    if kind == "lm_eval_logged_samples_adapter":
        hash_bindings.update(
            {
                "revision_report": "revision_report_sha256",
                "results": "results_sha256",
            }
        )
        if sample_source["artifacts"][0] != evidence["results"]:
            raise SchemaError("new full sample source must be its actual results.json")
    else:
        hash_bindings.update(
            {
                "source_run_manifest": "source_manifest_sha256",
                "source_revision_report": "revision_report_sha256",
                "source_results": "results_sha256",
            }
        )
        if sample_source["kind"] == "results_inline_samples" and (
            sample_source["artifacts"][0] != evidence["source_results"]
        ):
            raise SchemaError("Phase 1 inline sample source must be source_results")
    for evidence_key, producer_key in hash_bindings.items():
        if evidence[evidence_key]["sha256"] != producer[producer_key]:
            raise SchemaError("full producer evidence hash differs from producer summary")


def _validate_file_ref(value: Any, context: str) -> None:
    _exact_keys(value, _FILE_REF_KEYS, context)
    path = Path(_nonempty_string(value["path"], "%s.path" % context))
    if not path.is_absolute() or ".." in path.parts:
        raise SchemaError("%s path must be absolute and canonical" % context)
    _sha256(value["sha256"], "%s.sha256" % context)


def _reject_calibration_gold_fields(payload: Any, path: str = "root") -> None:
    # ``target`` is a legitimate label-free renderer provenance object
    # (source/split/subject) in the immutable Phase 1 pool.  Closed-world
    # rendering validation owns that one exception; every scientific sealed
    # object still rejects label-bearing target/gold/correctness fields.
    forbidden = {"gold", "gold_index", "correctness", "correct_margin_raw", "evaluator_acc_none"}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in forbidden:
                raise SchemaError("sealed calibration artifact contains forbidden label field: %s.%s" % (path, key))
            _reject_calibration_gold_fields(value, "%s.%s" % (path, key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _reject_calibration_gold_fields(value, "%s[%d]" % (path, index))


def validate_sealed_no_gold_fields(payload: Any) -> None:
    """Public recursive guard shared by every sealed calibration root."""

    _reject_calibration_gold_fields(payload)


def _scale_for_namespace(namespace: str) -> str:
    if namespace == CALIBRATION_IDENTITY_NAMESPACE:
        return "calibration_512"
    if namespace == FULL_IDENTITY_NAMESPACE:
        return "full_14042"
    raise SchemaError("unknown Phase 2 identity namespace")


def _validate_identity(value: Mapping[str, Any]) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise SchemaError("sample identity must be an object")
    _exact_keys(value, _IDENTITY_KEYS, "sample identity")
    return stable_sample_identity(value["task"], value["doc_id"], value["doc_hash"])


def _exact_mapping(payload: Mapping[str, Any], key: str, expected: Mapping[str, Any]) -> None:
    actual = _mapping(payload, key)
    if not strict_frozen_equal(actual, expected):
        raise SchemaError("%s differs from the exact frozen contract" % key)


def strict_frozen_equal(actual: Any, expected: Any) -> bool:
    """JSON-tree equality that never treats bool as numeric 0/1."""

    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(strict_frozen_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(strict_frozen_equal(left, right) for left, right in zip(actual, expected))
        )
    if type(actual) is not type(expected):
        return False
    return bool(actual == expected)


def _exact_keys(payload: Mapping[str, Any], expected: Iterable[str], context: str) -> None:
    if not isinstance(payload, Mapping):
        raise SchemaError("%s must be an object" % context)
    actual = set(payload)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        unknown = sorted(actual - wanted)
        raise SchemaError("%s exact-key mismatch; missing=%s unknown=%s" % (context, missing, unknown))


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SchemaError("%s must be an object" % key)
    return value


def _strict_int(value: Any, context: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError("%s must be an integer" % context)
    if minimum is not None and value < minimum:
        raise SchemaError("%s must be >= %d" % (context, minimum))
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError("%s must be numeric" % context)
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError("%s must be finite" % context)
    return result


def _positive_float(value: Any, context: str) -> float:
    result = _finite_float(value, context)
    if result <= 0.0:
        raise SchemaError("%s must be positive" % context)
    return result


def _nonnegative_float(value: Any, context: str) -> float:
    result = _finite_float(value, context)
    if result < 0.0:
        raise SchemaError("%s must be non-negative" % context)
    return result


def _bounded_cosine(value: Any, context: str) -> float:
    result = _finite_float(value, context)
    if result < -1.0 or result > 1.0:
        raise SchemaError("%s must be in [-1,1]" % context)
    return result


def _finite_float_list(value: Any, length: int, context: str) -> List[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaError("%s must contain exactly %d values" % (context, length))
    return [_finite_float(item, "%s[%d]" % (context, index)) for index, item in enumerate(value)]


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("%s must be a non-empty string" % context)
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA256" % context)
    return value


def _unique_sha256_list(values: Sequence[str], context: str) -> List[str]:
    if not isinstance(values, (list, tuple)) or len(values) != 16:
        raise SchemaError("%s must contain the baseline plus 15 sealed cell hashes" % context)
    normalized = [_sha256(value, "%s[%d]" % (context, index)) for index, value in enumerate(values)]
    if len(set(normalized)) != len(normalized):
        raise SchemaError("%s must contain unique hashes" % context)
    return normalized


def _utc_timestamp(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SchemaError("%s must be a real YYYY-MM-DDTHH:MM:SSZ UTC time" % context) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SchemaError("%s must use UTC" % context)
    return text


def _close_float(
    actual: Any, expected: float, context: str, *, abs_tol: float = 1e-9, rel_tol: float = 1e-9
) -> None:
    value = _finite_float(actual, context)
    if not math.isclose(value, float(expected), abs_tol=abs_tol, rel_tol=rel_tol):
        raise SchemaError("%s differs from its derived value" % context)


def _float32_softmax(scores: Sequence[float]) -> List[float]:
    rounded = [_f32(value) for value in scores]
    peak = max(rounded)
    weights = [_f32(math.exp(_f32(value - peak))) for value in rounded]
    total = _f32(math.fsum(weights))
    if total <= 0.0:
        raise SchemaError("softmax mass is not positive")
    return [_f32(value / total) for value in weights]


def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _reject_non_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_non_finite(nested, "%s.%s" % (path, key))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_non_finite(nested, "%s[%d]" % (path, index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise SchemaError("non-finite value at %s" % path)


def _format_float(value: float) -> str:
    return ("%.12g" % float(value)).replace("-", "m").replace(".", "p")


__all__ = [
    "CALIBRATION_IDENTITY_NAMESPACE",
    "CHOICE_LABELS",
    "DIRECT_PROBE_SCORE_SOURCE",
    "FIXED_HORIZON_CONTRAST_IDS",
    "FULL_IDENTITY_NAMESPACE",
    "FULL_LM_EVAL_SCORE_SOURCE",
    "MMLU_DATASET_REVISION",
    "PHASE1_INPUT_ROOT",
    "PHASE1_RUN_ROOT",
    "PHASE2_WORKSPACE_ROOT",
    "IMPLEMENTATION_FILES",
    "PHASE2_ANALYSIS_INPUT_SCHEMA_VERSION",
    "PHASE2_ANALYSIS_SCHEMA_VERSION",
    "PHASE2_CALIBRATION_BASELINE_SCHEMA_VERSION",
    "PHASE2_CALIBRATION_CELL_SCHEMA_VERSION",
    "PHASE2_CARD_ID",
    "PHASE2_CARD_SCHEMA_VERSION",
    "PHASE2_FINAL_OUTPUT_SCHEMA_VERSION",
    "PHASE2_FULL_FINAL_OUTPUT_SCHEMA_VERSION",
    "PHASE2_IDENTITY_MANIFEST_SCHEMA_VERSION",
    "PHASE2_REUSE_SCHEMA_VERSION",
    "PHASE2_TRACE_SCHEMA_VERSION",
    "PRIMARY_CONTRAST_IDS",
    "REUSE_STATUSES",
    "SchemaError",
    "atomic_write_new_json",
    "b2_logical_cells",
    "canonical_json_bytes",
    "file_sha256",
    "frozen_decision_rules",
    "expected_full_producer_kind",
    "make_calibration_cell_envelope",
    "make_calibration_baseline_envelope",
    "make_calibration_label_sidecar",
    "make_full_final_output_envelope",
    "make_hashed_manifest",
    "make_identity_manifest",
    "make_source_provenance",
    "make_unseal_authorization",
    "make_unseal_receipt",
    "manifest_sha256",
    "ordered_identity_sha256",
    "phase1_reuse_full_cell_ids",
    "phase2_new_full_cell_ids",
    "protocol_cell_id",
    "stable_sample_identity",
    "strict_frozen_equal",
    "validate_calibration_cell_envelope",
    "validate_calibration_baseline_envelope",
    "validate_calibration_baseline_sample",
    "validate_calibration_label_sidecar",
    "validate_contained_path",
    "validate_final_output_sample",
    "validate_full_final_output_envelope",
    "validate_identity_against_manifest",
    "validate_identity_manifest",
    "validate_no_persisted_vectors",
    "validate_ordered_identity",
    "validate_phase2_card",
    "validate_phase2_workspace_output_path",
    "validate_producer_provenance",
    "validate_protocol_cell",
    "validate_scalar_sidecar",
    "validate_sealed_no_gold_fields",
    "validate_source_provenance",
    "verify_live_source_provenance",
    "validate_trajectory_sample",
    "validate_unseal_authorization",
    "validate_unseal_receipt",
    "verify_phase2_implementation_hashes",
]
