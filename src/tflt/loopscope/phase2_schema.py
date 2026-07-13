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
PHASE2_CARD_SCHEMA_VERSION = "loopscope.phase2-h1-card.v3"
PHASE2_IDENTITY_MANIFEST_SCHEMA_VERSION = "loopscope.phase2-identity-manifest.v1"
PHASE2_TRACE_SCHEMA_VERSION = "loopscope.phase2-calibration-trajectory-sample.v2"
PHASE2_CALIBRATION_CELL_SCHEMA_VERSION = "loopscope.phase2-calibration-cell.v2"
PHASE2_CALIBRATION_BASELINE_SCHEMA_VERSION = "loopscope.phase2-calibration-baseline.v2"
PHASE2_FINAL_OUTPUT_SCHEMA_VERSION = "loopscope.phase2-final-output-sample.v2"
PHASE2_FULL_FINAL_OUTPUT_SCHEMA_VERSION = "loopscope.phase2-full-final-output-cell.v2"
PHASE2_CALIBRATION_LABEL_SCHEMA_VERSION = "loopscope.phase2-calibration-labels.v1"
PHASE2_UNSEAL_AUTHORIZATION_SCHEMA_VERSION = "loopscope.phase2-unseal-authorization.v1"
PHASE2_UNSEAL_RECEIPT_SCHEMA_VERSION = "loopscope.phase2-unseal-receipt.v1"
PHASE2_ANALYSIS_INPUT_SCHEMA_VERSION = "loopscope.phase2-h1-analysis-input.v2"
PHASE2_ANALYSIS_SCHEMA_VERSION = "loopscope.phase2-h1-analysis-report.v2"
PHASE2_REUSE_SCHEMA_VERSION = "loopscope.phase2-reuse-matrix.v1"

CALIBRATION_IDENTITY_NAMESPACE = "loopscope.phase2.calibration.validation512.v1"
FULL_IDENTITY_NAMESPACE = "loopscope.phase2.full.mmlu14042.v1"
DIRECT_PROBE_SCORE_SOURCE = (
    "direct_probe_next_token_log_probability_over_frozen_choice_token_ids"
)
FULL_LM_EVAL_SCORE_SOURCE = "lm_eval_acc_none_raw_per_choice_loglikelihood"

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
    "fh_12_15_k1_k2",
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
        "card_manifest_sha256", "revision", "manifest_sha256",
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
    _exact_mapping(card, "decision_rules", {
        "h1_priority": [
            "PERTURBATION", "REFINEMENT_SUPPORTED", "TRANSIENT_ONLY", "SUGGESTIVE", "INCONCLUSIVE"
        ],
        "nca_priority": [
            "NCA_INCONCLUSIVE", "NCA_NATIVE_FIDELITY_ONLY", "NCA_DIRECTION_SUPPORTED", "NCA_NONDISCRIMINATIVE"
        ],
        "significance_comparison": "adjusted_p_strictly_less_than_0.05",
        "nca_valid_fraction_min": 0.95,
        "native_fidelity_min_valid_per_subgroup": 10,
        "wrong_overconfidence": {
            "condition": "entropy_decreases_and_raw_top_margin_increases",
            "denominator": "final_wrong_pairs=wrong_to_wrong+right_to_wrong",
            "subgroups_reported": ["wrong_to_wrong", "right_to_wrong"],
            "role": "mechanism_diagnostic_not_H1_decision_rule",
        },
    })
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
    })
    _exact_mapping(card, "write_once_contract", {
        "workspace_root": "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope",
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
    if dict(counts) != expected_counts:
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
    _exact_keys(payload, _ENVELOPE_KEYS, "%s envelope" % artifact_kind)
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
    _validate_cell(_mapping(payload, "cell"), allow_baseline=evidence_scale == "full_14042")
    allowed_kinds = (
        {"gate_b_remote_calibration_probe"}
        if evidence_scale == "calibration_512"
        else {"lm_eval_logged_samples_adapter", "phase1_immutable_reuse_adapter"}
    )
    _validate_producer(_mapping(payload, "producer"), allowed_kinds=allowed_kinds)
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
    _exact_keys(producer, _PRODUCER_KEYS, "producer provenance")
    kind = _nonempty_string(producer["producer_kind"], "producer_kind")
    if allowed_kinds is not None and kind not in set(allowed_kinds):
        raise SchemaError("producer_kind is not allowed for this artifact")
    for key in _PRODUCER_KEYS - {"producer_kind"}:
        _sha256(producer[key], "producer.%s" % key)


def _reject_calibration_gold_fields(payload: Any, path: str = "root") -> None:
    forbidden = {"gold", "gold_index", "target", "correctness", "correct_margin_raw", "evaluator_acc_none"}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in forbidden:
                raise SchemaError("sealed calibration artifact contains forbidden label field: %s.%s" % (path, key))
            _reject_calibration_gold_fields(value, "%s.%s" % (path, key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _reject_calibration_gold_fields(value, "%s[%d]" % (path, index))


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
    if actual != expected:
        raise SchemaError("%s differs from the exact frozen contract" % key)


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
    if len(text) != 20 or text[4] != "-" or text[7] != "-" or text[10] != "T" or text[13] != ":" or text[16] != ":" or not text.endswith("Z"):
        raise SchemaError("%s must use YYYY-MM-DDTHH:MM:SSZ" % context)
    digits = text.replace("-", "").replace("T", "").replace(":", "").replace("Z", "")
    if not digits.isdigit():
        raise SchemaError("%s must use numeric UTC fields" % context)
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
    "make_calibration_cell_envelope",
    "make_calibration_baseline_envelope",
    "make_calibration_label_sidecar",
    "make_full_final_output_envelope",
    "make_hashed_manifest",
    "make_identity_manifest",
    "make_unseal_authorization",
    "make_unseal_receipt",
    "manifest_sha256",
    "ordered_identity_sha256",
    "protocol_cell_id",
    "stable_sample_identity",
    "validate_calibration_cell_envelope",
    "validate_calibration_baseline_envelope",
    "validate_calibration_baseline_sample",
    "validate_calibration_label_sidecar",
    "validate_final_output_sample",
    "validate_full_final_output_envelope",
    "validate_identity_against_manifest",
    "validate_identity_manifest",
    "validate_no_persisted_vectors",
    "validate_ordered_identity",
    "validate_phase2_card",
    "validate_producer_provenance",
    "validate_protocol_cell",
    "validate_scalar_sidecar",
    "validate_trajectory_sample",
    "validate_unseal_authorization",
    "validate_unseal_receipt",
    "verify_phase2_implementation_hashes",
]
