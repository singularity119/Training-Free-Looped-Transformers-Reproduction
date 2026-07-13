"""Conservative, local-only Phase 1 reuse classification for Phase 2."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from tflt.loopscope.phase2_schema import (
    PHASE2_CARD_ID,
    PHASE2_REUSE_SCHEMA_VERSION,
    REUSE_STATUSES,
    b2_logical_cells,
    make_hashed_manifest,
    validate_phase2_card,
    verify_phase2_implementation_hashes,
)


def build_phase1_reuse_matrix() -> Dict[str, Any]:
    """Return schema-confirmed facts without pretending to read live HPC artifacts."""

    entries: List[Dict[str, Any]] = [
        _entry(
            "calibration_pool_identity",
            "local_schema_confirmed",
            "Phase 1 probe-pool schema binds ordered sample IDs, prompt hashes, validation split, 512 count, and target-label exclusion.",
        ),
        _entry(
            "inclusive_boundary_and_k2_activity",
            "local_schema_confirmed",
            "Phase 1 probe/window schemas bind inclusive width-4 boundaries, K=2 body-call timeline, r/q, and validity.",
        ),
        _entry(
            "phase1_full_completion_and_revision_closure",
            "historical_only",
            "Control/history records baseline plus K2 completion and revision closure; Gate A performs no live HPC read.",
        ),
        _entry(
            "raw_four_choice_loglikelihood",
            "requires_gate_b_live_check",
            "Layer probe stores distributions, not the four raw lm-eval acc,none log-likelihood scores.",
        ),
        _entry(
            "full_ordered_doc_hash_identity",
            "requires_gate_b_live_check",
            "Local Phase 1 analysis accepts doc_id correctness but does not prove ordered task/doc_id/doc_hash closure.",
        ),
        _entry(
            "k2_per_step_nca_and_final_choice",
            "requires_gate_b_live_check",
            "Phase 1 window probe does not persist repeated-step NCA or final raw choice scores.",
        ),
        _entry(
            "logged_sample_exact_identity_and_raw_choice_join",
            "requires_gate_b_live_check",
            "The final-output adapter is order-independent and batch-size agnostic, but Gate B must verify that live lm-eval 0.4.11 logged samples expose exact task/doc_id/doc_hash and four raw choice scores.",
        ),
    ]
    return {
        "schema_version": PHASE2_REUSE_SCHEMA_VERSION,
        "verification_scope": "local_read_only_no_ssh",
        "allowed_statuses": sorted(REUSE_STATUSES),
        "entries": entries,
    }


def validate_reuse_matrix(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"schema_version", "verification_scope", "allowed_statuses", "entries"}:
        raise ValueError("reuse matrix has an exact-key mismatch")
    if payload.get("schema_version") != PHASE2_REUSE_SCHEMA_VERSION:
        raise ValueError("unsupported reuse matrix schema")
    if payload.get("verification_scope") != "local_read_only_no_ssh":
        raise ValueError("Gate A reuse matrix cannot claim live remote verification")
    if payload.get("allowed_statuses") != sorted(REUSE_STATUSES):
        raise ValueError("reuse matrix allowed statuses differ")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("reuse matrix requires entries")
    names = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"field", "status", "evidence"}:
            raise ValueError("reuse entry has an exact-key mismatch")
        if entry.get("status") not in REUSE_STATUSES:
            raise ValueError("reuse entry has invalid status")
        if not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip():
            raise ValueError("reuse entry evidence must be non-empty")
        names.append(str(entry.get("field") or ""))
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("reuse entry names must be non-empty and unique")


def build_phase2_freeze_candidate(card: Mapping[str, Any], repo_root: Any) -> Dict[str, Any]:
    """Single canonical prepare producer shared by the CLI and helper script."""

    validate_phase2_card(card)
    verify_phase2_implementation_hashes(card, repo_root)
    reuse = build_phase1_reuse_matrix()
    validate_reuse_matrix(reuse)
    return make_hashed_manifest(
        {
            "schema_version": "loopscope.phase2-preparation.v2",
            "status": "freeze_candidate_not_preregistered",
            "card_id": PHASE2_CARD_ID,
            "card_manifest_sha256": card["manifest_sha256"],
            "evidence_scale_contract": dict(card["evidence_scales"]),
            "workspace_root": card["write_once_contract"]["workspace_root"],
            "legacy_phase1_policy": "read_only_in_place_no_move_copy_replace_or_symlink",
            "b1_modes": {
                "smoke": "four_samples_with_k1_and_cross_k_prefix_proof",
                "calibration": "requires_successful_b1_proof_and_512_sealed_identity",
            },
            "no_loop_boundary": {
                "logical_cell_count": 1,
                "native_continuation_lifetime": "same_process_memory_only",
            },
            "loop_cells": list(b2_logical_cells(card)),
            "full_final_output_adapter": {
                "identity_join": "unordered_exact_task_doc_id_doc_hash_to_canonical_manifest",
                "batch_size_auto_allowed": True,
                "residual_collector_connected": False,
                "score_source": card["measurement"]["full_choice_score_source"],
            },
            "phase1_reuse_matrix": reuse,
            "gate_a_execution": {
                "remote_reads": 0,
                "gpu_hours": 0,
                "new_run_count": 0,
            },
        }
    )


def _entry(field: str, status: str, evidence: str) -> Dict[str, str]:
    if status not in REUSE_STATUSES:
        raise ValueError("invalid reuse status")
    return {"field": field, "status": status, "evidence": evidence}


__all__ = [
    "build_phase1_reuse_matrix",
    "build_phase2_freeze_candidate",
    "validate_reuse_matrix",
]
