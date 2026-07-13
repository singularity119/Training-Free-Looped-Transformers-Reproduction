"""Strict paired and NCA analysis for ``H1_ITERATIVE_REFINEMENT_ZONE_V2``."""

from __future__ import annotations

import math
import random
import struct
import json
import hashlib
import os
from datetime import datetime, timezone
from statistics import median
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase2_schema import (
    CALIBRATION_IDENTITY_NAMESPACE,
    CHOICE_LABELS,
    FIXED_HORIZON_CONTRAST_IDS,
    FULL_IDENTITY_NAMESPACE,
    FULL_LM_EVAL_SCORE_SOURCE,
    PHASE2_ANALYSIS_INPUT_SCHEMA_VERSION,
    PHASE2_ANALYSIS_SCHEMA_VERSION,
    PHASE2_FINAL_OUTPUT_SCHEMA_VERSION,
    PRIMARY_CONTRAST_IDS,
    SchemaError,
    atomic_write_new_json,
    b2_logical_cells,
    canonical_json_bytes,
    file_sha256,
    frozen_decision_rules,
    make_hashed_manifest,
    manifest_sha256,
    protocol_cell_id,
    stable_sample_identity,
    strict_frozen_equal,
    validate_calibration_baseline_envelope,
    validate_calibration_cell_envelope,
    validate_calibration_label_sidecar,
    validate_full_final_output_envelope,
    validate_identity_manifest,
    validate_ordered_identity,
    validate_phase2_card,
    validate_phase2_workspace_output_path,
    validate_scalar_sidecar,
    validate_source_provenance,
    verify_live_source_provenance,
    validate_unseal_authorization,
    validate_unseal_receipt,
)


class Phase2AnalysisError(ValueError):
    """Raised instead of analyzing incomplete or ambiguously paired evidence."""


def required_calibration_cell_ids(card: Mapping[str, Any]) -> List[str]:
    """Return all 15 sealed B2 cells in their frozen execution order."""

    validate_phase2_card(card)
    return [cell["cell_id"] for cell in b2_logical_cells(card)]


def required_full_cell_ids(card: Mapping[str, Any]) -> List[str]:
    """Return the exact 12 full final-output cells consumed by H1 analysis."""

    validate_phase2_card(card)
    result = [protocol_cell_id("baseline_no_loop", "none", 1, 1.0)]
    for window in card["science"]["windows"]:
        result.extend(
            (
                protocol_cell_id("shared_k2_anchor", window, 2, 1.0),
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
    if len(result) != 12 or len(set(result)) != 12:
        raise Phase2AnalysisError("frozen full evidence matrix is not 12 unique cells")
    return result


def make_analysis_input_manifest(
    card: Mapping[str, Any],
    sources: Mapping[str, Any],
    analysis_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a source-only analysis request; no derived statistics are accepted."""

    payload = make_hashed_manifest(
        {
            "schema_version": PHASE2_ANALYSIS_INPUT_SCHEMA_VERSION,
            "sources": dict(sources),
            "statistics_contract": dict(card["statistics"]),
            "analysis_provenance": dict(analysis_provenance),
        }
    )
    validate_analysis_input_manifest(payload, card)
    return payload


def validate_analysis_input_manifest(payload: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    """Validate the closed-world routing manifest before reading source files."""

    validate_phase2_card(card)
    _analysis_exact_keys(
        payload,
        {"schema_version", "sources", "statistics_contract", "analysis_provenance", "manifest_sha256"},
        "analysis input",
    )
    from tflt.loopscope.schema import verify_manifest_sha256

    verify_manifest_sha256(payload)
    if payload["schema_version"] != PHASE2_ANALYSIS_INPUT_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 2 analysis-input schema")
    if not strict_frozen_equal(payload["statistics_contract"], card["statistics"]):
        raise SchemaError("analysis statistics contract differs from the frozen card")
    sources = payload["sources"]
    _analysis_exact_keys(
        sources,
        {
            "card",
            "calibration_identity_manifest",
            "full_identity_manifest",
            "calibration_baseline",
            "calibration_aggregate",
            "calibration_cells",
            "full_final_output_cells",
            "calibration_labels",
            "unseal_authorization",
            "unseal_receipt",
        },
        "analysis sources",
    )
    for key in (
        "card",
        "calibration_identity_manifest",
        "full_identity_manifest",
        "calibration_baseline",
        "calibration_aggregate",
        "calibration_labels",
        "unseal_authorization",
        "unseal_receipt",
    ):
        _validate_source_ref(sources[key], "sources.%s" % key)
    if sources["card"]["sha256"] != card["manifest_sha256"]:
        raise SchemaError("analysis input card ref differs from the validated card")
    _validate_cell_refs(
        sources["calibration_cells"], required_calibration_cell_ids(card), "calibration_cells"
    )
    _validate_cell_refs(
        sources["full_final_output_cells"], required_full_cell_ids(card), "full_final_output_cells"
    )
    provenance = payload["analysis_provenance"]
    _analysis_exact_keys(
        provenance,
        {
            "authorization_manifest",
            "attempt_manifest",
            "receipt_manifest",
            "executor_thread_id",
        },
        "analysis provenance",
    )
    for key in ("authorization_manifest", "attempt_manifest", "receipt_manifest"):
        _validate_source_ref(provenance[key], "analysis_provenance.%s" % key)
    if not isinstance(provenance["executor_thread_id"], str) or not provenance["executor_thread_id"].strip():
        raise SchemaError("analysis executor_thread_id must be non-empty")


def _load_analysis_control_manifests(
    request: Mapping[str, Any], base: Path, card: Mapping[str, Any]
) -> Dict[str, Any]:
    provenance = request["analysis_provenance"]
    authorization = _load_ref(
        provenance["authorization_manifest"], base, "analysis authorization"
    )
    attempt = _load_ref(provenance["attempt_manifest"], base, "analysis attempt")
    receipt = _load_ref(provenance["receipt_manifest"], base, "analysis receipt")
    authorization_keys = {
        "schema_version", "artifact_kind", "card_manifest_sha256",
        "analysis_sources_sha256", "output_root", "executor_thread_id",
        "created_at_utc", "manifest_sha256",
    }
    attempt_keys = authorization_keys | {"authorization_manifest_sha256"}
    receipt_keys = authorization_keys | {
        "authorization_manifest_sha256", "attempt_manifest_sha256"
    }
    _analysis_exact_keys(authorization, authorization_keys, "analysis authorization")
    _analysis_exact_keys(attempt, attempt_keys, "analysis attempt")
    _analysis_exact_keys(receipt, receipt_keys, "analysis receipt")
    if (
        authorization["schema_version"] != "loopscope.phase2-analysis-authorization.v1"
        or authorization["artifact_kind"] != "phase2_h1_analysis_authorization"
        or attempt["schema_version"] != "loopscope.phase2-analysis-attempt.v1"
        or attempt["artifact_kind"] != "phase2_h1_analysis_attempt"
        or receipt["schema_version"] != "loopscope.phase2-analysis-receipt.v1"
        or receipt["artifact_kind"] != "phase2_h1_analysis_execution_receipt"
    ):
        raise SchemaError("unsupported analysis authorization/attempt/receipt schema")
    sources_sha = hashlib.sha256(canonical_json_bytes(request["sources"])).hexdigest()
    executor = provenance["executor_thread_id"]
    output_roots = set()
    for packet in (authorization, attempt, receipt):
        if (
            packet["card_manifest_sha256"] != card["manifest_sha256"]
            or packet["analysis_sources_sha256"] != sources_sha
            or packet["executor_thread_id"] != executor
        ):
            raise SchemaError("analysis control packet card/source/executor binding differs")
        _analysis_utc(packet["created_at_utc"])
        output_roots.add(
            str(
                validate_phase2_workspace_output_path(
                    packet["output_root"], card, context="analysis control output_root"
                )
            )
        )
    if len(output_roots) != 1:
        raise SchemaError("analysis control packets disagree on output_root")
    if attempt["authorization_manifest_sha256"] != authorization["manifest_sha256"]:
        raise SchemaError("analysis attempt is bound to a different authorization")
    if (
        receipt["authorization_manifest_sha256"] != authorization["manifest_sha256"]
        or receipt["attempt_manifest_sha256"] != attempt["manifest_sha256"]
    ):
        raise SchemaError("analysis receipt is bound to different control packets")
    if not (
        _analysis_utc(authorization["created_at_utc"])
        <= _analysis_utc(attempt["created_at_utc"])
        <= _analysis_utc(receipt["created_at_utc"])
    ):
        raise SchemaError("analysis control timestamps are out of order")
    return {
        "authorization": authorization,
        "attempt": attempt,
        "receipt": receipt,
        "output_root": Path(next(iter(output_roots))),
    }


def analysis_command_record(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    return {
        "schema_version": "loopscope.phase2-analysis-command.v1",
        "command": "analyze-phase2-h1",
        "input": str(Path(input_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
    }


def analysis_environment_record() -> Dict[str, Any]:
    keys = (
        "PYTHONPATH", "VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES", "HF_HOME",
        "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE",
    )
    return {
        "schema_version": "loopscope.phase2-analysis-environment.v1",
        "variables": {key: os.environ[key] for key in keys if key in os.environ},
    }


def derive_analysis_execution_provenance(
    evidence: Mapping[str, Any], output_dir: Path
) -> Dict[str, Any]:
    control = evidence.get("analysis_control")
    if not isinstance(control, Mapping):
        raise SchemaError("analysis evidence lacks loaded control manifests")
    resolved_output = validate_phase2_workspace_output_path(
        output_dir, evidence["card"], context="analysis report output directory"
    )
    if resolved_output != Path(control["output_root"]).resolve():
        raise SchemaError("analysis report directory differs from authorized output_root")
    command_path = resolved_output / "command_args.json"
    environment_path = resolved_output / "env.json"
    try:
        command = json.loads(command_path.read_text(encoding="utf-8"))
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SchemaError("analysis command/environment evidence is unreadable") from exc
    expected_command = analysis_command_record(evidence["request_path"], resolved_output)
    if not strict_frozen_equal(command, expected_command):
        raise SchemaError("analysis command_args.json differs from the actual request/output")
    _analysis_exact_keys(
        environment, {"schema_version", "variables"}, "analysis environment"
    )
    if environment["schema_version"] != "loopscope.phase2-analysis-environment.v1" or not isinstance(environment["variables"], Mapping):
        raise SchemaError("analysis environment schema differs")
    result = {
        "authorization_manifest_sha256": control["authorization"]["manifest_sha256"],
        "attempt_manifest_sha256": control["attempt"]["manifest_sha256"],
        "receipt_manifest_sha256": control["receipt"]["manifest_sha256"],
        "executor_thread_id": control["attempt"]["executor_thread_id"],
        "command_sha256": file_sha256(command_path),
        "environment_sha256": file_sha256(environment_path),
    }
    _validate_analysis_provenance(result)
    return result


def _analysis_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SchemaError("analysis control timestamp must be text")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SchemaError("analysis control timestamp must be a real UTC timestamp") from exc


def load_phase2_analysis_evidence(input_path: Path) -> Dict[str, Any]:
    """Load and validate every canonical per-sample source named by the request."""

    path = Path(input_path)
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping) or "sources" not in request:
        raise SchemaError("analysis input must be a source-only manifest")
    raw_sources = request["sources"]
    if not isinstance(raw_sources, Mapping) or "card" not in raw_sources:
        raise SchemaError("analysis input lacks a card source")
    base = path.parent
    card = _load_ref(raw_sources["card"], base, "card")
    validate_phase2_card(card)
    validate_analysis_input_manifest(request, card)
    analysis_control = _load_analysis_control_manifests(request, base, card)
    sources = request["sources"]

    calibration_identity = _load_ref(
        sources["calibration_identity_manifest"], base, "calibration_identity_manifest"
    )
    full_identity = _load_ref(sources["full_identity_manifest"], base, "full_identity_manifest")
    validate_identity_manifest(calibration_identity, card)
    validate_identity_manifest(full_identity, card)
    if calibration_identity["identity_namespace"] != CALIBRATION_IDENTITY_NAMESPACE:
        raise SchemaError("calibration source uses the wrong identity namespace")
    if full_identity["identity_namespace"] != FULL_IDENTITY_NAMESPACE:
        raise SchemaError("full source uses the wrong identity namespace")
    validate_source_provenance(
        calibration_identity["source_provenance"],
        CALIBRATION_IDENTITY_NAMESPACE,
        require_live=True,
    )
    validate_source_provenance(
        full_identity["source_provenance"], FULL_IDENTITY_NAMESPACE, require_live=True
    )
    verify_live_source_provenance(
        calibration_identity["source_provenance"],
        CALIBRATION_IDENTITY_NAMESPACE,
        calibration_identity["ordered_sample_identity"],
    )
    verify_live_source_provenance(
        full_identity["source_provenance"],
        FULL_IDENTITY_NAMESPACE,
        full_identity["ordered_sample_identity"],
    )

    calibration_baseline = _load_ref(
        sources["calibration_baseline"], base, "calibration_baseline"
    )
    validate_calibration_baseline_envelope(calibration_baseline, card, calibration_identity)

    calibration_cells: Dict[str, Mapping[str, Any]] = {}
    for ref in sources["calibration_cells"]:
        cell = _load_ref(ref, base, "calibration cell %s" % ref["cell_id"])
        if cell.get("cell", {}).get("cell_id") != ref["cell_id"]:
            raise SchemaError("calibration cell ref and artifact cell_id disagree")
        validate_calibration_cell_envelope(cell, card, calibration_identity)
        calibration_cells[ref["cell_id"]] = cell

    calibration_aggregate = _load_ref(
        sources["calibration_aggregate"], base, "calibration_aggregate"
    )
    aggregate_path = _source_ref_path(
        sources["calibration_aggregate"], base
    )
    prior_b1_ref = calibration_aggregate.get("prior_b1_admission_proof")
    if not isinstance(prior_b1_ref, Mapping):
        raise SchemaError("calibration aggregate lacks its prior B1 proof ref")
    b1_proof = _load_ref(prior_b1_ref, aggregate_path.parent, "prior B1 admission proof")
    _require_same_source_ref(
        calibration_aggregate["identity_manifest"],
        sources["calibration_identity_manifest"],
        aggregate_path.parent,
        base,
        "calibration identity manifest",
    )
    _require_same_source_ref(
        calibration_aggregate["baseline"],
        sources["calibration_baseline"],
        aggregate_path.parent,
        base,
        "calibration baseline",
    )
    aggregate_cell_refs = {
        ref["cell_id"]: ref for ref in calibration_aggregate.get("cells", [])
        if isinstance(ref, Mapping) and isinstance(ref.get("cell_id"), str)
    }
    for external_ref in sources["calibration_cells"]:
        cell_id = external_ref["cell_id"]
        if cell_id not in aggregate_cell_refs:
            raise SchemaError("calibration aggregate lacks cell ref %s" % cell_id)
        _require_same_source_ref(
            aggregate_cell_refs[cell_id],
            external_ref,
            aggregate_path.parent,
            base,
            "calibration cell %s" % cell_id,
            ignore_cell_id=True,
        )
    from tflt.loopscope.phase2_trajectory import validate_probe_aggregate

    validate_probe_aggregate(
        calibration_aggregate,
        card,
        b1_proof=b1_proof,
        identity_manifest=calibration_identity,
        baseline=calibration_baseline,
        cells=calibration_cells,
        aggregate_path=aggregate_path,
    )

    full_cells: Dict[str, Mapping[str, Any]] = {}
    for ref in sources["full_final_output_cells"]:
        cell = _load_ref(ref, base, "full final-output cell %s" % ref["cell_id"])
        if cell.get("cell", {}).get("cell_id") != ref["cell_id"]:
            raise SchemaError("full cell ref and artifact cell_id disagree")
        validate_full_final_output_envelope(cell, card, full_identity)
        full_cells[ref["cell_id"]] = cell

    authorization = _load_ref(sources["unseal_authorization"], base, "unseal_authorization")
    labels = _load_ref(sources["calibration_labels"], base, "calibration_labels")
    receipt = _load_ref(sources["unseal_receipt"], base, "unseal_receipt")
    sealed_hashes = [calibration_baseline["manifest_sha256"]] + [
        calibration_cells[cell_id]["manifest_sha256"]
        for cell_id in required_calibration_cell_ids(card)
    ]
    validate_unseal_authorization(
        authorization, card, calibration_identity, sealed_hashes
    )
    validate_calibration_label_sidecar(labels, card, calibration_identity, authorization)
    validate_unseal_receipt(
        receipt,
        card,
        calibration_identity,
        authorization,
        labels,
        sealed_hashes,
    )

    return {
        "request": request,
        "request_file_sha256": file_sha256(path),
        "request_path": path.resolve(),
        "analysis_control": analysis_control,
        "card": card,
        "calibration_identity": calibration_identity,
        "full_identity": full_identity,
        "calibration_baseline": calibration_baseline,
        "calibration_aggregate": calibration_aggregate,
        "b1_proof": b1_proof,
        "calibration_cells": calibration_cells,
        "full_cells": full_cells,
        "authorization": authorization,
        "labels": labels,
        "unseal_receipt": receipt,
    }


def analyze_phase2_evidence(
    evidence: Mapping[str, Any],
    *,
    execution_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Recompute H1 and NCA decisions exclusively from validated sample rows."""

    card = evidence["card"]
    validate_phase2_card(card)
    full_cells = evidence["full_cells"]
    calibration_cells = evidence["calibration_cells"]
    full_stats = card["statistics"]["full_paired_bootstrap"]
    nca_stats = card["statistics"]["nca_bootstrap"]

    def full_rows(cell_id: str) -> Sequence[Mapping[str, Any]]:
        try:
            return full_cells[cell_id]["samples"]
        except KeyError as exc:
            raise Phase2AnalysisError("missing canonical full cell %s" % cell_id) from exc

    primary_specs = _primary_contrast_specs()
    primary_raw: Dict[str, Dict[str, Any]] = {}
    for contrast_id, left_id, right_id in primary_specs:
        derived = paired_accuracy_contrast(
            full_rows(left_id),
            full_rows(right_id),
            bootstrap_replicates=full_stats["replicates"],
            bootstrap_seed=full_stats["seed"],
        )
        derived.update(
            {
                "contrast_id": contrast_id,
                "left_cell_id": left_id,
                "right_cell_id": right_id,
            }
        )
        primary_raw[contrast_id] = derived
    primary = apply_primary_holm(primary_raw)

    fixed_specs = _fixed_horizon_contrast_specs()
    fixed_raw: Dict[str, Dict[str, Any]] = {}
    for contrast_id, left_id, right_id in fixed_specs:
        derived = paired_accuracy_contrast(
            full_rows(left_id),
            full_rows(right_id),
            bootstrap_replicates=full_stats["replicates"],
            bootstrap_seed=full_stats["seed"],
        )
        derived.update(
            {
                "contrast_id": contrast_id,
                "left_cell_id": left_id,
                "right_cell_id": right_id,
            }
        )
        fixed_raw[contrast_id] = derived
    fixed_holm = holm_step_down(
        {identifier: fixed_raw[identifier]["mcnemar_raw_p"] for identifier in FIXED_HORIZON_CONTRAST_IDS},
        FIXED_HORIZON_CONTRAST_IDS,
        alpha=card["statistics"]["holm_familywise_alpha"],
    )
    fixed = {}
    for identifier in FIXED_HORIZON_CONTRAST_IDS:
        fixed[identifier] = dict(fixed_raw[identifier])
        fixed[identifier]["holm_adjusted_p"] = fixed_holm[identifier]["adjusted_p"]
        fixed[identifier]["holm_significant"] = fixed_holm[identifier]["significant"]
        fixed[identifier]["holm_rank"] = fixed_holm[identifier]["holm_rank"]

    baseline_id = protocol_cell_id("baseline_no_loop", "none", 1, 1.0)
    shared_12 = protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0)
    fs_12_k4 = protocol_cell_id("fixed_step", "12:15", 4, 2.0)
    phase1_anchor = paired_accuracy_contrast(
        full_rows(baseline_id),
        full_rows(shared_12),
        bootstrap_replicates=full_stats["replicates"],
        bootstrap_seed=full_stats["seed"],
    )
    d24 = paired_accuracy_contrast(
        full_rows(shared_12),
        full_rows(fs_12_k4),
        bootstrap_replicates=full_stats["replicates"],
        bootstrap_seed=full_stats["seed"],
    )
    cumulative_context = {
        "phase1_k2_vs_baseline_12_15": phase1_anchor,
        "fixed_step_k4_vs_k2_12_15": d24,
        "role": "secondary_sign_guard_and_context_only",
    }
    h1 = classify_h1(
        primary,
        d24_12_15_pp=d24["delta_acc_pp"],
        d24_12_15_ci_pp=d24["paired_ci_pp"],
        phase1_k2_vs_baseline_pp=phase1_anchor["delta_acc_pp"],
        decision_rules=card["decision_rules"],
    )

    mechanism = {}
    for contrast_id, left_id, right_id in primary_specs + fixed_specs:
        mechanism[contrast_id] = summarize_choice_pair(full_rows(left_id), full_rows(right_id))

    nca_primary: Dict[str, Dict[str, Any]] = {}
    nca_records_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for window in card["science"]["windows"]:
        for k, protocol, alpha in ((2, "shared_k2_anchor", 1.0), (3, "fixed_step", 1.5), (4, "fixed_step", 2.0)):
            cell_id = protocol_cell_id(protocol, window, k, alpha)
            records = _nca_records(calibration_cells[cell_id]["samples"])
            key = "%s_k%d" % (window, k)
            nca_records_by_key[key] = records
            nca_primary[key] = summarize_nca_cell(
                records,
                k=k,
                denominator=nca_stats["denominator"],
                bootstrap_replicates=nca_stats["replicates"],
                bootstrap_seed=nca_stats["seed"],
                canonical_identities=evidence["calibration_identity"]["ordered_sample_identity"],
            )
    nca_paired = {}
    for k in (2, 3, 4):
        nca_paired["k%d" % k] = paired_nca_cells(
            nca_records_by_key["12:15_k%d" % k],
            nca_records_by_key["13:16_k%d" % k],
            k=k,
            denominator=nca_stats["denominator"],
            bootstrap_replicates=nca_stats["replicates"],
            bootstrap_seed=nca_stats["seed"],
            canonical_identities=evidence["calibration_identity"]["ordered_sample_identity"],
        )
    nca_native = _native_fidelity_cells(evidence, nca_records_by_key)
    nca_decision = classify_nca(
        nca_primary, nca_paired, nca_native, decision_rules=card["decision_rules"]
    )
    nca = {
        "primary_cells": nca_primary,
        "paired_12_15_minus_13_16": nca_paired,
        "native_fidelity_cells": nca_native,
        "decision": nca_decision,
        "role": "independent_secondary_direction_diagnosis",
    }
    trajectory_summary = summarize_calibration_trajectory(calibration_cells, card)

    if execution_provenance is None:
        execution_provenance = evidence.get("execution_provenance")
    if execution_provenance is None:
        # Pure-Python analysis unit fixtures predate the routed producer but
        # cannot occur through a validated v4 analysis-input manifest.
        candidate = evidence.get("request", {}).get("analysis_provenance")
        if isinstance(candidate, Mapping) and "command_sha256" in candidate:
            execution_provenance = candidate
    _validate_analysis_provenance(execution_provenance)
    report = make_hashed_manifest(
        {
            "schema_version": PHASE2_ANALYSIS_SCHEMA_VERSION,
            "card_id": card["card_id"],
            "card_manifest_sha256": card["manifest_sha256"],
            "analysis_input_manifest_sha256": evidence["request"]["manifest_sha256"],
            "analysis_input_file_sha256": evidence["request_file_sha256"],
            "source_artifacts": _source_artifact_hashes(evidence),
            "identity_closure": {
                "calibration": _identity_closure(evidence["calibration_identity"]),
                "full": _identity_closure(evidence["full_identity"]),
            },
            "revision": card["science"]["revision"],
            "statistics_contract": dict(card["statistics"]),
            "primary_contrasts": primary,
            "fixed_horizon_contrasts": fixed,
            "cumulative_context": cumulative_context,
            "mechanism_diagnostics": mechanism,
            "calibration_trajectory_summary": trajectory_summary,
            "nca": nca,
            "h1": h1,
            "execution_provenance": dict(execution_provenance),
            "independent_decisions": True,
        }
    )
    validate_analysis_report(report, card)
    return report


def verify_phase2_analysis_report(
    input_path: Path, report_path: Path
) -> Dict[str, Any]:
    """Reload canonical sources, recompute the report, and require exact equality."""

    evidence = load_phase2_analysis_evidence(Path(input_path))
    observed = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(observed, Mapping):
        raise SchemaError("analysis report must contain a JSON object")
    validate_analysis_report(observed, evidence["card"])
    actual_provenance = derive_analysis_execution_provenance(
        evidence, Path(report_path).resolve().parent
    )
    expected = analyze_phase2_evidence(
        evidence, execution_provenance=actual_provenance
    )
    if not strict_frozen_equal(observed, expected):
        raise SchemaError(
            "analysis report differs from deterministic recomputation of canonical sources"
        )
    return expected


def choice_output(
    raw_scores: Sequence[float],
    *,
    identity: Mapping[str, Any],
    score_source: str = FULL_LM_EVAL_SCORE_SOURCE,
    labels: Sequence[str] = CHOICE_LABELS,
    gold_index: Optional[int] = None,
    evaluator_acc: Optional[bool] = None,
) -> Dict[str, Any]:
    """Derive the frozen four-choice final-output semantics from raw scores."""

    if tuple(labels) != CHOICE_LABELS or len(raw_scores) != 4:
        raise Phase2AnalysisError("choice labels/scores must be A,B,C,D in order")
    scores = [_finite(value, "raw choice score") for value in raw_scores]
    probabilities = _float32_softmax(scores)
    top1_index = max(range(4), key=lambda index: scores[index])
    ordered_scores = sorted(scores, reverse=True)
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)
    result: Dict[str, Any] = {
        "schema_version": PHASE2_FINAL_OUTPUT_SCHEMA_VERSION,
        "sample_identity": stable_sample_identity(
            identity.get("task"), identity.get("doc_id"), identity.get("doc_hash")
        ),
        "choice_labels": list(CHOICE_LABELS),
        "choice_score_source": str(score_source),
        "raw_choice_scores": scores,
        "choice_probabilities": probabilities,
        "probability_dtype": "float32",
        "entropy_nats": entropy,
        "top1_index": top1_index,
        "top1_label": CHOICE_LABELS[top1_index],
        "top_margin_raw": ordered_scores[0] - ordered_scores[1],
        "top1_tie_break": "first_maximum_A_B_C_D",
    }
    if gold_index is not None:
        if isinstance(gold_index, bool) or not isinstance(gold_index, int) or not 0 <= gold_index < 4:
            raise Phase2AnalysisError("gold_index must be an integer in [0,3]")
        correctness = top1_index == gold_index
        if evaluator_acc is not None and bool(evaluator_acc) != correctness:
            raise Phase2AnalysisError("derived top1/correctness disagrees with evaluator acc,none")
        result["gold_index"] = gold_index
        result["correct_margin_raw"] = scores[gold_index] - max(
            score for index, score in enumerate(scores) if index != gold_index
        )
        result["correctness"] = correctness
        result["evaluator_acc_none"] = correctness
    validate_scalar_sidecar(result)
    return result


def jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    lhs = _distribution(left)
    rhs = _distribution(right)
    if len(lhs) != len(rhs):
        raise Phase2AnalysisError("JS distributions must have equal length")
    middle = [(a + b) / 2.0 for a, b in zip(lhs, rhs)]
    return 0.5 * _kl(lhs, middle) + 0.5 * _kl(rhs, middle)


def compare_choice_outputs(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    validate_ordered_identity([left["sample_identity"]], [right["sample_identity"]])
    result = {
        "js_nats": jensen_shannon(left["choice_probabilities"], right["choice_probabilities"]),
        "entropy_delta_nats": float(right["entropy_nats"]) - float(left["entropy_nats"]),
        "top_margin_delta_raw": float(right["top_margin_raw"]) - float(left["top_margin_raw"]),
        "top1_retained": int(left["top1_index"]) == int(right["top1_index"]),
        "correct_margin_delta_raw": None,
    }
    if left.get("correct_margin_raw") is not None and right.get("correct_margin_raw") is not None:
        result["correct_margin_delta_raw"] = float(right["correct_margin_raw"]) - float(left["correct_margin_raw"])
    return result


def transition_counts(
    left_correct: Sequence[bool], right_correct: Sequence[bool]
) -> Dict[str, int]:
    if len(left_correct) != len(right_correct) or not left_correct:
        raise Phase2AnalysisError("paired correctness arrays must have equal positive length")
    counts = {"right_to_right": 0, "wrong_to_right": 0, "right_to_wrong": 0, "wrong_to_wrong": 0}
    for left, right in zip(left_correct, right_correct):
        if not isinstance(left, bool) or not isinstance(right, bool):
            raise Phase2AnalysisError("correctness values must be booleans")
        if left and right:
            counts["right_to_right"] += 1
        elif not left and right:
            counts["wrong_to_right"] += 1
        elif left and not right:
            counts["right_to_wrong"] += 1
        else:
            counts["wrong_to_wrong"] += 1
    return counts


def exact_mcnemar_p(wrong_to_right: int, right_to_wrong: int) -> float:
    for value in (wrong_to_right, right_to_wrong):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Phase2AnalysisError("McNemar discordant counts must be non-negative integers")
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    low = min(wrong_to_right, right_to_wrong)
    if discordant <= 1023:
        numerator = 2 * sum(math.comb(discordant, index) for index in range(low + 1))
        return min(1.0, numerator / float(1 << discordant))
    # Sum the exact binomial tail in log space.  ``comb / 2**n`` overflows or
    # underflows at the 14,042-sample scale even though the final probability
    # is representable.
    logs = []
    current = -discordant * math.log(2.0)
    logs.append(current)
    for index in range(1, low + 1):
        current += math.log(discordant - index + 1) - math.log(index)
        logs.append(current)
    peak = max(logs)
    log_tail = peak + math.log(math.fsum(math.exp(value - peak) for value in logs))
    return min(1.0, math.exp(math.log(2.0) + log_tail))


def holm_step_down(
    raw_p_by_id: Mapping[str, float],
    frozen_ids: Sequence[str] = PRIMARY_CONTRAST_IDS,
    *,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    ids = list(frozen_ids)
    if len(ids) != len(set(ids)) or set(raw_p_by_id) != set(ids):
        raise Phase2AnalysisError("Holm family must exactly match its frozen unique IDs")
    order_index = {identifier: index for index, identifier in enumerate(ids)}
    ranked = sorted(ids, key=lambda identifier: (_probability(raw_p_by_id[identifier]), order_index[identifier]))
    running = 0.0
    result: Dict[str, Dict[str, Any]] = {}
    family_size = len(ids)
    for rank, identifier in enumerate(ranked):
        raw = _probability(raw_p_by_id[identifier])
        running = max(running, min(1.0, (family_size - rank) * raw))
        result[identifier] = {
            "raw_p": raw,
            "adjusted_p": running,
            "significant": running < alpha,
            "holm_rank": rank + 1,
        }
    return {identifier: result[identifier] for identifier in ids}


def apply_primary_holm(
    primary: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    if set(primary) != set(PRIMARY_CONTRAST_IDS):
        raise Phase2AnalysisError("primary family does not match the six frozen IDs")
    holm = holm_step_down(
        {identifier: primary[identifier]["mcnemar_raw_p"] for identifier in PRIMARY_CONTRAST_IDS}
    )
    result = {}
    for identifier in PRIMARY_CONTRAST_IDS:
        result[identifier] = dict(primary[identifier])
        result[identifier]["holm_adjusted_p"] = holm[identifier]["adjusted_p"]
        result[identifier]["holm_significant"] = holm[identifier]["significant"]
        result[identifier]["holm_rank"] = holm[identifier]["holm_rank"]
    return result


def paired_accuracy_contrast(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260710,
) -> Dict[str, Any]:
    if len(left) != len(right) or not left:
        raise Phase2AnalysisError("paired analysis requires equal non-empty records")
    validate_ordered_identity(
        [record["sample_identity"] for record in left],
        [record["sample_identity"] for record in right],
    )
    lhs = [_correctness(record) for record in left]
    rhs = [_correctness(record) for record in right]
    counts = transition_counts(lhs, rhs)
    differences = [float(r) - float(l) for l, r in zip(lhs, rhs)]
    interval = paired_bootstrap_mean(
        differences, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    delta_pp = math.fsum(differences) * 100.0 / len(differences)
    net_pp = (
        counts["wrong_to_right"] - counts["right_to_wrong"]
    ) * 100.0 / len(differences)
    if not math.isclose(delta_pp, net_pp, abs_tol=1e-12, rel_tol=0.0):
        raise Phase2AnalysisError("accuracy delta and net correction disagree")
    return {
        "sample_count": len(left),
        "delta_acc_pp": delta_pp,
        "paired_ci_pp": [interval[0] * 100.0, interval[1] * 100.0],
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed, "paired_index": True},
        "transitions": counts,
        "right_to_right": counts["right_to_right"],
        "wrong_to_right": counts["wrong_to_right"],
        "right_to_wrong": counts["right_to_wrong"],
        "wrong_to_wrong": counts["wrong_to_wrong"],
        "mcnemar_raw_p": exact_mcnemar_p(counts["wrong_to_right"], counts["right_to_wrong"]),
    }


def paired_bootstrap_mean(values: Sequence[float], *, replicates: int, seed: int) -> Tuple[float, float]:
    items = [_finite(value, "bootstrap value") for value in values]
    if not items or replicates < 1:
        raise Phase2AnalysisError("bootstrap requires values and positive replicates")
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(replicates)):
        estimates.append(math.fsum(items[rng.randrange(len(items))] for _ in items) / len(items))
    return (_quantile(estimates, 0.025), _quantile(estimates, 0.975))


def bootstrap_median(values: Sequence[float], *, replicates: int = 10000, seed: int = 0) -> Dict[str, Any]:
    items = [_finite(value, "NCA value") for value in values]
    if not items or replicates < 1:
        return {"available": False, "reason": "no_valid_samples"}
    rng = random.Random(int(seed))
    estimates = [median(items[rng.randrange(len(items))] for _ in items) for _ in range(int(replicates))]
    return {
        "available": True,
        "median": median(items),
        "interval": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "replicates": int(replicates),
        "seed": int(seed),
        "pointwise": True,
        "multiplicity_adjustment": "none",
        "role": "secondary_diagnostic",
    }


def bootstrap_paired_median_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    lhs = [_finite(value, "left paired NCA") for value in left]
    rhs = [_finite(value, "right paired NCA") for value in right]
    if not lhs or len(lhs) != len(rhs) or replicates < 1:
        return {"available": False, "reason": "paired_valid_intersection_empty_or_misaligned"}
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(replicates)):
        indices = [rng.randrange(len(lhs)) for _ in lhs]
        estimates.append(median(lhs[index] for index in indices) - median(rhs[index] for index in indices))
    return {
        "available": True,
        "difference": median(lhs) - median(rhs),
        "interval": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "paired_index": True,
        "replicates": int(replicates),
        "seed": int(seed),
        "pointwise": True,
        "multiplicity_adjustment": "none",
        "role": "secondary_diagnostic",
    }


def bootstrap_stratified_median_difference(
    left_by_group: Mapping[str, Sequence[float]],
    right_by_group: Mapping[str, Sequence[float]],
    *,
    replicates: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Resample independently within each frozen subgroup, preserving sizes."""

    if set(left_by_group) != set(right_by_group) or not left_by_group:
        return {"available": False, "reason": "strata_mismatch"}
    groups = sorted(left_by_group)
    left = {key: [_finite(x, "stratified left") for x in left_by_group[key]] for key in groups}
    right = {key: [_finite(x, "stratified right") for x in right_by_group[key]] for key in groups}
    if any(not left[key] or not right[key] for key in groups) or replicates < 1:
        return {"available": False, "reason": "empty_stratum"}
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(replicates)):
        sampled_left = []
        sampled_right = []
        for key in groups:
            sampled_left.extend(left[key][rng.randrange(len(left[key]))] for _ in left[key])
            sampled_right.extend(right[key][rng.randrange(len(right[key]))] for _ in right[key])
        estimates.append(median(sampled_left) - median(sampled_right))
    flat_left = [value for key in groups for value in left[key]]
    flat_right = [value for key in groups for value in right[key]]
    return {
        "available": True,
        "difference": median(flat_left) - median(flat_right),
        "interval": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "stratified_resampling": True,
        "strata": groups,
        "replicates": int(replicates),
        "seed": int(seed),
    }


def summarize_nca_cell(
    records: Sequence[Mapping[str, Any]],
    *,
    k: int,
    denominator: int = 512,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 0,
    canonical_identities: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if len(records) != denominator:
        raise Phase2AnalysisError("NCA cell must retain the frozen denominator")
    identities = [record["sample_identity"] for record in records]
    if canonical_identities is not None:
        validate_ordered_identity(canonical_identities, identities)
    elif len({_identity_tuple(value) for value in identities}) != len(identities):
        raise Phase2AnalysisError("NCA cell identities contain duplicates")
    collapsed = [collapse_nca_sample_cell(record.get("repeated_steps", []), k) for record in records]
    values = [item["value"] for item in collapsed if item["valid"]]
    return {
        "denominator": denominator,
        "valid_count": len(values),
        "valid_fraction": float(len(values)) / denominator,
        "overall_interval": bootstrap_median(values, replicates=bootstrap_replicates, seed=bootstrap_seed),
    }


def paired_nca_cells(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    k: int,
    denominator: int = 512,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 0,
    canonical_identities: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if len(left) != denominator or len(right) != denominator:
        raise Phase2AnalysisError("paired NCA cells must retain the frozen denominator")
    validate_ordered_identity(
        [record["sample_identity"] for record in left],
        [record["sample_identity"] for record in right],
    )
    if canonical_identities is not None:
        validate_ordered_identity(
            canonical_identities, [record["sample_identity"] for record in left]
        )
    left_values = []
    right_values = []
    for lhs, rhs in zip(left, right):
        lvalue = collapse_nca_sample_cell(lhs.get("repeated_steps", []), k)
        rvalue = collapse_nca_sample_cell(rhs.get("repeated_steps", []), k)
        if lvalue["valid"] and rvalue["valid"]:
            left_values.append(lvalue["value"])
            right_values.append(rvalue["value"])
    return {
        "denominator": denominator,
        "paired_valid_count": len(left_values),
        "paired_valid_fraction": float(len(left_values)) / denominator,
        "difference_interval": bootstrap_paired_median_difference(
            left_values,
            right_values,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }


def collapse_nca_sample_cell(steps: Sequence[Mapping[str, Any]], k: int) -> Dict[str, Any]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise Phase2AnalysisError("K must be positive")
    if k == 1:
        return {"valid": False, "value": None, "reason": "K1_has_no_repeated_step_NCA"}
    expected = list(range(1, k))
    actual = [step.get("body_call_t", -1) for step in steps]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in actual):
        return {"valid": False, "value": None, "reason": "repeated_step_index_mismatch"}
    if actual != expected:
        return {"valid": False, "value": None, "reason": "repeated_step_index_mismatch"}
    if not all(step.get("valid") is True and step.get("value") is not None for step in steps):
        return {"valid": False, "value": None, "reason": "not_all_repeated_steps_valid"}
    return {"valid": True, "value": median(_finite(step["value"], "step NCA") for step in steps), "reason": None}


def classify_h1(
    primary: Mapping[str, Mapping[str, Any]],
    *,
    d24_12_15_pp: float,
    d24_12_15_ci_pp: Sequence[float],
    phase1_k2_vs_baseline_pp: float,
    decision_rules: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    if set(primary) != set(PRIMARY_CONTRAST_IDS) or len(d24_12_15_ci_pp) != 2:
        raise Phase2AnalysisError("H1 classification requires the complete frozen family")
    for identifier in PRIMARY_CONTRAST_IDS:
        _require_contrast(primary[identifier])
    rules = _validated_decision_rules(decision_rules)
    h1_rules = rules["h1"]
    significance_threshold = float(h1_rules["significance"]["threshold"])
    perturbation_rules = h1_rules["perturbation"]
    refinement_rules = h1_rules["refinement_supported"]
    transient_rules = h1_rules["transient_only"]
    suggestive_rules = h1_rules["suggestive"]
    saturation_rules = h1_rules["saturation"]
    c23 = primary[refinement_rules["target_contrast_id"]]
    c34 = primary[saturation_rules["target_contrast_id"]]
    perturbation = any(
        float(cell["delta_acc_pp"]) < float(perturbation_rules["delta_threshold_pp"])
        and float(cell["holm_adjusted_p"]) < significance_threshold
        and int(cell["right_to_wrong"]) > int(cell["wrong_to_right"])
        for cell in (
            primary[identifier]
            for identifier in perturbation_rules["target_contrast_ids"]
        )
    )
    if perturbation:
        label = "PERTURBATION"
    elif (
        float(c23["delta_acc_pp"]) > float(refinement_rules["delta_threshold_pp"])
        and float(c23["holm_adjusted_p"]) < significance_threshold
        and d24_12_15_pp >= float(refinement_rules["d24_threshold_pp"])
    ):
        label = "REFINEMENT_SUPPORTED"
    elif (
        phase1_k2_vs_baseline_pp
        > float(transient_rules["phase1_k2_vs_baseline_threshold_pp"])
        and float(d24_12_15_ci_pp[1])
        < float(transient_rules["d24_endpoint_threshold_pp"])
    ):
        label = "TRANSIENT_ONLY"
    elif (
        float(c23["delta_acc_pp"]) > float(suggestive_rules["delta_threshold_pp"])
        and d24_12_15_pp >= float(suggestive_rules["d24_threshold_pp"])
        and float(c23["holm_adjusted_p"]) >= significance_threshold
    ):
        label = "SUGGESTIVE"
    else:
        label = "INCONCLUSIVE"
    saturation = (
        saturation_rules["positive_label"]
        if float(c34["delta_acc_pp"]) > float(saturation_rules["delta_threshold_pp"])
        and float(c34["holm_adjusted_p"]) < significance_threshold
        else saturation_rules["otherwise_label"]
    )
    return {"h1_outcome": label, "k4_saturation": saturation}


def classify_nca(
    primary_cells: Mapping[str, Mapping[str, Any]],
    paired_cells: Mapping[str, Mapping[str, Any]],
    native_cells: Mapping[str, Mapping[str, Any]],
    decision_rules: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    rules = _validated_decision_rules(decision_rules)["nca"]
    if len(primary_cells) != rules["primary_cell_count"] or len(paired_cells) != rules["paired_cell_count"]:
        raise Phase2AnalysisError("NCA diagnosis requires 9 primary and 3 paired cells")
    valid_threshold = float(rules["valid_fraction_threshold"])
    incomplete = any(
        float(cell.get("valid_fraction", -1.0)) < valid_threshold
        or not _available_interval(cell.get("overall_interval"))
        for cell in primary_cells.values()
    ) or any(
        float(cell.get("paired_valid_fraction", -1.0)) < valid_threshold
        or not _available_interval(cell.get("difference_interval"))
        for cell in paired_cells.values()
    )
    if incomplete:
        return {"nca_diagnosis": "NCA_INCONCLUSIVE", "native_fidelity_check": "NOT_EVALUABLE"}
    targets = [
        primary_cells["%s_k%d" % (rules["target_window"], k)]
        for k in rules["primary_k"]
    ]
    all_target_positive = sum(
        float(cell["overall_interval"]["interval"][0])
        > float(rules["target_positive_threshold"])
        for cell in targets
    ) == int(rules["target_positive_required_count"])
    eligible = []
    native_rules = rules["native_fidelity"]
    for key in ("%s_k%d" % (rules["target_window"], k) for k in rules["primary_k"]):
        cell = native_cells.get(key, {})
        counts = cell.get("valid_subgroup_counts", {})
        if all(
            int(counts.get(group, 0)) >= int(native_rules["min_valid_per_subgroup"])
            for group in native_rules["required_subgroups"]
        ):
            eligible.append(cell)
    native_supported = sum(
        _available_interval(cell.get("right_to_right_interval"))
        and float(cell["right_to_right_interval"]["interval"][0])
        > float(native_rules["right_to_right_threshold"])
        and _available_interval(cell.get("wrong_to_right_minus_wrong_to_wrong_interval"))
        and float(cell["wrong_to_right_minus_wrong_to_wrong_interval"]["interval"][1])
        <= float(native_rules["wrong_to_right_minus_wrong_to_wrong_threshold"])
        for cell in eligible
    )
    if (
        all_target_positive
        and len(eligible) >= int(native_rules["eligible_cell_required_count"])
        and native_supported >= int(rules["native_supported_required_count"])
    ):
        diagnosis = "NCA_NATIVE_FIDELITY_ONLY"
    else:
        positive_differences = sum(
            float(paired_cells["k%d" % k]["difference_interval"]["interval"][0])
            > float(rules["paired_positive_threshold"])
            for k in rules["primary_k"]
        )
        diagnosis = (
            "NCA_DIRECTION_SUPPORTED"
            if all_target_positive
            and positive_differences >= int(rules["direction_positive_required_count"])
            else "NCA_NONDISCRIMINATIVE"
        )
    native_check = "NOT_EVALUABLE" if len(eligible) < int(native_rules["eligible_cell_required_count"]) else (
        "SUPPORTED"
        if native_supported >= int(rules["native_supported_required_count"])
        else "EVALUATED_NOT_SUPPORTED"
    )
    return {"nca_diagnosis": diagnosis, "native_fidelity_check": native_check}


def wrong_overconfidence(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Report the diagnostic over final-wrong pairs (WW + right-to-wrong)."""

    if len(left) != len(right):
        raise Phase2AnalysisError("overconfidence inputs must be paired")
    denominator = 0
    numerator = 0
    subgroup = {"wrong_to_wrong": [0, 0], "right_to_wrong": [0, 0]}
    for before, after in zip(left, right):
        before_correct = _correctness(before)
        after_correct = _correctness(after)
        if after_correct:
            continue
        name = "right_to_wrong" if before_correct else "wrong_to_wrong"
        denominator += 1
        subgroup[name][1] += 1
        event = (
            float(after["entropy_nats"]) < float(before["entropy_nats"])
            and float(after["top_margin_raw"]) > float(before["top_margin_raw"])
        )
        if event:
            numerator += 1
            subgroup[name][0] += 1
    return {
        "definition": "final_wrong_pairs_with_entropy_down_and_top_margin_up",
        "denominator_scope": "wrong_to_wrong_plus_right_to_wrong",
        "count": numerator,
        "denominator": denominator,
        "fraction": (float(numerator) / denominator if denominator else None),
        "subgroups": {
            key: {"count": value[0], "denominator": value[1], "fraction": (float(value[0]) / value[1] if value[1] else None)}
            for key, value in subgroup.items()
        },
        "role": "mechanism_diagnostic_not_H1_decision_rule",
    }


def summarize_choice_pair(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    if len(left) != len(right) or not left:
        raise Phase2AnalysisError("choice-pair diagnostics require equal non-empty records")
    validate_ordered_identity(
        [row["sample_identity"] for row in left],
        [row["sample_identity"] for row in right],
    )
    comparisons = [compare_choice_outputs(before, after) for before, after in zip(left, right)]
    transitions = transition_counts(
        [_correctness(row) for row in left], [_correctness(row) for row in right]
    )
    return {
        "sample_count": len(left),
        "mean_js_nats": math.fsum(row["js_nats"] for row in comparisons) / len(comparisons),
        "mean_entropy_delta_nats": math.fsum(row["entropy_delta_nats"] for row in comparisons) / len(comparisons),
        "mean_top_margin_delta_raw": math.fsum(row["top_margin_delta_raw"] for row in comparisons) / len(comparisons),
        "mean_correct_margin_delta_raw": math.fsum(
            row["correct_margin_delta_raw"] for row in comparisons
        ) / len(comparisons),
        "top1_retained_fraction": math.fsum(1.0 if row["top1_retained"] else 0.0 for row in comparisons) / len(comparisons),
        "transitions": transitions,
        "wrong_overconfidence": wrong_overconfidence(left, right),
    }


def summarize_calibration_trajectory(
    calibration_cells: Mapping[str, Mapping[str, Any]], card: Mapping[str, Any]
) -> Dict[str, Any]:
    """Recompute scalar residual/q/direction summaries for all 15 B2 cells."""

    validate_phase2_card(card)
    expected_cells = list(b2_logical_cells(card))
    if set(calibration_cells) != {cell["cell_id"] for cell in expected_cells}:
        raise Phase2AnalysisError("calibration trajectory matrix is missing or has rogue cells")
    result: Dict[str, Any] = {}
    for expected in expected_cells:
        samples = calibration_cells[expected["cell_id"]]["samples"]
        if len(samples) != 512:
            raise Phase2AnalysisError("calibration trajectory cell must retain 512 samples")
        valid_sample_count = sum(sample.get("valid") is True for sample in samples)
        steps = []
        for body_call_t in range(expected["k"]):
            rows = [sample["steps"][body_call_t] for sample in samples]
            indices = [row.get("body_call_t", -1) for row in rows]
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value != body_call_t
                for value in indices
            ):
                raise Phase2AnalysisError("trajectory step index differs from cell K")
            residual_norm = [_finite(row["residual_norm"], "residual norm") for row in rows]
            state_norm = [_finite(row["state_norm"], "state norm") for row in rows]
            activity = [_finite(row["relative_activity"], "relative activity") for row in rows]
            if any(value < 0.0 for value in residual_norm + state_norm + activity):
                raise Phase2AnalysisError("trajectory norms/activity must be non-negative")
            if body_call_t == 0:
                ratio_median = None
                cosine_median = None
                cosine_valid_count = 0
            else:
                ratios = [
                    _finite(row["residual_ratio_to_previous"], "residual ratio") for row in rows
                ]
                if any(value < 0.0 for value in ratios):
                    raise Phase2AnalysisError("residual ratios must be non-negative")
                ratio_median = median(ratios)
                cosines = [
                    _finite(row["adjacent_residual_cosine"], "adjacent cosine")
                    for row in rows
                    if row.get("adjacent_residual_cosine_valid") is True
                ]
                if any(value < -1.0 or value > 1.0 for value in cosines):
                    raise Phase2AnalysisError("adjacent cosine must be in [-1,1]")
                cosine_median = median(cosines) if cosines else None
                cosine_valid_count = len(cosines)
            steps.append(
                {
                    "body_call_t": body_call_t,
                    "residual_norm_median": median(residual_norm),
                    "state_norm_median": median(state_norm),
                    "relative_activity_median": median(activity),
                    "residual_ratio_to_previous_median": ratio_median,
                    "adjacent_residual_cosine_median": cosine_median,
                    "adjacent_residual_cosine_valid_count": cosine_valid_count,
                }
            )
        result[expected["cell_id"]] = {
            "cell": dict(expected),
            "sample_count": 512,
            "valid_sample_count": valid_sample_count,
            "valid_sample_fraction": valid_sample_count / 512.0,
            "steps": steps,
            "role": "calibration_512_mechanism_evidence_only",
        }
    return result


def bootstrap_independent_median_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    lhs = [_finite(value, "left subgroup NCA") for value in left]
    rhs = [_finite(value, "right subgroup NCA") for value in right]
    if not lhs or not rhs or replicates < 1:
        return {"available": False, "reason": "empty_frozen_subgroup"}
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(replicates)):
        estimates.append(
            median(lhs[rng.randrange(len(lhs))] for _ in lhs)
            - median(rhs[rng.randrange(len(rhs))] for _ in rhs)
        )
    return {
        "available": True,
        "difference": median(lhs) - median(rhs),
        "interval": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "stratified_within_frozen_subgroup": True,
        "left_count": len(lhs),
        "right_count": len(rhs),
        "replicates": int(replicates),
        "seed": int(seed),
        "pointwise": True,
        "multiplicity_adjustment": "none",
        "role": "secondary_diagnostic",
    }


def _primary_contrast_specs() -> List[Tuple[str, str, str]]:
    result = []
    for window, stem in (("11:14", "11_14"), ("12:15", "12_15"), ("13:16", "13_16")):
        k2 = protocol_cell_id("shared_k2_anchor", window, 2, 1.0)
        k3 = protocol_cell_id("fixed_step", window, 3, 1.5)
        k4 = protocol_cell_id("fixed_step", window, 4, 2.0)
        result.extend(
            (
                ("fs_%s_k2_k3" % stem, k2, k3),
                ("fs_%s_k3_k4" % stem, k3, k4),
            )
        )
    if [item[0] for item in result] != list(PRIMARY_CONTRAST_IDS):
        raise Phase2AnalysisError("primary contrast spec order drifted")
    return result


def _fixed_horizon_contrast_specs() -> List[Tuple[str, str, str]]:
    k2 = protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0)
    k3 = protocol_cell_id("fixed_horizon", "12:15", 3, 1.0)
    k4 = protocol_cell_id("fixed_horizon", "12:15", 4, 1.0)
    result = [
        ("fh_12_15_k2_k3", k2, k3),
        ("fh_12_15_k3_k4", k3, k4),
    ]
    if [item[0] for item in result] != list(FIXED_HORIZON_CONTRAST_IDS):
        raise Phase2AnalysisError("fixed-horizon contrast spec order drifted")
    return result


def _validated_decision_rules(
    value: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    expected = frozen_decision_rules()
    observed = expected if value is None else value
    if not isinstance(observed, Mapping) or not strict_frozen_equal(observed, expected):
        raise Phase2AnalysisError("decision rules differ from the versioned frozen contract")
    return observed


def _nca_records(samples: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for sample in samples:
        repeated = []
        for step in sample["steps"][1:]:
            nca = step["nca"]
            repeated.append(
                {
                    "body_call_t": nca["body_call_t"],
                    "valid": nca["valid"],
                    "value": nca["value"],
                }
            )
        result.append(
            {
                "sample_identity": sample["sample_identity"],
                "repeated_steps": repeated,
            }
        )
    return result


def _native_fidelity_cells(
    evidence: Mapping[str, Any], nca_records_by_key: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    card = evidence["card"]
    stats = card["statistics"]["nca_bootstrap"]
    baseline = evidence["calibration_baseline"]["samples"]
    labels = evidence["labels"]["samples"]
    result = {}
    for k, protocol, alpha in ((2, "shared_k2_anchor", 1.0), (3, "fixed_step", 1.5), (4, "fixed_step", 2.0)):
        key = "12:15_k%d" % k
        cell_id = protocol_cell_id(protocol, "12:15", k, alpha)
        final_rows = evidence["calibration_cells"][cell_id]["samples"]
        nca_rows = nca_records_by_key[key]
        groups: Dict[str, List[float]] = {
            "right_to_right": [],
            "wrong_to_right": [],
            "right_to_wrong": [],
            "wrong_to_wrong": [],
        }
        valid_counts = {name: 0 for name in groups}
        for base_row, final_row, label_row, nca_row in zip(baseline, final_rows, labels, nca_rows):
            if not (
                base_row["sample_identity"]
                == final_row["sample_identity"]
                == label_row["sample_identity"]
                == nca_row["sample_identity"]
            ):
                raise Phase2AnalysisError("native-fidelity sample identities are misaligned")
            gold = int(label_row["gold_index"])
            before = int(base_row["final_output"]["top1_index"]) == gold
            after = int(final_row["final_output"]["top1_index"]) == gold
            if before and after:
                group = "right_to_right"
            elif not before and after:
                group = "wrong_to_right"
            elif before and not after:
                group = "right_to_wrong"
            else:
                group = "wrong_to_wrong"
            collapsed = collapse_nca_sample_cell(nca_row["repeated_steps"], k)
            if collapsed["valid"]:
                groups[group].append(collapsed["value"])
                valid_counts[group] += 1
        result[key] = {
            "valid_subgroup_counts": valid_counts,
            "right_to_right_interval": bootstrap_median(
                groups["right_to_right"], replicates=stats["replicates"], seed=stats["seed"]
            ),
            "wrong_to_right_minus_wrong_to_wrong_interval": bootstrap_independent_median_difference(
                groups["wrong_to_right"],
                groups["wrong_to_wrong"],
                replicates=stats["replicates"],
                seed=stats["seed"],
            ),
            "resampling": "stratified_within_frozen_subgroup",
        }
    return result


def _identity_closure(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "identity_namespace": manifest["identity_namespace"],
        "identity_manifest_sha256": manifest["manifest_sha256"],
        "ordered_identity_sha256": manifest["ordered_identity_sha256"],
        "sample_count": manifest["sample_count"],
        "natural_order": manifest["natural_order"],
        "source_provenance": dict(manifest["source_provenance"]),
    }


def _source_artifact_hashes(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    card = evidence["card"]
    return {
        "card": card["manifest_sha256"],
        "calibration_identity_manifest": evidence["calibration_identity"]["manifest_sha256"],
        "full_identity_manifest": evidence["full_identity"]["manifest_sha256"],
        "calibration_baseline": evidence["calibration_baseline"]["manifest_sha256"],
        "calibration_aggregate": evidence["calibration_aggregate"]["manifest_sha256"],
        "calibration_cells": {
            cell_id: evidence["calibration_cells"][cell_id]["manifest_sha256"]
            for cell_id in required_calibration_cell_ids(card)
        },
        "full_final_output_cells": {
            cell_id: evidence["full_cells"][cell_id]["manifest_sha256"]
            for cell_id in required_full_cell_ids(card)
        },
        "calibration_labels": evidence["labels"]["manifest_sha256"],
        "unseal_authorization": evidence["authorization"]["manifest_sha256"],
        "unseal_receipt": evidence["unseal_receipt"]["manifest_sha256"],
    }


def _identity_tuple(value: Mapping[str, Any]) -> Tuple[str, str, str]:
    identity = stable_sample_identity(value.get("task"), value.get("doc_id"), value.get("doc_hash"))
    return identity["task"], identity["doc_id"], identity["doc_hash"]


def _float32_softmax(scores: Sequence[float]) -> List[float]:
    rounded = [_f32(value) for value in scores]
    peak = max(rounded)
    weights = [_f32(math.exp(_f32(value - peak))) for value in rounded]
    total = _f32(math.fsum(weights))
    if total <= 0.0:
        raise Phase2AnalysisError("softmax mass is not positive")
    return [_f32(value / total) for value in weights]


def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", _finite(value, "float32 value")))[0]


def _distribution(values: Sequence[float]) -> List[float]:
    items = [_finite(value, "probability") for value in values]
    if not items or any(value < 0.0 for value in items):
        raise Phase2AnalysisError("probabilities must be finite and non-negative")
    total = math.fsum(items)
    if total <= 0.0:
        raise Phase2AnalysisError("probability mass must be positive")
    return [value / total for value in items]


def _kl(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * math.log(a / b) for a, b in zip(left, right) if a > 0.0)


def _correctness(record: Mapping[str, Any]) -> bool:
    value = record.get("correctness")
    if not isinstance(value, bool):
        raise Phase2AnalysisError("paired record is missing boolean correctness")
    return value


def _require_contrast(cell: Mapping[str, Any]) -> None:
    required = ("delta_acc_pp", "holm_adjusted_p", "wrong_to_right", "right_to_wrong")
    if any(key not in cell for key in required):
        raise Phase2AnalysisError("primary contrast is incomplete")
    _finite(cell["delta_acc_pp"], "contrast delta")
    _probability(cell["holm_adjusted_p"])
    for key in ("wrong_to_right", "right_to_wrong"):
        if isinstance(cell[key], bool) or not isinstance(cell[key], int) or cell[key] < 0:
            raise Phase2AnalysisError("contrast transition count is invalid")


def _available_interval(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("available") is not True:
        return False
    interval = value.get("interval")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2 or not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        for item in interval
    ):
        raise Phase2AnalysisError("available interval must contain two finite numeric endpoints")
    if float(interval[0]) > float(interval[1]):
        raise Phase2AnalysisError("available interval endpoints are reversed")
    return True


def _analysis_exact_keys(payload: Any, expected: Iterable[str], context: str) -> None:
    if not isinstance(payload, Mapping):
        raise SchemaError("%s must be an object" % context)
    actual = set(payload)
    wanted = set(expected)
    if actual != wanted:
        raise SchemaError(
            "%s exact-key mismatch; missing=%s unknown=%s"
            % (context, sorted(wanted - actual), sorted(actual - wanted))
        )


def _analysis_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError("%s must be lowercase SHA256" % context)
    return value


def _validate_source_ref(value: Any, context: str) -> None:
    _analysis_exact_keys(value, {"path", "sha256"}, context)
    if not isinstance(value["path"], str) or not value["path"].strip():
        raise SchemaError("%s.path must be non-empty" % context)
    _analysis_sha256(value["sha256"], "%s.sha256" % context)


def _validate_cell_refs(value: Any, expected_ids: Sequence[str], context: str) -> None:
    if not isinstance(value, list):
        raise SchemaError("%s must be a list" % context)
    observed = []
    for index, item in enumerate(value):
        _analysis_exact_keys(item, {"cell_id", "path", "sha256"}, "%s[%d]" % (context, index))
        if not isinstance(item["cell_id"], str) or not item["cell_id"]:
            raise SchemaError("%s[%d].cell_id must be non-empty" % (context, index))
        _validate_source_ref(
            {"path": item["path"], "sha256": item["sha256"]}, "%s[%d]" % (context, index)
        )
        observed.append(item["cell_id"])
    if observed != list(expected_ids) or len(set(observed)) != len(observed):
        raise SchemaError("%s must exactly match the frozen ordered cell set" % context)


def _load_ref(ref: Mapping[str, Any], base: Path, context: str) -> Mapping[str, Any]:
    _validate_source_ref(ref, context)
    path = _source_ref_path(ref, base)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SchemaError("%s source must contain a JSON object" % context)
    from tflt.loopscope.schema import verify_manifest_sha256

    verify_manifest_sha256(payload)
    if payload.get("manifest_sha256") != ref["sha256"]:
        raise SchemaError("%s source hash differs from the analysis request" % context)
    return payload


def _source_ref_path(ref: Mapping[str, Any], base: Path) -> Path:
    path = Path(ref["path"])
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _require_same_source_ref(
    internal: Mapping[str, Any],
    external: Mapping[str, Any],
    internal_base: Path,
    external_base: Path,
    context: str,
    *,
    ignore_cell_id: bool = False,
) -> None:
    internal_ref = dict(internal)
    external_ref = dict(external)
    if ignore_cell_id:
        internal_ref.pop("cell_id", None)
        external_ref.pop("cell_id", None)
    _validate_source_ref(internal_ref, "%s internal ref" % context)
    _validate_source_ref(external_ref, "%s external ref" % context)
    if (
        internal_ref["sha256"] != external_ref["sha256"]
        or _source_ref_path(internal_ref, internal_base)
        != _source_ref_path(external_ref, external_base)
    ):
        raise SchemaError("%s path/hash differs between aggregate and analysis input" % context)


def validate_analysis_report(report: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    """Validate the complete derived report and recompute all decision pivots."""

    validate_phase2_card(card)
    _analysis_exact_keys(
        report,
        {
            "schema_version", "card_id", "card_manifest_sha256",
            "analysis_input_manifest_sha256", "analysis_input_file_sha256",
            "source_artifacts", "identity_closure", "revision", "statistics_contract",
            "primary_contrasts", "fixed_horizon_contrasts", "cumulative_context",
            "mechanism_diagnostics", "calibration_trajectory_summary", "nca", "h1", "execution_provenance",
            "independent_decisions", "manifest_sha256",
        },
        "analysis report",
    )
    from tflt.loopscope.schema import verify_manifest_sha256

    verify_manifest_sha256(report)
    if report["schema_version"] != PHASE2_ANALYSIS_SCHEMA_VERSION:
        raise SchemaError("unsupported Phase 2 analysis-report schema")
    if report["card_id"] != card["card_id"] or report["card_manifest_sha256"] != card["manifest_sha256"]:
        raise SchemaError("analysis report is bound to a different card")
    if report["revision"] != card["science"]["revision"]:
        raise SchemaError("analysis report revision differs from the card")
    if not strict_frozen_equal(report["statistics_contract"], card["statistics"]):
        raise SchemaError("analysis report statistics differ from the card")
    _analysis_sha256(report["analysis_input_manifest_sha256"], "analysis_input_manifest_sha256")
    _analysis_sha256(report["analysis_input_file_sha256"], "analysis_input_file_sha256")
    _validate_source_hash_tree(report["source_artifacts"], card)
    _validate_identity_closure(report["identity_closure"], card)

    primary = report["primary_contrasts"]
    fixed = report["fixed_horizon_contrasts"]
    _validate_contrast_family(primary, PRIMARY_CONTRAST_IDS, card)
    _validate_contrast_family(fixed, FIXED_HORIZON_CONTRAST_IDS, card)
    expected_primary_holm = holm_step_down(
        {identifier: primary[identifier]["mcnemar_raw_p"] for identifier in PRIMARY_CONTRAST_IDS},
        PRIMARY_CONTRAST_IDS,
        alpha=card["statistics"]["holm_familywise_alpha"],
    )
    expected_fixed_holm = holm_step_down(
        {identifier: fixed[identifier]["mcnemar_raw_p"] for identifier in FIXED_HORIZON_CONTRAST_IDS},
        FIXED_HORIZON_CONTRAST_IDS,
        alpha=card["statistics"]["holm_familywise_alpha"],
    )
    for family, expected in ((primary, expected_primary_holm), (fixed, expected_fixed_holm)):
        for identifier, cell in family.items():
            _assert_close(cell["holm_adjusted_p"], expected[identifier]["adjusted_p"], "Holm adjusted p")
            if cell["holm_significant"] is not expected[identifier]["significant"] or cell["holm_rank"] != expected[identifier]["holm_rank"]:
                raise SchemaError("Holm decision/rank differs from recomputed family")

    cumulative = report["cumulative_context"]
    _analysis_exact_keys(
        cumulative,
        {"phase1_k2_vs_baseline_12_15", "fixed_step_k4_vs_k2_12_15", "role"},
        "cumulative context",
    )
    if cumulative["role"] != "secondary_sign_guard_and_context_only":
        raise SchemaError("cumulative context role differs")
    for key in ("phase1_k2_vs_baseline_12_15", "fixed_step_k4_vs_k2_12_15"):
        _validate_contrast(cumulative[key], card, augmented=False)
    expected_h1 = classify_h1(
        primary,
        d24_12_15_pp=cumulative["fixed_step_k4_vs_k2_12_15"]["delta_acc_pp"],
        d24_12_15_ci_pp=cumulative["fixed_step_k4_vs_k2_12_15"]["paired_ci_pp"],
        phase1_k2_vs_baseline_pp=cumulative["phase1_k2_vs_baseline_12_15"]["delta_acc_pp"],
        decision_rules=card["decision_rules"],
    )
    if report["h1"] != expected_h1:
        raise SchemaError("H1 label differs from recomputed frozen priority rules")

    mechanism = report["mechanism_diagnostics"]
    expected_mechanism_ids = list(PRIMARY_CONTRAST_IDS) + list(FIXED_HORIZON_CONTRAST_IDS)
    if not isinstance(mechanism, Mapping) or set(mechanism) != set(expected_mechanism_ids):
        raise SchemaError("mechanism diagnostics do not match the frozen contrast order")
    for value in mechanism.values():
        _validate_mechanism_diagnostic(value, card)
    for identifier in expected_mechanism_ids:
        contrast = primary.get(identifier) or fixed.get(identifier)
        if mechanism[identifier]["transitions"] != contrast["transitions"]:
            raise SchemaError("mechanism transition table differs from its paired contrast")
    _validate_trajectory_summary(report["calibration_trajectory_summary"], card)

    nca = report["nca"]
    _analysis_exact_keys(
        nca,
        {"primary_cells", "paired_12_15_minus_13_16", "native_fidelity_cells", "decision", "role"},
        "NCA report",
    )
    if nca["role"] != "independent_secondary_direction_diagnosis":
        raise SchemaError("NCA report role differs")
    _validate_nca_report(nca, card)
    expected_nca = classify_nca(
        nca["primary_cells"],
        nca["paired_12_15_minus_13_16"],
        nca["native_fidelity_cells"],
        decision_rules=card["decision_rules"],
    )
    if nca["decision"] != expected_nca:
        raise SchemaError("NCA label differs from recomputed frozen rules")
    _validate_analysis_provenance(report["execution_provenance"])
    if report["independent_decisions"] is not True:
        raise SchemaError("H1 and NCA decisions must remain independent")


def _validate_contrast_family(
    family: Any, expected_ids: Sequence[str], card: Mapping[str, Any]
) -> None:
    if not isinstance(family, Mapping) or set(family) != set(expected_ids):
        raise SchemaError("contrast family IDs/order differ from the frozen contract")
    specs = {
        identifier: (left_id, right_id)
        for identifier, left_id, right_id in (
            _primary_contrast_specs()
            if tuple(expected_ids) == tuple(PRIMARY_CONTRAST_IDS)
            else _fixed_horizon_contrast_specs()
        )
    }
    for identifier in expected_ids:
        cell = family[identifier]
        _validate_contrast(cell, card, augmented=True)
        if cell["contrast_id"] != identifier:
            raise SchemaError("contrast_id differs from its family key")
        if (cell["left_cell_id"], cell["right_cell_id"]) != specs[identifier]:
            raise SchemaError("contrast source cells differ from the frozen family spec")


def _validate_contrast(cell: Any, card: Mapping[str, Any], *, augmented: bool) -> None:
    base_keys = {
        "sample_count", "delta_acc_pp", "paired_ci_pp", "bootstrap", "transitions",
        "right_to_right", "wrong_to_right", "right_to_wrong", "wrong_to_wrong",
        "mcnemar_raw_p",
    }
    keys = base_keys | (
        {"contrast_id", "left_cell_id", "right_cell_id", "holm_adjusted_p", "holm_significant", "holm_rank"}
        if augmented
        else set()
    )
    _analysis_exact_keys(cell, keys, "paired contrast")
    count = cell["sample_count"]
    if isinstance(count, bool) or count != 14042:
        raise SchemaError("full contrast must retain 14,042 samples")
    transitions = cell["transitions"]
    _analysis_exact_keys(
        transitions,
        {"right_to_right", "wrong_to_right", "right_to_wrong", "wrong_to_wrong"},
        "contrast transitions",
    )
    for key, value in transitions.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError("transition count must be a non-negative integer")
        if cell[key] != value:
            raise SchemaError("flat transition count differs from transition table")
    if sum(transitions.values()) != count:
        raise SchemaError("transition counts do not sum to the full denominator")
    expected_delta = (transitions["wrong_to_right"] - transitions["right_to_wrong"]) * 100.0 / count
    _assert_close(cell["delta_acc_pp"], expected_delta, "contrast delta")
    expected_p = exact_mcnemar_p(transitions["wrong_to_right"], transitions["right_to_wrong"])
    _assert_close(cell["mcnemar_raw_p"], expected_p, "McNemar p", abs_tol=1e-14)
    _validate_interval(cell["paired_ci_pp"], "paired_ci_pp")
    if (
        float(cell["paired_ci_pp"][0]) < -100.0
        or float(cell["paired_ci_pp"][1]) > 100.0
    ):
        raise SchemaError("paired accuracy interval must remain within [-100,100] pp")
    bootstrap = cell["bootstrap"]
    _analysis_exact_keys(bootstrap, {"replicates", "seed", "paired_index"}, "full bootstrap")
    expected_bootstrap = card["statistics"]["full_paired_bootstrap"]
    if not strict_frozen_equal(bootstrap, {
        "replicates": expected_bootstrap["replicates"],
        "seed": expected_bootstrap["seed"],
        "paired_index": True,
    }):
        raise SchemaError("full bootstrap seed/replicates differ from the card")
    if augmented:
        for key in ("contrast_id", "left_cell_id", "right_cell_id"):
            if not isinstance(cell[key], str) or not cell[key]:
                raise SchemaError("augmented contrast identity must be non-empty")
        _probability(cell["holm_adjusted_p"])
        if not isinstance(cell["holm_significant"], bool):
            raise SchemaError("Holm significance must be boolean")
        if isinstance(cell["holm_rank"], bool) or not isinstance(cell["holm_rank"], int):
            raise SchemaError("Holm rank must be integer")


def _validate_mechanism_diagnostic(value: Any, card: Mapping[str, Any]) -> None:
    _analysis_exact_keys(
        value,
        {
            "sample_count", "mean_js_nats", "mean_entropy_delta_nats",
            "mean_top_margin_delta_raw", "mean_correct_margin_delta_raw",
            "top1_retained_fraction", "transitions", "wrong_overconfidence",
        },
        "mechanism diagnostic",
    )
    if value["sample_count"] != 14042:
        raise SchemaError("mechanism diagnostic denominator differs")
    for key in (
        "mean_js_nats", "mean_entropy_delta_nats", "mean_top_margin_delta_raw",
        "mean_correct_margin_delta_raw", "top1_retained_fraction",
    ):
        _finite(value[key], key)
    if not 0.0 <= float(value["mean_js_nats"]) <= math.log(2.0):
        raise SchemaError("mean JS divergence must be in [0,ln(2)]")
    if not 0.0 <= float(value["top1_retained_fraction"]) <= 1.0:
        raise SchemaError("top1 retention must be in [0,1]")
    transitions = value["transitions"]
    _analysis_exact_keys(
        transitions,
        {"right_to_right", "wrong_to_right", "right_to_wrong", "wrong_to_wrong"},
        "mechanism transitions",
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in transitions.values()
    ) or sum(transitions.values()) != 14042:
        raise SchemaError("mechanism transition denominator differs")
    overconfidence = value["wrong_overconfidence"]
    _analysis_exact_keys(
        overconfidence,
        {"definition", "denominator_scope", "count", "denominator", "fraction", "subgroups", "role"},
        "wrong-overconfidence diagnostic",
    )
    if overconfidence.get("role") != "mechanism_diagnostic_not_H1_decision_rule":
        raise SchemaError("wrong-overconfidence role differs")
    if overconfidence["definition"] != "final_wrong_pairs_with_entropy_down_and_top_margin_up" or overconfidence["denominator_scope"] != "wrong_to_wrong_plus_right_to_wrong":
        raise SchemaError("wrong-overconfidence definition differs")
    denominator = overconfidence["denominator"]
    count = overconfidence["count"]
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (count, denominator)) or count > denominator:
        raise SchemaError("wrong-overconfidence counts are invalid")
    expected_fraction = float(count) / denominator if denominator else None
    if expected_fraction is None:
        if overconfidence["fraction"] is not None:
            raise SchemaError("empty wrong-overconfidence denominator must use null fraction")
    else:
        _assert_close(overconfidence["fraction"], expected_fraction, "wrong-overconfidence fraction")
    subgroups = overconfidence["subgroups"]
    _analysis_exact_keys(subgroups, {"wrong_to_wrong", "right_to_wrong"}, "wrong-overconfidence subgroups")
    subgroup_count = 0
    subgroup_denominator = 0
    for subgroup_name, subgroup in subgroups.items():
        _analysis_exact_keys(subgroup, {"count", "denominator", "fraction"}, "wrong-overconfidence subgroup")
        subgroup_value = subgroup["count"]
        subgroup_total = subgroup["denominator"]
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (subgroup_value, subgroup_total)
        ) or subgroup_value > subgroup_total:
            raise SchemaError("wrong-overconfidence subgroup counts are invalid")
        if subgroup_total != transitions[subgroup_name]:
            raise SchemaError("wrong-overconfidence subgroup denominator differs from transitions")
        expected = float(subgroup_value) / subgroup_total if subgroup_total else None
        if expected is None:
            if subgroup["fraction"] is not None:
                raise SchemaError("empty wrong-overconfidence subgroup requires null fraction")
        else:
            _assert_close(subgroup["fraction"], expected, "wrong-overconfidence subgroup fraction")
        subgroup_count += subgroup_value
        subgroup_denominator += subgroup_total
    if (
        subgroup_count != count
        or subgroup_denominator != denominator
        or denominator != transitions["wrong_to_wrong"] + transitions["right_to_wrong"]
    ):
        raise SchemaError("wrong-overconfidence subgroup totals differ from the aggregate")


def _validate_nca_report(nca: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    primary = nca["primary_cells"]
    expected_primary = [
        "%s_k%d" % (window, k)
        for window in card["science"]["windows"]
        for k in (2, 3, 4)
    ]
    if not isinstance(primary, Mapping) or set(primary) != set(expected_primary):
        raise SchemaError("NCA primary cell order differs")
    for cell in primary.values():
        _analysis_exact_keys(cell, {"denominator", "valid_count", "valid_fraction", "overall_interval"}, "NCA primary cell")
        valid_count = cell["valid_count"]
        if (
            cell["denominator"] != 512
            or isinstance(valid_count, bool)
            or not isinstance(valid_count, int)
            or not 0 <= valid_count <= 512
        ):
            raise SchemaError("NCA denominator/count differs")
        _assert_close(cell["valid_fraction"], valid_count / 512.0, "NCA valid fraction")
        _validate_optional_interval(cell["overall_interval"], -1.0, 1.0, card, kind="median")
        if (valid_count > 0) is not (cell["overall_interval"].get("available") is True):
            raise SchemaError("NCA valid count and interval availability disagree")
    paired = nca["paired_12_15_minus_13_16"]
    if not isinstance(paired, Mapping) or set(paired) != {"k2", "k3", "k4"}:
        raise SchemaError("NCA paired cell order differs")
    for key, cell in paired.items():
        _analysis_exact_keys(cell, {"denominator", "paired_valid_count", "paired_valid_fraction", "difference_interval"}, "paired NCA cell")
        paired_count = cell["paired_valid_count"]
        if (
            cell["denominator"] != 512
            or isinstance(paired_count, bool)
            or not isinstance(paired_count, int)
            or not 0 <= paired_count <= 512
        ):
            raise SchemaError("paired NCA denominator/count differs")
        _assert_close(cell["paired_valid_fraction"], paired_count / 512.0, "paired-valid fraction")
        _validate_optional_interval(cell["difference_interval"], -2.0, 2.0, card, kind="paired_difference")
        if (paired_count > 0) is not (cell["difference_interval"].get("available") is True):
            raise SchemaError("paired NCA valid count and interval availability disagree")
        k = int(key[1:])
        if paired_count > min(
            primary["12:15_k%d" % k]["valid_count"],
            primary["13:16_k%d" % k]["valid_count"],
        ):
            raise SchemaError("paired NCA valid count exceeds a source-cell valid count")
    native = nca["native_fidelity_cells"]
    if not isinstance(native, Mapping) or set(native) != {"12:15_k2", "12:15_k3", "12:15_k4"}:
        raise SchemaError("native-fidelity cell order differs")
    for cell_id, cell in native.items():
        _analysis_exact_keys(
            cell,
            {"valid_subgroup_counts", "right_to_right_interval", "wrong_to_right_minus_wrong_to_wrong_interval", "resampling"},
            "native-fidelity cell",
        )
        if cell["resampling"] != "stratified_within_frozen_subgroup":
            raise SchemaError("native-fidelity resampling differs")
        counts = cell["valid_subgroup_counts"]
        if not isinstance(counts, Mapping) or set(counts) != {
            "right_to_right", "wrong_to_right", "right_to_wrong", "wrong_to_wrong"
        }:
            raise SchemaError("native-fidelity subgroup set differs")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
            raise SchemaError("native-fidelity counts are invalid")
        if sum(counts.values()) != primary[cell_id]["valid_count"]:
            raise SchemaError(
                "native-fidelity subgroups must partition the primary valid samples"
            )
        _validate_optional_interval(cell["right_to_right_interval"], -1.0, 1.0, card, kind="median")
        _validate_optional_interval(
            cell["wrong_to_right_minus_wrong_to_wrong_interval"],
            -2.0,
            2.0,
            card,
            kind="independent_difference",
        )
        rr_available = cell["right_to_right_interval"].get("available") is True
        if rr_available is not (counts["right_to_right"] > 0):
            raise SchemaError("native-fidelity right-to-right count/interval disagree")
        difference = cell["wrong_to_right_minus_wrong_to_wrong_interval"]
        difference_available = difference.get("available") is True
        expected_available = counts["wrong_to_right"] > 0 and counts["wrong_to_wrong"] > 0
        if difference_available is not expected_available:
            raise SchemaError("native-fidelity subgroup counts/interval disagree")
        if difference_available and (
            difference["left_count"] != counts["wrong_to_right"]
            or difference["right_count"] != counts["wrong_to_wrong"]
        ):
            raise SchemaError("native-fidelity interval counts differ from frozen subgroups")


def _validate_trajectory_summary(value: Any, card: Mapping[str, Any]) -> None:
    expected_cells = list(b2_logical_cells(card))
    if not isinstance(value, Mapping) or set(value) != {cell["cell_id"] for cell in expected_cells}:
        raise SchemaError("trajectory summary does not match the frozen 15-cell matrix")
    for expected in expected_cells:
        cell = value[expected["cell_id"]]
        _analysis_exact_keys(
            cell,
            {"cell", "sample_count", "valid_sample_count", "valid_sample_fraction", "steps", "role"},
            "trajectory summary cell",
        )
        if not strict_frozen_equal(cell["cell"], expected) or cell["sample_count"] != 512:
            raise SchemaError("trajectory summary cell identity/count differs")
        valid_count = cell["valid_sample_count"]
        if isinstance(valid_count, bool) or not isinstance(valid_count, int) or not 0 <= valid_count <= 512:
            raise SchemaError("trajectory summary valid count is invalid")
        _assert_close(cell["valid_sample_fraction"], valid_count / 512.0, "trajectory valid fraction")
        if cell["role"] != "calibration_512_mechanism_evidence_only":
            raise SchemaError("trajectory summary role differs")
        steps = cell["steps"]
        if not isinstance(steps, list) or len(steps) != expected["k"]:
            raise SchemaError("trajectory summary step count differs from K")
        for index, step in enumerate(steps):
            _analysis_exact_keys(
                step,
                {
                    "body_call_t", "residual_norm_median", "state_norm_median",
                    "relative_activity_median", "residual_ratio_to_previous_median",
                    "adjacent_residual_cosine_median", "adjacent_residual_cosine_valid_count",
                },
                "trajectory summary step",
            )
            if isinstance(step["body_call_t"], bool) or not isinstance(step["body_call_t"], int) or step["body_call_t"] != index:
                raise SchemaError("trajectory summary step index differs")
            for key in ("residual_norm_median", "state_norm_median", "relative_activity_median"):
                if _finite(step[key], key) < 0.0:
                    raise SchemaError("trajectory summary norm/activity must be non-negative")
            cosine_count = step["adjacent_residual_cosine_valid_count"]
            if isinstance(cosine_count, bool) or not isinstance(cosine_count, int) or not 0 <= cosine_count <= 512:
                raise SchemaError("trajectory adjacent-cosine valid count is invalid")
            if index == 0:
                if step["residual_ratio_to_previous_median"] is not None or step["adjacent_residual_cosine_median"] is not None or cosine_count != 0:
                    raise SchemaError("trajectory t=0 cannot carry adjacent-step summaries")
            else:
                if _finite(step["residual_ratio_to_previous_median"], "residual ratio median") < 0.0:
                    raise SchemaError("trajectory residual ratio median must be non-negative")
                cosine = step["adjacent_residual_cosine_median"]
                if cosine is None:
                    if cosine_count != 0:
                        raise SchemaError("missing adjacent cosine must have zero valid count")
                elif not -1.0 <= _finite(cosine, "adjacent cosine median") <= 1.0:
                    raise SchemaError("trajectory adjacent cosine median must be in [-1,1]")


def _validate_optional_interval(
    value: Any, lower: float, upper: float, card: Mapping[str, Any], *, kind: str
) -> None:
    if not isinstance(value, Mapping) or value.get("available") is not True:
        if not isinstance(value, Mapping) or set(value) != {"available", "reason"} or value.get("available") is not False:
            raise SchemaError("unavailable interval must contain only available=false and reason")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise SchemaError("unavailable interval requires a non-empty reason")
        return
    common = {"available", "difference" if "difference" in kind else "median", "interval", "replicates", "seed", "pointwise", "multiplicity_adjustment", "role"}
    if kind == "paired_difference":
        expected_keys = common | {"paired_index"}
    elif kind == "independent_difference":
        expected_keys = common | {
            "stratified_within_frozen_subgroup", "left_count", "right_count"
        }
    elif kind == "median":
        expected_keys = common
    else:
        raise SchemaError("unknown interval contract kind")
    _analysis_exact_keys(value, expected_keys, "%s NCA interval" % kind)
    interval = value.get("interval")
    _validate_interval(interval, "bootstrap interval")
    if float(interval[0]) < lower or float(interval[1]) > upper:
        raise SchemaError("bootstrap interval exceeds its mathematical range")
    nca_contract = card["statistics"]["nca_bootstrap"]
    if not strict_frozen_equal(
        {"replicates": value.get("replicates"), "seed": value.get("seed")},
        {"replicates": nca_contract["replicates"], "seed": nca_contract["seed"]},
    ):
        raise SchemaError("NCA interval seed/replicates differ from the card")
    if value.get("pointwise") is not True or value.get("multiplicity_adjustment") != "none":
        raise SchemaError("NCA interval multiplicity role differs")
    if value.get("role") != "secondary_diagnostic":
        raise SchemaError("NCA interval role differs")
    point_key = "difference" if "difference" in kind else "median"
    point = _finite(value[point_key], "%s point estimate" % kind)
    if point < lower or point > upper:
        raise SchemaError("NCA point estimate exceeds its mathematical range")
    if kind == "paired_difference" and value.get("paired_index") is not True:
        raise SchemaError("paired NCA difference must reuse sample indices")
    if kind == "independent_difference":
        if value.get("stratified_within_frozen_subgroup") is not True:
            raise SchemaError("native-fidelity difference must resample within subgroups")
        for key in ("left_count", "right_count"):
            if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] < 1:
                raise SchemaError("native-fidelity subgroup count must be positive")


def _validate_interval(value: Any, context: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SchemaError("%s must have two endpoints" % context)
    low = _finite(value[0], "%s lower" % context)
    high = _finite(value[1], "%s upper" % context)
    if low > high:
        raise SchemaError("%s endpoints are reversed" % context)


def _validate_source_hash_tree(value: Any, card: Mapping[str, Any]) -> None:
    _analysis_exact_keys(
        value,
        {
            "card", "calibration_identity_manifest", "full_identity_manifest",
            "calibration_baseline", "calibration_aggregate", "calibration_cells", "full_final_output_cells",
            "calibration_labels", "unseal_authorization", "unseal_receipt",
        },
        "source artifact hashes",
    )
    if value["card"] != card["manifest_sha256"]:
        raise SchemaError("source card hash differs")
    for key in (
        "card", "calibration_identity_manifest", "full_identity_manifest", "calibration_baseline",
        "calibration_aggregate",
        "calibration_labels", "unseal_authorization", "unseal_receipt",
    ):
        _analysis_sha256(value[key], "source_artifacts.%s" % key)
    for key, expected_ids in (
        ("calibration_cells", required_calibration_cell_ids(card)),
        ("full_final_output_cells", required_full_cell_ids(card)),
    ):
        mapping = value[key]
        if not isinstance(mapping, Mapping) or set(mapping) != set(expected_ids):
            raise SchemaError("source cell-hash order differs")
        for cell_id, digest in mapping.items():
            _analysis_sha256(digest, "%s.%s" % (key, cell_id))


def _validate_identity_closure(value: Any, card: Mapping[str, Any]) -> None:
    _analysis_exact_keys(value, {"calibration", "full"}, "identity closure")
    for key, namespace, count in (
        ("calibration", CALIBRATION_IDENTITY_NAMESPACE, 512),
        ("full", FULL_IDENTITY_NAMESPACE, 14042),
    ):
        current = value[key]
        _analysis_exact_keys(
            current,
            {"identity_namespace", "identity_manifest_sha256", "ordered_identity_sha256", "sample_count", "natural_order", "source_provenance"},
            "%s identity closure" % key,
        )
        if current["identity_namespace"] != namespace or current["sample_count"] != count:
            raise SchemaError("%s identity namespace/count differs" % key)
        if current["natural_order"] != "canonical_manifest_list_order_zero_based":
            raise SchemaError("%s natural-order rule differs" % key)
        _analysis_sha256(current["identity_manifest_sha256"], "%s identity manifest" % key)
        _analysis_sha256(current["ordered_identity_sha256"], "%s ordered identity" % key)
        validate_source_provenance(
            current["source_provenance"], namespace, require_live=True
        )


def _validate_analysis_provenance(value: Any) -> None:
    _analysis_exact_keys(
        value,
        {
            "authorization_manifest_sha256", "attempt_manifest_sha256", "receipt_manifest_sha256",
            "executor_thread_id", "command_sha256", "environment_sha256",
        },
        "analysis execution provenance",
    )
    for key in (
        "authorization_manifest_sha256", "attempt_manifest_sha256", "receipt_manifest_sha256",
        "command_sha256", "environment_sha256",
    ):
        _analysis_sha256(value[key], "execution_provenance.%s" % key)
    if not isinstance(value["executor_thread_id"], str) or not value["executor_thread_id"]:
        raise SchemaError("analysis executor thread must be non-empty")


def _assert_close(
    actual: Any, expected: float, context: str, *, abs_tol: float = 1e-12
) -> None:
    if not math.isclose(_finite(actual, context), float(expected), abs_tol=abs_tol, rel_tol=1e-12):
        raise SchemaError("%s differs from recomputed value" % context)


def _probability(value: Any) -> float:
    result = _finite(value, "probability")
    if not 0.0 <= result <= 1.0:
        raise Phase2AnalysisError("probability must be in [0,1]")
    return result


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase2AnalysisError("%s must be a non-boolean number" % context)
    result = float(value)
    if not math.isfinite(result):
        raise Phase2AnalysisError("%s must be finite" % context)
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


__all__ = [
    "Phase2AnalysisError",
    "analyze_phase2_evidence",
    "analysis_command_record",
    "analysis_environment_record",
    "bootstrap_median",
    "bootstrap_paired_median_difference",
    "bootstrap_stratified_median_difference",
    "bootstrap_independent_median_difference",
    "choice_output",
    "apply_primary_holm",
    "compare_choice_outputs",
    "classify_h1",
    "classify_nca",
    "collapse_nca_sample_cell",
    "exact_mcnemar_p",
    "derive_analysis_execution_provenance",
    "holm_step_down",
    "jensen_shannon",
    "load_phase2_analysis_evidence",
    "make_analysis_input_manifest",
    "paired_accuracy_contrast",
    "paired_nca_cells",
    "summarize_nca_cell",
    "summarize_choice_pair",
    "summarize_calibration_trajectory",
    "transition_counts",
    "wrong_overconfidence",
    "required_calibration_cell_ids",
    "required_full_cell_ids",
    "validate_analysis_input_manifest",
    "validate_analysis_report",
    "verify_phase2_analysis_report",
]
