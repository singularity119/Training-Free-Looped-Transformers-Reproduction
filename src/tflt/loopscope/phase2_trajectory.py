"""Scalar-only trajectory measurements for the Phase 2 H1 V2 card.

The collector consumes the existing wrapper ``body_call`` and ``g_minus_x``
events.  Native-continuation and previous-residual vectors are deliberately
kept only in memory; returned records contain scalars and validity metadata.
"""

from __future__ import annotations

import math
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase2_schema import (
    CALIBRATION_IDENTITY_NAMESPACE,
    CHOICE_LABELS,
    DIRECT_PROBE_SCORE_SOURCE,
    atomic_write_new_json,
    protocol_cell_id,
    stable_sample_identity,
    validate_no_persisted_vectors,
    validate_phase2_card,
    validate_calibration_baseline_sample,
    validate_final_output_sample,
    validate_producer_provenance,
    validate_protocol_cell,
    b2_logical_cells,
    make_calibration_baseline_envelope,
    make_calibration_cell_envelope,
    make_identity_manifest,
    make_hashed_manifest,
    file_sha256,
)


NCA_MIN_NORM = 1e-8


class TrajectoryError(ValueError):
    """Raised when a trace cannot satisfy the frozen measurement contract."""


@dataclass(frozen=True)
class VectorMetadata:
    shape: Tuple[int, ...]
    dtype: str
    device: str


def cosine_measurement(
    left: Sequence[float],
    right: Sequence[float],
    *,
    left_meta: Optional[VectorMetadata] = None,
    right_meta: Optional[VectorMetadata] = None,
    min_norm: float = NCA_MIN_NORM,
) -> Dict[str, Any]:
    """Compute a float32-compatible cosine or explain why it is invalid."""

    if left_meta is not None and right_meta is not None:
        if left_meta.shape != right_meta.shape:
            raise TrajectoryError("vector shape mismatch")
        if left_meta.dtype != right_meta.dtype:
            raise TrajectoryError("vector dtype mismatch")
        if left_meta.device != right_meta.device:
            raise TrajectoryError("vector device mismatch")
    lhs = [_finite_float(value, "left vector") for value in left]
    rhs = [_finite_float(value, "right vector") for value in right]
    if not lhs or len(lhs) != len(rhs):
        raise TrajectoryError("cosine vectors must have equal positive length")
    left_norm = math.sqrt(math.fsum(value * value for value in lhs))
    right_norm = math.sqrt(math.fsum(value * value for value in rhs))
    valid = left_norm >= min_norm and right_norm >= min_norm
    cosine = (
        math.fsum(a * b for a, b in zip(lhs, rhs)) / (left_norm * right_norm)
        if valid
        else None
    )
    if cosine is not None:
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "value": cosine,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "valid": valid,
        "invalid_reason": None if valid else "norm_below_1e-8",
        "dtype": "float32",
    }


def baseline_native_alignment(
    boundary_before: Sequence[float],
    boundary_after: Sequence[float],
    native_continuation: Sequence[float],
) -> Dict[str, Any]:
    """Baseline B_(b+1)-B_a NCA, kept distinct from repeated-loop NCA."""

    if len(boundary_before) != len(boundary_after):
        raise TrajectoryError("baseline boundary vectors have different shapes")
    update = [float(after) - float(before) for before, after in zip(boundary_before, boundary_after)]
    result = cosine_measurement(update, native_continuation)
    result["residual_norm"] = result.pop("left_norm")
    result["native_continuation_norm"] = result.pop("right_norm")
    result["provenance"] = "baseline_single_forward_B_(b+1)-B_a"
    result["body_call_t"] = None
    return result


class Phase2TrajectoryCollector:
    """Collect answer-position residual scalars from unchanged wrapper events."""

    def __init__(self, *, window_width: int, expected_k: int) -> None:
        if int(window_width) != 4 or int(expected_k) < 1:
            raise TrajectoryError("Phase 2 requires inclusive width 4 and K>=1")
        self.window_width = int(window_width)
        self.expected_k = int(expected_k)
        self.samples: List[Dict[str, Any]] = []
        self._active: Optional[Dict[str, Any]] = None
        self._last_admission_vectors: Optional[Dict[str, Any]] = None

    def begin_forward(
        self,
        *,
        task: str,
        doc_id: Any,
        doc_hash: str,
        answer_position: int,
        native_continuation: Any,
        protocol: str,
        window: str,
        alpha: float,
    ) -> None:
        if self._active is not None:
            raise TrajectoryError("begin_forward called before end_forward")
        if hasattr(native_continuation, "detach"):
            native_value = native_continuation.detach()
            if getattr(native_value, "ndim", None) != 1:
                raise TrajectoryError("native continuation must be a vector")
            vector = native_value
            metadata = VectorMetadata(
                tuple(int(item) for item in native_value.shape),
                str(native_value.dtype),
                str(native_value.device),
            )
        else:
            vector, metadata = _answer_vector(native_continuation, 0, already_vector=True)
        self._active = {
            "sample_identity": stable_sample_identity(task, doc_id, doc_hash),
            "answer_position": int(answer_position),
            "native": vector,
            "native_meta": metadata,
            "protocol": str(protocol),
            "window": str(window),
            "alpha": float(alpha),
            "counts": Counter(),
            "timeline": [],
            "steps": [],
            "previous_residual": None,
            "previous_meta": None,
            "errors": [],
            "admission_states": [],
            "admission_residuals": [],
        }

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        active = self._require_active()
        active["counts"][str(event)] += 1
        active["timeline"].append(str(event))
        if event == "body_call":
            active["counts"]["operator_body_calls"] += 1
        elif event == "wrapper_forward":
            key = "bypass_true" if payload.get("bypass") else "bypass_false"
            active["counts"][key] += 1

    def record_tensor_diff(self, name: str, before: Any, after: Any) -> None:
        active = self._require_active()
        active["counts"][str(name)] += 1
        active["timeline"].append(str(name))
        if name != "g_minus_x":
            return
        try:
            if hasattr(before, "detach") and hasattr(after, "detach"):
                _record_device_tensor_step(active, before, after)
                return
            before_vector, before_meta = _answer_vector(before, active["answer_position"])
            after_vector, after_meta = _answer_vector(after, active["answer_position"])
            _require_same_metadata(before_meta, after_meta)
            residual = [right - left for left, right in zip(before_vector, after_vector)]
            residual_meta = before_meta
            residual_norm = _norm(residual)
            state_norm = _norm(before_vector)
            if state_norm < NCA_MIN_NORM:
                raise TrajectoryError("state norm below 1e-8")
            t = len(active["steps"])
            previous = active["previous_residual"]
            ratio = None
            adjacent_cosine = None
            adjacent_cosine_valid = None
            if previous is not None:
                previous_norm = _norm(previous)
                if previous_norm < NCA_MIN_NORM:
                    raise TrajectoryError("previous residual norm below 1e-8")
                ratio = residual_norm / previous_norm
                adjacent = cosine_measurement(
                    residual,
                    previous,
                    left_meta=residual_meta,
                    right_meta=active["previous_meta"],
                )
                adjacent_cosine = adjacent["value"]
                adjacent_cosine_valid = adjacent["valid"]
            nca = None
            if t >= 1:
                nca = cosine_measurement(
                    residual,
                    active["native"],
                    left_meta=residual_meta,
                    right_meta=active["native_meta"],
                )
                nca["residual_norm"] = nca.pop("left_norm")
                nca["native_continuation_norm"] = nca.pop("right_norm")
                nca["provenance"] = "actual_loop_repeated_step"
                nca["body_call_t"] = t
            active["steps"].append(
                {
                    "body_call_t": t,
                    "residual_norm": residual_norm,
                    "state_norm": state_norm,
                    "relative_activity": residual_norm / state_norm,
                    "residual_ratio_to_previous": ratio,
                    "adjacent_residual_cosine": adjacent_cosine,
                    "adjacent_residual_cosine_valid": adjacent_cosine_valid,
                    "nca": nca,
                    "valid": True,
                }
            )
            active["admission_states"].append(list(before_vector))
            active["admission_residuals"].append(list(residual))
            active["previous_residual"] = residual
            active["previous_meta"] = residual_meta
        except Exception as exc:
            message = "g_minus_x: %s" % exc
            active["errors"].append(message)
            raise TrajectoryError(message) from exc

    def end_forward(self, *, final_output: Mapping[str, Any]) -> Dict[str, Any]:
        active = self._require_active()
        counts = active["counts"]
        if counts.get("wrapper_forward", 0) != 1 or counts.get("bypass_false", 0) != 1:
            active["errors"].append("expected one non-bypass wrapper_forward")
        if counts.get("operator_body_calls", 0) != self.expected_k:
            active["errors"].append("body-call count differs from K")
        if counts.get("g_minus_x", 0) != self.expected_k:
            active["errors"].append("g_minus_x count differs from K")
        if len(active["steps"]) != self.expected_k:
            active["errors"].append("valid scalar step count differs from K")
        if counts.get("identity_forward", 0) != self.window_width - 1:
            active["errors"].append("identity_forward count differs from inclusive width")
        expected = ["wrapper_forward"]
        for _ in range(self.expected_k):
            expected.extend(("body_call", "g_minus_x"))
        expected.extend(("looped_hidden_vs_input", "stash_pass"))
        expected.extend(["identity_forward"] * (self.window_width - 1))
        if active["timeline"] != expected:
            active["errors"].append("unexpected wrapper event timeline")
        repeated = [step["nca"] for step in active["steps"] if step["body_call_t"] >= 1]
        all_repeated_valid = bool(repeated) and all(item and item.get("valid") for item in repeated)
        if self.expected_k == 1:
            all_repeated_valid = True
        result = {
            "sample_identity": active["sample_identity"],
            "answer_position": active["answer_position"],
            "position_rule": "final_pre_answer_prompt_token",
            "boundary_identity": "native_continuation=B_N-B_(b+1)",
            "body_call_indexing": "zero_based_t=0..K-1",
            "steps": active["steps"],
            "repeated_step_nca_all_valid": all_repeated_valid,
            "valid": not active["errors"] and all_repeated_valid,
            "errors": active["errors"],
            "event_counts": dict(counts),
            "final_output": dict(final_output),
        }
        cell = {
            "cell_id": protocol_cell_id(
                active["protocol"], active["window"], self.expected_k, active["alpha"]
            ),
            "protocol": active["protocol"],
            "window": active["window"],
            "k": self.expected_k,
            "alpha": active["alpha"],
            "step_size": active["alpha"] / self.expected_k,
        }
        from tflt.loopscope.phase2_schema import validate_trajectory_sample

        validate_trajectory_sample(result, cell, expected_identity=active["sample_identity"])
        self._last_admission_vectors = {
            "sample_identity": active["sample_identity"],
            "states": active["admission_states"],
            "residuals": active["admission_residuals"],
        }
        self.samples.append(result)
        self._active = None
        return result

    def pop_admission_vectors(self) -> Dict[str, Any]:
        """Return and clear the last sample's ephemeral prefix vectors."""

        if self._last_admission_vectors is None:
            raise TrajectoryError("no completed trajectory is available for B1 prefix proof")
        value = self._last_admission_vectors
        self._last_admission_vectors = None
        return value

    def _require_active(self) -> Dict[str, Any]:
        if self._active is None:
            raise TrajectoryError("collector event outside begin/end forward")
        return self._active


class InMemoryProbeSession:
    """Enforce one no-loop boundary pass plus 15 logical loop cells per process."""

    def __init__(self, expected_cells: Sequence[Mapping[str, Any]]) -> None:
        self.expected_cells = [dict(cell) for cell in expected_cells]
        identities = [
            protocol_cell_id(cell["protocol"], cell["window"], cell["k"], cell["alpha"])
            for cell in self.expected_cells
        ]
        if len(identities) != 15 or len(set(identities)) != 15:
            raise TrajectoryError("B2 session requires exactly 15 unique logical cells")
        self._native_by_sample_window: Dict[Tuple[str, str], Any] = {}
        self._baseline_complete = False

    def run(
        self,
        baseline_pass: Callable[[Dict[Tuple[str, str], Any]], None],
        loop_cell: Callable[[Mapping[str, Any], Mapping[Tuple[str, str], Any]], Any],
    ) -> List[Any]:
        if self._baseline_complete:
            raise TrajectoryError("probe session is write-once")
        baseline_pass(self._native_by_sample_window)
        self._baseline_complete = True
        if not self._native_by_sample_window:
            raise TrajectoryError("no-loop boundary pass produced no in-memory vectors")
        results = []
        for cell in self.expected_cells:
            results.append(loop_cell(cell, self._native_by_sample_window))
        self._native_by_sample_window.clear()
        return results


def add_probe_phase2_trajectory_args(parser: Any) -> None:
    parser.add_argument("--probe-mode", choices=("b1-smoke", "b2-calibration"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attempt-manifest", required=True)
    parser.add_argument("--receipt-manifest", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--b1-admission-proof", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")


def cmd_probe_phase2_trajectory(args: Any) -> int:
    """Run the controlled 1+15 calibration design in one loaded-model session."""

    from tflt.loopscope.probe import _write_failure, environment_snapshot, serializable_args
    from tflt.loopscope.schema import ensure_new_directory, write_new_json

    output_dir = ensure_new_directory(Path(args.output_dir))
    try:
        write_new_json(output_dir / "command_args.json", serializable_args(args))
        write_new_json(output_dir / "env.json", environment_snapshot())
        result = run_phase2_trajectory_probe(args)
        report = materialize_phase2_probe_artifacts(output_dir, args, result)
    except Exception as exc:
        _write_failure(output_dir, exc)
        raise
    print(str(output_dir / "phase2_probe_manifest.json"))
    return 0


def materialize_phase2_probe_artifacts(
    output_dir: Path, args: Any, result: Mapping[str, Any]
) -> Dict[str, Any]:
    """Seal scalar probe products as exclusive, independently hashed artifacts."""

    from tflt.loopscope.schema import verify_manifest_sha256

    card = result["card"]
    validate_phase2_card(card)
    attempt = json.loads(Path(args.attempt_manifest).read_text(encoding="utf-8"))
    receipt = json.loads(Path(args.receipt_manifest).read_text(encoding="utf-8"))
    verify_manifest_sha256(attempt)
    verify_manifest_sha256(receipt)
    revision_evidence = make_hashed_manifest(
        {
            "schema_version": "loopscope.phase2-probe-revision-evidence.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "manifest_revision": card["science"]["revision"],
            "revision_closure": result["revision_closure"],
        }
    )
    atomic_write_new_json(output_dir / "revision_evidence.json", revision_evidence)
    producer = {
        "producer_kind": (
            "gate_b_remote_b1_smoke_probe"
            if result["probe_mode"] == "b1-smoke"
            else "gate_b_remote_calibration_probe"
        ),
        "attempt_manifest_sha256": attempt["manifest_sha256"],
        "receipt_manifest_sha256": receipt["manifest_sha256"],
        "command_sha256": file_sha256(output_dir / "command_args.json"),
        "environment_sha256": file_sha256(output_dir / "env.json"),
        "revision_report_sha256": revision_evidence["manifest_sha256"],
    }
    if result["probe_mode"] == "b1-smoke":
        proof = result["b1_admission_proof"]
        validate_b1_admission_proof(proof, card)
        atomic_write_new_json(output_dir / "b1_admission_proof.json", proof)
        report = make_hashed_manifest(
            {
                "schema_version": "loopscope.phase2-b1-smoke-aggregate.v1",
                "artifact_kind": "b1_smoke_scalar_aggregate",
                "card_manifest_sha256": card["manifest_sha256"],
                "revision": card["science"]["revision"],
                "sample_count": 4,
                "probe_pool": result["pool"],
                "warnings": result["warnings"],
                "session_contract": result["session_contract"],
                "baseline_examples": result["baseline_examples"],
                "loop_cells": result["loop_cells"],
                "b1_admission_proof_sha256": proof["manifest_sha256"],
                "producer": producer,
                "vectors_persisted": False,
            }
        )
        validate_no_persisted_vectors(report)
        validate_probe_aggregate(report, card, b1_proof=proof)
        atomic_write_new_json(output_dir / "phase2_probe_manifest.json", report)
        return report

    identities = [_record_identity(record) for record in result["records"]]
    identity_manifest = make_identity_manifest(
        card,
        identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
        split="phase1_frozen_validation",
        identities=identities,
    )
    atomic_write_new_json(output_dir / "calibration_identity_manifest.json", identity_manifest)
    baseline = make_calibration_baseline_envelope(
        card,
        identity_manifest,
        samples=result["baseline_examples"],
        producer=producer,
    )
    atomic_write_new_json(output_dir / "calibration_baseline.json", baseline)
    cell_refs = []
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=False, exist_ok=False)
    for raw in result["loop_cells"]:
        envelope = make_calibration_cell_envelope(
            card,
            identity_manifest,
            cell=raw["cell"],
            samples=raw["trajectory_samples"],
            producer=producer,
        )
        relative = Path("cells") / (raw["cell"]["cell_id"] + ".json")
        atomic_write_new_json(output_dir / relative, envelope)
        cell_refs.append(
            {
                "cell_id": raw["cell"]["cell_id"],
                "path": str(relative),
                "sha256": envelope["manifest_sha256"],
            }
        )
    report = make_hashed_manifest(
        {
            "schema_version": "loopscope.phase2-calibration-aggregate.v1",
            "artifact_kind": "sealed_calibration_512_scalar_aggregate",
            "card_manifest_sha256": card["manifest_sha256"],
            "identity_manifest_sha256": identity_manifest["manifest_sha256"],
            "ordered_identity_sha256": identity_manifest["ordered_identity_sha256"],
            "revision": card["science"]["revision"],
            "label_state": "sealed",
            "sample_count": 512,
            "probe_pool": result["pool"],
            "warnings": result["warnings"],
            "session_contract": result["session_contract"],
            "baseline": {
                "path": "calibration_baseline.json",
                "sha256": baseline["manifest_sha256"],
            },
            "cells": cell_refs,
            "prior_b1_admission_proof_sha256": result["prior_b1_admission_proof_sha256"],
            "producer": producer,
            "vectors_persisted": False,
        }
    )
    validate_no_persisted_vectors(report)
    validate_probe_aggregate(report, card)
    atomic_write_new_json(output_dir / "phase2_probe_manifest.json", report)
    return report


def validate_probe_aggregate(
    report: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    b1_proof: Optional[Mapping[str, Any]] = None,
) -> None:
    """Closed-world validation for the B1 or B2 aggregate routing envelope."""

    from tflt.loopscope.schema import verify_manifest_sha256

    validate_phase2_card(card)
    verify_manifest_sha256(report)
    schema = report.get("schema_version")
    if schema == "loopscope.phase2-b1-smoke-aggregate.v1":
        _trajectory_exact_keys(
            report,
            {
                "schema_version", "artifact_kind", "card_manifest_sha256", "revision",
                "sample_count", "probe_pool", "warnings", "session_contract",
                "baseline_examples", "loop_cells", "b1_admission_proof_sha256",
                "producer", "vectors_persisted", "manifest_sha256",
            },
            "B1 aggregate",
        )
        if report["artifact_kind"] != "b1_smoke_scalar_aggregate" or report["sample_count"] != 4:
            raise TrajectoryError("B1 aggregate kind/count differs")
        if b1_proof is None:
            raise TrajectoryError("B1 aggregate validation requires the independent proof")
        validate_b1_admission_proof(b1_proof, card)
        if report["b1_admission_proof_sha256"] != b1_proof["manifest_sha256"]:
            raise TrajectoryError("B1 aggregate proof hash differs")
        if len(report["baseline_examples"]) != 4 or len(report["loop_cells"]) != 15:
            raise TrajectoryError("B1 aggregate scalar matrix is incomplete")
        proof_identities = b1_proof["ordered_sample_identity"]
        for baseline, expected_identity in zip(report["baseline_examples"], proof_identities):
            validate_calibration_baseline_sample(
                baseline, card, expected_identity=expected_identity
            )
        expected_cells = list(b2_logical_cells(card))
        if [raw.get("cell") for raw in report["loop_cells"]] != expected_cells:
            raise TrajectoryError("B1 aggregate cells differ from the frozen 15-cell order")
        for raw, expected_cell in zip(report["loop_cells"], expected_cells):
            _trajectory_exact_keys(
                raw, {"cell", "sample_count", "trajectory_samples"}, "B1 cell aggregate"
            )
            if raw["cell"] != expected_cell:
                raise TrajectoryError("B1 cell protocol identity differs")
            validate_protocol_cell(raw["cell"])
            if raw["sample_count"] != 4 or len(raw["trajectory_samples"]) != 4:
                raise TrajectoryError("B1 cell sample count differs")
            for sample, expected_identity in zip(raw["trajectory_samples"], proof_identities):
                from tflt.loopscope.phase2_schema import validate_trajectory_sample

                validate_trajectory_sample(
                    sample, raw["cell"], expected_identity=expected_identity
                )
        expected_session = {
            "model_load_count": 1,
            "no_loop_boundary_passes_per_sample": 1,
            "logical_loop_cell_count": 18,
            "k1_admission_cell_count": 3,
            "native_vectors_persisted": False,
        }
        allowed_producer_kinds = {"gate_b_remote_b1_smoke_probe"}
    elif schema == "loopscope.phase2-calibration-aggregate.v1":
        _trajectory_exact_keys(
            report,
            {
                "schema_version", "artifact_kind", "card_manifest_sha256",
                "identity_manifest_sha256", "ordered_identity_sha256", "revision",
                "label_state", "sample_count", "probe_pool", "warnings", "session_contract",
                "baseline", "cells", "prior_b1_admission_proof_sha256", "producer",
                "vectors_persisted", "manifest_sha256",
            },
            "B2 aggregate",
        )
        if report["artifact_kind"] != "sealed_calibration_512_scalar_aggregate" or report["label_state"] != "sealed" or report["sample_count"] != 512:
            raise TrajectoryError("B2 aggregate kind/seal/count differs")
        _trajectory_ref(report["baseline"], "B2 baseline ref")
        expected_ids = [cell["cell_id"] for cell in b2_logical_cells(card)]
        refs = report["cells"]
        if not isinstance(refs, list) or [ref.get("cell_id") for ref in refs] != expected_ids:
            raise TrajectoryError("B2 aggregate cell refs differ from the frozen 15-cell order")
        for ref in refs:
            _trajectory_exact_keys(ref, {"cell_id", "path", "sha256"}, "B2 cell ref")
            _trajectory_ref({"path": ref["path"], "sha256": ref["sha256"]}, "B2 cell ref")
        _trajectory_sha256(report["prior_b1_admission_proof_sha256"], "prior B1 proof")
        _trajectory_sha256(report["identity_manifest_sha256"], "B2 identity manifest")
        _trajectory_sha256(report["ordered_identity_sha256"], "B2 ordered identity")
        expected_session = {
            "model_load_count": 1,
            "no_loop_boundary_passes_per_sample": 1,
            "logical_loop_cell_count": 15,
            "k1_admission_cell_count": 0,
            "native_vectors_persisted": False,
        }
        allowed_producer_kinds = {"gate_b_remote_calibration_probe"}
    else:
        raise TrajectoryError("unsupported Phase 2 probe aggregate schema")
    if report["card_manifest_sha256"] != card["manifest_sha256"] or report["revision"] != card["science"]["revision"]:
        raise TrajectoryError("probe aggregate card/revision binding differs")
    if report["session_contract"] != expected_session:
        raise TrajectoryError("probe aggregate session contract differs")
    if report["vectors_persisted"] is not False:
        raise TrajectoryError("probe aggregate cannot persist vectors")
    if not isinstance(report["probe_pool"], Mapping) or not isinstance(report["warnings"], list):
        raise TrajectoryError("probe aggregate pool/warnings are malformed")
    validate_producer_provenance(
        report["producer"], allowed_kinds=allowed_producer_kinds
    )
    validate_no_persisted_vectors(report)


def _trajectory_exact_keys(value: Any, expected: Sequence[str], context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise TrajectoryError("%s exact-key contract differs" % context)


def _trajectory_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TrajectoryError("%s must be lowercase SHA256" % context)
    return value


def _trajectory_ref(value: Any, context: str) -> None:
    _trajectory_exact_keys(value, {"path", "sha256"}, context)
    if not isinstance(value["path"], str) or not value["path"]:
        raise TrajectoryError("%s path must be non-empty" % context)
    _trajectory_sha256(value["sha256"], "%s hash" % context)


def run_phase2_trajectory_probe(args: Any) -> Dict[str, Any]:
    """Remote-only producer; imports heavyweight dependencies lazily."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - Gate B remote dependency path.
        raise RuntimeError("Phase 2 trajectory probe requires torch and transformers") from exc

    from tflt.config import LoopConfig
    from tflt.loopscope.phase2_analysis import choice_output
    from tflt.loopscope.probe import (
        choice_tokenization,
        decoder_boundary_states,
        load_probe_records,
        probe_pool_metadata,
        resolve_answer_position,
        select_device,
        torch_dtype,
    )
    from tflt.loopscope.revisions import (
        load_tokenizer_with_resolved_commit,
        strict_revision_closure,
    )
    from tflt.models import resolve_model
    from tflt.wrapper import apply_loop_wrapper

    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    validate_phase2_card(card)
    probe_mode = str(args.probe_mode)
    if args.model != "qwen3-1.7b-base" or args.revision != card["science"]["revision"]:
        raise TrajectoryError("loaded model alias/revision differs from the frozen card")
    if args.dtype != card["science"]["dtype"]:
        raise TrajectoryError("probe dtype differs from the frozen card")
    records = load_probe_records(Path(args.input_jsonl), args.max_examples)
    pool_metadata, warnings = probe_pool_metadata(records, args.input_manifest)
    if probe_mode == "b1-smoke":
        if len(records) != 4 or args.max_examples != 4:
            raise TrajectoryError("B1 smoke producer requires exactly --max-examples 4")
        if args.b1_admission_proof is not None:
            raise TrajectoryError("B1 smoke cannot consume a prior B1 proof")
        prior_b1_proof = None
    else:
        if args.max_examples is not None or len(records) != card["science"]["calibration_count"]:
            raise TrajectoryError("B2 calibration requires the full frozen 512 records")
        if not args.b1_admission_proof:
            raise TrajectoryError("B2 calibration requires an audited B1 admission proof")
        prior_b1_proof = json.loads(Path(args.b1_admission_proof).read_text(encoding="utf-8"))
        validate_b1_admission_proof(prior_b1_proof, card)
        if not prior_b1_proof["k1_equivalence"]["all_match"] or not prior_b1_proof["fixed_step_prefix_consistency"]["all_match"]:
            raise TrajectoryError("B2 admission proof contains a failed K1/prefix check")
    model_info = resolve_model(args.model)
    load_kwargs = {"trust_remote_code": True, "revision": args.revision}
    tokenizer, tokenizer_revision = load_tokenizer_with_resolved_commit(
        AutoTokenizer, model_info["repo_id"], load_kwargs
    )
    _, choice_ids, _ = choice_tokenization(tokenizer, CHOICE_LABELS)
    device = select_device(torch, args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_info["repo_id"], torch_dtype=torch_dtype(torch, args.dtype), **load_kwargs
    )
    model.eval()
    model.to(device)
    closure = strict_revision_closure(model, tokenizer_revision, args.revision)
    layer_count = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    windows = list(card["science"]["windows"])
    if any(int(window.split(":")[1]) >= layer_count for window in windows):
        raise TrajectoryError("frozen window exceeds loaded decoder depth")

    native_store: Dict[Tuple[str, str], Any] = {}
    baseline_examples = []
    with torch.inference_mode():
        for record in records:
            encoded = tokenizer(record["text"], return_tensors="pt")
            inputs = {key: value.to(device) for key, value in encoded.items()}
            answer_position = resolve_answer_position(inputs["attention_mask"], record)
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            boundaries = decoder_boundary_states(tuple(outputs.hidden_states or ()), layer_count)
            identity = _record_identity(record)
            baseline_choice = _choice_from_logits(
                outputs.logits, answer_position, choice_ids, identity, torch, choice_output
            )
            boundary_rows = []
            for window in windows:
                start, end = (int(value) for value in window.split(":"))
                before = boundaries[start][0, answer_position, :].detach()
                after = boundaries[end + 1][0, answer_position, :].detach()
                native = boundaries[-1][0, answer_position, :].detach() - after
                key = (_identity_key(identity), window)
                native_store[key] = native
                baseline_update = after.float() - before.float()
                native32 = native.float()
                alignment = _torch_cosine_payload(baseline_update, native32, torch)
                alignment["provenance"] = "baseline_single_forward_B_(b+1)-B_a"
                alignment["body_call_t"] = None
                boundary_rows.append(
                    {
                        "window": window,
                        "inclusive_boundary": "B_%d_to_B_%d" % (start, end + 1),
                        "native_boundary_formula": "B_N-B_%d" % (end + 1),
                        "baseline_update_norm": float(baseline_update.norm().item()),
                        "native_continuation_norm": float(native32.norm().item()),
                        "baseline_nca": alignment,
                    }
                )
            baseline_examples.append(
                {
                    "sample_identity": identity,
                    "answer_position": answer_position,
                    "final_output": baseline_choice,
                    "window_boundaries": boundary_rows,
                }
            )
            del outputs, boundaries, inputs, encoded

    cell_reports = []
    prefix_captures_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for cell in b2_logical_cells(card):
        collector = Phase2TrajectoryCollector(window_width=4, expected_k=cell["k"])
        config = LoopConfig.from_window_string(
            model_alias=args.model,
            window=cell["window"],
            k=cell["k"],
            iteration_mode="block",
            strategy="damped_euler",
            alpha=cell["alpha"],
            beta=0.0,
            cache_strategy="last",
            decode_mode="bypass",
            audit_collector=collector,
        )
        handle = apply_loop_wrapper(model, config)
        try:
            with torch.inference_mode():
                for record in records:
                    encoded = tokenizer(record["text"], return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in encoded.items()}
                    answer_position = resolve_answer_position(inputs["attention_mask"], record)
                    identity = _record_identity(record)
                    native = native_store[(_identity_key(identity), cell["window"])]
                    collector.begin_forward(
                        task=identity["task"],
                        doc_id=identity["doc_id"],
                        doc_hash=identity["doc_hash"],
                        answer_position=answer_position,
                        native_continuation=native,
                        protocol=cell["protocol"],
                        window=cell["window"],
                        alpha=cell["alpha"],
                    )
                    outputs = model(**inputs, use_cache=False, return_dict=True)
                    final_output = _choice_from_logits(
                        outputs.logits, answer_position, choice_ids, identity, torch, choice_output
                    )
                    collector.end_forward(final_output=final_output)
                    capture = collector.pop_admission_vectors()
                    if probe_mode == "b1-smoke" and cell["protocol"] in ("shared_k2_anchor", "fixed_step"):
                        prefix_captures_by_cell.setdefault(cell["cell_id"], []).append(capture)
                    else:
                        del capture
                    del outputs, inputs, encoded
        finally:
            handle.restore()
        cell_reports.append(
            {
                "cell": cell,
                "sample_count": len(collector.samples),
                "trajectory_samples": collector.samples,
            }
        )
    b1_proof = None
    if probe_mode == "b1-smoke":
        k1_outputs_by_window: Dict[str, List[Dict[str, Any]]] = {}
        for window in windows:
            collector = Phase2TrajectoryCollector(window_width=4, expected_k=1)
            config = LoopConfig.from_window_string(
                model_alias=args.model,
                window=window,
                k=1,
                iteration_mode="block",
                strategy="damped_euler",
                alpha=1.0,
                beta=0.0,
                cache_strategy="last",
                decode_mode="bypass",
                audit_collector=collector,
            )
            handle = apply_loop_wrapper(model, config)
            rows = []
            try:
                with torch.inference_mode():
                    for record in records:
                        encoded = tokenizer(record["text"], return_tensors="pt")
                        inputs = {key: value.to(device) for key, value in encoded.items()}
                        answer_position = resolve_answer_position(inputs["attention_mask"], record)
                        identity = _record_identity(record)
                        native = native_store[(_identity_key(identity), window)]
                        collector.begin_forward(
                            task=identity["task"],
                            doc_id=identity["doc_id"],
                            doc_hash=identity["doc_hash"],
                            answer_position=answer_position,
                            native_continuation=native,
                            protocol="fixed_horizon",
                            window=window,
                            alpha=1.0,
                        )
                        outputs = model(**inputs, use_cache=False, return_dict=True)
                        final_output = _choice_from_logits(
                            outputs.logits, answer_position, choice_ids, identity, torch, choice_output
                        )
                        collector.end_forward(final_output=final_output)
                        collector.pop_admission_vectors()
                        rows.append(final_output)
                        del outputs, inputs, encoded
            finally:
                handle.restore()
            k1_outputs_by_window[window] = rows
        b1_proof = build_b1_admission_proof(
            card,
            [row["final_output"] for row in baseline_examples],
            k1_outputs_by_window,
            prefix_captures_by_cell,
        )
    native_store.clear()
    result = {
        "probe_mode": probe_mode,
        "card": card,
        "model": model_info,
        "revision_closure": closure,
        "pool": pool_metadata,
        "warnings": list(warnings),
        "records": records,
        "session_contract": {
            "model_load_count": 1,
            "no_loop_boundary_passes_per_sample": 1,
            "logical_loop_cell_count": 18 if probe_mode == "b1-smoke" else 15,
            "k1_admission_cell_count": 3 if probe_mode == "b1-smoke" else 0,
            "native_vectors_persisted": False,
        },
        "baseline_examples": baseline_examples,
        "loop_cells": cell_reports,
        "b1_admission_proof": b1_proof,
        "prior_b1_admission_proof_sha256": (
            prior_b1_proof["manifest_sha256"] if prior_b1_proof is not None else None
        ),
    }
    validate_no_persisted_vectors(
        {
            "baseline_examples": baseline_examples,
            "loop_cells": cell_reports,
            "b1_admission_proof": b1_proof,
        }
    )
    return result


def _record_identity(record: Mapping[str, Any]) -> Dict[str, str]:
    return stable_sample_identity(
        str(record["task_name"]),
        str(record["target_doc_id"]),
        str(record["target_doc_sha256"]),
    )


def _identity_key(identity: Mapping[str, str]) -> str:
    return "%s\0%s\0%s" % (identity["task"], identity["doc_id"], identity["doc_hash"])


def _choice_from_logits(
    logits: Any,
    answer_position: int,
    choice_ids: Sequence[int],
    identity: Mapping[str, Any],
    torch: Any,
    choice_builder: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    scores = torch.log_softmax(logits[0, answer_position, :].float(), dim=-1)[
        list(choice_ids)
    ].detach().cpu().tolist()
    return choice_builder(scores, identity=identity, score_source=DIRECT_PROBE_SCORE_SOURCE)


def _torch_cosine_payload(left: Any, right: Any, torch: Any) -> Dict[str, Any]:
    if left.shape != right.shape or left.device != right.device:
        raise TrajectoryError("baseline NCA shape/device mismatch")
    if not bool(torch.isfinite(left).all().item()) or not bool(torch.isfinite(right).all().item()):
        raise TrajectoryError("baseline NCA contains NaN/Inf")
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    valid = left_norm >= NCA_MIN_NORM and right_norm >= NCA_MIN_NORM
    return {
        "value": (
            max(-1.0, min(1.0, float((left @ right / (left.norm() * right.norm())).item())))
            if valid
            else None
        ),
        "residual_norm": left_norm,
        "native_continuation_norm": right_norm,
        "valid": valid,
        "invalid_reason": None if valid else "norm_below_1e-8",
        "dtype": "float32",
    }


def k1_choice_equivalent(
    baseline_scores: Sequence[float],
    k1_scores: Sequence[float],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> bool:
    if len(baseline_scores) != 4 or len(k1_scores) != 4:
        raise TrajectoryError("K1 equivalence requires four choice scores")
    return all(
        math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)
        for left, right in zip(baseline_scores, k1_scores)
    )


def prefix_consistent(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    if len(candidate) < len(reference):
        raise TrajectoryError("candidate trajectory is shorter than reference prefix")
    max_abs = 0.0
    for expected, actual in zip(reference, candidate):
        if len(expected) != len(actual):
            raise TrajectoryError("prefix vector shape mismatch")
        for left, right in zip(expected, actual):
            difference = abs(float(left) - float(right))
            max_abs = max(max_abs, difference)
            if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
                return {"match": False, "max_abs": max_abs, "atol": atol, "rtol": rtol}
    return {"match": True, "max_abs": max_abs, "atol": atol, "rtol": rtol}


def build_b1_admission_proof(
    card: Mapping[str, Any],
    baseline_outputs: Sequence[Mapping[str, Any]],
    k1_outputs_by_window: Mapping[str, Sequence[Mapping[str, Any]]],
    prefix_captures_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Derive B1 K1 and cross-K proofs from ephemeral in-memory vectors.

    The inputs may contain vectors because this function executes inside the
    controlled model process.  The returned proof contains scalar maxima and
    boolean results only.
    """

    validate_phase2_card(card)
    if len(baseline_outputs) != 4:
        raise TrajectoryError("B1 admission proof requires exactly four smoke samples")
    identities = [_proof_identity(row) for row in baseline_outputs]
    if len({_identity_key(value) for value in identities}) != 4:
        raise TrajectoryError("B1 admission identities must be unique")
    for row, expected_identity in zip(baseline_outputs, identities):
        validate_final_output_sample(
            row,
            score_source=DIRECT_PROBE_SCORE_SOURCE,
            gold_state="forbidden",
            expected_identity=expected_identity,
        )
    tolerances = card["tolerances"]
    k1_windows = []
    for window in card["science"]["windows"]:
        rows = k1_outputs_by_window.get(window)
        if not isinstance(rows, Sequence) or len(rows) != 4:
            raise TrajectoryError("B1 K1 outputs are incomplete for window %s" % window)
        records = []
        for baseline, candidate, expected_identity in zip(baseline_outputs, rows, identities):
            if _proof_identity(candidate) != expected_identity:
                raise TrajectoryError("B1 K1 identity/order differs from no-loop baseline")
            validate_final_output_sample(
                candidate,
                score_source=DIRECT_PROBE_SCORE_SOURCE,
                gold_state="forbidden",
                expected_identity=expected_identity,
            )
            baseline_scores = baseline.get("raw_choice_scores")
            candidate_scores = candidate.get("raw_choice_scores")
            match = k1_choice_equivalent(
                baseline_scores,
                candidate_scores,
                atol=tolerances["k1_scores_atol"],
                rtol=tolerances["k1_scores_rtol"],
            )
            maximum = max(abs(float(left) - float(right)) for left, right in zip(baseline_scores, candidate_scores))
            top1_match = int(baseline.get("top1_index", -1)) == int(candidate.get("top1_index", -2))
            records.append(
                {
                    "sample_identity": expected_identity,
                    "scores_allclose": match,
                    "top1_match": top1_match,
                    "max_abs_score_difference": maximum,
                }
            )
        k1_windows.append(
            {
                "window": window,
                "k": 1,
                "alpha": 1.0,
                "records": records,
                "all_match": all(row["scores_allclose"] and row["top1_match"] for row in records),
            }
        )

    prefix_windows = []
    for window in card["science"]["windows"]:
        reference_id = protocol_cell_id("shared_k2_anchor", window, 2, 1.0)
        reference = prefix_captures_by_cell.get(reference_id)
        if not isinstance(reference, Sequence) or len(reference) != 4:
            raise TrajectoryError("B1 K2 prefix capture is incomplete for window %s" % window)
        comparisons = []
        for k, alpha in ((3, 1.5), (4, 2.0)):
            candidate_id = protocol_cell_id("fixed_step", window, k, alpha)
            candidate = prefix_captures_by_cell.get(candidate_id)
            if not isinstance(candidate, Sequence) or len(candidate) != 4:
                raise TrajectoryError("B1 K%d prefix capture is incomplete for window %s" % (k, window))
            records = []
            for reference_row, candidate_row, expected_identity in zip(reference, candidate, identities):
                if _proof_identity(reference_row) != expected_identity or _proof_identity(candidate_row) != expected_identity:
                    raise TrajectoryError("B1 prefix identity/order mismatch")
                state = prefix_consistent(
                    reference_row.get("states", []),
                    candidate_row.get("states", []),
                    atol=tolerances["prefix_atol"],
                    rtol=tolerances["prefix_rtol"],
                )
                residual = prefix_consistent(
                    reference_row.get("residuals", []),
                    candidate_row.get("residuals", []),
                    atol=tolerances["prefix_atol"],
                    rtol=tolerances["prefix_rtol"],
                )
                records.append(
                    {
                        "sample_identity": expected_identity,
                        "state_match": state["match"],
                        "state_max_abs": state["max_abs"],
                        "residual_match": residual["match"],
                        "residual_max_abs": residual["max_abs"],
                    }
                )
            comparisons.append(
                {
                    "reference_cell_id": reference_id,
                    "candidate_cell_id": candidate_id,
                    "prefix_steps": 2,
                    "records": records,
                    "all_match": all(row["state_match"] and row["residual_match"] for row in records),
                }
            )
        prefix_windows.append({"window": window, "comparisons": comparisons})

    proof = make_hashed_manifest(
        {
            "schema_version": "loopscope.phase2-b1-admission-proof.v1",
            "artifact_kind": "b1_k1_and_fixed_step_prefix_proof",
            "card_id": card["card_id"],
            "card_manifest_sha256": card["manifest_sha256"],
            "revision": card["science"]["revision"],
            "sample_count": 4,
            "ordered_sample_identity": identities,
            "tolerances": dict(tolerances),
            "k1_equivalence": {
                "windows": k1_windows,
                "all_match": all(row["all_match"] for row in k1_windows),
            },
            "fixed_step_prefix_consistency": {
                "windows": prefix_windows,
                "all_match": all(
                    comparison["all_match"]
                    for row in prefix_windows
                    for comparison in row["comparisons"]
                ),
            },
            "vectors_persisted": False,
        }
    )
    validate_b1_admission_proof(proof, card)
    return proof


def validate_b1_admission_proof(proof: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    from tflt.loopscope.schema import verify_manifest_sha256

    validate_phase2_card(card)
    expected_keys = {
        "schema_version", "artifact_kind", "card_id", "card_manifest_sha256", "revision",
        "sample_count", "ordered_sample_identity", "tolerances", "k1_equivalence",
        "fixed_step_prefix_consistency", "vectors_persisted", "manifest_sha256",
    }
    if not isinstance(proof, Mapping) or set(proof) != expected_keys:
        raise TrajectoryError("B1 proof exact-key contract differs")
    verify_manifest_sha256(proof)
    if proof["schema_version"] != "loopscope.phase2-b1-admission-proof.v1" or proof["artifact_kind"] != "b1_k1_and_fixed_step_prefix_proof":
        raise TrajectoryError("unsupported B1 admission-proof schema")
    if proof["card_id"] != card["card_id"] or proof["card_manifest_sha256"] != card["manifest_sha256"]:
        raise TrajectoryError("B1 proof is bound to a different card")
    if proof["revision"] != card["science"]["revision"] or proof["tolerances"] != card["tolerances"]:
        raise TrajectoryError("B1 proof revision/tolerances differ from the card")
    if proof["sample_count"] != 4 or proof["vectors_persisted"] is not False:
        raise TrajectoryError("B1 proof scale/vector policy differs")
    identities = [_proof_identity({"sample_identity": value}) for value in proof["ordered_sample_identity"]]
    if len(identities) != 4 or len({_identity_key(value) for value in identities}) != 4:
        raise TrajectoryError("B1 proof identities are incomplete or duplicated")
    k1 = proof["k1_equivalence"]
    if not isinstance(k1, Mapping) or set(k1) != {"windows", "all_match"}:
        raise TrajectoryError("B1 K1 proof envelope differs")
    if [row.get("window") for row in k1["windows"]] != card["science"]["windows"]:
        raise TrajectoryError("B1 K1 windows differ from the card")
    k1_all = True
    for window_row, expected_window in zip(k1["windows"], card["science"]["windows"]):
        _trajectory_exact_keys(
            window_row, {"window", "k", "alpha", "records", "all_match"}, "B1 K1 window proof"
        )
        if window_row["window"] != expected_window or window_row["k"] != 1 or window_row["alpha"] != 1.0:
            raise TrajectoryError("B1 K1 window/K/alpha differs")
        records = window_row["records"]
        if not isinstance(records, list) or len(records) != 4:
            raise TrajectoryError("B1 K1 window proof must contain four samples")
        local_all = True
        for record, expected_identity in zip(records, identities):
            _trajectory_exact_keys(
                record,
                {"sample_identity", "scores_allclose", "top1_match", "max_abs_score_difference"},
                "B1 K1 sample proof",
            )
            if _proof_identity(record) != expected_identity:
                raise TrajectoryError("B1 K1 proof identity/order differs")
            if not isinstance(record["scores_allclose"], bool) or not isinstance(record["top1_match"], bool):
                raise TrajectoryError("B1 K1 proof match flags must be boolean")
            maximum = float(record["max_abs_score_difference"])
            if not math.isfinite(maximum) or maximum < 0.0:
                raise TrajectoryError("B1 K1 max-abs proof is invalid")
            local_all = local_all and record["scores_allclose"] and record["top1_match"]
        if window_row["all_match"] is not local_all:
            raise TrajectoryError("B1 K1 window summary disagrees with records")
        k1_all = k1_all and local_all
    if k1["all_match"] is not k1_all:
        raise TrajectoryError("B1 K1 global summary disagrees with windows")
    prefix = proof["fixed_step_prefix_consistency"]
    if not isinstance(prefix, Mapping) or set(prefix) != {"windows", "all_match"}:
        raise TrajectoryError("B1 prefix proof envelope differs")
    if [row.get("window") for row in prefix["windows"]] != card["science"]["windows"]:
        raise TrajectoryError("B1 prefix windows differ from the card")
    prefix_all = True
    for window_row, expected_window in zip(prefix["windows"], card["science"]["windows"]):
        _trajectory_exact_keys(window_row, {"window", "comparisons"}, "B1 prefix window proof")
        if window_row["window"] != expected_window:
            raise TrajectoryError("B1 prefix window differs")
        comparisons = window_row["comparisons"]
        if not isinstance(comparisons, list) or len(comparisons) != 2:
            raise TrajectoryError("B1 prefix proof requires K3 and K4 comparisons")
        reference_id = protocol_cell_id("shared_k2_anchor", expected_window, 2, 1.0)
        for comparison, (k, alpha) in zip(comparisons, ((3, 1.5), (4, 2.0))):
            _trajectory_exact_keys(
                comparison,
                {"reference_cell_id", "candidate_cell_id", "prefix_steps", "records", "all_match"},
                "B1 prefix comparison",
            )
            candidate_id = protocol_cell_id("fixed_step", expected_window, k, alpha)
            if comparison["reference_cell_id"] != reference_id or comparison["candidate_cell_id"] != candidate_id or comparison["prefix_steps"] != 2:
                raise TrajectoryError("B1 prefix comparison cell identity differs")
            records = comparison["records"]
            if not isinstance(records, list) or len(records) != 4:
                raise TrajectoryError("B1 prefix comparison must contain four samples")
            local_all = True
            for record, expected_identity in zip(records, identities):
                _trajectory_exact_keys(
                    record,
                    {"sample_identity", "state_match", "state_max_abs", "residual_match", "residual_max_abs"},
                    "B1 prefix sample proof",
                )
                if _proof_identity(record) != expected_identity:
                    raise TrajectoryError("B1 prefix proof identity/order differs")
                if not isinstance(record["state_match"], bool) or not isinstance(record["residual_match"], bool):
                    raise TrajectoryError("B1 prefix match flags must be boolean")
                for key in ("state_max_abs", "residual_max_abs"):
                    maximum = float(record[key])
                    if not math.isfinite(maximum) or maximum < 0.0:
                        raise TrajectoryError("B1 prefix max-abs proof is invalid")
                local_all = local_all and record["state_match"] and record["residual_match"]
            if comparison["all_match"] is not local_all:
                raise TrajectoryError("B1 prefix comparison summary disagrees with records")
            prefix_all = prefix_all and local_all
    if prefix["all_match"] is not prefix_all:
        raise TrajectoryError("B1 prefix global summary disagrees with comparisons")
    validate_no_persisted_vectors(proof)


def _proof_identity(row: Mapping[str, Any]) -> Dict[str, str]:
    value = row.get("sample_identity")
    if not isinstance(value, Mapping):
        value = row
    return stable_sample_identity(value.get("task"), value.get("doc_id"), value.get("doc_hash"))


def _answer_vector(value: Any, position: int, already_vector: bool = False) -> Tuple[List[float], VectorMetadata]:
    if hasattr(value, "detach"):
        tensor = value.detach()
        shape = tuple(int(item) for item in tensor.shape)
        dtype = str(tensor.dtype)
        device = str(tensor.device)
        if already_vector:
            if len(shape) != 1:
                raise TrajectoryError("native continuation must be a vector")
            selected = tensor.float().cpu().tolist()
        else:
            if len(shape) != 3 or shape[0] != 1 or not 0 <= position < shape[1]:
                raise TrajectoryError("wrapper tensors must be [1,seq,hidden]")
            selected = tensor[0, position, :].float().cpu().tolist()
            shape = (shape[2],)
        return [_finite_float(item, "tensor vector") for item in selected], VectorMetadata(shape, dtype, device)
    raw = list(value)
    if already_vector:
        selected = raw
    else:
        if len(raw) != 1 or not 0 <= position < len(raw[0]):
            raise TrajectoryError("nested wrapper values must be [1,seq,hidden]")
        selected = raw[0][position]
    vector = [_finite_float(item, "vector") for item in selected]
    return vector, VectorMetadata((len(vector),), "float32", "cpu")


def _record_device_tensor_step(active: Dict[str, Any], before: Any, after: Any) -> None:
    """Compute tensor metrics in device float32 and export scalars only."""

    lhs = before.detach()
    rhs = after.detach()
    if tuple(lhs.shape) != tuple(rhs.shape) or lhs.ndim != 3 or int(lhs.shape[0]) != 1:
        raise TrajectoryError("wrapper tensor shape mismatch; expected matching [1,seq,hidden]")
    if lhs.dtype != rhs.dtype:
        raise TrajectoryError("wrapper tensor dtype mismatch")
    if lhs.device != rhs.device:
        raise TrajectoryError("wrapper tensor device mismatch")
    position = active["answer_position"]
    if not 0 <= position < int(lhs.shape[1]):
        raise TrajectoryError("answer position is outside wrapper tensor")
    before_vector = lhs[0, position, :].float()
    residual = rhs[0, position, :].float() - before_vector
    native = active["native"]
    if not hasattr(native, "detach"):
        raise TrajectoryError("device trace requires device-resident native continuation")
    native_meta = active["native_meta"]
    residual_meta = VectorMetadata(
        (int(residual.shape[0]),), str(lhs.dtype), str(lhs.device)
    )
    _require_same_metadata(residual_meta, native_meta)
    if not bool(residual.isfinite().all().item()) or not bool(before_vector.isfinite().all().item()):
        raise TrajectoryError("wrapper tensor contains NaN/Inf")
    residual_norm = float(residual.norm().item())
    state_norm = float(before_vector.norm().item())
    if state_norm < NCA_MIN_NORM:
        raise TrajectoryError("state norm below 1e-8")
    t = len(active["steps"])
    previous = active["previous_residual"]
    ratio = None
    adjacent_cosine = None
    adjacent_cosine_valid = None
    if previous is not None:
        previous_norm = float(previous.norm().item())
        if previous_norm < NCA_MIN_NORM:
            raise TrajectoryError("previous residual norm below 1e-8")
        ratio = residual_norm / previous_norm
        if residual_norm >= NCA_MIN_NORM:
            adjacent_cosine = max(
                -1.0,
                min(1.0, float((residual @ previous / (residual.norm() * previous.norm())).item())),
            )
            adjacent_cosine_valid = True
        else:
            adjacent_cosine_valid = False
    nca = None
    if t >= 1:
        native32 = native.float()
        native_norm = float(native32.norm().item())
        valid = residual_norm >= NCA_MIN_NORM and native_norm >= NCA_MIN_NORM
        nca = {
            "value": (
                max(
                    -1.0,
                    min(1.0, float((residual @ native32 / (residual.norm() * native32.norm())).item())),
                )
                if valid
                else None
            ),
            "residual_norm": residual_norm,
            "native_continuation_norm": native_norm,
            "valid": valid,
            "invalid_reason": None if valid else "norm_below_1e-8",
            "dtype": "float32",
            "provenance": "actual_loop_repeated_step",
            "body_call_t": t,
        }
    active["steps"].append(
        {
            "body_call_t": t,
            "residual_norm": residual_norm,
            "state_norm": state_norm,
            "relative_activity": residual_norm / state_norm,
            "residual_ratio_to_previous": ratio,
            "adjacent_residual_cosine": adjacent_cosine,
            "adjacent_residual_cosine_valid": adjacent_cosine_valid,
            "nca": nca,
            "valid": True,
        }
    )
    active["admission_states"].append(before_vector.detach().clone())
    active["admission_residuals"].append(residual.detach().clone())
    active["previous_residual"] = residual
    active["previous_meta"] = residual_meta


def _require_same_metadata(left: VectorMetadata, right: VectorMetadata) -> None:
    if left.shape != right.shape:
        raise TrajectoryError("tensor shape mismatch")
    if left.dtype != right.dtype:
        raise TrajectoryError("tensor dtype mismatch")
    if left.device != right.device:
        raise TrajectoryError("tensor device mismatch")


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(math.fsum(_finite_float(value, "norm value") ** 2 for value in values))


def _finite_float(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TrajectoryError("%s contains NaN/Inf" % context)
    return result


__all__ = [
    "InMemoryProbeSession",
    "NCA_MIN_NORM",
    "Phase2TrajectoryCollector",
    "TrajectoryError",
    "VectorMetadata",
    "baseline_native_alignment",
    "build_b1_admission_proof",
    "add_probe_phase2_trajectory_args",
    "cmd_probe_phase2_trajectory",
    "cosine_measurement",
    "k1_choice_equivalent",
    "materialize_phase2_probe_artifacts",
    "prefix_consistent",
    "run_phase2_trajectory_probe",
    "validate_b1_admission_proof",
    "validate_probe_aggregate",
]
