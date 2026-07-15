"""Gate D fixed-horizon K1--K4 scalar trajectory producer.

This module is deliberately separate from the frozen H1 implementation.  It
uses the existing reversible wrapper and consumes its audit events, while
temporary hooks observe the five boundaries of each raw 12:15 block call.
Hidden, residual, and native-continuation vectors are retained only in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase2_schema import (
    CHOICE_LABELS,
    DIRECT_PROBE_SCORE_SOURCE,
    atomic_write_new_json,
    file_sha256,
    make_hashed_manifest,
    ordered_identity_sha256,
    stable_sample_identity,
    validate_no_persisted_vectors,
)


CARD_ID = "H2_FIXED_HORIZON_STEPWISE_PROFILE_V1"
EXECUTOR_THREAD_ID = "019f668f-e98d-7033-a28a-36d55d0cf5ac"
CONTROL_SHA256 = "ce7034b6274e7ae6d561f22eee515a093c3965a84ddcb6916edb64df71a6f7c7"
WINDOW = "12:15"
K_VALUES = (1, 2, 3, 4)
NCA_MIN_NORM = 1e-8


class FixedHorizonError(ValueError):
    """Raised when Gate D evidence cannot satisfy the frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_fixed_horizon_card(card: Mapping[str, Any]) -> None:
    """Validate the exact material science/statistics contract for Gate D."""

    required = {
        "schema_version",
        "card_id",
        "role",
        "control_sha256",
        "executor_thread_id",
        "science",
        "measurement",
        "statistics",
        "source_namespaces",
        "write_once_contract",
        "tolerances",
        "implementation",
        "manifest_sha256",
    }
    if set(card) != required:
        raise FixedHorizonError("fixed-horizon card keys differ from the frozen contract")
    if card["schema_version"] != "loopscope.h2-fixed-horizon-stepwise-card.v1":
        raise FixedHorizonError("unsupported fixed-horizon card schema")
    if card["card_id"] != CARD_ID or card["role"] != "post_hoc_exploratory_mechanism_profile":
        raise FixedHorizonError("fixed-horizon card identity/role differs")
    if card["control_sha256"] != CONTROL_SHA256 or card["executor_thread_id"] != EXECUTOR_THREAD_ID:
        raise FixedHorizonError("card control/executor binding differs")
    science = card["science"]
    expected_science = {
        "model_alias": "qwen3-1.7b-base",
        "model_repo": "Qwen/Qwen3-1.7B-Base",
        "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "task": "mmlu",
        "num_fewshot": 5,
        "dtype": "float16",
        "window": WINDOW,
        "window_inclusive_width": 4,
        "protocol": "fixed_horizon",
        "iteration_mode": "block",
        "strategy": "damped_euler",
        "k_values": [1, 2, 3, 4],
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "last",
        "decode_mode": "bypass",
        "total_horizon": 1.0,
        "calibration_count": 512,
        "maximum_new_full_run_count": 0,
        "gpu_count": 1,
        "gpu_type": "A40",
    }
    if science != expected_science:
        raise FixedHorizonError("card science differs from H2_FIXED_HORIZON_STEPWISE_PROFILE_V1")
    statistics_contract = {
        "new_scalar_bootstrap": {"replicates": 2000, "seed": 20260716, "confidence": 0.95, "method": "percentile"},
        "nca_bootstrap": {"replicates": 10000, "seed": 0},
        "full_paired_bootstrap": {"replicates": 2000, "seed": 20260710},
        "effective_rank": {"estimator_version": 2, "interval": "point_estimate_only"},
    }
    if card["statistics"] != statistics_contract:
        raise FixedHorizonError("card statistics differ from the frozen contract")
    measurement = card["measurement"]
    if measurement.get("body_call_indexing") != "zero_based_t=0..K-1":
        raise FixedHorizonError("body-call indexing differs")
    if measurement.get("step_size") != "h=alpha/K=1/K" or measurement.get("tau") != {"entry": "t/K", "applied": "(t+1)/K"}:
        raise FixedHorizonError("step-size/tau contract differs")
    if measurement.get("choice_score_source") != DIRECT_PROBE_SCORE_SOURCE:
        raise FixedHorizonError("direct-probe score namespace differs")
    if measurement.get("full_score_source") != "lm_eval_acc_none_raw_per_choice_loglikelihood":
        raise FixedHorizonError("full score namespace differs")
    if measurement.get("kl_orientation") != "KL(stage_choice_distribution || native_no_loop_reference)":
        raise FixedHorizonError("KL orientation differs")
    if measurement.get("flatness") != "-population_std([H(x_t),H(L12),H(L13),H(L14),H(z_t)])":
        raise FixedHorizonError("flatness definition differs")
    if card["source_namespaces"] != {
        "calibration": "phase1_frozen_validation_512_direct_probe",
        "full": "phase1_gatec_full_14042_lm_eval",
        "cross_namespace_sample_join": False,
    }:
        raise FixedHorizonError("source namespace separation differs")
    _verify_manifest(card)


def verify_implementation_hashes(card: Mapping[str, Any], repo_root: Path) -> None:
    validate_fixed_horizon_card(card)
    files = card["implementation"].get("files")
    if not isinstance(files, Mapping) or not files:
        raise FixedHorizonError("card implementation hash map is empty")
    for relative, expected in files.items():
        path = (Path(repo_root) / str(relative)).resolve()
        if file_sha256(path) != expected:
            raise FixedHorizonError("implementation hash drift: %s" % relative)


def _verify_manifest(payload: Mapping[str, Any]) -> None:
    expected = make_hashed_manifest({key: value for key, value in payload.items() if key != "manifest_sha256"})
    if payload.get("manifest_sha256") != expected["manifest_sha256"]:
        raise FixedHorizonError("manifest_sha256 mismatch")


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FixedHorizonError("%s contains NaN/Inf" % context)
    return result


def _probabilities(scores: Sequence[float]) -> List[float]:
    values = [_finite(value, "choice score") for value in scores]
    if len(values) != 4:
        raise FixedHorizonError("choice lens requires exactly four scores")
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = math.fsum(weights)
    return [value / total for value in weights]


def _entropy(probabilities: Sequence[float]) -> float:
    values = [_finite(value, "choice probability") for value in probabilities]
    if len(values) != 4 or any(value < 0.0 for value in values):
        raise FixedHorizonError("choice probabilities are invalid")
    return -math.fsum(value * math.log(value) for value in values if value > 0.0)


def _kl_to_reference(probabilities: Sequence[float], reference: Sequence[float]) -> float:
    lhs = [_finite(value, "KL distribution") for value in probabilities]
    rhs = [_finite(value, "KL reference") for value in reference]
    if len(lhs) != 4 or len(rhs) != 4 or any(value <= 0.0 for value in rhs):
        raise FixedHorizonError("KL distributions are invalid")
    return math.fsum(p * math.log(p / q) for p, q in zip(lhs, rhs) if p > 0.0)


def lens_metrics_from_scores(scores: Sequence[float]) -> Dict[str, Any]:
    """Pure four-choice lens semantics used by GPU code and local tests."""

    values = [_finite(value, "choice score") for value in scores]
    probabilities = _probabilities(values)
    order = sorted(range(4), key=lambda index: (-values[index], index))
    return {
        "choice_probabilities": probabilities,
        "entropy_nats": _entropy(probabilities),
        "top_margin_raw": values[order[0]] - values[order[1]],
        "top1_index": order[0],
    }


def scalar_step_from_vectors(
    entry: Sequence[float],
    raw: Sequence[float],
    *,
    h: float,
    previous_residual: Optional[Sequence[float]],
    native_continuation: Optional[Sequence[float]],
) -> Dict[str, Any]:
    """Pure-Python scalar contract fixture; actual GPU code mirrors it."""

    lhs = [_finite(value, "entry vector") for value in entry]
    rhs = [_finite(value, "raw vector") for value in raw]
    if not lhs or len(lhs) != len(rhs) or not 0.0 < h <= 1.0:
        raise FixedHorizonError("entry/raw vector shape or h is invalid")
    residual = [after - before for before, after in zip(lhs, rhs)]
    state_norm = math.sqrt(math.fsum(value * value for value in lhs))
    residual_norm = math.sqrt(math.fsum(value * value for value in residual))
    if state_norm < NCA_MIN_NORM:
        raise FixedHorizonError("entry state norm below 1e-8")
    ratio = cosine = nca = None
    if previous_residual is not None:
        previous = [_finite(value, "previous residual") for value in previous_residual]
        previous_norm = math.sqrt(math.fsum(value * value for value in previous))
        if len(previous) != len(residual) or previous_norm < NCA_MIN_NORM:
            raise FixedHorizonError("previous residual is invalid")
        ratio = residual_norm / previous_norm
        cosine = _cosine(residual, previous)
        if native_continuation is not None:
            nca = _cosine(residual, native_continuation)
    return {
        "raw_residual_norm": residual_norm,
        "raw_relative_activity": residual_norm / state_norm,
        "residual_ratio_to_previous": ratio,
        "adjacent_residual_cosine": cosine,
        "repeated_step_nca": nca,
        "h_scaled_applied_update_norm": h * residual_norm,
        "h_scaled_applied_relative_magnitude": h * residual_norm / state_norm,
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    lhs = [_finite(value, "cosine vector") for value in left]
    rhs = [_finite(value, "cosine vector") for value in right]
    if len(lhs) != len(rhs):
        raise FixedHorizonError("cosine vector shapes differ")
    left_norm = math.sqrt(math.fsum(value * value for value in lhs))
    right_norm = math.sqrt(math.fsum(value * value for value in rhs))
    if left_norm < NCA_MIN_NORM or right_norm < NCA_MIN_NORM:
        return None
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(lhs, rhs)) / (left_norm * right_norm)))


class FixedHorizonStepwiseCollector:
    """Audit collector plus temporary five-boundary hook consumer."""

    def __init__(
        self,
        *,
        torch_module: Any,
        k: int,
        lens: Callable[[Any], Dict[str, Any]],
    ) -> None:
        if k not in K_VALUES:
            raise FixedHorizonError("K must be one of 1,2,3,4")
        self.torch = torch_module
        self.k = int(k)
        self.h = 1.0 / self.k
        self.lens = lens
        self.samples: List[Dict[str, Any]] = []
        self.erank_vectors: Dict[Tuple[int, str], List[Any]] = {
            (t, stage): [] for t in range(self.k) for stage in ("entry", "raw", "applied")
        }
        self._active: Optional[Dict[str, Any]] = None

    def begin_sample(
        self,
        *,
        identity: Mapping[str, Any],
        answer_position: int,
        native_continuation: Any,
        reference_probabilities: Sequence[float],
    ) -> None:
        if self._active is not None:
            raise FixedHorizonError("begin_sample before prior sample ended")
        native = native_continuation.detach().float()
        if native.ndim != 1 or not bool(native.isfinite().all().item()):
            raise FixedHorizonError("native continuation vector is invalid")
        reference = [_finite(value, "reference probability") for value in reference_probabilities]
        if len(reference) != 4 or any(value <= 0.0 for value in reference):
            raise FixedHorizonError("native no-loop reference distribution is invalid")
        self._active = {
            "identity": dict(identity),
            "answer_position": int(answer_position),
            "native": native,
            "reference": reference,
            "events": {},
            "timeline": [],
            "steps": [],
            "previous_residual": None,
            "x0": None,
            "cumulative_path": 0.0,
            "current_boundaries": None,
            "ignore_stash": False,
            "loop_boundary_capture_count": 0,
            "stash_boundary_capture_count": 0,
        }

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        active = self._require_active()
        active["events"][event] = active["events"].get(event, 0) + 1
        active["timeline"].append(event)
        if event == "stash_pass":
            if active["current_boundaries"] is not None:
                raise FixedHorizonError("stash began with an unfinished raw call")
            active["ignore_stash"] = True

    def hook_entry(self, module: Any, inputs: Sequence[Any]) -> None:
        del module
        active = self._require_active()
        if not inputs:
            raise FixedHorizonError("first layer pre-hook lacks hidden state")
        vector = self._answer_vector(inputs[0])
        if active["ignore_stash"]:
            active["stash_boundary_capture_count"] += 1
            return
        if active["current_boundaries"] is not None:
            raise FixedHorizonError("raw call boundaries overlap")
        active["current_boundaries"] = [vector]
        active["loop_boundary_capture_count"] += 1

    def hook_layer(self, layer_offset: int) -> Callable[[Any, Sequence[Any], Any], None]:
        def callback(module: Any, inputs: Sequence[Any], output: Any) -> None:
            del module, inputs
            active = self._require_active()
            vector = self._answer_vector(_hidden(output))
            if active["ignore_stash"]:
                active["stash_boundary_capture_count"] += 1
                if layer_offset == 3:
                    active["ignore_stash"] = False
                return
            current = active["current_boundaries"]
            if current is None or len(current) != layer_offset + 1:
                raise FixedHorizonError("five-boundary hook order differs")
            current.append(vector)
            active["loop_boundary_capture_count"] += 1

        return callback

    def record_tensor_diff(self, name: str, before: Any, after: Any) -> None:
        active = self._require_active()
        active["events"][name] = active["events"].get(name, 0) + 1
        active["timeline"].append(name)
        if name != "g_minus_x":
            return
        entry = self._answer_vector(before)
        raw = self._answer_vector(after)
        boundaries = active["current_boundaries"]
        if boundaries is None or len(boundaries) != 5:
            raise FixedHorizonError("raw block call lacks exactly five boundary vectors")
        if not self._allclose(entry, boundaries[0]) or not self._allclose(raw, boundaries[-1]):
            raise FixedHorizonError("wrapper residual and five-boundary hooks disagree")
        t = len(active["steps"])
        if t >= self.k:
            raise FixedHorizonError("more g_minus_x events than K")
        residual = raw - entry
        state_norm = float(entry.norm().item())
        residual_norm = float(residual.norm().item())
        if state_norm < NCA_MIN_NORM:
            raise FixedHorizonError("entry state norm below 1e-8")
        previous = active["previous_residual"]
        ratio = adjacent_cosine = None
        adjacent_validity = "not_applicable" if t == 0 else "valid"
        nca_payload: Dict[str, Any]
        if t == 0:
            nca_payload = {"value": None, "valid": None, "reason": "not_applicable_initial_body_call"}
        else:
            previous_norm = float(previous.norm().item())
            if previous_norm < NCA_MIN_NORM:
                raise FixedHorizonError("previous residual norm below 1e-8")
            ratio = residual_norm / previous_norm
            adjacent_cosine = _torch_cosine(residual, previous)
            if adjacent_cosine is None:
                adjacent_validity = "invalid_norm_below_1e-8"
            native = active["native"].to(device=residual.device)
            nca_value = _torch_cosine(residual, native)
            nca_payload = {
                "value": nca_value,
                "valid": nca_value is not None,
                "reason": None if nca_value is not None else "norm_below_1e-8",
            }
        applied_native = entry.to(dtype=before.dtype) + self.h * (raw.to(dtype=before.dtype) - entry.to(dtype=before.dtype))
        applied = applied_native.float()
        actual_delta = applied - entry
        theoretical_applied = entry + self.h * residual
        rounding_gap = float((applied - theoretical_applied).norm().item())
        if active["x0"] is None:
            active["x0"] = entry.detach().clone()
        active["cumulative_path"] += self.h * residual_norm
        lens_rows = [self.lens(vector) for vector in boundaries]
        entry_lens = lens_rows[0]
        raw_lens = lens_rows[-1]
        applied_lens = self.lens(applied)
        entropy_curve = [row["entropy_nats"] for row in lens_rows]
        entry_kl = _kl_to_reference(entry_lens["choice_probabilities"], active["reference"])
        raw_kl = _kl_to_reference(raw_lens["choice_probabilities"], active["reference"])
        applied_kl = _kl_to_reference(applied_lens["choice_probabilities"], active["reference"])
        tau_entry = t / float(self.k)
        tau_applied = (t + 1) / float(self.k)
        step = {
            "body_call_t": t,
            "tau_entry": tau_entry,
            "tau_applied": tau_applied,
            "h": self.h,
            "raw_residual_norm": residual_norm,
            "entry_state_norm": state_norm,
            "raw_relative_activity": residual_norm / state_norm,
            "residual_ratio_to_previous": ratio,
            "adjacent_residual_cosine": adjacent_cosine,
            "adjacent_direction_validity": adjacent_validity,
            "repeated_step_nca": nca_payload,
            "h_scaled_applied_update_norm": self.h * residual_norm,
            "h_scaled_applied_relative_magnitude": self.h * residual_norm / state_norm,
            "actual_applied_state_delta_norm": float(actual_delta.norm().item()),
            "applied_state_rounding_gap_norm": rounding_gap,
            "cumulative_h_scaled_path_length": active["cumulative_path"],
            "cumulative_applied_displacement_norm": float((applied - active["x0"]).norm().item()),
            "intermediate_lens_entry": entry_lens,
            "intermediate_lens_raw_exit": raw_lens,
            "intermediate_lens_applied_state": applied_lens,
            "raw_entropy_drop_nats": entry_lens["entropy_nats"] - raw_lens["entropy_nats"],
            "applied_entropy_drop_nats": entry_lens["entropy_nats"] - applied_lens["entropy_nats"],
            "raw_call_boundary_entropy_curve_nats": entropy_curve,
            "raw_call_entropy_flatness_negative_population_std": -statistics.pstdev(entropy_curve),
            "entry_kl_to_native_no_loop_reference_nats": entry_kl,
            "raw_kl_to_native_no_loop_reference_nats": raw_kl,
            "applied_kl_to_native_no_loop_reference_nats": applied_kl,
            "raw_kl_to_reference_drop_nats": entry_kl - raw_kl,
            "applied_kl_to_reference_drop_nats": entry_kl - applied_kl,
            "finite": True,
        }
        self.erank_vectors[(t, "entry")].append(entry.detach().cpu())
        self.erank_vectors[(t, "raw")].append(raw.detach().cpu())
        self.erank_vectors[(t, "applied")].append(applied.detach().cpu())
        active["steps"].append(step)
        active["previous_residual"] = residual.detach().clone()
        active["current_boundaries"] = None

    def end_sample(self, final_output: Mapping[str, Any]) -> Dict[str, Any]:
        active = self._require_active()
        if active["current_boundaries"] is not None or active["ignore_stash"]:
            raise FixedHorizonError("sample ended with unfinished hook state")
        if len(active["steps"]) != self.k:
            raise FixedHorizonError("sample scalar step count differs from K")
        expected = {
            "wrapper_forward": 1,
            "body_call": self.k,
            "g_minus_x": self.k,
            "looped_hidden_vs_input": 1,
            "stash_pass": 1,
            "identity_forward": 3,
        }
        for key, value in expected.items():
            if active["events"].get(key, 0) != value:
                raise FixedHorizonError("wrapper event count differs for %s" % key)
        if active["loop_boundary_capture_count"] != 5 * self.k or active["stash_boundary_capture_count"] != 5:
            raise FixedHorizonError("five-boundary hook capture count differs")
        sample = {
            "sample_identity": active["identity"],
            "source_namespace": "phase1_frozen_validation_512_direct_probe",
            "position_rule": "final_pre_answer_prompt_token",
            "window": WINDOW,
            "k": self.k,
            "alpha": 1.0,
            "h": self.h,
            "total_horizon": 1.0,
            "body_call_count": self.k,
            "steps": active["steps"],
            "final_direct_probe_output": dict(final_output),
            "hook_capture_proof": {
                "loop_boundary_capture_count": active["loop_boundary_capture_count"],
                "expected_loop_boundary_capture_count": 5 * self.k,
                "stash_boundary_capture_count": active["stash_boundary_capture_count"],
            },
            "valid": True,
        }
        validate_no_persisted_vectors(sample)
        self.samples.append(sample)
        self._active = None
        return sample

    def effective_rank_summary(self) -> List[Dict[str, Any]]:
        from tflt.loopscope.probe import compute_effective_rank

        result = []
        x0_rank: Optional[float] = None
        for t in range(self.k):
            ranks = {
                stage: compute_effective_rank(self.torch, self.erank_vectors[(t, stage)])["effective_rank"]
                for stage in ("entry", "raw", "applied")
            }
            if x0_rank is None:
                x0_rank = ranks["entry"]
            result.append(
                {
                    "body_call_t": t,
                    "tau_entry": t / float(self.k),
                    "tau_applied": (t + 1) / float(self.k),
                    "entry_effective_rank": ranks["entry"],
                    "raw_exit_effective_rank": ranks["raw"],
                    "applied_state_effective_rank": ranks["applied"],
                    "raw_delta_from_entry": ranks["raw"] - ranks["entry"],
                    "applied_delta_from_entry": ranks["applied"] - ranks["entry"],
                    "entry_delta_from_x0": ranks["entry"] - x0_rank,
                    "raw_delta_from_x0": ranks["raw"] - x0_rank,
                    "applied_delta_from_x0": ranks["applied"] - x0_rank,
                    "estimator": "unit_normalize_center_across_samples_squared_singular_values_shannon_erank_v2",
                    "sample_count": len(self.erank_vectors[(t, "entry")]),
                }
            )
        return result

    def _answer_vector(self, tensor: Any) -> Any:
        value = tensor.detach()
        position = self._require_active()["answer_position"]
        if value.ndim != 3 or int(value.shape[0]) != 1 or not 0 <= position < int(value.shape[1]):
            raise FixedHorizonError("hook tensor must be [1,seq,hidden] at answer position")
        vector = value[0, position, :].float()
        if not bool(vector.isfinite().all().item()):
            raise FixedHorizonError("hook vector contains NaN/Inf")
        return vector

    def _allclose(self, left: Any, right: Any) -> bool:
        return bool(self.torch.allclose(left, right, atol=1e-3, rtol=1e-4))

    def _require_active(self) -> Dict[str, Any]:
        if self._active is None:
            raise FixedHorizonError("collector event outside active sample")
        return self._active


def register_five_boundary_hooks(handle: Any, collector: FixedHorizonStepwiseCollector) -> List[Any]:
    originals = [handle.originals[index] for index in sorted(handle.originals)]
    if len(originals) != 4:
        raise FixedHorizonError("12:15 block must expose exactly four original layers")
    handles = [originals[0].register_forward_pre_hook(collector.hook_entry)]
    handles.extend(layer.register_forward_hook(collector.hook_layer(offset)) for offset, layer in enumerate(originals))
    return handles


def remove_hooks(handles: Sequence[Any]) -> Dict[str, Any]:
    for handle in handles:
        handle.remove()
    return {
        "hook_handles_registered": len(handles),
        "hook_handles_removed": len(handles),
        "hooks_active_after_removal": 0,
    }


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def _torch_cosine(left: Any, right: Any) -> Optional[float]:
    if left.shape != right.shape or left.device != right.device:
        raise FixedHorizonError("cosine tensor shape/device mismatch")
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm < NCA_MIN_NORM or right_norm < NCA_MIN_NORM:
        return None
    return max(-1.0, min(1.0, float((left @ right / (left.norm() * right.norm())).item())))


def _record_identity(record: Mapping[str, Any]) -> Dict[str, str]:
    return stable_sample_identity(
        str(record["task_name"]), str(record["target_doc_id"]), str(record["target_doc_sha256"])
    )


def _identity_key(identity: Mapping[str, Any]) -> str:
    return "%s\0%s\0%s" % (identity["task"], identity["doc_id"], identity["doc_hash"])


def _resource_start(torch: Any, device: Any) -> float:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _resource_finish(torch: Any, device: Any, started: float, name: str) -> Dict[str, Any]:
    peak = 0
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    return {"name": name, "wall_clock_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": peak}


def _choice_from_logits(logits: Any, position: int, choice_ids: Sequence[int], identity: Mapping[str, Any], torch: Any) -> Dict[str, Any]:
    from tflt.loopscope.phase2_analysis import choice_output

    scores = torch.log_softmax(logits[0, position, :].float(), dim=-1)[list(choice_ids)].detach().cpu().tolist()
    return choice_output(scores, identity=identity, score_source=DIRECT_PROBE_SCORE_SOURCE)


def run_fixed_horizon_probe(args: Any, card: Mapping[str, Any]) -> Dict[str, Any]:
    """Load one model session and execute no-loop plus K1--K4 in order."""

    required_offline = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    if any(os.environ.get(key) != value for key, value in required_offline.items()):
        raise FixedHorizonError("all model/data runs require the three frozen offline flags")
    if os.environ.get("HF_ENDPOINT"):
        raise FixedHorizonError("HF_ENDPOINT must be unset for Gate D")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - remote-only dependency path
        raise RuntimeError("fixed-horizon probe requires torch and transformers") from exc
    from tflt.config import LoopConfig
    from tflt.loopscope.phase2_trajectory import k1_choice_equivalent
    from tflt.loopscope.probe import (
        choice_tokenization,
        decoder_boundary_states,
        find_final_norm,
        load_probe_records,
        probe_pool_metadata,
        resolve_answer_position,
        select_device,
        torch_dtype,
    )
    from tflt.loopscope.revisions import load_tokenizer_with_resolved_commit, strict_revision_closure
    from tflt.models import resolve_model
    from tflt.wrapper import apply_loop_wrapper

    validate_fixed_horizon_card(card)
    expected_count = 4 if args.mode == "d1-smoke" else 512
    records = load_probe_records(Path(args.input_jsonl), 4 if args.mode == "d1-smoke" else None)
    if len(records) != expected_count:
        raise FixedHorizonError("mode requires exactly %d frozen records" % expected_count)
    pool, warnings = probe_pool_metadata(records, args.input_manifest)
    if pool["count"] != expected_count or pool["uses_target_gold_labels"] is not False:
        raise FixedHorizonError("probe pool scale/label state differs")
    if args.mode == "d2-profile":
        if not args.d1_noninterference:
            raise FixedHorizonError("D2 requires a passing D1 noninterference proof")
        proof = json.loads(Path(args.d1_noninterference).read_text(encoding="utf-8"))
        _verify_manifest(proof)
        if proof.get("all_pass") is not True or proof.get("card_manifest_sha256") != card["manifest_sha256"]:
            raise FixedHorizonError("D1 noninterference admission proof is not pass-ready")
    model_info = resolve_model(card["science"]["model_alias"])
    load_kwargs = {"trust_remote_code": True, "revision": card["science"]["revision"]}
    tokenizer, tokenizer_revision = load_tokenizer_with_resolved_commit(AutoTokenizer, model_info["repo_id"], load_kwargs)
    _, choice_ids, _ = choice_tokenization(tokenizer, CHOICE_LABELS)
    device = select_device(torch, args.device)
    if not str(device).startswith("cuda"):
        raise FixedHorizonError("Gate D scientific probe requires CUDA")
    visible_device_count = int(torch.cuda.device_count())
    gpu_name = str(torch.cuda.get_device_name(device))
    if "A40" not in gpu_name.upper():
        raise FixedHorizonError("Gate D requires one A40; observed %s" % gpu_name)
    if visible_device_count != 1:
        raise FixedHorizonError("Gate D requires exactly one visible A40")
    model = AutoModelForCausalLM.from_pretrained(
        model_info["repo_id"], torch_dtype=torch_dtype(torch, card["science"]["dtype"]), **load_kwargs
    )
    model.eval()
    model.to(device)
    closure = strict_revision_closure(model, tokenizer_revision, card["science"]["revision"])
    layer_count = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if layer_count <= 15:
        raise FixedHorizonError("loaded model does not contain inclusive window 12:15")
    final_norm = find_final_norm(model)
    lm_head = model.get_output_embeddings()

    def lens(vector: Any) -> Dict[str, Any]:
        hidden = final_norm(vector.to(dtype=next(final_norm.parameters()).dtype).reshape(1, 1, -1))
        scores = lm_head(hidden)[0, 0, list(choice_ids)].float().detach().cpu().tolist()
        return lens_metrics_from_scores(scores)

    native_store: Dict[str, Any] = {}
    reference_store: Dict[str, List[float]] = {}
    baseline_rows = []
    resources = []
    baseline_started = _resource_start(torch, device)
    with torch.inference_mode():
        for record in records:
            encoded = tokenizer(record["text"], return_tensors="pt")
            inputs = {key: value.to(device) for key, value in encoded.items()}
            position = resolve_answer_position(inputs["attention_mask"], record)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            boundaries = decoder_boundary_states(tuple(outputs.hidden_states or ()), layer_count)
            identity = _record_identity(record)
            key = _identity_key(identity)
            after_native = boundaries[16][0, position, :].detach()
            before_native = boundaries[12][0, position, :].detach()
            native = (boundaries[-1][0, position, :].detach() - after_native).float()
            after = after_native.float()
            before = before_native.float()
            native_store[key] = native.cpu()
            output = _choice_from_logits(outputs.logits, position, choice_ids, identity, torch)
            reference_store[key] = list(output["choice_probabilities"])
            baseline_nca = _torch_cosine(after - before, native)
            baseline_rows.append(
                {
                    "sample_identity": identity,
                    "source_namespace": "phase1_frozen_validation_512_direct_probe",
                    "reference_role": "native_no_loop_final_direct_probe_distribution",
                    "final_direct_probe_output": output,
                    "baseline_window_update_norm": float((after - before).norm().item()),
                    "native_continuation_norm": float(native.norm().item()),
                    "baseline_nca": {"value": baseline_nca, "valid": baseline_nca is not None},
                }
            )
            del outputs, boundaries, encoded, inputs, before, after, native
    resources.append(_resource_finish(torch, device, baseline_started, "native_no_loop_reference"))

    cells = []
    total_calls = 0
    for k in K_VALUES:
        started = _resource_start(torch, device)
        collector = FixedHorizonStepwiseCollector(torch_module=torch, k=k, lens=lens)
        config = LoopConfig.from_window_string(
            model_alias=card["science"]["model_alias"], window=WINDOW, k=k,
            iteration_mode="block", strategy="damped_euler", alpha=1.0, beta=0.0,
            cache_strategy="last", decode_mode="bypass", audit_collector=collector,
        )
        handle = apply_loop_wrapper(model, config)
        hook_handles = register_five_boundary_hooks(handle, collector)
        hook_proof: Dict[str, Any]
        try:
            with torch.inference_mode():
                for record in records:
                    encoded = tokenizer(record["text"], return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in encoded.items()}
                    position = resolve_answer_position(inputs["attention_mask"], record)
                    identity = _record_identity(record)
                    key = _identity_key(identity)
                    collector.begin_sample(
                        identity=identity,
                        answer_position=position,
                        native_continuation=native_store[key].to(device),
                        reference_probabilities=reference_store[key],
                    )
                    outputs = model(**inputs, use_cache=False, return_dict=True)
                    final_output = _choice_from_logits(outputs.logits, position, choice_ids, identity, torch)
                    collector.end_sample(final_output)
                    del outputs, inputs, encoded
        finally:
            hook_proof = remove_hooks(hook_handles)
            handle.restore()
        total_calls += expected_count * k
        resource = _resource_finish(torch, device, started, "fixed_horizon_k%d" % k)
        resource.update(hook_proof)
        resources.append(resource)
        cells.append(
            {
                "cell": {"protocol": "fixed_horizon", "window": WINDOW, "k": k, "alpha": 1.0, "h": 1.0 / k},
                "samples": collector.samples,
                "cohort_effective_rank": collector.effective_rank_summary(),
                "hook_lifecycle": hook_proof,
                "restore_called": True,
                "restore_completed": handle.active is False,
            }
        )
    native_store.clear()
    k1_rows = cells[0]["samples"]
    k1_checks = []
    for baseline, candidate in zip(baseline_rows, k1_rows):
        left = baseline["final_direct_probe_output"]["raw_choice_scores"]
        right = candidate["final_direct_probe_output"]["raw_choice_scores"]
        k1_checks.append(k1_choice_equivalent(left, right, atol=card["tolerances"]["k1_choice_atol"], rtol=card["tolerances"]["k1_choice_rtol"]))
    if not all(k1_checks):
        raise FixedHorizonError("K1 final direct-probe output differs from native no-loop reference")
    result = {
        "mode": args.mode,
        "card": dict(card),
        "pool": pool,
        "warnings": warnings,
        "revision_closure": closure,
        "gpu": {"name": gpu_name, "visible_device_count": visible_device_count},
        "baseline_reference_samples": baseline_rows,
        "cells": cells,
        "session_contract": {
            "model_process_count": 1,
            "model_load_count": 1,
            "k_order": [1, 2, 3, 4],
            "sample_count": expected_count,
            "total_body_calls": total_calls,
            "expected_total_body_calls": expected_count * sum(K_VALUES),
            "native_vectors_persisted": False,
            "hidden_vectors_persisted": False,
            "residual_vectors_persisted": False,
        },
        "k1_noninterference": {
            "sample_count": expected_count,
            "all_choice_scores_match": all(k1_checks),
            "atol": card["tolerances"]["k1_choice_atol"],
            "rtol": card["tolerances"]["k1_choice_rtol"],
        },
        "resource_usage": resources,
    }
    validate_no_persisted_vectors(result)
    return result


def materialize_probe(output_dir: Path, args: Any, result: Mapping[str, Any]) -> Mapping[str, Any]:
    card = result["card"]
    atomic_write_new_json(output_dir / "fixed_horizon_stepwise_card.json", card)
    baseline = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-native-no-loop-reference.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "label_state": "sealed",
            "sample_count": len(result["baseline_reference_samples"]),
            "samples": result["baseline_reference_samples"],
            "vectors_persisted": False,
        }
    )
    atomic_write_new_json(output_dir / "native_no_loop_reference.json", baseline)
    samples_dir = output_dir / "calibration_stepwise_samples"
    samples_dir.mkdir(parents=False, exist_ok=False)
    cell_refs = []
    erank = []
    for cell in result["cells"]:
        k = cell["cell"]["k"]
        payload = make_hashed_manifest(
            {
                "schema_version": "loopscope.h2-fixed-horizon-stepwise-cell.v1",
                "card_manifest_sha256": card["manifest_sha256"],
                "label_state": "sealed",
                "source_namespace": "phase1_frozen_validation_512_direct_probe",
                "cell": cell["cell"],
                "sample_count": len(cell["samples"]),
                "samples": cell["samples"],
                "cohort_effective_rank": cell["cohort_effective_rank"],
                "hook_lifecycle": cell["hook_lifecycle"],
                "restore_called": cell["restore_called"],
                "restore_completed": cell["restore_completed"],
                "vectors_persisted": False,
            }
        )
        relative = Path("calibration_stepwise_samples") / ("k%d.json" % k)
        atomic_write_new_json(output_dir / relative, payload)
        cell_refs.append({"k": k, "path": str(relative), "sha256": payload["manifest_sha256"]})
        erank.append({"k": k, "steps": cell["cohort_effective_rank"]})
    summary = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-fixed-horizon-stepwise-summary.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "mode": result["mode"],
            "sample_count": result["session_contract"]["sample_count"],
            "ordered_identity_sha256": ordered_identity_sha256(
                [row["sample_identity"] for row in result["baseline_reference_samples"]]
            ),
            "label_state": "sealed",
            "session_contract": result["session_contract"],
            "gpu": result["gpu"],
            "k1_noninterference": result["k1_noninterference"],
            "cohort_effective_rank": erank,
            "resource_usage": result["resource_usage"],
            "vectors_persisted": False,
        }
    )
    atomic_write_new_json(output_dir / "calibration_stepwise_summary.json", summary)
    receipt = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-fixed-horizon-probe-receipt.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "mode": result["mode"],
            "completed_at_utc": utc_now(),
            "sample_count": result["session_contract"]["sample_count"],
            "total_body_calls": result["session_contract"]["total_body_calls"],
            "result": "complete",
        }
    )
    atomic_write_new_json(output_dir / "probe_receipt.json", receipt)
    manifest = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-fixed-horizon-probe-manifest.v1",
            "artifact_root": str(output_dir.resolve()),
            "card": {"path": "fixed_horizon_stepwise_card.json", "sha256": card["manifest_sha256"]},
            "mode": result["mode"],
            "sample_count": result["session_contract"]["sample_count"],
            "label_state": "sealed",
            "input_jsonl": {"path": str(Path(args.input_jsonl).resolve()), "sha256": file_sha256(Path(args.input_jsonl))},
            "input_manifest": {"path": str(Path(args.input_manifest).resolve()), "sha256": file_sha256(Path(args.input_manifest))},
            "ordered_identity_sha256": summary["ordered_identity_sha256"],
            "revision_closure": result["revision_closure"],
            "gpu": result["gpu"],
            "baseline_reference": {"path": "native_no_loop_reference.json", "sha256": baseline["manifest_sha256"]},
            "cells": cell_refs,
            "summary": {"path": "calibration_stepwise_summary.json", "sha256": summary["manifest_sha256"]},
            "receipt": {"path": "probe_receipt.json", "sha256": receipt["manifest_sha256"]},
            "vectors_persisted": False,
        }
    )
    validate_no_persisted_vectors(manifest)
    atomic_write_new_json(output_dir / "fixed_horizon_probe_manifest.json", manifest)
    return manifest


def _load_hashed(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FixedHorizonError("hashed JSON source must be an object")
    _verify_manifest(payload)
    return payload


def compare_d1_to_gate_b(d1_manifest_path: Path, gate_b_aggregate_path: Path, output_path: Path) -> Mapping[str, Any]:
    """Compare D1 K2--K4 core values with immutable Gate B B2 evidence."""

    manifest = _load_hashed(d1_manifest_path)
    if manifest.get("mode") != "d1-smoke" or manifest.get("sample_count") != 4:
        raise FixedHorizonError("D1 comparison requires the four-sample smoke manifest")
    root = Path(manifest["artifact_root"])
    card = _load_hashed(root / manifest["card"]["path"])
    gate_b = _load_hashed(gate_b_aggregate_path)
    gate_b_root = Path(gate_b["artifact_root"])
    refs = {ref["cell_id"]: ref for ref in gate_b["cells"]}
    from tflt.loopscope.phase2_schema import protocol_cell_id

    expected_ids = {
        2: protocol_cell_id("shared_k2_anchor", WINDOW, 2, 1.0),
        3: protocol_cell_id("fixed_horizon", WINDOW, 3, 1.0),
        4: protocol_cell_id("fixed_horizon", WINDOW, 4, 1.0),
    }
    d1_refs = {int(ref["k"]): ref for ref in manifest["cells"]}
    tolerance = card["tolerances"]
    comparisons = []
    for k in (2, 3, 4):
        d1_cell = _load_hashed(root / d1_refs[k]["path"])
        ref = refs.get(expected_ids[k])
        if ref is None:
            raise FixedHorizonError("Gate B aggregate lacks fixed-horizon K%d" % k)
        b2_cell = _load_hashed(gate_b_root / ref["path"])
        if len(d1_cell["samples"]) != 4 or len(b2_cell["samples"]) < 4:
            raise FixedHorizonError("D1/B2 sample scale is invalid")
        max_abs = 0.0
        all_match = True
        for observed, expected in zip(d1_cell["samples"], b2_cell["samples"][:4]):
            if observed["sample_identity"] != expected["sample_identity"]:
                raise FixedHorizonError("D1/B2 sample identity order differs")
            observed_scores = observed["final_direct_probe_output"]["raw_choice_scores"]
            expected_scores = expected["final_output"]["raw_choice_scores"]
            pairs: List[Tuple[float, float]] = list(zip(observed_scores, expected_scores))
            for t in range(k):
                left = observed["steps"][t]
                right = expected["steps"][t]
                pairs.extend(
                    [
                        (left["raw_residual_norm"], right["residual_norm"]),
                        (left["raw_relative_activity"], right["relative_activity"]),
                    ]
                )
                if t > 0:
                    pairs.extend(
                        [
                            (left["residual_ratio_to_previous"], right["residual_ratio_to_previous"]),
                            (left["adjacent_residual_cosine"], right["adjacent_residual_cosine"]),
                            (left["repeated_step_nca"]["value"], right["nca"]["value"]),
                        ]
                    )
            for left, right in pairs:
                difference = abs(float(left) - float(right))
                max_abs = max(max_abs, difference)
                if not math.isclose(float(left), float(right), abs_tol=tolerance["b2_core_atol"], rel_tol=tolerance["b2_core_rtol"]):
                    all_match = False
        comparisons.append({"k": k, "gate_b_cell_id": expected_ids[k], "all_match": all_match, "max_abs": max_abs})
    summary = _load_hashed(root / manifest["summary"]["path"])
    proof = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-d1-noninterference-proof.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "d1_manifest": {"path": str(Path(d1_manifest_path).resolve()), "sha256": manifest["manifest_sha256"]},
            "gate_b_aggregate": {"path": str(Path(gate_b_aggregate_path).resolve()), "sha256": gate_b["manifest_sha256"]},
            "k1_native_equivalence": summary["k1_noninterference"],
            "k2_k4_core_comparisons": comparisons,
            "hook_restore_all_complete": all(
                resource.get("hook_handles_registered") == resource.get("hook_handles_removed") == 5
                and resource.get("hooks_active_after_removal") == 0
                for resource in summary["resource_usage"] if resource["name"].startswith("fixed_horizon_k")
            ),
            "all_pass": summary["k1_noninterference"]["all_choice_scores_match"] and all(row["all_match"] for row in comparisons),
        }
    )
    if proof["all_pass"] is not True or proof["hook_restore_all_complete"] is not True:
        raise FixedHorizonError("D1 noninterference comparison failed")
    atomic_write_new_json(output_path, proof)
    return proof


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="run D1 smoke or D2 frozen-512 profile")
    probe.add_argument("--mode", choices=("d1-smoke", "d2-profile"), required=True)
    probe.add_argument("--card", required=True)
    probe.add_argument("--input-jsonl", required=True)
    probe.add_argument("--input-manifest", required=True)
    probe.add_argument("--output-dir", required=True)
    probe.add_argument("--device", choices=("cuda", "auto"), default="cuda")
    probe.add_argument("--d1-noninterference", default=None)
    compare = sub.add_parser("compare-d1", help="bind D1 outputs to immutable Gate B B2")
    compare.add_argument("--d1-manifest", required=True)
    compare.add_argument("--gate-b-aggregate", required=True)
    compare.add_argument("--output", required=True)
    check = sub.add_parser("check-card", help="validate card self/hash closure")
    check.add_argument("--card", required=True)
    check.add_argument("--repo-root", required=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.command == "check-card":
        card = _load_hashed(Path(args.card))
        verify_implementation_hashes(card, Path(args.repo_root))
        print(card["manifest_sha256"])
        return 0
    if args.command == "compare-d1":
        proof = compare_d1_to_gate_b(Path(args.d1_manifest), Path(args.gate_b_aggregate), Path(args.output))
        print(proof["manifest_sha256"])
        return 0
    card = _load_hashed(Path(args.card))
    repo_root = Path(__file__).resolve().parents[3]
    verify_implementation_hashes(card, repo_root)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FixedHorizonError("output directory already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    attempt = make_hashed_manifest(
        {
            "schema_version": "loopscope.h2-fixed-horizon-probe-attempt.v1",
            "card_manifest_sha256": card["manifest_sha256"],
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "mode": args.mode,
            "created_at_utc": utc_now(),
            "output_root": str(output_dir),
        }
    )
    atomic_write_new_json(output_dir / "attempt_manifest.json", attempt)
    atomic_write_new_json(output_dir / "command_args.json", vars(args))
    atomic_write_new_json(
        output_dir / "env.json",
        {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_ENDPOINT", "CUDA_VISIBLE_DEVICES")},
    )
    try:
        result = run_fixed_horizon_probe(args, card)
        manifest = materialize_probe(output_dir, args, result)
    except Exception as exc:
        atomic_write_new_json(
            output_dir / "failure.json",
            {"schema_version": "loopscope.h2-fixed-horizon-failure.v1", "type": type(exc).__name__, "message": str(exc), "failed_at_utc": utc_now()},
        )
        raise
    print(str(output_dir / "fixed_horizon_probe_manifest.json"))
    print(manifest["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
