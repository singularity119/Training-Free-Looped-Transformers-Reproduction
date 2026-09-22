"""Per-prompt Phase 9 residual transforms.

The callback is intentionally compatible with ``LoopConfig.residual_transform``
and the unchanged damped-Euler implementation.  At ``t=0`` it estimates one
rank-one direction from the valid retained prompt-token rows of the native
residual.  At later timesteps it changes only the answer-position row.  A
direction is cached only while the exact prompt identity and retained prefix
are unchanged, then released when the next prompt arrives.

Torch is imported lazily inside tensor execution so the pure-Python Gate A
tests remain runnable without the model runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .phase9_spectral_core import MATCHED_NORM, ONLINE_T0, STRENGTH


def identity_key(identity: Any) -> str:
    """Return a readable, order-stable key for a JSON-compatible identity."""

    return json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _finite_tensor(torch: Any, value: Any, message: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(message)


def _fit_direction(rows: Any, torch: Any) -> Tuple[Any, Dict[str, Any]]:
    """Fit a canonical FP32 reduced-SVD direction from ``rows``.

    CUDA uses the explicit ``gesvd`` backend required by the contract.  CPU
    uses PyTorch's exact SVD path without a CUDA-only driver argument.
    """

    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError("online direction requires a nonempty two-dimensional residual matrix")
    matrix = rows.float()
    _finite_tensor(torch, matrix, "online residual matrix contains nonfinite values")
    started = time.monotonic()
    if getattr(matrix.device, "type", None) == "cuda":
        # CUDA launches are asynchronous.  Use CUDA events around the SVD and
        # synchronize before reading the elapsed time so the recorded value is
        # an SVD time rather than a launch time.
        torch.cuda.synchronize(matrix.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")
        end_event.record()
        end_event.synchronize()
        elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        backend = "cuda_gesvd_fp32_reduced_svd"
    else:
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        elapsed = time.monotonic() - started
        backend = "cpu_fp32_reduced_svd"
    _finite_tensor(torch, singular_values, "online SVD returned nonfinite singular values")
    _finite_tensor(torch, vh, "online SVD returned nonfinite right singular vectors")
    energy = torch.sum(singular_values.square())
    if not bool(torch.isfinite(energy)) or float(energy) <= 0.0:
        raise ValueError("online prompt residuals have no finite nonzero energy")
    direction = vh[0].float()
    pivot = int(torch.argmax(torch.abs(direction)))
    if float(direction[pivot]) < 0.0:
        direction = -direction
    direction_norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(direction_norm)) or float(direction_norm) <= 0.0:
        raise ValueError("online SVD direction has zero or nonfinite norm")
    direction = direction / direction_norm
    singular_1 = float(singular_values[0])
    singular_2 = float(singular_values[1]) if singular_values.numel() > 1 else 0.0
    return direction, {
        "algorithm": backend,
        "row_count": int(matrix.shape[0]),
        "hidden_size": int(matrix.shape[1]),
        "top1_energy_fraction": float(singular_values[0].square() / energy),
        "singular_value_1": singular_1,
        "singular_value_2": singular_2,
        "singular_value_gap": singular_1 - singular_2,
        "singular_value_relative_gap": (singular_1 - singular_2) / singular_1 if singular_1 else None,
        "svd_seconds": elapsed,
        "svd_wall_seconds": time.monotonic() - started,
        "svd_timing": "cuda_event" if getattr(matrix.device, "type", None) == "cuda" else "cpu_wall",
    }


def _tensor_change(torch: Any, before: Any, after: Any, position: int) -> Dict[str, Any]:
    """Return bounded mutation diagnostics without retaining hidden tensors."""

    difference = (after - before).float().abs()
    other = difference.clone()
    other[0, position] = 0
    return {
        "same_object": bool(after is before),
        "answer_max_abs_change": float(difference[0, position].max()),
        "nonanswer_max_abs_change": float(other.max()),
        "changed_element_count": int((difference > 0).sum()),
    }


class Phase9DiagnosticCollector:
    """Collect scalar, label-free trajectory diagnostics for one cell.

    The t0 retained-token matrix and the CPU FP64 reference direction exist only
    for the active prompt.  The persisted record contains scalar norms,
    projections, spectral gaps, and timings, never the full residual matrix or
    direction vector.
    """

    def __init__(self, identities: Sequence[Any], k: int):
        if k not in (2, 3, 4):
            raise ValueError("Phase 9 diagnostic K must be 2, 3 or 4")
        self.identities = list(identities)
        self.keys = [identity_key(value) for value in self.identities]
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("diagnostic identities must be nonempty and unique")
        self.k = k
        self.rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.references: Dict[str, Dict[str, Any]] = {}
        self._answer_vectors: Dict[str, Any] = {}
        self.duplicate_observations = 0

    @staticmethod
    def _cpu_reference(rows: Any, torch: Any) -> Tuple[Any, Dict[str, Any]]:
        matrix = rows.detach().to(device="cpu", dtype=torch.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or not bool(torch.isfinite(matrix).all()):
            raise ValueError("CPU FP64 reference requires a finite nonempty matrix")
        started = time.monotonic()
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        elapsed = time.monotonic() - started
        energy = torch.sum(singular_values.square())
        if not bool(torch.isfinite(energy)) or float(energy) <= 0.0:
            raise ValueError("CPU FP64 reference has no finite nonzero energy")
        direction = vh[0].clone()
        pivot = int(torch.argmax(torch.abs(direction)))
        if float(direction[pivot]) < 0.0:
            direction = -direction
        direction = direction / torch.linalg.vector_norm(direction)
        first = float(singular_values[0])
        second = float(singular_values[1]) if singular_values.numel() > 1 else 0.0
        return direction, {
            "reference_algorithm": "cpu_fp64_reduced_svd_uncentered_unnormalized",
            "reference_row_count": int(matrix.shape[0]),
            "reference_hidden_size": int(matrix.shape[1]),
            "reference_top1_energy_fraction": float(singular_values[0].square() / energy),
            "reference_singular_value_1": first,
            "reference_singular_value_2": second,
            "reference_singular_value_gap": first - second,
            "reference_singular_value_relative_gap": (first - second) / first if first else None,
            "reference_cpu_svd_seconds": elapsed,
        }

    def observe(
        self,
        identity: Any,
        t: int,
        delta: Any,
        position: int,
        valid_positions: Sequence[int],
        direction: Any,
        fit_metadata: Optional[Dict[str, Any]],
        mutation: Dict[str, Any],
    ) -> None:
        """Record one pre-transform residual, deduplicated over A/B/C/D requests."""

        import torch

        key = identity_key(identity)
        if key not in self.keys or not 0 <= t < self.k:
            raise ValueError("diagnostic identity or timestep outside contract")
        row_key = (key, t)
        if row_key in self.rows:
            self.duplicate_observations += 1
            return
        index = torch.tensor(tuple(valid_positions), device=delta.device, dtype=torch.long)
        retained = delta[0].index_select(0, index).float()
        answer = delta[0, position].float()
        if not bool(torch.isfinite(retained).all()):
            raise ValueError("diagnostic residual contains nonfinite values")
        if t == 0:
            reference_direction, reference_metadata = self._cpu_reference(retained, torch)
            self.references[key] = {
                "direction": reference_direction,
                "metadata": reference_metadata,
                "position": position,
                "valid_positions": tuple(valid_positions),
            }
            self._answer_vectors[key] = answer.detach().cpu().double()
            diagnostic_direction = direction.detach().float()
            diagnostic_fit = dict(fit_metadata or {})
        else:
            # v_t is deliberately diagnostic-only.  The runtime's cached t0
            # direction remains the sole direction used by the intervention.
            diagnostic_direction, diagnostic_fit = _fit_direction(retained, torch)
        reference = self.references.get(key)
        if reference is None:
            raise RuntimeError("diagnostic t>=1 arrived before t0 reference")
        ref_direction = reference["direction"].to(dtype=torch.float64)
        diagnostic_cpu = diagnostic_direction.detach().to(device="cpu", dtype=torch.float64)
        diagnostic_dot = float(torch.dot(ref_direction, diagnostic_cpu))
        if diagnostic_dot < 0.0:
            diagnostic_dot = -diagnostic_dot
        answer64 = answer.detach().cpu().double()
        coefficient = float(torch.dot(ref_direction, answer64))
        answer_norm = float(torch.linalg.vector_norm(answer64))
        retained_norm = float(torch.linalg.vector_norm(retained))
        projected_norm = abs(coefficient)
        projection_fraction = (coefficient * coefficient) / (answer_norm * answer_norm) if answer_norm else None
        previous = self.rows.get((key, t - 1))
        previous_norm = previous.get("A_t", {}).get("retained_residual_norm") if previous else None
        self.rows[row_key] = {
            "identity": identity,
            "t": t,
            "position": int(position),
            "valid_token_count": len(tuple(valid_positions)),
            "token_length": int(delta.shape[1]),
            "E0": {
                "retained_residual_energy": float(torch.sum(retained.double().square())) if t == 0 else None,
                "answer_residual_norm": answer_norm if t == 0 else None,
            },
            "A_t": {
                "retained_residual_norm": retained_norm,
                "answer_residual_norm": answer_norm,
                "answer_norm_ratio_to_t0": None,
                "adjacent_retained_norm_ratio": (retained_norm / previous_norm if previous_norm else None),
            },
            "C0t": {
                "projection_coefficient": coefficient,
                "projected_answer_norm": projected_norm,
                "projection_energy_fraction": projection_fraction,
                "v0_vt_cosine_abs": diagnostic_dot,
            },
            "spectral": {
                "direction_used_for_intervention": "t0" if t == 0 else "t0_cached",
                "diagnostic_direction_used_for_intervention": False,
                "diagnostic_fit": diagnostic_fit,
            },
            "mutation": dict(mutation),
            "gpu_fit": dict(fit_metadata or {}) if t == 0 else None,
            "reference": dict(reference["metadata"]),
        }
        if t == 0:
            self.rows[row_key]["A_t"]["answer_norm_ratio_to_t0"] = 1.0
        else:
            initial = self.rows[(key, 0)]["A_t"]["answer_residual_norm"]
            self.rows[row_key]["A_t"]["answer_norm_ratio_to_t0"] = answer_norm / initial if initial else None

    def finish(self, identity: Any, direction: Any, fit_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Close one prompt and compare the GPU FP32 direction to CPU FP64."""

        import torch

        key = identity_key(identity)
        reference = self.references.pop(key, None)
        answer = self._answer_vectors.pop(key, None)
        if reference is None or answer is None:
            raise ValueError("cannot finish a prompt without a t0 CPU reference")
        gpu = direction.detach().to(device="cpu", dtype=torch.float64)
        ref = reference["direction"]
        dot = float(torch.dot(gpu, ref))
        if dot < 0.0:
            gpu = -gpu
            dot = -dot
        difference = torch.linalg.vector_norm(gpu - ref)
        ref_projection = ref * torch.dot(ref, answer)
        gpu_projection = gpu * torch.dot(gpu, answer)
        denominator = float(torch.linalg.vector_norm(ref_projection))
        projection_error = float(torch.linalg.vector_norm(gpu_projection - ref_projection))
        if denominator:
            projection_error /= denominator
        else:
            projection_error = None
        rows = [self.rows[(key, t)] for t in range(self.k)]
        comparison = {
            "direction_cosine": dot,
            "direction_l2_error": float(difference),
            "projection_relative_error": projection_error,
            "projection_error_denominator": denominator,
            "normal_spectral_gap": reference["metadata"]["reference_singular_value_gap"],
            "normal_spectral_relative_gap": reference["metadata"]["reference_singular_value_relative_gap"],
            "near_repeated_root": bool(
                reference["metadata"]["reference_singular_value_relative_gap"] is not None
                and reference["metadata"]["reference_singular_value_relative_gap"] <= 1e-3
            ),
            "gpu_fit": dict(fit_metadata or {}),
        }
        for row in rows:
            row["fit_comparison"] = comparison
        return {
            "identity": identity,
            "fit_comparison": comparison,
            "records": rows,
        }

    def cell_records(self) -> List[Dict[str, Any]]:
        if self.references:
            raise ValueError("diagnostic collector still has active prompt references")
        missing = [key for key in self.keys for t in range(self.k) if (key, t) not in self.rows]
        if missing:
            raise ValueError("missing diagnostic trajectory rows: %s" % missing[:4])
        result = []
        for identity in self.identities:
            key = identity_key(identity)
            result.append({"identity": identity, "records": [self.rows[(key, t)] for t in range(self.k)]})
        return result

    def summary(self) -> Dict[str, Any]:
        return {
            "identity_count": len(self.keys),
            "k": self.k,
            "row_count": len(self.rows),
            "duplicate_observations": self.duplicate_observations,
            "rows_by_t": {str(t): sum(key[1] == t for key in self.rows) for t in range(self.k)},
        }


class Phase9Runtime:
    """Online-t0 or Matched-norm callback for one model/window cell."""

    MODES = (ONLINE_T0, MATCHED_NORM)

    def __init__(
        self,
        mode: str,
        strength: float = STRENGTH,
        collector: Optional[Any] = None,
        intervene: bool = True,
    ):
        if mode not in self.MODES:
            raise ValueError("Phase 9 mode must be Online-t0 or Matched-norm")
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        self.mode = mode
        self.strength = float(strength)
        self.collector = collector
        self.intervene = bool(intervene)
        self._active: Optional[Dict[str, Any]] = None
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._direction: Any = None
        self._fit_metadata: Optional[Dict[str, Any]] = None
        self.calls = []
        self.diagnostics = []
        self.context_summaries = []
        self.cache_reset_count = 0

    @property
    def direction(self) -> Any:
        """Expose the current device direction for bounded diagnostic checks."""

        return self._direction

    @contextmanager
    def context(
        self,
        identity: Any,
        position: int,
        valid_positions: Iterable[int],
        token_ids: Optional[Sequence[int]] = None,
    ):
        if self._active is not None:
            raise RuntimeError("nested Phase 9 request context")
        if not isinstance(position, int) or position < 0:
            raise ValueError("pre-answer position must be a nonnegative integer")
        positions = tuple(valid_positions)
        if not positions or any(not isinstance(item, int) or item < 0 for item in positions):
            raise ValueError("valid prompt positions must be nonempty nonnegative integers")
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("valid prompt positions must be sorted and unique")
        if position not in positions:
            raise ValueError("answer position must be one of the valid prompt positions")
        if token_ids is None:
            retained_prefix = positions
        else:
            retained_prefix = tuple(token_ids[: position + 1])
        key = (identity_key(identity), position, positions, retained_prefix)
        if self._cache_key != key:
            if self._cache_key is not None:
                self.cache_reset_count += 1
            self._cache_key = key
            self._direction = None
            self._fit_metadata = None
        call_start = len(self.calls)
        self._active = {
            "identity": identity,
            "position": position,
            "valid_positions": positions,
            "token_ids": retained_prefix,
        }
        try:
            yield self
        finally:
            self.context_summaries.append(
                {
                    "identity": identity,
                    "position": position,
                    "valid_token_count": len(positions),
                    "call_count": len(self.calls) - call_start,
                    "timesteps": [call["t"] for call in self.calls[call_start:]],
                }
            )
            self._active = None

    def _validate_delta(self, delta: Any) -> None:
        if self._active is None:
            raise RuntimeError("Phase 9 residual callback requires evaluator context")
        if getattr(delta, "ndim", None) != 3:
            raise ValueError("Phase 9 expects a rank-3 residual tensor")
        shape = getattr(delta, "shape", ())
        if len(shape) != 3 or shape[0] != 1:
            raise ValueError("Phase 9 expects a batch-one residual tensor")
        position = self._active["position"]
        positions = self._active["valid_positions"]
        if position >= shape[1] or any(item >= shape[1] for item in positions):
            raise ValueError("Phase 9 prompt position is outside the residual sequence")

    def __call__(self, delta: Any, t: int) -> Any:
        self._validate_delta(delta)
        if not isinstance(t, int) or t < 0:
            raise ValueError("Phase 9 timestep must be a nonnegative integer")
        identity = self._active["identity"]
        position = self._active["position"]
        positions = self._active["valid_positions"]
        call = {
            "identity": identity,
            "t": t,
            "position": position,
            "valid_token_count": len(positions),
            "mode": self.mode,
            "fitted": False,
            "applied": False,
            "norm_ratio": None,
            "coverage": None,
        }
        if t == 0:
            import torch

            if self._direction is None:
                index = torch.tensor(positions, device=delta.device, dtype=torch.long)
                rows = delta[0].index_select(0, index)
                direction, metadata = _fit_direction(rows, torch)
                self._direction = direction
                self._fit_metadata = metadata
                self.diagnostics.append(
                    dict(
                        identity=identity,
                        position=position,
                        valid_token_count=len(positions),
                        **metadata,
                    )
                )
                call["fitted"] = True
            mutation = _tensor_change(torch, delta, delta, position)
            call.update(mutation)
            if self.collector is not None:
                self.collector.observe(
                    identity,
                    t,
                    delta,
                    position,
                    positions,
                    self._direction,
                    self._fit_metadata,
                    mutation,
                )
            self.calls.append(call)
            # t0 is an exact native path: no clone and no intervention.
            return delta

        if self._direction is None:
            raise RuntimeError("Phase 9 t>=1 callback arrived before t0 direction estimation")
        import torch

        direction = self._direction
        if direction.device != delta.device:
            direction = direction.to(device=delta.device)
            self._direction = direction
        if direction.numel() != delta.shape[-1]:
            raise ValueError("online direction hidden width differs from residual hidden width")
        raw = delta[0, position]
        raw32 = raw.float()
        _finite_tensor(torch, raw32, "answer-position residual contains nonfinite values")
        coefficient = torch.dot(direction, raw32)
        damped = raw32 - self.strength * direction * coefficient
        _finite_tensor(torch, damped, "Phase 9 damped residual is nonfinite")
        raw_norm = torch.linalg.vector_norm(raw32)
        if not bool(torch.isfinite(raw_norm)):
            raise ValueError("answer-position residual norm is nonfinite")
        if float(raw_norm) == 0.0:
            updated = torch.zeros_like(raw32)
        elif self.mode == ONLINE_T0:
            updated = damped
        else:
            scale = torch.linalg.vector_norm(damped) / raw_norm
            updated = scale * raw32
        _finite_tensor(torch, updated, "Phase 9 updated residual is nonfinite")
        if not self.intervene or self.strength == 0.0:
            result = delta
        else:
            result = delta.clone()
            result[0, position] = updated.to(dtype=delta.dtype)
        call["applied"] = self.intervene and self.strength != 0.0
        call["norm_ratio"] = float(torch.linalg.vector_norm(updated) / raw_norm) if float(raw_norm) else 1.0
        call["coverage"] = float((coefficient * coefficient) / (raw_norm * raw_norm)) if float(raw_norm) else None
        mutation = _tensor_change(torch, delta, result, position)
        call.update(mutation)
        if self.collector is not None:
            self.collector.observe(
                identity,
                t,
                delta,
                position,
                positions,
                self._direction,
                self._fit_metadata,
                mutation,
            )
        self.calls.append(call)
        return result

    def summary(self) -> Dict[str, Any]:
        """Return JSON-friendly in-memory callback diagnostics."""

        return {
            "mode": self.mode,
            "strength": self.strength,
            "call_count": len(self.calls),
            "calls": list(self.calls),
            "diagnostics": list(self.diagnostics),
            "context_summaries": list(self.context_summaries),
            "cache_reset_count": self.cache_reset_count,
            "intervene": self.intervene,
        }


def tensor_smoke_checks(torch: Any) -> Dict[str, bool]:
    """Run small real-tensor semantics for the future B debug entry point."""

    results = {}
    for mode in (ONLINE_T0, MATCHED_NORM):
        runtime = Phase9Runtime(mode)
        delta = torch.tensor([[[2.0, 3.0], [4.0, 6.0], [5.0, 7.0]]], dtype=torch.float32)
        with runtime.context("synthetic", 1, (0, 1), token_ids=(10, 11)):
            original = runtime(delta, 0)
            updated = runtime(delta, 1)
        results[mode] = bool(original.data_ptr() == delta.data_ptr()) and bool(
            torch.allclose(updated[0, 0], delta[0, 0])
        ) and bool(torch.isfinite(updated).all())
    return results
