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
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")
        backend = "cuda_gesvd_fp32_reduced_svd"
    else:
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        backend = "cpu_fp32_reduced_svd"
    elapsed = time.monotonic() - started
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
        "svd_seconds": elapsed,
    }


class Phase9Runtime:
    """Online-t0 or Matched-norm callback for one model/window cell."""

    MODES = (ONLINE_T0, MATCHED_NORM)

    def __init__(self, mode: str, strength: float = STRENGTH):
        if mode not in self.MODES:
            raise ValueError("Phase 9 mode must be Online-t0 or Matched-norm")
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        self.mode = mode
        self.strength = float(strength)
        self._active: Optional[Dict[str, Any]] = None
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._direction: Any = None
        self._fit_metadata: Optional[Dict[str, Any]] = None
        self.calls = []
        self.diagnostics = []

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
            self._cache_key = key
            self._direction = None
            self._fit_metadata = None
        self._active = {
            "identity": identity,
            "position": position,
            "valid_positions": positions,
            "token_ids": retained_prefix,
        }
        try:
            yield self
        finally:
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
        result = delta.clone()
        result[0, position] = updated.to(dtype=delta.dtype)
        call["applied"] = self.strength != 0.0
        call["norm_ratio"] = float(torch.linalg.vector_norm(updated) / raw_norm) if float(raw_norm) else 1.0
        call["coverage"] = float((coefficient * coefficient) / (raw_norm * raw_norm)) if float(raw_norm) else None
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
