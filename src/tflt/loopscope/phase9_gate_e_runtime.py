"""Per-prompt previous-round residual-direction transforms for Gate E.

This module is intentionally parallel to the frozen Phase 9 runtime.  It
reuses its numerically specified SVD and tensor diagnostics helpers but never
changes that historical runtime.  A direction is fitted from the current
native residual before the current-round intervention, while the intervention
uses only the direction fitted on the preceding round.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .phase9_gate_e_spectral_core import (
    DIRECTION_POLICIES,
    INTERVENTION_MODES,
    direction_schedule,
    validate_policy_and_mode,
)
from .phase9_runtime import _fit_direction, _finite_tensor, _tensor_change


def identity_key(identity: Any) -> str:
    return json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Phase9GateERuntime:
    """Shared fixed-t0/Lag1 callback with spectral/matched-norm intervention."""

    DIRECTION_POLICIES = DIRECTION_POLICIES
    INTERVENTION_MODES = INTERVENTION_MODES

    def __init__(
        self,
        direction_policy: str,
        intervention_mode: str,
        k: int,
        strength: float = 0.5,
        intervene: bool = True,
    ):
        validate_policy_and_mode(direction_policy, intervention_mode)
        if k not in (2, 3, 4):
            raise ValueError("Gate E runtime K must be 2, 3 or 4")
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        self.direction_policy = direction_policy
        self.intervention_mode = intervention_mode
        self.k = int(k)
        self.strength = float(strength)
        self.intervene = bool(intervene)
        self._active: Optional[Dict[str, Any]] = None
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._direction: Any = None
        self._fit_metadata: Dict[int, Dict[str, Any]] = {}
        self.calls = []
        self.context_summaries = []
        self.cache_reset_count = 0
        self._last_context_t: Optional[int] = None

    @property
    def direction(self) -> Any:
        """Expose the current previous-round direction for bounded debug checks."""

        return self._direction

    @property
    def fit_metadata(self) -> Dict[int, Dict[str, Any]]:
        return dict(self._fit_metadata)

    @contextmanager
    def context(
        self,
        identity: Any,
        position: int,
        valid_positions: Iterable[int],
        token_ids: Optional[Sequence[int]] = None,
    ):
        if self._active is not None:
            raise RuntimeError("nested Gate E request context")
        if not isinstance(position, int) or position < 0:
            raise ValueError("pre-answer position must be a nonnegative integer")
        positions = tuple(valid_positions)
        if not positions or any(not isinstance(item, int) or item < 0 for item in positions):
            raise ValueError("valid prompt positions must be nonempty nonnegative integers")
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("valid prompt positions must be sorted and unique")
        if position not in positions:
            raise ValueError("answer position must be one of the valid prompt positions")
        retained_prefix = positions if token_ids is None else tuple(token_ids[: position + 1])
        key = (identity_key(identity), position, positions, retained_prefix)
        if self._cache_key != key:
            if self._cache_key is not None:
                self.cache_reset_count += 1
            self._cache_key = key
            self._direction = None
            self._fit_metadata = {}
        call_start = len(self.calls)
        self._active = {
            "identity": identity,
            "position": position,
            "valid_positions": positions,
            "token_ids": retained_prefix,
        }
        self._last_context_t = None
        try:
            yield self
        finally:
            self.context_summaries.append({
                "identity": identity,
                "position": position,
                "valid_token_count": len(positions),
                "call_count": len(self.calls) - call_start,
                "timesteps": [call["t"] for call in self.calls[call_start:]],
            })
            self._active = None
            self._last_context_t = None

    def _validate_delta(self, delta: Any) -> None:
        if self._active is None:
            raise RuntimeError("Gate E residual callback requires evaluator context")
        if getattr(delta, "ndim", None) != 3:
            raise ValueError("Gate E expects a rank-3 residual tensor")
        shape = getattr(delta, "shape", ())
        if len(shape) != 3 or shape[0] != 1:
            raise ValueError("Gate E expects a batch-one residual tensor")
        position = self._active["position"]
        positions = self._active["valid_positions"]
        if position >= shape[1] or any(item >= shape[1] for item in positions):
            raise ValueError("Gate E prompt position is outside the residual sequence")

    def _fit_native_direction(self, delta: Any, t: int) -> None:
        import torch

        positions = self._active["valid_positions"]
        index = torch.tensor(positions, device=delta.device, dtype=torch.long)
        rows = delta[0].index_select(0, index)
        direction, metadata = _fit_direction(rows, torch)
        self._direction = direction
        self._fit_metadata[t] = metadata

    def __call__(self, delta: Any, t: int) -> Any:
        self._validate_delta(delta)
        if not isinstance(t, int) or t < 0 or t >= self.k:
            raise ValueError("Gate E timestep is outside the frozen K")
        identity = self._active["identity"]
        position = self._active["position"]
        positions = self._active["valid_positions"]
        if self._last_context_t is not None and t != self._last_context_t + 1:
            raise RuntimeError("Gate E callback timesteps must be strictly sequential")
        if t == 0:
            fitted_here = self._direction is None
            if fitted_here:
                self._fit_native_direction(delta, 0)
            import torch

            mutation = _tensor_change(torch, delta, delta, position)
            call = {
                "identity": identity,
                "t": t,
                "position": position,
                "valid_token_count": len(positions),
                "direction_policy": self.direction_policy,
                "intervention_mode": self.intervention_mode,
                "applied_t": None,
                "direction_fit_t": 0 if fitted_here else None,
                "direction_used_fit_t": None,
                "direction_used_from_t": None,
                "direction_fit_at_t": 0 if fitted_here else None,
                "applied": False,
                "norm_ratio": None,
                "coverage": None,
                **mutation,
            }
            self.calls.append(call)
            self._last_context_t = t
            return delta

        if self._direction is None:
            raise RuntimeError("Gate E t>=1 callback arrived before t0 direction estimation")
        import torch

        # Save the previous-round direction before fitting the current native
        # residual.  This is the central no-same-round-self-projection rule.
        used_direction = self._direction
        source_t = 0 if self.direction_policy == "fixed_t0" else t - 1
        should_fit_next = self.direction_policy == "lag1" and t < self.k - 1
        if should_fit_next:
            self._fit_native_direction(delta, t)
        if used_direction.device != delta.device:
            used_direction = used_direction.to(device=delta.device)
        if used_direction.numel() != delta.shape[-1]:
            raise ValueError("Gate E direction hidden width differs from residual hidden width")
        raw = delta[0, position]
        raw32 = raw.float()
        _finite_tensor(torch, raw32, "Gate E answer-position residual contains nonfinite values")
        coefficient = torch.dot(used_direction, raw32)
        damped = raw32 - self.strength * used_direction * coefficient
        _finite_tensor(torch, damped, "Gate E damped residual is nonfinite")
        raw_norm = torch.linalg.vector_norm(raw32)
        if not bool(torch.isfinite(raw_norm)):
            raise ValueError("Gate E answer-position residual norm is nonfinite")
        if float(raw_norm) == 0.0:
            updated = torch.zeros_like(raw32)
        elif self.intervention_mode == "spectral":
            updated = damped
        else:
            scale = torch.linalg.vector_norm(damped) / raw_norm
            updated = scale * raw32
        _finite_tensor(torch, updated, "Gate E updated residual is nonfinite")
        if not self.intervene or self.strength == 0.0:
            result = delta
        else:
            result = delta.clone()
            result[0, position] = updated.to(dtype=delta.dtype)
        call = {
            "identity": identity,
            "t": t,
            "position": position,
            "valid_token_count": len(positions),
            "direction_policy": self.direction_policy,
            "intervention_mode": self.intervention_mode,
            "applied_t": t if self.intervene and self.strength != 0.0 else None,
            "direction_fit_t": t if should_fit_next else None,
            "direction_used_fit_t": source_t,
            "direction_used_from_t": source_t,
            "direction_fit_at_t": t if should_fit_next else None,
            "applied": self.intervene and self.strength != 0.0,
            "norm_ratio": float(torch.linalg.vector_norm(updated) / raw_norm) if float(raw_norm) else 1.0,
            "coverage": float((coefficient * coefficient) / (raw_norm * raw_norm)) if float(raw_norm) else None,
            **_tensor_change(torch, delta, result, position),
        }
        self.calls.append(call)
        self._last_context_t = t
        return result

    def summary(self) -> Dict[str, Any]:
        return {
            "direction_policy": self.direction_policy,
            "intervention_mode": self.intervention_mode,
            "k": self.k,
            "strength": self.strength,
            "call_count": len(self.calls),
            "calls": list(self.calls),
            "fit_metadata_by_t": {str(t): dict(value) for t, value in sorted(self._fit_metadata.items())},
            "context_summaries": list(self.context_summaries),
            "cache_reset_count": self.cache_reset_count,
            "intervene": self.intervene,
            "schedule": [
                {"t": t, "direction_used_fit_t": used, "direction_fit_t": fit}
                for t, used, fit in direction_schedule(self.direction_policy, self.k)
            ],
        }
