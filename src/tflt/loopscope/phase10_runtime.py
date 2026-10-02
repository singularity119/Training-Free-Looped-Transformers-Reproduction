"""Phase 10 entry point over the immutable Phase 9 numerical callbacks.

No tensor library is imported until the historical callbacks execute.  Real
tensor/model equivalence belongs to Gate B; the schedules are checkable here.
"""
from contextlib import contextmanager
import math

from .phase9_runtime import Phase9Runtime, _finite_tensor, _tensor_change
from .phase9_gate_e_runtime import Phase9GateERuntime
from .phase9_gate_e_spectral_core import direction_schedule as _schedule


def direction_schedule(policy, k):
    if k not in (2, 3):
        raise ValueError("Phase 10 supports only K=2 or K=3")
    if policy == "current_t":
        return [(t, t, t) for t in range(k)]
    return _schedule(policy, k)


def applies_at(policy, t, strength):
    return strength != 0 and (policy == "current_t" or t >= 1)


class Phase10Lag1Runtime(Phase9GateERuntime):
    """Restart a lag1 trajectory for each full-continuation model call.

    Multiple ARC candidates can share a prefix but require distinct full
    forwards.  A preceding forward's v1 must never become the next one's v0.
    The frozen Gate E numerical callback itself is reused unchanged.
    """
    @contextmanager
    def context(self, identity, position, valid_positions, token_ids=None):
        if self._active is not None:
            raise RuntimeError("nested Phase 10 request context")
        self._direction = None
        self._fit_metadata = {}
        with super().context(identity, position, valid_positions, token_ids) as runtime:
            yield runtime


class Phase10CurrentRuntime(Phase10Lag1Runtime):
    """Fit the own raw D_t prompt rows before immediately damping p, including t0.

    Only the residual-transform callback is new. It makes no window/model calls
    and reuses the frozen FP32 SVD and scalar mutation helpers.
    """
    def __init__(self, k, strength):
        super().__init__("fixed_t0", "spectral", k, strength=strength)
        self.direction_policy = "current_t"

    def __call__(self, delta, t):
        self._validate_delta(delta)
        if not isinstance(t, int) or not 0 <= t < self.k:
            raise ValueError("Phase 10 timestep is outside K")
        expected_t = 0 if self._last_context_t is None else self._last_context_t + 1
        if t != expected_t:
            raise RuntimeError("current_t callbacks must follow the own ordered trajectory")
        # _fit_native_direction uses only active valid prompt positions. It
        # sees the original residual, before any clone or p-row assignment.
        self._fit_native_direction(delta, t)
        import torch

        direction = self._direction
        position = self._active["position"]
        if direction.device != delta.device:
            direction = direction.to(device=delta.device)
            self._direction = direction
        if direction.numel() != delta.shape[-1]:
            raise ValueError("current_t direction hidden width differs from residual")
        raw32 = delta[0, position].float()
        _finite_tensor(torch, raw32, "current_t answer residual contains nonfinite values")
        coefficient = torch.dot(direction, raw32)
        updated = raw32 - self.strength * direction * coefficient
        _finite_tensor(torch, updated, "current_t damped residual contains nonfinite values")
        raw_norm = torch.linalg.vector_norm(raw32)
        if not bool(torch.isfinite(raw_norm)):
            raise ValueError("current_t answer residual norm is nonfinite")
        if not self.intervene or self.strength == 0:
            result = delta
        else:
            result = delta.clone()
            result[0, position] = updated.to(dtype=delta.dtype)
        applied = bool(self.intervene and self.strength != 0)
        self.calls.append({"identity": self._active["identity"], "t": t,
            "position": position, "valid_token_count": len(self._active["valid_positions"]),
            "direction_policy": "current_t", "intervention_mode": "spectral",
            "direction_fit_t": t, "direction_used_fit_t": t,
            "direction_fit_at_t": t, "direction_used_from_t": t,
            "fit_from_unmodified_current_residual": True,
            "applied_t": t if applied else None, "applied": applied,
            "norm_ratio": float(torch.linalg.vector_norm(updated) / raw_norm) if float(raw_norm) else 1.0,
            "coverage": float(coefficient * coefficient / (raw_norm * raw_norm)) if float(raw_norm) else None,
            **_tensor_change(torch, delta, result, position)})
        self._last_context_t = t
        return result

    def summary(self):
        return {"direction_policy": self.direction_policy, "intervention_mode": self.intervention_mode,
            "k": self.k, "strength": self.strength, "intervene": self.intervene,
            "call_count": len(self.calls), "calls": list(self.calls),
            "fit_metadata_by_t": {str(t): dict(value) for t, value in sorted(self.fit_metadata.items())},
            "context_summaries": list(self.context_summaries), "cache_reset_count": self.cache_reset_count,
            "schedule": [{"t":t,"direction_used_fit_t":used,"direction_fit_t":fit}
                for t,used,fit in direction_schedule("current_t",self.k)]}


def make_runtime(cell):
    arm = cell.get("arm")
    if arm not in ("Native", "Loop", "Online"):
        raise ValueError("Phase 10 permits only Native, Loop and Online")
    if arm != "Online":
        return None
    policy, k, strength = cell["direction_policy"], cell["k"], cell["strength"]
    direction_schedule(policy, k)
    if cell.get("intervention_mode") != "spectral":
        raise ValueError("Phase 10 Online requires spectral intervention")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and in [0, 1]")
    if policy == "fixed_t0":
        return Phase9Runtime("Online-t0", strength=strength)
    if policy == "current_t":
        return Phase10CurrentRuntime(k, strength)
    return Phase10Lag1Runtime("lag1", "spectral", k, strength=strength)


def validate_calls(runtime, k):
    """Check all actual candidate contexts without assuming one forward/question."""
    contexts = runtime.context_summaries
    if not contexts or any(context["timesteps"] != list(range(k)) for context in contexts):
        raise RuntimeError("Online candidate forward lacks ordered K-step callbacks")
    if len(runtime.calls) != len(contexts) * k:
        raise RuntimeError("Online candidate callback count differs from K")
    for call in runtime.calls:
        policy = getattr(runtime, "direction_policy", "fixed_t0")
        if call["nonanswer_max_abs_change"] != 0:
            raise RuntimeError("Online callback mutated a non-answer position")
        if call["t"] == 0 and policy != "current_t" and call["answer_max_abs_change"] != 0:
            raise RuntimeError("fixed_t0/lag1 callback mutated t0")
        if runtime.strength == 0 and call["answer_max_abs_change"] != 0:
            raise RuntimeError("zero-strength callback mutated the residual")
        expected = applies_at(policy, call["t"], runtime.strength)
        if bool(call["applied"]) != expected:
            raise RuntimeError("Online callback intervention timing/strength differs")
        if policy == "current_t" and (call["direction_fit_t"] != call["t"] or call["direction_used_fit_t"] != call["t"]):
            raise RuntimeError("current_t direction was not fitted and used on the current round")
    return len(contexts)
