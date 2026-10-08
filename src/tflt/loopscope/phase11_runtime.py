"""Phase 11 prompt directions, frozen for actual incremental decode.

The unchanged Euler callback receives the raw original-dtype residual. Only
the prompt's last valid row (or the current decode row) can be changed. Tensor
imports remain lazy so lifecycle tests need no torch installation.
"""
from contextlib import contextmanager
import math

from .phase9_runtime import _finite_tensor, _fit_direction, _tensor_change


def direction_schedule(policy, k):
    if policy not in ("fixed_t0", "lag1", "current_t") or k not in (2, 3):
        raise ValueError("Phase 11 requires fixed_t0/lag1/current_t and K2/K3")
    if policy == "current_t":
        return [(t, t, t) for t in range(k)]
    return [(t, None if t == 0 else (0 if policy == "fixed_t0" else t - 1),
             t if t == 0 or (policy == "lag1" and t < k - 1) else None)
            for t in range(k)]


class Phase11Runtime:
    def __init__(self, policy, k, strength):
        self.schedule = direction_schedule(policy, k)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Phase 11 strength must be finite and in [0, 1]")
        self.direction_policy, self.k, self.strength = policy, k, float(strength)
        self.intervention_mode = "spectral"
        self._active = None
        self._forward = None
        self._directions = {}
        self._fit_metadata = {}
        self._last_fit_metadata = {}
        self._prefill_complete = False
        self.calls = []
        self.context_summaries = []
        self.cache_reset_count = 0

    @property
    def directions(self):
        return dict(self._directions)

    @property
    def fit_metadata(self):
        return dict(self._fit_metadata)

    @contextmanager
    def question(self, identity, position, valid_positions, token_ids=None):
        if self._active is not None:
            raise RuntimeError("nested Phase 11 question context")
        positions = tuple(valid_positions)
        if not positions or any(not isinstance(p, int) or p < 0 for p in positions):
            raise ValueError("valid prompt positions must be nonempty nonnegative integers")
        if positions != tuple(sorted(set(positions))) or position != positions[-1]:
            raise ValueError("intervention position must be the last valid prompt position")
        tokens = None if token_ids is None else tuple(token_ids)
        if tokens is not None and positions[-1] >= len(tokens):
            raise ValueError("prompt position lies outside token ids")
        self._directions = {}
        self._fit_metadata = {}
        self._last_fit_metadata = {}
        self._prefill_complete = False
        self._active = {"identity": identity, "position": position,
                        "valid_positions": positions, "token_ids": tokens,
                        "forward_count": 0}
        try:
            yield self
            if not self._prefill_complete:
                raise RuntimeError("Phase 11 question completed without a K-step prefill")
        finally:
            self._active = None
            self._forward = None
            self._directions = {}
            self._fit_metadata = {}
            self._prefill_complete = False
            self.cache_reset_count += 1

    @contextmanager
    def forward_context(self, kind, sequence_length):
        if self._active is None or self._forward is not None:
            raise RuntimeError("Phase 11 forward requires one active question")
        if kind not in ("prefill", "decode") or not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("invalid Phase 11 forward kind or sequence length")
        if kind == "prefill":
            if self._prefill_complete:
                raise RuntimeError("prefill cannot replay after prompt directions are frozen")
            if self._active["position"] >= sequence_length:
                raise ValueError("prompt intervention position is outside prefill")
        elif not self._prefill_complete or sequence_length != 1:
            raise RuntimeError("decode requires frozen prefill directions and one current token")
        self._forward = {"kind": kind, "sequence_length": sequence_length,
                         "index": self._active["forward_count"], "next_t": 0}
        call_start = len(self.calls)
        try:
            yield self
            if self._forward["next_t"] != self.k:
                raise RuntimeError("Phase 11 forward lacks ordered K residual callbacks")
            if kind == "prefill":
                self._prefill_complete = True
                # Scalar fit evidence survives question closure; direction
                # tensors are released in question's finally block.
                self._last_fit_metadata = {t: dict(m) for t, m in self._fit_metadata.items()}
            self.context_summaries.append({"identity": self._active["identity"],
                "kind": kind, "forward_index": self._forward["index"],
                "sequence_length": sequence_length, "call_count": len(self.calls)-call_start,
                "timesteps": [call["t"] for call in self.calls[call_start:]]})
            self._active["forward_count"] += 1
        finally:
            self._forward = None

    @contextmanager
    def context(self, identity, position, valid_positions, token_ids=None):
        """ARC candidates each have a fresh question and full prefill forward."""
        sequence_length = len(token_ids) if token_ids is not None else position + 1
        with self.question(identity, position, valid_positions, token_ids):
            with self.forward_context("prefill", sequence_length):
                yield self

    def _fit_native_direction(self, delta, t):
        import torch
        positions = self._active["valid_positions"]
        index = torch.tensor(positions, device=delta.device, dtype=torch.long)
        # Slice before FP32 conversion: D_t retains the model's original dtype.
        rows = delta[0].index_select(0, index)
        direction, metadata = _fit_direction(rows, torch)
        self._directions[t] = direction.detach()
        self._fit_metadata[t] = dict(metadata)

    def _damp(self, delta, position, direction):
        import torch
        if direction.device != delta.device:
            direction = direction.to(device=delta.device)
        if direction.numel() != delta.shape[-1]:
            raise ValueError("Phase 11 direction hidden width differs from residual")
        raw32 = delta[0, position].float()
        _finite_tensor(torch, raw32, "Phase 11 target residual is nonfinite")
        coefficient = torch.dot(direction, raw32)
        updated = raw32 - self.strength * direction * coefficient
        _finite_tensor(torch, updated, "Phase 11 damped residual is nonfinite")
        result = delta.clone()
        result[0, position] = updated.to(dtype=delta.dtype)
        return result

    def _mutation(self, before, after, position):
        import torch
        return _tensor_change(torch, before, after, position)

    def __call__(self, delta, t):
        if self._active is None or self._forward is None:
            raise RuntimeError("Phase 11 residual requires question and forward context")
        if getattr(delta, "ndim", None) != 3 or delta.shape[0] != 1:
            raise ValueError("Phase 11 expects a rank-three batch-one residual")
        if delta.shape[1] != self._forward["sequence_length"]:
            raise ValueError("residual sequence differs from actual forward input")
        if not isinstance(t, int) or t != self._forward["next_t"] or t >= self.k:
            raise RuntimeError("Phase 11 callbacks must follow ordered t=0..K-1 in each forward")
        kind = self._forward["kind"]
        position = self._active["position"] if kind == "prefill" else 0
        _, used_t, fit_t = self.schedule[t]
        fitted = kind == "prefill" and fit_t is not None
        if fitted:
            self._fit_native_direction(delta, fit_t)
        if used_t is not None and used_t not in self._directions:
            raise RuntimeError("required prefill direction was not fitted")
        applied = used_t is not None and self.strength != 0
        result = self._damp(delta, position, self._directions[used_t]) if applied else delta
        self.calls.append({"identity": self._active["identity"], "t": t,
            "forward_kind": kind, "forward_index": self._forward["index"], "position": position,
            "valid_token_count": len(self._active["valid_positions"]) if kind == "prefill" else 1,
            "direction_policy": self.direction_policy, "intervention_mode": "spectral",
            "direction_fit_t": fit_t if fitted else None, "direction_used_fit_t": used_t,
            "fit_from_unmodified_current_residual": fitted, "applied": applied,
            "applied_t": t if applied else None, **self._mutation(delta, result, position)})
        self._forward["next_t"] += 1
        return result

    def summary(self):
        return {"direction_policy": self.direction_policy, "intervention_mode": "spectral",
            "k": self.k, "strength": self.strength, "call_count": len(self.calls),
            "calls": list(self.calls), "context_summaries": list(self.context_summaries),
            "cache_reset_count": self.cache_reset_count,
            "fit_metadata_by_t": {str(t): dict(m) for t, m in self._last_fit_metadata.items()},
            "schedule": [{"t": t, "direction_used_fit_t": used, "direction_fit_t": fit}
                         for t, used, fit in self.schedule]}


def make_runtime(cell):
    if cell.get("arm") not in ("Native", "Loop", "Online"):
        raise ValueError("Phase 11 permits only Native, Loop and Online")
    if cell["arm"] != "Online":
        return None
    if cell.get("intervention_mode", "spectral") != "spectral":
        raise ValueError("Phase 11 Online requires spectral intervention")
    return Phase11Runtime(cell["direction_policy"], cell["k"], cell["strength"])


def validate_calls(runtime, k=None):
    k = runtime.k if k is None else k
    if k != runtime.k or not runtime.context_summaries:
        raise RuntimeError("missing Phase 11 forward evidence or mismatched K")
    if any(c["timesteps"] != list(range(k)) for c in runtime.context_summaries):
        raise RuntimeError("Phase 11 forward lacks ordered K callbacks")
    if len(runtime.calls) != len(runtime.context_summaries) * k:
        raise RuntimeError("Phase 11 callback count differs from K per forward")
    schedule = direction_schedule(runtime.direction_policy, k)
    for call in runtime.calls:
        _, used, fit = schedule[call["t"]]
        expected_fit = fit if call["forward_kind"] == "prefill" else None
        expected_applied = used is not None and runtime.strength != 0
        if call["direction_used_fit_t"] != used or call["direction_fit_t"] != expected_fit:
            raise RuntimeError("Phase 11 fit/use direction timing differs from frozen schedule")
        if call["nonanswer_max_abs_change"] != 0 or bool(call["applied"]) != expected_applied:
            raise RuntimeError("Phase 11 mutation position or timing differs")
        if not expected_applied and call["answer_max_abs_change"] != 0:
            raise RuntimeError("Phase 11 inactive callback mutated the target")
    return len(runtime.context_summaries)
