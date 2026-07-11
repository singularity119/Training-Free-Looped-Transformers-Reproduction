"""Torch-backed, in-memory Gate C tensor signal collection.

The module imports torch lazily so the repository's ``python -S`` tests remain
usable without site packages. Raw hidden states never leave this collector.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional


class TensorSignalCollector:
    """Collect R0/C0/N0 and paid second-call diagnostics for one document."""

    def __init__(self, question_mask: Any, epsilon: float = 1e-12) -> None:
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
            raise ValueError("epsilon must be a finite positive number")
        self.question_mask = question_mask
        self.epsilon = float(epsilon)
        self._calls = 0
        self._first: Optional[Dict[str, Any]] = None
        self._residual0: Any = None
        self._second: Optional[Dict[str, float]] = None

    def observe_operator_call(self, x: Any, y: Any, layer_updates: Iterable[Any]) -> None:
        """Observe one paid operator body call without mutating its tensors."""

        torch = _torch()
        if self._calls >= 2:
            raise RuntimeError("Gate C collector accepts exactly one or two operator calls")
        x_float, y_float, mask = _validate_tensors(torch, x, y, self.question_mask)
        residual = (y_float - x_float).detach()
        selected_x = x_float[mask]
        selected_y = y_float[mask]
        selected_residual = residual[mask]

        if self._calls == 0:
            updates = list(layer_updates)
            if not updates:
                raise ValueError("first operator call requires per-layer updates for C0")
            selected_updates = []
            for update in updates:
                update_float, _, update_mask = _validate_tensors(
                    torch, update, update, self.question_mask
                )
                if not torch.equal(update_mask, mask):
                    raise ValueError("layer update mask mismatch")
                selected_updates.append(update_float[mask])
            r0 = _row_norm(torch, selected_residual) / (
                _row_norm(torch, selected_x) + self.epsilon
            )
            n0 = (
                _row_norm(torch, selected_y) / (_row_norm(torch, selected_x) + self.epsilon)
                - 1.0
            ).abs()
            update_stack = torch.stack(selected_updates, dim=0)
            c0_rows = _row_norm(torch, update_stack.sum(dim=0)) / (
                _row_norm(torch, update_stack).sum(dim=0) + self.epsilon
            )
            self._first = {
                "valid_token_count": int(mask.sum().item()),
                "r0_median": _quantile(torch, r0, 0.5),
                "r0_p90": _quantile(torch, r0, 0.9),
                "r0_max": float(r0.max().item()),
                "c0": _quantile(torch, c0_rows, 0.5),
                "n0_median": _quantile(torch, n0, 0.5),
                "n0_p90": _quantile(torch, n0, 0.9),
                "n0_max": float(n0.max().item()),
            }
            self._residual0 = selected_residual
        else:
            if self._first is None or self._residual0 is None:
                raise RuntimeError("second operator call observed before first-call state")
            residual0 = self._residual0
            if residual0.shape != selected_residual.shape:
                raise ValueError("second-call selected residual shape mismatch")
            norm0 = _row_norm(torch, residual0)
            norm1 = _row_norm(torch, selected_residual)
            q1_rows = norm1 / (norm0 + self.epsilon)
            cosine_rows = (residual0 * selected_residual).sum(dim=-1) / (
                norm0 * norm1 + self.epsilon
            )
            self._second = {
                "q1": _quantile(torch, q1_rows, 0.5),
                "residual_cosine": _quantile(torch, cosine_rows, 0.5),
            }
        self._calls += 1

    def controller_probe(self, x0: Any, y0: Any) -> Dict[str, Any]:
        """Return detached first-call scalars after the required baseline call."""

        if self._calls != 1 or self._first is None:
            raise RuntimeError("controller probe requires exactly one observed operator call")
        return dict(self._first)

    def finalize(self, expected_calls: int) -> Dict[str, Any]:
        if expected_calls not in {1, 2}:
            raise ValueError("expected_calls must be 1 or 2")
        if self._calls != expected_calls or self._first is None:
            raise RuntimeError(
                "collector body-call mismatch: expected %d, observed %d"
                % (expected_calls, self._calls)
            )
        result = dict(self._first)
        result["operator_body_calls"] = self._calls
        if expected_calls == 1:
            if self._second is not None:
                raise RuntimeError("not-paid diagnostics unexpectedly exist")
            result.update(
                {"q1": None, "residual_cosine": None, "diagnostics_status": "not_paid"}
            )
        else:
            if self._second is None:
                raise RuntimeError("paid second-call diagnostics are missing")
            result.update(self._second)
            result["diagnostics_status"] = "paid"
        _validate_ranges(result)
        self._residual0 = None
        return result


def _torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - remote dependency contract
        raise RuntimeError("TensorSignalCollector requires torch") from exc
    return torch


def _validate_tensors(torch: Any, x: Any, y: Any, question_mask: Any) -> Any:
    if not hasattr(x, "detach") or not hasattr(y, "detach"):
        raise TypeError("collector inputs must be torch tensors")
    if tuple(x.shape) != tuple(y.shape) or len(x.shape) != 3:
        raise ValueError("collector tensors must share [batch, tokens, hidden] shape")
    x_float = x.detach().float()
    y_float = y.detach().float()
    if not bool(torch.isfinite(x_float).all()) or not bool(torch.isfinite(y_float).all()):
        raise ValueError("collector tensors contain NaN or Inf")
    mask = torch.as_tensor(question_mask, dtype=torch.bool, device=x.device)
    if len(mask.shape) == 1:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape) != tuple(x.shape[:2]):
        raise ValueError("question mask must match [batch, tokens]")
    if not bool(mask.any()):
        raise ValueError("question mask must select at least one token")
    return x_float, y_float, mask


def _row_norm(torch: Any, value: Any) -> Any:
    return torch.linalg.vector_norm(value, ord=2, dim=-1)


def _quantile(torch: Any, value: Any, quantile: float) -> float:
    return float(torch.quantile(value.float(), quantile).item())


def _validate_ranges(result: Dict[str, Any]) -> None:
    numeric = [
        result[name]
        for name in (
            "r0_median",
            "r0_p90",
            "r0_max",
            "c0",
            "n0_median",
            "n0_p90",
            "n0_max",
            "q1",
            "residual_cosine",
        )
        if result.get(name) is not None
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("computed tensor signal contains NaN or Inf")
    for name in ("r0_median", "r0_p90", "r0_max", "n0_median", "n0_p90", "n0_max"):
        if float(result[name]) < 0:
            raise ValueError("%s must be non-negative" % name)
    if result.get("q1") is not None and float(result["q1"]) < 0:
        raise ValueError("q1 must be non-negative")
    if not 0.0 <= float(result["c0"]) <= 1.0 + 1e-6:
        raise ValueError("c0 is outside [0, 1+1e-6]")
    cosine = result.get("residual_cosine")
    if cosine is not None and not -1.0 - 1e-6 <= float(cosine) <= 1.0 + 1e-6:
        raise ValueError("residual_cosine is outside [-1-1e-6, 1+1e-6]")
