"""Pure-Python reference aggregation for phase-one LoopPilot signals."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


Vector = Sequence[float]
TokenMatrix = Sequence[Vector]
LayerUpdates = Sequence[TokenMatrix]


def compute_signal_row(
    x0: TokenMatrix,
    y0: TokenMatrix,
    question_mask: Sequence[bool],
    layer_updates: Optional[LayerUpdates] = None,
    x1: Optional[TokenMatrix] = None,
    y1: Optional[TokenMatrix] = None,
    epsilon: float = 1e-12,
) -> Dict[str, Optional[float]]:
    """Compute one request row using only explicitly masked question tokens."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")
    _validate_pair(x0, y0, "x0", "y0")
    if len(question_mask) != len(x0):
        raise ValueError("question_mask length must match token count")
    selected = [idx for idx, include in enumerate(question_mask) if include]
    if not selected:
        raise ValueError("question_mask must select at least one token")

    r0_values = [_norm(_sub(y0[i], x0[i])) / (_norm(x0[i]) + epsilon) for i in selected]
    n0_values = [abs(_norm(y0[i]) / (_norm(x0[i]) + epsilon) - 1.0) for i in selected]

    c0 = None
    if layer_updates is not None:
        if not layer_updates:
            raise ValueError("layer_updates must contain at least one layer")
        for update in layer_updates:
            _validate_matrix(update, len(x0), len(x0[0]), "layer_updates")
        consistency = []
        for token_index in selected:
            token_updates = [layer[token_index] for layer in layer_updates]
            summed = [sum(values) for values in zip(*token_updates)]
            consistency.append(
                _norm(summed) / (sum(_norm(update) for update in token_updates) + epsilon)
            )
        c0 = _median(consistency)

    q1 = None
    residual_cosine = None
    if (x1 is None) != (y1 is None):
        raise ValueError("x1 and y1 must be supplied together")
    if x1 is not None and y1 is not None:
        _validate_pair(x1, y1, "x1", "y1")
        _validate_matrix(x1, len(x0), len(x0[0]), "x1")
        ratios = []
        cosines = []
        for i in selected:
            residual0 = _sub(y0[i], x0[i])
            residual1 = _sub(y1[i], x1[i])
            ratios.append(_norm(residual1) / (_norm(residual0) + epsilon))
            cosines.append(
                _dot(residual0, residual1)
                / ((_norm(residual0) * _norm(residual1)) + epsilon)
            )
        q1 = _median(ratios)
        residual_cosine = _median(cosines)

    result: Dict[str, Optional[float]] = {
        "valid_token_count": len(selected),
        "r0_median": _median(r0_values),
        "r0_p90": _percentile(r0_values, 0.9),
        "r0_max": max(r0_values),
        "c0": c0,
        "n0_median": _median(n0_values),
        "n0_p90": _percentile(n0_values, 0.9),
        "n0_max": max(n0_values),
        "q1": q1,
        "residual_cosine": residual_cosine,
    }
    for name, value in result.items():
        if value is not None and name != "valid_token_count" and not math.isfinite(float(value)):
            raise ValueError("computed signal %s is not finite" % name)
    return result


def _validate_pair(left: TokenMatrix, right: TokenMatrix, left_name: str, right_name: str) -> None:
    if not left:
        raise ValueError("%s must contain tokens" % left_name)
    width = len(left[0])
    if width < 1:
        raise ValueError("%s vectors must be non-empty" % left_name)
    _validate_matrix(left, len(left), width, left_name)
    _validate_matrix(right, len(left), width, right_name)


def _validate_matrix(matrix: TokenMatrix, rows: int, width: int, name: str) -> None:
    if len(matrix) != rows:
        raise ValueError("%s token count mismatch" % name)
    for vector in matrix:
        if len(vector) != width:
            raise ValueError("%s hidden width mismatch" % name)
        for value in vector:
            if not math.isfinite(float(value)):
                raise ValueError("%s contains NaN or Inf" % name)


def _sub(left: Vector, right: Vector) -> List[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def _dot(left: Vector, right: Vector) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _norm(vector: Vector) -> float:
    return math.sqrt(_dot(vector, vector))


def _median(values: Sequence[float]) -> float:
    return _percentile(values, 0.5)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
