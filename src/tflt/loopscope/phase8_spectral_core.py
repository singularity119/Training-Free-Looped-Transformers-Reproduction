"""Scalar Python references only; no fitted basis or runtime intervention."""

import math
from typing import Sequence, Tuple


def _vectors(delta: Sequence[float], direction: Sequence[float]):
    delta, direction = tuple(delta), tuple(direction)
    if not delta or len(delta) != len(direction):
        raise ValueError("delta and direction must have the same nonzero dimension")
    if not all(math.isfinite(x) for x in delta + direction):
        raise ValueError("vectors must be finite")
    if not math.isclose(math.hypot(*direction), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("direction must be unit length")
    return delta, direction


def rank1_projection(delta: Sequence[float], direction: Sequence[float]) -> Tuple[float, ...]:
    """Return v(v^T delta), requiring a finite unit v."""
    delta, direction = _vectors(delta, direction)
    coefficient = math.fsum(x * v for x, v in zip(delta, direction))
    return tuple(coefficient * v for v in direction)


def soft_damping(delta: Sequence[float], direction: Sequence[float], strength: float) -> Tuple[float, ...]:
    """Return delta - strength * v(v^T delta), for strength in [0, 1]."""
    delta, direction = _vectors(delta, direction)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    projection = rank1_projection(delta, direction)
    return tuple(x - strength * p for x, p in zip(delta, projection))


def matched_norm_uniform(delta: Sequence[float], direction: Sequence[float], strength: float) -> Tuple[float, ...]:
    """Scale this delta uniformly to its own spectral-damped norm.

    This is a per-vector control, not a fitted global alpha. Zero stays zero.
    """
    delta, direction = _vectors(delta, direction)
    damped = soft_damping(delta, direction, strength)
    original_norm = math.hypot(*delta)
    if original_norm == 0.0:
        return tuple(0.0 for _ in delta)
    scale = math.hypot(*damped) / original_norm
    return tuple(scale * x for x in delta)
