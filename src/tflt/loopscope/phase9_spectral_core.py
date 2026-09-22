"""Pure-Python references for the Phase 9 per-prompt direction methods.

The actual residual callback lives in :mod:`phase9_runtime` and imports torch
only when a tensor is being processed.  Keeping these scalar operations here
gives the local Gate A tests a torch-free contract for the two intervention
arms and for the prompt-prefix mask.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple


ONLINE_T0 = "Online-t0"
MATCHED_NORM = "Matched-norm"
STRENGTH = 0.5


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _vectors(delta: Sequence[float], direction: Sequence[float]):
    delta, direction = tuple(delta), tuple(direction)
    if not delta or len(delta) != len(direction):
        raise ValueError("delta and direction must have the same nonzero dimension")
    if not all(math.isfinite(value) for value in delta + direction):
        raise ValueError("vectors must be finite")
    direction_norm = _norm(direction)
    if not math.isclose(direction_norm, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("direction must be unit length")
    return delta, direction


def rank1_projection(delta: Sequence[float], direction: Sequence[float]) -> Tuple[float, ...]:
    """Return ``v(v^T delta)`` for a finite unit direction ``v``."""

    delta, direction = _vectors(delta, direction)
    coefficient = math.fsum(value * component for value, component in zip(delta, direction))
    return tuple(coefficient * component for component in direction)


def soft_damping(
    delta: Sequence[float], direction: Sequence[float], strength: float = STRENGTH
) -> Tuple[float, ...]:
    """Return ``delta - strength * v(v^T delta)``."""

    delta, direction = _vectors(delta, direction)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    projection = rank1_projection(delta, direction)
    return tuple(value - strength * projected for value, projected in zip(delta, projection))


def matched_norm_uniform(
    delta: Sequence[float], direction: Sequence[float], strength: float = STRENGTH
) -> Tuple[float, ...]:
    """Scale ``delta`` uniformly to the norm of its spectral-damped version.

    This is the Phase 9 per-state control.  It preserves the current residual
    direction, while matching the amount of norm reduction that the Online-t0
    rank-one damping would have produced for the same state.  A zero residual
    remains exactly zero.
    """

    delta, direction = _vectors(delta, direction)
    damped = soft_damping(delta, direction, strength)
    original_norm = _norm(delta)
    if original_norm == 0.0:
        return tuple(0.0 for _ in delta)
    scale = _norm(damped) / original_norm
    return tuple(scale * value for value in delta)


def prefix_positions(
    token_ids: Sequence[int],
    answer_position: int,
    special_token_ids: Iterable[int] = (),
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Return valid retained prompt positions up to and including ``p``.

    ``answer_position`` is the last retained prompt token, not the last model
    input position.  Continuation tokens are therefore excluded by construction
    even when a multi-token candidate is being scored.  Padding and tokenizer
    special tokens are removed from the actual retained prefix.
    """

    tokens = tuple(token_ids)
    if not tokens or not isinstance(answer_position, int):
        raise ValueError("token_ids must be nonempty and answer_position must be an integer")
    if not 0 <= answer_position < len(tokens):
        raise ValueError("answer_position must be inside the retained input")
    if attention_mask is not None and len(attention_mask) != len(tokens):
        raise ValueError("attention_mask length differs from retained input")
    special = set(special_token_ids)
    valid = []
    for index in range(answer_position + 1):
        if attention_mask is not None and not bool(attention_mask[index]):
            continue
        token = tokens[index]
        if token in special or (pad_token_id is not None and token == pad_token_id):
            continue
        valid.append(index)
    if not valid:
        raise ValueError("retained prompt has no valid non-special, non-padding positions")
    if answer_position not in valid:
        raise ValueError("answer_position is padding or a tokenizer special token")
    return tuple(valid)


def canonicalize_direction(direction: Sequence[float]) -> Tuple[float, ...]:
    """Normalize a direction and make its largest-absolute component positive."""

    values = tuple(float(value) for value in direction)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("direction must be finite and nonempty")
    norm = _norm(values)
    if norm == 0.0:
        raise ValueError("direction must have nonzero norm")
    pivot = max(range(len(values)), key=lambda index: abs(values[index]))
    sign = -1.0 if values[pivot] < 0.0 else 1.0
    return tuple(sign * value / norm for value in values)
