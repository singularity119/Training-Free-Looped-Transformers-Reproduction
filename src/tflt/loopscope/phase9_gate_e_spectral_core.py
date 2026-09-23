"""Pure-Python direction-policy and intervention-mode choices for Gate E."""

from __future__ import annotations

from typing import List, Optional, Tuple


LAG1_ONLINE = "Lag1-Online"
LAG1_MATCHED_NORM = "Lag1-Matched-norm"
LAG1_ARMS = (LAG1_ONLINE, LAG1_MATCHED_NORM)
DIRECTION_POLICIES = ("fixed_t0", "lag1")
INTERVENTION_MODES = ("spectral", "matched_norm")


def validate_policy_and_mode(direction_policy: str, intervention_mode: str) -> None:
    if direction_policy not in DIRECTION_POLICIES:
        raise ValueError("direction_policy must be explicitly fixed_t0 or lag1")
    if intervention_mode not in INTERVENTION_MODES:
        raise ValueError("intervention_mode must be explicitly spectral or matched_norm")


def direction_schedule(
    direction_policy: str, k: int
) -> List[Tuple[int, Optional[int], Optional[int]]]:
    """Return ``(t, direction_used_fit_t, direction_fit_t)`` for one policy.

    ``fixed_t0`` uses only the direction fitted at t0. ``lag1`` fits the
    current native residual for the next round, never the current one. Neither
    policy performs an unused fit at the final round.
    """

    if direction_policy not in DIRECTION_POLICIES:
        raise ValueError("direction_policy must be explicitly fixed_t0 or lag1")
    if k not in (2, 3, 4):
        raise ValueError("Gate E scheduling supports K=2, 3 or 4")
    result = [(0, None, 0)]
    for t in range(1, k):
        source_t = 0 if direction_policy == "fixed_t0" else t - 1
        fit_t = t if direction_policy == "lag1" and t < k - 1 else None
        result.append((t, source_t, fit_t))
    return result


def lag1_schedule(k: int) -> List[Tuple[int, Optional[int], Optional[int]]]:
    """Compatibility name for the frozen formal Lag1 schedule."""

    return direction_schedule("lag1", k)


def direction_source_for_t(k: int, t: int) -> Optional[int]:
    """Return the previous-round direction source for the formal Lag1 policy."""

    schedule = lag1_schedule(k)
    for observed_t, source_t, _ in schedule:
        if observed_t == t:
            return source_t
    raise ValueError("timestep is outside the Lag1 schedule")
