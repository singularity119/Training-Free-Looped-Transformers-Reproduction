"""Dependency-free paired summaries for frozen LoopPilot evaluations."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Sequence

from tflt.looppilot.offline import PairedSample


def paired_summary(pairs: Sequence[PairedSample]) -> Dict[str, object]:
    if not pairs:
        raise ValueError("paired analysis requires at least one sample")
    outcomes = Counter(pair.outcome for pair in pairs)
    total = len(pairs)
    baseline_correct = sum(pair.baseline_correct for pair in pairs)
    loop_correct = sum(pair.loop_correct for pair in pairs)
    n01 = outcomes["loop_helps"]
    n10 = outcomes["loop_hurts"]
    return {
        "sample_count": total,
        "baseline_accuracy": baseline_correct / total,
        "loop_accuracy": loop_correct / total,
        "paired_accuracy_delta": (loop_correct - baseline_correct) / total,
        "n01": n01,
        "n10": n10,
        "mcnemar_exact_p": mcnemar_exact_p(n01, n10),
        "outcomes": dict(outcomes),
    }


def mcnemar_exact_p(n01: int, n10: int) -> float:
    if n01 < 0 or n10 < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail = min(n01, n10)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)
