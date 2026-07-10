"""Pure numerical primitives used by LoopScope probes and selectors."""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Sequence


EFFECTIVE_RANK_ESTIMATOR = "gram_spectrum_shannon_effective_rank"
EFFECTIVE_RANK_ESTIMATOR_VERSION = "2"


class MetricError(ValueError):
    """Raised instead of silently replacing invalid measurements with zero."""


def normalize_distribution(values: Iterable[float]) -> List[float]:
    items = [_finite(value, "distribution value") for value in values]
    if not items:
        raise MetricError("distribution must not be empty")
    if any(value < 0.0 for value in items):
        raise MetricError("distribution values must be non-negative")
    total = sum(items)
    if total <= 0.0:
        raise MetricError("distribution mass must be positive")
    return [value / total for value in items]


def softmax(logits: Iterable[float]) -> List[float]:
    values = [_finite(value, "logit") for value in logits]
    if not values:
        raise MetricError("logits must not be empty")
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    return normalize_distribution(weights)


def choice_entropy(probabilities: Iterable[float], eps: float = 1e-12) -> float:
    probs = normalize_distribution(probabilities)
    return -sum(value * math.log(max(value, eps)) for value in probs)


def kl_to_reference(
    probabilities: Iterable[float], reference: Iterable[float], eps: float = 1e-12
) -> float:
    left = normalize_distribution(probabilities)
    right = normalize_distribution(reference)
    if len(left) != len(right):
        raise MetricError("KL distributions must have equal length")
    return sum(p * math.log(max(p, eps) / max(q, eps)) for p, q in zip(left, right))


def top1_agreement(probabilities: Sequence[float], reference: Sequence[float]) -> float:
    left = normalize_distribution(probabilities)
    right = normalize_distribution(reference)
    if len(left) != len(right):
        raise MetricError("top-1 distributions must have equal length")
    return 1.0 if _argmax(left) == _argmax(right) else 0.0


def relative_activity(
    residual_norm: float, state_norm: float, eps: float = 1e-12
) -> float:
    residual = _positive_norm(residual_norm, "first residual norm", eps)
    state = _positive_norm(state_norm, "state norm", eps)
    return residual / state


def contraction_ratio(
    second_residual_norm: float, first_residual_norm: float, eps: float = 1e-12
) -> float:
    second = _positive_norm(second_residual_norm, "second residual norm", eps)
    first = _positive_norm(first_residual_norm, "first residual norm", eps)
    return second / first


def activity_contraction_score(activity: float, contraction: float) -> float:
    r_value = _finite(activity, "activity")
    q_value = _finite(contraction, "contraction")
    if r_value < 0.0 or q_value < 0.0:
        raise MetricError("activity and contraction must be non-negative")
    return r_value * max(0.0, 1.0 - q_value)


def effective_rank_from_singular_values(singular_values: Iterable[float]) -> float:
    """Entropy effective rank of the Gram spectrum induced by singular values.

    If ``s_i`` are singular values of the centered representation matrix, the
    eigenvalues of its unscaled Gram/covariance spectrum are proportional to
    ``s_i**2``.  Scale cancels during normalization, so this function uses
    ``p_i = s_i**2 / sum_j(s_j**2)`` before exponentiating Shannon entropy.
    """

    return effective_rank_measurement_from_singular_values(singular_values)[
        "effective_rank"
    ]


def effective_rank_measurement_from_singular_values(
    singular_values: Iterable[float],
) -> Dict[str, Any]:
    """Measure effective rank and explicitly identify an exactly zero spectrum.

    Roy--Vetterli effective rank assumes positive spectral mass. Version 2
    extends only the exactly-zero centered spectrum: it reports zero varying
    dimensions and records the exceptional case without epsilon regularization
    or a near-zero threshold.
    """

    values = [_finite(value, "singular value") for value in singular_values]
    if any(value < 0.0 for value in values):
        raise MetricError("singular values must be non-negative")
    squared_spectrum = [
        _finite(value * value, "squared singular value") for value in values
    ]
    centered_spectrum_mass = _finite(
        math.fsum(squared_spectrum), "centered spectrum mass"
    )
    if centered_spectrum_mass == 0.0:
        return {
            "effective_rank": 0.0,
            "zero_centered_spectrum": True,
            "centered_spectrum_mass": 0.0,
        }
    probabilities = [value / centered_spectrum_mass for value in squared_spectrum]
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    return {
        "effective_rank": _finite(math.exp(entropy), "effective rank"),
        "zero_centered_spectrum": False,
        "centered_spectrum_mass": centered_spectrum_mass,
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    items = [_finite(value, "summary value") for value in values]
    if not items:
        raise MetricError("cannot summarize an empty measurement set")
    return {
        "count": float(len(items)),
        "mean": mean(items),
        "median": median(items),
        "p90": quantile(items, 0.9),
    }


def population_std(values: Iterable[float]) -> float:
    items = [_finite(value, "standard-deviation value") for value in values]
    if not items:
        raise MetricError("cannot compute standard deviation of an empty set")
    return pstdev(items)


def quantile(values: Iterable[float], probability: float) -> float:
    items = sorted(_finite(value, "quantile value") for value in values)
    if not items:
        raise MetricError("cannot compute a quantile of an empty set")
    if not 0.0 <= probability <= 1.0:
        raise MetricError("quantile probability must be in [0, 1]")
    if len(items) == 1:
        return items[0]
    position = probability * (len(items) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return items[low]
    fraction = position - low
    return items[low] * (1.0 - fraction) + items[high] * fraction


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _positive_norm(value: float, name: str, eps: float) -> float:
    item = _finite(value, name)
    if item <= eps:
        raise MetricError("%s must be greater than eps=%g, got %g" % (name, eps, item))
    return item


def _finite(value: float, name: str) -> float:
    item = float(value)
    if not math.isfinite(item):
        raise MetricError("%s must be finite, got %r" % (name, value))
    return item
