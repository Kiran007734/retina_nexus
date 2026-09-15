"""Shared, explicit clinical-rule computations used by training and runtime.

These are engineering mappings, not universal clinical guidance.  The
referable decision is intentionally independent from the five-class severity
argmax: severity is the most probable grade, while referability is the sum of
the configured referable-grade probabilities compared with the fixed 0.5
operating point.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


REFERABLE_PROBABILITY_THRESHOLD = 0.5


def referable_probability(probabilities: Sequence[float], referable_grades: Iterable[int] = (2, 3, 4)) -> float:
    return float(sum(float(probabilities[index]) for index in referable_grades if 0 <= index < len(probabilities)))


def is_referable_probability(
    probabilities: Sequence[float],
    referable_grades: Iterable[int] = (2, 3, 4),
    threshold: float = REFERABLE_PROBABILITY_THRESHOLD,
) -> bool:
    return referable_probability(probabilities, referable_grades) >= float(threshold)
