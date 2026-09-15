"""Backend-local copy of the shared referable probability rule.

The API is started from ``backend/`` in the supported local/deployment
workflow, so this module must remain importable without the repository root on
``sys.path``.
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
