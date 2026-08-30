"""Interfaces for future out-of-distribution validation.

No OOD decision is made here. Feature extraction and distribution summaries are
available so a validated detector can be added without changing API contracts.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class OODDetector(Protocol):
    def score(self, feature_vector: dict[str, float]) -> float: ...


def summarize_quality_distribution(feature_vectors: Iterable[dict[str, float]]) -> dict[str, dict[str, float | None]]:
    values: dict[str, list[float]] = {}
    for vector in feature_vectors:
        for key, value in vector.items():
            values.setdefault(key, []).append(float(value))
    return {key: {"count": len(items), "mean": sum(items) / len(items) if items else None, "min": min(items) if items else None, "max": max(items) if items else None} for key, items in sorted(values.items())}
