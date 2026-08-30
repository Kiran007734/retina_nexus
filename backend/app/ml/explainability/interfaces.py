"""Interfaces and error types for classifier explainability."""

from __future__ import annotations

from typing import Any, Protocol


class ExplainabilityNotConfiguredError(RuntimeError):
    """Raised when a model-linked explanation cannot be generated."""


class ExplainableClassifier(Protocol):
    async def explain_async(self, image_bytes: bytes, target_class: int | None = None) -> Any:
        """Return a prediction and normalized spatial attention map."""

    async def classify(self, image_bytes: bytes) -> Any:
        """Return the classifier prediction for a transformed image."""
