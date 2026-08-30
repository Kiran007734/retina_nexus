"""Interfaces for configurable retinal evidence models.

The protocols intentionally do not prescribe a framework. A future trained
segmentation or patch detector can be injected without changing the API or
the persistence contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class SegmentationModel(Protocol):
    name: str
    version: str

    def predict_mask(self, image_rgb: Any) -> Any: ...


class PatchDetector(Protocol):
    name: str
    version: str

    def predict_patches(self, image_rgb: Any, patches: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class LandmarkLocalizer(Protocol):
    name: str
    version: str

    def locate(self, image_rgb: Any) -> dict[str, Any] | None: ...


@dataclass
class EvidenceModuleResult:
    module: str
    category: str
    status: str
    supported: bool
    implementation: str
    confidence: float | None = None
    count: int | None = None
    mask_data_uri: str | None = None
    bounding_regions: list[dict[str, Any]] = field(default_factory=list)
    landmarks: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "category": self.category,
            "status": self.status,
            "supported": self.supported,
            "implementation": self.implementation,
            "confidence": self.confidence,
            "count": self.count,
            "mask_data_uri": self.mask_data_uri,
            "bounding_regions": self.bounding_regions,
            "landmarks": self.landmarks,
            "issues": self.issues,
            "metadata": self.metadata,
        }


class EvidenceModelAdapter(Protocol):
    """Optional trained adapter for replacing a baseline module."""

    module: str
    name: str
    version: str

    def analyze(self, image_rgb: Any, context: dict[str, Any]) -> EvidenceModuleResult: ...


class EvidenceModelNotConfiguredError(RuntimeError):
    """Raised when a requested trained evidence model is unavailable."""
