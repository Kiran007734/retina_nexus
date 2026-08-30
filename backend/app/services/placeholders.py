"""Explicit failure adapters retained for the legacy composition boundary.

The active HTTP routes use the concrete ML services in ``container.py``. These
adapters remain import-compatible for older integrations, but never emit a
synthetic clinical result.
"""

from app.services.interfaces import (
    ClassificationResult,
    ExplainabilityArtifact,
    ImagePreprocessingService,
    ImageQualityAssessment,
    ImageQualityService,
    LesionAnalysis,
)


class ServiceNotConfiguredError(RuntimeError):
    """Raised when a legacy integration attempts to use an absent service."""


class PlaceholderImageQualityService(ImageQualityService):
    async def assess(self, image_bytes: bytes) -> ImageQualityAssessment:
        raise ServiceNotConfiguredError("Image quality service is not configured; use the Image Trust Gate service.")


class PlaceholderImagePreprocessingService(ImagePreprocessingService):
    async def prepare(self, image_bytes: bytes, quality: ImageQualityAssessment) -> bytes:
        raise ServiceNotConfiguredError("Image preprocessing service is not configured; no fallback preprocessing was applied.")


class PlaceholderDRClassificationService:
    async def classify(self, image_bytes: bytes) -> ClassificationResult:
        raise ServiceNotConfiguredError("DR classifier is not configured; register a trained model artifact before inference.")


class PlaceholderLesionDetectionService:
    async def detect(self, image_bytes: bytes) -> LesionAnalysis:
        raise ServiceNotConfiguredError("Retinal evidence service is not configured; no lesion result was generated.")


class PlaceholderExplainabilityService:
    async def explain(self, image_bytes: bytes, classification: ClassificationResult) -> ExplainabilityArtifact:
        raise ServiceNotConfiguredError("Explainability service is not configured; no heatmap was generated.")


class PlaceholderEvidenceVerificationService:
    async def verify(self, lesions: LesionAnalysis, explanation: ExplainabilityArtifact) -> dict:
        raise ServiceNotConfiguredError("Evidence verification service is not configured; no agreement result was generated.")


class PlaceholderRetinaGuardService:
    async def score(self, quality, classification, evidence) -> float | None:
        raise ServiceNotConfiguredError("RetinaGuard service is not configured; no trust score was generated.")


class PlaceholderReportGenerationService:
    async def generate(self, screening_id: str, classification: ClassificationResult, trust_score: float | None) -> str:
        raise ServiceNotConfiguredError("Report generation service is not configured; no report was generated.")
