from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ImageQualityAssessment:
    score: float | None = None
    decision: str = "pending"
    reasons: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    dr_grade: int | None = None
    referable_dr: bool | None = None
    confidence: float | None = None
    calibrated_confidence: float | None = None
    uncertainty: float | None = None
    model_version: str | None = None


@dataclass
class LesionAnalysis:
    lesions: list[dict] = field(default_factory=list)


@dataclass
class ExplainabilityArtifact:
    artifact_path: str | None = None
    method: str = "grad-cam"


class ImageQualityService(Protocol):
    async def assess(self, image_bytes: bytes) -> ImageQualityAssessment: ...


class ImagePreprocessingService(Protocol):
    async def prepare(self, image_bytes: bytes, quality: ImageQualityAssessment) -> bytes: ...


class DRClassificationService(Protocol):
    async def classify(self, image_bytes: bytes) -> ClassificationResult: ...


class LesionDetectionService(Protocol):
    async def detect(self, image_bytes: bytes) -> LesionAnalysis: ...


class ExplainabilityService(Protocol):
    async def explain(self, image_bytes: bytes, classification: ClassificationResult) -> ExplainabilityArtifact: ...


class EvidenceVerificationService(Protocol):
    async def verify(self, lesions: LesionAnalysis, explanation: ExplainabilityArtifact) -> dict: ...


class RetinaGuardService(Protocol):
    async def score(self, quality: ImageQualityAssessment, classification: ClassificationResult, evidence: dict) -> float | None: ...


class ReportGenerationService(Protocol):
    async def generate(self, screening_id: str, classification: ClassificationResult, trust_score: float | None) -> str: ...
