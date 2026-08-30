from dataclasses import dataclass

from app.services.interfaces import (
    ClassificationResult,
    ExplainabilityArtifact,
    ImageQualityAssessment,
    LesionAnalysis,
)


@dataclass
class ScreeningPipelineOutput:
    quality: ImageQualityAssessment
    classification: ClassificationResult
    lesions: LesionAnalysis
    explainability: ExplainabilityArtifact
    evidence: dict
    trust_score: float | None


class ScreeningOrchestrator:
    """Coordinates the trust-first pipeline independently of HTTP and persistence."""

    def __init__(self, services: dict):
        self.quality_service = services["quality"]
        self.preprocessing_service = services["preprocessing"]
        self.classification_service = services["classification"]
        self.lesion_service = services["lesions"]
        self.explainability_service = services["explainability"]
        self.evidence_service = services["evidence"]
        self.trust_service = services["trust"]

    async def run(self, image_bytes: bytes) -> ScreeningPipelineOutput:
        quality = await self.quality_service.assess(image_bytes)
        prepared = await self.preprocessing_service.prepare(image_bytes, quality)
        classification = await self.classification_service.classify(prepared)
        lesions = await self.lesion_service.detect(prepared)
        explainability = await self.explainability_service.explain(prepared, classification)
        evidence = await self.evidence_service.verify(lesions, explainability)
        trust_score = await self.trust_service.score(quality, classification, evidence)
        return ScreeningPipelineOutput(quality, classification, lesions, explainability, evidence, trust_score)
