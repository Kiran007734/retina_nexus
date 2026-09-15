from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class EyeInput(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class ImageUploadResponse(BaseModel):
    image_id: UUID
    patient_id: UUID
    eye: EyeInput
    quality_decision: str
    message: str


class QualityIssueResponse(BaseModel):
    type: str
    severity: str
    message: str
    recommendation: str


class QualityAssessmentResponse(BaseModel):
    image_id: UUID
    quality_decision: str
    quality_score: float = Field(ge=0, le=1)
    final_quality_score: float = Field(ge=0, le=1)
    component_scores: dict[str, float]
    metrics: dict[str, float]
    issues: list[QualityIssueResponse]
    recommended_action: str
    enhancement_applied: bool
    enhancement_passes: int
    recheck_score: float | None = None
    recheck_decision: str | None = None
    recheck_issues: list[QualityIssueResponse] = Field(default_factory=list)
    next_action: str
    input_metadata: dict
    feature_vector: dict[str, float]
    quality_band: str | None = None
    ai_eligible: bool | None = None
    recoverable_issues: list[str] = Field(default_factory=list)
    non_recoverable_issues: list[str] = Field(default_factory=list)
    hard_focus_floor_passed: bool | None = None
    enhanced_image_available: bool = False
    enhanced_image_url: str | None = None
    quality_gate_version: str = "image-trust-gate-v2-adaptive"
