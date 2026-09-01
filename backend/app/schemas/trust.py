from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPredictionInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    predicted_grade: int = Field(ge=0, le=4)
    predicted_grade_label: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class TrustRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID | None = None
    model_predictions: list[ModelPredictionInput] = Field(default_factory=list)


class TrustResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID
    trust_score: float
    trust_category: str
    reliability_score: float | None = None
    reliability_state: str | None = None
    contributing_factors: list[dict]
    risk_flags: list[dict[str, str]]
    recommended_action: str
    calibration: dict
    uncertainty: dict
    model_disagreement: dict
    ood: dict
    signal_snapshot: dict
    configuration: dict
    reason_summary: list[str]
    confidence: dict = Field(default_factory=dict)
    image_quality_status: str = "UNAVAILABLE"
    ood_status: str = "UNAVAILABLE"
    evidence_status: str = "UNAVAILABLE"
    explanation_status: str = "UNAVAILABLE"
    warnings: list[dict[str, str]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    recommended_safe_action: str = "PROFESSIONAL_REVIEW_RECOMMENDED"
    assessment_status: str = "COMPLETED"
    available_signals: dict[str, str] = Field(default_factory=dict)
    decision_trace: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    note: str
