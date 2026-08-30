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
    note: str
