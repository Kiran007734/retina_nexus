from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.trust import ModelPredictionInput


class ScreeningCreate(BaseModel):
    patient_id: UUID
    fundus_image_id: UUID


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID | None = None
    model_version: str | None = None


class EvidenceAnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID | None = None


class ScreeningRunRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID | None = None
    run_stability: bool | None = None
    run_counterfactual: bool | None = None
    model_predictions: list[ModelPredictionInput] = Field(default_factory=list)


class ExplainabilityRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID | None = None
    run_stability: bool | None = None
    run_counterfactual: bool | None = None


class ScreeningResponse(BaseModel):
    session_id: UUID
    patient_id: UUID
    status: str
    message: str


class ScreeningRunResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    screening_id: UUID
    screening_session_id: UUID
    patient_id: UUID
    image_id: UUID
    status: str
    primary_status: str = "PENDING"
    evidence_status: str = "NOT_RUN"
    evidence_message: str = ""
    stage_status: dict = Field(default_factory=dict)
    stage_metrics: dict = Field(default_factory=dict)
    stage_errors: dict = Field(default_factory=dict)
    quality: dict | None = None
    classification: dict | None = None
    lesions: dict | None = None
    explainability: dict | None = None
    retinaguard: dict | None = None
    triage: dict | None = None
    model_versions: dict = Field(default_factory=dict)
    error: dict | None = None
    message: str = ""


class ScreeningHistoryItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    screening_id: UUID
    patient_id: UUID
    image_id: UUID
    eye: str
    status: str
    trust_category: str | None = None
    trust_score: float | None = None
    predicted_grade: int | None = None
    predicted_grade_label: str | None = None
    referable_dr: bool | None = None
    triage_recommendation: str | None = None
    created_at: datetime | None = None


class ScreeningResultResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: UUID
    status: str
    dr_grade: int | None = None
    referable_dr: bool | None = None
    confidence: float | None = None
    calibrated_confidence: float | None = None
    uncertainty: float | None = None
    trust_score: float | None = None
    final_decision: str | None = None
    model_version: str | None = None
    created_at: datetime | None = None


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID
    predicted_grade: int
    predicted_grade_label: str
    probabilities: dict[str, float]
    referable_dr: bool
    referable_probability: float
    raw_confidence: float
    model_name: str
    model_version: str
    backbone: str
    referable_mapping: dict
    hierarchical_probabilities: dict[str, dict[str, float]]
    ordinal_mode: bool
    note: str
