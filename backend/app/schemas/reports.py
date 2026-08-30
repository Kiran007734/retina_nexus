from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReportGenerateRequest(BaseModel):
    session_id: UUID


class ReportPayload(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    screening_id: UUID
    session_id: UUID
    patient_identifier: str | None = None
    eye: str
    generated_at: datetime | None = None
    image_quality: dict = Field(default_factory=dict)
    ai_assessment: dict = Field(default_factory=dict)
    clinical_evidence: dict = Field(default_factory=dict)
    explainability: dict = Field(default_factory=dict)
    retinaguard: dict = Field(default_factory=dict)
    recommended_action: str | None = None
    clinician_decision: dict | None = None
    disclaimer: str


class ReportResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    report_id: UUID
    session_id: UUID
    status: str
    download_url: str | None = None
    created_at: datetime | None = None
    report: ReportPayload | None = None
