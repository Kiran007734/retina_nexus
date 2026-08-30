from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReviewCreateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    decision: str = Field(pattern="^(approve|modify|reject|request_recapture)$")
    modified_grade: int | None = Field(default=None, ge=0, le=4)
    comments: str | None = Field(default=None, max_length=4000)


class ReviewResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    review_id: UUID
    session_id: UUID
    reviewer_id: UUID
    reviewer_name: str | None = None
    decision: str
    modified_grade: int | None = None
    comments: str | None = None
    created_at: datetime | None = None


class ReviewQueueItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: UUID
    patient_id: UUID
    image_id: UUID
    eye: str
    status: str
    trust_category: str | None = None
    trust_score: float | None = None
    predicted_grade: int | None = None
    predicted_grade_label: str | None = None
    referable_dr: bool | None = None
    reason: str
    created_at: datetime | None = None
    review: ReviewResponse | None = None
