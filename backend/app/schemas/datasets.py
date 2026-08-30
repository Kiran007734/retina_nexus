from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    slug: str
    name: str
    purpose: str
    status: str
    availability_status: str = "MISSING"
    raw_path: str
    latest_version: str | None = None
    image_count: int | None = None
    readiness_score: float | None = None
    created_at: datetime


class DatasetStatisticsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    dataset_id: UUID
    dataset_version: str | None = None
    total_files: int
    readable_files: int
    corrupted_files: int
    duplicate_exact_count: int
    duplicate_perceptual_count: int
    class_distribution: dict | None = None
    resolution_statistics: dict | None = None
    metadata_completeness: float | None = None
    readiness_score: float | None = None
    created_at: datetime | None = None


class DatasetValidationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    dataset_id: UUID
    dataset_version: str | None = None
    validation_run_id: UUID | None = None
    status: str
    report_path: str | None = None
    leakage_report_path: str | None = None
    summary: dict | None = None
    readiness_score: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
