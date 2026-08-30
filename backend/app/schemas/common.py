from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str


class PatientCreate(BaseModel):
    anonymized_identifier: str = Field(min_length=3, max_length=64)
    age_group: str | None = Field(default=None, max_length=32)
    clinical_metadata: str | None = Field(default=None, max_length=2000)


class PatientResponse(APIModel):
    id: UUID
    anonymized_identifier: str
    age_group: str | None
    created_at: datetime


class ModelVersionResponse(APIModel):
    id: UUID
    model_name: str
    model_type: str
    version: str
    training_dataset: str | None
    input_size: str | None
    performance_metrics: dict | None
    training_config: dict | None = None
    dataset_version: str | None = None
    file_path: str | None = None
    checksum: str | None = None
    artifact_kind: str = "FINE_TUNED_MODEL"
    artifact_status: str = "MODEL_MISSING"
    availability_status: str = "MODEL_MISSING"
    load_error: str | None = None
    is_active: bool
    created_at: datetime
