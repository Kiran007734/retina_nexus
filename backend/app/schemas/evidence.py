from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EvidenceModuleResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    module: str
    category: str
    status: str
    supported: bool
    implementation: str
    confidence: float | None = None
    count: int | None = None
    mask_data_uri: str | None = None
    probability_map_data_uri: str | None = None
    overlay_data_uri: str | None = None
    bounding_regions: list[dict] = Field(default_factory=list)
    landmarks: list[dict] = Field(default_factory=list)
    issues: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class EvidenceAnalysisResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID
    status: str
    image_metadata: dict
    coarse_to_fine: dict
    modules: dict[str, EvidenceModuleResponse]
    anatomical_landmarks: list[dict]
    evidence_map_data_uri: str | None = None
    dataset_support: dict
    note: str
