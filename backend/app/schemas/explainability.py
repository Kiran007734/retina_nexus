from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ExplainabilityResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_id: UUID
    screening_session_id: UUID
    predicted_class: int
    predicted_class_label: str
    model_version: str
    classification: dict
    grad_cam: dict
    lesion_evidence_map_data_uri: str | None = None
    attention_lesion_agreement: dict
    explanation_stability: dict
    counterfactual: dict
    note: str
