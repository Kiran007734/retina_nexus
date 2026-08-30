from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, UUIDType


class ExplainabilityResult(Base):
    """Persisted model-linked explanation and evidence-agreement artifact."""

    __tablename__ = "explainability_results"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    screening_session_id: Mapped[UUID] = mapped_column(ForeignKey("screening_sessions.id"), unique=True, index=True)
    fundus_image_id: Mapped[UUID] = mapped_column(ForeignKey("fundus_images.id"), index=True)
    predicted_class: Mapped[int] = mapped_column()
    predicted_class_label: Mapped[str] = mapped_column(String(80))
    model_version: Mapped[str] = mapped_column(String(80))
    heatmap_data_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    overlay_data_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_attention_map_data_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    lesion_evidence_map_data_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    attention_agreement_status: Mapped[str] = mapped_column(String(40))
    attention_agreement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    attention_agreement_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explanation_stability: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    counterfactual: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
