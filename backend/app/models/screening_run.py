from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, UUIDType


class ScreeningRun(Base):
    """Run-level state and complete orchestration artifact."""

    __tablename__ = "screening_runs"

    id: Mapped[UUID] = mapped_column(UUIDType, ForeignKey("screening_sessions.id"), primary_key=True)
    fundus_image_id: Mapped[UUID] = mapped_column(ForeignKey("fundus_images.id"), index=True)
    initiating_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    stage_status: Mapped[dict] = mapped_column(JSON)
    stage_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    stage_errors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    classification: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    lesions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explainability: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retinaguard: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    triage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_versions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
