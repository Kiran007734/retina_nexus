from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, UUIDType


class RetinaGuardResult(Base):
    """Versioned transparent self-checking output for one screening session."""

    __tablename__ = "retinaguard_results"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    screening_session_id: Mapped[UUID] = mapped_column(ForeignKey("screening_sessions.id"), unique=True, index=True)
    fundus_image_id: Mapped[UUID] = mapped_column(ForeignKey("fundus_images.id"), index=True)
    trust_score: Mapped[float] = mapped_column(Float)
    trust_category: Mapped[str] = mapped_column(String(40))
    contributing_factors: Mapped[list] = mapped_column(JSON)
    risk_flags: Mapped[list] = mapped_column(JSON)
    recommended_action: Mapped[str] = mapped_column(String(512))
    calibration: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    uncertainty: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_disagreement: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ood: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    signal_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    configuration: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reason_summary: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
