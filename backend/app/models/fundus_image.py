from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class Eye(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class QualityDecision(StrEnum):
    PENDING = "pending"
    GRADABLE = "gradable"
    BORDERLINE = "borderline"
    UNGRADABLE = "ungradable"
    # Legacy states are retained so existing records remain readable.
    ACCEPT = "accept"
    ENHANCE = "enhance"
    REJECT = "reject"


class FundusImage(Base):
    __tablename__ = "fundus_images"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id"), index=True)
    eye: Mapped[Eye] = mapped_column(Enum(Eye))
    storage_path: Mapped[str] = mapped_column(String(512))
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(64))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    image_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_decision: Mapped[QualityDecision] = mapped_column(Enum(QualityDecision), default=QualityDecision.PENDING)
    quality_assessment: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enhanced_storage_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    enhancement_passes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    patient = relationship("Patient", back_populates="fundus_images")
    screening_sessions = relationship("ScreeningSession", back_populates="fundus_image")
