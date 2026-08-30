from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, JSON, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class ScreeningStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class FinalDecision(StrEnum):
    ROUTINE = "routine"
    REFER = "refer"
    URGENT = "urgent"
    INCONCLUSIVE = "inconclusive"


class ScreeningSession(Base):
    __tablename__ = "screening_sessions"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id"), index=True)
    fundus_image_id: Mapped[UUID] = mapped_column(ForeignKey("fundus_images.id"), index=True)
    status: Mapped[ScreeningStatus] = mapped_column(Enum(ScreeningStatus), default=ScreeningStatus.QUEUED)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    patient = relationship("Patient", back_populates="screening_sessions")
    fundus_image = relationship("FundusImage", back_populates="screening_sessions")
    result = relationship("ScreeningResult", back_populates="session", uselist=False, cascade="all, delete-orphan")
    report = relationship("GeneratedReport", back_populates="session", uselist=False)


class ScreeningResult(Base):
    __tablename__ = "screening_results"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("screening_sessions.id"), unique=True, index=True)
    dr_grade: Mapped[int | None] = mapped_column(nullable=True)
    referable_dr: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_decision: Mapped[FinalDecision | None] = mapped_column(Enum(FinalDecision), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    lesion_evidence: Mapped[list | None] = mapped_column(JSON, nullable=True)
    explainability_artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    session = relationship("ScreeningSession", back_populates="result")
