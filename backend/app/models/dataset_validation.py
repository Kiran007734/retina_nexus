from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class DatasetValidationRun(Base):
    __tablename__ = "dataset_validation_run"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    dataset_version_id: Mapped[UUID] = mapped_column(ForeignKey("dataset_version.id"), index=True)
    status: Mapped[str] = mapped_column(String(40), default="not_run")
    report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    leakage_report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    readiness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    dataset_version = relationship("DatasetVersion", back_populates="validation_runs")
