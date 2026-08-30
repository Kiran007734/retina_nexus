from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class DatasetStatistics(Base):
    __tablename__ = "dataset_statistics"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    dataset_version_id: Mapped[UUID] = mapped_column(ForeignKey("dataset_version.id"), index=True)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    readable_files: Mapped[int] = mapped_column(Integer, default=0)
    corrupted_files: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_exact_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_perceptual_count: Mapped[int] = mapped_column(Integer, default=0)
    class_distribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    resolution_statistics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metadata_completeness: Mapped[float | None] = mapped_column(Float, nullable=True)
    readiness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    dataset_version = relationship("DatasetVersion", back_populates="statistics")
