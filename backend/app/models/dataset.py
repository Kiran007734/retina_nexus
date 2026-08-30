from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class DatasetStatus(StrEnum):
    NOT_ACQUIRED = "not_acquired"
    AVAILABLE = "available"
    VALIDATING = "validating"
    READY = "ready"
    BLOCKED = "blocked"


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(180))
    purpose: Mapped[str] = mapped_column(Text)
    status: Mapped[DatasetStatus] = mapped_column(Enum(DatasetStatus), default=DatasetStatus.NOT_ACQUIRED)
    raw_path: Mapped[str] = mapped_column(String(512))
    registry_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=datetime.utcnow)

    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")
    sources = relationship("DatasetSource", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base):
    __tablename__ = "dataset_version"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("dataset.id"), index=True)
    version: Mapped[str] = mapped_column(String(80))
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    file_count: Mapped[int | None] = mapped_column(nullable=True)
    image_count: Mapped[int | None] = mapped_column(nullable=True)
    label_count: Mapped[int | None] = mapped_column(nullable=True)
    manifest_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    dataset = relationship("Dataset", back_populates="versions")
    statistics = relationship("DatasetStatistics", back_populates="dataset_version", cascade="all, delete-orphan")
    validation_runs = relationship("DatasetValidationRun", back_populates="dataset_version", cascade="all, delete-orphan")
