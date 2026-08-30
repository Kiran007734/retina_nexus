from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, JSON, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, UUIDType


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    model_name: Mapped[str] = mapped_column(String(120), index=True)
    model_type: Mapped[str] = mapped_column(String(80))
    version: Mapped[str] = mapped_column(String(80), index=True)
    training_dataset: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_size: Mapped[str | None] = mapped_column(String(32), nullable=True)
    performance_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(120), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    artifact_kind: Mapped[str] = mapped_column(String(48), server_default=text("'FINE_TUNED_MODEL'"))
    artifact_status: Mapped[str] = mapped_column(String(40), server_default=text("'MODEL_MISSING'"))
    availability_status: Mapped[str] = mapped_column(String(40), server_default=text("'MODEL_MISSING'"))
    load_error: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
