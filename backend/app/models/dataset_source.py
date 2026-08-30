from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class DatasetSource(Base):
    __tablename__ = "dataset_source"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("dataset.id"), index=True)
    source_type: Mapped[str] = mapped_column(String(40))
    source_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    license: Mapped[str | None] = mapped_column(String(255), nullable=True)
    access_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    acquisition_status: Mapped[str] = mapped_column(String(40), default="not_started")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    dataset = relationship("Dataset", back_populates="sources")
