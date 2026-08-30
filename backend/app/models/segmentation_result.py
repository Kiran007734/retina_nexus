from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, UUIDType


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    screening_session_id: Mapped[UUID] = mapped_column(ForeignKey("screening_sessions.id"), index=True)
    fundus_image_id: Mapped[UUID] = mapped_column(ForeignKey("fundus_images.id"), index=True)
    structure_type: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(40))
    implementation: Mapped[str] = mapped_column(String(120))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    pixel_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mask_data_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    bounding_regions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    result_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
