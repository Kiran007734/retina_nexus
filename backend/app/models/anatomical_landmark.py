from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, UUIDType


class AnatomicalLandmark(Base):
    __tablename__ = "anatomical_landmarks"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    screening_session_id: Mapped[UUID] = mapped_column(ForeignKey("screening_sessions.id"), index=True)
    fundus_image_id: Mapped[UUID] = mapped_column(ForeignKey("fundus_images.id"), index=True)
    landmark_type: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(40))
    method: Mapped[str] = mapped_column(String(120))
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    radius: Mapped[float | None] = mapped_column(Float, nullable=True)
    x_normalized: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_normalized: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
