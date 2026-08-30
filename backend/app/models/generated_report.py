from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    screening_session_id: Mapped[UUID] = mapped_column(ForeignKey("screening_sessions.id"), unique=True, index=True)
    report_status: Mapped[str] = mapped_column(String(32), default="draft")
    report_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    session = relationship("ScreeningSession", back_populates="report")
