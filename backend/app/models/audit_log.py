from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, JSON, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, UUIDType


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    actor_id: Mapped[UUID | None] = mapped_column(nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(120), index=True)
    resource_type: Mapped[str] = mapped_column(String(80))
    resource_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
