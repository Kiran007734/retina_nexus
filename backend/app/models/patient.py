from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    anonymized_identifier: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    age_group: Mapped[str | None] = mapped_column(String(32), nullable=True)
    clinical_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=datetime.utcnow)

    fundus_images = relationship("FundusImage", back_populates="patient", cascade="all, delete-orphan")
    screening_sessions = relationship("ScreeningSession", back_populates="patient", cascade="all, delete-orphan")
