from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UUIDType


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"
    REQUEST_RECAPTURE = "request_recapture"
    # Legacy values are retained for existing review records.
    AGREE = "agree"
    DISAGREE = "disagree"
    INCONCLUSIVE = "inconclusive"


class ClinicalReview(Base):
    __tablename__ = "clinical_reviews"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    screening_session_id: Mapped[UUID] = mapped_column(ForeignKey("screening_sessions.id"), index=True)
    reviewer_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    reviewer_decision: Mapped[ReviewDecision] = mapped_column(Enum(ReviewDecision))
    agrees_with_ai: Mapped[bool] = mapped_column(Boolean)
    modified_grade: Mapped[int | None] = mapped_column(nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))

    reviewer = relationship("User", back_populates="reviews")
