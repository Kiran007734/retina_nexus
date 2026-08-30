"""Human-in-the-loop clinical review queue and decision capture."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_claims
from app.database.session import get_db
from app.models.audit_log import AuditLog
from app.models.clinical_review import ClinicalReview, ReviewDecision
from app.models.fundus_image import FundusImage
from app.models.retinaguard_result import RetinaGuardResult
from app.models.screening import FinalDecision, ScreeningResult, ScreeningSession, ScreeningStatus
from app.models.screening_run import ScreeningRun
from app.models.user import User
from app.schemas.reviews import ReviewCreateRequest, ReviewQueueItem, ReviewResponse

router = APIRouter(prefix="/reviews", tags=["clinical review"])


@router.get("/queue", response_model=list[ReviewQueueItem])
async def review_queue(db: AsyncSession = Depends(get_db)) -> list[ReviewQueueItem]:
    runs = (await db.execute(select(ScreeningRun).order_by(ScreeningRun.created_at.desc()))).scalars().all()
    queue: list[ReviewQueueItem] = []
    for run in runs:
        session = await db.get(ScreeningSession, run.id)
        image = await db.get(FundusImage, run.fundus_image_id)
        if session is None or image is None:
            continue
        trust = run.retinaguard or {}
        classification = run.classification or {}
        triage = run.triage or {}
        needs_review = trust.get("trust_category") in {"UNCERTAIN", "UNRELIABLE"} or classification.get("referable_dr") is True or triage.get("recommendation") in {"HUMAN_REVIEW_REQUIRED", "SPECIALIST_REVIEW_RECOMMENDED", "RECAPTURE_OR_SPECIALIST_REVIEW"}
        if run.status != "COMPLETED" or not needs_review:
            continue
        review = await _latest_review(db, session.id)
        queue.append(ReviewQueueItem(
            session_id=session.id, patient_id=session.patient_id, image_id=image.id, eye=image.eye.value,
            status="reviewed" if review else "open", trust_category=trust.get("trust_category"),
            trust_score=trust.get("trust_score"), predicted_grade=classification.get("predicted_grade"),
            predicted_grade_label=classification.get("predicted_grade_label"), referable_dr=classification.get("referable_dr"),
            reason=_review_reason(run), created_at=run.created_at,
            review=await _review_response(db, review) if review else None,
        ))
    return queue


@router.get("/{session_id}", response_model=list[ReviewResponse])
async def session_reviews(session_id: UUID, db: AsyncSession = Depends(get_db)) -> list[ReviewResponse]:
    session = await db.get(ScreeningSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Screening session not found")
    items = (await db.execute(select(ClinicalReview).where(ClinicalReview.screening_session_id == session_id).order_by(ClinicalReview.created_at.desc()))).scalars().all()
    return [await _review_response(db, item) for item in items]


@router.post("/{session_id}", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
async def create_review(
    session_id: UUID,
    payload: ReviewCreateRequest,
    claims: dict = Depends(get_current_claims),
    db: AsyncSession = Depends(get_db),
) -> ReviewResponse:
    session = await db.get(ScreeningSession, session_id)
    reviewer_id = _claim_uuid(claims)
    if session is None:
        raise HTTPException(status_code=404, detail="Screening session not found")
    if claims.get("role") not in {"clinician", "admin"}:
        raise HTTPException(status_code=403, detail="Only a clinician or administrator can record a review")
    if reviewer_id is None:
        raise HTTPException(status_code=401, detail="A clinician identity is required to record a review")
    reviewer = await db.get(User, reviewer_id)
    if reviewer is None or not reviewer.is_active:
        raise HTTPException(status_code=403, detail="Reviewer account is not active")
    decision = ReviewDecision(payload.decision)
    if decision == ReviewDecision.MODIFY and payload.modified_grade is None:
        raise HTTPException(status_code=422, detail="modified_grade is required for a modified AI grade")
    review = ClinicalReview(
        id=uuid4(), screening_session_id=session.id, reviewer_id=reviewer_id,
        reviewer_decision=decision, agrees_with_ai=decision == ReviewDecision.APPROVE,
        modified_grade=payload.modified_grade, feedback=payload.comments,
    )
    db.add(review)
    result = (await db.execute(select(ScreeningResult).where(ScreeningResult.session_id == session.id))).scalar_one_or_none()
    run = await db.get(ScreeningRun, session.id)
    ai_grade = (run.classification or {}).get("predicted_grade") if run else (result.dr_grade if result else None)
    final_grade = payload.modified_grade if decision == ReviewDecision.MODIFY else ai_grade
    if result is None:
        result = ScreeningResult(id=uuid4(), session_id=session.id)
        db.add(result)
    if final_grade is not None:
        result.dr_grade = final_grade
        result.final_decision = FinalDecision.REFER if final_grade >= 2 else FinalDecision.ROUTINE
    if decision == ReviewDecision.REQUEST_RECAPTURE:
        result.final_decision = FinalDecision.INCONCLUSIVE
        session.status = ScreeningStatus.NEEDS_REVIEW
    elif decision == ReviewDecision.REJECT:
        result.final_decision = FinalDecision.INCONCLUSIVE
        session.status = ScreeningStatus.NEEDS_REVIEW
    else:
        session.status = ScreeningStatus.COMPLETE
    session.completed_at = session.completed_at or datetime.now(timezone.utc)
    db.add(AuditLog(
        actor_id=reviewer_id, action="clinical_review.recorded", resource_type="screening_session",
        resource_id=str(session.id), details={"decision": decision.value, "modified_grade": payload.modified_grade, "timestamp": datetime.now(timezone.utc).isoformat()},
    ))
    await db.commit()
    return await _review_response(db, review)


async def _latest_review(db: AsyncSession, session_id: UUID) -> ClinicalReview | None:
    return (await db.execute(select(ClinicalReview).where(ClinicalReview.screening_session_id == session_id).order_by(ClinicalReview.created_at.desc()).limit(1))).scalar_one_or_none()


async def _review_response(db: AsyncSession, review: ClinicalReview | None) -> ReviewResponse:
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    reviewer = await db.get(User, review.reviewer_id)
    return ReviewResponse(
        review_id=review.id, session_id=review.screening_session_id, reviewer_id=review.reviewer_id,
        reviewer_name=reviewer.full_name if reviewer else None, decision=review.reviewer_decision.value,
        modified_grade=review.modified_grade, comments=review.feedback, created_at=review.created_at,
    )


def _claim_uuid(claims: dict) -> UUID | None:
    try:
        return UUID(str(claims.get("sub")))
    except (TypeError, ValueError):
        return None


def _review_reason(run: ScreeningRun) -> str:
    trust = run.retinaguard or {}
    reasons = trust.get("reason_summary") or []
    if reasons:
        return reasons[0]
    if (run.classification or {}).get("referable_dr"):
        return "Referable DR signal"
    return (run.triage or {}).get("recommendation", "Clinical review requested")
