"""Operational screening analytics for the workspace dashboard."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.fundus_image import FundusImage
from app.models.screening import ScreeningSession
from app.models.screening_run import ScreeningRun
from app.schemas.analytics import AnalyticsOverviewResponse

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview", response_model=AnalyticsOverviewResponse)
async def analytics_overview(db: AsyncSession = Depends(get_db)) -> AnalyticsOverviewResponse:
    sessions = (await db.execute(select(ScreeningSession).order_by(ScreeningSession.created_at.desc()))).scalars().all()
    runs = (await db.execute(select(ScreeningRun).order_by(ScreeningRun.created_at.desc()))).scalars().all()
    images = (await db.execute(select(FundusImage))).scalars().all()
    today = datetime.now(timezone.utc).date()
    run_by_id = {item.id: item for item in runs}
    status_distribution = Counter(item.status.value for item in sessions)
    severity_distribution: Counter[str] = Counter()
    recent_activity: list[dict] = []
    referable = human_review = 0
    for run in runs:
        classification = run.classification or {}
        trust = run.retinaguard or {}
        triage = run.triage or {}
        label = classification.get("predicted_grade_label")
        if label:
            severity_distribution[label] += 1
        if classification.get("referable_dr") is True:
            referable += 1
        if trust.get("trust_category") in {"UNCERTAIN", "REVIEW_RECOMMENDED", "INSUFFICIENT_EVIDENCE", "UNRELIABLE"} or triage.get("recommendation") in {"HUMAN_REVIEW_REQUIRED", "SPECIALIST_REVIEW_RECOMMENDED", "RECAPTURE_OR_SPECIALIST_REVIEW"}:
            human_review += 1
        if len(recent_activity) < 8:
            recent_activity.append({"screening_id": str(run.id), "status": run.status, "grade": label, "trust_category": trust.get("trust_category"), "referable_dr": classification.get("referable_dr"), "created_at": run.created_at})
    ungradable = sum(1 for image in images if getattr(image.quality_decision, "value", image.quality_decision) == "ungradable")
    today_screenings = sum(1 for session in sessions if session.created_at and session.created_at.date() == today)
    return AnalyticsOverviewResponse(
        total_screenings=len(sessions), today_screenings=today_screenings, referable_cases=referable,
        human_review_cases=human_review, ungradable_images=ungradable,
        completed_screenings=sum(1 for item in sessions if item.status.value == "complete"),
        status_distribution=dict(status_distribution), severity_distribution=dict(severity_distribution),
        recent_activity=recent_activity,
        system_health={"api": "operational", "database": "connected", "model_pipeline": "configured" if any((item.classification or {}).get("model_version") for item in runs) else "not_configured", "audit_logging": "ready"},
    )
