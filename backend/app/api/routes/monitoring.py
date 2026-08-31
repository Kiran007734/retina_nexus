"""Operational metrics and drift-monitoring preparation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.clinical_review import ClinicalReview
from app.models.screening_run import ScreeningRun

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary")
async def monitoring_summary(days: int = Query(default=30, ge=1, le=365), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    runs = (await db.execute(select(ScreeningRun).where(ScreeningRun.created_at >= cutoff).order_by(ScreeningRun.created_at.desc()))).scalars().all()
    reviewed_ids = set((await db.execute(select(ClinicalReview.screening_session_id).where(ClinicalReview.created_at >= cutoff))).scalars().all())
    completed = [run for run in runs if run.status == "COMPLETED"]
    quality_values = [_final_quality(run) for run in runs if _final_quality(run) is not None]
    classifications = [run.classification or {} for run in runs if run.classification]
    latency = [_duration(run, "dr_classification") for run in completed if _duration(run, "dr_classification") is not None]
    total_latency = [_total_duration(run) for run in completed if _total_duration(run) is not None]
    disagreement = [run for run in completed if float(((run.retinaguard or {}).get("model_disagreement") or {}).get("disagreement") or 0) > 0.2]
    ungradable = sum(1 for quality in quality_values if quality.get("quality_decision") == "UNGRADABLE")
    quality_distribution = Counter(str(quality.get("quality_decision", "UNKNOWN")) for quality in quality_values)
    prediction_distribution = Counter(str(item.get("predicted_grade_label", "UNKNOWN")) for item in classifications)
    model_distribution = Counter(str(item.get("model_version", "unversioned")) for item in classifications)
    bottlenecks = _stage_bottlenecks(completed)
    return {
        "status": "ready", "window_days": days, "sample_count": len(runs), "completed_count": len(completed),
        "model_pipeline": "configured" if classifications else "not_configured",
        "inference": {"mean_ms": _round_mean(latency), "median_ms": _round_median(latency), "p95_ms": _percentile(latency, 95), "sample_count": len(latency)},
        "pipeline": {"mean_ms": _round_mean(total_latency), "median_ms": _round_median(total_latency), "sample_count": len(total_latency), "bottlenecks": bottlenecks},
        "model_versions": dict(model_distribution), "prediction_distribution": dict(prediction_distribution),
        "quality_distribution": dict(quality_distribution),
        "rates": {"ungradable": _rate(ungradable, len(quality_values)), "review": _rate(len(reviewed_ids), len(completed)), "disagreement": _rate(len(disagreement), len(classifications)), "failure": _rate(sum(1 for run in runs if run.status == "FAILED"), len(runs))},
        "drift": {"input_distribution": _input_drift(completed), "prediction": _categorical_drift([item.get("predicted_grade_label") for item in classifications]), "quality": _quality_drift(quality_values)},
        "review_queue": {"open_signal_count": sum(1 for run in completed if _needs_review(run)), "reviewed_session_count": len(reviewed_ids)},
        "system_health": {"api": "operational", "database": "connected", "audit_logging": "ready", "worker_mode": "inline_queue_ready", "auto_retraining": "disabled"},
        "message": "Monitoring metrics are operational engineering signals. Drift flags require validation; no automatic retraining is performed.",
    }


def _final_quality(run: ScreeningRun) -> dict | None:
    value = (run.quality or {}).get("final") if isinstance(run.quality, dict) else None
    return value if isinstance(value, dict) else None


def _duration(run: ScreeningRun, stage: str) -> float | None:
    value = (run.stage_metrics or {}).get(stage, {}).get("duration_ms") if isinstance(run.stage_metrics, dict) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _total_duration(run: ScreeningRun) -> float | None:
    if run.started_at is None or run.completed_at is None:
        return None
    return max(0.0, (run.completed_at - run.started_at).total_seconds() * 1000)


def _stage_bottlenecks(runs: list[ScreeningRun]) -> list[dict[str, Any]]:
    values: dict[str, list[float]] = {}
    for run in runs:
        for stage, metric in (run.stage_metrics or {}).items():
            try:
                if metric.get("duration_ms") is not None:
                    values.setdefault(stage, []).append(float(metric["duration_ms"]))
            except (AttributeError, TypeError, ValueError):
                continue
    return [{"stage": stage, "mean_ms": _round_mean(samples), "sample_count": len(samples)} for stage, samples in sorted(values.items(), key=lambda item: mean(item[1]), reverse=True)[:8]]


def _needs_review(run: ScreeningRun) -> bool:
    trust = run.retinaguard or {}
    triage = run.triage or {}
    return trust.get("trust_category") in {"UNCERTAIN", "REVIEW_RECOMMENDED", "INSUFFICIENT_EVIDENCE", "UNRELIABLE"} or (run.classification or {}).get("referable_dr") is True or triage.get("recommendation") in {"HUMAN_REVIEW_REQUIRED", "SPECIALIST_REVIEW_RECOMMENDED", "RECAPTURE_OR_SPECIALIST_REVIEW"}


def _input_drift(runs: list[ScreeningRun]) -> dict[str, Any]:
    shifted = sum(1 for run in runs if (run.retinaguard or {}).get("ood", {}).get("status") == "SHIFTED")
    compared = sum(1 for run in runs if (run.retinaguard or {}).get("ood", {}).get("status") in {"SHIFTED", "IN_DISTRIBUTION"})
    if compared < 10:
        return {"status": "INSUFFICIENT_DATA", "flagged": False, "shifted_rate": _rate(shifted, compared), "sample_count": compared, "method": "configured_ood_signal_rate", "action": "Collect more reference-comparable runs before validation."}
    rate = shifted / compared
    return {"status": "FLAGGED" if rate >= 0.2 else "STABLE", "flagged": rate >= 0.2, "shifted_rate": round(rate, 4), "sample_count": compared, "method": "configured_ood_signal_rate", "action": "Validate image mix and reference distribution before any model change."}


def _categorical_drift(values: list[Any]) -> dict[str, Any]:
    usable = [str(value) for value in values if value]
    if len(usable) < 20:
        return {"status": "INSUFFICIENT_DATA", "flagged": False, "sample_count": len(usable), "method": "split_window_total_variation", "action": "Collect more predictions before validation."}
    midpoint = len(usable) // 2
    recent, baseline = usable[:midpoint], usable[midpoint:]
    categories = set(recent) | set(baseline)
    distance = sum(abs(recent.count(category) / len(recent) - baseline.count(category) / len(baseline)) for category in categories) / 2
    return {"status": "FLAGGED" if distance >= 0.2 else "STABLE", "flagged": distance >= 0.2, "sample_count": len(usable), "distance": round(distance, 4), "method": "split_window_total_variation", "action": "Validate case mix and labels before considering recalibration."}


def _quality_drift(values: list[dict]) -> dict[str, Any]:
    scores = [float(value["quality_score"]) for value in values if value.get("quality_score") is not None]
    if len(scores) < 20:
        return {"status": "INSUFFICIENT_DATA", "flagged": False, "sample_count": len(scores), "method": "split_window_quality_mean_and_ungradable_rate", "action": "Collect more quality assessments before validation."}
    midpoint = len(scores) // 2
    recent, baseline = scores[:midpoint], scores[midpoint:]
    difference = abs(mean(recent) - mean(baseline))
    return {"status": "FLAGGED" if difference >= 0.1 else "STABLE", "flagged": difference >= 0.1, "sample_count": len(scores), "mean_difference": round(difference, 4), "method": "split_window_quality_mean_and_ungradable_rate", "action": "Validate acquisition conditions and camera mix before any model change."}


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _round_mean(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def _round_median(values: list[float]) -> float | None:
    return round(median(values), 3) if values else None


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)
