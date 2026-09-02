"""Durable persistence helpers shared by inline and background screening work."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anatomical_landmark import AnatomicalLandmark
from app.models.explainability_result import ExplainabilityResult
from app.models.fundus_image import FundusImage
from app.models.lesion_result import LesionResult
from app.models.retinaguard_result import RetinaGuardResult
from app.models.screening import ScreeningResult, ScreeningSession, ScreeningStatus
from app.models.segmentation_result import SegmentationResult


async def persist_evidence_analysis(
    payload: dict[str, Any],
    image: FundusImage,
    session: ScreeningSession,
    db: AsyncSession,
    *,
    update_session_status: bool = True,
) -> None:
    """Persist evidence without turning unavailable evidence into a finding."""
    await db.execute(delete(SegmentationResult).where(SegmentationResult.screening_session_id == session.id))
    await db.execute(delete(LesionResult).where(LesionResult.screening_session_id == session.id))
    await db.execute(delete(AnatomicalLandmark).where(AnatomicalLandmark.screening_session_id == session.id))
    for module_name, module in payload["modules"].items():
        metadata = dict(module.get("metadata") or {})
        if module["category"] == "segmentation":
            db.add(SegmentationResult(
                id=uuid4(), screening_session_id=session.id, fundus_image_id=image.id,
                structure_type=module_name, status=module["status"], implementation=module["implementation"],
                confidence=module.get("confidence"), pixel_count=metadata.get("pixel_count"),
                mask_data_uri=module.get("mask_data_uri"), bounding_regions=module.get("bounding_regions"),
                result_metadata=metadata,
            ))
        elif module["category"] == "lesion_detection":
            db.add(LesionResult(
                id=uuid4(), screening_session_id=session.id, fundus_image_id=image.id,
                lesion_type=module_name, status=module["status"], implementation=module["implementation"],
                lesion_count=module.get("count"), confidence=module.get("confidence"),
                mask_data_uri=module.get("mask_data_uri"), bounding_regions=module.get("bounding_regions"),
                result_metadata=metadata,
            ))
    landmark_fields = {"landmark_type", "status", "method", "x", "y", "radius", "x_normalized", "y_normalized", "confidence"}
    for landmark in payload.get("anatomical_landmarks", []):
        if "x" not in landmark or "y" not in landmark:
            continue
        db.add(AnatomicalLandmark(
            id=uuid4(), screening_session_id=session.id, fundus_image_id=image.id,
            landmark_type=landmark["landmark_type"], status=landmark["status"], method=landmark["method"],
            x=landmark["x"], y=landmark["y"], radius=landmark.get("radius"),
            x_normalized=landmark.get("x_normalized"), y_normalized=landmark.get("y_normalized"),
            confidence=landmark.get("confidence"),
            result_metadata={key: value for key, value in landmark.items() if key not in landmark_fields},
        ))
    result = (await db.execute(select(ScreeningResult).where(ScreeningResult.session_id == session.id))).scalar_one_or_none()
    if result is None:
        result = ScreeningResult(id=uuid4(), session_id=session.id)
        db.add(result)
    result.lesion_evidence = {
        "status": payload["status"], "coarse_to_fine": payload["coarse_to_fine"],
        "evidence_map_data_uri": payload.get("evidence_map_data_uri"),
        "dataset_support": payload.get("dataset_support"), "note": payload.get("note"),
    }
    if update_session_status:
        session.status = ScreeningStatus.COMPLETE
        session.completed_at = session.completed_at or datetime.now(timezone.utc)
    await db.commit()


async def persist_explainability(
    payload: dict[str, Any],
    image: FundusImage,
    session: ScreeningSession,
    db: AsyncSession,
    *,
    update_session_status: bool = True,
) -> None:
    agreement = payload["attention_lesion_agreement"]
    record = (await db.execute(select(ExplainabilityResult).where(ExplainabilityResult.screening_session_id == session.id))).scalar_one_or_none()
    if record is None:
        record = ExplainabilityResult(
            id=uuid4(), screening_session_id=session.id, fundus_image_id=image.id,
            predicted_class=payload["predicted_class"], predicted_class_label=payload["predicted_class_label"],
            model_version=payload["model_version"], attention_agreement_status=agreement["status"],
        )
        db.add(record)
    record.fundus_image_id = image.id
    record.predicted_class = payload["predicted_class"]
    record.predicted_class_label = payload["predicted_class_label"]
    record.model_version = payload["model_version"]
    record.heatmap_data_uri = payload["grad_cam"].get("heatmap_data_uri")
    record.overlay_data_uri = payload["grad_cam"].get("overlay_data_uri")
    record.normalized_attention_map_data_uri = payload["grad_cam"].get("normalized_attention_map_data_uri")
    record.lesion_evidence_map_data_uri = payload.get("lesion_evidence_map_data_uri")
    record.attention_agreement_status = agreement["status"]
    record.attention_agreement_score = agreement.get("score")
    record.attention_agreement_metrics = agreement.get("metrics")
    record.explanation_stability = payload.get("explanation_stability")
    record.counterfactual = payload.get("counterfactual")
    record.result_metadata = {
        "classification": payload.get("classification"),
        "grad_cam": {key: value for key, value in payload.get("grad_cam", {}).items() if key not in {"heatmap_data_uri", "overlay_data_uri", "normalized_attention_map_data_uri"}},
        "note": payload.get("note"),
    }
    if update_session_status:
        session.status = ScreeningStatus.COMPLETE
        session.completed_at = session.completed_at or datetime.now(timezone.utc)
    await db.commit()


async def persist_retinaguard(
    payload: dict[str, Any],
    image: FundusImage,
    session: ScreeningSession,
    db: AsyncSession,
    prediction: Any,
    *,
    update_session_status: bool = True,
) -> None:
    record = (await db.execute(select(RetinaGuardResult).where(RetinaGuardResult.screening_session_id == session.id))).scalar_one_or_none()
    if record is None:
        record = RetinaGuardResult(
            id=uuid4(), screening_session_id=session.id, fundus_image_id=image.id,
            trust_score=payload["trust_score"], trust_category=payload["trust_category"],
            contributing_factors=payload["contributing_factors"], risk_flags=payload["risk_flags"],
            recommended_action=payload["recommended_action"],
        )
        db.add(record)
    record.fundus_image_id = image.id
    record.trust_score = payload["trust_score"]
    record.trust_category = payload["trust_category"]
    record.contributing_factors = payload["contributing_factors"]
    record.risk_flags = payload["risk_flags"]
    record.recommended_action = payload["recommended_action"]
    record.calibration = payload.get("calibration")
    record.uncertainty = payload.get("uncertainty")
    record.model_disagreement = payload.get("model_disagreement")
    record.ood = payload.get("ood")
    record.signal_snapshot = payload.get("signal_snapshot")
    record.configuration = {
        **(payload.get("configuration") or {}),
        "available_signals": payload.get("available_signals"),
        "decision_trace": payload.get("decision_trace"),
    }
    record.reason_summary = payload.get("reason_summary")
    screening_result = (await db.execute(select(ScreeningResult).where(ScreeningResult.session_id == session.id))).scalar_one_or_none()
    if screening_result is None:
        screening_result = ScreeningResult(id=uuid4(), session_id=session.id)
        db.add(screening_result)
    screening_result.dr_grade = prediction.predicted_grade
    screening_result.referable_dr = prediction.referable_dr
    screening_result.confidence = prediction.raw_confidence
    screening_result.calibrated_confidence = payload.get("calibration", {}).get("calibrated_confidence")
    screening_result.uncertainty = payload.get("uncertainty", {}).get("score")
    screening_result.trust_score = payload["trust_score"]
    screening_result.model_version = prediction.model_version
    if update_session_status:
        session.status = ScreeningStatus.COMPLETE if payload["trust_category"] == "TRUSTED" else ScreeningStatus.NEEDS_REVIEW
        session.completed_at = session.completed_at or datetime.now(timezone.utc)
    await db.commit()
