from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.api.deps import get_optional_claims
from app.models.audit_log import AuditLog
from app.models.fundus_image import FundusImage, QualityDecision
from app.models.screening import ScreeningResult, ScreeningSession, ScreeningStatus
from app.models.screening_run import ScreeningRun
from app.models.anatomical_landmark import AnatomicalLandmark
from app.models.lesion_result import LesionResult
from app.models.segmentation_result import SegmentationResult
from app.models.retinaguard_result import RetinaGuardResult
from app.models.explainability_result import ExplainabilityResult
from app.ml.evidence.service import RetinalEvidenceService
from app.ml.inference.classifier import ClassifierNotConfiguredError, TorchDRClassificationService
from app.ml.explainability.service import ExplainabilityService
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs, derive_lesion_evidence_strength, derive_vessel_evidence_status
from app.services.screening_pipeline import OPTIONAL_EVIDENCE_STAGES, RUN_STAGES, ScreeningPipelineService
from app.services.container import get_classifier_service
from app.storage.container import get_storage
from app.repositories.patients import get_patient
from app.schemas.screening import ClassifyRequest, ClassificationResponse, EvidenceAnalysisRequest, ExplainabilityRequest, ScreeningCreate, ScreeningHistoryItem, ScreeningResponse, ScreeningResultResponse, ScreeningRunRequest, ScreeningRunResponse
from app.schemas.evidence import EvidenceAnalysisResponse
from app.schemas.explainability import ExplainabilityResponse
from app.schemas.trust import TrustRequest, TrustResponse
from app.services.container import get_evidence_service, get_explainability_service, get_retinaguard_service, get_screening_pipeline_service
from app.services.screening_persistence import persist_evidence_analysis, persist_explainability, persist_retinaguard

router = APIRouter(prefix="/screening", tags=["screening"])


@router.post("", response_model=ScreeningResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_screening(payload: ScreeningCreate, db: AsyncSession = Depends(get_db)) -> ScreeningResponse:
    if await get_patient(db, payload.patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    session = ScreeningSession(
        id=uuid4(), patient_id=payload.patient_id, fundus_image_id=payload.fundus_image_id,
        status=ScreeningStatus.QUEUED, started_at=datetime.now(timezone.utc),
    )
    db.add(session)
    await db.commit()
    return ScreeningResponse(session_id=session.id, patient_id=session.patient_id, status=session.status.value, message="Screening queued; run the configured classifier after the Image Trust Gate")


@router.get("/history", response_model=list[ScreeningHistoryItem])
async def screening_history(db: AsyncSession = Depends(get_db)) -> list[ScreeningHistoryItem]:
    runs = (await db.execute(select(ScreeningRun).order_by(ScreeningRun.created_at.desc()).limit(100))).scalars().all()
    items: list[ScreeningHistoryItem] = []
    for run in runs:
        session = await db.get(ScreeningSession, run.id)
        image = await db.get(FundusImage, run.fundus_image_id)
        if session is None or image is None:
            continue
        classification = run.classification or {}
        trust = run.retinaguard or {}
        items.append(ScreeningHistoryItem(
            screening_id=run.id, patient_id=session.patient_id, image_id=image.id, eye=image.eye.value,
            status=run.status, trust_category=trust.get("trust_category"), trust_score=trust.get("trust_score"),
            predicted_grade=classification.get("predicted_grade"), predicted_grade_label=classification.get("predicted_grade_label"),
            referable_dr=classification.get("referable_dr"), triage_recommendation=(run.triage or {}).get("recommendation"), created_at=run.created_at,
        ))
    return items


@router.post("/run", response_model=ScreeningRunResponse)
async def run_screening(
    payload: ScreeningRunRequest,
    db: AsyncSession = Depends(get_db),
    claims: dict | None = Depends(get_optional_claims),
    pipeline: ScreeningPipelineService = Depends(get_screening_pipeline_service),
) -> ScreeningRunResponse:
    image = await db.get(FundusImage, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if payload.screening_session_id:
        session = await db.get(ScreeningSession, payload.screening_session_id)
        if session is None or session.fundus_image_id != image.id:
            raise HTTPException(status_code=404, detail="Screening session not found for this image")
        existing = await db.get(ScreeningRun, session.id)
        if existing is not None:
            if existing.status in {"QUEUED", "PROCESSING"}:
                raise HTTPException(status_code=409, detail="A screening run is already in progress for this session")
            raise HTTPException(status_code=409, detail="This screening session already has a terminal master run; create a new session to rerun it")
    else:
        session = ScreeningSession(id=uuid4(), patient_id=image.patient_id, fundus_image_id=image.id, status=ScreeningStatus.QUEUED, started_at=datetime.now(timezone.utc))
        db.add(session)
        await db.flush()
    actor_id = _actor_id(claims)
    run = ScreeningRun(
        id=session.id, fundus_image_id=image.id, initiating_user_id=actor_id, status="QUEUED",
        stage_status={stage: "PENDING" for stage in RUN_STAGES}, stage_metrics={},
    )
    db.add(run)
    db.add(AuditLog(actor_id=actor_id, action="screening.run.queued", resource_type="screening_run", resource_id=str(run.id), details={"timestamp": datetime.now(timezone.utc).isoformat(), "image_id": str(image.id), "initiating_user_id": str(actor_id) if actor_id else None}))
    await db.commit()
    output = await pipeline.execute_primary(
        db, run, session, image, actor_id,
        run_stability=payload.run_stability,
        run_counterfactual=payload.run_counterfactual,
        model_predictions=[item.model_dump() for item in payload.model_predictions],
    )
    if output.retinaguard_result is not None and output.prediction is not None:
        await persist_retinaguard(output.retinaguard_result.to_dict(), image, session, db, output.prediction)
    if output.classification is not None:
        pipeline.start_optional_processing(
            run.id, image.id, actor_id,
            run_stability=payload.run_stability,
            run_counterfactual=payload.run_counterfactual,
        )
    return _run_response(run, session)


@router.get("/{session_id}", response_model=ScreeningRunResponse)
async def screening_status(session_id: UUID, db: AsyncSession = Depends(get_db)) -> ScreeningRunResponse:
    run = await db.get(ScreeningRun, session_id)
    if run is not None:
        session = await db.get(ScreeningSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Screening session not found")
        return _run_response(run, session)
    session = await db.get(ScreeningSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Screening session not found")
    return ScreeningRunResponse(screening_id=session.id, screening_session_id=session.id, patient_id=session.patient_id, image_id=session.fundus_image_id, status=_public_session_status(session.status), primary_status="PENDING", evidence_status="NOT_RUN", stage_status={}, stage_metrics={}, stage_errors={}, model_versions={}, message="No master screening run has been started for this session")


@router.get("/{session_id}/result", response_model=ScreeningResultResponse)
async def screening_result(session_id: UUID, db: AsyncSession = Depends(get_db)) -> ScreeningResultResponse:
    session = await db.get(ScreeningSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Screening session not found")
    result = (await db.execute(select(ScreeningResult).where(ScreeningResult.session_id == session.id))).scalar_one_or_none()
    return ScreeningResultResponse(
        session_id=session.id, status=session.status.value,
        dr_grade=result.dr_grade if result else None,
        referable_dr=result.referable_dr if result else None,
        confidence=result.confidence if result else None,
        calibrated_confidence=result.calibrated_confidence if result else None,
        uncertainty=result.uncertainty if result else None,
        trust_score=result.trust_score if result else None,
        final_decision=result.final_decision.value if result and result.final_decision else None,
        model_version=result.model_version if result else None,
        created_at=result.created_at if result else None,
    )


@router.post("/classify", response_model=ClassificationResponse)
async def classify_screening(
    payload: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
    classifier: TorchDRClassificationService = Depends(get_classifier_service),
) -> ClassificationResponse:
    image = await db.get(FundusImage, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.quality_decision not in {QualityDecision.GRADABLE, QualityDecision.BORDERLINE, QualityDecision.ACCEPT}:
        if image.quality_decision == QualityDecision.UNGRADABLE:
            raise HTTPException(status_code=422, detail="Image Trust Gate marked this image UNGRADABLE; recapture before classification")
        raise HTTPException(status_code=409, detail="Run the Image Trust Gate before classification")
    if payload.screening_session_id:
        session = await db.get(ScreeningSession, payload.screening_session_id)
        if session is None or session.fundus_image_id != image.id:
            raise HTTPException(status_code=404, detail="Screening session not found for this image")
    else:
        session = (await db.execute(select(ScreeningSession).where(ScreeningSession.fundus_image_id == image.id).order_by(ScreeningSession.created_at.desc()).limit(1))).scalar_one_or_none()
        if session is None:
            session = ScreeningSession(id=uuid4(), patient_id=image.patient_id, fundus_image_id=image.id, status=ScreeningStatus.PROCESSING, started_at=datetime.now(timezone.utc))
            db.add(session)
            await db.flush()
    try:
        content = await get_storage().get(image.storage_path)
        prediction = await classifier.classify(content)
    except ClassifierNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if payload.model_version and payload.model_version != prediction.model_version:
        raise HTTPException(status_code=409, detail=f"Requested model version '{payload.model_version}' is not the configured inference artifact")
    result = (await db.execute(select(ScreeningResult).where(ScreeningResult.session_id == session.id))).scalar_one_or_none()
    if result is None:
        result = ScreeningResult(id=uuid4(), session_id=session.id)
        db.add(result)
    result.dr_grade = prediction.predicted_grade
    result.referable_dr = prediction.referable_dr
    result.confidence = prediction.raw_confidence
    result.model_version = prediction.model_version
    session.status = ScreeningStatus.COMPLETE
    session.completed_at = datetime.now(timezone.utc)
    await db.commit()
    return ClassificationResponse(
        image_id=image.id, screening_session_id=session.id, predicted_grade=prediction.predicted_grade,
        predicted_grade_label=prediction.predicted_grade_label, probabilities=prediction.probabilities,
        referable_dr=prediction.referable_dr, referable_probability=prediction.referable_probability,
        raw_confidence=prediction.raw_confidence, model_name=prediction.model_name,
        model_version=prediction.model_version, backbone=prediction.backbone,
        referable_mapping=prediction.referable_mapping, hierarchical_probabilities=prediction.hierarchical_probabilities,
        ordinal_mode=prediction.ordinal_mode,
        note="Raw model confidence only; this value is not a clinical trust guarantee and no final trust score is calculated at this stage.",
    )


@router.post("/analyze-structures", response_model=EvidenceAnalysisResponse)
async def analyze_structures(
    payload: EvidenceAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    evidence_service: RetinalEvidenceService = Depends(get_evidence_service),
) -> EvidenceAnalysisResponse:
    image = await db.get(FundusImage, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.quality_decision not in {QualityDecision.GRADABLE, QualityDecision.BORDERLINE, QualityDecision.ACCEPT}:
        if image.quality_decision == QualityDecision.UNGRADABLE:
            raise HTTPException(status_code=422, detail="Image Trust Gate marked this image UNGRADABLE; recapture before evidence analysis")
        raise HTTPException(status_code=409, detail="Run the Image Trust Gate before evidence analysis")
    session = await _session_for_image(image, payload.screening_session_id, db)
    try:
        content = await get_storage().get(image.storage_path)
        analysis = await evidence_service.analyze(content, str(image.id), str(session.id), image.eye.value)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Stored image is not available") from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"Evidence analysis could not read the stored image: {exc}") from exc
    await persist_evidence_analysis(analysis.to_dict(), image, session, db)
    return EvidenceAnalysisResponse.model_validate(analysis.to_dict())


@router.post("/explain", response_model=ExplainabilityResponse)
async def explain_screening(
    payload: ExplainabilityRequest,
    db: AsyncSession = Depends(get_db),
    evidence_service: RetinalEvidenceService = Depends(get_evidence_service),
    explainability_service: ExplainabilityService = Depends(get_explainability_service),
) -> ExplainabilityResponse:
    image = await db.get(FundusImage, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.quality_decision not in {QualityDecision.GRADABLE, QualityDecision.BORDERLINE, QualityDecision.ACCEPT}:
        if image.quality_decision == QualityDecision.UNGRADABLE:
            raise HTTPException(status_code=422, detail="Image Trust Gate marked this image UNGRADABLE; recapture before explainability analysis")
        raise HTTPException(status_code=409, detail="Run the Image Trust Gate before explainability analysis")
    session = await _session_for_image(image, payload.screening_session_id, db)
    try:
        content = await get_storage().get(image.storage_path)
        evidence = await evidence_service.analyze(content, str(image.id), str(session.id), image.eye.value)
        explanation = await explainability_service.analyze(
            content, str(image.id), str(session.id), evidence,
            run_stability=payload.run_stability,
            run_counterfactual=payload.run_counterfactual,
        )
    except ClassifierNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Stored image is not available") from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"Explainability analysis could not read the stored image: {exc}") from exc
    await persist_evidence_analysis(evidence.to_dict(), image, session, db)
    await persist_explainability(explanation.to_dict(), image, session, db)
    return ExplainabilityResponse.model_validate(explanation.to_dict())


@router.post("/trust", response_model=TrustResponse)
async def trust_screening(
    payload: TrustRequest,
    db: AsyncSession = Depends(get_db),
    classifier: TorchDRClassificationService = Depends(get_classifier_service),
    evidence_service: RetinalEvidenceService = Depends(get_evidence_service),
    retinaguard: RetinaGuardEngine = Depends(get_retinaguard_service),
) -> TrustResponse:
    image = await db.get(FundusImage, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.quality_decision not in {QualityDecision.GRADABLE, QualityDecision.BORDERLINE, QualityDecision.ACCEPT}:
        if image.quality_decision == QualityDecision.UNGRADABLE:
            raise HTTPException(status_code=422, detail="Image Trust Gate marked this image UNGRADABLE; recapture before RetinaGuard")
        raise HTTPException(status_code=409, detail="Run the Image Trust Gate before RetinaGuard")
    session = await _session_for_image(image, payload.screening_session_id, db)
    try:
        content = await get_storage().get(image.storage_path)
        prediction = await classifier.classify(content)
        evidence = await evidence_service.analyze(content, str(image.id), str(session.id), image.eye.value)
    except ClassifierNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Stored image is not available") from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"RetinaGuard could not read the stored image: {exc}") from exc

    explanation_record = (await db.execute(select(ExplainabilityResult).where(ExplainabilityResult.screening_session_id == session.id))).scalar_one_or_none()
    agreement = None
    stability = None
    if explanation_record is not None:
        agreement = {"status": explanation_record.attention_agreement_status, "score": explanation_record.attention_agreement_score, "metrics": explanation_record.attention_agreement_metrics or {}}
        stability = explanation_record.explanation_stability
    quality_assessment = image.quality_assessment or {}
    final_quality = quality_assessment.get("final") if isinstance(quality_assessment, dict) else {}
    feature_vector = final_quality.get("feature_vector", {}) if isinstance(final_quality, dict) else {}
    quality_score = image.quality_score
    if quality_score is None and isinstance(final_quality, dict):
        quality_score = final_quality.get("quality_score")
    additional_predictions = [item.model_dump() for item in payload.model_predictions]
    inputs = RetinaGuardInputs(
        quality_score=quality_score,
        raw_confidence=prediction.raw_confidence,
        probabilities=prediction.probabilities,
        classifier_logits=prediction.severity_logits,
        model_predictions=additional_predictions,
        lesion_evidence_strength=derive_lesion_evidence_strength(evidence),
        vessel_evidence_status=derive_vessel_evidence_status(evidence),
        attention_lesion_agreement=agreement,
        explanation_stability=stability,
        quality_feature_vector={key: float(value) for key, value in feature_vector.items() if isinstance(value, (int, float))},
        predicted_grade=prediction.predicted_grade,
        predicted_grade_label=prediction.predicted_grade_label,
        referable_dr=prediction.referable_dr,
        model_version=prediction.model_version,
    )
    result = await retinaguard.evaluate_async(inputs, content, classifier)
    await persist_evidence_analysis(evidence.to_dict(), image, session, db)
    await persist_retinaguard(result.to_dict(), image, session, db, prediction)
    response = {"image_id": image.id, "screening_session_id": session.id, **result.to_dict(), "note": "RetinaGuard is a transparent engineering self-check. Its score and category are not a medical diagnosis or a clinical trust guarantee."}
    return TrustResponse.model_validate(response)


async def _session_for_image(image: FundusImage, requested_id: UUID | None, db: AsyncSession) -> ScreeningSession:
    if requested_id:
        session = await db.get(ScreeningSession, requested_id)
        if session is None or session.fundus_image_id != image.id:
            raise HTTPException(status_code=404, detail="Screening session not found for this image")
        return session
    session = (await db.execute(select(ScreeningSession).where(ScreeningSession.fundus_image_id == image.id).order_by(ScreeningSession.created_at.desc()).limit(1))).scalar_one_or_none()
    if session is not None:
        return session
    session = ScreeningSession(id=uuid4(), patient_id=image.patient_id, fundus_image_id=image.id, status=ScreeningStatus.PROCESSING, started_at=datetime.now(timezone.utc))
    db.add(session)
    await db.flush()
    return session


async def _persist_evidence_analysis(payload: dict, image: FundusImage, session: ScreeningSession, db: AsyncSession) -> None:
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
    session.status = ScreeningStatus.COMPLETE
    session.completed_at = session.completed_at or datetime.now(timezone.utc)
    await db.commit()


async def _persist_explainability(payload: dict, image: FundusImage, session: ScreeningSession, db: AsyncSession) -> None:
    agreement = payload["attention_lesion_agreement"]
    record = (await db.execute(select(ExplainabilityResult).where(ExplainabilityResult.screening_session_id == session.id))).scalar_one_or_none()
    if record is None:
        record = ExplainabilityResult(id=uuid4(), screening_session_id=session.id, fundus_image_id=image.id, predicted_class=payload["predicted_class"], predicted_class_label=payload["predicted_class_label"], model_version=payload["model_version"], attention_agreement_status=agreement["status"])
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
    session.status = ScreeningStatus.COMPLETE
    session.completed_at = session.completed_at or datetime.now(timezone.utc)
    await db.commit()


async def _persist_retinaguard(payload: dict, image: FundusImage, session: ScreeningSession, db: AsyncSession, prediction: Any) -> None:
    record = (await db.execute(select(RetinaGuardResult).where(RetinaGuardResult.screening_session_id == session.id))).scalar_one_or_none()
    if record is None:
        record = RetinaGuardResult(id=uuid4(), screening_session_id=session.id, fundus_image_id=image.id, trust_score=payload["trust_score"], trust_category=payload["trust_category"], contributing_factors=payload["contributing_factors"], risk_flags=payload["risk_flags"], recommended_action=payload["recommended_action"])
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
    session.status = ScreeningStatus.COMPLETE if payload["trust_category"] == "TRUSTED" else ScreeningStatus.NEEDS_REVIEW
    session.completed_at = session.completed_at or datetime.now(timezone.utc)
    await db.commit()


def _actor_id(claims: dict | None) -> UUID | None:
    if not claims or not claims.get("sub"):
        return None
    try:
        return UUID(str(claims["sub"]))
    except (TypeError, ValueError):
        return None


def _public_session_status(value: ScreeningStatus) -> str:
    if value in {ScreeningStatus.COMPLETE, ScreeningStatus.NEEDS_REVIEW}:
        return "COMPLETED"
    return value.value.upper()


def _run_response(run: ScreeningRun, session: ScreeningSession) -> ScreeningRunResponse:
    message = "Screening run completed"
    primary_status = _primary_status(run)
    evidence_status, evidence_message = _evidence_status(run)
    if run.status == "FAILED":
        message = (run.error or {}).get("message", "Screening run failed")
    elif run.status == "COMPLETED" and run.triage and run.triage.get("recommendation") == "RECAPTURE_IMAGE":
        message = "Screening stopped after the Image Trust Gate; recapture is recommended"
    elif primary_status == "COMPLETED" and evidence_status == "PROCESSING":
        message = "Primary screening completed; optional evidence is processing"
    elif primary_status == "COMPLETED" and evidence_status == "TIMED_OUT":
        message = "Primary screening completed; optional evidence exceeded its runtime budget"
    elif primary_status == "COMPLETED" and evidence_status == "UNAVAILABLE":
        message = "Primary screening completed; optional evidence is unavailable"
    elif run.status in {"QUEUED", "PROCESSING"}:
        message = "Screening run is in progress"
    return ScreeningRunResponse(
        screening_id=run.id, screening_session_id=session.id, patient_id=session.patient_id, image_id=run.fundus_image_id,
        status=run.status, primary_status=primary_status, evidence_status=evidence_status, evidence_message=evidence_message,
        stage_status=run.stage_status or {}, stage_metrics=run.stage_metrics or {}, stage_errors=run.stage_errors or {},
        quality=run.quality, classification=run.classification, lesions=run.lesions,
        explainability=run.explainability, retinaguard=run.retinaguard, triage=run.triage,
        model_versions=run.model_versions or {}, error=run.error, message=message,
    )


def _primary_status(run: ScreeningRun) -> str:
    if run.status == "FAILED":
        return "FAILED"
    if run.triage and run.classification:
        return "COMPLETED"
    if run.status == "COMPLETED" and run.triage and not run.classification:
        return "QUALITY_BLOCKED"
    if run.status in {"QUEUED", "PROCESSING"}:
        return "PROCESSING"
    return "PENDING"


def _evidence_status(run: ScreeningRun) -> tuple[str, str]:
    if not run.classification:
        return "NOT_RUN", "Optional evidence is not run when the Image Trust Gate blocks clinical AI."
    statuses = [((run.stage_status or {}).get(stage) or "PENDING") for stage in OPTIONAL_EVIDENCE_STAGES]
    if any(value in {"QUEUED", "PROCESSING", "PENDING"} for value in statuses):
        return "PROCESSING", "Optional lesion, vessel, and explainability stages are still processing."
    if any(value == "TIMED_OUT" for value in statuses):
        return "TIMED_OUT", "Optional evidence did not complete within its runtime budget; it is not negative evidence."
    if any(value in {"UNAVAILABLE", "FAILED"} for value in statuses):
        return "UNAVAILABLE", "Optional evidence was unavailable; no evidence result was substituted."
    if all(value in {"COMPLETED", "SKIPPED"} for value in statuses):
        return "AVAILABLE", "Optional evidence is available for review."
    return "NOT_RUN", "Optional evidence has not started."
