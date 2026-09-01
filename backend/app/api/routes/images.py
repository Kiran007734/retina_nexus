from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.database.session import get_db
from app.ml.quality.trust_gate import ImageTrustGateError, ImageTrustGateService, TrustGateDecision, TrustGateOutcome
from app.models.fundus_image import Eye, FundusImage, QualityDecision
from app.repositories.patients import get_patient
from app.schemas.images import EyeInput, ImageUploadResponse, QualityAssessmentResponse, QualityIssueResponse
from app.services.container import get_image_quality_service
from app.storage.container import get_storage

router = APIRouter(prefix="/images", tags=["images"])
settings = get_settings()


@router.post("/upload", response_model=ImageUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    patient_id: UUID,
    eye: EyeInput = Query(...),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    quality_service: ImageTrustGateService = Depends(get_image_quality_service),
) -> ImageUploadResponse:
    if await get_patient(db, patient_id) is None:
        raise HTTPException(status_code=404, detail={"code": "PATIENT_NOT_FOUND", "message": "Patient not found"})
    filename = Path(image.filename or "").name
    if not filename or Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_FILE_EXTENSION", "message": "Use a .jpg, .jpeg, or .png fundus image"})
    if image.content_type not in settings.allowed_image_mime_types:
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": "Only JPEG and PNG uploads are supported"})
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await image.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds configured size limit")
    try:
        metadata = quality_service.validate_input(content)
    except ImageTrustGateError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_IMAGE", "message": str(exc)}) from exc

    image_id = uuid4()
    suffix = ".jpg" if metadata.format == "JPEG" else ".png"
    mime_type = "image/jpeg" if metadata.format == "JPEG" else "image/png"
    key = f"fundus/{patient_id}/{image_id}{suffix}"
    storage_path = await get_storage().save(key, content, mime_type)
    record = FundusImage(
        id=image_id, patient_id=patient_id, eye=Eye(eye.value), storage_path=storage_path,
        original_filename=filename, mime_type=mime_type, file_size_bytes=len(content),
        image_metadata={"width": metadata.width, "height": metadata.height, "channels": metadata.channels, "mode": metadata.mode, "format": metadata.format, "camera_metadata": metadata.camera_metadata},
        quality_decision=QualityDecision.PENDING,
    )
    db.add(record)
    await db.commit()
    return ImageUploadResponse(image_id=image_id, patient_id=patient_id, eye=eye, quality_decision="PENDING", message="Image accepted for the Image Trust Gate")


@router.post("/{image_id}/quality", response_model=QualityAssessmentResponse)
async def assess_quality(
    image_id: UUID,
    db: AsyncSession = Depends(get_db),
    quality_service: ImageTrustGateService = Depends(get_image_quality_service),
) -> QualityAssessmentResponse:
    record = await _get_image(image_id, db)
    if record.quality_assessment and record.enhancement_passes > 0:
        return _response_from_payload(image_id, record.quality_assessment)
    return await _assess_and_persist(record, db, quality_service)


@router.post("/{image_id}/enhance", response_model=QualityAssessmentResponse)
async def enhance_image(
    image_id: UUID,
    db: AsyncSession = Depends(get_db),
    quality_service: ImageTrustGateService = Depends(get_image_quality_service),
) -> QualityAssessmentResponse:
    record = await _get_image(image_id, db)
    if record.quality_assessment and record.enhancement_passes > 0:
        return _response_from_payload(image_id, record.quality_assessment)
    return await _assess_and_persist(record, db, quality_service)


@router.get("/{image_id}/content", response_class=Response)
async def image_content(
    image_id: UUID,
    variant: str = Query(default="original", pattern="^(original|enhanced)$"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    record = await _get_image(image_id, db)
    storage_path = record.enhanced_storage_path if variant == "enhanced" else record.storage_path
    if not storage_path:
        raise HTTPException(status_code=404, detail="Enhanced image is not available")
    try:
        content = await get_storage().get(storage_path)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="Stored image is not available") from exc
    media_type = "image/png" if variant == "enhanced" else record.mime_type
    return Response(content=content, media_type=media_type, headers={"Cache-Control": "no-store"})


async def _get_image(image_id: UUID, db: AsyncSession) -> FundusImage:
    record = await db.get(FundusImage, image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return record


async def _assess_and_persist(record: FundusImage, db: AsyncSession, quality_service: ImageTrustGateService) -> QualityAssessmentResponse:
    try:
        content = await get_storage().get(record.storage_path)
        initial = await quality_service.assess(content)
        enhanced_content = None
        if initial.quality_decision == TrustGateDecision.BORDERLINE:
            enhanced_content = quality_service.enhance(content)
            final = await quality_service.assess(enhanced_content)
            if final.quality_decision == TrustGateDecision.BORDERLINE:
                final.recommended_action = "One controlled enhancement pass was used; recapture is recommended."
                final.next_action = "RECAPTURE_IMAGE"
            outcome = TrustGateOutcome(initial, final, enhancement_applied=True, enhancement_passes=1)
        else:
            outcome = TrustGateOutcome(initial, initial)
        if outcome.enhancement_applied:
            key = f"fundus/{record.patient_id}/{record.id}/enhanced-pass-{outcome.enhancement_passes}.png"
            record.enhanced_storage_path = await get_storage().save(key, enhanced_content, "image/png")
        record.quality_score = outcome.final.quality_score
        record.quality_decision = QualityDecision(outcome.final.quality_decision.lower())
        record.enhancement_passes = outcome.enhancement_passes
        record.quality_checked_at = datetime.now(timezone.utc)
        record.quality_assessment = {"initial": outcome.initial.to_dict(), "final": outcome.final.to_dict(), "enhancement_applied": outcome.enhancement_applied, "enhancement_passes": outcome.enhancement_passes}
        await db.commit()
        return _response_from_outcome(record.id, outcome)
    except ImageTrustGateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _response_from_outcome(image_id: UUID, outcome: TrustGateOutcome) -> QualityAssessmentResponse:
    return QualityAssessmentResponse(
        image_id=image_id, quality_decision=outcome.final.quality_decision,
        quality_score=outcome.initial.quality_score, final_quality_score=outcome.final.quality_score,
        component_scores=outcome.initial.component_scores, metrics=outcome.initial.metrics,
        issues=[QualityIssueResponse(**issue.__dict__) for issue in outcome.initial.issues],
        recommended_action=outcome.final.recommended_action,
        enhancement_applied=outcome.enhancement_applied, enhancement_passes=outcome.enhancement_passes,
        recheck_score=outcome.final.quality_score if outcome.enhancement_applied else None,
        recheck_decision=outcome.final.quality_decision if outcome.enhancement_applied else None,
        recheck_issues=[QualityIssueResponse(**issue.__dict__) for issue in outcome.final.issues] if outcome.enhancement_applied else [],
        next_action=outcome.final.next_action, input_metadata=outcome.initial.input_metadata,
        feature_vector=outcome.initial.feature_vector,
    )


def _response_from_payload(image_id: UUID, payload: dict) -> QualityAssessmentResponse:
    initial = payload["initial"]
    final = payload["final"]
    enhanced = bool(payload.get("enhancement_applied"))
    return QualityAssessmentResponse(
        image_id=image_id, quality_decision=final["quality_decision"], quality_score=initial["quality_score"],
        final_quality_score=final["quality_score"], component_scores=initial["component_scores"], metrics=initial["metrics"],
        issues=[QualityIssueResponse(**issue) for issue in initial["issues"]], recommended_action=final["recommended_action"],
        enhancement_applied=enhanced, enhancement_passes=int(payload.get("enhancement_passes", 0)),
        recheck_score=final["quality_score"] if enhanced else None, recheck_decision=final["quality_decision"] if enhanced else None,
        recheck_issues=[QualityIssueResponse(**issue) for issue in final["issues"]] if enhanced else [], next_action=final["next_action"],
        input_metadata=initial["input_metadata"], feature_vector=initial["feature_vector"],
    )
