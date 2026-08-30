"""Screening report generation and PDF export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_optional_claims
from app.database.session import get_db
from app.models.audit_log import AuditLog
from app.models.clinical_review import ClinicalReview
from app.models.fundus_image import FundusImage
from app.models.generated_report import GeneratedReport
from app.models.patient import Patient
from app.models.retinaguard_result import RetinaGuardResult
from app.models.screening import ScreeningResult, ScreeningSession
from app.models.screening_run import ScreeningRun
from app.schemas.reports import ReportGenerateRequest, ReportPayload, ReportResponse
from app.storage.container import get_storage

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=list[ReportResponse])
async def reports(db: AsyncSession = Depends(get_db)) -> list[ReportResponse]:
    items = (await db.execute(select(GeneratedReport).order_by(GeneratedReport.created_at.desc()))).scalars().all()
    return [_response(item) for item in items]


@router.post("/generate", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def generate_report(
    payload: ReportGenerateRequest,
    db: AsyncSession = Depends(get_db),
    claims: dict | None = Depends(get_optional_claims),
) -> ReportResponse:
    session = await db.get(ScreeningSession, payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Screening session not found")
    run = await db.get(ScreeningRun, session.id)
    image = await db.get(FundusImage, session.fundus_image_id)
    patient = await db.get(Patient, session.patient_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Fundus image not found")
    if run is None:
        raise HTTPException(status_code=409, detail="The session has no master screening run to report")
    screening_result = (await db.execute(select(ScreeningResult).where(ScreeningResult.session_id == session.id))).scalar_one_or_none()
    guard = (await db.execute(select(RetinaGuardResult).where(RetinaGuardResult.screening_session_id == session.id))).scalar_one_or_none()
    review = (await db.execute(select(ClinicalReview).where(ClinicalReview.screening_session_id == session.id).order_by(ClinicalReview.created_at.desc()).limit(1))).scalar_one_or_none()
    existing_report = (await db.execute(select(GeneratedReport).where(GeneratedReport.screening_session_id == session.id))).scalar_one_or_none()
    report_id = existing_report.id if existing_report else uuid4()
    report = _build_payload(session, image, patient, run, screening_result, guard, review)
    body = json.dumps(report.model_dump(mode="json"), indent=2, default=str)
    pdf_path = f"reports/{session.id}/{report_id}.pdf"
    try:
        await get_storage().save(pdf_path, _pdf_bytes(report), "application/pdf")
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"Report storage is unavailable: {exc}") from exc
    record = existing_report or GeneratedReport(id=report_id, screening_session_id=session.id)
    record.report_status = "ready"
    record.report_body = body
    record.storage_path = pdf_path
    db.add(record)
    actor_id = _actor_id(claims)
    db.add(AuditLog(
        actor_id=actor_id, action="report.generated", resource_type="screening_session",
        resource_id=str(session.id), details={"report_id": str(report_id), "timestamp": datetime.now(timezone.utc).isoformat()},
    ))
    await db.commit()
    return _response(record, report)


@router.get("/{report_id}", response_model=ReportResponse)
async def report(report_id: UUID, db: AsyncSession = Depends(get_db)) -> ReportResponse:
    item = await db.get(GeneratedReport, report_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return _response(item)


@router.get("/{report_id}/pdf", response_class=Response)
async def report_pdf(report_id: UUID, db: AsyncSession = Depends(get_db)) -> Response:
    item = await db.get(GeneratedReport, report_id)
    if item is None or not item.storage_path:
        raise HTTPException(status_code=404, detail="PDF report not found")
    try:
        content = await get_storage().get(item.storage_path)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="PDF report is not available in storage") from exc
    return Response(content=content, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="retina-nexus-{report_id}.pdf"', "Cache-Control": "no-store"})


def _response(item: GeneratedReport, report: ReportPayload | None = None) -> ReportResponse:
    if report is None and item.report_body:
        try:
            report = ReportPayload.model_validate(json.loads(item.report_body))
        except (json.JSONDecodeError, ValueError):
            report = None
    return ReportResponse(
        report_id=item.id, session_id=item.screening_session_id, status=item.report_status,
        download_url=f"/api/v1/reports/{item.id}/pdf" if item.storage_path else None,
        created_at=item.created_at, report=report,
    )


def _build_payload(session: ScreeningSession, image: FundusImage, patient: Patient | None, run: ScreeningRun, screening_result: ScreeningResult | None, guard: RetinaGuardResult | None, review: ClinicalReview | None) -> ReportPayload:
    classification = run.classification or {}
    quality = run.quality or image.quality_assessment or {}
    triage = run.triage or {}
    clinician = None
    if review is not None:
        clinician = {"decision": review.reviewer_decision.value, "modified_grade": review.modified_grade, "comments": review.feedback, "reviewer_id": str(review.reviewer_id), "created_at": review.created_at}
    return ReportPayload(
        screening_id=session.id, session_id=session.id,
        patient_identifier=patient.anonymized_identifier if patient else None,
        eye=image.eye.value, generated_at=datetime.now(timezone.utc),
        image_quality={"decision": (quality.get("final") or {}).get("quality_decision", image.quality_decision.value), "score": (quality.get("final") or {}).get("quality_score", image.quality_score), "assessment": quality},
        ai_assessment={"predicted_grade": classification.get("predicted_grade", screening_result.dr_grade if screening_result else None), "predicted_grade_label": classification.get("predicted_grade_label"), "referable_dr": classification.get("referable_dr", screening_result.referable_dr if screening_result else None), "confidence": classification.get("raw_confidence", screening_result.confidence if screening_result else None), "model_version": classification.get("model_version")},
        clinical_evidence={"summary": _evidence_summary(run.lesions), "visualization": (run.lesions or {}).get("evidence_map_data_uri")},
        explainability={"summary": "Class-specific Grad-CAM and attention/evidence comparison.", "agreement": (run.explainability or {}).get("attention_lesion_agreement"), "visualization": (run.explainability or {}).get("grad_cam", {}).get("overlay_data_uri")},
        retinaguard=run.retinaguard or (guard.to_dict() if guard else {}),
        recommended_action=triage.get("recommendation"), clinician_decision=clinician,
        disclaimer="Prototype report. AI output is a screening recommendation, not a diagnosis or regulatory approval. Final clinical responsibility remains with the reviewing clinician.",
    )


def _evidence_summary(lesions: dict | None) -> list[dict]:
    values = []
    for name, module in ((lesions or {}).get("modules") or {}).items():
        values.append({"lesion_type": name, "status": module.get("status"), "count": module.get("count"), "confidence": module.get("confidence"), "supported": module.get("supported")})
    return values


def _actor_id(claims: dict | None) -> UUID | None:
    try:
        return UUID(str(claims["sub"])) if claims and claims.get("sub") else None
    except (TypeError, ValueError):
        return None


def _pdf_bytes(report: ReportPayload) -> bytes:
    """Create a dependency-free, text-first PDF prototype for the report."""
    ai = report.ai_assessment
    trust = report.retinaguard
    lines = [
        "RETINA-NEXUS SCREENING REPORT", "", f"Screening ID: {report.screening_id}",
        f"Eye: {report.eye.upper()}", f"Generated: {report.generated_at or '—'}", "",
        "AI SCREENING RECOMMENDATION", f"DR grade: {ai.get('predicted_grade_label') or ai.get('predicted_grade') or 'Unavailable'}",
        f"Referable DR: {ai.get('referable_dr', 'Unavailable')}", f"Confidence: {ai.get('confidence', 'Unavailable')}",
        f"Recommended action: {report.recommended_action or 'Unavailable'}", "",
        "IMAGE QUALITY", f"Decision: {report.image_quality.get('decision', 'Unavailable')}", f"Score: {report.image_quality.get('score', 'Unavailable')}", "",
        "RETINAGUARD", f"Trust score: {trust.get('trust_score', 'Unavailable')}", f"Trust category: {trust.get('trust_category', 'Unavailable')}",
        "Reasons: " + "; ".join(trust.get("reason_summary", []))[:400], "",
        "CLINICAL EVIDENCE", "Lesion summary: " + json.dumps(report.clinical_evidence.get("summary", []), default=str)[:600],
        "", "CLINICIAN DECISION", json.dumps(report.clinician_decision, default=str) if report.clinician_decision else "Pending human review.",
        "", report.disclaimer,
    ]
    escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")[:110] for line in lines]
    stream = "BT\n/F1 10 Tf\n50 760 Td\n" + "\n".join(f"({line}) Tj\n0 -16 Td" for line in escaped) + "\nET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", f"<< /Length {len(stream.encode('latin-1', 'replace'))} >>\nstream\n{stream}\nendstream",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n{obj}\nendobj\n".encode("latin-1", "replace")
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return pdf
