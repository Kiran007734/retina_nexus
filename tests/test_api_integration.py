"""Controlled API integration coverage for the screening workflow.

The test replaces model adapters with deterministic test doubles. It verifies
HTTP contracts, persistence boundaries, safe failure behavior, and report /
review integration without claiming model performance.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import sys
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.routes import images as image_routes  # noqa: E402
from app.api.routes import reports as report_routes  # noqa: E402
from app.api.routes import screening as screening_routes  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.security import UserRole, create_access_token  # noqa: E402
from app.database.base import Base  # noqa: E402
from app.database.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.ml.evidence.service import RetinalEvidenceAnalysis  # noqa: E402
from app.ml.explainability.service import ExplainabilityAnalysis  # noqa: E402
from app.ml.inference.classifier import DRPrediction  # noqa: E402
from app.ml.quality.trust_gate import ImageInputMetadata, QualityAssessment, QualityIssue, TrustGateDecision  # noqa: E402
from app.ml.trust.guard import RetinaGuardResult  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.container import get_classifier_service, get_evidence_service, get_explainability_service, get_image_quality_service, get_retinaguard_service, get_screening_pipeline_service  # noqa: E402
from app.services.screening_pipeline import ScreeningPipelineService  # noqa: E402


def test_http_screening_workflow_and_demo_mode(monkeypatch):
    asyncio.run(_run_workflow(monkeypatch))


async def _run_workflow(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    storage = MemoryStorage()
    quality = IntegrationQualityService()
    classifier = IntegrationClassifier()
    evidence = IntegrationEvidenceService()
    explanation = IntegrationExplainabilityService()
    guard = IntegrationGuardService()
    pipeline = ScreeningPipelineService(quality, classifier, evidence, explanation, guard, storage)

    async def override_db():
        async with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_image_quality_service] = lambda: quality
    app.dependency_overrides[get_screening_pipeline_service] = lambda: pipeline
    app.dependency_overrides[get_classifier_service] = lambda: classifier
    app.dependency_overrides[get_evidence_service] = lambda: evidence
    app.dependency_overrides[get_explainability_service] = lambda: explanation
    app.dependency_overrides[get_retinaguard_service] = lambda: guard
    monkeypatch.setattr(image_routes, "get_storage", lambda: storage)
    monkeypatch.setattr(screening_routes, "get_storage", lambda: storage)
    monkeypatch.setattr(report_routes, "get_storage", lambda: storage)

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            invalid_patient = await client.post("/api/v1/patients", json={"anonymized_identifier": "x"})
            assert invalid_patient.status_code == 422
            assert invalid_patient.json()["error_code"] == "REQUEST_VALIDATION_FAILURE"
            assert invalid_patient.json()["validation_errors"] == [{"loc": ["body", "anonymized_identifier"], "msg": "String should have at least 3 characters", "type": "string_too_short"}]

            patient_response = await client.post("/api/v1/patients", json={"anonymized_identifier": "integration-patient", "age_group": "adult"})
            assert patient_response.status_code == 201
            patient_id = patient_response.json()["id"]

            invalid_mime = await client.post(
                f"/api/v1/images/upload?patient_id={patient_id}&eye=right",
                files={"image": ("bad.txt", b"not-an-image", "text/plain")},
            )
            assert invalid_mime.status_code == 415
            corrupt_image = await client.post(
                f"/api/v1/images/upload?patient_id={patient_id}&eye=right",
                files={"image": ("corrupt.jpg", b"not-an-image", "image/jpeg")},
            )
            assert corrupt_image.status_code == 422

            high_upload = await _upload(client, patient_id, "high-quality.png", _image_bytes((210, 150, 100)))
            high_id = high_upload["image_id"]
            high_quality = await client.post(f"/api/v1/images/{high_id}/quality")
            assert high_quality.status_code == 200
            assert high_quality.json()["quality_decision"] == "GRADABLE"

            # Exercise the individual pre-master contracts as well as the
            # master orchestration path below.
            classification = await client.post("/api/v1/screening/classify", json={"image_id": high_id})
            assert classification.status_code == 200
            assert classification.json()["predicted_grade_label"] == "Moderate"
            evidence_response = await client.post("/api/v1/screening/analyze-structures", json={"image_id": high_id})
            assert evidence_response.status_code == 200
            assert "microaneurysm_detection" in evidence_response.json()["modules"]
            explanation_response = await client.post("/api/v1/screening/explain", json={"image_id": high_id})
            assert explanation_response.status_code == 200
            assert explanation_response.json()["attention_lesion_agreement"]["score"] == 0.86
            trust_response = await client.post("/api/v1/screening/trust", json={"image_id": high_id})
            assert trust_response.status_code == 200
            assert trust_response.json()["trust_category"] == "TRUSTED"

            master = await client.post("/api/v1/screening/run", json={"image_id": high_id})
            assert master.status_code == 200
            master_payload = master.json()
            assert master_payload["status"] == "COMPLETED"
            assert master_payload["primary_status"] == "COMPLETED"
            assert master_payload["classification"]["predicted_grade_label"] == "Moderate"
            assert master_payload["evidence_status"] == "PROCESSING"
            assert master_payload["lesions"] is None
            assert master_payload["explainability"] is None
            assert master_payload["retinaguard"]["trust_category"] == "TRUSTED"
            assert master_payload["triage"]["recommendation"] == "SPECIALIST_REVIEW_RECOMMENDED"
            assert master_payload["stage_metrics"]["dr_classification"]["duration_ms"] is not None

            borderline_upload = await _upload(client, patient_id, "borderline.png", _image_bytes((120, 120, 120)))
            borderline_quality = await client.post(f"/api/v1/images/{borderline_upload['image_id']}/quality")
            assert borderline_quality.status_code == 200
            assert borderline_quality.json()["enhancement_applied"] is True
            assert borderline_quality.json()["enhancement_passes"] == 1
            assert borderline_quality.json()["recheck_score"] == 0.84

            poor_upload = await _upload(client, patient_id, "poor.png", _image_bytes((20, 20, 20)))
            poor_quality = await client.post(f"/api/v1/images/{poor_upload['image_id']}/quality")
            assert poor_quality.status_code == 200
            assert poor_quality.json()["quality_decision"] == "UNGRADABLE"
            blocked = await client.post("/api/v1/screening/run", json={"image_id": poor_upload["image_id"]})
            assert blocked.status_code == 200
            blocked_payload = blocked.json()
            assert blocked_payload["classification"] is None
            assert blocked_payload["triage"]["recommendation"] == "RECAPTURE_IMAGE"
            assert blocked_payload["stage_status"]["dr_classification"] == "SKIPPED"

            async with session_factory() as db:
                clinician = User(id=uuid4(), email="integration-clinician@example.test", full_name="Integration Clinician", password_hash="unused", role=UserRole.CLINICIAN, is_active=True)
                db.add(clinician)
                await db.commit()
                token = create_access_token(str(clinician.id), UserRole.CLINICIAN)

            review = await client.post(
                f"/api/v1/reviews/{master_payload['screening_id']}",
                headers={"Authorization": f"Bearer {token}"},
                json={"decision": "approve", "comments": "Integration review recorded."},
            )
            assert review.status_code == 201
            assert review.json()["decision"] == "approve"
            report = await client.post("/api/v1/reports/generate", json={"session_id": master_payload["screening_id"]})
            assert report.status_code == 201
            assert report.json()["report"]["ai_assessment"]["predicted_grade_label"] == "Moderate"
            report_pdf = await client.get(f"/api/v1/reports/{report.json()['report_id']}/pdf")
            assert report_pdf.status_code == 200
            assert report_pdf.content.startswith(b"%PDF")

            settings = get_settings()
            original_demo, original_environment = settings.demo_mode_enabled, settings.environment
            settings.demo_mode_enabled = True
            settings.environment = "production"
            assert (await client.get("/api/v1/demo/scenarios")).status_code == 404
            settings.environment = "development"
            demo_scenarios = await client.get("/api/v1/demo/scenarios")
            assert demo_scenarios.status_code == 200
            assert demo_scenarios.json()["sample_data"] is True
            assert len(demo_scenarios.json()["scenarios"]) == 3
            demo_run = await client.post("/api/v1/demo/scenarios/image-1-trusted-refer/run")
            assert demo_run.status_code == 200
            assert demo_run.json()["scenario"]["retinaguard"]["trust_category"] == "TRUSTED"
            assert demo_run.json()["persisted_to_clinical_records"] is False
            settings.demo_mode_enabled, settings.environment = original_demo, original_environment
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


async def _upload(client: httpx.AsyncClient, patient_id: str, filename: str, content: bytes) -> dict[str, Any]:
    response = await client.post(
        f"/api/v1/images/upload?patient_id={patient_id}&eye=right",
        files={"image": (filename, content, "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


class MemoryStorage:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    async def save(self, key: str, content: bytes, _content_type: str) -> str:
        self.objects[key] = content
        return key

    async def get(self, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]


class IntegrationQualityService:
    def __init__(self):
        self.assessment_count: dict[str, int] = {}

    def validate_input(self, content: bytes) -> ImageInputMetadata:
        from app.ml.quality.trust_gate import ImageTrustGateService

        return ImageTrustGateService().validate_input(content)

    async def assess(self, content: bytes) -> QualityAssessment:
        key = hashlib.sha256(content).hexdigest()
        self.assessment_count[key] = self.assessment_count.get(key, 0) + 1
        with Image.open(io.BytesIO(content)) as image:
            brightness = sum(image.convert("RGB").resize((1, 1)).getpixel((0, 0))) / 3
        if brightness < 50:
            decision, score = TrustGateDecision.UNGRADABLE, 0.18
            issue = QualityIssue("severe_blur", "severe", "The retinal image has insufficient focus detail.", "Stabilize the camera and refocus before recapturing.")
        elif brightness < 180 and self.assessment_count[key] == 1:
            decision, score = TrustGateDecision.BORDERLINE, 0.74
            issue = QualityIssue("low_contrast", "moderate", "Tonal separation is limited.", "Apply one controlled enhancement pass and reassess.")
        else:
            decision, score = TrustGateDecision.GRADABLE, 0.84 if brightness < 180 else 0.94
            issue = None
        return QualityAssessment(
            quality_decision=decision, quality_score=score,
            component_scores={"focus": score, "illumination": score, "contrast": score, "field_of_view": score, "exposure": score, "artifacts": score},
            metrics={"laplacian_variance": 120.0, "mean_intensity": brightness},
            issues=[issue] if issue else [],
            recommended_action="Recapture" if decision == TrustGateDecision.UNGRADABLE else "Proceed",
            next_action="RECAPTURE_IMAGE" if decision == TrustGateDecision.UNGRADABLE else "ENHANCE_AND_REASSESS" if decision == TrustGateDecision.BORDERLINE else "CONTINUE_SCREENING",
            input_metadata={"width": 512, "height": 512, "channels": 3, "mode": "RGB", "format": "PNG", "camera_metadata": {}},
            feature_vector={"focus": score, "contrast": score, "mean_intensity": brightness / 255},
        )

    def enhance(self, content: bytes) -> bytes:
        return content


class IntegrationClassifier:
    async def classify(self, _content: bytes) -> DRPrediction:
        return _prediction()


class IntegrationEvidenceService:
    async def analyze(self, *_args) -> RetinalEvidenceAnalysis:
        return _evidence(str(_args[1]), str(_args[2]))


class IntegrationExplainabilityService:
    async def analyze(self, *_args, **_kwargs) -> ExplainabilityAnalysis:
        return _explanation(str(_args[1]), str(_args[2]))


class IntegrationGuardService:
    async def evaluate_async(self, *_args, **_kwargs) -> RetinaGuardResult:
        return _guard_result()


def _prediction() -> DRPrediction:
    return DRPrediction(
        predicted_grade=2, predicted_grade_label="Moderate",
        probabilities={"No DR": 0.02, "Mild": 0.03, "Moderate": 0.85, "Severe": 0.07, "Proliferative DR": 0.03},
        referable_dr=True, referable_probability=0.95, raw_confidence=0.85,
        model_name="integration-classifier", model_version="integration-classifier-v1", backbone="integration",
        referable_mapping={"threshold": 2}, hierarchical_probabilities={}, ordinal_mode=False, severity_logits=[0.0] * 5,
    )


def _evidence(image_id: str = "image", screening_session_id: str = "session") -> RetinalEvidenceAnalysis:
    module = {"module": "microaneurysm_detection", "category": "lesion_detection", "status": "experimental_test_adapter", "supported": True, "implementation": "integration-test", "confidence": 0.88, "count": 4, "mask_data_uri": None, "bounding_regions": [{"x": 10, "y": 10, "width": 12, "height": 12}], "landmarks": [], "issues": [], "metadata": {}}
    return RetinalEvidenceAnalysis(
        image_id=image_id, screening_session_id=screening_session_id, status="completed", image_metadata={"width": 512, "height": 512}, coarse_to_fine={},
        modules={"microaneurysm_detection": module}, anatomical_landmarks=[], evidence_map_data_uri=None, dataset_support={}, note="test adapter",
    )


def _explanation(image_id: str = "image", screening_session_id: str = "session") -> ExplainabilityAnalysis:
    return ExplainabilityAnalysis(
        image_id=image_id, screening_session_id=screening_session_id, predicted_class=2, predicted_class_label="Moderate", model_version="integration-classifier-v1",
        classification={"predicted_grade": 2, "predicted_grade_label": "Moderate"}, grad_cam={"heatmap_data_uri": "data:image/png;base64,test", "overlay_data_uri": "data:image/png;base64,test", "normalized_attention_map_data_uri": "data:image/png;base64,test"}, lesion_evidence_map_data_uri=None,
        attention_lesion_agreement={"status": "HIGH AGREEMENT", "score": 0.86, "metrics": {"dice": 0.86}}, explanation_stability={"status": "SKIPPED"}, counterfactual={"status": "SKIPPED"}, note="test adapter",
    )


def _guard_result() -> RetinaGuardResult:
    return RetinaGuardResult(
        trust_score=0.86, trust_category="TRUSTED", contributing_factors=[], risk_flags=[], recommended_action="AI triage may proceed",
        calibration={"calibrated_confidence": 0.85}, uncertainty={"score": 0.1}, model_disagreement={"disagreement": 0.0}, ood={"status": "UNAVAILABLE"},
        signal_snapshot={}, configuration={"version": "integration-guard-v1"}, reason_summary=["Integration test signals are aligned."],
    )


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (512, 512), color).save(output, format="PNG")
    return output.getvalue()
