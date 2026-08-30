import asyncio
import io
import sys
from pathlib import Path
from uuid import uuid4

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.evidence.service import RetinalEvidenceAnalysis  # noqa: E402
from app.ml.explainability.service import ExplainabilityAnalysis  # noqa: E402
from app.ml.inference.classifier import DRPrediction  # noqa: E402
from app.ml.quality.trust_gate import (  # noqa: E402
    ImageInputMetadata,
    QualityAssessment,
    TrustGateDecision,
)
from app.ml.trust.guard import RetinaGuardResult  # noqa: E402
from app.models.fundus_image import Eye, FundusImage, QualityDecision  # noqa: E402
from app.models.screening import ScreeningSession, ScreeningStatus  # noqa: E402
from app.models.screening_run import ScreeningRun  # noqa: E402
from app.services.screening_pipeline import RUN_STAGES, ScreeningPipelineService  # noqa: E402


class FakeDB:
    def __init__(self):
        self.events = []

    def add(self, value):
        self.events.append(value)

    async def commit(self):
        return None


class FakeStorage:
    def __init__(self, content: bytes):
        self.content = content
        self.saved = []

    async def get(self, _key: str) -> bytes:
        return self.content

    async def save(self, key: str, content: bytes, content_type: str) -> str:
        self.saved.append((key, content, content_type))
        return key


class FakeQualityService:
    def __init__(self, decision: str):
        self.decision = decision

    def validate_input(self, _content: bytes):
        return ImageInputMetadata(512, 512, 3, "RGB", "PNG", {})

    async def assess(self, _content: bytes):
        return QualityAssessment(
            quality_decision=self.decision,
            quality_score=0.9 if self.decision == TrustGateDecision.GRADABLE else 0.2,
            component_scores={"focus": 0.9},
            metrics={},
            issues=[],
            recommended_action="Proceed" if self.decision == TrustGateDecision.GRADABLE else "Recapture",
            next_action="CONTINUE_SCREENING" if self.decision == TrustGateDecision.GRADABLE else "RECAPTURE_IMAGE",
            input_metadata={},
            feature_vector={"focus": 0.9},
        )

    def enhance(self, content: bytes) -> bytes:
        return content


class FakeClassifier:
    def __init__(self, prediction: DRPrediction, error: Exception | None = None):
        self.prediction = prediction
        self.error = error
        self.called = False

    async def classify(self, _content: bytes):
        self.called = True
        if self.error:
            raise self.error
        return self.prediction


class FakeEvidenceService:
    def __init__(self, evidence: RetinalEvidenceAnalysis):
        self.evidence = evidence
        self.called = False

    async def analyze(self, *_args):
        self.called = True
        return self.evidence


class FakeExplainabilityService:
    def __init__(self, explanation: ExplainabilityAnalysis):
        self.explanation = explanation

    async def analyze(self, *_args, **_kwargs):
        return self.explanation


class FakeRetinaGuard:
    def __init__(self, result: RetinaGuardResult):
        self.result = result

    async def evaluate_async(self, *_args, **_kwargs):
        return self.result


def test_ungradable_image_blocks_all_clinical_ai_stages():
    content = _image_bytes()
    classifier = FakeClassifier(_prediction())
    evidence = FakeEvidenceService(_evidence())
    service = ScreeningPipelineService(
        FakeQualityService(TrustGateDecision.UNGRADABLE), classifier, evidence,
        FakeExplainabilityService(_explanation()), FakeRetinaGuard(_guard_result()), FakeStorage(content),
    )
    db, run, session, image = _records()

    output = asyncio.run(service.execute(db, run, session, image, None))

    assert output.status == "COMPLETED"
    assert output.classification is None
    assert output.triage["recommendation"] == "RECAPTURE_IMAGE"
    assert output.stage_status["dr_classification"] == "SKIPPED"
    assert output.stage_status["retinaguard"] == "SKIPPED"
    assert output.stage_status["triage"] == "COMPLETED"
    assert classifier.called is False
    assert evidence.called is False
    assert session.status == ScreeningStatus.NEEDS_REVIEW


def test_stage_failure_is_persisted_without_fake_downstream_results():
    content = _image_bytes()
    classifier = FakeClassifier(_prediction(), RuntimeError("registered classifier unavailable"))
    service = ScreeningPipelineService(
        FakeQualityService(TrustGateDecision.GRADABLE), classifier, FakeEvidenceService(_evidence()),
        FakeExplainabilityService(_explanation()), FakeRetinaGuard(_guard_result()), FakeStorage(content),
    )
    db, run, session, image = _records()

    output = asyncio.run(service.execute(db, run, session, image, None))

    assert output.status == "FAILED"
    assert output.error["stage"] == "dr_classification"
    assert "registered classifier unavailable" in output.error["message"]
    assert output.classification is None
    assert output.lesions is None
    assert output.stage_status["dr_classification"] == "FAILED"
    assert session.status == ScreeningStatus.FAILED


def _records():
    patient_id = uuid4()
    image = FundusImage(
        id=uuid4(), patient_id=patient_id, eye=Eye.RIGHT, storage_path="fundus/image.png",
        original_filename="image.png", mime_type="image/png", file_size_bytes=100,
        quality_decision=QualityDecision.PENDING,
    )
    session = ScreeningSession(id=uuid4(), patient_id=patient_id, fundus_image_id=image.id, status=ScreeningStatus.QUEUED)
    run = ScreeningRun(
        id=session.id, fundus_image_id=image.id, status="QUEUED",
        stage_status={stage: "PENDING" for stage in RUN_STAGES},
    )
    return FakeDB(), run, session, image


def _prediction():
    return DRPrediction(
        predicted_grade=2, predicted_grade_label="Moderate",
        probabilities={"No DR": 0.02, "Mild": 0.03, "Moderate": 0.85, "Severe": 0.07, "Proliferative DR": 0.03},
        referable_dr=True, referable_probability=0.95, raw_confidence=0.85,
        model_name="test-classifier", model_version="test-classifier-v1", backbone="test",
        referable_mapping={"threshold": 2}, hierarchical_probabilities={}, ordinal_mode=False,
        severity_logits=[0.0] * 5,
    )


def _evidence():
    return RetinalEvidenceAnalysis(
        image_id="image", screening_session_id="session", status="completed", image_metadata={},
        coarse_to_fine={}, modules={}, anatomical_landmarks=[], evidence_map_data_uri=None,
        dataset_support={}, note="test",
    )


def _explanation():
    return ExplainabilityAnalysis(
        image_id="image", screening_session_id="session", predicted_class=2,
        predicted_class_label="Moderate", model_version="test-classifier-v1",
        classification={}, grad_cam={}, lesion_evidence_map_data_uri=None,
        attention_lesion_agreement={"status": "UNAVAILABLE", "score": None},
        explanation_stability={"status": "SKIPPED"}, counterfactual={"status": "SKIPPED"},
        note="test",
    )


def _guard_result():
    return RetinaGuardResult(
        trust_score=0.8, trust_category="TRUSTED", contributing_factors=[], risk_flags=[],
        recommended_action="AI_TRIAGE_MAY_PROCEED", calibration={}, uncertainty={},
        model_disagreement={}, ood={}, signal_snapshot={}, configuration={"version": "test-guard-v1"},
        reason_summary=[],
    )


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (512, 512), (90, 110, 130)).save(output, format="PNG")
    return output.getvalue()
