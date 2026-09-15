import asyncio
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.inference.classifier import DRPrediction  # noqa: E402
from app.ml.inference.referable_fusion import ReferableFusionService  # noqa: E402
from app.services.screening_persistence import authoritative_referable_value  # noqa: E402
from app.services.screening_pipeline import ScreeningPipelineService  # noqa: E402
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs  # noqa: E402


def _prediction(probability: float, referable: bool | None = None) -> DRPrediction:
    return DRPrediction(
        predicted_grade=1,
        predicted_grade_label="Mild",
        probabilities={"No DR": 0.1, "Mild": 0.7, "Moderate": 0.1, "Severe": 0.05, "Proliferative DR": 0.05},
        referable_dr=probability >= 0.5 if referable is None else referable,
        referable_probability=probability,
        raw_confidence=0.7,
        model_name="test-primary",
        model_version="test-primary-v1",
        backbone="test",
        referable_mapping={"name": "test"},
        hierarchical_probabilities={},
        ordinal_mode=False,
        severity_logits=[0.0] * 5,
    )


def test_persisted_result_uses_fused_referable_value_without_changing_grade():
    prediction = _prediction(0.20, referable=False)
    payload = {"signal_snapshot": {"referable_fusion": {"fused_referable": True}}}
    assert prediction.predicted_grade == 1
    assert authoritative_referable_value(payload, prediction) is True


def test_persisted_result_falls_back_to_primary_when_fusion_is_disabled():
    prediction = _prediction(0.62, referable=True)
    payload = {"signal_snapshot": {"referable_fusion": {"status": "DISABLED"}}}
    assert authoritative_referable_value(payload, prediction) is True


class StubVerifier:
    def __init__(self, probability: float):
        self.probability = probability

    def predict(self, _image_bytes: bytes):
        return {"probability": self.probability, "referable": self.probability >= 0.204983, "model_version": "stub-verifier", "ood": {"status": "IN_DISTRIBUTION", "score": 1.0}}


class FailingVerifier:
    def predict(self, _image_bytes: bytes):
        raise RuntimeError("verifier unavailable")


def run(service, primary):
    return asyncio.run(service.evaluate(b"image", primary))


@pytest.mark.parametrize(
    ("primary_probability", "verifier_probability", "expected_referable", "expected_disagreement"),
    [
        (0.20, 0.10, False, False),
        (0.60, 0.10, True, True),
        (0.10, 0.50, True, True),
        (0.60, 0.50, True, False),
    ],
)
def test_maximum_probability_rule_and_disagreement(primary_probability, verifier_probability, expected_referable, expected_disagreement):
    result = run(ReferableFusionService(enabled=True, verifier=StubVerifier(verifier_probability)), _prediction(primary_probability))
    assert result.fused_referable is expected_referable
    assert result.fused_probability == pytest.approx(max(primary_probability, verifier_probability))
    assert result.disagreement is expected_disagreement


def test_severity_grade_remains_primary_when_fused_referable_is_true():
    result = run(ReferableFusionService(enabled=True, verifier=StubVerifier(0.50)), _prediction(0.10))
    payload = ScreeningPipelineService._classification_payload(_prediction(0.10), result)
    assert payload["predicted_grade"] == 1
    assert payload["referable_dr"] is True
    assert payload["referable_fusion"]["fused_referable"] is True


def test_disabled_fusion_preserves_primary_result():
    result = run(ReferableFusionService.disabled(), _prediction(0.20))
    assert result.status == "DISABLED"
    assert result.fused_probability == pytest.approx(0.20)
    assert result.fused_referable is False
    assert result.verifier_probability is None


def test_verifier_unavailable_keeps_primary_and_requests_review():
    result = run(ReferableFusionService(enabled=True, verifier=FailingVerifier()), _prediction(0.60))
    assert result.status == "VERIFIER_UNAVAILABLE"
    assert result.fused_referable is True
    assert result.review_recommended is True
    assert result.verifier_probability is None


def test_verifier_only_and_insufficient_evidence_fallbacks():
    verifier_only = run(ReferableFusionService(enabled=True, verifier=StubVerifier(0.50)), None)
    assert verifier_only.status == "VERIFIER_ONLY_FALLBACK"
    assert verifier_only.fused_referable is True
    assert verifier_only.review_recommended is True

    unavailable = run(ReferableFusionService(enabled=True, verifier=FailingVerifier()), None)
    assert unavailable.status == "INSUFFICIENT_EVIDENCE"
    assert unavailable.fused_referable is None
    assert unavailable.review_recommended is True


def test_retinaguard_receives_fusion_disagreement_as_review_signal():
    result = RetinaGuardEngine().evaluate(RetinaGuardInputs(
        quality_score=0.9,
        raw_confidence=0.8,
        probabilities={"No DR": 0.1, "Mild": 0.6, "Moderate": 0.1, "Severe": 0.1, "Proliferative DR": 0.1},
        predicted_grade=1,
        predicted_grade_label="Mild",
        model_version="primary-v1",
        referable_dr=True,
        referable_fusion={
            "status": "COMPLETED",
            "primary_referable": False,
            "verifier_referable": True,
            "disagreement": True,
            "fused_referable": True,
            "fused_probability": 0.62,
            "fusion_threshold": 0.4,
        },
    ))
    assert result.model_disagreement["referable_disagreement"] is True
    assert any(flag["code"] == "referable_model_disagreement" for flag in result.risk_flags)
