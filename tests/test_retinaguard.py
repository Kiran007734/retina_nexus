from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.trust.calibration import TemperatureScaler  # noqa: E402
from app.ml.trust.disagreement import calculate_model_disagreement  # noqa: E402
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs, derive_vessel_evidence_status  # noqa: E402
from app.ml.trust.ood import FeatureDistributionMonitor  # noqa: E402
from app.ml.trust.uncertainty import UncertaintyEstimator  # noqa: E402


def test_temperature_scaling_and_uncertainty_are_measurable():
    probabilities = {"No DR": 0.02, "Mild": 0.03, "Moderate": 0.90, "Severe": 0.03, "Proliferative DR": 0.02}
    calibrated = TemperatureScaler(temperature=2.0, version="heldout-v1", fitted=True).calibrate(probabilities)
    uncertainty = UncertaintyEstimator().estimate(probabilities)

    assert calibrated.fitted is True
    assert calibrated.calibrated_confidence < 0.90
    assert 0.0 <= uncertainty["score"] <= 1.0
    assert uncertainty["predictive_entropy"] < 0.5
    mc = UncertaintyEstimator().estimate(probabilities, [list(probabilities.values()), [0.01, 0.04, 0.87, 0.06, 0.02]])
    assert mc["mc_dropout"]["status"] == "COMPLETED"
    assert "+mc_dropout" in mc["method"]


def test_model_disagreement_increases_with_severity_distance():
    result = calculate_model_disagreement([
        {"model_version": "efficientnet-v1", "predicted_grade": 2},
        {"model_version": "resnet-v1", "predicted_grade": 0},
    ])

    assert result["status"] == "COMPLETED"
    assert result["disagreement"] >= 0.5
    assert result["severity"] == "high"


def test_ood_monitor_reports_shift_only_against_configured_reference(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text('{"features": {"focus": {"mean": 0.8, "std": 0.1}, "contrast": {"mean": 0.7, "std": 0.1}}}', encoding="utf-8")
    monitor = FeatureDistributionMonitor(str(reference), threshold=3.0)

    result = monitor.evaluate({"focus": 1.2, "contrast": 0.7})
    assert result["status"] == "SHIFTED"
    assert result["score"] < 0.2


def test_retinaguard_trusted_path_exposes_configuration():
    engine = RetinaGuardEngine()
    result = engine.evaluate(RetinaGuardInputs(
        quality_score=0.95,
        raw_confidence=0.92,
        probabilities={"No DR": 0.01, "Mild": 0.02, "Moderate": 0.90, "Severe": 0.05, "Proliferative DR": 0.02},
        model_predictions=[{"model_version": "resnet-v1", "predicted_grade": 2}],
        lesion_evidence_strength=0.85,
        attention_lesion_agreement={"status": "HIGH AGREEMENT", "score": 0.88},
        explanation_stability={"status": "COMPLETED", "prediction_stability": 1.0, "grad_cam_stability": 0.9},
        ood={"status": "IN_DISTRIBUTION", "score": 0.95},
        predicted_grade=2,
        predicted_grade_label="Moderate",
        model_version="efficientnet-v1",
    ))

    assert result.trust_category == "TRUSTED"
    assert result.trust_score >= 0.75
    assert result.configuration["version"] == "retinaguard-v2-reliability"
    assert result.to_dict()["reliability_state"] == "TRUSTED"
    assert result.to_dict()["recommended_safe_action"] == "AUTOMATED_RESULT_AVAILABLE"
    assert sum(item["weight"] for item in result.contributing_factors) == 1.0


def test_retinaguard_unreliable_path_reports_reasons():
    result = RetinaGuardEngine().evaluate(RetinaGuardInputs(
        quality_score=0.2,
        raw_confidence=0.4,
        probabilities={"No DR": 0.2, "Mild": 0.2, "Moderate": 0.2, "Severe": 0.2, "Proliferative DR": 0.2},
        model_predictions=[{"model_version": "a", "predicted_grade": 0}, {"model_version": "b", "predicted_grade": 4}],
        lesion_evidence_strength=0.1,
        attention_lesion_agreement={"status": "LOW AGREEMENT", "score": 0.1},
        explanation_stability={"status": "COMPLETED", "prediction_stability": 0.2, "grad_cam_stability": 0.2},
        ood={"status": "SHIFTED", "score": 0.05},
        predicted_grade=0,
        predicted_grade_label="No DR",
        model_version="a",
    ))

    codes = {flag["code"] for flag in result.risk_flags}
    assert result.trust_category == "UNRELIABLE"
    assert "low_image_quality" in codes
    assert "high_model_disagreement" in codes
    assert "distribution_shift" in codes
    assert result.recommended_action.startswith("Recapture")


def test_vessel_evidence_provenance_is_explicit_and_not_a_score_factor():
    assert derive_vessel_evidence_status({"modules": {"vessel_segmentation": {"supported": True, "status": "model_inference"}}}) == "REAL_MODEL_EVIDENCE"
    assert derive_vessel_evidence_status({"modules": {"vessel_segmentation": {"supported": True, "status": "experimental_heuristic"}}}) == "EXPERIMENTAL_BASELINE"
    assert derive_vessel_evidence_status({"modules": {"vessel_segmentation": {"supported": False, "status": "unsupported"}}}) == "UNAVAILABLE"
    result = RetinaGuardEngine().evaluate(RetinaGuardInputs(vessel_evidence_status="REAL_MODEL_EVIDENCE"))
    assert result.signal_snapshot["vessel_evidence_status"] == "REAL_MODEL_EVIDENCE"
    assert "vessel_evidence_policy" in result.configuration


def test_missing_signals_cannot_be_marked_trusted():
    result = RetinaGuardEngine().evaluate(RetinaGuardInputs(
        raw_confidence=0.99,
        probabilities={"No DR": 0.01, "Mild": 0.01, "Moderate": 0.90, "Severe": 0.06, "Proliferative DR": 0.02},
        predicted_grade=2,
        model_version="test-v1",
    ))

    assert result.trust_category != "TRUSTED"
    assert {flag["code"] for flag in result.risk_flags} >= {
        "quality_not_available", "attention_evidence_not_available", "ood_not_available",
    }


def test_retinaguard_review_recommended_is_distinct_from_trusted():
    result = RetinaGuardEngine().evaluate(RetinaGuardInputs(
        quality_score=0.82,
        raw_confidence=0.70,
        probabilities={"No DR": 0.03, "Mild": 0.05, "Moderate": 0.65, "Severe": 0.20, "Proliferative DR": 0.07},
        attention_lesion_agreement={"status": "MODERATE AGREEMENT", "score": 0.62},
        explanation_stability={"status": "COMPLETED", "prediction_stability": 0.82, "grad_cam_stability": 0.76},
        ood={"status": "IN_DISTRIBUTION", "score": 0.82},
        predicted_grade=2,
        predicted_grade_label="Moderate",
        model_version="efficientnet-v1",
    ))

    assert result.trust_category == "REVIEW_RECOMMENDED"
    assert result.configuration["safe_action"] == "PROFESSIONAL_REVIEW_RECOMMENDED"
    assert result.to_dict()["reliability_state"] == "REVIEW_RECOMMENDED"


def test_retinaguard_insufficient_evidence_is_safe_and_explicit():
    result = RetinaGuardEngine().evaluate(RetinaGuardInputs(
        quality_score=0.90,
        raw_confidence=0.90,
        probabilities={"No DR": 0.01, "Mild": 0.02, "Moderate": 0.90, "Severe": 0.05, "Proliferative DR": 0.02},
        predicted_grade=2,
        predicted_grade_label="Moderate",
        model_version="efficientnet-v1",
    ))

    assert result.trust_category == "INSUFFICIENT_EVIDENCE"
    payload = result.to_dict()
    assert payload["recommended_safe_action"] == "PROFESSIONAL_REVIEW_RECOMMENDED"
    assert payload["evidence_status"] == "UNAVAILABLE"
    assert payload["ood_status"] == "UNAVAILABLE"
    assert payload["provenance"]["clinical_validation_claim"] is False
