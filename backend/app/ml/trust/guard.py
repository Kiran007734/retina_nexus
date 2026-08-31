"""Transparent, versioned RetinaGuard decision and evidence fusion engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.ml.trust.calibration import TemperatureScaler
from app.ml.trust.disagreement import calculate_model_disagreement
from app.ml.trust.ood import FeatureDistributionMonitor
from app.ml.trust.uncertainty import UncertaintyEstimator

logger = logging.getLogger(__name__)


class ReliabilityState:
    """Operational reliability states, never diagnostic labels."""

    TRUSTED = "TRUSTED"
    REVIEW_RECOMMENDED = "REVIEW_RECOMMENDED"
    UNRELIABLE = "UNRELIABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SafeAction:
    AUTOMATED_RESULT_AVAILABLE = "AUTOMATED_RESULT_AVAILABLE"
    PROFESSIONAL_REVIEW_RECOMMENDED = "PROFESSIONAL_REVIEW_RECOMMENDED"
    AUTOMATED_INTERPRETATION_UNRELIABLE = "AUTOMATED_INTERPRETATION_UNRELIABLE"
    IMAGE_RECAPTURE_RECOMMENDED = "IMAGE_RECAPTURE_RECOMMENDED"


def _clamp(value: float) -> float:
    return round(float(max(0.0, min(1.0, value))), 6)


def derive_lesion_evidence_strength(evidence: Any) -> float | None:
    """Summarize available lesion-module confidence without making a finding."""
    modules = evidence.modules if hasattr(evidence, "modules") else (evidence or {}).get("modules", {})
    strengths: list[float] = []
    for name, module in modules.items():
        if not module.get("supported"):
            continue
        if module.get("category") != "lesion_detection" and name != "exudate_segmentation":
            continue
        if module.get("confidence") is not None:
            strengths.append(float(module["confidence"]))
    return _clamp(sum(strengths) / len(strengths)) if strengths else None


def derive_vessel_evidence_status(evidence: Any) -> str:
    """Classify vessel provenance for audit/display without changing trust score."""
    modules = evidence.modules if hasattr(evidence, "modules") else (evidence or {}).get("modules", {})
    vessel = modules.get("vessel_segmentation") or {}
    if not vessel.get("supported"):
        return "UNAVAILABLE"
    if vessel.get("status") == "model_inference":
        return "REAL_MODEL_EVIDENCE"
    if vessel.get("status") == "experimental_heuristic":
        return "EXPERIMENTAL_BASELINE"
    return "UNAVAILABLE"


@dataclass
class RetinaGuardInputs:
    quality_score: float | None = None
    raw_confidence: float | None = None
    calibrated_confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    classifier_logits: list[float] | None = None
    mc_probabilities: list[list[float]] | None = None
    mc_error: str | None = None
    model_predictions: list[dict[str, Any]] = field(default_factory=list)
    lesion_evidence_strength: float | None = None
    vessel_evidence_status: str = "UNAVAILABLE"
    attention_lesion_agreement: dict[str, Any] | None = None
    explanation_stability: dict[str, Any] | None = None
    ood: dict[str, Any] | None = None
    quality_feature_vector: dict[str, float] = field(default_factory=dict)
    predicted_grade: int | None = None
    predicted_grade_label: str | None = None
    referable_dr: bool | None = None
    model_version: str | None = None


@dataclass
class RetinaGuardResult:
    trust_score: float
    trust_category: str
    contributing_factors: list[dict[str, Any]]
    risk_flags: list[dict[str, str]]
    recommended_action: str
    calibration: dict[str, Any]
    uncertainty: dict[str, Any]
    model_disagreement: dict[str, Any]
    ood: dict[str, Any]
    signal_snapshot: dict[str, Any]
    configuration: dict[str, Any]
    reason_summary: list[str]

    def to_dict(self) -> dict[str, Any]:
        evidence_status = self.signal_snapshot.get("evidence_status", "UNAVAILABLE")
        explanation_status = self.signal_snapshot.get("explanation_status", "UNAVAILABLE")
        ood_status = self.ood.get("status", "UNAVAILABLE")
        safe_action = self.configuration.get("safe_action", SafeAction.PROFESSIONAL_REVIEW_RECOMMENDED)
        provenance = {
            "engine_version": self.configuration.get("version"),
            "calibration_version": self.configuration.get("calibration_version"),
            "uncertainty_method": self.uncertainty.get("method"),
            "ood_method": self.ood.get("method"),
            "evidence_status": evidence_status,
            "explanation_status": explanation_status,
            "clinical_validation_claim": False,
            "note": "RetinaGuard is an engineering reliability assessment. TRUSTED means no configured major warning was detected; it does not mean the prediction is correct.",
        }
        return {
            "trust_score": self.trust_score, "trust_category": self.trust_category,
            "reliability_score": self.trust_score, "reliability_state": self.trust_category,
            "contributing_factors": self.contributing_factors, "risk_flags": self.risk_flags,
            "recommended_action": self.recommended_action, "calibration": self.calibration,
            "uncertainty": self.uncertainty, "model_disagreement": self.model_disagreement,
            "ood": self.ood, "signal_snapshot": self.signal_snapshot,
            "configuration": self.configuration, "reason_summary": self.reason_summary,
            "confidence": {"raw": self.signal_snapshot.get("raw_confidence"), "calibrated": self.signal_snapshot.get("calibrated_confidence")},
            "image_quality_status": self.signal_snapshot.get("image_quality_status", "UNAVAILABLE"),
            "ood_status": ood_status,
            "evidence_status": evidence_status,
            "explanation_status": explanation_status,
            "warnings": self.risk_flags,
            "reasons": self.reason_summary,
            "recommended_safe_action": safe_action,
            "provenance": provenance,
        }


class RetinaGuardEngine:
    """Fuse measurable signals with disclosed weights and explicit safety rules."""

    VERSION = "retinaguard-v2-reliability"

    def __init__(
        self,
        version: str = VERSION,
        calibrator: TemperatureScaler | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        ood_monitor: FeatureDistributionMonitor | None = None,
        weights: dict[str, float] | None = None,
        missing_signal_score: float = 0.25,
        trusted_threshold: float = 0.75,
        unreliable_threshold: float = 0.45,
        mc_dropout_enabled: bool = False,
        mc_dropout_samples: int = 8,
    ):
        self.version = version
        self.calibrator = calibrator or TemperatureScaler()
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
        self.ood_monitor = ood_monitor or FeatureDistributionMonitor()
        self.weights = weights or {"quality": 0.20, "calibrated_confidence": 0.20, "uncertainty": 0.15, "model_agreement": 0.10, "lesion_evidence": 0.10, "attention_lesion_agreement": 0.15, "explanation_stability": 0.05, "ood": 0.05}
        if any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("RetinaGuard weights must be non-negative and non-zero")
        total = sum(self.weights.values())
        self.weights = {key: value / total for key, value in self.weights.items()}
        self.missing_signal_score = _clamp(missing_signal_score)
        self.trusted_threshold = trusted_threshold
        self.unreliable_threshold = unreliable_threshold
        self.mc_dropout_enabled = mc_dropout_enabled
        self.mc_dropout_samples = max(2, min(30, int(mc_dropout_samples)))

    async def evaluate_async(self, inputs: RetinaGuardInputs, image_bytes: bytes | None = None, mc_dropout_provider: Any = None) -> RetinaGuardResult:
        """Optionally obtain MC-dropout samples before running sync fusion."""
        if self.mc_dropout_enabled:
            if image_bytes is None or mc_dropout_provider is None or not hasattr(mc_dropout_provider, "predict_with_dropout"):
                inputs.mc_error = "MC-dropout is enabled but no compatible provider was supplied."
            else:
                try:
                    inputs.mc_probabilities = await mc_dropout_provider.predict_with_dropout(image_bytes, self.mc_dropout_samples)
                except Exception as exc:
                    inputs.mc_error = f"MC-dropout provider failed safely: {exc}"
        return self.evaluate(inputs)

    def evaluate(self, inputs: RetinaGuardInputs) -> RetinaGuardResult:
        calibration_result = self.calibrator.calibrate(inputs.probabilities, inputs.classifier_logits) if inputs.probabilities else None
        calibration = calibration_result.to_dict() if calibration_result is not None else {"calibrated_confidence": inputs.calibrated_confidence, "status": "UNAVAILABLE", "reason": "Classifier probabilities were not available."}
        calibrated_confidence = inputs.calibrated_confidence if inputs.calibrated_confidence is not None else calibration.get("calibrated_confidence")
        uncertainty = self.uncertainty_estimator.estimate(inputs.probabilities, inputs.mc_probabilities, inputs.mc_error) if inputs.probabilities else {"score": None, "status": "UNAVAILABLE", "reason": "Classifier probabilities were not available."}
        disagreement_inputs = list(inputs.model_predictions)
        if inputs.predicted_grade is not None and not any(item.get("model_version") == inputs.model_version for item in disagreement_inputs):
            disagreement_inputs.insert(0, {"model_version": inputs.model_version or "primary", "predicted_grade": inputs.predicted_grade, "predicted_grade_label": inputs.predicted_grade_label})
        disagreement = calculate_model_disagreement(disagreement_inputs)
        ood = inputs.ood or self.ood_monitor.evaluate(inputs.quality_feature_vector)
        agreement_score = (inputs.attention_lesion_agreement or {}).get("score")
        stability_score = self._stability_score(inputs.explanation_stability)
        factors = {
            "quality": inputs.quality_score,
            "calibrated_confidence": calibrated_confidence,
            "uncertainty": None if uncertainty.get("score") is None else 1.0 - float(uncertainty["score"]),
            "model_agreement": None if disagreement.get("agreement") is None else float(disagreement["agreement"]),
            "lesion_evidence": inputs.lesion_evidence_strength,
            "attention_lesion_agreement": agreement_score,
            "explanation_stability": stability_score,
            "ood": ood.get("score"),
        }
        factor_explanations = {
            "quality": "Image Trust Gate quality score.", "calibrated_confidence": "Temperature-scaled classifier confidence.",
            "uncertainty": "Inverse predictive uncertainty; higher means less uncertainty.", "model_agreement": "Inverse ensemble disagreement; higher means models agree.",
            "lesion_evidence": "Strength of available supporting lesion modules.", "attention_lesion_agreement": "Overlap between classifier attention and supported lesion regions.",
            "explanation_stability": "Combined prediction and Grad-CAM stability when perturbation tests ran.", "ood": "In-distribution score from the configured reference feature distribution.",
        }
        contributing: list[dict[str, Any]] = []
        weighted_total = 0.0
        for name, weight in self.weights.items():
            value = factors.get(name)
            available = value is not None
            used = _clamp(float(value)) if available else self.missing_signal_score
            contribution = weight * used
            weighted_total += contribution
            contributing.append({"factor": name, "score": used, "raw_value": value, "weight": round(weight, 6), "contribution": round(contribution, 6), "status": "available" if available else "missing_or_not_run", "explanation": factor_explanations[name]})
        trust_score = _clamp(weighted_total)
        risk_flags = self._risk_flags(inputs, calibrated_confidence, uncertainty, disagreement, agreement_score, stability_score, ood, factors)
        core_missing = [name for name in ("quality", "calibrated_confidence", "uncertainty") if factors.get(name) is None]
        evidence_missing = [name for name in ("attention_lesion_agreement", "ood") if factors.get(name) is None]
        required_missing = [*core_missing, *evidence_missing]
        hard_unreliable = any(flag["severity"] == "high" for flag in risk_flags)
        if hard_unreliable or trust_score < self.unreliable_threshold:
            category = ReliabilityState.UNRELIABLE
        elif core_missing or evidence_missing:
            category = ReliabilityState.INSUFFICIENT_EVIDENCE
        elif trust_score >= self.trusted_threshold and not risk_flags:
            category = ReliabilityState.TRUSTED
        else:
            category = ReliabilityState.REVIEW_RECOMMENDED
        reason_summary = [flag["reason"] for flag in risk_flags]
        if required_missing:
            reason_summary.append("Missing or not-run signals prevent a fully trusted decision: " + ", ".join(required_missing) + ".")
        if not reason_summary:
            reason_summary.append("All configured self-check signals are within the trusted operating thresholds.")
        action, safe_action = self._recommended_action(category, inputs, risk_flags)
        signal_snapshot = {
            "quality_score": inputs.quality_score, "raw_confidence": inputs.raw_confidence,
            "calibrated_confidence": calibrated_confidence, "uncertainty_score": uncertainty.get("score"),
            "lesion_evidence_strength": inputs.lesion_evidence_strength, "attention_lesion_agreement": inputs.attention_lesion_agreement,
            "explanation_stability": inputs.explanation_stability, "ood": ood,
            "vessel_evidence_status": inputs.vessel_evidence_status or "UNAVAILABLE",
            "image_quality_status": "AVAILABLE" if inputs.quality_score is not None else "UNAVAILABLE",
            "evidence_status": "AVAILABLE" if inputs.attention_lesion_agreement and agreement_score is not None else "UNAVAILABLE",
            "explanation_status": "AVAILABLE" if inputs.explanation_stability and inputs.explanation_stability.get("status") == "COMPLETED" else "LIMITED" if inputs.explanation_stability else "UNAVAILABLE",
        }
        configuration = {"version": self.version, "weights": self.weights, "missing_signal_score": self.missing_signal_score, "trusted_threshold": self.trusted_threshold, "unreliable_threshold": self.unreliable_threshold, "calibration_version": self.calibrator.version, "mc_dropout_enabled": self.mc_dropout_enabled, "mc_dropout_samples": self.mc_dropout_samples, "vessel_evidence_policy": "provenance_audit_only; no independent trust-score weight", "decision_policy_version": "retinaguard-state-policy-v2", "safe_action": safe_action, "clinical_validation_claim": False}
        result = RetinaGuardResult(trust_score, category, contributing, risk_flags, action, calibration, uncertainty, disagreement, ood, signal_snapshot, configuration, reason_summary)
        logger.info("retinaguard.score", extra={"trust_score": result.trust_score, "trust_category": result.trust_category, "configuration_version": self.version, "risk_flags": result.risk_flags})
        return result

    async def score(self, quality: Any, classification: Any, evidence: dict[str, Any]) -> float | None:
        """Compatibility method for the existing non-HTTP orchestrator."""
        result = self.evaluate(RetinaGuardInputs(
            quality_score=getattr(quality, "score", None), raw_confidence=getattr(classification, "confidence", None),
            calibrated_confidence=getattr(classification, "calibrated_confidence", None), predicted_grade=getattr(classification, "dr_grade", None),
            model_version=getattr(classification, "model_version", None), lesion_evidence_strength=evidence.get("lesion_evidence_strength") if isinstance(evidence, dict) else None,
        ))
        return result.trust_score

    @staticmethod
    def _stability_score(stability: dict[str, Any] | None) -> float | None:
        if not stability or stability.get("status") != "COMPLETED":
            return None
        values = [float(stability[key]) for key in ("prediction_stability", "grad_cam_stability") if stability.get(key) is not None]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _risk_flags(inputs: RetinaGuardInputs, calibrated: float | None, uncertainty: dict[str, Any], disagreement: dict[str, Any], agreement: float | None, stability: float | None, ood: dict[str, Any], factors: dict[str, float | None]) -> list[dict[str, str]]:
        flags: list[dict[str, str]] = []
        def add(code: str, severity: str, reason: str) -> None:
            flags.append({"code": code, "severity": severity, "reason": reason})
        if factors["quality"] is not None and factors["quality"] < 0.45:
            add("low_image_quality", "high", "Image quality is below the reliable operating threshold.")
        elif factors["quality"] is None:
            add("quality_not_available", "high", "Image quality was not available to RetinaGuard.")
        if calibrated is not None and calibrated < 0.35:
            add("low_calibrated_confidence", "high", "Calibrated model confidence is low.")
        elif calibrated is None:
            add("calibration_not_available", "high", "Calibrated confidence is unavailable; raw confidence is not used as a trust guarantee.")
        if uncertainty.get("score") is not None and uncertainty["score"] > 0.70:
            add("high_prediction_uncertainty", "high", "Predictive entropy and probability-margin uncertainty are high.")
        if disagreement.get("disagreement") is not None and disagreement["disagreement"] >= 0.50:
            add("high_model_disagreement", "high", "Multiple model predictions disagree materially in severity.")
        elif disagreement.get("disagreement") is None and len(inputs.model_predictions) > 0:
            add("model_disagreement_not_available", "medium", "Additional model predictions were supplied but could not be compared.")
        if agreement is not None and agreement < 0.30:
            add("low_attention_evidence_agreement", "high", "Classifier attention has low overlap with supported lesion evidence.")
        elif agreement is None:
            add("attention_evidence_not_available", "medium", "Attention-lesion agreement has not been calculated from supported evidence.")
        if stability is None:
            add("explanation_stability_not_run", "medium", "Explanation stability testing was not run for this real-time request.")
        elif stability < 0.40:
            add("low_explanation_stability", "high", "Controlled perturbations produced unstable prediction or Grad-CAM behavior.")
        if ood.get("status") == "SHIFTED":
            add("distribution_shift", "high", "Image quality features differ substantially from the configured reference distribution.")
        elif ood.get("score") is None:
            add("ood_not_available", "medium", "No authorized reference distribution is configured for OOD monitoring.")
        return flags

    @staticmethod
    def _recommended_action(category: str, inputs: RetinaGuardInputs, flags: list[dict[str, str]]) -> tuple[str, str]:
        if category == ReliabilityState.TRUSTED:
            return "AI triage may proceed to the configured workflow with human oversight.", SafeAction.AUTOMATED_RESULT_AVAILABLE
        if category == ReliabilityState.UNRELIABLE:
            if any(flag["code"] == "low_image_quality" for flag in flags):
                return "Recapture the fundus image; if the issue persists, route to specialist review.", SafeAction.IMAGE_RECAPTURE_RECOMMENDED
            return "Do not rely on automated triage; require specialist or human review.", SafeAction.AUTOMATED_INTERPRETATION_UNRELIABLE
        if category == ReliabilityState.INSUFFICIENT_EVIDENCE:
            return "Automated result requires professional review because required reliability evidence is unavailable.", SafeAction.PROFESSIONAL_REVIEW_RECOMMENDED
        return "Human review is required before relying on automated triage.", SafeAction.PROFESSIONAL_REVIEW_RECOMMENDED
