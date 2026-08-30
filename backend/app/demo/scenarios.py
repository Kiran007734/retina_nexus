"""Controlled, synthetic demo scenarios for local product walkthroughs.

These fixtures are intentionally separate from clinical records and model
inference. They must never be used as validation data or production fallback
results.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "scenario_id": "image-1-trusted-refer",
        "image_label": "IMAGE 1",
        "title": "High-quality moderate DR",
        "summary": "A synthetic gradable case with strong supporting lesion evidence.",
        "quality": {
            "quality_decision": "GRADABLE",
            "quality_score": 0.94,
            "component_scores": {"focus": 0.96, "illumination": 0.92, "contrast": 0.93, "field_of_view": 0.95, "exposure": 0.94, "artifacts": 0.98},
            "enhancement_applied": False,
            "issues": [],
        },
        "classification": {
            "predicted_grade": 2,
            "predicted_grade_label": "Moderate",
            "probabilities": {"No DR": 0.03, "Mild": 0.08, "Moderate": 0.78, "Severe": 0.08, "Proliferative DR": 0.03},
            "referable_dr": True,
            "referable_probability": 0.89,
            "raw_confidence": 0.78,
            "model_version": "demo-classifier-v1",
        },
        "lesions": {
            "status": "demo-supported",
            "evidence_strength": 0.88,
            "summary": [{"lesion_type": "microaneurysm_detection", "count": 12, "confidence": 0.86}, {"lesion_type": "hemorrhage_detection", "count": 2, "confidence": 0.81}],
        },
        "explainability": {"attention_lesion_agreement": {"status": "HIGH AGREEMENT", "score": 0.87}, "explanation_stability": {"status": "COMPLETED", "prediction_stability": 0.96, "grad_cam_stability": 0.91}},
        "retinaguard": {"trust_score": 0.89, "trust_category": "TRUSTED", "reason_summary": ["Quality, confidence, evidence, and explanation signals are aligned."], "risk_flags": []},
        "triage": {"recommendation": "SPECIALIST_REVIEW_RECOMMENDED", "display_action": "REFER", "priority": "high"},
        "model_versions": {"dr_classifier": "demo-classifier-v1", "retinaguard": "demo-guard-v1", "preprocessing": "demo-preprocessing-v1"},
    },
    {
        "scenario_id": "image-2-uncertain-review",
        "image_label": "IMAGE 2",
        "title": "Borderline image with uncertainty",
        "summary": "A synthetic borderline case that improves after one controlled enhancement pass but remains uncertain.",
        "quality": {
            "quality_decision": "BORDERLINE",
            "quality_score": 0.74,
            "recheck_score": 0.82,
            "component_scores": {"focus": 0.63, "illumination": 0.75, "contrast": 0.68, "field_of_view": 0.88, "exposure": 0.79, "artifacts": 0.91},
            "enhancement_applied": True,
            "enhancement_passes": 1,
            "issues": [{"type": "low_contrast", "severity": "moderate", "message": "Tonal separation is limited before enhancement."}],
        },
        "classification": {
            "predicted_grade": 1,
            "predicted_grade_label": "Mild",
            "probabilities": {"No DR": 0.25, "Mild": 0.31, "Moderate": 0.24, "Severe": 0.12, "Proliferative DR": 0.08},
            "referable_dr": False,
            "referable_probability": 0.44,
            "raw_confidence": 0.31,
            "model_version": "demo-classifier-v1",
        },
        "lesions": {"status": "demo-supported", "evidence_strength": 0.42, "summary": [{"lesion_type": "microaneurysm_detection", "count": 3, "confidence": 0.48}]},
        "explainability": {"attention_lesion_agreement": {"status": "MODERATE AGREEMENT", "score": 0.43}, "explanation_stability": {"status": "COMPLETED", "prediction_stability": 0.67, "grad_cam_stability": 0.58}},
        "retinaguard": {"trust_score": 0.58, "trust_category": "UNCERTAIN", "reason_summary": ["Prediction uncertainty remains elevated after enhancement."], "risk_flags": [{"code": "high_prediction_uncertainty", "severity": "high"}]},
        "triage": {"recommendation": "HUMAN_REVIEW_REQUIRED", "display_action": "HUMAN REVIEW", "priority": "high"},
        "model_versions": {"dr_classifier": "demo-classifier-v1", "retinaguard": "demo-guard-v1", "preprocessing": "demo-preprocessing-v1"},
    },
    {
        "scenario_id": "image-3-ungradable-recapture",
        "image_label": "IMAGE 3",
        "title": "Poor quality severe blur",
        "summary": "A synthetic ungradable case blocked before clinical AI and routed to recapture guidance.",
        "quality": {
            "quality_decision": "UNGRADABLE",
            "quality_score": 0.18,
            "component_scores": {"focus": 0.05, "illumination": 0.63, "contrast": 0.22, "field_of_view": 0.71, "exposure": 0.82, "artifacts": 0.88},
            "enhancement_applied": False,
            "issues": [{"type": "severe_blur", "severity": "severe", "message": "The retinal image has insufficient focus detail.", "recommendation": "Stabilize the camera and refocus before recapturing."}],
        },
        "classification": None,
        "lesions": None,
        "explainability": None,
        "retinaguard": None,
        "triage": {"recommendation": "RECAPTURE_IMAGE", "display_action": "SMART RECAPTURE", "priority": "high", "reasons": ["Severe blur blocked the Image Trust Gate.", "Clinical AI was not started."]},
        "model_versions": {"preprocessing": "demo-preprocessing-v1"},
    },
)


def list_scenarios() -> list[dict[str, Any]]:
    return [
        {"scenario_id": item["scenario_id"], "image_label": item["image_label"], "title": item["title"], "summary": item["summary"], "expected_category": (item["retinaguard"] or {}).get("trust_category", "UNGRADABLE"), "expected_action": item["triage"]["display_action"]}
        for item in _SCENARIOS
    ]


def get_scenario(scenario_id: str) -> dict[str, Any] | None:
    for item in _SCENARIOS:
        if item["scenario_id"] == scenario_id:
            return deepcopy(item)
    return None
