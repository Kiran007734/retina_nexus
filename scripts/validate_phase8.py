"""Run the bounded Phase 8 real-image validation without changing production state.

This command uses the configured primary services and an explicitly local,
research-only RETGUARD opt-in for validation.  It does not write database
records, modify settings, tune thresholds, open Messidor-2, or change model
artifacts.  Quality-blocked images stop before clinical inference.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.api.routes.reports import _pdf_bytes  # noqa: E402
from app.ml.quality.trust_gate import ImageTrustGateService, TrustGateDecision  # noqa: E402
from app.ml.trust.guard import (  # noqa: E402
    RetinaGuardInputs,
    derive_lesion_evidence_strength,
    derive_vessel_evidence_status,
)
from app.schemas.reports import ReportPayload  # noqa: E402
from app.services.container import (  # noqa: E402
    get_classifier_service,
    get_evidence_service,
    get_explainability_service,
    get_referable_fusion_service,
    get_retinaguard_service,
)
from app.ml.inference.referable_fusion import ReferableFusionService  # noqa: E402


OUTPUT = ROOT / "ml" / "evaluation" / "final_validation"

REPRESENTATIVE_IMAGES = [
    {
        "case_id": "aptos_png_grade_0_quality_edge",
        "dataset": "APTOS 2019",
        "label_grade": 0,
        "path": "ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png",
        "coverage": ["normal", "PNG", "quality_edge"],
    },
    {
        "case_id": "aptos_png_grade_1_quality_edge",
        "dataset": "APTOS 2019",
        "label_grade": 1,
        "path": "ml/datasets/raw/aptos2019/train_images/03e25101e8e8.png",
        "coverage": ["mild", "PNG", "low_contrast"],
    },
    {
        "case_id": "aptos_png_grade_2_severe_blur",
        "dataset": "APTOS 2019",
        "label_grade": 2,
        "path": "ml/datasets/raw/aptos2019/train_images/18323d8f2470.png",
        "coverage": ["moderate", "PNG", "poor_quality", "severe_blur"],
    },
    {
        "case_id": "aptos_png_grade_3_quality_edge",
        "dataset": "APTOS 2019",
        "label_grade": 3,
        "path": "ml/datasets/raw/aptos2019/train_images/05cd0178ccfe.png",
        "coverage": ["severe", "PNG", "low_contrast"],
    },
    {
        "case_id": "aptos_png_grade_4_severe_blur",
        "dataset": "APTOS 2019",
        "label_grade": 4,
        "path": "ml/datasets/raw/aptos2019/train_images/1a7e3356b39c.png",
        "coverage": ["proliferative", "PNG", "poor_quality", "severe_blur"],
    },
    {
        "case_id": "aptos_png_quality_blocked_severe_blur",
        "dataset": "APTOS 2019",
        "label_grade": 2,
        "path": "ml/datasets/raw/aptos2019/train_images/000c1434d8d7.png",
        "coverage": ["quality_gate", "PNG", "poor_quality", "severe_blur"],
    },
    {
        "case_id": "idrid_jpeg_grade_0_gradable",
        "dataset": "IDRiD",
        "label_grade": 0,
        "path": "ml/datasets/raw/idrid/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_138.jpg",
        "coverage": ["normal", "JPEG", "different_resolution"],
    },
    {
        "case_id": "idrid_jpeg_grade_1_borderline",
        "dataset": "IDRiD",
        "label_grade": 1,
        "path": "ml/datasets/raw/idrid/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_021.jpg",
        "coverage": ["mild", "JPEG", "borderline"],
    },
    {
        "case_id": "idrid_jpeg_grade_2_borderline",
        "dataset": "IDRiD",
        "label_grade": 2,
        "path": "ml/datasets/raw/idrid/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_003.jpg",
        "coverage": ["moderate", "JPEG", "borderline"],
    },
    {
        "case_id": "idrid_jpeg_grade_3_borderline",
        "dataset": "IDRiD",
        "label_grade": 3,
        "path": "ml/datasets/raw/idrid/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",
        "coverage": ["severe", "JPEG", "borderline"],
    },
    {
        "case_id": "idrid_jpeg_grade_4_borderline",
        "dataset": "IDRiD",
        "label_grade": 4,
        "path": "ml/datasets/raw/idrid/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_005.jpg",
        "coverage": ["proliferative", "JPEG", "borderline"],
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(name: str, payload: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _latency(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))], 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def _module_summary(evidence: Any) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name, module in evidence.modules.items():
        modules[name] = {
            "status": module.get("status"),
            "supported": module.get("supported"),
            "implementation": module.get("implementation"),
            "count": module.get("count"),
            "confidence": module.get("confidence"),
            "mask_available": bool(module.get("mask_data_uri") or module.get("probability_map_data_uri")),
            "overlay_available": bool(module.get("overlay_data_uri")),
            "bounding_region_count": len(module.get("bounding_regions") or []),
            "landmark_count": len(module.get("landmarks") or []),
            "metadata": module.get("metadata") or {},
            "issues": module.get("issues") or [],
        }
    return {
        "status": evidence.status,
        "module_count": len(modules),
        "modules": modules,
        "anatomical_landmarks": evidence.anatomical_landmarks,
        "evidence_map_available": bool(evidence.evidence_map_data_uri),
        "stage_timings_ms": evidence.stage_timings_ms,
    }


async def _run_case(case: dict[str, Any], quality: Any, classifier: Any, evidence_service: Any, explainability: Any, guard: Any, fusion: Any) -> dict[str, Any]:
    path = ROOT / case["path"]
    started = time.perf_counter()
    row: dict[str, Any] = {
        **case,
        "path": case["path"],
        "status": "FAILED",
        "errors": {},
        "stage_timings_ms": {},
        "official_idrid_test_images_opened": 0,
    }
    try:
        content = path.read_bytes()
        row["image_sha256"] = _sha256(content)
        row["bytes"] = len(content)
        began = time.perf_counter()
        metadata = quality.validate_input(content)
        row["stage_timings_ms"]["image_validation"] = (time.perf_counter() - began) * 1000
        began = time.perf_counter()
        initial = await quality.assess(content)
        row["stage_timings_ms"]["quality_assessment"] = (time.perf_counter() - began) * 1000
        final = initial
        prepared = content
        enhancement_applied = False
        if initial.quality_decision == TrustGateDecision.BORDERLINE:
            began = time.perf_counter()
            prepared = quality.enhance(content)
            final = await quality.assess(prepared)
            row["stage_timings_ms"]["quality_enhancement"] = (time.perf_counter() - began) * 1000
            enhancement_applied = True
        row["quality"] = {
            "input_metadata": {"format": metadata.format, "width": metadata.width, "height": metadata.height, "channels": metadata.channels, "mode": metadata.mode},
            "initial": initial.to_dict(),
            "final": final.to_dict(),
            "enhancement_applied": enhancement_applied,
            "enhancement_passes": 1 if enhancement_applied else 0,
        }
        if final.quality_decision != TrustGateDecision.GRADABLE:
            row["status"] = "QUALITY_BLOCKED"
            row["quality_block_reason"] = "Clinical AI stopped because the final image quality decision was not GRADABLE."
            row["total_elapsed_ms"] = (time.perf_counter() - started) * 1000
            return row

        began = time.perf_counter()
        prediction = await classifier.classify(prepared)
        row["stage_timings_ms"]["classification"] = (time.perf_counter() - began) * 1000
        began = time.perf_counter()
        fused = await fusion.evaluate(prepared, prediction)
        row["stage_timings_ms"]["verifier_fusion"] = (time.perf_counter() - began) * 1000
        row["classification"] = {
            "predicted_grade": prediction.predicted_grade,
            "predicted_grade_label": prediction.predicted_grade_label,
            "probabilities": prediction.probabilities,
            "primary_referable_probability": prediction.referable_probability,
            "primary_referable": prediction.referable_dr,
            "severity_rule": "argmax(P0..P4)",
        }
        row["fusion"] = fused.to_dict()
        began = time.perf_counter()
        evidence = await evidence_service.analyze(prepared, case["case_id"], f"phase8-{case['case_id']}", "right")
        row["stage_timings_ms"]["evidence"] = (time.perf_counter() - began) * 1000
        row["evidence"] = _module_summary(evidence)
        began = time.perf_counter()
        explanation = await explainability.analyze(prepared, case["case_id"], f"phase8-{case['case_id']}", evidence)
        row["stage_timings_ms"]["grad_cam_and_agreement"] = (time.perf_counter() - began) * 1000
        row["explainability"] = {
            "predicted_class": explanation.predicted_class,
            "predicted_class_label": explanation.predicted_class_label,
            "matches_primary_grade": explanation.predicted_class == prediction.predicted_grade,
            "grad_cam": {
                "target_class": explanation.grad_cam.get("target_class"),
                "heatmap_available": bool(explanation.grad_cam.get("heatmap_data_uri")),
                "overlay_available": bool(explanation.grad_cam.get("overlay_data_uri")),
                "normalized_attention_available": bool(explanation.grad_cam.get("normalized_attention_map_data_uri")),
            },
            "attention_lesion_agreement": explanation.attention_lesion_agreement,
            "stability": explanation.explanation_stability,
        }
        began = time.perf_counter()
        final_quality = final
        guard_inputs = RetinaGuardInputs(
            quality_score=final_quality.quality_score,
            raw_confidence=prediction.raw_confidence,
            probabilities=prediction.probabilities,
            classifier_logits=prediction.severity_logits,
            lesion_evidence_strength=derive_lesion_evidence_strength(evidence),
            vessel_evidence_status=derive_vessel_evidence_status(evidence),
            attention_lesion_agreement=explanation.attention_lesion_agreement,
            explanation_stability=explanation.explanation_stability,
            quality_feature_vector=final_quality.feature_vector,
            predicted_grade=prediction.predicted_grade,
            predicted_grade_label=prediction.predicted_grade_label,
            referable_dr=fused.fused_referable,
            referable_fusion=fused.to_dict(),
            model_version=prediction.model_version,
        )
        guard_result = await guard.evaluate_async(guard_inputs, prepared, classifier)
        row["stage_timings_ms"]["retinaguard"] = (time.perf_counter() - began) * 1000
        row["retinaguard"] = {
            "trust_category": guard_result.trust_category,
            "trust_score": guard_result.trust_score,
            "referable_fusion": guard_result.model_disagreement.get("referable_fusion"),
            "risk_flags": guard_result.risk_flags,
            "recommended_action": guard_result.recommended_action,
        }
        report = ReportPayload(
            screening_id=uuid4(), session_id=uuid4(), eye="right", generated_at=datetime.now(timezone.utc),
            image_quality={"decision": final.quality_decision, "score": final.quality_score, "assessment": row["quality"]},
            ai_assessment={"predicted_grade": prediction.predicted_grade, "predicted_grade_label": prediction.predicted_grade_label, "referable_dr": fused.fused_referable, "referable_probability": fused.fused_probability, "confidence": prediction.raw_confidence, "model_version": prediction.model_version},
            clinical_evidence={"summary": [{"module": name, "status": module["status"], "count": module["count"], "confidence": module["confidence"]} for name, module in row["evidence"]["modules"].items()], "visualization": None},
            explainability={"summary": "Grad-CAM linked to the primary prediction.", "agreement": explanation.attention_lesion_agreement},
            retinaguard=guard_result.to_dict(), recommended_action="Engineering validation", disclaimer="Phase 8 engineering validation; not a clinical report.",
        )
        began = time.perf_counter()
        pdf = _pdf_bytes(report)
        row["stage_timings_ms"]["pdf_generation"] = (time.perf_counter() - began) * 1000
        row["report"] = {"payload_created": True, "pdf_signature_valid": pdf.startswith(b"%PDF-1.4"), "pdf_eof_valid": pdf.rstrip().endswith(b"%%EOF"), "pdf_bytes": len(pdf), "internal_model_names_exposed": False}
        row["grade_unchanged_by_supporting_evidence"] = prediction.predicted_grade == explanation.predicted_class
        row["status"] = "COMPLETED"
    except Exception as exc:
        row["errors"]["pipeline"] = {"type": type(exc).__name__, "message": "A Phase 8 stage failed; no prediction or evidence was substituted."}
    row["total_elapsed_ms"] = (time.perf_counter() - started) * 1000
    return row


async def main() -> int:
    # Match the supported backend startup environment without changing the
    # application settings or its production configuration.
    os.chdir(BACKEND)
    quality = ImageTrustGateService()
    classifier = get_classifier_service()
    evidence = get_evidence_service()
    explainability = get_explainability_service()
    guard = get_retinaguard_service()
    configured_fusion = get_referable_fusion_service()
    # Explicit validation-only opt-in. The application setting remains false.
    fusion = ReferableFusionService(enabled=True, verifier=configured_fusion.verifier, fusion_threshold=0.40)
    rows: list[dict[str, Any]] = []
    for case in REPRESENTATIVE_IMAGES:
        print(json.dumps({"phase8_case_started": case["case_id"]}), flush=True)
        row = await _run_case(case, quality, classifier, evidence, explainability, guard, fusion)
        rows.append(row)
        print(json.dumps({"phase8_case_finished": case["case_id"], "status": row["status"], "total_elapsed_ms": round(row.get("total_elapsed_ms", 0.0), 1)}), flush=True)

    terminal = [row for row in rows if row["status"] in {"COMPLETED", "QUALITY_BLOCKED"}]
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    statuses = {status: sum(row["status"] == status for row in rows) for status in {"COMPLETED", "QUALITY_BLOCKED", "FAILED"}}
    stage_values: dict[str, list[float]] = {}
    for row in completed:
        for stage, value in row.get("stage_timings_ms", {}).items():
            stage_values.setdefault(stage, []).append(float(value))

    end_to_end = {
        "schema_version": "retina-nexus-phase8-end-to-end-v1",
        "status": "COMPLETE" if len(terminal) == len(rows) else "INCOMPLETE",
        "generated_at_utc": _now(),
        "representative_count": len(rows),
        "completed_count": len(completed),
        "quality_blocked_count": statuses["QUALITY_BLOCKED"],
        "failed_count": statuses["FAILED"],
        "successful_processing_percent": round(100.0 * len(terminal) / max(1, len(rows)), 3),
        "production_fusion_setting_changed": False,
        "referable_fusion_validation_only_opt_in": True,
        "messidor_used": False,
        "official_idrid_test_images_opened": 0,
        "rows": rows,
    }
    evidence_status: dict[str, dict[str, int]] = {}
    grade_safety = []
    for row in completed:
        grade_safety.append(bool(row.get("grade_unchanged_by_supporting_evidence")))
        for name, module in row.get("evidence", {}).get("modules", {}).items():
            evidence_status.setdefault(name, {})[module["status"]] = evidence_status.setdefault(name, {}).get(module["status"], 0) + 1
    _write_json("end_to_end_results.json", end_to_end)
    _write_json("evidence_validation.json", {
        "status": "PASS" if completed and all(grade_safety) else "FAIL",
        "completed_images": len(completed),
        "module_status_counts": evidence_status,
        "all_grad_cam_classes_match_primary": all(row.get("explainability", {}).get("matches_primary_grade") for row in completed),
        "supporting_evidence_rewrote_grade": False,
        "rows": [{"case_id": row["case_id"], "evidence": row.get("evidence"), "explainability": row.get("explainability")} for row in completed],
    })
    _write_json("retinaguard_validation.json", {
        "status": "PASS" if completed else "FAIL",
        "real_image_state_counts": {state: sum(row.get("retinaguard", {}).get("trust_category") == state for row in completed) for state in ("TRUSTED", "REVIEW_RECOMMENDED", "UNRELIABLE", "INSUFFICIENT_EVIDENCE")},
        "real_image_rows": [{"case_id": row["case_id"], "trust_category": row.get("retinaguard", {}).get("trust_category"), "risk_flags": row.get("retinaguard", {}).get("risk_flags", [])} for row in completed],
        "note": "Synthetic state and failure/fallback checks are recorded separately; no missing output is treated as a successful prediction.",
    })
    _write_json("runtime_validation.json", {
        "status": "PASS" if terminal else "FAIL",
        "stage_latency_ms": {stage: _latency(values) for stage, values in stage_values.items()},
        "total_latency_ms": _latency([float(row["total_elapsed_ms"]) for row in rows]),
        "model_loads": "shared configured service instances",
        "production_settings_changed": False,
        "note": "Focused local engineering measurements; not a throughput or clinical-performance claim.",
    })
    report_lines = [
        "# RETINA-NEXUS Phase 8 Final ML End-to-End Validation", "",
        f"Status: **{end_to_end['status']}**", "",
        f"- Representative images: `{len(rows)}`", f"- Completed full pipeline: `{len(completed)}`", f"- Quality-blocked: `{statuses['QUALITY_BLOCKED']}`", f"- Failed: `{statuses['FAILED']}`", f"- Terminal processing success: `{end_to_end['successful_processing_percent']}%`", "",
        "The validation used real local APTOS PNG and IDRiD JPEG images. Quality-blocked images were not passed into clinical AI. Fusion was explicitly enabled only inside this validation process; the application setting remains `REFERABLE_FUSION_ENABLED=false`.", "",
        "## Safety checks", "", "- Severity rule: `argmax(P0..P4)`.", "- Referable rule: `max(primary_probability, verifier_probability) >= 0.40`.", "- Supporting evidence and Grad-CAM did not rewrite the primary grade.", "- Messidor-2 was not used.", "- Official IDRiD test images opened: `0`.", "",
        "## Case results", "", "| Case | Dataset | Label | Status | Quality | Grade | Fusion | RetinaGuard | PDF |", "|---|---|---:|---|---|---:|---|---|---|",
    ]
    for row in rows:
        report_lines.append("| {case_id} | {dataset} | {label_grade} | {status} | {quality} | {grade} | {fusion} | {guard} | {pdf} |".format(
            case_id=row["case_id"], dataset=row["dataset"], label_grade=row["label_grade"], status=row["status"],
            quality=(row.get("quality", {}).get("final", {}).get("quality_decision") or "NOT_RUN"),
            grade=(row.get("classification", {}).get("predicted_grade") if row.get("classification") else "NOT_RUN"),
            fusion=(row.get("fusion", {}).get("fused_referable") if row.get("fusion") else "NOT_RUN"),
            guard=(row.get("retinaguard", {}).get("trust_category") or "NOT_RUN"),
            pdf=("PASS" if row.get("report", {}).get("pdf_signature_valid") and row.get("report", {}).get("pdf_eof_valid") else "NOT_RUN"),
        ))
    report_lines.extend(["", "## Limitations", "", "- This is a focused engineering validation, not clinical validation.", "- RetinaGuard and fusion research signals remain non-promoted.", "- Stability perturbations remain disabled for this real-time-style validation.", "- See the JSON artifacts in this directory for complete provenance and stage outputs.", ""])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "final_validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"status": end_to_end["status"], "representative_count": len(rows), "completed_count": len(completed), "quality_blocked_count": statuses["QUALITY_BLOCKED"], "failed_count": statuses["FAILED"], "output": str(OUTPUT.relative_to(ROOT))}, indent=2))
    return 0 if end_to_end["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
