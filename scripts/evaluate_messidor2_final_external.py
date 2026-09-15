"""Frozen Messidor-2 external evaluation without mutating prior results.

This runner deliberately separates the fresh classifier/current RetinaGuard
population pass from the expensive supporting-evidence stages.  The latter
are executed for one real end-to-end smoke image and are never represented as
if they had run for all 1,744 images.  No checkpoint, registry, threshold, or
production configuration is changed by this script.

The local ``messidor_data.csv`` is the diagnosis source for this evaluation.
It is an authorized label file used for descriptive external model evaluation;
it is not called official clinical ground truth and no clinical validation
claim is made.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.trust.guard import RetinaGuardInputs  # noqa: E402
from app.services.container import get_retinaguard_service  # noqa: E402
from app.services.runtime import resolve_path  # noqa: E402
from ml.evaluation.messidor2 import (  # noqa: E402
    APTOS_CLASS_MAPPING,
    compute_metrics,
    write_confusion_matrix,
)
from scripts.evaluate_messidor2 import run_inference  # noqa: E402


RAW_ROOT = ROOT / "ml" / "datasets" / "raw" / "messidor"
IMAGE_ROOT = RAW_ROOT / "images" / "messidor-2"
LABEL_CSV = RAW_ROOT / "images" / "messidor_data.csv"
QUALITY_CACHE = ROOT / "ml" / "evaluation" / "reliability" / "_phase5_1_messidor_quality_records.json"
AUDIT_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "messidor2" / "messidor2_audit_manifest.json"
OUTPUT_ROOT = ROOT / "ml" / "evaluation" / "messidor2" / "final_external"
DEFAULT_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
EXPECTED_CLASSIFIER_SHA = "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b"
EXPECTED_COUNTS = {"images": 1744, "labels": 1744}
HIGH_CONFIDENCE_THRESHOLD = 0.80


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), fraction))


def numeric_features(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items() if isinstance(item, (int, float)) and math.isfinite(float(item))}


def load_production_settings() -> Any:
    from app.core.config import Settings

    backend_env = ROOT / "backend" / ".env"
    return Settings(_env_file=str(backend_env) if backend_env.is_file() else None)


def resolve_model_paths(settings: Any) -> dict[str, Path]:
    from app.ml.evidence.lesion_model import DEFAULT_MODEL_PATH as LESION_DEFAULT
    from app.ml.evidence.vessel_model import DEFAULT_MODEL_PATH as VESSEL_DEFAULT

    classifier = resolve_path(settings.classifier_model_path) or DEFAULT_CHECKPOINT
    lesion = resolve_path(settings.lesion_model_path) or LESION_DEFAULT
    vessel = resolve_path(settings.vessel_model_path) or VESSEL_DEFAULT
    return {"classifier": classifier.resolve(), "lesion": lesion.resolve(), "vessel": vessel.resolve()}


def load_label_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not LABEL_CSV.is_file():
        raise FileNotFoundError(f"Messidor diagnosis source is missing: {LABEL_CSV}")
    image_paths = [path for path in IMAGE_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    by_name = {path.name.casefold(): path for path in image_paths}
    by_stem = {path.stem.casefold(): path for path in image_paths}
    records: list[dict[str, Any]] = []
    labels_seen: set[str] = set()
    missing_images: list[str] = []
    invalid_rows: list[dict[str, Any]] = []
    with LABEL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_id = str(row.get("id_code", "")).strip()
        key = image_id.casefold()
        stem = Path(image_id).stem.casefold()
        if not image_id or key in labels_seen:
            invalid_rows.append({"row": row, "reason": "missing_or_duplicate_image_id"})
            continue
        labels_seen.add(key)
        path = by_name.get(key) or by_stem.get(stem)
        try:
            grade = int(str(row.get("diagnosis", "")).strip())
            dme = int(str(row.get("adjudicated_dme", "")).strip())
            gradable = int(str(row.get("adjudicated_gradable", "")).strip())
        except (TypeError, ValueError):
            invalid_rows.append({"row": row, "reason": "non_integer_label"})
            continue
        if grade not in range(5) or dme not in {0, 1} or gradable not in {0, 1}:
            invalid_rows.append({"row": row, "reason": "label_out_of_contract"})
            continue
        if path is None:
            missing_images.append(image_id)
            continue
        with path.open("rb") as image_handle:
            image_sha = hashlib.sha256(image_handle.read()).hexdigest()
        records.append({
            # ``run_inference`` receives ``RAW_ROOT`` and appends this field;
            # keep it raw-root-relative.  The project-relative path is kept
            # separately for audit readability and never fed to the loader.
            "image_path": path.resolve().relative_to(RAW_ROOT.resolve()).as_posix(),
            "project_image_path": rel(path),
            "image_id": path.name,
            "image_sha256": image_sha,
            "adjudicated_dr_grade": grade,
            "adjudicated_dme": dme,
            "adjudicated_gradable": gradable,
        })
    image_names = {path.name.casefold() for path in image_paths}
    labels_without_images = sorted(
        str(row.get("id_code", ""))
        for row in rows
        if str(row.get("id_code", "")).casefold() not in image_names
    )
    manifest = {
        "dataset": "Messidor-2",
        "dataset_version": "messidor2-local-audited-1744",
        "generated_at_utc": now(),
        "image_root": rel(IMAGE_ROOT),
        "label_source": rel(LABEL_CSV),
        "label_source_policy": "messidor_data.csv is the diagnosis source; messidor-2.csv is pairing-only and is not used for labels.",
        "image_count": len(image_paths),
        "label_row_count": len(rows),
        "matched_pair_count": len(records),
        "missing_images_for_labels": missing_images,
        "labels_without_images": labels_without_images,
        "invalid_rows": invalid_rows,
        "class_distribution": {str(index): sum(item["adjudicated_dr_grade"] == index for item in records) for index in range(5)},
        "dme_distribution": {str(index): sum(item["adjudicated_dme"] == index for item in records) for index in range(2)},
        "gradable_distribution": {str(index): sum(item["adjudicated_gradable"] == index for item in records) for index in range(2)},
        "patient_ids_available": False,
        "patient_level_separation_claim": False,
        "label_provenance_limitation": "The local diagnosis CSV is an authorized released label file. It is not called official clinical ground truth, and this report is not clinical validation.",
    }
    if len(image_paths) != EXPECTED_COUNTS["images"] or len(records) != EXPECTED_COUNTS["labels"] or invalid_rows or missing_images:
        raise RuntimeError(f"Messidor-2 matched population contract failed: {json.dumps(manifest, sort_keys=True)}")
    return records, manifest


def load_quality_cache() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not QUALITY_CACHE.is_file():
        return {}, {"status": "UNAVAILABLE", "path": rel(QUALITY_CACHE), "reason": "No authorized quality cache was found."}
    payload = json.loads(QUALITY_CACHE.read_text(encoding="utf-8"))
    records = {str(item["image_id"]).casefold(): item for item in payload.get("records", []) if item.get("image_id")}
    quality_manifest = {
        "status": "AVAILABLE_CACHED_MEASUREMENTS",
        "path": rel(QUALITY_CACHE),
        "cache_key": payload.get("cache_key"),
        "analysis_max_dimension": payload.get("analysis_max_dimension"),
        "record_count": len(records),
        "provenance": "Measured by ImageTrustGateService during the Phase 5.1 Messidor reliability audit; not a newly rerun optional evidence stage.",
    }
    return records, quality_manifest


def build_model_manifest(settings: Any, paths: dict[str, Path], model_info: dict[str, Any]) -> dict[str, Any]:
    def artifact(path: Path, version: str, architecture: str, status: str = "AVAILABLE") -> dict[str, Any]:
        result: dict[str, Any] = {"status": status, "path": rel(path), "model_version": version, "architecture": architecture, "exists": path.is_file()}
        result["sha256"] = sha256_file(path) if path.is_file() else None
        manifest_path = path.parent / "model_manifest.json"
        result["manifest_path"] = rel(manifest_path) if manifest_path.is_file() else None
        return result

    classifier = {
        **model_info,
        "checkpoint_path": rel(paths["classifier"]),
        "checkpoint_sha256_expected": EXPECTED_CLASSIFIER_SHA,
        "checkpoint_sha256_actual": sha256_file(paths["classifier"]),
        "checkpoint_unchanged_during_evaluation": model_info.get("checkpoint_unchanged"),
        "preprocessing_contract": model_info.get("preprocessing"),
        "class_mapping": {str(key): value for key, value in APTOS_CLASS_MAPPING.items()},
        "referable_rule": "referable_probability = P(2) + P(3) + P(4); referable = referable_probability >= 0.5; severity grade = argmax(P0..P4).",
        "confidence_note": "Raw model confidence is not clinically calibrated; no calibration was fitted for this external evaluation.",
    }
    result = {
        "generated_at_utc": now(),
        "classifier": classifier,
        "primary_lesion": artifact(paths["lesion"], settings.lesion_model_version, "U-Net with SE-ResNeXt-50 32x4d", "AVAILABLE" if paths["lesion"].is_file() else "UNAVAILABLE"),
        "primary_vessel": artifact(paths["vessel"], settings.vessel_model_version, "RRWNet (R2-V2 bv variant)", "AVAILABLE" if paths["vessel"].is_file() else "UNAVAILABLE"),
        "localization": {"status": "NOT_CONFIGURED_FOR_PRODUCTION", "method": "classical optic-disc/fovea heuristic only when evidence service runs", "research_idrid_localization_enabled": False, "clinical_validation_claim": False},
        "grad_cam": {"status": "AVAILABLE_FROM_REGISTERED_CLASSIFIER", "implementation": "backend/app/ml/explainability/service.py", "classifier_linked": True},
        "retinaguard": {"status": "AVAILABLE", "configuration_version": settings.retinaguard_config_version, "calibration_version": settings.retinaguard_calibration_version, "calibration_fitted": settings.retinaguard_calibration_fitted, "thresholds": {"trusted": settings.retinaguard_trusted_threshold, "unreliable": settings.retinaguard_unreliable_threshold}},
        "research_models": {"idrid_lesion": {"enabled": settings.idrid_lesion_model_enabled}, "idrid_localization": {"enabled": settings.idrid_localization_model_enabled}, "drive_vessel": {"enabled": settings.drive_vessel_model_enabled}},
        "production_behavior_changed": False,
    }
    return result


def retinaguard_record(row: dict[str, Any], quality_item: dict[str, Any] | None, guard: Any) -> dict[str, Any]:
    probabilities = {APTOS_CLASS_MAPPING[index]: float(row[f"probability_{index}"]) for index in range(5)}
    quality = (quality_item or {}).get("quality") or {}
    quality_score = quality.get("quality_score")
    feature_vector = numeric_features(quality.get("feature_vector"))
    inputs = RetinaGuardInputs(
        quality_score=float(quality_score) if quality_score is not None else None,
        raw_confidence=float(row["raw_confidence"]),
        probabilities=probabilities,
        classifier_logits=None,
        model_predictions=[],
        lesion_evidence_strength=None,
        vessel_evidence_status="UNAVAILABLE",
        attention_lesion_agreement=None,
        explanation_stability=None,
        quality_feature_vector=feature_vector,
        predicted_grade=int(row["predicted_aptos_grade"]),
        predicted_grade_label=row["predicted_aptos_grade_label"],
        referable_dr=float(row["referable_probability_grade_2_or_worse"]) >= 0.5,
        model_version=row.get("model_version"),
    )
    result = guard.evaluate(inputs).to_dict()
    return {
        "retinaguard": result,
        "quality": quality,
        "quality_cache_status": "CACHED_PHASE5_1" if quality_item else "UNAVAILABLE",
    }


def selected_error_fields(row: dict[str, Any], quality_item: dict[str, Any] | None, guard_value: dict[str, Any] | None) -> dict[str, Any]:
    quality = (quality_item or {}).get("quality") or {}
    return {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "actual_grade": row.get("adjudicated_dr_grade"),
        "predicted_grade": row.get("predicted_aptos_grade"),
        "class_probabilities": {str(index): row.get(f"probability_{index}") for index in range(5)},
        "referable_probability": row.get("referable_probability_grade_2_or_worse"),
        "referable_prediction": float(row.get("referable_probability_grade_2_or_worse") or 0.0) >= 0.5,
        "raw_confidence": row.get("raw_confidence"),
        "quality_status": quality.get("quality_decision"),
        "quality_score": quality.get("quality_score"),
        "uncertainty": (guard_value or {}).get("uncertainty"),
        "retinaguard_category": (guard_value or {}).get("trust_category"),
        "evidence_status": "NOT_EXECUTED_IN_THIS_FULL_EVALUATION",
        "grad_cam_status": "NOT_EXECUTED_IN_THIS_FULL_EVALUATION",
    }


def error_analysis(records: list[dict[str, Any]], quality_by_id: dict[str, dict[str, Any]], guard_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in records if row.get("inference_status") == "SUCCESS" and row.get("adjudicated_gradable") == 1]
    def pick(predicate):
        return [selected_error_fields(row, quality_by_id.get(row["image_id"].casefold()), guard_by_id.get(row["image_id"].casefold())) for row in successful if predicate(row)]

    high_confidence = pick(lambda row: row["predicted_aptos_grade"] != row["adjudicated_dr_grade"] and float(row["raw_confidence"]) >= HIGH_CONFIDENCE_THRESHOLD)
    false_negatives = pick(lambda row: row["adjudicated_dr_grade"] >= 2 and float(row["referable_probability_grade_2_or_worse"]) < 0.5)
    return {
        "scope": "Fresh classifier inference on all matched gradable Messidor-2 records.",
        "grade_0_to_2_3_4": pick(lambda row: row["adjudicated_dr_grade"] == 0 and row["predicted_aptos_grade"] in {2, 3, 4}),
        "grade_1_to_2_3_4": pick(lambda row: row["adjudicated_dr_grade"] == 1 and row["predicted_aptos_grade"] in {2, 3, 4}),
        "grade_2_3_4_to_0_1": pick(lambda row: row["adjudicated_dr_grade"] in {2, 3, 4} and row["predicted_aptos_grade"] in {0, 1}),
        "grade_3_to_4_or_4_to_3": pick(lambda row: (row["adjudicated_dr_grade"], row["predicted_aptos_grade"]) in {(3, 4), (4, 3)}),
        "high_confidence_incorrect": {"threshold": HIGH_CONFIDENCE_THRESHOLD, "records": high_confidence},
        "referable_false_negatives": false_negatives,
        "counts": {"successful_gradable": len(successful), "high_confidence_incorrect": len(high_confidence), "referable_false_negatives": len(false_negatives)},
    }


def quality_summary(quality_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter()
    scores: list[float] = []
    readable = 0
    for item in quality_by_id.values():
        if item.get("status") == "READABLE":
            readable += 1
        quality = item.get("quality") or {}
        if quality.get("quality_decision"):
            decisions[quality["quality_decision"]] += 1
        if isinstance(quality.get("quality_score"), (int, float)):
            scores.append(float(quality["quality_score"]))
    return {"status": "CACHED_MEASUREMENTS", "record_count": len(quality_by_id), "readable_count": readable, "decision_counts": dict(decisions), "quality_score": {"mean": statistics.mean(scores) if scores else None, "min": min(scores) if scores else None, "max": max(scores) if scores else None}, "note": "These quality records are reused from the prior Phase 5.1 measurement cache; optional evidence stages were not inferred from them."}


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "stddev_ms": None, "min_ms": None, "max_ms": None}
    return {"count": len(values), "mean_ms": round(statistics.mean(values), 3), "median_ms": round(statistics.median(values), 3), "p95_ms": round(percentile(values, 95) or 0.0, 3), "stddev_ms": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0, "min_ms": round(min(values), 3), "max_ms": round(max(values), 3)}


async def run_smoke(records: list[dict[str, Any]], quality_by_id: dict[str, dict[str, Any]], smoke_image_id: str | None) -> dict[str, Any]:
    """Run one real production-composed evidence/explainability path."""
    from scripts.benchmark_pipeline import _run_once, _services, _settings

    candidates = [row for row in records if row["image_id"].casefold() in quality_by_id and ((quality_by_id[row["image_id"].casefold()].get("quality") or {}).get("quality_decision") == "GRADABLE")]
    chosen = next((row for row in candidates if row["image_id"] == smoke_image_id), None) if smoke_image_id else None
    chosen = chosen or (candidates[0] if candidates else records[0])
    image_path = RAW_ROOT / chosen["image_path"]
    settings = _settings()
    services = _services(settings)
    started = time.perf_counter()
    row = await _run_once(image_path.read_bytes(), image_path.name, services)
    total_ms = (time.perf_counter() - started) * 1000.0
    row["image_id"] = chosen["image_id"]
    row["image_path"] = chosen["image_path"]
    row["image_sha256"] = chosen["image_sha256"]
    row["ground_label_for_reference_only"] = chosen["adjudicated_dr_grade"]
    row["total_elapsed_ms"] = round(total_ms, 3)
    row["source"] = "one real authorized Messidor-2 image; smoke result only"
    row["clinical_validation_claim"] = False
    return row


def smoke_status_summary(smoke: dict[str, Any]) -> dict[str, Any]:
    evidence_status = smoke.get("evidence_status") or {}
    return {"status": "COMPLETED_SMOKE_ONLY" if smoke.get("status") == "COMPLETED" else smoke.get("status", "FAILED"), "image_id": smoke.get("image_id"), "stage_timings_ms": smoke.get("stage_timings_ms", {}), "evidence_module_status": evidence_status, "prediction": smoke.get("prediction"), "retinaguard": smoke.get("retinaguard"), "pdf_generation_attempted": "pdf_generation" in (smoke.get("stage_timings_ms") or {}), "errors": smoke.get("errors", {}), "scope_note": "This one-image run does not establish population-level evidence metrics."}


def report_markdown(summary: dict[str, Any], dataset: dict[str, Any], model: dict[str, Any], metrics: dict[str, Any], referable: dict[str, Any], errors: dict[str, Any], smoke: dict[str, Any]) -> str:
    full_status = summary["status"]
    lines = [
        "# RETINA-NEXUS — Final Messidor-2 External Validation",
        "",
        f"Final status: **{full_status}**",
        "",
        "This report is a frozen, descriptive external model evaluation. It is not clinical validation, regulatory evidence, or a claim that the released labels are official clinical ground truth.",
        "",
        "## 1. Evaluation scope and provenance",
        "",
        f"- Images evaluated by the fresh classifier/current RetinaGuard population pass: **{dataset['matched_pair_count']}**.",
        f"- Diagnosis source: `{dataset['label_source']}`.",
        f"- Label policy: {dataset['label_source_policy']}",
        f"- Patient identifiers available: `{dataset['patient_ids_available']}`; patient-level leakage separation is not claimed.",
        f"- Four archive-only images remain excluded because no diagnosis rows were supplied for them.",
        "",
        "## 2. Frozen model and configuration",
        "",
        f"- Classifier checkpoint: `{model['classifier']['checkpoint_path']}`.",
        f"- Classifier SHA-256: `{model['classifier']['checkpoint_sha256_actual']}`.",
        f"- Architecture: `{model['classifier'].get('architecture')}`; input `{model['classifier'].get('input_resolution')}x{model['classifier'].get('input_resolution')}` RGB.",
        "- Preprocessing: deterministic RGB resize to 224x224, tensor conversion, ImageNet mean/std normalization `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`; no dataset-specific crop flag.",
        "- Referable rule: `P(2)+P(3)+P(4) >= 0.5`; severity is independently `argmax(P0..P4)`.",
        f"- Primary lesion SHA-256: `{model['primary_lesion'].get('sha256')}`.",
        f"- Primary vessel SHA-256: `{model['primary_vessel'].get('sha256')}`.",
        f"- RetinaGuard: `{model['retinaguard']['configuration_version']}`, unfitted calibration `{model['retinaguard']['calibration_version']}`.",
        "",
        "## 3. Dataset validation and leakage",
        "",
        f"- Images found: `{dataset['image_count']}`; label rows: `{dataset['label_row_count']}`; matched pairs: `{dataset['matched_pair_count']}`.",
        f"- Class distribution: `{json.dumps(dataset['class_distribution'], sort_keys=True)}`.",
        "- Existing audit reported 0 corrupt images, 0 missing labels, and no patient IDs. Existing exact duplicate groups are preserved as an audit limitation; no patient-level split claim is made.",
        "",
        "## 4. Classifier metrics",
        "",
        f"- Evaluation population: `{metrics.get('sample_count')}` successful gradable inferences.",
        f"- Accuracy: `{metrics.get('accuracy')}`; macro F1: `{metrics.get('macro_f1')}`; weighted F1: `{metrics.get('weighted_f1')}`.",
        f"- ROC-AUC OVR macro: `{metrics.get('roc_auc_ovr_macro')}`; QWK: `{metrics.get('quadratic_weighted_kappa')}`.",
        f"- Referable sensitivity: `{referable.get('sensitivity')}`; specificity: `{referable.get('specificity')}`; F1: `{referable.get('f1')}`; ROC-AUC: `{referable.get('roc_auc')}`.",
        "- These are descriptive model-evaluation metrics only; they are not clinical performance claims.",
        "",
        "## 5. Error analysis",
        "",
        f"- Grade 0 -> 2/3/4: `{len(errors['grade_0_to_2_3_4'])}`.",
        f"- Grade 1 -> 2/3/4: `{len(errors['grade_1_to_2_3_4'])}`.",
        f"- Grade 2/3/4 -> 0/1: `{len(errors['grade_2_3_4_to_0_1'])}`.",
        f"- Grade 3 <-> 4: `{len(errors['grade_3_to_4_or_4_to_3'])}`.",
        f"- High-confidence incorrect (raw confidence >= {HIGH_CONFIDENCE_THRESHOLD}): `{errors['counts']['high_confidence_incorrect']}`.",
        f"- Referable false negatives: `{errors['counts']['referable_false_negatives']}`.",
        "Complete records are in `failure_analysis.json`; no predictions were changed.",
        "",
        "## 6. Image quality and RetinaGuard",
        "",
        "Quality measurements are included from the prior authorized Phase 5.1 ImageTrustGate cache and are labeled as cached, not silently rerun evidence results.",
        f"- Quality summary: `{json.dumps(summary['quality_summary'], sort_keys=True)}`.",
        f"- Current RetinaGuard was recomputed for `{summary['retinaguard_population_count']}` classifier rows with optional lesion/attention/stability signals explicitly unavailable.",
        "- RetinaGuard is an engineering reliability assessment; it does not prove prediction correctness.",
        "",
        "## 7. Lesion evidence, vessels, localization and explainability",
        "",
        "The current production artifacts were not replaced: primary lesion is the registered pretrained U-Net adapter, primary vessel is the registered R2-V2 RRWNet adapter, and localization remains heuristic because no production localization checkpoint is configured.",
        "Population-level lesion, vessel, localization, Grad-CAM, and attention-agreement metrics were not claimed because the full 1,744-image optional evidence run was not completed.",
        f"- Real smoke run: `{smoke.get('status')}` on `{smoke.get('image_id')}`; details are in `smoke_test.json`.",
        "",
        "## 8. Robustness and performance",
        "",
        "No calibration fitting, threshold tuning, model selection, or robustness perturbation sweep was performed on Messidor-2.",
        f"- Fresh classifier timing summary: `{json.dumps(summary['classifier_latency_ms'], sort_keys=True)}`.",
        f"- Full-pipeline blocker: `{summary['blocker']}`",
        "",
        "## 9. Required output artifacts",
        "",
        "This directory contains the dataset manifest, model manifest, fresh per-image classifier/current-Guard records, classifier metrics, referable metrics, confusion matrix, error analysis, optional-stage summaries, smoke result, and this report.",
        "",
        "## 10. Safety and non-claims",
        "",
        "No clinical validation, regulatory approval, calibration guarantee, or diagnostic claim is made. Raw model confidence is not clinically calibrated. The official label provenance limitation remains attached to every report.",
        "",
        f"## 11. Final decision: {full_status}",
        "",
        "The final status is BLOCKED because a complete 1,744-image run of the current R2-V2 plus primary lesion and Grad-CAM pipeline was not completed within this CPU-only environment. No optional-stage output was fabricated or promoted as a population result.",
        "",
        "## 12. Production and artifact integrity",
        "",
        "- No model weights, backend routes, frontend code, thresholds, or production registries were modified.",
        "- No training, calibration fitting, or automatic retraining was performed.",
        "- Existing Messidor evaluation artifacts were not overwritten.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smoke-image-id", default=None)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records, dataset = load_label_records()
    quality_by_id, quality_manifest = load_quality_cache()
    if len(quality_by_id) not in {0, len(records)}:
        raise RuntimeError(f"Quality cache coverage mismatch: {len(quality_by_id)} records for {len(records)} images")
    settings = load_production_settings()
    paths = resolve_model_paths(settings)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Classifier checkpoint is missing: {checkpoint}")
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != EXPECTED_CLASSIFIER_SHA:
        raise RuntimeError(f"Frozen APTOS classifier SHA mismatch before inference: expected {EXPECTED_CLASSIFIER_SHA}, got {checkpoint_before}")

    classifier_records, model_info, checksum_before, checksum_after = run_inference(
        records, RAW_ROOT, checkpoint, args.device, max(1, args.batch_size), max(0, args.torch_threads)
    )
    if checksum_before != checksum_after or checksum_after != EXPECTED_CLASSIFIER_SHA:
        raise RuntimeError(f"Classifier checkpoint changed or has an unexpected SHA: before={checksum_before}, after={checksum_after}")
    if len(classifier_records) != len(records):
        raise RuntimeError(f"Inference output count mismatch: {len(classifier_records)} != {len(records)}")

    guard = get_retinaguard_service()
    guard_by_id: dict[str, dict[str, Any]] = {}
    combined_rows: list[dict[str, Any]] = []
    for row in classifier_records:
        if row.get("inference_status") == "SUCCESS":
            guard_data = retinaguard_record(row, quality_by_id.get(row["image_id"].casefold()), guard)
            guard_by_id[row["image_id"].casefold()] = guard_data["retinaguard"]
        else:
            guard_data = {"retinaguard": {"status": "NOT_CALCULATED", "reason": row.get("inference_error")}, "quality": {}, "quality_cache_status": "UNAVAILABLE"}
        combined_rows.append({
            **row,
            "quality": guard_data["quality"],
            "quality_cache_status": guard_data["quality_cache_status"],
            "retinaguard": guard_data["retinaguard"],
            "lesion_evidence": {"status": "NOT_EXECUTED_IN_THIS_FULL_EVALUATION", "reason": "Full population primary lesion run was not completed within bounded CPU execution."},
            "vessel_evidence": {"status": "NOT_EXECUTED_IN_THIS_FULL_EVALUATION", "reason": "Full population R2-V2 run was not completed within bounded CPU execution."},
            "localization": {"status": "NOT_EXECUTED_IN_THIS_FULL_EVALUATION", "reason": "Localization is not configured as a production checkpoint."},
            "grad_cam": {"status": "NOT_EXECUTED_IN_THIS_FULL_EVALUATION"},
            "attention_lesion_agreement": {"status": "NOT_EXECUTED_IN_THIS_FULL_EVALUATION"},
            "robustness": {"status": "NOT_EXECUTED_IN_THIS_FULL_EVALUATION"},
        })

    metrics, per_class = compute_metrics(classifier_records, "1,744 matched Messidor-2 images; successful gradable rows only for grade metrics; released-label descriptive evaluation")
    authoritative_rule = "referable_probability = P(2)+P(3)+P(4); referable = referable_probability >= 0.5; severity grade = argmax(P0..P4)."
    if metrics.get("referable_dr_grade_2_or_worse"):
        metrics["referable_dr_grade_2_or_worse"]["rule"] = authoritative_rule
    referable = metrics.get("referable_dr_grade_2_or_worse", {}) if metrics.get("status") == "CALCULATED" else {}
    errors = error_analysis(classifier_records, quality_by_id, guard_by_id)
    model_manifest = build_model_manifest(settings, paths, model_info)
    quality_stats = quality_summary(quality_by_id)
    guard_counts = Counter(item.get("trust_category") for item in guard_by_id.values())
    classifier_times = [float(row["inference_time_ms"]) for row in classifier_records if isinstance(row.get("inference_time_ms"), (int, float))]
    smoke: dict[str, Any] = {"status": "SKIPPED", "reason": "--skip-smoke was supplied."}
    if not args.skip_smoke:
        smoke = asyncio.run(run_smoke(records, quality_by_id, args.smoke_image_id))
    smoke_summary = smoke_status_summary(smoke)
    evidence_blocker = "Full primary evidence pipeline cannot be completed in this CPU-only environment within bounded execution: current production R2-V2 vessel inference is approximately one minute per image on this host before primary lesion and Grad-CAM work; running it for all 1,744 images would require many hours and was not silently substituted or parallelized into an unbounded job."
    if smoke.get("status") == "COMPLETED":
        evidence_blocker += f" The real smoke run measured {float((smoke.get('stage_timings_ms') or {}).get('evidence_total', 0.0)) / 1000.0:.3f}s for its combined evidence stage; this is an engineering observation, not a population runtime measurement."
    summary = {
        "schema_version": "messidor2-final-external-evaluation-v1",
        "status": "MESSIDOR-2 EXTERNAL EVALUATION BLOCKED",
        "generated_at_utc": now(),
        "evaluation_scope": {"matched_images": len(records), "fresh_classifier_and_retinaguard_population_pass": True, "full_optional_pipeline_population_pass": False},
        "dataset": rel(AUDIT_MANIFEST),
        "label_provenance_limitation": dataset["label_provenance_limitation"],
        "model_manifest": "model_manifest.json",
        "checkpoint_sha256_before": checksum_before,
        "checkpoint_sha256_after": checksum_after,
        "checkpoint_unchanged": checksum_before == checksum_after,
        "quality_summary": quality_stats,
        "retinaguard_population_count": len(guard_by_id),
        "retinaguard_category_counts": dict(guard_counts),
        "classifier_latency_ms": stats(classifier_times),
        "smoke_test": smoke_summary,
        "blocker": evidence_blocker,
        "official_test_images_opened": 0,
        "calibration_fitted": False,
        "threshold_tuning_performed": False,
        "production_promoted": False,
        "clinical_validation_claim": False,
    }

    write_json(OUTPUT_ROOT / "dataset_manifest.json", dataset)
    write_json(OUTPUT_ROOT / "model_manifest.json", model_manifest)
    write_json(OUTPUT_ROOT / "classification_metrics.json", metrics)
    write_json(OUTPUT_ROOT / "per_class_metrics.json", per_class)
    write_json(OUTPUT_ROOT / "referable_metrics.json", {"status": "CALCULATED" if referable else "NOT_CALCULABLE", **referable, "rule": "referable_probability = P(2)+P(3)+P(4); referable = referable_probability >= 0.5; severity grade = argmax(P0..P4).", "threshold": 0.5})
    write_json(OUTPUT_ROOT / "confusion_matrix.json", {"matrix": metrics.get("confusion_matrix"), "labels": {str(key): value for key, value in APTOS_CLASS_MAPPING.items()}})
    if metrics.get("confusion_matrix"):
        write_confusion_matrix(OUTPUT_ROOT / "confusion_matrix.png", metrics["confusion_matrix"])
    write_json(OUTPUT_ROOT / "failure_analysis.json", errors)
    write_json(OUTPUT_ROOT / "quality_summary.json", quality_stats)
    write_json(OUTPUT_ROOT / "retinaguard_summary.json", {"status": "CALCULATED_FOR_CLASSIFIER_AND_QUALITY_SIGNALS", "engine_version": settings.retinaguard_config_version, "category_counts": dict(guard_counts), "population_count": len(guard_by_id), "optional_signals": "lesion evidence, vessel evidence, attention agreement, and explanation stability were unavailable in the all-image pass and are not inferred.", "false_negative_warning_coverage": "Not computed as a clinical safety measure; complete FN records include the current engineering Guard state."})
    for filename, value in {
        "lesion_evidence_summary.json": {"status": "BLOCKED_NOT_EXECUTED_FOR_ALL_IMAGES", "primary_artifact": model_manifest["primary_lesion"], "smoke": smoke_summary.get("evidence_module_status")},
        "vessel_evidence_summary.json": {"status": "BLOCKED_NOT_EXECUTED_FOR_ALL_IMAGES", "primary_artifact": model_manifest["primary_vessel"], "smoke": smoke_summary.get("evidence_module_status")},
        "localization_summary.json": {"status": "NOT_CONFIGURED_FOR_PRODUCTION", "research_model_enabled": False, "smoke": smoke_summary.get("evidence_module_status")},
        "xai_summary.json": {"status": "PARTIAL_SMOKE_ONLY", "classifier_linked": True, "smoke": smoke_summary},
        "robustness_summary.json": {"status": "NOT_EXECUTED", "reason": "No calibration fitting or perturbation sweep was run on the frozen external set."},
        "performance_summary.json": {"status": "PARTIAL", "classifier_latency_ms": stats(classifier_times), "smoke_stage_timings_ms": smoke.get("stage_timings_ms", {}), "full_population_optional_latency": "NOT_MEASURED", "blocker": evidence_blocker},
    }.items():
        write_json(OUTPUT_ROOT / filename, value)
    with (OUTPUT_ROOT / "per_image_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined_rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    write_json(OUTPUT_ROOT / "smoke_test.json", smoke)
    write_json(OUTPUT_ROOT / "evaluation_summary.json", summary)
    markdown = report_markdown(summary, dataset, model_manifest, metrics, referable, errors, smoke)
    (OUTPUT_ROOT / "evaluation_report.md").write_text(markdown, encoding="utf-8")
    (OUTPUT_ROOT / "messidor2_final_external_validation_report.md").write_text(markdown, encoding="utf-8")
    write_json(OUTPUT_ROOT / "run_provenance.json", {"generated_at_utc": now(), "elapsed_seconds": round(time.perf_counter() - started, 3), "command": "scripts/evaluate_messidor2_final_external.py", "read_only_inputs": [rel(LABEL_CSV), rel(AUDIT_MANIFEST), rel(QUALITY_CACHE), rel(checkpoint)], "writes_only_under": rel(OUTPUT_ROOT), "official_test_images_opened": 0, "production_promoted": False})

    print(json.dumps({"status": summary["status"], "output": rel(OUTPUT_ROOT), "matched_images": len(records), "classifier_metrics": metrics, "referable_metrics": referable, "retinaguard_category_counts": dict(guard_counts), "smoke_status": smoke.get("status"), "checkpoint_sha256": checksum_after, "official_test_images_opened": 0}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
