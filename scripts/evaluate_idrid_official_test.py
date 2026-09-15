"""Frozen official IDRiD test evaluation.

This command is deliberately not a training or model-selection command.  It
loads the registered IDRiD checkpoint, reads only the 103 official Disease
Grading testing records reserved in ``dr_training_split.json``, and writes a
final evaluation artifact.  It never changes the checkpoint, threshold,
preprocessing, production configuration, or model registry.

Run from the repository root:

    python scripts/evaluate_idrid_official_test.py

The resulting evaluation is a research prototype evaluation, not clinical
validation and not a production promotion decision.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "checkpoint_best.pt"
SPLIT_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"
SOURCE_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "source_manifest.json"
OUTPUT_JSON = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_official_test_evaluation.json"
OUTPUT_MD = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_official_test_evaluation.md"
EXPECTED_SHA = "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
MODEL_VERSION = "efficientnet-b0-idrid-20260912-v1"
REFERABLE_GRADES = (2, 3, 4)
REFERABLE_THRESHOLD = 0.5
GRADE_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
HIGH_CONFIDENCE_THRESHOLD = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def transform_metadata(transform: Any) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for item in getattr(transform, "transforms", []):
        entry: dict[str, Any] = {"name": item.__class__.__name__}
        for attribute in ("size", "mean", "std", "interpolation", "antialias"):
            if hasattr(item, attribute):
                entry[attribute] = json_safe(getattr(item, attribute))
        values.append(entry)
    return {"name": transform.__class__.__name__, "transforms": values, "repr": repr(transform)}


def ece_10_bins(actual: list[int], probabilities: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == np.asarray(actual, dtype=int)).astype(int)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            ece += abs(float(correct[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return float(ece)


def seed_deterministically(torch: Any) -> None:
    random.seed(20260912)
    np.random.seed(20260912)
    torch.manual_seed(20260912)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260912)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_test_records() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError(f"IDRiD split manifest is missing: {SPLIT_MANIFEST}")
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    records = manifest.get("reserved_official_test_records", [])
    if manifest.get("official_test_records_reserved") != 103 or len(records) != 103:
        raise RuntimeError(f"Expected exactly 103 reserved official test records, found {len(records)}")
    for record in records:
        path = record.get("path", "")
        if record.get("official_split") != "test":
            raise RuntimeError(f"Reserved test record has incorrect official split: {record}")
        if "/B. Disease Grading/" not in path.replace("\\", "/") or "/b. Testing Set/" not in path.replace("\\", "/"):
            raise RuntimeError(f"Record is not from the official IDRiD Disease Grading Testing Set: {path}")
        if record.get("label") is not None:
            raise RuntimeError("Official test labels must be read from the reserved disease-grading records, not fabricated fields")
        if not isinstance(record.get("dr_grade"), int) or record["dr_grade"] not in range(5):
            raise RuntimeError(f"Missing or invalid official test DR grade: {record}")
    return manifest, sorted(records, key=lambda item: item["path"])


def audit_test_records(manifest: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_root = ROOT / "ml" / "datasets" / "raw" / "idrid"
    training_validation = manifest.get("records", [])
    train_val_keys = {record.get("record_key") for record in training_validation}
    test_keys = {record.get("record_key") for record in records}
    key_overlap = sorted(item for item in test_keys & train_val_keys if item)
    source_paths: list[dict[str, Any]] = []
    test_hashes: dict[str, list[str]] = defaultdict(list)
    unreadable: list[dict[str, Any]] = []
    dimensions: Counter[str] = Counter()
    for record in records:
        relative = Path(record["path"]).relative_to("ml/datasets/raw/idrid")
        path = raw_root / relative
        item = {"image_id": record["image_id"], "relative_path": record["path"], "exists": path.is_file(), "sha256": None, "width": None, "height": None, "format": None, "mode": None}
        if not path.is_file():
            item["error"] = "missing_file"
            unreadable.append(item)
        else:
            try:
                from PIL import Image

                content = path.read_bytes()
                with Image.open(io.BytesIO(content)) as probe:
                    item["format"] = probe.format
                    probe.verify()
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    item["width"], item["height"], item["mode"] = image.width, image.height, image.mode
                    dimensions[f"{image.width}x{image.height}"] += 1
                item["sha256"] = bytes_sha256(content)
                test_hashes[item["sha256"]].append(record["image_id"])
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                unreadable.append(item)
        source_paths.append(item)

    used_train_val_hashes: dict[str, list[str]] = defaultdict(list)
    for record in training_validation:
        if record.get("sha256"):
            used_train_val_hashes[record["sha256"]].append(record.get("record_key", record.get("image_id", "unknown")))
    cross_split_hash_matches = [
        {"test_image_ids": sorted(test_hashes[content_hash]), "training_validation_record_keys": sorted(used_train_val_hashes[content_hash]), "sha256": content_hash}
        for content_hash in sorted(set(test_hashes) & set(used_train_val_hashes))
    ]
    duplicate_groups = [
        {"sha256": content_hash, "image_ids": sorted(image_ids), "labels": sorted({next(record["dr_grade"] for record in records if record["image_id"] == image_id) for image_id in image_ids})}
        for content_hash, image_ids in sorted(test_hashes.items()) if len(image_ids) > 1
    ]
    conflict_groups = [group for group in duplicate_groups if len(group["labels"]) > 1]
    reserved_payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "expected_record_count": 103,
        "actual_record_count": len(records),
        "source_dataset": "IDRiD Disease Grading official Testing Set",
        "source_path_validation": {"all_official_testing_paths": all(item["relative_path"].replace("\\", "/").find("/b. Testing Set/") >= 0 for item in source_paths), "records": source_paths},
        "missing_or_unreadable": unreadable,
        "readable_count": len(records) - len(unreadable),
        "record_key_overlap_with_train_validation": key_overlap,
        "cross_split_hash_matches_in_used_train_validation": cross_split_hash_matches,
        "duplicate_image_groups_within_test": duplicate_groups,
        "conflicting_label_duplicate_groups": conflict_groups,
        "dimensions": dict(dimensions),
        "test_record_manifest_sha256": bytes_sha256(reserved_payload),
        "split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST) if SOURCE_MANIFEST.is_file() else None,
        "known_excluded_cross_split_duplicate_policy": [item for item in manifest.get("excluded_records", []) if item.get("reason") == "TRAINING_COPY_OF_CROSS_SPLIT_EXACT_DUPLICATE"],
        "label_source": "dr_grade in the reserved official IDRiD Disease Grading records; no labels were inferred or changed",
    }


def collate_batch(batch: list[tuple[Any, dict[str, Any]]]):
    import torch

    images, records = zip(*batch)
    return torch.stack(list(images), dim=0), list(records)


async def assess_quality(records: list[dict[str, Any]], raw_root: Path) -> tuple[dict[str, Any], list[tuple[Any, dict[str, Any]]]]:
    from PIL import Image
    from app.ml.quality.trust_gate import ImageTrustGateService
    from scripts.train_classifier import make_transforms

    quality_service = ImageTrustGateService()
    _, validation_transform = make_transforms(224)
    assessments: dict[str, Any] = {}
    prepared: list[tuple[Any, dict[str, Any]]] = []
    for record in records:
        path = raw_root / Path(record["path"]).relative_to("ml/datasets/raw/idrid")
        try:
            content = path.read_bytes()
            assessment = await quality_service.assess(content)
            assessments[record["image_id"]] = {
                "status": "READABLE",
                "quality_decision": assessment.quality_decision,
                "quality_score": assessment.quality_score,
                "component_scores": assessment.component_scores,
                "issues": [issue.to_dict() if hasattr(issue, "to_dict") else {"type": issue.type, "severity": issue.severity, "message": issue.message} for issue in assessment.issues],
            }
            with Image.open(io.BytesIO(content)) as image:
                prepared.append((validation_transform(image.convert("RGB")), record))
        except Exception as exc:
            assessments[record["image_id"]] = {"status": "UNPROCESSABLE", "error": f"{type(exc).__name__}: {exc}"}
    return assessments, prepared


def infer(prepared: list[tuple[Any, dict[str, Any]]], model: Any, device: Any, torch: Any) -> list[dict[str, Any]]:
    from app.ml.models.classifier import severity_probabilities
    from torch.utils.data import DataLoader

    rows: list[dict[str, Any]] = []
    loader = DataLoader(prepared, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_batch)
    model.eval()
    with torch.inference_mode():
        for images, records in loader:
            outputs = model(images.to(device))
            logits = outputs["severity_logits"].detach().cpu().numpy()
            probabilities = severity_probabilities(outputs, False).detach().cpu().numpy()
            for index, record in enumerate(records):
                vector = probabilities[index].astype(float)
                grade = int(vector.argmax())
                referable_probability = float(vector[list(REFERABLE_GRADES)].sum())
                rows.append({
                    "image_id": record["image_id"],
                    "relative_path": record["path"],
                    "actual_grade": int(record["dr_grade"]),
                    "predicted_grade": grade,
                    "predicted_grade_label": GRADE_LABELS[grade],
                    "p0": round(float(vector[0]), 10), "p1": round(float(vector[1]), 10), "p2": round(float(vector[2]), 10), "p3": round(float(vector[3]), 10), "p4": round(float(vector[4]), 10),
                    "class_probabilities": [float(value) for value in vector],
                    "severity_logits": [round(float(value), 10) for value in logits[index]],
                    "confidence": round(float(vector.max()), 10),
                    "referable_probability": round(referable_probability, 10),
                    "referable_dr": bool(referable_probability >= REFERABLE_THRESHOLD),
                    "inference_status": "COMPLETED",
                })
    return rows


def add_runtime_signals(rows: list[dict[str, Any]], quality: dict[str, Any], guard_engine: Any, uncertainty_engine: Any) -> None:
    from app.ml.trust.guard import RetinaGuardInputs

    for row in rows:
        row_quality = quality.get(row["image_id"], {})
        probabilities = {GRADE_LABELS[index]: row[f"p{index}"] for index in range(5)}
        uncertainty = uncertainty_engine.estimate(probabilities)
        guard = guard_engine.evaluate(RetinaGuardInputs(
            quality_score=row_quality.get("quality_score"), raw_confidence=row["confidence"], probabilities=probabilities,
            classifier_logits=row["severity_logits"], predicted_grade=row["predicted_grade"], predicted_grade_label=row["predicted_grade_label"],
            referable_dr=row["referable_dr"], model_version=MODEL_VERSION,
        ))
        row["quality_status"] = row_quality.get("quality_decision", row_quality.get("status", "UNAVAILABLE"))
        row["quality_score"] = row_quality.get("quality_score")
        row["uncertainty"] = uncertainty
        row["retinaguard"] = {"trust_score": guard.trust_score, "trust_category": guard.trust_category, "risk_flags": guard.risk_flags, "reasons": guard.reason_summary, "assessment_status": guard.configuration.get("assessment_status")}


def error_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ("image_id", "actual_grade", "predicted_grade", "confidence", "referable_probability", "referable_dr", "quality_status", "quality_score", "uncertainty")}


def error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row.get("inference_status") == "COMPLETED" and row["actual_grade"] != row["predicted_grade"]]
    false_negatives = [row for row in rows if row["actual_grade"] in REFERABLE_GRADES and row["referable_dr"] is False]
    false_positives = [row for row in rows if row["actual_grade"] not in REFERABLE_GRADES and row["referable_dr"] is True]
    high_confidence = sorted([row for row in errors if row["confidence"] >= HIGH_CONFIDENCE_THRESHOLD], key=lambda row: row["confidence"], reverse=True)
    categories = {
        "referable_false_negatives": false_negatives,
        "referable_false_positives": false_positives,
        "grade_0_to_2_3_4": [row for row in errors if row["actual_grade"] == 0 and row["predicted_grade"] in REFERABLE_GRADES],
        "grade_1_to_2_3_4": [row for row in errors if row["actual_grade"] == 1 and row["predicted_grade"] in REFERABLE_GRADES],
        "grade_2_3_4_to_0_1": [row for row in errors if row["actual_grade"] in REFERABLE_GRADES and row["predicted_grade"] in (0, 1)],
        "grade_3_to_4": [row for row in errors if row["actual_grade"] == 3 and row["predicted_grade"] == 4],
        "grade_4_to_3": [row for row in errors if row["actual_grade"] == 4 and row["predicted_grade"] == 3],
        "high_confidence_incorrect_predictions": high_confidence,
    }
    return {
        "severity_error_count": len(errors),
        "referable_false_negative_rate": float(len(false_negatives) / max(1, sum(row["actual_grade"] in REFERABLE_GRADES for row in rows))),
        "categories": {name: [error_entry(row) for row in values] for name, values in categories.items()},
        "counts": {name: len(values) for name, values in categories.items()},
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
    }


def brier_diagnostics(actual: list[int], probabilities: np.ndarray) -> dict[str, float]:
    labels = np.eye(5, dtype=float)[np.asarray(actual, dtype=int)]
    referable_actual = np.isin(np.asarray(actual, dtype=int), REFERABLE_GRADES).astype(float)
    referable_probability = probabilities[:, list(REFERABLE_GRADES)].sum(axis=1)
    return {
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - labels) ** 2, axis=1))),
        "referable_brier_score": float(np.mean((referable_probability - referable_actual) ** 2)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["five_class_metrics"]
    ref = report["evaluation"]["referable_metrics"]
    return f"""# IDRiD Official Test Evaluation

Research prototype evaluation only; this is not clinical validation and does not promote the model.

## Dataset and model

- Dataset: {report['dataset']['name']}
- Official test samples: {report['dataset']['sample_count']}
- Model: {report['model']['model_version']}
- Checkpoint SHA-256: `{report['model']['checkpoint_sha256']}`
- Preprocessing: {report['model']['preprocessing']['repr']}
- Official test images evaluated: {report['integrity']['completed_inference_count']}
- Unprocessable images: {report['integrity']['unprocessable_count']}

## Five-class evaluation

| Metric | Value |
|---|---:|
| Accuracy | {metrics['accuracy']:.6f} |
| Macro Precision | {metrics['precision']:.6f} |
| Macro Recall | {metrics['recall']:.6f} |
| Macro F1 | {metrics['f1']:.6f} |
| QWK | {metrics['quadratic_weighted_kappa'] if metrics['quadratic_weighted_kappa'] is not None else 'N/A'} |
| ROC-AUC OVR macro | {metrics['roc_auc_ovr_macro'] if metrics['roc_auc_ovr_macro'] is not None else 'N/A'} |

Confusion matrix (actual rows, predicted columns):

```text
{json.dumps(metrics['confusion_matrix'])}
```

## Referable DR

Referable = grades 2/3/4, with `P2 + P3 + P4 >= 0.5`.

- TP/TN/FP/FN: {ref['true_positive']}/{ref['true_negative']}/{ref['false_positive']}/{ref['false_negative']}
- Sensitivity: {ref['sensitivity']:.6f}
- Specificity: {ref['specificity']:.6f}
- Precision: {ref['precision']:.6f}
- F1: {ref['f1']:.6f}
- ROC-AUC: {ref['roc_auc'] if ref['roc_auc'] is not None else 'N/A'}
- SIH target assessment: {report['sih_target_assessment']['overall']}

## Calibration diagnostics

Raw softmax confidence is not clinically calibrated. No calibration model was fitted on test data. Post-hoc diagnostics only: ECE-10 = `{report['calibration_diagnostics']['ece_10_bins']:.6f}`, multiclass Brier = `{report['calibration_diagnostics']['multiclass_brier_score']:.6f}`, referable Brier = `{report['calibration_diagnostics']['referable_brier_score']:.6f}`.

## Error analysis

{json.dumps(report['error_analysis']['counts'], indent=2)}

## Final model status

**{report['final_model_status']}**

The APTOS production model and production configuration were not changed. The official test result is immutable and must not be used for tuning.
"""


async def evaluate() -> dict[str, Any]:
    import torch
    from app.ml.models.classifier import build_classifier
    from app.ml.trust.guard import RetinaGuardEngine
    from app.ml.trust.uncertainty import UncertaintyEstimator
    from ml.evaluation.metrics import classification_metrics
    from scripts.train_classifier import make_transforms

    manifest, records = load_test_records()
    test_audit = audit_test_records(manifest, records)
    if test_audit["actual_record_count"] != 103:
        raise RuntimeError("The official test manifest audit did not produce exactly 103 records")
    if test_audit["missing_or_unreadable"]:
        raise RuntimeError(f"Official test contains unreadable/missing images; no metrics were fabricated: {test_audit['missing_or_unreadable'][:3]}")
    if test_audit["record_key_overlap_with_train_validation"] or test_audit["cross_split_hash_matches_in_used_train_validation"]:
        raise RuntimeError("Official test leakage detected in the used train/validation manifest; evaluation stopped")

    before_sha = sha256(CHECKPOINT)
    if before_sha != EXPECTED_SHA:
        raise RuntimeError(f"Frozen checkpoint SHA mismatch: expected {EXPECTED_SHA}, got {before_sha}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})
    if model_config != {"backbone": "efficientnet_b0", "num_classes": 5, "input_size": 224, "ordinal_mode": False}:
        raise RuntimeError(f"Frozen checkpoint configuration mismatch: {model_config}")
    model = build_classifier(backbone="efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cpu")
    model.to(device).eval()
    seed_deterministically(torch)

    raw_root = ROOT / "ml" / "datasets" / "raw" / "idrid"
    quality, prepared = await assess_quality(records, raw_root)
    rows = infer(prepared, model, device, torch)
    add_runtime_signals(rows, quality, RetinaGuardEngine(), UncertaintyEstimator())
    if len(rows) != 103:
        raise RuntimeError(f"Frozen inference did not return one result for each official test record: {len(rows)}")
    actual = [row["actual_grade"] for row in rows]
    probabilities = np.asarray([row["class_probabilities"] for row in rows], dtype=float)
    metrics = classification_metrics(actual, probabilities, referable_grades=REFERABLE_GRADES)
    diagnostics = {"ece_10_bins": ece_10_bins(actual, probabilities), **brier_diagnostics(actual, probabilities), "note": "Post-hoc diagnostics only; no calibration or threshold fitting was performed."}
    ref = metrics["referable_dr"]
    target_sensitivity = float(ref["sensitivity"]) > 0.90
    target_specificity = float(ref["specificity"]) > 0.85
    errors = error_analysis(rows)
    checkpoint_after = sha256(CHECKPOINT)
    aptos_path = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
    aptos_context_path = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_aptos_comparison.json"
    aptos_context = json.loads(aptos_context_path.read_text(encoding="utf-8")) if aptos_context_path.is_file() else None
    report = {
        "report_type": "IDRiD official test frozen evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "FINAL_IMMUTABLE_RESEARCH_PROTOTYPE_EVALUATION",
        "dataset": {"name": "IDRiD Disease Grading official Testing Set", "sample_count": 103, "test_data_audit": test_audit},
        "model": {
            "model_version": MODEL_VERSION,
            "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": before_sha,
            "architecture": "EfficientNet-B0 with RETINA-NEXUS hierarchical heads; severity head used for 5-class output",
            "model_config": model_config,
            "class_mapping": GRADE_LABELS,
            "preprocessing": transform_metadata(make_transforms(224)[1]),
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "authoritative_outputs": {"severity": "argmax(P0...P4)", "referable_probability": "P2 + P3 + P4", "referable": "referable_probability >= 0.5", "threshold": 0.5},
        },
        "evaluation": {"five_class_metrics": metrics, "referable_metrics": ref, "class_distribution": dict(sorted(Counter(actual).items())), "per_image_results": rows},
        "calibration_diagnostics": diagnostics,
        "error_analysis": errors,
        "sih_target_assessment": {
            "target": {"referable_sensitivity": "> 0.90", "referable_specificity": "> 0.85"},
            "observed": {"referable_sensitivity": ref["sensitivity"], "referable_specificity": ref["specificity"]},
            "sensitivity": "MET" if target_sensitivity else "NOT_MET",
            "specificity": "MET" if target_specificity else "NOT_MET",
            "overall": "MET" if target_sensitivity and target_specificity else "NOT_MET",
            "note": "SIH target comparison is a research prototype assessment, not clinical validation.",
        },
        "aptos_contextual_comparison": {
            "status": "CONTEXT_ONLY_NOT_DIRECTLY_EQUIVALENT",
            "source": str(aptos_context_path.relative_to(ROOT)).replace("\\", "/") if aptos_context_path.is_file() else None,
            "zero_shot_idrid_validation": aptos_context.get("aptos_model_zero_shot") if aptos_context else None,
            "note": "APTOS comparison is on the IDRiD validation split, not this official IDRiD test set. No APTOS tuning occurred.",
        },
        "integrity": {
            "completed_inference_count": len(rows),
            "unprocessable_count": 0,
            "checkpoint_sha_before": before_sha,
            "checkpoint_sha_after": checkpoint_after,
            "checkpoint_unchanged": before_sha == checkpoint_after == EXPECTED_SHA,
            "aptos_checkpoint_unchanged": sha256(aptos_path) == "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b" if aptos_path.is_file() else None,
            "training_executed": False,
            "threshold_tuning_executed": False,
            "production_promoted": False,
            "production_configuration_changed": False,
            "official_test_images_all_evaluated": len(rows) == 103,
            "test_results_immutable_final": True,
        },
        "reproducibility": {"seed": 20260912, "device": "cpu", "inference_passes": 1, "official_test_not_used_for_tuning": True},
        "limitations": [
            "This is an official IDRiD test evaluation of a research prototype, not clinical validation.",
            "Raw softmax confidence is not clinically calibrated.",
            "Post-hoc ECE and Brier values are diagnostics only and did not alter predictions.",
            "The dataset is small and may not represent deployment populations or camera distributions.",
            "RetinaGuard states were computed with available quality/classifier signals; unavailable optional signals remain explicitly unavailable.",
        ],
        "final_model_status": "KEEP EXPERIMENTAL",
        "final_model_status_reason": "The frozen official test result is recorded without promotion. Even if SIH operating targets are met, this research evaluation does not establish clinical validation or authorize production deployment.",
    }
    OUTPUT_JSON.write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(markdown_report(json_safe(report)), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run the frozen official IDRiD test evaluation without training or tuning").parse_args()


if __name__ == "__main__":
    result = asyncio.run(evaluate())
    print(json.dumps({
        "output": str(OUTPUT_JSON.relative_to(ROOT)).replace("\\", "/"),
        "markdown": str(OUTPUT_MD.relative_to(ROOT)).replace("\\", "/"),
        "sample_count": result["dataset"]["sample_count"],
        "accuracy": result["evaluation"]["five_class_metrics"]["accuracy"],
        "macro_f1": result["evaluation"]["five_class_metrics"]["f1"],
        "qwk": result["evaluation"]["five_class_metrics"]["quadratic_weighted_kappa"],
        "referable_sensitivity": result["evaluation"]["referable_metrics"]["sensitivity"],
        "referable_specificity": result["evaluation"]["referable_metrics"]["specificity"],
        "tp_tn_fp_fn": [result["evaluation"]["referable_metrics"][key] for key in ("true_positive", "true_negative", "false_positive", "false_negative")],
        "sih_target": result["sih_target_assessment"]["overall"],
        "final_model_status": result["final_model_status"],
        "checkpoint_sha256": result["model"]["checkpoint_sha256"],
    }, indent=2))
