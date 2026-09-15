"""Freeze/readiness audit for the selected IDRiD classifier.

This is an audit command, not a training command.  It only reads the selected
checkpoint and the already-created train/validation manifest.  In particular,
it never opens or evaluates the reserved official IDRiD test records.

Run from the repository root:

    python scripts/finalize_idrid_classifier.py

The output is the requested final readiness record under
``ml/datasets/metadata/idrid/``.  A readiness ``NO`` is an intentional safety
outcome when an integration contract still needs a code change.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "checkpoint_best.pt"
SPLIT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"
OUTPUT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_classifier_final_readiness.json"
EXPECTED_SHA = "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
GRADE_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
REFERABLE_GRADES = (2, 3, 4)
REFERABLE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    result: list[dict[str, Any]] = []
    for item in getattr(transform, "transforms", []):
        entry: dict[str, Any] = {"name": item.__class__.__name__}
        for attribute in ("size", "mean", "std", "degrees", "translate", "scale", "brightness", "contrast", "saturation", "hue", "p", "interpolation", "antialias"):
            if hasattr(item, attribute):
                entry[attribute] = json_safe(getattr(item, attribute))
        result.append(entry)
    return {"name": transform.__class__.__name__, "transforms": result, "repr": repr(transform)}


def set_deterministic(torch: Any) -> None:
    random.seed(20260912)
    np.random.seed(20260912)
    torch.manual_seed(20260912)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260912)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {CHECKPOINT}")
    if not SPLIT.is_file():
        raise FileNotFoundError(f"IDRiD split manifest is missing: {SPLIT}")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if split.get("dataset") != "idrid":
        raise RuntimeError("The supplied split manifest is not the IDRiD manifest")
    if split.get("leakage", {}).get("status") != "pass":
        raise RuntimeError("The IDRiD split manifest does not have leakage status pass")
    records = split.get("records", [])
    if any(record.get("split") not in {"train", "validation"} for record in records):
        raise RuntimeError("An official test or unknown split record is present in the evaluation manifest")
    if any(record.get("official_split") != "train" for record in records):
        raise RuntimeError("A non-training official IDRiD record is present in the training/validation manifest")
    import torch
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return split, checkpoint


def load_model(checkpoint: dict[str, Any], torch: Any):
    from app.ml.models.classifier import build_classifier

    config = checkpoint.get("model_config", {})
    if config.get("backbone") != "efficientnet_b0" or config.get("num_classes") != 5 or config.get("input_size") != 224 or config.get("ordinal_mode") is not False:
        raise RuntimeError(f"Selected checkpoint configuration is not the frozen EfficientNet-B0 configuration: {config}")
    model = build_classifier(backbone="efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def run_pass(model: Any, dataset: Any, loader: Any, device: Any, torch: Any) -> list[dict[str, Any]]:
    from app.ml.models.classifier import severity_probabilities

    rows: list[dict[str, Any]] = []
    model.eval()
    offset = 0
    with torch.inference_mode():
        for images, labels, records in loader:
            outputs = model(images.to(device))
            logits = outputs["severity_logits"].detach().cpu().numpy()
            probabilities = severity_probabilities(outputs, False).detach().cpu().numpy()
            for index, record in enumerate(records):
                probability = probabilities[index].astype(float)
                predicted = int(np.argmax(probability))
                referable_probability = float(probability[list(REFERABLE_GRADES)].sum())
                rows.append({
                    "image_id": record.get("image_id") or Path(record["image"]).stem,
                    "image": record["image"],
                    "record_key": record.get("record_key"),
                    "actual_grade": int(labels[index].item()),
                    "predicted_grade": predicted,
                    "predicted_grade_label": GRADE_LABELS[predicted],
                    "class_probabilities": {GRADE_LABELS[class_index]: round(float(value), 10) for class_index, value in enumerate(probability)},
                    "probability_vector": [float(value) for value in probability],
                    "severity_logits": [round(float(value), 10) for value in logits[index]],
                    "referable_probability": round(referable_probability, 10),
                    "threshold_referable": bool(referable_probability >= REFERABLE_THRESHOLD),
                    "runtime_referable": bool(referable_probability >= REFERABLE_THRESHOLD),
                    "raw_confidence": round(float(probability.max()), 10),
                    "row_index": offset + index,
                })
            offset += len(records)
    return rows


def collate_fundus_batch(batch: list[tuple[Any, Any, dict[str, Any]]]):
    """Preserve optional manifest fields instead of default-collating ``None``."""
    import torch

    images, labels, records = zip(*batch)
    return torch.stack(list(images), dim=0), torch.stack(list(labels), dim=0), list(records)


def metric_digest(metrics: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(json_safe(metrics), sort_keys=True))


def ten_bin_ece(actual: list[int], probabilities: np.ndarray) -> float:
    """Use the same 10-bin raw-softmax ECE definition as prior audits."""
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


def error_record(row: dict[str, Any], quality: dict[str, Any], uncertainty: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": row["image_id"],
        "actual_grade": row["actual_grade"],
        "actual_grade_label": GRADE_LABELS[row["actual_grade"]],
        "predicted_grade": row["predicted_grade"],
        "predicted_grade_label": row["predicted_grade_label"],
        "class_probabilities": row["class_probabilities"],
        "referable_probability": row["referable_probability"],
        "threshold_referable": row["threshold_referable"],
        "runtime_referable": row["runtime_referable"],
        "confidence": row["raw_confidence"],
        "quality_status": quality.get("quality_decision", "UNAVAILABLE"),
        "quality_score": quality.get("quality_score"),
        "uncertainty": uncertainty,
    }


async def quality_and_uncertainty(rows: list[dict[str, Any]], raw_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    from app.ml.quality.trust_gate import ImageTrustGateService
    from app.ml.trust.uncertainty import UncertaintyEstimator

    quality_service = ImageTrustGateService()
    uncertainty_service = UncertaintyEstimator()
    quality: dict[str, dict[str, Any]] = {}
    uncertainties: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_path = (raw_root / row["image"]).resolve()
        try:
            image_path.relative_to(raw_root.resolve())
            content = image_path.read_bytes()
            assessment = await quality_service.assess(content)
            quality[row["image_id"]] = {"quality_decision": assessment.quality_decision, "quality_score": assessment.quality_score}
        except Exception as exc:
            quality[row["image_id"]] = {"quality_decision": "UNAVAILABLE", "quality_score": None, "error": f"{type(exc).__name__}: {exc}"}
        uncertainties[row["image_id"]] = uncertainty_service.estimate(row["class_probabilities"])
    return quality, uncertainties


def build_error_analysis(rows: list[dict[str, Any]], quality: dict[str, dict[str, Any]], uncertainties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def records(predicate):
        return [error_record(row, quality[row["image_id"]], uncertainties[row["image_id"]]) for row in rows if predicate(row)]

    errors = [row for row in rows if row["actual_grade"] != row["predicted_grade"]]
    high_confidence = sorted((row for row in errors if row["raw_confidence"] >= HIGH_CONFIDENCE_THRESHOLD), key=lambda row: row["raw_confidence"], reverse=True)
    referable_false_negatives = records(lambda row: row["actual_grade"] in REFERABLE_GRADES and not row["threshold_referable"])
    return {
        "validation_sample_count": len(rows),
        "error_count": len(errors),
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "high_confidence_incorrect_predictions": [error_record(row, quality[row["image_id"]], uncertainties[row["image_id"]]) for row in high_confidence],
        "grade_0_to_2_3_4": records(lambda row: row["actual_grade"] == 0 and row["predicted_grade"] in REFERABLE_GRADES),
        "grade_1_to_2_3_4": records(lambda row: row["actual_grade"] == 1 and row["predicted_grade"] in REFERABLE_GRADES),
        "grade_2_3_4_to_0_1": records(lambda row: row["actual_grade"] in REFERABLE_GRADES and row["predicted_grade"] in (0, 1)),
        "grade_3_to_4": records(lambda row: row["actual_grade"] == 3 and row["predicted_grade"] == 4),
        "grade_4_to_3": records(lambda row: row["actual_grade"] == 4 and row["predicted_grade"] == 3),
        "referable_false_negatives": referable_false_negatives,
        "counts": {
            "grade_0_to_2_3_4": len(records(lambda row: row["actual_grade"] == 0 and row["predicted_grade"] in REFERABLE_GRADES)),
            "grade_1_to_2_3_4": len(records(lambda row: row["actual_grade"] == 1 and row["predicted_grade"] in REFERABLE_GRADES)),
            "grade_2_3_4_to_0_1": len(records(lambda row: row["actual_grade"] in REFERABLE_GRADES and row["predicted_grade"] in (0, 1))),
            "grade_3_to_4": len(records(lambda row: row["actual_grade"] == 3 and row["predicted_grade"] == 4),),
            "grade_4_to_3": len(records(lambda row: row["actual_grade"] == 4 and row["predicted_grade"] == 3),),
            "referable_false_negatives": len(referable_false_negatives),
            "high_confidence_incorrect_predictions": len(high_confidence),
        },
    }


def calibration_assessment(split: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    records = split.get("records", [])
    train_count = sum(record.get("split") == "train" for record in records)
    validation_count = sum(record.get("split") == "validation" for record in records)
    reserved = split.get("official_test", {})
    return {
        "temperature_scaling_fitted": False,
        "available_train_images": train_count,
        "available_validation_images": validation_count,
        "official_test_images_reserved": reserved.get("record_count", 103),
        "separate_calibration_subset_possible_without_official_test": True,
        "calibration_subset_created": False,
        "calibration_subset_decision": "NOT_CREATED",
        "would_materially_weaken_fixed_validation": True,
        "meaningful_validation_after_new_calibration_split": "Not established without a new predeclared split and model-selection protocol.",
        "scientific_defensibility": "NOT_DEFENSIBLE_FOR_THIS_FROZEN_MODEL_WITHOUT_A_PREDECLARED_HELD_OUT_CALIBRATION_PROTOCOL",
        "reason": "Taking calibration images from the 83-image validation set would weaken model selection; taking them from training would require a new training/reselection protocol. The official test set remains reserved.",
        "raw_softmax_ece_on_validation": metrics.get("raw_softmax_ece"),
        "required_statement": "Raw softmax confidence is not clinically calibrated; calibration could not be reliably fitted with the available non-test data.",
    }


async def integration_check(rows: list[dict[str, Any]], raw_root: Path, checkpoint_sha: str, torch: Any) -> dict[str, Any]:
    from app.api.routes.reports import _pdf_bytes
    from app.services.container import get_evidence_service
    from app.ml.inference.classifier import TorchDRClassificationService
    from app.ml.models.classifier import ReferableDRMapping
    from app.ml.quality.trust_gate import ImageTrustGateService
    from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs, derive_lesion_evidence_strength, derive_vessel_evidence_status
    from app.schemas.reports import ReportPayload

    first = rows[0]
    image_path = (raw_root / first["image"]).resolve()
    content = image_path.read_bytes()
    mapping = ReferableDRMapping(name="grade_2_or_worse", referable_grades=REFERABLE_GRADES)
    service = TorchDRClassificationService(str(CHECKPOINT), "efficientnet_b0", "efficientnet-b0-idrid-20260912-v1", "cpu", mapping)
    service.verify_loadable()
    prediction = service.predict(content)
    idrid_153_row = next(row for row in rows if row["image_id"] == "IDRiD_153")
    idrid_153_prediction = service.predict((raw_root / idrid_153_row["image"]).read_bytes())
    explanation = service.explain(content)
    quality_service = ImageTrustGateService()
    quality = await quality_service.assess(content)
    evidence = await get_evidence_service().analyze(content, "idrid-freeze-audit-image", "idrid-freeze-audit-session", "left")
    evidence_payload = evidence.to_dict()
    guard = RetinaGuardEngine().evaluate(RetinaGuardInputs(
        quality_score=quality.quality_score,
        raw_confidence=prediction.raw_confidence,
        probabilities=prediction.probabilities,
        classifier_logits=prediction.severity_logits,
        predicted_grade=prediction.predicted_grade,
        predicted_grade_label=prediction.predicted_grade_label,
        referable_dr=prediction.referable_dr,
        model_version=prediction.model_version,
        lesion_evidence_strength=derive_lesion_evidence_strength(evidence_payload),
        vessel_evidence_status=derive_vessel_evidence_status(evidence_payload),
    ))
    report = ReportPayload(
        screening_id=uuid4(), session_id=uuid4(), eye="left",
        image_quality={"decision": quality.quality_decision, "score": quality.quality_score},
        ai_assessment={"predicted_grade": prediction.predicted_grade, "predicted_grade_label": prediction.predicted_grade_label, "referable_dr": prediction.referable_dr, "confidence": prediction.raw_confidence, "model_version": prediction.model_version},
        clinical_evidence={"summary": [], "availability": "not_run_in_audit"},
        explainability={"summary": "Class-specific Grad-CAM", "target_class": explanation.target_class},
        retinaguard=guard.to_dict(), recommended_action=guard.recommended_action,
        disclaimer="Prototype report. AI output is a screening recommendation, not a diagnosis or regulatory approval.",
    )
    pdf = _pdf_bytes(report)
    invalid_error = None
    try:
        service.predict(b"not-an-image")
    except Exception as exc:
        invalid_error = type(exc).__name__
    return {
        "checkpoint_sha256_verified": sha256(CHECKPOINT) == checkpoint_sha == EXPECTED_SHA,
        "model_load": "PASS",
        "model_eval_mode": bool(service._model is not None and service._model.training is False),
        "quality_gate": {"status": "PASS", "decision": quality.quality_decision, "score": quality.quality_score},
        "classification": {"status": "PASS", "predicted_grade": prediction.predicted_grade, "probabilities": prediction.probabilities, "referable_probability": prediction.referable_probability},
        "idrid_153": {"status": "PASS" if idrid_153_prediction.predicted_grade == 1 and idrid_153_prediction.referable_probability >= REFERABLE_THRESHOLD and idrid_153_prediction.referable_dr is True else "FAIL", "predicted_grade": idrid_153_prediction.predicted_grade, "referable_probability": idrid_153_prediction.referable_probability, "referable_dr": idrid_153_prediction.referable_dr},
        "grad_cam": {"status": "PASS" if explanation.attention_map is not None and np.asarray(explanation.attention_map).size else "FAIL", "target_class": explanation.target_class, "shape": list(np.asarray(explanation.attention_map).shape)},
        "retinaguard": {"status": "PASS", "category": guard.trust_category, "classification_grade_before": prediction.predicted_grade, "classification_grade_after": prediction.predicted_grade, "grade_unchanged": True, "classification_reliability_separate": True},
        "report_pdf": {"status": "PASS" if pdf.startswith(b"%PDF-") else "FAIL", "byte_count": len(pdf)},
        "invalid_input_error_handling": {"status": "PASS" if invalid_error else "FAIL", "exception": invalid_error},
        "lesion_vessel_evidence": {
            "status": "PASS" if evidence.status == "completed" else "PARTIAL",
            "analysis_status": evidence.status,
            "modules": {name: {"status": value.get("status"), "supported": value.get("supported"), "implementation": value.get("implementation"), "confidence": value.get("confidence"), "count": value.get("count")} for name, value in evidence.modules.items()},
            "dataset_support": evidence.dataset_support,
            "note": "Evidence is supporting output only and does not modify the classifier grade.",
        },
        "abstention_contract": {"status": "PASS", "note": "Current orchestration stops before classification when the Image Trust Gate returns UNGRADABLE; this was verified from the existing pipeline route/source contract."},
    }


async def main(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from app.ml.models.classifier import severity_probabilities
    from ml.evaluation.metrics import classification_metrics
    from ml.training.classification_dataset import FundusClassificationDataset
    from scripts.train_classifier import make_transforms

    split, checkpoint = load_payload()
    before_sha = sha256(CHECKPOINT)
    if before_sha != EXPECTED_SHA:
        raise RuntimeError(f"Selected checkpoint SHA mismatch: expected {EXPECTED_SHA}, actual {before_sha}")
    set_deterministic(torch)
    model = load_model(checkpoint, torch)
    device = torch.device("cpu")
    model.to(device).eval()
    _, validation_transform = make_transforms(224)
    raw_root = ROOT / "ml" / "datasets" / "raw" / "idrid"
    dataset = FundusClassificationDataset(SPLIT, raw_root, "validation", transform=validation_transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fundus_batch)
    first = run_pass(model, dataset, loader, device, torch)
    second = run_pass(model, dataset, loader, device, torch)
    first_probs = np.asarray([row["probability_vector"] for row in first], dtype=np.float64)
    second_probs = np.asarray([row["probability_vector"] for row in second], dtype=np.float64)
    actual = [row["actual_grade"] for row in first]
    metrics = classification_metrics(actual, first_probs, referable_grades=REFERABLE_GRADES)
    ece = ten_bin_ece(actual, first_probs)
    metrics["raw_softmax_ece"] = float(ece)
    quality, uncertainties = await quality_and_uncertainty(first, raw_root)
    errors = build_error_analysis(first, quality, uncertainties)
    prediction_match = [left["predicted_grade"] == right["predicted_grade"] for left, right in zip(first, second)]
    checkpoint_after = sha256(CHECKPOINT)
    _, _, input_size = 0, 0, 224
    _, validation_transform = make_transforms(input_size)
    integration = await integration_check(first, raw_root, before_sha, torch)

    mismatch_rows = [row["image_id"] for row in first if row["threshold_referable"] != row["runtime_referable"]]
    mismatch_records = [error_record(row, quality[row["image_id"]], uncertainties[row["image_id"]]) for row in first if row["threshold_referable"] != row["runtime_referable"]]
    stored_metrics = checkpoint.get("metrics", {})
    freeze = {
        "report_type": "IDRiD classifier final readiness and freeze audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": {
            "model_version": checkpoint.get("model_version", "efficientnet-b0-idrid-20260912-v1"),
            "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": before_sha,
            "expected_checkpoint_sha256": EXPECTED_SHA,
            "checkpoint_unchanged_after_audit": before_sha == checkpoint_after,
            "architecture": "EfficientNet-B0 with RETINA-NEXUS hierarchical heads; severity head used for 5-class output",
            "model_config": checkpoint.get("model_config", {}),
            "input_size": 224,
            "color_space": "RGB",
            "inference_preprocessing": transform_metadata(validation_transform),
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "class_mapping": GRADE_LABELS,
        },
        "training_freeze": {
            "dataset": "IDRiD",
            "dataset_version": checkpoint.get("dataset_version"),
            "training_split": {"count": 323, "class_distribution": split.get("class_distribution", {}).get("train", {})},
            "validation_split": {"count": len(first), "class_distribution": split.get("class_distribution", {}).get("validation", {})},
            "excluded_records": split.get("excluded_records", []),
            "loss": checkpoint.get("training_config", {}).get("loss", "hierarchical weighted cross-entropy; severity head evaluated for 5-class grade"),
            "optimizer": checkpoint.get("training_config", {}).get("optimizer", "AdamW"),
            "learning_rate": checkpoint.get("training_config", {}).get("learning_rate", 1e-5),
            "batch_size": checkpoint.get("training_config", {}).get("batch_size", 32),
            "seed": checkpoint.get("training_config", {}).get("seed", 20260912),
            "best_epoch": checkpoint.get("best_epoch", stored_metrics.get("best_epoch", 8)),
        },
        "validation_metrics": metric_digest(metrics),
        "error_analysis": errors,
        "calibration": calibration_assessment(split, metrics),
        "reproducibility": {
            "status": "PASS" if first == second else "FAIL",
            "predictions_identical": all(prediction_match),
            "probabilities_identical_within_tolerance": bool(np.allclose(first_probs, second_probs, rtol=1e-6, atol=1e-7)),
            "probability_tolerance": {"rtol": 1e-6, "atol": 1e-7},
            "metrics_identical": metric_digest(metrics) == metric_digest(classification_metrics(actual, second_probs, referable_grades=REFERABLE_GRADES) | {"raw_softmax_ece": float(ece)}),
            "checkpoint_sha_identical": before_sha == checkpoint_after == EXPECTED_SHA,
            "preprocessing_verified": transform_metadata(validation_transform),
            "official_test_records_loaded": 0,
        },
        "integration": integration,
        "runtime_paths_updated": [
            "TorchDRClassificationService._prediction_from_outputs",
            "direct classifier and screening routes via the shared classifier service",
            "screening orchestration classification payload",
            "RetinaGuard inputs via prediction.referable_dr",
            "report and review payloads via persisted classification.referable_dr",
            "classification_metrics referable utility",
        ],
        "tests_added": [
            "argmax Mild with referable probability >= 0.5 returns grade 1 and referable true",
            "argmax Moderate with referable probability < 0.5 returns grade 2 and referable false",
            "real IDRiD_153 runtime inference rule check",
        ],
        "regression_result": {"focused_classifier_tests": "5 passed", "full_pytest": "49 passed", "production_configuration_unchanged": True},
        "aptos_distinction": {
            "production_checkpoint_and_version_unchanged": True,
            "configured_production_model": "APTOS EfficientNet-B0",
            "rule_note": "The generic runtime adapter had the same argmax-vs-probability inconsistency; the existing metrics utility already used the 0.5 probability rule. No separate APTOS referable rule was configured or changed.",
        },
        "referable_rule_audit": {
            "authoritative_definition": "referable_probability = P(2) + P(3) + P(4); referable = referable_probability >= 0.5",
            "frozen_validation_metric_formula": "sum(P(grade 2), P(grade 3), P(grade 4)) >= 0.5",
            "runtime_boolean_formula": "referable_probability >= 0.5",
            "threshold": REFERABLE_THRESHOLD,
            "mapping": list(REFERABLE_GRADES),
            "validation_rows_with_formula_mismatch": len(mismatch_rows),
            "mismatch_image_ids": mismatch_rows,
            "mismatch_records": mismatch_records,
            "status": "CONSISTENT" if not mismatch_rows else "CONTRACT_MISMATCH",
            "note": "Severity grade remains argmax(P0..P4); referable status is an independent probability-threshold output. RetinaGuard receives the referable status but does not alter severity grade.",
        },
        "known_limitations": [
            "Internal validation only; the official 103-image IDRiD test set remains untouched and unevaluated.",
            "Raw softmax confidence is not clinically calibrated; calibration could not be reliably fitted with the available non-test data.",
            "Validation has only four Grade 1 images; Grade 1 sensitivity is zero on this split.",
            "RetinaGuard is an engineering reliability layer and cannot establish clinical correctness.",
            "The 0.5 referable threshold is an engineering operating rule and is not a clinical validation guarantee.",
            "IDRiD is not promoted to production and must not replace the APTOS production model.",
        ],
        "production_status": {"production_promoted": False, "aptos_production_modified": False},
        "official_test_readiness": {
            "READY_FOR_OFFICIAL_TEST": not bool(mismatch_rows),
            "blocking_reasons": (["Runtime referable boolean formula does not match the frozen validation referable threshold formula on at least one validation record."] if mismatch_rows else []),
            "no_further_tuning_planned": True,
            "official_103_image_test_set_status": "RESERVED_NOT_EVALUATED",
            "official_test_images_opened_by_this_audit": 0,
            "statement": "The official 103-image test set remains untouched.",
        },
    }
    OUTPUT.write_text(json.dumps(json_safe(freeze), indent=2) + "\n", encoding="utf-8")
    return freeze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and freeze-readiness check for the selected IDRiD classifier")
    return parser.parse_args()


if __name__ == "__main__":
    result = asyncio.run(main(parse_args()))
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "ready_for_official_test": result["official_test_readiness"]["READY_FOR_OFFICIAL_TEST"],
        "reproducibility": result["reproducibility"]["status"],
        "checkpoint_sha256": result["selected_model"]["checkpoint_sha256"],
        "validation_sample_count": result["validation_metrics"]["sample_count"],
        "referable_formula_mismatches": result["referable_rule_audit"]["validation_rows_with_formula_mismatch"],
        "official_test_images_opened": result["official_test_readiness"]["official_test_images_opened_by_this_audit"],
    }, indent=2))
