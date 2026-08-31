"""Validate RetinaGuard reliability signals on the authorized Phase 4B run.

This command is retrospective and deliberately does not train, tune, or alter
the APTOS classifier. It measures the reliability signals that are available
from the existing Messidor-2 predictions, runs the real quality gate over the
same images, and optionally performs a small deterministic robustness sample
with real classifier inference and Grad-CAM.

The resulting artifacts are engineering diagnostics. They are not clinical
validation, calibration, or proof that an explanation is causal.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.inference.classifier import TorchDRClassificationService  # noqa: E402
from app.ml.quality.trust_gate import ImageTrustGateError, ImageTrustGateService  # noqa: E402
from app.ml.trust.guard import RetinaGuardEngine, RetinaGuardInputs  # noqa: E402


DEFAULT_PREDICTIONS = ROOT / "ml" / "evaluation" / "messidor" / "zero_shot_predictions.csv"
DEFAULT_RAW_DIR = ROOT / "ml" / "datasets" / "raw" / "messidor"
DEFAULT_OUTPUT = ROOT / "ml" / "evaluation" / "reliability"
DEFAULT_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
EXPECTED_CHECKPOINT_SHA256 = "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b"
PROBABILITY_COLUMNS = ["probability_0", "probability_1", "probability_2", "probability_3", "probability_4"]
GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(float(mean(values)), 8),
        "std": round(float(stdev(values)), 8) if len(values) > 1 else 0.0,
        "min": round(float(min(values)), 8),
        "max": round(float(max(values)), 8),
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Messidor-2 predictions CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Messidor-2 predictions CSV is empty: {path}")
    required = {"image_path", "image_id", "predicted_aptos_grade", "adjudicated_dr_grade", "inference_status", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {', '.join(missing)}")
    return rows


def _manifest_index() -> dict[str, str]:
    path = ROOT / "ml" / "evaluation" / "messidor" / "dataset_manifest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(record.get("image_path", "")).casefold().replace("\\", "/"): str(record.get("image_root_relative_path", ""))
        for record in payload.get("matched_records", [])
        if record.get("image_path") and record.get("image_root_relative_path")
    }


def _image_path(raw_dir: Path, row: dict[str, str], manifest: dict[str, str]) -> Path:
    candidate = Path(row["image_path"])
    if candidate.is_absolute():
        return candidate
    direct = raw_dir / candidate
    if direct.is_file():
        return direct
    relative = manifest.get(row["image_path"].casefold().replace("\\", "/"))
    if relative:
        alternate = raw_dir / "images" / "messidor-2" / relative
        if alternate.is_file():
            return alternate
    return direct


async def _assess_quality(rows: list[dict[str, str]], raw_dir: Path) -> list[dict[str, Any]]:
    service = ImageTrustGateService()
    manifest = _manifest_index()
    output: list[dict[str, Any]] = []
    for row in rows:
        path = _image_path(raw_dir, row, manifest)
        item: dict[str, Any] = {"image_id": row.get("image_id"), "image_path": row.get("image_path"), "local_path": str(path), "status": "UNREADABLE", "quality": None, "error": None}
        try:
            content = path.read_bytes()
            assessment = await service.assess(content)
            item.update({"status": "READABLE", "quality": assessment.to_dict()})
        except (OSError, ImageTrustGateError, ValueError) as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        output.append(item)
    return output


def _probabilities(row: dict[str, str]) -> dict[str, float] | None:
    values = [_float(row.get(column)) for column in PROBABILITY_COLUMNS]
    if any(value is None for value in values):
        return None
    total = float(sum(value for value in values if value is not None))
    if total <= 0:
        return None
    return {GRADE_LABELS[index]: round(float(values[index] / total), 8) for index in range(5)}


def _duplicate_summary() -> dict[str, Any]:
    path = ROOT / "ml" / "evaluation" / "messidor" / "duplicate_audit.json"
    if not path.is_file():
        return {"status": "UNAVAILABLE", "source": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "UNAVAILABLE", "source": str(path), "error": str(exc)}
    fields = {key: payload.get(key) for key in ("exact_duplicate_group_count", "perceptual_duplicate_group_count", "canonical_representative_count", "deduplicated_image_count") if key in payload}
    return {"status": "AVAILABLE", "source": str(path), **fields, "note": "Duplicate results are inherited from the separately generated Phase 4B audit; the original predictions were not rewritten."}


def _quality_summary(quality_rows: list[dict[str, Any]]) -> dict[str, Any]:
    readable = [item for item in quality_rows if item["status"] == "READABLE"]
    decisions = Counter((item.get("quality") or {}).get("quality_decision") for item in readable)
    scores = [_float((item.get("quality") or {}).get("quality_score")) for item in readable]
    scores = [value for value in scores if value is not None]
    component_values: dict[str, list[float]] = {}
    resolutions: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    metadata_count = 0
    for item in readable:
        quality = item.get("quality") or {}
        for name, value in (quality.get("component_scores") or {}).items():
            numeric = _float(value)
            if numeric is not None:
                component_values.setdefault(name, []).append(numeric)
        metadata = quality.get("input_metadata") or {}
        width, height = _float(metadata.get("width")), _float(metadata.get("height"))
        if width is not None and height is not None:
            resolutions.append(width * height)
            widths.append(width)
            heights.append(height)
        if metadata.get("camera_metadata"):
            metadata_count += 1
    return {
        "file_count": len(quality_rows),
        "readable_file_count": len(readable),
        "unreadable_file_count": len(quality_rows) - len(readable),
        "decision_counts": dict(sorted(decisions.items(), key=lambda item: str(item[0]))),
        "quality_score": _summary(scores),
        "component_scores": {name: _summary(values) for name, values in sorted(component_values.items())},
        "resolution": {"width": _summary(widths), "height": _summary(heights), "pixel_count": _summary(resolutions)},
        "camera_metadata_present_count": metadata_count,
    }


def _engine_rows(rows: list[dict[str, str]], quality_rows: list[dict[str, Any]], engine: RetinaGuardEngine) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, quality_item in zip(rows, quality_rows):
        quality = quality_item.get("quality") or {}
        probabilities = _probabilities(row)
        grade = _int(row.get("predicted_aptos_grade"))
        inputs = RetinaGuardInputs(
            quality_score=_float(quality.get("quality_score")),
            raw_confidence=_float(row.get("raw_confidence")),
            probabilities=probabilities or {},
            predicted_grade=grade,
            predicted_grade_label=GRADE_LABELS[grade] if grade in range(5) else None,
            model_version=row.get("model_version"),
        )
        result = engine.evaluate(inputs)
        output.append({
            "image_id": row.get("image_id"), "image_path": row.get("image_path"),
            "ground_truth_grade": _int(row.get("adjudicated_dr_grade")),
            "ground_truth_referable": (_int(row.get("adjudicated_dr_grade")) or 0) >= 2 if _int(row.get("adjudicated_dr_grade")) is not None else None,
            "predicted_grade": grade,
            "predicted_referable": (_float(row.get("referable_probability_grade_2_or_worse"), 0.0) or 0.0) >= 0.5,
            "prediction_status": row.get("inference_status"),
            "quality_decision": quality.get("quality_decision"),
            "quality_score": quality.get("quality_score"),
            "raw_confidence": _float(row.get("raw_confidence")),
            "retinaguard": result.to_dict(),
        })
    return output


def _false_negative_analysis(engine_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in engine_rows if item["prediction_status"] == "SUCCESS" and item["ground_truth_grade"] in range(5)]
    false_negatives = [item for item in eligible if item["ground_truth_referable"] and not item["predicted_referable"]]
    warning_counts: Counter[str] = Counter()
    for item in false_negatives:
        warning_counts.update(flag.get("code", "unknown") for flag in item["retinaguard"].get("risk_flags", []))
    flagged = sum(bool(item["retinaguard"].get("risk_flags")) for item in false_negatives)
    return {
        "status": "CALCULATED",
        "evaluation_type": "retrospective_false_negative_warning_analysis",
        "source": str(DEFAULT_PREDICTIONS),
        "rule": "false negative means adjudicated grade >= 2 and original grade>=2 probability < 0.50, matching Phase 4B; no warning threshold was optimized here.",
        "eligible_sample_count": len(eligible),
        "false_negative_count": len(false_negatives),
        "false_negative_warning_coverage": round(flagged / len(false_negatives), 8) if false_negatives else None,
        "warning_counts_among_false_negatives": dict(sorted(warning_counts.items())),
        "false_negatives_with_multiple_warnings": sum(sum(1 for flag in item["retinaguard"].get("risk_flags", [])) > 1 for item in false_negatives),
        "examples": [
            {
                "image_id": item["image_id"], "image_path": item["image_path"],
                "ground_truth_grade": item["ground_truth_grade"], "predicted_grade": item["predicted_grade"],
                "warnings": item["retinaguard"].get("risk_flags", []),
                "reliability_state": item["retinaguard"].get("reliability_state"),
            }
            for item in false_negatives[:25]
        ],
        "limitations": [
            "This is a retrospective warning audit, not a prospective false-negative detector.",
            "Warnings were not tuned on the false-negative labels and must not be interpreted as sensitivity or clinical safety performance.",
            "Messidor-2 reference labels are separately acquired third-party labels as documented by Phase 4B.",
        ],
    }


def _risk_coverage(engine_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in engine_rows if item["prediction_status"] == "SUCCESS" and item["ground_truth_grade"] in range(5)]
    total = len(eligible)
    for item in eligible:
        flags = item["retinaguard"].get("risk_flags", [])
        item["_warning_burden"] = len(flags)

    def slice_metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"count": 0, "coverage": 0.0, "five_class_accuracy": None, "referable_false_negative_rate": None}
        correct = sum(item["predicted_grade"] == item["ground_truth_grade"] for item in selected)
        referable = [item for item in selected if item["ground_truth_referable"]]
        referable_fn = sum(not item["predicted_referable"] for item in referable)
        return {"count": len(selected), "coverage": round(len(selected) / total, 8) if total else 0.0, "five_class_accuracy": round(correct / len(selected), 8), "referable_false_negative_rate": round(referable_fn / len(referable), 8) if referable else None}

    burden_points = [{"max_warning_burden": limit, **slice_metrics([item for item in eligible if item["_warning_burden"] <= limit])} for limit in (0, 1, 2, 3)]
    score_points = [{"minimum_trust_score": threshold, **slice_metrics([item for item in eligible if (_float(item["retinaguard"].get("trust_score"), 0.0) or 0.0) >= threshold])} for threshold in (0.45, 0.60, 0.75)]
    for item in eligible:
        item.pop("_warning_burden", None)
    return {
        "status": "CALCULATED",
        "evaluation_type": "retrospective_descriptive_risk_coverage",
        "eligible_sample_count": total,
        "operating_points": {"warning_burden": burden_points, "trust_score": score_points},
        "state_distribution": dict(Counter(item["retinaguard"].get("reliability_state") for item in eligible)),
        "note": "Descriptive retrospective risk-coverage analysis. Operating points are fixed engineering views, not optimized thresholds, and do not establish clinical risk coverage.",
    }


def _software() -> dict[str, str]:
    result = {"python": platform.python_version()}
    try:
        from importlib.metadata import version
        for package in ("numpy", "Pillow", "torch", "torchvision"):
            try:
                result[package] = version(package)
            except Exception:
                result[package] = "unavailable"
    except ImportError:
        pass
    return result


def _dataset_version() -> str:
    path = ROOT / "ml" / "evaluation" / "messidor" / "dataset_manifest.json"
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("dataset_version")
            if value:
                return str(value)
        except (OSError, json.JSONDecodeError):
            pass
    return "UNAVAILABLE"


def _variant_bytes(image: Image.Image, name: str, seed: int) -> bytes:
    if name == "brightness_minus_15pct":
        output_image = ImageEnhance.Brightness(image).enhance(0.85)
    elif name == "brightness_plus_15pct":
        output_image = ImageEnhance.Brightness(image).enhance(1.15)
    elif name == "contrast_minus_20pct":
        output_image = ImageEnhance.Contrast(image).enhance(0.80)
    elif name == "contrast_plus_20pct":
        output_image = ImageEnhance.Contrast(image).enhance(1.20)
    elif name == "gaussian_blur_mild":
        output_image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    elif name == "gaussian_blur_moderate":
        output_image = image.filter(ImageFilter.GaussianBlur(radius=2.8))
    elif name == "minor_noise":
        rng = np.random.default_rng(seed)
        values = np.asarray(image, dtype=np.int16)
        noisy = np.clip(values + rng.normal(0, 3.0, values.shape), 0, 255).astype(np.uint8)
        output_image = Image.fromarray(noisy, mode="RGB")
    elif name == "rotation_plus_3deg":
        output_image = image.rotate(3.0, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    else:
        raise ValueError(f"Unknown perturbation: {name}")
    buffer = __import__("io").BytesIO()
    output_image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _save_gradcam_overlay(path: Path, image: Image.Image, attention: np.ndarray) -> None:
    attention = np.asarray(attention, dtype=np.float32)
    if attention.shape != (image.height, image.width):
        attention = np.asarray(Image.fromarray(attention, mode="F").resize((image.width, image.height), Image.Resampling.BILINEAR), dtype=np.float32)
    attention = np.clip(attention, 0.0, 1.0)
    heat = np.stack([np.clip(attention * 2.0, 0, 1), np.clip(1.0 - np.abs(attention - 0.5) * 2.0, 0, 1), np.clip(1.0 - attention * 2.0, 0, 1)], axis=-1) * 255
    rgb = np.asarray(image, dtype=np.float32)
    weight = (0.12 + 0.58 * attention)[..., None]
    overlay = np.clip(rgb * (1.0 - weight) + heat * weight, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(path, format="PNG", optimize=True)


async def _robustness(rows: list[dict[str, str]], raw_dir: Path, output: Path, checkpoint: Path, samples: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    valid = [row for row in rows if _image_path(raw_dir, row, _manifest_index()).is_file()]
    selected: list[dict[str, str]] = []
    for grade in range(5):
        selected.extend(sorted([row for row in valid if _int(row.get("adjudicated_dr_grade")) == grade], key=lambda item: item.get("image_id", ""))[:1])
    selected = sorted(selected, key=lambda item: item.get("image_id", ""))[: max(1, samples)]
    classifier = TorchDRClassificationService(str(checkpoint), "efficientnet_b0", model_version=None, device="cpu")
    quality = ImageTrustGateService()
    engine = RetinaGuardEngine()
    perturbations = ["brightness_minus_15pct", "brightness_plus_15pct", "contrast_minus_20pct", "contrast_plus_20pct", "gaussian_blur_mild", "gaussian_blur_moderate", "minor_noise", "rotation_plus_3deg"]
    records: list[dict[str, Any]] = []
    base_gradcam_count = 0
    variant_gradcam_count = 0
    prediction_changes = 0
    cam_similarities: list[float] = []
    for index, row in enumerate(selected, start=1):
        path = _image_path(raw_dir, row, _manifest_index())
        source_bytes = path.read_bytes()
        with Image.open(__import__("io").BytesIO(source_bytes)) as source:
            image = source.convert("RGB")
        try:
            base = await classifier.explain_async(source_bytes)
            base_gradcam_count += 1
        except Exception as exc:
            records.append({"image_id": row.get("image_id"), "image_path": row.get("image_path"), "status": "BASE_FAILED", "error": f"{type(exc).__name__}: {exc}"})
            continue
        base_attention = np.asarray(base.attention_map, dtype=np.float32)
        sample_record: dict[str, Any] = {
            "image_id": row.get("image_id"), "image_path": row.get("image_path"), "ground_truth_grade": _int(row.get("adjudicated_dr_grade")),
            "base_prediction": {"predicted_grade": base.prediction.predicted_grade, "predicted_grade_label": base.prediction.predicted_grade_label, "raw_confidence": base.prediction.raw_confidence, "probabilities": base.prediction.probabilities, "model_version": base.prediction.model_version},
            "base_gradcam_artifact": str((output / "visuals" / f"sample_{index:02d}_base_gradcam.png").relative_to(output)),
            "variants": [],
        }
        _save_gradcam_overlay(output / "visuals" / f"sample_{index:02d}_base_gradcam.png", image, base_attention)
        base_quality = await quality.assess(source_bytes)
        base_guard = engine.evaluate(RetinaGuardInputs(quality_score=base_quality.quality_score, raw_confidence=base.prediction.raw_confidence, probabilities=base.prediction.probabilities, predicted_grade=base.prediction.predicted_grade, predicted_grade_label=base.prediction.predicted_grade_label, model_version=base.prediction.model_version))
        sample_record["base_reliability_state"] = base_guard.trust_category
        for perturbation in perturbations:
            variant_bytes = _variant_bytes(image, perturbation, seed=17 + index)
            variant_record: dict[str, Any] = {"name": perturbation, "status": "FAILED", "evidence_status": "NOT_RUN_IN_ROBUSTNESS_SCOPE", "lesion_evidence_status": "NOT_RUN_IN_ROBUSTNESS_SCOPE", "vessel_evidence_status": "NOT_RUN_IN_ROBUSTNESS_SCOPE"}
            try:
                assessed = await quality.assess(variant_bytes)
                explanation = await classifier.explain_async(variant_bytes, target_class=base.prediction.predicted_grade)
                variant_gradcam_count += 1
                variant_attention = np.asarray(explanation.attention_map, dtype=np.float32)
                if variant_attention.shape != base_attention.shape:
                    variant_attention = np.asarray(Image.fromarray(variant_attention, mode="F").resize((base_attention.shape[1], base_attention.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
                difference = float(np.mean(np.abs(base_attention - variant_attention)))
                similarity = max(0.0, min(1.0, 1.0 - difference))
                cam_similarities.append(similarity)
                if explanation.prediction.predicted_grade != base.prediction.predicted_grade:
                    prediction_changes += 1
                guard = engine.evaluate(RetinaGuardInputs(quality_score=assessed.quality_score, raw_confidence=explanation.prediction.raw_confidence, probabilities=explanation.prediction.probabilities, predicted_grade=explanation.prediction.predicted_grade, predicted_grade_label=explanation.prediction.predicted_grade_label, model_version=explanation.prediction.model_version))
                artifact_name = f"sample_{index:02d}_{perturbation}_gradcam.png"
                _save_gradcam_overlay(output / "visuals" / artifact_name, image, variant_attention)
                variant_record.update({"status": "COMPLETED", "predicted_grade": explanation.prediction.predicted_grade, "predicted_grade_label": explanation.prediction.predicted_grade_label, "raw_confidence": explanation.prediction.raw_confidence, "quality_decision": assessed.quality_decision, "quality_score": assessed.quality_score, "prediction_unchanged": explanation.prediction.predicted_grade == base.prediction.predicted_grade, "grad_cam_mean_absolute_difference": round(difference, 8), "grad_cam_similarity": round(similarity, 8), "gradcam_artifact": str((output / "visuals" / artifact_name).relative_to(output)), "retinaguard_state": guard.trust_category, "retinaguard_score": guard.trust_score})
            except Exception as exc:
                variant_record["error"] = f"{type(exc).__name__}: {exc}"
            sample_record["variants"].append(variant_record)
        records.append(sample_record)
    completed = sum(1 for record in records for variant in record.get("variants", []) if variant.get("status") == "COMPLETED")
    requested = len(selected) * len(perturbations)
    aggregate = {
        "status": "CALCULATED" if records else "BLOCKED",
        "model": {"checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "architecture": "efficientnet_b0", "model_version": next((row.get("model_version") for row in rows if row.get("model_version")), "unversioned")},
        "sample_selection": {"requested_samples": samples, "selected_samples": len(selected), "policy": "one deterministic lexicographically first image per available adjudicated grade, then image_id sort; capped by requested samples"},
        "perturbations": perturbations,
        "requested_variant_count": requested,
        "completed_variant_count": completed,
        "failed_variant_count": requested - completed,
        "base_gradcam_count": base_gradcam_count,
        "variant_gradcam_count": variant_gradcam_count,
        "prediction_stability": round(1.0 - prediction_changes / completed, 8) if completed else None,
        "grad_cam_stability": round(float(mean(cam_similarities)), 8) if cam_similarities else None,
        "evidence_status": "NOT_RUN_IN_ROBUSTNESS_SCOPE",
        "note": "Controlled perturbation diagnostic on a small deterministic sample. Real classifier and Grad-CAM outputs were used; this is not a clinical robustness validation.",
    }
    stability = {"status": "COMPLETED" if completed else "FAILED", "method": "controlled brightness/contrast/blur/noise/rotation perturbations", "prediction_stability": aggregate["prediction_stability"], "grad_cam_stability": aggregate["grad_cam_stability"], "sample_count": len(selected), "variant_count": completed, "records": records, "note": "Explanation stability is an engineering diagnostic and does not establish causal faithfulness or clinical safety."}
    return aggregate, records, stability


def _write_report(output: Path, metrics: dict[str, Any], false_negatives: dict[str, Any], risk_coverage: dict[str, Any], robustness: dict[str, Any], checkpoint: Path) -> None:
    quality = metrics["quality"]
    lines = [
        "# RetinaGuard reliability validation",
        "",
        "Status: retrospective engineering validation of reliability signals. This document is not a clinical validation report and TRUSTED is not a correctness guarantee.",
        "",
        "## Immutable model contract",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Checkpoint SHA-256: `{metrics['checkpoint']['sha256']}`",
        f"- Expected SHA-256 unchanged: `{metrics['checkpoint']['sha256'] == EXPECTED_CHECKPOINT_SHA256}`",
        "- Classifier training weights were not retrained, fine-tuned, replaced, or modified by this validation.",
        "",
        "## Retrospective population",
        "",
        f"- Source predictions: `{metrics['source']['predictions']}`",
        f"- Dataset version: `{metrics['source']['dataset_version']}`",
        f"- Prediction rows: **{metrics['source']['row_count']}**",
        f"- Successful five-class reference rows: **{metrics['source']['eligible_row_count']}**",
        f"- Quality-readable images: **{quality['readable_file_count']}**; unreadable: **{quality['unreadable_file_count']}**",
        f"- Quality decisions: `{json.dumps(quality['decision_counts'], sort_keys=True)}`",
        f"- Reference class distribution: `{json.dumps(metrics['ground_truth_class_distribution'], sort_keys=True)}`",
        f"- Reference prediction distribution: `{json.dumps(metrics['prediction_class_distribution'], sort_keys=True)}`",
        f"- Duplicate audit: `{json.dumps(metrics['duplicates'], sort_keys=True)}`",
        "",
        "## Reliability signals",
        "",
        f"- Reliability states: `{json.dumps(metrics['reliability_state_counts'], sort_keys=True)}`",
        f"- Warning counts: `{json.dumps(metrics['warning_counts'], sort_keys=True)}`",
        f"- Mean raw confidence: `{metrics['raw_confidence']['mean']}`",
        f"- Mean calibrated confidence (identity/unfitted runtime unless configured otherwise): `{metrics['calibrated_confidence']['mean']}`",
        f"- Mean predictive uncertainty: `{metrics['uncertainty']['mean']}`",
        "- OOD reference: **UNAVAILABLE** unless an authorized reference is configured; no unfamiliar-image detection guarantee is claimed.",
        "- Lesion/evidence agreement: **UNAVAILABLE** in this retrospective source because Phase 4B predictions do not contain a supported lesion comparison.",
        "- Explanation stability: **NOT RUN** for the full retrospective population; controlled robustness sample is reported separately.",
        "",
        "## False-negative warning audit",
        "",
        f"- Reference false negatives under the unchanged Phase 4B grade-2 probability rule: **{false_negatives['false_negative_count']}**",
        f"- Warning coverage among those false negatives: `{false_negatives['false_negative_warning_coverage']}`",
        f"- Warning counts: `{json.dumps(false_negatives['warning_counts_among_false_negatives'], sort_keys=True)}`",
        "- No threshold, weight, model, or classifier output was optimized against these labels.",
        "",
        "## Risk coverage",
        "",
        f"`{json.dumps(risk_coverage['operating_points'], sort_keys=True)}`",
        "These fixed operating views are descriptive only and do not establish clinical risk coverage.",
        "",
        "## Robustness and Grad-CAM",
        "",
        f"- Robustness status: `{robustness['status']}`",
        f"- Controlled sample: **{robustness['sample_selection']['selected_samples']}** images; completed variants: **{robustness['completed_variant_count']} / {robustness['requested_variant_count']}**",
        f"- Prediction stability: `{robustness['prediction_stability']}`",
        f"- Grad-CAM stability: `{robustness['grad_cam_stability']}`",
        "- Grad-CAM artifacts are stored under `visuals/` and are linked to real checkpoint inference. Lesion and vessel evidence were intentionally not rerun in this robustness scope and are marked unavailable, not inferred.",
        "",
        "## Safety interpretation",
        "",
        "RetinaGuard is a transparent operating/review decision layer. TRUSTED means configured signals did not raise a major warning; it does not mean the model is correct. REVIEW_RECOMMENDED requires professional review, UNRELIABLE blocks automated interpretation or recommends recapture, and INSUFFICIENT_EVIDENCE indicates that required evidence was not available. Final clinical responsibility remains with a qualified clinician.",
    ]
    (output / "reliability_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_checkpoint(explicit: str | None) -> Path:
    configured = explicit or os.environ.get("CLASSIFIER_MODEL_PATH")
    path = Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"A real classifier checkpoint is required: {path}. Set CLASSIFIER_MODEL_PATH or pass --checkpoint.")
    actual = _sha256(path)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Refusing reliability validation because checkpoint SHA-256 changed. Expected {EXPECTED_CHECKPOINT_SHA256}, found {actual}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrospective RetinaGuard reliability and controlled robustness validation")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--robustness-samples", type=int, default=3)
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()
    checkpoint = _resolve_checkpoint(args.checkpoint)
    rows = _load_rows(args.predictions)
    quality_rows = asyncio.run(_assess_quality(rows, args.raw_dir))
    engine = RetinaGuardEngine()
    evaluated = _engine_rows(rows, quality_rows, engine)
    valid_reference = [item for item in evaluated if item["prediction_status"] == "SUCCESS" and item["ground_truth_grade"] in range(5)]
    state_counts = Counter(item["retinaguard"].get("reliability_state") for item in evaluated)
    warnings = Counter(flag.get("code", "unknown") for item in evaluated for flag in item["retinaguard"].get("risk_flags", []))
    uncertainties = [_float(item["retinaguard"].get("uncertainty", {}).get("score")) for item in evaluated]
    uncertainties = [value for value in uncertainties if value is not None]
    confidences = [item["raw_confidence"] for item in evaluated if item["raw_confidence"] is not None]
    source_dataset_version = _dataset_version()
    calibrated = [_float(item["retinaguard"].get("calibration", {}).get("calibrated_confidence")) for item in evaluated]
    calibrated = [value for value in calibrated if value is not None]
    disagreement_statuses = Counter(item["retinaguard"].get("model_disagreement", {}).get("status", "UNAVAILABLE") for item in evaluated)
    signal_statuses = {
        "image_quality": Counter(item["retinaguard"].get("image_quality_status", "UNAVAILABLE") for item in evaluated),
        "evidence": Counter(item["retinaguard"].get("evidence_status", "UNAVAILABLE") for item in evaluated),
        "explanation": Counter(item["retinaguard"].get("explanation_status", "UNAVAILABLE") for item in evaluated),
        "ood": Counter(item["retinaguard"].get("ood_status", "UNAVAILABLE") for item in evaluated),
    }
    ground_truth_distribution = Counter(str(item["ground_truth_grade"]) for item in valid_reference)
    prediction_distribution = Counter(str(item["predicted_grade"]) for item in valid_reference)
    metrics = {
        "status": "CALCULATED",
        "evaluation_type": "retrospective_reliability_validation",
        "clinical_validation_claim": False,
        "generated_by": "scripts/evaluate_reliability.py",
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "software": _software(),
        "source": {"predictions": str(args.predictions), "raw_dataset": str(args.raw_dir), "dataset_version": source_dataset_version, "row_count": len(rows), "eligible_row_count": len(valid_reference)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint), "expected_sha256": EXPECTED_CHECKPOINT_SHA256, "unchanged": _sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256},
        "quality": _quality_summary(quality_rows),
        "ground_truth_class_distribution": dict(sorted(ground_truth_distribution.items())),
        "prediction_class_distribution": dict(sorted(prediction_distribution.items())),
        "duplicates": _duplicate_summary(),
        "reliability_state_counts": dict(sorted(state_counts.items(), key=lambda item: str(item[0]))),
        "warning_counts": dict(sorted(warnings.items())),
        "raw_confidence": _summary(confidences),
        "calibrated_confidence": _summary(calibrated),
        "uncertainty": _summary(uncertainties),
        "model_disagreement_status_counts": dict(sorted(disagreement_statuses.items())),
        "signal_status_counts": {name: dict(sorted(values.items())) for name, values in signal_statuses.items()},
        "configuration": {"engine_version": engine.VERSION, "weights": engine.weights, "missing_signal_score": engine.missing_signal_score, "trusted_threshold": engine.trusted_threshold, "unreliable_threshold": engine.unreliable_threshold, "clinical_validation_claim": False},
        "reference_model_metrics": json.loads((ROOT / "ml" / "evaluation" / "messidor" / "zero_shot_metrics.json").read_text(encoding="utf-8")) if (ROOT / "ml" / "evaluation" / "messidor" / "zero_shot_metrics.json").is_file() else {"status": "UNAVAILABLE"},
        "note": "Retrospective reliability signals only. Missing evidence is represented explicitly and never replaced with a fabricated positive signal.",
    }
    false_negatives = _false_negative_analysis(evaluated)
    risk_coverage = _risk_coverage(evaluated)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "reliability_metrics.json", metrics)
    _json(output / "false_negative_warning_analysis.json", false_negatives)
    _json(output / "risk_coverage_analysis.json", risk_coverage)
    _json(output / "reliability_configuration.json", {"format": "retinaguard-reliability-configuration-v1", **metrics["configuration"], "state_policy": {"TRUSTED": "AUTOMATED_RESULT_AVAILABLE", "REVIEW_RECOMMENDED": "PROFESSIONAL_REVIEW_RECOMMENDED", "UNRELIABLE": "AUTOMATED_INTERPRETATION_UNRELIABLE or IMAGE_RECAPTURE_RECOMMENDED when quality is low", "INSUFFICIENT_EVIDENCE": "PROFESSIONAL_REVIEW_RECOMMENDED"}, "source": "RetinaGuardEngine runtime configuration", "note": "Weights and thresholds are disclosed engineering settings, not clinically validated operating points."})
    if args.skip_robustness:
        robustness = {"status": "SKIPPED", "reason": "--skip-robustness was supplied; no Grad-CAM perturbation inference was run.", "sample_selection": {"selected_samples": 0}, "completed_variant_count": 0, "requested_variant_count": 0, "prediction_stability": None, "grad_cam_stability": None}
        perturbations: list[dict[str, Any]] = []
        stability = {"status": "SKIPPED", "reason": "--skip-robustness was supplied."}
    else:
        robustness, perturbations, stability = asyncio.run(_robustness(rows, args.raw_dir, output, checkpoint, max(1, min(5, args.robustness_samples))))
    _json(output / "robustness_test_results.json", robustness)
    _json(output / "perturbation_results.json", perturbations)
    _json(output / "explanation_stability.json", stability)
    _write_report(output, metrics, false_negatives, risk_coverage, robustness, checkpoint)
    print(json.dumps({"output": str(output), "dataset_rows": len(rows), "quality_readable": metrics["quality"]["readable_file_count"], "reliability_states": metrics["reliability_state_counts"], "robustness": robustness}, indent=2, default=str))


if __name__ == "__main__":
    main()
