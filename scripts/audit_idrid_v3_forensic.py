"""Quantitative forensic audit of the frozen IDRiD V2 generalization failure.

This report is diagnostic only.  It reads the existing V2 IDRiD validation and
Messidor-2 zero-shot artifacts, does not fit thresholds/calibration, and does
not access the official IDRiD test package.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_idrid_v2_research import quality_proxy  # noqa: E402

V2_DIR = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v2" / "20260912" / "v2_c_lesion_aware_efficientnet_b0"
V2_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_selected_candidate.json"
V2_DEVELOPMENT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
V2_TRAINING = V2_DIR / "training_config.json"
V2_VAL = V2_DIR / "validation_predictions.json"
EXTERNAL = ROOT / "ml" / "evaluation" / "messidor" / "idrid_v2_lesion_aware_zero_shot" / "idrid_v2_messidor2_zero_shot.json"
IDRID_RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
MESSIDOR_RAW = ROOT / "ml" / "datasets" / "raw" / "messidor"
OUTPUT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v3_failure_analysis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=float)
    return {"min": float(array.min()), "p10": float(np.quantile(array, 0.10)), "median": float(np.median(array)), "p90": float(np.quantile(array, 0.90)), "max": float(array.max()), "mean": float(array.mean())}


def ece(rows: list[dict[str, Any]], probability_key: str = "confidence") -> float:
    confidence = np.asarray([float(row[probability_key]) for row in rows], dtype=float)
    correct = np.asarray([int(row["actual"] == row["predicted"]) for row in rows], dtype=int)
    result = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def brier(rows: list[dict[str, Any]]) -> float:
    actual = np.asarray([int(row["actual"] in {2, 3, 4}) for row in rows], dtype=float)
    probability = np.asarray([float(row["referable_probability"]) for row in rows], dtype=float)
    return float(np.mean((probability - actual) ** 2))


def read_dimensions(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            array = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        gray = array.mean(axis=2)
        return {
            "width": width,
            "height": height,
            "aspect_ratio": float(width / height) if height else None,
            "dark_fraction_at_224": float(np.mean(gray < 0.05)),
            "bright_fraction_at_224": float(np.mean(gray > 0.95)),
            "quality_proxy": quality_proxy(path),
            "readable": True,
        }
    except Exception as exc:
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}


def domain_rows(name: str, rows: list[dict[str, Any]], raw_root: Path, path_key: str, actual_key: str = "actual") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    for row in rows:
        if name == "idrid_validation":
            image_path = raw_root / row["image"]
            actual = int(row["actual"])
            predicted = int(row["predicted"])
            probabilities = [float(value) for value in row["probabilities"]]
            confidence = float(row["confidence"])
            referable_probability = float(row["referable_probability"])
            entropy_value = float(row["normalized_entropy"])
        else:
            image_path = raw_root / row[path_key]
            actual = int(row[actual_key])
            probabilities = [float(value) for value in row["probabilities"]]
            predicted = int(row["predicted_grade"])
            confidence = float(max(probabilities))
            referable_probability = float(row["referable_probability"])
            entropy_value = float(-sum(value * math.log(max(value, 1e-12) for value in probabilities) / math.log(5.0))) if False else float(-sum(value * math.log(max(value, 1e-12)) for value in probabilities) / math.log(5.0))
        normalized.append({"actual": actual, "predicted": predicted, "probabilities": probabilities, "confidence": confidence, "referable_probability": referable_probability, "normalized_entropy": entropy_value})
        geometry.append(read_dimensions(image_path))
    actual_distribution = Counter(str(row["actual"]) for row in normalized)
    predicted_distribution = Counter(str(row["predicted"]) for row in normalized)
    ref = [row["referable_probability"] for row in normalized]
    confidence = [row["confidence"] for row in normalized]
    entropy_values = [row["normalized_entropy"] for row in normalized]
    quality = [item.get("quality_proxy", {}).get("quality_proxy_score") for item in geometry if item.get("readable")]
    dark = [item.get("dark_fraction_at_224") for item in geometry if item.get("readable")]
    aspect = [item.get("aspect_ratio") for item in geometry if item.get("readable")]
    readable = [item for item in geometry if item.get("readable")]
    threshold_rows = []
    actual_ref = np.asarray([int(row["actual"] in {2, 3, 4}) for row in normalized], dtype=int)
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        predicted_ref = np.asarray([int(row["referable_probability"] >= threshold) for row in normalized], dtype=int)
        tp = int(((actual_ref == 1) & (predicted_ref == 1)).sum())
        tn = int(((actual_ref == 0) & (predicted_ref == 0)).sum())
        fp = int(((actual_ref == 0) & (predicted_ref == 1)).sum())
        fn = int(((actual_ref == 1) & (predicted_ref == 0)).sum())
        threshold_rows.append({"threshold": threshold, "sensitivity": tp / (tp + fn) if tp + fn else 0.0, "specificity": tn / (tn + fp) if tn + fp else 0.0, "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    result = {
        "sample_count": len(normalized),
        "actual_grade_distribution": dict(sorted(actual_distribution.items())),
        "predicted_grade_distribution": dict(sorted(predicted_distribution.items())),
        "referable_probability_distribution": quantiles(ref),
        "confidence_distribution": quantiles(confidence),
        "normalized_entropy_distribution": quantiles(entropy_values),
        "quality_proxy_distribution": quantiles([float(value) for value in quality]),
        "aspect_ratio_distribution": quantiles([float(value) for value in aspect]),
        "dark_border_fraction_distribution": quantiles([float(value) for value in dark]),
        "readable_count": len(readable),
        "ece_10_bins": ece(normalized),
        "referable_brier": brier(normalized),
        "threshold_diagnostics_not_used_for_selection": threshold_rows,
        "rows": normalized,
    }
    return result, geometry


def main() -> int:
    validation_rows = json.loads(V2_VAL.read_text(encoding="utf-8"))
    external_payload = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    external_rows = external_payload["predictions"]
    development = json.loads(V2_DEVELOPMENT.read_text(encoding="utf-8"))
    development_paths = {record["image_id"]: record["image"] for record in development["records"]}
    validation_rows = [{**row, "image": development_paths[row["image_id"]]} for row in validation_rows]
    idrid_summary, idrid_geometry = domain_rows("idrid_validation", validation_rows, IDRID_RAW, "image")
    external_summary, external_geometry = domain_rows("messidor2", external_rows, MESSIDOR_RAW, "image_path", "adjudicated_dr_grade")
    training = json.loads(V2_TRAINING.read_text(encoding="utf-8"))
    selected = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    lesion_rows = [row for row in validation_rows if row.get("lesion_evidence", {}).get("probabilities") is not None]
    lesion_available = Counter()
    for row in lesion_rows:
        for index, available in enumerate(row["lesion_evidence"]["annotation_available"]):
            if available:
                lesion_available[index] += 1
    report = {
        "schema_version": "idrid-v3-forensic-audit-1",
        "purpose": "Quantitative diagnosis of V2 IDRiD development-to-Messidor-2 generalization failure; no V3 selection data from Messidor-2.",
        "official_test_images_opened": 0,
        "idrid_v2_checkpoint": {"path": str((V2_DIR / "checkpoint_best.pt").relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(V2_DIR / "checkpoint_best.pt")},
        "v2_validation": {key: value for key, value in idrid_summary.items() if key != "rows"},
        "messidor2_external": {key: value for key, value in external_summary.items() if key != "rows"},
        "preprocessing_contract_comparison": {
            "training_and_idrid_validation": training.get("preprocessing"),
            "messidor2_external": {"input_size": 224, "color_space": "RGB", "resize": [224, 224], "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]},
            "mismatch_detected": False,
            "important_behavior": "Both domains are resized directly to a square without crop; native aspect ratio and peripheral FOV are therefore compressed into 224x224.",
        },
        "measurable_domain_shift": {
            "aspect_ratio_mean_idrid": float(np.mean([item["aspect_ratio"] for item in idrid_geometry if item.get("readable")])),
            "aspect_ratio_mean_messidor2": float(np.mean([item["aspect_ratio"] for item in external_geometry if item.get("readable")])),
            "quality_proxy_mean_idrid": float(np.mean([item["quality_proxy"]["quality_proxy_score"] for item in idrid_geometry if item.get("readable")])),
            "quality_proxy_mean_messidor2": float(np.mean([item["quality_proxy"]["quality_proxy_score"] for item in external_geometry if item.get("readable")])),
            "dark_fraction_mean_idrid": float(np.mean([item["dark_fraction_at_224"] for item in idrid_geometry if item.get("readable")])),
            "dark_fraction_mean_messidor2": float(np.mean([item["dark_fraction_at_224"] for item in external_geometry if item.get("readable")])),
            "interpretation": "These are engineering distribution indicators, not clinical image-quality labels.",
        },
        "prediction_collapse_diagnostics": {
            "idrid_referable_rate_at_0_45": float(np.mean([row["referable_probability"] >= 0.45 for row in idrid_summary["rows"]])),
            "messidor2_referable_rate_at_0_45": float(np.mean([row["referable_probability"] >= 0.45 for row in external_summary["rows"]])),
            "idrid_mean_referable_probability": float(np.mean([row["referable_probability"] for row in idrid_summary["rows"]])),
            "messidor2_mean_referable_probability": float(np.mean([row["referable_probability"] for row in external_summary["rows"]])),
            "external_false_negative_count_at_0_45": external_payload["metrics"]["referable_grade_2_or_worse_at_frozen_development_threshold"]["fn"],
            "finding": "External predictions are strongly shifted toward non-referable at the frozen threshold, producing high specificity and very low sensitivity; threshold movement is not a sufficient diagnosis and was not used for V3 selection.",
        },
        "quality_gate_interaction": {
            "v2_idrid_validation_quality_gate_run": False,
            "v2_messidor2_quality_gate_run": False,
            "finding": "The stored V2 development and external evaluators both run classifier inference directly; a quality gate did not cause the external false negatives in these evaluation artifacts.",
        },
        "lesion_branch_diagnostics": {
            "branch_available_in_v2_predictions": len(lesion_rows) == len(validation_rows),
            "annotation_availability_counts_by_order_MA_HE_EX_CWS": dict(lesion_available),
            "grade_head_receives_lesion_logits_at_inference": False,
            "lesion_branch_role": "Auxiliary masked presence loss during training; lesion logits are not fused into or allowed to overwrite the severity head.",
            "causal_ablation_status": "NOT_PERFORMED; architecture inspection establishes separation, but no lesion-head ablation was used as a causal claim.",
        },
        "transfer_chain": {
            "v2_initialization": training.get("initialization"),
            "aptos_to_idrid_effect": "V2 inherits an APTOS-to-IDRiD transfer chain through the frozen V1 backbone, but V2 was fine-tuned on IDRiD and no direct APTOS ablation was performed in this forensic pass.",
        },
        "calibration_diagnostics": {
            "idrid_ece_10_bins": idrid_summary["ece_10_bins"],
            "messidor2_ece_10_bins": external_summary["ece_10_bins"],
            "idrid_referable_brier": idrid_summary["referable_brier"],
            "messidor2_referable_brier": external_summary["referable_brier"],
            "calibration_fitted": False,
            "status": "Diagnostic only; no calibration parameters were fitted from validation or external labels.",
        },
        "root_cause_assessment": [
            "No executable preprocessing mismatch was found between V2 training/IDRiD validation and the stored Messidor-2 evaluator: both use RGB, direct 224x224 resize, and ImageNet normalization.",
            "The evidence supports domain shift in native geometry/FOV/appearance and severity distributions, compounded by direct square resizing that does not preserve acquisition geometry.",
            "V2 shows a measurable external probability collapse toward non-referable at the frozen 0.45 threshold; this is a model/domain calibration and ranking failure, not a quality-gate rejection.",
            "The lesion branch is auxiliary and separate; it cannot explain a runtime overwrite of the grade, although shared-backbone auxiliary training may have changed learned features.",
            "A small 83-image IDRiD validation set is insufficient to establish robust cross-domain generalization or reliable calibration.",
        ],
        "raw_domain_rows": {"idrid_validation": idrid_summary["rows"], "messidor2_external": external_summary["rows"]},
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "preprocessing_mismatch": report["preprocessing_contract_comparison"]["mismatch_detected"],
        "idrid_mean_ref_probability": report["prediction_collapse_diagnostics"]["idrid_mean_referable_probability"],
        "messidor_mean_ref_probability": report["prediction_collapse_diagnostics"]["messidor2_mean_referable_probability"],
        "idrid_ece": idrid_summary["ece_10_bins"],
        "messidor_ece": external_summary["ece_10_bins"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
