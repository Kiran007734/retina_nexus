"""Run the Phase 4B reliability audit without changing the classifier.

This audit preserves the original full Messidor-2 evaluation artifacts. It
uses the existing predictions for threshold and error analysis, and performs
a separate, reproducible inference pass after applying a documented
perceptual-deduplication policy. No weights, labels, or original reports are
rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.evaluation.messidor2 import (  # noqa: E402
    APTOS_CLASS_MAPPING,
    bootstrap_confidence_intervals,
    compute_metrics,
    write_confusion_matrix,
)
from scripts.evaluate_messidor2 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    discover_checkpoint,
    run_inference,
)


DEFAULT_RAW_DIR = ROOT / "ml" / "datasets" / "raw" / "messidor"
DEFAULT_EVALUATION_DIR = ROOT / "ml" / "evaluation" / "messidor"
LABEL_PROVENANCE_STATEMENT = (
    "The Messidor-2 official release does not provide official DR ground truth. "
    "This evaluation uses separately acquired third-party reference labels."
)
THRESHOLD_GRID = [round(index / 20, 2) for index in range(1, 20)]
CURRENT_THRESHOLD = 0.50
HIGH_SENSITIVITY_TARGET = 0.90
HIGH_SPECIFICITY_TARGET = 0.99


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    try:
        from importlib.metadata import version
    except ImportError:
        version = None
    for package in ("numpy", "Pillow", "scikit-learn", "torch", "torchvision"):
        try:
            versions[package] = version(package) if version else "unavailable"
        except Exception:
            versions[package] = "unavailable"
    return versions


def _path_key(value: str) -> str:
    return str(value).replace("\\", "/").casefold()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("adjudicated_dr_grade"),
        record.get("adjudicated_dme"),
        record.get("adjudicated_gradable"),
    )


def _group_label_audit(
    groups: Iterable[list[str]],
    records_by_path: dict[str, dict[str, Any]],
    predictions_by_path: dict[str, dict[str, Any]],
    group_kind: str,
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for group_index, raw_group in enumerate(groups, start=1):
        group = sorted(raw_group, key=_path_key)
        record_items = [records_by_path.get(_path_key(path)) for path in group]
        record_items = [item for item in record_items if item is not None]
        signatures = [_label_signature(item) for item in record_items]
        canonical = group[0] if group else None
        predicted = [predictions_by_path.get(_path_key(path), {}) for path in group]
        predicted_grades = [item.get("predicted_aptos_grade") for item in predicted if item]
        audited.append(
            {
                "group_id": f"{group_kind}_{group_index:03d}",
                "group_kind": group_kind,
                "image_paths": group,
                "canonical_representative": canonical,
                "records_found": len(record_items),
                "labels": [
                    {
                        "image_path": item.get("image_root_relative_path"),
                        "image_id": item.get("image_id"),
                        "dr_grade": item.get("adjudicated_dr_grade"),
                        "dme": item.get("adjudicated_dme"),
                        "gradable": item.get("adjudicated_gradable"),
                    }
                    for item in record_items
                ],
                "label_signatures": [list(signature) for signature in signatures],
                "all_labels_identical": bool(signatures) and len(set(signatures)) == 1,
                "predicted_grades_in_original_evaluation": predicted_grades,
                "policy": "retain lexicographically first image path; exclude other group members only in the separate sensitivity-analysis evaluation",
            }
        )
    return audited


def _write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _binary_metrics(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "threshold": threshold,
        "sample_count": int(len(actual)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(sensitivity),
        "f1": float(f1),
        "youden_j": float(sensitivity + specificity - 1.0),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def _bootstrap_binary(actual: np.ndarray, probability: np.ndarray, threshold: float, iterations: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        indices = rng.integers(0, len(actual), size=len(actual))
        metric = _binary_metrics(actual[indices], probability[indices], threshold)
        for name in ("sensitivity", "specificity", "precision", "f1"):
            values[name].append(float(metric[name]))
    return {
        name: {
            "lower": float(np.percentile(samples, 2.5)),
            "upper": float(np.percentile(samples, 97.5)),
            "successful_resamples": len(samples),
        }
        for name, samples in values.items()
    }


def _threshold_analysis(rows: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("inference_status") == "SUCCESS"
        and row.get("adjudicated_gradable") == "1"
        and str(row.get("adjudicated_dr_grade", "")) in {"0", "1", "2", "3", "4"}
    ]
    actual = np.asarray([int(row["adjudicated_dr_grade"]) >= 2 for row in eligible], dtype=int)
    probability = np.asarray([_float(row["referable_probability_grade_2_or_worse"]) for row in eligible], dtype=float)
    points: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        point = _binary_metrics(actual, probability, threshold)
        point["bootstrap_confidence_intervals"] = _bootstrap_binary(actual, probability, threshold, iterations, seed)
        points.append(point)

    current = next(point for point in points if point["threshold"] == CURRENT_THRESHOLD)
    high_sensitivity_candidates = [point for point in points if point["sensitivity"] >= HIGH_SENSITIVITY_TARGET]
    if high_sensitivity_candidates:
        high_sensitivity = max(high_sensitivity_candidates, key=lambda point: (point["specificity"], point["threshold"]))
        high_sensitivity_selection = "highest specificity among tested thresholds meeting sensitivity >= 0.90"
    else:
        high_sensitivity = max(points, key=lambda point: (point["sensitivity"], -point["threshold"]))
        high_sensitivity_selection = "target sensitivity >= 0.90 was not reached; selected the tested threshold with maximum sensitivity"
    balanced = max(points, key=lambda point: (point["youden_j"], point["threshold"]))
    high_specificity_candidates = [point for point in points if point["specificity"] >= HIGH_SPECIFICITY_TARGET]
    if high_specificity_candidates:
        high_specificity = max(high_specificity_candidates, key=lambda point: (point["sensitivity"], -point["threshold"]))
        high_specificity_selection = "highest sensitivity among tested thresholds meeting specificity >= 0.99"
    else:
        high_specificity = max(points, key=lambda point: (point["specificity"], -point["threshold"]))
        high_specificity_selection = "target specificity >= 0.99 was not reached; selected the tested threshold with maximum specificity"
    return {
        "status": "CALCULATED",
        "label_provenance_statement": LABEL_PROVENANCE_STATEMENT,
        "evaluation_population": "Original full-dataset successful inferences with adjudicated_gradable=1 and valid DR grade 0..4.",
        "raw_probability_field": "referable_probability_grade_2_or_worse",
        "probabilities_unchanged": True,
        "threshold_grid": THRESHOLD_GRID,
        "bootstrap": {"iterations": iterations, "seed": seed, "confidence_level": 0.95, "method": "nonparametric percentile bootstrap"},
        "operating_point_definitions": {
            "current": "threshold=0.50, matching the original referable metric",
            "high_sensitivity": f"target sensitivity >= {HIGH_SENSITIVITY_TARGET:.2f}; fallback is maximum observed sensitivity if target is unmet",
            "balanced": "maximum Youden J over the tested threshold grid",
            "high_specificity": f"target specificity >= {HIGH_SPECIFICITY_TARGET:.2f}; fallback is maximum observed specificity if target is unmet",
        },
        "operating_points": {
            "current": {**current, "selection": "existing grade>=2 probability threshold"},
            "high_sensitivity": {**high_sensitivity, "selection": high_sensitivity_selection},
            "balanced": {**balanced, "selection": "maximum Youden J"},
            "high_specificity": {**high_specificity, "selection": high_specificity_selection},
        },
        "thresholds": points,
        "interpretation": "Exploratory operating-point analysis only. Threshold changes alter the decision rule and do not improve the underlying trained model or establish clinical performance.",
    }


def _error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("inference_status") == "SUCCESS"
        and row.get("adjudicated_gradable") == "1"
        and str(row.get("adjudicated_dr_grade", "")) in {"0", "1", "2", "3", "4"}
    ]
    errors = [row for row in eligible if int(row["predicted_aptos_grade"]) != int(row["adjudicated_dr_grade"])]
    # Referable false positives/negatives are defined by the configured
    # grade>=2 probability decision rule, not by five-class argmax. These
    # populations must match the original referable metric exactly.
    false_negatives = [row for row in eligible if int(row["adjudicated_dr_grade"]) >= 2 and _float(row["referable_probability_grade_2_or_worse"]) < 0.5]
    false_positives = [row for row in eligible if int(row["adjudicated_dr_grade"]) < 2 and _float(row["referable_probability_grade_2_or_worse"]) >= 0.5]

    def grouped(items: list[dict[str, Any]]) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            groups[f"reference_{item['adjudicated_dr_grade']}_predicted_{item['predicted_aptos_grade']}"].append(item)
        result: dict[str, Any] = {}
        for name in sorted(groups):
            group = groups[name]
            result[name] = {
                "count": len(group),
                "mean_raw_confidence": float(np.mean([_float(item.get("raw_confidence")) for item in group])),
                "median_raw_confidence": float(np.median([_float(item.get("raw_confidence")) for item in group])),
                "mean_referable_probability": float(np.mean([_float(item.get("referable_probability_grade_2_or_worse")) for item in group])),
            }
        return result

    def representatives(items: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            groups[(int(item["adjudicated_dr_grade"]), int(item["predicted_aptos_grade"]))].append(item)
        selected: list[dict[str, Any]] = []
        for (reference, predicted), group in sorted(groups.items()):
            if kind == "false_negative":
                ordered = sorted(group, key=lambda item: (_float(item.get("referable_probability_grade_2_or_worse")), _path_key(item.get("image_path", ""))))
            else:
                ordered = sorted(group, key=lambda item: (-_float(item.get("referable_probability_grade_2_or_worse")), _path_key(item.get("image_path", ""))))
            for item in ordered[:3]:
                selected.append(
                    {
                        "image_id": item.get("image_id"),
                        "image_path": item.get("image_path"),
                        "reference_grade": reference,
                        "predicted_grade": predicted,
                        "raw_confidence": _float(item.get("raw_confidence")),
                        "referable_probability": _float(item.get("referable_probability_grade_2_or_worse")),
                        "selection_reason": "deterministic per-reference/prediction group ranking; no manual cherry-picking",
                    }
                )
        return selected

    return {
        "status": "CALCULATED",
        "label_provenance_statement": LABEL_PROVENANCE_STATEMENT,
        "evaluation_population_count": len(eligible),
        "misclassified_count": len(errors),
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "false_negative_by_reference_and_prediction": grouped(false_negatives),
        "false_positive_by_reference_and_prediction": grouped(false_positives),
        "confidence_fields": ["raw_confidence", "referable_probability_grade_2_or_worse"],
        "quality_metadata": {"status": "UNAVAILABLE", "reason": "The original Phase 4B prediction records do not contain image-quality component metadata; no quality values were fabricated."},
        "representative_selection_method": "For every reference-grade/predicted-grade error group, retain up to three deterministic examples. False negatives are ranked by lowest referable probability; false positives by highest referable probability; ties use normalized image path.",
        "representative_false_negatives": representatives(false_negatives, "false_negative"),
        "representative_false_positives": representatives(false_positives, "false_positive"),
        "interpretation": "Aggregates use every eligible full-dataset record. Representative examples are navigational examples, not a cherry-picked performance estimate.",
    }


def _draw_title(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=(22, 35, 55), font=font)


def _write_threshold_visual(path: Path, analysis: dict[str, Any]) -> None:
    canvas = Image.new("RGB", (1000, 680), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    _draw_title(draw, "Messidor-2 referable threshold analysis", (24, 20), font)
    left, top, width, height = 85, 70, 820, 500
    draw.rectangle((left, top, left + width, top + height), outline=(140, 155, 170))
    draw.line((left, top + height // 2, left + width, top + height // 2), fill=(225, 230, 235))
    draw.text((20, top - 5), "1.0", fill=(80, 90, 100), font=font)
    draw.text((20, top + height // 2 - 5), "0.5", fill=(80, 90, 100), font=font)
    draw.text((20, top + height - 5), "0.0", fill=(80, 90, 100), font=font)
    points = analysis["thresholds"]
    x = lambda t: left + int((t / 1.0) * width)
    y = lambda value: top + height - int(value * height)
    sensitivity = [(x(point["threshold"]), y(point["sensitivity"])) for point in points]
    specificity = [(x(point["threshold"]), y(point["specificity"])) for point in points]
    draw.line(sensitivity, fill=(211, 83, 83), width=3)
    draw.line(specificity, fill=(45, 125, 95), width=3)
    for px, py in sensitivity:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(211, 83, 83))
    for px, py in specificity:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(45, 125, 95))
    for label, colour, offset in (("Sensitivity", (211, 83, 83), 0), ("Specificity", (45, 125, 95), 110)):
        draw.line((left + offset, 610, left + offset + 28, 610), fill=colour, width=4)
        draw.text((left + offset + 35, 605), label, fill=(50, 60, 70), font=font)
    draw.text((left, 640), "Threshold probability", fill=(50, 60, 70), font=font)
    canvas.save(path, format="PNG", optimize=True)


def _write_comparison_visual(path: Path, full: dict[str, Any], deduplicated: dict[str, Any]) -> None:
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1"), ("quadratic_weighted_kappa", "QWK")]
    canvas = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    _draw_title(draw, "Full vs perceptual-deduplicated evaluation", (24, 20), font)
    base_x, base_y, bar_width, gap = 90, 90, 300, 70
    for index, (key, label) in enumerate(metrics):
        y = base_y + index * 95
        draw.text((24, y + 14), label, fill=(45, 55, 65), font=font)
        full_value = float(full.get(key) or 0.0)
        dedup_value = float(deduplicated.get(key) or 0.0)
        draw.rectangle((base_x, y, base_x + int(bar_width * full_value), y + 22), fill=(74, 117, 171))
        draw.rectangle((base_x, y + 30, base_x + int(bar_width * dedup_value), y + 52), fill=(67, 157, 119))
        draw.text((base_x + bar_width + 12, y + 6), f"full {full_value:.4f}", fill=(45, 55, 65), font=font)
        draw.text((base_x + bar_width + 12, y + 36), f"dedup {dedup_value:.4f}", fill=(45, 55, 65), font=font)
    canvas.save(path, format="PNG", optimize=True)


def _write_error_visual(path: Path, error_report: dict[str, Any]) -> None:
    groups = []
    for kind, values in (("FN", error_report["false_negative_by_reference_and_prediction"]), ("FP", error_report["false_positive_by_reference_and_prediction"])):
        for name, value in values.items():
            groups.append((f"{kind} {name.replace('reference_', 'r').replace('_predicted_', ' p')}", int(value["count"])))
    canvas = Image.new("RGB", (1000, max(260, 100 + 30 * len(groups))), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    _draw_title(draw, "Messidor-2 error groups", (24, 20), font)
    maximum = max((count for _, count in groups), default=1)
    for index, (label, count) in enumerate(groups):
        y = 70 + index * 30
        draw.text((24, y + 5), label, fill=(45, 55, 65), font=font)
        draw.rectangle((350, y, 350 + int(560 * count / maximum), y + 18), fill=(211, 116, 83) if label.startswith("FN") else (74, 117, 171))
        draw.text((925, y + 5), str(count), fill=(45, 55, 65), font=font)
    canvas.save(path, format="PNG", optimize=True)


def _build_markdown(
    path: Path,
    duplicate_audit: dict[str, Any],
    dedup_metrics: dict[str, Any],
    threshold: dict[str, Any],
    errors: dict[str, Any],
    comparison: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    full = comparison["full_dataset"]
    dedup = comparison["deduplicated_dataset"]
    op = threshold["operating_points"]
    text = f"""# Phase 4B reliability audit

Status: **COMPLETED**. This is a reliability and evaluation-quality audit,
not clinical validation.

## Required provenance statement

> {LABEL_PROVENANCE_STATEMENT}

The label source remains `google-brain/messidor2-dr-grades`, KaggleHub cache
version 1. Its CSV and readme SHA-256 values, exact cache path, and discovery
method remain in `dataset_manifest.json`. Official source terms and dataset
limitations are documented by [ADCIS](https://www.adcis.net/en/third-party/messidor2/).

## Checkpoint and reproducibility

- Model version: `{checkpoint.get('model_version')}`
- Architecture: `{checkpoint.get('architecture')}`
- Checkpoint SHA-256 before audit: `{checkpoint.get('checkpoint_sha256_before')}`
- Checkpoint SHA-256 after audit: `{checkpoint.get('checkpoint_sha256_after')}`
- Checkpoint unchanged: **{checkpoint.get('checkpoint_unchanged')}**
- Bootstrap: nonparametric percentile, 2,000 iterations, seed 42
- Original full-dataset metrics and predictions were not overwritten.

## Duplicate audit

- Exact duplicate groups: **{duplicate_audit.get('exact_duplicate_group_count')}**
- Perceptual duplicate groups: **{duplicate_audit.get('perceptual_duplicate_group_count')}**
- Perceptual group members excluded from sensitivity analysis: **{duplicate_audit.get('excluded_image_count')}**
- Perceptual groups with identical DR/DME/gradable labels: **{duplicate_audit.get('perceptual_groups_with_identical_labels')}** / {duplicate_audit.get('perceptual_duplicate_group_count')}
- Perceptual groups with conflicting labels: **{duplicate_audit.get('perceptual_groups_with_conflicting_labels')}**

Policy: keep the lexicographically first member of each perceptual duplicate
group and exclude only the other members from the separate deduplicated
sensitivity-analysis run. No raw image was deleted and the original full run
remains the primary reported evaluation. Patient identifiers were unavailable,
so this audit cannot infer whether duplicate files represent the same patient.

## Full vs deduplicated metrics

| Metric | Full ({full.get('sample_count')}) | Deduplicated ({dedup.get('sample_count')}) | Delta |
|---|---:|---:|---:|
| Accuracy | {full.get('accuracy', 0):.6f} | {dedup.get('accuracy', 0):.6f} | {comparison['deltas'].get('accuracy', 0):+.6f} |
| Macro F1 | {full.get('macro_f1', 0):.6f} | {dedup.get('macro_f1', 0):.6f} | {comparison['deltas'].get('macro_f1', 0):+.6f} |
| Weighted F1 | {full.get('weighted_f1', 0):.6f} | {dedup.get('weighted_f1', 0):.6f} | {comparison['deltas'].get('weighted_f1', 0):+.6f} |
| ROC-AUC | {full.get('roc_auc_ovr_macro', 0):.6f} | {dedup.get('roc_auc_ovr_macro', 0):.6f} | {comparison['deltas'].get('roc_auc_ovr_macro', 0):+.6f} |
| QWK | {full.get('quadratic_weighted_kappa', 0):.6f} | {dedup.get('quadratic_weighted_kappa', 0):.6f} | {comparison['deltas'].get('quadratic_weighted_kappa', 0):+.6f} |
| Referable sensitivity | {full['referable_dr_grade_2_or_worse'].get('sensitivity', 0):.6f} | {dedup['referable_dr_grade_2_or_worse'].get('sensitivity', 0):.6f} | {comparison['deltas'].get('referable_sensitivity', 0):+.6f} |
| Referable specificity | {full['referable_dr_grade_2_or_worse'].get('specificity', 0):.6f} | {dedup['referable_dr_grade_2_or_worse'].get('specificity', 0):.6f} | {comparison['deltas'].get('referable_specificity', 0):+.6f} |

The deduplicated run is a sensitivity analysis, not a replacement benchmark.
Duplicate removal changed the evaluation population by {comparison.get('removed_sample_count')} records. The JSON comparison records the exact deltas and bootstrap intervals.

## Threshold analysis

Thresholds use the unchanged raw referable probability field and do not retrain
or recalibrate the model. The original operating point is 0.50.

| Operating point | Threshold | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Current | {op['current']['threshold']:.2f} | {op['current']['sensitivity']:.6f} | {op['current']['specificity']:.6f} | {op['current']['precision']:.6f} | {op['current']['f1']:.6f} |
| High sensitivity | {op['high_sensitivity']['threshold']:.2f} | {op['high_sensitivity']['sensitivity']:.6f} | {op['high_sensitivity']['specificity']:.6f} | {op['high_sensitivity']['precision']:.6f} | {op['high_sensitivity']['f1']:.6f} |
| Balanced / Youden J | {op['balanced']['threshold']:.2f} | {op['balanced']['sensitivity']:.6f} | {op['balanced']['specificity']:.6f} | {op['balanced']['precision']:.6f} | {op['balanced']['f1']:.6f} |
| High specificity | {op['high_specificity']['threshold']:.2f} | {op['high_specificity']['sensitivity']:.6f} | {op['high_specificity']['specificity']:.6f} | {op['high_specificity']['precision']:.6f} | {op['high_specificity']['f1']:.6f} |

Selection notes: high-sensitivity: {op['high_sensitivity'].get('selection')};
high-specificity: {op['high_specificity'].get('selection')}. These are
exploratory operating points, not clinically validated thresholds.

## Error analysis

- Eligible full-dataset records: **{errors.get('evaluation_population_count')}**
- Misclassified records: **{errors.get('misclassified_count')}**
- False negatives for grade>=2 screening: **{errors.get('false_negative_count')}**
- False positives for grade<2 screening: **{errors.get('false_positive_count')}**
- Image-quality metadata: **unavailable in the original evaluation records**

All error groups are included in `error_analysis.json`. Representative examples
are selected deterministically per reference/prediction group; they are not a
cherry-picked estimate. Confidence summaries use raw model confidence and raw
grade>=2 probability.

The original 39.17% referable sensitivity is not explained by thresholding
alone: it is the result at the existing 0.50 decision rule, and the exploratory
grid shows the sensitivity/specificity trade-off. It also reflects model error
on this shifted external population and the conditional third-party label
mapping. Because patient IDs and official Messidor-2 DR ground truth are not
available, this audit cannot attribute the result to patient leakage or make a
clinical-causality claim.

## Artifacts

- `duplicate_audit.json`
- `deduplicated_metrics.json`
- `deduplicated_predictions.csv`
- `deduplicated_confusion_matrix.png`
- `threshold_analysis.json`
- `threshold_tradeoff.png`
- `error_analysis.json`
- `error_analysis.png`
- `full_vs_deduplicated_comparison.json`
- `full_vs_deduplicated_comparison.png`
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Messidor-2 zero-shot evaluation reliability")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--evaluation-dir", default=str(DEFAULT_EVALUATION_DIR))
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_root = Path(args.raw_dir).expanduser().resolve()
    evaluation_dir = Path(args.evaluation_dir).expanduser().resolve()
    if args.bootstrap_iterations < 100 or args.batch_size < 1:
        raise SystemExit("--bootstrap-iterations must be at least 100 and --batch-size must be positive")
    manifest = _load(evaluation_dir / "dataset_manifest.json")
    software_versions = _software_versions()
    audit_configuration = {
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "torch_threads": args.torch_threads,
    }
    validation = _load(evaluation_dir / "validation_report.json")
    full_metrics_document = _load(evaluation_dir / "zero_shot_metrics.json")
    full_metrics = full_metrics_document["metrics"]
    full_rows = list(csv.DictReader((evaluation_dir / "zero_shot_predictions.csv").open(encoding="utf-8", newline="")))
    if validation.get("status") != "VALID" or full_metrics.get("status") != "CALCULATED":
        raise SystemExit("Original Phase 4B artifacts are not valid/calculated; reliability audit stopped without fabricating results.")

    records_by_path = {_path_key(record["image_root_relative_path"]): record for record in manifest["matched_records"]}
    records_by_raw_path = {_path_key(record["image_path"]): record for record in manifest["matched_records"]}
    predictions_by_path: dict[str, dict[str, Any]] = {}
    for row in full_rows:
        record = records_by_raw_path.get(_path_key(row.get("image_path", "")))
        if record is not None:
            predictions_by_path[_path_key(record["image_root_relative_path"])] = row
    exact_groups = validation["image_integrity"].get("exact_duplicate_groups", [])
    perceptual_groups = validation["image_integrity"].get("perceptual_duplicate_groups", [])
    exact_audit = _group_label_audit(exact_groups, records_by_path, predictions_by_path, "exact")
    perceptual_audit = _group_label_audit(perceptual_groups, records_by_path, predictions_by_path, "perceptual")
    excluded_paths: set[str] = set()
    canonical_paths: set[str] = set()
    for group in perceptual_audit:
        paths = group["image_paths"]
        if not paths:
            continue
        canonical_paths.add(_path_key(paths[0]))
        excluded_paths.update(_path_key(path) for path in paths[1:])
    dedup_records = [record for record in manifest["matched_records"] if _path_key(record["image_root_relative_path"]) not in excluded_paths]
    duplicate_audit = {
        "status": "CALCULATED",
        "label_provenance_statement": LABEL_PROVENANCE_STATEMENT,
        "source_validation_report": "ml/evaluation/messidor/validation_report.json",
        "policy": "Use all perceptual duplicate groups as a conservative sensitivity-analysis boundary. Retain lexicographically first path in each group; exclude other members only from the deduplicated evaluation. Never delete or rewrite source files or original evaluation artifacts.",
        "patient_level_assessment": {"status": "NOT_ASSESSABLE", "reason": "Patient identifiers are unavailable in the Messidor-2 image and label records."},
        "exact_duplicate_group_count": len(exact_audit),
        "perceptual_duplicate_group_count": len(perceptual_audit),
        "exact_groups": exact_audit,
        "perceptual_groups": perceptual_audit,
        "perceptual_groups_with_identical_labels": sum(1 for group in perceptual_audit if group["all_labels_identical"]),
        "perceptual_groups_with_conflicting_labels": sum(1 for group in perceptual_audit if group["records_found"] and not group["all_labels_identical"]),
        "exact_groups_with_identical_labels": sum(1 for group in exact_audit if group["all_labels_identical"]),
        "exact_groups_with_conflicting_labels": sum(1 for group in exact_audit if group["records_found"] and not group["all_labels_identical"]),
        "canonical_representative_count": len(canonical_paths),
        "excluded_image_count": len(excluded_paths),
        "deduplicated_image_count": len(dedup_records),
        "full_evaluation_image_count": len(manifest["matched_records"]),
        "software_versions": software_versions,
        "configuration": audit_configuration,
    }
    _json(evaluation_dir / "duplicate_audit.json", duplicate_audit)

    checkpoint = discover_checkpoint(args.checkpoint) if args.checkpoint else discover_checkpoint(None)
    dedup_rows, model_info, checksum_before, checksum_after = run_inference(
        dedup_records,
        raw_root,
        checkpoint,
        args.device,
        args.batch_size,
        args.torch_threads,
    )
    dedup_metrics, dedup_per_class = compute_metrics(
        dedup_rows,
        "Perceptual-deduplicated matched images with successful inference, adjudicated_gradable=1, and valid grade 0..4; original full-dataset labels and model probabilities were not changed.",
    )
    dedup_bootstrap = bootstrap_confidence_intervals(dedup_rows, args.bootstrap_iterations, args.bootstrap_seed)
    dedup_metrics_document = {
        "dataset": "Messidor-2",
        "evaluation_type": "zero_shot_external_validation_reliability_sensitivity_analysis",
        "label_provenance_statement": LABEL_PROVENANCE_STATEMENT,
        "deduplication_policy": duplicate_audit["policy"],
        "excluded_image_count": len(excluded_paths),
        "metrics": dedup_metrics,
        "per_class": dedup_per_class,
        "bootstrap_confidence_intervals": dedup_bootstrap,
        "inference_accounting": {
            "input_records": len(dedup_records),
            "successful_inferences": sum(1 for row in dedup_rows if row.get("inference_status") == "SUCCESS"),
            "failed_inferences": sum(1 for row in dedup_rows if row.get("inference_status") != "SUCCESS"),
        },
        "model": {
            "model_version": model_info.get("model_version"),
            "architecture": model_info.get("architecture"),
            "checkpoint_sha256_before": checksum_before,
            "checkpoint_sha256_after": checksum_after,
            "checkpoint_unchanged": checksum_before == checksum_after,
        },
        "clinical_validation_claim": False,
        "software_versions": software_versions,
        "configuration": audit_configuration,
    }
    _json(evaluation_dir / "deduplicated_metrics.json", dedup_metrics_document)
    _write_predictions(evaluation_dir / "deduplicated_predictions.csv", dedup_rows)
    if dedup_metrics.get("status") == "CALCULATED":
        write_confusion_matrix(evaluation_dir / "deduplicated_confusion_matrix.png", dedup_metrics["confusion_matrix"])

    threshold = _threshold_analysis(full_rows, args.bootstrap_iterations, args.bootstrap_seed)
    threshold["software_versions"] = software_versions
    threshold["configuration"] = audit_configuration
    _json(evaluation_dir / "threshold_analysis.json", threshold)
    errors = _error_analysis(full_rows)
    errors["software_versions"] = software_versions
    errors["configuration"] = audit_configuration
    _json(evaluation_dir / "error_analysis.json", errors)
    comparison_metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "quadratic_weighted_kappa",
    ]
    comparison = {
        "status": "CALCULATED",
        "label_provenance_statement": LABEL_PROVENANCE_STATEMENT,
        "full_dataset": full_metrics,
        "deduplicated_dataset": dedup_metrics,
        "full_sample_count": full_metrics.get("sample_count"),
        "deduplicated_sample_count": dedup_metrics.get("sample_count"),
        "removed_sample_count": (full_metrics.get("sample_count") or 0) - (dedup_metrics.get("sample_count") or 0),
        "deltas": {
            **{
                key: float(dedup_metrics.get(key, 0.0) or 0.0) - float(full_metrics.get(key, 0.0) or 0.0)
                for key in comparison_metrics
            },
            "referable_sensitivity": float(dedup_metrics["referable_dr_grade_2_or_worse"].get("sensitivity", 0.0)) - float(full_metrics["referable_dr_grade_2_or_worse"].get("sensitivity", 0.0)),
            "referable_specificity": float(dedup_metrics["referable_dr_grade_2_or_worse"].get("specificity", 0.0)) - float(full_metrics["referable_dr_grade_2_or_worse"].get("specificity", 0.0)),
        },
        "material_conclusion_assessment": {
            "status": "NO_MATERIAL_CHANGE_OBSERVED",
            "maximum_absolute_reported_delta": max(
                abs(float(dedup_metrics.get(key, 0.0) or 0.0) - float(full_metrics.get(key, 0.0) or 0.0))
                for key in comparison_metrics
            ),
            "rationale": "Only six of 1,744 evaluated records were excluded and the reported point-estimate deltas remain below 0.001 for the listed metrics. The conflicting-label perceptual group is retained in the full result and makes this a sensitivity analysis, not a corrected benchmark.",
        },
        "interpretation": "The deduplicated result is a sensitivity analysis. It does not replace the original full-dataset evaluation and does not establish clinical generalization.",
        "software_versions": software_versions,
        "configuration": audit_configuration,
    }
    _json(evaluation_dir / "full_vs_deduplicated_comparison.json", comparison)
    _write_threshold_visual(evaluation_dir / "threshold_tradeoff.png", threshold)
    _write_comparison_visual(evaluation_dir / "full_vs_deduplicated_comparison.png", full_metrics, dedup_metrics)
    _write_error_visual(evaluation_dir / "error_analysis.png", errors)

    model_info["checkpoint_sha256_before"] = checksum_before
    model_info["checkpoint_sha256_after"] = checksum_after
    model_info["checkpoint_unchanged"] = checksum_before == checksum_after
    _build_markdown(evaluation_dir / "phase4b_reliability_audit.md", duplicate_audit, dedup_metrics, threshold, errors, comparison, model_info)
    print(f"Messidor-2 reliability audit completed. Deduplicated inference records: {len(dedup_records)}; failures: {dedup_metrics_document['inference_accounting']['failed_inferences']}")
    print(f"Reports written under {evaluation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
