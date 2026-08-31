"""Genuine zero-shot Messidor-2 evaluation helpers.

The evaluator consumes the already-trained APTOS checkpoint and never writes
to it. Labels are discovered from the KaggleHub cache when available, while
images remain in the authorized raw dataset directory. All metrics are
descriptive external model-evaluation results, not clinical validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from scripts.dataset_common import scan_images, sha256_file


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
REQUIRED_LABEL_COLUMNS = {
    "image_id",
    "adjudicated_dr_grade",
    "adjudicated_dme",
    "adjudicated_gradable",
}
LABEL_DATASET_HANDLE = "google-brain/messidor2-dr-grades"
LABEL_CACHE_FILENAME = "messidor_data.csv"
LABEL_README_FILENAME = "messidor_readme.txt"
OFFICIAL_SOURCE_URL = "https://www.adcis.net/en/third-party/messidor2/"
APTOS_CLASS_MAPPING = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def normalize_stem(value: str) -> str:
    return Path(str(value).strip().replace("\\", "/")).stem.casefold()


def _safe_int(value: Any) -> int | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


@dataclass(frozen=True)
class LabelSource:
    path: Path
    discovery_method: str
    cache_root: str | None
    candidates: tuple[str, ...]


def _csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(next(csv.reader(handle), []))


def discover_label_source(raw_root: Path, explicit: str | None = None) -> tuple[LabelSource | None, list[str]]:
    """Find the expected label CSV without embedding a user-specific cache path."""
    errors: list[str] = []
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            return None, [f"Explicit label file does not exist: {path}"]
        try:
            columns = set(_csv_columns(path))
        except (OSError, UnicodeError, csv.Error) as exc:
            return None, [f"Could not inspect explicit label file {path}: {exc}"]
        if not REQUIRED_LABEL_COLUMNS.issubset(columns):
            return None, [f"Explicit label file is missing required columns: {sorted(REQUIRED_LABEL_COLUMNS - columns)}"]
        return LabelSource(path, "explicit_cli_path", None, (str(path),)), errors

    cache_roots: list[Path] = []
    try:
        import kagglehub.config as kaggle_config

        cache_roots.append(Path(kaggle_config.get_cache_folder()).expanduser())
    except Exception as exc:
        errors.append(f"KaggleHub cache discovery unavailable: {type(exc).__name__}: {exc}")
    cache_roots.extend([Path.home() / ".cache" / "kagglehub", Path.home() / ".kagglehub"])
    candidates: list[Path] = []
    for cache_root in dict.fromkeys(cache_roots):
        if not cache_root.is_dir():
            continue
        try:
            candidates.extend(sorted(path for path in cache_root.rglob(LABEL_CACHE_FILENAME) if path.is_file()))
        except OSError as exc:
            errors.append(f"Could not inspect KaggleHub cache {cache_root}: {exc}")
    valid_candidates: list[Path] = []
    for path in dict.fromkeys(candidates):
        try:
            if REQUIRED_LABEL_COLUMNS.issubset(set(_csv_columns(path))):
                valid_candidates.append(path)
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"Could not inspect candidate label file {path}: {exc}")
    if valid_candidates:
        chosen = sorted(valid_candidates, key=lambda item: str(item).casefold())[-1]
        cache_root = next((root for root in cache_roots if root in chosen.parents), None)
        return LabelSource(chosen, "kagglehub_cache", str(cache_root) if cache_root else None, tuple(str(item) for item in valid_candidates)), errors

    # A copied, schema-compatible label file is a safe fallback for an
    # offline checkout, but the report records that cache discovery did not
    # provide it.
    raw_candidates = sorted(path for path in raw_root.rglob(LABEL_CACHE_FILENAME) if path.is_file()) if raw_root.is_dir() else []
    for path in raw_candidates:
        try:
            if REQUIRED_LABEL_COLUMNS.issubset(set(_csv_columns(path))):
                return LabelSource(path, "raw_directory_fallback", None, tuple(str(item) for item in raw_candidates)), errors
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"Could not inspect raw label candidate {path}: {exc}")
    errors.append(f"No schema-compatible {LABEL_CACHE_FILENAME} was found in the KaggleHub cache or raw directory.")
    return None, errors


def discover_image_root(raw_root: Path) -> tuple[Path, list[Path], str]:
    preferred = raw_root / "images" / "messidor-2"
    search_root = preferred if preferred.is_dir() else raw_root
    method = "expected_messidor2_image_directory" if preferred.is_dir() else "raw_directory_fallback"
    images = sorted(path for path in search_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return search_root, images, method


def validate_images(image_root: Path, image_paths: list[Path]) -> dict[str, Any]:
    if not image_paths:
        return {
            "total_images": 0,
            "readable_images": 0,
            "corrupt_images": [],
            "resolution_statistics": {"count": 0},
            "exact_duplicate_groups": [],
            "perceptual_duplicate_groups": [],
            "exact_duplicate_count": 0,
            "perceptual_duplicate_count": 0,
            "readable_inventory": [],
        }
    scan = scan_images(image_paths, image_root)
    return {
        "total_images": len(image_paths),
        "readable_images": scan["readable_files"],
        "corrupt_images": scan["corrupted"],
        "resolution_statistics": scan["resolution_statistics"],
        "exact_duplicate_groups": scan["exact_duplicate_groups"],
        "perceptual_duplicate_groups": scan["perceptual_duplicate_groups"],
        "exact_duplicate_count": scan["duplicate_exact_count"],
        "perceptual_duplicate_count": scan["duplicate_perceptual_count"],
        "readable_inventory": scan["readable"],
    }


def load_labels(source: LabelSource) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if not REQUIRED_LABEL_COLUMNS.issubset(set(columns)):
            raise ValueError(f"Label file is missing required columns: {sorted(REQUIRED_LABEL_COLUMNS - set(columns))}")
        for row_number, raw in enumerate(reader, start=2):
            image_id = str(raw.get("image_id") or "").strip()
            gradable = _safe_int(raw.get("adjudicated_gradable"))
            grade = _safe_int(raw.get("adjudicated_dr_grade"))
            dme = _safe_int(raw.get("adjudicated_dme"))
            item = {
                "row_number": row_number,
                "image_id": image_id,
                "image_stem": normalize_stem(image_id),
                "adjudicated_dr_grade": grade,
                "adjudicated_dme": dme,
                "adjudicated_gradable": gradable,
            }
            if not image_id:
                errors.append({"row": row_number, "field": "image_id", "reason": "missing"})
            if gradable not in {0, 1}:
                errors.append({"row": row_number, "field": "adjudicated_gradable", "value": raw.get("adjudicated_gradable"), "reason": "must be 0 or 1"})
            if gradable == 1 and grade not in range(5):
                errors.append({"row": row_number, "field": "adjudicated_dr_grade", "value": raw.get("adjudicated_dr_grade"), "reason": "gradable row must contain an integer grade 0..4"})
            if gradable == 1 and dme not in {0, 1}:
                errors.append({"row": row_number, "field": "adjudicated_dme", "value": raw.get("adjudicated_dme"), "reason": "gradable row must contain 0 or 1"})
            if gradable == 0 and (grade is not None or dme is not None):
                errors.append({"row": row_number, "field": "ungradable_labels", "reason": "ungradable rows should not contain DR or DME labels"})
            rows.append(item)
    return {
        "columns": columns,
        "row_count": len(rows),
        "rows": rows,
        "errors": errors,
        "sha256": sha256_file(source.path),
    }


def build_matched_manifest(raw_root: Path, image_root: Path, image_paths: list[Path], labels: dict[str, Any], source: LabelSource, image_validation: dict[str, Any]) -> dict[str, Any]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        index[path.stem.casefold()].append(path)
    readable = {item["path"]: item for item in image_validation["readable_inventory"]}
    records: list[dict[str, Any]] = []
    unmatched_labels: list[dict[str, Any]] = []
    ambiguous_labels: list[dict[str, Any]] = []
    matched_stems: set[str] = set()
    for row in labels["rows"]:
        matches = index.get(row["image_stem"], [])
        if not matches:
            unmatched_labels.append({"row_number": row["row_number"], "image_id": row["image_id"], "reason": "no_image_with_matching_case_insensitive_stem"})
            continue
        if len(matches) > 1:
            ambiguous_labels.append({"row_number": row["row_number"], "image_id": row["image_id"], "matches": [relative(path, raw_root) for path in matches]})
            continue
        path = matches[0]
        matched_stems.add(row["image_stem"])
        scan_item = readable.get(relative(path, image_root))
        records.append({
            "image_path": relative(path, raw_root),
            "image_root_relative_path": relative(path, image_root),
            "image_id": row["image_id"],
            "image_stem": row["image_stem"],
            "adjudicated_dr_grade": row["adjudicated_dr_grade"],
            "adjudicated_dme": row["adjudicated_dme"],
            "adjudicated_gradable": row["adjudicated_gradable"],
            "label_row_number": row["row_number"],
            "dataset_source": LABEL_DATASET_HANDLE,
            "image_sha256": scan_item["sha256"] if scan_item else None,
        })
    image_without_labels = [relative(path, raw_root) for path in image_paths if path.stem.casefold() not in matched_stems]
    duplicate_image_ids = [
        {"image_stem": stem, "rows": [item["label_row_number"] for item in records if item["image_stem"] == stem]}
        for stem in sorted({item["image_stem"] for item in records})
        if sum(1 for item in records if item["image_stem"] == stem) > 1
    ]
    return {
        "records": records,
        "matched_pair_count": len(records),
        "unmatched_label_records": unmatched_labels,
        "ambiguous_label_records": ambiguous_labels,
        "images_without_labels": image_without_labels,
        "duplicate_image_ids": duplicate_image_ids,
        "label_source": {
            "handle": LABEL_DATASET_HANDLE,
            "path": str(source.path),
            "discovery_method": source.discovery_method,
            "cache_root": source.cache_root,
            "candidate_paths": list(source.candidates),
            "sha256": labels["sha256"],
        },
        "dataset_source": LABEL_DATASET_HANDLE,
    }


def _safe_metric_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_stats(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    precision = _safe_metric_divide(tp, tp + fp)
    recall = _safe_metric_divide(tp, tp + fn)
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": recall,
        "specificity": _safe_metric_divide(tn, tn + fp),
        "precision": precision,
        "recall": recall,
        "f1": _safe_metric_divide(2 * precision * recall, precision + recall),
    }


def _qwk(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import cohen_kappa_score

        if len(np.unique(actual)) < 2:
            return None
        return float(cohen_kappa_score(actual, predicted, weights="quadratic"))
    except (ImportError, ValueError):
        return None


def _auc(actual: np.ndarray, probabilities: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(actual)) < 2:
            return None
        return float(roc_auc_score(actual, probabilities, multi_class="ovr", average="macro"))
    except (ImportError, ValueError):
        return None


def _binary_auc(actual: np.ndarray, probability: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(actual)) < 2:
            return None
        return float(roc_auc_score(actual, probability))
    except (ImportError, ValueError):
        return None


def compute_metrics(records: list[dict[str, Any]], population_note: str) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluated = [item for item in records if item.get("inference_status") == "SUCCESS" and item.get("adjudicated_gradable") == 1 and item.get("adjudicated_dr_grade") in range(5)]
    if not evaluated:
        return ({"status": "NOT_CALCULABLE", "reason": "No successful gradable inferences with valid labels.", "population": population_note}, {"status": "NOT_CALCULABLE", "per_class": {}})
    actual = np.asarray([item["adjudicated_dr_grade"] for item in evaluated], dtype=int)
    probabilities = np.asarray([[item[f"probability_{index}"] for index in range(5)] for item in evaluated], dtype=float)
    predicted = probabilities.argmax(axis=1)
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

        matrix = confusion_matrix(actual, predicted, labels=list(range(5))).tolist()
        accuracy = float(accuracy_score(actual, predicted))
        macro_f1 = float(f1_score(actual, predicted, labels=list(range(5)), average="macro", zero_division=0))
        weighted_f1 = float(f1_score(actual, predicted, labels=list(range(5)), average="weighted", zero_division=0))
        per_class: dict[str, Any] = {}
        for index, label in APTOS_CLASS_MAPPING.items():
            one_actual = (actual == index).astype(int)
            one_predicted = (predicted == index).astype(int)
            stats = _binary_stats(one_actual, one_predicted)
            per_class[str(index)] = {
                "label": label,
                "support": int(np.sum(actual == index)),
                "precision": float(precision_score(one_actual, one_predicted, zero_division=0)),
                "recall": float(recall_score(one_actual, one_predicted, zero_division=0)),
                "sensitivity": stats["sensitivity"],
                "specificity": stats["specificity"],
                "f1": float(f1_score(one_actual, one_predicted, zero_division=0)),
            }
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for Messidor-2 metrics. Install backend/requirements-ml.txt.") from exc
    referable_actual = (actual >= 2).astype(int)
    referable_probability = probabilities[:, 2:].sum(axis=1)
    referable_predicted = (referable_probability >= 0.5).astype(int)
    referable = _binary_stats(referable_actual, referable_predicted)
    referable["roc_auc"] = _binary_auc(referable_actual, referable_probability)
    point = {
        "status": "CALCULATED",
        "evaluation_population": population_note,
        "sample_count": len(evaluated),
        "label_mapping": "Messidor-2 adjudicated five-point ICDR grade 0..4 mapped 1:1 to the existing APTOS output labels defined in the downloaded label readme.",
        "exclusions": "Unmatched label rows are excluded; ungradable rows and unsuccessful inferences are excluded from grade metrics.",
        "averaging": {"macro_f1": "unweighted mean across five classes", "weighted_f1": "support-weighted mean across five classes", "roc_auc_ovr_macro": "one-vs-rest macro average"},
        "class_distribution": {str(index): int(np.sum(actual == index)) for index in range(5)},
        "prediction_distribution": {str(index): int(np.sum(predicted == index)) for index in range(5)},
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_macro": float(precision_score(actual, predicted, labels=list(range(5)), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(actual, predicted, labels=list(range(5)), average="macro", zero_division=0)),
        "roc_auc_ovr_macro": _auc(actual, probabilities),
        "quadratic_weighted_kappa": _qwk(actual, predicted),
        "confusion_matrix": matrix,
        "referable_dr_grade_2_or_worse": {"rule": "true and predicted referable if grade >= 2", **referable},
        "clinical_validation_claim": False,
    }
    per_class_report = {"status": "CALCULATED", "sample_count": len(evaluated), "per_class": per_class, "clinical_validation_claim": False}
    return point, per_class_report


def bootstrap_confidence_intervals(records: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    evaluated = [item for item in records if item.get("inference_status") == "SUCCESS" and item.get("adjudicated_gradable") == 1 and item.get("adjudicated_dr_grade") in range(5)]
    if not evaluated:
        return {"status": "NOT_CALCULABLE", "iterations": iterations, "seed": seed, "confidence_level": 0.95}
    actual = np.asarray([item["adjudicated_dr_grade"] for item in evaluated], dtype=int)
    probabilities = np.asarray([[item[f"probability_{index}"] for index in range(5)] for item in evaluated], dtype=float)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        indices = rng.integers(0, len(actual), size=len(actual))
        sample_actual = actual[indices]
        sample_probabilities = probabilities[indices]
        sample_predicted = sample_probabilities.argmax(axis=1)
        try:
            from sklearn.metrics import accuracy_score, f1_score

            values["accuracy"].append(float(accuracy_score(sample_actual, sample_predicted)))
            values["macro_f1"].append(float(f1_score(sample_actual, sample_predicted, labels=list(range(5)), average="macro", zero_division=0)))
            values["weighted_f1"].append(float(f1_score(sample_actual, sample_predicted, labels=list(range(5)), average="weighted", zero_division=0)))
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for bootstrap metrics.") from exc
        qwk = _qwk(sample_actual, sample_predicted)
        if qwk is not None:
            values["quadratic_weighted_kappa"].append(qwk)
        referable_actual = (sample_actual >= 2).astype(int)
        referable_probability = sample_probabilities[:, 2:].sum(axis=1)
        referable = _binary_stats(referable_actual, (referable_probability >= 0.5).astype(int))
        values["referable_sensitivity"].append(float(referable["sensitivity"]))
        values["referable_specificity"].append(float(referable["specificity"]))
        auc = _binary_auc(referable_actual, referable_probability)
        if auc is not None:
            values["referable_roc_auc"].append(auc)
    intervals = {
        metric: {"lower": float(np.percentile(samples, 2.5)), "upper": float(np.percentile(samples, 97.5)), "successful_resamples": len(samples)}
        for metric, samples in values.items()
        if samples
    }
    return {
        "status": "CALCULATED",
        "confidence_level": 0.95,
        "method": "nonparametric percentile bootstrap",
        "iterations": iterations,
        "seed": seed,
        "sample_count": len(evaluated),
        "intervals": intervals,
        "note": "Intervals quantify sampling variability for this external evaluation population; they are not clinical uncertainty guarantees.",
    }


def distribution_comparison(actual: Iterable[int], predicted: Iterable[int], internal_metrics: dict[str, Any] | None) -> dict[str, Any]:
    actual_counts = Counter(str(item) for item in actual)
    predicted_counts = Counter(str(item) for item in predicted)
    reference_counts = (internal_metrics or {}).get("class_distribution") or {}
    keys = [str(index) for index in range(5)]
    actual_total = max(1, sum(actual_counts.values()))
    predicted_total = max(1, sum(predicted_counts.values()))
    reference_total = max(1, sum(int(value) for value in reference_counts.values()))
    actual_distribution = {key: actual_counts.get(key, 0) / actual_total for key in keys}
    predicted_distribution = {key: predicted_counts.get(key, 0) / predicted_total for key in keys}
    reference_distribution = {key: int(reference_counts.get(key, 0)) / reference_total for key in keys}
    def js_divergence(first: dict[str, float], second: dict[str, float]) -> float:
        p = np.asarray([first[key] for key in keys], dtype=float) + 1e-12
        q = np.asarray([second[key] for key in keys], dtype=float) + 1e-12
        p /= p.sum()
        q /= q.sum()
        midpoint = 0.5 * (p + q)
        return float(0.5 * np.sum(p * np.log2(p / midpoint)) + 0.5 * np.sum(q * np.log2(q / midpoint)))
    return {
        "external_true_distribution": actual_distribution,
        "external_prediction_distribution": predicted_distribution,
        "internal_reference_distribution": reference_distribution if reference_counts else None,
        "js_divergence_external_true_vs_internal_reference": js_divergence(actual_distribution, reference_distribution) if reference_counts else None,
        "js_divergence_external_true_vs_external_prediction": js_divergence(actual_distribution, predicted_distribution),
        "interpretation": "Descriptive distribution comparisons only; no drift or clinical generalization conclusion is made from these values alone.",
    }


def write_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    """Write a deterministic PNG without adding a plotting dependency."""
    from PIL import ImageDraw, ImageFont

    labels = [APTOS_CLASS_MAPPING[index] for index in range(5)]
    canvas = Image.new("RGB", (900, 760), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    draw.text((24, 20), "Messidor-2 zero-shot confusion matrix", fill=(20, 35, 55), font=title_font)
    left, top, cell = 230, 100, 82
    maximum = max((value for row in matrix for value in row), default=1)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            intensity = int(245 - 190 * (value / max(1, maximum)))
            colour = (intensity, intensity + 5, 255)
            x0 = left + column_index * cell
            y0 = top + row_index * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=colour, outline=(120, 140, 160))
            text_colour = "white" if value > maximum * 0.45 else (15, 25, 40)
            bounds = draw.textbbox((0, 0), str(value), font=font)
            draw.text((x0 + (cell - (bounds[2] - bounds[0])) / 2, y0 + (cell - (bounds[3] - bounds[1])) / 2), str(value), fill=text_colour, font=font)
    for index, label in enumerate(labels):
        x = left + index * cell + 12
        y = top + 5 * cell + 14
        draw.text((x, y), label[:12], fill=(20, 35, 55), font=font)
        draw.text((left - 205, top + index * cell + 34), label, fill=(20, 35, 55), font=font)
    draw.text((left + 2 * cell, 650), "Predicted APTOS grade", fill=(20, 35, 55), font=font)
    draw.text((20, 120), "True Messidor-2 ICDR grade", fill=(20, 35, 55), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def model_snapshot(checkpoint: Path, before_checksum: str, checkpoint_metadata: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint_metadata.get("model_config") or {}
    artifact = checkpoint_metadata.get("artifact") or {}
    return {
        "model_name": artifact.get("model_name", "RETINA-NEXUS DR classifier"),
        "model_version": checkpoint_metadata.get("model_version") or artifact.get("model_version", "unversioned"),
        "architecture": config.get("backbone", artifact.get("backbone", "unknown")),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256_before": before_checksum,
        "checkpoint_sha256_after": None,
        "output_classes": int(config.get("num_classes", 5)),
        "input_resolution": int(config.get("input_size", 224)),
        "ordinal_mode": bool(config.get("ordinal_mode", False)),
        "preprocessing": {
            "resize": [int(config.get("input_size", 224)), int(config.get("input_size", 224))],
            "color_conversion": "RGB",
            "to_tensor": True,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "dataset_specific_optimization": False,
        },
        "class_mapping": {str(key): value for key, value in APTOS_CLASS_MAPPING.items()},
        "referable_mapping": artifact.get("referable_mapping", {"name": "grade_2_or_worse", "referable_grades": [2, 3, 4]}),
        "dataset_version": checkpoint_metadata.get("dataset_version"),
        "clinical_validation_claim": False,
    }


__all__ = [
    "APTOS_CLASS_MAPPING",
    "LABEL_DATASET_HANDLE",
    "LabelSource",
    "bootstrap_confidence_intervals",
    "build_matched_manifest",
    "compute_metrics",
    "discover_image_root",
    "discover_label_source",
    "distribution_comparison",
    "load_labels",
    "model_snapshot",
    "relative",
    "sha256_bytes",
    "validate_images",
    "write_confusion_matrix",
]
