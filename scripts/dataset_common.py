"""Shared, dependency-light dataset governance utilities.

These utilities intentionally operate on local files only. Acquisition is a
separate command so validation can never imply that a dataset was downloaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "ml" / "datasets"
REGISTRY_PATH = DATA_ROOT / "metadata" / "dataset_registry.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


class DatasetError(RuntimeError):
    """An actionable dataset setup or validation error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        raise DatasetError(f"Dataset registry is missing: {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def get_definition(slug: str) -> dict[str, Any]:
    registry = load_registry()
    for definition in registry.get("datasets", []):
        if definition["slug"] == slug:
            return definition
    available = ", ".join(item["slug"] for item in registry.get("datasets", []))
    raise DatasetError(f"Unknown dataset '{slug}'. Choose one of: {available}")


def raw_path_for(definition: dict[str, Any], override: str | None = None) -> Path:
    path = Path(override) if override else REPO_ROOT / definition["raw_path"]
    return path.expanduser().resolve()


def report_directory(slug: str) -> Path:
    path = DATA_ROOT / "metadata" / "reports" / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_directory(slug: str) -> Path:
    path = DATA_ROOT / "metadata" / "splits" / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def image_files(raw_path: Path, definition: dict[str, Any]) -> list[Path]:
    extensions = {ext.lower() for ext in definition.get("expected_image_extensions", IMAGE_EXTENSIONS)}
    excluded = [pattern.lower() for pattern in definition.get("image_exclude_patterns", [])]
    return sorted(
        path for path in raw_path.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
        and not any(pattern in path.name.lower() for pattern in excluded)
    )


def annotation_files(raw_path: Path, definition: dict[str, Any]) -> list[Path]:
    found: set[Path] = set()
    for pattern in definition.get("annotation_globs", []):
        found.update(path for path in raw_path.glob(pattern) if path.is_file())
    return sorted(found)


def _pick(row: dict[str, Any], candidates: list[str]) -> tuple[str | None, str | None]:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for candidate in candidates:
        value = normalized.get(candidate.lower())
        if value is not None and str(value).strip() != "":
            return candidate, str(value).strip()
    return None, None


def read_annotation_rows(raw_path: Path, definition: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for annotation in annotation_files(raw_path, definition):
        if annotation.suffix.lower() != ".csv":
            continue
        try:
            with annotation.open("r", encoding="utf-8-sig", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle), start=2):
                    image_field, image_ref = _pick(row, definition.get("image_field_candidates", []))
                    label_field, label = _pick(row, definition.get("label_field_candidates", []))
                    group_field, group = _pick(row, definition.get("group_field_candidates", []))
                    rows.append({
                        "annotation_file": str(annotation.relative_to(raw_path)),
                        "annotation_row": index,
                        "image_field": image_field,
                        "image_ref": image_ref,
                        "label_field": label_field,
                        "label": label,
                        "group_field": group_field,
                        "group": group,
                        "raw": {str(key): value for key, value in row.items()},
                    })
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"{annotation.name}: {exc}")
    return rows, errors


def build_image_index(files: list[Path], raw_path: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in files:
        relative = str(path.relative_to(raw_path)).replace("\\", "/").lower()
        index[relative] = path
        index[path.name.lower()] = path
        index[path.stem.lower()] = path
    return index


def resolve_image_ref(value: str | None, index: dict[str, Path], raw_path: Path, definition: dict[str, Any]) -> Path | None:
    if not value:
        return None
    cleaned = value.strip().replace("\\", "/")
    candidates = [cleaned.lower(), Path(cleaned).name.lower(), Path(cleaned).stem.lower()]
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    for ext in definition.get("expected_image_extensions", IMAGE_EXTENSIONS):
        candidate = f"{Path(cleaned).stem}{ext}".lower()
        if candidate in index:
            return index[candidate]
    direct = (raw_path / cleaned).resolve()
    try:
        direct.relative_to(raw_path)
    except ValueError:
        return None
    return direct if direct.exists() and direct.is_file() else None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    """Return a small average hash suitable for near-duplicate screening."""
    grayscale = ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    average = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
    return f"{int(bits, 2):0{len(bits) // 4}x}"


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def scan_images(files: list[Path], raw_path: Path) -> dict[str, Any]:
    readable: list[dict[str, Any]] = []
    corrupted: list[dict[str, str]] = []
    exact_groups: dict[str, list[str]] = defaultdict(list)
    perceptual_values: dict[str, str] = {}
    resolutions: list[dict[str, Any]] = []
    for path in files:
        relative = str(path.relative_to(raw_path)).replace("\\", "/")
        try:
            exact = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                phash = perceptual_hash(image)
            readable.append({"path": relative, "sha256": exact, "perceptual_hash": phash, "width": width, "height": height})
            exact_groups[exact].append(relative)
            perceptual_values[relative] = phash
            resolutions.append({"width": width, "height": height, "pixels": width * height})
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            corrupted.append({"path": relative, "error": str(exc)})

    perceptual_groups: list[list[str]] = []
    assigned: set[str] = set()
    for path, value in perceptual_values.items():
        if path in assigned:
            continue
        group = [other for other, other_value in perceptual_values.items() if hamming_distance(value, other_value) <= 4]
        if len(group) > 1:
            perceptual_groups.append(sorted(group))
            assigned.update(group)

    exact_duplicate_groups = [sorted(group) for group in exact_groups.values() if len(group) > 1]
    exact_duplicate_count = sum(len(group) - 1 for group in exact_duplicate_groups)
    perceptual_duplicate_count = sum(len(group) - 1 for group in perceptual_groups)
    widths = [item["width"] for item in resolutions]
    heights = [item["height"] for item in resolutions]
    pixels = [item["pixels"] for item in resolutions]
    resolution_statistics = {
        "count": len(resolutions),
        "width": _numeric_stats(widths),
        "height": _numeric_stats(heights),
        "pixels": _numeric_stats(pixels),
    }
    return {
        "total_files": len(files),
        "readable_files": len(readable),
        "corrupted_files": len(corrupted),
        "corrupted": corrupted,
        "readable": readable,
        "exact_duplicate_groups": exact_duplicate_groups,
        "perceptual_duplicate_groups": perceptual_groups,
        "duplicate_exact_count": exact_duplicate_count,
        "duplicate_perceptual_count": perceptual_duplicate_count,
        "resolution_statistics": resolution_statistics,
        "phash_by_path": perceptual_values,
    }


def _numeric_stats(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {"min": min(values), "max": max(values), "mean": round(statistics.mean(values), 2), "median": statistics.median(values)}


def _label_summary(rows: list[dict[str, Any]], image_index: dict[str, Path], raw_path: Path, definition: dict[str, Any]) -> dict[str, Any]:
    distribution: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    missing_images: list[dict[str, Any]] = []
    missing_labels: list[dict[str, Any]] = []
    with_image = 0
    with_group = 0
    label_candidates = definition.get("label_field_candidates", [])
    allowed = {str(value) for value in definition.get("label_values", [])}
    for row in rows:
        if resolve_image_ref(row.get("image_ref"), image_index, raw_path, definition):
            with_image += 1
        else:
            missing_images.append({"annotation_file": row["annotation_file"], "row": row["annotation_row"], "image_ref": row.get("image_ref")})
        if row.get("group"):
            with_group += 1
        if label_candidates:
            label = row.get("label")
            if label is None:
                missing_labels.append({"annotation_file": row["annotation_file"], "row": row["annotation_row"]})
            else:
                distribution[label] += 1
                if allowed and label not in allowed:
                    invalid.append({"annotation_file": row["annotation_file"], "row": row["annotation_row"], "label": label, "allowed": sorted(allowed)})
    expected_metadata = 1 + (1 if label_candidates else 0) + (1 if definition.get("group_field_candidates") else 0)
    supplied_metadata = with_image + (sum(1 for row in rows if row.get("label")) if label_candidates else 0) + (with_group if definition.get("group_field_candidates") else 0)
    denominator = max(1, len(rows) * expected_metadata)
    return {
        "annotation_rows": len(rows),
        "rows_with_resolved_images": with_image,
        "missing_image_references": missing_images,
        "missing_labels": missing_labels,
        "invalid_labels": invalid,
        "class_distribution": dict(sorted(distribution.items())),
        "label_completeness": (sum(1 for row in rows if row.get("label")) / len(rows)) if rows and label_candidates else None,
        "metadata_completeness": supplied_metadata / denominator if rows else None,
        "group_field_available": with_group > 0,
    }


def load_split(slug: str) -> dict[str, Any] | None:
    path = split_directory(slug) / "splits.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def leakage_report(slug: str, records: list[dict[str, Any]], split_payload: dict[str, Any] | None) -> dict[str, Any]:
    limitations: list[str] = []
    patient_level = any(record.get("group_source") == "patient_id" for record in records)
    if not patient_level:
        limitations.append("No patient-level identifier was found in the source annotations; patient-level separation cannot be guaranteed.")
    if split_payload is None:
        return {
            "dataset": slug, "status": "not_run", "patient_level_guarantee": patient_level,
            "cross_split_patient_groups": [], "cross_split_duplicate_groups": [], "limitations": limitations + ["No split manifest exists. Run create_splits.py first."],
        }
    by_group: dict[str, set[str]] = defaultdict(set)
    by_duplicate: dict[str, set[str]] = defaultdict(set)
    for record in split_payload.get("records", []):
        by_group[record["group_id"]].add(record["split"])
        by_duplicate[record["duplicate_group_id"]].add(record["split"])
    cross_groups = sorted(group for group, splits in by_group.items() if len(splits) > 1)
    cross_duplicates = sorted(group for group, splits in by_duplicate.items() if len(splits) > 1)
    if cross_duplicates:
        limitations.append("Duplicate-image groups cross split boundaries.")
    return {
        "dataset": slug, "status": "pass" if not cross_groups and not cross_duplicates else "blocked",
        "patient_level_guarantee": patient_level, "cross_split_patient_groups": cross_groups,
        "cross_split_duplicate_groups": cross_duplicates, "limitations": limitations,
    }


def readiness_score(scan: dict[str, Any], labels: dict[str, Any], split_report: dict[str, Any]) -> dict[str, Any]:
    total = scan["total_files"]
    readable = scan["readable_files"] / total if total else 0.0
    duplicate_free = max(0.0, 1 - ((scan["duplicate_exact_count"] + scan["duplicate_perceptual_count"]) / max(1, total)))
    class_distribution = labels.get("class_distribution", {})
    if labels.get("label_completeness") is None:
        label_completeness = None
        class_balance = None
    else:
        label_completeness = labels["label_completeness"]
        counts = list(class_distribution.values())
        class_balance = min(counts) / max(counts) if counts else 0.0
    split_integrity = 1.0 if split_report.get("status") == "pass" else 0.0
    metadata = labels.get("metadata_completeness")
    dimensions = {
        "readable_files": readable, "duplicate_free": duplicate_free,
        "label_completeness": label_completeness, "class_balance": class_balance,
        "split_integrity": split_integrity, "metadata_completeness": metadata,
    }
    weights = load_registry().get("readiness_metric", {}).get("weights", {})
    applicable_weight = sum(weights.get(name, 0) for name, value in dimensions.items() if value is not None)
    weighted = sum(weights.get(name, 0) * value for name, value in dimensions.items() if value is not None)
    score = round(100 * weighted / applicable_weight, 2) if applicable_weight else 0.0
    return {"name": "Dataset Readiness Score", "score": score, "dimensions": dimensions, "excluded_dimensions": [name for name, value in dimensions.items() if value is None], "note": "Engineering readiness metric only; not a clinical validation metric."}


def validate_dataset(slug: str, raw_override: str | None = None) -> dict[str, Any]:
    definition = get_definition(slug)
    raw_path = raw_path_for(definition, raw_override)
    if not raw_path.exists():
        raise DatasetError(f"Dataset directory does not exist: {raw_path}. Acquire or manually place the authorized files first.")
    files = image_files(raw_path, definition)
    if not files:
        raise DatasetError(f"No image files found under {raw_path}. Nothing was downloaded or the directory layout is not recognized.")
    scan = scan_images(files, raw_path)
    rows, annotation_errors = read_annotation_rows(raw_path, definition)
    labels = _label_summary(rows, build_image_index(files, raw_path), raw_path, definition)
    split_payload = load_split(slug)
    records = [{"image": item["path"], "group_source": "patient_id" if row.get("group") else "duplicate_or_image"} for row in rows for item in scan["readable"] if Path(item["path"]).stem.lower() == Path(row.get("image_ref") or "").stem.lower()]
    leakage = leakage_report(slug, records, split_payload)
    score = readiness_score(scan, labels, leakage)
    status = "pass" if scan["corrupted_files"] == 0 and not labels["invalid_labels"] and not labels["missing_image_references"] and leakage["status"] in {"pass", "not_run"} else "blocked"
    return {
        "dataset": {"slug": slug, "name": definition["name"], "purpose": definition["purpose"], "raw_path": str(raw_path), "registry_status": definition["status"]},
        "generated_at": utc_now(), "status": status, "annotation_errors": annotation_errors,
        "files": {key: value for key, value in scan.items() if key not in {"readable", "phash_by_path"}},
        "labels": labels, "leakage": leakage, "readiness": score,
        "limitations": definition.get("limitations", []),
    }
