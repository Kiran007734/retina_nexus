"""Shared, dependency-light dataset governance utilities.

These utilities intentionally operate on local files only. Acquisition is a
separate command so validation can never imply that a dataset was downloaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import cv2
    import numpy as np
except ImportError:  # Keep manual governance utilities importable before ML dependencies are installed.
    cv2 = None
    np = None

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


def scan_fingerprint(files: list[Path], raw_path: Path) -> str:
    """Fingerprint the file inventory so cached scans cannot go stale silently."""
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        relative = str(path.relative_to(raw_path)).replace("\\", "/")
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def load_cached_scan(slug: str, raw_path: Path, files: list[Path]) -> dict[str, Any] | None:
    """Load a complete scan cached by validate_dataset when the inventory matches."""
    report_path = DATA_ROOT / "metadata" / "reports" / slug / "dataset_validation_report.json"
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_files = payload.get("files", {})
        if report_files.get("scan_fingerprint") != scan_fingerprint(files, raw_path):
            return None
        if "readable" not in report_files:
            return None
        return report_files
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def perceptual_hash(image: Image.Image) -> str:
    """Return a low-frequency DCT hash for near-duplicate screening.

    A whole-image average hash is too sensitive to the large dark background
    common in fundus photographs and can classify unrelated images as close
    duplicates. The DCT hash compares low-frequency structure instead.
    """
    grayscale = ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS)
    if cv2 is not None and np is not None:
        pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
        coefficients = cv2.dct(pixels)[:8, :8]
        dct_values = coefficients.flatten()
        median = float(np.median(dct_values[1:]))
        dct_bits = "".join("1" if value >= median else "0" for value in dct_values)
        # Add low-resolution horizontal and vertical gradients. This makes
        # the candidate hash sensitive to retinal structure instead of only
        # the low-frequency brightness pattern of the circular field of view.
        horizontal = cv2.resize(pixels, (9, 8), interpolation=cv2.INTER_AREA)
        vertical = cv2.resize(pixels, (8, 9), interpolation=cv2.INTER_AREA)
        horizontal_bits = "".join("1" if value >= 0 else "0" for value in (horizontal[:, 1:] - horizontal[:, :-1]).flatten())
        vertical_bits = "".join("1" if value >= 0 else "0" for value in (vertical[1:, :] - vertical[:-1, :]).flatten())
        bits = dct_bits + horizontal_bits + vertical_bits
        return f"{int(bits, 2):0{len(bits) // 4}x}"
    pixels = list(grayscale.getdata())
    average = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
    return f"{int(bits, 2):0{len(bits) // 4}x}"


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


class _PerceptualHashIndex:
    """BK-tree index for bounded Hamming-distance hash lookup."""

    def __init__(self) -> None:
        self._values: list[tuple[str, int, dict[int, int]]] = []
        self._root: int | None = None

    def add(self, path: str, value: str) -> None:
        numeric = int(value, 16)
        if self._root is None:
            self._values.append((path, numeric, {}))
            self._root = 0
            return
        node_index = self._root
        while True:
            _, node_value, children = self._values[node_index]
            distance = (numeric ^ node_value).bit_count()
            child_index = children.get(distance)
            if child_index is None:
                children[distance] = len(self._values)
                self._values.append((path, numeric, {}))
                return
            node_index = child_index

    def query(self, value: str, max_distance: int) -> list[str]:
        numeric = int(value, 16)
        if self._root is None:
            return []
        matches: list[str] = []
        pending = [self._root]
        while pending:
            node_index = pending.pop()
            path, node_value, children = self._values[node_index]
            distance = (numeric ^ node_value).bit_count()
            if distance <= max_distance:
                matches.append(path)
            lower = max(0, distance - max_distance)
            upper = distance + max_distance
            pending.extend(child for edge, child in children.items() if lower <= edge <= upper)
        return matches


def _scan_single_image(path: Path, raw_path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    relative = str(path.relative_to(raw_path)).replace("\\", "/")
    try:
        exact = sha256_file(path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            phash = perceptual_hash(image)
        return (
            {"path": relative, "sha256": exact, "perceptual_hash": phash, "width": width, "height": height},
            None,
        )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return None, {"path": relative, "error": str(exc)}


def scan_images(files: list[Path], raw_path: Path) -> dict[str, Any]:
    readable: list[dict[str, Any]] = []
    corrupted: list[dict[str, str]] = []
    exact_groups: dict[str, list[str]] = defaultdict(list)
    perceptual_values: dict[str, str] = {}
    resolutions: list[dict[str, Any]] = []
    worker_count = min(4, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(lambda item: _scan_single_image(item, raw_path), files)
        for readable_item, corrupted_item in results:
            if corrupted_item is not None:
                corrupted.append(corrupted_item)
                continue
            assert readable_item is not None
            readable.append(readable_item)
            exact_groups[readable_item["sha256"]].append(readable_item["path"])
            perceptual_values[readable_item["path"]] = readable_item["perceptual_hash"]
            resolutions.append({"width": readable_item["width"], "height": readable_item["height"], "pixels": readable_item["width"] * readable_item["height"]})

    # Use a BK-tree plus union-find so duplicate detection remains complete
    # for the configured Hamming threshold without comparing every image pair.
    hash_index = _PerceptualHashIndex()
    parent = {path: path for path in perceptual_values}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for path, value in perceptual_values.items():
        for other in hash_index.query(value, max_distance=4):
            union(path, other)
        hash_index.add(path, value)

    grouped_paths: dict[str, list[str]] = defaultdict(list)
    for path in perceptual_values:
        grouped_paths[find(path)].append(path)
    perceptual_groups = [sorted(group) for group in grouped_paths.values() if len(group) > 1]

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


def duplicate_label_conflicts(
    scan: dict[str, Any],
    rows: list[dict[str, Any]],
    image_index: dict[str, Path],
    raw_path: Path,
    definition: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find duplicate candidates whose source labels disagree.

    These files are not relabeled automatically. Split generation can exclude
    them from supervised training while retaining the evidence in the report.
    """
    labels_by_path: dict[Path, int] = {}
    for row in rows:
        path = resolve_image_ref(row.get("image_ref"), image_index, raw_path, definition)
        if path is None or row.get("label") in (None, ""):
            continue
        try:
            labels_by_path[path] = int(str(row["label"]).strip())
        except (KeyError, TypeError, ValueError):
            continue

    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for duplicate_type, groups in (
        ("exact", scan.get("exact_duplicate_groups", [])),
        ("perceptual", scan.get("perceptual_duplicate_groups", [])),
    ):
        for index, group in enumerate(groups, start=1):
            labeled = []
            for relative in group:
                path = (raw_path / relative).resolve()
                if path in labels_by_path:
                    labeled.append({"image": relative, "label": labels_by_path[path]})
            label_values = sorted({item["label"] for item in labeled})
            key = tuple(sorted(item["image"] for item in labeled))
            if len(label_values) > 1 and key not in seen:
                seen.add(key)
                conflicts.append({
                    "type": duplicate_type,
                    "group_id": f"{duplicate_type}-duplicate-{index:04d}",
                    "images": labeled,
                    "labels": label_values,
                    "action": "Exclude from supervised split generation pending source-label review; do not relabel automatically.",
                })
    return conflicts


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
    scan = load_cached_scan(slug, raw_path, files) or scan_images(files, raw_path)
    rows, annotation_errors = read_annotation_rows(raw_path, definition)
    labels = _label_summary(rows, build_image_index(files, raw_path), raw_path, definition)
    duplicate_conflicts = duplicate_label_conflicts(scan, rows, build_image_index(files, raw_path), raw_path, definition)
    split_payload = load_split(slug)
    readable_by_stem = {Path(item["path"]).stem.lower(): item for item in scan["readable"]}
    records = []
    for row in rows:
        image_ref = row.get("image_ref")
        item = readable_by_stem.get(Path(image_ref).stem.lower()) if image_ref else None
        if item is not None:
            records.append({"image": item["path"], "group_source": "patient_id" if row.get("group") else "duplicate_or_image"})
    leakage = leakage_report(slug, records, split_payload)
    score = readiness_score(scan, labels, leakage)
    status = "pass" if scan["corrupted_files"] == 0 and not labels["invalid_labels"] and not labels["missing_image_references"] and leakage["status"] in {"pass", "not_run"} else "blocked"
    return {
        "dataset": {"slug": slug, "name": definition["name"], "purpose": definition["purpose"], "raw_path": str(raw_path), "registry_status": definition["status"]},
        "generated_at": utc_now(), "status": status, "annotation_errors": annotation_errors,
        "files": {
            key: value for key, value in scan.items() if key not in {"phash_by_path"}
        } | {"scan_fingerprint": scan_fingerprint(files, raw_path)},
        "labels": labels, "duplicate_label_conflicts": duplicate_conflicts,
        "training_eligibility": {
            "status": "requires_conflict_exclusion" if duplicate_conflicts else "eligible",
            "excluded_label_conflict_groups": len(duplicate_conflicts),
            "note": "Conflicting duplicate candidates are retained in the raw dataset and excluded from supervised manifests without relabeling." if duplicate_conflicts else "No conflicting duplicate candidates were found.",
        },
        "leakage": leakage, "readiness": score,
        "limitations": definition.get("limitations", []),
    }
