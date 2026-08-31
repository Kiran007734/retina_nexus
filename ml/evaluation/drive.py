"""DRIVE discovery, validation, hashing, and segmentation metric helpers.

The DRIVE release has separate image, field-of-view, and manual-vessel-mask
directories.  These helpers deliberately discover files from names and path
tokens instead of assuming one particular archive layout.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MASK_EXTENSIONS = IMAGE_EXTENSIONS | {".gif"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".gif"}
_SUFFIXES = (
    "_test_mask",
    "_training_mask",
    "_manual1",
    "_manual2",
    "_manual",
    "_1st_manual",
    "_2nd_manual",
    "_training",
    "_test",
)


def specimen_id(path: Path) -> str:
    """Return the stable DRIVE specimen identifier from a filename."""
    value = path.stem.lower()
    for suffix in _SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def split_name(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "training" in parts:
        return "training"
    if "test" in parts:
        return "test"
    return "unknown"


def _is_manual_vessel(path: Path) -> bool:
    lowered = str(path).lower().replace("\\", "/")
    name = path.name.lower()
    return "1st_manual" in lowered or "2nd_manual" in lowered or "manual" in name


def _is_fov_mask(path: Path) -> bool:
    lowered = str(path).lower().replace("\\", "/")
    return "/mask/" in f"/{lowered}/" or path.stem.lower().endswith("_mask") or "fov" in path.name.lower()


def classify_file(path: Path) -> str:
    """Classify a supported file as image, vessel mask, or FOV mask."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return "unsupported"
    if _is_manual_vessel(path):
        return "vessel_mask"
    if _is_fov_mask(path):
        return "fov_mask"
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def discover_drive_files(root: str | Path) -> dict[str, list[Path]]:
    root_path = Path(root).expanduser().resolve()
    discovered: dict[str, list[Path]] = defaultdict(list)
    if not root_path.is_dir():
        return {"image": [], "vessel_mask": [], "fov_mask": [], "unknown": [], "unsupported": []}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        discovered[classify_file(path)].append(path)
    return {key: discovered.get(key, []) for key in ("image", "vessel_mask", "fov_mask", "unknown", "unsupported")}


def group_drive_files(root: str | Path) -> dict[str, dict[str, list[Path]]]:
    """Group discovered files by specimen and category."""
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for category, paths in discover_drive_files(root).items():
        for path in paths:
            if category in {"unknown", "unsupported"}:
                continue
            grouped[specimen_id(path)][category].append(path)
    return {key: {category: sorted(paths) for category, paths in value.items()} for key, value in sorted(grouped.items())}


def read_file_info(path: Path, root: Path, category: str) -> dict[str, Any]:
    """Read and verify one image or annotation without changing it."""
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info: dict[str, Any] = {
        "path": relative,
        "category": category,
        "split": split_name(path),
        "specimen_id": specimen_id(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "readable": False,
        "format": None,
        "mode": None,
        "width": None,
        "height": None,
        "channels": None,
        "unique_values": None,
        "annotation_valid": None,
        "error": None,
    }
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            info.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                    "channels": len(image.getbands()),
                }
            )
            array = np.asarray(image.convert("L" if category != "image" else "RGB"))
        if not np.isfinite(array.astype(np.float32)).all():
            raise ValueError("decoded pixels contain non-finite values")
        info["readable"] = True
        if category != "image":
            values = np.unique(array)
            info["unique_values"] = [int(value) for value in values.tolist()]
            info["annotation_valid"] = bool(values.size > 0 and np.all(np.isin(values, [0, 1, 255])))
            if not info["annotation_valid"]:
                info["error"] = "annotation contains values outside {0, 1, 255}"
    except Exception as exc:  # PIL raises several format-specific exceptions.
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def perceptual_hash(path: Path) -> str:
    """Return a compact DCT perceptual hash for an image file."""
    with Image.open(path) as image:
        gray = np.asarray(ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    n = 32
    coordinates = np.arange(n, dtype=np.float32)
    basis = np.cos(np.pi * (2 * coordinates[:, None] + 1) * coordinates[None, :] / (2 * n)).astype(np.float32)
    dct = basis.T @ gray @ basis
    low = dct[:8, :8]
    median = float(np.median(low.reshape(-1)[1:]))
    bits = (low >= median).reshape(-1)
    return "".join("1" if bit else "0" for bit in bits)


def hash_distance(first: str, second: str) -> int:
    if len(first) != len(second):
        return max(len(first), len(second))
    return sum(left != right for left, right in zip(first, second))


def perceptual_duplicate_candidates(records: Iterable[dict[str, Any]], hash_key: str = "perceptual_hash", max_distance: int = 4) -> list[dict[str, Any]]:
    """Return pairwise pHash candidates; visual similarity must be checked separately."""
    usable = [record for record in records if record.get(hash_key)]
    candidates: list[dict[str, Any]] = []
    for index, first in enumerate(usable):
        for second in usable[index + 1 :]:
            distance = hash_distance(str(first[hash_key]), str(second[hash_key]))
            if distance <= max_distance:
                candidates.append({"path_a": str(first["path"]), "path_b": str(second["path"]), "hamming_distance": distance})
    return candidates


def resized_grayscale_mae(first_path: Path, second_path: Path, size: tuple[int, int] = (64, 64)) -> float:
    """Compare normalized grayscale thumbnails for pHash candidate confirmation."""
    arrays = []
    for path in (first_path, second_path):
        with Image.open(path) as image:
            value = np.asarray(ImageOps.grayscale(image).resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        arrays.append(value)
    first, second = arrays
    first = (first - float(first.mean())) / max(float(first.std()), 1e-6)
    second = (second - float(second.mean())) / max(float(second.std()), 1e-6)
    return float(np.mean(np.abs(first - second)))


def duplicate_groups(records: Iterable[dict[str, Any]], hash_key: str, max_distance: int | None = None) -> list[list[str]]:
    usable = [record for record in records if record.get(hash_key)]
    if max_distance is None:
        buckets: dict[str, list[str]] = defaultdict(list)
        for record in usable:
            buckets[str(record[hash_key])].append(str(record["path"]))
        return [sorted(paths) for paths in buckets.values() if len(paths) > 1]
    groups: list[list[str]] = []
    for record in usable:
        path = str(record["path"])
        # The explicit pairwise pass keeps the group representation readable
        # and avoids treating unrelated hashes as metadata.
        placed = False
        for group in groups:
            if any(hash_distance(str(other["hash"]), str(record[hash_key])) <= max_distance for other in group):
                group.append({"hash": record[hash_key], "path": path})
                placed = True
                break
        if not placed:
            groups.append([{"hash": record[hash_key], "path": path}])
    return [sorted(str(item["path"]) for item in group) for group in groups if len(group) > 1]


def segmentation_metrics(target: np.ndarray, probability: np.ndarray, fov: np.ndarray | None = None, threshold: float = 0.5) -> dict[str, float | int]:
    """Calculate binary segmentation metrics within an optional FOV."""
    actual = np.asarray(target).astype(bool)
    predicted = np.asarray(probability) >= threshold
    if actual.shape != predicted.shape:
        raise ValueError(f"target/probability shape mismatch: {actual.shape} vs {predicted.shape}")
    valid = np.ones(actual.shape, dtype=bool) if fov is None else np.asarray(fov).astype(bool)
    if valid.shape != actual.shape:
        raise ValueError(f"FOV/target shape mismatch: {valid.shape} vs {actual.shape}")
    actual = actual[valid]
    predicted = predicted[valid]
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    dice = 2 * tp / max(1, int(actual.sum()) + int(predicted.sum()))
    iou = tp / max(1, int(np.sum(actual | predicted)))
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_accuracy": float((tp + tn) / max(1, len(actual))),
        "sensitivity": float(recall),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "evaluated_pixels": int(len(actual)),
        "ground_truth_vessel_pixels": int(actual.sum()),
        "predicted_vessel_pixels": int(predicted.sum()),
    }
