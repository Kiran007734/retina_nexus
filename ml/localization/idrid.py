"""IDRiD optic-disc/fovea localization data and model utilities.

Coordinates remain in the original image convention: pixel coordinates with
the origin at the top-left of the 4288x2848 fundus image. Training transforms
are explicitly recorded so padded/resized coordinates remain reversible.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
LOCALIZATION_ROOT = ROOT / "ml" / "datasets" / "raw" / "idrid" / "C. Localization" / "C. Localization"
IMAGE_ROOT = LOCALIZATION_ROOT / "1. Original Images"
GROUNDTRUTH_ROOT = LOCALIZATION_ROOT / "2. Groundtruths"
TARGET_W = 512
TARGET_H = 352
HEATMAP_W = TARGET_W // 4
HEATMAP_H = TARGET_H // 4
LANDMARKS = ("optic_disc", "fovea")

CSV_SPECS = {
    ("train", "optic_disc"): GROUNDTRUTH_ROOT / "1. Optic Disc Center Location" / "a. IDRiD_OD_Center_Training Set_Markups.csv",
    ("test", "optic_disc"): GROUNDTRUTH_ROOT / "1. Optic Disc Center Location" / "b. IDRiD_OD_Center_Testing Set_Markups.csv",
    ("train", "fovea"): GROUNDTRUTH_ROOT / "2. Fovea Center Location" / "IDRiD_Fovea_Center_Training Set_Markups.csv",
    ("test", "fovea"): GROUNDTRUTH_ROOT / "2. Fovea Center Location" / "IDRiD_Fovea_Center_Testing Set_Markups.csv",
}
IMAGE_DIRS = {"train": IMAGE_ROOT / "a. Training Set", "test": IMAGE_ROOT / "b. Testing Set"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path, size: int = 16) -> str:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    threshold = float(np.median(gray))
    value = 0
    for bit in (gray >= threshold).reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def read_image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return {"width": image.width, "height": image.height, "mode": image.mode, "format": image.format, "channels": len(image.getbands())}


def read_annotations(path: Path) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    values: dict[str, tuple[float, float]] = {}
    summary: dict[str, Any] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "columns": [], "blank_rows_ignored": 0, "invalid_rows": [], "duplicate_records": [], "conflicting_records": []}
    seen: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        rows = list(reader)
    summary["columns"] = rows[0] if rows else []
    for row_number, row in enumerate(rows[1:], start=2):
        cells = [str(value).strip() for value in row[:3]]
        if not any(cells):
            summary["blank_rows_ignored"] += 1
            continue
        if len(cells) < 3 or not cells[0]:
            summary["invalid_rows"].append({"row": row_number, "reason": "missing image or coordinate columns"})
            continue
        image_id = cells[0]
        try:
            point = (float(cells[1]), float(cells[2]))
        except ValueError:
            summary["invalid_rows"].append({"row": row_number, "image_id": image_id, "reason": "non-numeric coordinates"})
            continue
        if image_id in seen:
            if seen[image_id] == point:
                summary["duplicate_records"].append({"row": row_number, "image_id": image_id, "coordinates": point})
            else:
                summary["conflicting_records"].append({"row": row_number, "image_id": image_id, "previous": seen[image_id], "current": point})
            continue
        seen[image_id] = point
        values[image_id] = point
    summary["nonempty_records"] = len(values)
    return values, summary


def image_path(image_id: str, split: str) -> Path:
    return IMAGE_DIRS[split] / f"{image_id}.jpg"


def build_records(include_test: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotation_values: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    csv_summaries = []
    for (split, landmark), path in CSV_SPECS.items():
        values, summary = read_annotations(path)
        annotation_values[(split, landmark)] = values
        summary.update({"split": split, "landmark": landmark})
        csv_summaries.append(summary)
    records: list[dict[str, Any]] = []
    splits = ("train", "test") if include_test else ("train",)
    for split in splits:
        for path in sorted(IMAGE_DIRS[split].glob("*.jpg")):
            image_id = path.stem
            try:
                metadata = read_image_metadata(path)
                image_error = None
                image_sha = sha256(path)
                phash = perceptual_hash(path)
            except Exception as exc:
                metadata = {"width": None, "height": None, "mode": None, "format": None, "channels": None}
                image_error = f"{type(exc).__name__}: {exc}"
                image_sha = None
                phash = None
            annotations: dict[str, Any] = {}
            for landmark in LANDMARKS:
                point = annotation_values[(split, landmark)].get(image_id)
                width, height = metadata.get("width"), metadata.get("height")
                valid = point is not None and width is not None and height is not None and 0 <= point[0] < width and 0 <= point[1] < height
                annotations[landmark] = {
                    "x": point[0] if point else None,
                    "y": point[1] if point else None,
                    "x_norm": point[0] / width if valid else None,
                    "y_norm": point[1] / height if valid else None,
                    "valid": valid,
                    "coordinate_convention": "original-image pixels, origin top-left, x horizontal/y vertical",
                    "error": None if valid else ("missing annotation" if point is None else "coordinate outside image bounds"),
                }
            records.append({
                "image_id": image_id,
                "split": split,
                "image_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "image": {**metadata, "sha256": image_sha, "perceptual_hash": phash, "read_error": image_error},
                "annotations": annotations,
            })
    exact_groups: dict[str, str] = {}
    phash_groups: dict[str, str] = {}
    for record in records:
        exact = record["image"].get("sha256") or record["image_id"]
        phash = record["image"].get("perceptual_hash") or record["image_id"]
        exact_groups.setdefault(exact, f"exact_{len(exact_groups):04d}")
        phash_groups.setdefault(phash, f"phash_{len(phash_groups):04d}")
        record["duplicate_group"] = exact_groups[exact]
        record["perceptual_duplicate_group"] = phash_groups[phash]
    return records, {"csv_files": csv_summaries, "coordinate_convention": "Pixel coordinates refer to the original JPEG dimensions; origin is top-left; x increases rightward and y increases downward.", "units": "pixels", "normalized_coordinates": "x/image_width and y/image_height", "patient_ids_available": False}


def split_training_records(records: list[dict[str, Any]], folds: int = 5, seed: int = 20260913) -> dict[str, Any]:
    test_duplicate_groups = {record["duplicate_group"] for record in records if record["split"] == "test"}
    all_train = [record for record in records if record["split"] == "train"]
    excluded = [record for record in all_train if record["duplicate_group"] in test_duplicate_groups]
    train = [record for record in all_train if record["duplicate_group"] not in test_duplicate_groups]
    groups = [record["duplicate_group"] for record in train]
    # Stratify by coarse landmark geometry so each development fold sees a
    # range of disc/fovea positions while preserving exact duplicate groups.
    strata = []
    for record in train:
        od = record["annotations"]["optic_disc"]
        fv = record["annotations"]["fovea"]
        strata.append(f"{int(od['x_norm'] * 3)}_{int(od['y_norm'] * 3)}_{int(fv['x_norm'] * 3)}_{int(fv['y_norm'] * 3)}")
    method = "StratifiedGroupKFold"
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        generated = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(train)), strata, groups))
    except Exception:
        from sklearn.model_selection import GroupKFold

        method = "GroupKFold_fallback"
        generated = list(GroupKFold(n_splits=folds).split(np.zeros(len(train)), strata, groups))
    fold_by_id: dict[str, int] = {}
    for fold, (_training_indices, validation_indices) in enumerate(generated, start=1):
        for index in validation_indices:
            fold_by_id[train[index]["image_id"]] = fold
    return {
        "schema_version": "idrid-localization-split-1",
        "seed": seed,
        "fold_count": folds,
        "method": method,
        "patient_grouping": "unavailable; no patient IDs in IDRiD localization package",
        "official_test_used": False,
        "raw_training_count": len(all_train),
        "development_count": len(train),
        "excluded_cross_split_duplicates": [{"image_id": record["image_id"], "image_path": record["image_path"], "duplicate_group": record["duplicate_group"], "reason": "exact/perceptual duplicate of an official test image"} for record in excluded],
        "records": [{"image_id": record["image_id"], "fold": fold_by_id[record["image_id"]], "duplicate_group": record["duplicate_group"], "perceptual_duplicate_group": record["perceptual_duplicate_group"]} for record in train],
        "official_test_records": [{"image_id": record["image_id"], "duplicate_group": record["duplicate_group"], "perceptual_duplicate_group": record["perceptual_duplicate_group"]} for record in records if record["split"] == "test"],
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def transform_points(points: np.ndarray, width: int, height: int, target_w: int = TARGET_W, target_h: int = TARGET_H) -> tuple[np.ndarray, dict[str, float]]:
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    pad_left = (target_w - resized_w) / 2.0
    pad_top = (target_h - resized_h) / 2.0
    output = points.astype(np.float32).copy()
    output[:, 0] = output[:, 0] * scale + pad_left
    output[:, 1] = output[:, 1] * scale + pad_top
    return output, {"scale": scale, "resized_width": resized_w, "resized_height": resized_h, "pad_left": pad_left, "pad_top": pad_top, "target_width": target_w, "target_height": target_h}


def letterbox(image: np.ndarray, points: np.ndarray, target_w: int = TARGET_W, target_h: int = TARGET_H) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    height, width = image.shape[:2]
    transformed, transform = transform_points(points, width, height, target_w, target_h)
    resized_w, resized_h = int(transform["resized_width"]), int(transform["resized_height"])
    resized = np.asarray(Image.fromarray(image, mode="RGB").resize((resized_w, resized_h), Image.Resampling.LANCZOS), dtype=np.uint8)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = np.asarray([0, 0, 0], dtype=np.uint8)
    x0, y0 = int(round(transform["pad_left"])), int(round(transform["pad_top"]))
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas, transformed, transform


def rotate_with_points(image: Image.Image, points: np.ndarray, angle: float) -> tuple[Image.Image, np.ndarray]:
    width, height = image.size
    radians = math.radians(angle)
    cos_value, sin_value = math.cos(radians), math.sin(radians)
    centered = points - np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    transformed = np.empty_like(centered)
    transformed[:, 0] = centered[:, 0] * cos_value + centered[:, 1] * sin_value
    transformed[:, 1] = -centered[:, 0] * sin_value + centered[:, 1] * cos_value
    transformed += np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)), transformed


def gaussian_heatmaps(points: np.ndarray, sigma: float = 2.5, height: int = HEATMAP_H, width: int = HEATMAP_W) -> np.ndarray:
    heatmaps = np.zeros((len(points), height, width), dtype=np.float32)
    for index, (x, y) in enumerate(points):
        hx = x / TARGET_W * width
        hy = y / TARGET_H * height
        x0, x1 = max(0, int(hx - 4 * sigma)), min(width, int(hx + 4 * sigma + 1))
        y0, y1 = max(0, int(hy - 4 * sigma)), min(height, int(hy + 4 * sigma + 1))
        xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        heatmaps[index, y0:y1, x0:x1] = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sigma * sigma))
    return heatmaps


def localization_tensor(image: np.ndarray):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class LandmarkDataset:
    def __init__(self, records: list[dict[str, Any]], augment: bool = False, seed: int = 20260913):
        self.records = records
        self.augment = augment
        self.seed = seed
        # The source files are 4288x2848 JPEGs.  Decode and letterbox once per
        # dataset instance so CV does not repeat expensive full-resolution
        # JPEG work for every epoch.  Augmentation is still applied to a copy
        # of the cached model-space canvas, so it remains stochastic and
        # geometry-aware.
        self._base_cache: list[tuple[np.ndarray, np.ndarray, dict[str, float]]] = []
        for record in records:
            image = load_rgb(ROOT / record["image_path"])
            points = np.asarray([[record["annotations"][name]["x"], record["annotations"][name]["y"]] for name in LANDMARKS], dtype=np.float32)
            canvas, transformed, transform = letterbox(image, points)
            self._base_cache.append((canvas, transformed, transform))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        base_canvas, base_points, transform = self._base_cache[index]
        image = Image.fromarray(base_canvas.copy(), mode="RGB")
        points = base_points.copy()
        if self.augment:
            rng = random.Random(self.seed + index)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                points[:, 0] = image.width - 1 - points[:, 0]
            if rng.random() < 0.35:
                image, points = rotate_with_points(image, points, rng.uniform(-4.0, 4.0))
            if rng.random() < 0.4:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
            if rng.random() < 0.4:
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.08))
        canvas = np.asarray(image, dtype=np.uint8)
        heatmaps = gaussian_heatmaps(points)
        coords = points / np.asarray([TARGET_W, TARGET_H], dtype=np.float32)
        return localization_tensor(canvas), torch.from_numpy(heatmaps), torch.from_numpy(coords), record["image_id"], transform


class SharedLandmarkHeatmapNet:
    """Factory wrapper for a compact shared encoder with two landmark heads."""

    @staticmethod
    def build():
        import torch.nn as nn

        class ConvBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
                super().__init__()
                self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

            def forward(self, value):
                return self.block(value)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(ConvBlock(3, 24, 2), ConvBlock(24, 48, 2), ConvBlock(48, 96, 1))
                self.heatmap_head = nn.Conv2d(96, 2, 1)
                self.coordinate_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 64), nn.ReLU(inplace=True), nn.Linear(64, 4))

            def forward(self, value):
                features = self.encoder(value)
                return {"heatmaps": self.heatmap_head(features), "coordinates": self.coordinate_head(features).sigmoid().view(-1, 2, 2)}

        return Model()


def decode_heatmaps(logits, temperature: float = 5.0):
    import torch

    batch, channels, height, width = logits.shape
    probabilities = torch.softmax(logits.flatten(2) * temperature, dim=-1).view(batch, channels, height, width)
    x_grid = torch.linspace(0, TARGET_W, width, device=logits.device).view(1, 1, 1, width)
    y_grid = torch.linspace(0, TARGET_H, height, device=logits.device).view(1, 1, height, 1)
    x = (probabilities * x_grid).sum(dim=(2, 3))
    y = (probabilities * y_grid).sum(dim=(2, 3))
    points = torch.stack([x, y], dim=-1)
    peak = torch.sigmoid(logits.flatten(2).amax(dim=-1))
    entropy = -(probabilities.flatten(2).clamp_min(1e-8) * probabilities.flatten(2).clamp_min(1e-8).log()).sum(dim=-1) / math.log(height * width)
    confidence = peak * (1.0 - entropy)
    return points, confidence, probabilities


def inverse_points(points: np.ndarray, transform: dict[str, float]) -> np.ndarray:
    """Map model-canvas points back to original image pixels."""
    output = np.asarray(points, dtype=np.float32).copy()
    output[:, 0] = (output[:, 0] - transform["pad_left"]) / transform["scale"]
    output[:, 1] = (output[:, 1] - transform["pad_top"]) / transform["scale"]
    return output


def localization_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate landmark errors without hiding failed predictions."""
    output: dict[str, Any] = {}
    for landmark_index, landmark in enumerate(LANDMARKS):
        measured = [row for row in rows if row.get("predicted") is not None and row["ground_truth"][landmark_index] is not None]
        errors = np.asarray([np.linalg.norm(np.asarray(row["predicted"])[landmark_index] - np.asarray(row["ground_truth"])[landmark_index]) for row in measured], dtype=np.float64)
        diagonal = np.asarray([math.hypot(row["image_width"], row["image_height"]) for row in measured], dtype=np.float64)
        normalized = errors / np.maximum(diagonal, 1.0)
        x_errors = np.asarray([abs(float(row["predicted"][landmark_index][0]) - float(row["ground_truth"][landmark_index][0])) for row in measured], dtype=np.float64)
        y_errors = np.asarray([abs(float(row["predicted"][landmark_index][1]) - float(row["ground_truth"][landmark_index][1])) for row in measured], dtype=np.float64)
        output[landmark] = {
            "status": "MEASURED" if len(errors) else "UNAVAILABLE",
            "support": len(measured),
            "failures": len(rows) - len(measured),
            "mean_euclidean_error_px": float(np.mean(errors)) if len(errors) else None,
            "median_euclidean_error_px": float(np.median(errors)) if len(errors) else None,
            "p90_euclidean_error_px": float(np.percentile(errors, 90)) if len(errors) else None,
            "max_euclidean_error_px": float(np.max(errors)) if len(errors) else None,
            "mean_normalized_error": float(np.mean(normalized)) if len(errors) else None,
            "median_normalized_error": float(np.median(normalized)) if len(errors) else None,
            "p90_normalized_error": float(np.percentile(normalized, 90)) if len(errors) else None,
            "x_coordinate_mae_px": float(np.mean(x_errors)) if len(errors) else None,
            "y_coordinate_mae_px": float(np.mean(y_errors)) if len(errors) else None,
            "within_1pct_diagonal": float(np.mean(normalized <= 0.01)) if len(errors) else None,
            "within_2pct_diagonal": float(np.mean(normalized <= 0.02)) if len(errors) else None,
            "within_5pct_diagonal": float(np.mean(normalized <= 0.05)) if len(errors) else None,
            "within_10pct_diagonal": float(np.mean(normalized <= 0.10)) if len(errors) else None,
        }
    return output


def localization_loss(outputs, heatmaps, coords, coordinate_weight: float = 0.25):
    import torch.nn.functional as F

    heatmap_loss = F.mse_loss(outputs["heatmaps"].sigmoid(), heatmaps)
    coordinate_loss = F.smooth_l1_loss(outputs["coordinates"], coords)
    return heatmap_loss + coordinate_weight * coordinate_loss, {"heatmap_mse": float(heatmap_loss.detach().cpu()), "coordinate_smooth_l1": float(coordinate_loss.detach().cpu())}
