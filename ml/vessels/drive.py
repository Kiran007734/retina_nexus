"""Leak-safe DRIVE vessel data, preprocessing, training, and metrics helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ml.evaluation.drive import group_drive_files, specimen_id

ROOT = Path(__file__).resolve().parents[2]
LANDMARK = "drive"


def resolve_dataset_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    direct = (ROOT / path).resolve()
    if direct.is_file():
        return direct
    return (ROOT / "ml" / "datasets" / "raw" / "drive" / path).resolve()


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pairs(raw_root: Path, split: str = "training") -> list[dict[str, Any]]:
    grouped = group_drive_files(raw_root)
    rows = []
    for image_id, categories in grouped.items():
        images = [path for path in categories.get("image", []) if ("training" if split == "training" else "test") in {part.lower() for part in path.parts}]
        fovs = categories.get("fov_mask", [])
        vessels = categories.get("vessel_mask", [])
        if len(images) != 1 or len(fovs) != 1:
            continue
        vessel = vessels[0] if vessels else None
        if split == "training" and vessel is None:
            continue
        rows.append({"image_id": image_id, "split": split, "image_path": str(images[0].resolve().relative_to(ROOT)).replace("\\", "/"), "vessel_mask_path": str(vessel.resolve().relative_to(ROOT)).replace("\\", "/") if vessel else None, "fov_mask_path": str(fovs[0].resolve().relative_to(ROOT)).replace("\\", "/")})
    return sorted(rows, key=lambda row: row["image_id"])


def normalize_input(image: np.ndarray, mode: str = "rgb") -> np.ndarray:
    value = image.astype(np.float32) / 255.0
    if mode == "rgb":
        output = value
    elif mode == "green":
        green = value[..., 1]
        output = np.stack([green, green, green], axis=-1)
    elif mode == "clahe_green":
        try:
            import cv2

            green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image[..., 1])
            green = green.astype(np.float32) / 255.0
        except Exception:
            green = value[..., 1]
        output = np.stack([green, green, green], axis=-1)
    else:
        raise ValueError(f"Unsupported DRIVE preprocessing mode: {mode}")
    return np.clip(output, 0.0, 1.0)


def resize_triplet(image: np.ndarray, vessel: np.ndarray, fov: np.ndarray, input_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_out = np.asarray(Image.fromarray(image, mode="RGB").resize((input_size, input_size), Image.Resampling.BILINEAR), dtype=np.uint8)
    vessel_out = np.asarray(Image.fromarray(vessel.astype(np.uint8) * 255, mode="L").resize((input_size, input_size), Image.Resampling.NEAREST), dtype=np.uint8) >= 128
    fov_out = np.asarray(Image.fromarray(fov.astype(np.uint8) * 255, mode="L").resize((input_size, input_size), Image.Resampling.NEAREST), dtype=np.uint8) >= 128
    return image_out, vessel_out, fov_out


def threshold_metrics(target: np.ndarray, probability: np.ndarray, fov: np.ndarray, threshold: float) -> dict[str, float | int]:
    valid = np.asarray(fov).astype(bool)
    actual = np.asarray(target).astype(bool)[valid]
    predicted = (np.asarray(probability) >= threshold)[valid]
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    dice = 2 * tp / max(1, int(actual.sum()) + int(predicted.sum()))
    iou = tp / max(1, int(np.sum(actual | predicted)))
    return {"dice": float(dice), "f1": float(2 * precision * sensitivity / max(1e-12, precision + sensitivity)), "iou": float(iou), "pixel_accuracy": float((tp + tn) / max(1, len(actual))), "sensitivity": float(sensitivity), "specificity": float(specificity), "precision": float(precision), "false_positive_rate": float(fp / max(1, fp + tn)), "false_negative_rate": float(fn / max(1, fn + tp)), "vessel_prevalence": float(actual.mean()) if len(actual) else 0.0, "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "evaluated_pixels": int(len(actual)), "ground_truth_vessel_pixels": int(actual.sum()), "predicted_vessel_pixels": int(predicted.sum())}


def aggregate_metrics(rows: list[dict[str, Any]], keys: tuple[str, ...] = ("dice", "iou", "sensitivity", "specificity", "precision", "pixel_accuracy", "f1", "false_positive_rate", "false_negative_rate")) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "mean": {}, "std": {}, "min": {}, "max": {}}
    return {"sample_count": len(rows), "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys}, "std": {key: float(np.std([row[key] for row in rows])) for key in keys}, "min": {key: float(np.min([row[key] for row in rows])) for key in keys}, "max": {key: float(np.max([row[key] for row in rows])) for key in keys}, "micro_confusion": {key: int(sum(int(row[key]) for row in rows)) for key in ("true_positive", "true_negative", "false_positive", "false_negative", "evaluated_pixels", "ground_truth_vessel_pixels", "predicted_vessel_pixels")}}


def choose_threshold(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    results = []
    for threshold in thresholds:
        metrics = aggregate_metrics([{**threshold_metrics(row["target"], row["probability"], row["fov"], threshold)} for row in rows])
        mean_values = metrics["mean"]
        balanced = float(np.mean([mean_values["dice"], mean_values["iou"], mean_values["sensitivity"], mean_values["specificity"], mean_values["precision"]]))
        results.append({"threshold": threshold, "metrics": metrics, "selection_score": balanced})
    return {"thresholds": results, "selected": max(results, key=lambda item: item["selection_score"])}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class DriveSegmentationDataset:
    def __init__(self, records: list[dict[str, Any]], input_size: int = 512, preprocessing: str = "rgb", augment: bool = False, seed: int = 20260913):
        self.records = records
        self.input_size = input_size
        self.preprocessing = preprocessing
        self.augment = augment
        self.seed = seed
        self._cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for record in records:
            image = read_rgb(resolve_dataset_path(record["image_path"]))
            vessel = read_gray(resolve_dataset_path(record["vessel_mask_path"])) >= 128 if record.get("vessel_mask_path") else np.zeros(image.shape[:2], dtype=bool)
            fov = read_gray(resolve_dataset_path(record["fov_mask_path"])) >= 128
            self._cache.append(resize_triplet(image, vessel, fov, input_size))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        image, vessel, fov = self._cache[index]
        image = Image.fromarray(image.copy(), mode="RGB")
        vessel_image = Image.fromarray(vessel.astype(np.uint8) * 255, mode="L")
        fov_image = Image.fromarray(fov.astype(np.uint8) * 255, mode="L")
        if self.augment:
            rng = random.Random(self.seed + index)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                vessel_image = vessel_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                fov_image = fov_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < 0.35:
                angle = rng.uniform(-7.0, 7.0)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                vessel_image = vessel_image.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
                fov_image = fov_image.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            if rng.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.90, 1.10))
            if rng.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.10))
            if rng.random() < 0.15:
                image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
        image_array = normalize_input(np.asarray(image, dtype=np.uint8), self.preprocessing)
        vessel_array = np.asarray(vessel_image, dtype=np.uint8) >= 128
        fov_array = np.asarray(fov_image, dtype=np.uint8) >= 128
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        return image_tensor, torch.from_numpy(vessel_array[None].astype(np.float32)), torch.from_numpy(fov_array[None].astype(np.float32)), self.records[index]["image_id"]


def make_five_fold_split(records: list[dict[str, Any]], seed: int = 20260913) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: row["image_id"])
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    fold_by_id = {record["image_id"]: (index % 5) + 1 for index, record in enumerate(shuffled)}
    return {"schema_version": "drive-split-1", "seed": seed, "method": "deterministic duplicate-aware five-fold development split", "official_test_used": False, "patient_ids_available": False, "records": [{"image_id": record["image_id"], "fold": fold_by_id[record["image_id"]], "split": "development"} for record in ordered], "official_test_reserved": True}
