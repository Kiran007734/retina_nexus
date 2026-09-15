"""IDRiD lesion segmentation data, model, and metric utilities.

The official IDRiD segmentation test set is represented in the manifest for
audit purposes, but training helpers accept only the official training split.
Missing annotations are availability flags, never negative labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
IDRID_ROOT = ROOT / "ml" / "datasets" / "raw" / "idrid" / "A. Segmentation" / "A. Segmentation"
IMAGE_ROOT = IDRID_ROOT / "1. Original Images"
MASK_ROOT = IDRID_ROOT / "2. All Segmentation Groundtruths"
LESION_CLASSES = ("microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates")
CLASS_CODES = {"microaneurysms": "MA", "haemorrhages": "HE", "hard_exudates": "EX", "soft_exudates": "SE"}
CLASS_DIRS = {
    "microaneurysms": "1. Microaneurysms",
    "haemorrhages": "2. Haemorrhages",
    "hard_exudates": "3. Hard Exudates",
    "soft_exudates": "4. Soft Exudates",
}
IMAGE_SPLIT_DIRS = {"train": "a. Training Set", "test": "b. Testing Set"}


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


def image_path(image_id: str, split: str) -> Path:
    return IMAGE_ROOT / IMAGE_SPLIT_DIRS[split] / f"{image_id}.jpg"


def mask_path(image_id: str, split: str, lesion_class: str) -> Path:
    return MASK_ROOT / IMAGE_SPLIT_DIRS[split] / CLASS_DIRS[lesion_class] / f"{image_id}_{CLASS_CODES[lesion_class]}.tif"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path, size: int = 16) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(gray, dtype=np.float32)
    threshold = float(np.median(array))
    bits = (array >= threshold).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def read_image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return {"width": image.width, "height": image.height, "mode": image.mode, "format": image.format, "channels": len(image.getbands())}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_binary_mask(path: Path) -> np.ndarray:
    """Load an IDRiD mask without rewriting raw files.

    IDRiD_81_EX.tif is RGBA with the varying annotation in red. For other
    multi-channel files, the first channel carrying non-zero information is
    selected; the choice is recorded by the audit.
    """
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
        mode = image.mode
    if array.ndim == 2:
        return array > 0
    channels = [array[..., index] for index in range(array.shape[-1])]
    if mode == "RGBA" and np.unique(channels[0]).size > 1 and np.all(channels[1] == 0) and np.all(channels[2] == 0):
        return channels[0] > 0
    varying = [channel for channel in channels if np.unique(channel).size > 1]
    selected = max(varying or channels, key=lambda channel: int(np.count_nonzero(channel)))
    return selected > 0


def mask_metadata(path: Path | None, image_size: tuple[int, int]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "UNAVAILABLE", "path": None, "readable": None, "active_pixels": None, "pixel_percentage": None}
    try:
        with Image.open(path) as image:
            image.load()
            raw = np.asarray(image)
            mode = image.mode
            bands = image.getbands()
            dimensions = image.size
            unique = {str(band): sorted(int(value) for value in np.unique(raw[..., index] if raw.ndim == 3 else raw)) for index, band in enumerate(bands)}
        mask = load_binary_mask(path)
        active = int(np.count_nonzero(mask))
        total = int(mask.size)
        return {
            "status": "AVAILABLE",
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
            "readable": True,
            "mode": mode,
            "channels": len(bands),
            "width": dimensions[0],
            "height": dimensions[1],
            "dimension_match": dimensions == image_size,
            "unique_values": unique,
            "active_pixels": active,
            "pixel_percentage": round(active / max(1, total), 10),
            "active_channel": "R" if mode == "RGBA" and path.name.endswith("_EX.tif") and "R" in bands else (bands[0] if bands else None),
        }
    except Exception as exc:
        return {"status": "UNREADABLE", "path": str(path.relative_to(ROOT)).replace("\\", "/"), "readable": False, "error": f"{type(exc).__name__}: {exc}"}


def build_records(include_test: bool = True, include_pixel_stats: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    splits = ("train", "test") if include_test else ("train",)
    for split in splits:
        directory = IMAGE_ROOT / IMAGE_SPLIT_DIRS[split]
        for path in sorted(directory.glob("*.jpg")):
            image_id = path.stem
            try:
                metadata = read_image_metadata(path)
                image_error = None
                image_hash = sha256(path)
                phash = perceptual_hash(path)
            except Exception as exc:
                metadata = {"width": None, "height": None, "mode": None, "format": None, "channels": None}
                image_error = f"{type(exc).__name__}: {exc}"
                image_hash = None
                phash = None
            image_size = (metadata.get("width"), metadata.get("height"))
            masks: dict[str, dict[str, Any]] = {}
            for lesion_class in LESION_CLASSES:
                candidate = mask_path(image_id, split, lesion_class)
                masks[lesion_class] = mask_metadata(candidate if candidate.is_file() else None, image_size) if include_pixel_stats else {
                    "status": "AVAILABLE" if candidate.is_file() else "UNAVAILABLE",
                    "path": str(candidate.relative_to(ROOT)).replace("\\", "/") if candidate.is_file() else None,
                }
            records.append({
                "image_id": image_id,
                "split": split,
                "image_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "image": {**metadata, "sha256": image_hash, "perceptual_hash": phash, "read_error": image_error},
                "masks": masks,
                "fov": {"status": "UNAVAILABLE", "reason": "No official IDRiD FOV mask was present in the segmentation package."},
            })
    groups: dict[str, str] = {}
    phash_groups: dict[str, str] = {}
    for record in records:
        image_hash = record["image"].get("sha256") or record["image_id"]
        phash = record["image"].get("perceptual_hash") or record["image_id"]
        groups.setdefault(image_hash, f"exact_{len(groups):03d}")
        phash_groups.setdefault(phash, f"phash_{len(phash_groups):03d}")
        record["duplicate_group"] = groups[image_hash]
        record["perceptual_duplicate_group"] = phash_groups[phash]
    return records


def split_training_records(records: list[dict[str, Any]], folds: int = 5, seed: int = 20260913) -> dict[str, Any]:
    train = [record for record in records if record["split"] == "train"]
    groups = [record["duplicate_group"] for record in train]
    strata = []
    for record in train:
        available = [name for name in LESION_CLASSES if record["masks"][name]["status"] == "AVAILABLE"]
        positive = sum(1 for name in available if (record["masks"][name].get("active_pixels") or 0) > 0)
        strata.append(str(positive))
    fold_assignments: dict[str, int] = {}
    method = "StratifiedGroupKFold"
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        generated = list(splitter.split(np.zeros(len(train)), strata, groups))
    except Exception:
        from sklearn.model_selection import GroupKFold

        method = "GroupKFold_fallback_due_to_small_strata"
        generated = list(GroupKFold(n_splits=folds).split(np.zeros(len(train)), strata, groups))
    for fold, (_train_indices, validation_indices) in enumerate(generated, start=1):
        for index in validation_indices:
            fold_assignments[train[index]["image_id"]] = fold
    for record in train:
        record["fold"] = fold_assignments[record["image_id"]]
    return {
        "schema_version": "idrid-lesion-split-1",
        "seed": seed,
        "fold_count": folds,
        "method": method,
        "official_test_used": False,
        "records": [{"image_id": record["image_id"], "fold": record["fold"], "duplicate_group": record["duplicate_group"], "perceptual_duplicate_group": record["perceptual_duplicate_group"]} for record in train],
    }


def resize_pair(image: np.ndarray, masks: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image_pil = Image.fromarray(image, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)
    target = np.zeros((masks.shape[0], size, size), dtype=np.float32)
    for index, mask in enumerate(masks):
        target[index] = np.asarray(Image.fromarray(mask.astype(np.uint8), mode="L").resize((size, size), Image.Resampling.NEAREST), dtype=np.float32) > 0
    return np.asarray(image_pil, dtype=np.uint8), target


def load_training_sample(record: dict[str, Any], size: int, augment: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = load_rgb(ROOT / record["image_path"])
    masks = np.zeros((len(LESION_CLASSES), image.shape[0], image.shape[1]), dtype=np.float32)
    availability = np.zeros(len(LESION_CLASSES), dtype=np.float32)
    for index, lesion_class in enumerate(LESION_CLASSES):
        info = record["masks"][lesion_class]
        if info["status"] != "AVAILABLE":
            continue
        path = ROOT / info["path"]
        masks[index] = load_binary_mask(path)
        availability[index] = 1.0
    image, masks = resize_pair(image, masks, size)
    if augment:
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            masks = np.ascontiguousarray(masks[:, :, ::-1])
        if random.random() < 0.35:
            image = np.asarray(ImageEnhance.Brightness(Image.fromarray(image)).enhance(random.uniform(0.9, 1.1)), dtype=np.uint8)
        if random.random() < 0.35:
            image = np.asarray(ImageEnhance.Contrast(Image.fromarray(image)).enhance(random.uniform(0.9, 1.1)), dtype=np.uint8)
    return image, masks, availability


def image_tensor(image: np.ndarray):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def build_idrid_model(initial_checkpoint: Path | None = None):
    import torch
    import torchseg

    model = torchseg.create_model(arch="unet", encoder_name="se_resnext50_32x4d", encoder_weights=None, in_channels=3, classes=4)
    if initial_checkpoint is None:
        return model, {"initialized_from": None, "transferred_keys": 0}
    try:
        from safetensors.torch import load_file

        source = load_file(str(initial_checkpoint), device="cpu")
    except Exception as exc:
        raise RuntimeError(f"Could not load baseline lesion checkpoint for transfer initialization: {exc}") from exc
    target = model.state_dict()
    mapped: dict[str, Any] = {}
    for key, value in source.items():
        normalized = key[6:] if key.startswith("model.") else key
        if normalized.startswith("encoder.model."):
            normalized = "encoder." + normalized[len("encoder.model.") :]
        normalized = normalized.replace(".se.", ".se_module.")
        if normalized.startswith("encoder.conv1."):
            normalized = "encoder.layer0." + normalized[len("encoder.") :]
        elif normalized.startswith("encoder.bn1."):
            normalized = "encoder.layer0." + normalized[len("encoder.") :]
        if normalized in target and target[normalized].shape == value.shape:
            mapped[normalized] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    # Transfer the published four lesion heads into the new multi-label order.
    source_weight = source.get("segmentation_head.0.weight")
    if source_weight is None:
        source_weight = source.get("model.segmentation_head.0.weight")
    source_bias = source.get("segmentation_head.0.bias")
    if source_bias is None:
        source_bias = source.get("model.segmentation_head.0.bias")
    if source_weight is not None and source_bias is not None:
        with torch.no_grad():
            mapping = [4, 3, 2, 1]  # MA, HE, hard EX, soft EX/cotton wool
            model.segmentation_head[0].weight.copy_(source_weight[mapping])
            model.segmentation_head[0].bias.copy_(source_bias[mapping])
    return model, {"initialized_from": str(initial_checkpoint.relative_to(ROOT)).replace("\\", "/"), "transferred_keys": len(mapped), "missing_keys": len(missing), "unexpected_keys": len(unexpected), "head_mapping": {"0": "published_class_4_microaneurysm", "1": "published_class_3_hemorrhage", "2": "published_class_2_exudate", "3": "published_class_1_cotton_wool_spot"}}


def masked_focal_dice_loss(logits, targets, availability, pos_weight=None, gamma: float = 2.0):
    import torch
    import torch.nn.functional as F

    available = availability.view(-1, logits.shape[1], 1, 1)
    if pos_weight is None:
        pos_weight = torch.ones(logits.shape[1], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight.view(1, -1, 1, 1))
    probability = torch.sigmoid(logits)
    focal_factor = torch.where(targets > 0.5, (1 - probability) ** gamma, probability ** gamma)
    # Normalize over valid pixels, not merely valid class channels.  Dividing
    # by ``available.sum()`` alone scales the loss by H*W and can produce
    # million-scale losses on full retinal frames.
    valid_pixel_count = (available.sum() * logits.shape[-2] * logits.shape[-1]).clamp_min(1.0)
    bce = (bce * focal_factor * available).sum() / valid_pixel_count
    intersection = (probability * targets * available).flatten(2).sum(2)
    denominator = ((probability + targets) * available).flatten(2).sum(2)
    dice = ((2 * intersection + 1.0) / (denominator + 1.0))
    dice_loss = 1.0 - dice.sum() / available.flatten(1).sum().clamp_min(1.0)
    return bce + dice_loss


def segmentation_metrics(probabilities: np.ndarray, targets: np.ndarray, availability: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets).astype(bool)
    availability = np.asarray(availability).astype(bool)
    per_class: dict[str, Any] = {}
    dice_values: list[float] = []
    iou_values: list[float] = []
    for index, name in enumerate(LESION_CLASSES):
        valid = availability[:, index]
        if not np.any(valid):
            per_class[name] = {"status": "UNAVAILABLE", "support_images": 0}
            continue
        pred = probabilities[valid, index] >= threshold
        truth = targets[valid, index]
        tp = int(np.logical_and(pred, truth).sum())
        fp = int(np.logical_and(pred, ~truth).sum())
        fn = int(np.logical_and(~pred, truth).sum())
        tn = int(np.logical_and(~pred, ~truth).sum())
        dice = (2 * tp) / max(1, 2 * tp + fp + fn)
        iou = tp / max(1, tp + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        pixel_accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
        positive_images = int(np.any(truth.reshape(truth.shape[0], -1), axis=1).sum())
        predicted_positive_images = int(np.any(pred.reshape(pred.shape[0], -1), axis=1).sum())
        positive_detection = int(np.logical_and(np.any(truth.reshape(truth.shape[0], -1), axis=1), np.any(pred.reshape(pred.shape[0], -1), axis=1)).sum()) / max(1, positive_images)
        per_class[name] = {"status": "MEASURED", "support_images": int(valid.sum()), "positive_images": positive_images, "predicted_positive_images": predicted_positive_images, "dice": dice, "iou": iou, "precision": precision, "recall": recall, "sensitivity": recall, "specificity": specificity, "f1": 2 * precision * recall / max(1e-12, precision + recall), "pixel_accuracy": pixel_accuracy, "lesion_positive_detection_rate": positive_detection, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        dice_values.append(dice)
        iou_values.append(iou)
    return {"threshold": threshold, "per_class": per_class, "macro_dice": float(np.mean(dice_values)) if dice_values else None, "macro_iou": float(np.mean(iou_values)) if iou_values else None}
