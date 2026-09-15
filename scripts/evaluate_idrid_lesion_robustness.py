"""Run a bounded, development-only perturbation robustness check."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.lesions.idrid import LESION_CLASSES, build_idrid_model, image_tensor, load_training_sample  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MANIFEST = META / "idrid_lesion_manifest.json"
MODEL_MANIFEST = ROOT / "ml" / "weights" / "lesions" / "idrid" / "model_manifest.json"
CHECKPOINT = ROOT / "ml" / "weights" / "lesions" / "idrid" / "checkpoint_best.pt"
OUTPUT = META / "idrid_lesion_robustness.json"


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perturbations(image: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    source = Image.fromarray(image, mode="RGB")
    rng = np.random.default_rng(seed)
    noisy = np.clip(image.astype(np.float32) + rng.normal(0.0, 2.0, image.shape), 0, 255).astype(np.uint8)
    rotated = source.rotate(2.0, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    jpeg = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=0.35))
    return {
        "brightness_minus_5pct": np.asarray(ImageEnhance.Brightness(source).enhance(0.95), dtype=np.uint8),
        "contrast_plus_5pct": np.asarray(ImageEnhance.Contrast(source).enhance(1.05), dtype=np.uint8),
        "gaussian_noise_std_2": noisy,
        "rotation_plus_2deg": np.asarray(rotated, dtype=np.uint8),
        "mild_blur": np.asarray(jpeg, dtype=np.uint8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    actual_sha = sha256(CHECKPOINT)
    if actual_sha != model_manifest.get("checkpoint_sha256"):
        raise SystemExit("Frozen checkpoint SHA mismatch")
    import torch

    torch.set_num_threads(args.torch_threads)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model, _transfer = build_idrid_model(None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    records = [record for record in json.loads(MANIFEST.read_text(encoding="utf-8"))["records"] if record["split"] == "train"][: args.sample_count]
    threshold = float(model_manifest["threshold"])
    rows = []
    with torch.inference_mode():
        for record in records:
            image, _masks, _available = load_training_sample(record, int(model_manifest["input_size"]), augment=False)
            variants = {"original": image, **perturbations(image, seed=20260913 + len(rows))}
            predictions: dict[str, np.ndarray] = {}
            for name, variant in variants.items():
                output = model(image_tensor(variant).unsqueeze(0).to(args.device))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                predictions[name] = torch.sigmoid(output).squeeze(0).cpu().numpy()
            base = predictions["original"]
            variant_results = {}
            for name, probability in predictions.items():
                if name == "original":
                    continue
                class_rows = []
                for index, lesion_class in enumerate(LESION_CLASSES):
                    base_mask = base[index] >= threshold
                    variant_mask = probability[index] >= threshold
                    intersection = np.logical_and(base_mask, variant_mask).sum()
                    union = np.logical_or(base_mask, variant_mask).sum()
                    dice = 2 * intersection / max(1, base_mask.sum() + variant_mask.sum())
                    class_rows.append({"class": lesion_class, "mask_dice": float(dice), "mask_iou": float(intersection / max(1, union)), "mean_absolute_probability_change": float(np.mean(np.abs(base[index] - probability[index])))})
                variant_results[name] = class_rows
            rows.append({"image_id": record["image_id"], "variants": variant_results})
    aggregate = {}
    for variant in rows[0]["variants"] if rows else []:
        values = {name: [] for name in ("mask_dice", "mask_iou", "mean_absolute_probability_change")}
        for row in rows:
            for item in row["variants"][variant]:
                for key in values:
                    values[key].append(item[key])
        aggregate[variant] = {key: {"mean": float(np.mean(value)), "std": float(np.std(value)), "min": float(np.min(value)), "max": float(np.max(value))} for key, value in values.items()}
    dump(OUTPUT, {"schema_version": "idrid-lesion-robustness-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "model_version": model_manifest["model_version"], "checkpoint_sha256": actual_sha, "development_only": True, "sample_count": len(records), "sample_image_ids": [record["image_id"] for record in records], "variants": list(aggregate), "aggregate": aggregate, "per_image": rows, "interpretation": "Engineering perturbation consistency only; not clinical validation and not a guarantee of explanation or prediction stability.", "official_test_images_opened": 0, "production_promoted": False})
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"), "sample_count": len(records), "aggregate": aggregate, "official_test_images_opened": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
