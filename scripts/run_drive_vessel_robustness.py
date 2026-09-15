"""Run a small, fixed development-only robustness probe for the frozen vessel model."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from ml.vessels.drive import normalize_input, read_gray, read_rgb, resolve_dataset_path, threshold_metrics  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "drive"
MODEL_PATH = ROOT / "ml" / "weights" / "vessels" / "drive" / "checkpoint_best.pt"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def perturb(image: np.ndarray, name: str) -> np.ndarray:
    pil = Image.fromarray(image, mode="RGB")
    if name == "brightness_minus_10":
        pil = ImageEnhance.Brightness(pil).enhance(0.90)
    elif name == "brightness_plus_10":
        pil = ImageEnhance.Brightness(pil).enhance(1.10)
    elif name == "contrast_minus_10":
        pil = ImageEnhance.Contrast(pil).enhance(0.90)
    elif name == "contrast_plus_10":
        pil = ImageEnhance.Contrast(pil).enhance(1.10)
    elif name == "mild_blur":
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.6))
    elif name == "jpeg_quality_70":
        stream = io.BytesIO()
        pil.save(stream, format="JPEG", quality=70)
        stream.seek(0)
        pil = Image.open(stream).convert("RGB")
    elif name == "rotation_plus_3":
        pil = pil.rotate(3.0, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    elif name == "gaussian_noise":
        rng = np.random.default_rng(20260913)
        return np.clip(np.asarray(pil, dtype=np.float32) + rng.normal(0.0, 3.0, np.asarray(pil).shape), 0, 255).astype(np.uint8)
    elif name == "clean":
        pass
    else:
        raise ValueError(f"Unknown perturbation {name}")
    return np.asarray(pil, dtype=np.uint8)


def model_predict(model, image: np.ndarray, input_size: int, preprocessing: str, torch) -> np.ndarray:
    resized = np.asarray(Image.fromarray(image, mode="RGB").resize((input_size, input_size), Image.Resampling.BILINEAR), dtype=np.uint8)
    normalized = normalize_input(resized, preprocessing)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.inference_mode():
        return torch.sigmoid(model(tensor))[0, 0].cpu().numpy()


def main() -> int:
    import torch
    from app.ml.models.evidence import build_vessel_segmentation_model

    manifest = json.loads((META / "drive_manifest.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((ROOT / "ml" / "weights" / "vessels" / "drive" / "model_manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == "training"]
    records = sorted(records, key=lambda row: row["image_id"])
    sample_records = records[::4]
    if len(sample_records) != 5:
        raise SystemExit(f"Expected five deterministic development samples, found {len(sample_records)}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = build_vessel_segmentation_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    names = ["clean", "brightness_minus_10", "brightness_plus_10", "contrast_minus_10", "contrast_plus_10", "mild_blur", "jpeg_quality_70", "rotation_plus_3", "gaussian_noise"]
    all_rows = []
    for record in sample_records:
        image = read_rgb(resolve_dataset_path(record["image_path"]))
        target = np.asarray(Image.fromarray((read_gray(resolve_dataset_path(record["vessel_mask_path"])) >= 128).astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), dtype=bool)
        fov = np.asarray(Image.fromarray((read_gray(resolve_dataset_path(record["fov_mask_path"])) >= 128).astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST), dtype=bool)
        for name in names:
            probability = model_predict(model, perturb(image, name), 512, model_manifest["preprocessing"]["mode"], torch)
            row = threshold_metrics(target, probability, fov, float(model_manifest["threshold"]))
            all_rows.append({"image_id": record["image_id"], "perturbation": name, "metrics": row, "mean_probability_in_fov": float(np.mean(probability[fov])), "max_probability": float(np.max(probability))})
    summary = {}
    for name in names:
        rows = [row for row in all_rows if row["perturbation"] == name]
        summary[name] = {key: {"mean": float(np.mean([row["metrics"][key] for row in rows])), "std": float(np.std([row["metrics"][key] for row in rows]))} for key in ("dice", "iou", "sensitivity", "specificity", "precision", "f1", "pixel_accuracy")}
    dump(META / "drive_robustness.json", {"status": "COMPLETED_DEVELOPMENT_SAMPLE", "model_version": model_manifest["model_version"], "checkpoint": model_manifest["checkpoint"], "threshold": model_manifest["threshold"], "sample_image_ids": [record["image_id"] for record in sample_records], "perturbations": names, "summary": summary, "per_image": all_rows, "official_test_images_opened": 0, "interpretation": "A fixed engineering robustness probe on five labeled development images; not a clinical robustness or external validation result."})
    print(json.dumps({"status": "COMPLETED_DEVELOPMENT_SAMPLE", "sample_count": len(sample_records), "perturbation_count": len(names), "clean_dice": summary["clean"]["dice"], "official_test_images_opened": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
