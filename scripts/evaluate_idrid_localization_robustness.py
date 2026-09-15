"""Development-only perturbation robustness check for the frozen landmark model."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.localization.idrid import LANDMARKS, decode_heatmaps, inverse_points, letterbox, localization_tensor, load_rgb, SharedLandmarkHeatmapNet  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MODEL_DIR = ROOT / "ml" / "weights" / "localization" / "idrid"
OUTPUT = META / "idrid_localization_robustness.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variants(image: Image.Image) -> dict[str, Image.Image]:
    array = np.asarray(image, dtype=np.float32)
    rng = np.random.default_rng(20260913)
    noisy = np.clip(array + rng.normal(0.0, 2.0, array.shape), 0, 255).astype(np.uint8)
    rotated = image.rotate(2.0, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    compressed_buffer = io.BytesIO()
    image.save(compressed_buffer, format="JPEG", quality=70)
    compressed_buffer.seek(0)
    return {
        "brightness_minus_10pct": ImageEnhance.Brightness(image).enhance(0.90),
        "brightness_plus_10pct": ImageEnhance.Brightness(image).enhance(1.10),
        "contrast_minus_10pct": ImageEnhance.Contrast(image).enhance(0.90),
        "contrast_plus_10pct": ImageEnhance.Contrast(image).enhance(1.10),
        "mild_blur": image.filter(ImageFilter.GaussianBlur(radius=0.8)),
        "mild_noise": Image.fromarray(noisy, mode="RGB"),
        "small_rotation": rotated,
        "jpeg_quality_70": Image.open(compressed_buffer).convert("RGB"),
    }


def predict(model, image: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    points = np.zeros((2, 2), dtype=np.float32)
    canvas, _dummy, transform = letterbox(image, points)
    with torch.inference_mode():
        output = model(localization_tensor(canvas).unsqueeze(0).to(device))
        decoded, confidence, _probabilities = decode_heatmaps(output["heatmaps"])
    return inverse_points(decoded[0].detach().cpu().numpy(), transform), confidence[0].detach().cpu().numpy()


def main() -> int:
    import torch

    manifest = json.loads((META / "idrid_localization_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((META / "idrid_localization_split.json").read_text(encoding="utf-8"))
    checkpoint = MODEL_DIR / "checkpoint_best.pt"
    model_manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    if not checkpoint.is_file() or sha256(checkpoint) != model_manifest.get("checkpoint_sha256"):
        raise SystemExit("Frozen checkpoint is missing or SHA-256 does not match its manifest")
    development_ids = {item["image_id"] for item in split["records"]}
    records = sorted([item for item in manifest["records"] if item["split"] == "train" and item["image_id"] in development_ids], key=lambda item: item["image_id"])[:20]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SharedLandmarkHeatmapNet.build().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    model.eval()
    rows = []
    for record in records:
        source = Image.fromarray(load_rgb(ROOT / record["image_path"]), mode="RGB")
        base_points, base_confidence = predict(model, np.asarray(source), device)
        perturbation_rows = []
        for name, altered in variants(source).items():
            points, confidence = predict(model, np.asarray(altered), device)
            diagonal = max(float(np.hypot(source.width, source.height)), 1.0)
            displacement = np.linalg.norm(points - base_points, axis=1) / diagonal
            perturbation_rows.append({"variant": name, "normalized_displacement": {landmark: float(displacement[index]) for index, landmark in enumerate(LANDMARKS)}, "confidence": {landmark: float(confidence[index]) for index, landmark in enumerate(LANDMARKS)}})
        rows.append({"image_id": record["image_id"], "base_confidence": {landmark: float(base_confidence[index]) for index, landmark in enumerate(LANDMARKS)}, "perturbations": perturbation_rows})
    all_displacements = []
    for row in rows:
        for perturbation in row["perturbations"]:
            for landmark in LANDMARKS:
                all_displacements.append(float(perturbation["normalized_displacement"][landmark]))
    payload = {"schema_version": "idrid-localization-robustness-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "model_version": model_manifest["model_version"], "checkpoint_sha256": model_manifest["checkpoint_sha256"], "development_images_sampled": len(records), "variants_per_image": 8, "mean_normalized_coordinate_displacement": float(np.mean(all_displacements)) if all_displacements else None, "p95_normalized_coordinate_displacement": float(np.percentile(all_displacements, 95)) if all_displacements else None, "rows": rows, "official_test_images_opened": 0, "production_promoted": False, "clinical_validation_claim": False, "note": "Engineering perturbation stability only; not clinical robustness validation."}
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("model_version", "development_images_sampled", "variants_per_image", "mean_normalized_coordinate_displacement", "p95_normalized_coordinate_displacement", "official_test_images_opened")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
