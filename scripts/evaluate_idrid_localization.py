"""One-time frozen evaluation on the official 103-image IDRiD test split.

This command is intentionally guarded.  It refuses to run before a frozen
manifest exists, after production promotion, or when a completed official
report already exists.  It must be invoked exactly once after research freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.localization.idrid import LANDMARKS, LandmarkDataset, SharedLandmarkHeatmapNet, decode_heatmaps, inverse_points, localization_metrics  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MODEL_DIR = ROOT / "ml" / "weights" / "localization" / "idrid"
OUTPUT = META / "idrid_localization_official_test.json"
VISUAL_DIR = ROOT / "ml" / "evaluation" / "idrid_localization" / "visuals"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if OUTPUT.is_file():
        raise SystemExit(f"Official evaluation already completed: {OUTPUT}")
    manifest_path = MODEL_DIR / "model_manifest.json"
    checkpoint_path = MODEL_DIR / "checkpoint_best.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit("Frozen localization model manifest/checkpoint is required before official evaluation")
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("production_promoted") is not False:
        raise SystemExit("Official evaluation requires production_promoted=false")
    if int(model_manifest.get("training", {}).get("official_test_images_opened", 0)) != 0:
        raise SystemExit("Frozen manifest indicates official test images were already opened")
    expected_sha = model_manifest.get("checkpoint_sha256")
    actual_sha = sha256(checkpoint_path)
    if expected_sha != actual_sha:
        raise SystemExit(f"Frozen checkpoint SHA mismatch: expected {expected_sha}, got {actual_sha}")

    import torch

    torch.set_num_threads(args.torch_threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads((META / "idrid_localization_split.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((META / "idrid_localization_manifest.json").read_text(encoding="utf-8"))
    official_ids = {record["image_id"] for record in data_manifest["records"] if record["split"] == "test"}
    records = [record for record in data_manifest["records"] if record["split"] == "test"]
    if len(records) != 103 or official_ids != {item["image_id"] for item in split["official_test_records"]}:
        raise SystemExit(f"Official test inventory mismatch: {len(records)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SharedLandmarkHeatmapNet.build().to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    rows = []
    started = time.perf_counter()
    for record in records:
        dataset = LandmarkDataset([record], augment=False, seed=0)
        image_tensor, _heatmap, _coords, image_id, transform = dataset[0]
        with torch.inference_mode():
            outputs = model(image_tensor.unsqueeze(0).to(args.device))
            predicted_canvas, confidence, _probabilities = decode_heatmaps(outputs["heatmaps"])
        predicted = inverse_points(predicted_canvas[0].detach().cpu().numpy(), transform)
        width = int(record["image"]["width"])
        height = int(record["image"]["height"])
        predicted[:, 0] = np.clip(predicted[:, 0], 0, width - 1)
        predicted[:, 1] = np.clip(predicted[:, 1], 0, height - 1)
        ground_truth = np.asarray([[record["annotations"][name]["x"], record["annotations"][name]["y"]] for name in LANDMARKS], dtype=np.float32)
        rows.append({"image_id": image_id, "predicted": predicted.tolist(), "ground_truth": ground_truth.tolist(), "confidence": {name: float(confidence[0, index].detach().cpu()) for index, name in enumerate(LANDMARKS)}, "image_width": width, "image_height": height})

    metrics = localization_metrics(rows)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda row: sum(np.linalg.norm(np.asarray(row["predicted"])[i] - np.asarray(row["ground_truth"])[i]) / max(np.hypot(row["image_width"], row["image_height"]), 1.0) for i in range(2)))
    selected = {"best": ranked[:3], "worst": ranked[-3:]}
    visual_files = []
    for group, group_rows in selected.items():
        for row in group_rows:
            record = next(item for item in records if item["image_id"] == row["image_id"])
            image_path = ROOT / record["image_path"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image.thumbnail((1200, 800), Image.Resampling.LANCZOS)
                scale_x = image.width / row["image_width"]
                scale_y = image.height / row["image_height"]
                draw = ImageDraw.Draw(image)
                for index, name in enumerate(LANDMARKS):
                    gt = row["ground_truth"][index]
                    pred = row["predicted"][index]
                    gx, gy = gt[0] * scale_x, gt[1] * scale_y
                    px, py = pred[0] * scale_x, pred[1] * scale_y
                    draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), outline=(0, 255, 0), width=3)
                    draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=(255, 0, 0), width=3)
                    draw.line((gx, gy, px, py), fill=(255, 255, 0), width=2)
                    draw.text((px + 7, py + 7), f"{name} pred", fill=(255, 0, 0))
                    draw.text((gx + 7, gy - 15), f"{name} gt", fill=(0, 255, 0))
                output = VISUAL_DIR / f"{group}_{row['image_id']}.png"
                image.save(output)
                visual_files.append(str(output.relative_to(ROOT)).replace("\\", "/"))
    payload = {"schema_version": "idrid-localization-official-test-1", "evaluation_status": "COMPLETED_ONCE", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "model_version": model_manifest["model_version"], "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": actual_sha, "dataset": "IDRiD C. Localization official testing set", "official_test_count": len(records), "official_test_images_opened": len(records), "development_images_used_for_training": 412, "metrics": metrics, "per_image": rows, "best_examples": [row["image_id"] for row in selected["best"]], "worst_examples": [row["image_id"] for row in selected["worst"]], "visual_files": visual_files, "inference_seconds": round(time.perf_counter() - started, 3), "production_promoted": False, "clinical_validation_claim": False, "note": "Official test metrics are reported once after model freeze; they were not used for training, model selection, threshold tuning, or production promotion."}
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation_status": payload["evaluation_status"], "official_test_count": len(records), "metrics": metrics, "production_promoted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
