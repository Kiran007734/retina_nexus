"""Analyze development-only DRIVE vessel predictions and create visual QA artifacts.

This script consumes only the five-fold out-of-fold predictions generated from
the 20 labeled DRIVE training images. It never opens the official test split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.vessels.drive import resolve_dataset_path, read_rgb, read_gray, threshold_metrics  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "drive"
VISUALS = ROOT / "ml" / "evaluation" / "drive" / "research_dev_visuals"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _image(path: Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    return Image.fromarray(read_rgb(path), mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _gray(values: np.ndarray, size: tuple[int, int] = (512, 512)) -> Image.Image:
    return Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8), mode="L").resize(size, Image.Resampling.NEAREST)


def _overlay(base: Image.Image, target: np.ndarray, prediction: np.ndarray, fov: np.ndarray) -> Image.Image:
    canvas = np.asarray(base.convert("RGBA"), dtype=np.uint8).copy()
    gt = target.astype(bool) & fov.astype(bool)
    pred = prediction.astype(bool) & fov.astype(bool)
    canvas[gt & pred] = (255, 220, 0, 255)      # overlap
    canvas[gt & ~pred] = (30, 110, 255, 255)    # false negative
    canvas[pred & ~gt] = (255, 40, 40, 255)     # false positive
    return Image.fromarray(canvas, mode="RGBA")


def main() -> int:
    threshold_report = json.loads((META / "drive_threshold_analysis.json").read_text(encoding="utf-8"))
    threshold = float(threshold_report["selected"]["threshold"])
    manifest = json.loads((META / "drive_manifest.json").read_text(encoding="utf-8"))
    records = {record["image_id"]: record for record in manifest["records"] if record["split"] == "training"}
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "ml" / "weights" / "vessels" / "drive" / "cv" / "scratch-green-focal-dice-512-v1").glob("fold_*/oof_predictions.npz")):
        with np.load(path, allow_pickle=False) as data:
            for index, image_id in enumerate(data["image_ids"].tolist()):
                target = data["targets"][index]
                probability = data["probabilities"][index]
                fov = data["fovs"][index]
                metrics = threshold_metrics(target, probability, fov, threshold)
                rows.append({"image_id": str(image_id), "fold_file": str(path.relative_to(ROOT)).replace("\\", "/"), "threshold": threshold, "metrics": metrics, "max_probability": float(np.max(probability)), "mean_probability_in_fov": float(np.mean(probability[fov.astype(bool)]))})
    if len(rows) != 20:
        raise SystemExit(f"Expected 20 development OOF rows, found {len(rows)}")
    rows.sort(key=lambda row: str(row["image_id"]))
    ranked = sorted(rows, key=lambda row: float(row["metrics"]["dice"]))
    selected = {"best": ranked[-1], "average": rows[len(rows) // 2], "difficult": ranked[len(ranked) // 2], "worst": ranked[0]}
    VISUALS.mkdir(parents=True, exist_ok=True)
    for label, row in selected.items():
        image_id = str(row["image_id"])
        record = records[image_id]
        image = read_rgb(resolve_dataset_path(record["image_path"]))
        target = (read_gray(resolve_dataset_path(record["vessel_mask_path"])) >= 128)
        fov = (read_gray(resolve_dataset_path(record["fov_mask_path"])) >= 128)
        probability = next(
            data["probabilities"][index]
            for path in sorted((ROOT / "ml" / "weights" / "vessels" / "drive" / "cv" / "scratch-green-focal-dice-512-v1").glob("fold_*/oof_predictions.npz"))
            for data in [np.load(path, allow_pickle=False)]
            for index, value in enumerate(data["image_ids"].tolist())
            if str(value) == image_id
        )
        base = Image.fromarray(image, mode="RGB").resize((512, 512), Image.Resampling.BILINEAR)
        target_small = np.asarray(Image.fromarray(target.astype(np.uint8) * 255, mode="L").resize((512, 512), Image.Resampling.NEAREST)) >= 128
        fov_small = np.asarray(Image.fromarray(fov.astype(np.uint8) * 255, mode="L").resize((512, 512), Image.Resampling.NEAREST)) >= 128
        prediction_small = probability >= threshold
        Image.fromarray(image, mode="RGB").save(VISUALS / f"{label}_{image_id}_original.png")
        _gray(target_small).save(VISUALS / f"{label}_{image_id}_ground_truth.png")
        _gray(prediction_small.astype(np.float32)).save(VISUALS / f"{label}_{image_id}_prediction.png")
        _overlay(base, target_small, prediction_small, fov_small).save(VISUALS / f"{label}_{image_id}_error_overlay.png")
    aggregate = {"dice_mean": float(np.mean([row["metrics"]["dice"] for row in rows])), "dice_std": float(np.std([row["metrics"]["dice"] for row in rows])), "iou_mean": float(np.mean([row["metrics"]["iou"] for row in rows])), "sensitivity_mean": float(np.mean([row["metrics"]["sensitivity"] for row in rows])), "specificity_mean": float(np.mean([row["metrics"]["specificity"] for row in rows]))}
    dump(META / "drive_error_analysis.json", {"status": "COMPLETED_DEVELOPMENT_ONLY", "dataset": "DRIVE", "development_images": 20, "official_test_images_opened": 0, "threshold": threshold, "aggregate": aggregate, "per_image": rows, "ranked_best_to_worst": [row["image_id"] for row in reversed(ranked)], "visual_examples": {label: {"image_id": row["image_id"], "dice": row["metrics"]["dice"], "files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in VISUALS.glob(f"{label}_{row['image_id']}_*.png")]} for label, row in selected.items()}, "interpretation": "Development-only engineering error analysis. It is not a clinical validation result."})
    print(json.dumps({"status": "COMPLETED_DEVELOPMENT_ONLY", "rows": len(rows), "threshold": threshold, "aggregate": aggregate, "official_test_images_opened": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
