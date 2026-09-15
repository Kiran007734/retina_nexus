"""One-time, guarded inference audit on the unlabeled DRIVE official test split.

The official DRIVE test images have no manual vessel masks in this checkout.
This script therefore records only provenance and engineering output summaries;
all accuracy metrics remain explicitly unavailable.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from ml.vessels.drive import normalize_input, read_gray, read_rgb, resolve_dataset_path  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "drive"
MODEL_PATH = ROOT / "ml" / "weights" / "vessels" / "drive" / "checkpoint_best.pt"
MANIFEST_PATH = MODEL_PATH.with_name("model_manifest.json")
REPORT_PATH = META / "drive_official_test.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    if REPORT_PATH.exists():
        raise SystemExit(f"Refusing to rerun official DRIVE evaluation: {REPORT_PATH} already exists")
    model_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if model_manifest.get("production_promoted") is not False:
        raise SystemExit("Refusing official evaluation unless the research model is explicitly production_promoted=false")
    actual_hash = sha256(MODEL_PATH)
    if actual_hash != model_manifest.get("checkpoint_sha256"):
        raise SystemExit(f"Frozen checkpoint SHA mismatch: manifest={model_manifest.get('checkpoint_sha256')} actual={actual_hash}")
    manifest = json.loads((META / "drive_manifest.json").read_text(encoding="utf-8"))
    records = sorted([record for record in manifest["records"] if record["split"] == "test"], key=lambda row: row["image_id"])
    if len(records) != 20 or any(record.get("vessel_mask_path") for record in records):
        raise SystemExit("Expected exactly 20 official test records with no manual vessel masks")
    import torch
    from app.ml.models.evidence import build_vessel_segmentation_model

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = build_vessel_segmentation_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    rows = []
    for record in records:
        image = read_rgb(resolve_dataset_path(record["image_path"]))
        fov = read_gray(resolve_dataset_path(record["fov_mask_path"])) >= 128
        resized = np.asarray(Image.fromarray(image, mode="RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8)
        values = normalize_input(resized, model_manifest["preprocessing"]["mode"])
        tensor = torch.from_numpy(values.transpose(2, 0, 1)).float().unsqueeze(0)
        started = time.perf_counter()
        with torch.inference_mode():
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        elapsed = time.perf_counter() - started
        prediction = probability >= float(model_manifest["threshold"])
        rows.append({"image_id": record["image_id"], "image_path": record["image_path"], "fov_mask_path": record["fov_mask_path"], "width": int(image.shape[1]), "height": int(image.shape[0]), "fov_pixels": int(fov.sum()), "predicted_vessel_pixels_512": int(prediction.sum()), "predicted_vessel_density_512": float(prediction.mean()), "predicted_vessel_density_within_fov_512": float(prediction[np.asarray(Image.fromarray(fov.astype(np.uint8) * 255, mode="L").resize((512, 512), Image.Resampling.NEAREST)) >= 128].mean()), "mean_probability_512": float(probability.mean()), "max_probability_512": float(probability.max()), "inference_seconds": float(elapsed), "ground_truth_available": False})
    densities = np.asarray([row["predicted_vessel_density_within_fov_512"] for row in rows], dtype=float)
    runtimes = np.asarray([row["inference_seconds"] for row in rows], dtype=float)
    density_order = np.argsort(densities)
    runtime_order = np.argsort(runtimes)
    report = {"status": "COMPLETED_ONCE", "dataset": "DRIVE", "dataset_version": manifest["dataset_version"], "evaluation_type": "official_test_inference_audit_without_manual_ground_truth", "model_version": model_manifest["model_version"], "checkpoint": model_manifest["checkpoint"], "checkpoint_sha256": actual_hash, "threshold": model_manifest["threshold"], "official_test_images_opened": len(records), "official_test_images_inferred": len(records), "manual_vessel_masks_available": 0, "accuracy_metrics_available": False, "metrics": None, "ground_truth_limitation": "The supplied DRIVE official test split contains FOV masks but no manual vessel segmentation masks. Dice, IoU, pixel accuracy, sensitivity, specificity, precision, and F1 are therefore unavailable and intentionally not calculated.", "per_image": rows, "summary": {"mean_predicted_vessel_density_within_fov_512": float(densities.mean()), "std_predicted_vessel_density_within_fov_512": float(densities.std()), "mean_inference_seconds": float(runtimes.mean()), "p95_inference_seconds": float(np.percentile(runtimes, 95))}, "representative_examples": {"lowest_predicted_density": rows[int(density_order[0])]["image_id"], "median_predicted_density": rows[int(density_order[len(rows) // 2])]["image_id"], "highest_predicted_density": rows[int(density_order[-1])]["image_id"], "fastest_inference": rows[int(runtime_order[0])]["image_id"], "slowest_inference": rows[int(runtime_order[-1])]["image_id"]}, "official_test_used_for_model_selection": False, "production_promoted": False, "clinical_validation_claim": False}
    dump(REPORT_PATH, report)
    print(json.dumps({"status": report["status"], "official_test_images_opened": report["official_test_images_opened"], "manual_vessel_masks_available": 0, "accuracy_metrics_available": False, "mean_inference_seconds": report["summary"]["mean_inference_seconds"], "checkpoint_sha256": actual_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
