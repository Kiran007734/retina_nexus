"""Fit the selected DRIVE vessel configuration on all 20 training images.

This is a final research fit after development-only CV selection.  It never
loads the official test images or masks and writes production_promoted=false.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.models.evidence import build_vessel_segmentation_model  # noqa: E402
from ml.vessels.drive import DriveSegmentationDataset, aggregate_metrics, threshold_metrics, set_seed  # noqa: E402
from scripts.run_drive_vessel_research import segmentation_loss, train_epochs  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "drive"
MODEL_DIR = ROOT / "ml" / "weights" / "vessels" / "drive"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def evaluate_fit(model, records, input_size, preprocessing, threshold, device):
    import torch
    from torch.utils.data import DataLoader

    dataset = DriveSegmentationDataset(records, input_size, preprocessing, augment=False, seed=0)
    rows = []
    model.eval()
    with torch.inference_mode():
        for images, targets, fovs, ids in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0):
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            for index, image_id in enumerate(ids):
                rows.append({"image_id": image_id, **threshold_metrics(targets[index, 0].numpy(), probabilities[index], fovs[index, 0].numpy(), threshold)})
    return aggregate_metrics(rows), rows


def main() -> int:
    import torch

    candidate = "scratch-green-focal-dice-512-v1"
    preprocessing = "green"
    loss_name = "focal_dice"
    input_size = 512
    threshold_report = json.loads((META / "drive_threshold_analysis.json").read_text(encoding="utf-8"))
    threshold = float(threshold_report["selected"]["threshold"])
    manifest = json.loads((META / "drive_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((META / "drive_split.json").read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == "training"]
    if len(records) != 20 or len(split["official_test_records"]) != 20:
        raise SystemExit("Expected 20 training records and 20 reserved official test records")
    device = torch.device("cpu")
    torch.set_num_threads(8)
    set_seed(20260913)
    dataset = DriveSegmentationDataset(records, input_size, preprocessing, augment=True, seed=20260913)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(20260913))
    model = build_vessel_segmentation_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    history = []
    best_loss = float("inf")
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, 7):
        dataset.seed = 20260913 + epoch
        train_metrics = train_epochs(model, loader, optimizer, device, loss_name, torch)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "learning_rate": scheduler.get_last_lr()[0]})
        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"final DRIVE vessel epoch={epoch}/6 loss={train_metrics['loss']:.6f}", flush=True)
    if best_state is None:
        raise SystemExit("No final vessel state was produced")
    model.load_state_dict(best_state, strict=True)
    fit_metrics, fit_rows = evaluate_fit(model, records, input_size, preprocessing, threshold, device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "checkpoint_best.pt"
    torch.save({"state_dict": best_state, "model_config": {"architecture": "lightweight_unet", "encoder": "two-stage convolutional encoder", "input_size": input_size, "preprocessing": preprocessing}, "training_config": {"candidate": candidate, "loss": loss_name, "optimizer": "AdamW", "learning_rate": 5e-4, "weight_decay": 1e-4, "epochs": 6, "batch_size": 2, "seed": 20260913, "augmentation": "synchronized horizontal flip, small rotation, brightness, contrast, mild blur", "initialization": "random_initialization"}, "threshold": threshold, "production_promoted": False, "official_test_images_opened": 0}, checkpoint_path)
    digest = sha256(checkpoint_path)
    cv = json.loads((META / "drive_cv_report.json").read_text(encoding="utf-8"))
    audit = json.loads((META / "drive_data_audit.json").read_text(encoding="utf-8"))
    model_manifest = {"schema_version": "drive-vessel-model-manifest-1", "model_version": "drive-vessel-scratch-green-focal-dice-512-20260913-v1", "checkpoint": "ml/weights/vessels/drive/checkpoint_best.pt", "checkpoint_sha256": digest, "architecture": "lightweight U-Net", "encoder": "two-stage convolutional encoder", "initialization": "random_initialization", "input_size": input_size, "preprocessing": {"mode": preprocessing, "color": "green channel replicated to three input channels", "normalization": "float32 [0,1]"}, "augmentation": "synchronized horizontal flip, +/-7 degree rotation, brightness/contrast, mild blur", "loss": loss_name, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "learning_rate": 5e-4, "batch_size": 2, "seed": 20260913, "threshold": threshold, "post_processing": "probability threshold only; FOV is derived at runtime and supplied explicitly during DRIVE evaluation", "cv_protocol": {"folds": 5, "development_images": 20, "official_test_images_opened": 0, "cv_report": "ml/datasets/metadata/drive/drive_cv_report.json", "threshold_report": "ml/datasets/metadata/drive/drive_threshold_analysis.json"}, "cv_metrics": cv["selected_summary"], "final_fit_metrics_not_for_model_selection": fit_metrics, "dataset_version": audit["dataset_version"], "training_seconds": round(time.perf_counter() - started, 3), "production_promoted": False, "clinical_validation_claim": False, "known_limitations": ["The learned candidate is weaker than the protected R2-V2 reference on development CV.", "The supplied official DRIVE test split has no manual vessel masks, so official test accuracy cannot be computed honestly.", "Vessel density and reliability are engineering evidence outputs, not clinical diagnoses."]}
    dump(MODEL_DIR / "model_manifest.json", model_manifest)
    dump(META / "drive_final_training.json", {"model_version": model_manifest["model_version"], "checkpoint": model_manifest["checkpoint"], "checkpoint_sha256": digest, "fit_metrics": fit_metrics, "history": history, "official_test_images_opened": 0, "production_promoted": False})
    print(json.dumps({"model_version": model_manifest["model_version"], "checkpoint": model_manifest["checkpoint"], "checkpoint_sha256": digest, "threshold": threshold, "fit_metrics": fit_metrics["mean"], "official_test_images_opened": 0, "production_promoted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
