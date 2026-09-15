"""Train the selected IDRiD localization candidate on the leak-safe dev pool.

The official 103-image localization test package is never loaded here.  Model
selection is performed by the preceding development-only CV report; this
script only fits the selected configuration on all 412 legitimate development
images and writes an immutable research artifact with production promotion
disabled.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.localization.idrid import LANDMARKS, SharedLandmarkHeatmapNet, localization_loss, set_seed  # noqa: E402
from scripts.run_idrid_localization_research import evaluate, loader  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
FINAL_DIR = ROOT / "ml" / "weights" / "localization" / "idrid"


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default="idrid-localization-shared-heatmap-512x352-20260913-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coordinate-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)

    split = json.loads((META / "idrid_localization_split.json").read_text(encoding="utf-8"))
    manifest = json.loads((META / "idrid_localization_manifest.json").read_text(encoding="utf-8"))
    development_ids = {item["image_id"] for item in split["records"]}
    records = [item for item in manifest["records"] if item["split"] == "train" and item["image_id"] in development_ids]
    if len(records) != 412 or len(records) != int(split["development_count"]):
        raise SystemExit(f"Unexpected development pool: {len(records)}")

    import torch.utils.data

    train_loader, train_dataset = loader(records, augment=True, batch_size=args.batch_size, seed=args.seed)
    model = SharedLandmarkHeatmapNet.build().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    history = []
    best_loss = float("inf")
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_dataset.seed = args.seed + epoch * 31
        model.train()
        losses = []
        heatmap_losses = []
        coordinate_losses = []
        for images, heatmaps, coords, _ids, _transforms in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(args.device))
            loss, parts = localization_loss(outputs, heatmaps.to(args.device), coords.to(args.device), coordinate_weight=args.coordinate_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            heatmap_losses.append(parts["heatmap_mse"])
            coordinate_losses.append(parts["coordinate_smooth_l1"])
        scheduler.step()
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": epoch_loss, "heatmap_mse": float(np.mean(heatmap_losses)), "coordinate_smooth_l1": float(np.mean(coordinate_losses)), "learning_rate": scheduler.get_last_lr()[0]})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"idrid localization final epoch={epoch}/{args.epochs} loss={epoch_loss:.6f}", flush=True)

    if best_state is None:
        raise SystemExit("No final localization checkpoint was produced")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = FINAL_DIR / "checkpoint_best.pt"
    torch.save({
        "state_dict": best_state,
        "model_config": {"architecture": "shared compact heatmap network with auxiliary coordinate regression", "landmarks": list(LANDMARKS), "input_width": 512, "input_height": 352, "heatmap_width": 128, "heatmap_height": 88, "heatmap_sigma": 2.5, "prediction_method": "heatmap spatial soft-argmax"},
        "training_config": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "coordinate_weight": args.coordinate_weight, "seed": args.seed, "loss": "heatmap MSE + auxiliary Smooth L1 coordinate loss", "preprocessing": "aspect-ratio-preserving RGB letterbox to 512x352; ImageNet normalization", "augmentation": "horizontal flip, +/-4 degree geometry-aware rotation, brightness and contrast"},
        "production_promoted": False,
        "official_test_images_opened": 0,
    }, checkpoint_path)
    digest = sha256(checkpoint_path)
    model.load_state_dict(best_state, strict=True)
    train_metrics, _rows = evaluate(model, records, args.batch_size, args.device)
    cv = json.loads((META / "idrid_localization_cv_report.json").read_text(encoding="utf-8"))
    manifest_payload = {
        "schema_version": "idrid-localization-model-manifest-1",
        "model_version": args.model_version,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest,
        "architecture": "shared compact heatmap network with auxiliary coordinate regression",
        "landmarks": list(LANDMARKS),
        "input_resolution": [512, 352],
        "heatmap_resolution": [128, 88],
        "preprocessing": {"color": "RGB", "resize": "aspect-ratio-preserving letterbox", "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "coordinate_convention": "original image pixels, x right and y down"},
        "prediction": {"heatmap_decode": "temperature-scaled spatial soft-argmax", "temperature": 5.0, "confidence": "peak sigmoid multiplied by one minus normalized spatial entropy"},
        "training": {"development_images": len(records), "raw_training_images": int(split["raw_training_count"]), "excluded_cross_split_duplicates": split["excluded_cross_split_duplicates"], "official_test_images_opened": 0, "patient_ids_available": False, "fold_count": cv["fold_count"], "loss": "heatmap MSE + auxiliary Smooth L1 coordinate loss", "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "history": history, "seconds": round(time.perf_counter() - started, 3)},
        "validation": {"cv_report": "ml/datasets/metadata/idrid/idrid_localization_cv_report.json", "cv_summary": cv["summary"], "development_fit_metrics_not_for_model_selection": train_metrics},
        "production_promoted": False,
        "clinical_validation_claim": False,
        "known_limitations": ["IDRiD localization has no patient identifiers; duplicate-aware image-level splitting was used.", "IDRiD_118 was excluded because it is an exact duplicate of official test IDRiD_064.", "This artifact is research-only and does not establish clinical validity or regulatory approval."],
    }
    dump(FINAL_DIR / "model_manifest.json", manifest_payload)
    dump(META / "idrid_localization_final_training.json", {"model_version": args.model_version, "checkpoint": manifest_payload["checkpoint"], "checkpoint_sha256": digest, "development_images": len(records), "official_test_images_opened": 0, "production_promoted": False, "train_fit_metrics": train_metrics, "history": history})
    print(json.dumps({"model_version": args.model_version, "checkpoint": manifest_payload["checkpoint"], "checkpoint_sha256": digest, "development_images": len(records), "official_test_images_opened": 0, "production_promoted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
