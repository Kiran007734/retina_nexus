"""Bounded from-scratch DRIVE vessel experiments with five-fold CV.

The script only loads records from ``drive_split.json`` (the 20 declared
training images).  The official 20-image test split is never loaded here.
Each candidate is random-initialized; controlled low-learning-rate fine-tuning
is performed only from that candidate's fold-specific scratch checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.models.evidence import build_vessel_segmentation_model  # noqa: E402
from ml.vessels.drive import DriveSegmentationDataset, aggregate_metrics, set_seed, threshold_metrics  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "drive"
CV_ROOT = ROOT / "ml" / "weights" / "vessels" / "drive" / "cv"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="scratch-rgb-bce-dice-512-v1")
    parser.add_argument("--preprocessing", choices=["rgb", "green", "clahe_green"], default="rgb")
    parser.add_argument("--loss", choices=["bce_dice", "focal_dice"], default="bce_dice")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--fine-tune-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def device_for(torch, requested: str):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def segmentation_loss(logits, target, fov, loss_name: str, torch):
    import torch.nn.functional as F

    valid = fov.bool()
    if loss_name == "bce_dice":
        pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    else:
        probability = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        alpha = 0.75 * target + 0.25 * (1.0 - target)
        pixel_loss = alpha * torch.pow(1.0 - pt, 2.0) * bce
    pixel_loss = pixel_loss[valid].mean()
    probability = torch.sigmoid(logits)
    intersection = (probability * target * fov).sum(dim=(1, 2, 3))
    denominator = (probability * fov).sum(dim=(1, 2, 3)) + (target * fov).sum(dim=(1, 2, 3))
    dice_loss = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return pixel_loss + dice_loss, {"pixel_loss": float(pixel_loss.detach().cpu()), "dice_loss": float(dice_loss.detach().cpu())}


def evaluate(model, records, input_size, preprocessing, batch_size, device, seed=0):
    import torch
    from torch.utils.data import DataLoader

    dataset = DriveSegmentationDataset(records, input_size=input_size, preprocessing=preprocessing, augment=False, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    with torch.inference_mode():
        offset = 0
        for images, targets, fovs, ids in loader:
            probabilities = torch.sigmoid(model(images.to(device))).detach().cpu().numpy()[:, 0]
            targets_np = targets.numpy()[:, 0]
            fovs_np = fovs.numpy()[:, 0]
            for index, image_id in enumerate(ids):
                metrics = threshold_metrics(targets_np[index], probabilities[index], fovs_np[index], threshold=0.5)
                rows.append({"image_id": image_id, "probability": probabilities[index], "target": targets_np[index], "fov": fovs_np[index], "metrics": metrics})
            offset += len(ids)
    aggregate = aggregate_metrics([row["metrics"] for row in rows])
    return aggregate, rows


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"image_id": row["image_id"], **row["metrics"]} for row in rows]


def train_epochs(model, loader, optimizer, device, loss_name, torch):
    model.train()
    losses = []
    pixel_losses = []
    dice_losses = []
    for images, targets, fovs, _ids in loader:
        images, targets, fovs = images.to(device), targets.to(device), fovs.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, parts = segmentation_loss(logits, targets, fovs, loss_name, torch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        pixel_losses.append(parts["pixel_loss"])
        dice_losses.append(parts["dice_loss"])
    return {"loss": float(np.mean(losses)), "pixel_loss": float(np.mean(pixel_losses)), "dice_loss": float(np.mean(dice_losses))}


def save_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    np.savez_compressed(path, image_ids=np.asarray([row["image_id"] for row in rows]), probabilities=np.stack([row["probability"] for row in rows]), targets=np.stack([row["target"] for row in rows]), fovs=np.stack([row["fov"] for row in rows]))


def run_fold(args, candidate, fold, train_records, validation_records, torch, device):
    fold_dir = CV_ROOT / candidate / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model = build_vessel_segmentation_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    from torch.utils.data import DataLoader

    train_dataset = DriveSegmentationDataset(train_records, args.input_size, args.preprocessing, augment=True, seed=args.seed + fold)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed + fold))
    history = []
    best_score = -1.0
    best_state = None
    best_metrics = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_dataset.seed = args.seed + fold * 100 + epoch
        train_metrics = train_epochs(model, train_loader, optimizer, device, args.loss, torch)
        scheduler.step()
        validation_metrics, validation_rows = evaluate(model, validation_records, args.input_size, args.preprocessing, args.batch_size, device, seed=0)
        history.append({"phase": "scratch", "epoch": epoch, "train": train_metrics, "validation": validation_metrics, "learning_rate": scheduler.get_last_lr()[0]})
        score = float(validation_metrics["mean"].get("dice", 0.0))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = validation_metrics
            torch.save({"state_dict": best_state, "model_config": {"architecture": "lightweight_unet", "encoder": "two-stage convolutional encoder", "input_size": args.input_size, "preprocessing": args.preprocessing}, "training_config": vars(args), "initialization": "random_initialization", "production_promoted": False, "official_test_images_opened": 0}, fold_dir / "checkpoint_scratch_best.pt")
        print(f"candidate={candidate} fold={fold}/{args.folds} scratch_epoch={epoch}/{args.epochs} dice={validation_metrics['mean'].get('dice', 0):.6f} iou={validation_metrics['mean'].get('iou', 0):.6f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"No scratch checkpoint generated for fold {fold}")
    model.load_state_dict(best_state, strict=True)
    fine_optimizer = torch.optim.AdamW(model.parameters(), lr=args.fine_tune_learning_rate, weight_decay=args.weight_decay)
    fine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fine_optimizer, T_max=max(1, args.fine_tune_epochs))
    fine_best_score = best_score
    fine_best_state = best_state
    fine_best_metrics = best_metrics
    for epoch in range(1, args.fine_tune_epochs + 1):
        train_dataset.seed = args.seed + fold * 1000 + epoch
        train_metrics = train_epochs(model, train_loader, fine_optimizer, device, args.loss, torch)
        fine_scheduler.step()
        validation_metrics, validation_rows = evaluate(model, validation_records, args.input_size, args.preprocessing, args.batch_size, device, seed=0)
        history.append({"phase": "controlled_fine_tune", "epoch": epoch, "train": train_metrics, "validation": validation_metrics, "learning_rate": fine_scheduler.get_last_lr()[0]})
        score = float(validation_metrics["mean"].get("dice", 0.0))
        if score > fine_best_score:
            fine_best_score = score
            fine_best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            fine_best_metrics = validation_metrics
        print(f"candidate={candidate} fold={fold}/{args.folds} fine_tune_epoch={epoch}/{args.fine_tune_epochs} dice={validation_metrics['mean'].get('dice', 0):.6f} iou={validation_metrics['mean'].get('iou', 0):.6f}", flush=True)
    model.load_state_dict(fine_best_state, strict=True)
    final_metrics, final_rows = evaluate(model, validation_records, args.input_size, args.preprocessing, args.batch_size, device, seed=0)
    torch.save({"state_dict": fine_best_state, "model_config": {"architecture": "lightweight_unet", "encoder": "two-stage convolutional encoder", "input_size": args.input_size, "preprocessing": args.preprocessing}, "training_config": vars(args), "initialization": "random_initialization_then_controlled_low_lr_fine_tuning", "production_promoted": False, "official_test_images_opened": 0}, fold_dir / "checkpoint_best.pt")
    save_npz(fold_dir / "oof_predictions.npz", final_rows)
    result = {"fold": fold, "train_count": len(train_records), "validation_count": len(validation_records), "scratch_best_metrics": best_metrics, "fine_tuned_best_metrics": fine_best_metrics, "selected_metrics": final_metrics, "history": history, "validation_predictions": compact_rows(final_rows), "seconds": round(time.perf_counter() - started, 3), "official_test_images_opened": 0, "production_promoted": False}
    dump(fold_dir / "metrics.json", result)
    return result


def summarize(folds, key):
    metrics = [fold[key]["mean"] for fold in folds]
    keys = sorted(set().union(*(item.keys() for item in metrics)))
    return {metric: {"mean": float(np.mean([item[metric] for item in metrics])), "std": float(np.std([item[metric] for item in metrics])), "min": float(np.min([item[metric] for item in metrics])), "max": float(np.max([item[metric] for item in metrics])), "values": [float(item[metric]) for item in metrics]} for metric in keys}


def main() -> int:
    args = parse_args()
    import torch

    torch.set_num_threads(max(1, args.torch_threads))
    device = device_for(torch, args.device)
    set_seed(args.seed)
    manifest = json.loads((META / "drive_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((META / "drive_split.json").read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == "training"]
    fold_by_id = {record["image_id"]: int(record["fold"]) for record in split["records"]}
    if len(records) != 20 or any(record["image_id"] not in fold_by_id for record in records):
        raise SystemExit("DRIVE development manifest does not contain exactly 20 fold-assigned training records")
    candidate = args.candidate
    folds = []
    for fold in range(1, args.folds + 1):
        validation = [record for record in records if fold_by_id[record["image_id"]] == fold]
        training = [record for record in records if fold_by_id[record["image_id"]] != fold]
        if not validation or not training:
            raise SystemExit(f"Invalid fold {fold}: train={len(training)} validation={len(validation)}")
        cached = CV_ROOT / candidate / f"fold_{fold}" / "metrics.json"
        if cached.is_file() and (CV_ROOT / candidate / f"fold_{fold}" / "checkpoint_best.pt").is_file() and (CV_ROOT / candidate / f"fold_{fold}" / "oof_predictions.npz").is_file():
            folds.append(json.loads(cached.read_text(encoding="utf-8")))
            print(f"candidate={candidate} fold={fold}/{args.folds} reused_cached=true", flush=True)
        else:
            folds.append(run_fold(args, candidate, fold, training, validation, torch, device))
    report = {"schema_version": "drive-vessel-cv-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "candidate": candidate, "architecture": "lightweight U-Net", "initialization": "random_initialization", "input_size": args.input_size, "preprocessing": args.preprocessing, "loss": args.loss, "fold_count": args.folds, "development_images": len(records), "folds": folds, "scratch_summary": summarize(folds, "scratch_best_metrics"), "fine_tuned_summary": summarize(folds, "fine_tuned_best_metrics"), "selected_summary": summarize(folds, "selected_metrics"), "official_test_images_opened": 0, "production_promoted": False, "note": "All fold metrics are development-only engineering measurements within the supplied FOV masks."}
    dump(META / f"drive_cv_report_{candidate}.json", report)
    dump(META / "drive_cv_report.json", report)
    training_registry = {"schema_version": "drive-vessel-training-1", "generated_at_utc": report["generated_at_utc"], "candidate": candidate, "configuration": vars(args), "cv_report": "ml/datasets/metadata/drive/drive_cv_report.json", "official_test_images_opened": 0, "production_promoted": False}
    dump(META / "drive_scratch_training.json", training_registry)
    print(json.dumps({"candidate": candidate, "scratch_summary": report["scratch_summary"], "fine_tuned_summary": report["fine_tuned_summary"], "selected_summary": report["selected_summary"], "official_test_images_opened": 0, "production_promoted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
