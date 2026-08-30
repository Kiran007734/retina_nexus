"""Train a compact vessel segmentation baseline using paired DRIVE files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.evidence import build_vessel_segmentation_model  # noqa: E402
from ml.evidence.drive import DriveVesselDataset, find_drive_pairs  # noqa: E402
from ml.evidence.vessel import split_pairs, vessel_metrics  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Train RETINA-NEXUS DRIVE vessel segmentation")
    value.add_argument("--raw-dir", default="ml/datasets/raw/drive")
    value.add_argument("--dataset-version", default="unspecified")
    value.add_argument("--input-size", type=int, default=512)
    value.add_argument("--batch-size", type=int, default=2)
    value.add_argument("--learning-rate", type=float, default=1e-3)
    value.add_argument("--epochs", type=int, default=20)
    value.add_argument("--validation-ratio", type=float, default=0.2)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    value.add_argument("--model-version", default="drive-vessel-baseline-v1")
    value.add_argument("--output-dir")
    return value


def device_for(torch, requested: str):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu or install a CUDA-enabled PyTorch build.")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def run_epoch(model, loader, optimizer, device, torch):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[Any] = []
    probabilities: list[Any] = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
        probability = torch.sigmoid(logits)
        intersection = (probability * masks).sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + 1) / (probability.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1)).mean()
        loss = binary_loss + dice_loss
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(images)
        targets.extend(masks.detach().cpu().numpy())
        probabilities.extend(probability.detach().cpu().numpy())
    metrics = vessel_metrics(targets, probabilities)
    metrics["loss"] = total_loss / max(1, len(targets))
    return metrics


def main() -> int:
    args = parser().parse_args()
    if args.input_size < 32 or args.batch_size < 1 or args.epochs < 1:
        print("VESSEL TRAINING ERROR: input size, batch size, and epochs must be positive and reasonable.", file=sys.stderr)
        return 2
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        print("VESSEL TRAINING ERROR: install backend/requirements-ml.txt before training; no model was created.", file=sys.stderr)
        print(f"Original error: {exc}", file=sys.stderr)
        return 2
    try:
        raw_root = (ROOT / args.raw_dir).resolve() if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir).resolve()
        pairs = find_drive_pairs(raw_root)
        if len(pairs) < 2:
            raise ValueError(f"Fewer than two paired DRIVE images/masks were found under {raw_root}. Place authorized DRIVE files first; annotations were not fabricated.")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_pairs, validation_pairs = split_pairs(pairs, args.validation_ratio, args.seed)
        train_loader = DataLoader(DriveVesselDataset(train_pairs, args.input_size), batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(DriveVesselDataset(validation_pairs, args.input_size), batch_size=args.batch_size, shuffle=False)
        device = device_for(torch, args.device)
        model = build_vessel_segmentation_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        version = args.model_version
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / "ml" / "weights" / "evidence" / "vessels" / "drive" / version
        output_dir.mkdir(parents=True, exist_ok=True)
        best_score = -1.0
        best_epoch = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, device, torch)
            with torch.no_grad():
                validation_metrics = run_epoch(model, validation_loader, None, device, torch)
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
            if validation_metrics["dice"] > best_score:
                best_score = float(validation_metrics["dice"])
                best_epoch = epoch
                torch.save({"state_dict": model.state_dict(), "model_config": {"input_size": args.input_size, "architecture": "lightweight_unet"}, "dataset": "drive", "dataset_version": args.dataset_version, "model_version": version, "metrics": validation_metrics, "clinical_validation_claim": False}, output_dir / "checkpoint_best.pt")
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} validation_dice={validation_metrics['dice']:.4f}")
        checkpoint = output_dir / "checkpoint_best.pt"
        metadata = {"model_version": version, "dataset": "drive", "dataset_version": args.dataset_version, "checkpoint": checkpoint.name, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "best_epoch": best_epoch, "validation_metrics": history[best_epoch - 1]["validation"], "training_config": vars(args), "clinical_validation_claim": False, "note": "DRIVE split metrics are engineering results only."}
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "model_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"Vessel training complete: {checkpoint}")
        print("No clinical validation claim is made.")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"VESSEL TRAINING ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
