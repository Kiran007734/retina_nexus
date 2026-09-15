"""Bounded IDRiD lesion segmentation research cycle.

Only the 54 official IDRiD training images are used for optimization and
development selection. The official 27-image test split is never loaded by
this command. The published lesion checkpoint is used as a baseline and as
transfer initialization; it is never overwritten.
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

from ml.lesions.idrid import (  # noqa: E402
    LESION_CLASSES,
    ROOT as PROJECT_ROOT,
    build_idrid_model,
    image_tensor,
    load_training_sample,
    masked_focal_dice_loss,
    segmentation_metrics,
    set_seed,
)

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MANIFEST = META / "idrid_lesion_manifest.json"
SPLIT = META / "idrid_lesion_split.json"
BASELINE = ROOT / "ml" / "weights" / "lesion_segmentation" / "fundus-lesions-unet-seresnext50-all-v1" / "model.safetensors"
OUTPUT_ROOT = ROOT / "ml" / "weights" / "lesions" / "idrid"
CV_ROOT = OUTPUT_ROOT / "cv"


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    folds = {item["image_id"]: int(item["fold"]) for item in split["records"]}
    records = [record for record in manifest["records"] if record["split"] == "train"]
    if len(records) != 54 or set(folds) != {record["image_id"] for record in records}:
        raise RuntimeError("Expected exactly 54 IDRiD segmentation training records and a complete development split")
    return records, folds


class LesionDataset:
    def __init__(self, records: list[dict[str, Any]], size: int, augment: bool):
        self.records = records
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        image, masks, availability = load_training_sample(self.records[index], self.size, self.augment)
        return image_tensor(image), masks.astype(np.float32), availability.astype(np.float32), self.records[index]["image_id"]


def collate(batch):
    import torch

    images, masks, availability, ids = zip(*batch)
    return torch.stack(list(images)), torch.from_numpy(np.stack(masks)), torch.from_numpy(np.stack(availability)), list(ids)


def loader(dataset: LesionDataset, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate, generator=generator)


def class_pos_weights(records: list[dict[str, Any]], size: int) -> np.ndarray:
    weights = []
    for name in LESION_CLASSES:
        available = [record["masks"][name] for record in records if record["masks"][name]["status"] == "AVAILABLE"]
        positive = sum(int(item.get("active_pixels") or 0) for item in available) / max(1, 4288 * 2848) * size * size
        total = len(available) * size * size
        negative = max(1.0, total - positive)
        weights.append(float(min(25.0, max(1.0, negative / max(1.0, positive)))))
    return np.asarray(weights, dtype=np.float32)


def evaluate(model, records: list[dict[str, Any]], size: int, batch_size: int, device: str) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    import torch

    model.eval()
    data = loader(LesionDataset(records, size, augment=False), batch_size, shuffle=False, seed=0)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    per_image: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, target, available, ids in data:
            outputs = model(images.to(device))
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            probs = torch.sigmoid(outputs).cpu().numpy()
            target_np = target.numpy()
            available_np = available.numpy()
            probabilities.append(probs)
            targets.append(target_np)
            availability.append(available_np)
            for row, image_id in enumerate(ids):
                image_metrics = segmentation_metrics(probs[row : row + 1], target_np[row : row + 1], available_np[row : row + 1])
                per_image.append({"image_id": image_id, "metrics": image_metrics})
    arrays = {"probabilities": np.concatenate(probabilities), "targets": np.concatenate(targets), "availability": np.concatenate(availability)}
    return segmentation_metrics(arrays["probabilities"], arrays["targets"], arrays["availability"]), arrays, per_image


def train_fold(candidate: str, train_records: list[dict[str, Any]], validation_records: list[dict[str, Any]], args, torch) -> dict[str, Any]:
    fold_dir = CV_ROOT / candidate / f"fold_{args.fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model, transfer = build_idrid_model(BASELINE)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    pos_weight = torch.from_numpy(class_pos_weights(train_records, args.size)).to(args.device)
    train_loader = loader(LesionDataset(train_records, args.size, augment=True), args.batch_size, shuffle=True, seed=args.seed + args.fold_number)
    history = []
    best_score = -1.0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, availability, _ids in train_loader:
            images = images.to(args.device)
            targets = targets.to(args.device)
            availability = availability.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = masked_focal_dice_loss(outputs, targets, availability, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_metrics, arrays, per_image = evaluate(model, validation_records, args.size, args.batch_size, args.device)
        score = float(validation_metrics.get("macro_dice") or 0.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation_metrics, "learning_rate": scheduler.get_last_lr()[0]})
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            np.savez_compressed(fold_dir / "validation_outputs.npz", **arrays)
            dump(fold_dir / "validation_per_image.json", per_image)
    if best_state is None:
        raise RuntimeError(f"No checkpoint was produced for fold {args.fold_number}")
    checkpoint = fold_dir / "checkpoint_best.pt"
    torch.save({"state_dict": best_state, "model_config": {"architecture": "U-Net", "encoder": "se_resnext50_32x4d", "classes": list(LESION_CLASSES), "input_size": args.size, "multi_label": True}, "training_config": {"candidate": candidate, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "seed": args.seed + args.fold_number, "loss": "masked focal BCE + Dice", "pos_weight": pos_weight.detach().cpu().tolist(), "preprocessing": "RGB -> resize -> ImageNet normalization; no FOV mask available", "augmentation": "synchronized horizontal flip, brightness and contrast"}, "transfer": transfer, "production_promoted": False, "official_test_images_opened": 0}, checkpoint)
    model.load_state_dict(best_state)
    final_metrics, _arrays, _per_image = evaluate(model, validation_records, args.size, args.batch_size, args.device)
    result = {"fold": args.fold_number, "train_count": len(train_records), "validation_count": len(validation_records), "metrics": final_metrics, "history": history, "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "seconds": round(time.perf_counter() - started, 3), "official_test_images_opened": 0, "production_promoted": False}
    dump(fold_dir / "metrics.json", result)
    print(f"lesion candidate={candidate} fold={args.fold_number}/{args.folds} macro_dice={final_metrics.get('macro_dice')} macro_iou={final_metrics.get('macro_iou')}", flush=True)
    return result


def summarize(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["macro_dice", "macro_iou"]
    summary = {}
    for field in fields:
        values = np.asarray([float(item["metrics"].get(field) or 0.0) for item in fold_results], dtype=float)
        summary[field] = {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max()), "values": values.tolist()}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.torch_threads)
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not BASELINE.is_file():
        raise RuntimeError(f"Published baseline checkpoint is missing: {BASELINE}")
    records, fold_by_id = load_records()
    set_seed(args.seed)
    candidate = f"idrid-unet-seresnext50-{args.size}-focaldice-v2"
    fold_results = []
    for fold in range(1, args.folds + 1):
        args.fold_number = fold
        validation = [record for record in records if fold_by_id[record["image_id"]] == fold]
        training = [record for record in records if fold_by_id[record["image_id"]] != fold]
        cached_metrics = CV_ROOT / candidate / f"fold_{fold}" / "metrics.json"
        cached_checkpoint = CV_ROOT / candidate / f"fold_{fold}" / "checkpoint_best.pt"
        cached_outputs = CV_ROOT / candidate / f"fold_{fold}" / "validation_outputs.npz"
        if cached_metrics.exists() and cached_checkpoint.exists() and cached_outputs.exists():
            fold_results.append(json.loads(cached_metrics.read_text(encoding="utf-8")))
            print(f"lesion candidate={candidate} fold={fold}/{args.folds} reused_cached=true", flush=True)
            continue
        fold_results.append(train_fold(candidate, training, validation, args, torch))
    report = {"schema_version": "idrid-lesion-cv-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "candidate": candidate, "architecture": "U-Net with SE-ResNeXt-50 32x4d encoder", "target_classes": list(LESION_CLASSES), "input_size": args.size, "fold_count": args.folds, "folds": fold_results, "summary": summarize(fold_results), "training_images": len(records), "official_test_images_opened": 0, "production_promoted": False}
    dump(META / "idrid_lesion_cv_report.json", report)
    dump(META / "idrid_lesion_experiment_comparison.json", {
        "baseline": {
            "model_version": "fundus-lesions-unet-seresnext50-all-v1",
            "checkpoint": str(BASELINE.relative_to(ROOT)).replace("\\", "/"),
            "note": "Baseline evaluation is produced by scripts/evaluate_external_idrid_lesion_baseline.py; it is not overwritten."
        },
        "invalid_experiment": {
            "candidate": "idrid-unet-seresnext50-768-focaldice-v1",
            "status": "INVALID_NOT_SELECTED",
            "reason": "The masked BCE term was normalized by valid class channels rather than valid pixels, producing million-scale losses on full-resolution masks. Its metrics must not be used for model selection."
        },
        "idrid_candidate": report,
        "selection_metric": "macro_dice_then_macro_iou_then_class_recall_then_false_positive_burden",
        "official_test_images_opened": 0,
        "production_promoted": False,
    })
    print(json.dumps({"candidate": candidate, "summary": report["summary"], "official_test_images_opened": 0, "production_promoted": False}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
