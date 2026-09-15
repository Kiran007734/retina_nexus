"""Bounded IDRiD localization development CV.

Only the 412-image training-side development pool is used. The one training
image that is an exact duplicate of an official test image is excluded by the
frozen split artifact. The official 103-image test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.localization.idrid import (  # noqa: E402
    HEATMAP_H,
    HEATMAP_W,
    LANDMARKS,
    LandmarkDataset,
    SharedLandmarkHeatmapNet,
    decode_heatmaps,
    inverse_points,
    localization_loss,
    localization_metrics,
    set_seed,
)

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
OUTPUT = ROOT / "ml" / "weights" / "localization" / "idrid" / "cv"


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def collate(batch):
    import torch

    images, heatmaps, coords, ids, transforms = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(heatmaps)), torch.stack(list(coords)), list(ids), list(transforms)


def loader(records, augment, batch_size, seed):
    import torch
    from torch.utils.data import DataLoader

    dataset = LandmarkDataset(records, augment=augment, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=augment, num_workers=0, collate_fn=collate, generator=generator), dataset


def evaluate(model, records, batch_size, device):
    import torch

    model.eval()
    data, _dataset = loader(records, augment=False, batch_size=batch_size, seed=0)
    rows = []
    with torch.inference_mode():
        for images, _heatmaps, _coords, ids, transforms in data:
            outputs = model(images.to(device))
            heatmap_points, confidence, _probabilities = decode_heatmaps(outputs["heatmaps"])
            coordinate_points = outputs["coordinates"].cpu().numpy() * np.asarray([512.0, 352.0], dtype=np.float32)
            heatmap_points = heatmap_points.cpu().numpy()
            confidence = confidence.cpu().numpy()
            for index, image_id in enumerate(ids):
                record = next(record for record in records if record["image_id"] == image_id)
                predicted = inverse_points(heatmap_points[index], transforms[index])
                coordinate_predicted = inverse_points(coordinate_points[index], transforms[index])
                ground_truth = np.asarray([[record["annotations"][name]["x"], record["annotations"][name]["y"]] for name in LANDMARKS], dtype=np.float32)
                rows.append({"image_id": image_id, "predicted": predicted.tolist(), "coordinate_head_predicted": coordinate_predicted.tolist(), "ground_truth": ground_truth.tolist(), "confidence": {name: float(confidence[index, landmark_index]) for landmark_index, name in enumerate(LANDMARKS)}, "image_width": record["image"]["width"], "image_height": record["image"]["height"]})
    return localization_metrics(rows), rows


def train_fold(candidate, fold, train_records, validation_records, args, torch):
    fold_dir = OUTPUT / candidate / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model = SharedLandmarkHeatmapNet.build().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    training_loader, training_dataset = loader(train_records, augment=True, batch_size=args.batch_size, seed=args.seed + fold)
    history = []
    best_score = float("inf")
    best_state = None
    best_metrics = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        training_dataset.seed = args.seed + fold * 1000 + epoch * 31
        model.train()
        losses = []
        heatmap_losses = []
        coordinate_losses = []
        for images, heatmaps, coords, _ids, _transforms in training_loader:
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
        metrics, rows = evaluate(model, validation_records, args.batch_size, args.device)
        score = float(metrics["optic_disc"]["mean_normalized_error"] or 1.0) + float(metrics["fovea"]["mean_normalized_error"] or 1.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "heatmap_mse": float(np.mean(heatmap_losses)), "coordinate_smooth_l1": float(np.mean(coordinate_losses)), "learning_rate": scheduler.get_last_lr()[0], "validation": metrics})
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = metrics
            dump(fold_dir / "validation_predictions.json", rows)
        print(f"localization candidate={candidate} fold={fold}/{args.folds} epoch={epoch}/{args.epochs} loss={history[-1]['train_loss']:.6f} od_norm={metrics['optic_disc']['mean_normalized_error']:.6f} fovea_norm={metrics['fovea']['mean_normalized_error']:.6f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"No best state generated for localization fold {fold}")
    checkpoint = fold_dir / "checkpoint_best.pt"
    torch.save({"state_dict": best_state, "model_config": {"architecture": "shared compact heatmap network", "landmarks": list(LANDMARKS), "input_width": 512, "input_height": 352, "heatmap_width": HEATMAP_W, "heatmap_height": HEATMAP_H, "heatmap_sigma": 2.5, "prediction_method": "heatmap spatial soft-argmax"}, "training_config": {"candidate": candidate, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "coordinate_weight": args.coordinate_weight, "seed": args.seed + fold, "loss": "heatmap MSE + auxiliary Smooth L1 coordinate loss", "preprocessing": "aspect-ratio-preserving RGB letterbox to 512x352; ImageNet normalization", "augmentation": "horizontal flip, +/-4 degree geometry-aware rotation, brightness and contrast"}, "production_promoted": False, "official_test_images_opened": 0}, checkpoint)
    model.load_state_dict(best_state)
    final_metrics, _rows = evaluate(model, validation_records, args.batch_size, args.device)
    result = {"fold": fold, "train_count": len(train_records), "validation_count": len(validation_records), "metrics": final_metrics, "history": history, "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "seconds": round(time.perf_counter() - started, 3), "official_test_images_opened": 0, "production_promoted": False}
    dump(fold_dir / "metrics.json", result)
    return result


def summarize(folds):
    output = {}
    for landmark in LANDMARKS:
        output[landmark] = {}
        for metric in ("mean_euclidean_error_px", "p90_euclidean_error_px", "mean_normalized_error", "within_1pct_diagonal", "within_2pct_diagonal", "within_5pct_diagonal", "within_10pct_diagonal"):
            values = np.asarray([fold["metrics"][landmark][metric] for fold in folds], dtype=np.float64)
            output[landmark][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values)), "values": values.tolist()}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="idrid-shared-heatmap-coord-512x352-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coordinate-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.torch_threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    set_seed(args.seed)
    manifest = json.loads((META / "idrid_localization_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((META / "idrid_localization_split.json").read_text(encoding="utf-8"))
    fold_by_id = {item["image_id"]: int(item["fold"]) for item in split["records"]}
    records = [record for record in manifest["records"] if record["split"] == "train" and record["image_id"] in fold_by_id]
    if len(records) != split["development_count"]:
        raise SystemExit(f"Development record mismatch: expected {split['development_count']}, found {len(records)}")
    results = []
    for fold in range(1, args.folds + 1):
        validation = [record for record in records if fold_by_id[record["image_id"]] == fold]
        training = [record for record in records if fold_by_id[record["image_id"]] != fold]
        fold_dir = OUTPUT / args.candidate / f"fold_{fold}"
        cached = fold_dir / "metrics.json"
        if cached.is_file() and (fold_dir / "checkpoint_best.pt").is_file() and (fold_dir / "validation_predictions.json").is_file():
            results.append(json.loads(cached.read_text(encoding="utf-8")))
            print(f"localization candidate={args.candidate} fold={fold}/{args.folds} reused_cached=true", flush=True)
        else:
            results.append(train_fold(args.candidate, fold, training, validation, args, torch))
    report = {"schema_version": "idrid-localization-cv-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "candidate": args.candidate, "architecture": "shared compact heatmap network with auxiliary coordinate regression", "input_resolution": [512, 352], "fold_count": args.folds, "development_images": len(records), "excluded_cross_split_duplicates": split["excluded_cross_split_duplicates"], "folds": results, "summary": summarize(results), "official_test_images_opened": 0, "production_promoted": False}
    dump(META / "idrid_localization_cv_report.json", report)
    comparison = {"baseline": {"artifact": "ml/datasets/metadata/idrid/idrid_localization_baseline.json", "note": "Existing classical evidence baseline; not a trained model."}, "idrid_candidate": report, "selection_priority": ["optic_disc_error", "fovea_error", "p90_error", "success_rates", "cross_fold_stability", "robustness", "inference_time"], "official_test_images_opened": 0, "production_promoted": False}
    dump(META / "idrid_localization_experiment_comparison.json", comparison)
    print(json.dumps({"candidate": args.candidate, "development_images": len(records), "summary": report["summary"], "official_test_images_opened": 0, "production_promoted": False}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
