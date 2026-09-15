"""Freeze the selected IDRiD lesion model using all 54 training images.

This command is development-only. It does not open the official 27-image
segmentation test images; official evaluation is a separate one-time command
that must be run only after this manifest is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.lesions.idrid import (  # noqa: E402
    LESION_CLASSES,
    build_idrid_model,
    image_tensor,
    load_training_sample,
    masked_focal_dice_loss,
    set_seed,
)

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
OUTPUT = ROOT / "ml" / "weights" / "lesions" / "idrid"
BASELINE = ROOT / "ml" / "weights" / "lesion_segmentation" / "fundus-lesions-unet-seresnext50-all-v1" / "model.safetensors"


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


class Dataset:
    def __init__(self, records, size, augment):
        self.records, self.size, self.augment = records, size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, masks, availability = load_training_sample(self.records[index], self.size, self.augment)
        return image_tensor(image), masks.astype(np.float32), availability.astype(np.float32)


def collate(batch):
    import torch

    images, masks, availability = zip(*batch)
    return torch.stack(list(images)), torch.from_numpy(np.stack(masks)), torch.from_numpy(np.stack(availability))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="idrid-unet-seresnext50-768-focaldice-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(args.torch_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but this PyTorch installation has no CUDA runtime")
    records = [record for record in json.loads((META / "idrid_lesion_manifest.json").read_text(encoding="utf-8")).get("records", []) if record["split"] == "train"]
    if len(records) != 54:
        raise SystemExit(f"Expected 54 training records, found {len(records)}")
    threshold_artifact = json.loads((META / "idrid_lesion_threshold_analysis.json").read_text(encoding="utf-8"))
    if threshold_artifact.get("candidate") != args.candidate:
        raise SystemExit("Threshold artifact does not belong to the selected candidate")
    selected_threshold = float(threshold_artifact["selected"]["threshold"])
    set_seed(args.seed)
    model, transfer = build_idrid_model(BASELINE)
    model.to(args.device)
    dataset = Dataset(records, args.size, augment=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate, generator=generator)
    positive = []
    for name in LESION_CLASSES:
        available = [record["masks"][name] for record in records if record["masks"][name]["status"] == "AVAILABLE"]
        positive.append(sum(int(item.get("active_pixels") or 0) for item in available) / max(1, 4288 * 2848) * args.size * args.size)
    total = len(records) * args.size * args.size
    pos_weight = torch.tensor([min(25.0, max(1.0, (total - value) / max(1.0, value))) for value in positive], dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, availability in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(args.device))
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = masked_focal_dice_loss(outputs, targets.to(args.device), availability.to(args.device), pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": scheduler.get_last_lr()[0]})
        print(f"final lesion epoch={epoch}/{args.epochs} loss={history[-1]['train_loss']:.6f}", flush=True)
    checkpoint = OUTPUT / "checkpoint_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": {"architecture": "U-Net", "encoder": "se_resnext50_32x4d", "classes": list(LESION_CLASSES), "input_size": args.size, "multi_label": True}, "training_config": {"candidate": args.candidate, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "seed": args.seed, "loss": "masked focal BCE + Dice with valid-pixel normalization", "threshold": selected_threshold, "preprocessing": "RGB -> resize -> ImageNet normalization", "augmentation": "synchronized horizontal flip, brightness and contrast", "training_images": 54, "official_test_images_opened": 0}, "transfer": transfer, "production_promoted": False}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {"schema_version": "idrid-lesion-model-manifest-1", "model_version": "idrid-lesion-unet-seresnext50-768-focaldice-20260913-v2", "model_type": "multi_label_semantic_segmentation", "architecture": "U-Net", "encoder": "se_resnext50_32x4d", "classes": list(LESION_CLASSES), "input_size": args.size, "preprocessing": "RGB resize to 768x768; ImageNet mean/std normalization; synchronized retinal-safe augmentation", "loss": "masked focal BCE + Dice with valid-pixel normalization", "threshold": selected_threshold, "post_processing": "threshold plus connected-component object analysis; no FOV mask available", "training_data": "IDRiD official segmentation training split, 54 images only", "validation_strategy": "5-fold development CV on the 54-image training split; duplicate-aware fixed seed", "cv_report": "ml/datasets/metadata/idrid/idrid_lesion_cv_report.json", "threshold_analysis": "ml/datasets/metadata/idrid/idrid_lesion_threshold_analysis.json", "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": digest, "official_test_images_opened": 0, "production_promoted": False, "clinical_validation_claim": False, "known_limitations": ["Small development set", "No official FOV masks", "Soft-exudate labels unavailable for 28 training images", "No clinical validation", "CPU-only PyTorch runtime in this environment"]}
    dump(OUTPUT / "model_manifest.json", manifest)
    dump(META / "idrid_lesion_final_report.json", {"status": "FROZEN_RESEARCH_MODEL", "model_manifest": "ml/weights/lesions/idrid/model_manifest.json", "checkpoint_sha256": digest, "development_only_freeze": True, "official_test_images_opened": 0, "production_promoted": False, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "note": "Official test evaluation is not included in this artifact until a separate one-time evaluation command is run after freeze."})
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
