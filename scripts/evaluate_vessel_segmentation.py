"""Evaluate a registered DRIVE vessel segmentation checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.evidence import build_vessel_segmentation_model  # noqa: E402
from ml.evidence.drive import DriveVesselDataset, find_drive_pairs  # noqa: E402
from ml.evidence.vessel import vessel_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a DRIVE vessel segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-dir", default="ml/datasets/raw/drive")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="ml/experiments/vessel_evaluation.json")
    args = parser.parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        input_size = args.input_size or int(checkpoint.get("model_config", {}).get("input_size", 512))
        raw_root = (ROOT / args.raw_dir).resolve() if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir).resolve()
        pairs = find_drive_pairs(raw_root)
        if not pairs:
            raise ValueError(f"No paired DRIVE images/masks found under {raw_root}; annotations were not fabricated.")
        device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
        if args.device == "cuda" and device.type != "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
        model = build_vessel_segmentation_model().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loader = DataLoader(DriveVesselDataset(pairs, input_size), batch_size=args.batch_size, shuffle=False)
        targets, probabilities = [], []
        with torch.inference_mode():
            for images, masks in loader:
                probability = torch.sigmoid(model(images.to(device))).cpu().numpy()
                targets.extend(masks.numpy())
                probabilities.extend(probability)
        report = {"dataset": "drive", "dataset_version": checkpoint.get("dataset_version", "unspecified"), "model_version": checkpoint.get("model_version", "unversioned"), "checkpoint": str(checkpoint_path), "metrics": vessel_metrics(targets, probabilities), "clinical_validation_claim": False, "note": "Measured mask metrics only; no clinical validation claim."}
        output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Vessel evaluation report written: {output}")
        return 0
    except Exception as exc:
        print(f"VESSEL EVALUATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
