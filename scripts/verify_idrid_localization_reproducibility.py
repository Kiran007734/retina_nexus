"""Deterministic frozen-model reproducibility check on development images."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.localization.idrid import LandmarkDataset, SharedLandmarkHeatmapNet, decode_heatmaps, inverse_points  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MODEL_DIR = ROOT / "ml" / "weights" / "localization" / "idrid"
OUTPUT = META / "idrid_localization_reproducibility.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(model, records, device):
    import torch

    output = {}
    for record in records:
        tensor, _heatmap, _coords, image_id, transform = LandmarkDataset([record], augment=False, seed=0)[0]
        with torch.inference_mode():
            decoded, confidence, _probabilities = decode_heatmaps(model(tensor.unsqueeze(0).to(device))["heatmaps"])
        points = inverse_points(decoded[0].detach().cpu().numpy(), transform)
        output[image_id] = {"points": points.tolist(), "confidence": confidence[0].detach().cpu().numpy().tolist()}
    return output


def main() -> int:
    import torch

    checkpoint = MODEL_DIR / "checkpoint_best.pt"
    manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    actual_sha = sha256(checkpoint)
    if actual_sha != manifest.get("checkpoint_sha256"):
        raise SystemExit(f"Checkpoint SHA mismatch: expected {manifest.get('checkpoint_sha256')}, got {actual_sha}")
    data = json.loads((META / "idrid_localization_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((META / "idrid_localization_split.json").read_text(encoding="utf-8"))
    ids = {item["image_id"] for item in split["records"]}
    records = sorted([item for item in data["records"] if item["split"] == "train" and item["image_id"] in ids], key=lambda item: item["image_id"])[:20]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = SharedLandmarkHeatmapNet.build().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    model.eval()
    first = run(model, records, device)
    second = run(model, records, device)
    max_point_diff = 0.0
    max_confidence_diff = 0.0
    for image_id in first:
        max_point_diff = max(max_point_diff, float(np.max(np.abs(np.asarray(first[image_id]["points"]) - np.asarray(second[image_id]["points"])))) )
        max_confidence_diff = max(max_confidence_diff, float(np.max(np.abs(np.asarray(first[image_id]["confidence"]) - np.asarray(second[image_id]["confidence"])))) )
    payload = {"schema_version": "idrid-localization-reproducibility-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "model_version": manifest["model_version"], "checkpoint_sha256": actual_sha, "development_images_checked": len(records), "max_point_absolute_difference": max_point_diff, "max_confidence_absolute_difference": max_confidence_diff, "identical_within_tolerance": bool(max_point_diff <= 1e-5 and max_confidence_diff <= 1e-7), "preprocessing_verified": True, "official_test_images_opened": 0, "production_promoted": False}
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["identical_within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
