"""Verify deterministic inference for the frozen DRIVE research checkpoint."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from ml.vessels.drive import normalize_input, read_rgb, resolve_dataset_path  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "drive"
MODEL_PATH = ROOT / "ml" / "weights" / "vessels" / "drive" / "checkpoint_best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def predict(model, image: np.ndarray, preprocessing: str, torch) -> np.ndarray:
    resized = np.asarray(Image.fromarray(image, mode="RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8)
    normalized = normalize_input(resized, preprocessing)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.inference_mode():
        return torch.sigmoid(model(tensor))[0, 0].cpu().numpy()


def main() -> int:
    import torch
    from app.ml.models.evidence import build_vessel_segmentation_model

    manifest = json.loads((META / "drive_manifest.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((ROOT / "ml" / "weights" / "vessels" / "drive" / "model_manifest.json").read_text(encoding="utf-8"))
    records = sorted([record for record in manifest["records"] if record["split"] == "training"], key=lambda row: row["image_id"])
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model_a = build_vessel_segmentation_model()
    model_b = build_vessel_segmentation_model()
    model_a.load_state_dict(checkpoint["state_dict"], strict=True)
    model_b.load_state_dict(checkpoint["state_dict"], strict=True)
    model_a.eval()
    model_b.eval()
    rows = []
    passed = True
    maximum_probability_delta = 0.0
    for record in records:
        image = read_rgb(resolve_dataset_path(record["image_path"]))
        first = predict(model_a, image, model_manifest["preprocessing"]["mode"], torch)
        second = predict(model_b, image, model_manifest["preprocessing"]["mode"], torch)
        delta = float(np.max(np.abs(first - second)))
        exact = bool(np.array_equal(first, second))
        maximum_probability_delta = max(maximum_probability_delta, delta)
        passed = passed and exact
        rows.append({"image_id": record["image_id"], "probability_shape": list(first.shape), "exact_probability_match": exact, "max_abs_probability_delta": delta, "binary_mask_exact_match": bool(np.array_equal(first >= float(model_manifest["threshold"]), second >= float(model_manifest["threshold"])))})
    checkpoint_hash = sha256(MODEL_PATH)
    if checkpoint_hash != model_manifest["checkpoint_sha256"]:
        raise SystemExit(f"Checkpoint checksum mismatch: manifest={model_manifest['checkpoint_sha256']} actual={checkpoint_hash}")
    result = {"status": "PASS" if passed and maximum_probability_delta == 0.0 else "FAIL", "model_version": model_manifest["model_version"], "checkpoint": model_manifest["checkpoint"], "checkpoint_sha256": checkpoint_hash, "preprocessing": model_manifest["preprocessing"], "image_count": len(records), "probabilities_identical": passed, "maximum_abs_probability_delta": maximum_probability_delta, "per_image": rows, "official_test_images_opened": 0, "production_promoted": model_manifest["production_promoted"]}
    dump(META / "drive_reproducibility.json", result)
    print(json.dumps({"status": result["status"], "image_count": len(records), "probabilities_identical": passed, "maximum_abs_probability_delta": maximum_probability_delta, "checkpoint_sha256": checkpoint_hash, "official_test_images_opened": 0}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
