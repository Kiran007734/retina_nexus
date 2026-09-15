"""Regression checks for the frozen IDRiD lesion research artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.lesions.idrid import LESION_CLASSES, masked_focal_dice_loss  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MODEL = ROOT / "ml" / "weights" / "lesions" / "idrid"
EXTERNAL = ROOT / "ml" / "weights" / "lesion_segmentation" / "fundus-lesions-unet-seresnext50-all-v1" / "model.safetensors"


def read(name: str) -> dict:
    return json.loads((META / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_idrid_manifest_preserves_train_test_and_missing_annotation_policy():
    manifest = read("idrid_lesion_manifest.json")
    train = [record for record in manifest["records"] if record["split"] == "train"]
    test = [record for record in manifest["records"] if record["split"] == "test"]
    assert len(train) == 54
    assert len(test) == 27
    assert sum(record["masks"]["haemorrhages"]["status"] == "UNAVAILABLE" for record in train) == 1
    assert sum(record["masks"]["soft_exudates"]["status"] == "UNAVAILABLE" for record in train) == 28
    assert read("idrid_lesion_split.json")["official_test_used"] is False


def test_frozen_model_checksum_and_official_evaluation_provenance():
    model_manifest = json.loads((MODEL / "model_manifest.json").read_text(encoding="utf-8"))
    assert model_manifest["checkpoint_sha256"] == sha256(MODEL / "checkpoint_best.pt")
    assert model_manifest["production_promoted"] is False
    official = read("idrid_lesion_official_test.json")
    assert official["evaluation_completed"] is True
    assert official["official_test_images_opened"] == 27
    assert official["threshold_tuning_on_test"] is False


def test_preserved_external_checkpoint_checksum_is_unchanged():
    assert sha256(EXTERNAL) == "a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2"


def test_masked_loss_uses_valid_pixel_normalization():
    import torch

    logits = torch.zeros((1, len(LESION_CLASSES), 64, 64), dtype=torch.float32, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, :, 16:24, 16:24] = 1.0
    availability = torch.ones((1, len(LESION_CLASSES)), dtype=torch.float32)
    loss = masked_focal_dice_loss(logits, targets, availability)
    assert np.isfinite(float(loss.detach()))
    assert float(loss.detach()) < 10.0
    loss.backward()
