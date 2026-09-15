from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

CHECKPOINT = ROOT / "ml/weights/vessels/drive/checkpoint_best.pt"
MODEL_MANIFEST = CHECKPOINT.with_name("model_manifest.json")
REAL_IMAGE = ROOT / "ml/datasets/raw/drive/datasets/training/training/images/21_training.tif"
R2_CHECKPOINT = ROOT / "ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not CHECKPOINT.is_file() or not MODEL_MANIFEST.is_file(), reason="frozen DRIVE research artifact is not installed")
def test_drive_research_manifest_is_frozen_and_non_production():
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["production_promoted"] is False
    assert manifest["checkpoint_sha256"] == _sha256(CHECKPOINT)
    assert manifest["threshold"] == 0.3
    assert manifest["input_size"] == 512


@pytest.mark.skipif(not CHECKPOINT.is_file() or not MODEL_MANIFEST.is_file() or not REAL_IMAGE.is_file(), reason="frozen DRIVE research artifact or DRIVE training image is not installed")
def test_drive_research_adapter_returns_real_supporting_evidence():
    import numpy as np

    from app.ml.evidence.drive_research_model import DriveResearchVesselAdapter

    image = np.asarray(Image.open(REAL_IMAGE).convert("RGB"), dtype=np.uint8)
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    adapter = DriveResearchVesselAdapter(expected_sha256=manifest["checkpoint_sha256"])
    result = adapter.analyze(image, {})
    assert result.supported is True
    assert result.status == "model_inference"
    assert result.metadata["research_only"] is True
    assert result.metadata["production_promoted"] is False
    probability = adapter.predict_probability(image)
    assert probability.shape == image.shape[:2]
    assert float(probability.min()) >= 0.0
    assert float(probability.max()) <= 1.0
    assert result.mask_data_uri and result.overlay_data_uri and result.probability_map_data_uri


@pytest.mark.skipif(not R2_CHECKPOINT.is_file(), reason="protected R2-V2 artifact is not installed")
def test_protected_r2_v2_checkpoint_remains_unchanged():
    assert _sha256(R2_CHECKPOINT) == "ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a"


def test_official_drive_report_does_not_fabricate_accuracy():
    report_path = ROOT / "ml/datasets/metadata/drive/drive_official_test.json"
    if not report_path.is_file():
        pytest.skip("one-time official DRIVE audit has not been run")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["official_test_images_opened"] == 20
    assert report["manual_vessel_masks_available"] == 0
    assert report["accuracy_metrics_available"] is False
    assert report["metrics"] is None


def test_drive_research_default_is_opt_in():
    from app.core.config import Settings

    settings = Settings(_env_file=None)
    assert settings.drive_vessel_model_enabled is False
