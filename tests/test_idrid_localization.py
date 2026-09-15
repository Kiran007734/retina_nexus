from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"

from ml.localization.idrid import HEATMAP_H, HEATMAP_W, LANDMARKS, SharedLandmarkHeatmapNet, gaussian_heatmaps, inverse_points, transform_points


def read(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_localization_inventory_is_duplicate_safe():
    audit = read("idrid_localization_data_audit.json")
    split = read("idrid_localization_split.json")
    assert audit["training_images"] == 413
    assert audit["development_images_after_leakage_exclusion"] == 412
    assert audit["official_test_images"] == 103
    assert split["development_count"] == 412
    assert len(split["official_test_records"]) == 103
    assert [item["image_id"] for item in split["excluded_cross_split_duplicates"]] == ["IDRiD_118"]
    assert audit["official_test_used_for_training_or_selection"] is False


def test_coordinate_transform_round_trip():
    points = np.asarray([[100.0, 200.0], [3000.0, 1800.0]], dtype=np.float32)
    transformed, transform = transform_points(points, 4288, 2848)
    restored = inverse_points(transformed, transform)
    assert np.allclose(restored, points, atol=1e-4)


def test_heatmap_and_model_output_shapes():
    points = np.asarray([[100.0, 120.0], [300.0, 200.0]], dtype=np.float32)
    heatmaps = gaussian_heatmaps(points)
    assert heatmaps.shape == (len(LANDMARKS), HEATMAP_H, HEATMAP_W)
    import torch

    model = SharedLandmarkHeatmapNet.build()
    output = model(torch.zeros((1, 3, 352, 512), dtype=torch.float32))
    assert tuple(output["heatmaps"].shape) == (1, 2, HEATMAP_H, HEATMAP_W)
    assert tuple(output["coordinates"].shape) == (1, 2, 2)


def test_frozen_manifest_integrity_when_present():
    manifest_path = ROOT / "ml/weights/localization/idrid/model_manifest.json"
    checkpoint_path = ROOT / "ml/weights/localization/idrid/checkpoint_best.pt"
    assert manifest_path.is_file(), "Run final localization freeze before the localization regression suite"
    assert checkpoint_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["production_promoted"] is False
    assert manifest["training"]["official_test_images_opened"] == 0
    assert sha256(checkpoint_path) == manifest["checkpoint_sha256"]


def test_frozen_adapter_emits_landmark_evidence_only():
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "backend"))
    from app.ml.evidence.idrid_localization_model import IDRiDLocalizationAdapter

    image_path = ROOT / "ml/datasets/raw/idrid/C. Localization/C. Localization/1. Original Images/a. Training Set/IDRiD_001.jpg"
    from PIL import Image

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    manifest = json.loads((ROOT / "ml/weights/localization/idrid/model_manifest.json").read_text(encoding="utf-8"))
    adapter = IDRiDLocalizationAdapter(device="cpu", expected_sha256=manifest["checkpoint_sha256"])
    result = adapter.analyze(image, {"requested_module": "optic_disc_localization"})
    assert result.supported is True
    assert result.landmarks and result.landmarks[0]["landmark_type"] == "optic_disc"
    assert result.metadata["production_promoted"] is False


def test_official_evaluation_is_single_completed_artifact_when_present():
    report_path = META / "idrid_localization_official_test.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["evaluation_status"] == "COMPLETED_ONCE"
    assert report["official_test_count"] == 103
    assert report["official_test_images_opened"] == 103
    assert report["production_promoted"] is False


def test_protected_classifier_hashes_unchanged():
    protected = {
        "ml/weights/classifiers/aptos2019/efficientnet-b0-aptos2019-20260830-v1/checkpoint_best.pt": "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b",
        "ml/weights/classifiers/idrid/efficientnet-b0-idrid-20260912-v1/checkpoint_best.pt": "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de",
    }
    for relative, expected in protected.items():
        path = ROOT / relative
        if path.is_file():
            assert sha256(path) == expected
