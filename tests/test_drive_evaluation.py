"""Focused tests for DRIVE discovery and pixel-metric correctness."""

from pathlib import Path

import numpy as np
from PIL import Image

from ml.evaluation.drive import discover_drive_files, group_drive_files, segmentation_metrics


def _write_drive_fixture(root: Path) -> None:
    folders = [
        root / "DRIVE" / "training" / "images",
        root / "DRIVE" / "training" / "1st_manual",
        root / "DRIVE" / "training" / "mask",
        root / "DRIVE" / "test" / "images",
        root / "DRIVE" / "test" / "mask",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (100, 100, 100)).save(root / "DRIVE" / "training" / "images" / "01_training.tif")
    Image.new("L", (4, 4), 255).save(root / "DRIVE" / "training" / "1st_manual" / "01_manual1.gif")
    Image.new("L", (4, 4), 255).save(root / "DRIVE" / "training" / "mask" / "01_training_mask.gif")
    Image.new("RGB", (4, 4), (120, 120, 120)).save(root / "DRIVE" / "test" / "images" / "02_test.tif")
    Image.new("L", (4, 4), 255).save(root / "DRIVE" / "test" / "mask" / "02_test_mask.gif")


def test_drive_discovery_pairs_training_ground_truth_and_test_fov(tmp_path):
    _write_drive_fixture(tmp_path)
    root = tmp_path / "DRIVE"
    discovered = discover_drive_files(root)
    assert len(discovered["image"]) == 2
    assert len(discovered["vessel_mask"]) == 1
    assert len(discovered["fov_mask"]) == 2
    grouped = group_drive_files(root)
    assert grouped["01"]["vessel_mask"]
    assert grouped["01"]["fov_mask"]
    assert "vessel_mask" not in grouped["02"]


def test_segmentation_metrics_apply_fov_and_return_expected_values():
    target = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    probability = np.array([[0.9, 0.1, 0.9], [0.1, 0.1, 0.1]], dtype=np.float32)
    fov = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
    metrics = segmentation_metrics(target, probability, fov, threshold=0.5)
    assert metrics["true_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["true_negative"] == 2
    assert metrics["dice"] == 2 / 3
    assert metrics["iou"] == 0.5
    assert metrics["pixel_accuracy"] == 0.75
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 2 / 3
