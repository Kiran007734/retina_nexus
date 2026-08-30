import asyncio
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.ml.evidence.service import RetinalEvidenceService  # noqa: E402
from ml.evidence.drive import find_drive_pairs  # noqa: E402


def evidence_image() -> bytes:
    rng = np.random.default_rng(3)
    image = np.clip(rng.normal(105, 28, (512, 512, 3)), 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def test_evidence_service_emits_standardized_coarse_to_fine_modules():
    result = asyncio.run(RetinalEvidenceService(max_dimension=512).analyze(evidence_image(), "image-1", "session-1", "right"))
    assert result.status == "completed"
    assert set(result.modules) == {
        "vessel_segmentation", "optic_disc_localization", "fovea_localization",
        "cotton_wool_spot_detection",
        "microaneurysm_detection", "hemorrhage_detection", "exudate_segmentation",
        "neovascularization_detection",
    }
    assert "global_context" in result.coarse_to_fine
    assert "suspicious_region_proposals" in result.coarse_to_fine
    assert "high_resolution_patch_extraction" in result.coarse_to_fine
    assert result.modules["neovascularization_detection"]["supported"] is False
    assert result.evidence_map_data_uri is not None
    assert result.dataset_support["drive"]["vessel_segmentation"]["status"] in {"available", "unsupported"}


def test_drive_pairing_requires_explicit_image_and_mask_files(tmp_path):
    image = Image.new("RGB", (32, 32), (100, 100, 100))
    mask = Image.new("L", (32, 32), 255)
    image.save(tmp_path / "21_training.tif")
    mask.save(tmp_path / "21_manual1.gif")
    pairs = find_drive_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][0].name == "21_training.tif"
    assert pairs[0][1].name == "21_manual1.gif"
