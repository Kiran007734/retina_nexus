import asyncio
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from app.ml.quality.trust_gate import ImageTrustGateError, ImageTrustGateService, QualityBand


def image_bytes(color=(100, 110, 120), size=(512, 512), blur=0):
    image = Image.new("RGB", size, color)
    if blur:
        array = np.array(image)
        array[::8, ::8] = (240, 40, 40)
        image = Image.fromarray(array)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_rejects_small_image():
    with pytest.raises(ImageTrustGateError, match="below the minimum"):
        ImageTrustGateService().validate_input(image_bytes(size=(32, 32)))


def test_rejects_non_image_bytes():
    with pytest.raises(ImageTrustGateError):
        ImageTrustGateService().validate_input(b"not-an-image")


def test_assessment_contains_component_scores_and_action():
    result = asyncio.run(ImageTrustGateService().assess(image_bytes()))
    assert set(result.component_scores) == {"focus", "illumination", "contrast", "field_of_view", "exposure", "artifacts"}
    assert result.quality_decision in {"GRADABLE", "BORDERLINE", "UNGRADABLE"}
    assert result.quality_band in {QualityBand.GREEN, QualityBand.YELLOW, QualityBand.RED}
    assert result.ai_eligible is (result.quality_band == QualityBand.GREEN)
    assert result.next_action in {"CONTINUE_SCREENING", "ENHANCE_AND_REASSESS", "RECAPTURE_IMAGE"}


def test_hard_focus_floor_cannot_be_bypassed_by_enhancement():
    result = asyncio.run(ImageTrustGateService().assess(image_bytes()))
    assert result.quality_band == QualityBand.RED
    assert result.ai_eligible is False
    assert result.hard_focus_floor_passed is False
    assert "focus_acquisition_floor" in result.non_recoverable_issues


def test_known_representative_quality_bands_when_dataset_is_available():
    root = Path(__file__).resolve().parents[1]
    paths = {
        "good": root / "ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png",
        "borderline": root / "ml/datasets/raw/aptos2019/train_images/005b95c28852.png",
        "ungradable": root / "ml/datasets/raw/aptos2019/train_images/000c1434d8d7.png",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("APTOS representative fixtures are not available")

    service = ImageTrustGateService()
    good = asyncio.run(service.assess(paths["good"].read_bytes()))
    borderline = asyncio.run(service.assess(paths["borderline"].read_bytes()))
    ungradable = asyncio.run(service.assess(paths["ungradable"].read_bytes()))

    assert good.quality_band == QualityBand.GREEN
    assert good.ai_eligible is True
    assert borderline.quality_band == QualityBand.YELLOW
    assert borderline.ai_eligible is False
    assert ungradable.quality_band == QualityBand.RED
    assert ungradable.ai_eligible is False


def test_ood_distribution_summary():
    from app.ml.quality.ood import summarize_quality_distribution
    summary = summarize_quality_distribution([{"focus": 0.4}, {"focus": 0.8}])
    assert summary["focus"]["count"] == 2
    assert summary["focus"]["mean"] == 0.6000000000000001
