import asyncio
import io

import numpy as np
from PIL import Image
import pytest

from app.ml.quality.trust_gate import ImageTrustGateError, ImageTrustGateService


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
    assert result.next_action in {"CONTINUE_SCREENING", "ENHANCE_AND_REASSESS", "RECAPTURE_IMAGE"}


def test_ood_distribution_summary():
    from app.ml.quality.ood import summarize_quality_distribution
    summary = summarize_quality_distribution([{"focus": 0.4}, {"focus": 0.8}])
    assert summary["focus"]["count"] == 2
    assert summary["focus"]["mean"] == 0.6000000000000001
