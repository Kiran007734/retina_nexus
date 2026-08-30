import asyncio
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.ml.evidence.service import _png_data_uri, RetinalEvidenceAnalysis  # noqa: E402
from app.ml.explainability.service import ExplainabilityService  # noqa: E402
from app.ml.inference.classifier import DRPrediction  # noqa: E402


def test_attention_agreement_is_linked_to_supported_lesion_regions():
    height, width = 64, 64
    lesion_mask = np.zeros((height, width), dtype=bool)
    lesion_mask[20:36, 24:40] = True
    evidence = RetinalEvidenceAnalysis(
        image_id="image-1", screening_session_id="session-1", status="completed",
        image_metadata={}, coarse_to_fine={}, modules={
            "microaneurysm_detection": {
                "module": "microaneurysm_detection", "category": "lesion_detection", "supported": True,
                "mask_data_uri": _png_data_uri(lesion_mask), "bounding_regions": [],
            },
        }, anatomical_landmarks=[], evidence_map_data_uri=_png_data_uri(lesion_mask), dataset_support={}, note="test",
    )
    attention = np.full((height, width), 0.05, dtype=np.float32)
    attention[20:36, 24:40] = 0.95
    classifier = FakeClassifier(attention)
    result = asyncio.run(ExplainabilityService(classifier).analyze(_image_bytes(), "image-1", "session-1", evidence))

    assert result.attention_lesion_agreement["status"] == "HIGH AGREEMENT"
    assert result.attention_lesion_agreement["score"] > 0.9
    assert result.explanation_stability["status"] == "SKIPPED"
    assert result.counterfactual["status"] == "SKIPPED"
    assert result.grad_cam["normalized_attention_map_data_uri"].startswith("data:image/png;base64,")


def test_stability_and_counterfactual_are_explicit_opt_in():
    attention = np.zeros((48, 48), dtype=np.float32)
    attention[12:28, 12:28] = 0.9
    classifier = FakeClassifier(attention)
    evidence = RetinalEvidenceAnalysis(
        image_id="image-2", screening_session_id="session-2", status="completed", image_metadata={}, coarse_to_fine={},
        modules={}, anatomical_landmarks=[], evidence_map_data_uri=None, dataset_support={}, note="test",
    )
    result = asyncio.run(ExplainabilityService(classifier, max_stability_variants=2).analyze(
        _image_bytes(48), "image-2", "session-2", evidence, run_stability=True, run_counterfactual=True,
    ))

    assert result.explanation_stability["status"] == "COMPLETED"
    assert result.explanation_stability["variant_count"] == 2
    assert result.explanation_stability["prediction_stability"] == 1.0
    assert result.counterfactual["status"] == "COMPLETED"
    assert result.counterfactual["experimental"] is True


def _image_bytes(size: int = 64) -> bytes:
    image = Image.new("RGB", (size, size), (90, 110, 130))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeClassifier:
    def __init__(self, attention: np.ndarray):
        self.attention = attention
        self.prediction = DRPrediction(
            predicted_grade=2, predicted_grade_label="Moderate", probabilities={"No DR": 0.05, "Mild": 0.1, "Moderate": 0.7, "Severe": 0.1, "Proliferative DR": 0.05},
            referable_dr=True, referable_probability=0.85, raw_confidence=0.7, model_name="test", model_version="test-1", backbone="test", referable_mapping={}, hierarchical_probabilities={}, ordinal_mode=False,
        )

    async def explain_async(self, image_bytes: bytes, target_class: int | None = None):
        return SimpleNamespace(prediction=self.prediction, attention_map=self.attention.copy(), target_class=target_class or 2, target_layer="test-layer")

    async def classify(self, image_bytes: bytes):
        return self.prediction
