import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.classifier import ReferableDRMapping  # noqa: E402
from app.ml.inference.classifier import ClassifierNotConfiguredError, TorchDRClassificationService  # noqa: E402
from ml.evaluation.metrics import classification_metrics  # noqa: E402


def test_referable_mapping_is_explicit_and_configurable():
    default = ReferableDRMapping()
    assert default.to_dict() == {"name": "moderate_or_worse", "referable_grades": [2, 3, 4]}
    assert default.is_referable(1) is False
    assert default.is_referable(2) is True
    mild_or_worse = ReferableDRMapping(name="mild_or_worse", referable_grades=(1, 2, 3, 4))
    assert mild_or_worse.is_referable(1) is True


def _runtime_prediction(probabilities: list[float]):
    service = TorchDRClassificationService(model_path=None, backbone="efficientnet_b0")
    service._torch = torch
    service._ordinal_mode = False
    service._artifact_config = {}
    outputs = {
        "severity_logits": torch.log(torch.tensor([probabilities], dtype=torch.float32)),
        "stage1_logits": torch.zeros((1, 2), dtype=torch.float32),
        "stage2_logits": torch.zeros((1, 2), dtype=torch.float32),
    }
    return service._prediction_from_outputs(outputs)


def test_referable_status_is_probability_rule_independent_of_severity_argmax():
    mild_argmax = _runtime_prediction([0.10, 0.38, 0.21, 0.20, 0.11])
    assert mild_argmax.predicted_grade == 1
    assert mild_argmax.referable_probability == pytest.approx(0.52, abs=1e-6)
    assert mild_argmax.referable_dr is True

    moderate_argmax = _runtime_prediction([0.25, 0.26, 0.30, 0.10, 0.09])
    assert moderate_argmax.predicted_grade == 2
    assert moderate_argmax.referable_probability == pytest.approx(0.49, abs=1e-6)
    assert moderate_argmax.referable_dr is False


def test_idrid_153_uses_probability_referable_rule():
    image_path = ROOT / "ml" / "datasets" / "raw" / "idrid" / "B. Disease Grading" / "B. Disease Grading" / "1. Original Images" / "a. Training Set" / "IDRiD_153.jpg"
    checkpoint = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "checkpoint_best.pt"
    if not image_path.is_file() or not checkpoint.is_file():
        pytest.skip("The selected local IDRiD validation artifact is not available")
    service = TorchDRClassificationService(
        model_path=checkpoint, backbone="efficientnet_b0", model_version="efficientnet-b0-idrid-20260912-v1", device="cpu",
        referable_mapping=ReferableDRMapping(name="grade_2_or_worse", referable_grades=(2, 3, 4)),
    )
    prediction = service.predict(image_path.read_bytes())
    assert prediction.predicted_grade == 1
    assert prediction.referable_probability == pytest.approx(0.56822, abs=1e-4)
    assert prediction.referable_dr is True


def test_classification_metrics_include_five_class_and_referable_results():
    labels = [0, 1, 2, 3, 4]
    probabilities = np.eye(5, dtype=float)
    report = classification_metrics(labels, probabilities)
    assert report["accuracy"] == 1.0
    assert report["confusion_matrix"] == np.eye(5, dtype=int).tolist()
    assert report["referable_dr"]["sensitivity"] == 1.0
    assert report["referable_dr"]["specificity"] == 1.0


def test_inference_does_not_fabricate_without_a_registered_artifact():
    service = TorchDRClassificationService(model_path=None, backbone="efficientnet_b0")
    with pytest.raises(ClassifierNotConfiguredError):
        asyncio.run(service.classify(b"not-an-image"))
