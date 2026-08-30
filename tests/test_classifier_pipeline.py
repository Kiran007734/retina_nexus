import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

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
