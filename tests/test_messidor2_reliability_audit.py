from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_messidor2_reliability import _error_analysis  # noqa: E402


def _row(reference: int, predicted: int, referable_probability: float) -> dict[str, str]:
    return {
        "inference_status": "SUCCESS",
        "adjudicated_gradable": "1",
        "adjudicated_dr_grade": str(reference),
        "predicted_aptos_grade": str(predicted),
        "raw_confidence": "0.8",
        "referable_probability_grade_2_or_worse": str(referable_probability),
        "image_id": f"image-{reference}-{predicted}",
        "image_path": f"images/image-{reference}-{predicted}.png",
    }


def test_error_analysis_uses_referable_rule_independently_of_grade_argmax():
    report = _error_analysis(
        [
            _row(reference=0, predicted=0, referable_probability=0.60),
            _row(reference=2, predicted=2, referable_probability=0.40),
        ]
    )

    assert report["misclassified_count"] == 0
    assert report["false_positive_count"] == 1
    assert report["false_negative_count"] == 1
