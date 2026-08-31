from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.evaluation.messidor2 import (  # noqa: E402
    bootstrap_confidence_intervals,
    compute_metrics,
    discover_label_source,
    normalize_stem,
)


def _record(grade: int, *, status: str = "SUCCESS", gradable: int = 1) -> dict[str, object]:
    probabilities = {f"probability_{index}": 1.0 if index == grade else 0.0 for index in range(5)}
    return {
        "inference_status": status,
        "adjudicated_gradable": gradable,
        "adjudicated_dr_grade": grade if gradable else None,
        **probabilities,
    }


def test_messidor2_metrics_are_calculated_from_successful_gradable_records():
    records = [_record(index) for index in range(5)]
    records.append(_record(4, status="ERROR"))
    records.append(_record(3, gradable=0))

    metrics, per_class = compute_metrics(records, "test population")

    assert metrics["status"] == "CALCULATED"
    assert metrics["sample_count"] == 5
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["quadratic_weighted_kappa"] == 1.0
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    assert per_class["per_class"]["4"]["support"] == 1


def test_messidor2_bootstrap_is_seeded_and_reports_intervals():
    records = [_record(index) for index in range(5)] * 3

    first = bootstrap_confidence_intervals(records, iterations=100, seed=42)
    second = bootstrap_confidence_intervals(records, iterations=100, seed=42)

    assert first == second
    assert first["status"] == "CALCULATED"
    assert first["sample_count"] == 15
    assert first["iterations"] == 100
    assert "accuracy" in first["intervals"]
    assert first["intervals"]["accuracy"]["lower"] == 1.0
    assert first["intervals"]["accuracy"]["upper"] == 1.0


def test_label_source_requires_schema_and_stem_matching_is_extension_independent(tmp_path: Path):
    label_file = tmp_path / "messidor_data.csv"
    label_file.write_text(
        "image_id,adjudicated_dr_grade,adjudicated_dme,adjudicated_gradable\n"
        "IM0001.JPG,0,0,1\n",
        encoding="utf-8",
    )

    source, errors = discover_label_source(tmp_path, str(label_file))

    assert not errors
    assert source is not None
    assert source.discovery_method == "explicit_cli_path"
    assert normalize_stem("folder/IM0001.JPG") == "im0001"
    assert normalize_stem("IM0001.png") == "im0001"
