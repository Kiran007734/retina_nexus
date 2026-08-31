from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.evaluation.messidor import build_reports


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path, format="TIFF")


def test_empty_messidor_copy_is_blocked_without_fabricating_data(tmp_path: Path):
    raw = tmp_path / "raw"
    output = tmp_path / "reports"
    reports = build_reports(raw, output, "messidor2")

    readiness = reports["phase4_readiness_report"]
    assert readiness["status"] == "NOT_READY"
    assert readiness["ready_for_zero_shot_external_validation"] is False
    assert reports["validation_report"]["image_count"] == 0
    assert reports["validation_report"]["annotations"]["valid_grade_rows"] == 0


def test_messidor_grade_rows_and_conflicts_are_reported_without_relabeling(tmp_path: Path):
    raw = tmp_path / "raw"
    _image(raw / "Base11" / "a.tif", (10, 20, 30))
    _image(raw / "Base11" / "b.tif", (30, 20, 10))
    (raw / "Base11" / "diagnosis.csv").write_text(
        "Image,Retinopathy grade\n"
        "a.tif,2\n"
        "a.tif,3\n"
        "b.tif,0\n",
        encoding="utf-8",
    )
    reports = build_reports(raw, tmp_path / "reports", "messidor")

    validation = reports["validation_report"]
    annotations = validation["annotations"]
    assert annotations["valid_grade_rows"] == 3
    assert annotations["class_distribution"] == {0: 1, 2: 1, 3: 1}
    assert len(annotations["annotation_conflicts"]) == 1
    assert validation["status"] == "BLOCKED"
    compatibility = reports["grading_compatibility"]
    assert compatibility["five_class_evaluation"]["status"] == "INVALID"
    assert compatibility["binary_evaluation"]["status"] == "CONDITIONALLY_SUPPORTED"


def test_official_messidor2_pairing_metadata_is_not_treated_as_ground_truth(tmp_path: Path):
    raw = tmp_path / "raw"
    _image(raw / "IMAGES" / "left.png", (10, 20, 30))
    (raw / "pairing.csv").write_text("left,right\nleft.png,right.png\n", encoding="utf-8")
    reports = build_reports(raw, tmp_path / "reports", "messidor2")

    compatibility = reports["grading_compatibility"]
    assert compatibility["external_dataset"]["source_status"] == "official_release_has_no_labels"
    assert compatibility["binary_evaluation"]["status"] == "NOT_CALCULABLE"
    assert reports["validation_report"]["annotations"]["valid_grade_rows"] == 0

    saved = json.loads((tmp_path / "reports" / "phase4_readiness_report.json").read_text(encoding="utf-8"))
    assert saved["clinical_validation_claim"] is False
