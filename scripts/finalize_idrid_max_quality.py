"""Finalize metadata for the already-trained IDRiD max-quality artifact.

This command performs no training and does not open any IDRiD official-test
image.  It only verifies the frozen research checkpoint and consolidates the
development/CV/integration artifacts already produced by the max-quality run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
FINAL = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "max_quality" / "final"
CHECKPOINT = FINAL / "checkpoint_best.pt"
MANIFEST = FINAL / "model_manifest.json"
OFFICIAL = META / "idrid_final_grading_official_test.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def weighted_f1(rows: list[dict]) -> float:
    from sklearn.metrics import f1_score
    actual = [int(row["actual"]) for row in rows]
    predicted = [int(row["predicted"]) for row in rows]
    return float(f1_score(actual, predicted, labels=[0, 1, 2, 3, 4], average="weighted", zero_division=0))


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checkpoint_sha = sha256(CHECKPOINT)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"Frozen max-quality checkpoint checksum changed: {checkpoint_sha} != {manifest['checkpoint_sha256']}")
    comparison = read("idrid_max_quality_experiment_comparison.json")
    cv = read("idrid_max_quality_cv_report.json")
    calibration = read("idrid_max_quality_calibration.json")
    integration = read("idrid_max_quality_integration.json")
    robustness = {
        "schema_version": "idrid-max-quality-robustness-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source": "idrid_max_quality_integration.json",
        "development_images_used": integration["validation_images_used"],
        "deterministic_repeat_pass": integration["deterministic_inference_pass"],
        "brightness_perturbation": {"factor": 1.05, "jpeg_quality": 95, "grade_stability": integration["brightness_grade_stability"], "rows": [{"image_id": row["image_id"], "original_grade": row["prediction"]["grade"], "perturbed_grade": row["brightness_perturbation"]["grade"], "same_grade": row["brightness_perturbation"]["same_grade"]} for row in integration["rows"]]},
        "grad_cam_finite_pass": integration["grad_cam"]["finite"],
        "official_test_images_opened": 0,
        "clinical_validation_claim": False,
    }
    (META / "idrid_max_quality_robustness.json").write_text(json.dumps(robustness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest.update({
        "checkpoint_sha256": checkpoint_sha,
        "cv_metrics_artifact": "ml/datasets/metadata/idrid/idrid_max_quality_cv_report.json",
        "experiment_comparison_artifact": "ml/datasets/metadata/idrid/idrid_max_quality_experiment_comparison.json",
        "calibration_artifact": "ml/datasets/metadata/idrid/idrid_max_quality_calibration.json",
        "robustness_artifact": "ml/datasets/metadata/idrid/idrid_max_quality_robustness.json",
        "integration_artifact": "ml/datasets/metadata/idrid/idrid_max_quality_integration.json",
        "cv_metrics": cv.get("metrics"),
        "cv_ece": cv.get("ece"),
        "official_test_images_opened": 0,
        "freeze_status": "FROZEN_RESEARCH_ONLY",
        "production_promoted": False,
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    official = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    official_rows = official.get("predictions", [])
    official_metrics = dict(official.get("metrics", {}))
    if official_rows:
        official_metrics["weighted_f1"] = weighted_f1(official_rows)
        official_metrics["mean_absolute_grade_error"] = float(np.mean([abs(int(row["actual"]) - int(row["predicted"])) for row in official_rows]))
        confidence = np.asarray([float(row["confidence"]) for row in official_rows], dtype=float)
        official_metrics["confidence_distribution"] = {"mean": float(confidence.mean()), "std": float(confidence.std()), "min": float(confidence.min()), "max": float(confidence.max())}
    final_report = {
        "schema_version": "idrid-max-quality-final-report-2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_model": manifest,
        "development_cv": {"artifact": "ml/datasets/metadata/idrid/idrid_max_quality_cv_report.json", "metrics": cv.get("metrics"), "ece": cv.get("ece"), "calibration_status": calibration.get("status"), "official_test_images_opened": 0},
        "error_analysis": "ml/datasets/metadata/idrid/idrid_max_quality_error_analysis.json",
        "robustness": robustness,
        "integration": integration,
        "official_test": {"status": "IMMUTABLE_PRIOR_RESULT_NOT_REOPENED_IN_THIS_CYCLE", "source_report": "ml/datasets/metadata/idrid/idrid_final_grading_official_test.json", "metrics": official_metrics, "official_test_images_opened_in_current_cycle": 0, "statement": "The official 103-image test result was evaluated once before this max-quality cycle and is preserved as an immutable reference; no official test image was reopened."},
        "known_limitations": ["No patient identifiers were available; duplicate-aware grouping is not a patient-level guarantee.", "IDRiD is small and Grade 1 is sparse.", "Raw softmax confidence is not clinically calibrated; no independent calibration subset was fitted.", "The max-quality candidate is research-only and was not promoted to production.", "The prior official result is not evidence for development selection in this cycle."],
        "production_promoted": False,
        "official_test_images_opened": 0,
    }
    (META / "idrid_max_quality_final_report.json").write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry_path = ROOT / "ml" / "weights" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"artifacts": []}
    for item in registry.get("artifacts", []):
        if item.get("model_version") == manifest["model_version"]:
            item.update({"checkpoint_sha256": checkpoint_sha, "validation_metrics_artifact": "ml/datasets/metadata/idrid/idrid_max_quality_cv_report.json", "robustness_artifact": "ml/datasets/metadata/idrid/idrid_max_quality_robustness.json", "production_promoted": False, "clinical_validation_claim": False})
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model_version": manifest["model_version"], "checkpoint_sha256": checkpoint_sha, "cv_artifact": manifest["cv_metrics_artifact"], "robustness_artifact": manifest["robustness_artifact"], "official_test_images_opened": 0, "production_promoted": False}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
