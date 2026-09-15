"""Consolidate frozen IDRiD localization artifacts without hiding limitations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MODEL_DIR = ROOT / "ml" / "weights" / "localization" / "idrid"
OUTPUT = META / "idrid_localization_final_report.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    cv = read(META / "idrid_localization_cv_report.json")
    baseline = read(META / "idrid_localization_baseline.json")
    model = read(MODEL_DIR / "model_manifest.json")
    official = read(META / "idrid_localization_official_test.json")
    robustness = read(META / "idrid_localization_robustness.json")
    reproducibility = read(META / "idrid_localization_reproducibility.json")
    errors = read(META / "idrid_localization_error_analysis.json")
    if not cv or not model:
        raise SystemExit("CV report and frozen model manifest are required")
    baseline_metrics = baseline.get("metrics", {}) if baseline else {}
    cv_summary = cv.get("summary", {})
    cv_od = cv_summary.get("optic_disc", {}).get("mean_normalized_error", {}).get("mean")
    cv_fovea = cv_summary.get("fovea", {}).get("mean_normalized_error", {}).get("mean")
    baseline_od = baseline_metrics.get("optic_disc", {}).get("mean_normalized_error")
    baseline_fovea = baseline_metrics.get("fovea", {}).get("mean_normalized_error")
    aggregate_improved = all(value is not None for value in (cv_od, cv_fovea, baseline_od, baseline_fovea)) and (cv_od + cv_fovea) < (baseline_od + baseline_fovea)
    official_completed = bool(official and official.get("evaluation_status") == "COMPLETED_ONCE" and official.get("official_test_images_opened") == 103)
    reproducibility_pass = bool(reproducibility and reproducibility.get("identical_within_tolerance"))
    # The status is deliberately conservative: a frozen research artifact may
    # exist even when its evidence quality does not justify a broader claim.
    status = "LOCALIZATION COMPLETE — READY FOR VESSEL MODULE" if official_completed and reproducibility_pass and aggregate_improved else "LOCALIZATION REQUIRES FURTHER RESEARCH"
    report = {
        "schema_version": "idrid-localization-final-report-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selected_model": model,
        "cv": {"report": "ml/datasets/metadata/idrid/idrid_localization_cv_report.json", "summary": cv_summary, "baseline": baseline_metrics, "aggregate_normalized_error_improved_vs_baseline": aggregate_improved},
        "official_test": official,
        "robustness": robustness,
        "reproducibility": reproducibility,
        "error_analysis": {"artifact": "ml/datasets/metadata/idrid/idrid_localization_error_analysis.json", "available": errors is not None},
        "integration": {"adapter": "backend/app/ml/evidence/idrid_localization_model.py", "opt_in": True, "changes_dr_classification": False, "changes_retinaguard": False, "production_promoted": False},
        "known_limitations": ["No patient identifiers were available; duplicate-aware image-level grouping was used.", "IDRiD_118 was excluded from development because it duplicates official test IDRiD_064.", "Classical optic-disc baseline was stronger on optic-disc normalized error in the development audit; the learned model's fovea result must not be interpreted as clinical validation.", "No production promotion or clinical performance claim is made."],
        "official_test_images_opened": int(official.get("official_test_images_opened", 0)) if official else 0,
        "production_promoted": False,
        "clinical_validation_claim": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "aggregate_normalized_error_improved_vs_baseline": aggregate_improved, "official_test_images_opened": report["official_test_images_opened"], "production_promoted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
