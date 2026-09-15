"""Assemble the final IDRiD lesion research report from measured artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"


def read(name: str) -> dict:
    return json.loads((META / name).read_text(encoding="utf-8"))


def main() -> int:
    audit = read("idrid_lesion_data_audit.json")
    manifest = read("idrid_lesion_manifest.json")
    split = read("idrid_lesion_split.json")
    cv = read("idrid_lesion_cv_report.json")
    threshold = read("idrid_lesion_threshold_analysis.json")
    errors = read("idrid_lesion_error_analysis.json")
    official = read("idrid_lesion_official_test.json")
    robustness = read("idrid_lesion_robustness.json")
    benchmark_path = META / "idrid_lesion_inference_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.is_file() else {"status": "NOT_RUN"}
    baseline_path = META / "idrid_lesion_external_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {"status": "NOT_RUN"}
    model_manifest = json.loads((ROOT / "ml/weights/lesions/idrid/model_manifest.json").read_text(encoding="utf-8"))
    train_records = [item for item in manifest["records"] if item["split"] == "train"]
    test_records = [item for item in manifest["records"] if item["split"] == "test"]
    payload = {
        "schema_version": "idrid-lesion-final-report-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LESION DETECTION REQUIRES FURTHER RESEARCH",
        "status_reason": "The model is frozen and measurable, but the small development set, low microaneurysm Dice, missing FOV masks, incomplete soft-exudate annotations, and lack of clinical validation prevent a production or clinical-readiness claim.",
        "dataset": {
            "package": "IDRiD A. Segmentation",
            "training_images": len(train_records),
            "official_test_images": len(test_records),
            "mask_availability_training": {name: sum(record["masks"][name]["status"] == "AVAILABLE" for record in train_records) for name in model_manifest["classes"]},
            "mask_availability_official_test": {name: sum(record["masks"][name]["status"] == "AVAILABLE" for record in test_records) for name in model_manifest["classes"]},
            "fov_masks": "UNAVAILABLE",
            "corrupt_images": audit.get("summary", {}).get("unreadable_images", []),
            "duplicate_groups": audit.get("summary", {}).get("exact_duplicate_groups", []),
        },
        "selected_model": model_manifest,
        "development_cv": cv,
        "threshold_selection": threshold,
        "development_error_analysis": errors,
        "external_baseline": baseline,
        "official_test": official,
        "robustness": robustness,
        "inference_benchmark": benchmark,
        "integration": {
            "adapter": "backend/app/ml/evidence/idrid_lesion_model.py",
            "enabled_by_default": False,
            "classification_grade_unchanged": True,
            "retinaguard_grade_override": False,
            "production_promoted": False,
        },
        "data_split": split,
        "official_test_images_opened": 27,
        "official_test_evaluated_once": official.get("evaluation_completed") is True,
        "production_promoted": False,
        "clinical_validation_claim": False,
        "next_step": "Further lesion model research and/or localization work may proceed only as research; no production promotion without a new validation decision.",
    }
    path = META / "idrid_lesion_final_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "status": payload["status"], "production_promoted": False, "official_test_images_opened": 27}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
