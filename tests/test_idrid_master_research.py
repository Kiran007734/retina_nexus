"""Guards for the development-only IDRiD master research cycle."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
V3_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"


def read(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def test_master_data_audit_is_complete_and_development_only():
    audit = read("idrid_final_data_audit.json")
    assert audit["official_test_images_opened"] == 0
    assert audit["no_external_datasets_used"] is True
    assert audit["idrid_development"]["record_count"] == 406
    assert audit["idrid_development"]["readable_count"] == 406
    assert audit["idrid_development"]["unreadable"] == []
    assert audit["idrid_development"]["class_distribution"] == {"0": 133, "1": 19, "2": 133, "3": 73, "4": 48}
    assert audit["leakage"]["status"] == "PASS"
    assert audit["patient_ids_available"] is False


def test_cv_is_five_fold_grouped_and_reports_variability():
    study = read("idrid_final_cv_stability.json")
    assert study["fold_count"] == 5
    assert study["official_test_images_opened"] == 0
    assert study["duplicate_group_overlap_any_fold"] is False
    assert len(study["folds"]) == 5
    assert sum(fold["validation_count"] for fold in study["folds"]) == 406
    assert study["summary"]["referable_sensitivity"]["min"] < study["summary"]["referable_sensitivity"]["max"]
    assert study["summary"]["referable_specificity"]["mean"] >= 0.85


def test_forensic_audit_records_v1_v2_v3_changes():
    audit = read("idrid_final_data_audit.json")["implementation_audit"]
    assert {"v1", "v2", "v3"} <= set(audit)
    assert len(audit["exact_changes"]) >= 3
    assert audit["official_test_used_for_this_audit"] is False


def test_current_cycle_does_not_rerun_official_test_or_promote_model():
    report = read("idrid_final_official_test.json")
    assert report["status"] == "ALREADY_EVALUATED_ONCE_AFTER_PRIOR_FREEZE; NOT_RERUN"
    assert report["official_test_images_opened_in_current_cycle"] == 0
    assert report["do_not_rerun"] is True
    selected = read("idrid_final_selected_candidate.json")
    assert selected["checkpoint_sha256"] == V3_SHA
    assert selected["production_promoted"] is False


def test_master_artifact_set_exists():
    required = {
        "idrid_final_cv_stability.json",
        "idrid_final_data_audit.json",
        "idrid_final_experiment_registry.json",
        "idrid_final_model_comparison.json",
        "idrid_final_threshold_analysis.json",
        "idrid_final_calibration.json",
        "idrid_final_robustness.json",
        "idrid_final_reliability.json",
        "idrid_final_selected_candidate.json",
        "idrid_final_reproducibility.json",
        "idrid_final_official_test.json",
    }
    assert required <= {path.name for path in META.glob("idrid_final_*.json")}
