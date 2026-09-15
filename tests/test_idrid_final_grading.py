"""Artifact-level guards for the final IDRiD grading research cycle."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
FINAL = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "final"
V1_SHA = "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
V2_SHA = "2c5bebd6be18184c56443d8d6a8f590d8b31c8904caea6a0c8b31667f9ac9967"
V3_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"


def read(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def test_final_data_audit_is_development_only_and_clean():
    audit = read("idrid_final_grading_data_audit.json")
    assert audit["official_test_images_opened"] == 0
    assert audit["no_external_datasets_used"] is True
    assert audit["leakage"]["status"] == "PASS"
    assert audit["cross_dataset_exact_sha256_matches"] == []
    assert audit["aptos"]["inventory_summary"]["readable_count"] == 3046
    assert audit["idrid"]["inventory_summary"]["readable_count"] == 406


def test_four_bounded_candidates_and_selection_are_recorded():
    comparison = read("idrid_final_grading_comparison.json")
    assert {row["candidate"] for row in comparison["candidate_rows"]} == {"a", "b", "c", "d"}
    assert comparison["selected_candidate"] == "a"
    assert comparison["production_promoted"] is False
    assert comparison["official_test_images_opened"] == 0


def test_final_checkpoint_and_manifest_are_frozen_without_overwriting_prior_versions():
    manifest = json.loads((FINAL / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["freeze_status"] == "FROZEN_BEFORE_OFFICIAL_TEST"
    assert manifest["production_promoted"] is False
    assert manifest["checkpoint_sha256"] == V3_SHA
    assert (FINAL / "checkpoint_best.pt.sha256").read_text(encoding="utf-8").strip() == V3_SHA
    assert (FINAL / "training_config.json").is_file()
    assert json.loads((FINAL / "training_config.json").read_text(encoding="utf-8"))["referable_threshold"] == 0.4
    assert json.loads((ROOT / "ml/weights/classifiers/idrid/efficientnet-b0-idrid-20260912-v1/model_manifest.json").read_text(encoding="utf-8"))["model_version"] == "efficientnet-b0-idrid-20260912-v1"
    assert V1_SHA == "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
    assert V2_SHA == "2c5bebd6be18184c56443d8d6a8f590d8b31c8904caea6a0c8b31667f9ac9967"


def test_reproducibility_and_robustness_artifacts_pass():
    repro = read("idrid_final_grading_reproducibility.json")
    robust = read("idrid_final_grading_robustness.json")
    assert repro["status"] == "PASS"
    assert repro["validation_count"] == 83
    assert repro["probabilities_identical_within_1e-6"] is True
    assert robust["status"] == "COMPLETED"
    assert {"brightness_minus", "brightness_plus", "contrast_minus", "contrast_plus", "mild_blur", "jpeg_compression", "small_fov_crop", "illumination_change"} <= set(robust["perturbations"])


def test_official_evaluation_is_post_freeze_and_not_tuned():
    report = read("idrid_final_grading_official_test.json")
    assert report["evaluation_type"] == "ONE_POST_FREEZE_OFFICIAL_IDRID_TEST_EVALUATION"
    assert report["official_test_images_opened"] == 103
    assert report["dataset"]["readable_count"] == 103
    assert report["checkpoint_unchanged"] is True
    assert report["post_test_tuning"] is False
    assert report["metrics"]["sample_count"] == 103
    assert report["metrics"]["referable_dr"]["threshold"] == 0.4
