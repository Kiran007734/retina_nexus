"""Regression guards for the isolated IDRiD max-quality research artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
FINAL = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "max_quality" / "final"


def read(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def test_max_quality_audit_is_development_only_and_readable():
    audit = read("idrid_max_quality_data_audit.json")
    assert audit["official_idrid_test_images_opened"] == 0
    assert audit["aptos_competition_test_images_opened"] == 0
    assert audit["idrid_development"]["record_count"] == 406
    assert audit["idrid_development"]["readable_count"] == 406
    assert audit["aptos_training"]["record_count"] == 2509
    assert audit["unreadable"] == []
    assert audit["leakage"]["status"] == "PASS"


def test_max_quality_cv_and_freeze_are_not_production():
    comparison = read("idrid_max_quality_experiment_comparison.json")
    assert comparison["selected_candidate"] == "b0_crop_384"
    assert comparison["official_test_used_for_selection"] is False
    assert comparison["official_test_images_opened"] == 0
    manifest = json.loads((FINAL / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_version"] == "idrid-max-quality-20260912-v1"
    assert manifest["production_promoted"] is False
    assert manifest["official_test_images_opened"] == 0
    assert manifest["input_size"] == 384
    assert manifest["retinal_field_crop"] is True


def test_max_quality_checkpoint_checksum_matches_manifest():
    manifest = json.loads((FINAL / "model_manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((FINAL / "checkpoint_best.pt").read_bytes()).hexdigest()
    assert digest == manifest["checkpoint_sha256"]
    assert digest == "075d7e41714cac28cfd0b60eba39fc50b81f7a8788d743004cd9dda702e1cbab"


def test_max_quality_official_result_is_not_reopened():
    report = read("idrid_max_quality_final_report.json")
    assert report["official_test"]["official_test_images_opened_in_current_cycle"] == 0
    assert report["official_test"]["status"] == "IMMUTABLE_PRIOR_RESULT_NOT_REOPENED_IN_THIS_CYCLE"
    assert report["production_promoted"] is False
