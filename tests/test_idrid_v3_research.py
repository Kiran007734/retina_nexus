"""Non-training integrity tests for the isolated IDRiD V3 research artifacts."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
V1 = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "checkpoint_best.pt"
V2 = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v2" / "20260912" / "v2_c_lesion_aware_efficientnet_b0" / "checkpoint_best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def test_v3_required_artifacts_and_selection_are_research_only():
    registry = read("idrid_v3_experiment_registry.json")
    selected = read("idrid_v3_selected_candidate.json")
    assert registry["selection"]["selected_experiment_id"] == "v3_b"
    assert selected["selected_candidate"]["experiment_id"] == "v3_b"
    assert selected["production_promoted"] is False
    assert selected["official_test_images_opened"] == 0
    assert registry["messidor2_used_for_selection"] is False
    assert registry["final_conclusion"] == "V3 FAILED — RETURN TO V2/V1"


def test_v3_checkpoint_and_frozen_baseline_hashes():
    selected = read("idrid_v3_selected_candidate.json")
    candidate = selected["selected_candidate"]
    checkpoint = ROOT / candidate["checkpoint_path"]
    assert checkpoint.is_file()
    assert sha256(checkpoint) == candidate["checkpoint_sha256"]
    assert sha256(V1) == "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
    assert sha256(V2) == "2c5bebd6be18184c56443d8d6a8f590d8b31c8904caea6a0c8b31667f9ac9967"


def test_v3_development_leakage_and_external_policy():
    audit = read("idrid_v3_data_audit.json")
    reproducibility = read("idrid_v3_reproducibility.json")
    calibration = read("idrid_v3_calibration.json")
    assert audit["idrid_integrity"]["status"] == "PASS"
    assert audit["idrid_integrity"]["reserved_test_record_keys_overlap_development"] == []
    assert audit["official_idrid_test_images_opened"] == 0
    assert all(item["predictions_identical"] and item["probabilities_identical_within_1e-6"] for item in reproducibility["candidates"].values())
    assert calibration["status"] == "UNCALIBRATED"
    assert calibration["messidor2_used_for_calibration"] is False


def test_v3_external_evaluation_is_single_and_not_selection_data():
    comparison = read("idrid_v3_external_comparison.json")
    external_path = ROOT / "ml" / "evaluation" / "messidor" / "idrid_v3_domain_robust_zero_shot" / "idrid_v3_messidor2_zero_shot.json"
    assert external_path.is_file()
    assert comparison["messidor2_used_for_selection"] is False
    assert comparison["official_idrid_test_images_opened"] == 0
