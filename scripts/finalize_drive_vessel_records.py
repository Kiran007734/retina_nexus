"""Write the frozen DRIVE research comparison, report, and registry entry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "drive"
CHECKPOINT = ROOT / "ml" / "weights" / "vessels" / "drive" / "checkpoint_best.pt"


def read(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = read("drive_manifest.json")
    audit = read("drive_data_audit.json")
    final_training = read("drive_final_training.json")
    model_manifest = json.loads((CHECKPOINT.parent / "model_manifest.json").read_text(encoding="utf-8"))
    threshold = read("drive_threshold_analysis.json")
    official = read("drive_official_test.json")
    rgb = read("drive_cv_report_scratch-rgb-bce-dice-512-v1.json")
    green = read("drive_cv_report_scratch-green-focal-dice-512-v1.json")
    r2 = json.loads((ROOT / "ml" / "evaluation" / "drive" / "r2-v2-evaluation.json").read_text(encoding="utf-8"))
    actual_hash = sha256(CHECKPOINT)
    if actual_hash != model_manifest["checkpoint_sha256"]:
        raise SystemExit("Frozen DRIVE research checkpoint SHA does not match its manifest")
    selected_threshold = threshold["selected"]
    comparison = {
        "schema_version": "drive-vessel-experiment-comparison-1",
        "dataset": "DRIVE",
        "development_protocol": "20 labeled training images, deterministic five-fold OOF development split, FOV applied",
        "official_test_used_for_model_selection": False,
        "candidates": {
            "scratch_rgb_bce_dice_512": {"candidate": rgb["candidate"], "preprocessing": rgb["preprocessing"], "loss": rgb["loss"], "cv_selected_summary": rgb["selected_summary"]},
            "scratch_green_focal_dice_512": {"candidate": green["candidate"], "preprocessing": green["preprocessing"], "loss": green["loss"], "cv_selected_summary": green["selected_summary"], "threshold_selected_from_oof": selected_threshold},
            "protected_r2_v2_reference": {"model_version": "r2-v2-bv-2025", "evaluation_scope": "20 labeled DRIVE training images from the existing protected R2-V2 evaluation artifact", "metrics": r2.get("metrics", {}).get("mean", {}), "checkpoint_sha256": "ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a", "production_path": "default vessel adapter"},
        },
        "selected_research_candidate": {"model_version": model_manifest["model_version"], "checkpoint": model_manifest["checkpoint"], "checkpoint_sha256": actual_hash, "threshold": model_manifest["threshold"], "cv_metrics": model_manifest["cv_metrics"], "threshold_selection": model_manifest["threshold_selection"], "final_fit_metrics_not_for_model_selection": model_manifest["final_fit_metrics_not_for_model_selection"]},
        "interpretation": "The newly trained candidate is substantially weaker than the protected R2-V2 reference on the available development evidence and is not promoted. The comparison is engineering evidence, not a clinical validation claim.",
    }
    dump(META / "drive_experiment_comparison.json", comparison)
    final_report = {
        "schema_version": "drive-vessel-final-report-1",
        "status": "VESSEL SEGMENTATION REQUIRES FURTHER RESEARCH",
        "dataset": "DRIVE",
        "dataset_version": manifest["dataset_version"],
        "audit": {"training_images": audit["counts"]["training_images"], "official_test_images": audit["counts"]["test_images"], "manual_training_vessel_masks": audit["counts"]["training_manual_vessel_masks"], "manual_test_vessel_masks": audit["counts"]["test_manual_vessel_masks"], "corrupt_files": audit["counts"]["corrupt_or_unreadable_files"], "missing_pairs": audit["validation"]["missing_pairs"], "exact_duplicates": audit["validation"]["cross_split_exact_duplicates"], "confirmed_perceptual_duplicates": []},
        "research_model": {"model_version": model_manifest["model_version"], "checkpoint": model_manifest["checkpoint"], "checkpoint_sha256": actual_hash, "architecture": model_manifest["architecture"], "preprocessing": model_manifest["preprocessing"], "threshold": model_manifest["threshold"], "threshold_selection": model_manifest["threshold_selection"], "development_cv": model_manifest["cv_metrics"], "final_fit": final_training["fit_metrics"], "production_promoted": False},
        "protected_reference": {"model_version": "r2-v2-bv-2025", "checkpoint_sha256": "ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a", "role": "primary/default vessel model; unchanged", "development_metrics": r2.get("metrics", {}).get("mean", {})},
        "official_test": {"status": official["status"], "official_test_images_opened": official["official_test_images_opened"], "manual_vessel_masks_available": official["manual_vessel_masks_available"], "accuracy_metrics_available": official["accuracy_metrics_available"], "metrics": None, "limitation": official["ground_truth_limitation"]},
        "reproducibility": read("drive_reproducibility.json"),
        "robustness": {"status": read("drive_robustness.json")["status"], "sample_image_ids": read("drive_robustness.json")["sample_image_ids"], "perturbations": read("drive_robustness.json")["perturbations"]},
        "reasons_for_further_research": ["Five-fold development Dice and IoU for the new candidate are materially below the protected R2-V2 reference artifact.", "The official test split lacks manual vessel masks, so external segmentation accuracy cannot be measured.", "The new model is random-initialized and trained on only 20 labeled DRIVE images; its output is research evidence only."],
        "official_test_used_for_model_selection": False,
        "clinical_validation_claim": False,
    }
    dump(META / "drive_final_report.json", final_report)
    registry_path = ROOT / "ml" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["artifacts"] = [artifact for artifact in registry.get("artifacts", []) if artifact.get("model_version") != model_manifest["model_version"]]
    registry["artifacts"].append({"artifact_kind": "EXPERIMENTAL", "artifact_status": "MODEL_FROZEN_AND_EVALUATED", "availability_status": "MODEL_AVAILABLE", "checkpoint": model_manifest["checkpoint"], "checkpoint_sha256": actual_hash, "model_name": "DRIVE research lightweight U-Net vessel segmentor", "model_type": "binary_vessel_segmentation", "model_version": model_manifest["model_version"], "architecture": model_manifest["architecture"], "dataset_version": manifest["dataset_version"], "model_config": {"input_size": model_manifest["input_size"], "preprocessing": model_manifest["preprocessing"], "threshold": model_manifest["threshold"], "threshold_selection": model_manifest["threshold_selection"], "loss": model_manifest["loss"]}, "evaluation": {"development_cv": model_manifest["cv_metrics"], "official_test": {"images_opened": official["official_test_images_opened"], "manual_vessel_masks_available": 0, "metrics": None, "status": "INFERENCE_AUDIT_ONLY"}}, "production_promoted": False, "clinical_validation_claim": False, "note": "Research-only DRIVE candidate. Protected R2-V2 remains the default vessel model; official test accuracy is unavailable because manual test masks are not present."})
    registry["registry_version"] = registry.get("registry_version", "1.0.0")
    dump(registry_path, registry)
    print(json.dumps({"status": final_report["status"], "checkpoint_sha256": actual_hash, "registry_artifacts": len(registry["artifacts"]), "official_test_metrics": None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
