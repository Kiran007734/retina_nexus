"""Append/update the frozen IDRiD localization research artifact in registry."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "ml" / "model_registry.json"
MODEL = ROOT / "ml" / "weights" / "localization" / "idrid" / "model_manifest.json"
REPORT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_localization_final_report.json"


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    artifact = {
        "artifact_kind": "EXPERIMENTAL",
        "artifact_status": "MODEL_FROZEN_AND_EVALUATED",
        "availability_status": "MODEL_AVAILABLE",
        "checkpoint": model["checkpoint"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "clinical_validation_claim": False,
        "dataset_version": "IDRiD-C-Localization-412-development-images-leak-safe",
        "evaluation": {
            "development_cv": report["cv"]["summary"],
            "official_test": report["official_test"]["metrics"],
            "robustness": report["robustness"],
            "reproducibility": report["reproducibility"],
            "status": report["status"],
            "clinical_validation_claim": False,
        },
        "model_config": {
            "architecture": model["architecture"],
            "landmarks": {str(index): name for index, name in enumerate(model["landmarks"])},
            "input_resolution": model["input_resolution"],
            "heatmap_resolution": model["heatmap_resolution"],
            "prediction": model["prediction"],
        },
        "model_name": "IDRiD optic-disc and fovea research localizer",
        "model_type": "anatomical_landmark_localization",
        "model_version": model["model_version"],
        "note": "Research supporting evidence only. The model does not alter DR classification or RetinaGuard and is not production promoted.",
        "training_config": model["training"],
        "production_promoted": False,
        "official_test_images_opened": 103,
    }
    artifacts = [item for item in registry.get("artifacts", []) if item.get("model_type") != artifact["model_type"]]
    artifacts.append(artifact)
    registry["artifacts"] = artifacts
    registry["registry_version"] = registry.get("registry_version", "1.0.0")
    REGISTRY.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model_version": artifact["model_version"], "checkpoint_sha256": artifact["checkpoint_sha256"], "production_promoted": False, "status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
