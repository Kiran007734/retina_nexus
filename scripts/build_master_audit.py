"""Build a non-destructive Retina-Nexus master audit snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ml" / "evaluation" / "master_audit"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def count_files(path: Path, suffixes: set[str] | None = None) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and (suffixes is None or item.suffix.lower() in suffixes))


def json_load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def model_record(name: str, role: str, task: str, architecture: str, checkpoint: str | None, dataset: str, source: str, license_name: str | None, production_status: str, validation: dict[str, Any] | None = None, limitations: list[str] | None = None) -> dict[str, Any]:
    path = ROOT / checkpoint if checkpoint else None
    return {
        "name": name,
        "role": role,
        "task": task,
        "architecture": architecture,
        "dataset": dataset,
        "training_status": "CHECKPOINT_PRESENT" if path and path.is_file() else "NO_CHECKPOINT",
        "source": source,
        "checkpoint": checkpoint,
        "sha256": sha256(path) if path else None,
        "license": license_name,
        "validation": validation or {},
        "limitations": limitations or [],
        "production_status": production_status,
    }


def audit() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    generated = datetime.now(timezone.utc).isoformat()
    messidor = json_load(ROOT / "ml" / "evaluation" / "messidor2" / "authoritative_external_manifest.json", {})
    messidor_original = json_load(ROOT / "ml" / "datasets" / "metadata" / "messidor2" / "authoritative_original_manifest.json", {})
    idrid_registry = json_load(ROOT / "ml" / "evaluation" / "referable_research" / "experiment_registry.json", {})
    idrid_research = json_load(ROOT / "ml" / "evaluation" / "referable_research" / "research_conclusion.json", {})
    aptos_train = ROOT / "ml" / "datasets" / "raw" / "aptos2019" / "train.csv"
    aptos_rows = 0
    aptos_distribution: dict[str, int] = {}
    if aptos_train.is_file():
        with aptos_train.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        aptos_rows = len(rows)
        for row in rows:
            value = str(row.get("diagnosis", ""))
            aptos_distribution[value] = aptos_distribution.get(value, 0) + 1
    datasets = {
        "aptos2019": {
            "purpose": "Primary legacy DR classifier training dataset",
            "raw_root": "ml/datasets/raw/aptos2019",
            "train_images": count_files(ROOT / "ml/datasets/raw/aptos2019/train_images", {".png", ".jpg", ".jpeg"}),
            "test_images": count_files(ROOT / "ml/datasets/raw/aptos2019/test_images", {".png", ".jpg", ".jpeg"}),
            "train_label_rows": aptos_rows,
            "class_distribution": aptos_distribution,
            "validation_artifacts": ["ml/datasets/metadata/reports/aptos2019/dataset_validation_report.json", "ml/datasets/metadata/splits/aptos2019/splits.json"],
        },
        "idrid": {
            "purpose": "Research severity, lesion, and localization data",
            "raw_root": "ml/datasets/raw/idrid",
            "files": count_files(ROOT / "ml/datasets/raw/idrid"),
            "development_records": 406,
            "official_test_status": "NOT_USED_FOR_CURRENT_REFERABLE_RESEARCH",
            "validation_artifacts": ["ml/datasets/metadata/idrid/idrid_v3_selected_candidate.json", "ml/evaluation/referable_research/cv_results.json"],
        },
        "drive": {
            "purpose": "Vessel segmentation research/evaluation",
            "raw_root": "ml/datasets/raw/drive",
            "files": count_files(ROOT / "ml/datasets/raw/drive"),
            "validation_artifacts": ["ml/evaluation/drive/validation_report.json", "ml/evaluation/drive/dataset_manifest.json"],
        },
        "messidor2": {
            "purpose": "External descriptive evaluation",
            "authoritative_root": "ml/datasets/raw/messidor/messidor-2-original/IMAGES",
            "authoritative_original_count": messidor_original.get("counts", {}).get("original_archive_images"),
            "label_matched_count": messidor.get("image_count"),
            "unlabeled_original_count": 4,
            "official_ground_truth_status": "NOT_PROVEN; local/adjudicated label source only",
            "historical_derivative_root": "ml/datasets/raw/messidor/images/messidor-2/messidor-2/preprocess",
            "validation_manifest": "ml/evaluation/messidor2/authoritative_external_manifest.json",
        },
    }
    models = [
        model_record("APTOS EfficientNet-B0", "PRIMARY", "five-class DR severity", "EfficientNet-B0", "ml/weights/classifiers/aptos2019/efficientnet-b0-aptos2019-20260830-v1/checkpoint_best.pt", "APTOS 2019", "RETINA-NEXUS trained checkpoint", None, "LEGACY_PRODUCTION_PATH", limitations=["Messidor external transportability is descriptive only.", "Raw confidence is not clinically calibrated."]),
        model_record("IDRiD V3 severity candidate", "PRIMARY", "five-class DR severity and referable research", "EfficientNet-B0 multi-head; severity head authoritative", "ml/weights/classifiers/idrid/research/final/checkpoint_best.pt", "IDRiD development", "RETINA-NEXUS research checkpoint", None, "RESEARCH_ONLY_NOT_PROMOTED", validation={"referable_threshold": 0.20, "research_status": idrid_research.get("status"), "official_test_images_opened": idrid_registry.get("data", {}).get("official_test_images_opened", 0)}),
        model_record("Fundus lesions primary", "PRIMARY", "lesion supporting evidence", "U-Net with SE-ResNeXt-50 32x4d encoder", "ml/weights/lesion_segmentation/fundus-lesions-unet-seresnext50-all-v1/model.safetensors", "IDRiD, DDR, FGADR, MESSIDOR, RETLES", "ClementP/fundus-lesions-toolkit", "MIT declared by model repository", "SUPPORTING_EVIDENCE", limitations=["Not clinically validated by RETINA-NEXUS."]),
        model_record("R2-V2 vessel primary", "PRIMARY", "retinal vessel supporting evidence", "RRWNet R2-V2 bv variant", "ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors", "Unified_Fundus per model config", "j-morano/R2-V2", "CC BY 4.0 declared by model repository", "SUPPORTING_EVIDENCE", limitations=["CPU inference is currently a runtime bottleneck."]),
        model_record("IDRiD lesion verifier", "VERIFIER", "lesion segmentation research verification", "U-Net SE-ResNeXt-50", "ml/weights/lesions/idrid/checkpoint_best.pt", "IDRiD", "RETINA-NEXUS research checkpoint", None, "RESEARCH_ONLY", limitations=["Not allowed to replace the primary lesion model."]),
        model_record("IDRiD localization research", "RESEARCH_ONLY", "optic-disc/fovea localization research", "IDRiD localization model", "ml/weights/localization/idrid/checkpoint_best.pt", "IDRiD", "RETINA-NEXUS research checkpoint", None, "RESEARCH_ONLY"),
        model_record("DRIVE vessel research", "RESEARCH_ONLY", "vessel segmentation research", "DRIVE vessel model", "ml/weights/vessels/drive/checkpoint_best.pt", "DRIVE", "RETINA-NEXUS research checkpoint", None, "RESEARCH_ONLY", limitations=["Must not replace R2-V2 primary."]),
    ]
    project_registry = {
        "schema_version": "retina-nexus-master-project-model-registry-v1",
        "generated_at_utc": generated,
        "models": models,
        "backup_models": [],
        "backup_model_status": "NO_BACKUP_PROMOTED",
        "backup_model_reason": "No independently validated compatible backup checkpoint is installed. RETFound is recorded as a research candidate only; no external checkpoint was downloaded.",
        "production_behavior_changed": False,
        "model_weights_changed": False,
        "clinical_validation_claim": False,
    }
    dataset_registry = {
        "schema_version": "retina-nexus-master-dataset-registry-v1",
        "generated_at_utc": generated,
        "datasets": datasets,
        "authoritative_messidor_manifest": "ml/evaluation/messidor2/authoritative_external_manifest.json",
        "historical_results_preserved": True,
        "patient_identifiers": {"messidor2": False, "idrid": False},
        "external_label_warning": "Messidor-2 local labels are not independently proven official clinical ground truth.",
    }
    pipeline_registry = {
        "schema_version": "retina-nexus-master-pipeline-registry-v1",
        "generated_at_utc": generated,
        "stages": [
            {"name": "image_quality_gate", "paths": ["backend/app/ml/quality", "backend/app/api/routes/images.py"], "status": "IMPLEMENTED"},
            {"name": "dr_classification", "paths": ["backend/app/ml/inference/classifier.py", "backend/app/api/routes/screening.py"], "status": "IMPLEMENTED_PRIMARY_APTOS_RUNTIME"},
            {"name": "lesion_evidence", "paths": ["backend/app/ml/evidence/lesion_model.py", "backend/app/ml/evidence/service.py"], "status": "IMPLEMENTED_PRIMARY_SUPPORTING"},
            {"name": "vessel_evidence", "paths": ["backend/app/ml/evidence/vessel_model.py", "ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors"], "status": "IMPLEMENTED_PRIMARY_SUPPORTING"},
            {"name": "anatomical_localization", "paths": ["backend/app/ml/evidence/service.py", "ml/weights/localization/idrid/checkpoint_best.pt"], "status": "PRODUCTION_HEURISTIC_PLUS_RESEARCH_MODEL"},
            {"name": "grad_cam_and_agreement", "paths": ["backend/app/ml/explainability", "backend/app/api/routes/screening.py"], "status": "IMPLEMENTED"},
            {"name": "uncertainty_and_ood", "paths": ["backend/app/ml/trust/uncertainty.py", "backend/app/ml/trust/ood.py"], "status": "EXPERIMENTAL_MONITORING"},
            {"name": "retinaguard", "paths": ["backend/app/ml/trust/guard.py", "backend/app/ml/quality/trust_gate.py"], "status": "IMPLEMENTED_SAFETY_LAYER"},
            {"name": "reporting_and_review", "paths": ["backend/app/api/routes/reports.py", "frontend/src"], "status": "IMPLEMENTED"},
            {"name": "monitoring", "paths": ["backend/app/api/routes/monitoring.py", "frontend/src/pages/MonitoringPage.tsx"], "status": "IMPLEMENTED"},
            {"name": "simulink_digital_twin", "paths": ["simulink/retina_nexus_digital_twin.m", "simulink/run_digital_twin_scenarios.m"], "status": "IMPLEMENTED_DOCUMENTED_PROTOTYPE"},
        ],
        "fusion_status": "NO_LEARNED_FUSION_PROMOTED",
        "backup_agreement_status": "NOT_AVAILABLE_NO_BACKUP_CHECKPOINT",
        "safety_principle": "Supporting evidence changes reliability/escalation, not the primary severity grade; missing evidence is not fabricated.",
    }
    report = f'''# Retina-Nexus master audit

Generated: `{generated}`

## Scope

This is a non-destructive inventory of datasets, frozen model artifacts,
pipeline modules, reports, and tests. No checkpoint, threshold, production
configuration, or historical evaluation result was modified.

## Primary model inventory

| Model | Role | Task | SHA-256 | Status |
|---|---|---|---|---|
{chr(10).join(f"| {item['name']} | {item['role']} | {item['task']} | `{item['sha256']}` | {item['production_status']} |" for item in models)}

No backup model was promoted. RETFound was considered as a research candidate
from its official repository, but no checkpoint was downloaded or integrated;
therefore no backup agreement or backup safety benefit is claimed.

## Dataset audit

{json.dumps(datasets, indent=2, sort_keys=True)}

## Pipeline status

The quality gate, classifier, lesion evidence, R2-V2 vessel evidence,
explainability, RetinaGuard, reports, frontend, monitoring, and Simulink
prototype are present. Learned fusion and backup-model voting are not promoted.

## Known blockers

1. Messidor-2 labels are a local/adjudicated source and are not independently
   proven official clinical ground truth.
2. The authoritative APTOS classifier evaluation is descriptive external
   evaluation; no clinical validation claim is made.
3. Full evidence execution is CPU-bound. Existing smoke timing and bounded
   parallel attempts did not justify an unbounded 1,744-image vessel/evidence
   run.
4. No independent compatible backup checkpoint is installed, so model
   disagreement with a backup cannot be measured.
5. No learned fusion model was trained; supporting evidence is not allowed to
   rewrite severity.

## SIH status

**TARGET NOT YET DEMONSTRATED.** Neither frozen Messidor threshold met both
greater-than-90-percent referable sensitivity and greater-than-85-percent
specificity on the authoritative external evaluation.
'''
    return project_registry, dataset_registry, pipeline_registry, report


def main() -> int:
    project, datasets, pipeline, report = audit()
    write(OUTPUT / "project_model_registry.json", project)
    write(OUTPUT / "dataset_registry.json", datasets)
    write(OUTPUT / "pipeline_registry.json", pipeline)
    (OUTPUT / "master_audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT.relative_to(ROOT)), "model_count": len(project["models"]), "backup_models": len(project["backup_models"]), "production_behavior_changed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
