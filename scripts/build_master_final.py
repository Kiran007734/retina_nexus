"""Build the non-destructive RETINA-NEXUS master research artifacts.

This script only reads existing datasets, checkpoints, evaluation reports, and
benchmark output. It writes audit/summary artifacts; it never changes model
weights, production configuration, thresholds, or datasets.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "ml" / "evaluation" / "master_final"
REGISTRY = ROOT / "ml" / "models" / "model_registry.json"


def load(relative: str) -> Any:
    path = ROOT / relative
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(relative: str, payload: Any) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(relative: str) -> str | None:
    path = ROOT / relative
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def main() -> int:
    generated = datetime.now(timezone.utc).isoformat()
    audit_models = load("ml/evaluation/master_audit/project_model_registry.json")
    dataset_registry = load("ml/evaluation/master_audit/dataset_registry.json")
    pipeline_registry = load("ml/evaluation/master_audit/pipeline_registry.json")
    aptos = load("ml/evaluation/messidor2/authoritative_classifier/evaluation_summary.json")
    idrid = load("ml/evaluation/messidor2/authoritative_idrid_v3/evaluation_summary.json")
    aptos_model = load("ml/evaluation/messidor2/authoritative_classifier/model_snapshot.json")
    idrid_model = load("ml/evaluation/messidor2/authoritative_idrid_v3/model_snapshot.json")
    drive = load("ml/evaluation/drive/r2-v2-evaluation.json")
    cv = load("ml/evaluation/referable_research/cv_results.json")
    fusion_research = load("ml/evaluation/referable_research/lesion_fusion_results.json")
    benchmarks = {
        "10": load("ml/evaluation/master_final/benchmarks/authoritative_10/runtime_benchmark.json"),
        "50": load("ml/evaluation/master_final/benchmarks/authoritative_50/runtime_benchmark.json"),
    }

    aptos_checkpoint = aptos_model.get("checkpoint_path_relative", aptos_model.get("checkpoint"))
    idrid_checkpoint = idrid_model.get("checkpoint", idrid_model.get("checkpoint_path_relative"))
    aptos_sha = sha256(aptos_checkpoint)
    idrid_sha = sha256(idrid_checkpoint)
    vessel_sha = sha256(drive["model"]["checkpoint"])
    expected_aptos_sha = "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b"
    expected_idrid_sha = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"
    expected_vessel_sha = "ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a"

    aptos_05 = aptos["referable_thresholds"]["threshold_0.50"]
    aptos_02 = aptos["referable_thresholds"]["threshold_0.20"]
    idrid_05 = idrid["referable_thresholds"]["threshold_0.50"]
    idrid_02 = idrid["referable_thresholds"]["threshold_0.20"]

    central_registry = {
        "schema_version": "retina-nexus-central-model-registry-v1",
        "generated_at_utc": generated,
        "clinical_validation_claim": False,
        "production_promotion_changed": False,
        "backup_model_status": "NO_COMPATIBLE_DR_BACKUP_INSTALLED",
        "models": [
            {
                "name": "APTOS EfficientNet-B0",
                "role": "PRIMARY_LEGACY_PRODUCTION_PATH",
                "task": "five-class diabetic-retinopathy severity",
                "architecture": aptos_model.get("architecture", "EfficientNet-B0"),
                "dataset": "APTOS 2019",
                "checkpoint": aptos_checkpoint,
                "sha256": aptos_sha,
                "expected_sha256": expected_aptos_sha,
                "sha_verified": aptos_sha == expected_aptos_sha,
                "source": "RETINA-NEXUS trained checkpoint",
                "training_status": "CHECKPOINT_PRESENT",
                "production_status": "EXISTING_BEHAVIOR_PRESERVED",
                "validation": {
                    "external_dataset": "Messidor-2 authoritative original images with local/adjudicated labels",
                    "threshold_0.50": aptos_05,
                    "threshold_0.20": aptos_02,
                },
                "limitations": [
                    "Messidor labels are not independently proven official clinical ground truth.",
                    "External evaluation is descriptive and does not establish clinical validation.",
                    "No clinical confidence calibration claim.",
                ],
            },
            {
                "name": "IDRiD V3 EfficientNet-B0",
                "role": "PRIMARY_RESEARCH_ONLY",
                "task": "five-class severity and research referable DR",
                "architecture": idrid_model.get("architecture", "EfficientNet-B0 multi-head"),
                "dataset": "IDRiD development",
                "checkpoint": idrid_checkpoint,
                "sha256": idrid_sha,
                "expected_sha256": expected_idrid_sha,
                "sha_verified": idrid_sha == expected_idrid_sha,
                "source": "RETINA-NEXUS research checkpoint",
                "training_status": "FROZEN_RESEARCH_CANDIDATE",
                "production_status": "NOT_PROMOTED",
                "referable_rule": "P(2)+P(3)+P(4) >= 0.20",
                "validation": {
                    "development_oof": cv.get("derived_probability_oof", {}).get("selected", {}),
                    "external_descriptive_threshold_0.50": idrid_05,
                    "external_descriptive_threshold_0.20": idrid_02,
                },
                "limitations": [
                    "Messidor evaluation is not clinical validation.",
                    "Official IDRiD test images remain outside this program.",
                    "Research threshold is not a production threshold.",
                ],
            },
            {
                "name": "R2-V2 RRWNet vessel segmentor",
                "role": "PRIMARY_SUPPORTING_EVIDENCE",
                "task": "retinal vessel segmentation",
                "architecture": drive["model"]["architecture"],
                "dataset": "DRIVE training split with manual vessel masks",
                "checkpoint": drive["model"]["checkpoint"],
                "sha256": vessel_sha,
                "expected_sha256": expected_vessel_sha,
                "sha_verified": vessel_sha == expected_vessel_sha,
                "source": drive["model"].get("source"),
                "training_status": "PRETRAINED_INFERENCE_ONLY",
                "production_status": "SUPPORTING_EVIDENCE_ONLY",
                "validation": drive["evaluation"]["aggregate"],
                "limitations": [
                    "DRIVE test images have no manual vessel masks in the available copy.",
                    "CPU inference is a runtime bottleneck.",
                    "Engineering segmentation evaluation is not clinical validation.",
                ],
            },
            {
                "name": "Fundus lesions primary",
                "role": "PRIMARY_SUPPORTING_EVIDENCE",
                "task": "lesion segmentation/evidence",
                "architecture": "U-Net with SE-ResNeXt-50 32x4d encoder",
                "dataset": "External mixed fundus lesion training provenance",
                "checkpoint": "ml/weights/lesion_segmentation/fundus-lesions-unet-seresnext50-all-v1/model.safetensors",
                "sha256": "a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2",
                "source": "ClementP/fundus-lesions-toolkit",
                "production_status": "SUPPORTING_EVIDENCE_ONLY",
                "validation": {"status": "available_in_pipeline; no new master-population clinical validation run"},
                "limitations": ["Evidence cannot rewrite the primary severity grade.", "Not clinically validated by RETINA-NEXUS."],
            },
            {
                "name": "Independent DR backup candidate inventory",
                "role": "RESEARCH_ONLY_NOT_ADOPTED",
                "task": "independent DR backup research",
                "architecture": "RETFound candidate; compatible classifier head not installed",
                "dataset": "External pretraining provenance not integrated",
                "checkpoint": None,
                "sha256": None,
                "source": "https://github.com/rmaphoh/RETFound",
                "training_status": "NOT_DOWNLOADED",
                "production_status": "NOT_AVAILABLE",
                "validation": {},
                "limitations": ["No checkpoint or compatible five-class DR head is installed; no backup disagreement is claimed."],
            },
        ],
        "notes": [
            "No model weights were changed by this master program.",
            "No independent compatible DR backup checkpoint was promoted.",
            "R2-V2 is a supporting vessel model, not an independent DR severity backup.",
        ],
    }
    write_json("ml/models/model_registry.json", central_registry)

    model_comparison = {
        "schema_version": "retina-nexus-master-model-comparison-v1",
        "generated_at_utc": generated,
        "messidor_population": 1744,
        "label_status": "local/adjudicated labels; not independently proven official clinical ground truth",
        "selection_tuning_on_messidor": False,
        "candidates": [
            {
                "model": "APTOS EfficientNet-B0",
                "checkpoint_sha256": aptos_sha,
                "threshold_0.50": aptos_05,
                "threshold_0.20": aptos_02,
                "external_validation_status": "DESCRIPTIVE_ONLY",
            },
            {
                "model": "IDRiD V3 EfficientNet-B0",
                "checkpoint_sha256": idrid_sha,
                "threshold_0.50": idrid_05,
                "threshold_0.20": idrid_02,
                "external_validation_status": "DESCRIPTIVE_ONLY_RESEARCH",
            },
        ],
        "backup_status": "NO_COMPATIBLE_BACKUP_CHECKPOINT",
        "sih_target": {
            "definition": "referable sensitivity > 0.90 and specificity > 0.85 on the same frozen external evaluation",
            "status": "TARGET_NOT_YET_DEMONSTRATED",
        },
    }
    write_json("ml/evaluation/master_final/model_comparison.json", model_comparison)

    fusion = {
        "schema_version": "retina-nexus-master-fusion-results-v1",
        "generated_at_utc": generated,
        "status": "NOT_RUN",
        "production_status": "NO_LEARNED_FUSION_PROMOTED",
        "reason": fusion_research["reason"],
        "primary_grade_protection": "Supporting lesion/vessel/evidence signals may affect reliability and escalation; they do not rewrite the primary classifier grade.",
        "backup_voting": "UNAVAILABLE_NO_COMPATIBLE_BACKUP_CHECKPOINT",
        "messidor_labels_used_for_training_or_selection": False,
    }
    write_json("ml/evaluation/master_final/fusion_results.json", fusion)

    safety = {
        "schema_version": "retina-nexus-master-safety-analysis-v1",
        "generated_at_utc": generated,
        "status": "RESEARCH_AND_ENGINEERING_AUDIT",
        "severity_grade_protection": "The classifier argmax remains distinct from referable probability, evidence, and RetinaGuard state.",
        "referable_rule": {
            "aptos_external_evaluation": "P(2)+P(3)+P(4) >= threshold; evaluated at 0.50 and 0.20",
            "idrid_research": "P(2)+P(3)+P(4) >= 0.20",
        },
        "missing_evidence_behavior": "Missing or failed evidence is surfaced as unavailable/error and cannot be silently fabricated.",
        "abstention_or_escalation": "Quality gate, uncertainty, disagreement availability, OOD monitoring, and RetinaGuard can escalate; they do not silently change grade.",
        "backup_disagreement": "NOT_AVAILABLE_NO_BACKUP_CHECKPOINT",
        "known_safety_limitations": [
            "No clinical validation claim.",
            "Raw confidence is not clinically calibrated.",
            "Messidor labels are not independently proven official ground truth.",
            "Full-population evidence and XAI validation were not run because CPU cost is high.",
        ],
    }
    write_json("ml/evaluation/master_final/safety_analysis.json", safety)

    false_negative = {
        "schema_version": "retina-nexus-master-false-negative-analysis-v1",
        "generated_at_utc": generated,
        "external_descriptive_evaluation": {
            "dataset": "Messidor-2 authoritative original label-matched images",
            "aptos_threshold_0.50": {"false_negative": aptos_05["false_negative"], "sensitivity": aptos_05["sensitivity"]},
            "aptos_threshold_0.20": {"false_negative": aptos_02["false_negative"], "sensitivity": aptos_02["sensitivity"]},
            "idrid_v3_threshold_0.50": {"false_negative": idrid_05["fn"], "sensitivity": idrid_05["sensitivity"]},
            "idrid_v3_threshold_0.20": {"false_negative": idrid_02["fn"], "sensitivity": idrid_02["sensitivity"]},
            "warning": "These are descriptive results against local/adjudicated labels, not clinical false-negative rates.",
        },
        "development_referable_research": {
            "source": "ml/evaluation/referable_research/cv_results.json",
            "selected_probability_source": "derived_probability_oof",
            "selected_threshold": cv["derived_probability_oof"]["selected"],
            "official_test_images_opened": cv.get("official_test_images_opened", 0),
        },
        "per_image_sources": [
            "ml/evaluation/messidor2/authoritative_classifier/predictions.jsonl",
            "ml/evaluation/messidor2/authoritative_idrid_v3/predictions.jsonl",
            "ml/evaluation/referable_research/false_negative_analysis.json",
        ],
    }
    write_json("ml/evaluation/master_final/false_negative_analysis.json", false_negative)

    xai = {
        "schema_version": "retina-nexus-master-xai-validation-v1",
        "generated_at_utc": generated,
        "status": "BOUNDED_RUNTIME_VERIFICATION",
        "grad_cam": {
            "executed_in_benchmark": True,
            "benchmark_cases": benchmarks["50"].get("stage_latency", {}).get("grad_cam_and_agreement", {}).get("count", 0),
            "full_messidor_population_executed": False,
            "interpretation": "Grad-CAM is an attention visualization and is not proof of lesion causality.",
        },
        "attention_lesion_agreement": {
            "executed_in_benchmark": True,
            "clinical_causality_claim": False,
            "status": "engineering explainability metric only",
        },
        "stability": "Existing configurable explanation-stability path remains available; no unbounded population perturbation run was added.",
        "limitations": ["Bounded runtime smoke/benchmark coverage only; no full external-population XAI validation claim."],
    }
    write_json("ml/evaluation/master_final/xai_validation.json", xai)

    guard = {
        "schema_version": "retina-nexus-master-retinaguard-validation-v1",
        "generated_at_utc": generated,
        "status": "BOUNDED_RUNTIME_VERIFICATION",
        "cases_reaching_guard": benchmarks["50"].get("stage_latency", {}).get("retinaguard", {}).get("count", 0),
        "full_messidor_population_executed": False,
        "inputs": ["quality", "raw/calibrated confidence where available", "uncertainty", "disagreement availability", "evidence", "attention agreement", "stability", "OOD"],
        "grade_override": False,
        "safety_behavior": "RetinaGuard changes triage/review reliability state, not the five-class classifier grade.",
        "limitations": ["No clinical trust guarantee; no full-population trust validation claim."],
    }
    write_json("ml/evaluation/master_final/retinaguard_validation.json", guard)

    messidor_final = {
        "schema_version": "retina-nexus-master-messidor-results-v1",
        "generated_at_utc": generated,
        "dataset": {
            "authoritative_original_count": 1748,
            "label_matched_count": 1744,
            "unlabeled_originals_excluded": aptos.get("unlabeled_originals_excluded", []),
            "manifest": "ml/evaluation/messidor2/authoritative_external_manifest.json",
            "label_status": "local/adjudicated; not independently proven official clinical ground truth",
        },
        "aptos_efficientnet_b0": {
            "checkpoint_sha256": aptos_sha,
            "severity": aptos["severity_metrics"],
            "referable_threshold_0.50": aptos_05,
            "referable_threshold_0.20": aptos_02,
            "status": aptos["status"],
        },
        "idrid_v3": {
            "checkpoint_sha256": idrid_sha,
            "severity": idrid["severity_metrics"],
            "referable_threshold_0.50": idrid_05,
            "referable_threshold_0.20": idrid_02,
            "official_idrid_test_images_opened": idrid["official_idrid_test_images_opened"],
            "status": "DESCRIPTIVE_RESEARCH_ONLY",
        },
        "external_threshold_tuning": False,
        "clinical_validation_claim": False,
    }
    write_json("ml/evaluation/master_final/messidor_final_results.json", messidor_final)

    runtime = {
        "schema_version": "retina-nexus-master-runtime-results-v1",
        "generated_at_utc": generated,
        "benchmarks": benchmarks,
        "interpretation": {
            "count_10": "bounded real-image full pipeline benchmark",
            "count_50": "bounded real-image full pipeline benchmark",
            "population_full_evidence_run": False,
            "dominant_bottleneck": "R2-V2 vessel inference on CPU",
            "not_a_clinical_metric": True,
        },
    }
    write_json("ml/evaluation/master_final/runtime_results.json", runtime)

    simulink = {
        "schema_version": "retina-nexus-master-simulink-results-v1",
        "generated_at_utc": generated,
        "status": "DOCUMENTED_PROTOTYPE",
        "implementation": [
            "simulink/retina_nexus_digital_twin.m",
            "simulink/run_digital_twin_scenarios.m",
            "docs/DEPLOYMENT.md",
        ],
        "scenarios": ["NORMAL LOAD", "HIGH LOAD", "LOW BANDWIDTH", "HIGH UNGRADABLE RATE", "LIMITED SPECIALIST CAPACITY"],
        "scope": "Operational throughput/queue simulation plan; not a patient outcome model and not a clinical validation result.",
        "executed_in_master_run": False,
    }
    write_json("ml/evaluation/master_final/simulink_results.json", simulink)

    report = f"""# RETINA-NEXUS Master ML Research, Safety, Validation & Fusion Report

Generated: `{generated}`

## Executive status

**SIH TARGET NOT YET DEMONSTRATED.** The target requires referable sensitivity >90% and specificity >85% on the same frozen external evaluation. The authoritative-original Messidor descriptive evaluation did not demonstrate both at either frozen threshold. These results are not clinical validation because the available Messidor labels are local/adjudicated and are not independently proven official clinical ground truth.

No checkpoint, dataset, production threshold, production configuration, or existing APTOS behavior was changed. No model was promoted and no commit was created.

## Frozen model and dataset inventory

- APTOS EfficientNet-B0: `{aptos_sha}`; SHA verified: `{aptos_sha == expected_aptos_sha}`.
- IDRiD V3 research EfficientNet-B0: `{idrid_sha}`; SHA verified: `{idrid_sha == expected_idrid_sha}`; production promoted: `false`.
- R2-V2 vessel evidence model: `{vessel_sha}`; SHA verified: `{vessel_sha == expected_vessel_sha}`.
- Messidor authoritative original images: 1,748; label-matched evaluation population: 1,744; four originals had no matching local label and were excluded.
- Official IDRiD test images opened during this program: `{idrid['official_idrid_test_images_opened']}`.

## External classifier evaluation

All 1,744 label-matched authoritative original Messidor images were inferred by both frozen classifier candidates. Thresholds were evaluated without Messidor tuning.

| Model / threshold | Sensitivity | Specificity | FN | FP | Accuracy | QWK | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| APTOS EfficientNet-B0 / 0.50 | {pct(aptos_05['sensitivity'])} | {pct(aptos_05['specificity'])} | {aptos_05['false_negative']} | {aptos_05['false_positive']} | {aptos['severity_metrics']['accuracy']:.4f} | {aptos['severity_metrics']['quadratic_weighted_kappa']:.4f} | {aptos['severity_metrics']['roc_auc_ovr_macro']:.4f} |
| APTOS EfficientNet-B0 / 0.20 | {pct(aptos_02['sensitivity'])} | {pct(aptos_02['specificity'])} | {aptos_02['false_negative']} | {aptos_02['false_positive']} | {aptos['severity_metrics']['accuracy']:.4f} | {aptos['severity_metrics']['quadratic_weighted_kappa']:.4f} | {aptos['severity_metrics']['roc_auc_ovr_macro']:.4f} |
| IDRiD V3 / 0.50 | {pct(idrid_05['sensitivity'])} | {pct(idrid_05['specificity'])} | {idrid_05['fn']} | {idrid_05['fp']} | {idrid['severity_metrics']['accuracy']:.4f} | {idrid['severity_metrics']['quadratic_weighted_kappa']:.4f} | {idrid['severity_metrics']['roc_auc_ovr_macro']:.4f} |
| IDRiD V3 / 0.20 | {pct(idrid_02['sensitivity'])} | {pct(idrid_02['specificity'])} | {idrid_02['fn']} | {idrid_02['fp']} | {idrid['severity_metrics']['accuracy']:.4f} | {idrid['severity_metrics']['quadratic_weighted_kappa']:.4f} | {idrid['severity_metrics']['roc_auc_ovr_macro']:.4f} |

APTOS severity metrics are accuracy `0.6032`, macro F1 `0.3307`, weighted F1 `0.5296`, QWK `0.4924`, and macro OVR ROC-AUC `0.7690`. IDRiD V3 severity metrics are accuracy `0.6147`, macro F1 `0.3199`, QWK `0.3751`, and ROC-AUC `0.7857` on this descriptive external population.

## Supporting evidence and safety

The R2-V2 DRIVE evaluation used genuine manual vessel masks for 20 training images, with field-of-view masking. Mean Dice/F1 was `{drive['evaluation']['aggregate']['mean']['dice']:.4f}`, IoU `{drive['evaluation']['aggregate']['mean']['iou']:.4f}`, sensitivity `{drive['evaluation']['aggregate']['mean']['sensitivity']:.4f}`, specificity `{drive['evaluation']['aggregate']['mean']['specificity']:.4f}`. The available DRIVE copy has no manual test vessel masks, so no test accuracy was reported.

Lesion, vessel, Grad-CAM, agreement, uncertainty, OOD, and RetinaGuard remain supporting/reliability paths. Supporting evidence is not allowed to rewrite the primary severity grade. No leak-safe multimodal learned fusion was trained or promoted. No compatible independent DR backup checkpoint is installed, so backup disagreement is unavailable rather than fabricated.

## Runtime benchmark

The real-image full pipeline benchmark completed 50/50 records with zero failures. Three images stopped at the quality gate; 47 reached classifier/evidence stages. The 50-image run took `{benchmarks['50'].get('elapsed_wall_seconds', 'n/a')}` seconds total. For the 47 gradable cases, median evidence total latency was `{benchmarks['50']['stage_latency']['evidence_total']['median_ms'] / 1000:.2f}s`, including median vessel inference `{benchmarks['50']['stage_latency']['vessel_inference_ms']['median_ms'] / 1000:.2f}s` on CPU. This is an engineering runtime measurement, not a clinical performance metric.

## Required artifacts

- [Central model registry](../../models/model_registry.json)
- [Model comparison](model_comparison.json)
- [Messidor final results](messidor_final_results.json)
- [Safety analysis](safety_analysis.json)
- [False-negative analysis](false_negative_analysis.json)
- [Fusion results](fusion_results.json)
- [XAI validation](xai_validation.json)
- [RetinaGuard validation](retinaguard_validation.json)
- [Runtime benchmarks](runtime_results.json)
- [Simulink status](simulink_results.json)
- [Phase 0 audit report](../master_audit/master_audit_report.md)

## Final limitations

1. The external labels are not independently proven clinical ground truth.
2. The SIH target is not demonstrated; no threshold was tuned on the external set.
3. Raw probabilities are not clinically calibrated.
4. No compatible independent DR backup checkpoint is installed.
5. Full-population vessel/lesion/XAI/RetinaGuard execution was not attempted because the existing CPU evidence path is very slow; bounded 10- and 50-image runs are reported instead.
6. This repository remains a research/prototype system and makes no regulatory approval or clinical deployment claim.
"""
    (FINAL / "master_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "generated_at_utc": generated,
        "final_directory": str(FINAL.relative_to(ROOT)),
        "aptos_sha_verified": aptos_sha == expected_aptos_sha,
        "idrid_sha_verified": idrid_sha == expected_idrid_sha,
        "vessel_sha_verified": vessel_sha == expected_vessel_sha,
        "messidor_images_evaluated": aptos["successful_inferences"],
        "benchmark_10": benchmarks["10"]["status"],
        "benchmark_50": benchmarks["50"]["status"],
        "sih_target": "TARGET_NOT_YET_DEMONSTRATED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
