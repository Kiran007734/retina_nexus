# Retina-Nexus master audit

Generated: `2026-09-14T04:46:58.110253+00:00`

## Scope

This is a non-destructive inventory of datasets, frozen model artifacts,
pipeline modules, reports, and tests. No checkpoint, threshold, production
configuration, or historical evaluation result was modified.

## Primary model inventory

| Model | Role | Task | SHA-256 | Status |
|---|---|---|---|---|
| APTOS EfficientNet-B0 | PRIMARY | five-class DR severity | `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b` | LEGACY_PRODUCTION_PATH |
| IDRiD V3 severity candidate | PRIMARY | five-class DR severity and referable research | `97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049` | RESEARCH_ONLY_NOT_PROMOTED |
| Fundus lesions primary | PRIMARY | lesion supporting evidence | `a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2` | SUPPORTING_EVIDENCE |
| R2-V2 vessel primary | PRIMARY | retinal vessel supporting evidence | `ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a` | SUPPORTING_EVIDENCE |
| IDRiD lesion verifier | VERIFIER | lesion segmentation research verification | `8f3c64a7aae23318f08bb34199aee079b16ff155db92912e97543aa670246a2c` | RESEARCH_ONLY |
| IDRiD localization research | RESEARCH_ONLY | optic-disc/fovea localization research | `91d49e43c5ea87fdc4adf36f9687ba7f07eeac546f89f09e1203ab56b083d569` | RESEARCH_ONLY |
| DRIVE vessel research | RESEARCH_ONLY | vessel segmentation research | `f199543db550f7cd84548cb0e2ab2cf7efbad99ffb41af8649b249d4433a3133` | RESEARCH_ONLY |

No backup model was promoted. RETFound was considered as a research candidate
from its official repository, but no checkpoint was downloaded or integrated;
therefore no backup agreement or backup safety benefit is claimed.

## Dataset audit

{
  "aptos2019": {
    "class_distribution": {
      "0": 1805,
      "1": 370,
      "2": 999,
      "3": 193,
      "4": 295
    },
    "purpose": "Primary legacy DR classifier training dataset",
    "raw_root": "ml/datasets/raw/aptos2019",
    "test_images": 1928,
    "train_images": 3662,
    "train_label_rows": 3662,
    "validation_artifacts": [
      "ml/datasets/metadata/reports/aptos2019/dataset_validation_report.json",
      "ml/datasets/metadata/splits/aptos2019/splits.json"
    ]
  },
  "drive": {
    "files": 101,
    "purpose": "Vessel segmentation research/evaluation",
    "raw_root": "ml/datasets/raw/drive",
    "validation_artifacts": [
      "ml/evaluation/drive/validation_report.json",
      "ml/evaluation/drive/dataset_manifest.json"
    ]
  },
  "idrid": {
    "development_records": 406,
    "files": 1488,
    "official_test_status": "NOT_USED_FOR_CURRENT_REFERABLE_RESEARCH",
    "purpose": "Research severity, lesion, and localization data",
    "raw_root": "ml/datasets/raw/idrid",
    "validation_artifacts": [
      "ml/datasets/metadata/idrid/idrid_v3_selected_candidate.json",
      "ml/evaluation/referable_research/cv_results.json"
    ]
  },
  "messidor2": {
    "authoritative_original_count": 1748,
    "authoritative_root": "ml/datasets/raw/messidor/messidor-2-original/IMAGES",
    "historical_derivative_root": "ml/datasets/raw/messidor/images/messidor-2/messidor-2/preprocess",
    "label_matched_count": 1744,
    "official_ground_truth_status": "NOT_PROVEN; local/adjudicated label source only",
    "purpose": "External descriptive evaluation",
    "unlabeled_original_count": 4,
    "validation_manifest": "ml/evaluation/messidor2/authoritative_external_manifest.json"
  }
}

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
