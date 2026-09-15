# RETINA-NEXUS — Final Messidor-2 External Validation

Final status: **MESSIDOR-2 EXTERNAL EVALUATION BLOCKED**

This report is a frozen, descriptive external model evaluation. It is not clinical validation, regulatory evidence, or a claim that the released labels are official clinical ground truth.

## 1. Evaluation scope and provenance

- Images evaluated by the fresh classifier/current RetinaGuard population pass: **1744**.
- Diagnosis source: `ml/datasets/raw/messidor/images/messidor_data.csv`.
- Label policy: messidor_data.csv is the diagnosis source; messidor-2.csv is pairing-only and is not used for labels.
- Patient identifiers available: `False`; patient-level leakage separation is not claimed.
- Four archive-only images remain excluded because no diagnosis rows were supplied for them.

## 2. Frozen model and configuration

- Classifier checkpoint: `ml/weights/classifiers/aptos2019/efficientnet-b0-aptos2019-20260830-v1/checkpoint_best.pt`.
- Classifier SHA-256: `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`.
- Architecture: `efficientnet_b0`; input `224x224` RGB.
- Preprocessing: deterministic RGB resize to 224x224, tensor conversion, ImageNet mean/std normalization `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`; no dataset-specific crop flag.
- Referable rule: `P(2)+P(3)+P(4) >= 0.5`; severity is independently `argmax(P0..P4)`.
- Primary lesion SHA-256: `a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2`.
- Primary vessel SHA-256: `ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a`.
- RetinaGuard: `retinaguard-v3-graceful-degradation`, unfitted calibration `temperature-scaling-unfitted`.

## 3. Dataset validation and leakage

- Images found: `1744`; label rows: `1744`; matched pairs: `1744`.
- Class distribution: `{"0": 1017, "1": 270, "2": 347, "3": 75, "4": 35}`.
- Existing audit reported 0 corrupt images, 0 missing labels, and no patient IDs. Existing exact duplicate groups are preserved as an audit limitation; no patient-level split claim is made.

## 4. Classifier metrics

- Evaluation population: `1744` successful gradable inferences.
- Accuracy: `0.6238532110091743`; macro F1: `0.34438069917298414`; weighted F1: `0.5182738878127344`.
- ROC-AUC OVR macro: `0.7750565082447286`; QWK: `0.5181399386379085`.
- Referable sensitivity: `0.3916849015317287`; specificity: `0.9805749805749806`; F1: `0.5416036308623298`; ROC-AUC: `0.8296804095491184`.
- These are descriptive model-evaluation metrics only; they are not clinical performance claims.

## 5. Error analysis

- Grade 0 -> 2/3/4: `18`.
- Grade 1 -> 2/3/4: `2`.
- Grade 2/3/4 -> 0/1: `292`.
- Grade 3 <-> 4: `33`.
- High-confidence incorrect (raw confidence >= 0.8): `402`.
- Referable false negatives: `278`.
Complete records are in `failure_analysis.json`; no predictions were changed.

## 6. Image quality and RetinaGuard

Quality measurements are included from the prior authorized Phase 5.1 ImageTrustGate cache and are labeled as cached, not silently rerun evidence results.
- Quality summary: `{"decision_counts": {"BORDERLINE": 1711, "GRADABLE": 27, "UNGRADABLE": 6}, "note": "These quality records are reused from the prior Phase 5.1 measurement cache; optional evidence stages were not inferred from them.", "quality_score": {"max": 0.9005, "mean": 0.7269479357798165, "min": 0.6095}, "readable_count": 1744, "record_count": 1744, "status": "CACHED_MEASUREMENTS"}`.
- Current RetinaGuard was recomputed for `1744` classifier rows with optional lesion/attention/stability signals explicitly unavailable.
- RetinaGuard is an engineering reliability assessment; it does not prove prediction correctness.

## 7. Lesion evidence, vessels, localization and explainability

The current production artifacts were not replaced: primary lesion is the registered pretrained U-Net adapter, primary vessel is the registered R2-V2 RRWNet adapter, and localization remains heuristic because no production localization checkpoint is configured.
Population-level lesion, vessel, localization, Grad-CAM, and attention-agreement metrics were not claimed because the full 1,744-image optional evidence run was not completed.
- Real smoke run: `COMPLETED` on `20051020_43882_0100_PP.png`; details are in `smoke_test.json`.

## 8. Robustness and performance

No calibration fitting, threshold tuning, model selection, or robustness perturbation sweep was performed on Messidor-2.
- Fresh classifier timing summary: `{"count": 1744, "max_ms": 19.259, "mean_ms": 16.202, "median_ms": 16.173, "min_ms": 14.499, "p95_ms": 18.016, "stddev_ms": 1.01}`.
- Full-pipeline blocker: `Full primary evidence pipeline cannot be completed in this CPU-only environment within bounded execution: current production R2-V2 vessel inference is approximately one minute per image on this host before primary lesion and Grad-CAM work; running it for all 1,744 images would require many hours and was not silently substituted or parallelized into an unbounded job. The real smoke run measured 71.623s for its combined evidence stage; this is an engineering observation, not a population runtime measurement.`

## 9. Required output artifacts

This directory contains the dataset manifest, model manifest, fresh per-image classifier/current-Guard records, classifier metrics, referable metrics, confusion matrix, error analysis, optional-stage summaries, smoke result, and this report.

## 10. Safety and non-claims

No clinical validation, regulatory approval, calibration guarantee, or diagnostic claim is made. Raw model confidence is not clinically calibrated. The official label provenance limitation remains attached to every report.

## 11. Final decision: MESSIDOR-2 EXTERNAL EVALUATION BLOCKED

The final status is BLOCKED because a complete 1,744-image run of the current R2-V2 plus primary lesion and Grad-CAM pipeline was not completed within this CPU-only environment. No optional-stage output was fabricated or promoted as a population result.

## 12. Production and artifact integrity

- No model weights, backend routes, frontend code, thresholds, or production registries were modified.
- No training, calibration fitting, or automatic retraining was performed.
- Existing Messidor evaluation artifacts were not overwritten.
