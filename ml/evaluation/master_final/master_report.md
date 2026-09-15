# RETINA-NEXUS Master ML Research, Safety, Validation & Fusion Report

Generated: `2026-09-14T06:01:04.975179+00:00`

## Executive status

**SIH TARGET NOT YET DEMONSTRATED.** The target requires referable sensitivity >90% and specificity >85% on the same frozen external evaluation. The authoritative-original Messidor descriptive evaluation did not demonstrate both at either frozen threshold. These results are not clinical validation because the available Messidor labels are local/adjudicated and are not independently proven official clinical ground truth.

No checkpoint, dataset, production threshold, production configuration, or existing APTOS behavior was changed. No model was promoted and no commit was created.

## Frozen model and dataset inventory

- APTOS EfficientNet-B0: `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`; SHA verified: `True`.
- IDRiD V3 research EfficientNet-B0: `97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049`; SHA verified: `True`; production promoted: `false`.
- R2-V2 vessel evidence model: `ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a`; SHA verified: `True`.
- Messidor authoritative original images: 1,748; label-matched evaluation population: 1,744; four originals had no matching local label and were excluded.
- Official IDRiD test images opened during this program: `0`.

## External classifier evaluation

All 1,744 label-matched authoritative original Messidor images were inferred by both frozen classifier candidates. Thresholds were evaluated without Messidor tuning.

| Model / threshold | Sensitivity | Specificity | FN | FP | Accuracy | QWK | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| APTOS EfficientNet-B0 / 0.50 | 57.33% | 87.80% | 195 | 157 | 0.6032 | 0.4924 | 0.7690 |
| APTOS EfficientNet-B0 / 0.20 | 86.87% | 57.34% | 60 | 549 | 0.6032 | 0.4924 | 0.7690 |
| IDRiD V3 / 0.50 | 26.91% | 99.38% | 334 | 8 | 0.6147 | 0.3751 | 0.7857 |
| IDRiD V3 / 0.20 | 45.73% | 94.87% | 248 | 66 | 0.6147 | 0.3751 | 0.7857 |

APTOS severity metrics are accuracy `0.6032`, macro F1 `0.3307`, weighted F1 `0.5296`, QWK `0.4924`, and macro OVR ROC-AUC `0.7690`. IDRiD V3 severity metrics are accuracy `0.6147`, macro F1 `0.3199`, QWK `0.3751`, and ROC-AUC `0.7857` on this descriptive external population.

## Supporting evidence and safety

The R2-V2 DRIVE evaluation used genuine manual vessel masks for 20 training images, with field-of-view masking. Mean Dice/F1 was `0.7176`, IoU `0.5624`, sensitivity `0.6098`, specificity `0.9885`. The available DRIVE copy has no manual test vessel masks, so no test accuracy was reported.

Lesion, vessel, Grad-CAM, agreement, uncertainty, OOD, and RetinaGuard remain supporting/reliability paths. Supporting evidence is not allowed to rewrite the primary severity grade. No leak-safe multimodal learned fusion was trained or promoted. No compatible independent DR backup checkpoint is installed, so backup disagreement is unavailable rather than fabricated.

## Runtime benchmark

The real-image full pipeline benchmark completed 50/50 records with zero failures. Three images stopped at the quality gate; 47 reached classifier/evidence stages. The 50-image run took `2546.3478263000143` seconds total. For the 47 gradable cases, median evidence total latency was `44.97s`, including median vessel inference `41.66s` on CPU. This is an engineering runtime measurement, not a clinical performance metric.

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
