# RETINA-NEXUS Fusion Research Report

Generated: `2026-09-14T13:41:50.854941+00:00`

## Governance

Research-only analysis. No checkpoint, production route, threshold, or RetinaGuard production rule was modified. Messidor labels were not used for fitting, threshold selection, calibration, or fusion selection. Official IDRiD test images opened: `0`.

## Development results

Primary IDRiD V3: sensitivity `0.8937`, specificity `0.9013`, F1 `0.9153`, FN `27` at frozen threshold `0.1`.
RETGUARD verifier: sensitivity `1.0000`, specificity `0.9474`, F1 `0.9845`, FN `0` at published threshold `0.204983`.
Disagreement: `42` / 406 (`0.1034`). Verifier-caught primary FNs: `27`. Both-model FNs: `0`.
Best research fusion operating point: `maximum_probability` with sensitivity `1.0000`, specificity `0.9803`, FN `0`, FP `3`.
Learned fusion: `COMPLETED`. It is strict fold-held-out logistic regression only and is not promoted. Calibration ECE/Brier: `0.039930412056562996` / `0.00838441430365663`.

## False-negative and evidence interpretation

Relative to primary, the selected research fusion changes sensitivity by `0.1063` and specificity by `0.0789`.
The verifier checkpoint is architecture/checkpoint-independent from the primary, but its published training datasets include IDRiD; sample-level membership is unknown. Therefore the 406-image verifier and learned-fusion results are overlap-risk development evidence, not independent validation.
Lesion, vessel, localization, Grad-CAM, quality, and uncertainty signals remain supporting or reliability evidence. They are not converted into disease probabilities and do not rewrite severity. A complete leak-safe 406-image lesion evidence table was unavailable, so lesion fusion was not fabricated.
Research RetinaGuard review flag (disagreement OR verifier OOD) sensitivity: `0.1181`; specificity: `0.8618`. This is a review-priority analysis, not a clinical trust guarantee.

## External evaluation

Messidor application used `1744` cached label-matched records exactly once after development selection was frozen. Frozen `maximum_probability` result: sensitivity `0.9278`, specificity `0.8609`, FN `33`, FP `179`. Results are descriptive only; see `messidor_fusion_results.json`. SIH >90% sensitivity and >85% specificity is **TARGET_NOT_YET_DEMONSTRATED** because the verifier has known IDRiD training-overlap risk and the external labels are not clinical validation.

## Final decision

Production promotion is not justified. The primary remains the severity controller; the verifier and evidence modules remain research-only complementary signals pending independent prospective and target-device validation.

## Provenance

Primary checkpoint SHA-256: `97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049`.
RETGUARD ONNX SHA-256: `f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b`.

Artifacts: `fusion_prediction_table.json`, `disagreement_analysis.json`, `fusion_baselines.json`, `fusion_thresholds.json`, `fusion_cv_results.json`, `false_negative_analysis.json`, `evidence_fusion_results.json`, `retinaguard_fusion_results.json`, and `messidor_fusion_results.json`.
