# RETINA-NEXUS Referable DR V2 Research Report

Generated: `2026-09-14T09:27:53.557151+00:00`

## Status

**RESEARCH ONLY. Production behavior is unchanged.** Messidor-2 was used only after the primary threshold and independent verifier were frozen. No Messidor threshold, model, augmentation, calibration, or fusion rule was selected from these labels.

The SIH target remains **TARGET NOT YET DEMONSTRATED**. The target requires sensitivity above 90% and specificity above 85% on the same appropriate evaluation.

## Primary binary candidate

The best available binary candidate is the existing IDRiD V3 shared EfficientNet-B0 stage2 referable head. It explicitly predicts referable probability rather than deriving it from severity argmax.

- Positive: IDRiD grade 2/3/4
- Negative: IDRiD grade 0/1
- Data: 406 development records, five-fold leak-safe OOF
- Frozen threshold: `0.1`
- Checkpoint SHA: `97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049`
- Production promoted: `false`

Development OOF: sensitivity `0.8937`, specificity `0.9013`, ROC-AUC `0.9630`, PR-AUC `0.9808`, FN `27`. OOF ECE is `0.1579` and Brier score `0.1193`; raw probabilities are not clinically calibrated.

## Domain generalization

Previously completed development-only domain candidates were reused. The comparison is recorded in `domain_generalization.json`; no uncontrolled new sweep was run. Existing candidate comparisons do not establish external generalization.

## Independent verifier

RETGUARD DR v1.0.0 was selected as the research verifier because it provides a documented binary referable task, official release weights, checksum, preprocessing, and model card. It is licensed for research/noncommercial use and is not integrated into production. [Official repository](https://github.com/anchor-neuro/retguard), [release](https://github.com/anchor-neuro/retguard/releases/tag/v1.0.0).

- Architecture: EfficientNetV2-M
- Input: 480x480 fundus photograph
- Protocol: official preprocessing and 8-view D4 TTA
- ONNX SHA-256: `f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b`
- Published threshold: `0.204983`
- External inference: `1744` / `1744`, zero failures
- External metrics: sensitivity `0.9716`, specificity `0.7599`, ROC-AUC `0.9690`, PR-AUC `0.9398`, FN `13`, FP `309`

The verifier's released model card reports known research limitations, including absent gradability handling and failed logit-parity tolerance for the released ONNX export. Those limitations prevent production promotion.

## Safety disagreement analysis

The fixed safety layer does not average probabilities and does not invent a diagnosis. It recommends review when primary and verifier decisions disagree or when the verifier OOD flag is raised.

- Agreement: `1224` / 1,744
- Disagreement rate: `0.2982`
- Primary false negatives: `243`
- Verifier-caught primary false negatives: `230`
- Both missed: `13`
- Safety review-flag sensitivity: `0.5033`
- Safety review-flag specificity: `0.7638`
- Sensitivity change: `0.0350`
- Specificity change: `-0.1795`

These are descriptive external results against local/adjudicated labels, not clinical safety estimates. Lesion evidence was not fused because a complete leak-safe OOF lesion table was unavailable; missing evidence was not fabricated.

## Model comparison on Messidor-2

| Candidate | Frozen operating point | Sensitivity | Specificity | FN | FP |
|---|---:|---:|---:|---:|---:|
| APTOS EfficientNet-B0 | 0.50 | 0.5733 | 0.8780 | 195 | 157 |
| APTOS EfficientNet-B0 | 0.20 | 0.8687 | 0.5734 | 60 | 549 |
| IDRiD V3 severity-derived | 0.50 | 0.2691 | 0.9938 | 334 | 8 |
| IDRiD V3 severity-derived | 0.20 | 0.4573 | 0.9487 | 248 | 66 |
| IDRiD V3 stage2 binary head | 0.10 | 0.4683 | 0.9433 | 243 | 73 |
| RETGUARD DR verifier | 0.204983 | 0.9716 | 0.7599 | 13 | 309 |

## Final decision

No model should replace the current production model. The binary head and RETGUARD verifier remain research-only. The next step is independent prospective/target-device validation and a pre-registered safety review protocol before any promotion decision.

## Artifacts

- `research_registry.json`
- `cv_results.json`
- `threshold_sweep.json`
- `false_negative_analysis.json`
- `domain_generalization.json`
- `verifier_candidates.json`
- `verifier_validation.json`
- `fusion_research.json`
- `messidor_external_results.json`
- `oof_predictions.jsonl`
- `primary_binary_messidor/`
- `retguard_messidor/`
