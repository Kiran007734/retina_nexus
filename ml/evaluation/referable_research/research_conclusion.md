# Referable DR sensitivity research

Status: **RESEARCH CANDIDATE IMPROVED — EXTERNAL VALIDATION PENDING**

## Protocol

- Development data: 406 governed IDRiD records.
- Evaluation: five-fold leak-safe OOF predictions, 406 unique images.
- Official IDRiD test images opened: 0.
- Messidor labels used for selection: false.
- Production checkpoint unchanged: `97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049`.
- Selection rule: highest OOF sensitivity subject to specificity >= 0.90.

## Threshold result

- Existing 0.40 threshold: sensitivity `0.818898`, specificity `0.980263`, FN `46`.
- Selected development-only threshold: `0.2`; sensitivity `0.893701`, specificity `0.907895`, precision `0.941909`, F1 `0.917172`, ROC-AUC `0.963919`, PR-AUC `0.981013`, FN `27`.
- This is not a Messidor result and is not a clinical validation result.

## Dedicated referable head

The existing shared EfficientNet-B0 architecture exposes a stage2 referable head. Its five-fold OOF analysis is recorded in `binary_referable_results.json`; no new binary checkpoint was trained or promoted.

## Other research directions

Existing domain-generalization candidates are reported from prior development-only artifacts. Learned lesion fusion was not run because there was no complete leak-safe OOF evidence table and IDRiD annotations are incomplete; no fusion output was fabricated.

## False negatives

Development OOF false-negative records, grade breakdown, quality proxy, confidence, and uncertainty are in `false_negative_analysis.json`. Grad-CAM and model disagreement are marked unavailable because those signals are not present in the OOF artifacts.

## Safety boundary

The selected threshold is research-only. It does not change the APTOS production threshold, production model, backend, frontend, or RetinaGuard. External Messidor evaluation remains pending until the research candidate is formally frozen.
