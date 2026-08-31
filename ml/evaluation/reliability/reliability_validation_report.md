# RetinaGuard reliability validation

Status: retrospective engineering validation of reliability signals. This document is not a clinical validation report and TRUSTED is not a correctness guarantee.

## Immutable model contract

- Checkpoint: `C:\Users\kiran\Downloads\retina_nexus\ml\weights\classifiers\aptos2019\efficientnet-b0-aptos2019-20260830-v1\checkpoint_best.pt`
- Checkpoint SHA-256: `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`
- Expected SHA-256 unchanged: `True`
- Classifier training weights were not retrained, fine-tuned, replaced, or modified by this validation.

## Retrospective population

- Source predictions: `C:\Users\kiran\Downloads\retina_nexus\ml\evaluation\messidor\zero_shot_predictions.csv`
- Dataset version: `messidor2-external-f7abdb1e0276`
- Prediction rows: **1744**
- Successful five-class reference rows: **1744**
- Quality-readable images: **1744**; unreadable: **0**
- Quality decisions: `{"BORDERLINE": 1711, "GRADABLE": 27, "UNGRADABLE": 6}`
- Reference class distribution: `{"0": 1017, "1": 270, "2": 347, "3": 75, "4": 35}`
- Reference prediction distribution: `{"0": 1556, "1": 3, "2": 71, "3": 17, "4": 97}`
- Duplicate audit: `{"canonical_representative_count": 6, "deduplicated_image_count": 1738, "exact_duplicate_group_count": 4, "note": "Duplicate results are inherited from the separately generated Phase 4B audit; the original predictions were not rewritten.", "perceptual_duplicate_group_count": 6, "source": "C:\\Users\\kiran\\Downloads\\retina_nexus\\ml\\evaluation\\messidor\\duplicate_audit.json", "status": "AVAILABLE"}`

## Reliability signals

- Reliability states: `{"INSUFFICIENT_EVIDENCE": 1444, "UNRELIABLE": 300}`
- Warning counts: `{"attention_evidence_not_available": 1744, "explanation_stability_not_run": 1744, "high_prediction_uncertainty": 232, "low_calibrated_confidence": 51, "ood_not_available": 1744}`
- Mean raw confidence: `0.85524812`
- Mean calibrated confidence (identity/unfitted runtime unless configured otherwise): `0.85524811`
- Mean predictive uncertainty: `0.25190356`
- OOD reference: **UNAVAILABLE** unless an authorized reference is configured; no unfamiliar-image detection guarantee is claimed.
- Lesion/evidence agreement: **UNAVAILABLE** in this retrospective source because Phase 4B predictions do not contain a supported lesion comparison.
- Explanation stability: **NOT RUN** for the full retrospective population; controlled robustness sample is reported separately.

## False-negative warning audit

- Reference false negatives under the unchanged Phase 4B grade-2 probability rule: **278**
- Warning coverage among those false negatives: `1.0`
- Warning counts: `{"attention_evidence_not_available": 278, "explanation_stability_not_run": 278, "high_prediction_uncertainty": 49, "low_calibrated_confidence": 5, "ood_not_available": 278}`
- No threshold, weight, model, or classifier output was optimized against these labels.

## Risk coverage

`{"trust_score": [{"count": 1444, "coverage": 0.82798165, "five_class_accuracy": 0.67313019, "minimum_trust_score": 0.45, "referable_false_negative_rate": 0.9055794}, {"count": 359, "coverage": 0.20584862, "five_class_accuracy": 0.72423398, "minimum_trust_score": 0.6, "referable_false_negative_rate": 1.0}, {"count": 0, "coverage": 0.0, "five_class_accuracy": null, "minimum_trust_score": 0.75, "referable_false_negative_rate": null}], "warning_burden": [{"count": 0, "coverage": 0.0, "five_class_accuracy": null, "max_warning_burden": 0, "referable_false_negative_rate": null}, {"count": 0, "coverage": 0.0, "five_class_accuracy": null, "max_warning_burden": 1, "referable_false_negative_rate": null}, {"count": 0, "coverage": 0.0, "five_class_accuracy": null, "max_warning_burden": 2, "referable_false_negative_rate": null}, {"count": 1512, "coverage": 0.86697248, "five_class_accuracy": 0.66203704, "max_warning_burden": 3, "referable_false_negative_rate": 0.84501845}]}`
These fixed operating views are descriptive only and do not establish clinical risk coverage.

## Robustness and Grad-CAM

- Robustness status: `CALCULATED`
- Controlled sample: **3** images; completed variants: **24 / 24**
- Prediction stability: `1.0`
- Grad-CAM stability: `0.98136886`
- Grad-CAM artifacts are stored under `visuals/` and are linked to real checkpoint inference. Lesion and vessel evidence were intentionally not rerun in this robustness scope and are marked unavailable, not inferred.

## Safety interpretation

RetinaGuard is a transparent operating/review decision layer. TRUSTED means configured signals did not raise a major warning; it does not mean the model is correct. REVIEW_RECOMMENDED requires professional review, UNRELIABLE blocks automated interpretation or recommends recapture, and INSUFFICIENT_EVIDENCE indicates that required evidence was not available. Final clinical responsibility remains with a qualified clinician.
