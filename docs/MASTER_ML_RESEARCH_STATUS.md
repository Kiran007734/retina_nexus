# RETINA-NEXUS Master ML Research Status

This document records the non-destructive master research pass completed on
2026-09-14. It is an engineering/research record, not a clinical validation
or regulatory submission.

## Frozen roles

- APTOS EfficientNet-B0 remains the existing classifier runtime path. Its
  checkpoint and SHA are recorded in `ml/models/model_registry.json`.
- IDRiD V3 EfficientNet-B0 remains a frozen research-only candidate. It was
  evaluated descriptively but was not promoted.
- The fundus-lesion model and R2-V2 vessel model remain supporting evidence
  models. They cannot rewrite the primary five-class severity grade.
- No compatible independent DR backup checkpoint is installed. RETFound was
  reviewed as a research candidate, but no classifier head/checkpoint was
  downloaded or integrated.

## External evaluation

The authoritative Messidor migration contains 1,748 original images. The
local/adjudicated label manifest matches 1,744 images; four original images
remain unlabeled and were excluded. Both frozen classifiers were evaluated on
all 1,744 matched originals at thresholds 0.50 and 0.20 without tuning on the
external data.

The results are descriptive only. The available label source is not
independently proven official clinical ground truth. The SIH target is therefore
`TARGET NOT YET DEMONSTRATED` because neither frozen model/threshold showed
both referable sensitivity above 90% and specificity above 85% on this set.

See [master_report.md](../ml/evaluation/master_final/master_report.md) and
[messidor_final_results.json](../ml/evaluation/master_final/messidor_final_results.json).

## Fusion and safety

No learned multimodal fusion was trained or promoted because a leak-safe
out-of-fold evidence table was not available for all development records.
RetinaGuard remains a transparent reliability/escalation layer. Quality,
uncertainty, missing evidence, OOD monitoring, and model disagreement affect
review/triage state; they do not silently change the classifier grade.

No clinical trust guarantee is made. Raw probabilities are not clinically
calibrated.

## Runtime

The real-image full pipeline benchmark completed 10/10 and 50/50 bounded runs
with zero terminal failures. The 50-image run reached all classifier/evidence
stages for 47 images and stopped three at the quality gate. CPU R2-V2 vessel
inference was the dominant measured bottleneck. The complete measured stage
latencies are in [runtime_results.json](../ml/evaluation/master_final/runtime_results.json).

## Data and test boundaries

- No model weights or dataset files were modified.
- Official IDRiD test images were not opened by this program.
- No Messidor threshold was tuned.
- No production promotion, commit, or push was performed.
- DRIVE metrics use genuine manual masks for the available 20 training images;
  the available test copy has no manual vessel masks, so test accuracy is not
  reported.

## Primary artifacts

- [Central model registry](../ml/models/model_registry.json)
- [Phase 0 audit](../ml/evaluation/master_audit/master_audit_report.md)
- [Model comparison](../ml/evaluation/master_final/model_comparison.json)
- [Safety analysis](../ml/evaluation/master_final/safety_analysis.json)
- [Fusion results](../ml/evaluation/master_final/fusion_results.json)
- [XAI validation](../ml/evaluation/master_final/xai_validation.json)
- [RetinaGuard validation](../ml/evaluation/master_final/retinaguard_validation.json)

