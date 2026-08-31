# Phase 4B reliability audit

Status: **COMPLETED**. This is a reliability and evaluation-quality audit,
not clinical validation.

## Required provenance statement

> The Messidor-2 official release does not provide official DR ground truth. This evaluation uses separately acquired third-party reference labels.

The label source remains `google-brain/messidor2-dr-grades`, KaggleHub cache
version 1. Its CSV and readme SHA-256 values, exact cache path, and discovery
method remain in `dataset_manifest.json`. Official source terms and dataset
limitations are documented by [ADCIS](https://www.adcis.net/en/third-party/messidor2/).

## Checkpoint and reproducibility

- Model version: `efficientnet-b0-aptos2019-20260830-v1`
- Architecture: `efficientnet_b0`
- Checkpoint SHA-256 before audit: `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`
- Checkpoint SHA-256 after audit: `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`
- Checkpoint unchanged: **True**
- Bootstrap: nonparametric percentile, 2,000 iterations, seed 42
- Original full-dataset metrics and predictions were not overwritten.

## Duplicate audit

- Exact duplicate groups: **4**
- Perceptual duplicate groups: **6**
- Perceptual group members excluded from sensitivity analysis: **6**
- Perceptual groups with identical DR/DME/gradable labels: **5** / 6
- Perceptual groups with conflicting labels: **1**

Policy: keep the lexicographically first member of each perceptual duplicate
group and exclude only the other members from the separate deduplicated
sensitivity-analysis run. No raw image was deleted and the original full run
remains the primary reported evaluation. Patient identifiers were unavailable,
so this audit cannot infer whether duplicate files represent the same patient.

## Full vs deduplicated metrics

| Metric | Full (1744) | Deduplicated (1738) | Delta |
|---|---:|---:|---:|
| Accuracy | 0.623853 | 0.624281 | +0.000428 |
| Macro F1 | 0.344381 | 0.343862 | -0.000519 |
| Weighted F1 | 0.518274 | 0.518601 | +0.000327 |
| ROC-AUC | 0.775057 | 0.775299 | +0.000242 |
| QWK | 0.518140 | 0.518437 | +0.000297 |
| Referable sensitivity | 0.391685 | 0.391209 | -0.000476 |
| Referable specificity | 0.980575 | 0.980514 | -0.000061 |

The deduplicated run is a sensitivity analysis, not a replacement benchmark.
Duplicate removal changed the evaluation population by 6 records. The JSON comparison records the exact deltas and bootstrap intervals.

## Threshold analysis

Thresholds use the unchanged raw referable probability field and do not retrain
or recalibrate the model. The original operating point is 0.50.

| Operating point | Threshold | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Current | 0.50 | 0.391685 | 0.980575 | 0.877451 | 0.541604 |
| High sensitivity | 0.05 | 0.728665 | 0.785548 | 0.546798 | 0.624765 |
| Balanced / Youden J | 0.10 | 0.643326 | 0.876457 | 0.649007 | 0.646154 |
| High specificity | 0.70 | 0.308534 | 0.990676 | 0.921569 | 0.462295 |

Selection notes: high-sensitivity: target sensitivity >= 0.90 was not reached; selected the tested threshold with maximum sensitivity;
high-specificity: highest sensitivity among tested thresholds meeting specificity >= 0.99. These are
exploratory operating points, not clinically validated thresholds.

## Error analysis

- Eligible full-dataset records: **1744**
- Misclassified records: **656**
- False negatives for grade>=2 screening: **278**
- False positives for grade<2 screening: **25**
- Image-quality metadata: **unavailable in the original evaluation records**

All error groups are included in `error_analysis.json`. Representative examples
are selected deterministically per reference/prediction group; they are not a
cherry-picked estimate. Confidence summaries use raw model confidence and raw
grade>=2 probability.

The original 39.17% referable sensitivity is not explained by thresholding
alone: it is the result at the existing 0.50 decision rule, and the exploratory
grid shows the sensitivity/specificity trade-off. It also reflects model error
on this shifted external population and the conditional third-party label
mapping. Because patient IDs and official Messidor-2 DR ground truth are not
available, this audit cannot attribute the result to patient leakage or make a
clinical-causality claim.

## Artifacts

- `duplicate_audit.json`
- `deduplicated_metrics.json`
- `deduplicated_predictions.csv`
- `deduplicated_confusion_matrix.png`
- `threshold_analysis.json`
- `threshold_tradeoff.png`
- `error_analysis.json`
- `error_analysis.png`
- `full_vs_deduplicated_comparison.json`
- `full_vs_deduplicated_comparison.png`
