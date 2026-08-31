# Phase 4B — Messidor-2 zero-shot external validation

Status: **COMPLETED AS EXTERNAL MODEL EVALUATION**. This is not a clinical diagnostic validation claim.

## Dataset and matching

- Dataset version: `messidor2-external-f7abdb1e0276`
- Image files discovered: **1744**
- Label records: **1748**
- Matched image-label pairs: **1744**
- Unmatched label records excluded: **4**
- Images without labels: **0**
- Image-label matching: case-insensitive, extension-independent filename stems
- Label source: `google-brain/messidor2-dr-grades`, KaggleHub cache version 1
- Official ADCIS source: https://www.adcis.net/en/third-party/messidor2/

The four unmatched label IDs are recorded verbatim in `dataset_manifest.json` and were not replaced: `im002385`, `im004176`, `im003718`, and `20060411_58550_0200_pp`.

## Model contract

- Model: `RETINA-NEXUS DR classifier`
- Version: `efficientnet-b0-aptos2019-20260830-v1`
- Architecture: `efficientnet_b0`
- Input: `224 × 224` RGB
- Normalization: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- Checkpoint SHA-256 before: `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`
- Checkpoint SHA-256 after: `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`
- Checkpoint unchanged: **True**

## Compatibility

The downloaded readme documents `adjudicated_dr_grade` as a five-point ICDR scale: 0 None, 1 Mild DR, 2 Moderate DR, 3 Severe DR, and 4 PDR. This supports a documented one-to-one mapping to the existing APTOS output labels for descriptive external evaluation. It does not establish clinical interchangeability or clinical validation.

## Inference accounting

- Matched images inferred: **1744**
- Gradable matched images: **1744**
- Ungradable matched images: **0**
- Successful inferences: **1744**
- Failed inferences: **0**

## Valid five-class metrics

Evaluation population: successful inference + `adjudicated_gradable=1` + valid grade 0–4. Unmatched labels, ungradable rows, and failed inferences are excluded.

- `accuracy`: 0.623853 (95% CI 0.601491–0.645642)
- `macro_f1`: 0.344381 (95% CI 0.309375–0.380267)
- `weighted_f1`: 0.518274 (95% CI 0.490656–0.545870)
- `roc_auc_ovr_macro`: 0.775057
- `quadratic_weighted_kappa`: 0.518140 (95% CI 0.463147–0.564614)

Referable DR uses the existing grade-2-or-worse rule and is reported as a grade-based external metric:

- `sensitivity`: 0.391685
- `specificity`: 0.980575
- `precision`: 0.877451
- `recall`: 0.391685
- `f1`: 0.541604
- `roc_auc`: 0.829680

## Unsupported claims/metrics

- No prospective clinical accuracy or clinical validation claim is made.
- DME metrics are not calculated because this classifier does not predict DME.
- No metrics include the four unmatched labels.
- No threshold or model setting was optimized on Messidor-2.

## Failure analysis

- Low-confidence threshold: 0.60; count: **258**
- Label/prediction disagreements: **656**
- Duplicate groups: exact **4**, perceptual **6**

Bootstrap intervals use a fixed-seed nonparametric percentile bootstrap with `2000` iterations and seed `42`.

No training, fine-tuning, retraining, model selection, or weight modification occurred.
