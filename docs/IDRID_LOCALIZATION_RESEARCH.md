# IDRiD optic-disc and fovea localization research

This document describes the IDRiD C. Localization research track. It is an
anatomical evidence module and is not the DR classifier, a clinical diagnosis,
or a production-promoted model.

## Dataset governance

The audit found 413 official training images and 103 official testing images.
All four coordinate CSVs were present and matched their respective images:
optic-disc and fovea coordinates for both train and test. Images are RGB
4288x2848 JPEGs and coordinates are original-image pixels with origin at the
top-left.

An exact duplicate was found across the package boundary: training
`IDRiD_118.jpg` is identical to official testing `IDRiD_064.jpg`. It is
excluded from the development pool. The resulting development pool contains
412 images. Other exact duplicates are retained only within the development
side and are kept together by duplicate-aware folds. IDRiD does not provide
patient identifiers in this package, so patient-level grouping is unavailable.

The authoritative artifacts are:

- `ml/datasets/metadata/idrid/idrid_localization_data_audit.json`
- `ml/datasets/metadata/idrid/idrid_localization_manifest.json`
- `ml/datasets/metadata/idrid/idrid_localization_split.json`

The official test split is reserved until the frozen-model gate passes.

## Model and preprocessing

The experimental model is a shared compact CNN with two heatmap channels and
an auxiliary two-landmark coordinate head. RGB images are aspect-ratio
preserving letterboxed to 512x352 and normalized with ImageNet mean/std
`[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]`. Training uses horizontal
flip, geometry-aware +/-4 degree rotation, brightness and contrast changes.
Heatmaps are decoded with temperature-scaled spatial soft-argmax. Reported
coordinates are mapped back to original image pixels.

The loss is heatmap MSE plus auxiliary Smooth L1 coordinate loss. The model
is evidence-only and must not change DR grade, referable status, or RetinaGuard
rules.

## Evaluation protocol

Development model selection uses the leak-safe fold artifact and reports mean,
median, P90, maximum and normalized landmark error, axis-specific MAE, and
success rates at 1%, 2%, 5% and 10% of image diagonal. The existing classical
evidence baseline is retained as a comparison, not silently replaced.

After the selected model is frozen, the official 103-image test evaluator may
be run exactly once. Its output is
`ml/datasets/metadata/idrid/idrid_localization_official_test.json` and includes
per-image metrics, aggregate metrics and visual comparisons. The official test
must not be used for tuning, threshold selection or production promotion.

Robustness and reproducibility checks are engineering checks only. They do not
establish clinical validity, generalization, or regulatory approval.

## Runtime integration

The backend adapter is opt-in through:

```text
IDRID_LOCALIZATION_MODEL_ENABLED=true
IDRID_LOCALIZATION_MODEL_PATH=ml/weights/localization/idrid/checkpoint_best.pt
IDRID_LOCALIZATION_MODEL_SHA256=<manifest checkpoint SHA-256>
```

When enabled and loadable, it replaces only the optic-disc and fovea evidence
modules. The default configuration remains unchanged. A missing artifact or
checksum mismatch is reported as unavailable evidence; it never produces a
fake landmark and never alters the classification or trust decision.

## Research status

The final status is recorded in
`ml/datasets/metadata/idrid/idrid_localization_final_report.json` after the CV,
freeze, one-time official evaluation, robustness, reproducibility, and
regression gates have completed. `production_promoted` must remain `false`.
