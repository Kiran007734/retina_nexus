# DRIVE Vessel Research Cycle

This document records the research-only vessel segmentation cycle performed on
the local DRIVE checkout. It does not promote a new vessel model and does not
change the protected R2-V2 production adapter.

## Data and leakage controls

- The discovered dataset contains 20 labeled training images, 20 official test
  images, 20 training manual vessel masks, and 40 field-of-view masks.
- The test split has no manual vessel masks. Accuracy metrics for that split
  are therefore unavailable and are intentionally not fabricated.
- Development selection used a deterministic five-fold split of the 20 labeled
  training images. Exact and confirmed perceptual duplicate checks found none.
- Patient identifiers are not supplied by DRIVE; the manifest documents that
  patient-level splitting is not possible.

## Candidate and freeze

The selected experimental candidate is a randomly initialized compact U-Net
using the green channel replicated to three channels, 512x512 input, focal plus
Dice loss, synchronized retinal augmentations, AdamW, six final-fit epochs, and
a threshold of 0.30 selected only from development OOF predictions.

- Model version: `drive-vessel-scratch-green-focal-dice-512-20260913-v1`
- Checkpoint: `ml/weights/vessels/drive/checkpoint_best.pt`
- SHA-256: `f199543db550f7cd84548cb0e2ab2cf7efbad99ffb41af8649b249d4433a3133`
- Production promoted: `false`

The frozen candidate's development OOF mean Dice is 0.2046 and mean IoU is
0.1166 before threshold analysis; at the OOF-selected threshold 0.30, mean
Dice is 0.2749 and mean IoU is 0.1655. These are engineering measurements on a
small development set, not clinical validation results.

## Protected reference

R2-V2 remains the default vessel model and is unchanged. Its existing DRIVE
training-mask reference artifact reports mean Dice 0.7176 and IoU 0.5624. The
new candidate is therefore marked **VESSEL SEGMENTATION REQUIRES FURTHER
RESEARCH** and is not automatically substituted into production.

## Integration policy

The new adapter is opt-in with `DRIVE_VESSEL_MODEL_ENABLED=false` by default.
When explicitly enabled, it verifies the manifest/checksum and emits vessel
supporting evidence only. It cannot alter DR severity, referable status, or
RetinaGuard classification logic.

## Official test limitation

The guarded one-time official pass opened and inferred all 20 test images and
used their FOV masks for engineering density summaries. Manual vessel masks
were unavailable, so Dice, IoU, accuracy, sensitivity, specificity, precision,
and F1 are all `null`/unavailable. No test metric was used for selection or
tuning.

See the generated artifacts:

- `ml/datasets/metadata/drive/drive_data_audit.json`
- `ml/datasets/metadata/drive/drive_manifest.json`
- `ml/datasets/metadata/drive/drive_split.json`
- `ml/datasets/metadata/drive/drive_cv_report_*.json`
- `ml/datasets/metadata/drive/drive_threshold_analysis.json`
- `ml/datasets/metadata/drive/drive_error_analysis.json`
- `ml/datasets/metadata/drive/drive_robustness.json`
- `ml/datasets/metadata/drive/drive_reproducibility.json`
- `ml/datasets/metadata/drive/drive_official_test.json`
- `ml/datasets/metadata/drive/drive_final_report.json`
