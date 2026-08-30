# Real retinal vessel segmentation integration

## Selected model

RETINA-NEXUS integrates the published R2-V2 `bv` checkpoint:

- Model card: <https://huggingface.co/j-morano/R2-V2>
- Source implementation: <https://github.com/j-morano/R2-V2>
- Architecture: RRWNet, R2-V2 `bv` variant
- License declared by the model repository/model card: CC BY 4.0
- Published use: blood-vessel segmentation and artery/vein classification in
  retinal fundus images
- Training provenance stated in the artifact configuration:
  `Unified_Fundus`. RETINA-NEXUS does not infer a constituent dataset list.

The model card supplies a `bv.safetensors` checkpoint, `bv_config.json`, and
the exact inference source files. The `bv` output has three sigmoid channels;
channel 2 is the blood-vessel probability map used by RETINA-NEXUS. Artery and
vein channels are retained in the model output contract but are not exposed as
clinical artery/vein findings.

## Acquisition and verification

Weights and downloaded source files are ignored by Git. Install the optional
dependencies and acquire the complete artifact from the repository root:

```powershell
pip install -r backend/requirements-ml.txt
python scripts/acquire_vessel_model.py
python scripts/acquire_vessel_model.py --verify-only
```

The verified local checkpoint is:

```text
ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors
```

The acquisition manifest records the requested ref, immutable resolved
revision, source files, configuration, license, and SHA-256 checksum. The
current verified revision is `bb8c6c9346054df749f8bd48a9da41ba974e6a6b` and the
current checkpoint checksum is:

```text
ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a
```

No credential is required for the public default repository. If a future
revision is gated, configure the user's Hugging Face authentication outside
the repository; never place tokens in `.env` files committed to Git.

## Preprocessing and inference contract

The adapter follows the published preprocessing source rather than reusing
APTOS classifier preprocessing:

1. Convert the RGB fundus to `[0, 1]`.
2. Resize to the published 1408-pixel width while preserving aspect ratio.
3. Build the non-black field-of-view mask.
4. Run the published enhancement and CLAHE preprocessing.
5. Concatenate enhanced RGB and original RGB into six channels.
6. Pad spatial dimensions to the published U-Net multiple of 32.
7. Load `RRWNet` from the published source and the safetensors checkpoint.
8. Use sigmoid output channel 2, restore it to the evidence working image,
   threshold it for a binary mask, and zero pixels outside the field of view.

The adapter validates the configuration, source files, checkpoint checksum,
model output shape, and finite probability range before returning evidence.
It loads lazily, shares one process-local model instance, supports CPU and
CUDA when available, and returns explicit `MODEL_MISSING`, `MODEL_INVALID`, or
`MODEL_LOAD_FAILED` metadata when it cannot produce real output.

## Evidence and safety boundary

Each successful vessel result includes:

- probability map data URI;
- transparent binary mask data URI;
- original-image composite overlay;
- threshold and probability statistics;
- pixel coverage and connected-component count labelled
  `ENGINEERING_ESTIMATE`;
- model version, revision, checksum, source, and license metadata.

The primary pipeline never falls back to the classical-CV vessel mask. The
previous baseline remains available only through
`EVIDENCE_ENABLE_VESSEL_BASELINE=true` and is labelled
`EXPERIMENTAL BASELINE — NOT MODEL-BACKED`. The default is false. If the real
artifact is unavailable, the API returns an unsupported vessel module without
fabricating a mask.

RetinaGuard records one of `REAL_MODEL_EVIDENCE`, `EXPERIMENTAL_BASELINE`, or
`UNAVAILABLE` in its signal snapshot. This is provenance/audit information,
not an additional trust-score weight, so the presence of a vessel model does
not artificially raise reliability.

## Evaluation boundary

The real model is load-tested and run on an authorized local retinal image.
The repository does not claim vessel Dice, IoU, sensitivity, specificity, or
clinical accuracy because an authorized DRIVE ground-truth evaluation set is
not available in the local workspace. When DRIVE images and vessel masks are
placed under `ml/datasets/raw/drive/`, use the existing pairing and evaluation
utilities to calculate genuine metrics; until then the status is:

> Real inference verified; quantitative segmentation evaluation pending
> authorized ground-truth dataset.
