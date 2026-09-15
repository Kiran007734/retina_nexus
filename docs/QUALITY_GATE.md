# Adaptive Image Quality Gate

The Image Trust Gate is an acquisition-quality control, not a disease model.
It retains the original image and may create one PNG derivative for controlled
preprocessing. The original is never overwritten.

## Decision flow

```text
upload
  -> validate format, integrity, dimensions, and channels
  -> assess focus, contrast, illumination, field of view, exposure, artifacts
  -> GREEN / ACCEPTABLE: continue with the original image
  -> YELLOW / ENHANCEMENT_CANDIDATE: one controlled enhancement and recheck
  -> RED / UNGRADABLE: stop clinical AI and recommend recapture
```

`GRADABLE`, `BORDERLINE`, and `UNGRADABLE` remain in the API for backward
compatibility. The adaptive `quality_band` is `GREEN`, `YELLOW`, or `RED`.
Only a post-check with `ai_eligible=true` may enter clinical AI.

## Safety rules

- The existing severe-focus floor is retained at focus score `0.10`.
- A below-floor focus capture is red and cannot be rescued by enhancement.
- Insufficient field of view, severe illumination problems, clipping, and
  artifacts that obscure the field are non-recoverable.
- Contrast and mild illumination/tonal issues are enhancement candidates when
  no hard acquisition failure is present.
- Enhancement is limited to one pass using the existing CLAHE, illumination
  normalization, denoising, and contrast pipeline.
- A high aggregate score cannot override a critical component failure.
- A yellow post-check does not enter AI; it recommends recapture.

## Provenance

The quality response includes `quality_band`, `ai_eligible`, recoverable and
non-recoverable issue codes, `hard_focus_floor_passed`, and
`quality_gate_version=image-trust-gate-v2-adaptive`. Direct classifier,
evidence, explainability, and RetinaGuard routes reject images that did not
reach the post-check and use the enhanced derivative when one was accepted.

Thresholds remain engineering heuristics. They are not clinical gradability
validation and should only be recalibrated against an independently labelled,
camera-representative gradability set.
