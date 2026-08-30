# Image Trust Gate

The Image Trust Gate is the first vision stage after acquisition. It answers only whether a fundus image is suitable for a later clinical AI stage; it does not classify diabetic retinopathy and it does not make a clinical recommendation.

## Input contract

The upload endpoint accepts JPEG and PNG files only. Pillow verifies the file, decodes the image, checks RGB/RGBA channels, and enforces dimensions from 256×256 through 12,000×12,000 pixels. The decoded image is then evaluated with OpenCV when installed, with a Pillow/NumPy fallback for minimal local development environments.

## Measured signals

- Focus: Laplacian variance (or an equivalent NumPy second-derivative fallback).
- Illumination: mean intensity, low/high intensity percentiles, and clipped-pixel ratios.
- Contrast: percentile spread and grayscale standard deviation.
- Field of view: non-background retinal coverage heuristic.
- Exposure: low/high clipping ratios.
- Artifacts: dark border and highly saturated-pixel ratios.

Signals are normalized and combined with explicit weights. `GRADABLE`, `BORDERLINE`, and `UNGRADABLE` thresholds are engineering defaults and must be calibrated against representative, reviewed data before any clinical use.

## Controlled enhancement

Borderline images receive at most one enhancement pass. With OpenCV available, the pass uses CLAHE, illumination normalization, mild denoising, and contrast normalization. The local fallback uses conservative Pillow contrast/color normalization and median filtering. The output is reassessed and both pre/post scores are returned. No endless enhancement loop is possible because `enhancement_passes` is persisted and capped at one.

Ungradable images return issue type, severity, explanation, and recapture recommendation. The frontend blocks continuation to the DR screening stage when the final decision is `UNGRADABLE`.

## API

- `POST /api/v1/images/upload?patient_id=<id>&eye=left|right`
- `POST /api/v1/images/{id}/quality`
- `POST /api/v1/images/{id}/enhance`
- `GET /api/v1/images/{id}/content?variant=original|enhanced`

The result includes component scores, raw metrics, feature vectors, camera EXIF fields when available, issues, action, and next action. `backend/app/ml/quality/ood.py` provides a future OOD detector protocol and quality-distribution summarizer; it makes no OOD decision today.
