# Image Quality Gate Forensic Report

Date: 2026-09-15
Scope: the live upload reported as `WhatsApp Image 2026-09-15 at 10.20.58 AM.jpeg`

## QUALITY GATE ISSUE

The image was accepted by the upload endpoint, decoded successfully, and then
classified by the Image Trust Gate as `UNGRADABLE` with score `0.6634` (66%).
The recorded reasons were:

- `severe_blur` — focus component `0.0793`
- `low_contrast` — contrast component `0.3436`

This was a quality-gate decision, not an HTTP validation failure, model-load
failure, or frontend/backend request-contract failure.

## FALSE REJECTION YES/NO/UNKNOWN

**UNKNOWN — no human gradability label is available for this image.**

The runtime behavior is internally consistent and the available engineering
evidence supports a conservative rejection: the image's focus metric is close
to the known ungradable reference and far below the known gradable reference.
That does not establish clinical gradability, because the repository has no
labelled gradability study for this camera/source population.

The image has a large visible retinal field and usable-looking global anatomy,
but visual usability alone cannot demonstrate preservation of tiny lesion
detail. No threshold change is justified by this single unlabeled image.

## ORIGINAL METRICS

Stored source:

`backend/storage/fundus/f8581203-9746-4b3c-83e8-cb0a261ddaf8/1b02c577-d947-4b7f-b7df-b5e90a4e7092.jpg`

- image id: `1b02c577-d947-4b7f-b7df-b5e90a4e7092`
- bytes: `158154`
- SHA-256: `c5e84951231369563dac183d48498a3ebb5e2ab2ae32c2e45fca5ee73cad2fcf`
- decoded format: JPEG
- dimensions: 1280 x 1280
- channels: 3, RGB
- Laplacian variance in the runtime retinal ROI: `7.1834`
- mean intensity: `114.2624`
- intensity standard deviation: `16.5747`
- retinal coverage ratio: `0.7686`
- low/high clipped ratios: `0.0 / 0.0`
- border/saturated artifact ratios: `0.0 / 0.0`

Component scores:

| Component | Score |
|---|---:|
| Focus | 0.0793 |
| Illumination | 0.9472 |
| Contrast | 0.3436 |
| Field of view | 1.0000 |
| Exposure | 1.0000 |
| Artifacts | 1.0000 |

## CURRENT THRESHOLDS AND FORMULAS

The current implementation is in
`backend/app/ml/quality/trust_gate.py`.

- Focus score: `log_score(laplacian_variance, low=5, high=300)`.
- Weights: focus `0.25`, illumination `0.15`, contrast `0.15`, field of view
  `0.20`, exposure `0.15`, artifacts `0.10`.
- A focus score below `0.30` creates a focus issue.
- A focus score below `0.10` makes that issue severe.
- A weighted score below `0.45` is ungradable.
- Any severe issue is ungradable.
- A weighted score below `0.75`, or any issue, is borderline.

The target weighted score is `0.6634`; the severe focus issue independently
forces `UNGRADABLE`.

These thresholds are documented in
`ml/evaluation/reliability/quality_threshold_rationale.json` as
`HEURISTIC_CODE_CONFIGURATION_ONLY`, not as thresholds learned from labelled
clinical gradability outcomes.

## REFERENCE COMPARISON

All values below were measured with the same current runtime service:

| Reference | Format / size | Laplacian variance | Decision | Score |
|---|---|---:|---|---:|
| Reported upload | JPEG, 1280x1280 | 7.1834 | UNGRADABLE | 0.6634 |
| Known gradable fixture `04efb1a284cc.png` | PNG, 1050x1050 | 36.2912 | GRADABLE | 0.7783 |
| Known borderline fixture `005b95c28852.png` | PNG, 2048x1536 | 55.9106 | BORDERLINE | 0.7420 |
| Known ungradable fixture `000c1434d8d7.png` | PNG, 3216x2136 | 7.6864 | UNGRADABLE | 0.6238 |

The target's focus signal is therefore much closer to the known ungradable
case than to the known gradable case. This is not a clinical validation of the
threshold; it is a runtime consistency check.

## FRONTEND/BACKEND TRANSPORT AUDIT

The frontend upload client in `frontend/src/services/api.ts` sends:

- `POST /api/v1/images/upload`
- query parameters `patient_id` and `eye`
- `multipart/form-data` with field name `image`
- the original browser `File` object
- no frontend resize or re-encoding
- no manually set multipart `Content-Type` header, so the browser owns the boundary

The OpenAPI contract agrees: `patient_id` and `eye` are required query
parameters and `image` is the required binary multipart field.

Live checks:

- `GET /api/v1/images/1b02c577-d947-4b7f-b7df-b5e90a4e7092/content?variant=original`: HTTP 200, `image/jpeg`, 158154 bytes.
- Local stored-file SHA-256 equals API content SHA-256 exactly.
- Valid multipart upload using the same field/query contract: HTTP 201.
- Deliberately declaring the JPEG bytes as `image/png`: HTTP 415,
  `UNSUPPORTED_MEDIA_TYPE`.
- `POST /api/v1/images/1b02c577-d947-4b7f-b7df-b5e90a4e7092/quality`: HTTP 200 with the metrics above.

There is no evidence that the frontend sent a different image, resized the
image, or caused a MIME mismatch.

## PIPELINE RESULT

`POST /api/v1/screening/run` with the target image returned HTTP 200 and:

- overall status: `COMPLETED`
- primary status: `QUALITY_BLOCKED`
- quality assessment: `COMPLETED`
- DR classification: `SKIPPED`
- lesion/evidence analysis: `SKIPPED`
- Grad-CAM: `SKIPPED`
- RetinaGuard: `SKIPPED`
- triage: `RECAPTURE_IMAGE`
- clinical AI started: `false`

This is the intended safety behavior for an ungradable image. There is no
clinical prediction to render or place in a report for this run.

## ENHANCEMENT CHECK

The current implementation only enhances `BORDERLINE` images. It does not
apply enhancement to an image that already has a severe issue. This prevents a
single aggressive preprocessing pass from converting a severely soft image
into an apparently acceptable input.

For diagnostic purposes only, forcing the existing one-pass enhancer on the
target produced `BORDERLINE`, not `GRADABLE`, and introduced a lower field of
view score (`0.2639`) plus a border artifact signal. That result is not used by
the production path and does not justify bypassing the severe-blur guard.

## ROOT CAUSE / PROBLEM

**Root cause of the observed rejection:** the decoded target image has low
measured retinal edge detail and low tonal separation under the configured
quality heuristics. The rejection is not caused by a frontend request bug,
wrong image bytes, MIME mismatch, or an API validation error.

**Known limitation:** the focus and aggregate thresholds are heuristic and
camera/population dependent. Their scientific calibration requires an
independent, labelled gradability dataset. Lowering them from this one case
would weaken the gate without evidence that the target preserves lesion-level
detail.

## FIX

No production code was changed. No safe code defect was identified. The
classifier, checkpoints, disease thresholds, lesion/vessel modules,
Grad-CAM, fusion, and RetinaGuard logic were not modified.

The request contract is unchanged before/after because no integration fix was
required.

## VALIDATION STATUS

- Quality gate service regression: passed before this audit.
- Backend test suite: previously `103 passed`.
- Python compilation: previously passed for `backend`, `scripts`, and `ml`.
- Frontend lint: previously passed.
- Frontend production build: previously passed.
- `git diff --check`: previously passed.
- Local frontend: `http://localhost:5173/`.
- Local backend: `http://localhost:8000/`.
- Readiness: `http://localhost:8000/api/v1/health/ready`.
- API docs: `http://localhost:8000/docs`.

The target's safe quality rejection was verified live. Downstream clinical AI
was correctly not run, so a full inference result cannot honestly be claimed
for this image.

