# Adaptive Image Quality + Enhancement Gate Report

Date: 2026-09-15
Version: `image-trust-gate-v2-adaptive`

## QUALITY ARCHITECTURE

Previous behavior:

```text
image -> aggregate quality score -> pass/reject
```

Current behavior:

```text
image -> component assessment -> GREEN/YELLOW/RED band
      -> one controlled enhancement only for YELLOW
      -> post-enhancement assessment
      -> AI only when the post-check is GREEN and ai_eligible=true
```

The existing metrics remain active: focus/sharpness, contrast, illumination,
field of view, exposure, artifacts, dimensions, and retinal-field visibility.
The weighted score remains a supporting signal; it cannot override a critical
component failure.

## CURRENT PROBLEM IMAGE

Image SHA-256:
`c5e84951231369563dac183d48498a3ebb5e2ab2ae32c2e45fca5ee73cad2fcf`

- dimensions: 1280 x 1280
- channels: 3, RGB
- format: JPEG
- original quality score: `0.6634`
- focus score: `0.0793`
- Laplacian variance: `7.1834`
- contrast score: `0.3436`
- illumination score: `0.9472`
- field-of-view score: `1.0000`
- hard focus floor: `0.10`
- hard focus floor passed: `false`
- initial band: `RED`
- enhancement attempted: `NO`
- post-enhancement metrics: not applicable; red images are not enhanced
- final decision: `RECAPTURE`
- final AI eligibility: `false`

Reason: severe blur is a non-recoverable acquisition failure under the
existing conservative focus floor. Low contrast is separately recorded as a
recoverable issue, but it cannot override the focus floor.

Result: **the problem image remains ungradable and is not forced through AI**.

## GOOD IMAGE

Fixture: `ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png`

- original assessment: `0.7783`
- band: `GREEN`
- decision: `GRADABLE`
- AI eligible: `true`
- enhancement: not attempted

Result: **PASS — direct AI path**.

## BORDERLINE IMAGE

Fixture: `ml/datasets/raw/aptos2019/train_images/005b95c28852.png`

- initial score: `0.7420`
- initial band: `YELLOW`
- initial issue: `low_contrast` (recoverable)
- enhancement attempted: `YES`
- enhancement passes: `1`
- post-enhancement score: `0.7622`
- post-enhancement focus: `1.0000`
- post-enhancement contrast: `0.3208`
- post-enhancement field of view: `0.3629`
- post-enhancement issue: `insufficient_field_of_view` (non-recoverable)
- final band: `RED`
- final decision: `UNGRADABLE`
- final AI eligibility: `false`

Result: **PASS — enhancement candidate was rechecked and rejected when the
minimum component requirements were not met**.

## UNGRADABLE IMAGE

Fixture: `ml/datasets/raw/aptos2019/train_images/000c1434d8d7.png`

- score: `0.6238`
- band: `RED`
- focus score: `0.0945`
- Laplacian variance: `7.6864`
- issues: `severe_blur`, `low_contrast`
- enhancement: not attempted
- AI eligible: `false`

Result: **PASS — rejected without enhancement**.

## LIVE API VERIFICATION

- target content endpoint: HTTP `200`, `image/jpeg`
- target content SHA equals local stored SHA exactly
- target quality endpoint: HTTP `200`, `RED`, `ai_eligible=false`
- target master screening: HTTP `200`, `primary_status=QUALITY_BLOCKED`
- target classification: `null`
- target triage: `RECAPTURE_IMAGE`
- target direct classifier route: HTTP `422`
- live borderline upload: HTTP `201`
- live borderline quality: HTTP `200`, one enhancement, final `RED`
- live borderline direct classifier route: HTTP `422`

## ORIGINAL/PRESERVED IMAGE SECURITY

For the live borderline case, the original uploaded PNG SHA-256 remained:
`4af20410946f8730d1a9c13361a0b4604c5934c325890b6ddd85b39627c6a0e8`.

The enhanced derivative was served separately as PNG with SHA-256:
`7c3cc7aaa249525e1d3d0817b93cd22463d3485c398492509b85366459afbf88`.

The original and enhanced bytes differ, and the original was not overwritten.

## MIME SECURITY

An intentionally mislabeled JPEG upload declared as `image/png` returned HTTP
`415` with `UNSUPPORTED_MEDIA_TYPE`. Valid JPEG and PNG multipart uploads
returned HTTP `201`.

Result: **PASS**.

## FULL AI PIPELINE

For the current problem image: **NOT RUN BY SAFETY DESIGN**. It did not pass
the quality gate, so classification, lesions, vessels, localization,
explainability, fusion, RetinaGuard, report, and PDF were not started. No
fabricated result was produced.

The existing full-pipeline regression path remains covered by the backend
integration suite; no model, checkpoint, model architecture, disease
threshold, fusion rule, or RetinaGuard rule was changed.

## PERFORMANCE

Measured local wall-clock timings using the runtime service:

| Case | Original assessment | Enhancement | Post-assessment | Total |
|---|---:|---:|---:|---:|
| Current problem image | 86.14 ms | not run | not run | 86.14 ms |
| Good fixture | 72.00 ms | not run | not run | 72.00 ms |
| Borderline fixture | 147.49 ms | 6010.91 ms | 696.15 ms | 6854.55 ms |
| Ungradable fixture | 1092.96 ms | not run | not run | 1092.96 ms |

## REGRESSION

- backend tests: `105 passed`
- Python compileall (`backend`, `scripts`, `ml`): passed
- frontend lint: passed
- frontend production build: passed
- model preflight: `READY`
- `git diff --check`: passed
- no model or weight files changed
- no retraining performed

## LOCALHOST URL

- frontend: [http://localhost:5173](http://localhost:5173)
- backend: [http://localhost:8000](http://localhost:8000)
- readiness: [http://localhost:8000/api/v1/health/ready](http://localhost:8000/api/v1/health/ready)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

All checked endpoints returned HTTP `200` where applicable.

## GIT STATUS

`DIRTY` — expected review changes are uncommitted. No commit or push was
performed.

