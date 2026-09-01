# RETINA-NEXUS Phase 6 deployment readiness report

## Scope

This report records the deployment-hardening verification performed against
the existing RETINA-NEXUS prototype. No model was retrained, tuned, replaced,
or rewritten. The report is an engineering deployment assessment, not a
clinical validation, security certification, or production-scale capacity
claim.

## Verdict

**READY FOR PHASE 7 prototype work, with explicit local-demo limitations.**

The native local deployment path is operational and the complete screening
workflow was exercised with a real authorized APTOS image. Docker verification
could not be executed because Docker is not installed in the verification
environment; the Compose configuration remains an unbuilt local reference.

## Architecture verified

- FastAPI/Uvicorn backend with SQLAlchemy and SQLite for local development;
  PostgreSQL is the Compose target.
- React/Vite frontend with an API base URL configured through `VITE_API_URL`.
- Registered EfficientNet-B0 classifier plus independently versioned optional
  lesion and R2-V2 vessel evidence adapters.
- Inline database-backed screening pipeline with durable stage status. Redis is
  reserved as the future worker boundary; the current endpoint is synchronous.
- Local object storage for development and a documented future encrypted,
  resumable edge-sync boundary.

## Model verification

The explicit full verification command completed successfully:

```text
python scripts/verify_models.py
status: READY
classifier: AVAILABLE, manifest present, checksum valid, loadable
lesion_segmentation: AVAILABLE, manifest present, checksum valid, loadable
vessel_segmentation: AVAILABLE, manifest present, checksum valid, loadable
```

The startup readiness check load-verifies the required classifier and exposes
optional artifacts as capabilities. The demo preflight performs the complete
optional-model load/checksum verification before a demonstration.

Classifier checkpoint SHA-256 before Phase 6:

`ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`

Classifier checkpoint SHA-256 after Phase 6:

`ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`

The existing model manifest and historical provenance also record the same
before/after value and `checkpoint_unchanged: true`.

## Input hardening and safety

- Empty, unsupported, corrupt, undecodable, wrong-channel, undersized,
  oversized-dimension, and excessive-pixel images are rejected before costly
  processing.
- Uploads require JPEG/PNG suffix and MIME type, followed by actual Pillow
  verification/decoding.
- Filename path components are sanitized before persistence.
- HTTP errors use stable safe codes; generic failures return an internal error
  message and request ID without stack traces, paths, secrets, or image bytes.
- `MAX_UPLOAD_SIZE_MB`, `MAX_IMAGE_PIXELS`,
  `MAX_CONCURRENT_SCREENINGS`, and `SCREENING_TIMEOUT_SECONDS` bound obvious
  resource risks.
- Production settings reject placeholder/short `SECRET_KEY` values and use an
  explicit configured CORS allowlist.

## Health, observability, and failure behavior

- `GET /api/v1/health` returned HTTP 200 with database `ok`.
- `GET /api/v1/health/ready` returned HTTP 200 with required classifier and
  report generator available; optional capability status was exposed without
  filesystem paths.
- Requests receive an `X-Request-ID`; JSON logs include request, route,
  status, stage, duration, reliability state, and missing capability fields.
- Stage failures, timeouts, and optional-model gaps are persisted and do not
  create substituted predictions or evidence.
- A development/test schema compatibility repair was added for legacy local
  SQLite files; controlled deployments still use Alembic migrations.

## Real live end-to-end verification

The following path completed against a real local APTOS image:

```text
patient -> upload -> quality -> screening/run -> all 11 pipeline stages
-> EfficientNet prediction -> lesion/vessel evidence -> Grad-CAM/agreement
-> RetinaGuard -> triage -> report -> PDF -> screening status
```

Observed live result:

- image: `0d0b8fc9ab5c.png`, 2048x1536
- quality: `GRADABLE`, final score `0.8274`
- classifier: EfficientNet-B0, model version
  `efficientnet-b0-aptos2019-20260830-v1`, real prediction grade `0`
- evidence: lesion and vessel modules returned `model_inference`; experimental
  optic-disc/fovea statuses and unsupported neovascularization remained
  explicitly labelled
- explainability: Grad-CAM and attention/evidence agreement completed
- RetinaGuard: `UNRELIABLE` with an explicit review/recapture-or-specialist
  recommendation because optional reliability signals were missing/not run
- report/PDF: HTTP 201/200; PDF began with `%PDF-`

The result is a system verification only. It is not a claim about clinical
correctness or generalization.

## Performance benchmark

The real local benchmark is in
[`performance_benchmark.json`](performance_benchmark.json) and
[`performance_benchmark_report.md`](performance_benchmark_report.md). It used
three warm repetitions after separate model loads on Windows 10, Python
3.11.9, CPU-only PyTorch 2.11.0+cpu, 16 logical CPUs, and no CUDA.

Measured means (milliseconds):

| Stage | Mean | Median | P95 |
| --- | ---: | ---: | ---: |
| Image validation | 108.960 | 92.562 | 140.997 |
| Quality assessment | 325.221 | 209.857 | 525.097 |
| Classification | 196.175 | 112.106 | 358.538 |
| Lesion inference | 742.361 | 69.634 | 1,911.310 |
| Vessel inference | 30,532.666 | 274.706 | 81,973.217 |
| Grad-CAM/agreement | 4,501.129 | 4,582.757 | 4,602.057 |
| RetinaGuard | 1.369 | 0.406 | 3.134 |
| PDF generation | 0.135 | 0.142 | 0.176 |
| Full pipeline | 36,785.044 | 5,651.577 | 89,996.158 |

The very high warm variance is genuine local CPU behavior, primarily in the
R2-V2 vessel stage. It is a deployment bottleneck and must be addressed by
validated runtime/hardware work in a later phase; it is not hidden or averaged
away. Three repetitions make P95 descriptive only.

## Regression checks

- Python compilation: **PASS**
- Backend/API/ML/integration/reliability tests: **44 passed**
- Frontend TypeScript lint: **PASS**
- Frontend production build: **PASS**
- Model verification: **PASS** for all three local artifacts
- Demo preflight: **PASS**
- Live end-to-end upload-to-PDF: **PASS**
- Docker version/Compose build: **NOT RUN** — Docker executable unavailable

## Remaining limitations

- The current master endpoint runs inline; a durable worker/queue is not yet
  deployed.
- Local storage and SQLite are suitable for the prototype only.
- Optional evidence is capability-dependent; neovascularization remains
  unsupported and some landmarks are heuristic/approximate.
- OOD monitoring requires an authorized reference distribution and is not a
  guarantee of unfamiliar-image detection.
- The benchmark is CPU-only and not a throughput/SLA result.
- Clinical, regulatory, privacy, security, human-factors, and prospective
  validation remain required before real clinical use.
