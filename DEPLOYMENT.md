# RETINA-NEXUS deployment guide

This guide describes the reproducible local/SIH prototype deployment. The
system is an engineering prototype: it is not clinically validated, a medical
device, or a security-certified service.

## 1. Requirements

- Windows PowerShell 5+ or a POSIX shell
- Python 3.11+ (the verified local runtime is Python 3.11)
- Node.js 20+ and npm
- 8 GB RAM minimum for the API shell; more is recommended when the optional
  lesion and vessel models are loaded
- Docker Desktop 24+ is optional for the compose reference deployment

The ML runtime is pinned in `backend/requirements-ml.txt`; the API-only runtime
is pinned in `backend/requirements.txt`. Frontend versions are locked by
`frontend/package-lock.json` and installed with `npm ci`.

## 2. Installation

```powershell
cd C:\path\to\retina_nexus
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements-ml.txt
cd frontend
npm.cmd ci
cd ..
```

Use a separate virtual environment if the host already manages Python
packages. No dependency command downloads datasets or model weights.

## 3. Environment setup

For the documented local launcher:

```powershell
Copy-Item backend/.env.example backend/.env
```

Set `CLASSIFIER_MODEL_PATH` to the local registered EfficientNet checkpoint.
The example uses a repository-relative path, so it works when the backend is
started from `backend/`. Set `CLASSIFIER_MODEL_SHA256` when an operator wants
an explicit checksum assertion; the model registry checksum is used otherwise.

Never commit `.env`, credentials, datasets, uploaded images, or weights. A
production-like environment must replace the example secret, database URL,
CORS origins, and local storage with managed configuration. Production startup
rejects placeholder secrets shorter than 32 characters.

## 4. Model acquisition and verification

The classifier checkpoint and optional evidence artifacts are external files.
They are intentionally ignored by Git. Verify all artifacts before a demo:

```powershell
python scripts/verify_models.py
```

The command checks artifact presence, model manifests, configured or registry
checksums, and loadability. Exit code `2` means the required classifier is not
ready. Optional lesion/vessel failures are reported as unavailable and never
replaced by a hidden heuristic model.

## 5. Dataset policy

Only authorized datasets may be placed under `ml/datasets/raw/`. The dataset
directories and processed data are ignored by Git. Follow the governance
instructions in `docs/data_governance.md`; deployment does not download data
automatically.

## 6. One-command local startup

After installation and model verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local.ps1
```

The launcher checks Python/npm, environment files, frontend dependencies, and
the required model, starts the API, waits for `/api/v1/health/ready`, starts
Vite, waits for the frontend, and prints URLs and log locations. It does not
modify models or evaluation results. Use `-SkipModelLoad` only for an artifact
presence check; readiness still requires a loadable classifier.

POSIX environments can use:

```bash
bash scripts/start_local.sh
```

The current worker executes the master screening pipeline inline. Its durable
`QUEUED`/`PROCESSING`/`COMPLETED`/`FAILED` state is ready for a future Redis
worker, but the prototype must not be advertised as asynchronous processing.

## 7. Docker startup

```powershell
Copy-Item .env.example .env
# Replace SECRET_KEY, POSTGRES_PASSWORD, and the database password.
docker compose up --build
```

The compose file is a local integration reference. It mounts `ml/weights` and
`ml/datasets` read-only rather than embedding them into images, runs Alembic
migrations through the backend entrypoint, and exposes backend/frontend health
checks. Set `CLASSIFIER_MODEL_PATH` in the root `.env` to a container-visible
path such as `/app/ml/weights/...` or `./ml/weights/...`.

Before shared deployment add TLS termination, managed secrets, least privilege,
rate limiting, image scanning, resource quotas, backups, restore testing, and
an authenticated gateway. Docker configuration is not a security certification.

## 8. Health, readiness, and demo preflight

```text
GET http://127.0.0.1:8000/api/v1/health
GET http://127.0.0.1:8000/api/v1/health/ready
```

Health reports service/database liveness. Readiness reports whether the
required classifier and report generator are available plus optional model
status. It intentionally does not expose file paths, environment variables,
or secrets.

Run the non-mutating SIH preflight after both services are running:

```powershell
python scripts/demo_preflight.py
```

The command checks dependencies, model loadability, API health/readiness,
frontend availability, and report generator importability. Demo scenarios are
synthetic fixtures only and remain gated behind development/test mode.

## 9. Input and failure behavior

Uploads accept only JPEG/PNG content with matching supported filename suffixes,
valid decoding, RGB/RGBA channels, bounded dimensions, and a configurable
pixel/file-size limit. Errors use safe codes and do not echo filesystem paths,
stack traces, secrets, or image bytes.

| Condition | Expected behavior |
| --- | --- |
| Missing classifier/configuration | Readiness `503`; classification returns service-unavailable; no grade is fabricated |
| Missing optional model | Service continues; capability is marked unavailable; RetinaGuard records the limitation |
| Checksum or manifest mismatch | Artifact is unavailable; required model blocks readiness |
| Corrupt/empty/unsupported/oversized image | Upload rejected with `4xx` error code |
| Grad-CAM or evidence failure | Run is marked with stage error; missing output is not substituted |
| PDF storage/generation failure | Report request returns service-unavailable; no fake PDF is emitted |
| Pipeline timeout/concurrency saturation | Run fails safely with a durable error; no partial prediction is promoted |

Each request receives an `X-Request-ID`. Logs record route, status, stage,
duration, reliability state, and missing capability names where available; they
do not record image bytes or patient-identifying metadata.

## 10. Performance benchmark

Use a real authorized local image:

```powershell
python scripts/benchmark_pipeline.py --image ml/datasets/raw/aptos2019/train_images/0d0b8fc9ab5c.png --runs 3
```

The benchmark writes `ml/evaluation/deployment/performance_benchmark.json`
and the generated report. It records warm stage mean/median/P95, model-load
time, evidence substage timings, hardware/runtime, repetition count, and any
unavailable optional components. These are local engineering measurements, not
production-scale or clinical performance claims.

## 11. Model, clinical, and connectivity limitations

The EfficientNet-B0 output is a screening recommendation. RetinaGuard is an
engineering self-check, not a correctness guarantee. OOD monitoring requires an
authorized reference distribution; optional lesion/vessel and explanation
signals may be unavailable. The current API uses local object storage and an
inline worker. Low-bandwidth/offline operation requires the future encrypted,
resumable edge queue described in `docs/DEPLOYMENT_ARCHITECTURE.md`.

Complete clinical, privacy, security, regulatory, and human-factors review is
required before use with real patients.
