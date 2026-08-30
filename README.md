# RETINA-NEXUS

RETINA-NEXUS is a privacy-conscious prototype for explainable diabetic
retinopathy screening. It separates image trust, disease classification,
clinical evidence, explainability, self-checking, triage, and human review.
It does not claim clinical validation, regulatory approval, or diagnostic
performance.

## Architecture

```text
JPEG/PNG upload
  -> Image Trust Gate
  -> DR classifier (registered artifact only)
  -> retinal evidence modules
  -> Grad-CAM and evidence agreement
  -> uncertainty and model disagreement
  -> RetinaGuard
  -> triage recommendation
  -> clinician review and report
```

The FastAPI backend owns durable workflow state, audit events, model-version
metadata, reports, and review decisions. The React frontend consumes the
versioned API. PostgreSQL is the production database target; SQLite is
supported for local development. Local object storage can be replaced by the
S3-compatible adapter. Redis is provisioned as the future worker boundary.

Read the detailed maps in [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md)
and [docs/MODEL_PIPELINE.md](docs/MODEL_PIPELINE.md).

## Installation

Requirements: Python 3.11+, Node.js 20+, npm, and optionally Docker Desktop.

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env

cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head

cd ..\frontend
npm.cmd install
```

The backend API uses `http://localhost:8000`; the frontend uses
`http://localhost:5173` in development.

## Environment setup

Copy the relevant example file for local values. Never commit `.env`,
passwords, signing keys, model weights, patient data, or uploaded images. For
a backend-only SQLite run, use `backend/.env.example`. For shared or
production environments, provide a randomly generated `SECRET_KEY` of at
least 32 characters, managed database credentials, TLS, and an external
secret manager.

Demo mode is disabled by default. To enable only the synthetic walkthrough:

```text
ENVIRONMENT=development
DEMO_MODE_ENABLED=true
```

The demo API is environment-gated, does not write clinical records, and must
not be used as a model fallback.

## Database setup

```powershell
cd backend
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Development startup creates missing tables for convenience. Controlled
deployments should run Alembic explicitly. The health endpoint is
`GET /api/v1/health`; interactive OpenAPI is at `/docs`.

## Dataset setup

Datasets are not bundled. Place only authorized copies under
`ml/datasets/raw/` and follow [docs/data_governance.md](docs/data_governance.md).
Acquisition supports environment credentials, authorized Kaggle API access,
manual placement, checksums, file counts, validation, duplicate detection,
patient-aware splits, and leakage reporting. Missing credentials or
unavailable access fails with setup instructions; no dataset is fabricated.

## Training

Install the optional ML dependencies only when needed:

```powershell
pip install -r backend/requirements-ml.txt
python scripts/validate_dataset.py --dataset aptos2019
python scripts/create_splits.py --dataset aptos2019
python scripts/train_classifier.py --dataset aptos2019 --dataset-version aptos2019-authorized-v1 --backbone efficientnet_b0
python scripts/benchmark_classifiers.py
```

Training supports EfficientNet, ResNet, MobileNet, weighted/focal loss,
weighted sampling, mixed precision, checkpoints, early stopping, seeds, and
experimental ordinal mode. Official pretrained-weight downloads fail clearly
when unavailable. No target metric is claimed without measured validation.

To install and verify the optional pretrained retinal-lesion evidence model:

```powershell
pip install -r backend/requirements-ml.txt
python scripts/acquire_lesion_model.py
python scripts/acquire_lesion_model.py --verify-only
```

See [docs/LESION_MODEL_INTEGRATION.md](docs/LESION_MODEL_INTEGRATION.md) for
the source, license, architecture, checksum, supported classes, and the
explicit Phase 3 vessel-segmentation boundary. The checkpoint is ignored by
Git and is supporting evidence only; it is not a clinical validation claim.

## Inference

Set `CLASSIFIER_MODEL_PATH` to a trained and registered artifact, then start
the API. Without a usable artifact, classification returns a clear HTTP 503
and no fake grade is produced. The master endpoint is:

```text
POST /api/v1/screening/run
GET  /api/v1/screening/{screening_id}
```

The run records stage status, errors, stage timing, model versions, audit
events, and the complete quality/classification/evidence/explainability/
RetinaGuard/triage artifact.

## Docker deployment

Create a root `.env` from `.env.example`, replace the placeholder secret and
PostgreSQL password, then run:

```powershell
docker compose up --build
```

The compose file is a local integration reference. Review
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) before any shared deployment; it
covers TLS, secrets, backups, object storage, worker migration, offline-first
synchronization, model promotion, and monitoring hardening.

## API documentation

The complete route index is in [docs/API_REFERENCE.md](docs/API_REFERENCE.md).
Important groups include image upload and quality, screening orchestration,
evidence, explainability, RetinaGuard, reviews, reports, datasets, analytics,
monitoring, and explicitly gated demo scenarios.

## Frontend instructions

```powershell
cd frontend
Copy-Item .env.example .env
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:5173`. The New Screening flow uses the upload and
master-run APIs. Results, Clinical Review, Reports, Dataset Management, and
Operations Dashboard read backend data rather than production fixtures. The
controlled synthetic walkthrough is available at `/demo` only when demo mode
is enabled.

## Verification

```powershell
pytest -q
cd frontend; npm.cmd run lint; npm.cmd run build
```

The integration test uses deterministic test doubles to exercise the HTTP
workflow without claiming model performance. See
`tests/test_api_integration.py`.

## Safety boundary

AI values are screening recommendations. Quality, confidence, trust,
explainability overlap, readiness, and drift values are engineering signals;
they are not clinical guarantees. Final clinical responsibility remains with
an authorized reviewer.
