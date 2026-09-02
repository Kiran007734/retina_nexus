# Deployment guide

The repository includes a local integration deployment. A clinical or shared
deployment requires an additional security, validation, and operations review.
No regulatory or clinical deployment claim is made here.

## Local API

```powershell
Copy-Item backend/.env.example backend/.env
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Use SQLite and local storage for a single-machine development run. Do not use
the default development values on a shared network.

## Docker Compose

The root compose file runs the API, frontend, PostgreSQL, and Redis. It now
requires `SECRET_KEY`, `DATABASE_URL`, and `POSTGRES_PASSWORD` from the root
`.env`; it has no embedded database password or signing key. Start it with:

```powershell
Copy-Item .env.example .env
# Replace SECRET_KEY, POSTGRES_PASSWORD, and the password in DATABASE_URL.
docker compose up --build
```

The backend image installs the optional model-serving dependencies so an
authorized checkpoint can be mounted at `/app/ml/weights`. The repository does
not contain model weights; set `CLASSIFIER_MODEL_PATH` to a registered mounted
artifact after training and checksum review. Without that configuration,
clinical classification fails explicitly rather than returning a fallback.

Compose is a development reference. Add TLS termination, managed secrets,
network policy, non-default users, image scanning, resource limits, health and
readiness probes, and a backup/restore procedure before a shared deployment.

## Edge and intermittent connectivity

The intended topology is:

```text
PHC acquisition
  -> encrypted local intake queue
  -> edge validation / optional local quality gate
  -> resumable secure sync
  -> district/cloud API and worker pool
  -> specialist review dashboard
```

The future edge adapter should use device-plus-capture idempotency keys,
SHA-256 content hashes, encrypted at-rest queue storage, chunked transfer with
backoff, explicit retry states, bounded retention, and server acknowledgement
only after durable metadata and object storage. Conflict resolution is
server-authoritative for screening and clinician decisions.

The current HTTP worker returns the mandatory primary result inline and uses a
small in-process background task for optional evidence. Run state is durable
and queue-ready. This prototype mechanism is intentionally not a replacement
for a supervised worker: a process restart can interrupt optional work, and a
future Redis/Celery or equivalent worker can replace it without changing the
master API contract.

## Primary and optional screening runtime

`POST /api/v1/screening/run` completes the primary path after image validation,
quality assessment, EfficientNet classification, uncertainty/model checks,
RetinaGuard, and triage. Lesion/vessel evidence and Grad-CAM/agreement are
optional enrichment and are scheduled after the primary response. The status
endpoint exposes `primary_status` and `evidence_status`, as well as each
optional stage status.

The default local budgets are `60` seconds for primary screening, `240` seconds
for combined retinal evidence, and `30` seconds for explainability. They are
engineering limits selected from recorded warm-CPU timings, not clinical or
hard latency targets. A timed-out or unavailable optional stage is persisted as
`TIMED_OUT`/`UNAVAILABLE` with provenance. It never becomes a successful empty
result and is never interpreted as negative evidence. The primary result stays
valid and available for review while the optional status is shown separately.

## Model and data release controls

1. Acquire only authorized datasets and validate their manifests.
2. Generate patient-aware splits and inspect the leakage report.
3. Train and benchmark candidates on a declared dataset version.
4. Register checksum, metrics, configuration, and model version.
5. Promote explicitly with a rollback target.
6. Monitor latency, quality, prediction, review, disagreement, and drift.

Drift flags require human validation. This system never automatically retrains,
recalibrates, promotes, or changes a deployed model.

## Production checklist

- enforce a managed `SECRET_KEY` and secret rotation;
- enable TLS and short-lived device/user credentials;
- use encrypted managed object storage and immutable original images;
- apply least privilege, rate limits, audit retention, and access review;
- run Alembic migrations in a controlled release step;
- operate worker, API, database, storage, and queue health probes;
- ship structured logs and monitoring metrics without patient/image contents;
- test database, object storage, and queue restore procedures;
- validate model artifact checksums and record every runtime version;
- keep demo mode disabled;
- complete clinical, privacy, security, and regulatory review before use with
  real patients.
