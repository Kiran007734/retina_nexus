# Deployment architecture and synchronization plan

RETINA-NEXUS is designed as a trust-first edge-to-cloud workflow. The
current Docker Compose stack is a development reference deployment; production
must add managed secrets, TLS termination, backups, observability, and a
validated identity boundary.

```text
PHC / clinic
  ├─ fundus acquisition device
  ├─ local encrypted intake queue (future offline-first adapter)
  └─ edge API / quality gate
          │ secure, resumable synchronization
          ▼
District / cloud boundary
  ├─ API gateway + authentication
  ├─ screening worker pool (Redis-backed queue boundary)
  ├─ PostgreSQL metadata and audit store
  ├─ object storage for original/enhanced images and reports
  └─ monitoring + model registry
          │
          ▼
Specialist review dashboard
```

## Runtime boundaries

- The acquisition client submits JPEG/PNG through the existing upload
  contract. It never bypasses the Image Trust Gate.
- The API persists a screening run and exposes `QUEUED`, `PROCESSING`,
  `COMPLETED`, and `FAILED`. The current runner is inline but its durable stage
  state is ready to move to a Redis/Celery or equivalent worker.
- PostgreSQL stores metadata, model versions, run state, clinical review, and
  audit events. Object storage stores image and PDF bytes. Originals should be
  immutable; enhanced derivatives are separate objects.
- The model registry is the deployment control plane. A worker loads only an
  explicitly registered, checksummed artifact and records its version for each
  run.

## Low-bandwidth and intermittent connectivity

The future clinic adapter should use an encrypted local queue with:

- an idempotency key per capture (`device_id + local_capture_id`);
- a SHA-256 content hash and image metadata before upload;
- resumable, chunked transfer with exponential backoff;
- a small preview/status payload before the original image transfer;
- explicit local states: `CAPTURED`, `VALIDATED`, `QUEUED_FOR_SYNC`, `SYNCING`,
  `ACKNOWLEDGED`, `FAILED_RETRYABLE`, and `FAILED_PERMANENT`;
- a bounded retention policy and operator-visible failed-sync queue.

The server must acknowledge a payload only after metadata and object storage
are durable. A local edge queue can run the quality gate while offline, but
clinical AI results should be marked `PENDING_SYNC` until the versioned worker
artifact and audit event are confirmed.

## Security and synchronization

Use TLS for every network hop, short-lived JWTs or device credentials, secret
management outside source control, encryption at rest, least-privilege roles,
and audit events for upload, sync, model selection, review, and export. Do not
put direct patient identifiers or image bytes in logs. Conflict resolution is
server-authoritative for screening and review decisions; the client may retry
an idempotent request but may not overwrite a clinician decision silently.

## Model version management

Every inference record should include classifier, evidence, explainability,
RetinaGuard, preprocessing, and calibration versions. Promotion should be an
explicit registry action with a rollback target. A model update must not
retroactively change stored screening results. Monitoring flags drift for
validation; it does not automatically retrain or promote a model.

## Operational deployment checklist

Before a production pilot, add a real worker process, health/readiness probes,
structured log shipping, metrics storage, object-storage lifecycle rules,
database backups and restore tests, rate limiting, vulnerability scanning,
device enrollment, and an incident runbook. The included `docker-compose.yml`
is suitable for local integration only and uses development credentials by
default.
