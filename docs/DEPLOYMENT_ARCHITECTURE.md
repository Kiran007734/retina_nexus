# RETINA-NEXUS deployment architecture

## Boundaries

```text
Clinic / PHC
  -> acquisition client
  -> API upload + Image Trust Gate
  -> local object storage and metadata DB
  -> inline screening worker (current prototype)
  -> clinician review and PDF report

Future edge sync:
  encrypted local queue -> resumable transfer -> district/cloud API
  -> Redis worker pool -> PostgreSQL + object storage -> review dashboard
```

The current deployment is a single API process with a durable database-backed
screening run. The worker boundary is intentionally explicit so it can later be
replaced with Redis/Celery or an equivalent queue without changing the master
screening API contract.

## Startup sequence

1. Load environment settings and reject an invalid production secret.
2. Create the configured local upload/evaluation directories.
3. Create development tables and seed registries only in development/test.
4. Verify the required classifier artifact, manifest, checksum, and loadability.
5. Verify optional artifact presence and manifests; optional runtime loading is
   performed by `scripts/verify_models.py` and `scripts/demo_preflight.py`.
6. Expose liveness at `/api/v1/health` and readiness at
   `/api/v1/health/ready`.
7. Accept screening only through the validated upload and quality-gate path.

Readiness is not a clinical claim. A missing required classifier makes the
service `NOT_READY`; a missing optional model is visible as an unavailable
capability and does not result in fabricated evidence.

## Model loading and provenance

The classifier is loaded lazily by the inference service from the explicitly
configured checkpoint. Startup verification uses the same architecture and
registered model metadata. Lesion and vessel adapters are separate optional
artifacts with their own manifests, versions, and checksums. Every completed
run records model versions, preprocessing version, calibration/RetinaGuard
version, stage timing, and audit events. Historical evaluation files and
weights are never rewritten by deployment checks.

## Request and resource controls

- JPEG/PNG upload content is decoded and verified; extensions are not trusted
  alone.
- File size, dimensions, channels, and total pixels are bounded before costly
  processing.
- Master screening work is limited by `MAX_CONCURRENT_SCREENINGS` and
  `SCREENING_TIMEOUT_SECONDS`.
- Unhandled failures return a generic safe error and a request ID; internal
  diagnostics stay in structured logs.
- CORS is configured from `CORS_ORIGINS`; production settings must use an
  explicit allowlist and managed TLS/authentication boundary.
- The API does not log uploaded bytes, secrets, or patient-identifying fields.

## Failure handling

Every pipeline stage has durable status, timing, and error fields. A failed
classifier, Grad-CAM, evidence adapter, or report operation cannot create a
synthetic downstream result. Quality failures stop clinical AI and return
recapture guidance. Optional model failures produce `unsupported` evidence and
are reflected in RetinaGuard's availability trace.

## Deployment topology and synchronization plan

For low-bandwidth clinics, the future acquisition adapter should store an
encrypted bounded queue with `device_id + local_capture_id` idempotency,
content hash, chunked/resumable upload, exponential backoff, and explicit retry
states. The server should acknowledge only after metadata and image storage are
durable. Conflict resolution is server-authoritative for screening and
clinician decisions.

## Production boundary

The included Docker Compose file is a reproducible local reference. A shared
deployment still needs TLS, managed secrets, an authenticated API gateway,
least-privilege database/object-storage roles, rate limiting, backups and
restore tests, container scanning, resource quotas, metrics shipping, alerting,
and an incident runbook. None of those additions should be inferred from a
successful local health check.
