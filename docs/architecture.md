# Architecture

## Request lifecycle

1. A healthcare worker submits an anonymized patient reference and fundus image.
2. The image route validates MIME type, file size, and decodability before storage.
3. A screening session is queued and records the image, patient, and model-version boundary.
4. An orchestrator can run quality assessment, optional enhancement, classification, lesion analysis, Grad-CAM, evidence verification, and RetinaGuard trust scoring.
5. Uncertain or referable results move to a clinician review queue.
6. A report service creates a reviewable clinical summary and audit event.

The API imports service contracts rather than concrete model code. `backend/app/services/container.py` is the composition point for swapping implementations.

## Data boundaries

PostgreSQL is the production database target, while SQLite is supported for local development. The storage protocol supports local files now and an S3-compatible adapter later. Redis is provisioned for caching and task infrastructure; asynchronous workers are intentionally not enabled in the first scaffold.

## Privacy-conscious defaults

Patient records use anonymized identifiers, minimal metadata, validated image uploads, path traversal checks in local storage, and an audit log table. Production deployments should add a managed secrets provider, network policy, encrypted object storage, key rotation, retention policy, and a reviewed access-control implementation.
