# Project architecture

RETINA-NEXUS is organized as a layered screening system. Each layer has a
separate contract, persisted output, and safety boundary.

## System flow

```text
Clinic / PHC
  -> image acquisition and upload
  -> Image Trust Gate
      -> UNGRADABLE: recapture guidance, no clinical AI
      -> BORDERLINE: one controlled enhancement and reassessment
      -> GRADABLE
  -> DR classifier
  -> clinical evidence modules
  -> explainability and evidence verification
  -> RetinaGuard self-check
  -> triage recommendation
  -> clinician review
  -> report and audit trail
```

## Repository boundaries

| Boundary | Responsibility |
| --- | --- |
| `frontend/` | React workflow, result visualization, admin and clinician surfaces |
| `backend/app/api/` | Versioned FastAPI contracts and HTTP validation |
| `backend/app/services/` | Composition root and master orchestration |
| `backend/app/ml/quality/` | Image validation, quality metrics, controlled enhancement |
| `backend/app/ml/inference/` | Registered classifier loading and inference |
| `backend/app/ml/evidence/` | Separate retinal structure and lesion evidence |
| `backend/app/ml/explainability/` | Grad-CAM, overlap, stability, counterfactual boundary |
| `backend/app/ml/trust/` | Calibration, uncertainty, disagreement, OOD preparation, RetinaGuard |
| `ml/` and `scripts/` | Dataset governance, training, evaluation, and artifact preparation |
| `simulink/` | Operational capacity-planning digital twin specification |

## Persistence

PostgreSQL is the shared deployment target and SQLite is supported for local
development. Screening runs store quality, classification, lesions,
explainability, RetinaGuard, triage, model versions, per-stage timing, and
stage errors. Audit records capture run initiation, stage transitions,
completion/failure, report generation, and clinician review.

Image and PDF bytes use the storage abstraction. Original images are kept
separate from enhanced derivatives. The local adapter is path-safe; an
S3-compatible adapter is available for a managed deployment.

## Runtime contracts

`POST /api/v1/screening/run` returns the mandatory primary result inline and
persists status as `QUEUED`, `PROCESSING`, `COMPLETED`, or `FAILED`. Optional
retinal evidence and explainability run in a bounded in-process worker after
the response; `primary_status`, `evidence_status`, per-stage status, budgets,
and provenance are exposed by `GET /api/v1/screening/{id}`. This worker is a
local prototype boundary and can move behind Redis/Celery later without
changing the master API contract.

Registered model artifacts are loaded only when explicitly configured. Missing
or invalid artifacts return a setup error instead of a fabricated prediction.

## Security and safety boundaries

- Patient identifiers are anonymized at the application boundary.
- Uploads are MIME, size, integrity, format, dimensions, and channel checked.
- Authentication and clinician/admin role checks protect review decisions.
- Production secrets are environment/secret-manager inputs, never source values.
- Demo scenarios are disabled by default and do not write clinical records.
- AI output is a screening recommendation; a clinician owns the final decision.
