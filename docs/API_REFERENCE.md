# API reference

Base URL: `/api/v1`. OpenAPI is generated at `/docs` and `/openapi.json`.
Responses containing AI values are screening artifacts, not diagnoses.

## Core workflow

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/images/upload?patient_id=&eye=` | Validate and store JPEG/PNG |
| `POST` | `/images/{image_id}/quality` | Run or retrieve Image Trust Gate result |
| `POST` | `/images/{image_id}/enhance` | Run controlled enhancement/reassessment |
| `GET` | `/images/{image_id}/content?variant=` | Retrieve original/enhanced image |
| `POST` | `/screening/run` | Orchestrate the complete screening pipeline |
| `GET` | `/screening/{screening_id}` | Poll a run artifact/status |
| `GET` | `/screening/history` | List screening history |
| `GET` | `/screening/{session_id}/result` | Read persisted screening result |

## Individual AI stages

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/screening/classify` | DR grade and referable result |
| `POST` | `/screening/analyze-structures` | Vessel, landmark, and lesion evidence |
| `POST` | `/screening/explain` | Grad-CAM, agreement, stability, counterfactual boundary |
| `POST` | `/screening/trust` | RetinaGuard score and risk factors |

The classifier returns HTTP 503 when no usable registered model is configured.
The Trust Gate returns HTTP 422 for corrupt/unsupported images and the
individual AI stages return HTTP 422/409 when quality prerequisites are not
met. These responses never contain fabricated predictions.

`POST /screening/trust` returns backward-compatible `trust_score` and
`trust_category` fields plus `reliability_score`, `reliability_state`,
`recommended_safe_action`, `assessment_status`, `available_signals`, a
machine-readable `decision_trace`, warnings, reasons, and versioned
provenance. New reliability states are `TRUSTED`,
`REVIEW_RECOMMENDED`, `UNRELIABLE`, and `INSUFFICIENT_EVIDENCE`; legacy
`UNCERTAIN` values may appear in older stored runs. `TRUSTED` is an engineering
operating state, not a correctness or clinical safety guarantee.

Under the current `retinaguard-v3-graceful-degradation` policy, image quality,
calibrated confidence, and uncertainty are core signals. Optional lesion,
explanation-stability, model-agreement, and OOD capabilities may be
`NOT_AVAILABLE`/`UNAVAILABLE`; the response remains explicit and routes to
`REVIEW_RECOMMENDED`. A core missing signal or reported pipeline failure is
`INSUFFICIENT_EVIDENCE`.

## Human review and reports

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/reviews/queue` | Review signals requiring clinician attention |
| `GET` | `/reviews/{session_id}` | Review history |
| `POST` | `/reviews/{session_id}` | Approve, modify, reject, or request recapture |
| `GET` | `/reports` | List generated reports |
| `POST` | `/reports/generate` | Generate a report from a screening session |
| `GET` | `/reports/{report_id}` | Read report payload |
| `GET` | `/reports/{report_id}/pdf` | Download the prototype PDF |

Review writes require an authenticated clinician or administrator. The report
labels AI recommendation and clinician decision separately.

## Governance and operations

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/datasets` and dataset subroutes | Dataset registry/statistics/validation |
| `GET` | `/models` | Registered model artifacts |
| `GET` | `/analytics/overview` | Screening and queue overview |
| `GET` | `/monitoring/summary?days=30` | Latency, distributions, rates, queues, drift flags |
| `GET` | `/health` | API/database liveness |
| `GET` | `/health/ready` | Required-model/report readiness and optional capability status |

Monitoring drift states are `STABLE`, `FLAGGED`, or `INSUFFICIENT_DATA`.
`FLAGGED` opens a validation task; it does not retrain or promote a model.

Every request receives an `X-Request-ID` response header. The backend emits
structured JSON logs keyed by that identifier, with route/status/duration and
screening-stage fields but without image bytes or patient-identifying data.
Errors use safe `error_code` values and do not expose filesystem paths,
secrets, or stack traces.

Request-schema failures return HTTP `422` with
`error_code: "REQUEST_VALIDATION_FAILURE"`. In development and test they also
include sanitized `validation_errors` entries containing only `loc`, `msg`, and
`type`; raw request values are never returned. The frontend maps failures to
`REQUEST_VALIDATION_FAILURE`, `API_CONNECTION_FAILURE`, `MODEL_UNAVAILABLE`,
`INFERENCE_FAILURE`, `QUALITY_GATE_REJECTION`, or `INTERNAL_SERVER_ERROR` so a
pre-inference request rejection is not presented as a model failure.

## Controlled demo mode

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/demo/scenarios` | List three synthetic walkthroughs |
| `POST` | `/demo/scenarios/{scenario_id}/run` | Return a labeled, non-persistent fixture |

These endpoints return 404 unless `DEMO_MODE_ENABLED=true` and
`ENVIRONMENT=development` or `test`. They must not be enabled in production.
