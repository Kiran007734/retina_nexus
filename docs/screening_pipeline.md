# End-to-end screening pipeline

The master runner is `POST /api/v1/screening/run`. It composes the existing
modules; it does not replace their contracts or duplicate their model logic.
The current implementation executes in the API process, but its durable run
record and explicit stage boundaries are designed for a future queue worker.

## Stage order

```text
image_validation
  -> quality_assessment
  -> dr_classification
  -> retinal_structure_analysis + lesion_detection
  -> grad_cam + attention_lesion_agreement + uncertainty
  -> model_disagreement
  -> retinaguard
  -> triage
```

The Image Trust Gate is a hard boundary. `UNGRADABLE` images complete with a
recapture recommendation and all clinical-AI stages are marked `SKIPPED`.
`BORDERLINE` images receive at most one controlled enhancement pass and are
reassessed; only a final `GRADABLE` result proceeds to clinical AI. No
prediction is fabricated when a stage fails.

## Run state and failure behavior

The public run status is one of `QUEUED`, `PROCESSING`, `COMPLETED`, or
`FAILED`. Each run is stored in `screening_runs`, keyed to the screening session
ID so the existing session URL remains stable. The row contains stage status,
stage errors, each module response, model/preprocessing versions, timestamps,
and the initiating user when a valid authenticated subject is available.

On failure, the runner logs the exception, marks the failing stage and run as
`FAILED`, stores an error type/message, updates the screening session, and
returns the failed run artifact. Downstream results are left null; clients
must not interpret a partial run as a prediction.

## Audit trail

The runner records queueing, processing, each stage transition, completion or
failure, and the final trust/triage decision in `audit_logs`. Audit details
include an event timestamp, run ID, initiating user where available, model
versions, preprocessing version, RetinaGuard configuration version, and final
decision.

## API example

```json
{
  "image_id": "<fundus-image-uuid>",
  "run_stability": false,
  "run_counterfactual": false,
  "model_predictions": []
}
```

Stability and counterfactual analysis are opt-in because they add inference
cost. The normal production run still returns the classifier's uncertainty and
the other available evidence signals. `GET /api/v1/screening/{id}` can be
polled by a future frontend or worker-backed implementation without changing
the response shape.

Triage values are workflow recommendations only: `AI_TRIAGE_MAY_PROCEED`,
`SPECIALIST_REVIEW_RECOMMENDED`, `HUMAN_REVIEW_REQUIRED`, or
`RECAPTURE_OR_SPECIALIST_REVIEW`. They are not diagnoses or clinical
disposition guarantees.
