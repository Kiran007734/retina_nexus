# RETINA-NEXUS Architecture

RETINA-NEXUS is a research and SIH prototype for explainable diabetic
retinopathy screening. It is not a diagnostic device, clinical validation
study, or regulatory-approved system.

## End-to-end flow

```text
Fundus JPEG/PNG
  -> upload integrity and MIME validation
  -> Image Trust Gate
  -> controlled borderline enhancement and reassessment
  -> primary DR severity classifier
  -> supporting lesion, vessel, and anatomical evidence
  -> Grad-CAM and attention/evidence agreement
  -> uncertainty and model-disagreement signals
  -> transparent RetinaGuard decision layer
  -> triage recommendation
  -> clinician review
  -> audit trail and PDF report
```

The primary five-class severity result remains the classifier argmax. Supporting
evidence and RetinaGuard may recommend review or recapture; they do not
silently rewrite the severity grade.

## Runtime topology

- `backend/`: FastAPI API, SQLAlchemy persistence, Alembic migrations, image
  storage, orchestration, model adapters, report/PDF generation, review flow,
  monitoring, and readiness checks.
- `frontend/`: React/Vite clinician and administrator experience. It consumes
  `/api/v1` and does not contain production prediction fixtures.
- `ml/`: preprocessing, dataset governance metadata, model registries,
  evidence utilities, evaluation artifacts, and research manifests.
- `simulink/`: the operational digital-twin model, MATLAB scripts, scenario
  configuration, plots, and simulation results.
- `scripts/`: authorized dataset/model acquisition, validation, training,
  evaluation, deployment preflight, localhost validation, and release audit.

## Model roles

The default localhost classifier is the registered APTOS EfficientNet-B0
checkpoint. R2-V2 vessel segmentation and the published lesion model are
supporting evidence capabilities and can degrade explicitly when unavailable.
IDRiD grading, lesion/localization candidates, DRIVE research candidates, and
the independent RETGUARD verifier are research-only unless an explicit
configuration and governance decision enables them.

## Data and security boundaries

Raw datasets, local uploads, checkpoints, `.env` files, databases, logs, and
runtime caches are excluded from Git. Dataset metadata and evaluation reports
must still be reviewed for sensitive paths and redistribution constraints.
Upload validation checks decoded image type against the declared MIME type,
rejects invalid/oversized content, and stores files under generated identifiers.

Authentication/authorization for PHI-bearing routes remains a deployment
hardening requirement. Localhost CORS is deliberately limited to local origins.

## Reliability boundary

RetinaGuard is a transparent evidence-fusion and review-priority engine. Its
score, quality state, uncertainty, evidence agreement, and OOD availability
are engineering signals. They are not a clinical trust guarantee or proof of
causality.
