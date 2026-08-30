# Model and evidence pipeline

This document describes the runtime sequence. It is an engineering pipeline,
not a clinical validation report.

## Runtime stages

1. **Input validation** checks JPEG/PNG decoding, dimensions, integrity, and
   RGB/RGBA channels.
2. **Image Trust Gate** measures focus, illumination, contrast, field of view,
   exposure, and artifacts. It returns `GRADABLE`, `BORDERLINE`, or
   `UNGRADABLE` with scores and recapture guidance.
3. **Controlled enhancement** runs once for a borderline image using CLAHE,
   illumination normalization, denoising, and contrast normalization. Quality
   is reassessed and enhancement is never looped indefinitely.
4. **DR classification** predicts levels 0 through 4 and a configurable
   referable result. No inference occurs for an ungradable image.
5. **Clinical evidence** independently analyzes vessels, landmarks, and
   supported lesion modules with coarse-to-fine regions and evidence maps.
6. **Explainability** produces class-specific Grad-CAM and compares attention
   to supported lesion regions. Overlap is an engineering metric, not proof of
   causality.
7. **Self-checking** calculates calibrated confidence, uncertainty, model
   disagreement, optional OOD signal, and a versioned RetinaGuard score.
8. **Triage** returns a workflow recommendation such as recapture, human
   review, specialist review, or routine AI triage.
9. **Review/report** keeps the AI screening recommendation distinct from the
   final clinician decision and stores both in the audit trail.

## Classifier strategy

Training supports EfficientNet, ResNet, and MobileNet baselines through a
benchmarking interface. Transfer-learning weights are obtained through
official package mechanisms when authorized and available. Download or import
failure stops the run with instructions. The repository does not bundle model
weights.

Weighted loss, focal loss, weighted sampling, mixed precision, checkpoints,
early stopping, reproducible seeds, and an experimental ordinal objective are
configurable. Evaluation reports accuracy, sensitivity, specificity, precision,
recall, F1, ROC-AUC where appropriate, confusion matrices, and referable DR
sensitivity/specificity only from measured validation data.

## Model registry and versioning

Training writes a checkpoint, training configuration, metrics, dataset version,
and checksum. Runtime records classifier, backbone, preprocessing, evidence,
explainability, calibration, and RetinaGuard versions per run. Updating an
artifact does not rewrite historical results.

## Failure semantics

If a stage fails, the master run becomes `FAILED`, records the stage, exception
type, message, timestamp, and audit event, and does not synthesize downstream
results. If the Trust Gate blocks an image, downstream clinical stages are
`SKIPPED`, triage is recapture-oriented, and the frontend prevents progression
to a DR result.

## Demo boundary

The `/api/v1/demo` route is a separately gated synthetic fixture surface. Its
three scenarios demonstrate trusted referral, uncertain review, and smart
recapture. It is unavailable in production and is not a substitute for model
inference or validation.
