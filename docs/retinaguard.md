# RetinaGuard self-checking engine

RetinaGuard is a transparent decision and evidence-fusion layer. It is not a
second neural network and it does not replace the DR classifier, Image Trust
Gate, retinal evidence modules, or clinician review.

## Signals

The engine consumes:

- Image Trust Gate quality score;
- classifier raw and calibrated confidence;
- predictive uncertainty from normalized entropy and probability margin;
- optional model ensemble disagreement;
- supporting lesion evidence strength;
- attention-lesion agreement from the explainability layer;
- explanation stability when perturbation tests have run;
- an OOD in-distribution score from an authorized reference distribution.

Missing or not-run signals are returned in the factor list, receive the
configured `missing_signal_score`, and create a risk flag. Raw confidence is
kept separate from calibrated confidence.

## Calibration and uncertainty

Runtime calibration uses configurable temperature scaling. A temperature must
be fitted on held-out calibration data and its version recorded before it can
be treated as a calibrated model quantity. The default temperature is `1.0`
with `fitted=false`, which is an identity/unfitted configuration, not a
calibration claim.

Uncertainty combines normalized predictive entropy and inverse top-two
probability margin with disclosed component weights. An MC-dropout provider
is implemented on the registered PyTorch classifier and can be enabled with
`RETINAGUARD_MC_DROPOUT_ENABLED=true`; its sample count is controlled by
`RETINAGUARD_MC_DROPOUT_SAMPLES`. It is not run by default because it adds
multiple inference passes.

## Disagreement and OOD

When at least two model predictions are supplied to `/screening/trust`,
disagreement combines pairwise severity distance and majority disagreement.
Large grade gaps, such as No DR versus Severe, increase review priority.

OOD monitoring uses robust feature z-scores and RMS distance against an
authorized reference JSON. No reference distribution is bundled. Build one
only from approved data:

```powershell
python scripts/build_ood_reference.py --input <authorized-feature-vectors.jsonl> --output <reference.json>
```

Configure `RETINAGUARD_OOD_REFERENCE_PATH`. If it is absent, OOD is returned as
`UNAVAILABLE`; the engine does not claim to detect all unfamiliar medical
images.

## Score and decision policy

The default normalized weights are:

```text
quality                         0.20
calibrated_confidence           0.20
inverse_uncertainty             0.15
model_agreement                 0.10
lesion_evidence                 0.10
attention_lesion_agreement      0.15
explanation_stability            0.05
ood                             0.05
```

Every response returns these weights, contributions, thresholds, calibration
version, and engine version. They are configurable through `RETINAGUARD_*`
settings; there are no hidden weights.

- `TRUSTED`: score meets the configured trusted threshold and no risk flags or
  required signals are missing. AI triage may proceed with human oversight.
- `REVIEW_RECOMMENDED`: the core assessment completed, but a review limitation
  or non-major warning remains. An optional capability being unavailable is
  disclosed and routes here; it is not silently treated as a positive signal.
- `UNRELIABLE`: automated interpretation is blocked or recapture/specialist
  review is required, especially for low quality, high disagreement, high
  uncertainty, low agreement, or detected distribution shift.
- `INSUFFICIENT_EVIDENCE`: a core signal is missing or a pipeline failure was
  reported, so a complete reliability assessment could not be made. It is
  never silently treated as a positive signal and requires professional review.

`UNCERTAIN` remains accepted when reading legacy stored runs, but new engine
outputs use `REVIEW_RECOMMENDED` and `INSUFFICIENT_EVIDENCE` explicitly.

Safe actions are returned separately from the human-readable message:

- `AUTOMATED_RESULT_AVAILABLE`
- `PROFESSIONAL_REVIEW_RECOMMENDED`
- `AUTOMATED_INTERPRETATION_UNRELIABLE`
- `IMAGE_RECAPTURE_RECOMMENDED`

These categories are engineering operating decisions. They are not medical
diagnoses, clinical validation results, or guarantees of model correctness.

## API and persistence

Call `POST /api/v1/screening/trust` with an image ID and optional screening
session ID. Optional `model_predictions` can provide additional registered
model outputs for disagreement measurement. The endpoint returns
`trust_score`, `trust_category`, `contributing_factors`, `risk_flags`,
`recommended_action`, calibration, uncertainty, disagreement, OOD, and the
versioned configuration.

Results are stored in `retinaguard_results` and the current screening result
also records calibrated confidence, uncertainty, and trust score. Non-trusted
categories move the session to `needs_review`. API responses retain the
backward-compatible `trust_*` fields and also expose `reliability_*`, signal
availability, warnings, reasons, safe action, and provenance fields. The
current policy is `retinaguard-v3-graceful-degradation`: quality, calibrated
confidence, and uncertainty are core signals; lesion agreement, explanation
stability, model agreement, and OOD are optional capabilities. Their absence
produces `COMPLETED_LIMITED` plus explicit `NOT_AVAILABLE`/`UNAVAILABLE`
statuses. A supplied pipeline failure produces `FAILED` and
`INSUFFICIENT_EVIDENCE`.

Phase 5.1's audit is reproducible with:

```powershell
python scripts/audit_reliability_usability.py --analysis-max-dimension 512 --workers 12
```

It writes the quality distribution, per-result reliability trace, state
comparison, threshold rationale, false-negative safety comparison, and the
versioned graceful-degradation configuration under
`ml/evaluation/reliability/`. The working-copy dimension only controls metric
performance; original files are decoded and validated first.
