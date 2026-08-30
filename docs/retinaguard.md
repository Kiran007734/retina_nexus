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
- `UNCERTAIN`: human review is required before relying on automated triage.
- `UNRELIABLE`: recapture or specialist review is required, especially for
  low quality, high disagreement, high uncertainty, low agreement, or detected
  distribution shift.

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
categories move the session to `needs_review`.
