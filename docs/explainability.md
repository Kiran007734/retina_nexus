# Explainability and evidence verification

The explainability layer is a model-linked diagnostic view around the DR
classifier. It does not change the predicted grade, referable mapping, or
final screening decision.

## Grad-CAM

`POST /api/v1/screening/explain` loads the registered classifier artifact and
backpropagates the predicted class through its final spatial feature map. The
response includes:

- a transparent class-specific heatmap;
- a blended overlay;
- a normalized attention map;
- the predicted class and model version used to create the artifact.

If no compatible trained classifier is configured, the endpoint returns a
clear `503` setup error. It never creates a synthetic attention map.

## Attention-lesion agreement

The service compares the Grad-CAM high-attention region with supported lesion
regions from the separate retinal evidence layer. It reports IoU, Dice,
attention-in-lesion, lesion-in-attention, and a combined score with:

- `HIGH AGREEMENT` for scores at least 0.60;
- `MODERATE AGREEMENT` for scores from 0.30 to below 0.60;
- `LOW AGREEMENT` below 0.30;
- `UNAVAILABLE` when no supported lesion region exists.

This is an engineering explainability metric. It is not proof that a lesion
caused a classification, and it is not a clinical validation metric.

## Stability and counterfactuals

Stability checks are skipped by default for real-time use. Send
`run_stability: true` to run controlled brightness, rotation, and noise
perturbations. The output measures unchanged predictions and normalized
Grad-CAM similarity; it is not a robustness or clinical performance claim.

The optional `run_counterfactual: true` path masks supported lesion regions,
or the highest-attention region when lesion evidence is unavailable, then
reruns inference. This is an experimental region-masking diagnostic and a
prediction change does not establish causality.

Environment defaults are configurable with:

```text
EXPLAINABILITY_STABILITY_ENABLED=false
EXPLAINABILITY_COUNTERFACTUAL_ENABLED=false
EXPLAINABILITY_MAX_STABILITY_VARIANTS=3
```

The request flags can opt into a run without changing the production default.

## API example

```json
{
  "image_id": "...",
  "screening_session_id": "...",
  "run_stability": false,
  "run_counterfactual": false
}
```

The response includes `grad_cam`, `lesion_evidence_map_data_uri`,
`attention_lesion_agreement`, `explanation_stability`, and `counterfactual`.
Explainability artifacts are persisted in `explainability_results` and linked
to the same screening session as the classifier and retinal evidence records.
