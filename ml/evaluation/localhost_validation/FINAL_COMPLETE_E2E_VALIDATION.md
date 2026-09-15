# RETINA-NEXUS final complete E2E validation

Validation date: 2026-09-15

This report records one real local run. It is an engineering/integration
verification record and does not establish clinical validity or performance.

## Live environment

- Frontend: `http://127.0.0.1:5173/` — HTTP 200
- Backend: `http://127.0.0.1:8000/` — HTTP 200
- Readiness: `http://127.0.0.1:8000/api/v1/health/ready` — HTTP 200
- API docs: `http://127.0.0.1:8000/docs` — HTTP 200
- Model verification: `READY`; required classifier and configured optional
  model artifacts were present, checksum-valid, and loadable.

## Real input

- Fixture: `ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png`
- SHA-256: `440b3ad644a4c0b250809522ab7a20f4755dd1ea63dc3355d346b5621c76567e`
- Decoded metadata: 1050 × 1050, RGB, PNG
- Screening ID: `4e625cce-89f3-4d00-8740-82eacf95ed40`

## Result and pipeline state

- Quality: `GRADABLE`, GREEN, score `0.7783`, no enhancement, AI eligible
- DR grade: Level 0, No DR
- Class probabilities: No DR `0.973709`; Mild `0.014758`; Moderate
  `0.004699`; Severe `0.004004`; Proliferative DR `0.002830`
- Referable DR: `false`; referable probability `0.011533`
- Raw confidence: `0.973709`
- RetinaGuard: score `0.627420`, category `UNRELIABLE`
- Triage: `RECAPTURE_OR_SPECIALIST_REVIEW` (high priority), because the
  engineering reliability signals included low attention/evidence agreement,
  skipped stability testing, and no configured OOD reference distribution.
- Final state: `FINAL_RESULT_READY`
- Evidence status: `AVAILABLE`
- No stage errors were recorded.

Observed lifecycle:

`PRIMARY_RESULT_READY → EVIDENCE_PROCESSING → FINAL_RESULT_READY`

The public API did not expose a terminal completed state while optional
evidence was still processing.

## Model and evidence provenance

| Component | Version | SHA-256 | Result |
|---|---|---|---|
| DR classifier | `efficientnet-b0-aptos2019-20260830-v1` | `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b` | real inference |
| Lesion evidence | `fundus-lesions-unet-seresnext50-all-v1` | `a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2` | real model inference |
| Vessel evidence | `r2-v2-bv-2025` | `ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a` | real model inference |

The run also returned optic-disc and fovea supporting heuristics, while
neovascularization remained explicitly unsupported. The lesion/vessel/evidence
modules support the classifier and do not alter its predicted severity grade.

Grad-CAM was available. Attention-lesion agreement was computed as an
engineering explainability metric: score `0.0`, `LOW AGREEMENT`; this is not
proof of clinical causality. Stability was explicitly `SKIPPED` because the
real-time request disabled perturbation testing.

## Timing

- Upload: `0.298 s`
- Primary screening request: `2.309 s` total request time
- Primary screening stage: `1781.947 ms`
- Vessel/evidence stage: `328849.450 ms` recorded stage duration
- Lesion stage: `328877.784 ms` recorded stage duration
- Grad-CAM: `5679.846 ms`
- Attention agreement: `5749.489 ms`
- End-to-end polling completion: `342.256 s`

## Report and PDF

- Report generation: HTTP 201 after `FINAL_RESULT_READY`
- PDF retrieval: HTTP 200
- Content type: `application/pdf`
- PDF size: 2059 bytes

The API correctly rejected premature report generation with `409` and
`REPORT_NOT_READY` while optional stages were pending. The final report was
generated only after all optional stages reached a terminal state.

## Reliability safeguards verified

- Missing RetinaGuard signals are represented as unavailable/not-run and have
  `score: null`, `contribution: 0`, and no fabricated fallback score.
- Available factor weights are renormalized and the policy is logged as
  `excluded_and_weight_renormalized`.
- Primary classifier grade and probabilities remain separate from RetinaGuard.
- Optional evidence timeout/failure paths close to an explicit final state and
  explain that unavailable evidence is not negative evidence.
- Clinical UI types no longer require displaying internal model names.

## Verification status

- Python compilation: PASS
- Backend tests: PASS — `105 passed`
- Focused regression after changes: PASS — `16 passed`
- Frontend TypeScript lint: PASS
- Frontend production build: PASS
- Live localhost health/readiness/docs checks: PASS
- Real image upload and primary inference: PASS
- Real evidence completion including Grad-CAM: PASS
- Final report and PDF generation: PASS

The repository remains uncommitted by instruction. No model weights, datasets,
clinical thresholds, or production classifier behavior were changed.

Full machine-readable output is in
`ml/evaluation/localhost_validation/final_e2e_live_result.json`.
