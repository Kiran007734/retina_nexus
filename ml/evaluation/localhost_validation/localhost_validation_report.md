# RETINA-NEXUS Localhost Validation Report

Validation window: `2026-09-15T04:45:19.180852+00:00` to `2026-09-15T04:48:30.240714+00:00`

This is an engineering integration report. It is not a clinical validation or regulatory approval claim.

## Verdict

`LOCALHOST PROTOTYPE READY: NO`

The backend, frontend assets, real-image API pipeline, reports, PDF, model preflight, and automated regression checks passed. Full interactive browser click-through could not be independently automated because no browser automation tool is installed in this environment; the UI contract, SPA routes, asset delivery, and production build were verified instead.

## Service and model checks

- Backend readiness: HTTP `200`, `backend_ready=see response`.
- Frontend root: HTTP `200`.
- API docs/OpenAPI: HTTP `200` / `200`.
- CORS preflight: HTTP `200`, origin `http://localhost:5173`.
- Model preflight: `PASS (classifier, lesion, vessel checksums/loadability)`.
- Production ML defaults were not changed by this validation pass.

## Real-image cases

| Case | Source | Quality | Screening | Primary | Evidence | Grade | Referable | Trust |
|---|---|---|---|---|---|---:|---|---|
| gradable | `ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png` | GRADABLE | COMPLETED | COMPLETED | AVAILABLE | 0 | False | REVIEW_RECOMMENDED (0.601936) |
| borderline | `ml/datasets/raw/aptos2019/train_images/005b95c28852.png` | BORDERLINE | COMPLETED | QUALITY_BLOCKED | NOT_RUN | None | None | None (None) |
| ungradable | `ml/datasets/raw/aptos2019/train_images/000c1434d8d7.png` | UNGRADABLE | COMPLETED | QUALITY_BLOCKED | NOT_RUN | None | None | None (None) |

The gradable case was processed by the registered EfficientNet-B0 classifier and optional evidence modules. Borderline and ungradable cases were blocked from clinical AI after the quality gate and returned recapture-oriented flow results.

## Reports and artifacts

- Report generation: HTTP `201`, report ID `1a1e7dff-f5ec-4887-a17a-2f4e856b4fa2`.
- PDF: HTTP `200`, `2058 bytes`, header `True`, EOF `True`.
- Artifact safety: `True`; reports contain summaries rather than image bytes.

## API contract and security probes

- Frontend upload contract: `POST` `/images/upload`, field `image`, browser-managed multipart boundary `True`.
- Frontend run contract: `POST` `/screening/run` with `application/json`.
- Invalid image: HTTP `422` (expected rejection).
- Mismatched decoded MIME: HTTP `415` (expected rejection after upload hardening).
- Oversized transport fixture: HTTP `413` (expected rejection).
- Traversal filename probe: HTTP `201`; generated storage ID `True`; server-side storage uses generated IDs, not client paths.

## Controlled concurrency

- Requested workers: `2`; all requests HTTP 200: `True`.
- Note: Controlled two-request probe; optional evidence was not polled for these probe sessions.

## Regression checks

- Pytest: `103 passed in 63.05s`.
- Python compileall: `PASS`.
- Frontend lint/typecheck: `PASS (tsc --noEmit)`.
- Frontend production build: `PASS (Vite production build)`.
- Browser automation: `False`; Open http://localhost:5173/screening/new and repeat the gradable upload if interactive browser evidence is required.

## Root cause fixed during this pass

The upload route validated decoded image bytes but accepted a declared media type that disagreed with the decoded format. A valid PNG sent as `image/jpeg` could therefore enter storage with inconsistent metadata. The route now rejects that request with HTTP 415 and `UNSUPPORTED_MEDIA_TYPE`; normal frontend uploads continue to use the browser-generated multipart boundary and the `image` field.

No model weights, model architecture, inference thresholds, quality thresholds, RetinaGuard rules, datasets, or evaluation metrics were changed.
