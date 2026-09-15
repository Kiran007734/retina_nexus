# RETINA-NEXUS / RetinaGuard Final Live Localhost Demo

Run date: `2026-09-15`

This is an engineering prototype validation. It is not clinical validation,
diagnosis, regulatory approval, or a clinical trust guarantee.

## Live URLs

- **Primary frontend:** http://localhost:5173
- Backend: http://localhost:8000
- Readiness: http://localhost:8000/api/v1/health/ready
- API docs: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/openapi.json

## Startup

The required processes were already running and were verified without restart:

```text
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
vite --configLoader runner --host 127.0.0.1 --port 5173
```

Both processes remained running after validation.

## Service and model preflight

All service checks returned HTTP 200:

- Frontend root and SPA routes: `/`, `/screening/new`, `/screening/results`
- Backend root: `/`
- Readiness: `/api/v1/health/ready`
- Documentation: `/docs` and `/openapi.json`

Model preflight: **PASS / READY**.

| Runtime artifact | Version | SHA-256 | Status |
|---|---|---|---|
| APTOS EfficientNet-B0 | `efficientnet-b0-aptos2019-20260830-v1` | `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b` | Loaded and verified |
| Lesion segmentation | `fundus-lesions-unet-seresnext50-all-v1` | `a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2` | Loaded and verified |
| R2-V2 vessel segmentation | `r2-v2-bv-2025` | `ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a` | Loaded and verified |

No model was retrained, replaced, downloaded, or modified.

## Real-image validation

Existing local APTOS images were used; no dataset was downloaded.

| Case | Source | Quality | Enhancement | Clinical AI | Result |
|---|---|---|---|---|---|
| Gradable | `04efb1a284cc.png` | `GRADABLE`, score `0.7783` | None | Completed | Primary result plus optional evidence |
| Borderline | `005b95c28852.png` | `BORDERLINE`, final score `0.7622` | One controlled pass | Blocked | Recapture; no prediction fabricated |
| Ungradable | `000c1434d8d7.png` | `UNGRADABLE`, score `0.6238` | None | Blocked | Severe blur/low contrast; recapture |

Gradable screening ID: `604c729f-d103-46e4-8e13-993c28aed9fc`.

The actual browser flow also passed: upload image, fill patient context, click
`Run secure screening`, render `Primary screening complete`, open the results
page, and display the real classifier result.

## Disease grading

EfficientNet-B0 completed real inference.

- Grade: `0` / `No DR`
- Probabilities: `P0=0.973709`, `P1=0.014758`, `P2=0.004699`, `P3=0.004004`, `P4=0.002830`
- Severity rule: primary `argmax(P0...P4)`
- Referable probability: `0.011533`
- Referable: `false`
- Raw confidence: `0.973709`
- Calibrated confidence: `0.973709` (`temperature-scaling-unfitted`)
- Uncertainty: `0.073107`
- Model version: `efficientnet-b0-aptos2019-20260830-v1`
- Production fusion: `DISABLED`; severity remained primary classifier output.

## Evidence and explainability

- Lesion detection: **PASS** — lesion model inference completed; masks were returned for supported lesion modules. Counts for this image were zero for the detected lesion classes and are not interpreted as clinical absence.
- Vessel segmentation: **PASS** — R2-V2/RRWNet inference completed; vessel mask returned, count `99`, confidence `0.815`.
- Coarse-to-fine evidence: **PASS** — global context, 16 suspicious proposals, and high-resolution patches were recorded.
- Grad-CAM: **PASS** — predicted class `0`, heatmap, overlay, and normalized attention map returned.
- Attention-lesion agreement: **PASS** — score `0.0`, category `LOW AGREEMENT`; engineering explainability metric only.
- Neovascularization: explicitly `UNSUPPORTED`; no fabricated output.
- Optic disc/fovea: **LIMITED** — the runtime registry surfaced `experimental_heuristic` and `approximate` implementations, but this real response did not contain dedicated landmark coordinates in the evidence payload. No landmark evidence was claimed.

Supporting evidence did not rewrite the severity grade.

## RetinaGuard and triage

RetinaGuard completed with:

- Trust score: `0.601936`
- State: `REVIEW_RECOMMENDED`
- Safe action: `PROFESSIONAL_REVIEW_RECOMMENDED`
- Triage: `HUMAN_REVIEW_REQUIRED`

The warnings explicitly identified unavailable real-time stability/OOD/model
agreement signals. The existing safe-failure validation also covers `TRUSTED`,
`REVIEW_RECOMMENDED`, `UNRELIABLE`, and `INSUFFICIENT_EVIDENCE` without
substituting predictions.

## Reports and PDF

- Report generation: HTTP `201`.
- PDF export: HTTP `200`, `application/pdf`, `2058` bytes, valid `%PDF` header and `%%EOF`.
- Browser PDF export: **PASS** after correcting the frontend URL construction.
- Final browser URL: `http://localhost:8000/api/v1/reports/{report_id}/pdf`.
- Frontend/API/report values were consistent for grade, referable status, confidence, model version, and RetinaGuard state.

The corrected frontend file is `frontend/src/pages/ResultsPage.tsx`. The backend
returns a root-relative `/api/v1/reports/.../pdf` URL; the previous frontend
concatenation duplicated `/api/v1`. The fix uses the existing `reportPdfUrl`
helper and does not change the backend or screening logic.

## Security and safe failure

- Invalid image: HTTP `422`, `INVALID_IMAGE`.
- Declared MIME/content mismatch: HTTP `415`, `UNSUPPORTED_MEDIA_TYPE`.
- Oversized upload: HTTP `413`, `PAYLOAD_TOO_LARGE`.
- Traversal filename probe: safely stored with a generated storage ID.
- Optional evidence unavailable: explicit unavailable/partial states; no fake masks or predictions.
- Controlled two-request concurrency probe: both screening requests HTTP `200`.

## Performance

Measured on the local CPU runtime; these are engineering timings only.

| Component | Time |
|---|---:|
| Image validation | `99.9 ms` |
| Quality assessment | `237.8 ms` |
| DR classification | `768.0 ms` |
| Uncertainty | `18.7 ms` |
| Model disagreement | `18.2 ms` |
| RetinaGuard | `24.0 ms` |
| Triage | `18.8 ms` |
| Lesion inference | `2,759.4 ms` |
| Vessel inference | `78,631.0 ms` |
| Optional evidence stage | `82,756.3 ms` |
| Grad-CAM | `2,664.5 ms` |
| Attention agreement | `2,702.6 ms` |
| Full gradable run elapsed | approximately `87.2 s` |
| Report generation | `113.4 ms` |
| PDF generation/request | `1,637.7 ms` |

Slowest component: **R2-V2 vessel inference on CPU**, approximately `78.6 s`.
No performance optimization was performed during this validation.

## Regression

- Backend tests: **PASS**, `103 passed in 63.05s`.
- Python compileall: **PASS**.
- Model preflight: **PASS / READY**.
- Frontend lint/typecheck: **PASS** (`tsc --noEmit`).
- Frontend production build: **PASS** (Vite).
- Git diff check: **PASS**.
- Browser results/PDF flow: **PASS** with no console errors or failed responses after the URL fix.

## Limitations

- Browser automation was performed with the locally available Playwright runtime and Chrome; this is not a clinical usability study.
- CPU vessel inference is slow and remains the main local bottleneck.
- Optic-disc/fovea outputs are not currently emitted as dedicated landmark
  evidence in this real run.
- RetinaGuard signals are engineering review/abstention signals, not a medical
  diagnosis guarantee.
- This demo used real local project images and real configured checkpoints;
  results must not be interpreted as clinical performance.

## Final live status

The frontend and backend remain running on localhost after validation.

**OPEN THIS URL:** http://localhost:5173

