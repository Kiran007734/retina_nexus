# SIH Localhost Demo

This workflow uses the real local API and the registered local models. It is an
engineering demonstration, not a clinical result.

## Start

```powershell
.\scripts\start_local.ps1
```

Open `http://localhost:5173/screening/new`. The backend is available at
`http://localhost:8000`, with readiness at `/api/v1/health/ready` and OpenAPI
docs at `/docs`.

## Three quality-gate paths

The existing local APTOS images used by the validation harness are:

1. `ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png` — gradable path:
   real EfficientNet-B0 classification, evidence, Grad-CAM, RetinaGuard,
   triage, report, and PDF.
2. `ml/datasets/raw/aptos2019/train_images/005b95c28852.png` — borderline
   path: one controlled enhancement/reassessment pass, followed by recapture
   guidance when it remains blocked.
3. `ml/datasets/raw/aptos2019/train_images/000c1434d8d7.png` — ungradable
   path: severe-quality failure stops clinical AI and returns smart recapture
   guidance.

Run the compact harness:

```powershell
python scripts/validate_localhost.py
python scripts/render_localhost_report.py `
  --pytest "103 passed" `
  --compileall "PASS" `
  --frontend-lint "PASS" `
  --frontend-build "PASS" `
  --model-preflight "PASS"
```

The generated artifacts under `ml/evaluation/localhost_validation/` contain
compact summaries only. They do not store uploaded image bytes or base64
evidence maps.

## Safety language

The UI and reports distinguish AI screening recommendation from clinician
decision. RetinaGuard `TRUSTED`, `REVIEW_RECOMMENDED`, and `UNRELIABLE` states
are reliability/review signals, not guarantees. Human review remains required
where configured by the result.

