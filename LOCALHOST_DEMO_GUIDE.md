# RETINA-NEXUS Localhost Demo Guide

This guide exercises the real local prototype. It does not enable demo mode,
generate synthetic clinical predictions, or replace registered model weights.

## Start

From the repository root in PowerShell:

```powershell
.\scripts\start_local.ps1
```

The script requires `backend/.env`, the existing frontend dependencies, and
the registered model preflight to pass. It starts:

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs
- Readiness: http://localhost:8000/api/v1/health/ready

## Real-image workflow

Open `http://localhost:5173/screening/new`, upload an existing retinal image,
and select **Run secure screening**. The frontend uses the real API client:
the multipart field is `image`, the browser owns the multipart boundary, and
the master request is `POST /api/v1/screening/run`.

For local validation, these already-present APTOS images exercise three
quality-gate paths:

| Case | Image | Expected flow |
|---|---|---|
| Gradable | `ml/datasets/raw/aptos2019/train_images/04efb1a284cc.png` | Real classifier, evidence, RetinaGuard, triage, report/PDF |
| Borderline | `ml/datasets/raw/aptos2019/train_images/005b95c28852.png` | One controlled enhancement pass, reassessment, recapture if still blocked |
| Ungradable | `ml/datasets/raw/aptos2019/train_images/000c1434d8d7.png` | Clinical AI stops and smart recapture guidance is returned |

These images remain local ignored dataset files and are not committed.

## Automated validation

```powershell
python scripts/validate_localhost.py
python scripts/render_localhost_report.py `
  --pytest "103 passed" `
  --compileall "PASS" `
  --frontend-lint "PASS" `
  --frontend-build "PASS" `
  --model-preflight "PASS"
```

The validation artifacts are written to
`ml/evaluation/localhost_validation/`. They contain compact summaries only;
uploaded image bytes and base64 evidence maps are not written to the reports.

## Stop

The startup script prints the process IDs it launches. Stop only those local
processes when the demo is complete. Do not delete model weights, datasets, or
the local database unless a separate maintenance task explicitly requires it.

## Safety note

The prototype output is an AI screening/review aid, not a diagnosis, clinical
trust guarantee, or regulatory approval. Human review remains required for
uncertain or unreliable cases.
