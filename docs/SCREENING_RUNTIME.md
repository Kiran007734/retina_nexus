# Screening runtime contract

RETINA-NEXUS separates the mandatory primary screening response from optional
evidence enrichment so CPU-heavy evidence cannot invalidate a classification.

## Stage ownership

Primary stages are image validation, image quality assessment, DR
classification, uncertainty estimation, model disagreement calculation,
RetinaGuard, and triage. A gradable image must finish these stages before the
API returns a primary screening result. An ungradable image completes safely
with recapture guidance and does not start clinical AI.

Optional stages are retinal structure/vessel analysis, lesion detection,
Grad-CAM, and attention-lesion agreement. They start only after a primary
classification exists and run in the local in-process background worker.

## Runtime budgets

The default configuration is:

| Path | Setting | Default |
| --- | --- | ---: |
| Primary screening | `SCREENING_PRIMARY_TIMEOUT_SECONDS` | 60 s |
| Retinal evidence | `SCREENING_OPTIONAL_EVIDENCE_TIMEOUT_SECONDS` | 240 s |
| Grad-CAM/agreement | `SCREENING_OPTIONAL_EXPLAINABILITY_TIMEOUT_SECONDS` | 30 s |

The values are based on persisted local measurements: classification roughly
0.6–3.3 s, completed Grad-CAM/agreement roughly 0.9–8.4 s, and combined vessel
plus lesion evidence roughly 1.4–310 s, with one prior whole-run observation
exceeding the legacy 900-second limit. These are prototype engineering
budgets, not promises and not clinical validation.

## Honest degradation

The run remains `COMPLETED` when primary stages finish. Optional stages are
reported as `QUEUED`, `PROCESSING`, `COMPLETED`, `TIMED_OUT`, or `UNAVAILABLE`.
Timeouts include the stage, budget, reason, and `evidence_is_not_negative` in
the durable audit/status record. No mask, heatmap, lesion count, agreement
score, or other placeholder is created for work that did not complete.

The frontend polls while evidence is processing and distinguishes primary
completion from optional evidence availability. A process restart may interrupt
in-process optional work; a supervised queue worker is the next deployment
hardening step.
