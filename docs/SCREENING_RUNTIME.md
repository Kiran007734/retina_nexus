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
| Retinal evidence | `SCREENING_OPTIONAL_EVIDENCE_TIMEOUT_SECONDS` | 600 s |
| Grad-CAM/agreement | `SCREENING_OPTIONAL_EXPLAINABILITY_TIMEOUT_SECONDS` | 30 s |

The values are based on persisted local measurements: classification roughly
0.6–3.3 s, completed Grad-CAM/agreement roughly 0.9–8.4 s, and combined vessel
plus lesion evidence roughly 1.4–310 s, with one prior whole-run observation
exceeding the legacy 900-second limit. These are prototype engineering
budgets, not promises and not clinical validation.

## Honest degradation

The public run lifecycle is explicit: `PRIMARY_RESULT_READY` means the
classifier, reliability assessment, and triage are ready while optional work
is queued; `EVIDENCE_PROCESSING` means optional work is active;
`FINAL_RESULT_READY` means every optional stage is terminal. Optional stages
are reported as `QUEUED`, `PROCESSING`, `COMPLETED`, `TIMED_OUT`, or
`UNAVAILABLE`. A run is never exposed as a terminal completed result while
evidence is still processing.
Timeouts include the stage, budget, reason, and `evidence_is_not_negative` in
the durable audit/status record. No mask, heatmap, lesion count, agreement
score, or other placeholder is created for work that did not complete.

The frontend polls while evidence is processing and distinguishes primary
completion from optional evidence availability. Missing RetinaGuard signals are
marked `NOT_RUN`, `NOT_AVAILABLE`, or `UNAVAILABLE`; they have no fabricated
score and no contribution, while available configured weights are
renormalized. A process restart may interrupt in-process optional work; a
supervised queue worker is the next deployment hardening step.

## Optional local Qwen secondary verifier

`Qwen/Qwen3-VL-4B-Instruct` is an optional local-only secondary visual
verification capability. It is never the primary classifier and cannot rewrite
the primary severity grade or referable result. The adapter reports
`NOT_CONFIGURED` until all official model shards, tokenizer, processor, and a
successful real image inference are present. A failed or timed-out verifier
must leave the primary screening result operational and produces no fabricated
agreement score.
