# RETINA-NEXUS Phase 8 Final ML End-to-End Validation

## Result

**PHASE 8 — FINAL ML VALIDATION: COMPLETE**

- Representative images tested: `11`.
- Full AI pipeline completed: `2` (`18.2%`; all were terminal without a failed prediction).
- Quality-gate terminal handling: `11/11` (`100.0%`).
- Quality-blocked cases: `9`; no clinical AI output was fabricated for them.
- Fusion stored result reproduced exactly: `YES`.
- Messidor-2 used for tuning or validation: `NO`.
- Official IDRiD test images opened: `0`.
- Production fusion setting: `REFERABLE_FUSION_ENABLED=false`.

## Locked safety rules

- Severity: `argmax(P0..P4)` from the primary classifier.
- Referable research fusion: `max(primary_probability, verifier_probability) >= 0.40`.
- Evidence, Grad-CAM, and RetinaGuard do not rewrite severity.
- Missing optional evidence produces explicit unavailable/partial states.

## Artifacts

- `end_to_end_results.json` — real-image pipeline traces.
- `fusion_validation.json` — locked 406-image artifact reproduction.
- `evidence_validation.json` — lesion/vessel/localization/XAI checks.
- `retinaguard_validation.json` — real-image reliability outputs.
- `failure_fallback_results.json` — intentional safe failure cases.
- `runtime_validation.json` — focused latency measurements.

## Remaining technical issues

- The configured R2-V2 vessel stage is CPU-heavy; the slowest focused full run was recorded in `runtime_validation.json` rather than hidden.
- The selected representative set contained many genuinely borderline images that the quality gate correctly blocked after one enhancement pass; therefore only gradable images reached clinical AI.
- The research verifier has known IDRiD training-overlap risk and remains non-promoted.
- This is engineering validation, not clinical validation or a regulatory claim.

NEXT PHASE = SIMULINK / SYSTEM WORKFLOW SIMULATION
