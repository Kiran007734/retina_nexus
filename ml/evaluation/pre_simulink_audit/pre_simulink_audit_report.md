# RETINA-NEXUS Pre-Simulink Final Audit

## Overall status

**PRE-SIMULINK AUDIT PASSED**

The locked image-to-report pipeline was traced using the existing backend,
Phase 8 real-image artifacts, model registries, API integration coverage, and
frontend build gates. No new dataset, training run, threshold change, model
replacement, fusion enablement, commit, or push was performed.

## Findings

### Critical

None.

### High

None after the persistence consistency fix.

### Medium

- Legacy/generated registries contain conflicting availability/selection metadata; `ml/weights/model_registry.json` is the runtime-authoritative registry.
- Route-level authentication/authorization is incomplete for PHI-bearing prototype endpoints; this is not production access control.
- The in-process optional worker can be interrupted by backend restart.
- R2-V2/evidence execution is CPU-heavy and must be modeled in Simulink.

### Informational

Dedicated neovascularization detection is not currently validated in the prototype and remains future work.

## Locked flow and safety

Severity remains `argmax(P0..P4)`. Research referable fusion remains disabled
by default and, when explicitly enabled, uses
`max(primary_probability, verifier_probability) >= 0.40`. Evidence, Grad-CAM,
and RetinaGuard do not rewrite severity. Missing optional evidence produces an
explicit unavailable/limited state.

## Real-image smoke results

- Representative images: 11.
- Full AI pipelines completed: 2.
- Quality-gate terminal handling: 11/11.
- Failed cases: 0.
- Official IDRiD test images opened: 0.
- Fusion reproduction: `PASS`.

## Readiness gates

| Gate | Status |
|---|---|
| ML | READY |
| Backend | READY |
| Frontend | READY |
| Report/PDF | READY |
| Security baseline | PARTIAL BASELINE PASS |
| Simulink | READY |

## Required boundary

This is engineering/prototype readiness evidence, not clinical validation,
diagnostic accuracy, regulatory approval, or a clinical trust guarantee.

**NEXT PHASE: SIMULINK / SYSTEM WORKFLOW SIMULATION**
