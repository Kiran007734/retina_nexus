# Phase 5.1 RetinaGuard Reliability Usability Audit

Generated: `2026-09-01T17:15:54.666680+00:00`

## Scope and safety statement

This audit evaluates the existing RetinaGuard decision layer against actual
local APTOS and Messidor-2 image bytes and the authorized Phase 4B prediction
CSV. It does not train or tune a model, does not establish clinical validity,
and does not treat `TRUSTED` as proof that a prediction is correct. `TRUSTED`
means that no configured major warning was detected among the checks available
to that run.

## Audit result

The Phase 5 result was operationally safe but too conservative in one specific
way: the legacy policy treated optional attention-lesion agreement and OOD
monitoring absence as required evidence. That produced `1444`
`INSUFFICIENT_EVIDENCE` results in the Messidor-2 retrospective run even
though the core quality/classifier/uncertainty path completed and no pipeline
failure was recorded. The quality gate itself produced
`1711` BORDERLINE
results, so the high borderline rate is a separate conservative quality-gate
finding.

Phase 5.1 keeps optional-signal warnings and routes those results to
`REVIEW_RECOMMENDED`; it does not convert them to `TRUSTED`, remove high-risk
warnings, alter probabilities, or alter the referable rule. Updated states are:

* Phase 5: `{"INSUFFICIENT_EVIDENCE": 1444, "UNRELIABLE": 300}`
* Phase 5.1: `{"REVIEW_RECOMMENDED": 1444, "UNRELIABLE": 300}`

## Quality distributions

APTOS measured `5590` local images
(`5590` readable). Messidor-2 measured
`1744` images (`1744`
readable). The complete descriptive statistics, percentiles, issue counts,
and histograms are in `quality_distribution_audit.json`.

The controlled demo has no known-good image bytes; its fixture trust scores are
reported separately as synthetic values only and are not included in dataset
statistics.

## Root-cause classification

* **Scientifically correct:** missing OOD reference data must be reported as
  unavailable; real-time explanation stability is intentionally optional; the
  classifier source probabilities and high uncertainty warnings remain intact.
* **Conservative behavior:** the Image Trust Gate's any-issue-to-BORDERLINE
  rule and the legacy optional-capability requirement drove the observed state
  distribution.
* **Dependency/capability limitation:** no authorized OOD reference was
  configured, no lesion evidence/attention agreement was present in the Phase
  4B prediction-only records, and stability was not run in real time.
* **Pipeline failure:** none was recorded in the audited Messidor-2 rows.

The full machine-readable per-image trace is in `reliability_state_trace.json`.
Each row records signal availability, legacy and updated state, warning codes,
assessment status, optional capability limitations, and pipeline-failure
classification.

## Threshold decision

The current quality thresholds remain unchanged. The threshold rationale and
the actual APTOS/Messidor component distributions are in
`quality_threshold_rationale.json`. A labelled gradability study is required
before recalibration; this audit alone cannot justify relaxing a safety gate.

## False-negative safety audit

The Phase 4B referable false-negative rule identified
`278` cases. The before/after
warning and state comparison is in `false_negative_safety_comparison.json`.
The policy change preserves high-severity warning coverage and only changes
the label applied when optional capabilities are unavailable.

## Recommended next steps

1. Keep Phase 5.1's versioned graceful-degradation configuration for current
   operation and display unavailable checks explicitly in the UI.
2. Provide an authorized reference distribution for OOD monitoring before
   interpreting OOD status as available.
3. Run lesion evidence and explanation stability when the workflow requests
   those capabilities; do not infer them from absent records.
4. Collect independent gradability labels before changing quality thresholds.

This remains an engineering readiness/usability audit and is not a clinical
validation or regulatory approval claim.
