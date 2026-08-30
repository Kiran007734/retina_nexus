# Monitoring and drift preparation

The Operations Dashboard reads `GET /api/v1/monitoring/summary?days=30`.
Metrics are derived from durable `screening_runs`, stage timing, model
metadata, quality decisions, RetinaGuard signals, and clinical review rows.

## Tracked signals

- inference and total pipeline latency: mean, median, p95, and sample count;
- model version and predicted-grade distributions;
- quality decision distribution and ungradable rate;
- review rate from recorded clinician reviews;
- model disagreement rate from completed RetinaGuard ensemble signals;
- per-stage mean duration to expose acquisition, AI, or review bottlenecks;
- open review signal count and reviewed session count;
- API, database, audit, worker-mode, and auto-retraining status.

Stage timings are recorded in `screening_runs.stage_metrics`. They are
engineering telemetry, not a clinical performance claim.

## Drift signals

Three prototype detectors return `STABLE`, `FLAGGED`, or
`INSUFFICIENT_DATA`:

1. Input distribution uses the configured RetinaGuard/OOD signal rate.
2. Prediction drift compares recent and baseline halves of the observation
   window with categorical total variation distance.
3. Quality drift compares recent and baseline quality-score means.

The thresholds and minimum sample requirements are intentionally visible in
the API implementation. A `FLAGGED` result opens a validation task: inspect
camera mix, acquisition conditions, labels, reference distribution, and model
version. It does not retrain, recalibrate, or promote a model automatically.

With fewer than 10 comparable OOD observations or 20 categorical/quality
observations, the detector reports `INSUFFICIENT_DATA` rather than implying
stability. This is a monitoring prototype and cannot guarantee detection of
all distribution shifts.

## Recommended production extension

Export the JSON response to a time-series backend with a stable metric schema,
attach site/device/model dimensions, define alert ownership and retention, and
review flagged windows against approved validation data before changing any
artifact.
