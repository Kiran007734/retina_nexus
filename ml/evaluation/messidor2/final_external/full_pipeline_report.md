# Messidor-2 resumable parallel full-pipeline run

Status: **BENCHMARK_INTERRUPTED**

This runner composes the frozen production services only. It does not tune or replace models, thresholds, preprocessing, or production behavior.

- Evaluation scope: **BENCHMARK_SUBSET** (`0` of `1744` source jobs returned terminal results).
- Bounded jobs submitted: `1`.
- Workers: `2`; torch threads per worker: `1`.
- Result: the five-minute execution limit was reached before the full pipeline result returned.
- No prediction was substituted and no clinical metric was calculated.

The resumable runner remains available. A later run can resume from terminal cache entries, but this run did not create any terminal cache entries. No official test images were opened, no external labels were used for optimization, and no clinical validation claim is made.
