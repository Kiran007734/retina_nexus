# Authoritative Messidor-2 full-pipeline benchmark

Status: **COMPLETE**

- Requested: `50`
- Processed: `50`
- Terminal: `50`
- Failed: `0`
- Elapsed seconds: `2546.348`
- Max wall seconds: `3600.0`

This is an engineering benchmark of the existing pipeline. It does not establish clinical performance and does not change models or thresholds.

## Stage latency

```json
{
  "classification": {
    "count": 47,
    "max_ms": 280.0170000118669,
    "mean_ms": 128.5328553138707,
    "median_ms": 125.01359998714179,
    "min_ms": 77.49699999112636,
    "p95_ms": 164.5213000010699
  },
  "evidence_total": {
    "count": 47,
    "max_ms": 47491.23389998567,
    "mean_ms": 44969.496400003874,
    "median_ms": 44974.61979999207,
    "min_ms": 43660.159799968824,
    "p95_ms": 46326.74370001769
  },
  "grad_cam_and_agreement": {
    "count": 47,
    "max_ms": 6967.756300000474,
    "mean_ms": 4162.220951061364,
    "median_ms": 4109.2961999820545,
    "min_ms": 3498.38569998974,
    "p95_ms": 4728.667700022925
  },
  "image_validation": {
    "count": 50,
    "max_ms": 201.45950000733137,
    "mean_ms": 60.730890000704676,
    "median_ms": 56.222949991934,
    "min_ms": 41.70459997840226,
    "p95_ms": 88.50220002932474
  },
  "lesion_inference_ms": {
    "count": 47,
    "max_ms": 2612.687,
    "mean_ms": 2024.4673829787234,
    "median_ms": 2005.849,
    "min_ms": 1903.678,
    "p95_ms": 2106.881
  },
  "pdf_generation": {
    "count": 47,
    "max_ms": 0.2142000012099743,
    "mean_ms": 0.10888936045143674,
    "median_ms": 0.09609997505322099,
    "min_ms": 0.0801000278443098,
    "p95_ms": 0.16309996135532856
  },
  "quality_assessment": {
    "count": 50,
    "max_ms": 314.8036999627948,
    "mean_ms": 240.04018799576443,
    "median_ms": 234.9114000244299,
    "min_ms": 198.00540001597255,
    "p95_ms": 284.98530000797473
  },
  "quality_enhancement": {
    "count": 49,
    "max_ms": 4848.246800014749,
    "mean_ms": 4400.449451020496,
    "median_ms": 4399.261099984869,
    "min_ms": 3872.9361000005156,
    "p95_ms": 4808.06950002443
  },
  "retinaguard": {
    "count": 47,
    "max_ms": 0.9200999920722097,
    "mean_ms": 0.34485106044349834,
    "median_ms": 0.29090000316500664,
    "min_ms": 0.23210002109408379,
    "p95_ms": 0.6602000212296844
  },
  "structure_analysis_ms": {
    "count": 47,
    "max_ms": 46.029,
    "mean_ms": 26.277659574468085,
    "median_ms": 24.087,
    "min_ms": 14.646,
    "p95_ms": 38.942
  },
  "vessel_inference_ms": {
    "count": 47,
    "max_ms": 43502.496,
    "mean_ms": 41688.047638297874,
    "median_ms": 41655.647,
    "min_ms": 40461.51,
    "p95_ms": 43125.901
  }
}
```
