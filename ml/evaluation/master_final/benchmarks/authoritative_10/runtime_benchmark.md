# Authoritative Messidor-2 full-pipeline benchmark

Status: **COMPLETE**

- Requested: `10`
- Processed: `10`
- Terminal: `10`
- Failed: `0`
- Elapsed seconds: `752.043`
- Max wall seconds: `900.0`

This is an engineering benchmark of the existing pipeline. It does not establish clinical performance and does not change models or thresholds.

## Stage latency

```json
{
  "classification": {
    "count": 10,
    "max_ms": 616.3123000005726,
    "mean_ms": 221.82481000490952,
    "median_ms": 197.46024999767542,
    "min_ms": 84.08120000967756,
    "p95_ms": 247.53059999784455
  },
  "evidence_total": {
    "count": 10,
    "max_ms": 95001.08430002001,
    "mean_ms": 62086.08323999797,
    "median_ms": 51434.60474998574,
    "min_ms": 45031.06529999059,
    "p95_ms": 87167.50080001657
  },
  "grad_cam_and_agreement": {
    "count": 10,
    "max_ms": 8324.413199996343,
    "mean_ms": 5874.127799997223,
    "median_ms": 5176.5896000142675,
    "min_ms": 4090.554800000973,
    "p95_ms": 7630.309800006216
  },
  "image_validation": {
    "count": 10,
    "max_ms": 363.2825000095181,
    "mean_ms": 123.06706999661401,
    "median_ms": 97.70064998883754,
    "min_ms": 69.82319999951869,
    "p95_ms": 134.91379999322817
  },
  "lesion_inference_ms": {
    "count": 10,
    "max_ms": 4831.99,
    "mean_ms": 2655.2086,
    "median_ms": 2258.199,
    "min_ms": 1965.571,
    "p95_ms": 3341.38
  },
  "pdf_generation": {
    "count": 10,
    "max_ms": 0.4217000096105039,
    "mean_ms": 0.18893000087700784,
    "median_ms": 0.1700000138953328,
    "min_ms": 0.11170000652782619,
    "p95_ms": 0.27879999834112823
  },
  "quality_assessment": {
    "count": 10,
    "max_ms": 609.3969999928959,
    "mean_ms": 383.12510999676306,
    "median_ms": 302.3525999888079,
    "min_ms": 241.04939997778274,
    "p95_ms": 551.6528000007384
  },
  "quality_enhancement": {
    "count": 9,
    "max_ms": 10303.079200006323,
    "mean_ms": 7203.821122235644,
    "median_ms": 6252.756300003966,
    "min_ms": 4644.953700044425,
    "p95_ms": 10210.74530002079
  },
  "retinaguard": {
    "count": 10,
    "max_ms": 27.86879998166114,
    "mean_ms": 3.2717700029024854,
    "median_ms": 0.48755001625977457,
    "min_ms": 0.31170001602731645,
    "p95_ms": 1.2676999904215336
  },
  "structure_analysis_ms": {
    "count": 10,
    "max_ms": 58.204,
    "mean_ms": 39.0674,
    "median_ms": 33.462500000000006,
    "min_ms": 24.958,
    "p95_ms": 53.823
  },
  "vessel_inference_ms": {
    "count": 10,
    "max_ms": 86941.121,
    "mean_ms": 57460.5351,
    "median_ms": 47531.5205,
    "min_ms": 41715.519,
    "p95_ms": 81332.368
  }
}
```
