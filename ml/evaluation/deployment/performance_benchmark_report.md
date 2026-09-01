# RETINA-NEXUS deployment performance benchmark

Generated: `2026-09-01T18:32:34.680058+00:00`

This is a local engineering benchmark using an authorized image. It is not a production-scale throughput claim, clinical validation, or safety guarantee.

## Environment

- OS: `Windows-10-10.0.26200-SP0`
- Python: `3.11.9`
- Processor: `Intel64 Family 6 Model 191 Stepping 2, GenuineIntel`
- CPU count: `16`
- Torch: `2.11.0+cpu`
- Torchvision: `0.26.0+cpu`
- CUDA available: `False`

## Method

- Warm repetitions: `3`
- Model loads are measured separately; stage samples run warm after the first load.
- No database records, model artifacts, datasets, or evaluation reports were modified.

## Stage latency

| Stage | n | Mean ms | Median ms | P95 ms | Stddev ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| image_validation | 3 | 108.96 | 92.562 | 140.997 | 32.488 |
| quality_assessment | 3 | 325.221 | 209.857 | 525.097 | 203.442 |
| classification | 3 | 196.175 | 112.106 | 358.538 | 164.678 |
| evidence_total | 3 | 31652.054 | 662.545 | 84367.039 | 53706.988 |
| vessel_inference_ms | 3 | 30532.666 | 274.706 | 81973.217 | 52410.266 |
| structure_analysis_ms | 3 | 24.397 | 27.574 | 27.738 | 5.661 |
| lesion_inference_ms | 3 | 742.361 | 69.634 | 1911.31 | 1189.638 |
| grad_cam_and_agreement | 3 | 4501.129 | 4582.757 | 4602.057 | 160.314 |
| retinaguard | 3 | 1.369 | 0.406 | 3.134 | 1.793 |
| pdf_generation | 3 | 0.135 | 0.142 | 0.176 | 0.048 |
| classification_only | 3 | 196.175 | 112.106 | 358.538 | 164.678 |
| full_pipeline | 3 | 36785.044 | 5651.577 | 89996.158 | 54198.468 |

## Artifact/model status

- Run statuses: `COMPLETED, COMPLETED, COMPLETED`
- Completed runs: `3`
- Evidence module statuses: `{"cotton_wool_spot_detection": "model_inference", "exudate_segmentation": "model_inference", "fovea_localization": "approximate", "hemorrhage_detection": "model_inference", "microaneurysm_detection": "model_inference", "neovascularization_detection": "unsupported", "optic_disc_localization": "experimental_heuristic", "vessel_segmentation": "model_inference"}`

## Limitations

- Local engineering latency only; no throughput or clinical performance claim.
- P95 is descriptive and based on the configured repetition count.
- Unavailable optional evidence is reported as unavailable and is not replaced by a heuristic model.
