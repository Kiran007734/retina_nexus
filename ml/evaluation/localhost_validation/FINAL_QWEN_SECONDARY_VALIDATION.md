# Qwen secondary validation

Date: 2026-09-15

## Environment preflight

- Python: 3.11.9
- PyTorch: 2.11.0+cpu
- CUDA: unavailable
- NVIDIA GPU: none detected
- Disk: approximately 230.36 GB free on C:
- Transformers: 4.57.6
- Hugging Face Hub: 0.36.2

## Official model acquisition

- Model ID: `Qwen/Qwen3-VL-4B-Instruct`
- Official repository revision observed: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Target: `ml/weights/qwen/Qwen3-VL-4B-Instruct/`
- Runtime: local Transformers adapter
- Device/dtype selected: CPU / float32 if the complete model becomes available
- Download status: `BLOCKED_INCOMPLETE`

The official repository metadata and small files were retrieved. The two
required safetensors shards (approximately 8.88 GB total) remained at zero
bytes during two resumable official download attempts, including an attempt
with Xet disabled. The local directory therefore lacks:

- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`

No unofficial mirror, API key, credential, or mock model was used.

## Preflight result

The preflight command is:

`python scripts/preflight_qwen.py`

Result: `NOT_CONFIGURED`. Tokenizer/config/processor metadata files exist, but
the complete model shards do not. Model loading and image inference were not
attempted because doing so would not be a valid model preflight.

Machine-readable result: `ml/evaluation/localhost_validation/qwen_preflight.json`.

The implemented adapter is [qwen_vl_verifier.py](../../backend/app/services/qwen_vl_verifier.py).
It is lazy-loaded, local-only, strict-JSON validated, timeout-bounded, and has
no fake fallback. The primary pipeline remains unchanged and operational.

## Integration status

- Qwen status: `NOT_CONFIGURED`
- Real Qwen image inference: not run because required shards are absent
- Qwen model agreement: not available
- Explanation stability: existing primary Grad-CAM path remains available;
  Qwen does not produce Grad-CAM
- Distribution check: existing OOD reference remains `NOT_CONFIGURED`; no
  reference distribution was fabricated
- RetinaGuard: existing graceful missing-signal policy remains active
- Primary EfficientNet-B0, lesion model, R2-V2 vessel model, thresholds,
  datasets, and weights: unchanged

## Existing regression verification

- Backend tests: 105 passed
- Focused tests: 16 passed
- Python compilation: passed
- Frontend TypeScript lint: passed
- Frontend production build: passed
- Existing localhost services remained healthy

## Genuine blocker and next action

The remaining blocker is external model-shard transfer, not an application
fallback. Retry the official Hugging Face download when the environment permits
large-file transfer, then run `python scripts/preflight_qwen.py` again. Qwen
must not be marked `AVAILABLE` until the processor/model loads and one real
retinal-image inference returns schema-valid JSON.

Qwen is not clinically validated and is not a replacement for an ophthalmologist.
