# Model Setup and Provenance Boundary

Model files are intentionally not bundled in the Git repository. A fresh clone
needs an authorized local checkpoint for real inference. Do not copy weights
from an unofficial mirror, bypass access controls, or infer redistribution
permission from a checksum.

## Required primary runtime artifact

| Role | Expected path | Version | SHA-256 | Status |
|---|---|---|---|---|
| Primary DR classifier | `ml/weights/classifiers/aptos2019/efficientnet-b0-aptos2019-20260830-v1/checkpoint_best.pt` | `efficientnet-b0-aptos2019-20260830-v1` | `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b` | Required for real localhost classification |

The classifier is the RETINA-NEXUS-trained APTOS EfficientNet-B0 artifact
recorded by `ml/model_registry.json`. Its redistribution terms are not
recorded as an open-source license; human provenance/redistribution approval is
required before bundling it.

Configure it in `backend/.env`:

```text
CLASSIFIER_MODEL_PATH=../ml/weights/classifiers/aptos2019/efficientnet-b0-aptos2019-20260830-v1/checkpoint_best.pt
CLASSIFIER_MODEL_VERSION=efficientnet-b0-aptos2019-20260830-v1
CLASSIFIER_BACKBONE=efficientnet_b0
VERIFY_MODELS_ON_STARTUP=true
```

## Supporting runtime artifacts

These are used for optional clinical evidence in the full local prototype and
are not needed for the primary grade if unavailable. The backend reports an
explicit unavailable/degraded capability; it does not fabricate evidence.

| Role | Expected path | SHA-256 | Provenance status |
|---|---|---|---|
| Lesion evidence | `ml/weights/lesion_segmentation/fundus-lesions-unet-seresnext50-all-v1/model.safetensors` | `a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2` | Model manifest declares MIT; human review still required |
| R2-V2 vessel evidence | `ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors` | `ea219b13b03984b8d454f999343c5bda1a8a7cb8586aeb3639a29814cab2330a` | Model manifest declares CC BY 4.0; human review still required |

The acquisition scripts use the documented official source mechanisms:

```powershell
python scripts/acquire_lesion_model.py
python scripts/acquire_vessel_model.py
python scripts/verify_models.py --json
```

These commands fail clearly when the source, package, or local authorization is
unavailable. They do not create placeholder weights.

## Research-only artifacts

IDRiD severity/lesion/localization checkpoints, DRIVE vessel candidates, CV
folds, and the RETGUARD ONNX verifier are ignored external artifacts. Their
actual local SHA-256 values are recorded in
`ml/evaluation/github_release_audit/MODEL_WEIGHT_INVENTORY.md`; they are not
production-promoted by default.

## Verification behavior

```powershell
python scripts/verify_models.py --json
```

The preflight checks artifact presence, registry/manifest checksum, required
model loadability, and optional capability availability. A missing required
classifier prevents real clinical AI inference. Missing optional evidence
models remain visible as unavailable and do not become fake predictions.

## Fresh clone expectation

Source code, migrations, frontend dependencies, and documentation can be
cloned normally. Real inference is not ready until the required primary
checkpoint is installed through an authorized, documented process and
`verify_models.py` passes. Raw datasets are not required for ordinary
inference; they are required for training and dataset evaluation only.

