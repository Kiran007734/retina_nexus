# Pretrained lesion segmentation integration

## Selected artifact

RETINA-NEXUS integrates the publicly published research checkpoint:

- Repository: `ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d`
- Model card: <https://huggingface.co/ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d>
- Source implementation: <https://github.com/ClementPla/fundus-lesions-toolkit>
- Architecture: U-Net with a SE-ResNeXt-50 32x4d encoder
- Input: RGB fundus image fitted to 1024 × 1024 with black-border cropping,
  padding, and ImageNet normalization
- Output classes: background, cotton-wool spot, exudate, hemorrhage, and
  microaneurysm
- Training sources stated by the authors: IDRiD, DDR, FGADR, Messidor, and
  RETLES
- License stated by the model repository/model card: MIT

The model is supporting evidence only. It has not been validated by
RETINA-NEXUS and is not a diagnostic or regulatory claim.

## Acquisition and verification

Weights are not committed to Git. From the repository root:

```powershell
pip install -r backend/requirements-ml.txt
python scripts/acquire_lesion_model.py
```

The script uses the official `huggingface_hub` download mechanism, retrieves
`model.safetensors` and `config.json`, checks the architecture/class count,
calculates SHA-256, and writes a local `model_manifest.json` under:

```text
ml/weights/lesion_segmentation/fundus-lesions-unet-seresnext50-all-v1/
```

The verified artifact checksum is:

```text
a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2
```

Use `python scripts/acquire_lesion_model.py --verify-only` to verify an
existing local artifact without downloading. Do not put Hugging Face tokens
or other secrets in this repository. The selected repository is public, so no
token is expected for the default download; a gated revision would require
the user's `huggingface_hub` authentication in their own environment.

## Runtime behavior

`PretrainedRetinalLesionAdapter` loads the safetensors checkpoint lazily on
the first evidence request. The checkpoint was published with older TorchSeg
state names. The adapter applies an explicit compatibility mapping for the
encoder stem and squeeze-excitation module names, then requires zero missing
and zero unexpected state keys. It never uses `torch.load` on an untrusted
pickle.

For each supported class the adapter returns a real model-derived probability
mask restored to the evidence working image, a thresholded visualization,
connected-component regions, pixel count/coverage, and model/source/license/
class metadata. The modules are `cotton_wool_spot_detection`,
`exudate_segmentation`, `hemorrhage_detection`, and
`microaneurysm_detection`. Neovascularization is still interface-only and
unsupported. Vessel segmentation remains an experimental classical-CV
baseline in this phase; no vessel model is downloaded or trained.

If the weights, config, runtime dependency, or architecture are unavailable,
the affected module returns `supported=false` and an actionable error. The
evidence pipeline does not silently fall back to a synthetic or heuristic
mask for a configured pretrained-model failure.

## Evaluation boundary

The repository records model provenance and performs inference smoke tests.
It does not claim segmentation accuracy on IDRiD or any clinical validation
result. A future evaluation task must use authorized lesion annotations and
report lesion-specific metrics separately from the APTOS DR classifier.
