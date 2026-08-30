# Pretrained retinal lesion segmentation artifacts

RETINA-NEXUS can load the following externally published research model when
the optional ML dependencies are installed:

- Model: `ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d`
- Architecture: U-Net with a SE-ResNeXt-50 32x4d encoder
- Classes: background, cotton-wool spots, exudates, hemorrhages,
  microaneurysms
- Training sources stated by the authors: IDRiD, DDR, FGADR, Messidor, and
  RETLES
- Source code: <https://github.com/ClementPla/fundus-lesions-toolkit>
- Model card and weights: <https://huggingface.co/ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d>
- License declared by the source repository/model card: MIT

The `model.safetensors` file is intentionally excluded from Git. Acquire it
with:

```powershell
python scripts/acquire_lesion_model.py
```

The script verifies the published configuration, calculates a SHA-256
checksum, and writes `model_manifest.json` beside the local artifact. The
manifest is the provenance record used when registering the model. The
checkpoint is a research model used for supporting evidence only. It is not a
clinical diagnosis, regulatory claim, or RETINA-NEXUS validation result.

If the artifact is absent, the evidence API returns an explicit unsupported
model status. It does not substitute a fabricated mask.
