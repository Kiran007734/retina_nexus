# Pretrained retinal vessel segmentation artifacts

RETINA-NEXUS uses the published R2-V2 `bv` vessel segmentor when its local
artifact is available:

- Model: `j-morano/R2-V2`
- Architecture: RRWNet, R2-V2 `bv` variant
- Classes: artery, vein, blood vessels (channel 2 is used for vessel mask)
- Training provenance stated by the artifact: `Unified_Fundus`
- Model card: <https://huggingface.co/j-morano/R2-V2>
- Source code: <https://github.com/j-morano/R2-V2>
- License declared by the source: CC BY 4.0

Acquire the ignored checkpoint and published inference files with:

```powershell
python scripts/acquire_vessel_model.py
```

The acquisition script validates `bv_config.json`, downloads the source
implementation, records the resolved repository revision, and writes a
checksum manifest. The real model is supporting engineering evidence only;
it is not a diagnostic or clinical validation claim.

If the artifact is absent or fails validation, the API returns an explicit
unsupported vessel module. It does not substitute a fabricated mask or the
old classical-CV baseline. The baseline is opt-in only through
`EVIDENCE_ENABLE_VESSEL_BASELINE=true` and is labelled as not model-backed.
