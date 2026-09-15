# IDRiD lesion segmentation research

This research artifact is separate from DR severity classification. It emits
supporting lesion evidence and cannot change the classifier grade, referable
rule, or RetinaGuard decision.

## Data contract

- Official IDRiD segmentation training split: 54 images.
- Official segmentation test split: 27 images.
- Training masks: microaneurysms 54/54, haemorrhages 53/54, hard exudates
  54/54, soft exudates 26/54.
- Test masks: microaneurysms 27/27, haemorrhages 27/27, hard exudates 27/27,
  soft exudates 14/27.
- Missing annotations are `UNAVAILABLE`, never all-zero masks.
- No official field-of-view masks were present.

Development training and threshold selection used only the 54-image training
split. The frozen model was then evaluated once on the 27-image official test
split. The official report is
`ml/datasets/metadata/idrid/idrid_lesion_official_test.json`.

## Frozen model

`ml/weights/lesions/idrid/checkpoint_best.pt`

- Version: `idrid-lesion-unet-seresnext50-768-focaldice-20260913-v2`
- Architecture: U-Net with SE-ResNeXt-50 32x4d encoder
- Multi-label outputs: microaneurysms, haemorrhages, hard exudates, soft
  exudates
- Input: RGB resized to 768x768, ImageNet normalization
- Fixed threshold: 0.70, selected from out-of-fold development predictions
- SHA-256: `8f3c64a7aae23318f08bb34199aee079b16ff155db92912e97543aa670246a2c`
- `production_promoted=false`; no clinical validation claim

The final measured report is
`ml/datasets/metadata/idrid/idrid_lesion_final_report.json`. Current status is
`LESION DETECTION REQUIRES FURTHER RESEARCH`, driven by the small dataset,
low microaneurysm performance, incomplete annotations, missing FOV masks, and
absence of clinical validation.

## Commands

```powershell
python scripts/audit_idrid_lesions.py
python scripts/run_idrid_lesion_research.py --epochs 2 --folds 5 --size 768 --device cpu
python scripts/analyze_idrid_lesion_cv.py --candidate idrid-unet-seresnext50-768-focaldice-v2 --folds 5
python scripts/finalize_idrid_lesion.py --candidate idrid-unet-seresnext50-768-focaldice-v2 --epochs 3 --size 768 --device cpu
python scripts/evaluate_idrid_lesions.py --device cpu
```

`evaluate_idrid_lesions.py` is guarded: it requires the frozen manifest and
refuses to overwrite a completed official evaluation. The opt-in backend
adapter is `backend/app/ml/evidence/idrid_lesion_model.py`; enable it only
with `IDRID_LESION_MODEL_ENABLED=true` and a verified checkpoint path. The
default evidence service continues to use the preserved external lesion model.
