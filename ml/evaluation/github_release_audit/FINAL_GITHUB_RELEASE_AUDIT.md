# FINAL GitHub Release Audit

Audit date: `2026-09-15T04:30:36.402940+00:00`
Commit audited: `08b98f549a69febe7293bdcbfcf0a829fba06ff0`
Worktree at audit start: `DIRTY`

## Release decision

**GITHUB READY: NO**

This is a packaging/release decision, not a clinical or model-quality decision. No model, threshold, preprocessing, fusion rule, RetinaGuard rule, dataset, or screening flow was changed by this audit.

## Inventory

- Frontend, backend, ML source, adapters, registries, XAI, evidence, RetinaGuard, reports, tests, Simulink model, plots, configs, scripts, and deployment files are present in the repository inventory.
- Local weight artifacts: `116` weight/support files; `3962.78 MB` total.
- Raw datasets: `aptos2019, drive, idrid, messidor`; all raw dataset roots are ignored and untracked.
- See `MODEL_WEIGHT_INVENTORY.md` and `DATASET_INVENTORY.md` for complete records.

## Model preflight

- Status: `READY`.
- The current APTOS EfficientNet-B0 checkpoint is the only required primary classifier runtime artifact.
- Lesion and R2-V2 vessel models are supporting runtime capabilities and degrade explicitly when unavailable.
- IDRiD, DRIVE research candidates, and RETGUARD verifier artifacts are not production-promoted.

## Datasets and tracking

- Raw image datasets, labels, masks, and archives are not tracked by Git.
- Metadata/manifests are retained selectively, but any artifact containing local absolute paths or sensitive provenance should be reviewed before release.
- No redistribution permission is inferred.

## Security and portability

- Secret-pattern audit: `PASS - no matches found`; secret values were not emitted.
- Machine-specific path references: `0` report files identified for packaging cleanup.
- Upload MIME/content mismatch, oversized upload, traversal filename, and invalid image checks are covered by the existing validation/test suite.
- PHI-bearing route authentication/authorization remains incomplete for a clinical deployment.

## Large files

| Path | Size | Tracked | Ignored | Recommendation |
|---|---:|---|---|---|
| `ml/datasets/raw/messidor/IMAGES.zip.001` | 700.0 MB | False | True | External dataset/weight; do not push. |
| `ml/datasets/raw/messidor/IMAGES.zip.002` | 700.0 MB | False | True | External dataset/weight; do not push. |
| `ml/datasets/raw/messidor/IMAGES.zip.003` | 700.0 MB | False | True | External dataset/weight; do not push. |
| `ml/datasets/raw/messidor/IMAGES.zip.004` | 241.45 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors` | 236.77 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/backup_verifier/retguard/v1.0.0/retguard_dr_v1.0.0.onnx` | 202.63 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/backup_verifier/retguard/retguard-dr-v1.0.0.zip` | 195.22 MB | False | True | External dataset/weight; do not push. |
| `backend/retina_nexus.db` | 164.61 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/checkpoint_best.pt` | 132.12 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_1/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_2/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_3/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_4/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_5/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_1/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_2/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_3/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_4/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_5/checkpoint_best.pt` | 132.1 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesion_segmentation/fundus-lesions-unet-seresnext50-all-v1/model.safetensors` | 132.02 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_1/validation_outputs.npz` | 90.21 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_1/validation_outputs.npz` | 90.2 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_3/validation_outputs.npz` | 89.23 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_3/validation_outputs.npz` | 89.06 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_4/validation_outputs.npz` | 87.46 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_4/validation_outputs.npz` | 86.63 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_2/validation_outputs.npz` | 84.61 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_2/validation_outputs.npz` | 84.6 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v2/fold_5/validation_outputs.npz` | 78.06 MB | False | True | External dataset/weight; do not push. |
| `ml/weights/lesions/idrid/cv/idrid-unet-seresnext50-768-focaldice-v1/fold_5/validation_outputs.npz` | 77.82 MB | False | True | External dataset/weight; do not push. |

## Localhost and regression evidence

- Previous recorded real-image localhost validation: `{'gradable': {'quality': None, 'status': 'COMPLETED', 'primary_status': 'COMPLETED', 'evidence_status': 'AVAILABLE'}, 'borderline': {'quality': None, 'status': 'COMPLETED', 'primary_status': 'QUALITY_BLOCKED', 'evidence_status': 'NOT_RUN'}, 'ungradable': {'quality': None, 'status': 'COMPLETED', 'primary_status': 'QUALITY_BLOCKED', 'evidence_status': 'NOT_RUN'}}`.
- Report/PDF statuses: `201` / `200`.
- Current audit regression result: 103 backend tests passed in 72.66 seconds; Python compileall passed; model preflight reported `READY`; frontend lint/typecheck passed from `frontend/`; and the frontend production build passed from `frontend/`.
- Browser automation was unavailable; HTTP/API, SPA routes, assets, and build were verified.

## Documentation readiness

- `README.md`: `READY` for project overview, installation, runtime boundaries, measured results, limitations, and release caveats.
- `docs/ARCHITECTURE.md`: `READY` for the end-to-end module and deployment boundaries.
- `docs/MODEL_SETUP.md`: `READY` for required checkpoint setup, SHA verification, external artifact handling, and preflight commands.
- `docs/DATASETS.md`: `READY` for dataset locations, local inventory, validation boundaries, and non-redistribution guidance.
- `docs/DEMO.md`: `READY` for the controlled demo workflow and its non-clinical status.
- `docs/THIRD_PARTY_PROVENANCE.md`: `READY` as a provenance record; legal/redistribution approval remains a human blocker.

## Simulink

- `simulink/RETINA_NEXUS_SYSTEM.slx`, scenario scripts, results, plots, and documentation are present.
- Simulation outputs are operational modeling results and require real-world site calibration; they are not clinical or financial claims.

## Exact blockers before push

- Required runtime weights are external/ignored; a fresh clone cannot run real inference until authorized checkpoints are installed.
- Several model/dataset redistribution permissions are not established by the repository; human legal/provenance review is required.
- No project LICENSE or NOTICE file is present.
- GitHub push was previously blocked by unavailable github.com:443; this audit intentionally does not retry or push.

## Required actions

1. Human-review and approve model/dataset provenance, redistribution, and attribution terms.
2. Decide which external checkpoints will be distributed, hosted separately, or installed manually; retain the current ignore policy unless legal approval changes.
3. Add an appropriate project license/NOTICE after human/legal approval.
4. Review the generated path cleanup and audit artifacts, then create a new packaging commit if accepted.
5. Restore GitHub network/authentication and push only after the above approvals. This audit does not push.
