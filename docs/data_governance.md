# Data management and governance

## Acquisition

The repository never downloads or commits dataset content implicitly. Dataset definitions live in `ml/datasets/metadata/dataset_registry.json` and start as `not_acquired`.

For APTOS, an authorized Kaggle account can be used after installing/configuring the Kaggle CLI:

```powershell
$env:KAGGLE_USERNAME = "your-authorized-username"
$env:KAGGLE_KEY = "your-authorized-key"
python scripts/acquire_dataset.py --dataset aptos2019 --mode kaggle
```

Alternatively configure `KAGGLE_CONFIG_DIR` or `~/.kaggle/kaggle.json`. If the account or competition permission is unavailable, use the manual flow:

```powershell
# Place files obtained under their source terms in ml/datasets/raw/<slug>
python scripts/acquire_dataset.py --dataset aptos2019 --mode manual --verify-only
```

IDRiD, DRIVE, and Messidor are registered for manual placement because a public Kaggle identifier is not assumed. The acquisition command fails with setup instructions when credentials, the Kaggle CLI, the source slug, or local files are missing.

## Governance commands

Run from the repository root:

```powershell
python scripts/analyze_dataset.py --dataset aptos2019
python scripts/detect_duplicates.py --dataset aptos2019
python scripts/validate_dataset.py --dataset aptos2019
python scripts/create_splits.py --dataset aptos2019 --seed 42
```

Commands write dataset-scoped reports under `ml/datasets/metadata/reports/<slug>/` and split manifests under `ml/datasets/metadata/splits/<slug>/`. The root `ml/datasets/metadata/data_leakage_report.json` is updated with the latest leakage result for convenient CI or review consumption.

Validation covers readable/corrupted images, exact SHA-256 duplicates, perceptual near-duplicates, missing annotation references, allowed labels, class distribution, image resolution, metadata completeness, and split integrity. The perceptual hash is an average hash; near matches should be reviewed before exclusion.

## Split guarantees and limitations

When an annotation file exposes a patient/subject identifier, split generation groups all records for that identifier. Duplicate-image groups are kept together regardless of patient metadata. When patient identifiers are absent, the report explicitly sets `patient_level_guarantee` to `false` and records that only duplicate/image grouping was enforced.

Splits are manifests only; source files are never copied by the command. A split is blocked if patient groups or duplicate groups cross train, validation, or test boundaries.

## Dataset Readiness Score

The prototype score is a weighted engineering signal, not a clinical validation metric:

`25% readable files + 15% duplicate-free + 20% label completeness + 15% class balance + 15% split integrity + 10% metadata completeness`

Dimensions that are not applicable to a dataset, such as class labels for a segmentation-only source, are excluded from the denominator and reported in `excluded_dimensions`. This score must not be used as evidence of model safety, generalization, or clinical performance.
