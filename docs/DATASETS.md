# Dataset and Research Data Boundary

Raw datasets are local-only. This repository does not claim permission to
redistribute APTOS, IDRiD, DRIVE, Messidor, or Messidor-2 images, labels, masks,
or archives.

## Local datasets found during the release audit

| Dataset | Local image count | Use | Release decision |
|---|---:|---|---|
| APTOS 2019 | 5,590 | Primary DR classifier training/inference validation | Keep out of Git |
| IDRiD | 1,476 | DR, lesion, and anatomical research | Keep out of Git |
| DRIVE | 100 | Vessel segmentation research/evaluation | Keep out of Git |
| Messidor/Messidor-2 | 3,492 | External evaluation preparation | Keep out of Git |

Counts describe the current local checkout and are not clinical performance
claims. The complete file counts, sizes, source notes, and ignore checks are in
`ml/evaluation/github_release_audit/DATASET_INVENTORY.md`.

## Expected layout

```text
ml/datasets/raw/aptos2019/
ml/datasets/raw/idrid/
ml/datasets/raw/drive/
ml/datasets/raw/messidor/
```

Training/evaluation manifests and measured reports may be retained under
`ml/datasets/metadata/` when they contain no unauthorized image bytes or
credentials. Review any generated report containing absolute paths before
distribution.

## Acquisition rules

- APTOS acquisition uses the authorized Kaggle/KaggleHub competition mechanism
  and never stores credentials in the repository.
- IDRiD and DRIVE are used only from authorized local copies.
- Messidor/Messidor-2 access follows the documented ADCIS/source constraints;
  official Messidor-2 has no official DR ground-truth labels in the source
  release described by the project documentation.
- Missing access or labels must produce a clear setup/unsupported status, not
  fabricated files or metrics.

