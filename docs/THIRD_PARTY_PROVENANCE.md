# Third-Party Provenance and License Status

This file records only information already present in the repository’s model
documentation and manifests. It does not grant permission to redistribute any
dataset or model.

| Artifact | Source recorded in repository | Architecture/purpose | License recorded | Redistribution status |
|---|---|---|---|---|
| Fundus lesion model | `https://huggingface.co/ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d` and `https://github.com/ClementPla/fundus-lesions-toolkit` | U-Net with SE-ResNeXt-50 encoder; supporting lesion evidence | MIT declared by model repository/model card | Requires human verification before bundling |
| R2-V2 vessel model | `https://huggingface.co/j-morano/R2-V2` and `https://github.com/j-morano/R2-V2` | RRWNet `bv` variant; supporting vessel evidence | CC BY 4.0 declared by model repository/model card | Requires human verification and attribution review |
| APTOS classifier | RETINA-NEXUS-trained checkpoint; no external license recorded in repository | EfficientNet-B0 DR classifier | Not recorded | Requires human provenance/redistribution decision |
| IDRiD/DRIVE research checkpoints | RETINA-NEXUS research artifacts and local dataset annotations | Research-only grading, lesion, localization, and vessel candidates | Not recorded | Requires human verification; not production-promoted |
| APTOS, IDRiD, DRIVE, Messidor/Messidor-2 datasets | Dataset-specific source/access notes in `docs/data_governance.md` and `docs/MESSIDOR_EXTERNAL_VALIDATION.md` | Training, evidence, and external evaluation data | Dataset terms are not reproduced here | Keep raw data external; do not infer redistribution permission |

## Project license

No root `LICENSE`, `NOTICE`, or `COPYING` file was found during the release
audit. The project owner should choose and add an appropriate project license
before public distribution. That decision must not be used to override the
terms of third-party models or datasets.

