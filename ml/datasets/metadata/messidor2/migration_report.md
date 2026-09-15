# Messidor-2 authoritative-original migration report

Generated: `2026-09-13T21:13:09.507744+00:00`

## Decision summary

- **Authoritative future image source:** `ml/datasets/raw/messidor/messidor-2-original/IMAGES/`
- **Archive source:** four-part `IMAGES.zip.001` through `IMAGES.zip.004`; CRC validation passed.
- **Original archive image entries:** `1748`
- **Originals available and readable:** `1748`
- **Exact filename mappings to existing derivatives:** `1744`
- **Matched local labels:** `1744`
- **Unlabeled originals:** `4`
- **Extraction requested in this run:** `True`

The 1,744 existing files are 512x512 preprocessed derivatives under
`ml/datasets/raw/messidor/images/messidor-2/messidor-2/preprocess/`. Their
filenames map exactly to 1,744 archive originals, but their bytes are not
byte-identical. The four archive-only images are retained as originals and are
marked `LABEL_UNAVAILABLE`; no labels were inferred.

## Label provenance

The diagnosis/DME/gradable values come from
`ml/datasets/raw/messidor/images/messidor_data.csv`. The CSV has 1,744 unique
rows and is preserved as a project-local/adjudicated reference source. This
report does not call it official Messidor-2 clinical ground truth. The
`messidor-2.csv` file is pairing metadata only and is not used as diagnosis
ground truth.

## Future supervised evaluation

`ml/evaluation/messidor2/authoritative_external_manifest.json` contains only
the `1744` readable, label-matched ORIGINAL records. The
four unlabeled original images are intentionally absent from supervised metric
calculations.

## Reference migration matrix

| Repository reference | Role | Current reference | Required future action |
|---|---|---|---|
| `scripts/evaluate_messidor2.py` | historical external evaluation | raw/messidor/images/messidor-2 by default; accepts an explicit image directory | Keep unchanged for reproducibility. Future runs should use the authoritative manifest or an explicit original image root. |
| `scripts/evaluate_messidor2_final_external.py` | historical final external evaluation | raw/messidor/images/messidor-2 and raw/messidor/images/messidor_data.csv | Do not rewrite historical behavior. Add a separate authoritative-manifest evaluation path before any new run. |
| `scripts/audit_messidor2_reliability.py` | historical reliability audit | raw/messidor as a discovery root | Keep historical audit inputs stable; future audits should explicitly consume authoritative_external_manifest.json. |
| `scripts/audit_reliability_usability.py` | historical Phase 5.1 usability audit | raw/messidor plus historical prediction CSVs | Preserve existing outputs; do not mix authoritative-original measurements into historical comparisons without a new run identifier. |
| `scripts/run_messidor2_parallel.py` | historical/full-pipeline execution infrastructure | historical per_image_results.jsonl image paths under the existing raw root | Keep cache identity and historical results stable; add a future manifest-driven run rather than changing existing cache provenance. |
| `scripts/evaluate_idrid_v2_messidor.py` | research-only IDRiD v2 external evaluation | raw/messidor discovery root | Use an explicit authoritative manifest for future research runs; do not regenerate prior artifacts in place. |
| `scripts/evaluate_idrid_v3_messidor.py` | research-only IDRiD v3 external evaluation | raw/messidor discovery root | Use an explicit authoritative manifest for future research runs; preserve existing zero-shot artifacts. |
| `scripts/audit_messidor2_ingestion.py` | dataset ingestion and provenance audit | split archive, historical derivative root, and local label CSV | Retain as the archive integrity audit. This migration manifest is the authoritative mapping output. |
| `ml/datasets/metadata/messidor2/messidor2_audit_manifest.json` | historical ingestion audit artifact | records existing derivatives and archive comparison | Preserve unchanged as historical audit evidence; use authoritative_original_manifest.json for future original-image provenance. |
| `ml/evaluation/messidor2/final_external/*` | historical final external artifacts | existing 1,744-image derivative evaluation | Never overwrite. New authoritative-original runs need a separate versioned output directory. |
| `README.md and docs/MESSIDOR_EXTERNAL_VALIDATION.md` | documentation | historical/Kaggle-derived evaluation instructions | Update in a later documentation-only change after the first authoritative-original evaluation is verified. |

Production backend/frontend paths do not depend on the Messidor dataset. No
production model configuration was changed. Historical evaluation outputs must
continue to resolve against the existing derivative tree and must not be
rewritten in place.

## Storage

- Split archive parts: `2,455,185,147` bytes (2.287 GiB)
- Extracted original images: `2,467,041,260` bytes (2.298 GiB)
- Existing preprocessed derivatives: `402,142,860` bytes (0.375 GiB)
- Safe to reclaim now: **0 bytes**
- Potential derivative reclaim after a separately approved migration: `402,142,860` bytes (0.375 GiB)

The archive parts should also be retained as provenance/recovery material. No
directory or file was deleted or moved by this migration.

## Deletion decision

**SAFE TO DELETE NOW: NO.**

The existing derivative dataset cannot be deleted yet because historical
evaluation manifests, cached predictions, research artifacts, and scripts still
depend on it. Deletion can be reconsidered only after a separately approved
migration proves that historical results are archived and reproducible, all
tests are independent of the old paths, and a new authoritative-original
evaluation has completed successfully.

## Recommended next action

Run a new versioned Messidor evaluation that consumes
`authoritative_external_manifest.json` and writes to a new output directory.
Compare it with the historical derivative evaluation without overwriting either
result. Only after that comparison and dependency migration should deletion be
reviewed.
