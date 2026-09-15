# Messidor-2 Data Audit and Safe Ingestion

Generated: `2026-09-13T19:39:13.402127+00:00`  
Clinical validation claim: **false**

## Archive

- Parts present: **4**
- Virtual concatenated size: **2,455,185,147 bytes**
- Image entries: **1748**
- CRC validation: **PASS**
- Duplicate archive entry names: **0**
- Archive was inspected without joining or renaming the parts.

## Existing images

- Existing image count: **1744**
- Readable: **1744**
- Corrupt/unreadable: **0**
- Extensions: `{'.JPG': 687, '.png': 1057}`
- Formats: `{'JPEG': 687, 'PNG': 1057}`
- Dimensions: `{'512x512': 1744}`
- Exact duplicate groups: **4**

The existing files are under `ml/datasets/raw/messidor/images/` and are 512px
preprocessed assets. Their filenames correspond to archive originals, but their
bytes are not identical to the archive originals; they were not overwritten.

## Archive comparison and extraction

- Archive image count: **1748**
- Existing/archive filename matches: **1744**
- New archive-only images: **4**
- Cross-source exact SHA-256 matches: **0**
- New originals extracted: **4**
- Extraction directory: `ml/datasets/raw/messidor/messidor-2-original/IMAGES/`

Only archive-only images were extracted. Existing data was not duplicated,
renamed, deleted, or overwritten.

## CSV audit

`ml/datasets/raw/messidor/messidor-2.csv` contains **874** rows and is a left/right pairing index with no DR, DME, or gradability labels.

`ml\datasets\raw\messidor\images\messidor_data.csv` contains **1744** rows and columns `['adjudicated_dme', 'adjudicated_gradable', 'diagnosis', 'id_code']`. It has **1744** unique image identifiers, **0** validation errors, and diagnosis distribution `{0: 1017, 1: 270, 2: 347, 3: 75, 4: 35}`.

The label CSV does not itself declare official Messidor-2 provenance. It is
retained as a local/adjudicated label cache and is not called official clinical
ground truth by this audit.

## Matching and readiness

- Existing images with local labels: **1744**
- Existing images without local labels: **0**
- Archive images without local labels: **4**
- Patient IDs available: **no**
- Patient-level separation claim: **no**
- Validation set scope: **Existing 1744-image matched validation set only; the four archive-only images remain excluded until labels are legitimately supplied.**

## Final status

**MESSIDOR-2 READY FOR EXTERNAL VALIDATION**

This readiness applies to the existing 1,744-image, label-matched dataset for
descriptive external evaluation only. It is not a clinical validation claim.
The four newly extracted originals remain excluded from labeled evaluation until
their labels are legitimately available.
