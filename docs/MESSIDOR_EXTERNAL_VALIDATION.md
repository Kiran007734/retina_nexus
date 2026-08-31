# Phase 4: Messidor/Messidor-2 acquisition and readiness

Phase 4 prepares an independent dataset for zero-shot evaluation of the
already-trained APTOS EfficientNet-B0 classifier. It does not train, fine-tune,
modify, or select the APTOS model.

## Authorized source and access

The official source is the [ADCIS Messidor page](https://www.adcis.net/en/third-party/messidor/)
and its [ADCIS Messidor-2 page](https://www.adcis.net/en/third-party/messidor2/).
Both pages state that a form with personal information must be completed before
download. Their terms restrict use to research and educational purposes and
prohibit copying, redistribution, and unauthorized commercial use.

The repository does not bypass the form, authentication, licensing, or any
protected download mechanism. There is intentionally no Kaggle or scraper
path for this dataset.

## Current local status

The project raw directory is:

```text
ml/datasets/raw/messidor/
```

The supplied Messidor-2 copy contains 1,744 images under
`ml/datasets/raw/messidor/images/messidor-2/`. The official release has no DR
ground truth, so this checkout uses the separately obtained KaggleHub label
package `google-brain/messidor2-dr-grades` only after schema, provenance, and
compatibility checks. The exact cache path and SHA-256 values are recorded in
the generated manifest; no label file was rewritten.

## Official dataset facts

The ADCIS documentation describes the following variants:

| Variant | Images | Image formats | Official DR labels |
|---|---:|---|---|
| Messidor original | 1,200 | TIFF | Per-subset medical-diagnosis Excel files; retinopathy grades 0–3 and macular-edema risk 0–2 |
| Messidor-2 | 1,748 images from 874 examinations | 1,058 PNG + 690 JPG | None in the official release; the pairing spreadsheet is metadata, not DR ground truth |

Messidor original grades are defined from microaneurysm, hemorrhage, and
neovascularization findings. The official grade 3 combines severe findings and
neovascularization. It therefore cannot be split into APTOS grade 3 (Severe)
and grade 4 (Proliferative DR) without information that is not present in the
source grade.

## Manual setup

1. Complete the download form at the appropriate ADCIS page.
2. Retain the original archive and source documentation.
3. Extract or place the authorized files, without changing labels, under
   `ml/datasets/raw/messidor/`.
4. Select the variant explicitly after inspecting the acquired copy:

```powershell
python scripts/acquire_messidor.py --variant messidor --verify-only
python scripts/acquire_messidor.py --variant messidor2 --verify-only
```

`--variant auto` is available for an initial inventory, but an explicit variant
is required for a final readiness decision:

```powershell
python scripts/acquire_messidor.py --variant auto --verify-only
```

The command is local-only. It writes:

- `ml/evaluation/messidor/dataset_manifest.json`
- `ml/evaluation/messidor/validation_report.json`
- `ml/evaluation/messidor/grading_compatibility.json`
- `ml/evaluation/messidor/phase4_readiness_report.json`

It checks actual image formats, dimensions, readability, SHA-256 inventory,
exact and perceptual duplicates, label-file parsing, image/label matching,
malformed grades, duplicate annotation rows, conflicting annotations, and
missing labels. Raw files and original labels are never rewritten.

### Messidor-2 label provenance

The official Messidor-2 release has no DR ground truth. A third-party label file
may only be considered after separate human review of its source, authorization,
license, grading semantics, and relationship to the exact image release. Record
that information in `messidor_label_provenance.json` at the raw root. The
validator will not treat an unprovenanced third-party file as official ground
truth.

## Phase 4B zero-shot evaluation

Run the unchanged APTOS EfficientNet-B0 checkpoint against all matched,
gradable Messidor-2 images:

```powershell
python scripts/evaluate_messidor2.py --device cpu --batch-size 32 --torch-threads 8 --bootstrap-iterations 2000
```

The evaluator discovers the schema-compatible label CSV in the KaggleHub
cache, validates the raw image tree, performs inference only, and writes
reports and predictions under `ml/evaluation/messidor/`. The verified run
found 1,744 images, 1,748 label rows, 1,744 matched gradable images, zero
corrupt images, and four unmatched ungradable label rows. The APTOS
checkpoint was unchanged. Results are descriptive zero-shot external
evaluation only; they are not clinical validation or a claim that the
Messidor-2 official release contains DR ground truth.

## Compatibility decision

APTOS outputs five classes:

```text
0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR
```

For Messidor original, five-class metrics are blocked. A severity-only binary
proxy `Messidor grade >= 2` can be calculated conditionally, corresponding to
the APTOS grade-2-or-worse threshold, but it must be named a severity-only
proxy and must not be presented as full referable DR unless a separate,
documented macular-edema policy is established. Legitimate conditional metrics
are sensitivity, specificity, precision, recall, F1, ROC-AUC, and a confusion
matrix for that binary proxy.

For official Messidor-2, neither five-class nor binary metrics are calculable
because there are no official DR labels. A documented third-party label source
can move the result to conditional review; it does not automatically become an
official ground truth.

No model training or fine-tuning is performed by this phase. Repeatable
zero-shot evaluation is implemented by `scripts/evaluate_messidor2.py`.

## Phase 4B reliability audit

After the original evaluation, run the non-destructive reliability audit:

```powershell
python scripts/audit_messidor2_reliability.py --device cpu --batch-size 32 --torch-threads 8 --bootstrap-iterations 2000 --bootstrap-seed 42
```

This preserves the original full-dataset results, audits exact and perceptual
duplicates, reruns inference on a separate perceptual-deduplicated population,
analyses fixed referable thresholds, and produces deterministic error samples.
It does not retrain, fine-tune, alter probabilities, delete images, or replace
the original benchmark. Audit outputs are written alongside the original
artifacts under `ml/evaluation/messidor/`.
