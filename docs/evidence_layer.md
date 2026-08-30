# Retinal clinical evidence layer

The retinal evidence layer is a supporting analysis path distinct from the
five-class diabetic-retinopathy classifier. It never changes the classifier's
grade or referable result.

## Coarse-to-fine workflow

The runtime keeps the following stages explicit in the API response:

1. Full fundus image and global context statistics
2. Suspicious region proposals
3. High-resolution patch coordinates
4. Local module analysis
5. A combined evidence overlay

The default service downsizes only its working copy to a bounded analysis
resolution. Region proposals retain coordinates in that working image so
patch-level adapters can process small candidate regions without relying on a
classifier-sized whole-image resize.

## Module status and safety boundary

The runtime uses the verified R2-V2 `bv` model for vessel masks when the local
artifact is present. See [VESSEL_MODEL_INTEGRATION.md](VESSEL_MODEL_INTEGRATION.md).
If explicitly enabled, the previous classical-CV vessel mask is labelled
`EXPERIMENTAL BASELINE — NOT MODEL-BACKED`; it is disabled by default and is
never substituted for a configured model failure. Optic-disc localization,
the optic-disc-relative fovea approximation, and experimental candidate-region
heuristics for microaneurysms, hemorrhages, and exudates remain explicitly
labelled engineering outputs. They are not clinical findings.

Neovascularization is currently an experimental interface only and returns
unsupported until a validated model and compatible annotations are configured.
The same unsupported status is used when heuristic processing is disabled.

The model interfaces are framework-neutral protocols:

- SegmentationModel for pixel masks
- PatchDetector for high-resolution lesion proposals
- LandmarkLocalizer for optic-disc/fovea coordinates
- EvidenceModelAdapter for injecting a trained module without changing the API

Lesion and vessel weights are not bundled. They are acquired through their
documented scripts and are loaded only after artifact/configuration checks.

## Dataset annotation support

Run:

    python scripts/validate_evidence_annotations.py

The capability report checks local authorized files only. DRIVE vessel
training/evaluation requires paired images and vessel masks. IDRiD lesion
support requires compatible, present annotations for the requested lesion
module. A missing label produces unsupported status; the system never creates
or infers annotations to make a module appear supported.

DRIVE quantitative evaluation is available when authorized paired data exists:

    python scripts/train_vessel_segmentation.py --raw-dir ml/datasets/raw/drive
    python scripts/evaluate_vessel_segmentation.py --checkpoint <checkpoint>

These scripts target the existing compact DRIVE training checkpoint. The
pretrained R2-V2 adapter has separate real-inference verification; no DRIVE
metrics are reported until authorized ground-truth masks are present.

## API

After the Image Trust Gate, call:

    POST /api/v1/screening/analyze-structures

with an image_id and optional screening_session_id. The response contains
standardized module objects with status, support, implementation, confidence,
counts, masks, bounding regions, landmarks, coarse-to-fine metadata, dataset
capability status, and a combined evidence-map PNG data URI.

UNGRADABLE images are blocked before structure analysis. Evidence analysis may
run independently of classifier availability, but neither path issues a
final clinical decision or trust score.
