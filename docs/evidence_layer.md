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

The development runtime includes transparent classical-CV baselines for
vessel masks and optic-disc localization, an optic-disc-relative fovea
approximation, and experimental candidate-region heuristics for
microaneurysms, hemorrhages, and exudates. These outputs are labelled
experimental and include clinical_validation_claim=false. They are not
clinical findings.

Neovascularization is currently an experimental interface only and returns
unsupported until a validated model and compatible annotations are configured.
The same unsupported status is used when heuristic processing is disabled.

The model interfaces are framework-neutral protocols:

- SegmentationModel for pixel masks
- PatchDetector for high-resolution lesion proposals
- LandmarkLocalizer for optic-disc/fovea coordinates
- EvidenceModelAdapter for injecting a trained module without changing the API

No trained evidence weights are bundled. An adapter must be explicitly
injected or configured by a future model promotion process.

## Dataset annotation support

Run:

    python scripts/validate_evidence_annotations.py

The capability report checks local authorized files only. DRIVE vessel
training/evaluation requires paired images and vessel masks. IDRiD lesion
support requires compatible, present annotations for the requested lesion
module. A missing label produces unsupported status; the system never creates
or infers annotations to make a module appear supported.

DRIVE vessel experiments are available through:

    python scripts/train_vessel_segmentation.py --raw-dir ml/datasets/raw/drive
    python scripts/evaluate_vessel_segmentation.py --checkpoint <checkpoint>

These scripts write measured mask metrics and artifact provenance, not
clinical validation claims.

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
