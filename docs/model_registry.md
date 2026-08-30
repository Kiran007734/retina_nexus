# Model registry

`model_versions` stores the metadata required to swap models without changing route code:

- model name and type
- semantic version
- training dataset reference
- input size
- performance metrics
- training configuration and dataset version
- artifact path and checksum
- artifact kind: `PRETRAINED_BACKBONE`, `FINE_TUNED_MODEL`, `DEMO_MODEL`,
  `PRODUCTION_CANDIDATE`, or `EXPERIMENTAL`
- artifact lifecycle: `MODEL_DOWNLOADED`, `MODEL_TRAINED`, or
  `MODEL_MISSING`
- runtime availability: `MODEL_AVAILABLE`, `MODEL_MISSING`, or
  `MODEL_FAILED_TO_LOAD`, with a persisted load error when applicable
- active/inactive state
- created timestamp

Each inference adapter should receive a registry record, emit its model version with the result, and refuse to run when a required active version is unavailable. Performance metrics should be accompanied by dataset and evaluation provenance; the UI must not present them as clinical guarantees.

The API resolves relative artifact paths against the repository root and
checks the file on every registry read. A database row therefore does not
imply that weights are present. A pretrained backbone is also not a trained
RETINA-NEXUS classifier; the two artifact kinds remain distinct. Demo fixtures
are kept outside the clinical model registry and cannot be selected by the
production classifier.

The classifier training command also writes a self-contained model manifest
next to each checkpoint. It records the five-class model configuration, the
hierarchical or ordinal mode, referable mapping, training configuration,
dataset version, measured metrics, and a SHA-256 checksum. The generated
local index is ignored by Git until an authorized artifact promotion process
is defined.
