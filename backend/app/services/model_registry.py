"""Small, explicit model-artifact lifecycle helpers used by the registry API."""

from pathlib import Path

from app.models.model_version import ModelVersion

ARTIFACT_KINDS = {
    "PRETRAINED_BACKBONE",
    "FINE_TUNED_MODEL",
    "DEMO_MODEL",
    "PRODUCTION_CANDIDATE",
    "EXPERIMENTAL",
}
ARTIFACT_STATUSES = {
    "MODEL_DOWNLOADED",
    "MODEL_TRAINED",
    "MODEL_AVAILABLE",
    "MODEL_MISSING",
    "MODEL_FAILED_TO_LOAD",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_model_availability(model: ModelVersion) -> str:
    """Resolve availability from the recorded lifecycle and actual artifact path.

    A registry row is never treated as available merely because it exists. A
    relative path is resolved from the repository root for local deployments.
    Explicit load failures remain visible until an operator updates the row.
    """
    if model.availability_status == "MODEL_FAILED_TO_LOAD" or model.load_error:
        return "MODEL_FAILED_TO_LOAD"
    if not model.file_path:
        return "MODEL_MISSING"
    path = Path(model.file_path).expanduser()
    if not path.is_absolute():
        path = _repository_root() / path
    return "MODEL_AVAILABLE" if path.is_file() else "MODEL_MISSING"


def model_response_payload(model: ModelVersion) -> dict:
    """Return a JSON-safe registry payload with current file availability."""
    return {
        "id": model.id,
        "model_name": model.model_name,
        "model_type": model.model_type,
        "version": model.version,
        "training_dataset": model.training_dataset,
        "input_size": model.input_size,
        "performance_metrics": model.performance_metrics,
        "training_config": model.training_config,
        "dataset_version": model.dataset_version,
        "file_path": model.file_path,
        "checksum": model.checksum,
        "artifact_kind": model.artifact_kind,
        "artifact_status": model.artifact_status,
        "availability_status": resolve_model_availability(model),
        "load_error": model.load_error,
        "is_active": model.is_active,
        "created_at": model.created_at,
    }
