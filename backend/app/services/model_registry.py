"""Small, explicit model-artifact lifecycle helpers used by the registry API."""

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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


def artifact_registry_path() -> Path:
    """Locate the generated ML registry in both source and container layouts."""
    candidates = (
        _repository_root() / "ml" / "weights" / "model_registry.json",
        _repository_root() / "ml" / "model_registry.json",
        Path.cwd() / "ml" / "weights" / "model_registry.json",
        Path.cwd() / "ml" / "model_registry.json",
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


async def seed_model_registry(db: AsyncSession) -> None:
    """Synchronize trained artifacts into the API registry without inventing rows.

    The generated JSON registry is the source of truth for ML artifacts. This
    startup sync makes the same real checkpoint visible through ``/models``;
    missing or failed artifacts remain explicitly unavailable.
    """
    path = artifact_registry_path()
    if not path.is_file():
        return
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    changed = False
    for artifact in registry.get("artifacts", []):
        version = artifact.get("model_version")
        checkpoint = artifact.get("checkpoint")
        if not version or not checkpoint:
            continue
        existing = (await db.execute(select(ModelVersion).where(ModelVersion.version == version))).scalar_one_or_none()
        config = artifact.get("model_config") or {}
        training_config = artifact.get("training_config") or {}
        evaluation = artifact.get("evaluation") or {}
        measured_metrics = artifact.get("validation_metrics") or evaluation.get("metrics")
        values = {
            "model_name": artifact.get(
                "model_name",
                "RETINA-NEXUS DR classifier" if artifact.get("model_type", "classification") == "classification" else "RETINA-NEXUS evidence model",
            ),
            "model_type": artifact.get("model_type", "classification"),
            "version": version,
            "training_dataset": training_config.get("dataset") or artifact.get("dataset_version") or "not specified",
            "input_size": str(config.get("input_size", "")),
            "performance_metrics": {
                "validation": measured_metrics,
                "evaluation": evaluation or None,
                "evaluation_status": artifact.get("evaluation_status"),
                "clinical_validation_claim": False,
            },
            "training_config": training_config,
            "dataset_version": artifact.get("dataset_version"),
            "file_path": checkpoint,
            "checksum": artifact.get("checkpoint_sha256"),
            "artifact_kind": artifact.get("artifact_kind", "FINE_TUNED_MODEL"),
            "artifact_status": artifact.get("artifact_status", "MODEL_TRAINED"),
            "availability_status": artifact.get("availability_status", "MODEL_MISSING"),
            "is_active": artifact.get("availability_status") == "MODEL_AVAILABLE",
        }
        if existing is None:
            db.add(ModelVersion(**values))
            changed = True
        else:
            for key, value in values.items():
                if getattr(existing, key) != value:
                    setattr(existing, key, value)
                    changed = True
    if changed:
        await db.commit()


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
