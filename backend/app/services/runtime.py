"""Runtime readiness and model-artifact verification.

This module is deliberately separate from model inference.  It verifies the
deployment contract before a request reaches the screening pipeline and keeps
operator diagnostics out of public health responses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.core.config import Settings, get_settings


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_CHECK_VERSION = "deployment-runtime-check-v1"


@dataclass
class ModelCheck:
    name: str
    required: bool
    version: str | None
    artifact_present: bool
    manifest_present: bool
    checksum_configured: bool
    checksum_valid: bool | None
    loadable: bool | None
    status: str
    error_code: str | None = None
    _artifact_path: str | None = None

    def to_dict(self, include_path: bool = False) -> dict[str, Any]:
        value = {
            "name": self.name,
            "required": self.required,
            "version": self.version,
            "artifact_present": self.artifact_present,
            "manifest_present": self.manifest_present,
            "checksum_configured": self.checksum_configured,
            "checksum_valid": self.checksum_valid,
            "loadable": self.loadable,
            "status": self.status,
            "error_code": self.error_code,
        }
        if include_path:
            value["artifact_path"] = self._artifact_path
        return value


_last_model_check: dict[str, Any] | None = None


def repository_root() -> Path:
    return REPOSITORY_ROOT


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = ((Path.cwd() / path).resolve(), (REPOSITORY_ROOT / path).resolve(), (REPOSITORY_ROOT / "backend" / path).resolve())
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def ensure_runtime_directories(settings: Settings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    upload_root = resolve_path(settings.local_storage_path or settings.upload_directory)
    if upload_root is None:
        raise RuntimeError("UPLOAD_DIRECTORY is not configured")
    upload_root.mkdir(parents=True, exist_ok=True)
    (REPOSITORY_ROOT / "ml" / "evaluation" / "deployment").mkdir(parents=True, exist_ok=True)
    return {"upload_directory": str(upload_root)}


def _registry_path() -> Path:
    return REPOSITORY_ROOT / "ml" / "model_registry.json"


def _registry_artifact(version: str | None, artifact_path: Path | None) -> dict[str, Any] | None:
    path = _registry_path()
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    normalized = artifact_path.as_posix().lower() if artifact_path else ""
    for artifact in registry.get("artifacts", []):
        if version and artifact.get("model_version") == version:
            return artifact
        registered = str(artifact.get("checkpoint", "")).replace("\\", "/").lower()
        if normalized and registered and (normalized.endswith(registered) or registered.endswith(normalized)):
            return artifact
    return None


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_artifact(
    name: str,
    required: bool,
    path: Path | None,
    version: str | None,
    configured_checksum: str | None,
    manifest_names: tuple[str, ...],
    loader: Callable[[], None] | None,
    checksum_enabled: bool = True,
) -> ModelCheck:
    artifact_present = path is not None and path.is_file()
    manifest_present = bool(path and path.parent.is_dir() and all((path.parent / item).is_file() for item in manifest_names))
    registry = _registry_artifact(version, path)
    expected_checksum = ((configured_checksum or (registry or {}).get("checkpoint_sha256") or "").strip().lower() or None) if checksum_enabled else None
    actual_checksum: str | None = None
    checksum_valid: bool | None = None
    if artifact_present and expected_checksum:
        try:
            actual_checksum = _checksum(path)
            checksum_valid = actual_checksum == expected_checksum
        except OSError:
            checksum_valid = False
    elif expected_checksum:
        checksum_valid = False

    resolved_version = version or (registry or {}).get("model_version")
    base = {
        "name": name,
        "required": required,
        "version": resolved_version,
        "artifact_present": artifact_present,
        "manifest_present": manifest_present,
        "checksum_configured": expected_checksum is not None,
        "checksum_valid": checksum_valid,
        "loadable": None,
        "status": "OPTIONAL_MODEL_UNAVAILABLE" if not required else "REQUIRED_MODEL_UNAVAILABLE",
        "_artifact_path": str(path) if path else None,
    }
    if not artifact_present:
        base["error_code"] = "ARTIFACT_MISSING"
        return ModelCheck(**base)
    if not manifest_present:
        base["status"] = "MODEL_MANIFEST_MISSING"
        base["error_code"] = "MANIFEST_MISSING"
        base["loadable"] = False
        return ModelCheck(**base)
    if checksum_valid is False:
        base["status"] = "CHECKSUM_MISMATCH"
        base["error_code"] = "CHECKSUM_MISMATCH"
        base["loadable"] = False
        return ModelCheck(**base)
    if loader is None:
        base["status"] = "PRESENT_NOT_LOADED"
        base["loadable"] = None
        return ModelCheck(**base)
    try:
        loader()
        base["loadable"] = True
        base["status"] = "AVAILABLE"
    except Exception:
        # Detailed exceptions stay in local logs/CLI diagnostics.  Health and
        # readiness responses expose only a stable error code.
        base["loadable"] = False
        base["status"] = "MODEL_LOAD_FAILED" if required else "OPTIONAL_MODEL_UNAVAILABLE"
        base["error_code"] = "LOAD_FAILED"
    return ModelCheck(**base)


def verify_models(settings: Settings | None = None, load_models: bool = True, load_optional_models: bool = True, verify_optional_checksums: bool = True) -> dict[str, Any]:
    """Verify required and optional artifacts without returning filesystem paths."""
    global _last_model_check
    settings = settings or get_settings()
    from app.ml.evidence.lesion_model import DEFAULT_MODEL_PATH, PretrainedRetinalLesionAdapter
    from app.ml.evidence.vessel_model import DEFAULT_MODEL_PATH as VESSEL_DEFAULT_MODEL_PATH, PretrainedRetinalVesselAdapter
    from app.ml.inference.classifier import TorchDRClassificationService
    from app.ml.models.classifier import ReferableDRMapping

    classifier = TorchDRClassificationService(
        model_path=settings.classifier_model_path,
        backbone=settings.classifier_backbone,
        model_version=settings.classifier_model_version,
        device=settings.classifier_device,
        referable_mapping=ReferableDRMapping(
            name=f"grade_{settings.referable_min_grade}_or_worse",
            referable_grades=tuple(range(settings.referable_min_grade, 5)),
        ),
    )
    classifier_path = resolve_path(settings.classifier_model_path)
    if classifier_path is not None:
        classifier.model_path = classifier_path
    classifier_version = settings.classifier_model_version
    classifier_check = _check_artifact(
        "classifier", True, classifier_path, classifier_version, settings.classifier_model_sha256,
        ("model_manifest.json",), classifier.verify_loadable if load_models else None,
    )

    lesion_path = resolve_path(settings.lesion_model_path) or DEFAULT_MODEL_PATH
    lesion_adapter = PretrainedRetinalLesionAdapter(model_path=lesion_path, device=settings.lesion_model_device, threshold=settings.lesion_model_threshold, version=settings.lesion_model_version)
    lesion_check = _check_artifact(
        "lesion_segmentation", False, lesion_adapter.model_path, settings.lesion_model_version, settings.lesion_model_sha256,
        ("model_manifest.json", "config.json"), lesion_adapter.verify_loadable if load_models and load_optional_models else None,
        verify_optional_checksums,
    )

    vessel_path = resolve_path(settings.vessel_model_path) or VESSEL_DEFAULT_MODEL_PATH
    vessel_adapter = PretrainedRetinalVesselAdapter(model_path=vessel_path, device=settings.vessel_model_device, threshold=settings.vessel_model_threshold, version=settings.vessel_model_version)
    vessel_check = _check_artifact(
        "vessel_segmentation", False, vessel_adapter.model_path, settings.vessel_model_version, settings.vessel_model_sha256,
        ("model_manifest.json", "bv_config.json", "model.py", "preprocessing.py"), vessel_adapter.verify_loadable if load_models and load_optional_models else None,
        verify_optional_checksums,
    )
    checks = {item.name: item.to_dict() for item in (classifier_check, lesion_check, vessel_check)}
    result = {
        "check_version": RUNTIME_CHECK_VERSION,
        "status": "READY" if classifier_check.status == "AVAILABLE" else "NOT_READY",
        "required_models_available": classifier_check.status == "AVAILABLE",
        "models": checks,
        "optional_capabilities": {name: value["status"] == "AVAILABLE" for name, value in checks.items() if name != "classifier"},
        "optional_models_loaded_at_check": bool(load_models and load_optional_models),
        "optional_checksums_verified_at_check": bool(verify_optional_checksums),
        "note": "Model checks verify local deployment artifacts and loadability. They do not establish clinical validity.",
    }
    _last_model_check = result
    return result


def last_model_check() -> dict[str, Any] | None:
    return _last_model_check


def reset_runtime_state() -> None:
    global _last_model_check
    _last_model_check = None
