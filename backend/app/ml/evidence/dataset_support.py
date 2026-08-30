"""Dataset/annotation capability checks for evidence modules.

This code only inspects local authorized files. It never creates labels or
marks a dataset available when its source files are absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATH = ROOT / "ml" / "datasets" / "metadata" / "dataset_registry.json"

LESION_MODULES = (
    "microaneurysm_detection",
    "hemorrhage_detection",
    "exudate_segmentation",
    "neovascularization_detection",
)


def _registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"datasets": []}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"datasets": []}


def _definition(slug: str) -> dict[str, Any] | None:
    return next((item for item in _registry().get("datasets", []) if item.get("slug") == slug), None)


def _path(definition: dict[str, Any]) -> Path:
    value = Path(str(definition.get("raw_path", "")))
    return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _files_exist(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def _drive_support() -> dict[str, Any]:
    definition = _definition("drive")
    if definition is None:
        return {"status": "unsupported", "reason": "DRIVE is not present in the dataset registry."}
    path = _path(definition)
    if not _files_exist(path):
        return {"status": "unsupported", "reason": "DRIVE is not acquired. Place authorized images and vessel masks under ml/datasets/raw/drive/."}
    mask_files = [item for item in path.rglob("*") if item.is_file() and any(token in item.name.lower() for token in ("manual", "mask", "1st", "2nd"))]
    if not mask_files:
        return {"status": "unsupported", "reason": "DRIVE images were found but no vessel-mask files were found; annotations were not fabricated."}
    return {"status": "available", "reason": "DRIVE image and vessel-mask files were found.", "annotation_file_count": len(mask_files)}


def _idrid_support(module: str) -> dict[str, Any]:
    definition = _definition("idrid")
    if definition is None:
        return {"status": "unsupported", "reason": "IDRiD is not present in the dataset registry."}
    path = _path(definition)
    if not _files_exist(path):
        return {"status": "unsupported", "reason": "IDRiD is not acquired. Place authorized images and compatible lesion annotations under ml/datasets/raw/idrid/."}
    keywords = {
        "microaneurysm_detection": ("microaneurysm", "micro_aneurysm", "micro-aneurysm"),
        "hemorrhage_detection": ("hemorrhage", "hemorrhages", "haemorrhage"),
        "exudate_segmentation": ("exudate", "exudates", "hardexudate", "soft_exudate", "hard_exudate"),
        "neovascularization_detection": ("neovascular", "neovascularization", "new-vessel", "new_vessel"),
    }[module]
    annotation_files = []
    for item in path.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in {".csv", ".json", ".xml", ".mat", ".tif", ".tiff", ".png"}:
            continue
        haystack = item.name.lower()
        if item.suffix.lower() in {".csv", ".json", ".xml"}:
            try:
                haystack += " " + item.read_text(encoding="utf-8", errors="ignore")[:2_000_000].lower()
            except OSError:
                pass
        if any(token in haystack for token in keywords):
            annotation_files.append(item)
    if not annotation_files:
        return {"status": "unsupported", "reason": f"No compatible IDRiD annotation was found for {module}; annotations were not fabricated."}
    return {"status": "available", "reason": "Compatible IDRiD annotation files were found.", "annotation_file_count": len(annotation_files)}


def evidence_dataset_support() -> dict[str, Any]:
    """Return support status for training/evaluation data, separately from inference."""
    support: dict[str, Any] = {"drive": {"vessel_segmentation": _drive_support()}, "idrid": {}}
    for module in LESION_MODULES:
        support["idrid"][module] = _idrid_support(module)
    return support
