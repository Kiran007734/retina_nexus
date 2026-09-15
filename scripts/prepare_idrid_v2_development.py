"""Audit and prepare the isolated IDRiD v2 development manifest.

This command is deliberately conservative.  It reads only the already approved
IDRiD ``records`` from ``dr_training_split.json`` (323 train + 83 validation),
plus the labelled APTOS training images for a cross-dataset duplicate audit.
It never opens any record in ``reserved_official_test_records`` and it never
changes raw data or the frozen v1 artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IDRID_RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
APTOS_RAW = ROOT / "ml" / "datasets" / "raw" / "aptos2019"
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
SOURCE_SPLIT = META / "dr_training_split.json"
SEGMENTATION = META / "segmentation_manifest.json"
LOCALIZATION = META / "localization_manifest.json"
APTOS_REPORT = ROOT / "ml" / "datasets" / "metadata" / "reports" / "aptos2019" / "dataset_validation_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(value: str) -> str:
    match = re.search(r"IDRiD[_-]0*(\d+)", str(value), re.IGNORECASE)
    return f"IDRiD_{int(match.group(1)):03d}" if match else str(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ROOT / path


def raw_relative(value: str | Path) -> str:
    path = project_path(value).resolve()
    return path.relative_to(IDRID_RAW.resolve()).as_posix()


def inspect_image(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": path.is_file(),
        "readable": False,
        "sha256": None,
        "width": None,
        "height": None,
        "mode": None,
        "channels": None,
        "error": None,
    }
    if not path.is_file():
        result["error"] = "missing_file"
        return result
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result.update({
                "readable": True,
                "sha256": sha256(path),
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "channels": len(image.getbands()),
            })
    except Exception as exc:  # pragma: no cover - data-dependent path
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit_idrid(split: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dev_records = list(split.get("records", []))
    reserved = list(split.get("reserved_official_test_records", []))
    dev_ids = {canonical_id(record.get("image_id", "")) for record in dev_records}
    # IDRiD numbers the training and official-testing packages independently
    # (for example both packages contain an ``IDRiD_001.jpg``).  Numeric IDs
    # therefore are not cross-split identities.  The approved source manifest
    # uses split-qualified record keys and hashes to govern leakage.
    reserved_keys = {record.get("record_key") for record in reserved}
    dev_keys = {record.get("record_key") for record in dev_records}
    test_overlap = sorted(dev_keys & reserved_keys)

    segmentation = load_json(SEGMENTATION)
    segmentation_by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in segmentation.get("records", []):
        if record.get("split") != "train":
            continue
        segmentation_by_id[canonical_id(record.get("image_id", ""))][record.get("mask_class", "")] = record

    localization = load_json(LOCALIZATION)
    localization_by_id: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in localization.get("records", []):
        if record.get("split") == "train":
            localization_by_id[canonical_id(record.get("image_id", ""))][record.get("annotation_type", "")] = record

    inspection: list[dict[str, Any]] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    dimensions: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    missing: list[str] = []
    corrupt: list[dict[str, Any]] = []
    for record in dev_records:
        source_path = project_path(record["source_path"])
        item = inspect_image(source_path)
        item.update({
            "image_id": record["image_id"],
            "canonical_image_id": canonical_id(record["image_id"]),
            "split": record["split"],
            "label": int(record["label"]),
            "dr_grade": int(record["dr_grade"]),
            "dme_grade": record.get("dme_grade"),
        })
        inspection.append(item)
        labels[str(record["label"])] += 1
        if item["sha256"]:
            by_hash[item["sha256"]].append(record["image_id"])
        if item["width"] is not None:
            dimensions[f"{item['width']}x{item['height']}x{item['channels']}"] += 1
        if not item["exists"]:
            missing.append(record["image_id"])
        elif not item["readable"]:
            corrupt.append(item)

    duplicate_groups = [
        {"sha256": digest, "image_ids": sorted(ids), "count": len(ids)}
        for digest, ids in sorted(by_hash.items()) if len(ids) > 1
    ]
    split_hashes = defaultdict(set)
    for item in inspection:
        if item["sha256"]:
            split_hashes[item["split"]].add(item["sha256"])
    split_duplicate_overlap = sorted(split_hashes["train"] & split_hashes["validation"])

    enriched: list[dict[str, Any]] = []
    for record in dev_records:
        cid = canonical_id(record["image_id"])
        lesions: dict[str, dict[str, Any]] = {}
        for lesion_name in ("Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc"):
            annotation = segmentation_by_id.get(cid, {}).get(lesion_name)
            available = bool(annotation and annotation.get("annotation_status") == "AVAILABLE" and annotation.get("readable"))
            lesions[lesion_name] = {
                "available": available,
                "annotation_status": annotation.get("annotation_status") if annotation else "UNAVAILABLE",
                "mask_path": raw_relative(annotation["mask_path"]) if available else None,
                "mask_class": annotation.get("mask_class") if annotation else lesion_name,
                "source_image_path": raw_relative(annotation["original_image_path"]) if annotation else None,
                "missing_is_not_negative": True,
            }
        landmarks: dict[str, Any] = {}
        for kind in ("optic_disc_center", "fovea_center"):
            annotation = localization_by_id.get(cid, {}).get(kind)
            landmarks[kind] = {
                "available": bool(annotation and annotation.get("coordinate_valid")),
                "x": annotation.get("x") if annotation else None,
                "y": annotation.get("y") if annotation else None,
                "width": annotation.get("image_width") if annotation else None,
                "height": annotation.get("image_height") if annotation else None,
                "source": annotation.get("source_csv") if annotation else None,
            }
        enriched.append({
            "record_key": record["record_key"],
            "image_id": record["image_id"],
            "canonical_image_id": cid,
            "image": raw_relative(record["source_path"]),
            "source_path": record["source_path"],
            "sha256": next(item["sha256"] for item in inspection if item["image_id"] == record["image_id"]),
            "split": record["split"],
            "label": int(record["label"]),
            "dr_grade": int(record["dr_grade"]),
            "dme_grade": record.get("dme_grade"),
            "duplicate_group_id": record.get("duplicate_group_id"),
            "lesion_annotations": lesions,
            "landmarks": landmarks,
        })

    return {
        "dataset": "idrid",
        "purpose": "Isolated v2 development manifest; official test remains reserved",
        "source_split_manifest": str(SOURCE_SPLIT.relative_to(ROOT)).replace("\\", "/"),
        "source_split_manifest_sha256": sha256(SOURCE_SPLIT),
        "official_test_policy": "No official test image is opened or used for v2 development, tuning, or threshold selection.",
        "official_test_record_count_reserved": len(reserved),
        "official_test_image_accessed": False,
        "records": enriched,
        "counts": {
            "total": len(enriched),
            "train": sum(record["split"] == "train" for record in enriched),
            "validation": sum(record["split"] == "validation" for record in enriched),
            "class_distribution": {split_name: dict(Counter(str(r["label"]) for r in enriched if r["split"] == split_name)) for split_name in ("train", "validation")},
        },
        "image_audit": {
            "missing": missing,
            "corrupt_or_unreadable": corrupt,
            "readable_count": sum(item["readable"] for item in inspection),
            "dimensions": dict(dimensions),
            "exact_duplicate_groups_within_development": duplicate_groups,
            "train_validation_hash_overlap": split_duplicate_overlap,
            "patient_ids_available": False,
            "patient_level_leakage_guarantee": False,
            "limitation": "IDRiD does not provide patient identifiers; exact image hashes and the source governance exclusions are used.",
        },
        "annotation_availability": {
            lesion: sum(record["lesion_annotations"][lesion]["available"] for record in enriched)
            for lesion in ("Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc")
        },
        "landmark_availability": {
            kind: sum(record["landmarks"][kind]["available"] for record in enriched)
            for kind in ("optic_disc_center", "fovea_center")
        },
        "integrity": {
            "development_record_count_matches_source": len(enriched) == split.get("eligible_records"),
            "reserved_test_record_keys_overlap_development": test_overlap,
            "numeric_id_overlap_is_expected": True,
            "source_leakage_status": split.get("leakage", {}).get("status"),
            # Exact duplicates within one split are retained only when they
            # have the same governed label and are kept together by the source
            # manifest.  They are reported, but do not by themselves fail the
            # split-integrity gate; any train/validation hash overlap does.
            "status": "PASS" if not missing and not corrupt and not split_duplicate_overlap and not test_overlap else "REVIEW_REQUIRED",
        },
    }, enriched


def audit_aptos(dev_hashes: set[str]) -> dict[str, Any]:
    train_csv = APTOS_RAW / "train.csv"
    image_dir = APTOS_RAW / "train_images"
    rows: list[dict[str, str]] = []
    with train_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    label_counts = Counter()
    missing: list[str] = []
    invalid: list[dict[str, Any]] = []
    readable = 0
    hashes: dict[str, list[str]] = defaultdict(list)
    dimensions: Counter[str] = Counter()
    for row in rows:
        image_id = row.get("id_code", "")
        raw_label = row.get("diagnosis", "")
        path = image_dir / f"{image_id}.png"
        if raw_label not in {"0", "1", "2", "3", "4"}:
            invalid.append({"id_code": image_id, "diagnosis": raw_label})
        else:
            label_counts[raw_label] += 1
        info = inspect_image(path)
        if not info["exists"]:
            missing.append(image_id)
        elif info["readable"]:
            readable += 1
            hashes[info["sha256"]].append(image_id)
            dimensions[f"{info['width']}x{info['height']}x{info['channels']}"] += 1
    duplicate_groups = [{"sha256": h, "ids": sorted(ids), "count": len(ids)} for h, ids in sorted(hashes.items()) if len(ids) > 1]
    cross_dataset = [
        {"sha256": h, "idrid_image_ids": sorted(ids), "aptos_image_ids": sorted(hashes[h])}
        for h, ids in sorted({h: ids for h, ids in []}.items())
    ]
    # Keep the cross-dataset check explicit and memory-bounded: only hashes from
    # eligible IDRiD records are candidates for a match.
    aptos_matches = [{"sha256": h, "aptos_image_ids": sorted(ids)} for h, ids in sorted(hashes.items()) if h in dev_hashes]
    report = load_json(APTOS_REPORT) if APTOS_REPORT.is_file() else {}
    return {
        "dataset": "aptos2019",
        "scope": "labelled training images only; test images excluded",
        "train_csv": str(train_csv.relative_to(ROOT)).replace("\\", "/"),
        "image_directory": str(image_dir.relative_to(ROOT)).replace("\\", "/"),
        "label_rows": len(rows),
        "resolved_images": len(rows) - len(missing),
        "readable_images": readable,
        "missing_images": missing,
        "invalid_labels": invalid,
        "class_distribution": dict(label_counts),
        "dimensions": dict(dimensions),
        "exact_duplicate_groups": duplicate_groups,
        "existing_governance_report": {
            "sha256": sha256(APTOS_REPORT) if APTOS_REPORT.is_file() else None,
            "duplicate_label_conflicts": len(report.get("duplicate_label_conflicts", [])),
            "patient_level_guarantee": report.get("leakage", {}).get("patient_level_guarantee"),
        },
        "exact_hash_matches_with_idrid_development": aptos_matches,
        "note": "APTOS labels are audited for optional research initialization only; its test split is not used.",
    }


def main() -> int:
    for required in (SOURCE_SPLIT, SEGMENTATION, LOCALIZATION):
        if not required.is_file():
            raise FileNotFoundError(required)
    split = load_json(SOURCE_SPLIT)
    audit, records = audit_idrid(split)
    aptos = audit_aptos({record["sha256"] for record in records if record.get("sha256")})
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "idrid-v2-development-1",
        "generated_at_utc": generated,
        "dataset": "idrid",
        "development_only": True,
        "official_test_used": False,
        "official_test_images_opened": 0,
        "class_mapping": {"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative DR"},
        "source_governance": audit,
        "aptos_cross_dataset_audit": aptos,
        "records": records,
    }
    audit_report = {
        "schema_version": "idrid-v2-audit-1",
        "generated_at_utc": generated,
        "idrid": audit,
        "aptos2019": aptos,
        "conclusions": {
            "idrid_development_ready_for_isolated_research": audit["integrity"]["status"] == "PASS",
            "aptos_optional_initialization_ready": not aptos["missing_images"] and not aptos["invalid_labels"],
            "patient_level_split_guarantee": False,
            "official_test_untouched": True,
        },
    }
    META.mkdir(parents=True, exist_ok=True)
    (META / "idrid_v2_development_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (META / "idrid_v2_data_audit.json").write_text(json.dumps(audit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str((META / "idrid_v2_development_manifest.json").relative_to(ROOT)).replace("\\", "/"),
        "audit": str((META / "idrid_v2_data_audit.json").relative_to(ROOT)).replace("\\", "/"),
        "idrid_counts": audit["counts"],
        "idrid_integrity": audit["integrity"],
        "aptos": {key: aptos[key] for key in ("label_rows", "readable_images", "class_distribution", "exact_hash_matches_with_idrid_development")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"idrid v2 development audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
