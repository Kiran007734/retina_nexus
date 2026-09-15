"""Audit the actual DRIVE layout and build the vessel research manifest.

The raw dataset is read-only.  This script discovers the nested archive
structure, validates image/annotation correspondence, and creates metadata
without moving or rewriting any DRIVE files.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluation.drive import duplicate_groups, discover_drive_files, group_drive_files, perceptual_duplicate_candidates, perceptual_hash, read_file_info, resized_grayscale_mae  # noqa: E402
from ml.vessels.drive import make_five_fold_split, sha256  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "drive"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    raw_root = ROOT / "ml" / "datasets" / "raw" / "drive"
    discovered = discover_drive_files(raw_root)
    all_files = []
    for category, paths in discovered.items():
        for path in paths:
            record = read_file_info(path, raw_root, category)
            if category == "image" and record["readable"]:
                record["perceptual_hash"] = perceptual_hash(path)
            else:
                record["perceptual_hash"] = None
            all_files.append(record)
    grouped = group_drive_files(raw_root)
    file_by_path = {record["path"]: record for record in all_files}
    relationships = []
    missing = []
    mismatches = []
    invalid = []
    manifest_records = []
    for image_id, categories in grouped.items():
        images = categories.get("image", [])
        if len(images) != 1:
            continue
        image = images[0]
        image_info = file_by_path[image.relative_to(raw_root).as_posix()]
        vessels = categories.get("vessel_mask", [])
        fovs = categories.get("fov_mask", [])
        vessel = vessels[0] if len(vessels) == 1 else None
        fov = fovs[0] if len(fovs) == 1 else None
        vessel_info = file_by_path[vessel.relative_to(raw_root).as_posix()] if vessel else None
        fov_info = file_by_path[fov.relative_to(raw_root).as_posix()] if fov else None
        split = image_info["split"]
        if split == "training" and vessel is None:
            missing.append({"image_id": image_id, "missing": "manual_vessel_mask"})
        if fov is None:
            missing.append({"image_id": image_id, "missing": "field_of_view_mask"})
        for annotation, info in (("vessel_mask", vessel_info), ("field_of_view_mask", fov_info)):
            if not info:
                continue
            if info["annotation_valid"] is False:
                invalid.append({"image_id": image_id, "annotation": annotation, "path": info["path"], "reason": info["error"]})
            if image_info["readable"] and info["readable"] and (image_info["width"], image_info["height"]) != (info["width"], info["height"]):
                mismatches.append({"image_id": image_id, "annotation": annotation, "image_size": [image_info["width"], image_info["height"]], "annotation_size": [info["width"], info["height"]]})
        vessel_pixels = None
        fov_pixels = None
        if vessel_info and vessel_info["readable"]:
            with Image.open(raw_root / vessel_info["path"]) as annotation:
                vessel_pixels = int((np.asarray(annotation.convert("L")) >= 128).sum())
        if fov_info and fov_info["readable"]:
            with Image.open(raw_root / fov_info["path"]) as annotation:
                fov_pixels = int((np.asarray(annotation.convert("L")) >= 128).sum())
        manifest_records.append({"image_id": image_id, "split": split, "image_path": image_info["path"], "vessel_mask_path": vessel_info["path"] if vessel_info else None, "fov_mask_path": fov_info["path"] if fov_info else None, "width": image_info["width"], "height": image_info["height"], "channels": image_info["channels"], "format": image_info["format"], "image_sha256": image_info["sha256"], "image_perceptual_hash": image_info.get("perceptual_hash"), "vessel_pixel_count": vessel_pixels, "vessel_percentage": float(vessel_pixels / max(1, image_info["width"] * image_info["height"])) if vessel_pixels is not None else None, "fov_pixel_count": fov_pixels, "fov_percentage": float(fov_pixels / max(1, image_info["width"] * image_info["height"])) if fov_pixels is not None else None, "vessel_mask_sha256": vessel_info["sha256"] if vessel_info else None, "fov_mask_sha256": fov_info["sha256"] if fov_info else None})
        relationships.append({"image_id": image_id, "split": split, "image": image_info["path"], "vessel_mask": vessel_info["path"] if vessel_info else None, "field_of_view_mask": fov_info["path"] if fov_info else None})

    image_records = [record for record in all_files if record["category"] == "image"]
    exact = duplicate_groups(image_records, "sha256")
    candidates = perceptual_duplicate_candidates(image_records, "perceptual_hash", max_distance=4)
    for candidate in candidates:
        candidate["normalized_thumbnail_mae"] = resized_grayscale_mae(raw_root / candidate["path_a"], raw_root / candidate["path_b"])
    confirmed_perceptual = [candidate for candidate in candidates if candidate["normalized_thumbnail_mae"] <= 0.02]
    split_counts = Counter(record["split"] for record in manifest_records)
    unreadable = [record for record in all_files if record["category"] in {"image", "vessel_mask", "fov_mask"} and not record["readable"]]
    cross_split = [group for group in exact if len({file_by_path[path]["split"] for path in group}) > 1]
    validation_status = "PASS" if not unreadable and not missing and not mismatches and not invalid and not cross_split else "REVIEW_REQUIRED"
    dataset_material = "\n".join(f"{record['image_path']}|{record['image_sha256']}" for record in sorted(manifest_records, key=lambda row: row["image_path"]))
    dataset_version = "drive-vessel-" + sha256_from_text(dataset_material)[:16]
    audit = {"schema_version": "drive-vessel-audit-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "DRIVE", "raw_root": "ml/datasets/raw/drive/datasets", "detected_layout": {"training_images": sorted(record["image_path"] for record in manifest_records if record["split"] == "training"), "test_images": sorted(record["image_path"] for record in manifest_records if record["split"] == "test"), "training_vessel_masks": sorted(record["vessel_mask_path"] for record in manifest_records if record["split"] == "training" and record["vessel_mask_path"]), "fov_masks": sorted(record["fov_mask_path"] for record in manifest_records if record["fov_mask_path"])}, "dataset_version": dataset_version, "counts": {"training_images": split_counts["training"], "test_images": split_counts["test"], "training_manual_vessel_masks": sum(record["split"] == "training" and record["vessel_mask_path"] is not None for record in manifest_records), "test_manual_vessel_masks": sum(record["split"] == "test" and record["vessel_mask_path"] is not None for record in manifest_records), "fov_masks": sum(record["fov_mask_path"] is not None for record in manifest_records), "corrupt_or_unreadable_files": len(unreadable)}, "dimensions": sorted({(record["width"], record["height"], record["channels"]) for record in manifest_records}), "encoding": {"vessel_mask_foreground": ">=128", "fov_mask_foreground": ">=128", "observed_annotation_values": sorted({value for record in all_files if record["category"] != "image" for value in (record.get("unique_values") or [])})}, "validation": {"status": validation_status, "unreadable_files": unreadable, "missing_pairs": missing, "dimension_mismatches": mismatches, "invalid_annotations": invalid, "cross_split_exact_duplicates": cross_split, "fov_masks_available_for_training_and_test": all(record["fov_mask_path"] for record in manifest_records), "test_manual_vessel_ground_truth_available": any(record["split"] == "test" and record["vessel_mask_path"] for record in manifest_records), "test_accuracy_limitation": "The supplied DRIVE test split contains FOV masks but no manual vessel masks; official test vessel accuracy cannot be measured until a genuine test mask package is supplied."}, "duplicates": {"exact_image_duplicate_groups": exact, "perceptual_candidates": candidates, "confirmed_perceptual_duplicate_pairs": confirmed_perceptual, "cross_split_duplicate_groups": cross_split}, "patient_ids_available": False, "official_test_used_for_model_selection": False, "note": "FOV masks are supplied and used. No FOV masks or vessel labels were fabricated."}
    split = make_five_fold_split([item for item in manifest_records if item["split"] == "training"], seed=20260913)
    split["official_test_records"] = [{"image_id": record["image_id"], "split": "official_test_reserved"} for record in manifest_records if record["split"] == "test"]
    dump(META / "drive_data_audit.json", audit)
    dump(META / "drive_manifest.json", {"schema_version": "drive-manifest-1", "dataset": "DRIVE", "dataset_version": dataset_version, "records": sorted(manifest_records, key=lambda row: row["image_id"]), "raw_files": all_files, "official_test_used_for_model_selection": False})
    dump(META / "drive_split.json", split)
    print(json.dumps({"status": validation_status, "dataset_version": dataset_version, "training_images": split_counts["training"], "test_images": split_counts["test"], "fov_masks": audit["counts"]["fov_masks"], "test_manual_vessel_masks": audit["counts"]["test_manual_vessel_masks"], "exact_duplicates": len(exact), "confirmed_perceptual_duplicates": len(confirmed_perceptual), "official_test_used_for_model_selection": False}, indent=2))
    return 0 if validation_status == "PASS" else 2


def sha256_from_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
