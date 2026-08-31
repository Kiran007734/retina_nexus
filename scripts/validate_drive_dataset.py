"""Validate a manually placed DRIVE dataset and write a truthful manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.evaluation.drive import (  # noqa: E402
    duplicate_groups,
    discover_drive_files,
    group_drive_files,
    perceptual_duplicate_candidates,
    perceptual_hash,
    read_file_info,
    resized_grayscale_mae,
    split_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="ml/datasets/raw/drive")
    parser.add_argument("--output-dir", default="ml/evaluation/drive")
    return parser.parse_args()


def _path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _inventory_version(records: list[dict[str, Any]]) -> str:
    material = "\n".join(f"{row['path']}|{row['size_bytes']}|{row['sha256']}" for row in sorted(records, key=lambda item: item["path"]))
    return "drive-inventory-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _relative(root: Path, paths: list[Path]) -> list[str]:
    return [path.resolve().relative_to(root.resolve()).as_posix() for path in paths]


def main() -> int:
    args = parse_args()
    raw_root = _path(ROOT, args.raw_dir)
    output_root = _path(ROOT, args.output_dir)
    if not raw_root.is_dir():
        print(f"DRIVE VALIDATION ERROR: dataset directory does not exist: {raw_root}", file=sys.stderr)
        return 2

    discovered = discover_drive_files(raw_root)
    file_records: list[dict[str, Any]] = []
    for category in ("image", "vessel_mask", "fov_mask"):
        for path in discovered[category]:
            record = read_file_info(path, raw_root, category)
            if category == "image" and record["readable"]:
                try:
                    record["perceptual_hash"] = perceptual_hash(path)
                except Exception as exc:
                    record["perceptual_hash"] = None
                    record["error"] = f"perceptual hash failed: {type(exc).__name__}: {exc}"
            else:
                record["perceptual_hash"] = None
            file_records.append(record)

    by_category_split = Counter((record["category"], record["split"]) for record in file_records)
    grouped = group_drive_files(raw_root)
    relationships: list[dict[str, Any]] = []
    missing_masks: list[dict[str, Any]] = []
    dimension_mismatches: list[dict[str, Any]] = []
    invalid_annotations: list[dict[str, Any]] = []
    ambiguous_pairs: list[dict[str, Any]] = []
    info_by_path = {record["path"]: record for record in file_records}
    for key, categories in grouped.items():
        images = categories.get("image", [])
        vessels = categories.get("vessel_mask", [])
        fovs = categories.get("fov_mask", [])
        if len(images) > 1 or len(vessels) > 1 or len(fovs) > 1:
            ambiguous_pairs.append({"specimen_id": key, "images": _relative(raw_root, images), "vessel_masks": _relative(raw_root, vessels), "fov_masks": _relative(raw_root, fovs)})
        if not images:
            continue
        image = images[0]
        image_record = info_by_path[image.resolve().relative_to(raw_root.resolve()).as_posix()]
        vessel = vessels[0] if vessels else None
        fov = fovs[0] if fovs else None
        vessel_record = info_by_path[vessel.resolve().relative_to(raw_root.resolve()).as_posix()] if vessel else None
        fov_record = info_by_path[fov.resolve().relative_to(raw_root.resolve()).as_posix()] if fov else None
        expected_vessel = image_record["split"] == "training"
        if expected_vessel and vessel is None:
            missing_masks.append({"specimen_id": key, "split": image_record["split"], "image": image_record["path"], "missing": "vessel_mask"})
        if fov is None:
            missing_masks.append({"specimen_id": key, "split": image_record["split"], "image": image_record["path"], "missing": "field_of_view_mask"})
        for annotation_name, annotation_record in (("vessel_mask", vessel_record), ("field_of_view_mask", fov_record)):
            if annotation_record is None:
                continue
            if annotation_record.get("annotation_valid") is False:
                invalid_annotations.append({"specimen_id": key, "annotation": annotation_record["path"], "reason": annotation_record.get("error")})
            if image_record["readable"] and annotation_record["readable"] and (image_record["width"], image_record["height"]) != (annotation_record["width"], annotation_record["height"]):
                dimension_mismatches.append({"specimen_id": key, "image": image_record["path"], "annotation": annotation_record["path"], "image_size": [image_record["width"], image_record["height"]], "annotation_size": [annotation_record["width"], annotation_record["height"]]})
        relationships.append({
            "specimen_id": key,
            "split": image_record["split"],
            "image": image_record["path"],
            "vessel_mask": vessel_record["path"] if vessel_record else None,
            "field_of_view_mask": fov_record["path"] if fov_record else None,
            "has_vessel_ground_truth": vessel_record is not None,
            "has_field_of_view_mask": fov_record is not None,
        })

    image_records = [record for record in file_records if record["category"] == "image"]
    exact_duplicates = duplicate_groups(image_records, "sha256")
    perceptual_candidates = perceptual_duplicate_candidates(image_records, "perceptual_hash", max_distance=4)
    image_by_path = {record["path"]: record for record in image_records}
    for candidate in perceptual_candidates:
        candidate["normalized_thumbnail_mae"] = resized_grayscale_mae(raw_root / candidate["path_a"], raw_root / candidate["path_b"])
    # pHash is intentionally a candidate generator.  Fundus images share a
    # circular field and can have small pHash distances without being copies.
    # Confirm candidates with a normalized thumbnail distance before treating
    # them as duplicates or leakage.
    perceptual_duplicates = [candidate for candidate in perceptual_candidates if float(candidate["normalized_thumbnail_mae"]) <= 0.02]
    cross_split_duplicates = [
        [candidate["path_a"], candidate["path_b"]]
        for candidate in [{"path_a": path_a, "path_b": path_b} for group in exact_duplicates for path_a, path_b in zip(group, group[1:])]
        if image_by_path[candidate["path_a"]]["split"] != image_by_path[candidate["path_b"]]["split"]
    ]
    cross_split_duplicates += [
        [candidate["path_a"], candidate["path_b"]]
        for candidate in perceptual_duplicates
        if image_by_path[candidate["path_a"]]["split"] != image_by_path[candidate["path_b"]]["split"]
    ]
    corrupt_files = [record for record in file_records if not record["readable"]]
    unreadable_annotations = [record for record in file_records if record["category"] != "image" and not record["readable"]]
    class_counts = Counter(record["split"] for record in image_records)
    dataset_version = _inventory_version(file_records)
    report = {
        "report_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "DRIVE",
        "dataset_version": dataset_version,
        "raw_directory": "ml/datasets/raw/drive",
        "source_layout": {
            "training_images": _relative(raw_root, [path for path in discovered["image"] if split_name(path) == "training"]),
            "test_images": _relative(raw_root, [path for path in discovered["image"] if split_name(path) == "test"]),
            "manual_vessel_masks": _relative(raw_root, discovered["vessel_mask"]),
            "field_of_view_masks": _relative(raw_root, discovered["fov_mask"]),
            "unknown_supported_files": _relative(raw_root, discovered["unknown"]),
            "unsupported_files": _relative(raw_root, discovered["unsupported"]),
        },
        "counts": {
            "all_supported_images": len(image_records),
            "training_images": class_counts["training"],
            "test_images": class_counts["test"],
            "manual_vessel_masks": len(discovered["vessel_mask"]),
            "field_of_view_masks": len(discovered["fov_mask"]),
            "valid_files": sum(record["readable"] for record in file_records),
            "corrupt_or_unreadable_files": len(corrupt_files),
            "training_vessel_ground_truth_pairs": sum(item["split"] == "training" and item["has_vessel_ground_truth"] for item in relationships),
            "test_vessel_ground_truth_pairs": sum(item["split"] == "test" and item["has_vessel_ground_truth"] for item in relationships),
        },
        "file_counts_by_category_and_split": {f"{category}:{split}": count for (category, split), count in sorted(by_category_split.items())},
        "validation": {
            "status": "VALID" if not corrupt_files and not missing_masks and not dimension_mismatches and not invalid_annotations and not ambiguous_pairs else "INVALID",
            "corrupt_files": corrupt_files,
            "unreadable_annotation_files": unreadable_annotations,
            "missing_required_masks": missing_masks,
            "dimension_mismatches": dimension_mismatches,
            "invalid_annotations": invalid_annotations,
            "ambiguous_pairs": ambiguous_pairs,
            "test_manual_vessel_ground_truth_available": bool(any(item["split"] == "test" and item["has_vessel_ground_truth"] for item in relationships)),
            "test_ground_truth_limitation": "The discovered DRIVE copy contains test field-of-view masks but no manual test vessel masks; test images are not included in accuracy metrics.",
        },
        "duplicates": {
            "exact_image_duplicate_groups": exact_duplicates,
            "perceptual_hash_candidates_hamming_le_4": perceptual_candidates,
            "validated_perceptual_duplicate_pairs_hamming_le_4_and_normalized_mae_le_0_02": perceptual_duplicates,
            "cross_split_duplicate_groups": cross_split_duplicates,
            "exact_duplicate_image_count": sum(len(group) for group in exact_duplicates),
            "perceptual_duplicate_image_count": len(perceptual_duplicates),
        },
        "leakage": {
            "patient_identifiers_available": False,
            "patient_level_check": "not_run",
            "image_duplicate_check": "completed",
            "cross_split_duplicate_groups": cross_split_duplicates,
            "limitation": "The discovered DRIVE files do not include patient identifiers. Leakage checks are limited to exact and perceptual duplicate image detection and the dataset's declared training/test specimen IDs.",
        },
        "relationships": relationships,
    }
    manifest = {
        "manifest_version": "1.0.0",
        "dataset": "DRIVE",
        "dataset_version": dataset_version,
        "generated_at": report["generated_at"],
        "root": "ml/datasets/raw/drive",
        "files": sorted(file_records, key=lambda item: item["path"]),
        "relationships": relationships,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["validation"]["status"], "dataset_version": dataset_version, "counts": report["counts"], "duplicates": report["duplicates"], "output_dir": str(output_root)}, indent=2))
    return 0 if report["validation"]["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
