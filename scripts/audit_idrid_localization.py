"""Read-only forensic audit and manifest builder for IDRiD localization."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.localization.idrid import LANDMARKS, build_records, split_training_records  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    records, audit_info = build_records(include_test=True)
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    split = split_training_records(records, folds=5, seed=20260913)
    excluded_cross_split = split["excluded_cross_split_duplicates"]
    unreadable = [record["image_id"] for record in records if record["image"].get("read_error")]
    missing = {name: [record["image_id"] for record in records if not record["annotations"][name]["valid"]] for name in LANDMARKS}
    invalid = {name: [record["image_id"] for record in records if record["annotations"][name].get("error") == "coordinate outside image bounds"] for name in LANDMARKS}
    exact_groups: dict[str, list[dict]] = defaultdict(list)
    phash_groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        exact_groups[record["duplicate_group"]].append({"image_id": record["image_id"], "split": record["split"], "path": record["image_path"]})
        phash_groups[record["perceptual_duplicate_group"]].append({"image_id": record["image_id"], "split": record["split"], "path": record["image_path"]})
    exact_duplicates = [items for items in exact_groups.values() if len(items) > 1]
    perceptual_duplicates = [items for items in phash_groups.values() if len(items) > 1]
    coordinate_ranges = {}
    for name in LANDMARKS:
        points = [record["annotations"][name] for record in records if record["annotations"][name]["valid"]]
        coordinate_ranges[name] = {"x_min": min(point["x"] for point in points), "x_max": max(point["x"] for point in points), "y_min": min(point["y"] for point in points), "y_max": max(point["y"] for point in points), "x_norm_mean": float(np.mean([point["x_norm"] for point in points])), "y_norm_mean": float(np.mean([point["y_norm"] for point in points]))}
    audit = {
        "schema_version": "idrid-localization-audit-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "IDRiD C. Localization",
        "raw_files_untouched": True,
        "training_images": len(train),
        "development_images_after_leakage_exclusion": split["development_count"],
        "official_test_images": len(test),
        "image_dimensions": [list(item) for item in sorted({(record["image"].get("width"), record["image"].get("height")) for record in records if not record["image"].get("read_error")})],
        "image_formats": dict(Counter(record["image"].get("format") for record in records)),
        "image_modes": dict(Counter(record["image"].get("mode") for record in records)),
        "unreadable_images": unreadable,
        "annotation_counts": {split: {name: sum(record["annotations"][name]["valid"] for record in records if record["split"] == split) for name in LANDMARKS} for split in ("train", "test")},
        "missing_coordinates": missing,
        "out_of_image_coordinates": invalid,
        "duplicate_images_exact": exact_duplicates,
        "duplicate_images_perceptual": perceptual_duplicates,
        "cross_split_duplicate_exclusions": excluded_cross_split,
        "duplicate_coordinate_records": {item["landmark"]: item["duplicate_records"] for item in audit_info["csv_files"]},
        "conflicting_coordinate_records": {item["landmark"]: item["conflicting_records"] for item in audit_info["csv_files"]},
        "coordinate_ranges": coordinate_ranges,
        "coordinate_system": audit_info["coordinate_convention"],
        "coordinate_units": audit_info["units"],
        "patient_ids_available": False,
        "csv_files": audit_info["csv_files"],
        "official_test_used_for_training_or_selection": False,
        "status": "PASS" if not unreadable and not any(missing.values()) and not any(invalid.values()) else "REVIEW_REQUIRED",
    }
    dump(META / "idrid_localization_data_audit.json", audit)
    dump(META / "idrid_localization_manifest.json", {"schema_version": "idrid-localization-manifest-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "IDRiD C. Localization", "records": records, "coordinate_system": audit_info["coordinate_convention"], "official_test_used_for_training_or_selection": False})
    dump(META / "idrid_localization_split.json", split)
    print(json.dumps({"training_images": len(train), "development_images": split["development_count"], "official_test_images": len(test), "annotation_counts": audit["annotation_counts"], "exact_duplicate_groups": len(exact_duplicates), "perceptual_duplicate_groups": len(perceptual_duplicates), "cross_split_duplicate_exclusions": excluded_cross_split, "status": audit["status"], "official_test_used_for_training_or_selection": False}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
