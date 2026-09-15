"""Audit and manifest the official IDRiD segmentation package.

This command reads raw files only. It never renames, moves, rewrites, or
trains on the official testing split. Missing annotations are recorded as
UNAVAILABLE rather than converted to all-zero masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
sys.path.insert(0, str(ROOT))

from ml.lesions.idrid import LESION_CLASSES, build_records, split_training_records  # noqa: E402


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--skip-pixel-stats", action="store_true")
    args = parser.parse_args()
    records = build_records(include_test=True, include_pixel_stats=not args.skip_pixel_stats)
    train_records = [record for record in records if record["split"] == "train"]
    test_records = [record for record in records if record["split"] == "test"]
    split = split_training_records(records, folds=args.folds, seed=args.seed)
    fold_by_id = {item["image_id"]: item["fold"] for item in split["records"]}
    for record in records:
        if record["split"] == "train":
            record["fold"] = fold_by_id[record["image_id"]]
        else:
            record["fold"] = None
    exact_groups = defaultdict(list)
    phash_groups = defaultdict(list)
    for record in records:
        exact_groups[record["duplicate_group"]].append(record["image_id"])
        phash_groups[record["perceptual_duplicate_group"]].append(record["image_id"])
    mask_summary = {}
    for lesion_class in LESION_CLASSES:
        available = [record for record in records if record["masks"][lesion_class]["status"] == "AVAILABLE"]
        unavailable = [record for record in records if record["masks"][lesion_class]["status"] == "UNAVAILABLE"]
        unreadable = [record for record in records if record["masks"][lesion_class]["status"] == "UNREADABLE"]
        mask_summary[lesion_class] = {
            "available_total": len(available),
            "available_train": sum(record["split"] == "train" for record in available),
            "available_test": sum(record["split"] == "test" for record in available),
            "unavailable_total": len(unavailable),
            "unavailable_train": sum(record["split"] == "train" for record in unavailable),
            "unavailable_test": sum(record["split"] == "test" for record in unavailable),
            "unreadable_total": len(unreadable),
            "positive_train_images": sum(bool(record["masks"][lesion_class].get("active_pixels")) for record in train_records if record["masks"][lesion_class]["status"] == "AVAILABLE"),
            "active_pixels_train": sum(int(record["masks"][lesion_class].get("active_pixels") or 0) for record in train_records if record["masks"][lesion_class]["status"] == "AVAILABLE"),
        }
    audit = {
        "schema_version": "idrid-lesion-audit-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "IDRiD A. Segmentation",
        "raw_root": "ml/datasets/raw/idrid/A. Segmentation/A. Segmentation",
        "training_images": len(train_records),
        "official_testing_images": len(test_records),
        "official_testing_predictions_or_model_selection": False,
        "image_formats": dict(Counter(record["image"].get("format") for record in records)),
        "image_modes": dict(Counter(record["image"].get("mode") for record in records)),
        "image_dimensions": dict(Counter(f"{record['image'].get('width')}x{record['image'].get('height')}" for record in records)),
        "corrupt_images": [record["image_id"] for record in records if record["image"].get("read_error")],
        "mask_summary": mask_summary,
        "mask_missing_policy": "UNAVAILABLE masks are excluded from supervised loss and class-specific evaluation; they are never treated as negative masks.",
        "fov": {"status": "UNAVAILABLE", "reason": "No official FOV masks were found in the IDRiD segmentation package."},
        "exact_duplicate_groups": {key: value for key, value in exact_groups.items() if len(value) > 1},
        "perceptual_duplicate_groups": {key: value for key, value in phash_groups.items() if len(value) > 1},
        "duplicate_group_cross_split": [],
        "special_encoding": {"image_id": "IDRiD_81", "file": "IDRiD_81_EX.tif", "handling": "RGBA hard-exudate mask uses the varying red channel; raw TIFF is not rewritten."},
        "split_artifact": "ml/datasets/metadata/idrid/idrid_lesion_split.json",
        "official_test_set_status": "FROZEN_UNTIL_FINAL_MODEL_SELECTION",
    }
    manifest = {
        "schema_version": "idrid-lesion-manifest-1",
        "generated_at_utc": audit["generated_at_utc"],
        "dataset": "IDRiD A. Segmentation",
        "target_classes": list(LESION_CLASSES),
        "optic_disc_policy": "Optic Disc is audited separately and is excluded from lesion model targets.",
        "missing_mask_policy": audit["mask_missing_policy"],
        "official_test_set_status": audit["official_test_set_status"],
        "records": records,
    }
    dump(META / "idrid_lesion_data_audit.json", audit)
    dump(META / "idrid_lesion_manifest.json", manifest)
    dump(META / "idrid_lesion_split.json", split)
    print(json.dumps({"audit": str((META / "idrid_lesion_data_audit.json").relative_to(ROOT)), "manifest": str((META / "idrid_lesion_manifest.json").relative_to(ROOT)), "split": str((META / "idrid_lesion_split.json").relative_to(ROOT)), "training_images": len(train_records), "official_testing_images": len(test_records), "official_test_used_for_training_or_selection": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
