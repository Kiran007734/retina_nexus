from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from dataset_common import (
    DatasetError,
    DATA_ROOT,
    build_image_index,
    get_definition,
    image_files,
    leakage_report,
    perceptual_hash,
    raw_path_for,
    read_annotation_rows,
    resolve_image_ref,
    scan_images,
    split_directory,
    utc_now,
    write_json,
)


def build_records(raw: Path, definition: dict, files: list[Path], scan: dict) -> tuple[list[dict], bool]:
    index = build_image_index(files, raw)
    rows, _ = read_annotation_rows(raw, definition)
    by_image: dict[Path, dict] = {}
    for row in rows:
        path = resolve_image_ref(row.get("image_ref"), index, raw, definition)
        if path:
            by_image[path] = row
    phash_groups: dict[str, str] = {}
    for duplicate_index, group in enumerate(scan["perceptual_duplicate_groups"], start=1):
        group_id = f"duplicate-{duplicate_index:04d}"
        for relative in group:
            phash_groups[relative] = group_id
    records: list[dict] = []
    patient_groups_available = False
    for item in scan["readable"]:
        path = raw / item["path"]
        row = by_image.get(path, {})
        patient_group = row.get("group")
        if patient_group:
            patient_groups_available = True
            group_id = f"patient:{patient_group}"
            group_source = "patient_id"
        else:
            group_id = phash_groups.get(item["path"], f"image:{item['path']}")
            group_source = "duplicate_or_image"
        records.append({
            "image": item["path"], "label": row.get("label"),
            "group_id": group_id, "group_source": group_source,
            "duplicate_group_id": phash_groups.get(item["path"], f"unique:{item['path']}"),
        })
    return records, patient_groups_available


def assign_splits(records: list[dict], ratios: dict[str, float], seed: int) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    randomized = list(groups.items())
    random.Random(seed).shuffle(randomized)
    randomized.sort(key=lambda pair: len(pair[1]), reverse=True)
    targets = {split: len(records) * ratio for split, ratio in ratios.items()}
    counts = {split: 0 for split in ratios}
    for _, group_records in randomized:
        selected = min(ratios, key=lambda split: counts[split] - targets[split])
        for record in group_records:
            record["split"] = selected
        counts[selected] += len(group_records)


def validate_split_manifest(records: list[dict]) -> dict:
    groups: dict[str, set[str]] = defaultdict(set)
    duplicates: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record["group_id"]].add(record["split"])
        duplicates[record["duplicate_group_id"]].add(record["split"])
    cross_groups = sorted(group for group, splits in groups.items() if len(splits) > 1)
    cross_duplicates = sorted(group for group, splits in duplicates.items() if len(splits) > 1)
    patient_level = any(record["group_source"] == "patient_id" for record in records)
    return {
        "status": "pass" if not cross_groups and not cross_duplicates else "blocked",
        "patient_level_guarantee": patient_level,
        "cross_split_patient_groups": cross_groups,
        "cross_split_duplicate_groups": cross_duplicates,
        "limitations": [] if patient_level else ["No patient-level identifier was found; only image/duplicate grouping was enforced."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create group-aware train/validation/test split manifests")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-dir")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if abs(args.train_ratio + args.validation_ratio + args.test_ratio - 1) > 0.001:
            raise DatasetError("Split ratios must sum to 1.0")
        definition = get_definition(args.dataset)
        raw = raw_path_for(definition, args.raw_dir)
        if not raw.exists():
            raise DatasetError(f"Dataset directory does not exist: {raw}")
        files = image_files(raw, definition)
        if not files:
            raise DatasetError(f"No image files found under {raw}.")
        scan = scan_images(files, raw)
        records, patient_groups_available = build_records(raw, definition, files, scan)
        if not records:
            raise DatasetError("No readable images are available for splitting.")
        assign_splits(records, {"train": args.train_ratio, "validation": args.validation_ratio, "test": args.test_ratio}, args.seed)
        leakage = validate_split_manifest(records)
        payload = {
            "dataset": args.dataset, "created_at": utc_now(), "seed": args.seed,
            "ratios": {"train": args.train_ratio, "validation": args.validation_ratio, "test": args.test_ratio},
            "counts": {split: sum(1 for record in records if record["split"] == split) for split in ["train", "validation", "test"]},
            "records": sorted(records, key=lambda record: record["image"]),
            "leakage": leakage,
            "note": "Manifest only; source images are not copied. Group-aware separation is guaranteed only for identifiers present in source metadata.",
        }
        output = write_json(split_directory(args.dataset) / "splits.json", payload)
        top_level = write_json(DATA_ROOT / "metadata" / "data_leakage_report.json", {"dataset": args.dataset, **leakage})
        print(f"Created split manifest: {output}")
        print(f"Split counts: {payload['counts']}")
        print(f"Patient-level guarantee: {patient_groups_available}")
        print(f"Leakage report: {top_level}")
        return 0 if leakage["status"] == "pass" else 1
    except DatasetError as exc:
        print(f"SPLIT CREATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
