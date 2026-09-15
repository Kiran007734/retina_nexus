"""Create a leakage-aware train/validation manifest for IDRiD grading.

The official IDRiD testing records are intentionally not written into the
training manifest. Exact duplicate groups are kept together; conflicting
training duplicate groups and the training-side copy of the known
train/test duplicate are excluded without changing their raw records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRADING_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "grading_manifest.json"
DEFAULT_DUPLICATE_REPORT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "duplicate_report.json"
DEFAULT_OUTPUT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the IDRiD DR grading train/validation manifest")
    parser.add_argument("--grading-manifest", type=Path, default=DEFAULT_GRADING_MANIFEST)
    parser.add_argument("--duplicate-report", type=Path, default=DEFAULT_DUPLICATE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    return parser.parse_args()


def choose_validation_groups(groups: list[dict[str, Any]], labels: list[int], fraction: float, seed: int) -> set[str]:
    """Choose deterministic class-stratified validation groups.

    The current eligible duplicate groups contain only same-label duplicates,
    but this routine remains group-aware and never separates group members.
    """
    targets = {
        label: max(1, round(count * fraction))
        for label, count in Counter(labels).items()
    }
    selected: set[str] = set()
    selected_counts = Counter()
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        group_labels = set(group["labels"])
        if len(group_labels) == 1:
            by_label[next(iter(group_labels))].append(group)
        else:
            # This is defensive; conflicting groups are excluded before this
            # function is called.
            by_label[min(group_labels)].append(group)
    for label in sorted(by_label):
        rng = random.Random(seed + label * 1009)
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        while candidates and selected_counts[label] < targets.get(label, 0):
            current = selected_counts[label]
            candidates.sort(key=lambda group: (abs(targets[label] - current - group["size"]), group["group_id"]))
            group = candidates.pop(0)
            selected.add(group["group_id"])
            selected_counts[label] += group["size"]
    return selected


def main() -> int:
    args = parse_args()
    grading_path = args.grading_manifest.resolve()
    duplicate_path = args.duplicate_report.resolve()
    output_path = args.output.resolve()
    grading = json.loads(grading_path.read_text(encoding="utf-8"))
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    records = list(grading["records"])
    official_train = [record for record in records if record["split"] == "train"]
    official_test = [record for record in records if record["split"] == "test"]
    duplicate_groups = {group["duplicate_group_id"]: group for group in duplicate["duplicate_groups"]}

    excluded: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for record in official_train:
        group_id = record.get("duplicate_group_id")
        group = duplicate_groups.get(group_id) if group_id else None
        if record.get("potential_train_test_leakage"):
            excluded.append({
                "image_id": record["image_id"],
                "path": record["path"],
                "reason": "TRAINING_COPY_OF_CROSS_SPLIT_EXACT_DUPLICATE",
                "duplicate_group_id": group_id,
                "paired_official_test_ids": [item["image_id"] for item in group["records"] if item["split"] == "test"] if group else [],
                "dr_grade": record["dr_grade"],
                "dme_grade": record["dme_grade"],
            })
        elif record.get("conflicting_duplicate_labels"):
            excluded.append({
                "image_id": record["image_id"],
                "path": record["path"],
                "reason": "CONFLICTING_LABELS_IN_EXACT_TRAINING_DUPLICATE_GROUP",
                "duplicate_group_id": group_id,
                "group_records": group["records"] if group else [],
                "dr_grade": record["dr_grade"],
                "dme_grade": record["dme_grade"],
            })
        else:
            eligible.append(record)

    group_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        group_id = record.get("duplicate_group_id") or f"unique_{record['image_id']}"
        group_records[group_id].append(record)
    groups = []
    for group_id, group in sorted(group_records.items()):
        groups.append({
            "group_id": group_id,
            "records": group,
            "size": len(group),
            "labels": sorted({int(record["dr_grade"]) for record in group}),
        })
    validation_group_ids = choose_validation_groups(groups, [int(record["dr_grade"]) for record in eligible], args.validation_fraction, args.seed)

    split_records: list[dict[str, Any]] = []
    for group in groups:
        split = "validation" if group["group_id"] in validation_group_ids else "train"
        for record in group["records"]:
            split_records.append({
                "record_key": f"official_train:{record['image_id']}",
                "image_id": record["image_id"],
                "original_filename": record["original_filename"],
                "image": record["path"].split("ml/datasets/raw/idrid/", 1)[-1],
                "source_path": record["path"],
                "split": split,
                "official_split": "train",
                "label": int(record["dr_grade"]),
                "dr_grade": int(record["dr_grade"]),
                "dme_grade": int(record["dme_grade"]),
                "sha256": record["sha256"],
                "duplicate_group_id": group["group_id"] if record.get("duplicate_group_id") else None,
            })
    split_records.sort(key=lambda record: (record["split"], record["image_id"]))
    by_split = {
        split: [record for record in split_records if record["split"] == split]
        for split in ("train", "validation")
    }
    cross_split_duplicate_groups = []
    for group_id, group in group_records.items():
        splits = {record["split"] for record in split_records if (record.get("duplicate_group_id") or f"unique_{record['image_id']}") == group_id}
        if len(splits) > 1:
            cross_split_duplicate_groups.append(group_id)
    if cross_split_duplicate_groups:
        raise RuntimeError(f"Duplicate groups crossed the generated split: {cross_split_duplicate_groups}")

    payload = {
        "dataset": "idrid",
        "purpose": "Experimental DR severity grading train/validation split",
        "generated_from": {
            "grading_manifest": str(grading_path.relative_to(ROOT)).replace("\\", "/"),
            "grading_manifest_sha256": digest(grading_path),
            "duplicate_report": str(duplicate_path.relative_to(ROOT)).replace("\\", "/"),
            "duplicate_report_sha256": digest(duplicate_path),
        },
        "seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "split_method": "Deterministic class-stratified group assignment; exact duplicate groups remain together.",
        "official_train_records": len(official_train),
        "official_test_records_reserved": len(official_test),
        "eligible_records": len(eligible),
        "excluded_record_count": len(excluded),
        "excluded_records": excluded,
        "reserved_official_test_records": [
            {
                "record_key": f"official_test:{record['image_id']}",
                "image_id": record["image_id"],
                "original_filename": record["original_filename"],
                "path": record["path"],
                "official_split": "test",
                "dr_grade": record["dr_grade"],
                "dme_grade": record["dme_grade"],
            }
            for record in sorted(official_test, key=lambda value: value["path"])
        ],
        "class_mapping": {
            "0": "No DR",
            "1": "Mild",
            "2": "Moderate",
            "3": "Severe",
            "4": "Proliferative DR",
        },
        "class_distribution": {
            split: {str(label): count for label, count in sorted(Counter(record["label"] for record in records_for_split).items())}
            for split, records_for_split in by_split.items()
        },
        "leakage": {
            "status": "pass",
            "cross_split_duplicate_groups": cross_split_duplicate_groups,
            "official_test_in_training_manifest": False,
            "patient_level_guarantee": False,
            "limitations": ["IDRiD does not provide patient identifiers; grouping is exact-image based only."],
        },
        "official_test_policy": "The official 103-image test set is reserved and is not used for training, validation, augmentation, threshold selection, or hyperparameter selection.",
        "training_policy": {
            "cross_split_duplicate_training_copy_excluded": True,
            "conflicting_duplicate_groups_excluded": True,
            "same_label_duplicate_groups_kept_together": True,
            "labels_modified": False,
            "raw_files_modified": False,
        },
        "records": split_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "train": len(by_split["train"]),
        "validation": len(by_split["validation"]),
        "excluded": len(excluded),
        "class_distribution": payload["class_distribution"],
        "cross_split_duplicate_groups": cross_split_duplicate_groups,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
