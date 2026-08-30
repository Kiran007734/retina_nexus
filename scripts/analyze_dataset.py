from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset_common import DatasetError, build_image_index, duplicate_label_conflicts, get_definition, image_files, load_cached_scan, raw_path_for, read_annotation_rows, report_directory, scan_fingerprint, scan_images, _label_summary, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dataset statistics and image resolution summaries")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-dir")
    args = parser.parse_args()
    try:
        definition = get_definition(args.dataset)
        raw = raw_path_for(definition, args.raw_dir)
        if not raw.exists():
            raise DatasetError(f"Dataset directory does not exist: {raw}. Acquire or manually place the authorized files first.")
        files = image_files(raw, definition)
        if not files:
            raise DatasetError(f"No image files found under {raw}.")
        scan = load_cached_scan(args.dataset, raw, files) or scan_images(files, raw)
        rows, annotation_errors = read_annotation_rows(raw, definition)
        labels = _label_summary(rows, build_image_index(files, raw), raw, definition)
        stats = {"dataset": args.dataset, "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "raw_path": str(raw), "annotation_errors": annotation_errors, "total_files": scan["total_files"], "readable_files": scan["readable_files"], "corrupted_files": scan["corrupted_files"], "duplicate_exact_count": scan["duplicate_exact_count"], "duplicate_perceptual_count": scan["duplicate_perceptual_count"], "class_distribution": labels["class_distribution"], "resolution_statistics": scan["resolution_statistics"], "metadata_completeness": labels["metadata_completeness"], "label_completeness": labels["label_completeness"], "duplicate_label_conflicts": duplicate_label_conflicts(scan, rows, build_image_index(files, raw), raw, definition), "scan_fingerprint": scan_fingerprint(files, raw), "note": "Statistics are descriptive engineering outputs; they are not clinical validation."}
        output = write_json(report_directory(args.dataset) / "dataset_statistics.json", stats)
        print(f"Statistics report: {output}")
        return 0
    except DatasetError as exc:
        print(f"DATASET ANALYSIS ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
