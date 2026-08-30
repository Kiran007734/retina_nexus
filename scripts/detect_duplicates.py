from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset_common import DatasetError, get_definition, image_files, raw_path_for, report_directory, scan_images, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect exact and perceptual duplicate images")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-dir")
    args = parser.parse_args()
    try:
        definition = get_definition(args.dataset)
        raw = raw_path_for(definition, args.raw_dir)
        if not raw.exists():
            raise DatasetError(f"Dataset directory does not exist: {raw}.")
        files = image_files(raw, definition)
        if not files:
            raise DatasetError(f"No image files found under {raw}.")
        scan = scan_images(files, raw)
        output = write_json(report_directory(args.dataset) / "duplicate_report.json", {"dataset": args.dataset, "total_files": scan["total_files"], "readable_files": scan["readable_files"], "corrupted_files": scan["corrupted_files"], "duplicate_exact_count": scan["duplicate_exact_count"], "duplicate_perceptual_count": scan["duplicate_perceptual_count"], "exact_duplicate_groups": scan["exact_duplicate_groups"], "perceptual_duplicate_groups": scan["perceptual_duplicate_groups"], "note": "Perceptual duplicates use an average hash with Hamming distance <= 4; review near matches before exclusion."})
        print(f"Exact duplicate images: {scan['duplicate_exact_count']}")
        print(f"Perceptual duplicate images: {scan['duplicate_perceptual_count']}")
        print(f"Duplicate report: {output}")
        return 0
    except DatasetError as exc:
        print(f"DUPLICATE DETECTION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
