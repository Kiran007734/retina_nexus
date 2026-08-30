from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset_common import DatasetError, DATA_ROOT, leakage_report, validate_dataset, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate readability, duplicates, labels, metadata, and split integrity")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-dir")
    args = parser.parse_args()
    try:
        result = validate_dataset(args.dataset, args.raw_dir)
        output = write_json(Path(__file__).resolve().parents[1] / "ml" / "datasets" / "metadata" / "reports" / args.dataset / "dataset_validation_report.json", result)
        leakage = write_json(DATA_ROOT / "metadata" / "data_leakage_report.json", result["leakage"])
        print(f"Validation status: {result['status']}")
        print(f"Dataset Readiness Score: {result['readiness']['score']}/100 (engineering metric only)")
        print(f"Validation report: {output}")
        print(f"Leakage report: {leakage}")
        return 0 if result["status"] == "pass" else 1
    except DatasetError as exc:
        print(f"DATASET VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
