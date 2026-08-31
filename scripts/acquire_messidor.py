"""Prepare an authorized Messidor/Messidor-2 copy for external validation.

The official ADCIS source requires a human-completed download form.  This
command intentionally performs no web download and never fabricates labels.
It inventories and validates only files already placed in the raw directory.

Examples:
  python scripts/acquire_messidor.py --variant messidor
  python scripts/acquire_messidor.py --variant messidor2 --verify-only
  python scripts/acquire_messidor.py --variant auto --raw-dir <authorized-copy>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.evaluation.messidor import OFFICIAL_MESSIDOR2_URL, OFFICIAL_SOURCE_URL, build_reports


DEFAULT_RAW_DIR = ROOT / "ml" / "datasets" / "raw" / "messidor"
DEFAULT_OUTPUT_DIR = ROOT / "ml" / "evaluation" / "messidor"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory and validate an authorized Messidor dataset copy")
    parser.add_argument("--variant", choices=["auto", "messidor", "messidor2"], default="auto")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Authorized local dataset directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Phase 4 report directory")
    parser.add_argument("--verify-only", action="store_true", help="Compatibility flag; all operation is local verification")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        reports = build_reports(raw_dir, output_dir, args.variant)
    except Exception as exc:  # Report an actionable setup failure without hiding it.
        print(f"MESSIDOR SETUP ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    readiness = reports["phase4_readiness_report"]
    manifest = reports["dataset_manifest"]
    print(f"Dataset variant: {manifest.get('dataset_variant') or 'undetermined'}")
    print(f"Images discovered: {manifest['actual_files']['image_count']}")
    print(f"Label files discovered: {manifest['actual_files']['label_file_count']}")
    print(f"Readiness: {readiness['status']}")
    print(f"Inventory: {output_dir / 'dataset_manifest.json'}")
    print(f"Validation: {output_dir / 'validation_report.json'}")
    print(f"Compatibility: {output_dir / 'grading_compatibility.json'}")
    print(f"Readiness report: {output_dir / 'phase4_readiness_report.json'}")
    if readiness["status"] != "READY_FOR_NEXT_PHASE":
        print("\nNo external evaluation was run.")
        print("Obtain the authorized files through the official source and rerun:")
        print(f"  Messidor:   {OFFICIAL_SOURCE_URL}")
        print(f"  Messidor-2: {OFFICIAL_MESSIDOR2_URL}")
        print(f"  python scripts/acquire_messidor.py --variant {args.variant} --verify-only")
        return 1
    print("\nREADY FOR NEXT PHASE: ZERO-SHOT EXTERNAL VALIDATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
