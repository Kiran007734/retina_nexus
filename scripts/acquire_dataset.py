"""Acquire an authorized dataset or verify a manually placed copy.

Examples:
  python scripts/acquire_dataset.py --dataset aptos2019 --mode manual
  python scripts/acquire_dataset.py --dataset aptos2019 --mode kaggle
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dataset_common import DatasetError, get_definition, raw_path_for, sha256_file, write_json


def kaggle_is_configured() -> tuple[bool, str]:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, "environment credentials"
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    config_file = config_dir / "kaggle.json"
    if config_file.exists():
        return True, str(config_file)
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire or verify an authorized RETINA-NEXUS dataset")
    parser.add_argument("--dataset", required=True, help="Registry slug, e.g. aptos2019")
    parser.add_argument("--mode", choices=["manual", "kaggle"], required=True)
    parser.add_argument("--source", help="Override the Kaggle competition slug or dataset slug")
    parser.add_argument("--raw-dir", help="Override the registry raw directory")
    parser.add_argument("--expected-files", type=int, help="Optional exact file count check")
    parser.add_argument("--expected-sha256", help="Optional SHA-256 for a source archive")
    parser.add_argument("--archive", help="Archive to verify before manual extraction")
    parser.add_argument("--verify-only", action="store_true", help="Do not download; only verify the local placement")
    args = parser.parse_args()
    try:
        definition = get_definition(args.dataset)
        raw_dir = raw_path_for(definition, args.raw_dir)
        if args.mode == "kaggle" and not args.verify_only:
            configured, credential_source = kaggle_is_configured()
            if not configured:
                raise DatasetError("Kaggle access is not configured. Set KAGGLE_USERNAME and KAGGLE_KEY, or place an authorized kaggle.json under ~/.kaggle (or KAGGLE_CONFIG_DIR). If access is unavailable, use --mode manual and follow the registry instructions.")
            slug = args.source or definition.get("kaggle_competition")
            if not slug:
                raise DatasetError(f"No Kaggle identifier is registered for {args.dataset}. Obtain it under the dataset's source terms and pass --source <authorized-slug>.")
            if shutil.which("kaggle") is None:
                raise DatasetError("The Kaggle CLI is not installed. Install it in your environment with pip install kaggle after reviewing its terms, then retry.")
            raw_dir.mkdir(parents=True, exist_ok=True)
            command = ["kaggle", "competitions", "download", "-c", slug, "-p", str(raw_dir), "--unzip"]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode != 0:
                raise DatasetError(f"Kaggle acquisition failed for '{slug}'. Confirm your authorized access and CLI configuration.\n{result.stderr.strip()}")
            print(f"Acquisition completed using {credential_source}.")
        if args.archive and args.expected_sha256:
            if not Path(args.archive).exists():
                raise DatasetError(f"Archive does not exist: {args.archive}")
            actual = sha256_file(Path(args.archive))
            if actual.lower() != args.expected_sha256.lower():
                raise DatasetError(f"Archive checksum mismatch. Expected {args.expected_sha256}, got {actual}.")
        if not raw_dir.exists():
            raise DatasetError(f"Dataset directory does not exist: {raw_dir}. Place the authorized files there, then rerun with --mode manual --verify-only.")
        image_count = sum(1 for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
        if image_count == 0:
            raise DatasetError(f"No image files found under {raw_dir}. Nothing was downloaded and no files were verified.")
        if args.expected_files is not None and image_count != args.expected_files:
            raise DatasetError(f"File-count check failed: expected {args.expected_files} images, found {image_count}.")
        verification = {"dataset": args.dataset, "raw_path": str(raw_dir), "status": "available", "image_count": image_count, "verified_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "source": args.mode, "note": "Availability does not imply label correctness, license review, or clinical validity."}
        report = write_json(Path(__file__).resolve().parents[1] / "ml" / "datasets" / "metadata" / f"{args.dataset}_acquisition.json", verification)
        print(f"Verified {image_count} images. Wrote {report}.")
        return 0
    except DatasetError as exc:
        print(f"DATASET SETUP ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
