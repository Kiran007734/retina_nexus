"""Safely materialize and manifest the authoritative Messidor-2 originals.

The split archive is treated as the source of truth for image bytes.  The
historical 512px tree is read only and is never used as the future original
dataset.  Existing files in the target directory are verified and never
overwritten.

Examples:
    python scripts/migrate_messidor2_authoritative.py --dry-run
    python scripts/migrate_messidor2_authoritative.py --extract-originals
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW_ROOT = ROOT / "ml" / "datasets" / "raw" / "messidor"
DERIVATIVE_ROOT = RAW_ROOT / "images"
ORIGINAL_ROOT = RAW_ROOT / "messidor-2-original" / "IMAGES"
LABEL_CSV = DERIVATIVE_ROOT / "messidor_data.csv"
PAIR_CSV = RAW_ROOT / "messidor-2.csv"
METADATA_ROOT = ROOT / "ml" / "datasets" / "metadata" / "messidor2"
EVALUATION_ROOT = ROOT / "ml" / "evaluation" / "messidor2"
ORIGINAL_MANIFEST = METADATA_ROOT / "authoritative_original_manifest.json"
FUTURE_MANIFEST = EVALUATION_ROOT / "authoritative_external_manifest.json"
REPORT_PATH = METADATA_ROOT / "migration_report.md"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}

REFERENCE_ROWS = [
    {
        "path": "scripts/evaluate_messidor2.py",
        "role": "historical external evaluation",
        "current_reference": "raw/messidor/images/messidor-2 by default; accepts an explicit image directory",
        "future_action": "Keep unchanged for reproducibility. Future runs should use the authoritative manifest or an explicit original image root.",
    },
    {
        "path": "scripts/evaluate_messidor2_final_external.py",
        "role": "historical final external evaluation",
        "current_reference": "raw/messidor/images/messidor-2 and raw/messidor/images/messidor_data.csv",
        "future_action": "Do not rewrite historical behavior. Add a separate authoritative-manifest evaluation path before any new run.",
    },
    {
        "path": "scripts/audit_messidor2_reliability.py",
        "role": "historical reliability audit",
        "current_reference": "raw/messidor as a discovery root",
        "future_action": "Keep historical audit inputs stable; future audits should explicitly consume authoritative_external_manifest.json.",
    },
    {
        "path": "scripts/audit_reliability_usability.py",
        "role": "historical Phase 5.1 usability audit",
        "current_reference": "raw/messidor plus historical prediction CSVs",
        "future_action": "Preserve existing outputs; do not mix authoritative-original measurements into historical comparisons without a new run identifier.",
    },
    {
        "path": "scripts/run_messidor2_parallel.py",
        "role": "historical/full-pipeline execution infrastructure",
        "current_reference": "historical per_image_results.jsonl image paths under the existing raw root",
        "future_action": "Keep cache identity and historical results stable; add a future manifest-driven run rather than changing existing cache provenance.",
    },
    {
        "path": "scripts/evaluate_idrid_v2_messidor.py",
        "role": "research-only IDRiD v2 external evaluation",
        "current_reference": "raw/messidor discovery root",
        "future_action": "Use an explicit authoritative manifest for future research runs; do not regenerate prior artifacts in place.",
    },
    {
        "path": "scripts/evaluate_idrid_v3_messidor.py",
        "role": "research-only IDRiD v3 external evaluation",
        "current_reference": "raw/messidor discovery root",
        "future_action": "Use an explicit authoritative manifest for future research runs; preserve existing zero-shot artifacts.",
    },
    {
        "path": "scripts/audit_messidor2_ingestion.py",
        "role": "dataset ingestion and provenance audit",
        "current_reference": "split archive, historical derivative root, and local label CSV",
        "future_action": "Retain as the archive integrity audit. This migration manifest is the authoritative mapping output.",
    },
    {
        "path": "ml/datasets/metadata/messidor2/messidor2_audit_manifest.json",
        "role": "historical ingestion audit artifact",
        "current_reference": "records existing derivatives and archive comparison",
        "future_action": "Preserve unchanged as historical audit evidence; use authoritative_original_manifest.json for future original-image provenance.",
    },
    {
        "path": "ml/evaluation/messidor2/final_external/*",
        "role": "historical final external artifacts",
        "current_reference": "existing 1,744-image derivative evaluation",
        "future_action": "Never overwrite. New authoritative-original runs need a separate versioned output directory.",
    },
    {
        "path": "README.md and docs/MESSIDOR_EXTERNAL_VALIDATION.md",
        "role": "documentation",
        "current_reference": "historical/Kaggle-derived evaluation instructions",
        "future_action": "Update in a later documentation-only change after the first authoritative-original evaluation is verified.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels() -> dict[str, dict[str, Any]]:
    if not LABEL_CSV.is_file():
        raise FileNotFoundError(f"Messidor diagnosis source is missing: {LABEL_CSV}")
    with LABEL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = Path((row.get("id_code") or "").strip()).name.casefold()
        if not key:
            raise ValueError("A label row has an empty id_code")
        if key in labels:
            raise ValueError(f"Duplicate label id_code: {key}")
        labels[key] = {
            "image_id": Path((row.get("id_code") or "").strip()).name,
            "diagnosis": int(row["diagnosis"]),
            "dme": int(row["adjudicated_dme"]),
            "gradable": int(row["adjudicated_gradable"]),
        }
    return labels


def load_derivative_records() -> dict[str, dict[str, Any]]:
    audit_path = METADATA_ROOT / "messidor2_audit_manifest.json"
    if audit_path.is_file():
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        records: dict[str, dict[str, Any]] = {}
        for row in payload.get("records", []):
            if row.get("asset_status") != "EXISTING_PREPROCESSED":
                continue
            normalized = dict(row)
            derivative_path = ROOT / str(row.get("image_path", ""))
            normalized["size_bytes"] = derivative_path.stat().st_size if derivative_path.is_file() else None
            records[Path(str(row["image_filename"])).name.casefold()] = normalized
        if records:
            return records
    records = {}
    for path in sorted(DERIVATIVE_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            records[path.name.casefold()] = {
                "image_filename": path.name,
                "image_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(data),
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "readable": True,
            }
    return records


def open_archive():
    from scripts.audit_messidor2_ingestion import open_archive

    return open_archive()


def archive_infos(archive: Any) -> list[Any]:
    infos = [info for info in archive.infolist() if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES]
    names = [Path(info.filename).name.casefold() for info in infos]
    if len(names) != len(set(names)):
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        raise RuntimeError(f"Archive has duplicate image basenames: {duplicates[:5]}")
    if any(Path(info.filename).parent.name != "IMAGES" for info in infos):
        raise RuntimeError("Archive image entries are not all direct IMAGES children")
    return sorted(infos, key=lambda info: Path(info.filename).name.casefold())


def materialize_originals(archive: Any, infos: list[Any], extract: bool) -> dict[str, dict[str, Any]]:
    ORIGINAL_ROOT.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for index, info in enumerate(infos, start=1):
        filename = Path(info.filename).name
        destination = ORIGINAL_ROOT / filename
        if extract and not destination.is_file():
            temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
            try:
                with archive.open(info, "r") as source, temporary.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        if not destination.is_file():
            records[filename.casefold()] = {
                "filename": filename,
                "original_path": destination.relative_to(ROOT).as_posix(),
                "availability": "ARCHIVE_MEMBER_NOT_EXTRACTED",
                "archive_member": info.filename,
                "archive_crc32": f"{info.CRC:08x}",
                "archive_compressed_size_bytes": info.compress_size,
                "archive_uncompressed_size_bytes": info.file_size,
                "sha256": None,
                "readable": None,
                "error": None,
                "width": None,
                "height": None,
                "format": None,
                "mode": None,
            }
            continue
        data_hash = sha256_file(destination)
        try:
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                metadata = {"width": image.width, "height": image.height, "format": image.format, "mode": image.mode}
            record = {
                "filename": filename,
                "original_path": destination.relative_to(ROOT).as_posix(),
                "availability": "AVAILABLE_ORIGINAL",
                "archive_member": info.filename,
                "archive_crc32": f"{info.CRC:08x}",
                "archive_compressed_size_bytes": info.compress_size,
                "archive_uncompressed_size_bytes": info.file_size,
                "sha256": data_hash,
                "readable": True,
                "error": None,
                **metadata,
            }
        except Exception as exc:
            record = {
                "filename": filename,
                "original_path": destination.relative_to(ROOT).as_posix(),
                "availability": "CORRUPT_ORIGINAL",
                "archive_member": info.filename,
                "archive_crc32": f"{info.CRC:08x}",
                "archive_compressed_size_bytes": info.compress_size,
                "archive_uncompressed_size_bytes": info.file_size,
                "sha256": data_hash,
                "readable": False,
                "error": f"{type(exc).__name__}: {exc}",
                "width": None,
                "height": None,
                "format": None,
                "mode": None,
            }
        records[filename.casefold()] = record
        if index % 250 == 0:
            print(f"verified_originals={index}/{len(infos)}", flush=True)
    return records


def duplicate_groups(records: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key, record in records.items():
        if record.get("sha256"):
            groups[str(record["sha256"])].append(str(record["filename"]))
    return {digest: sorted(names) for digest, names in groups.items() if len(names) > 1}


def build_manifests(originals: dict[str, dict[str, Any]], labels: dict[str, dict[str, Any]], derivatives: dict[str, dict[str, Any]], archive_parts: list[dict[str, Any]], archive_inventory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    duplicate_by_sha = duplicate_groups(originals)
    duplicate_digest_by_name = {name.casefold(): digest for digest, names in duplicate_by_sha.items() for name in names}
    records: list[dict[str, Any]] = []
    for key in sorted(originals):
        original = originals[key]
        label = labels.get(key)
        derivative = derivatives.get(key)
        has_label = label is not None
        mapping_status = "MAPPED_BY_EXACT_FILENAME" if derivative else "ORIGINAL_ONLY_LABEL_UNAVAILABLE"
        record = {
            "image_id": original["filename"],
            "original_image_id": original["filename"],
            "original_image_path": original["original_path"],
            "original_source": "AUTHORITATIVE_MESSIDOR_2_SPLIT_ARCHIVE",
            "original_archive_member": original["archive_member"],
            "original_sha256": original["sha256"],
            "original_width": original["width"],
            "original_height": original["height"],
            "original_format": original["format"],
            "original_mode": original["mode"],
            "original_readable": original["readable"],
            "original_availability": original["availability"],
            "corresponding_existing_preprocessed_image_path": derivative.get("image_path") if derivative else None,
            "preprocessed_derivative_sha256": derivative.get("sha256") if derivative else None,
            "preprocessed_derivative_width": derivative.get("width") if derivative else None,
            "preprocessed_derivative_height": derivative.get("height") if derivative else None,
            "preprocessed_derivative_format": derivative.get("format") if derivative else None,
            "preprocessed_derivative_readable": derivative.get("readable") if derivative else None,
            "diagnosis": label.get("diagnosis") if label else None,
            "dme": label.get("dme") if label else None,
            "gradable": label.get("gradable") if label else None,
            "label_source": LABEL_CSV.relative_to(ROOT).as_posix() if label else None,
            "label_status": "MATCHED" if label else "LABEL_UNAVAILABLE",
            "label_provenance": "Project-local diagnosis/DME/gradable CSV; values are preserved without calling them official Messidor-2 clinical ground truth." if label else "No row exists in the project-local diagnosis CSV; no label was fabricated.",
            "mapping_status": mapping_status,
            "mapping_method": "Exact case-insensitive filename match between archive member and existing derivative/label id_code." if derivative else "Archive filename has no corresponding derivative or local label row.",
            "duplicate_status": "DUPLICATE_SHA256_WITHIN_ORIGINALS" if key in duplicate_digest_by_name else "UNIQUE_SHA256_WITHIN_ORIGINALS",
            "provenance_notes": "Original bytes are from the verified split archive. The existing 512px file is a non-byte-identical derivative and is retained only for historical reproducibility." if derivative else "Archive-only original; excluded from supervised metric calculations because the local label source has no matching row.",
            "archive_crc32": original["archive_crc32"],
        }
        records.append(record)
    manifest = {
        "schema_version": "messidor2-authoritative-original-manifest-v1",
        "generated_at_utc": utc_now(),
        "dataset": "Messidor-2",
        "authoritative_source": "ml/datasets/raw/messidor/messidor-2-original/IMAGES/",
        "authoritative_source_basis": "Original image bytes extracted from the verified four-part IMAGES.zip archive; archive CRC validation passed.",
        "historical_derivative_source": "ml/datasets/raw/messidor/images/messidor-2/messidor-2/preprocess/",
        "label_source": LABEL_CSV.relative_to(ROOT).as_posix(),
        "pairing_source": PAIR_CSV.relative_to(ROOT).as_posix(),
        "label_source_warning": "messidor_data.csv is a project-local/adjudicated label source. It is not called official Messidor-2 ground truth by this manifest.",
        "archive": {"parts": archive_parts, "inventory": archive_inventory},
        "counts": {
            "original_archive_images": len(records),
            "originals_available_and_readable": sum(row["original_readable"] is True for row in records),
            "originals_corrupt_or_unreadable": sum(row["original_readable"] is False for row in records),
            "originals_not_extracted": sum(row["original_availability"] == "ARCHIVE_MEMBER_NOT_EXTRACTED" for row in records),
            "exact_filename_mappings_to_derivatives": sum(row["mapping_status"] == "MAPPED_BY_EXACT_FILENAME" for row in records),
            "label_matched_originals": sum(row["label_status"] == "MATCHED" for row in records),
            "label_unavailable_originals": sum(row["label_status"] == "LABEL_UNAVAILABLE" for row in records),
        },
        "class_distribution_from_local_label_source": dict(sorted(Counter(str(row["diagnosis"]) for row in records if row["diagnosis"] is not None).items())),
        "dme_distribution_from_local_label_source": dict(sorted(Counter(str(row["dme"]) for row in records if row["dme"] is not None).items())),
        "duplicate_sha256_groups_in_originals": duplicate_by_sha,
        "patient_ids_available": False,
        "records": records,
    }
    evaluation_records = [
        {
            "image_path": row["original_image_path"],
            "image_id": row["image_id"],
            "label": row["diagnosis"],
            "dme": row["dme"],
            "sha256": row["original_sha256"],
            "label_source": row["label_source"],
            "label_provenance": row["label_provenance"],
            "duplicate_status": row["duplicate_status"],
            "availability": "AVAILABLE_ORIGINAL_AND_LABEL",
            "gradable": row["gradable"],
        }
        for row in records
        if row["label_status"] == "MATCHED" and row["original_readable"] is True
    ]
    future_manifest = {
        "schema_version": "messidor2-authoritative-external-manifest-v1",
        "generated_at_utc": utc_now(),
        "dataset": "Messidor-2",
        "evaluation_image_source": "ORIGINAL_ONLY",
        "image_root": "ml/datasets/raw/messidor/messidor-2-original/IMAGES/",
        "label_source": LABEL_CSV.relative_to(ROOT).as_posix(),
        "label_source_warning": "Labels are local/adjudicated reference labels and not independently proven official Messidor-2 ground truth.",
        "supervised_metric_policy": "Only records in this manifest are included; the four original images without local labels are excluded.",
        "patient_ids_available": False,
        "image_count": len(evaluation_records),
        "records": evaluation_records,
    }
    return manifest, future_manifest


def storage_report(archive_parts: list[dict[str, Any]], originals: dict[str, dict[str, Any]], derivatives: dict[str, dict[str, Any]]) -> dict[str, Any]:
    archive_bytes = sum(int(item["size_bytes"]) for item in archive_parts)
    original_bytes = sum(path.stat().st_size for path in ORIGINAL_ROOT.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    derivative_bytes = sum(int(row.get("size_bytes") or 0) for row in derivatives.values())
    return {
        "archive_parts_bytes": archive_bytes,
        "archive_parts_gib": archive_bytes / (1024 ** 3),
        "extracted_original_images_bytes": original_bytes,
        "extracted_original_images_gib": original_bytes / (1024 ** 3),
        "existing_preprocessed_derivatives_bytes": derivative_bytes,
        "existing_preprocessed_derivatives_gib": derivative_bytes / (1024 ** 3),
        "safe_to_reclaim_now_bytes": 0,
        "safe_to_reclaim_now_gib": 0,
        "potential_derivative_reclaim_after_explicit_approval_bytes": derivative_bytes,
        "potential_derivative_reclaim_after_explicit_approval_gib": derivative_bytes / (1024 ** 3),
        "note": "No deletion is safe now. The potential derivative reclaim is informational only and remains blocked by historical reproducibility dependencies.",
    }


def report_markdown(manifest: dict[str, Any], future: dict[str, Any], storage: dict[str, Any], extract_requested: bool) -> str:
    counts = manifest["counts"]
    refs = "\n".join(f"| `{row['path']}` | {row['role']} | {row['current_reference']} | {row['future_action']} |" for row in REFERENCE_ROWS)
    return f'''# Messidor-2 authoritative-original migration report

Generated: `{manifest["generated_at_utc"]}`

## Decision summary

- **Authoritative future image source:** `ml/datasets/raw/messidor/messidor-2-original/IMAGES/`
- **Archive source:** four-part `IMAGES.zip.001` through `IMAGES.zip.004`; CRC validation passed.
- **Original archive image entries:** `{counts["original_archive_images"]}`
- **Originals available and readable:** `{counts["originals_available_and_readable"]}`
- **Exact filename mappings to existing derivatives:** `{counts["exact_filename_mappings_to_derivatives"]}`
- **Matched local labels:** `{counts["label_matched_originals"]}`
- **Unlabeled originals:** `{counts["label_unavailable_originals"]}`
- **Extraction requested in this run:** `{extract_requested}`

The 1,744 existing files are 512x512 preprocessed derivatives under
`ml/datasets/raw/messidor/images/messidor-2/messidor-2/preprocess/`. Their
filenames map exactly to 1,744 archive originals, but their bytes are not
byte-identical. The four archive-only images are retained as originals and are
marked `LABEL_UNAVAILABLE`; no labels were inferred.

## Label provenance

The diagnosis/DME/gradable values come from
`ml/datasets/raw/messidor/images/messidor_data.csv`. The CSV has 1,744 unique
rows and is preserved as a project-local/adjudicated reference source. This
report does not call it official Messidor-2 clinical ground truth. The
`messidor-2.csv` file is pairing metadata only and is not used as diagnosis
ground truth.

## Future supervised evaluation

`ml/evaluation/messidor2/authoritative_external_manifest.json` contains only
the `{future["image_count"]}` readable, label-matched ORIGINAL records. The
four unlabeled original images are intentionally absent from supervised metric
calculations.

## Reference migration matrix

| Repository reference | Role | Current reference | Required future action |
|---|---|---|---|
{refs}

Production backend/frontend paths do not depend on the Messidor dataset. No
production model configuration was changed. Historical evaluation outputs must
continue to resolve against the existing derivative tree and must not be
rewritten in place.

## Storage

- Split archive parts: `{storage["archive_parts_bytes"]:,}` bytes ({storage["archive_parts_gib"]:.3f} GiB)
- Extracted original images: `{storage["extracted_original_images_bytes"]:,}` bytes ({storage["extracted_original_images_gib"]:.3f} GiB)
- Existing preprocessed derivatives: `{storage["existing_preprocessed_derivatives_bytes"]:,}` bytes ({storage["existing_preprocessed_derivatives_gib"]:.3f} GiB)
- Safe to reclaim now: **0 bytes**
- Potential derivative reclaim after a separately approved migration: `{storage["potential_derivative_reclaim_after_explicit_approval_bytes"]:,}` bytes ({storage["potential_derivative_reclaim_after_explicit_approval_gib"]:.3f} GiB)

The archive parts should also be retained as provenance/recovery material. No
directory or file was deleted or moved by this migration.

## Deletion decision

**SAFE TO DELETE NOW: NO.**

The existing derivative dataset cannot be deleted yet because historical
evaluation manifests, cached predictions, research artifacts, and scripts still
depend on it. Deletion can be reconsidered only after a separately approved
migration proves that historical results are archived and reproducible, all
tests are independent of the old paths, and a new authoritative-original
evaluation has completed successfully.

## Recommended next action

Run a new versioned Messidor evaluation that consumes
`authoritative_external_manifest.json` and writes to a new output directory.
Compare it with the historical derivative evaluation without overwriting either
result. Only after that comparison and dependency migration should deletion be
reviewed.
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-originals", action="store_true", help="Extract all original image members into the authoritative target directory; existing matching files are verified and never overwritten.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and generate manifests without extracting missing original files.")
    args = parser.parse_args()
    if args.extract_originals and args.dry_run:
        raise SystemExit("Choose only one of --extract-originals or --dry-run")
    extract = bool(args.extract_originals)
    labels = load_labels()
    derivatives = load_derivative_records()
    virtual, archive, inventory = open_archive()
    try:
        infos = archive_infos(archive)
        archive_parts = [{"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in [RAW_ROOT / f"IMAGES.zip.{index:03d}" for index in range(1, 5)]]
        originals = materialize_originals(archive, infos, extract)
    finally:
        archive.close()
        virtual.close()
    manifest, future_manifest = build_manifests(originals, labels, derivatives, archive_parts, inventory)
    storage = storage_report(archive_parts, originals, derivatives)
    manifest["storage"] = storage
    manifest["extraction_mode"] = "EXTRACT_ALL_ORIGINALS" if extract else "DRY_RUN_NO_NEW_EXTRACTION"
    manifest["historical_data_preserved"] = True
    manifest["production_model_changed"] = False
    manifest["model_weights_changed"] = False
    manifest["historical_results_overwritten"] = False
    future_manifest["storage"] = storage
    future_manifest["source_manifest"] = ORIGINAL_MANIFEST.relative_to(ROOT).as_posix()
    future_manifest["historical_derivative_manifest_preserved"] = True
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    ORIGINAL_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    FUTURE_MANIFEST.write_text(json.dumps(future_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report_markdown(manifest, future_manifest, storage, extract), encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if manifest["counts"]["originals_not_extracted"] == 0 and manifest["counts"]["label_matched_originals"] == 1744 else "INCOMPLETE",
        "archive_images": manifest["counts"]["original_archive_images"],
        "originals_available_and_readable": manifest["counts"]["originals_available_and_readable"],
        "label_matched_originals": manifest["counts"]["label_matched_originals"],
        "label_unavailable_originals": manifest["counts"]["label_unavailable_originals"],
        "authoritative_manifest": str(ORIGINAL_MANIFEST.relative_to(ROOT)),
        "future_evaluation_manifest": str(FUTURE_MANIFEST.relative_to(ROOT)),
        "migration_report": str(REPORT_PATH.relative_to(ROOT)),
        "safe_to_reclaim_now_bytes": storage["safe_to_reclaim_now_bytes"],
    }, indent=2))
    return 0 if manifest["counts"]["originals_not_extracted"] == 0 and manifest["counts"]["label_matched_originals"] == 1744 else 2


if __name__ == "__main__":
    raise SystemExit(main())
