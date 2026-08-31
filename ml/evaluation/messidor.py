"""Messidor/Messidor-2 acquisition inventory and external-validation readiness.

This module is deliberately local-only.  The official ADCIS download requires
completion of a form and the official Messidor-2 release does not include DR
ground truth.  Nothing here downloads, relabels, or silently promotes
third-party annotations.  The output is preparation metadata for a later
zero-shot evaluation step.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.dataset_common import scan_images


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx"}
DOCUMENTATION_EXTENSIONS = {".txt", ".md", ".pdf", ".bib", ".doc", ".docx"}

OFFICIAL_SOURCE_URL = "https://www.adcis.net/en/third-party/messidor/"
OFFICIAL_MESSIDOR2_URL = "https://www.adcis.net/en/third-party/messidor2/"

APTOS_GRADES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

MESSIDOR_GRADE_DEFINITIONS = {
    0: "Normal: zero microaneurysms and zero hemorrhages",
    1: "0 < microaneurysms <= 5 and no hemorrhages",
    2: "5 < microaneurysms < 15 or 0 < hemorrhages < 5, with no neovascularization",
    3: "microaneurysms >= 15, or hemorrhages >= 5, or neovascularization present",
}

IMAGE_FIELD_KEYS = {
    "image",
    "imageid",
    "image_id",
    "imagename",
    "image_name",
    "filename",
    "file",
    "name",
    "id",
    "idcode",
    "id_code",
}
GRADE_FIELD_KEYS = {
    "grade",
    "class",
    "diagnosis",
    "retinopathy",
    "retinopathygrade",
    "retinopathy_grade",
    "drgrade",
    "dr_grade",
    "severity",
}
EDEMA_FIELD_KEYS = {
    "riskofmacularedema",
    "risk_of_macular_edema",
    "macularedema",
    "macular_edema",
    "edema",
    "megrade",
    "me_grade",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _pick(row: dict[str, Any], accepted: set[str]) -> tuple[str | None, Any]:
    normalized = {_normalized_key(key): value for key, value in row.items()}
    accepted_normalized = {_normalized_key(key) for key in accepted}
    for key, value in normalized.items():
        if key in accepted_normalized and value is not None and str(value).strip():
            return key, value
    return None, None


def _parse_grade(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    named = {
        "normal": 0,
        "no dr": 0,
        "no diabetic retinopathy": 0,
        "mild": 1,
        "mild dr": 1,
        "moderate": 2,
        "moderate dr": 2,
        "severe": 3,
        "severe dr": 3,
        "proliferative": 4,
        "proliferative dr": 4,
    }
    if text in named:
        return named[text]
    match = re.fullmatch(r"[+]?(\d+)(?:\.0+)?", text)
    return int(match.group(1)) if match else None


def _read_csv(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        encoding = "utf-8-sig"
        raw = path.read_bytes()
        try:
            raw.decode(encoding)
        except UnicodeDecodeError:
            encoding = "latin-1"
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        text = raw.decode(encoding)
        if delimiter == "," and text.count(";") > text.count(","):
            delimiter = ";"
        with path.open("r", encoding=encoding, newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)], None
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _read_excel(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read legacy Excel labels when the optional engine is installed."""
    try:
        import pandas as pd
    except ImportError as exc:
        return [], f"pandas is required to parse {path.name}: {exc}"
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    except Exception as exc:  # pandas wraps engine-specific errors inconsistently.
        return [], (
            f"Could not parse {path.name}: {type(exc).__name__}: {exc}. "
            "Install the Excel engines from backend/requirements-ml.txt and rerun."
        )
    rows: list[dict[str, Any]] = []
    for sheet_name, frame in sheets.items():
        frame = frame.where(frame.notna(), None)
        for index, row in frame.iterrows():
            item = {str(key): value for key, value in row.to_dict().items()}
            item["__sheet__"] = str(sheet_name)
            item["__row__"] = int(index) + 2
            rows.append(item)
    return rows, None


def _read_label_file(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return _read_csv(path)
    return _read_excel(path)


def _image_index(image_paths: list[Path], root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for path in image_paths:
        relative = _relative(path, root)
        keys = {relative.lower(), path.name.lower(), path.stem.lower()}
        for key in keys:
            index[key].append(relative)
    return index


def _resolve_image(reference: Any, index: dict[str, list[str]]) -> tuple[str | None, str | None]:
    if reference is None or not str(reference).strip():
        return None, "missing_image_reference"
    cleaned = str(reference).strip().replace("\\", "/")
    candidates = [cleaned.lower(), Path(cleaned).name.lower(), Path(cleaned).stem.lower()]
    matches: list[str] = []
    for candidate in candidates:
        matches.extend(index.get(candidate, []))
        if matches:
            break
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0], None
    if len(unique) > 1:
        return None, "ambiguous_image_reference"
    return None, "missing_image_file"


def _fingerprint(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(_relative(path, root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def detect_variant(root: Path, requested: str) -> dict[str, Any]:
    if requested != "auto":
        return {"requested": requested, "detected": requested, "method": "explicit_cli_argument", "candidates": [requested]}
    names = [path.name.lower() for path in root.rglob("*") if path.is_file()] if root.exists() else []
    joined = " ".join(names)
    candidates: list[str] = []
    if "messidor-2" in joined or "messidor2" in joined:
        candidates.append("messidor2")
    if any(name.endswith(('.xls', '.xlsx')) for name in names):
        candidates.append("messidor")
    if len(set(candidates)) == 1:
        return {"requested": requested, "detected": candidates[0], "method": "filename_layout_heuristic", "candidates": candidates}
    return {"requested": requested, "detected": None, "method": "undetermined_without_explicit_variant", "candidates": sorted(set(candidates))}


def _provenance(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    for filename in ("messidor_label_provenance.json", "label_provenance.json"):
        path = root / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                errors.append(f"{filename} must contain a JSON object")
            else:
                return payload, errors
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
    return None, errors


def _local_files(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep") if root.exists() else []
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [path for path in files if path.suffix.lower() in LABEL_EXTENSIONS]
    documentation = [path for path in files if path.suffix.lower() in DOCUMENTATION_EXTENSIONS]
    return {
        "all_files": files,
        "images": images,
        "label_files": labels,
        "documentation_files": documentation,
        "file_count": len(files),
        "image_count": len(images),
        "label_file_count": len(labels),
        "documentation_file_count": len(documentation),
        "format_counts": dict(sorted(Counter(path.suffix.lower() for path in files).items())),
        "directory_counts": dict(sorted(Counter(_relative(path.parent, root) for path in files).items())),
    }


def _parse_annotations(root: Path, label_files: list[Path], image_paths: list[Path], variant: str | None) -> dict[str, Any]:
    index = _image_index(image_paths, root)
    file_summaries: list[dict[str, Any]] = []
    extracted: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for path in label_files:
        rows, error = _read_label_file(path)
        summary = {"path": _relative(path, root), "format": path.suffix.lower().lstrip("."), "row_count": len(rows), "recognized_image_rows": 0, "recognized_grade_rows": 0, "parse_error": error}
        if error:
            parse_errors.append(f"{_relative(path, root)}: {error}")
        for row_index, row in enumerate(rows, start=2):
            image_field, image_ref = _pick(row, IMAGE_FIELD_KEYS)
            grade_field, grade_raw = _pick(row, GRADE_FIELD_KEYS)
            edema_field, edema_raw = _pick(row, EDEMA_FIELD_KEYS)
            resolved, resolve_error = _resolve_image(image_ref, index) if image_ref is not None else (None, "missing_image_reference")
            if image_field:
                summary["recognized_image_rows"] += 1
            if grade_field:
                summary["recognized_grade_rows"] += 1
            item = {
                "annotation_file": _relative(path, root),
                "annotation_row": row.get("__row__", row_index),
                "sheet": row.get("__sheet__"),
                "image_field": image_field,
                "image_ref": None if image_ref is None else str(image_ref).strip(),
                "resolved_image": resolved,
                "resolve_error": resolve_error,
                "grade_field": grade_field,
                "grade_raw": None if grade_raw is None else str(grade_raw).strip(),
                "grade": _parse_grade(grade_raw),
                "edema_field": edema_field,
                "edema_raw": None if edema_raw is None else str(edema_raw).strip(),
            }
            if image_field or grade_field or edema_field:
                extracted.append(item)
        file_summaries.append(summary)

    grade_rows = [item for item in extracted if item["grade_field"]]
    valid_grade_rows = [item for item in grade_rows if item["grade"] is not None]
    allowed = set(range(4)) if variant == "messidor" else set(range(5))
    invalid_grades = [
        {"annotation_file": item["annotation_file"], "row": item["annotation_row"], "value": item["grade_raw"], "allowed": sorted(allowed)}
        for item in grade_rows
        if item["grade"] is None or item["grade"] not in allowed
    ]
    missing_image_files = [
        {"annotation_file": item["annotation_file"], "row": item["annotation_row"], "image_ref": item["image_ref"], "reason": item["resolve_error"]}
        for item in extracted
        if item["image_field"] and item["resolved_image"] is None
    ]
    labels_by_image: dict[str, list[int]] = defaultdict(list)
    for item in valid_grade_rows:
        if item["resolved_image"] is not None:
            labels_by_image[item["resolved_image"]].append(item["grade"])
    annotation_conflicts = [
        {"image": image, "grades": sorted(set(grades)), "rows": len(grades), "action": "Do not relabel automatically; review the source annotation."}
        for image, grades in sorted(labels_by_image.items())
        if len(set(grades)) > 1
    ]
    duplicate_annotation_rows = [
        {"image": image, "grades": sorted(set(grades)), "rows": len(grades)}
        for image, grades in sorted(labels_by_image.items())
        if len(grades) > 1 and len(set(grades)) == 1
    ]
    labelled = set(labels_by_image)
    missing_labels = sorted(_relative(path, root) for path in image_paths if _relative(path, root) not in labelled)
    return {
        "files": file_summaries,
        "parse_errors": parse_errors,
        "extracted_rows": extracted,
        "annotation_rows": len(extracted),
        "grade_rows": len(grade_rows),
        "valid_grade_rows": len(valid_grade_rows),
        "labelled_image_count": len(labelled),
        "unlabelled_image_count": len(missing_labels),
        "missing_labels": missing_labels,
        "missing_image_files": missing_image_files,
        "invalid_grades": invalid_grades,
        "annotation_conflicts": annotation_conflicts,
        "duplicate_annotation_rows": duplicate_annotation_rows,
        "class_distribution": dict(sorted(Counter(item["grade"] for item in valid_grade_rows).items())),
        "grade_values_observed": sorted({item["grade"] for item in valid_grade_rows}),
    }


def _compatibility(variant: str | None, annotations: dict[str, Any], provenance: dict[str, Any] | None) -> dict[str, Any]:
    external = {
        "variant": variant,
        "grading_scheme": None,
        "grade_values": annotations.get("grade_values_observed", []),
        "definitions": {},
        "source_status": "unknown",
    }
    if variant == "messidor":
        external.update({
            "grading_scheme": "Official Messidor retinopathy grade 0-3",
            "grade_values": [0, 1, 2, 3],
            "definitions": {str(key): value for key, value in MESSIDOR_GRADE_DEFINITIONS.items()},
            "source_status": "official_documented",
        })
        five_class = {
            "status": "INVALID",
            "mapping": None,
            "reason": "Messidor has four retinopathy grades. Its grade 3 combines severe findings and neovascularization, so it cannot be split into APTOS grades 3 and 4 without source information that is not present in the Messidor grade.",
        }
        binary = {
            "status": "CONDITIONALLY_SUPPORTED",
            "target": "severity_only_referable_proxy",
            "rule": "messidor_retinopathy_grade >= 2",
            "basis": "The official grade definitions identify grade 2 as the next severity category after mild disease; this is a severity-only proxy aligned to the APTOS grade_2_or_worse rule.",
            "caveat": "This is not a complete referable-DR label unless macular-edema risk policy is separately established and reported. Do not merge edema risk into this proxy silently.",
            "metrics": ["sensitivity", "specificity", "precision", "recall", "f1", "roc_auc", "confusion_matrix"],
        } if annotations.get("valid_grade_rows", 0) else {
            "status": "NOT_CALCULABLE",
            "target": "severity_only_referable_proxy",
            "rule": "messidor_retinopathy_grade >= 2",
            "reason": "No valid Messidor grade rows are available.",
            "metrics": [],
        }
        metrics = ["binary severity-only sensitivity", "binary severity-only specificity"] if annotations.get("valid_grade_rows", 0) else []
    elif variant == "messidor2":
        external.update({
            "grading_scheme": "Official Messidor-2 release: no DR ground truth annotations",
            "source_status": "official_release_has_no_labels",
        })
        five_class = {
            "status": "NOT_SUPPORTED",
            "mapping": None,
            "reason": "The official Messidor-2 download page states that the dataset contains no DR ground truth; a third-party label file cannot be treated as official without separately documented provenance and authorization.",
        }
        binary = {
            "status": "NOT_CALCULABLE",
            "target": "referable_dr",
            "mapping": None,
            "reason": "No official labels are supplied. Binary metrics would require an independently authorized, documented label source.",
            "metrics": [],
        }
        metrics = []
        if annotations.get("valid_grade_rows", 0):
            five_class["status"] = "CONDITIONAL_REVIEW_REQUIRED"
            five_class["reason"] = "Labels were found, but they are not part of the official Messidor-2 release. Require a provenance sidecar, license/access confirmation, and documented grading semantics before evaluation."
            binary["status"] = "CONDITIONAL_REVIEW_REQUIRED"
            binary["reason"] = "Labels were found, but official Messidor-2 has no labels. Binary evaluation is blocked pending third-party provenance and grading review."
    else:
        external.update({"grading_scheme": "Undetermined until a dataset variant is selected", "source_status": "not_selected"})
        five_class = {"status": "NOT_DECIDABLE", "mapping": None, "reason": "Select --variant messidor or --variant messidor2 after inspecting the authorized files."}
        binary = {"status": "NOT_DECIDABLE", "mapping": None, "reason": "Select a dataset variant and validate its labels first.", "metrics": []}
        metrics = []

    return {
        "external_dataset": external,
        "aptos_classifier": {
            "grades": {str(key): value for key, value in APTOS_GRADES.items()},
            "referable_mapping": "APTOS grades 2, 3, and 4 (grade_2_or_worse)",
        },
        "five_class_evaluation": five_class,
        "binary_evaluation": binary,
        "legitimate_metrics_if_ready": metrics,
        "provenance_sidecar_present": provenance is not None,
        "clinical_claim": False,
        "note": "Compatibility is an evaluation-design decision, not proof that the source labels are clinically interchangeable with APTOS.",
    }


def build_reports(raw_dir: Path, output_dir: Path, requested_variant: str = "auto") -> dict[str, Any]:
    raw_dir = raw_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exists = raw_dir.is_dir()
    variant_info = detect_variant(raw_dir, requested_variant)
    variant = variant_info["detected"]
    local = _local_files(raw_dir)
    image_paths = local["images"]
    label_files = local["label_files"]
    provenance, provenance_errors = _provenance(raw_dir) if exists else (None, [])
    annotation_report = _parse_annotations(raw_dir, label_files, image_paths, variant) if exists else {
        "files": [], "parse_errors": [], "extracted_rows": [], "annotation_rows": 0, "grade_rows": 0,
        "valid_grade_rows": 0, "labelled_image_count": 0, "unlabelled_image_count": 0, "missing_labels": [],
        "missing_image_files": [], "invalid_grades": [], "annotation_conflicts": [], "duplicate_annotation_rows": [],
        "class_distribution": {}, "grade_values_observed": [],
    }
    if image_paths:
        scan = scan_images(image_paths, raw_dir)
    else:
        scan = {
            "total_files": 0, "readable_files": 0, "corrupted_files": 0, "corrupted": [], "readable": [],
            "exact_duplicate_groups": [], "perceptual_duplicate_groups": [], "duplicate_exact_count": 0,
            "duplicate_perceptual_count": 0, "resolution_statistics": {"count": 0, "width": {"min": None, "max": None, "mean": None, "median": None}, "height": {"min": None, "max": None, "mean": None, "median": None}, "pixels": {"min": None, "max": None, "mean": None, "median": None}},
            "phash_by_path": {},
        }
    compatibility = _compatibility(variant, annotation_report, provenance)
    expected = {
        "messidor": {"image_count": 1200, "image_formats": ["tif"], "label_requirement": "Official per-subset Excel diagnoses expected."},
        "messidor2": {"image_count": 1748, "image_formats": ["png", "jpg"], "label_requirement": "Official release has no DR ground-truth labels; pairing spreadsheet is metadata only."},
    }.get(variant)
    missing_required: list[str] = []
    if not exists:
        missing_required.append("Authorized dataset directory is absent.")
    if not image_paths:
        missing_required.append("No retinal image files were found.")
    if variant == "messidor" and annotation_report["valid_grade_rows"] == 0:
        missing_required.append("No valid Messidor retinopathy grade rows were found.")
    if variant == "messidor2" and annotation_report["valid_grade_rows"] == 0:
        missing_required.append("Official Messidor-2 has no DR ground-truth labels; an authorized documented label source is required for evaluation.")
    if annotation_report["parse_errors"]:
        missing_required.append("One or more label files could not be parsed.")
    if annotation_report["missing_image_files"]:
        missing_required.append("One or more annotation rows reference missing or ambiguous image files.")
    if annotation_report["invalid_grades"]:
        missing_required.append("One or more annotation grades are malformed or outside the selected scheme.")
    if annotation_report["annotation_conflicts"]:
        missing_required.append("Conflicting annotations exist for one or more images.")
    if annotation_report["unlabelled_image_count"]:
        missing_required.append(f"{annotation_report['unlabelled_image_count']} image(s) have no valid retinopathy grade label.")
    if provenance_errors:
        missing_required.append("Label provenance metadata could not be read.")
    if variant_info["detected"] is None:
        missing_required.append("Dataset variant could not be determined; rerun with --variant messidor or --variant messidor2.")
    if scan["corrupted_files"]:
        missing_required.append("One or more image files are corrupt or unreadable.")
    if expected and image_paths and len(image_paths) != expected["image_count"]:
        missing_required.append(f"Observed image count {len(image_paths)} differs from the documented {expected['image_count']} for {variant}.")
    labels_ready = bool(annotation_report["valid_grade_rows"]) and not annotation_report["parse_errors"] and not annotation_report["missing_image_files"] and not annotation_report["invalid_grades"] and not annotation_report["annotation_conflicts"]
    ready_for_zero_shot = bool(image_paths) and scan["corrupted_files"] == 0 and labels_ready and compatibility["binary_evaluation"]["status"] in {"CONDITIONALLY_SUPPORTED", "SUPPORTED"} and not missing_required
    fingerprint = _fingerprint(local["all_files"], raw_dir) if local["all_files"] else None
    dataset_version = f"{variant or 'messidor'}-inventory-{fingerprint[:12]}" if fingerprint else None
    manifest = {
        "report_type": "messidor_phase4_acquisition_inventory",
        "generated_at": utc_now(),
        "dataset_name": "Messidor" if variant == "messidor" else "Messidor-2" if variant == "messidor2" else "Messidor / Messidor-2",
        "dataset_variant": variant,
        "dataset_version": dataset_version,
        "raw_path": str(raw_dir),
        "source": {
            "official_messidor": OFFICIAL_SOURCE_URL,
            "official_messidor2": OFFICIAL_MESSIDOR2_URL,
            "access_status": "manual_form_required",
            "license_status": "research_and_educational_use_only; copying, redistribution, and unauthorized commercial use prohibited per official source",
            "download_method": "manual authorized download; no automated downloader is used",
        },
        "variant_detection": variant_info,
        "expected_from_official_documentation": expected,
        "actual_files": {
            "file_count": local["file_count"],
            "image_count": local["image_count"],
            "label_file_count": local["label_file_count"],
            "documentation_file_count": local["documentation_file_count"],
            "format_counts": local["format_counts"],
            "directory_counts": local["directory_counts"],
            "image_paths_sample": [_relative(path, raw_dir) for path in image_paths[:20]],
            "label_file_paths": [_relative(path, raw_dir) for path in label_files],
            "documentation_file_paths": [_relative(path, raw_dir) for path in local["documentation_files"]],
        },
        "official_documentation": {
            "source_pages": [OFFICIAL_SOURCE_URL, OFFICIAL_MESSIDOR2_URL],
            "grading_documentation": "Messidor official page documents retinopathy grades 0-3 and macular-edema risk 0-2; official Messidor-2 page states that no DR ground truth annotations are included.",
        },
        "preservation": "Raw files are inspected in place; labels are never rewritten or inferred.",
    }
    validation = {
        "report_type": "messidor_phase4_validation",
        "generated_at": utc_now(),
        "dataset": manifest["dataset_name"],
        "dataset_variant": variant,
        "dataset_version": dataset_version,
        "status": "VALID" if not missing_required and ready_for_zero_shot else "BLOCKED",
        "image_count": len(image_paths),
        "readable_image_count": scan["readable_files"],
        "corrupt_or_unreadable_images": scan["corrupted"],
        "missing_required_components": missing_required,
        "image_resolution_statistics": scan["resolution_statistics"],
        "image_sha256_and_inventory": [{key: value for key, value in item.items() if key != "perceptual_hash"} for item in scan["readable"]],
        "duplicate_detection": {
            "exact_duplicate_groups": scan["exact_duplicate_groups"],
            "perceptual_duplicate_groups": scan["perceptual_duplicate_groups"],
            "exact_duplicate_count": scan["duplicate_exact_count"],
            "perceptual_duplicate_count": scan["duplicate_perceptual_count"],
        },
        "annotations": {key: value for key, value in annotation_report.items() if key != "extracted_rows"},
        "annotation_parse_errors": annotation_report["parse_errors"],
        "label_provenance": provenance,
        "label_provenance_errors": provenance_errors,
        "patient_level_leakage": {"status": "NOT_ASSESSABLE", "reason": "The official Messidor documentation states that identifying information was discarded; no patient identifier is expected in the released files."},
        "note": "This is an acquisition/readiness validation, not an external model evaluation and not a clinical validation.",
    }
    compatibility["dataset_version"] = dataset_version
    compatibility["variant"] = variant
    if variant is None:
        compatibility["variant_compatibility_options"] = {
            "messidor": _compatibility("messidor", annotation_report, provenance),
            "messidor2": _compatibility("messidor2", annotation_report, provenance),
        }
    readiness = {
        "report_type": "messidor_phase4_readiness",
        "generated_at": utc_now(),
        "dataset": manifest["dataset_name"],
        "dataset_variant": variant,
        "dataset_version": dataset_version,
        "status": "READY_FOR_NEXT_PHASE" if ready_for_zero_shot else "NOT_READY",
        "ready_for_zero_shot_external_validation": ready_for_zero_shot,
        "acquisition_status": "AVAILABLE" if exists and image_paths else "NOT_ACQUIRED",
        "inventory_report": "ml/evaluation/messidor/dataset_manifest.json",
        "validation_report": "ml/evaluation/messidor/validation_report.json",
        "compatibility_report": "ml/evaluation/messidor/grading_compatibility.json",
        "blocking_reasons": missing_required,
        "compatibility": compatibility,
        "required_human_actions": [
            f"Obtain the authorized dataset through the ADCIS download form: {OFFICIAL_SOURCE_URL if variant != 'messidor2' else OFFICIAL_MESSIDOR2_URL}",
            "Keep the original archive, labels, and source documentation unchanged under ml/datasets/raw/messidor/.",
            "For Messidor-2, provide a separately authorized label source with explicit grading semantics; the official release has no DR ground truth.",
        ] if not ready_for_zero_shot else [],
        "clinical_validation_claim": False,
    }
    reports = {"dataset_manifest": manifest, "validation_report": validation, "grading_compatibility": compatibility, "phase4_readiness_report": readiness}
    for name, payload in reports.items():
        (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return reports


__all__ = ["build_reports", "OFFICIAL_SOURCE_URL", "OFFICIAL_MESSIDOR2_URL"]
