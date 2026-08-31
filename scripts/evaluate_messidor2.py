"""Run zero-shot external validation of the unchanged APTOS classifier.

The command discovers Messidor-2 labels from the KaggleHub cache, matches them
to the locally authorized images, validates the matched set, and performs
inference only. It never trains, fine-tunes, selects a threshold from
Messidor-2, or writes model weights.

Example:
    python scripts/evaluate_messidor2.py --device cpu --batch-size 32 --torch-threads 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.inference.classifier import ClassifierNotConfiguredError, TorchDRClassificationService  # noqa: E402
from app.ml.models.classifier import severity_probabilities  # noqa: E402
from ml.evaluation.messidor2 import (  # noqa: E402
    APTOS_CLASS_MAPPING,
    LABEL_DATASET_HANDLE,
    OFFICIAL_SOURCE_URL,
    bootstrap_confidence_intervals,
    build_matched_manifest,
    compute_metrics,
    discover_image_root,
    discover_label_source,
    distribution_comparison,
    load_labels,
    model_snapshot,
    relative,
    sha256_bytes,
    utc_now,
    validate_images,
    write_confusion_matrix,
)


DEFAULT_RAW_DIR = ROOT / "ml" / "datasets" / "raw" / "messidor"
DEFAULT_OUTPUT_DIR = ROOT / "ml" / "evaluation" / "messidor"
DEFAULT_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
EXPECTED_IMAGE_COUNT = 1744
EXPECTED_LABEL_COUNT = 1748
EXPECTED_UNMATCHED_LABEL_IDS = {"im002385", "im004176", "im003718", "20060411_58550_0200_pp"}


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoint(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Requested classifier checkpoint does not exist: {path}")
        return path
    configured = os.environ.get("CLASSIFIER_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CLASSIFIER_MODEL_PATH does not exist: {path}")
        return path
    registry_paths = [ROOT / "ml" / "weights" / "model_registry.json", ROOT / "ml" / "model_registry.json"]
    for registry_path in registry_paths:
        if not registry_path.is_file():
            continue
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for artifact in registry.get("artifacts", []):
            if artifact.get("model_type") != "classification":
                continue
            checkpoint = Path(str(artifact.get("checkpoint", ""))).expanduser()
            path = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
            if path.is_file():
                return path.resolve()
    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT.resolve()
    raise FileNotFoundError("No usable APTOS classifier checkpoint was found. Set CLASSIFIER_MODEL_PATH or pass --checkpoint.")


def _read_readme(label_path: Path) -> dict[str, Any]:
    readme = label_path.parent / "messidor_readme.txt"
    if not readme.is_file():
        return {"status": "MISSING", "path": str(readme), "reason": "The downloaded label package has no messidor_readme.txt beside the CSV."}
    text = readme.read_text(encoding="utf-8", errors="replace")
    required_phrases = ["5 point ICDR", "0=None", "1=Mild DR", "2=Moderate DR", "3=Severe DR", "4=PDR", "adjudicated_gradable"]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    return {
        "status": "VERIFIED" if not missing else "INCOMPLETE",
        "path": str(readme),
        "sha256": _sha256_file(readme),
        "missing_phrases": missing,
        "grading_scheme": "five-point ICDR",
        "definitions": {"0": "None", "1": "Mild DR", "2": "Moderate DR", "3": "Severe DR", "4": "PDR"},
        "dme_definition": "Referable DME defined by hard exudates within 1 disc diameter, according to the downloaded readme.",
        "gradability_definition": "0 = ungradable and DR/DME fields are empty; 1 = gradable and both fields are graded.",
        "citation": "Krause et al., Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy, Ophthalmology (2018), doi:10.1016/j.ophtha.2018.01.034",
    }


def _ignored_raw_label_candidates(raw_root: Path, chosen: Path | None) -> list[dict[str, Any]]:
    candidates = sorted(path for path in raw_root.rglob("messidor_data.csv") if path.is_file()) if raw_root.is_dir() else []
    result: list[dict[str, Any]] = []
    for path in candidates:
        if chosen and path.resolve() == chosen.resolve():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle), [])
            result.append({"path": str(path), "columns": columns, "reason": "Not used because the cache-discovered label source takes precedence or the required adjudicated schema is absent."})
        except (OSError, UnicodeError, csv.Error) as exc:
            result.append({"path": str(path), "reason": f"Could not inspect candidate: {type(exc).__name__}: {exc}"})
    return result


def _compatibility(labels: dict[str, Any], documentation: dict[str, Any], dataset_version: str) -> dict[str, Any]:
    valid_grades = sorted({row["adjudicated_dr_grade"] for row in labels["rows"] if row["adjudicated_gradable"] == 1 and row["adjudicated_dr_grade"] in range(5)})
    doc_verified = documentation.get("status") == "VERIFIED"
    supported = doc_verified and not labels["errors"] and valid_grades == [0, 1, 2, 3, 4]
    return {
        "status": "SUPPORTED_FOR_DESCRIPTIVE_ZERO_SHOT_EXTERNAL_EVALUATION" if supported else "BLOCKED",
        "external_dataset": "Messidor-2 adjudicated label package",
        "label_source": {"handle": LABEL_DATASET_HANDLE, "dataset_version": "1", "documentation": documentation.get("path"), "documentation_sha256": documentation.get("sha256")},
        "external_grading_scheme": {
            "name": "five-point ICDR",
            "values": {"0": "None", "1": "Mild DR", "2": "Moderate DR", "3": "Severe DR", "4": "PDR"},
            "basis": "The downloaded label package readme explicitly defines adjudicated_dr_grade as a five-point ICDR grade with these semantics; numeric equivalence was not assumed without this documentation.",
        },
        "aptos_grading_scheme": {str(key): value for key, value in APTOS_CLASS_MAPPING.items()},
        "proposed_mapping": {str(key): key for key in range(5)},
        "mapping_status": "ONE_TO_ONE_DOCUMENTED_SEMANTIC_MAPPING" if supported else "NOT_SUPPORTED",
        "valid_metrics": ["accuracy", "macro_f1", "weighted_f1", "per_class_precision", "per_class_recall", "per_class_f1", "confusion_matrix", "quadratic_weighted_kappa", "multiclass_roc_auc_ovr_macro", "sensitivity", "specificity", "referable_grade_2_or_worse_roc_auc"],
        "invalid_or_unsupported_metrics": [
            {"metric": "clinical diagnostic accuracy", "reason": "This is an external model evaluation on a released dataset, not prospective clinical validation."},
            {"metric": "DME sensitivity/specificity", "reason": "The model outputs DR grades and no DME prediction is being evaluated."},
            {"metric": "metrics including unmatched label records", "reason": "Four label rows have no corresponding image and are explicitly excluded."},
        ],
        "evaluation_population_policy": "Run inference on all matched images; calculate DR grade metrics only on successful inference rows with adjudicated_gradable=1 and a valid 0..4 label.",
        "dataset_version": dataset_version,
        "clinical_validation_claim": False,
    }


def _checkpoint_metadata(path: Path) -> tuple[dict[str, Any], str]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for zero-shot inference. Install backend/requirements-ml.txt.") from exc
    before = _sha256_file(path)
    metadata = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(metadata, dict):
        raise RuntimeError("The classifier checkpoint does not contain a metadata dictionary.")
    config = metadata.get("model_config") or {}
    if config.get("backbone") != "efficientnet_b0" or int(config.get("input_size", 0)) != 224 or int(config.get("num_classes", 0)) != 5:
        raise RuntimeError(f"The selected checkpoint does not match the required APTOS EfficientNet-B0 224px five-class contract: {config}")
    return metadata, before


def _prediction_record(record: dict[str, Any], probabilities: list[float], model_info: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    predicted = int(np.argmax(np.asarray(probabilities)))
    return {
        **record,
        "predicted_aptos_grade": predicted,
        "predicted_aptos_grade_label": APTOS_CLASS_MAPPING[predicted],
        **{f"probability_{index}": float(probabilities[index]) for index in range(5)},
        "raw_confidence": float(max(probabilities)),
        "referable_probability_grade_2_or_worse": float(sum(probabilities[2:])),
        "inference_status": "SUCCESS",
        "inference_error": None,
        "inference_time_ms": round(float(elapsed_ms), 3),
        "model_version": model_info["model_version"],
        "gradability_status": "GRADABLE" if record.get("adjudicated_gradable") == 1 else "UNGRADABLE",
    }


def run_inference(records: list[dict[str, Any]], raw_root: Path, checkpoint: Path, device: str, batch_size: int, torch_threads: int) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    import torch

    metadata, checksum_before = _checkpoint_metadata(checkpoint)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    service = TorchDRClassificationService(model_path=str(checkpoint), backbone="efficientnet_b0", device=device)
    try:
        service._load()
    except ClassifierNotConfiguredError:
        raise
    model = service._model
    transform = service._transform
    assert model is not None and transform is not None
    model_info = model_snapshot(checkpoint, checksum_before, metadata)
    results: list[dict[str, Any] | None] = [None] * len(records)
    for start in range(0, len(records), max(1, batch_size)):
        batch_records = records[start:start + max(1, batch_size)]
        tensors = []
        valid: list[tuple[int, dict[str, Any]]] = []
        for offset, record in enumerate(batch_records):
            try:
                image_path = raw_root / record["image_path"]
                with Image.open(image_path) as image:
                    tensor = transform(image.convert("RGB"))
                tensors.append(tensor)
                valid.append((start + offset, record))
            except Exception as exc:
                results[start + offset] = {
                    **record,
                    "predicted_aptos_grade": None,
                    "predicted_aptos_grade_label": None,
                    **{f"probability_{index}": None for index in range(5)},
                    "raw_confidence": None,
                    "referable_probability_grade_2_or_worse": None,
                    "inference_status": "FAILED",
                    "inference_error": f"{type(exc).__name__}: {exc}",
                    "inference_time_ms": None,
                    "model_version": model_info["model_version"],
                    "gradability_status": "GRADABLE" if record.get("adjudicated_gradable") == 1 else "UNGRADABLE",
                }
        if not tensors:
            continue
        tensor_batch = torch.stack(tensors).to(service._device)
        began = time.perf_counter()
        try:
            with torch.inference_mode():
                outputs = model(tensor_batch)
                probabilities = severity_probabilities(outputs, service._ordinal_mode).detach().cpu().tolist()
            elapsed = (time.perf_counter() - began) * 1000.0 / len(valid)
            for (index, record), probability in zip(valid, probabilities):
                results[index] = _prediction_record(record, probability, model_info, elapsed)
        except Exception as batch_exc:
            # Retry one at a time so a batch-level failure is visible per image.
            for index, record in valid:
                try:
                    started = time.perf_counter()
                    prediction = service.predict((raw_root / record["image_path"]).read_bytes())
                    elapsed = (time.perf_counter() - started) * 1000.0
                    probabilities = [prediction.probabilities[APTOS_CLASS_MAPPING[item]] for item in range(5)]
                    results[index] = _prediction_record(record, probabilities, model_info, elapsed)
                except Exception as exc:
                    results[index] = {
                        **record,
                        "predicted_aptos_grade": None,
                        "predicted_aptos_grade_label": None,
                        **{f"probability_{item}": None for item in range(5)},
                        "raw_confidence": None,
                        "referable_probability_grade_2_or_worse": None,
                        "inference_status": "FAILED",
                        "inference_error": f"{type(exc).__name__}: {exc}; batch_error={type(batch_exc).__name__}: {batch_exc}",
                        "inference_time_ms": None,
                        "model_version": model_info["model_version"],
                        "gradability_status": "GRADABLE" if record.get("adjudicated_gradable") == 1 else "UNGRADABLE",
                    }
    final = [item for item in results if item is not None]
    checksum_after = _sha256_file(checkpoint)
    model_info["checkpoint_sha256_after"] = checksum_after
    model_info["checkpoint_unchanged"] = checksum_before == checksum_after
    return final, model_info, checksum_before, checksum_after


def write_predictions_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "image_path", "image_id", "image_sha256", "adjudicated_dr_grade", "adjudicated_dme", "adjudicated_gradable",
        "predicted_aptos_grade", "predicted_aptos_grade_label", "probability_0", "probability_1", "probability_2", "probability_3", "probability_4",
        "referable_probability_grade_2_or_worse", "raw_confidence", "gradability_status", "inference_status", "inference_error", "inference_time_ms", "model_version",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_failure_analysis(path: Path, records: list[dict[str, Any]], metrics: dict[str, Any], labels: dict[str, Any], validation: dict[str, Any], internal_metrics: dict[str, Any] | None) -> None:
    successes = [item for item in records if item["inference_status"] == "SUCCESS"]
    failures = [item for item in records if item["inference_status"] != "SUCCESS"]
    disagreements = [item for item in successes if item.get("adjudicated_gradable") == 1 and item.get("predicted_aptos_grade") != item.get("adjudicated_dr_grade")]
    low_confidence = [item for item in successes if item.get("raw_confidence", 1.0) < 0.6]
    ungradable = [item for item in records if item.get("adjudicated_gradable") == 0]
    _json(path, {
        "inference_failures": {"count": len(failures), "records": failures},
        "low_confidence": {"threshold": 0.6, "count": len(low_confidence), "records": low_confidence[:100]},
        "label_prediction_disagreements": {"count": len(disagreements), "records": disagreements[:100]},
        "ungradable_matched_images": {"count": len(ungradable), "records": ungradable},
        "unmatched_label_records": validation["matching"]["unmatched_label_records"],
        "label_validation_errors": labels["errors"],
        "duplicate_and_integrity_summary": validation["image_integrity"],
        "distribution_comparison": distribution_comparison(
            [item["adjudicated_dr_grade"] for item in successes if item.get("adjudicated_gradable") == 1],
            [item["predicted_aptos_grade"] for item in successes if item.get("adjudicated_gradable") == 1],
            internal_metrics,
        ),
        "interpretation": "These are dataset/model error-analysis records only. They do not diagnose individual patients or establish causality.",
    })


def write_markdown(path: Path, manifest: dict[str, Any], validation: dict[str, Any], compatibility: dict[str, Any], model_info: dict[str, Any], metrics: dict[str, Any], bootstrap: dict[str, Any], failures: dict[str, Any]) -> None:
    mean_lines = []
    for key in ["accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr_macro", "quadratic_weighted_kappa"]:
        value = metrics.get(key)
        ci = bootstrap.get("intervals", {}).get(key)
        if value is None:
            mean_lines.append(f"- `{key}`: not calculable")
        elif ci:
            mean_lines.append(f"- `{key}`: {value:.6f} (95% CI {ci['lower']:.6f}–{ci['upper']:.6f})")
        else:
            mean_lines.append(f"- `{key}`: {value:.6f}")
    referable = metrics.get("referable_dr_grade_2_or_worse", {})
    referable_lines = [f"- `{key}`: {referable[key]:.6f}" for key in ["sensitivity", "specificity", "precision", "recall", "f1", "roc_auc"] if referable.get(key) is not None]
    text = f"""# Phase 4B — Messidor-2 zero-shot external validation

Status: **COMPLETED AS EXTERNAL MODEL EVALUATION**. This is not a clinical diagnostic validation claim.

## Dataset and matching

- Dataset version: `{manifest['dataset_version']}`
- Image files discovered: **{manifest['image_count']}**
- Label records: **{manifest['label_record_count']}**
- Matched image-label pairs: **{manifest['matched_pair_count']}**
- Unmatched label records excluded: **{len(manifest['unmatched_label_records'])}**
- Images without labels: **{len(manifest['images_without_labels'])}**
- Image-label matching: case-insensitive, extension-independent filename stems
- Label source: `{LABEL_DATASET_HANDLE}`, KaggleHub cache version 1
- Official ADCIS source: {OFFICIAL_SOURCE_URL}

The four unmatched label IDs are recorded verbatim in `dataset_manifest.json` and were not replaced: `im002385`, `im004176`, `im003718`, and `20060411_58550_0200_pp`.

## Model contract

- Model: `{model_info['model_name']}`
- Version: `{model_info['model_version']}`
- Architecture: `{model_info['architecture']}`
- Input: `{model_info['input_resolution']} × {model_info['input_resolution']}` RGB
- Normalization: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- Checkpoint SHA-256 before: `{model_info['checkpoint_sha256_before']}`
- Checkpoint SHA-256 after: `{model_info['checkpoint_sha256_after']}`
- Checkpoint unchanged: **{model_info['checkpoint_unchanged']}**

## Compatibility

The downloaded readme documents `adjudicated_dr_grade` as a five-point ICDR scale: 0 None, 1 Mild DR, 2 Moderate DR, 3 Severe DR, and 4 PDR. This supports a documented one-to-one mapping to the existing APTOS output labels for descriptive external evaluation. It does not establish clinical interchangeability or clinical validation.

## Inference accounting

- Matched images inferred: **{len(records_for_count := manifest['matched_records'])}**
- Gradable matched images: **{validation['counts']['gradable_matched_images']}**
- Ungradable matched images: **{validation['counts']['ungradable_matched_images']}**
- Successful inferences: **{failures['successful_inferences']}**
- Failed inferences: **{failures['failed_inferences']}**

## Valid five-class metrics

Evaluation population: successful inference + `adjudicated_gradable=1` + valid grade 0–4. Unmatched labels, ungradable rows, and failed inferences are excluded.

{chr(10).join(mean_lines)}

Referable DR uses the existing grade-2-or-worse rule and is reported as a grade-based external metric:

{chr(10).join(referable_lines)}

## Unsupported claims/metrics

- No prospective clinical accuracy or clinical validation claim is made.
- DME metrics are not calculated because this classifier does not predict DME.
- No metrics include the four unmatched labels.
- No threshold or model setting was optimized on Messidor-2.

## Failure analysis

- Low-confidence threshold: 0.60; count: **{failures['low_confidence_count']}**
- Label/prediction disagreements: **{failures['disagreement_count']}**
- Duplicate groups: exact **{validation['image_integrity']['exact_duplicate_count']}**, perceptual **{validation['image_integrity']['perceptual_duplicate_count']}**

Bootstrap intervals use a fixed-seed nonparametric percentile bootstrap with `{bootstrap.get('iterations')}` iterations and seed `{bootstrap.get('seed')}`.

No training, fine-tuning, retraining, model selection, or weight modification occurred.
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the unchanged APTOS EfficientNet-B0 on Messidor-2")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--image-dir", help="Optional exact image directory; defaults to raw/images/messidor-2")
    parser.add_argument("--labels", help="Optional explicit schema-compatible label CSV; cache discovery is preferred by default")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    raw_root = Path(args.raw_dir).expanduser().resolve()
    try:
        if args.batch_size < 1 or args.bootstrap_iterations < 100:
            raise ValueError("--batch-size must be positive and --bootstrap-iterations must be at least 100")
        image_root, image_paths, image_discovery_method = discover_image_root(raw_root)
        label_source, discovery_errors = discover_label_source(raw_root, args.labels)
        if label_source is None:
            raise RuntimeError("Messidor-2 labels could not be discovered. " + " ".join(discovery_errors))
        labels = load_labels(label_source)
        image_validation = validate_images(image_root, image_paths)
        matched = build_matched_manifest(raw_root, image_root, image_paths, labels, label_source, image_validation)
        image_sha_digest = hashlib.sha256("".join(sorted(item["image_sha256"] or "" for item in matched["records"])).encode("ascii")).hexdigest()
        dataset_version = f"messidor2-external-{hashlib.sha256((image_sha_digest + labels['sha256']).encode('ascii')).hexdigest()[:12]}"
        documentation = _read_readme(label_source.path)
        compatibility = _compatibility(labels, documentation, dataset_version)
        ignored_raw = _ignored_raw_label_candidates(raw_root, label_source.path if label_source.discovery_method == "raw_directory_fallback" else None)
        matched_expected = len(matched["records"]) == EXPECTED_IMAGE_COUNT
        labels_expected = labels["row_count"] == EXPECTED_LABEL_COUNT
        unmatched_ids = {normalize for normalize in (Path(item["image_id"]).stem.casefold() for item in matched["unmatched_label_records"])}
        expected_unmatched = {item.casefold() for item in EXPECTED_UNMATCHED_LABEL_IDS}
        matching_status = "VALID" if matched_expected and labels_expected and len(matched["images_without_labels"]) == 0 and unmatched_ids == expected_unmatched else "BLOCKED"
        manifest = {
            "report_type": "messidor2_zero_shot_dataset_manifest",
            "generated_at": utc_now(),
            "dataset": "Messidor-2",
            "dataset_version": dataset_version,
            "source": {"official_adc_is": OFFICIAL_SOURCE_URL, "label_package_handle": LABEL_DATASET_HANDLE, "label_package_version": "1", "label_discovery_method": label_source.discovery_method, "label_path": str(label_source.path), "label_sha256": labels["sha256"], "label_readme": documentation},
            "image_root": str(image_root),
            "image_discovery_method": image_discovery_method,
            "image_count": len(image_paths),
            "label_record_count": labels["row_count"],
            "matched_pair_count": matched["matched_pair_count"],
            "images_without_labels": matched["images_without_labels"],
            "unmatched_label_records": matched["unmatched_label_records"],
            "unmatched_label_ids_expected": sorted(EXPECTED_UNMATCHED_LABEL_IDS),
            "matched_records": matched["records"],
            "ignored_raw_label_candidates": ignored_raw,
            "discovery_errors": discovery_errors,
            "matching_status": matching_status,
            "preservation": "The raw image tree and all source label files remain unchanged. The four unmatched label rows are excluded, not substituted.",
        }
        validation = {
            "report_type": "messidor2_zero_shot_validation",
            "generated_at": utc_now(),
            "status": matching_status if compatibility["status"] == "SUPPORTED_FOR_DESCRIPTIVE_ZERO_SHOT_EXTERNAL_EVALUATION" else "BLOCKED",
            "dataset_version": dataset_version,
            "counts": {
                "images_discovered": len(image_paths),
                "readable_images": image_validation["readable_images"],
                "corrupt_images": len(image_validation["corrupt_images"]),
                "label_records": labels["row_count"],
                "matched_pairs": matched["matched_pair_count"],
                "images_without_labels": len(matched["images_without_labels"]),
                "label_records_without_images": len(matched["unmatched_label_records"]),
                "gradable_matched_images": sum(1 for item in matched["records"] if item["adjudicated_gradable"] == 1),
                "ungradable_matched_images": sum(1 for item in matched["records"] if item["adjudicated_gradable"] == 0),
            },
            "image_integrity": image_validation,
            "label_validation": {"columns": labels["columns"], "errors": labels["errors"], "class_distribution": dict(sorted(Counter(row["adjudicated_dr_grade"] for row in labels["rows"] if row["adjudicated_dr_grade"] is not None).items())), "gradable_distribution": dict(sorted(Counter(row["adjudicated_gradable"] for row in labels["rows"]).items())), "dme_distribution": dict(sorted(Counter(row["adjudicated_dme"] for row in labels["rows"] if row["adjudicated_dme"] is not None).items()))},
            "matching": {"matched_pairs": matched["matched_pair_count"], "images_without_labels": matched["images_without_labels"], "unmatched_label_records": matched["unmatched_label_records"], "ambiguous_label_records": matched["ambiguous_label_records"], "duplicate_image_ids": matched["duplicate_image_ids"], "method": "case-insensitive extension-independent filename stem"},
            "patient_level_leakage": {"status": "NOT_ASSESSABLE", "reason": "The released Messidor-2 labels and image files do not provide a patient identifier. Duplicate image checks were still performed."},
            "expected_verification": {"images": EXPECTED_IMAGE_COUNT, "label_records": EXPECTED_LABEL_COUNT, "matched_pairs": EXPECTED_IMAGE_COUNT, "unmatched_label_ids": sorted(EXPECTED_UNMATCHED_LABEL_IDS)},
            "clinical_validation_claim": False,
        }
        _json(output / "dataset_manifest.json", manifest)
        _json(output / "validation_report.json", validation)
        _json(output / "grading_compatibility.json", compatibility)
        if validation["status"] != "VALID":
            print(f"MESSIDOR-2 VALIDATION BLOCKED: {validation['status']}", file=sys.stderr)
            print(f"Reports written under {output}", file=sys.stderr)
            return 2

        checkpoint = discover_checkpoint(args.checkpoint)
        records, model_info, checksum_before, checksum_after = run_inference(matched["records"], raw_root, checkpoint, args.device, args.batch_size, args.torch_threads)
        internal_metrics = None
        try:
            checkpoint_payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
            internal_metrics = (checkpoint_payload.get("metrics") or {}).get("class_distribution") and {"class_distribution": (checkpoint_payload.get("metrics") or {}).get("class_distribution")}
        except Exception:
            internal_metrics = None
        population_note = "Matched Messidor-2 images with successful inference, adjudicated_gradable=1, and valid adjudicated_dr_grade 0..4; four unmatched label records excluded."
        metrics, per_class = compute_metrics(records, population_note)
        bootstrap = bootstrap_confidence_intervals(records, args.bootstrap_iterations, args.bootstrap_seed)
        if metrics.get("status") == "CALCULATED":
            write_confusion_matrix(output / "confusion_matrix.png", metrics["confusion_matrix"])
        write_predictions_csv(output / "zero_shot_predictions.csv", records)
        _json(output / "zero_shot_metrics.json", {"dataset_version": dataset_version, "evaluation_type": "zero_shot_external_validation", "metrics": metrics, "bootstrap_confidence_intervals": bootstrap, "clinical_validation_claim": False})
        _json(output / "per_class_metrics.json", per_class)
        success = [item for item in records if item["inference_status"] == "SUCCESS"]
        gradable_success = [item for item in success if item.get("adjudicated_gradable") == 1 and item.get("adjudicated_dr_grade") in range(5)]
        failure_summary = {
            "successful_inferences": len(success),
            "failed_inferences": len(records) - len(success),
            "low_confidence_count": sum(1 for item in success if item.get("raw_confidence", 1.0) < 0.6),
            "disagreement_count": sum(1 for item in gradable_success if item["predicted_aptos_grade"] != item["adjudicated_dr_grade"]),
        }
        write_failure_analysis(output / "failure_analysis.json", records, metrics, labels, validation, internal_metrics)
        failure_payload = json.loads((output / "failure_analysis.json").read_text(encoding="utf-8"))
        model_info["checkpoint_path_relative"] = str(checkpoint.relative_to(ROOT)).replace("\\", "/") if checkpoint.is_relative_to(ROOT) else str(checkpoint)
        model_info["checkpoint_sha256_before"] = checksum_before
        model_info["checkpoint_sha256_after"] = checksum_after
        model_info["checkpoint_unchanged"] = checksum_before == checksum_after
        _json(output / "model_snapshot.json", model_info)
        write_markdown(output / "phase4b_external_validation_report.md", manifest, validation, compatibility, model_info, metrics, bootstrap, {**failure_summary, **failure_payload})
        update_registry(model_info, dataset_version, metrics, bootstrap, output)
        readiness = {"status": "EXTERNAL_VALIDATION_COMPLETED", "dataset": "Messidor-2", "dataset_version": dataset_version, "ready_for_zero_shot_external_validation": True, "zero_shot_external_validation_report": "ml/evaluation/messidor/phase4b_external_validation_report.md", "clinical_validation_claim": False}
        _json(output / "phase4_readiness_report.json", readiness)
        print(f"Messidor-2 zero-shot evaluation completed on {len(gradable_success)} gradable images.")
        print(f"Successful inferences: {len(success)}; failed: {len(records) - len(success)}")
        print(f"Reports written under {output}")
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError, KeyError, ClassifierNotConfiguredError) as exc:
        print(f"MESSIDOR-2 EVALUATION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def update_registry(model_info: dict[str, Any], dataset_version: str, metrics: dict[str, Any], bootstrap: dict[str, Any], output: Path) -> None:
    evaluation = {
        "evaluation_type": "zero_shot_external_validation",
        "dataset": "Messidor-2",
        "dataset_version": dataset_version,
        "label_source": LABEL_DATASET_HANDLE,
        "sample_count": metrics.get("sample_count", 0),
        "metrics": metrics,
        "bootstrap_confidence_intervals": bootstrap,
        "checkpoint_sha256_before": model_info.get("checkpoint_sha256_before"),
        "checkpoint_sha256_after": model_info.get("checkpoint_sha256_after"),
        "checkpoint_unchanged": model_info.get("checkpoint_unchanged"),
        "report": "ml/evaluation/messidor/phase4b_external_validation_report.md",
        "clinical_validation_claim": False,
    }
    for registry_path in (ROOT / "ml" / "model_registry.json", ROOT / "ml" / "weights" / "model_registry.json"):
        if not registry_path.is_file():
            continue
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        changed = False
        for artifact in registry.get("artifacts", []):
            version = artifact.get("model_version")
            if version != model_info.get("model_version"):
                continue
            artifact["evaluation"] = evaluation
            artifact["evaluation_status"] = "EXTERNAL_VALIDATION_COMPLETED"
            artifact["clinical_validation_claim"] = False
            # Keep the registry self-describing even when the checkpoint is
            # intentionally excluded from Git. These fields are copied from
            # the validated checkpoint contract, not inferred from predictions.
            artifact["architecture"] = model_info.get("architecture")
            artifact["input_size"] = model_info.get("input_resolution")
            artifact["preprocessing"] = model_info.get("preprocessing")
            artifact["checkpoint_sha256"] = model_info.get("checkpoint_sha256_after") or model_info.get("checkpoint_sha256_before")
            artifact["evaluation_type"] = "zero_shot_external_validation"
            changed = True
        if changed:
            registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
