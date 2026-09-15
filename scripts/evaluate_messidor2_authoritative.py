"""Evaluate the frozen APTOS classifier on authoritative original Messidor-2 images.

This is a separate, descriptive external-evaluation path.  It consumes the
authoritative external manifest, preserves both frozen referable thresholds
(0.50 and the IDRiD-development research threshold 0.20), and never updates
production registries, checkpoints, thresholds, or model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from scripts.evaluate_messidor2 import run_inference  # noqa: E402
from ml.evaluation.messidor2 import APTOS_CLASS_MAPPING, compute_metrics  # noqa: E402

DEFAULT_MANIFEST = ROOT / "ml" / "evaluation" / "messidor2" / "authoritative_external_manifest.json"
DEFAULT_OUTPUT = ROOT / "ml" / "evaluation" / "messidor2" / "authoritative_classifier"
DEFAULT_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
THRESHOLDS = (0.50, 0.20)
UNLABELED_ORIGINALS = ["20060411_58550_0200_PP.png", "IM002385.JPG", "IM003718.JPG", "IM004176.JPG"]


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def binary_stats(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        roc_auc = float(roc_auc_score(actual, probability)) if len(np.unique(actual)) > 1 else None
        pr_auc = float(average_precision_score(actual, probability)) if len(np.unique(actual)) > 1 else None
    except (ImportError, ValueError):
        roc_auc = None
        pr_auc = None
    return {
        "threshold": None,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(sensitivity),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sample_count": int(len(actual)),
    }


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_image_source") != "ORIGINAL_ONLY":
        raise RuntimeError("The evaluation manifest is not marked ORIGINAL_ONLY")
    records = list(payload.get("records", []))
    if len(records) != 1744:
        raise RuntimeError(f"Expected 1744 labeled original records, found {len(records)}")
    for record in records:
        image_path = ROOT / str(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if record.get("availability") != "AVAILABLE_ORIGINAL_AND_LABEL":
            raise RuntimeError(f"Manifest contains a non-evaluable record: {record.get('image_id')}")
    return payload, records


def to_inference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_root = ROOT / "ml" / "datasets" / "raw" / "messidor"
    return [
        {
            "image_path": str((ROOT / str(record["image_path"])).relative_to(raw_root)).replace("\\", "/"),
            "image_id": record["image_id"],
            "image_sha256": record["sha256"],
            "adjudicated_dr_grade": int(record["label"]),
            "adjudicated_dme": int(record["dme"]),
            "adjudicated_gradable": int(record["gradable"]),
        }
        for record in records
    ]


def evaluate_thresholds(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in predictions if row.get("inference_status") == "SUCCESS" and row.get("adjudicated_gradable") == 1]
    actual = np.asarray([int(row["adjudicated_dr_grade"]) >= 2 for row in rows], dtype=int)
    probability = np.asarray([float(row["referable_probability_grade_2_or_worse"]) for row in rows], dtype=float)
    result: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        predicted = (probability >= threshold).astype(int)
        stats = binary_stats(actual, predicted, probability)
        stats["threshold"] = threshold
        stats["rule"] = "referable_probability = P(2)+P(3)+P(4); referable = referable_probability >= threshold; severity grade = argmax(P0..P4)."
        result[f"threshold_{threshold:.2f}"] = stats
    return result


def write_report(path: Path, manifest: dict[str, Any], model_info: dict[str, Any], severity: dict[str, Any], thresholds: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    lines = [
        "# Messidor-2 authoritative-original classifier evaluation",
        "",
        "Status: **DESCRIPTIVE_EXTERNAL_EVALUATION**",
        "",
        "This run uses the frozen APTOS EfficientNet-B0 checkpoint against the authoritative extracted original images. It is not clinical validation and no threshold was tuned on Messidor-2.",
        "",
        f"- Original evaluation records: `{len(predictions)}`",
        f"- Successful inferences: `{sum(row.get('inference_status') == 'SUCCESS' for row in predictions)}`",
        f"- Checkpoint: `{model_info.get('checkpoint_path_relative')}`",
        f"- Checkpoint SHA-256: `{model_info.get('checkpoint_sha256_after')}`",
        f"- Checkpoint unchanged: `{model_info.get('checkpoint_unchanged')}`",
        f"- Label source: `{manifest.get('label_source')}`",
        f"- Label provenance warning: {manifest.get('label_source_warning')}",
        "",
        "## Severity metrics",
        "",
        f"```json\n{json.dumps(severity, indent=2, sort_keys=True)}\n```",
        "",
        "## Referable threshold comparison",
        "",
        f"```json\n{json.dumps(thresholds, indent=2, sort_keys=True)}\n```",
        "",
        "Threshold 0.20 is an IDRiD-development research threshold. Threshold 0.50 is the established grade-2-or-worse rule. Neither was selected using Messidor-2 results.",
        "",
        "The four archive originals without local labels are excluded by the authoritative manifest and are not present in this evaluation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.torch_threads < 1:
        raise SystemExit("--batch-size and --torch-threads must be positive")
    manifest_path = Path(args.manifest).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    manifest, manifest_records = load_manifest(manifest_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    inference_records = to_inference_records(manifest_records)
    raw_root = ROOT / "ml" / "datasets" / "raw" / "messidor"
    predictions, model_info, before, after = run_inference(inference_records, raw_root, checkpoint, args.device, args.batch_size, args.torch_threads)
    model_info["checkpoint_sha256_before"] = before
    model_info["checkpoint_sha256_after"] = after
    model_info["checkpoint_unchanged"] = before == after
    model_info["checkpoint_path_relative"] = checkpoint.relative_to(ROOT).as_posix() if checkpoint.is_relative_to(ROOT) else str(checkpoint)
    if not model_info["checkpoint_unchanged"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    severity, per_class = compute_metrics(predictions, "Successful inference on readable, label-matched, gradable authoritative original images")
    thresholds = evaluate_thresholds(predictions)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "dataset_manifest.json", {"source_manifest": manifest_path.relative_to(ROOT).as_posix(), "image_source": "ORIGINAL_ONLY", "record_count": len(manifest_records), "label_source": manifest.get("label_source"), "label_source_warning": manifest.get("label_source_warning"), "unlabeled_originals_excluded": 4, "clinical_validation_claim": False})
    dump(output / "model_snapshot.json", model_info)
    dump(output / "severity_metrics.json", severity)
    dump(output / "per_class_metrics.json", per_class)
    dump(output / "threshold_comparison.json", thresholds)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    dump(output / "evaluation_summary.json", {
        "status": "COMPLETED_DESCRIPTIVE_EXTERNAL_EVALUATION",
        "dataset": "Messidor-2",
        "image_source": "ORIGINAL_ONLY",
        "image_count": len(manifest_records),
        "successful_inferences": sum(row.get("inference_status") == "SUCCESS" for row in predictions),
        "failed_inferences": sum(row.get("inference_status") != "SUCCESS" for row in predictions),
        "severity_metrics": severity,
        "referable_thresholds": thresholds,
        "model_snapshot": "model_snapshot.json",
        "official_clinical_validation_claim": False,
        "external_label_tuning": False,
        "unlabeled_originals_excluded": UNLABELED_ORIGINALS,
    })
    write_report(output / "evaluation_report.md", manifest, model_info, severity, thresholds, predictions)
    print(json.dumps({"status": "PASS", "images": len(manifest_records), "successful": sum(row.get("inference_status") == "SUCCESS" for row in predictions), "thresholds": thresholds, "checkpoint_sha256": after, "output": str(output.relative_to(ROOT))}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
