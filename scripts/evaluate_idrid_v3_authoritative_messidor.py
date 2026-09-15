"""Run frozen IDRiD V3 zero-shot evaluation on original Messidor-2 images."""

from __future__ import annotations

import argparse
import hashlib
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

from ml.evaluation.metrics import classification_metrics  # noqa: E402
from scripts.run_idrid_v3_research import build_model, make_transforms  # noqa: E402

MANIFEST = ROOT / "ml" / "evaluation" / "messidor2" / "authoritative_external_manifest.json"
CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "final" / "checkpoint_best.pt"
OUTPUT = ROOT / "ml" / "evaluation" / "messidor2" / "authoritative_idrid_v3"
THRESHOLDS = (0.50, 0.20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def binary(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "recall": sensitivity, "f1": f1, "roc_auc": float(roc_auc_score(actual, probability)), "pr_auc": float(average_precision_score(actual, probability)), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "support": int(actual.sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    manifest_path = Path(args.manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if manifest.get("evaluation_image_source") != "ORIGINAL_ONLY" or len(records) != 1744:
        raise RuntimeError("Authoritative original manifest is missing, not ORIGINAL_ONLY, or not 1,744 records")
    before = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_class = build_model(torch, torch.nn)
    model = model_class()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.set_num_threads(args.torch_threads)
    model.eval()
    _, transform = make_transforms(224)
    class OriginalDataset(Dataset):
        def __len__(self):
            return len(records)
        def __getitem__(self, index):
            record = records[index]
            with Image.open(ROOT / record["image_path"]) as image:
                return transform(image.convert("RGB")), index
    loader = DataLoader(OriginalDataset(), batch_size=args.batch_size, shuffle=False, num_workers=0)
    probabilities: list[list[float] | None] = [None] * len(records)
    with torch.inference_mode():
        for images, indices in loader:
            vectors = torch.softmax(model(images)["severity_logits"], dim=1).cpu().tolist()
            for index, vector in zip(indices.tolist(), vectors):
                probabilities[index] = vector
    after = sha256(checkpoint_path)
    if before != after:
        raise RuntimeError("IDRiD V3 checkpoint changed during evaluation")
    evaluated = [(record, vector) for record, vector in zip(records, probabilities) if vector is not None]
    actual_grade = np.asarray([int(record["label"]) for record, _ in evaluated], dtype=int)
    vectors = [vector for _, vector in evaluated]
    severity = classification_metrics(actual_grade.tolist(), vectors, referable_grades=(2, 3, 4))
    actual_referable = (actual_grade >= 2).astype(int)
    probability_referable = np.asarray([sum(vector[2:]) for vector in vectors], dtype=float)
    threshold_metrics = {f"threshold_{threshold:.2f}": binary(actual_referable, probability_referable, threshold) for threshold in THRESHOLDS}
    prediction_rows = [{"image_id": record["image_id"], "image_path": record["image_path"], "label": record["label"], "probabilities": vector, "predicted_grade": int(np.argmax(vector)), "referable_probability": float(sum(vector[2:])), "referable_at_0_20": bool(sum(vector[2:]) >= 0.20), "referable_at_0_50": bool(sum(vector[2:]) >= 0.50)} for record, vector in evaluated]
    model_snapshot = {"model_version": checkpoint.get("model_version"), "architecture": checkpoint.get("model_config", {}).get("architecture", "EfficientNet-B0 multi-head"), "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(), "checkpoint_sha256_before": before, "checkpoint_sha256_after": after, "checkpoint_unchanged": before == after, "preprocessing": {"input_size": 224, "color_space": "RGB", "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]}}
    dump(output / "model_snapshot.json", model_snapshot)
    dump(output / "severity_metrics.json", severity)
    dump(output / "threshold_comparison.json", threshold_metrics)
    dump(output / "evaluation_summary.json", {"status": "COMPLETED_DESCRIPTIVE_EXTERNAL_EVALUATION", "dataset": "Messidor-2", "image_source": "ORIGINAL_ONLY", "image_count": len(evaluated), "severity_metrics": severity, "referable_thresholds": threshold_metrics, "model_snapshot": "model_snapshot.json", "official_idrid_test_images_opened": 0, "production_promoted": False, "messidor_labels_used_for_selection": False, "clinical_validation_claim": False})
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "evaluation_report.md").write_text(f"# IDRiD V3 on authoritative Messidor-2 originals\n\nDescriptive external evaluation only. No Messidor threshold selection or production promotion occurred.\n\n- Images: `{len(evaluated)}`\n- Checkpoint SHA-256: `{after}`\n- Threshold 0.50: `{json.dumps(threshold_metrics['threshold_0.50'], sort_keys=True)}`\n- Threshold 0.20: `{json.dumps(threshold_metrics['threshold_0.20'], sort_keys=True)}`\n\nOfficial IDRiD test images opened: `0`.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "images": len(evaluated), "severity": severity, "thresholds": threshold_metrics, "checkpoint_sha256": after}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
