"""Evaluate the frozen IDRiD hierarchical referable head zero-shot on Messidor.

The threshold is fixed from the existing leak-safe development OOF analysis
(0.10). No Messidor labels are used for selection, calibration, or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.evaluation.metrics import classification_metrics  # noqa: E402
from scripts.run_idrid_v3_research import build_model, make_transforms  # noqa: E402

MANIFEST = ROOT / "ml" / "evaluation" / "messidor2" / "authoritative_external_manifest.json"
CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912" / "v3_b_domain_robust" / "checkpoint_best.pt"
OUTPUT = ROOT / "ml" / "evaluation" / "referable_v2" / "primary_binary_messidor"
THRESHOLD = 0.10
EXPECTED_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    predicted = probability >= threshold
    tp = int(np.sum((actual == 1) & predicted))
    tn = int(np.sum((actual == 0) & ~predicted))
    fp = int(np.sum((actual == 0) & predicted))
    fn = int(np.sum((actual == 1) & ~predicted))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {"threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "roc_auc": float(roc_auc_score(actual, probability)), "pr_auc": float(average_precision_score(actual, probability)), "support": int(actual.sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    manifest_path = Path(args.manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if payload.get("evaluation_image_source") != "ORIGINAL_ONLY" or len(records) != 1744:
        raise RuntimeError("Authoritative original manifest is missing or incomplete")
    before = sha256(checkpoint_path)
    if before != EXPECTED_SHA:
        raise RuntimeError(f"Checkpoint SHA mismatch before inference: {before}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_class = build_model(torch, torch.nn)
    model = model_class()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    torch.set_num_threads(max(1, args.torch_threads))
    _, transform = make_transforms(224)

    class OriginalDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            with Image.open(ROOT / record["image_path"]) as image:
                return transform(image.convert("RGB")), index

    loader = DataLoader(OriginalDataset(), batch_size=args.batch_size, shuffle=False, num_workers=0)
    probabilities: list[float | None] = [None] * len(records)
    severity_vectors: list[list[float] | None] = [None] * len(records)
    with torch.inference_mode():
        for images, indices in loader:
            output_batch = model(images)
            stage2_probability = torch.softmax(output_batch["stage2_logits"], dim=1)[:, 1].cpu().tolist()
            severity = torch.softmax(output_batch["severity_logits"], dim=1).cpu().tolist()
            for index, binary_probability, vector in zip(indices.tolist(), stage2_probability, severity):
                probabilities[index] = float(binary_probability)
                severity_vectors[index] = [float(value) for value in vector]
    after = sha256(checkpoint_path)
    if before != after:
        raise RuntimeError("IDRiD checkpoint changed during evaluation")

    actual_grade = np.asarray([int(record["label"]) for record in records], dtype=int)
    actual_referable = (actual_grade >= 2).astype(int)
    probability = np.asarray([float(value) for value in probabilities], dtype=float)
    threshold_metrics = {f"threshold_{threshold:.2f}": binary(actual_referable, probability, threshold) for threshold in THRESHOLDS}
    rows = []
    for record, value, severity in zip(records, probability.tolist(), severity_vectors):
        rows.append({"image_id": record["image_id"], "image_path": record["image_path"], "label": int(record["label"]), "actual_referable": int(int(record["label"]) >= 2), "referable_probability": value, "referable_at_frozen_0_10": bool(value >= THRESHOLD), "severity_probabilities": severity, "predicted_grade": int(np.argmax(severity))})
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {"status": "COMPLETED_DESCRIPTIVE_EXTERNAL_EVALUATION", "model": "IDRiD V3 shared EfficientNet-B0 stage2 binary referable head", "architecture": checkpoint.get("model_config", {}).get("architecture", "EfficientNet-B0 multi-head"), "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": after, "checkpoint_unchanged": before == after, "dataset": "Messidor-2 authoritative original images", "image_count": len(rows), "fixed_development_threshold": THRESHOLD, "threshold_metrics": threshold_metrics, "messidor_labels_used_for_selection": False, "clinical_validation_claim": False, "production_promoted": False, "official_idrid_test_images_opened": 0, "preprocessing": {"input_size": 224, "color_space": "RGB", "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]}}
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "image_count": len(rows), "fixed_threshold": THRESHOLD, "metrics": threshold_metrics["threshold_0.10"], "checkpoint_sha256": after}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
