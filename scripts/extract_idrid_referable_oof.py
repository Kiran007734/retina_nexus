"""Extract leak-safe five-fold OOF probabilities for the IDRiD stage2 head."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from scripts.run_idrid_v3_research import make_transforms  # noqa: E402
from app.ml.models.classifier import build_classifier  # noqa: E402

MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
OOF_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "master_cv" / "20260912"
OUTPUT = ROOT / "ml" / "evaluation" / "referable_v2"
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def metric(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
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
    return {"threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "roc_auc": float(roc_auc_score(actual, probability)), "pr_auc": float(average_precision_score(actual, probability))}


def ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1])
        if not np.any(mask):
            continue
        result += float(mask.mean()) * abs(float(actual[mask].mean()) - float(probability[mask].mean()))
    return result


def main() -> int:
    import torch
    from PIL import Image

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {str(row["image_id"]): row for row in manifest.get("records", [])}
    if len(records) != 406 or manifest.get("official_test_images_opened") != 0:
        raise RuntimeError("IDRiD development manifest is not the governed 406-record set")
    torch.set_num_threads(4)
    all_rows: list[dict[str, Any]] = []
    _, transform = make_transforms(224)
    for fold in range(1, 6):
        fold_rows = json.loads((OOF_ROOT / f"fold_{fold}" / "validation_predictions.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(OOF_ROOT / f"fold_{fold}" / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            for source_row in fold_rows:
                record = records[str(source_row["image_id"])]
                with Image.open(RAW / record["image"]) as image:
                    tensor = transform(image.convert("RGB")).unsqueeze(0)
                output = model(tensor)
                stage2_probability = float(torch.softmax(output["stage2_logits"], dim=1)[0, 1].item())
                severity = torch.softmax(output["severity_logits"], dim=1)[0].tolist()
                all_rows.append({"image_id": str(source_row["image_id"]), "fold": fold, "actual_grade": int(record["label"]), "actual_referable": int(int(record["label"]) >= 2), "referable_probability": stage2_probability, "severity_probabilities": [float(value) for value in severity]})
        del model
    if len(all_rows) != 406 or len({row["image_id"] for row in all_rows}) != 406:
        raise RuntimeError("OOF extraction did not produce exactly one row per development image")
    all_rows.sort(key=lambda row: row["image_id"])
    actual = np.asarray([row["actual_referable"] for row in all_rows], dtype=int)
    probability = np.asarray([row["referable_probability"] for row in all_rows], dtype=float)
    entries = [metric(actual, probability, threshold) for threshold in THRESHOLDS]
    eligible = [entry for entry in entries if entry["specificity"] >= 0.85]
    selected = max(eligible, key=lambda row: (row["sensitivity"], row["precision"], row["f1"], row["specificity"], -row["threshold"])) if eligible else None
    from sklearn.metrics import brier_score_loss

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "oof_predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows), encoding="utf-8")
    summary = {"status": "COMPLETED", "development_records": 406, "folds": 5, "official_test_images_opened": 0, "messidor_labels_used": False, "thresholds": entries, "selection_criterion": "highest OOF sensitivity subject to specificity >= 0.85; ties precision, F1, specificity, then lower threshold", "selected": selected, "roc_auc": float(entries[0]["roc_auc"]), "pr_auc": float(entries[0]["pr_auc"]), "ece_10_bins": ece(actual, probability), "brier_score": float(brier_score_loss(actual, probability)), "raw_probability_is_not_clinically_calibrated": True}
    (OUTPUT / "oof_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
