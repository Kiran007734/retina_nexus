"""Run the single post-freeze zero-shot Messidor-2 evaluation for IDRiD V3.

The threshold is already frozen from IDRiD development validation.  This
script never searches thresholds, reads no official IDRiD test image, and does
not modify any model registry or production configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.evaluation.metrics import classification_metrics  # noqa: E402
from ml.evaluation.messidor2 import build_matched_manifest, discover_image_root, discover_label_source, load_labels, validate_images  # noqa: E402
from scripts.run_idrid_v3_research import build_model, make_transforms  # noqa: E402

RAW_ROOT = ROOT / "ml" / "datasets" / "raw" / "messidor"
CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912" / "v3_b_domain_robust" / "checkpoint_best.pt"
OUTPUT = ROOT / "ml" / "evaluation" / "messidor" / "idrid_v3_domain_robust_zero_shot" / "idrid_v3_messidor2_zero_shot.json"
FROZEN_THRESHOLD = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def binary_metrics(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    from sklearn.metrics import roc_auc_score
    return {"threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "recall": sensitivity, "f1": f1, "roc_auc": float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "support": int(actual.sum())}


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Dataset

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    checkpoint_sha_before = sha256(CHECKPOINT)
    image_root, image_paths, discovery_method = discover_image_root(RAW_ROOT)
    label_source, errors = discover_label_source(RAW_ROOT, None)
    if label_source is None:
        raise RuntimeError("Messidor-2 label source unavailable: " + " ".join(errors))
    labels = load_labels(label_source)
    image_validation = validate_images(image_root, image_paths)
    matched = build_matched_manifest(RAW_ROOT, image_root, image_paths, labels, label_source, image_validation)
    records = matched["records"]
    _, transform = make_transforms(224)

    class MessidorDataset(Dataset):
        def __len__(self):
            return len(records)
        def __getitem__(self, index):
            with Image.open(RAW_ROOT / records[index]["image_path"]) as image:
                return transform(image.convert("RGB")), index

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model_class = build_model(torch, torch.nn)
    model = model_class()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model.eval()
    loader = DataLoader(MessidorDataset(), batch_size=32, shuffle=False, num_workers=0)
    probabilities: list[list[float] | None] = [None] * len(records)
    with torch.inference_mode():
        for tensors, indices in loader:
            output = torch.softmax(model(tensors)["severity_logits"], dim=1).cpu().tolist()
            for index, vector in zip(indices.tolist(), output):
                probabilities[index] = vector
    checkpoint_sha_after = sha256(CHECKPOINT)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("V3 checkpoint changed during external evaluation")
    evaluated = [index for index, vector in enumerate(probabilities) if vector is not None and records[index].get("adjudicated_gradable") == 1 and records[index].get("adjudicated_dr_grade") in range(5)]
    actual = [int(records[index]["adjudicated_dr_grade"]) for index in evaluated]
    vectors = [probabilities[index] for index in evaluated]
    assert all(vector is not None for vector in vectors)
    five_class = classification_metrics(actual, vectors, referable_grades=(2, 3, 4))
    actual_referable = np.asarray([int(value in (2, 3, 4)) for value in actual], dtype=int)
    referable_probability = np.asarray([float(sum(vector[2:5])) for vector in vectors], dtype=float)
    referable = binary_metrics(actual_referable, referable_probability, FROZEN_THRESHOLD)
    prediction_rows = []
    for index, vector in enumerate(probabilities):
        prediction_rows.append({
            "image_id": records[index]["image_id"],
            "image_path": records[index]["image_path"],
            "adjudicated_dr_grade": records[index].get("adjudicated_dr_grade"),
            "adjudicated_gradable": records[index].get("adjudicated_gradable"),
            "probabilities": vector,
            "predicted_grade": int(np.argmax(vector)) if vector is not None else None,
            "referable_probability": float(sum(vector[2:5])) if vector is not None else None,
            "referable_at_frozen_threshold": bool(sum(vector[2:5]) >= FROZEN_THRESHOLD) if vector is not None else None,
            "normalized_entropy": float(-sum(value * math.log(max(value, 1e-12)) for value in vector) / math.log(5.0)) if vector is not None else None,
        })
    v1_path = ROOT / "ml" / "evaluation" / "messidor" / "zero_shot_metrics.json"
    v2_path = ROOT / "ml" / "evaluation" / "messidor" / "idrid_v2_lesion_aware_zero_shot" / "idrid_v2_messidor2_zero_shot.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.is_file() else None
    v2 = json.loads(v2_path.read_text(encoding="utf-8")) if v2_path.is_file() else None
    comparison = {
        "protocol": "Same discovered Messidor-2 image tree and adjudicated label package; successful gradable matched images; each model uses only its threshold frozen from non-Messidor development data.",
        "messidor2_labels_used_for_v3_selection": False,
        "v1": {"accuracy": v1["metrics"].get("accuracy"), "macro_f1": v1["metrics"].get("macro_f1"), "qwk": v1["metrics"].get("quadratic_weighted_kappa"), "referable": v1["metrics"].get("referable_dr_grade_2_or_worse"), "threshold_source": "existing APTOS production rule/report"} if v1 else None,
        "v2": {"accuracy": v2["metrics"].get("five_class", {}).get("accuracy"), "macro_f1": v2["metrics"].get("five_class", {}).get("f1"), "qwk": v2["metrics"].get("five_class", {}).get("quadratic_weighted_kappa"), "referable": v2["metrics"].get("referable_grade_2_or_worse_at_frozen_development_threshold"), "threshold_source": v2.get("threshold_policy")} if v2 else None,
        "v3": {"accuracy": five_class.get("accuracy"), "macro_f1": five_class.get("f1"), "qwk": five_class.get("quadratic_weighted_kappa"), "referable": referable, "threshold_source": "IDRiD development validation frozen threshold 0.40"},
    }
    report = {
        "schema_version": "idrid-v3-messidor2-zero-shot-1",
        "evaluation_type": "single_post_freeze_zero_shot_external_evaluation",
        "clinical_validation_claim": False,
        "dataset": {"name": "Messidor-2", "image_root": str(image_root), "discovery_method": discovery_method, "image_count": len(image_paths), "label_count": labels["row_count"], "matched_pairs": matched["matched_pair_count"], "gradable_evaluated": len(evaluated), "unmatched_label_records": matched["unmatched_label_records"], "image_validation": image_validation},
        "model": {"model_version": checkpoint.get("model_version"), "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256_before": checkpoint_sha_before, "checkpoint_sha256_after": checkpoint_sha_after, "checkpoint_unchanged": checkpoint_sha_before == checkpoint_sha_after, "model_config": checkpoint.get("model_config")},
        "preprocessing": {"input_size": 224, "color_space": "RGB", "resize": [224, 224], "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]},
        "metrics": {"five_class": five_class, "referable_at_frozen_development_threshold": referable},
        "threshold_policy": {"threshold": FROZEN_THRESHOLD, "source": "IDRiD development validation only", "external_label_tuning": False, "threshold_search_run": False},
        "comparison_v1_v2_v3": comparison,
        "official_idrid_test_images_opened": 0,
        "production_promoted": False,
        "limitations": ["Single descriptive external evaluation; not clinical validation.", "Messidor-2 was not used for training, calibration, threshold selection, or model selection.", "The result does not authorize production deployment."],
        "predictions": prediction_rows,
    }
    json_dump(OUTPUT, report)
    json_dump(ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v3_external_comparison.json", {"schema_version": "idrid-v3-external-comparison-1", "report": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"), "comparison": comparison, "messidor2_used_for_selection": False, "official_idrid_test_images_opened": 0, "production_promoted": False})
    print(json.dumps({"status": "COMPLETED", "images": len(image_paths), "gradable_evaluated": len(evaluated), "accuracy": five_class.get("accuracy"), "macro_f1": five_class.get("f1"), "qwk": five_class.get("quadratic_weighted_kappa"), "referable": referable, "checkpoint_sha256": checkpoint_sha_after, "threshold": FROZEN_THRESHOLD}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
