"""Zero-shot external evaluation of the frozen IDRiD v2 research candidate.

This is descriptive external evaluation only.  The Messidor-2 labels are
never used to tune weights or thresholds; the threshold is the value frozen
from IDRiD development validation.  The script writes only a v2 research
report and does not update production registries or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.evaluation.messidor2 import (  # noqa: E402
    build_matched_manifest,
    discover_image_root,
    discover_label_source,
    load_labels,
    validate_images,
)
from ml.evaluation.metrics import classification_metrics  # noqa: E402
from scripts.run_idrid_v2_research import build_model, make_transforms  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v2" / "20260912" / "v2_c_lesion_aware_efficientnet_b0" / "checkpoint_best.pt"
DEFAULT_RAW = ROOT / "ml" / "datasets" / "raw" / "messidor"
DEFAULT_OUTPUT = ROOT / "ml" / "evaluation" / "messidor" / "idrid_v2_lesion_aware_zero_shot"
FROZEN_THRESHOLD = 0.45


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen IDRiD v2 candidate on Messidor-2 without tuning")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def calculate_binary(actual: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(actual, probabilities)) if len(np.unique(actual)) == 2 else None
    except (ImportError, ValueError):
        auc = None
    return {"threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "recall": sensitivity, "f1": f1, "roc_auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "support": int(actual.sum())}


def main() -> int:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader, Dataset

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    raw_root = Path(args.raw_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checksum_before = sha256(checkpoint_path)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    image_root, image_paths, image_method = discover_image_root(raw_root)
    label_source, errors = discover_label_source(raw_root, None)
    if label_source is None:
        raise RuntimeError("Messidor-2 labels unavailable: " + " ".join(errors))
    labels = load_labels(label_source)
    validation = validate_images(image_root, image_paths)
    matched = build_matched_manifest(raw_root, image_root, image_paths, labels, label_source, validation)
    records = matched["records"]
    _, validation_transform = make_transforms(224)

    class ExternalDataset(Dataset):
        def __len__(self):
            return len(records)
        def __getitem__(self, index):
            record = records[index]
            with Image.open(raw_root / record["image_path"]) as image:
                tensor = validation_transform(image.convert("RGB"))
            return tensor, index

    loader = DataLoader(ExternalDataset(), batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})
    if model_config.get("backbone") != "efficientnet_b0" or int(model_config.get("num_classes", 0)) != 5:
        raise RuntimeError(f"Unexpected v2 model configuration: {model_config}")
    model_class = build_model(torch, torch.nn)
    model = model_class()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model.to(device).eval()
    probabilities: list[list[float] | None] = [None] * len(records)
    with torch.inference_mode():
        for tensors, indices in loader:
            outputs = model(tensors.to(device))
            batch = torch.softmax(outputs["severity_logits"], dim=1).detach().cpu().numpy()
            for row_index, probability in zip(indices.tolist(), batch.tolist()):
                probabilities[row_index] = probability
    checksum_after = sha256(checkpoint_path)
    if checksum_before != checksum_after:
        raise RuntimeError("v2 checkpoint changed during external evaluation")
    successful = [index for index, probability in enumerate(probabilities) if probability is not None]
    gradable = [index for index in successful if records[index].get("adjudicated_gradable") == 1 and records[index].get("adjudicated_dr_grade") in range(5)]
    actual = [int(records[index]["adjudicated_dr_grade"]) for index in gradable]
    vectors = [probabilities[index] for index in gradable]
    assert all(vector is not None for vector in vectors)
    metrics = classification_metrics(actual, vectors, referable_grades=(2, 3, 4))
    actual_ref = np.asarray([int(value in (2, 3, 4)) for value in actual], dtype=int)
    ref_prob = np.asarray([float(sum(vector[2:5])) for vector in vectors], dtype=float)
    referable_external = calculate_binary(actual_ref, ref_prob, FROZEN_THRESHOLD)
    predictions = []
    for index, vector in enumerate(probabilities):
        predictions.append({
            "image_id": records[index]["image_id"],
            "image_path": records[index]["image_path"],
            "adjudicated_gradable": records[index].get("adjudicated_gradable"),
            "adjudicated_dr_grade": records[index].get("adjudicated_dr_grade"),
            "probabilities": vector,
            "predicted_grade": int(np.argmax(vector)) if vector is not None else None,
            "referable_probability": float(sum(vector[2:5])) if vector is not None else None,
            "referable_at_frozen_threshold": bool(sum(vector[2:5]) >= FROZEN_THRESHOLD) if vector is not None else None,
        })
    report = {
        "schema_version": "idrid-v2-messidor2-zero-shot-1",
        "evaluation_type": "zero_shot_external_evaluation",
        "clinical_validation_claim": False,
        "dataset": {"name": "Messidor-2", "image_root": str(image_root), "image_discovery_method": image_method, "image_count": len(image_paths), "label_count": labels["row_count"], "matched_pairs": matched["matched_pair_count"], "gradable_evaluated": len(gradable), "unmatched_label_records": matched["unmatched_label_records"], "validation_errors": labels["errors"]},
        "model": {"model_version": checkpoint.get("model_version"), "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/") if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path), "checkpoint_sha256_before": checksum_before, "checkpoint_sha256_after": checksum_after, "checkpoint_unchanged": checksum_before == checksum_after, "model_config": model_config},
        "preprocessing": {"input_size": 224, "color_space": "RGB", "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]},
        "metrics": {"five_class": metrics, "referable_grade_2_or_worse_at_frozen_development_threshold": referable_external},
        "threshold_policy": {"threshold": FROZEN_THRESHOLD, "source": "IDRiD development validation only", "external_tuning": False, "external_label_tuning": False},
        "limitations": ["External evaluation is descriptive and not prospective clinical validation.", "Messidor-2 was not used for training, model selection, or threshold selection.", "This result does not promote the v2 candidate to production."],
        "predictions": predictions,
    }
    json_dump(output / "idrid_v2_messidor2_zero_shot.json", report)
    print(json.dumps({"status": "COMPLETED", "images": len(image_paths), "gradable_evaluated": len(gradable), "accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("f1"), "qwk": metrics.get("quadratic_weighted_kappa"), "referable": referable_external, "checkpoint_sha256": checksum_after, "threshold": FROZEN_THRESHOLD}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
