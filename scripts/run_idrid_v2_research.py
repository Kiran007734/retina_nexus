"""Run the isolated IDRiD v2 research experiments.

The v2 runner has a narrow scope: it uses the 406-record governed
development manifest only, writes under a new research namespace, and never
opens the reserved official 103-image test package.  It compares the frozen v1
checkpoint, reuses the already measured class-balanced focal control, and runs
one lesion-aware multi-task EfficientNet-B0 experiment.  It is not a
production-training or model-promotion command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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

from app.ml.models.classifier import build_backbone  # noqa: E402
from ml.evaluation.metrics import classification_metrics  # noqa: E402
from ml.training.losses import build_class_weights  # noqa: E402
from scripts.train_classifier import make_transforms, select_device  # noqa: E402

DEV_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
SPLIT_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"
V1_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "checkpoint_best.pt"
PRIOR_B_DIR = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "20260912" / "experiment_b_class_balanced_focal"
OUTPUT_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v2" / "20260912"
METADATA_ROOT = ROOT / "ml" / "datasets" / "metadata" / "idrid"
RAW_ROOT = ROOT / "ml" / "datasets" / "raw" / "idrid"
CLASS_MAPPING = {"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative DR"}
REFERABLE_GRADES = (2, 3, 4)
LESION_NAMES = ("Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates")
LESION_KEYS = ("microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates")
V1_SHA = "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
V1_OFFICIAL_EVALUATION = METADATA_ROOT / "idrid_official_test_evaluation.json"
V2_EXTERNAL_EVALUATION = ROOT / "ml" / "evaluation" / "messidor" / "idrid_v2_lesion_aware_zero_shot" / "idrid_v2_messidor2_zero_shot.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated IDRiD v2 research experiments")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-training", action="store_true", help="Only rebuild comparison artifacts from existing v2 output")
    parser.add_argument("--resume-existing", action="store_true", help="Resume the isolated v2 experiment from its last checkpoint")
    return parser.parse_args()


def load_manifest() -> dict[str, Any]:
    payload = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("official_test_used") or payload.get("official_test_images_opened") != 0:
        raise RuntimeError("The v2 development manifest is not marked official-test untouched")
    records = payload.get("records", [])
    if len(records) != 406 or {record.get("split") for record in records} != {"train", "validation"}:
        raise RuntimeError("Expected exactly 406 governed train/validation development records")
    if any(record.get("label") not in range(5) for record in records):
        raise RuntimeError("Development labels must be in 0..4")
    if payload.get("source_governance", {}).get("integrity", {}).get("status") != "PASS":
        raise RuntimeError("Development manifest integrity gate is not PASS")
    return payload


def load_checkpoint(torch: Any, path: Path, expected_sha: str | None = None) -> tuple[Any, dict[str, Any]]:
    actual = sha256(path)
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f"Checkpoint SHA mismatch for {path}: expected {expected_sha}, got {actual}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint, {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": actual}


class ResearchDataset:
    def __init__(self, records: list[dict[str, Any]], split: str, transform: Any, mask_cache: dict[str, Any]):
        import torch
        self.records = [record for record in records if record["split"] == split]
        self.transform = transform
        self.mask_cache = mask_cache
        self.torch = torch
        if not self.records:
            raise ValueError(f"No records for split {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = RAW_ROOT / record["image"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        cached = self.mask_cache[record["record_key"]]
        return (
            tensor,
            self.torch.tensor(int(record["label"]), dtype=self.torch.long),
            self.torch.tensor(cached["lesion_targets"], dtype=self.torch.float32),
            self.torch.tensor(cached["lesion_available"], dtype=self.torch.float32),
            self.torch.tensor(cached["structure_target"], dtype=self.torch.float32),
            self.torch.tensor(cached["structure_available"], dtype=self.torch.float32),
            record,
        )


def collate(batch: list[tuple[Any, ...]]):
    import torch
    images, labels, lesion_targets, lesion_available, structure_target, structure_available, records = zip(*batch)
    return (
        torch.stack(images), torch.stack(labels), torch.stack(lesion_targets),
        torch.stack(lesion_available), torch.stack(structure_target),
        torch.stack(structure_available), list(records),
    )


def make_mask_cache(manifest: dict[str, Any]) -> dict[str, Any]:
    """Read only available IDRiD masks and derive presence targets.

    Missing soft-exudate annotations (and any other unavailable mask) remain
    unavailable in the mask; no all-zero target is created for them.
    """
    cache: dict[str, Any] = {}
    for record in manifest["records"]:
        targets: list[float] = []
        available: list[float] = []
        for name in LESION_NAMES:
            annotation = record["lesion_annotations"][name]
            is_available = bool(annotation.get("available"))
            available.append(1.0 if is_available else 0.0)
            if not is_available:
                targets.append(0.0)
                continue
            path = RAW_ROOT / annotation["mask_path"]
            with Image.open(path) as mask:
                array = np.asarray(mask)
                if array.ndim == 3:
                    array = array[:, :, 0]
                targets.append(1.0 if bool(np.any(array > 0)) else 0.0)
        landmark = record["landmarks"]["optic_disc_center"]
        structure_available = [1.0, 1.0] if landmark.get("available") else [0.0, 0.0]
        width = float(landmark.get("width") or 4288)
        height = float(landmark.get("height") or 2848)
        structure_target = [float(landmark.get("x") or 0.0) / width, float(landmark.get("y") or 0.0) / height]
        cache[record["record_key"]] = {
            "lesion_targets": targets,
            "lesion_available": available,
            "structure_target": structure_target,
            "structure_available": structure_available,
        }
    return cache


def build_model(torch: Any, nn: Any):
    class LesionAwareDRClassifier(nn.Module):
        def __init__(self, initialize_from: dict[str, Any] | None = None):
            super().__init__()
            self.backbone_name = "efficientnet_b0"
            self.num_classes = 5
            self.feature_extractor, feature_dim = build_backbone("efficientnet_b0", pretrained=False)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(p=0.2)
            self.stage1_head = nn.Linear(feature_dim, 2)
            self.stage2_head = nn.Linear(feature_dim, 2)
            self.severity_head = nn.Linear(feature_dim, 5)
            self.lesion_head = nn.Linear(feature_dim, 4)
            self.structure_head = nn.Linear(feature_dim, 2)
            if initialize_from is not None:
                current = self.state_dict()
                compatible = {key: value for key, value in initialize_from.items() if key in current and tuple(value.shape) == tuple(current[key].shape)}
                result = self.load_state_dict(compatible, strict=False)
                allowed_missing = {"lesion_head.weight", "lesion_head.bias", "structure_head.weight", "structure_head.bias"}
                if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
                    raise RuntimeError(f"Unexpected v1-to-v2 initialization mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")

        def forward(self, inputs):
            features = self.pool(self.feature_extractor(inputs)).flatten(1)
            features = self.dropout(features)
            return {
                "features": features,
                "stage1_logits": self.stage1_head(features),
                "stage2_logits": self.stage2_head(features),
                "severity_logits": self.severity_head(features),
                "lesion_logits": self.lesion_head(features),
                "structure_prediction": torch.sigmoid(self.structure_head(features)),
            }

    return LesionAwareDRClassifier


def quality_proxy(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    gx = np.diff(gray, n=2, axis=1)
    gy = np.diff(gray, n=2, axis=0)
    laplacian_proxy = float(np.var(gx) + np.var(gy))
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))
    # These are transparent engineering proxies for correlation analysis, not
    # the production quality gate and not clinical quality labels.
    score = float(np.clip(0.55 * min(1.0, laplacian_proxy / 0.08) + 0.35 * min(1.0, contrast / 0.25) + 0.10 * (1.0 - abs(brightness - 0.5) / 0.5), 0.0, 1.0))
    return {"focus_proxy": laplacian_proxy, "contrast_proxy": contrast, "brightness_proxy": brightness, "quality_proxy_score": score}


def entropy(probabilities: list[float]) -> tuple[float, float, float]:
    values = np.asarray(probabilities, dtype=float)
    raw = float(-np.sum(values * np.log(np.clip(values, 1e-12, 1.0))))
    normalized = float(raw / math.log(5.0))
    ordered = np.sort(values)
    margin = float(ordered[-1] - ordered[-2])
    return raw, normalized, margin


def run_inference(model: Any, loader: Any, device: Any, torch: Any, lesion_aware: bool) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, labels, lesion_targets, lesion_available, structure_target, structure_available, records in loader:
            outputs = model(images.to(device))
            probabilities = torch.softmax(outputs["severity_logits"], dim=1)
            for index, record in enumerate(records):
                vector = probabilities[index].detach().cpu().tolist()
                raw_entropy, normalized_entropy, margin = entropy(vector)
                predicted = int(np.argmax(vector))
                lesion_probabilities = torch.sigmoid(outputs["lesion_logits"][index]).detach().cpu().tolist() if lesion_aware else None
                row = {
                    "image_id": record["image_id"],
                    "record_key": record["record_key"],
                    "actual": int(labels[index].item()),
                    "logits": outputs["severity_logits"][index].detach().cpu().tolist(),
                    "probabilities": vector,
                    "predicted": predicted,
                    "confidence": float(max(vector)),
                    "referable_probability": float(sum(vector[2:5])),
                    "referable_predicted": int(sum(vector[2:5]) >= 0.5),
                    "entropy_nats": raw_entropy,
                    "normalized_entropy": normalized_entropy,
                    "probability_margin": margin,
                    "quality": quality_proxy(RAW_ROOT / record["image"]),
                    "lesion_evidence": {
                        "probabilities": lesion_probabilities,
                        "target_presence": lesion_targets[index].detach().cpu().tolist(),
                        "annotation_available": lesion_available[index].detach().cpu().tolist(),
                    },
                    "structure_evidence": {
                        "predicted_normalized_xy": outputs["structure_prediction"][index].detach().cpu().tolist() if lesion_aware else None,
                        "target_normalized_xy": structure_target[index].detach().cpu().tolist(),
                        "annotation_available": structure_available[index].detach().cpu().tolist(),
                    },
                }
                row["abstention"] = {
                    "review_recommended": bool(normalized_entropy >= 0.70 or margin < 0.15),
                    "policy": "research_only: normalized entropy >= 0.70 OR top-two probability margin < 0.15",
                    "does_not_change_grade": True,
                }
                rows.append(row)
    return rows


def ece(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    correct = np.asarray([int(row["actual"] == row["predicted"]) for row in rows], dtype=int)
    value = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if selected.any():
            value += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(value)


def threshold_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = np.asarray([int(row["actual"] in REFERABLE_GRADES) for row in rows], dtype=int)
    entries: list[dict[str, Any]] = []
    for threshold in [round(0.30 + i * 0.05, 2) for i in range(7)]:
        predicted = np.asarray([int(row["referable_probability"] >= threshold) for row in rows], dtype=int)
        tp = int(((actual == 1) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
        entries.append({"threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    selected = sorted(entries, key=lambda item: (item["f1"], item["sensitivity"], item["specificity"], -item["threshold"]), reverse=True)[0]
    return {"criterion": "max referable F1; ties sensitivity, specificity, then lower threshold", "range": [0.30, 0.60], "entries": entries, "selected": selected}


def risk_coverage(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    coverage_rows = [row for row in rows if not row["abstention"]["review_recommended"]]
    accepted = len(coverage_rows)
    correct = sum(row["actual"] == row["predicted"] for row in coverage_rows)
    selected_actual = np.asarray([int(row["actual"] in REFERABLE_GRADES) for row in coverage_rows], dtype=int)
    selected_predicted = np.asarray([int(row["referable_probability"] >= threshold) for row in coverage_rows], dtype=int)
    fn = int(((selected_actual == 1) & (selected_predicted == 0)).sum()) if accepted else 0
    return {
        "abstention_policy": "research_only: normalized entropy >= 0.70 OR top-two probability margin < 0.15",
        "total": len(rows),
        "accepted_without_review": accepted,
        "coverage": accepted / len(rows) if rows else 0.0,
        "accepted_grade_accuracy": correct / accepted if accepted else None,
        "accepted_referable_false_negatives": fn,
        "threshold": threshold,
        "no_grade_overwrite": True,
    }


def quality_error_correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    error = np.asarray([int(row["actual"] != row["predicted"]) for row in rows], dtype=float)
    quality = np.asarray([row["quality"]["quality_proxy_score"] for row in rows], dtype=float)
    entropy_values = np.asarray([row["normalized_entropy"] for row in rows], dtype=float)
    def corr(a: np.ndarray, b: np.ndarray) -> float | None:
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else None
    correct_rows = [row for row in rows if row["actual"] == row["predicted"]]
    incorrect_rows = [row for row in rows if row["actual"] != row["predicted"]]
    return {
        "method": "engineering proxy analysis; not clinical validation",
        "quality_error_point_biserial_proxy": corr(quality, error),
        "uncertainty_error_point_biserial_proxy": corr(entropy_values, error),
        "mean_quality_correct": float(np.mean([row["quality"]["quality_proxy_score"] for row in correct_rows])) if correct_rows else None,
        "mean_quality_incorrect": float(np.mean([row["quality"]["quality_proxy_score"] for row in incorrect_rows])) if incorrect_rows else None,
        "mean_normalized_entropy_correct": float(np.mean([row["normalized_entropy"] for row in correct_rows])) if correct_rows else None,
        "mean_normalized_entropy_incorrect": float(np.mean([row["normalized_entropy"] for row in incorrect_rows])) if incorrect_rows else None,
    }


def error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_id": row["image_id"],
            "record_key": row["record_key"],
            "actual_grade": row["actual"],
            "predicted_grade": row["predicted"],
            "probabilities": row["probabilities"],
            "referable_probability": row["referable_probability"],
            "referable_predicted_at_0_5": row["referable_predicted"],
            "confidence": row["confidence"],
            "uncertainty": {"entropy_nats": row["entropy_nats"], "normalized_entropy": row["normalized_entropy"], "probability_margin": row["probability_margin"]},
            "quality_proxy": row["quality"],
        }
    def selected(predicate):
        return [public(row) for row in rows if predicate(row)]
    return {
        "grade_0_to_2_3_4": selected(lambda row: row["actual"] == 0 and row["predicted"] in {2, 3, 4}),
        "grade_1_to_2_3_4": selected(lambda row: row["actual"] == 1 and row["predicted"] in {2, 3, 4}),
        "grade_2_3_4_to_0_1": selected(lambda row: row["actual"] in {2, 3, 4} and row["predicted"] in {0, 1}),
        "grade_3_to_4_or_4_to_3": selected(lambda row: {row["actual"], row["predicted"]} == {3, 4}),
        "referable_false_negatives": selected(lambda row: row["actual"] in REFERABLE_GRADES and row["referable_predicted"] == 0),
        "high_confidence_incorrect": sorted(selected(lambda row: row["actual"] != row["predicted"]), key=lambda item: (-item["confidence"], item["image_id"])),
    }


def metrics_for(rows: list[dict[str, Any]], threshold: float | None = None) -> dict[str, Any]:
    metric = classification_metrics([row["actual"] for row in rows], [row["probabilities"] for row in rows], referable_grades=REFERABLE_GRADES)
    chosen = threshold if threshold is not None else 0.5
    return {
        **metric,
        "raw_softmax_ece_10_bins": ece(rows),
        "referable_threshold_used_for_research_diagnostics": chosen,
        "referable_rule": "P(2)+P(3)+P(4) >= threshold; severity remains argmax(P0..P4)",
        "threshold_sweep": threshold_sweep(rows),
        "risk_coverage": risk_coverage(rows, chosen),
        "quality_uncertainty_error_analysis": quality_error_correlation(rows),
        "error_analysis": error_analysis(rows),
        "high_confidence_errors": sorted([row for row in rows if row["actual"] != row["predicted"]], key=lambda row: (-row["confidence"], row["image_id"]))[:10],
    }


def load_prior_control() -> dict[str, Any]:
    metrics = json.loads((PRIOR_B_DIR / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((PRIOR_B_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    predictions = json.loads((PRIOR_B_DIR / "validation_predictions.json").read_text(encoding="utf-8"))
    return {
        "experiment_id": "v2_b_prior_class_balanced_focal_control",
        "status": "REUSED_PRIOR_CONTROLLED_EXPERIMENT",
        "model_version": manifest.get("model_version", "experiment_b_class_balanced_focal"),
        "checkpoint_path": str((PRIOR_B_DIR / "checkpoint_best.pt").relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(PRIOR_B_DIR / "checkpoint_best.pt"),
        "metrics": metrics.get("validation_metrics", metrics.get("metrics", metrics)),
        "audit": metrics.get("audit", {}),
        "predictions": predictions,
        "training_config": manifest.get("training_config", {}),
        "limitations": ["This control was trained in the prior fixed-split research cycle and reused without retraining or official-test access."],
    }


def load_frozen_v1_official_reference() -> dict[str, Any]:
    """Read the already-frozen v1 test artifact as metadata only.

    This function deliberately does not open any official test image.  The
    v1 test artifact is immutable historical context for the v2 comparison.
    """
    payload = json.loads(V1_OFFICIAL_EVALUATION.read_text(encoding="utf-8"))
    evaluation = payload.get("evaluation", {})
    return {
        "model_version": payload.get("model", {}).get("model_version"),
        "checkpoint_sha256": payload.get("model", {}).get("checkpoint_sha256"),
        "sample_count": payload.get("dataset", {}).get("sample_count"),
        "five_class_metrics": evaluation.get("five_class_metrics"),
        "referable_metrics": evaluation.get("referable_metrics"),
        "integrity": payload.get("integrity"),
        "official_test_images_opened_by_v2_run": 0,
        "note": "Historical frozen v1 official-test reference; not recomputed or used for v2 selection.",
    }


def load_v2_external_reference() -> dict[str, Any]:
    if not V2_EXTERNAL_EVALUATION.is_file():
        return {"status": "NOT_RUN", "reason": "No authorized external evaluation artifact was available at v2 freeze time."}
    payload = json.loads(V2_EXTERNAL_EVALUATION.read_text(encoding="utf-8"))
    return {
        "status": "COMPLETED_ZERO_SHOT_DESCRIPTIVE_EVALUATION",
        "report": str(V2_EXTERNAL_EVALUATION.relative_to(ROOT)).replace("\\", "/"),
        "dataset": payload.get("dataset"),
        "model": payload.get("model"),
        "metrics": payload.get("metrics"),
        "threshold_policy": payload.get("threshold_policy"),
        "clinical_validation_claim": False,
    }


def evaluate_v1(manifest: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader
    from app.ml.models.classifier import build_classifier

    checkpoint, identity = load_checkpoint(torch, V1_CHECKPOINT, V1_SHA)
    model_config = checkpoint.get("model_config", {})
    if model_config.get("backbone") != "efficientnet_b0" or model_config.get("num_classes") != 5:
        raise RuntimeError(f"Frozen v1 configuration mismatch: {model_config}")
    model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = select_device(torch, args.device)
    model.to(device)
    _, validation_transform = make_transforms(args.input_size)
    dataset = ResearchDataset(manifest["records"], "validation", validation_transform, make_mask_cache(manifest))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    rows = run_inference(model, loader, device, torch, lesion_aware=False)
    for row in rows:
        row["lesion_evidence"] = {"availability": "not_run_for_v1_baseline", "separate_from_grade": True}
        row["structure_evidence"] = {"availability": "not_run_for_v1_baseline", "separate_from_grade": True}
    return {
        "experiment_id": "v2_a_v1_frozen_baseline",
        "status": "FROZEN_BASELINE_REPRODUCED_ON_DEVELOPMENT_VALIDATION",
        "model_version": checkpoint.get("model_version", "efficientnet-b0-idrid-20260912-v1"),
        "checkpoint_path": identity["path"],
        "checkpoint_sha256": identity["sha256"],
        "architecture": model_config,
        "metrics": metrics_for(rows),
        "training_config": checkpoint.get("training_config", {}),
        "predictions": rows,
        "limitations": ["Raw softmax confidence is not clinically calibrated; no held-out calibration subset was fitted."],
    }, rows


def train_lesion_aware(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    seed_everything(args.seed, torch)
    device = select_device(torch, args.device)
    mask_cache = make_mask_cache(manifest)
    train_transform, validation_transform = make_transforms(args.input_size)
    train_dataset = ResearchDataset(manifest["records"], "train", train_transform, mask_cache)
    validation_dataset = ResearchDataset(manifest["records"], "validation", validation_transform, mask_cache)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, collate_fn=collate)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    v1_checkpoint, v1_identity = load_checkpoint(torch, V1_CHECKPOINT, V1_SHA)
    model_class = build_model(torch, nn)
    model = model_class(initialize_from=v1_checkpoint["state_dict"]).to(device)
    labels = [int(record["label"]) for record in train_dataset.records]
    class_weights = build_class_weights(labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7)
    output_dir = OUTPUT_ROOT / "v2_c_lesion_aware_efficientnet_b0"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": "v2_c_lesion_aware_efficientnet_b0",
        "description": "EfficientNet-B0 shared backbone with separate severity, hierarchical, lesion-presence and optic-disc structure heads.",
        "dataset": "idrid",
        "dataset_manifest": str(DEV_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "dataset_manifest_sha256": sha256(DEV_MANIFEST),
        "training_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "official_test_used": False,
        "architecture": "EfficientNet-B0 shared feature extractor; grade/stage heads; lesion presence head (MA/HE/EX/CWS); optic-disc coordinate head",
        "model_config": {"backbone": "efficientnet_b0", "num_classes": 5, "input_size": args.input_size, "ordinal_mode": False, "lesion_outputs": 4, "structure_outputs": 2},
        "initialization": {"source_checkpoint": v1_identity["path"], "source_sha256": v1_identity["sha256"], "matching_layers_loaded": "feature extractor, pool, dropout, stage1, stage2, severity", "new_heads": ["lesion_head", "structure_head"]},
        "preprocessing": {"color_space": "RGB", "resize": [args.input_size, args.input_size], "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225], "augmentation": ["RandomHorizontalFlip(p=0.5)", "RandomRotation(8 degrees)", "RandomAffine(translate=0.03, scale=0.95..1.05)", "ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)"], "validation": "resize + tensor + ImageNet normalization"},
        "loss": {"severity": "class-weighted cross entropy", "stage1": "cross entropy", "stage2": "cross entropy", "lesion": "masked BCE; unavailable masks excluded, not treated as negative", "structure": "masked SmoothL1 on normalized optic-disc x/y", "weights": {"severity": 1.0, "stage1": 0.25, "stage2": 0.25, "lesion": 0.25, "structure": 0.10}, "class_weights": [float(value) for value in class_weights.detach().cpu().tolist()]},
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "early_stopping": {"metric": "validation macro F1", "patience": args.patience},
        "seed": args.seed,
        "device": str(device),
        "production_promoted": False,
        "clinical_validation_claim": False,
    }
    json_dump(output_dir / "training_config.json", config)
    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    stale = 0
    start_epoch = 1
    resumed_from_epoch = None
    if args.resume_existing and last_path.is_file():
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["state_dict"], strict=True)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        resumed_from_epoch = int(resume_payload.get("epoch", 0))
        if best_path.is_file():
            previous_best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_score = float(previous_best.get("metrics", {}).get("f1", float("-inf")))
            best_epoch = int(previous_best.get("best_epoch", previous_best.get("epoch", 0)))
        config["resume"] = {"enabled": True, "from_checkpoint": str(last_path.relative_to(ROOT)).replace("\\", "/"), "from_epoch": resumed_from_epoch, "optimizer_state_restored": False}
        json_dump(output_dir / "training_config.json", config)
    started = time.perf_counter()
    last_completed_epoch = resumed_from_epoch or 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, labels_tensor, lesion_targets, lesion_available, structure_targets, structure_available, _records in train_loader:
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)
            lesion_targets = lesion_targets.to(device)
            lesion_available = lesion_available.to(device)
            structure_targets = structure_targets.to(device)
            structure_available = structure_available.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            stage1_targets = (labels_tensor > 0).long()
            stage2_targets = (labels_tensor >= 2).long()
            severity_loss = torch.nn.functional.cross_entropy(outputs["severity_logits"], labels_tensor, weight=class_weights)
            stage1_loss = torch.nn.functional.cross_entropy(outputs["stage1_logits"], stage1_targets)
            stage2_loss = torch.nn.functional.cross_entropy(outputs["stage2_logits"], stage2_targets)
            lesion_raw = torch.nn.functional.binary_cross_entropy_with_logits(outputs["lesion_logits"], lesion_targets, reduction="none")
            lesion_loss = (lesion_raw * lesion_available).sum() / lesion_available.sum().clamp_min(1.0)
            structure_raw = torch.nn.functional.smooth_l1_loss(outputs["structure_prediction"], structure_targets, reduction="none")
            structure_loss = (structure_raw * structure_available).sum() / structure_available.sum().clamp_min(1.0)
            loss = severity_loss + 0.25 * stage1_loss + 0.25 * stage2_loss + 0.25 * lesion_loss + 0.10 * structure_loss
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(labels_tensor)
            train_count += len(labels_tensor)
        validation_rows = run_inference(model, validation_loader, device, torch, lesion_aware=True)
        validation_metrics = metrics_for(validation_rows)
        scheduler.step(validation_metrics["f1"])
        record = {"epoch": epoch, "train_loss": train_loss / max(1, train_count), "validation_metrics": validation_metrics, "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(record)
        last_completed_epoch = epoch
        payload = {
            "state_dict": model.state_dict(),
            "model_config": config["model_config"],
            "training_config": config,
            "metrics": validation_metrics,
            "model_version": config["experiment_id"],
            "dataset": "idrid",
            "dataset_version": "idrid-v2-development-20260912",
            "epoch": epoch,
            "best_epoch": best_epoch,
            "production_promoted": False,
            "clinical_validation_claim": False,
        }
        torch.save(payload, last_path)
        if float(validation_metrics["f1"]) > best_score:
            best_score = float(validation_metrics["f1"])
            best_epoch = epoch
            payload["best_epoch"] = best_epoch
            torch.save(payload, best_path)
            stale = 0
        else:
            stale += 1
        print(f"v2_c epoch={epoch} loss={record['train_loss']:.4f} val_f1={validation_metrics['f1']:.4f} val_acc={validation_metrics['accuracy']:.4f} lr={record['learning_rate']:.8f}", flush=True)
        if stale >= args.patience:
            break
    if not best_path.is_file():
        raise RuntimeError("Lesion-aware experiment did not produce a checkpoint")
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["state_dict"], strict=True)
    final_rows = run_inference(model, validation_loader, device, torch, lesion_aware=True)
    checkpoint_sha = sha256(best_path)
    json_dump(output_dir / "history.json", history)
    json_dump(output_dir / "validation_predictions.json", final_rows)
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "model_version": config["experiment_id"],
        "checkpoint_path": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha,
        "metrics": metrics_for(final_rows),
        "training_config": config,
        "best_epoch": int(best_payload.get("best_epoch", best_epoch)),
        "epochs_completed": last_completed_epoch,
        "epochs_run_in_final_invocation": len(history),
        "training_seconds": time.perf_counter() - started,
        "predictions": final_rows,
        "limitations": ["Lesion heads supervise annotation presence, not full pixel segmentation; missing annotations are masked as unavailable.", "Raw softmax confidence is not clinically calibrated; no separate calibration set was fitted."],
    }
    json_dump(output_dir / "metrics.json", {"metrics": result["metrics"], "best_epoch": result["best_epoch"], "epochs_completed": result["epochs_completed"]})
    json_dump(output_dir / "model_manifest.json", {key: value for key, value in result.items() if key not in {"predictions"}})
    return result


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "predictions"}


def main() -> int:
    args = parse_args()
    import torch

    if not V1_CHECKPOINT.is_file() or sha256(V1_CHECKPOINT) != V1_SHA:
        raise RuntimeError("Frozen v1 checkpoint is missing or its SHA-256 changed; v2 run stopped")
    manifest = load_manifest()
    v1_before = sha256(V1_CHECKPOINT)
    baseline, baseline_rows = evaluate_v1(manifest, args)
    json_dump(METADATA_ROOT / "idrid_v2_v1_baseline_validation_predictions.json", baseline_rows)
    control = load_prior_control()
    if args.skip_training:
        output_dir = OUTPUT_ROOT / "v2_c_lesion_aware_efficientnet_b0"
        if not (output_dir / "model_manifest.json").is_file():
            raise RuntimeError("--skip-training requested but the v2 lesion-aware artifact is missing")
        lesion = json.loads((output_dir / "model_manifest.json").read_text(encoding="utf-8"))
        lesion["predictions"] = json.loads((output_dir / "validation_predictions.json").read_text(encoding="utf-8"))
    else:
        lesion = train_lesion_aware(manifest, args)
    v1_after = sha256(V1_CHECKPOINT)
    if v1_before != v1_after or v1_after != V1_SHA:
        raise RuntimeError("Frozen v1 checkpoint changed during v2 research")

    candidates = [baseline, control, lesion]
    for candidate in candidates:
        candidate["metrics_summary"] = compact_candidate(candidate).get("metrics", {})
    # Select only among v2 research candidates.  The rule is transparent and
    # validation-only: maximize QWK, then referable sensitivity, then macro F1,
    # while requiring that a candidate has no more referable FN than v1 when
    # possible.  If no candidate satisfies that guard, retain v1 explicitly.
    v1_fn = baseline["metrics"]["referable_dr"]["false_negative"]
    v2_candidates = [control, lesion]
    eligible = [candidate for candidate in v2_candidates if candidate["metrics"]["referable_dr"]["false_negative"] <= v1_fn]
    ranked = sorted(v2_candidates, key=lambda candidate: (candidate["metrics"].get("quadratic_weighted_kappa") or -1.0, candidate["metrics"]["referable_dr"].get("sensitivity", 0.0), candidate["metrics"].get("f1", 0.0)), reverse=True)
    selected = sorted(eligible, key=lambda candidate: (candidate["metrics"].get("quadratic_weighted_kappa") or -1.0, candidate["metrics"]["referable_dr"].get("sensitivity", 0.0), candidate["metrics"].get("f1", 0.0)), reverse=True)[0] if eligible else baseline
    selection_reason = "v2 candidate selected by validation-only ranking (QWK, referable sensitivity, macro F1) with a referable-FN guard against v1." if selected is not baseline else "No v2 candidate met the validation referable-FN guard (<= frozen v1); frozen v1 remains the research baseline."
    official_v1_reference = load_frozen_v1_official_reference()
    external_v2_reference = load_v2_external_reference()
    registry = {
        "schema_version": "idrid-v2-experiment-registry-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status": "EXPERIMENTAL_NOT_PRODUCTION",
        "official_test_used": False,
        "official_test_images_opened": 0,
        "development_manifest": str(DEV_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "development_manifest_sha256": sha256(DEV_MANIFEST),
        "experiments": [compact_candidate(candidate) for candidate in candidates],
        "selection": {"selected_experiment_id": selected["experiment_id"], "reason": selection_reason, "selection_data": "IDRiD development validation only", "production_promoted": False},
        "frozen_v1_official_test_reference": official_v1_reference,
        "v2_external_validation_reference": external_v2_reference,
        "frozen_v1_checkpoint_sha256_after_run": v1_after,
        "known_limitation": "IDRiD has no patient identifiers; source governance uses exact image hashes and split-qualified records.",
    }
    comparison = {
        "schema_version": "idrid-v2-comparison-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development validation only; official 103-image test untouched",
        "v1_frozen": compact_candidate(baseline),
        "v2_candidates": [compact_candidate(candidate) for candidate in v2_candidates],
        "ranking": [{"experiment_id": candidate["experiment_id"], "quadratic_weighted_kappa": candidate["metrics"].get("quadratic_weighted_kappa"), "macro_f1": candidate["metrics"].get("f1"), "referable_sensitivity": candidate["metrics"]["referable_dr"].get("sensitivity"), "referable_specificity": candidate["metrics"]["referable_dr"].get("specificity"), "referable_fn": candidate["metrics"]["referable_dr"].get("false_negative")} for candidate in ranked],
        "selection": registry["selection"],
        "frozen_v1_official_test_reference": official_v1_reference,
        "v2_external_validation_reference": external_v2_reference,
        "no_clinical_claim": True,
    }
    selected_artifact = {
        "schema_version": "idrid-v2-selected-candidate-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate": compact_candidate(selected),
        "selected_research_referable_threshold": selected["metrics"]["threshold_sweep"]["selected"] if selected is not baseline else {"threshold": 0.5, "status": "v1_authoritative_threshold_retained", "reason": "No v2 candidate passed selection guard; no v2 threshold adopted."},
        "severity_definition": "predicted_grade = argmax(P0..P4); never replaced by referable status",
        "referable_definition": "referable_probability = P2 + P3 + P4; referable = probability >= selected development threshold (production default remains separately governed)",
        "calibration_status": "Raw softmax confidence is not clinically calibrated; no separate calibration subset was fitted.",
        "production_promoted": False,
        "official_test_used": False,
        "official_test_images_opened": 0,
        "frozen_v1_official_test_results_unchanged": True,
        "frozen_v1_official_test_reference": official_v1_reference,
        "v2_external_validation_reference": external_v2_reference,
        "known_limitations": ["No patient identifiers; exact-image split governance only.", "Small validation set (83 images) makes selection and uncertainty estimates exploratory.", "Lesion annotations are incomplete, especially soft exudates; unavailable masks are not treated as negatives.", "This candidate is not approved for production or clinical use."],
    }
    json_dump(METADATA_ROOT / "idrid_v2_experiment_registry.json", registry)
    json_dump(METADATA_ROOT / "idrid_v2_comparison.json", comparison)
    json_dump(METADATA_ROOT / "idrid_v2_selected_candidate.json", selected_artifact)
    print(json.dumps({
        "selected_experiment_id": selected["experiment_id"],
        "selected_checkpoint": selected.get("checkpoint_path"),
        "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
        "selected_threshold": selected_artifact["selected_research_referable_threshold"],
        "v1_sha_unchanged": v1_after == V1_SHA,
        "official_test_images_opened": 0,
        "candidate_metrics": {candidate["experiment_id"]: {"accuracy": candidate["metrics"].get("accuracy"), "macro_f1": candidate["metrics"].get("f1"), "qwk": candidate["metrics"].get("quadratic_weighted_kappa"), "referable_sensitivity": candidate["metrics"]["referable_dr"].get("sensitivity"), "referable_specificity": candidate["metrics"]["referable_dr"].get("specificity"), "referable_fn": candidate["metrics"]["referable_dr"].get("false_negative")} for candidate in candidates},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
