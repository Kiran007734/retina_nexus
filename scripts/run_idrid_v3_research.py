"""Run a small, development-only IDRiD V3 robustness study.

V3 candidates are trained only on the governed IDRiD development training
split.  The fixed 83-image validation split is used for comparison and
threshold selection.  This script does not import, read, or score Messidor-2
and never opens the official IDRiD test images.
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
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.classifier import build_classifier  # noqa: E402
from ml.evaluation.metrics import classification_metrics  # noqa: E402
from ml.training.losses import build_class_weights, build_focal_loss  # noqa: E402
# Importing the V2 utilities keeps the architecture and annotation handling
# identical while the V3 runner writes to an entirely new namespace.
from scripts.run_idrid_v2_research import (  # noqa: E402
    LESION_NAMES,
    RAW_ROOT,
    V1_SHA,
    ResearchDataset,
    build_model,
    collate,
    make_mask_cache,
    metrics_for,
    sha256,
)
from scripts.train_classifier import make_transforms, select_device  # noqa: E402

DEV_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
V2_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v2" / "20260912" / "v2_c_lesion_aware_efficientnet_b0" / "checkpoint_best.pt"
V1_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "checkpoint_best.pt"
APTOS_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
V2_COMPARISON = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_comparison.json"
V2_FAILURE = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v3_failure_analysis.json"
OUTPUT_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912"
META_ROOT = ROOT / "ml" / "datasets" / "metadata" / "idrid"
REFERABLE_GRADES = (2, 3, 4)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled IDRiD V3 research experiments")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--experiments", default="b,c,d", help="Controlled candidates to run: b,c,d")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse completed B/C/D artifacts not selected for retraining")
    parser.add_argument("--finalize-existing", action="store_true", help="Rebuild all V3 reports from completed artifacts without training")
    return parser.parse_args()


def load_manifest() -> dict[str, Any]:
    payload = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("official_test_images_opened") != 0 or payload.get("official_test_used"):
        raise RuntimeError("Development manifest is not official-test untouched")
    if payload.get("source_governance", {}).get("integrity", {}).get("status") != "PASS":
        raise RuntimeError("IDRiD development integrity gate is not PASS")
    if len(payload.get("records", [])) != 406:
        raise RuntimeError("Expected 406 governed development records")
    return payload


def robust_train_transform(input_size: int):
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18, hue=0.04),
        transforms.RandomApply([transforms.RandomAutocontrast()], p=0.25),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.20),
        transforms.ToTensor(),
        normalize,
    ])


def validation_transform(input_size: int):
    return make_transforms(input_size)[1]


def load_state_into_custom(model: Any, checkpoint_path: Path, torch: Any, expected_sha: str | None = None) -> dict[str, Any]:
    actual_sha = sha256(checkpoint_path)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint_path} expected {expected_sha}, got {actual_sha}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"No state_dict in {checkpoint_path}")
    return {"payload": payload, "sha256": actual_sha, "state_dict": state, "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/")}


def init_model(torch: Any, nn: Any, source: Path, expected_sha: str | None = None):
    source_info = load_state_into_custom(None, source, torch, expected_sha)
    model_class = build_model(torch, nn)
    model = model_class()
    source_state = source_info["state_dict"]
    current = model.state_dict()
    compatible = {key: value for key, value in source_state.items() if key in current and tuple(value.shape) == tuple(current[key].shape)}
    result = model.load_state_dict(compatible, strict=False)
    expected_new = {"lesion_head.weight", "lesion_head.bias", "structure_head.weight", "structure_head.bias"}
    if set(result.missing_keys) - expected_new or result.unexpected_keys:
        raise RuntimeError(f"V3 initialization mismatch from {source}: missing={result.missing_keys} unexpected={result.unexpected_keys}")
    source_info["matching_layers_loaded"] = len(compatible)
    source_info["new_heads"] = sorted(set(result.missing_keys) & expected_new)
    return model, source_info


def run_inference(model: Any, loader: Any, device: Any, torch: Any, include_quality: bool = True) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, labels, lesion_targets, lesion_available, structure_targets, structure_available, records in loader:
            outputs = model(images.to(device))
            probabilities = torch.softmax(outputs["severity_logits"], dim=1)
            for index, record in enumerate(records):
                vector = probabilities[index].detach().cpu().tolist()
                confidence = float(max(vector))
                entropy = float(-sum(value * math.log(max(value, 1e-12)) for value in vector))
                row = {
                    "image_id": record["image_id"],
                    "record_key": record["record_key"],
                    "actual": int(labels[index].item()),
                    "logits": outputs["severity_logits"][index].detach().cpu().tolist(),
                    "probabilities": vector,
                    "predicted": int(np.argmax(vector)),
                    "confidence": confidence,
                    "referable_probability": float(sum(vector[2:5])),
                    "referable_predicted": int(sum(vector[2:5]) >= 0.5),
                    "entropy_nats": entropy,
                    "normalized_entropy": entropy / math.log(5.0),
                    "probability_margin": float(sorted(vector)[-1] - sorted(vector)[-2]),
                    "abstention": {"review_recommended": bool(entropy / math.log(5.0) >= 0.70 or sorted(vector)[-1] - sorted(vector)[-2] < 0.15), "does_not_change_grade": True},
                }
                if include_quality:
                    from scripts.run_idrid_v2_research import quality_proxy
                    row["quality"] = quality_proxy(RAW_ROOT / record["image"])
                rows.append(row)
    return rows


def losses(outputs: dict[str, Any], labels: Any, lesion_targets: Any, lesion_available: Any, structure_targets: Any, structure_available: Any, class_weights: Any, torch: Any, strategy: str) -> tuple[Any, dict[str, float]]:
    stage1 = torch.nn.functional.cross_entropy(outputs["stage1_logits"], (labels > 0).long())
    stage2 = torch.nn.functional.cross_entropy(outputs["stage2_logits"], (labels >= 2).long())
    if strategy == "focal":
        severity = build_focal_loss(gamma=2.0, alpha=class_weights)(outputs["severity_logits"], labels)
    else:
        severity = torch.nn.functional.cross_entropy(outputs["severity_logits"], labels, weight=class_weights)
    lesion_raw = torch.nn.functional.binary_cross_entropy_with_logits(outputs["lesion_logits"], lesion_targets, reduction="none")
    lesion = (lesion_raw * lesion_available).sum() / lesion_available.sum().clamp_min(1.0)
    structure_raw = torch.nn.functional.smooth_l1_loss(outputs["structure_prediction"], structure_targets, reduction="none")
    structure = (structure_raw * structure_available).sum() / structure_available.sum().clamp_min(1.0)
    total = severity + 0.25 * stage1 + 0.25 * stage2 + 0.25 * lesion + 0.10 * structure
    return total, {"severity": float(severity.detach().cpu()), "stage1": float(stage1.detach().cpu()), "stage2": float(stage2.detach().cpu()), "lesion": float(lesion.detach().cpu()), "structure": float(structure.detach().cpu())}


def threshold_priority(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = np.asarray([int(row["actual"] in REFERABLE_GRADES) for row in rows], dtype=int)
    entries: list[dict[str, Any]] = []
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
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
    eligible = [entry for entry in entries if entry["specificity"] >= 0.85]
    selected = sorted(eligible, key=lambda entry: (entry["sensitivity"], entry["specificity"], -entry["fn"], entry["f1"], -entry["threshold"]), reverse=True)[0] if eligible else None
    return {"candidate_thresholds": entries, "selection_priority": ["sensitivity", "specificity >= 0.85", "false negatives", "QWK", "macro F1", "Grade 3/4 sensitivity", "calibration"], "selected": selected, "external_labels_used": False}


def referable_brier(rows: list[dict[str, Any]]) -> float:
    actual = np.asarray([int(row["actual"] in REFERABLE_GRADES) for row in rows], dtype=float)
    probabilities = np.asarray([float(row["referable_probability"]) for row in rows], dtype=float)
    return float(np.mean((probabilities - actual) ** 2))


def robustness(model: Any, records: list[dict[str, Any]], mask_cache: dict[str, Any], device: Any, torch: Any, input_size: int, batch_size: int) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    base_transform = validation_transform(input_size)

    def transform_for(name: str):
        from torchvision import transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        class Perturb:
            def __call__(self, image):
                if name == "brightness_minus":
                    image = ImageEnhance.Brightness(image).enhance(0.85)
                elif name == "brightness_plus":
                    image = ImageEnhance.Brightness(image).enhance(1.15)
                elif name == "contrast_minus":
                    image = ImageEnhance.Contrast(image).enhance(0.80)
                elif name == "contrast_plus":
                    image = ImageEnhance.Contrast(image).enhance(1.20)
                elif name == "mild_blur":
                    image = image.filter(ImageFilter.GaussianBlur(radius=0.8))
                elif name == "fov_crop":
                    width, height = image.size
                    margin_x, margin_y = int(width * 0.04), int(height * 0.04)
                    image = image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
                return normalize(transforms.ToTensor()(transforms.Resize((input_size, input_size))(image)))
        return Perturb()

    base_dataset = ResearchDataset(records, "validation", base_transform, mask_cache)
    base_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    base_rows = run_inference(model, base_loader, device, torch, include_quality=False)
    output: dict[str, Any] = {"perturbations": {}}
    names = ("brightness_minus", "brightness_plus", "contrast_minus", "contrast_plus", "mild_blur", "fov_crop")
    for name in names:
        dataset = ResearchDataset(records, "validation", transform_for(name), mask_cache)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        rows = run_inference(model, loader, device, torch, include_quality=False)
        grade_stability = float(np.mean([a["predicted"] == b["predicted"] for a, b in zip(base_rows, rows)]))
        referable_stability = float(np.mean([(a["referable_probability"] >= 0.5) == (b["referable_probability"] >= 0.5) for a, b in zip(base_rows, rows)]))
        probability_shift = float(np.mean([np.mean(np.abs(np.asarray(a["probabilities"]) - np.asarray(b["probabilities"]))) for a, b in zip(base_rows, rows)]))
        output["perturbations"][name] = {"grade_prediction_stability": grade_stability, "referable_decision_stability_at_0_5": referable_stability, "mean_probability_l1_shift": probability_shift}
    return output


def train_candidate(candidate: str, manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(args.seed, torch)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = select_device(torch, args.device)
    mask_cache = make_mask_cache(manifest)
    train_transform = robust_train_transform(args.input_size) if candidate == "b" else make_transforms(args.input_size)[0]
    val_transform = validation_transform(args.input_size)
    train_dataset = ResearchDataset(manifest["records"], "train", train_transform, mask_cache)
    val_dataset = ResearchDataset(manifest["records"], "validation", val_transform, mask_cache)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    if candidate == "d":
        source_path, expected_sha, source_label = APTOS_CHECKPOINT, None, "APTOS EfficientNet-B0 checkpoint; direct controlled transfer"
    else:
        source_path, expected_sha, source_label = V2_CHECKPOINT, None, "V2 lesion-aware checkpoint; controlled continuation"
    model, source = init_model(torch, torch.nn, source_path, expected_sha)
    model.to(device)
    labels = [int(record["label"]) for record in train_dataset.records]
    class_weights = build_class_weights(labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = OUTPUT_ROOT / {"b": "v3_b_domain_robust", "c": "v3_c_class_balanced", "d": "v3_d_aptos_transfer"}[candidate]
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": f"v3_{candidate}",
        "candidate": candidate,
        "dataset": "idrid",
        "development_manifest": str(DEV_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "development_manifest_sha256": sha256(DEV_MANIFEST),
        "official_test_used": False,
        "external_evaluation_used_for_training_or_selection": False,
        "architecture": "EfficientNet-B0 shared backbone with separate severity, hierarchical, lesion-presence and optic-disc heads",
        "model_config": {"backbone": "efficientnet_b0", "num_classes": 5, "input_size": args.input_size, "ordinal_mode": False, "lesion_outputs": 4, "structure_outputs": 2},
        "initialization": {"source": source_label, "checkpoint": source["path"], "checkpoint_sha256": source["sha256"], "matching_layers_loaded": source["matching_layers_loaded"], "new_heads": source["new_heads"]},
        "preprocessing": {"input_size": args.input_size, "color_space": "RGB", "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225], "validation": "direct resize to square + tensor + ImageNet normalization", "train": "domain-robust augmentation" if candidate == "b" else "existing retinal augmentation"},
        "augmentation": (["Resize(224,224)", "RandomHorizontalFlip(0.5)", "RandomRotation(10)", "RandomAffine(translate=0.05, scale=0.90..1.10)", "ColorJitter(0.25,0.25,0.18,0.04)", "RandomAutocontrast(0.25)", "GaussianBlur(0.20)"] if candidate == "b" else ["Resize(224,224)", "RandomHorizontalFlip(0.5)", "RandomRotation(8)", "RandomAffine(translate=0.03, scale=0.95..1.05)", "ColorJitter(0.12,0.12,0.08,0.02)"]),
        "loss": {"severity": "class-weighted focal loss gamma=2" if candidate == "c" else "class-weighted cross entropy", "stage1": "cross entropy", "stage2": "cross entropy", "lesion": "masked BCE; unavailable masks excluded", "structure": "masked SmoothL1", "weights": {"severity": 1.0, "stage1": 0.25, "stage2": 0.25, "lesion": 0.25, "structure": 0.10}, "class_weights": [float(value) for value in class_weights.detach().cpu().tolist()]},
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "production_promoted": False,
        "clinical_validation_claim": False,
    }
    json_dump(output_dir / "training_config.json", config)
    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    best_score = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        count = 0
        for images, labels_tensor, lesion_targets, lesion_available, structure_targets, structure_available, _records in train_loader:
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)
            lesion_targets = lesion_targets.to(device)
            lesion_available = lesion_available.to(device)
            structure_targets = structure_targets.to(device)
            structure_available = structure_available.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss, components = losses(outputs, labels_tensor, lesion_targets, lesion_available, structure_targets, structure_available, class_weights, torch, "focal" if candidate == "c" else "ce")
            loss.backward()
            optimizer.step()
            totals["total"] += float(loss.detach().cpu()) * len(labels_tensor)
            for key, value in components.items():
                totals[key] += value * len(labels_tensor)
            count += len(labels_tensor)
        validation_rows = run_inference(model, val_loader, device, torch)
        validation_metrics = metrics_for(validation_rows)
        threshold = threshold_priority(validation_rows)
        validation_metrics["threshold_policy_v3"] = threshold
        history.append({"epoch": epoch, "train_loss": totals["total"] / max(1, count), "loss_components": {key: value / max(1, count) for key, value in totals.items() if key != "total"}, "validation_metrics": validation_metrics})
        payload = {"state_dict": model.state_dict(), "model_config": config["model_config"], "training_config": config, "metrics": validation_metrics, "model_version": config["experiment_id"], "dataset": "idrid", "dataset_version": "idrid-v3-development-20260912", "epoch": epoch, "best_epoch": best_epoch, "production_promoted": False, "clinical_validation_claim": False}
        torch.save(payload, last_path)
        if float(validation_metrics["f1"]) > best_score:
            best_score = float(validation_metrics["f1"])
            best_epoch = epoch
            payload["best_epoch"] = best_epoch
            torch.save(payload, best_path)
        print(f"{config['experiment_id']} epoch={epoch} loss={totals['total']/max(1,count):.4f} val_f1={validation_metrics['f1']:.4f} val_acc={validation_metrics['accuracy']:.4f}", flush=True)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["state_dict"], strict=True)
    final_rows = run_inference(model, val_loader, device, torch)
    final_metrics = metrics_for(final_rows)
    final_metrics["threshold_policy_v3"] = threshold_priority(final_rows)
    robustness_report = robustness(model, manifest["records"], mask_cache, device, torch, args.input_size, args.batch_size)
    checkpoint_sha = sha256(best_path)
    result = {
        "experiment_id": config["experiment_id"],
        "candidate": candidate,
        "status": "COMPLETED",
        "checkpoint_path": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha,
        "metrics": final_metrics,
        "robustness": robustness_report,
        "training_config": config,
        "best_epoch": int(best_payload.get("best_epoch", best_epoch)),
        "epochs_completed": args.epochs,
        "training_seconds": time.perf_counter() - started,
        "validation_predictions": final_rows,
        "production_promoted": False,
        "official_test_images_opened": 0,
        "limitations": ["Small fixed validation set (83 images).", "IDRiD has no patient identifiers.", "Lesion annotations are incomplete; unavailable masks are never treated as negatives.", "This is a research candidate, not a clinical validation or production model."],
    }
    json_dump(output_dir / "history.json", history)
    json_dump(output_dir / "validation_predictions.json", final_rows)
    json_dump(output_dir / "metrics.json", {"metrics": final_metrics, "best_epoch": result["best_epoch"], "robustness": robustness_report})
    json_dump(output_dir / "robustness.json", robustness_report)
    json_dump(output_dir / "model_manifest.json", {key: value for key, value in result.items() if key != "validation_predictions"})
    return result


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "validation_predictions"}


def load_existing_candidate(candidate: str) -> dict[str, Any]:
    directory = OUTPUT_ROOT / {"b": "v3_b_domain_robust", "c": "v3_c_class_balanced", "d": "v3_d_aptos_transfer"}[candidate]
    manifest_path = directory / "model_manifest.json"
    predictions_path = directory / "validation_predictions.json"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise RuntimeError(f"Cannot reuse incomplete V3 candidate {candidate}: {directory}")
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    result["validation_predictions"] = json.loads(predictions_path.read_text(encoding="utf-8"))
    robustness_path = directory / "robustness.json"
    result["robustness"] = json.loads(robustness_path.read_text(encoding="utf-8")) if robustness_path.is_file() else None
    result["status"] = "REUSED_COMPLETED_ARTIFACT"
    return result


def reproducibility_check(result: dict[str, Any], manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    seed_everything(args.seed, torch)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    checkpoint_path = ROOT / result["checkpoint_path"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_class = build_model(torch, torch.nn)
    model = model_class()
    model.load_state_dict(payload["state_dict"], strict=True)
    device = select_device(torch, args.device)
    model.to(device)
    cache = make_mask_cache(manifest)
    dataset = ResearchDataset(manifest["records"], "validation", validation_transform(args.input_size), cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    first = run_inference(model, loader, device, torch, include_quality=False)
    second = run_inference(model, loader, device, torch, include_quality=False)
    probability_delta = max(float(np.max(np.abs(np.asarray(a["probabilities"]) - np.asarray(b["probabilities"])))) for a, b in zip(first, second))
    return {"checkpoint_sha256": sha256(checkpoint_path), "checkpoint_unchanged": sha256(checkpoint_path) == result["checkpoint_sha256"], "rows_first": len(first), "rows_second": len(second), "predictions_identical": all(a["predicted"] == b["predicted"] for a, b in zip(first, second)), "probabilities_max_abs_delta": probability_delta, "probabilities_identical_within_1e-6": probability_delta <= 1e-6, "preprocessing": "same validation transform and ImageNet normalization", "official_test_images_opened": 0}


def write_data_audit(manifest: dict[str, Any], forensic: dict[str, Any] | None) -> dict[str, Any]:
    source = json.loads((META_ROOT / "idrid_v2_data_audit.json").read_text(encoding="utf-8"))
    audit = {
        "schema_version": "idrid-v3-data-audit-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_manifest": str(DEV_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "development_manifest_sha256": sha256(DEV_MANIFEST),
        "idrid_development_counts": manifest["source_governance"]["counts"],
        "idrid_integrity": manifest["source_governance"]["integrity"],
        "duplicate_and_conflict_policy": "Existing seven governed exclusions and same-split duplicate handling remain unchanged; no labels are modified.",
        "aptos_training_only_audit": source.get("aptos2019"),
        "aptos_test_images_used": False,
        "messidor2_used_for_training_or_selection": False,
        "official_idrid_test_images_opened": 0,
        "patient_level_leakage_limitation": "No patient identifiers are provided; exact-image governance is the available leakage control.",
        "cross_dataset_exact_hash_matches_in_existing_audit": source.get("aptos2019", {}).get("exact_hash_matches_with_idrid_development", []),
        "forensic_reference": "idrid_v3_failure_analysis.json",
        "forensic_not_used_for_external_selection": True,
    }
    json_dump(META_ROOT / "idrid_v3_data_audit.json", audit)
    return audit


def main() -> int:
    args = parse_args()
    import torch

    manifest = load_manifest()
    if sha256(V1_CHECKPOINT) != V1_SHA:
        raise RuntimeError("Frozen V1 checkpoint SHA changed before V3")
    if not V2_CHECKPOINT.is_file():
        raise RuntimeError("Frozen V2 checkpoint is missing")
    forensic = json.loads(V2_FAILURE.read_text(encoding="utf-8")) if V2_FAILURE.is_file() else None
    write_data_audit(manifest, forensic)
    candidates: list[dict[str, Any]] = []
    v2_manifest = json.loads((V2_CHECKPOINT.parent / "model_manifest.json").read_text(encoding="utf-8"))
    v2_predictions = json.loads((V2_CHECKPOINT.parent / "validation_predictions.json").read_text(encoding="utf-8"))
    v2_metrics = json.loads((V2_CHECKPOINT.parent / "metrics.json").read_text(encoding="utf-8"))
    v2_selected = json.loads((META_ROOT / "idrid_v2_selected_candidate.json").read_text(encoding="utf-8"))
    v2_threshold = v2_selected.get("selected_research_referable_threshold", {}).get("threshold", 0.45)
    v2_control_metrics = dict(v2_metrics["metrics"])
    v2_control_metrics["threshold_policy_v3"] = {"selected": next((entry for entry in threshold_priority(v2_predictions)["candidate_thresholds"] if entry["threshold"] == v2_threshold), None), "source": "frozen IDRiD V2 development threshold", "external_labels_used": False}
    candidates.append({"experiment_id": "v3_a_v2_reproduction_control", "status": "FROZEN_CONTROL_REUSED", "checkpoint_path": str(V2_CHECKPOINT.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": sha256(V2_CHECKPOINT), "metrics": v2_control_metrics, "robustness": json.loads((V2_CHECKPOINT.parent / "robustness.json").read_text(encoding="utf-8")) if (V2_CHECKPOINT.parent / "robustness.json").is_file() else None, "validation_predictions": v2_predictions, "training_config": v2_manifest.get("training_config", {}), "production_promoted": False, "official_test_images_opened": 0, "limitations": ["Frozen V2 control; no retraining in V3."]})
    requested = set() if args.finalize_existing else {item.strip() for item in args.experiments.split(",") if item.strip()}
    if args.finalize_existing:
        args.reuse_existing = True
    if requested - {"b", "c", "d"}:
        raise ValueError(f"Unsupported V3 candidate(s) {sorted(requested - {'b', 'c', 'd'})}")
    for candidate in ("b", "c", "d"):
        if candidate in requested:
            candidates.append(train_candidate(candidate, manifest, args))
        elif args.reuse_existing:
            candidates.append(load_existing_candidate(candidate))
    reproducibility = {candidate["experiment_id"]: reproducibility_check(candidate, manifest, args) for candidate in candidates if candidate["experiment_id"] != "v3_a_v2_reproduction_control"}
    json_dump(META_ROOT / "idrid_v3_reproducibility.json", {"schema_version": "idrid-v3-reproducibility-1", "official_test_images_opened": 0, "candidates": reproducibility})
    robustness_artifact = {candidate["experiment_id"]: candidate.get("robustness") for candidate in candidates}
    json_dump(META_ROOT / "idrid_v3_robustness.json", {"schema_version": "idrid-v3-robustness-1", "method": "fixed development validation perturbations; no external images or labels", "candidates": robustness_artifact})
    calibration = {
        "schema_version": "idrid-v3-calibration-1",
        "status": "UNCALIBRATED",
        "decision": "Raw softmax confidence is not clinically calibrated; a separate calibration subset was not fitted because the 83-image validation set is too small and creating one would weaken controlled model selection.",
        "official_test_used": False,
        "messidor2_used_for_calibration": False,
        "candidates": {candidate["experiment_id"]: {"raw_softmax_ece_10_bins": candidate["metrics"].get("raw_softmax_ece_10_bins"), "referable_brier": referable_brier(candidate["validation_predictions"]), "calibrator_fitted": False} for candidate in candidates},
    }
    json_dump(META_ROOT / "idrid_v3_calibration.json", calibration)
    v2_sensitivity = candidates[0]["metrics"]["referable_dr"]["sensitivity"]
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        threshold = candidate["metrics"].get("threshold_policy_v3", {}).get("selected")
        sensitivity = threshold["sensitivity"] if threshold else candidate["metrics"]["referable_dr"]["sensitivity"]
        specificity = threshold["specificity"] if threshold else candidate["metrics"]["referable_dr"]["specificity"]
        fn = threshold["fn"] if threshold else candidate["metrics"]["referable_dr"]["false_negative"]
        scored.append({"experiment_id": candidate["experiment_id"], "threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "false_negatives": fn, "qwk": candidate["metrics"].get("quadratic_weighted_kappa"), "macro_f1": candidate["metrics"].get("f1"), "grade_3_4_sensitivity": float(np.mean([candidate["metrics"].get("per_class", {}).get(label, {}).get("sensitivity", 0.0) for label in ("Severe", "Proliferative DR")])), "ece": candidate["metrics"].get("raw_softmax_ece_10_bins")})
    eligible = [entry for entry in scored if entry["specificity"] >= 0.85 and entry["sensitivity"] >= v2_sensitivity]
    selected_entry = sorted(eligible, key=lambda entry: (entry["sensitivity"], entry["specificity"], -entry["false_negatives"], entry["qwk"] or -1.0, entry["macro_f1"] or -1.0, entry["grade_3_4_sensitivity"], -(entry["ece"] or 1.0)), reverse=True)[0] if eligible else None
    selected = next((candidate for candidate in candidates if candidate["experiment_id"] == selected_entry["experiment_id"]), candidates[0]) if selected_entry else candidates[0]
    selection_reason = "Selected using IDRiD development validation only: sensitivity >= V2 control, specificity >=85%, then FN, QWK, macro F1, Grade 3/4 sensitivity, calibration." if selected_entry else "No V3 candidate met the predefined sensitivity and specificity gate; frozen V2 remains the research control."
    registry = {
        "schema_version": "idrid-v3-experiment-registry-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status": "EXPERIMENTAL_NOT_PRODUCTION",
        "experiments": [compact(candidate) for candidate in candidates],
        "selection": {"selected_experiment_id": selected["experiment_id"], "reason": selection_reason, "development_only": True, "production_promoted": False},
        "messidor2_used_for_selection": False,
        "official_test_images_opened": 0,
        "v1_checkpoint_sha256_verified": sha256(V1_CHECKPOINT),
        "v2_checkpoint_sha256": sha256(V2_CHECKPOINT),
        "candidate_evaluation_priority": ["sensitivity", "specificity >=85%", "false negatives", "QWK", "macro F1", "Grade 3/4 sensitivity", "calibration"],
        "candidate_e_not_run": "No separate combination experiment was run; the limited study avoided a sweep and no pre-external evidence justified another combination before freezing.",
    }
    comparison = {"schema_version": "idrid-v3-comparison-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "IDRiD development validation only; official test and Messidor-2 excluded from selection", "v1_official_reference": json.loads((META_ROOT / "idrid_official_test_evaluation.json").read_text(encoding="utf-8")).get("evaluation"), "v2_control": compact(candidates[0]), "v3_candidates": [compact(candidate) for candidate in candidates[1:]], "ranking": scored, "selection": registry["selection"], "no_clinical_claim": True}
    selected_artifact = {"schema_version": "idrid-v3-selected-candidate-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "selected_candidate": compact(selected), "selected_threshold": selected_entry["threshold"] if selected_entry else {"threshold": 0.5, "status": "V2_CONTROL_THRESHOLD_RETAINED"}, "production_promoted": False, "official_test_images_opened": 0, "messidor2_used_for_selection": False, "severity_definition": "argmax(P0..P4); referable status is separate", "referable_definition": "P2 + P3 + P4 >= development-only frozen threshold", "known_limitations": ["Small 83-image development validation set.", "No patient IDs; patient-level leakage cannot be guaranteed.", "External generalization is not established until one post-freeze Messidor-2 evaluation.", "No clinical validation claim."]}
    json_dump(META_ROOT / "idrid_v3_experiment_registry.json", registry)
    json_dump(META_ROOT / "idrid_v3_comparison.json", comparison)
    json_dump(META_ROOT / "idrid_v3_selected_candidate.json", selected_artifact)
    print(json.dumps({"selected": selected["experiment_id"], "selected_checkpoint": selected["checkpoint_path"], "selected_sha256": selected["checkpoint_sha256"], "selected_threshold": selected_artifact["selected_threshold"], "scored": scored, "official_test_images_opened": 0, "messidor2_used_for_selection": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
