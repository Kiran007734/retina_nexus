"""Run a small, fixed-split IDRiD classifier research comparison.

This runner is intentionally separate from the production APTOS/IDRiD
trainers. It never reads the official IDRiD test records, never updates the
production environment, and stores all candidate artifacts under the research
directory. The current IDRiD checkpoint is evaluated as a frozen baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.classifier import (  # noqa: E402
    ReferableDRMapping,
    build_classifier,
    severity_probabilities,
)
from ml.evaluation.metrics import classification_metrics  # noqa: E402
from ml.training.classification_dataset import FundusClassificationDataset  # noqa: E402
from ml.training.losses import (  # noqa: E402
    build_class_weights,
    build_focal_loss,
    build_ordinal_loss,
    build_weighted_sampler,
    hierarchical_loss,
)
from scripts.train_classifier import make_transforms, seed_everything, select_device  # noqa: E402


DEFAULT_SPLIT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"
DEFAULT_APTOS = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
DEFAULT_CURRENT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "checkpoint_best.pt"
DEFAULT_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "20260912"
DEFAULT_METADATA_ROOT = ROOT / "ml" / "datasets" / "metadata" / "idrid"
CLASS_MAPPING = {"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative DR"}
REFERABLE_GRADES = (2, 3, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "torchvision", "numpy", "scikit-learn", "Pillow"):
        try:
            result[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            result[package] = "UNAVAILABLE"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-split IDRiD research classifier experiments")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--aptos-checkpoint", type=Path, default=DEFAULT_APTOS)
    parser.add_argument("--current-checkpoint", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--experiments", default="a,b,c", help="Comma-separated experiments: a,b,c")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse completed experiment metrics unless forced")
    parser.add_argument("--force-experiments", default="", help="Comma-separated experiments to rerun when --reuse-existing is set")
    return parser.parse_args()


def load_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "idrid":
        raise RuntimeError("The split manifest is not an IDRiD manifest")
    if payload.get("leakage", {}).get("status") != "pass":
        raise RuntimeError("The IDRiD split manifest does not have leakage status pass")
    records = payload.get("records", [])
    if not records or any(record.get("split") not in {"train", "validation"} for record in records):
        raise RuntimeError("The research manifest may contain only train/validation records")
    if any(record.get("official_split") != "train" for record in records):
        raise RuntimeError("A non-training official record was found in the research manifest")
    if any(record.get("label") not in range(5) for record in records):
        raise RuntimeError("IDRiD labels must be integers in the range 0..4")
    return payload


def collate_fundus_batch(batch):
    import torch

    images, labels, records = zip(*batch)
    return torch.stack(list(images), dim=0), torch.stack(list(labels), dim=0), list(records)


def load_initialization(model, checkpoint_path: Path, ordinal_mode: bool, torch) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Initialization checkpoint does not exist: {checkpoint_path}")
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})
    if model_config.get("backbone") != "efficientnet_b0" or model_config.get("num_classes") != 5 or model_config.get("input_size") != 224:
        raise RuntimeError(f"Incompatible initialization checkpoint model configuration: {model_config}")
    source_state = checkpoint.get("state_dict")
    if not isinstance(source_state, dict):
        raise RuntimeError("Initialization checkpoint has no state_dict")

    if not ordinal_mode:
        result = model.load_state_dict(source_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Strict initialization mismatch: {result}")
        return {
            "source": "APTOS EfficientNet-B0 checkpoint, strict initialization",
            "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": checkpoint_hash,
            "model_version": checkpoint.get("model_version"),
            "state_dict_loading": "strict",
            "missing_keys": [],
            "unexpected_keys": [],
        }

    # The ordinal head has four outputs and cannot consume the APTOS five-class
    # severity head. Reuse only matching backbone and hierarchical head tensors;
    # the ordinal head remains freshly initialized and is recorded explicitly.
    target_state = model.state_dict()
    compatible = {key: value for key, value in source_state.items() if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)}
    result = model.load_state_dict(compatible, strict=False)
    expected_random = sorted(key for key in target_state if key.startswith("ordinal_head."))
    if sorted(result.missing_keys) != expected_random or result.unexpected_keys:
        raise RuntimeError(f"Ordinal partial initialization mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    return {
        "source": "APTOS EfficientNet-B0 matching layers; ordinal head freshly initialized",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash,
        "model_version": checkpoint.get("model_version"),
        "state_dict_loading": "partial_matching_layers",
        "missing_keys": sorted(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def build_loaders(split_path: Path, batch_size: int, input_size: int, device, experiment: str, seed: int):
    import torch
    from torch.utils.data import DataLoader

    train_transform, validation_transform = make_transforms(input_size)
    raw_root = ROOT / "ml" / "datasets" / "raw" / "idrid"
    train_dataset = FundusClassificationDataset(split_path, raw_root, "train", train_transform)
    validation_dataset = FundusClassificationDataset(split_path, raw_root, "validation", validation_transform)
    train_labels = [int(record["label"]) for record in train_dataset.records]
    generator = torch.Generator()
    generator.manual_seed(seed)
    if experiment == "a":
        sampler = build_weighted_sampler(train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_fundus_batch)
        sampling = "weighted_sampler_with_replacement"
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_fundus_batch)
        sampling = "shuffle"
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_fundus_batch)
    return train_dataset, validation_dataset, train_loader, validation_loader, train_labels, sampling


def experiment_spec(experiment: str) -> dict[str, Any]:
    specs = {
        "a": {
            "name": "experiment_a_sampler_balanced_ce",
            "description": "EfficientNet-B0 with weighted sampling and unweighted severity cross-entropy.",
            "ordinal_mode": False,
            "ordinal_parameterization": "softmax_5_class",
            "loss_strategy": "plain_cross_entropy",
            "sampler": "weighted_sampler_with_replacement",
        },
        "b": {
            "name": "experiment_b_class_balanced_focal",
            "description": "EfficientNet-B0 with class-weighted focal severity loss (gamma=2).",
            "ordinal_mode": False,
            "ordinal_parameterization": "softmax_5_class",
            "loss_strategy": "class_weighted_focal_loss",
            "sampler": "shuffle",
        },
        "c": {
            "name": "experiment_c_ordinal_monotonic_bce",
            "description": "EfficientNet-B0 with monotonic cumulative ordinal thresholds and weighted BCE.",
            "ordinal_mode": True,
            "ordinal_parameterization": "monotonic_cumulative_logit",
            "loss_strategy": "ordinal_weighted_bce",
            "sampler": "shuffle",
        },
    }
    if experiment not in specs:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose a, b, or c")
    return specs[experiment]


def build_criterion(experiment: str, train_labels: list[int], class_weights, torch):
    if experiment == "a":
        return torch.nn.CrossEntropyLoss()
    if experiment == "b":
        return build_focal_loss(gamma=2.0, alpha=class_weights)
    return build_ordinal_loss("weighted_loss", train_labels)


def prepare_outputs(raw_outputs: dict[str, Any], ordinal_mode: bool, ordinal_parameterization: str, torch) -> dict[str, Any]:
    if not ordinal_mode or ordinal_parameterization != "monotonic_cumulative_logit":
        return raw_outputs
    raw_logits = raw_outputs["ordinal_logits"]
    base = raw_logits[:, :1]
    positive_steps = torch.nn.functional.softplus(raw_logits[:, 1:])
    cumulative_logits = torch.cat((base, base - torch.cumsum(positive_steps, dim=1)), dim=1)
    outputs = dict(raw_outputs)
    outputs["ordinal_logits"] = cumulative_logits
    return outputs


def run_epoch(model, loader, optimizer, criterion, mapping, device, torch, ordinal_mode: bool, ordinal_parameterization: str, amp_enabled: bool) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    components = {"stage1": 0.0, "stage2": 0.0, "severity": 0.0}
    actual: list[int] = []
    probabilities: list[list[float]] = []
    for images, labels, _records in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = prepare_outputs(model(images), ordinal_mode, ordinal_parameterization, torch)
            loss, batch_components = hierarchical_loss(outputs, labels, mapping, criterion, ordinal_mode=ordinal_mode)
        if training:
            loss.backward()
            optimizer.step()
        count = len(labels)
        total_loss += float(loss.detach().cpu()) * count
        for key in components:
            components[key] += batch_components[key] * count
        with torch.inference_mode():
            probabilities.extend(severity_probabilities(outputs, ordinal_mode).detach().cpu().tolist())
        actual.extend(labels.detach().cpu().tolist())
    count = max(1, len(actual))
    metrics = classification_metrics(actual, probabilities, referable_grades=REFERABLE_GRADES)
    metrics["loss"] = total_loss / count
    metrics["loss_components"] = {key: value / count for key, value in components.items()}
    return metrics


def infer_rows(model, loader, device, torch, ordinal_mode: bool, ordinal_parameterization: str = "softmax_5_class") -> tuple[list[int], list[list[float]], list[dict[str, Any]]]:
    model.eval()
    actual: list[int] = []
    probabilities: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, labels, records in loader:
            outputs = prepare_outputs(model(images.to(device, non_blocking=device.type == "cuda")), ordinal_mode, ordinal_parameterization, torch)
            logits = outputs["ordinal_logits"] if ordinal_mode else outputs["severity_logits"]
            probs = severity_probabilities(outputs, ordinal_mode)
            for index, record in enumerate(records):
                vector = probs[index].detach().cpu().tolist()
                row_actual = int(labels[index].item())
                predicted = int(np.argmax(vector))
                confidence = float(max(vector))
                referable_probability = float(sum(vector[2:5]))
                entropy = float(-sum(value * math.log(max(value, 1e-12)) for value in vector))
                rows.append({
                    "image_id": record["image_id"],
                    "record_key": record.get("record_key"),
                    "actual": row_actual,
                    "logits": logits[index].detach().cpu().tolist(),
                    "probabilities": vector,
                    "predicted": predicted,
                    "confidence": confidence,
                    "referable_probability": referable_probability,
                    "referable_predicted": int(referable_probability >= 0.5),
                    "entropy_nats": entropy,
                })
                actual.append(row_actual)
                probabilities.append(vector)
    return actual, probabilities, rows


def calibration_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    correct = np.asarray([int(row["actual"] == row["predicted"]) for row in rows], dtype=int)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            ece += abs(float(correct[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return {
        "temperature_scaling_fitted": False,
        "validation_confidence_is_raw_softmax": True,
        "calibration_data_exists": False,
        "ece_10_bins": float(ece),
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
        "confidence_mean": float(confidence.mean()),
        "confidence_median": float(np.median(confidence)),
        "clinical_statement": "confidence is NOT clinically calibrated",
    }


def audit_metrics(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual_ref = np.asarray([int(row["actual"] in REFERABLE_GRADES) for row in rows], dtype=int)
    predicted_ref = np.asarray([row["referable_predicted"] for row in rows], dtype=int)
    tp = int(((actual_ref == 1) & (predicted_ref == 1)).sum())
    tn = int(((actual_ref == 0) & (predicted_ref == 0)).sum())
    fp = int(((actual_ref == 0) & (predicted_ref == 1)).sum())
    fn = int(((actual_ref == 1) & (predicted_ref == 0)).sum())
    errors = sorted((row for row in rows if row["actual"] != row["predicted"]), key=lambda row: (-row["confidence"], row["image_id"]))
    return {
        "calibration": calibration_audit(rows),
        "referable_counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "predicted_distribution": dict(sorted(Counter(row["predicted"] for row in rows).items())),
        "mean_referable_probability_actual_referable": float(np.mean([row["referable_probability"] for row in rows if row["actual"] in REFERABLE_GRADES])),
        "mean_referable_probability_actual_nonreferable": float(np.mean([row["referable_probability"] for row in rows if row["actual"] not in REFERABLE_GRADES])),
        "false_negative_ids": [row["image_id"] for row in rows if row["actual"] in REFERABLE_GRADES and row["referable_predicted"] == 0],
        "top_10_high_confidence_errors": errors[:10],
        "metrics_referable": metrics["referable_dr"],
    }


def config_for(spec: dict[str, Any], args, split: dict[str, Any], split_path: Path, initialization: dict[str, Any], train_labels: list[int], class_weights, device, sampling: str) -> dict[str, Any]:
    dataset_version = f"idrid-grading-{split['generated_from']['grading_manifest_sha256'][:16]}-split-{split['seed']}"
    return {
        "experiment_name": spec["name"],
        "description": spec["description"],
        "dataset": "idrid",
        "dataset_version": dataset_version,
        "dataset_source": "ml/datasets/raw/idrid/B. Disease Grading/",
        "split_manifest": str(split_path.relative_to(ROOT)).replace("\\", "/"),
        "split_manifest_sha256": sha256(split_path),
        "official_test_set": {"status": "RESERVED_NOT_EVALUATED", "record_count": len(split.get("reserved_official_test_records", []))},
        "architecture": "EfficientNet-B0 with RETINA-NEXUS hierarchical heads",
        "backbone": "efficientnet_b0",
        "ordinal_mode": spec["ordinal_mode"],
        "ordinal_parameterization": spec["ordinal_parameterization"],
        "initialization": initialization,
        "input_size": args.input_size,
        "input_channels": 3,
        "color_space": "RGB",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "augmentation": ["Resize(224,224)", "RandomHorizontalFlip(p=0.5)", "RandomRotation(8 degrees)", "RandomAffine(translate=0.03, scale=0.95..1.05)", "ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)"],
        "validation_transform": ["Resize(224,224)", "ToTensor", "ImageNet normalization"],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=1e-7)",
        "epochs_requested": args.epochs,
        "early_stopping": {"metric": "validation_macro_f1", "patience": args.patience},
        "loss_strategy": spec["loss_strategy"],
        "sampler": sampling,
        "focal_gamma": 2.0 if spec["loss_strategy"] == "class_weighted_focal_loss" else None,
        "class_weights_from_train_subset_only": [float(value) for value in class_weights.detach().cpu().tolist()],
        "class_weight_counts_from_train_subset": dict(sorted(Counter(train_labels).items())),
        "class_mapping": CLASS_MAPPING,
        "referable_mapping": {"name": "grade_2_or_worse", "referable_grades": list(REFERABLE_GRADES)},
        "seed": args.seed,
        "device_requested": args.device,
        "device_used": str(device),
        "mixed_precision_requested": args.mixed_precision,
        "amp_enabled": bool(args.mixed_precision and device.type == "cuda"),
        "fine_tune_all_layers": True,
        "train_record_count": len(train_labels),
        "validation_record_count": 83,
        "software_versions": versions(),
        "excluded_records": split.get("excluded_records", []),
        "clinical_validation_claim": False,
        "production_promoted": False,
    }


def run_training_experiment(experiment: str, args, split: dict[str, Any], split_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    spec = experiment_spec(experiment)
    seed_everything(args.seed, torch)
    device = select_device(torch, args.device)
    train_dataset, validation_dataset, train_loader, validation_loader, train_labels, sampling = build_loaders(split_path, args.batch_size, args.input_size, device, experiment, args.seed)
    model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=spec["ordinal_mode"]).to(device)
    initialization = load_initialization(model, args.aptos_checkpoint, spec["ordinal_mode"], torch)
    class_weights = build_class_weights(train_labels).to(device)
    criterion = build_criterion(experiment, train_labels, class_weights, torch)
    mapping = ReferableDRMapping(name="grade_2_or_worse", referable_grades=REFERABLE_GRADES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7)
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    config = config_for(spec, args, split, split_path, initialization, train_labels, class_weights, device, sampling)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    start = time.perf_counter()
    best_score = float("-inf")
    best_epoch = 0
    patience_count = 0
    history: list[dict[str, Any]] = []
    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, mapping, device, torch, spec["ordinal_mode"], spec["ordinal_parameterization"], amp_enabled)
        validation_metrics = run_epoch(model, validation_loader, None, criterion, mapping, device, torch, spec["ordinal_mode"], spec["ordinal_parameterization"], False)
        scheduler.step(validation_metrics["f1"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "learning_rate": learning_rate, "train": train_metrics, "validation": validation_metrics})
        payload = {
            "state_dict": model.state_dict(),
            "model_config": {"backbone": "efficientnet_b0", "num_classes": 5, "input_size": args.input_size, "ordinal_mode": spec["ordinal_mode"]},
            "training_config": config,
            "metrics": validation_metrics,
            "dataset": "idrid",
            "dataset_version": config["dataset_version"],
            "model_version": spec["name"],
            "epoch": epoch,
            "best_epoch": best_epoch,
            "clinical_validation_claim": False,
            "production_promoted": False,
        }
        torch.save(payload, last_path)
        score = float(validation_metrics["f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            payload["best_epoch"] = best_epoch
            torch.save(payload, best_path)
            patience_count = 0
        else:
            patience_count += 1
        print(f"{spec['name']} epoch={epoch} train_loss={train_metrics['loss']:.4f} validation_f1={validation_metrics['f1']:.4f} validation_accuracy={validation_metrics['accuracy']:.4f} lr={learning_rate:.8f}", flush=True)
        if patience_count >= args.patience:
            print(f"{spec['name']} early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    training_seconds = time.perf_counter() - start
    if not best_path.is_file():
        raise RuntimeError(f"{spec['name']} ended without a best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    model.to(device)
    actual, probabilities, rows = infer_rows(model, validation_loader, device, torch, spec["ordinal_mode"], spec["ordinal_parameterization"])
    metrics = classification_metrics(actual, probabilities, referable_grades=REFERABLE_GRADES)
    audit = audit_metrics(metrics, rows)
    history_path = output_dir / "history.json"
    predictions_path = output_dir / "validation_predictions.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "experiment": experiment,
        "experiment_name": spec["name"],
        "description": spec["description"],
        "model_version": spec["name"],
        "artifact_directory": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(best_path),
        "initialization": initialization,
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_config": config,
        "validation_metrics": metrics,
        "audit": audit,
        "history": str(history_path.relative_to(ROOT)).replace("\\", "/"),
        "validation_predictions": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
        "official_test": "RESERVED_NOT_EVALUATED",
        "production_promoted": False,
        "clinical_validation_claim": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "model_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def evaluate_frozen_checkpoint(checkpoint_path: Path, split_path: Path, args, name: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    _, validation_transform = make_transforms(args.input_size)
    dataset = FundusClassificationDataset(split_path, ROOT / "ml" / "datasets" / "raw" / "idrid", "validation", validation_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fundus_batch)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=bool(model_config.get("ordinal_mode", False)))
    result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Frozen checkpoint mismatch: {result}")
    model.eval()
    actual, probabilities, rows = infer_rows(model, loader, torch.device("cpu"), torch, bool(model_config.get("ordinal_mode", False)), "softmax_5_class")
    metrics = classification_metrics(actual, probabilities, referable_grades=REFERABLE_GRADES)
    return {
        "experiment": "baseline",
        "experiment_name": name,
        "model_version": checkpoint.get("model_version", name),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_config": model_config,
        "validation_metrics": metrics,
        "audit": audit_metrics(metrics, rows),
        "training_seconds": 0.0,
        "best_epoch": checkpoint.get("best_epoch"),
        "epochs_completed": checkpoint.get("epoch"),
        "production_promoted": False,
        "official_test": "RESERVED_NOT_EVALUATED",
        "clinical_validation_claim": False,
    }


def compare_results(results: list[dict[str, Any]], metadata_root: Path, split: dict[str, Any]) -> dict[str, Any]:
    baseline = next(result for result in results if result["experiment"] == "baseline")
    rows = []
    for result in results:
        metrics = result["validation_metrics"]
        referable = metrics["referable_dr"]
        per_class = metrics["per_class"]
        rows.append({
            "experiment": result["experiment"],
            "model_version": result["model_version"],
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["precision"],
            "macro_recall": metrics["recall"],
            "macro_f1": metrics["f1"],
            "qwk": metrics["quadratic_weighted_kappa"],
            "roc_auc_ovr": metrics["roc_auc_ovr_macro"],
            "referable_tp": referable["true_positive"],
            "referable_tn": referable["true_negative"],
            "referable_fp": referable["false_positive"],
            "referable_fn": referable["false_negative"],
            "referable_sensitivity": referable["sensitivity"],
            "referable_specificity": referable["specificity"],
            "referable_precision": referable["precision"],
            "referable_f1": referable["f1"],
            "referable_roc_auc": referable["roc_auc"],
            "grade_3_sensitivity": per_class["Severe"]["sensitivity"],
            "grade_4_sensitivity": per_class["Proliferative DR"]["sensitivity"],
            "ece": result["audit"]["calibration"]["ece_10_bins"],
            "training_seconds": result["training_seconds"],
            "best_epoch": result["best_epoch"],
            "official_test": result["official_test"],
        })

    baseline_row = next(row for row in rows if row["experiment"] == "baseline")
    eligible = [row for row in rows if row["referable_sensitivity"] >= 0.90 and row["referable_specificity"] >= 0.85]
    ranked = sorted(eligible, key=lambda row: (row["referable_fn"], -row["qwk"], -row["macro_f1"], -(row["grade_3_sensitivity"] + row["grade_4_sensitivity"]), row["ece"]))
    ranked_candidate = ranked[0] if ranked else baseline_row
    better_than_baseline = (
        ranked_candidate["experiment"] != "baseline"
        and ranked_candidate["referable_fn"] <= baseline_row["referable_fn"]
        and ranked_candidate["qwk"] >= baseline_row["qwk"]
        and ranked_candidate["macro_f1"] >= baseline_row["macro_f1"]
    )
    selected = ranked_candidate if better_than_baseline else baseline_row
    selection = {
        "selected_experiment": selected["experiment"],
        "selected_model_version": selected["model_version"],
        "decision": "KEEP_CURRENT_IDRID_MODEL" if selected["experiment"] == "baseline" else "KEEP_SELECTED_CANDIDATE_EXPERIMENTAL",
        "priority_order": ["leakage_and_defects", "referable_sensitivity", "referable_specificity", "referable_false_negatives", "qwk", "macro_f1", "grade_3_4_sensitivity", "ece", "reproducibility"],
        "eligible_target_candidates": [row["experiment"] for row in eligible],
        "ranked_candidates": [row["experiment"] for row in ranked],
        "rationale": "No candidate was selected over the current checkpoint unless it met referable targets, did not increase referable false negatives, and improved both QWK and macro F1 on the fixed validation split.",
        "production_promoted": False,
        "official_test": "RESERVED_NOT_EVALUATED",
    }
    report = {
        "report_type": "IDRiD controlled research experiment comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "idrid",
        "validation_count": 83,
        "train_count": 323,
        "official_test_set": {"status": "RESERVED_NOT_EVALUATED", "record_count": len(split.get("reserved_official_test_records", []))},
        "fixed_split_manifest": str((ROOT / "ml/datasets/metadata/idrid/dr_training_split.json").relative_to(ROOT)).replace("\\", "/"),
        "experiments": rows,
        "selection": selection,
        "clinical_validation_claim": False,
    }
    metadata_root.mkdir(parents=True, exist_ok=True)
    (metadata_root / "idrid_experiment_registry.json").write_text(json.dumps({"experiments": results, "official_test": "RESERVED_NOT_EVALUATED", "clinical_validation_claim": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (metadata_root / "idrid_experiment_comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (metadata_root / "idrid_selected_candidate.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise SystemExit("batch size, epochs, and patience must be positive")
    split_path = args.split_manifest.resolve()
    split = load_split(split_path)
    selected_experiments = [value.strip().lower() for value in args.experiments.split(",") if value.strip()]
    if not selected_experiments:
        raise SystemExit("At least one experiment must be selected")
    for experiment in selected_experiments:
        experiment_spec(experiment)
    results = [evaluate_frozen_checkpoint(args.current_checkpoint.resolve(), split_path, args, "current_idrid_frozen_baseline")]
    forced_experiments = {value.strip().lower() for value in args.force_experiments.split(",") if value.strip()}
    for experiment in selected_experiments:
        spec = experiment_spec(experiment)
        output_dir = args.output_root.resolve() / spec["name"]
        existing_metrics = output_dir / "metrics.json"
        if args.reuse_existing and experiment not in forced_experiments and existing_metrics.is_file():
            print(f"{spec['name']} reusing_existing_artifact", flush=True)
            results.append(json.loads(existing_metrics.read_text(encoding="utf-8")))
            continue
        results.append(run_training_experiment(experiment, args, split, split_path, output_dir))
    report = compare_results(results, args.metadata_root.resolve(), split)
    selected = report["selection"]["selected_experiment"]
    selected_result = next(result for result in results if result["experiment"] == selected)
    print(json.dumps({
        "status": "RESEARCH_EXPERIMENTS_COMPLETE",
        "selected_experiment": selected,
        "selected_model_version": selected_result["model_version"],
        "selected_checkpoint": selected_result["checkpoint"],
        "selected_checkpoint_sha256": selected_result["checkpoint_sha256"],
        "comparison_report": "ml/datasets/metadata/idrid/idrid_experiment_comparison.json",
        "official_test": "RESERVED_NOT_EVALUATED",
        "production_promoted": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
