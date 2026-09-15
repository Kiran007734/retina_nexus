"""Train the experimental IDRiD-only DR severity classifier.

This command is intentionally separate from scripts/train_classifier.py. It
uses the IDRiD clean manifest, can initialize from the existing APTOS
EfficientNet-B0 checkpoint without modifying it, never loads official IDRiD
testing records, and writes artifacts only under the IDRiD model directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

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
from ml.training.losses import build_class_weights, hierarchical_loss  # noqa: E402
from scripts.train_classifier import make_transforms, seed_everything, select_device  # noqa: E402


DEFAULT_SPLIT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"
DEFAULT_APTOS = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
DEFAULT_VERSION = "efficientnet-b0-idrid-20260912-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the experimental IDRiD EfficientNet-B0 DR classifier")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--init-source", choices=("aptos_checkpoint", "imagenet"), default="aptos_checkpoint")
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_APTOS)
    parser.add_argument("--model-version", default=DEFAULT_VERSION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-report", type=Path, default=ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_training_evaluation.json")
    parser.add_argument("--comparison-report", type=Path, default=ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_aptos_comparison.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "torchvision", "numpy", "scikit-learn", "Pillow"):
        try:
            result[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            result[package] = "UNAVAILABLE"
    return result


def load_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "idrid":
        raise RuntimeError("The split manifest is not an IDRiD manifest")
    if payload.get("leakage", {}).get("status") != "pass":
        raise RuntimeError("The IDRiD split manifest does not have leakage status pass")
    if payload.get("leakage", {}).get("official_test_in_training_manifest") is True:
        raise RuntimeError("The official IDRiD test set is present in the training manifest")
    records = payload.get("records", [])
    if not records or any(record.get("split") not in {"train", "validation"} for record in records):
        raise RuntimeError("The IDRiD split manifest must contain only train and validation records")
    if any(record.get("official_split") != "train" for record in records):
        raise RuntimeError("A non-training official record was found in the IDRiD training manifest")
    if any(record.get("label") not in range(5) for record in records):
        raise RuntimeError("IDRiD labels must be integers in the range 0..4")
    return payload


def load_initialization(model, args: argparse.Namespace, torch) -> dict[str, Any]:
    if args.init_source == "imagenet":
        return {
            "source": "torchvision ImageNet EfficientNet-B0 weights",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "model_version": None,
            "state_dict_loaded": False,
        }
    checkpoint_path = args.init_checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Initialization checkpoint does not exist: {checkpoint_path}")
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    if config.get("backbone") != "efficientnet_b0" or config.get("num_classes") != 5 or config.get("input_size") != args.input_size or config.get("ordinal_mode") is not False:
        raise RuntimeError(f"Initialization checkpoint is incompatible with IDRiD EfficientNet-B0 configuration: {config}")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Initialization checkpoint has no state_dict")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Initialization checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "source": "local APTOS EfficientNet-B0 checkpoint used as initialization only",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash,
        "model_version": checkpoint.get("model_version"),
        "dataset": checkpoint.get("dataset"),
        "dataset_version": checkpoint.get("dataset_version"),
        "state_dict_loaded": True,
    }


def collate_fundus_batch(batch):
    """Stack tensors while preserving optional governance metadata as a list."""
    import torch

    images, labels, records = zip(*batch)
    return torch.stack(list(images), dim=0), torch.stack(list(labels), dim=0), list(records)


def run_epoch(model, loader, optimizer, criterion, mapping, device, torch, amp_enabled: bool) -> dict[str, Any]:
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
            outputs = model(images)
            loss, batch_components = hierarchical_loss(outputs, labels, mapping, criterion, ordinal_mode=False)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = len(labels)
        total_loss += float(loss.detach().cpu()) * batch_size
        for key in components:
            components[key] += batch_components[key] * batch_size
        with torch.inference_mode():
            probabilities.extend(severity_probabilities(outputs, False).detach().cpu().tolist())
        actual.extend(labels.detach().cpu().tolist())
    count = max(1, len(actual))
    result = classification_metrics(actual, probabilities, referable_grades=(2, 3, 4))
    result["loss"] = total_loss / count
    result["loss_components"] = {key: value / count for key, value in components.items()}
    return result


def checkpoint_payload(model, config: dict[str, Any], metrics: dict[str, Any], epoch: int, best_epoch: int) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "model_config": {
            "backbone": "efficientnet_b0",
            "num_classes": 5,
            "input_size": config["input_size"],
            "ordinal_mode": False,
        },
        "training_config": config,
        "metrics": metrics,
        "dataset": "idrid",
        "dataset_version": config["dataset_version"],
        "model_version": config["model_version"],
        "epoch": epoch,
        "best_epoch": best_epoch,
        "artifact": {
            "model_name": "RETINA-NEXUS experimental IDRiD DR classifier",
            "model_version": config["model_version"],
            "backbone": "efficientnet_b0",
            "referable_mapping": {"name": "grade_2_or_worse", "referable_grades": [2, 3, 4]},
            "clinical_validation_claim": False,
            "production_promoted": False,
        },
    }


def evaluate_checkpoint(model, loader, device, torch) -> dict[str, Any]:
    model.eval()
    actual: list[int] = []
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for images, labels, _records in loader:
            outputs = model(images.to(device, non_blocking=device.type == "cuda"))
            probabilities.extend(severity_probabilities(outputs, False).cpu().tolist())
            actual.extend(labels.tolist())
    return classification_metrics(actual, probabilities, referable_grades=(2, 3, 4))


def update_experimental_registry(output_dir: Path, model_manifest: dict[str, Any]) -> None:
    registry_path = ROOT / "ml" / "weights" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"artifacts": []}
    artifacts = [item for item in registry.get("artifacts", []) if item.get("model_version") != model_manifest["model_version"]]
    artifacts.append({
        "model_version": model_manifest["model_version"],
        "dataset": "idrid",
        "dataset_version": model_manifest["dataset_version"],
        "artifact_kind": "EXPERIMENTAL_FINE_TUNED_MODEL",
        "artifact_status": "MODEL_TRAINED",
        "availability_status": "MODEL_AVAILABLE",
        "production_promoted": False,
        "artifact_directory": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint": str((output_dir / model_manifest["checkpoint"]).relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": model_manifest["checkpoint_sha256"],
        "model_config": model_manifest["model_config"],
        "training_config": model_manifest["training_config"],
        "validation_metrics": model_manifest["metrics"],
        "class_mapping": model_manifest["class_mapping"],
        "clinical_validation_claim": False,
    })
    registry_path.write_text(json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise SystemExit("batch size, epochs, and patience must be positive")
    split_path = args.split_manifest.resolve()
    split = load_split(split_path)
    output_dir = (args.output_dir or ROOT / "ml" / "weights" / "classifiers" / "idrid" / args.model_version).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = args.evaluation_report.resolve()
    comparison_path = args.comparison_report.resolve()

    import torch
    from torch.utils.data import DataLoader

    seed_everything(args.seed, torch)
    device = select_device(torch, args.device)
    train_transform, validation_transform = make_transforms(args.input_size)
    raw_root = ROOT / "ml" / "datasets" / "raw" / "idrid"
    train_dataset = FundusClassificationDataset(split_path, raw_root, "train", train_transform)
    validation_dataset = FundusClassificationDataset(split_path, raw_root, "validation", validation_transform)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fundus_batch)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fundus_batch)

    model = build_classifier("efficientnet_b0", num_classes=5, pretrained=args.init_source == "imagenet", ordinal_mode=False).to(device)
    initialization = load_initialization(model, args, torch)
    initialization_hash_before = initialization.get("checkpoint_sha256")
    train_labels = [int(record["label"]) for record in train_dataset.records]
    class_weights = build_class_weights(train_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    mapping = ReferableDRMapping(name="grade_2_or_worse", referable_grades=(2, 3, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7)
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    dataset_version = f"idrid-grading-{split['generated_from']['grading_manifest_sha256'][:16]}-split-{split['seed']}"
    config = {
        "dataset": "idrid",
        "dataset_version": dataset_version,
        "dataset_source": "ml/datasets/raw/idrid/B. Disease Grading/",
        "split_manifest": str(split_path.relative_to(ROOT)).replace("\\", "/"),
        "split_manifest_sha256": sha256(split_path),
        "model_version": args.model_version,
        "architecture": "EfficientNet-B0 with RETINA-NEXUS hierarchical heads; severity head used for 5-class output",
        "backbone": "efficientnet_b0",
        "initialization": initialization,
        "input_size": args.input_size,
        "input_channels": 3,
        "color_space": "RGB",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "augmentation": ["Resize(224,224)", "RandomHorizontalFlip(p=0.5)", "RandomRotation(8 degrees)", "RandomAffine(translate=0.03, scale=0.95..1.05)", "ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)"],
        "validation_transform": ["Resize(224,224)", "ToTensor", "ImageNet normalization"],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=1e-7)",
        "epochs_requested": args.epochs,
        "early_stopping": {"metric": "validation_macro_f1", "patience": args.patience},
        "loss": "hierarchical weighted cross-entropy; severity weight=1.0, stage1 weight=0.25, stage2 weight=0.25",
        "class_weights_from_train_subset_only": [float(value) for value in class_weights.detach().cpu().tolist()],
        "class_weight_counts_from_train_subset": dict(sorted(Counter(train_labels).items())),
        "class_mapping": {"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative DR"},
        "referable_mapping": mapping.to_dict(),
        "seed": args.seed,
        "device_requested": args.device,
        "device_used": str(device),
        "mixed_precision_requested": args.mixed_precision,
        "amp_enabled": amp_enabled,
        "fine_tune_all_layers": True,
        "official_test_set": {"status": "RESERVED_NOT_EVALUATED", "record_count": len(split["reserved_official_test_records"])},
        "excluded_records": split["excluded_records"],
        "train_record_count": len(train_dataset),
        "validation_record_count": len(validation_dataset),
        "software_versions": versions(),
        "clinical_validation_claim": False,
        "production_promoted": False,
    }
    (output_dir / "class_mapping.json").write_text(json.dumps(config["class_mapping"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps({"training_config": config, "official_test_set": config["official_test_set"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_score = float("-inf")
    best_epoch = 0
    patience_count = 0
    history: list[dict[str, Any]] = []
    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, mapping, device, torch, amp_enabled)
        validation_metrics = run_epoch(model, validation_loader, None, criterion, mapping, device, torch, False)
        scheduler.step(validation_metrics["f1"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "learning_rate": learning_rate, "train": train_metrics, "validation": validation_metrics})
        last_payload = checkpoint_payload(model, config, validation_metrics, epoch, best_epoch)
        torch.save(last_payload, last_path)
        score = float(validation_metrics["f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(checkpoint_payload(model, config, validation_metrics, epoch, best_epoch), best_path)
            patience_count = 0
        else:
            patience_count += 1
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} validation_f1={validation_metrics['f1']:.4f} validation_accuracy={validation_metrics['accuracy']:.4f} lr={learning_rate:.8f}", flush=True)
        if patience_count >= args.patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("Training ended without a best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    model.to(device)
    final_idrid_metrics = evaluate_checkpoint(model, validation_loader, device, torch)
    aptos_hash_after = sha256(args.init_checkpoint.resolve()) if args.init_source == "aptos_checkpoint" else None
    if initialization_hash_before and aptos_hash_after != initialization_hash_before:
        raise RuntimeError("The APTOS initialization checkpoint SHA-256 changed during IDRiD training")

    # Zero-shot comparison on the same IDRiD validation records only.
    aptos_model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False).to(device)
    aptos_checkpoint = torch.load(args.init_checkpoint.resolve(), map_location="cpu", weights_only=False)
    aptos_model.load_state_dict(aptos_checkpoint["state_dict"], strict=True)
    aptos_metrics = evaluate_checkpoint(aptos_model, validation_loader, device, torch)
    comparison = {
        "dataset": "idrid",
        "evaluation_population": "same IDRiD validation split for both models",
        "validation_record_count": len(validation_dataset),
        "idrid_model": {"model_version": args.model_version, "metrics": final_idrid_metrics},
        "aptos_model_zero_shot": {"model_version": aptos_checkpoint.get("model_version"), "checkpoint_sha256": initialization_hash_before, "metrics": aptos_metrics},
        "metric_selection_or_tuning": False,
        "official_test_set_used": False,
        "clinical_validation_claim": False,
        "note": "Engineering comparison only; the APTOS model was not tuned, changed, or promoted.",
    }
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checkpoint_hash = sha256(best_path)
    metrics = {
        "validation": final_idrid_metrics,
        "official_test": {"status": "RESERVED_NOT_EVALUATED", "record_count": len(split["reserved_official_test_records"])},
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "clinical_validation_claim": False,
        "note": "IDRiD internal validation / engineering evaluation only; no clinical validation claim.",
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_manifest = {
        "model_version": args.model_version,
        "dataset": "idrid",
        "dataset_version": dataset_version,
        "checkpoint": "checkpoint_best.pt",
        "checkpoint_sha256": checkpoint_hash,
        "initialization_checkpoint_sha256": initialization_hash_before,
        "model_config": best_checkpoint["model_config"],
        "training_config": config,
        "metrics": metrics,
        "class_mapping": config["class_mapping"],
        "comparison_report": str(comparison_path.relative_to(ROOT)).replace("\\", "/"),
        "clinical_validation_claim": False,
        "production_promoted": False,
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_experimental_registry(output_dir, model_manifest)
    evaluation_report = {
        "report_type": "IDRiD internal validation / engineering evaluation",
        "dataset": "idrid",
        "model_manifest": str((output_dir / "model_manifest.json").relative_to(ROOT)).replace("\\", "/"),
        "training_history": str((output_dir / "history.json").relative_to(ROOT)).replace("\\", "/"),
        "metrics": metrics,
        "comparison": comparison,
        "excluded_records": split["excluded_records"],
        "official_test_set": config["official_test_set"],
        "clinical_validation_claim": False,
    }
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "TRAINING_COMPLETE",
        "model_version": args.model_version,
        "checkpoint": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash,
        "initialization_sha256": initialization_hash_before,
        "train_count": len(train_dataset),
        "validation_count": len(validation_dataset),
        "best_epoch": best_epoch,
        "validation_metrics": final_idrid_metrics,
        "aptos_zero_shot_metrics": aptos_metrics,
        "official_test": "RESERVED_NOT_EVALUATED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
