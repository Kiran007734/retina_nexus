"""Train a reproducible, configurable hierarchical DR classifier.

This command consumes a governance split manifest. It does not acquire data,
create labels, or claim clinical performance targets. Install the optional ML
dependencies before invoking it:

    pip install -r backend/requirements-ml.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.classifier import (  # noqa: E402
    SUPPORTED_BACKBONES,
    ReferableDRMapping,
    build_classifier,
    severity_probabilities,
)
from ml.evaluation.metrics import classification_metrics  # noqa: E402
from ml.training.classification_dataset import FundusClassificationDataset  # noqa: E402
from ml.training.losses import (  # noqa: E402
    build_class_weights,
    build_ordinal_loss,
    build_severity_loss,
    build_weighted_sampler,
    hierarchical_loss,
)
from scripts.dataset_common import (  # noqa: E402
    DatasetError,
    get_definition,
    raw_path_for,
    split_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the RETINA-NEXUS hierarchical DR classifier")
    parser.add_argument("--dataset", required=True, help="Dataset registry slug, for example aptos2019")
    parser.add_argument("--manifest", help="Governance split manifest; defaults to the dataset split manifest")
    parser.add_argument("--raw-dir", help="Dataset raw directory; defaults to the registry path")
    parser.add_argument("--dataset-version", default="unspecified", help="Immutable dataset/version label recorded in the artifact")
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default="efficientnet_b0")
    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument("--pretrained", dest="pretrained", action="store_true", help="Load official torchvision weights (default)")
    pretrained.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Start the backbone without pretrained weights")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--ordinal-mode", action="store_true", help="Use cumulative ordinal thresholds for the fine-grade head")
    parser.add_argument("--loss-strategy", choices=["plain", "weighted_loss", "focal_loss", "weighted_sampler"], default="weighted_loss")
    parser.add_argument("--referable-min-grade", type=int, choices=range(1, 5), default=2, help="Configurable lower grade included in referable DR")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping epochs without validation F1 improvement")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true", help="Use CUDA mixed precision when supported")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-version", help="Version label; defaults to backbone-seed")
    parser.add_argument("--output-dir", help="Artifact directory; defaults to ml/weights/classifiers/<dataset>/<version>")
    return parser


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_manifest(manifest_path: Path, raw_root: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise DatasetError(f"Split manifest does not exist: {manifest_path}. Run scripts/create_splits.py first.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not read split manifest {manifest_path}: {exc}") from exc
    records = payload.get("records", [])
    if not records:
        raise DatasetError(f"Split manifest contains no records: {manifest_path}")
    if payload.get("leakage", {}).get("status") != "pass":
        raise DatasetError("Split manifest leakage status is not pass; refusing to train across an unsafe split.")
    group_splits: dict[str, set[str]] = {}
    duplicate_splits: dict[str, set[str]] = {}
    for record in records:
        split = record.get("split")
        if split not in {"train", "validation", "test"}:
            raise DatasetError(f"Invalid split value in manifest: {split!r}")
        for key, target in (("group_id", group_splits), ("duplicate_group_id", duplicate_splits)):
            value = record.get(key)
            if value:
                target.setdefault(str(value), set()).add(split)
        image = record.get("image")
        if not image:
            raise DatasetError("A split record has no image path")
        path = (raw_root / image).resolve()
        try:
            path.relative_to(raw_root.resolve())
        except ValueError as exc:
            raise DatasetError(f"Manifest image escapes raw directory: {image}") from exc
        if not path.is_file():
            raise DatasetError(f"Manifest image is missing: {path}")
        try:
            label = int(record.get("label"))
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Missing or non-integer label for {image}") from exc
        if label not in range(5):
            raise DatasetError(f"DR label for {image} must be in 0..4, got {label}")
    crossed_groups = [key for key, splits in group_splits.items() if len(splits) > 1]
    crossed_duplicates = [key for key, splits in duplicate_splits.items() if len(splits) > 1]
    if crossed_groups or crossed_duplicates:
        raise DatasetError(f"Split leakage detected: groups={crossed_groups[:3]}, duplicates={crossed_duplicates[:3]}")
    for required in ("train", "validation"):
        if not any(record.get("split") == required and record.get("label") not in (None, "") for record in records):
            raise DatasetError(f"No labeled {required} records are available in {manifest_path}")
    return payload


def make_transforms(input_size: int):
    try:
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("torchvision transforms are unavailable. Install backend/requirements-ml.txt and verify its binary dependencies.") from exc
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize,
    ])
    return train_transform, validation_transform


def select_device(torch, requested: str):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or install a CUDA-enabled PyTorch build.")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def run_epoch(model, loader, optimizer, criterion, mapping, ordinal_mode, device, torch, scaler=None, amp_enabled=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    component_totals = {"stage1": 0.0, "stage2": 0.0, "severity": 0.0}
    actual: list[int] = []
    probabilities: list[list[float]] = []
    for images, labels, _records in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss, components = hierarchical_loss(outputs, labels, mapping, criterion, ordinal_mode=ordinal_mode)
        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(labels)
        for key in component_totals:
            component_totals[key] += components[key] * len(labels)
        with torch.inference_mode():
            batch_probabilities = severity_probabilities(outputs, ordinal_mode).detach().cpu().tolist()
        actual.extend(labels.detach().cpu().tolist())
        probabilities.extend(batch_probabilities)
    count = max(1, len(actual))
    metrics = classification_metrics(actual, probabilities, referable_grades=mapping.referable_grades)
    metrics["loss"] = total_loss / count
    metrics["loss_components"] = {key: value / count for key, value in component_totals.items()}
    return metrics


def checkpoint_payload(model, args, metrics, dataset_version: str, model_version: str, epoch: int, best_epoch: int):
    mapping = ReferableDRMapping(
        name=f"grade_{args.referable_min_grade}_or_worse",
        referable_grades=tuple(range(args.referable_min_grade, 5)),
    )
    return {
        "state_dict": model.state_dict(),
        "model_config": {
            "backbone": args.backbone, "num_classes": 5,
            "input_size": args.input_size, "ordinal_mode": args.ordinal_mode,
        },
        "training_config": vars(args),
        "metrics": metrics,
        "dataset_version": dataset_version,
        "model_version": model_version,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "artifact": {
            "model_name": "RETINA-NEXUS DR classifier",
            "model_version": model_version,
            "backbone": args.backbone,
            "referable_mapping": mapping.to_dict(),
            "clinical_validation_claim": False,
        },
    }


def write_artifact_registry(output_dir: Path, model_version: str, dataset_version: str, metrics: dict[str, Any], checkpoint: Path) -> None:
    registry_path = ROOT / "ml" / "weights" / "model_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry: dict[str, Any] = {"artifacts": []}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registry = {"artifacts": []}
    items = [item for item in registry.get("artifacts", []) if item.get("model_version") != model_version]
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    items.append({
        "model_version": model_version, "dataset_version": dataset_version,
        "artifact_kind": "FINE_TUNED_MODEL", "artifact_status": "MODEL_TRAINED",
        "availability_status": "MODEL_AVAILABLE" if checkpoint.is_file() else "MODEL_MISSING",
        "artifact_directory": display_path(output_dir),
        "checkpoint": display_path(checkpoint),
        "validation_metrics": metrics,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "clinical_validation_claim": False,
    })
    registry_path.write_text(json.dumps({"artifacts": items}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.input_size < 32:
        print("TRAINING SETUP ERROR: batch size and epochs must be positive; input size must be at least 32.", file=sys.stderr)
        return 2
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        print("TRAINING SETUP ERROR: PyTorch/torchvision could not be imported. Install backend/requirements-ml.txt; no weights were downloaded or assumed.", file=sys.stderr)
        print(f"Original error: {exc}", file=sys.stderr)
        return 2
    try:
        definition = get_definition(args.dataset)
        raw_root = raw_path_for(definition, args.raw_dir)
        manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else split_directory(args.dataset) / "splits.json"
        payload = load_manifest(manifest_path, raw_root)
        train_transform, validation_transform = make_transforms(args.input_size)
        train_dataset = FundusClassificationDataset(manifest_path, raw_root, "train", train_transform)
        validation_dataset = FundusClassificationDataset(manifest_path, raw_root, "validation", validation_transform)
        test_dataset = None
        if any(record.get("split") == "test" and record.get("label") not in (None, "") for record in payload.get("records", [])):
            test_dataset = FundusClassificationDataset(manifest_path, raw_root, "test", validation_transform)
        train_labels = [int(record["label"]) for record in train_dataset.records]
        sampler = build_weighted_sampler(train_labels) if args.loss_strategy == "weighted_sampler" else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if test_dataset else None
        device = select_device(torch, args.device)
        seed_everything(args.seed, torch)
        class_weights = build_class_weights(train_labels).to(device)
        criterion = build_ordinal_loss(args.loss_strategy, train_labels) if args.ordinal_mode else build_severity_loss(args.loss_strategy, class_weights)
        model = build_classifier(args.backbone, num_classes=5, pretrained=args.pretrained, ordinal_mode=args.ordinal_mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")
        amp_enabled = bool(args.mixed_precision and device.type == "cuda")
        mapping = ReferableDRMapping(name=f"grade_{args.referable_min_grade}_or_worse", referable_grades=tuple(range(args.referable_min_grade, 5)))
        model_version = args.model_version or f"{args.backbone}-seed{args.seed}"
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / "ml" / "weights" / "classifiers" / args.dataset / model_version
        output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        best_score = float("-inf")
        best_epoch = 0
        patience_count = 0
        best_path = output_dir / "checkpoint_best.pt"
        last_path = output_dir / "checkpoint_last.pt"
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, criterion, mapping, args.ordinal_mode, device, torch, scaler, amp_enabled)
            with torch.no_grad():
                validation_metrics = run_epoch(model, validation_loader, None, criterion, mapping, args.ordinal_mode, device, torch, None, False)
            history_item = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            history.append(history_item)
            last_payload = checkpoint_payload(model, args, validation_metrics, args.dataset_version, model_version, epoch, best_epoch)
            torch.save(last_payload, last_path)
            score = float(validation_metrics.get("f1", 0.0))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(checkpoint_payload(model, args, validation_metrics, args.dataset_version, model_version, epoch, best_epoch), best_path)
                patience_count = 0
            else:
                patience_count += 1
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} validation_f1={validation_metrics['f1']:.4f} validation_accuracy={validation_metrics['accuracy']:.4f}")
            if patience_count >= args.patience:
                print(f"Early stopping after epoch {epoch}; best validation epoch={best_epoch}.")
                break
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        with torch.no_grad():
            final_validation = run_epoch(model, validation_loader, None, criterion, mapping, args.ordinal_mode, device, torch, None, False)
            final_test = run_epoch(model, test_loader, None, criterion, mapping, args.ordinal_mode, device, torch, None, False) if test_loader else None
        config = {**vars(args), "dataset": args.dataset, "manifest": str(manifest_path), "raw_dir": str(raw_root), "device_used": str(device), "amp_enabled": amp_enabled, "referable_mapping": mapping.to_dict(), "clinical_validation_claim": False}
        metrics = {"validation": final_validation, "test": final_test, "best_epoch": best_epoch, "note": "Measured local split metrics only; no clinical performance claim."}
        (output_dir / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        manifest = {"model_version": model_version, "dataset": args.dataset, "dataset_version": args.dataset_version, "checkpoint": best_path.name, "checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(), "model_config": checkpoint["model_config"], "training_config": config, "metrics": metrics, "clinical_validation_claim": False}
        (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        write_artifact_registry(output_dir, model_version, args.dataset_version, final_validation, best_path)
        print(f"Training complete. Best artifact: {best_path}")
        print("Metrics are measured engineering results, not a clinical validation claim.")
        return 0
    except (DatasetError, ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"TRAINING SETUP ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
