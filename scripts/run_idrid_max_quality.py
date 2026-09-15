"""Maximum-quality IDRiD disease-grading research cycle.

This command is deliberately isolated from production and from the frozen
IDRiD V1/V2/V3/final artifacts.  It reads only the governed IDRiD development
records and the labeled APTOS training records used for transfer-learning
initialization.  It never enumerates, opens, or evaluates the reserved IDRiD
official test records or the APTOS competition test images.

The study is intentionally bounded rather than a blind sweep.  It compares
the already measured V3 control with two retinal-field-preserving EfficientNet
experiments (224px and 384px).  The purpose is to test whether removing large
black borders and preserving more retinal detail improves development-only
five-class grading.  Raw softmax values remain uncalibrated unless an
independent calibration set is defensible; this script does not fit one.

Usage:
    python scripts/run_idrid_max_quality.py --audit-only
    python scripts/run_idrid_max_quality.py --epochs 2 --folds 5

No official-test images are opened by this command.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import random
import shutil
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

from app.ml.models.classifier import build_classifier  # noqa: E402
from ml.evaluation.metrics import classification_metrics  # noqa: E402
from ml.training.losses import build_class_weights, build_focal_loss, build_weighted_sampler  # noqa: E402
from ml.training.retinal_preprocessing import RetinalFieldCrop  # noqa: E402
from scripts.train_classifier import make_transforms  # noqa: E402


IDRID_RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
IDRID_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
APTOS_RAW = ROOT / "ml" / "datasets" / "raw" / "aptos2019"
APTOS_SPLIT = ROOT / "ml" / "datasets" / "metadata" / "splits" / "aptos2019" / "splits.json"
APTOS_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
V3_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912" / "v3_b_domain_robust" / "checkpoint_best.pt"
V3_PREDICTIONS = V3_CHECKPOINT.parent / "validation_predictions.json"
V3_CV = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_cv_stability.json"
OLD_OFFICIAL = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_grading_official_test.json"
META_ROOT = ROOT / "ml" / "datasets" / "metadata" / "idrid"
OUTPUT_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "max_quality"
CV_ROOT = OUTPUT_ROOT / "cv"
FINAL_ROOT = OUTPUT_ROOT / "final"

REFERABLE = (2, 3, 4)
THRESHOLD = 0.40
APTOS_SHA = "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b"
V1_SHA = "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
V2_SHA = "2c5bebd6be18184c56443d8d6a8f590d8b31c8904caea6a0c8b31667f9ac9967"
V3_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def seed_everything(seed_value: int, torch: Any) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_idrid_development() -> list[dict[str, Any]]:
    manifest = json.loads(IDRID_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("official_test_used") or manifest.get("official_test_images_opened") != 0:
        raise RuntimeError("IDRiD manifest is not official-test untouched")
    records = manifest.get("records", [])
    if len(records) != 406 or {record.get("split") for record in records} != {"train", "validation"}:
        raise RuntimeError("Expected exactly 406 governed IDRiD development records")
    for record in records:
        if int(record.get("label", -1)) not in range(5):
            raise RuntimeError(f"Invalid IDRiD development label: {record}")
        path = IDRID_RAW / record["image"]
        if not path.is_file():
            raise RuntimeError(f"Missing IDRiD development image: {path}")
    return records


def load_aptos_training() -> list[dict[str, Any]]:
    manifest = json.loads(APTOS_SPLIT.read_text(encoding="utf-8"))
    records = [record for record in manifest.get("records", []) if record.get("split") == "train"]
    if not records:
        raise RuntimeError("No APTOS labeled training records are available")
    for record in records:
        if int(record.get("label", -1)) not in range(5):
            raise RuntimeError(f"Invalid APTOS label: {record}")
        path = APTOS_RAW / record["image"]
        if not path.is_file():
            raise RuntimeError(f"Missing APTOS training image: {path}")
    return records


def _feature_record(path: Path, label: int | None, source: str, image_id: str) -> dict[str, Any]:
    content = path.read_bytes()
    exact = hashlib.sha256(content).hexdigest()
    with Image.open(io.BytesIO(content)) as probe:
        probe.verify()
    with Image.open(io.BytesIO(content)) as image:
        original_width, original_height = image.width, image.height
        image = image.convert("RGB")
        thumb = image.copy()
        thumb.thumbnail((512, 512), Image.Resampling.BILINEAR)
        array = np.asarray(thumb, dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    edges = np.concatenate((gray[0].ravel(), gray[-1].ravel(), gray[:, 0].ravel(), gray[:, -1].ravel()))
    # A transparent, low-resolution engineering proxy.  It is not a clinical
    # quality label and is not used as a target.
    fov = gray > 0.035
    yy, xx = np.where(fov)
    if len(xx):
        fov_area = float((xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1) / gray.size)
    else:
        fov_area = 0.0
    return {
        "source": source,
        "image_id": image_id,
        "path": rel(path),
        "label": label,
        "sha256": exact,
        "width": original_width,
        "height": original_height,
        "channels": 3,
        "aspect_ratio": float(original_width / max(1, original_height)),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "black_border_ratio": float(np.mean(edges < 0.03)),
        "fov_bbox_area_proxy": fov_area,
        "focus_laplacian_proxy": float(np.var(np.diff(gray, n=2, axis=0)) + np.var(np.diff(gray, n=2, axis=1))),
        "perceptual_hash": "".join(f"{byte:02x}" for byte in np.packbits((np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((16, 16), Image.Resampling.BILINEAR)) >= np.mean(gray) * 255).astype(np.uint8)).tolist()),
    }


def audit_development_data(idrid: list[dict[str, Any]], aptos: list[dict[str, Any]]) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for source, root, records in (("idrid_development", IDRID_RAW, idrid), ("aptos_training", APTOS_RAW, aptos)):
        for record in records:
            image_id = record.get("image_id") or Path(record["image"]).stem
            try:
                inventory.append(_feature_record(root / record["image"], int(record["label"]), source, image_id))
            except Exception as exc:
                unreadable.append({"source": source, "image": record["image"], "error": f"{type(exc).__name__}: {exc}"})
    by_sha: dict[str, list[str]] = {}
    by_phash: dict[str, list[str]] = {}
    labels: dict[str, int] = {}
    for item in inventory:
        key = f"{item['source']}:{item['image_id']}"
        by_sha.setdefault(item["sha256"], []).append(key)
        by_phash.setdefault(item["perceptual_hash"], []).append(key)
        if item["label"] is not None:
            labels[key] = int(item["label"])
    exact_groups = [sorted(group) for group in by_sha.values() if len(group) > 1]
    perceptual_groups = [sorted(group) for group in by_phash.values() if len(group) > 1]
    conflicts = [{"records": group, "labels": sorted({labels[item] for item in group if item in labels})} for group in exact_groups if len({labels[item] for item in group if item in labels}) > 1]
    idrid_sha = {item["sha256"] for item in inventory if item["source"] == "idrid_development"}
    aptos_sha = {item["sha256"] for item in inventory if item["source"] == "aptos_training"}
    cross_dataset = sorted(idrid_sha & aptos_sha)
    def summary(source: str) -> dict[str, Any]:
        rows = [item for item in inventory if item["source"] == source]
        values = {key: [float(item[key]) for item in rows] for key in ("width", "height", "aspect_ratio", "brightness", "contrast", "black_border_ratio", "fov_bbox_area_proxy", "focus_laplacian_proxy")}
        return {
            "record_count": len(rows),
            "readable_count": len(rows),
            "class_distribution": dict(Counter(str(item["label"]) for item in rows if item["label"] is not None)),
            "feature_summary": {key: {"mean": float(np.mean(value)), "std": float(np.std(value)), "min": float(np.min(value)), "max": float(np.max(value))} for key, value in values.items()},
            "dimensions": dict(Counter(f"{item['width']}x{item['height']}" for item in rows)),
        }
    payload = {
        "schema_version": "idrid-max-quality-data-audit-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "IDRiD governed development records and APTOS labeled training records only",
        "official_idrid_test_images_opened": 0,
        "aptos_competition_test_images_opened": 0,
        "patient_ids_available": False,
        "patient_level_limitation": "No patient identifiers are supplied; exact/perceptual image grouping is the available leakage control.",
        "idrid_development": summary("idrid_development"),
        "aptos_training": summary("aptos_training"),
        "unreadable": unreadable,
        "exact_duplicate_groups": exact_groups,
        "perceptual_duplicate_groups": perceptual_groups,
        "duplicate_conflicting_label_groups": conflicts,
        "cross_dataset_exact_sha256_matches": cross_dataset,
        "leakage": {
            "status": "PASS" if not unreadable and not conflicts and not cross_dataset else "REVIEW_REQUIRED",
            "controls": ["governed development manifests", "exact SHA-256 groups", "coarse perceptual hash groups", "duplicate-group-aware CV"],
            "patient_level_guarantee": False,
        },
        "no_external_datasets_used": True,
        "no_labels_modified": True,
        "inventory": inventory,
    }
    dump(META_ROOT / "idrid_max_quality_data_audit.json", payload)
    return payload


def max_quality_transforms(size: int, training: bool):
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    common: list[Any] = [RetinalFieldCrop(), transforms.Resize((size, size))]
    if training:
        common += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.12),
        ]
    common += [transforms.ToTensor(), normalize]
    return transforms.Compose(common)


class FundusDataset:
    def __init__(self, records: list[dict[str, Any]], root: Path, transform: Any):
        self.records = records
        self.root = root
        self.transform = transform
        import torch
        self.torch = torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.root / record["image"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.torch.tensor(int(record["label"]), dtype=self.torch.long), record


def collate(batch: list[tuple[Any, ...]]):
    import torch
    images, labels, records = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(labels)), list(records)


def _metrics(actual: list[int], probabilities: list[list[float]]) -> dict[str, Any]:
    base = classification_metrics(actual, probabilities, referable_grades=REFERABLE)
    actual_array = np.asarray(actual, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    referable_actual = np.isin(actual_array, REFERABLE).astype(int)
    referable_probability = probability_array[:, 2:5].sum(axis=1)
    referable_predicted = (referable_probability >= THRESHOLD).astype(int)
    tp = int(((referable_actual == 1) & (referable_predicted == 1)).sum())
    tn = int(((referable_actual == 0) & (referable_predicted == 0)).sum())
    fp = int(((referable_actual == 0) & (referable_predicted == 1)).sum())
    fn = int(((referable_actual == 1) & (referable_predicted == 0)).sum())
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    base["referable_dr"].update({"threshold": THRESHOLD, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "f1": 2 * precision * sensitivity / max(1e-12, precision + sensitivity), "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "false_negative_rate": fn / max(1, tp + fn)})
    base["mean_absolute_grade_error"] = float(np.mean(np.abs(actual_array - probability_array.argmax(axis=1))))
    base["severe_error_count"] = int(sum((a in (0, 1) and p in (3, 4)) or (a in (3, 4) and p in (0, 1)) for a, p in zip(actual_array.tolist(), probability_array.argmax(axis=1).tolist())))
    base["threshold_used_for_research_report"] = THRESHOLD
    return base


def _ece(actual: list[int], probabilities: list[list[float]], bins: int = 10) -> float:
    labels = np.asarray(actual, dtype=int)
    matrix = np.asarray(probabilities, dtype=float)
    confidence = matrix.max(axis=1)
    correct = (matrix.argmax(axis=1) == labels).astype(float)
    total = 0.0
    for low, high in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1.0 else confidence < high)
        if mask.any():
            total += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return total


def build_model_from_aptos(torch: Any, backbone: str = "efficientnet_b0"):
    model = build_classifier(backbone, num_classes=5, pretrained=False, ordinal_mode=False)
    checkpoint = torch.load(APTOS_CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise RuntimeError("APTOS checkpoint has no state_dict")
    if backbone != "efficientnet_b0":
        raise RuntimeError("This bounded cycle uses the verified APTOS EfficientNet-B0 initialization only")
    result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"APTOS initialization mismatch: {result}")
    return model


def infer(model: Any, loader: Any, device: Any, torch: Any) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, labels, records in loader:
            output = model(images.to(device))
            probabilities = torch.softmax(output["severity_logits"], dim=1).cpu().numpy()
            logits = output["severity_logits"].cpu().numpy()
            for index, record in enumerate(records):
                vector = probabilities[index].astype(float).tolist()
                entropy_value = float(-sum(p * math.log(max(p, 1e-12)) for p in vector))
                predicted = int(np.argmax(vector))
                rows.append({
                    "image_id": record.get("image_id") or Path(record["image"]).stem,
                    "image": record["image"],
                    "actual": int(labels[index].item()),
                    "predicted": predicted,
                    "logits": logits[index].astype(float).tolist(),
                    "probabilities": vector,
                    "confidence": float(max(vector)),
                    "referable_probability": float(sum(vector[2:5])),
                    "referable": bool(sum(vector[2:5]) >= THRESHOLD),
                    "uncertainty": {"entropy_nats": entropy_value, "normalized_entropy": entropy_value / math.log(5.0), "probability_margin": float(np.sort(vector)[-1] - np.sort(vector)[-2])},
                })
    return rows


def folds_for(records: list[dict[str, Any]], n_splits: int, seed_value: int):
    from sklearn.model_selection import StratifiedGroupKFold
    labels = np.asarray([int(record["label"]) for record in records], dtype=int)
    groups = np.asarray([record.get("duplicate_group_id") or record.get("record_key") or record["image"] for record in records])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)
    return list(splitter.split(np.zeros(len(records)), labels, groups)), groups


def train_candidate(name: str, records: list[dict[str, Any]], args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    folds, groups = folds_for(records, args.folds, args.seed)
    fold_results: list[dict[str, Any]] = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        cached_fold_dir = CV_ROOT / name / f"fold_{fold_number}"
        cached_metrics_path = cached_fold_dir / "metrics.json"
        cached_predictions_path = cached_fold_dir / "validation_predictions.json"
        if cached_metrics_path.exists() and cached_predictions_path.exists():
            cached = json.loads(cached_metrics_path.read_text(encoding="utf-8"))
            fold_results.append({
                "fold": fold_number,
                "train_count": cached.get("train_count", len(train_indices)),
                "validation_count": cached.get("validation_count", len(validation_indices)),
                "metrics": cached["metrics"],
                "history": cached.get("history", []),
                "checkpoint_sha256": cached.get("checkpoint_sha256"),
                "duplicate_group_overlap": cached.get("duplicate_group_overlap", []),
                "training_seconds": cached.get("training_seconds", 0.0),
            })
            print(f"max_quality candidate={name} fold={fold_number}/{args.folds} reused_cached=true", flush=True)
            continue
        fold_seed = args.seed + fold_number
        seed_everything(fold_seed, torch)
        train_records = [records[index] for index in train_indices]
        validation_records = [records[index] for index in validation_indices]
        size = 384 if "384" in name else 224
        train_dataset = FundusDataset(train_records, IDRID_RAW, max_quality_transforms(size, True))
        validation_dataset = FundusDataset(validation_records, IDRID_RAW, max_quality_transforms(size, False))
        loader_kwargs = {"batch_size": args.batch_size if size == 224 else max(2, args.batch_size // 2), "num_workers": 0, "collate_fn": collate}
        if name.endswith("_sampler"):
            train_sampler = build_weighted_sampler([int(record["label"]) for record in train_records])
            train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        else:
            train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(fold_seed), **loader_kwargs)
        validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
        device = torch.device("cpu")
        model = build_model_from_aptos(torch).to(device)
        class_weights = build_class_weights([int(record["label"]) for record in train_records]).to(device)
        criterion = build_focal_loss(gamma=2.0, alpha=class_weights) if name.endswith("_focal") else torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history: list[dict[str, Any]] = []
        best_qwk = -float("inf")
        best_state: dict[str, Any] | None = None
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: list[float] = []
            for images, labels, _records in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(images.to(device))
                loss = criterion(output["severity_logits"], labels.to(device))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            rows = infer(model, validation_loader, device, torch)
            metrics = _metrics([row["actual"] for row in rows], [row["probabilities"] for row in rows])
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_metrics": metrics})
            qwk = float(metrics.get("quadratic_weighted_kappa") or -1.0)
            if qwk > best_qwk:
                best_qwk = qwk
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError(f"Candidate {name} fold {fold_number} did not produce a checkpoint")
        model.load_state_dict(best_state, strict=True)
        rows = infer(model, validation_loader, device, torch)
        metrics = _metrics([row["actual"] for row in rows], [row["probabilities"] for row in rows])
        fold_dir = CV_ROOT / name / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_dir / "checkpoint_best.pt"
        torch.save({"state_dict": best_state, "model_config": {"backbone": "efficientnet_b0", "num_classes": 5, "input_size": size, "retinal_field_crop": True, "ordinal_mode": False}, "training_config": {"candidate": name, "epochs": args.epochs, "batch_size": loader_kwargs["batch_size"], "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "seed": fold_seed, "loss": "class-weighted cross-entropy", "preprocessing": "RetinalFieldCrop -> Resize -> retinal-safe augmentation -> ImageNet normalization"}, "model_version": f"idrid-max-quality-{name}-fold-{fold_number}", "production_promoted": False, "official_test_images_opened": 0}, checkpoint_path)
        dump(fold_dir / "validation_predictions.json", rows)
        dump(fold_dir / "metrics.json", {"fold": fold_number, "train_count": len(train_records), "validation_count": len(validation_records), "metrics": metrics, "history": history, "checkpoint_sha256": sha256(checkpoint_path), "duplicate_group_overlap": sorted(set(groups[train_indices]) & set(groups[validation_indices])), "official_test_images_opened": 0})
        fold_results.append({"fold": fold_number, "train_count": len(train_records), "validation_count": len(validation_records), "metrics": metrics, "history": history, "checkpoint_sha256": sha256(checkpoint_path), "duplicate_group_overlap": sorted(set(groups[train_indices]) & set(groups[validation_indices])), "training_seconds": time.perf_counter() - started})
        print(f"max_quality candidate={name} fold={fold_number}/{args.folds} qwk={metrics['quadratic_weighted_kappa']:.4f} macro_f1={metrics['f1']:.4f} accuracy={metrics['accuracy']:.4f}", flush=True)
        del model, optimizer, train_loader, validation_loader
        gc.collect()
    metric_names = ["accuracy", "precision", "recall", "f1", "quadratic_weighted_kappa", "roc_auc_ovr_macro", "mean_absolute_grade_error", "severe_error_count"]
    summary: dict[str, Any] = {}
    for metric in metric_names:
        values = np.asarray([float(row["metrics"].get(metric) or 0.0) for row in fold_results], dtype=float)
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "min": float(values.min()), "max": float(values.max()), "values": values.tolist()}
    for metric in ("sensitivity", "specificity", "f1"):
        values = np.asarray([float(row["metrics"]["referable_dr"][metric]) for row in fold_results], dtype=float)
        summary[f"referable_{metric}"] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "min": float(values.min()), "max": float(values.max()), "values": values.tolist()}
    result = {"candidate": name, "architecture": "EfficientNet-B0 initialized from unchanged APTOS production checkpoint", "preprocessing": {"input_size": 384 if "384" in name else 224, "retinal_field_crop": True, "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]}, "loss": "class-balanced focal loss" if name.endswith("_focal") else "class-weighted cross-entropy", "fold_count": args.folds, "folds": fold_results, "summary": summary, "official_test_images_opened": 0, "production_promoted": False}
    dump(OUTPUT_ROOT / f"{name}_cv.json", result)
    return result


def final_train(selected: str, records: list[dict[str, Any]], args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    size = 384 if "384" in selected else 224
    train_dataset = FundusDataset(records, IDRID_RAW, max_quality_transforms(size, True))
    loader_kwargs = {"batch_size": args.batch_size if size == 224 else max(2, args.batch_size // 2), "num_workers": 0, "collate_fn": collate}
    if selected.endswith("_sampler"):
        loader = DataLoader(train_dataset, sampler=build_weighted_sampler([int(record["label"]) for record in records]), **loader_kwargs)
    else:
        loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_kwargs)
    seed_everything(args.seed, torch)
    model = build_model_from_aptos(torch).to(torch.device("cpu"))
    weights = build_class_weights([int(record["label"]) for record in records])
    criterion = build_focal_loss(gamma=2.0, alpha=weights) if selected.endswith("_focal") else torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.final_epochs + 1):
        model.train()
        losses = []
        for images, labels, _records in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output["severity_logits"], labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = FINAL_ROOT / "checkpoint_best.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": {"backbone": "efficientnet_b0", "num_classes": 5, "input_size": size, "retinal_field_crop": True, "ordinal_mode": False}, "training_config": {"candidate": selected, "epochs": args.final_epochs, "batch_size": loader.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "seed": args.seed, "loss": "class-weighted cross-entropy", "preprocessing": "RetinalFieldCrop -> Resize -> retinal-safe augmentation -> ImageNet normalization", "training_data": "IDRiD governed development images only; no official test"}, "model_version": "idrid-max-quality-20260912-v1", "production_promoted": False, "clinical_validation_claim": False, "official_test_images_opened": 0}, checkpoint_path)
    checkpoint_sha = sha256(checkpoint_path)
    (FINAL_ROOT / "checkpoint_best.pt.sha256").write_text(checkpoint_sha + "  checkpoint_best.pt\n", encoding="utf-8")
    config = {"model_version": "idrid-max-quality-20260912-v1", "candidate": selected, "architecture": "EfficientNet-B0", "input_size": size, "retinal_field_crop": True, "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225], "class_mapping": {"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative DR"}, "referable_rule": "P(2)+P(3)+P(4) >= 0.40; severity=argmax(P0..P4)", "calibration": {"status": "UNCALIBRATED", "statement": "Raw softmax confidence is not clinically calibrated; no independent calibration subset was fitted."}, "dataset": "IDRiD governed development records only (406); official test unopened in this cycle", "training_config": {"epochs": args.final_epochs, "batch_size": loader.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "seed": args.seed, "loss": "class-weighted cross-entropy", "history": history}, "checkpoint": rel(checkpoint_path), "checkpoint_sha256": checkpoint_sha, "production_promoted": False, "official_test_images_opened": 0, "freeze_status": "FROZEN_RESEARCH_ONLY"}
    dump(FINAL_ROOT / "training_config.json", config["training_config"])
    dump(FINAL_ROOT / "model_manifest.json", config)
    return config


def error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row["actual"] != row["predicted"]]
    important = [row for row in errors if (row["actual"] in (3, 4) and row["predicted"] in (0, 1)) or (row["actual"] in (0, 1) and row["predicted"] in (3, 4)) or (row["actual"] == 2 and row["predicted"] in (0, 1))]
    high_confidence = sorted((row for row in errors if row["confidence"] >= 0.80), key=lambda row: row["confidence"], reverse=True)
    false_negatives = [row for row in rows if row["actual"] in REFERABLE and not row["referable"]]
    return {"source": "development-only out-of-fold predictions", "error_count": len(errors), "important_error_count": len(important), "important_errors": important, "high_confidence_incorrect": high_confidence, "referable_false_negatives": false_negatives, "severe_grade_error_count": sum(int((row["actual"] in (0, 1) and row["predicted"] in (3, 4)) or (row["actual"] in (3, 4) and row["predicted"] in (0, 1))) for row in errors), "official_test_images_opened": 0}


def pooled_oof(candidate: str, folds: int) -> list[dict[str, Any]]:
    rows = []
    for fold in range(1, folds + 1):
        path = CV_ROOT / candidate / f"fold_{fold}" / "validation_predictions.json"
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def choose_candidate(results: dict[str, Any]) -> str:
    # Development-only selection priority.  A candidate must improve QWK and
    # cannot materially collapse macro-F1 relative to the frozen control.
    control = results["control"]
    candidates = [value for key, value in results.items() if key != "control"]
    eligible = [value for value in candidates if value["summary"]["quadratic_weighted_kappa"]["mean"] >= control["summary"]["qwk"]["mean"] and value["summary"]["f1"]["mean"] >= control["summary"]["macro_f1"]["mean"] * 0.90]
    if not eligible:
        return "control_v3"
    eligible.sort(key=lambda value: (value["summary"]["quadratic_weighted_kappa"]["mean"], value["summary"]["f1"]["mean"], value["summary"]["accuracy"]["mean"]), reverse=True)
    return str(eligible[0]["candidate"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded maximum-quality IDRiD development research")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--experiments", default="b0_crop_224,b0_crop_384,b0_crop_384_focal")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.final_epochs < 1 or args.folds < 2:
        raise SystemExit("epochs, final-epochs and folds must be positive; folds must be at least 2")
    import torch
    torch.set_num_threads(args.torch_threads)
    if sha256(APTOS_CHECKPOINT) != APTOS_SHA:
        raise RuntimeError("APTOS production checkpoint SHA changed; refusing research run")
    records = load_idrid_development()
    aptos_records = load_aptos_training()
    audit = audit_development_data(records, aptos_records)
    if args.audit_only:
        print(json.dumps({"audit": rel(META_ROOT / "idrid_max_quality_data_audit.json"), "idrid_records": len(records), "aptos_training_records": len(aptos_records), "official_test_images_opened": 0}, indent=2), flush=True)
        return 0
    results: dict[str, Any] = {}
    # The frozen V3 control is imported as a development reference only.  It
    # is not copied, altered, reopened, or used as a production artifact.
    if V3_CV.is_file():
        prior = json.loads(V3_CV.read_text(encoding="utf-8"))
        results["control"] = {"candidate": "control_v3", "architecture": "frozen V3 reference", "summary": {"qwk": prior["summary"]["qwk"], "macro_f1": prior["summary"]["macro_f1"], "accuracy": prior["summary"]["accuracy"]}, "checkpoint_sha256": V3_SHA, "official_test_images_opened": 0}
    else:
        raise RuntimeError("Missing existing development-only V3 control report")
    requested_candidates = {item.strip() for item in args.experiments.split(",") if item.strip()}
    for candidate in requested_candidates:
        if candidate not in {"b0_crop_224", "b0_crop_384", "b0_crop_384_focal"}:
            raise SystemExit(f"Unsupported bounded experiment: {candidate}")
        results[candidate] = train_candidate(candidate, records, args, torch)
    # Reuse completed development CV artifacts when this command is invoked to
    # add one bounded candidate later; never retrain or mutate them.
    for candidate in ("b0_crop_224", "b0_crop_384"):
        if candidate not in results:
            prior_path = OUTPUT_ROOT / f"{candidate}_cv.json"
            if prior_path.is_file():
                results[candidate] = json.loads(prior_path.read_text(encoding="utf-8"))
    selection = choose_candidate(results)
    if selection == "control_v3":
        # No new model is frozen when the new preprocessing candidates do not
        # beat the prior development control.  The report remains explicit.
        final_manifest = {"status": "NO_NEW_CANDIDATE_BEAT_FROZEN_V3_CONTROL", "selected_candidate": "control_v3", "production_promoted": False, "official_test_images_opened": 0}
    else:
        existing_manifest = FINAL_ROOT / "model_manifest.json"
        if selection in {"b0_crop_224", "b0_crop_384"} and existing_manifest.is_file():
            final_manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
        else:
            final_manifest = final_train(selection, records, args, torch)
    oof_rows = pooled_oof(selection, args.folds) if selection != "control_v3" else []
    comparison = {"schema_version": "idrid-max-quality-experiment-comparison-1", "generated_at": datetime.now(timezone.utc).isoformat(), "results": results, "selected_candidate": selection, "selection_priority": ["QWK", "macro-F1", "accuracy", "per-class stability", "severe-error count"], "official_test_used_for_selection": False, "official_test_images_opened": 0, "production_promoted": False}
    dump(META_ROOT / "idrid_max_quality_experiment_comparison.json", comparison)
    if oof_rows:
        actual = [row["actual"] for row in oof_rows]
        probs = [row["probabilities"] for row in oof_rows]
        metrics = _metrics(actual, probs)
        dump(META_ROOT / "idrid_max_quality_cv_report.json", {"candidate": selection, "metrics": metrics, "ece": _ece(actual, probs), "row_count": len(oof_rows), "out_of_fold": True, "fold_count": args.folds, "official_test_images_opened": 0, "duplicate_group_overlap_any_fold": any(row["duplicate_group_overlap"] for row in results[selection]["folds"]), "calibration_status": "UNCALIBRATED"})
        dump(META_ROOT / "idrid_max_quality_error_analysis.json", error_analysis(oof_rows))
        dump(META_ROOT / "idrid_max_quality_calibration.json", {"status": "UNCALIBRATED", "ece_from_out_of_fold_predictions": _ece(actual, probs), "calibration_fit": False, "reason": "No independent calibration subset was fitted; carving one from 406 development images would weaken the already small model-selection sample.", "statement": "Raw softmax confidence is not clinically calibrated; calibration could not be reliably fitted with the available non-test data.", "official_test_images_opened": 0})
    else:
        dump(META_ROOT / "idrid_max_quality_cv_report.json", {"candidate": selection, "status": "NO_NEW_CANDIDATE_SELECTED", "control_report": rel(V3_CV), "official_test_images_opened": 0})
        dump(META_ROOT / "idrid_max_quality_error_analysis.json", {"status": "NOT_RECOMPUTED_FOR_CONTROL", "source": "frozen V3 artifacts", "official_test_images_opened": 0})
        dump(META_ROOT / "idrid_max_quality_calibration.json", {"status": "UNCALIBRATED", "calibration_fit": False, "official_test_images_opened": 0, "statement": "Raw softmax confidence is not clinically calibrated; calibration could not be reliably fitted with the available non-test data."})
    prior_official = json.loads(OLD_OFFICIAL.read_text(encoding="utf-8")) if OLD_OFFICIAL.is_file() else None
    final_report = {"schema_version": "idrid-max-quality-final-report-1", "generated_at": datetime.now(timezone.utc).isoformat(), "best_model": final_manifest, "selection": selection, "development_only_selection": True, "official_test": {"status": "IMMUTABLE_PRIOR_RESULT_NOT_REOPENED_IN_THIS_CYCLE", "report": rel(OLD_OFFICIAL) if OLD_OFFICIAL.is_file() else None, "metrics": prior_official.get("metrics") if prior_official else None, "official_test_images_opened_in_current_cycle": 0}, "production_promoted": False, "known_limitations": ["No patient IDs were available; duplicate-aware grouping is not a patient-level guarantee.", "IDRiD is small and Grade 1 is sparse.", "Raw softmax confidence is not clinically calibrated.", "No external data was used for training or selection."], "official_test_images_opened": 0}
    dump(META_ROOT / "idrid_max_quality_final_report.json", final_report)
    # A registry entry is research metadata only; production configuration is
    # not changed and no frozen prior artifact is overwritten.
    if final_manifest.get("checkpoint"):
        registry_path = ROOT / "ml" / "weights" / "model_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"artifacts": []}
        registry["artifacts"] = [item for item in registry.get("artifacts", []) if item.get("model_version") != final_manifest.get("model_version")]
        registry["artifacts"].append({"model_version": final_manifest["model_version"], "dataset_version": "idrid-development-20260912", "artifact_kind": "RESEARCH_MODEL", "artifact_status": "FROZEN_RESEARCH_ONLY", "checkpoint": final_manifest["checkpoint"], "checkpoint_sha256": final_manifest["checkpoint_sha256"], "model_config": {"architecture": final_manifest["architecture"], "input_size": final_manifest["input_size"]}, "validation_metrics_artifact": rel(META_ROOT / "idrid_max_quality_cv_report.json"), "production_promoted": False, "clinical_validation_claim": False})
        registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_candidate": selection, "final_manifest": final_manifest, "comparison": rel(META_ROOT / "idrid_max_quality_experiment_comparison.json"), "official_test_images_opened": 0, "production_promoted": False}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
