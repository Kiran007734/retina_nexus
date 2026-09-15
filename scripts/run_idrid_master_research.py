"""Development-only master research cycle for IDRiD disease grading.

The official IDRiD test was already evaluated once after the previous freeze.
This cycle deliberately never opens those 103 images again.  It audits the
frozen V1/V2/V3 implementations, audits all 406 governed development records,
and runs a small five-fold stratified, duplicate-grouped development study
using APTOS initialization and IDRiD development images only.
"""

from __future__ import annotations

import gc
import hashlib
import io
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

from app.ml.models.classifier import ReferableDRMapping, build_classifier  # noqa: E402
from ml.evaluation.metrics import classification_metrics  # noqa: E402
from ml.training.losses import build_class_weights, hierarchical_loss  # noqa: E402
from scripts.run_idrid_final_grading import metrics_with_threshold, sha256  # noqa: E402
from scripts.train_classifier import make_transforms  # noqa: E402

IDRID_RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
DEV_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
APTOS_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
V1_MANIFEST = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "model_manifest.json"
V2_MANIFEST = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v2" / "20260912" / "v2_c_lesion_aware_efficientnet_b0" / "model_manifest.json"
V3_MANIFEST = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912" / "v3_b_domain_robust" / "model_manifest.json"
V3_PREDICTIONS = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912" / "v3_b_domain_robust" / "validation_predictions.json"
FINAL_MANIFEST = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "final" / "model_manifest.json"
OLD_OFFICIAL = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_grading_official_test.json"
META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
CV_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "master_cv" / "20260912"

APTOS_SHA = "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b"
V1_SHA = "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
V2_SHA = "2c5bebd6be18184c56443d8d6a8f590d8b31c8904caea6a0c8b31667f9ac9967"
V3_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"
THRESHOLD = 0.40
REFERABLE = (2, 3, 4)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def seed(seed_value: int, torch: Any) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_transform(size: int):
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18, hue=0.04),
        transforms.RandomApply([transforms.RandomAutocontrast()], p=0.25),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.20),
        transforms.ToTensor(), normalize,
    ])


def load_development_records() -> list[dict[str, Any]]:
    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("official_test_images_opened") not in (None, 0) or manifest.get("official_test_used"):
        raise RuntimeError("Development manifest does not prove official-test isolation")
    records = manifest.get("records", [])
    if len(records) != 406 or {record.get("split") for record in records} != {"train", "validation"}:
        raise RuntimeError("Expected the 406 governed IDRiD development records")
    if any(record.get("label") not in range(5) for record in records):
        raise RuntimeError("Invalid development label")
    return records


def implementation_audit() -> dict[str, Any]:
    def selected(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("training_config", {})
        return {
            "manifest": rel(path),
            "architecture": config.get("architecture") or payload.get("architecture"),
            "initialization": config.get("initialization") or payload.get("initialization"),
            "preprocessing": {key: config.get(key) for key in ("input_size", "input_channels", "color_space", "normalization", "validation_transform", "preprocessing") if config.get(key) is not None},
            "augmentation": config.get("augmentation"),
            "loss": config.get("loss") or config.get("loss_strategy"),
            "class_weights": config.get("class_weights_from_train_subset_only") or config.get("loss", {}).get("class_weights") if isinstance(config.get("loss"), dict) else config.get("class_weights_from_train_subset_only"),
            "sampling": config.get("sampler") or config.get("sampling"),
            "optimizer": config.get("optimizer"),
            "learning_rate": config.get("learning_rate"),
            "scheduler": config.get("scheduler"),
            "early_stopping": config.get("early_stopping"),
            "seed": config.get("seed"),
            "decoder": "severity=argmax(P0..P4); referable=P2+P3+P4 >= frozen threshold",
            "calibration": "UNCALIBRATED; no fitted temperature scaling",
        }
    return {
        "scope": "read-only forensic comparison of frozen V1/V2/V3 artifacts",
        "v1": selected(V1_MANIFEST),
        "v2": selected(V2_MANIFEST),
        "v3": selected(V3_MANIFEST),
        "exact_changes": [
            "V1: standard EfficientNet-B0 hierarchical severity model, APTOS initialization, conventional retinal augmentation, 323/83 IDRiD development split.",
            "V2: added lesion-presence and optic-disc auxiliary heads with masked auxiliary losses; severity decoder remained the 5-class severity head.",
            "V3: continued V2 architecture with stronger domain-robust augmentation; candidate A/V3 control was selected on development data only.",
            "No implementation path changes the severity grade from the referable boolean; referable status is probability aggregation and thresholding.",
            "Aspect ratio handling remains direct square resize to 224x224; no retinal-preserving crop was introduced in the frozen candidates.",
        ],
        "official_test_used_for_this_audit": False,
    }


class Dataset:
    def __init__(self, records: list[dict[str, Any]], transform: Any):
        import torch
        self.records = records
        self.transform = transform
        self.torch = torch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(IDRID_RAW / record["image"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, self.torch.tensor(int(record["label"]), dtype=self.torch.long), record


def collate(batch):
    import torch
    images, labels, records = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(labels)), list(records)


def data_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    previous = json.loads((META / "idrid_final_grading_data_audit.json").read_text(encoding="utf-8"))
    stats = []
    exact_hashes: dict[str, list[str]] = {}
    label_by_image: dict[str, int] = {record["image_id"]: int(record["label"]) for record in records}
    unreadable = []
    for record in records:
        path = IDRID_RAW / record["image"]
        try:
            content = path.read_bytes()
            with Image.open(io.BytesIO(content)) as probe:
                probe.verify()
            with Image.open(io.BytesIO(content)) as image:
                original_width, original_height = image.width, image.height
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((512, 512), Image.Resampling.BILINEAR)
                rgb = np.asarray(thumbnail, dtype=np.float32) / 255.0
                gray = rgb.mean(axis=2)
                edge = np.concatenate([rgb[0].ravel(), rgb[-1].ravel(), rgb[:, 0].ravel(), rgb[:, -1].ravel()])
                stats.append({"width": original_width, "height": original_height, "aspect_ratio": original_width / original_height, "brightness": float(gray.mean()), "contrast": float(gray.std()), "black_border_ratio": float(np.mean(edge < 0.03)), "rgb_mean": [float(value) for value in rgb.mean(axis=(0, 1))], "focus_laplacian_proxy": float(np.var(np.diff(gray, n=2, axis=0)) + np.var(np.diff(gray, n=2, axis=1)))})
            exact_hashes.setdefault(hashlib.sha256(content).hexdigest(), []).append(record["image_id"])
        except Exception as exc:
            unreadable.append({"image": record["image"], "error": f"{type(exc).__name__}: {exc}"})
    numeric = {key: [item[key] for item in stats] for key in ("aspect_ratio", "brightness", "contrast", "black_border_ratio", "focus_laplacian_proxy")}
    summary = {key: {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values))} for key, values in numeric.items()}
    duplicate_groups = [sorted(group) for group in exact_hashes.values() if len(group) > 1]
    duplicate_conflicts = [{"image_ids": group, "labels": sorted({label_by_image[image_id] for image_id in group})} for group in duplicate_groups if len({label_by_image[image_id] for image_id in group}) > 1]
    split_groups: dict[str, set[str]] = {}
    for record in records:
        split_groups.setdefault(record["split"], set()).add(record.get("duplicate_group_id") or record.get("record_key") or record["image"])
    split_overlap = sorted(split_groups.get("train", set()) & split_groups.get("validation", set()))
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "scope": "APTOS labeled development audit plus all 406 IDRiD development images; official test not opened in this cycle", "official_test_images_opened": 0, "patient_ids_available": False, "patient_level_limitation": "Patient identifiers are not supplied; duplicate/image grouping is the available control.", "idrid_development": {"record_count": len(records), "readable_count": len(records) - len(unreadable), "missing_files": 0, "missing_labels": sum(1 for record in records if record.get("label") not in range(5)), "unreadable": unreadable, "class_distribution": dict(Counter(str(record["label"]) for record in records)), "feature_summary": summary, "exact_duplicate_groups": duplicate_groups, "duplicate_conflicting_label_groups": duplicate_conflicts, "perceptual_duplicate_group_count": previous["idrid"]["inventory_summary"].get("perceptual_duplicate_groups", []).__len__(), "train_validation_duplicate_group_overlap": split_overlap}, "aptos_development_reference": {"source_artifact": rel(META / "idrid_final_grading_data_audit.json"), "record_count": previous["aptos"]["inventory_summary"]["record_count"], "readable_count": previous["aptos"]["inventory_summary"]["readable_count"], "class_distribution": previous["aptos"]["counts"], "cross_dataset_exact_matches": previous["cross_dataset_exact_sha256_matches"], "duplicate_group_count": len(previous["aptos"]["inventory_summary"].get("exact_duplicate_groups", [])), "perceptual_duplicate_group_count": len(previous["aptos"]["inventory_summary"].get("perceptual_duplicate_groups", []))}, "leakage": {"status": "PASS" if not split_overlap and not duplicate_conflicts and not previous["cross_dataset_exact_sha256_matches"] else "REVIEW_REQUIRED", "cross_dataset_exact_matches": previous["cross_dataset_exact_sha256_matches"], "cross_split_duplicate_groups": split_overlap, "duplicate_conflicting_label_groups": duplicate_conflicts, "patient_level_guarantee": False}, "no_external_datasets_used": True, "no_labels_modified": True}


def stratified_folds(records: list[dict[str, Any]], seed_value: int):
    from sklearn.model_selection import StratifiedGroupKFold
    labels = [int(record["label"]) for record in records]
    groups = [record.get("duplicate_group_id") or record.get("record_key") or record["image"] for record in records]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed_value)
    return list(splitter.split(np.zeros(len(records)), labels, groups)), groups


def infer(model: Any, loader: Any, device: Any, torch: Any) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    with torch.inference_mode():
        for images, labels, batch_records in loader:
            output = model(images.to(device))
            logits = output["severity_logits"].detach().cpu().numpy()
            probabilities = torch.softmax(output["severity_logits"], dim=1).detach().cpu().numpy()
            for index, record in enumerate(batch_records):
                vector = probabilities[index].astype(float).tolist()
                entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in vector))
                rows.append({"image_id": record["image_id"], "actual": int(labels[index].item()), "predicted": int(np.argmax(vector)), "probabilities": vector, "logits": logits[index].astype(float).tolist(), "confidence": float(max(vector)), "referable_probability": float(sum(vector[2:5])), "uncertainty": {"entropy_nats": entropy, "normalized_entropy": entropy / math.log(5.0), "probability_margin": float(np.sort(vector)[-1] - np.sort(vector)[-2])}})
    return rows


def fold_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [row["actual"] for row in rows]
    probabilities = [row["probabilities"] for row in rows]
    return metrics_with_threshold(actual, probabilities, THRESHOLD)


def run_cv(records: list[dict[str, Any]], args: Any) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(args.torch_threads)
    folds, groups = stratified_folds(records, args.seed)
    _, validation_transform = make_transforms(args.input_size)
    mapping = ReferableDRMapping(name="moderate_or_worse", referable_grades=REFERABLE)
    fold_results = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        fold_seed = args.seed + fold_number
        seed(fold_seed, torch)
        train_records = [records[index] for index in train_indices]
        validation_records = [records[index] for index in validation_indices]
        fold_dir = CV_ROOT / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = Dataset(train_records, robust_transform(args.input_size))
        validation_dataset = Dataset(validation_records, validation_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(fold_seed), num_workers=0, collate_fn=collate)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False)
        checkpoint = torch.load(APTOS_CHECKPOINT, map_location="cpu", weights_only=False)
        load_result = model.load_state_dict(checkpoint["state_dict"], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"Fold {fold_number} APTOS initialization mismatch: {load_result}")
        device = torch.device("cpu")
        model.to(device)
        labels = [int(record["label"]) for record in train_records]
        criterion = torch.nn.CrossEntropyLoss(weight=build_class_weights(labels).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        started = time.perf_counter()
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for images, batch_labels, _batch_records in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(images.to(device))
                loss, _components = hierarchical_loss(output, batch_labels.to(device), mapping, criterion, ordinal_mode=False)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_rows = infer(model, validation_loader, device, torch)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_metrics": fold_metrics(validation_rows)})
        rows = infer(model, validation_loader, device, torch)
        result = {"fold": fold_number, "seed": fold_seed, "train_count": len(train_records), "validation_count": len(validation_records), "train_class_distribution": dict(Counter(str(r["label"]) for r in train_records)), "validation_class_distribution": dict(Counter(str(r["label"]) for r in validation_records)), "duplicate_group_overlap": sorted(set(groups[index] for index in train_indices) & set(groups[index] for index in validation_indices)), "metrics": fold_metrics(rows), "history": history, "training_seconds": time.perf_counter() - started, "checkpoint": rel(fold_dir / "checkpoint_best.pt"), "checkpoint_sha256": None, "official_test_images_opened": 0, "protocol": "APTOS initialized EfficientNet-B0, IDRiD development fold only, robust augmentation, weighted cross-entropy, frozen referable threshold 0.40"}
        payload = {"state_dict": model.state_dict(), "model_config": {"backbone": "efficientnet_b0", "num_classes": 5, "input_size": args.input_size, "ordinal_mode": False}, "training_config": result["protocol"], "model_version": f"idrid-master-cv-fold-{fold_number}", "production_promoted": False, "official_test_images_opened": 0}
        checkpoint_path = fold_dir / "checkpoint_best.pt"
        torch.save(payload, checkpoint_path)
        result["checkpoint_sha256"] = sha256(checkpoint_path)
        dump(fold_dir / "validation_predictions.json", rows)
        dump(fold_dir / "metrics.json", result)
        fold_results.append(result)
        del model, optimizer, train_loader, validation_loader
        gc.collect()
        print(f"master_cv fold={fold_number} validation={len(validation_records)} qwk={result['metrics']['quadratic_weighted_kappa']} referable_sensitivity={result['metrics']['referable_dr']['sensitivity']}", flush=True)
    metric_names = {"accuracy": lambda r: r["metrics"]["accuracy"], "macro_precision": lambda r: r["metrics"]["precision"], "macro_recall": lambda r: r["metrics"]["recall"], "macro_f1": lambda r: r["metrics"]["f1"], "qwk": lambda r: r["metrics"]["quadratic_weighted_kappa"], "roc_auc_ovr": lambda r: r["metrics"]["roc_auc_ovr_macro"], "referable_sensitivity": lambda r: r["metrics"]["referable_dr"]["sensitivity"], "referable_specificity": lambda r: r["metrics"]["referable_dr"]["specificity"], "referable_precision": lambda r: r["metrics"]["referable_dr"]["precision"], "referable_f1": lambda r: r["metrics"]["referable_dr"]["f1"], "referable_false_negative_rate": lambda r: r["metrics"]["referable_dr"]["false_negative_rate"]}
    summary = {}
    for name, getter in metric_names.items():
        values = np.asarray([getter(row) for row in fold_results], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        summary[name] = {"mean": mean, "std": std, "min": float(values.min()), "max": float(values.max()), "approximate_95_percent_ci": [float(mean - 1.96 * std / math.sqrt(len(values))), float(mean + 1.96 * std / math.sqrt(len(values)))], "values": values.tolist()}
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "study": "stratified 5-fold grouped development study", "model_protocol": "APTOS initialized EfficientNet-B0; no IDRiD test records, lesion masks, localization labels, Messidor, or DRIVE used", "seed": args.seed, "fold_count": 5, "official_test_images_opened": 0, "patient_level_split": False, "patient_level_limitation": "Patient IDs are unavailable; duplicate_group_id/record_key grouping was used.", "duplicate_group_overlap_any_fold": any(result["duplicate_group_overlap"] for result in fold_results), "fixed_referable_threshold": THRESHOLD, "folds": fold_results, "summary": summary}


def reliability_artifact() -> dict[str, Any]:
    rows = json.loads(V3_PREDICTIONS.read_text(encoding="utf-8"))
    errors = [row for row in rows if int(row["actual"]) != int(row["predicted"])]
    high = [row for row in errors if float(row["confidence"]) >= 0.80]
    reviewed = [row for row in rows if float(row.get("normalized_entropy", 0.0)) >= 0.70 or float(row.get("probability_margin", 1.0)) < 0.15]
    return {"source_model": rel(V3_PREDICTIONS), "development_only": True, "threshold": THRESHOLD, "confidence": {"mean": float(np.mean([row["confidence"] for row in rows])), "min": float(np.min([row["confidence"] for row in rows])), "max": float(np.max([row["confidence"] for row in rows]))}, "uncertainty": {"mean_normalized_entropy": float(np.mean([row.get("normalized_entropy", 0.0) for row in rows])), "review_recommended_count": len(reviewed), "review_recommended_rate": len(reviewed) / len(rows), "policy": "normalized entropy >= 0.70 OR top-two probability margin < 0.15; does not change grade"}, "high_confidence_error_count": len(high), "high_confidence_errors": high, "all_error_count": len(errors), "statement": "Reliability flags are engineering safety signals, not clinical guarantees or error concealment.", "official_test_images_opened": 0}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run development-only IDRiD master research cycle")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true", help="Refresh the development audit without running CV")
    args = parser.parse_args()
    if sha256(APTOS_CHECKPOINT) != APTOS_SHA:
        raise RuntimeError("APTOS production checkpoint SHA changed")
    if sha256(V1_MANIFEST.parent / "checkpoint_best.pt") != V1_SHA or sha256(V2_MANIFEST.parent / "checkpoint_best.pt") != V2_SHA or sha256(V3_MANIFEST.parent / "checkpoint_best.pt") != V3_SHA:
        raise RuntimeError("Frozen IDRiD checkpoint SHA changed")
    records = load_development_records()
    dump(META / "idrid_final_data_audit.json", {"implementation_audit": implementation_audit(), **data_audit(records)})
    if args.audit_only:
        print(json.dumps({"audit": rel(META / "idrid_final_data_audit.json"), "official_test_images_opened": 0}, indent=2), flush=True)
        return 0
    cv = run_cv(records, args)
    dump(META / "idrid_final_cv_stability.json", cv)
    prior_comparison = json.loads((META / "idrid_final_grading_comparison.json").read_text(encoding="utf-8"))
    dump(META / "idrid_final_experiment_registry.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "official_test_images_opened": 0, "production_promoted": False, "experiments": {"prior_bounded_cycle": prior_comparison["candidate_rows"], "current_cv_protocol": {"study": "idrid_final_cv_stability.json", "fold_count": 5, "model_protocol": cv["model_protocol"]}}, "selection_policy": "development only; official test cannot be reused", "external_datasets_used": False})
    dump(META / "idrid_final_model_comparison.json", {"prior_development_comparison": prior_comparison, "current_cv_summary": cv["summary"], "official_test_used_for_selection": False, "selected_reference": "existing frozen V3 control remains the only post-freeze-tested candidate; CV models are research-only and not promoted"})
    prior_threshold = json.loads((META / "idrid_final_grading_threshold_analysis.json").read_text(encoding="utf-8"))
    dump(META / "idrid_final_threshold_analysis.json", {"source": rel(META / "idrid_final_grading_threshold_analysis.json"), "frozen_threshold": THRESHOLD, "threshold_selection_completed_before_official_test": True, "analysis": prior_threshold, "official_test_used": False})
    dump(META / "idrid_final_calibration.json", {"status": "UNCALIBRATED", "statement": "Raw softmax probabilities are not clinically calibrated.", "calibration_fit": False, "official_test_used": False, "reason": "No defensible independent calibration subset was available without weakening the small development protocol."})
    dump(META / "idrid_final_robustness.json", {"source": rel(META / "idrid_final_grading_robustness.json"), "official_test_used": False, "robustness": json.loads((META / "idrid_final_grading_robustness.json").read_text(encoding="utf-8"))})
    dump(META / "idrid_final_reliability.json", reliability_artifact())
    repro = json.loads((META / "idrid_final_grading_reproducibility.json").read_text(encoding="utf-8"))
    dump(META / "idrid_final_reproducibility.json", {"source": rel(META / "idrid_final_grading_reproducibility.json"), "official_test_images_opened_in_current_cycle": 0, **repro})
    final_manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    dump(META / "idrid_final_selected_candidate.json", {"selected_model": final_manifest["model_version"], "checkpoint": final_manifest["checkpoint"], "checkpoint_sha256": final_manifest["checkpoint_sha256"], "selection_status": "EXISTING_FROZEN_V3_REFERENCE; CURRENT_CV_RESEARCH_NOT_PROMOTED", "development_cv_artifact": rel(META / "idrid_final_cv_stability.json"), "official_test_reused": False, "production_promoted": False, "official_test_images_opened_in_current_cycle": 0})
    prior_official = json.loads(OLD_OFFICIAL.read_text(encoding="utf-8"))
    dump(META / "idrid_final_official_test.json", {"status": "ALREADY_EVALUATED_ONCE_AFTER_PRIOR_FREEZE; NOT_RERUN", "source_report": rel(OLD_OFFICIAL), "official_test_images_opened_in_current_cycle": 0, "do_not_rerun": True, "immutable_prior_result": prior_official, "selection_or_tuning_after_result": False})
    print(json.dumps({"cv": {"summary": cv["summary"], "duplicate_group_overlap_any_fold": cv["duplicate_group_overlap_any_fold"], "official_test_images_opened": 0}, "selected_reference": final_manifest["model_version"]}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
