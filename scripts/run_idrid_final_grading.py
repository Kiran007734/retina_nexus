"""Run the bounded final IDRiD disease-grading research cycle.

This command is intentionally isolated from production.  It uses only the
governed APTOS labeled training split and the governed IDRiD development
split.  The reserved 103-image IDRiD test records are never loaded by the
development phase.  Candidate A reuses the already measured V3 control;
candidates B/C/D are the only new controlled experiments.

The script writes experiment outputs below
``ml/weights/classifiers/idrid/research/final/20260912`` and metadata below
``ml/datasets/metadata/idrid``.  It does not overwrite V1, V2, V3, or the
APTOS production checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.classifier import ReferableDRMapping, build_classifier  # noqa: E402
from ml.evaluation.metrics import classification_metrics  # noqa: E402
from ml.training.losses import build_class_weights, build_focal_loss, hierarchical_loss  # noqa: E402
from scripts.train_classifier import make_transforms, select_device  # noqa: E402


IDRID_SPLIT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"
IDRID_DEV_MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
APTOS_SPLIT = ROOT / "ml" / "datasets" / "metadata" / "splits" / "aptos2019" / "splits.json"
IDRID_RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
APTOS_RAW = ROOT / "ml" / "datasets" / "raw" / "aptos2019"
APTOS_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "aptos2019" / "efficientnet-b0-aptos2019-20260830-v1" / "checkpoint_best.pt"
V1_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "efficientnet-b0-idrid-20260912-v1" / "checkpoint_best.pt"
V2_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v2" / "20260912" / "v2_c_lesion_aware_efficientnet_b0" / "checkpoint_best.pt"
V3_CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912" / "v3_b_domain_robust" / "checkpoint_best.pt"
V3_PREDICTIONS = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "v3" / "20260912" / "v3_b_domain_robust" / "validation_predictions.json"
OUTPUT_ROOT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "final" / "20260912"
META_ROOT = ROOT / "ml" / "datasets" / "metadata" / "idrid"

EXPECTED_APTOS_SHA = "ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b"
EXPECTED_V1_SHA = "d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de"
EXPECTED_V2_SHA = "2c5bebd6be18184c56443d8d6a8f590d8b31c8904caea6a0c8b31667f9ac9967"
EXPECTED_V3_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"
REFERABLE_GRADES = (2, 3, 4)
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CLASS_MAPPING = {"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative DR"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded final IDRiD grading research cycle")
    parser.add_argument("--experiments", default="b,c,d", help="New candidates to run; A is always the frozen V3 control")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--combination-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_protected_checkpoints() -> dict[str, Any]:
    expected = {
        "aptos_production": (APTOS_CHECKPOINT, EXPECTED_APTOS_SHA),
        "idrid_v1": (V1_CHECKPOINT, EXPECTED_V1_SHA),
        "idrid_v2": (V2_CHECKPOINT, EXPECTED_V2_SHA),
        "idrid_v3": (V3_CHECKPOINT, EXPECTED_V3_SHA),
    }
    result: dict[str, Any] = {}
    for name, (path, expected_sha) in expected.items():
        if not path.is_file():
            raise RuntimeError(f"Protected checkpoint is missing: {path}")
        actual = sha256(path)
        if actual != expected_sha:
            raise RuntimeError(f"Protected checkpoint SHA mismatch for {name}: {actual} != {expected_sha}")
        result[name] = {"path": relative(path), "sha256": actual, "unchanged": True}
    return result


def records_for_data_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aptos = load_json(APTOS_SPLIT)
    idrid = load_json(IDRID_SPLIT)
    if idrid.get("official_test_images_opened") not in (None, 0) or idrid.get("official_test_used"):
        raise RuntimeError("IDRiD split manifest no longer proves official-test isolation")
    idrid_records = [r for r in idrid.get("records", []) if r.get("split") in {"train", "validation"}]
    aptos_records = [r for r in aptos.get("records", []) if r.get("split") in {"train", "validation"}]
    if len(idrid_records) != 406 or len(aptos_records) != 3046:
        raise RuntimeError(f"Unexpected governed audit counts: IDRiD={len(idrid_records)}, APTOS={len(aptos_records)}")
    return aptos_records, idrid_records


def image_inventory(root: Path, records: Iterable[dict[str, Any]], source: str) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    sha_groups: dict[str, list[str]] = defaultdict(list)
    phash_groups: dict[str, list[str]] = defaultdict(list)
    dimensions: Counter[str] = Counter()
    for record in records:
        rel = record["image"]
        path = root / rel
        item: dict[str, Any] = {"source": source, "image": rel, "label": record.get("label"), "exists": path.is_file()}
        if not path.is_file():
            item["error"] = "missing_file"
            unreadable.append(item)
            inventory.append(item)
            continue
        try:
            content = path.read_bytes()
            with Image.open(io.BytesIO(content)) as probe:
                image_format = probe.format
                mode = probe.mode
                probe.verify()
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                rgb = image.convert("RGB")
                small = np.asarray(rgb.resize((8, 8)), dtype=np.float32)
                gray = small.mean(axis=2)
                median = float(np.median(gray))
                phash = "".join("1" if value >= median else "0" for value in gray.ravel())
                item.update({"sha256": hashlib.sha256(content).hexdigest(), "format": image_format, "mode": mode, "width": image.width, "height": image.height, "phash": phash})
                dimensions[f"{image.width}x{image.height}"] += 1
                sha_groups[item["sha256"]].append(rel)
                phash_groups[phash].append(rel)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            unreadable.append(item)
        inventory.append(item)
    return {
        "source": source,
        "record_count": len(inventory),
        "readable_count": len(inventory) - len(unreadable),
        "unreadable": unreadable,
        "dimensions": dict(dimensions),
        "exact_duplicate_groups": [sorted(paths) for paths in sha_groups.values() if len(paths) > 1],
        "perceptual_duplicate_groups": [sorted(paths) for paths in phash_groups.values() if len(paths) > 1],
        "inventory": inventory,
    }


def data_audit() -> dict[str, Any]:
    aptos_records, idrid_records = records_for_data_audit()
    aptos = image_inventory(APTOS_RAW, aptos_records, "APTOS")
    idrid = image_inventory(IDRID_RAW, idrid_records, "IDRiD")
    aptos_by_sha = {item.get("sha256"): item for item in aptos["inventory"] if item.get("sha256")}
    idrid_by_sha = {item.get("sha256"): item for item in idrid["inventory"] if item.get("sha256")}
    exact_cross = sorted(set(aptos_by_sha) & set(idrid_by_sha))
    aptos_class = Counter(str(r["label"]) for r in aptos_records)
    idrid_class = Counter(str(r["label"]) for r in idrid_records)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "APTOS labeled train/validation records and IDRiD governed development records only",
        "official_test_images_opened": 0,
        "patient_identifier_available": False,
        "patient_level_limitation": "Neither source provides a patient identifier in the governed manifests; exact/perceptual image grouping is enforced where available.",
        "aptos": {"counts": dict(aptos_class), "inventory_summary": {k: v for k, v in aptos.items() if k != "inventory"}},
        "idrid": {"counts": dict(idrid_class), "inventory_summary": {k: v for k, v in idrid.items() if k != "inventory"}},
        "cross_dataset_exact_sha256_matches": [
            {"sha256": value, "aptos_image": aptos_by_sha[value]["image"], "idrid_image": idrid_by_sha[value]["image"]}
            for value in exact_cross
        ],
        "cross_dataset_perceptual_check": "Coarse 8x8 median hash groups were computed independently; perceptual matches are screening candidates, not proof of same patient or image.",
        "leakage": {
            "aptos_split_status": load_json(APTOS_SPLIT).get("leakage", {}).get("status"),
            "idrid_split_status": load_json(IDRID_SPLIT).get("leakage", {}).get("status"),
            "cross_split_exact_duplicate_count": 0,
            "status": "PASS" if not exact_cross else "REVIEW_REQUIRED",
        },
        "no_external_datasets_used": True,
    }


class RecordDataset:
    def __init__(self, records: list[dict[str, Any]], root: Path, transform: Any, domain: str):
        import torch
        self.records = records
        self.root = root
        self.transform = transform
        self.domain = domain
        self.torch = torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = self.root / record["image"]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.torch.tensor(int(record["label"]), dtype=self.torch.long), {**record, "domain": self.domain}


class CombinedDataset:
    def __init__(self, datasets: list[RecordDataset]):
        self.datasets = datasets
        self.offsets = []
        total = 0
        for dataset in datasets:
            self.offsets.append(total)
            total += len(dataset)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        for offset, dataset in reversed(list(zip(self.offsets, self.datasets))):
            if index >= offset:
                return dataset[index - offset]
        raise IndexError(index)


def collate(batch):
    import torch
    images, labels, records = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(labels)), list(records)


def robust_transform(input_size: int):
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
        transforms.ToTensor(), normalize,
    ])


def load_split_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aptos = load_json(APTOS_SPLIT)
    idrid = load_json(IDRID_SPLIT)
    if idrid.get("official_test_images_opened") not in (None, 0) or idrid.get("official_test_used"):
        raise RuntimeError("Development run cannot proceed: IDRiD official-test isolation flag failed")
    aptos_train = [r for r in aptos["records"] if r.get("split") == "train"]
    idrid_train = [r for r in idrid["records"] if r.get("split") == "train"]
    idrid_val = [r for r in idrid["records"] if r.get("split") == "validation"]
    if len(aptos_train) != 2509 or len(idrid_train) != 323 or len(idrid_val) != 83:
        raise RuntimeError(f"Unexpected development split sizes: APTOS train={len(aptos_train)}, IDRiD train={len(idrid_train)}, IDRiD val={len(idrid_val)}")
    return aptos_train, idrid_train, idrid_val


def load_aptos_state(model: Any, checkpoint: Path, torch: Any) -> dict[str, Any]:
    actual = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = model.load_state_dict(payload["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"APTOS strict initialization failed: {result}")
    return {"source": relative(checkpoint), "sha256": actual, "model_version": payload.get("model_version"), "state_dict_loading": "strict"}


def load_state(model: Any, checkpoint: Path, torch: Any, expected_sha: str | None = None) -> dict[str, Any]:
    actual = sha256(checkpoint)
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint} {actual} != {expected_sha}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = model.load_state_dict(payload["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint loading failed: {result}")
    return {"source": relative(checkpoint), "sha256": actual, "model_version": payload.get("model_version"), "state_dict_loading": "strict"}


def metrics_with_threshold(actual: list[int], probabilities: list[list[float]], threshold: float) -> dict[str, Any]:
    base = classification_metrics(actual, probabilities, referable_grades=REFERABLE_GRADES)
    truth = np.isin(np.asarray(actual, dtype=int), REFERABLE_GRADES).astype(int)
    probability = np.asarray([sum(row[2:5]) for row in probabilities], dtype=float)
    predicted = (probability >= threshold).astype(int)
    tp = int(((truth == 1) & (predicted == 1)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    base["referable_dr"] = {**base["referable_dr"], "threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "f1": f1, "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn, "false_negative_rate": 1.0 - sensitivity}
    return base


def threshold_table(actual: list[int], probabilities: list[list[float]]) -> dict[str, Any]:
    entries = []
    for threshold in THRESHOLDS:
        metrics = metrics_with_threshold(actual, probabilities, threshold)["referable_dr"]
        entries.append({"threshold": threshold, **{key: metrics[key] for key in ("sensitivity", "specificity", "precision", "f1", "true_positive", "true_negative", "false_positive", "false_negative", "false_negative_rate")}})
    eligible = [row for row in entries if row["specificity"] >= 0.85]
    selected = sorted(eligible, key=lambda row: (-row["sensitivity"], -row["specificity"], -row["f1"], row["false_positive"], -row["threshold"]))[0] if eligible else None
    return {"candidate_thresholds": list(THRESHOLDS), "selection_rule": "highest development sensitivity among thresholds with specificity >= 0.85; ties specificity, F1, fewer false positives, then higher threshold", "entries": entries, "selected": selected}


def ece_brier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=float)
    actual = np.asarray([row["actual"] for row in rows], dtype=int)
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == actual).astype(int)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    one_hot = np.eye(5, dtype=float)[actual]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {"status": "UNCALIBRATED", "temperature_scaling_fitted": False, "calibration_subset": None, "ece_10_bins_diagnostic": float(ece), "multiclass_brier_diagnostic": brier, "statement": "Raw softmax confidence is not clinically calibrated; calibration could not be reliably fitted with the available non-test data."}


def error_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    def public(row: dict[str, Any]) -> dict[str, Any]:
        return {"image_id": row["image_id"], "actual_grade": row["actual"], "predicted_grade": row["predicted"], "probabilities": row["probabilities"], "referable_probability": row["referable_probability"], "confidence": row["confidence"], "uncertainty": row["uncertainty"], "quality_status": "NOT_REASSESSED_IN_RESEARCH_RUN"}
    return {
        "grade_0_to_2_3_4": [public(r) for r in rows if r["actual"] == 0 and r["predicted"] in {2, 3, 4}],
        "grade_1_to_2_3_4": [public(r) for r in rows if r["actual"] == 1 and r["predicted"] in {2, 3, 4}],
        "grade_2_3_4_to_0_1": [public(r) for r in rows if r["actual"] in REFERABLE_GRADES and r["predicted"] in {0, 1}],
        "grade_3_to_4_or_4_to_3": [public(r) for r in rows if {r["actual"], r["predicted"]} == {3, 4}],
        "referable_false_negatives": [public(r) for r in rows if r["actual"] in REFERABLE_GRADES and r["referable_probability"] < threshold],
        "high_confidence_incorrect": sorted([public(r) for r in rows if r["actual"] != r["predicted"]], key=lambda r: (-r["confidence"], r["image_id"])),
    }


def prediction_rows(model: Any, loader: Any, device: Any, torch: Any) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, labels, records in loader:
            outputs = model(images.to(device))
            logits = outputs["severity_logits"].detach().cpu().numpy()
            probabilities = torch.softmax(outputs["severity_logits"], dim=1).detach().cpu().numpy()
            for index, record in enumerate(records):
                vector = probabilities[index].astype(float).tolist()
                entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in vector))
                rows.append({"image_id": record.get("image_id", record.get("image")), "record_key": record.get("record_key", record.get("image")), "actual": int(labels[index].item()), "logits": logits[index].astype(float).tolist(), "probabilities": vector, "predicted": int(np.argmax(vector)), "confidence": float(max(vector)), "referable_probability": float(sum(vector[2:5])), "uncertainty": {"entropy_nats": entropy, "normalized_entropy": entropy / math.log(5.0), "probability_margin": float(np.sort(vector)[-1] - np.sort(vector)[-2])}})
    return rows


def build_loader(dataset: Any, batch_size: int, sampler: Any = None):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=0, collate_fn=collate)


def train_candidate(candidate: str, args: argparse.Namespace, train_aptos: list[dict[str, Any]], train_idrid: list[dict[str, Any]], val_idrid: list[dict[str, Any]], device: Any, torch: Any) -> dict[str, Any]:
    from torch.utils.data import WeightedRandomSampler

    candidate_dir = OUTPUT_ROOT / {"b": "candidate_b_aptos_finetune", "c": "candidate_c_domain_balanced", "d": "candidate_d_combined_finetune"}[candidate]
    if args.reuse_existing and (candidate_dir / "metrics.json").is_file() and (candidate_dir / "checkpoint_best.pt").is_file():
        return load_json(candidate_dir / "metrics.json")
    input_size = args.input_size
    train_transform = robust_transform(input_size)
    _, validation_transform = make_transforms(input_size)
    idrid_train = RecordDataset(train_idrid, IDRID_RAW, train_transform, "idrid")
    aptos_train = RecordDataset(train_aptos, APTOS_RAW, train_transform, "aptos")
    validation = RecordDataset(val_idrid, IDRID_RAW, validation_transform, "idrid")
    if candidate == "b":
        train_dataset = idrid_train
        train_labels = [int(r["label"]) for r in train_idrid]
        sampler = WeightedRandomSampler(torch.as_tensor([1.0 / max(1, Counter(train_labels)[label]) for label in train_labels], dtype=torch.double), num_samples=len(train_labels), replacement=True)
        sampling = "IDRiD class-balanced weighted sampler"
    elif candidate == "c":
        train_dataset = CombinedDataset([idrid_train, aptos_train])
        weights = [0.5 / len(idrid_train)] * len(idrid_train) + [0.5 / len(aptos_train)] * len(aptos_train)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=2 * len(idrid_train), replacement=True)
        train_labels = [int(r["label"]) for r in train_idrid]
        sampling = "domain-balanced sampler: 50% IDRiD / 50% APTOS per epoch"
    else:
        train_dataset = idrid_train
        train_labels = [int(r["label"]) for r in train_idrid]
        sampler = WeightedRandomSampler(torch.as_tensor([1.0 / max(1, Counter(train_labels)[label]) for label in train_labels], dtype=torch.double), num_samples=len(train_labels), replacement=True)
        sampling = "IDRiD class-balanced weighted sampler after candidate C initialization"
    train_loader = build_loader(train_dataset, args.batch_size, sampler)
    val_loader = build_loader(validation, args.batch_size)
    model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False).to(device)
    if candidate in {"b", "c"}:
        initialization = load_aptos_state(model, APTOS_CHECKPOINT, torch)
    else:
        source = OUTPUT_ROOT / "candidate_c_domain_balanced" / "checkpoint_best.pt"
        initialization = load_state(model, source, torch)
    class_weights = build_class_weights(train_labels).to(device)
    criterion = build_focal_loss(gamma=2.0, alpha=class_weights) if candidate in {"b", "d"} else torch.nn.CrossEntropyLoss(weight=class_weights)
    mapping = ReferableDRMapping(name="moderate_or_worse", referable_grades=REFERABLE_GRADES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-7)
    epochs = args.combination_epochs if candidate == "d" else args.epochs
    config = {"candidate": candidate, "architecture": "EfficientNet-B0 hierarchical classifier; severity head is the grade output", "backbone": "efficientnet_b0", "input_size": input_size, "input_channels": 3, "color_space": "RGB", "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "augmentation": "controlled robust retinal augmentation", "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "epochs_requested": epochs, "early_stopping_patience": args.patience, "loss": "class-weighted focal gamma=2" if candidate in {"b", "d"} else "class-weighted cross-entropy", "sampling": sampling, "initialization": initialization, "dataset": "APTOS train + IDRiD development train according to candidate", "seed": args.seed, "referable_rule": "P(2)+P(3)+P(4) >= threshold; severity=argmax(P0..P4)", "production_promoted": False, "clinical_validation_claim": False}
    candidate_dir.mkdir(parents=True, exist_ok=True)
    dump(candidate_dir / "training_config.json", config)
    best_score = -float("inf")
    best_epoch = 0
    patience_count = 0
    history: list[dict[str, Any]] = []
    best_path = candidate_dir / "checkpoint_best.pt"
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels, _records in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(device))
            loss, _ = hierarchical_loss(outputs, labels.to(device), mapping, criterion, ordinal_mode=False)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        rows = prediction_rows(model, val_loader, device, torch)
        val_metrics = metrics_with_threshold([r["actual"] for r in rows], [r["probabilities"] for r in rows], 0.5)
        scheduler.step(val_metrics["f1"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": val_metrics, "learning_rate": optimizer.param_groups[0]["lr"]})
        score = float(val_metrics["f1"])
        payload = {"state_dict": model.state_dict(), "model_config": {"backbone": "efficientnet_b0", "num_classes": 5, "input_size": input_size, "ordinal_mode": False}, "training_config": config, "metrics": val_metrics, "model_version": f"idrid-final-candidate-{candidate}", "epoch": epoch, "best_epoch": best_epoch, "dataset": "idrid_development", "production_promoted": False, "clinical_validation_claim": False}
        torch.save(payload, candidate_dir / "checkpoint_last.pt")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            payload["best_epoch"] = best_epoch
            torch.save(payload, best_path)
            patience_count = 0
        else:
            patience_count += 1
        print(f"candidate_{candidate} epoch={epoch} loss={np.mean(losses):.4f} val_f1={val_metrics['f1']:.4f} val_qwk={val_metrics['quadratic_weighted_kappa']}", flush=True)
        if patience_count >= args.patience:
            break
    elapsed = time.perf_counter() - start
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["state_dict"], strict=True)
    rows = prediction_rows(model, val_loader, device, torch)
    actual = [r["actual"] for r in rows]
    probabilities = [r["probabilities"] for r in rows]
    base_metrics = classification_metrics(actual, probabilities, referable_grades=REFERABLE_GRADES)
    table = threshold_table(actual, probabilities)
    selected_threshold = float(table["selected"]["threshold"] if table["selected"] else 0.5)
    final_metrics = metrics_with_threshold(actual, probabilities, selected_threshold)
    result = {"candidate": candidate, "experiment_id": f"final_candidate_{candidate}", "status": "COMPLETED", "checkpoint_path": relative(best_path), "checkpoint_sha256": sha256(best_path), "best_epoch": best_epoch, "epochs_completed": len(history), "training_seconds": elapsed, "training_config": config, "metrics_at_0_5": base_metrics, "metrics": final_metrics, "threshold_analysis": table, "calibration": ece_brier(rows), "error_analysis": error_rows(rows, selected_threshold), "validation_predictions": relative(candidate_dir / "validation_predictions.json"), "production_promoted": False, "official_test_images_opened": 0, "clinical_validation_claim": False}
    dump(candidate_dir / "history.json", history)
    dump(candidate_dir / "validation_predictions.json", rows)
    dump(candidate_dir / "metrics.json", result)
    dump(candidate_dir / "model_manifest.json", result)
    return result


def load_control() -> dict[str, Any]:
    if not V3_PREDICTIONS.is_file():
        raise RuntimeError(f"Frozen V3 control predictions are missing: {V3_PREDICTIONS}")
    rows = json.loads(V3_PREDICTIONS.read_text(encoding="utf-8"))
    # V3 stored uncertainty as flat fields; normalize it at the artifact
    # boundary without changing the frozen control predictions.
    for row in rows:
        if "uncertainty" not in row:
            entropy = float(row.get("entropy_nats", 0.0))
            row["uncertainty"] = {
                "entropy_nats": entropy,
                "normalized_entropy": float(row.get("normalized_entropy", entropy / math.log(5.0))),
                "probability_margin": float(row.get("probability_margin", np.sort(np.asarray(row["probabilities"], dtype=float))[-1] - np.sort(np.asarray(row["probabilities"], dtype=float))[-2])),
            }
    actual = [int(r["actual"]) for r in rows]
    probabilities = [list(map(float, r["probabilities"])) for r in rows]
    table = threshold_table(actual, probabilities)
    selected_threshold = float(table["selected"]["threshold"] if table["selected"] else 0.5)
    return {"candidate": "a", "experiment_id": "final_candidate_a_v3_control", "status": "REUSED_FROZEN_CONTROL", "checkpoint_path": relative(V3_CHECKPOINT), "checkpoint_sha256": sha256(V3_CHECKPOINT), "best_epoch": 1, "epochs_completed": 0, "training_seconds": 0.0, "training_config": {"source": "V3 domain-robust control; no retraining in final cycle", "preprocessing": "RGB -> Resize(224,224) -> ToTensor -> ImageNet mean/std", "augmentation": "V3 robust augmentation recorded in prior manifest", "production_promoted": False}, "metrics_at_0_5": classification_metrics(actual, probabilities, referable_grades=REFERABLE_GRADES), "metrics": metrics_with_threshold(actual, probabilities, selected_threshold), "threshold_analysis": table, "calibration": ece_brier(rows), "error_analysis": error_rows(rows, selected_threshold), "validation_predictions": relative(V3_PREDICTIONS), "production_promoted": False, "official_test_images_opened": 0, "clinical_validation_claim": False}


def compare(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        metrics = result["metrics"]
        referable = metrics["referable_dr"]
        per_class = metrics["per_class"]
        rows.append({"candidate": result["candidate"], "experiment_id": result["experiment_id"], "checkpoint_path": result["checkpoint_path"], "checkpoint_sha256": result["checkpoint_sha256"], "accuracy": metrics["accuracy"], "macro_precision": metrics["precision"], "macro_recall": metrics["recall"], "macro_f1": metrics["f1"], "qwk": metrics["quadratic_weighted_kappa"], "roc_auc_ovr": metrics["roc_auc_ovr_macro"], "referable_sensitivity": referable["sensitivity"], "referable_specificity": referable["specificity"], "referable_precision": referable["precision"], "referable_f1": referable["f1"], "referable_fn": referable["false_negative"], "referable_fp": referable["false_positive"], "grade_2_sensitivity": per_class["Moderate"]["sensitivity"], "grade_3_sensitivity": per_class["Severe"]["sensitivity"], "grade_4_sensitivity": per_class["Proliferative DR"]["sensitivity"], "ece": result["calibration"]["ece_10_bins_diagnostic"], "official_test": "NOT_EVALUATED"})
    baseline = next(row for row in rows if row["candidate"] == "a")
    eligible = [row for row in rows if row["referable_specificity"] >= 0.85]
    ranked = sorted(eligible, key=lambda row: (row["referable_sensitivity"], -row["referable_fn"], row["qwk"], row["macro_f1"], row["grade_2_sensitivity"] + row["grade_3_sensitivity"] + row["grade_4_sensitivity"], -row["ece"]), reverse=True)
    selected = ranked[0] if ranked else baseline
    # The selection is development-only and is not a production promotion.
    return {"candidate_rows": rows, "selection_priority": ["no leakage", "referable sensitivity", "specificity >= 0.85", "false negatives", "QWK", "macro F1", "Grade 2/3/4 sensitivity", "calibration/reliability", "reproducibility"], "selected_candidate": selected["candidate"], "selected_experiment_id": selected["experiment_id"], "selected_checkpoint_path": selected["checkpoint_path"], "selection_reason": "Development-only ranking; no official test or external dataset was accessed.", "production_promoted": False, "official_test_images_opened": 0}


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.combination_epochs < 1 or args.patience < 1:
        raise SystemExit("epochs, combination-epochs, and patience must be positive")
    import torch
    torch.set_num_threads(max(1, args.torch_threads))
    seed_everything(args.seed, torch)
    protected = verify_protected_checkpoints()
    audit_path = META_ROOT / "idrid_final_grading_data_audit.json"
    if audit_path.is_file():
        audit = load_json(audit_path)
    else:
        audit = data_audit()
        dump(audit_path, {"protected_checkpoints": protected, **audit})
    aptos_train, idrid_train, idrid_val = load_split_records()
    device = select_device(torch, args.device)
    results = [load_control()]
    for candidate in [item.strip().lower() for item in args.experiments.split(",") if item.strip()]:
        if candidate not in {"b", "c", "d"}:
            raise SystemExit(f"Unsupported candidate {candidate}; choose b,c,d")
        if candidate == "d" and not (OUTPUT_ROOT / "candidate_c_domain_balanced" / "checkpoint_best.pt").is_file() and "c" not in [item.strip().lower() for item in args.experiments.split(",")]:
            raise SystemExit("Candidate D requires candidate C checkpoint; include c in --experiments or use --reuse-existing")
        results.append(train_candidate(candidate, args, aptos_train, idrid_train, idrid_val, device, torch))
    dump(META_ROOT / "idrid_final_grading_experiment_registry.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "scope": "development only", "experiments": results, "protected_checkpoints": protected, "production_promoted": False, "official_test_images_opened": 0})
    comparison = compare(results)
    dump(META_ROOT / "idrid_final_grading_comparison.json", comparison)
    selected = next(r for r in results if r["candidate"] == comparison["selected_candidate"])
    dump(META_ROOT / "idrid_final_grading_threshold_analysis.json", {"selected_candidate": selected["candidate"], "threshold_analysis": selected["threshold_analysis"], "all_candidates": {r["candidate"]: r["threshold_analysis"] for r in results}})
    dump(META_ROOT / "idrid_final_grading_calibration.json", {"selected_candidate": selected["candidate"], "calibration": selected["calibration"], "decision": "No calibration fit; non-test calibration sample is not defensible for this small fixed development set."})
    dump(META_ROOT / "idrid_final_grading_selected_candidate.json", {"selected_candidate": selected, "selection": comparison, "official_test_status": "RESERVED_NOT_EVALUATED", "production_promoted": False})
    print(json.dumps({"selected_candidate": comparison["selected_candidate"], "selected_checkpoint": selected["checkpoint_path"], "selected_metrics": selected["metrics"], "official_test_images_opened": 0}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
