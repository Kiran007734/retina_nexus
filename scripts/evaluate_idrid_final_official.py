"""Perform the one authorized post-freeze official IDRiD test evaluation.

This script refuses to run unless the final research manifest is frozen, the
checkpoint SHA matches, production promotion is false, and the threshold is
already recorded.  It does not train, tune, or modify the frozen checkpoint.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.classifier import build_classifier  # noqa: E402
from scripts.run_idrid_final_grading import metrics_with_threshold  # noqa: E402
from scripts.run_idrid_v2_research import build_model as build_lesion_aware_model  # noqa: E402
from scripts.train_classifier import make_transforms  # noqa: E402

SPLIT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"
FINAL_MANIFEST = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "final" / "model_manifest.json"
CHECKPOINT = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "final" / "checkpoint_best.pt"
OUTPUT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_grading_official_test.json"
RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def load_model(torch: Any):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    if any(key.startswith("lesion_head.") or key.startswith("structure_head.") for key in state):
        import torch.nn as nn
        model = build_lesion_aware_model(torch, nn)()
    else:
        model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Frozen checkpoint state mismatch: {result}")
    model.eval()
    return model


def load_and_audit_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    records = sorted(split.get("reserved_official_test_records", []), key=lambda item: item["path"])
    if len(records) != 103:
        raise RuntimeError(f"Expected 103 official IDRiD test records, found {len(records)}")
    train_val = split.get("records", [])
    train_val_hashes = {r.get("sha256"): r.get("record_key") for r in train_val if r.get("sha256")}
    details = []
    hashes: dict[str, list[str]] = defaultdict(list)
    unreadable = []
    dimensions = Counter()
    for record in records:
        relative = Path(record["path"]).relative_to("ml/datasets/raw/idrid")
        path = RAW / relative
        item = {"image_id": record["image_id"], "path": record["path"], "exists": path.is_file(), "sha256": None, "width": None, "height": None, "format": None, "mode": None}
        try:
            content = path.read_bytes()
            with Image.open(io.BytesIO(content)) as probe:
                item["format"] = probe.format
                probe.verify()
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                item["width"], item["height"], item["mode"] = image.width, image.height, image.mode
                dimensions[f"{image.width}x{image.height}"] += 1
            item["sha256"] = hashlib.sha256(content).hexdigest()
            hashes[item["sha256"]].append(record["image_id"])
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            unreadable.append(item)
        details.append(item)
    duplicate_groups = [{"sha256": content_hash, "image_ids": sorted(image_ids)} for content_hash, image_ids in hashes.items() if len(image_ids) > 1]
    cross_split = [{"sha256": item["sha256"], "test_image_ids": sorted(hashes[item["sha256"]]), "development_record_key": train_val_hashes[item["sha256"]]} for item in details if item.get("sha256") in train_val_hashes]
    audit = {"expected_count": 103, "readable_count": len(records) - len(unreadable), "unreadable": unreadable, "dimensions": dict(dimensions), "duplicate_groups_within_official_test": duplicate_groups, "cross_split_exact_hash_matches": cross_split, "record_manifest_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "official_test_images_opened": 103}
    if unreadable or cross_split:
        raise RuntimeError(f"Official test integrity gate failed: unreadable={len(unreadable)}, cross_split={len(cross_split)}")
    return records, audit


def main() -> int:
    import torch
    if not FINAL_MANIFEST.is_file() or not CHECKPOINT.is_file():
        raise SystemExit("Final candidate is not frozen; run finalize_idrid_final_grading.py first")
    manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_status") != "FROZEN_BEFORE_OFFICIAL_TEST":
        raise SystemExit("Official evaluation requires freeze_status=FROZEN_BEFORE_OFFICIAL_TEST")
    if manifest.get("production_promoted") or manifest.get("official_test_images_opened") != 0:
        raise SystemExit("Production promotion or pre-freeze test access flag is invalid")
    expected_sha = manifest.get("checkpoint_sha256")
    before_sha = sha256(CHECKPOINT)
    if before_sha != expected_sha:
        raise SystemExit(f"Frozen checkpoint SHA mismatch: {before_sha} != {expected_sha}")
    threshold = float(manifest["referable_threshold"])
    if threshold not in {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60}:
        raise SystemExit(f"Threshold {threshold} was not from the pre-specified development candidate set")
    records, audit = load_and_audit_records()
    _, transform = make_transforms(int(manifest["input_size"]))
    model = load_model(torch)
    rows = []
    with torch.inference_mode():
        for record in records:
            path = RAW / Path(record["path"]).relative_to("ml/datasets/raw/idrid")
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            output = model(tensor)
            logits = output["severity_logits"][0].detach().cpu().numpy().astype(float)
            probabilities = torch.softmax(output["severity_logits"], dim=1)[0].detach().cpu().numpy().astype(float).tolist()
            entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probabilities))
            rows.append({"image_id": record["image_id"], "actual": int(record["dr_grade"]), "predicted": int(np.argmax(probabilities)), "logits": logits.tolist(), "probabilities": probabilities, "referable_probability": float(sum(probabilities[2:5])), "referable": bool(sum(probabilities[2:5]) >= threshold), "confidence": float(max(probabilities)), "uncertainty": {"entropy_nats": entropy, "normalized_entropy": entropy / math.log(5.0), "probability_margin": float(np.sort(probabilities)[-1] - np.sort(probabilities)[-2])}})
    actual = [row["actual"] for row in rows]
    probabilities = [row["probabilities"] for row in rows]
    metrics = metrics_with_threshold(actual, probabilities, threshold)
    high_confidence_errors = [row for row in rows if row["actual"] != row["predicted"] and row["confidence"] >= 0.80]
    result = {"evaluation_type": "ONE_POST_FREEZE_OFFICIAL_IDRID_TEST_EVALUATION", "generated_at": datetime.now(timezone.utc).isoformat(), "model_version": manifest["model_version"], "checkpoint": rel(CHECKPOINT), "checkpoint_sha256_before": before_sha, "checkpoint_sha256_after": sha256(CHECKPOINT), "checkpoint_unchanged": before_sha == sha256(CHECKPOINT), "frozen_manifest": rel(FINAL_MANIFEST), "threshold_frozen_from_development": threshold, "referable_rule": "P(2)+P(3)+P(4) >= frozen threshold; severity=argmax(P0..P4)", "dataset": audit, "metrics": metrics, "class_distribution": dict(Counter(map(str, actual))), "high_confidence_errors": high_confidence_errors, "calibration": {"status": "UNCALIBRATED", "raw_softmax_ece_not_a_calibration_claim": True, "statement": "No calibration was fitted; raw softmax values are not clinically calibrated."}, "predictions": rows, "production_promoted": False, "official_test_images_opened": 103, "post_test_tuning": False}
    dump(OUTPUT, result)
    print(json.dumps({"output": rel(OUTPUT), "checkpoint_unchanged": result["checkpoint_unchanged"], "metrics": metrics, "official_test_images_opened": 103}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
