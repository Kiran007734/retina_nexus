"""Freeze the selected final IDRiD development candidate.

This script performs no training and reads only the 83-image development
validation split.  It creates the final research checkpoint namespace,
records provenance, and measures deterministic validation reproducibility plus
controlled perturbation stability.  The official 103-image test set is not
opened here; it is handled by the separate post-freeze evaluator.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models.classifier import build_classifier  # noqa: E402
from scripts.run_idrid_v2_research import build_model as build_lesion_aware_model  # noqa: E402
from scripts.train_classifier import make_transforms  # noqa: E402

SPLIT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "dr_training_split.json"
SELECTED = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_grading_selected_candidate.json"
DATA_AUDIT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_grading_data_audit.json"
V3_REPRO = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v3_reproducibility.json"
V3_ROBUST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v3_robustness.json"
RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
FINAL_DIR = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "final"
META_ROOT = ROOT / "ml" / "datasets" / "metadata" / "idrid"
FINAL_CHECKPOINT = FINAL_DIR / "checkpoint_best.pt"
FINAL_MANIFEST = FINAL_DIR / "model_manifest.json"
FINAL_TRAINING_CONFIG = FINAL_DIR / "training_config.json"
FINAL_SHA_FILE = FINAL_DIR / "checkpoint_best.pt.sha256"
FINAL_REPRO = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_grading_reproducibility.json"
FINAL_ROBUST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_final_grading_robustness.json"

EXPECTED_V3_SHA = "97bd6a9933ca3d09c1e41bb16ba7452b5988eb1818fc7f1c87054471c4c33049"
THRESHOLD = 0.40


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
    payload = torch.load(FINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    if any(key.startswith("lesion_head.") or key.startswith("structure_head.") for key in state):
        import torch.nn as nn
        model = build_lesion_aware_model(torch, nn)()
        architecture = "EfficientNet-B0 shared backbone with severity, hierarchical, lesion-presence, and optic-disc heads; severity head is authoritative"
    else:
        model = build_classifier("efficientnet_b0", num_classes=5, pretrained=False, ordinal_mode=False)
        architecture = "EfficientNet-B0 hierarchical classifier; severity head is authoritative"
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Final checkpoint load mismatch: {result}")
    model.eval()
    return model, payload, architecture


def image_records() -> list[dict[str, Any]]:
    manifest = json.loads(SPLIT.read_text(encoding="utf-8"))
    if manifest.get("official_test_images_opened") not in (None, 0) or manifest.get("official_test_used"):
        raise RuntimeError("Development reproducibility cannot proceed: official-test isolation flag failed")
    records = [r for r in manifest.get("records", []) if r.get("split") == "validation"]
    if len(records) != 83:
        raise RuntimeError(f"Expected 83 IDRiD development validation records, found {len(records)}")
    return sorted(records, key=lambda row: row["image"])


def infer_records(model: Any, records: list[dict[str, Any]], transform: Any, torch: Any) -> list[dict[str, Any]]:
    rows = []
    with torch.inference_mode():
        for record in records:
            path = RAW / record["image"]
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            output = model(tensor)
            logits = output["severity_logits"][0].detach().cpu().numpy().astype(float)
            probabilities = torch.softmax(output["severity_logits"], dim=1)[0].detach().cpu().numpy().astype(float)
            vector = probabilities.tolist()
            rows.append({"image_id": record["image_id"], "actual": int(record["label"]), "predicted": int(np.argmax(vector)), "probabilities": vector, "logits": logits.tolist(), "confidence": float(max(vector)), "referable_probability": float(sum(vector[2:5])), "referable": bool(sum(vector[2:5]) >= THRESHOLD)})
    return rows


def perturbations(image: Image.Image) -> dict[str, Image.Image]:
    output: dict[str, Image.Image] = {
        "brightness_minus": ImageEnhance.Brightness(image).enhance(0.90),
        "brightness_plus": ImageEnhance.Brightness(image).enhance(1.10),
        "contrast_minus": ImageEnhance.Contrast(image).enhance(0.90),
        "contrast_plus": ImageEnhance.Contrast(image).enhance(1.10),
        "mild_blur": image.filter(ImageFilter.GaussianBlur(radius=0.6)),
        "illumination_change": ImageEnhance.Color(ImageEnhance.Brightness(image).enhance(1.08)).enhance(0.92),
    }
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    buffer.seek(0)
    output["jpeg_compression"] = Image.open(buffer).convert("RGB")
    width, height = image.size
    margin_x, margin_y = int(width * 0.03), int(height * 0.03)
    crop = image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
    output["small_fov_crop"] = crop
    return output


def robustness(model: Any, records: list[dict[str, Any]], transform: Any, baseline: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    per_type: dict[str, list[dict[str, float]]] = {}
    with torch.inference_mode():
        for name in ("brightness_minus", "brightness_plus", "contrast_minus", "contrast_plus", "mild_blur", "jpeg_compression", "small_fov_crop", "illumination_change"):
            values = []
            for index, record in enumerate(records):
                with Image.open(RAW / record["image"]) as image:
                    altered = perturbations(image.convert("RGB"))[name]
                    tensor = transform(altered).unsqueeze(0)
                output = model(tensor)
                probability = torch.softmax(output["severity_logits"], dim=1)[0].detach().cpu().numpy().astype(float)
                base = baseline[index]
                values.append({"grade_stable": float(int(int(np.argmax(probability)) == base["predicted"])), "referable_stable": float(int(bool(float(sum(probability[2:5])) >= THRESHOLD) == base["referable"])), "confidence_abs_delta": abs(float(probability.max()) - base["confidence"]), "probability_l1_delta": float(np.abs(probability - np.asarray(base["probabilities"])).sum())})
            per_type[name] = values
    summary = {}
    for name, values in per_type.items():
        summary[name] = {"sample_count": len(values), "grade_stability": float(np.mean([v["grade_stable"] for v in values])), "referable_stability": float(np.mean([v["referable_stable"] for v in values])), "mean_confidence_abs_delta": float(np.mean([v["confidence_abs_delta"] for v in values])), "mean_probability_l1_delta": float(np.mean([v["probability_l1_delta"] for v in values]))}
    return {"status": "COMPLETED", "method": "fixed validation perturbations; no threshold or weight changes", "perturbations": summary, "clinical_claim": False}


def main() -> int:
    import torch
    if not SELECTED.is_file():
        raise SystemExit("Development comparison is missing; run scripts/run_idrid_final_grading.py first")
    selected_payload = json.loads(SELECTED.read_text(encoding="utf-8"))
    selection = selected_payload.get("selection", selected_payload)
    selected = selected_payload.get("selected_candidate", selected_payload)
    chosen = selection["selected_candidate"] if isinstance(selection.get("selected_candidate"), str) else selected.get("candidate")
    selected_checkpoint = ROOT / selection["selected_checkpoint_path"]
    if chosen != "a":
        raise SystemExit("This finalizer currently supports the selected V3 control only; no checkpoint was frozen")
    if sha256(selected_checkpoint) != EXPECTED_V3_SHA:
        raise SystemExit("Selected V3 control SHA changed; refusing to freeze")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_checkpoint, FINAL_CHECKPOINT)
    final_sha = sha256(FINAL_CHECKPOINT)
    if final_sha != EXPECTED_V3_SHA:
        raise SystemExit("Copied final checkpoint SHA mismatch")
    architecture = "EfficientNet-B0 shared backbone with severity, hierarchical, lesion-presence, and optic-disc heads; severity head is authoritative"
    if FINAL_REPRO.is_file() and FINAL_ROBUST.is_file() and FINAL_MANIFEST.is_file():
        repro = json.loads(FINAL_REPRO.read_text(encoding="utf-8"))
        robust = json.loads(FINAL_ROBUST.read_text(encoding="utf-8"))
        architecture = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8")).get("architecture", architecture)
    else:
        model, payload, architecture = load_model(torch)
        records = image_records()
        _, validation_transform = make_transforms(224)
        first = infer_records(model, records, validation_transform, torch)
        second = infer_records(model, records, validation_transform, torch)
        prediction_identical = [row["predicted"] for row in first] == [row["predicted"] for row in second]
        probability_delta = max(float(np.max(np.abs(np.asarray(a["probabilities"]) - np.asarray(b["probabilities"])))) for a, b in zip(first, second))
        repro = {"status": "PASS" if prediction_identical and probability_delta <= 1e-6 else "FAIL", "validation_count": len(records), "predictions_identical": prediction_identical, "probabilities_identical_within_1e-6": probability_delta <= 1e-6, "max_probability_abs_delta": probability_delta, "checkpoint_sha256": final_sha, "official_test_images_opened": 0, "preprocessing": "RGB -> Resize(224,224) -> ToTensor -> ImageNet mean/std", "source_reproducibility_artifact": rel(V3_REPRO) if V3_REPRO.is_file() else None}
        dump(FINAL_REPRO, repro)
        robust = robustness(model, records, validation_transform, first, torch)
        robust["source_v3_robustness_artifact"] = rel(V3_ROBUST) if V3_ROBUST.is_file() else None
        dump(FINAL_ROBUST, robust)
    selected_metrics = selected["metrics"]
    manifest = {
        "model_version": "idrid-final-development-candidate-a-v3-control-20260912",
        "production_promoted": False,
        "clinical_validation_claim": False,
        "checkpoint": rel(FINAL_CHECKPOINT),
        "checkpoint_sha256": final_sha,
        "architecture": architecture,
        "backbone": "efficientnet_b0",
        "input_size": 224,
        "input_channels": 3,
        "color_space": "RGB",
        "preprocessing": {"resize": [224, 224], "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225], "inference_transform": "Resize -> ToTensor -> ImageNet normalization"},
        "class_mapping": {"0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative DR"},
        "training_split": {"dataset": "IDRiD", "train_count": 323, "validation_count": 83, "official_test_count": 103, "official_test_status": "FROZEN_AND_NOT_EVALUATED_AT_FREEZE"},
        "excluded_records": json.loads(SPLIT.read_text(encoding="utf-8")).get("excluded_records", []),
        "loss": "V3 domain-robust hierarchical training loss; severity head is authoritative",
        "optimizer": "AdamW",
        "learning_rate": 1e-5,
        "batch_size": 32,
        "seed": 20260912,
        "best_epoch": 1,
        "validation_metrics": selected_metrics,
        "referable_definition": "Referable DR = grades 2, 3, or 4",
        "referable_probability": "P(2)+P(3)+P(4)",
        "referable_threshold": THRESHOLD,
        "severity_decoder": "argmax(P0..P4), independent of referable decision",
        "calibration_status": "UNCALIBRATED; raw softmax confidence is not clinically calibrated",
        "reproducibility_artifact": rel(FINAL_REPRO),
        "robustness_artifact": rel(FINAL_ROBUST),
        "known_limitations": ["Patient identifiers are unavailable; leakage control is image/duplicate based.", "Development validation has only 83 images and four Grade 1 examples.", "No calibration subset was fitted without materially weakening model selection.", "This is a research candidate and is not production promoted or clinically validated."],
        "official_test_images_opened": 0,
        "freeze_status": "FROZEN_BEFORE_OFFICIAL_TEST",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    dump(FINAL_MANIFEST, manifest)
    dump(FINAL_TRAINING_CONFIG, {
        "model_version": manifest["model_version"],
        "architecture": manifest["architecture"],
        "backbone": manifest["backbone"],
        "input_size": manifest["input_size"],
        "preprocessing": manifest["preprocessing"],
        "training_split": manifest["training_split"],
        "loss": manifest["loss"],
        "optimizer": manifest["optimizer"],
        "learning_rate": manifest["learning_rate"],
        "batch_size": manifest["batch_size"],
        "seed": manifest["seed"],
        "referable_rule": manifest["referable_probability"],
        "referable_threshold": manifest["referable_threshold"],
        "calibration_status": manifest["calibration_status"],
        "production_promoted": False,
    })
    FINAL_SHA_FILE.write_text(final_sha + "\n", encoding="utf-8")
    dump(META_ROOT / "idrid_final_grading_selected_candidate.json", {**selected_payload, "final_freeze": manifest, "production_promoted": False, "official_test_status": "FROZEN_NOT_YET_EVALUATED"})
    print(json.dumps({"freeze_status": manifest["freeze_status"], "checkpoint": manifest["checkpoint"], "sha256": final_sha, "reproducibility": repro, "robustness": robust}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
