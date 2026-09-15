"""Evaluate the frozen IDRiD lesion model on the official test split once.

This command is intentionally separate from development training and threshold
selection. It refuses to run before a frozen model manifest exists and refuses
to overwrite a completed official evaluation artifact. Missing annotations are
reported as UNAVAILABLE; they are never converted to negative masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.lesions.idrid import (  # noqa: E402
    LESION_CLASSES,
    build_idrid_model,
    image_tensor,
    load_training_sample,
    segmentation_metrics,
)
from scripts.analyze_idrid_lesion_cv import object_metrics  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MANIFEST_PATH = META / "idrid_lesion_manifest.json"
MODEL_DIR = ROOT / "ml" / "weights" / "lesions" / "idrid"
MODEL_MANIFEST = MODEL_DIR / "model_manifest.json"
CHECKPOINT = MODEL_DIR / "checkpoint_best.pt"
OUTPUT_DIR = ROOT / "ml" / "evaluation" / "idrid_lesions"
REPORT_PATH = META / "idrid_lesion_official_test.json"


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _class_stats(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in LESION_CLASSES:
        values: list[float] = []
        for record in per_image:
            detail = record["metrics"]["per_class"].get(name, {})
            if detail.get("status") == "MEASURED" and detail.get("dice") is not None:
                values.append(float(detail["dice"]))
        output[name] = {
            "status": "MEASURED" if values else "UNAVAILABLE",
            "support_images": len(values),
            "dice_mean": _finite(np.mean(values)) if values else None,
            "dice_std": _finite(np.std(values)) if values else None,
            "dice_min": _finite(np.min(values)) if values else None,
            "dice_max": _finite(np.max(values)) if values else None,
        }
    measured = [item["dice_mean"] for item in output.values() if item["status"] == "MEASURED"]
    return {"per_class": output, "macro_dice_mean": _finite(np.mean(measured)) if measured else None}


def _save_visual(record: dict[str, Any], image: np.ndarray, targets: np.ndarray, probabilities: np.ndarray, threshold: float, path: Path) -> None:
    """Save a compact prediction/ground-truth comparison, not a clinical claim."""
    base = Image.fromarray(image).convert("RGB")
    overlay = np.asarray(base).copy()
    prediction = np.any(probabilities >= threshold, axis=0)
    truth = np.any(targets > 0.5, axis=0)
    overlay[truth & ~prediction] = (0, 210, 70)       # ground truth only
    overlay[prediction & ~truth] = (220, 40, 40)       # prediction only
    overlay[prediction & truth] = (235, 190, 30)      # overlap
    panel = Image.new("RGB", (base.width * 2, base.height + 30), "white")
    panel.paste(base, (0, 30))
    panel.paste(Image.fromarray(overlay), (base.width, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{record['image_id']} original", fill="black")
    draw.text((base.width + 8, 8), "green=GT red=prediction gold=overlap", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    if REPORT_PATH.is_file():
        previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        if previous.get("evaluation_completed") is True:
            raise SystemExit(f"Official test evaluation already completed; refusing a second run: {REPORT_PATH}")
    if not MODEL_MANIFEST.is_file() or not CHECKPOINT.is_file():
        raise SystemExit("Frozen IDRiD lesion model is missing. Run scripts/finalize_idrid_lesion.py before official evaluation.")

    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if model_manifest.get("production_promoted") is not False:
        raise SystemExit("Official evaluation requires production_promoted=false in the frozen model manifest.")
    if int(model_manifest.get("official_test_images_opened", 0)) != 0:
        raise SystemExit("Frozen manifest already records official test access; refusing evaluation.")
    actual_sha = checkpoint_sha256(CHECKPOINT)
    if actual_sha != model_manifest.get("checkpoint_sha256"):
        raise SystemExit("Checkpoint SHA-256 does not match the frozen model manifest.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    test_records = [record for record in manifest.get("records", []) if record.get("split") == "test"]
    if len(test_records) != 27:
        raise SystemExit(f"Expected exactly 27 official IDRiD segmentation test records, found {len(test_records)}")
    threshold = float(model_manifest["threshold"])

    import torch

    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable in the current PyTorch runtime")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model, _transfer = build_idrid_model(None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device)
    model.eval()

    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    per_image: list[dict[str, Any]] = []
    raw_for_visuals: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    with torch.inference_mode():
        for record in test_records:
            image, masks, available = load_training_sample(record, int(model_manifest["input_size"]), augment=False)
            output = model(image_tensor(image).unsqueeze(0).to(args.device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy().astype(np.float32)
            targets.append(masks.astype(np.float32))
            probabilities.append(probs)
            availability.append(available.astype(np.float32))
            metrics = segmentation_metrics(probs[None, ...], masks[None, ...], available[None, ...], threshold)
            per_image.append({"image_id": record["image_id"], "metrics": metrics, "available_classes": [LESION_CLASSES[i] for i, value in enumerate(available) if value > 0.5]})
            raw_for_visuals.append((record, image, masks, probs))

    arrays = {"probabilities": np.stack(probabilities), "targets": np.stack(targets), "availability": np.stack(availability)}
    aggregate = segmentation_metrics(arrays["probabilities"], arrays["targets"], arrays["availability"], threshold)
    object_level = object_metrics(arrays["probabilities"], arrays["targets"], arrays["availability"], threshold)
    ranked = []
    for index, record in enumerate(per_image):
        values = [detail.get("dice") for detail in record["metrics"]["per_class"].values() if detail.get("status") == "MEASURED" and detail.get("dice") is not None]
        ranked.append((float(np.mean(values)) if values else None, index))
    ranked = sorted(ranked, key=lambda item: item[0] if item[0] is not None else 1.0)
    visuals: list[dict[str, Any]] = []
    selected_indices = []
    if ranked:
        selected_indices.append(ranked[0][1])
        selected_indices.append(ranked[-1][1])
    if len(ranked) > 2:
        selected_indices.append(ranked[len(ranked) // 2][1])
    for index in dict.fromkeys(selected_indices):
        record, image, target, probs = raw_for_visuals[index]
        visual_path = OUTPUT_DIR / "visuals" / f"{record['image_id']}_comparison.png"
        _save_visual(record, image, target, probs, threshold, visual_path)
        visuals.append({"image_id": record["image_id"], "path": str(visual_path.relative_to(ROOT)).replace("\\", "/"), "mean_dice": ranked[[item[1] for item in ranked].index(index)][0]})

    report = {
        "schema_version": "idrid-lesion-official-test-1",
        "evaluation_completed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_manifest.get("model_version"),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": actual_sha,
        "evaluation_dataset": "IDRiD official segmentation testing split",
        "test_image_count": len(test_records),
        "input_size": model_manifest.get("input_size"),
        "threshold": threshold,
        "fov_masks": "UNAVAILABLE; no official IDRiD FOV mask was present",
        "aggregate_metrics": aggregate,
        "per_class_distribution": _class_stats(per_image),
        "object_level_metrics": object_level,
        "per_image_metrics": per_image,
        "best_and_worst_examples": visuals,
        "official_test_images_opened": len(test_records),
        "threshold_tuning_on_test": False,
        "production_promoted": False,
        "clinical_validation_claim": False,
        "known_limitations": [
            "The official split has no FOV masks in the downloaded package.",
            "Soft-exudate ground truth is unavailable for some official test images and those records are excluded from that class metric.",
            "This is a research evaluation on a small dataset and is not clinical validation.",
        ],
    }
    dump(REPORT_PATH, report)
    print(json.dumps({"report": str(REPORT_PATH.relative_to(ROOT)).replace("\\", "/"), "aggregate_metrics": aggregate, "object_level_metrics": object_level, "official_test_images_opened": len(test_records), "production_promoted": False}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
