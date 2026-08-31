"""Evaluate the real R2-V2 ``bv`` vessel model on discovered DRIVE labels."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.evidence.vessel_model import PretrainedRetinalVesselAdapter  # noqa: E402
from ml.evaluation.drive import group_drive_files, segmentation_metrics  # noqa: E402


METRIC_KEYS = ("dice", "iou", "pixel_accuracy", "sensitivity", "specificity", "precision", "recall", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="ml/datasets/raw/drive")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="ml/evaluation/drive")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=1, help="Parallel model workers; use only when memory permits")
    parser.add_argument("--torch-threads", type=int, default=4, help="Torch CPU threads per worker")
    parser.add_argument("--include-unlabeled-test-inference", action="store_true", help="Also run inference on test images without reporting accuracy metrics")
    return parser.parse_args()


def _resolve(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _save_comparison(output: Path, image: np.ndarray, fov: np.ndarray, target: np.ndarray, probability: np.ndarray, threshold: float, title: str) -> None:
    predicted = (probability >= threshold) & fov
    target = target & fov
    true_positive = target & predicted
    false_positive = ~target & predicted & fov
    false_negative = target & ~predicted
    base = Image.fromarray(image, mode="RGB").convert("RGBA")
    rgba = np.zeros((*target.shape, 4), dtype=np.uint8)
    rgba[true_positive, :3] = (0, 220, 100)
    rgba[false_positive, :3] = (245, 80, 70)
    rgba[false_negative, :3] = (255, 190, 0)
    rgba[target | predicted, 3] = 155
    overlay = Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")
    gt = np.where(target, 255, 0).astype(np.uint8)
    pred = np.where(predicted, 255, 0).astype(np.uint8)
    panels = [Image.fromarray(image), Image.fromarray(gt, mode="L").convert("RGB"), Image.fromarray(pred, mode="L").convert("RGB"), overlay]
    panel_width, panel_height = image.shape[1], image.shape[0]
    canvas = Image.new("RGB", (panel_width * 2, panel_height * 2 + 34), "white")
    labels = ["Original", "Ground truth", "R2-V2 prediction", "Overlay TP green / FP red / FN amber"]
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = (index % 2) * panel_width
        y = (index // 2) * panel_height
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + panel_width - 1, y + panel_height - 1), outline=(80, 80, 80), width=2)
        draw.rectangle((x, y, x + min(panel_width, len(label) * 8 + 12), y + 22), fill=(255, 255, 255))
        draw.text((x + 6, y + 4), label, fill=(20, 20, 20), font=ImageFont.load_default())
    draw.text((8, panel_height * 2 + 8), title, fill=(20, 20, 20), font=ImageFont.load_default())
    canvas.save(output, format="PNG", optimize=True)


_WORKER_ADAPTER: PretrainedRetinalVesselAdapter | None = None
_WORKER_COMPARISON_ROOT: Path | None = None
_WORKER_RAW_ROOT: Path | None = None
_WORKER_THRESHOLD = 0.5


def _init_worker(model_path: str, device: str, threshold: float, torch_threads: int, raw_root: str, comparison_root: str) -> None:
    global _WORKER_ADAPTER, _WORKER_COMPARISON_ROOT, _WORKER_RAW_ROOT, _WORKER_THRESHOLD
    import torch

    torch.set_num_threads(max(1, int(torch_threads)))
    _WORKER_ADAPTER = PretrainedRetinalVesselAdapter(model_path=model_path, device=device, threshold=threshold, version="r2-v2-bv-2025")
    _WORKER_COMPARISON_ROOT = Path(comparison_root)
    _WORKER_RAW_ROOT = Path(raw_root)
    _WORKER_THRESHOLD = threshold


def _evaluate_labeled_pair(task: tuple[str, str, str, str]) -> dict[str, Any]:
    if _WORKER_ADAPTER is None or _WORKER_RAW_ROOT is None or _WORKER_COMPARISON_ROOT is None:
        raise RuntimeError("R2-V2 worker was not initialized")
    specimen, image_relative, vessel_relative, fov_relative = task
    image_path = _WORKER_RAW_ROOT / image_relative
    vessel_path = _WORKER_RAW_ROOT / vessel_relative
    fov_path = _WORKER_RAW_ROOT / fov_relative
    image = _read_rgb(image_path)
    target = _read_mask(vessel_path)
    fov = _read_mask(fov_path)
    result = _WORKER_ADAPTER.analyze(image, {"drive_specimen_id": specimen})
    if not result.supported or not result.probability_map_data_uri:
        raise RuntimeError(f"R2-V2 inference failed for DRIVE specimen {specimen}: {result.to_dict()}")
    probability = _WORKER_ADAPTER.predict_probability(image)
    if probability.shape != target.shape or fov.shape != target.shape:
        raise RuntimeError(f"Shape mismatch after R2-V2 inference for {specimen}: probability={probability.shape}, target={target.shape}, fov={fov.shape}")
    metrics = segmentation_metrics(target, probability, fov, _WORKER_THRESHOLD)
    row = {"specimen_id": specimen, "split": "training", "image": image_relative, "vessel_mask": vessel_relative, "field_of_view_mask": fov_relative, "model_status": result.status, "model_confidence": result.confidence, "model_metadata": result.metadata, **metrics}
    _save_comparison(_WORKER_COMPARISON_ROOT / f"{specimen}_comparison.png", image, fov, target, probability, _WORKER_THRESHOLD, f"DRIVE {specimen} | Dice {metrics['dice']:.4f} | IoU {metrics['iou']:.4f}")
    return row


def _aggregate(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(per_image),
        "mean": {key: float(mean(float(item[key]) for item in per_image)) for key in METRIC_KEYS},
        "std": {key: float(pstdev(float(item[key]) for item in per_image)) for key in METRIC_KEYS},
        "micro_confusion": {key: int(sum(int(item[key]) for item in per_image)) for key in ("true_positive", "true_negative", "false_positive", "false_negative", "evaluated_pixels", "ground_truth_vessel_pixels", "predicted_vessel_pixels")},
    }


def _update_registry(report: dict[str, Any]) -> None:
    registry_path = ROOT / "ml" / "model_registry.json"
    if not registry_path.is_file():
        return
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for artifact in registry.get("artifacts", []):
        if artifact.get("model_version") != report["model"]["model_version"]:
            continue
        artifact["evaluation"] = {
            "dataset": "DRIVE",
            "dataset_version": report["dataset"]["dataset_version"],
            "ground_truth_scope": report["dataset"]["ground_truth_scope"],
            "threshold": report["evaluation"]["threshold"],
            "metrics": report["evaluation"]["aggregate"],
            "clinical_validation_claim": False,
            "report": "ml/evaluation/drive/r2-v2-evaluation.json",
        }
        artifact["note"] = "Real model-backed vessel evidence. DRIVE metrics are measured engineering segmentation results only; no clinical validation claim."
        break
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    runtime_registry = ROOT / "ml" / "weights" / "model_registry.json"
    if runtime_registry.is_file():
        runtime = json.loads(runtime_registry.read_text(encoding="utf-8"))
        for artifact in runtime.get("artifacts", []):
            if artifact.get("model_version") == report["model"]["model_version"]:
                artifact["evaluation"] = {
                    "dataset": "DRIVE",
                    "dataset_version": report["dataset"]["dataset_version"],
                    "ground_truth_scope": report["dataset"]["ground_truth_scope"],
                    "threshold": report["evaluation"]["threshold"],
                    "metrics": report["evaluation"]["aggregate"],
                    "clinical_validation_claim": False,
                    "report": "ml/evaluation/drive/r2-v2-evaluation.json",
                }
                break
        runtime_registry.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_root = _resolve(args.raw_dir, ROOT / "ml" / "datasets" / "raw" / "drive")
    output_root = _resolve(args.output_dir, ROOT / "ml" / "evaluation" / "drive")
    model_path = _resolve(args.model_path, ROOT / "ml" / "weights" / "vessel_segmentation" / "r2-v2-bv-2025" / "bv.safetensors")
    validation_path = output_root / "validation_report.json"
    if not validation_path.is_file():
        raise SystemExit("DRIVE validation_report.json is missing. Run scripts/validate_drive_dataset.py first.")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("validation", {}).get("status") != "VALID":
        raise SystemExit("DRIVE validation did not pass; R2-V2 evaluation was not started.")
    grouped = group_drive_files(raw_root)
    pairs = []
    inference_only = []
    for specimen, categories in grouped.items():
        images = categories.get("image", [])
        if len(images) != 1:
            continue
        image = images[0]
        fovs = categories.get("fov_mask", [])
        vessels = categories.get("vessel_mask", [])
        if len(fovs) != 1:
            continue
        if len(vessels) == 1:
            pairs.append((specimen, image, vessels[0], fovs[0]))
        else:
            inference_only.append((specimen, image, fovs[0]))
    if not pairs:
        raise SystemExit("No image/manual-vessel/FOV pairs found; no metrics were fabricated.")
    adapter = PretrainedRetinalVesselAdapter(model_path=model_path, device=args.device, threshold=args.threshold, version="r2-v2-bv-2025")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_root = output_root / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (specimen, image.relative_to(raw_root).as_posix(), vessel.relative_to(raw_root).as_posix(), fov.relative_to(raw_root).as_posix())
        for specimen, image, vessel, fov in pairs
    ]
    per_image: list[dict[str, Any]] = []
    if args.workers == 1:
        _init_worker(str(model_path), args.device, args.threshold, args.torch_threads, str(raw_root), str(comparison_root))
        for index, task in enumerate(tasks, start=1):
            row = _evaluate_labeled_pair(task)
            per_image.append(row)
            print(f"[{index}/{len(tasks)}] {row['specimen_id']}: Dice={row['dice']:.6f} IoU={row['iou']:.6f}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(model_path), args.device, args.threshold, args.torch_threads, str(raw_root), str(comparison_root)),
        ) as executor:
            futures = [executor.submit(_evaluate_labeled_pair, task) for task in tasks]
            for index, future in enumerate(futures, start=1):
                row = future.result()
                per_image.append(row)
                print(f"[{index}/{len(tasks)}] {row['specimen_id']}: Dice={row['dice']:.6f} IoU={row['iou']:.6f}", flush=True)
    inference_records: list[dict[str, Any]] = []
    if args.include_unlabeled_test_inference:
        for specimen, image_path, fov_path in inference_only:
            _init_worker(str(model_path), args.device, args.threshold, args.torch_threads, str(raw_root), str(comparison_root))
            image = _read_rgb(image_path)
            fov = _read_mask(fov_path)
            result = _WORKER_ADAPTER.analyze(image, {"drive_specimen_id": specimen})
            if not result.supported or not result.probability_map_data_uri:
                raise RuntimeError(f"R2-V2 inference failed for inference-only DRIVE specimen {specimen}: {result.to_dict()}")
            probability = _WORKER_ADAPTER.predict_probability(image)
            if probability.shape != fov.shape:
                raise RuntimeError(f"Shape mismatch after R2-V2 inference for test specimen {specimen}: probability={probability.shape}, fov={fov.shape}")
            inference_records.append({"specimen_id": specimen, "split": "test", "image": image_path.relative_to(raw_root).as_posix(), "field_of_view_mask": fov_path.relative_to(raw_root).as_posix(), "model_status": result.status, "model_confidence": result.confidence, "predicted_vessel_coverage_inside_fov": float(((probability >= args.threshold) & fov).sum() / max(1, fov.sum())), "ground_truth_available": False})
            print(f"[inference-only] {specimen}: no manual vessel ground truth; metrics omitted", flush=True)
    per_image_sorted = sorted(per_image, key=lambda item: float(item["dice"]))
    checksum = adapter._checksum()
    report = {
        "report_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": "DRIVE", "dataset_version": validation["dataset_version"], "manifest": "ml/evaluation/drive/dataset_manifest.json", "validation_report": "ml/evaluation/drive/validation_report.json", "ground_truth_scope": {"training_images_evaluated": len(per_image), "test_images_without_ground_truth": len(inference_only), "test_images_inferred_without_metrics": len(inference_records), "manual_vessel_masks": validation["counts"]["manual_vessel_masks"], "field_of_view_masks_applied": True, "limitation": "The discovered test split has no manual vessel ground-truth masks; no test accuracy metrics are reported."}},
        "model": {"model_name": adapter.name, "model_version": adapter.version, "architecture": "RRWNet (R2-V2 bv variant)", "checkpoint": "ml/weights/vessel_segmentation/r2-v2-bv-2025/bv.safetensors", "checkpoint_sha256": checksum, "source": "https://huggingface.co/j-morano/R2-V2", "source_code": "https://github.com/j-morano/R2-V2", "license": "CC BY 4.0", "vessel_channel": 2, "clinical_validation_claim": False},
        "evaluation": {"threshold": args.threshold, "device": adapter.device_name, "ground_truth_mask_threshold": 128, "aggregate": _aggregate(per_image), "best_performing_examples": per_image_sorted[-3:][::-1], "worst_performing_examples": per_image_sorted[:3], "per_image": per_image, "inference_only": inference_records, "comparison_directory": "ml/evaluation/drive/comparisons", "metrics_note": "All reported metrics are pixel-level engineering measurements within the genuine DRIVE FOV masks; they are not clinical validation results."},
    }
    (output_root / "r2-v2-evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _update_registry(report)
    print(json.dumps({"status": "completed", "dataset_version": report["dataset"]["dataset_version"], "model_version": report["model"]["model_version"], "checkpoint_sha256": checksum, "ground_truth_images": len(per_image), "inference_only_images": len(inference_records), "mean_metrics": report["evaluation"]["aggregate"]["mean"], "std_metrics": report["evaluation"]["aggregate"]["std"], "best": [item["specimen_id"] for item in report["evaluation"]["best_performing_examples"]], "worst": [item["specimen_id"] for item in report["evaluation"]["worst_performing_examples"]]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"R2-V2 DRIVE EVALUATION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
