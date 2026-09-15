"""Analyze saved IDRiD lesion development predictions.

This command consumes only cross-validation outputs from the official 54-image
training split. It selects thresholds using development data and computes
pixel- and object-level diagnostics; it never reads the official test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.lesions.idrid import LESION_CLASSES, segmentation_metrics  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def components(mask: np.ndarray, minimum_area: int = 2) -> list[dict[str, Any]]:
    import cv2

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        # Keep component masks local to their bounding boxes.  A full-image
        # boolean array per component is needlessly expensive on 768x768
        # outputs and can make object-level diagnostics appear hung.
        local_mask = labels[y : y + height, x : x + width] == index
        result.append({"label": index, "area": area, "mask": local_mask, "x": x, "y": y, "width": width, "height": height, "center_x": round(float(centroids[index][0]), 2), "center_y": round(float(centroids[index][1]), 2)})
    return result


def object_metrics(probabilities: np.ndarray, targets: np.ndarray, availability: np.ndarray, threshold: float, iou_threshold: float = 0.1) -> dict[str, Any]:
    result = {}
    for index, name in enumerate(LESION_CLASSES):
        valid_indices = np.flatnonzero(availability[:, index] > 0.5)
        if valid_indices.size == 0:
            result[name] = {"status": "UNAVAILABLE"}
            continue
        predicted_count = reference_count = matched = false_positive = missed = 0
        for image_index in valid_indices:
            predicted = components(probabilities[image_index, index] >= threshold)
            reference = components(targets[image_index, index] > 0.5)
            predicted_count += len(predicted)
            reference_count += len(reference)
            used = set()
            for truth in reference:
                best = 0.0
                best_index = None
                for candidate_index, candidate in enumerate(predicted):
                    if candidate_index in used:
                        continue
                    x0 = max(truth["x"], candidate["x"])
                    y0 = max(truth["y"], candidate["y"])
                    x1 = min(truth["x"] + truth["width"], candidate["x"] + candidate["width"])
                    y1 = min(truth["y"] + truth["height"], candidate["y"] + candidate["height"])
                    if x1 <= x0 or y1 <= y0:
                        continue
                    truth_slice = truth["mask"][y0 - truth["y"] : y1 - truth["y"], x0 - truth["x"] : x1 - truth["x"]]
                    candidate_slice = candidate["mask"][y0 - candidate["y"] : y1 - candidate["y"], x0 - candidate["x"] : x1 - candidate["x"]]
                    intersection = np.logical_and(truth_slice, candidate_slice).sum()
                    union = truth["area"] + candidate["area"] - intersection
                    score = float(intersection / max(1, union))
                    if score > best:
                        best, best_index = score, candidate_index
                if best_index is not None and best >= iou_threshold:
                    used.add(best_index)
                    matched += 1
                else:
                    missed += 1
            false_positive += max(0, len(predicted) - len(used))
        precision = matched / max(1, matched + false_positive)
        recall = matched / max(1, matched + missed)
        result[name] = {"status": "MEASURED", "iou_match_threshold": iou_threshold, "predicted_lesions": predicted_count, "reference_lesions": reference_count, "matched_lesions": matched, "missed_lesions": missed, "false_positive_lesions": false_positive, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="idrid-unet-seresnext50-768-focaldice-v2")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    root = ROOT / "ml" / "weights" / "lesions" / "idrid" / "cv" / args.candidate
    probabilities, targets, availability = [], [], []
    for fold in range(1, args.folds + 1):
        artifact = root / f"fold_{fold}" / "validation_outputs.npz"
        if not artifact.is_file():
            raise SystemExit(f"Missing development artifact: {artifact}")
        data = np.load(artifact)
        probabilities.append(data["probabilities"])
        targets.append(data["targets"])
        availability.append(data["availability"])
    arrays = {"probabilities": np.concatenate(probabilities), "targets": np.concatenate(targets), "availability": np.concatenate(availability)}
    thresholds = [round(value, 2) for value in np.arange(0.20, 0.71, 0.05)]
    threshold_results = []
    for threshold in thresholds:
        metrics = segmentation_metrics(arrays["probabilities"], arrays["targets"], arrays["availability"], threshold)
        threshold_results.append({"threshold": threshold, "macro_dice": metrics["macro_dice"], "macro_iou": metrics["macro_iou"], "per_class": {name: {key: value for key, value in details.items() if key in {"dice", "iou", "precision", "recall", "specificity", "lesion_positive_detection_rate", "status"}} for name, details in metrics["per_class"].items()}})
    selected = max(threshold_results, key=lambda item: (item["macro_dice"] or 0.0, item["macro_iou"] or 0.0, sum((item["per_class"].get(name, {}).get("recall") or 0.0) for name in LESION_CLASSES)))
    selected_metrics = segmentation_metrics(arrays["probabilities"], arrays["targets"], arrays["availability"], selected["threshold"])
    objects = object_metrics(arrays["probabilities"], arrays["targets"], arrays["availability"], selected["threshold"])
    errors = []
    for index in range(arrays["probabilities"].shape[0]):
        image_score = segmentation_metrics(arrays["probabilities"][index:index + 1], arrays["targets"][index:index + 1], arrays["availability"][index:index + 1], selected["threshold"])
        scores = [details.get("dice") for details in image_score["per_class"].values() if details.get("status") == "MEASURED"]
        errors.append({"oof_index": index, "macro_dice": float(np.mean(scores)) if scores else None, "metrics": image_score})
    errors = sorted(errors, key=lambda item: item["macro_dice"] if item["macro_dice"] is not None else 1.0)[:10]
    generated = datetime.now(timezone.utc).isoformat()
    dump(META / "idrid_lesion_threshold_analysis.json", {"schema_version": "idrid-lesion-threshold-1", "candidate": args.candidate, "thresholds": threshold_results, "selected": selected, "official_test_images_opened": 0, "generated_at_utc": generated})
    dump(META / "idrid_lesion_error_analysis.json", {"schema_version": "idrid-lesion-error-1", "candidate": args.candidate, "selected_threshold": selected["threshold"], "worst_oof_cases": errors, "object_level": objects, "official_test_images_opened": 0, "generated_at_utc": generated})
    print(json.dumps({"candidate": args.candidate, "selected_threshold": selected["threshold"], "development_metrics": selected_metrics, "object_level": objects, "official_test_images_opened": 0}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
