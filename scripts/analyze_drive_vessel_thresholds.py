"""Select a vessel probability threshold from development OOF predictions only."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.vessels.drive import aggregate_metrics, threshold_metrics  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "drive"
CV_ROOT = ROOT / "ml" / "weights" / "vessels" / "drive" / "cv"


def main() -> int:
    candidate = sys.argv[1] if len(sys.argv) > 1 else "scratch-green-focal-dice-512-v1"
    arrays = []
    for fold in range(1, 6):
        path = CV_ROOT / candidate / f"fold_{fold}" / "oof_predictions.npz"
        if not path.is_file():
            raise SystemExit(f"Missing development OOF predictions: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays.append({"image_ids": data["image_ids"].astype(str).tolist(), "probabilities": data["probabilities"], "targets": data["targets"], "fovs": data["fovs"]})
    image_ids = [image_id for part in arrays for image_id in part["image_ids"]]
    probabilities = np.concatenate([part["probabilities"] for part in arrays], axis=0)
    targets = np.concatenate([part["targets"] for part in arrays], axis=0)
    fovs = np.concatenate([part["fovs"] for part in arrays], axis=0)
    if len(image_ids) != 20 or len(set(image_ids)) != 20:
        raise SystemExit(f"Expected one OOF prediction for each of 20 development images, found {len(image_ids)}")
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []
    for threshold in thresholds:
        rows = [threshold_metrics(targets[index], probabilities[index], fovs[index], threshold) for index in range(len(image_ids))]
        aggregate = aggregate_metrics(rows)
        mean_values = aggregate["mean"]
        selection_score = float(np.mean([mean_values["dice"], mean_values["iou"], mean_values["sensitivity"], mean_values["specificity"], mean_values["precision"]]))
        results.append({"threshold": threshold, "metrics": aggregate, "selection_score": selection_score})
    selected = max(results, key=lambda item: item["selection_score"])
    report = {"schema_version": "drive-vessel-threshold-analysis-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "candidate": candidate, "source": "five-fold development out-of-fold predictions", "thresholds": results, "selected": selected, "official_test_images_opened": 0, "production_promoted": False, "note": "Threshold was selected without reading official DRIVE test vessel masks."}
    (META / "drive_threshold_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": candidate, "selected_threshold": selected["threshold"], "selected_mean_metrics": selected["metrics"]["mean"], "official_test_images_opened": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
