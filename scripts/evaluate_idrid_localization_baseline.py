"""Evaluate the existing classical optic-disc/fovea evidence baseline.

The baseline is the current bright-region optic-disc localizer plus its
optic-disc-relative fovea approximation. It is measured only on the 412-image
development pool after cross-split duplicate exclusion.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from ml.localization.idrid import LANDMARKS, localization_metrics, load_rgb  # noqa: E402
from app.ml.evidence.service import RetinalEvidenceService  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    manifest = json.loads((META / "idrid_localization_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((META / "idrid_localization_split.json").read_text(encoding="utf-8"))
    development_ids = {item["image_id"] for item in split["records"]}
    records = [record for record in manifest["records"] if record["split"] == "train" and record["image_id"] in development_ids]
    service = RetinalEvidenceService(max_dimension=768, enable_heuristics=True, model_adapters={}, enable_vessel_baseline=False)
    rows = []
    for index, record in enumerate(records, start=1):
        image = load_rgb(ROOT / record["image_path"])
        scale = min(1.0, 768.0 / max(image.shape[:2]))
        if scale < 1.0:
            working = np.asarray(Image.fromarray(image, mode="RGB").resize((round(image.shape[1] * scale), round(image.shape[0] * scale)), Image.Resampling.LANCZOS), dtype=np.uint8)
        else:
            working = image
        gray = np.asarray(np.dot(working[..., :3], [0.299, 0.587, 0.114]), dtype=np.float32)
        retina_mask = service._retina_mask(gray)
        disc_mask, disc = service._optic_disc(gray, retina_mask)
        predicted = None
        confidence = {name: None for name in LANDMARKS}
        if disc is not None:
            disc_point = [float(disc["x"] / scale), float(disc["y"] / scale)]
            fovea_point = [float((disc["x"] + disc["radius"] * 2.2) / scale), float(disc["y"] / scale)]
            predicted = [disc_point, fovea_point]
            confidence["optic_disc"] = float(disc.get("brightness_score", 0.0))
            confidence["fovea"] = 0.1
        ground_truth = [[record["annotations"][name]["x"], record["annotations"][name]["y"]] for name in LANDMARKS]
        rows.append({"image_id": record["image_id"], "predicted": predicted, "ground_truth": ground_truth, "confidence": confidence, "image_width": record["image"]["width"], "image_height": record["image"]["height"], "failure_reason": None if predicted is not None else "no bright optic-disc candidate"})
        if index % 50 == 0:
            print(f"baseline localization {index}/{len(records)}", flush=True)
    report = {
        "schema_version": "idrid-localization-baseline-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "classical-cv-optic-disc-relative-fovea-baseline",
        "model_type": "existing_evidence_service_baseline",
        "development_images": len(records),
        "metrics": localization_metrics(rows),
        "per_image": rows,
        "official_test_images_opened": 0,
        "production_promoted": False,
        "clinical_validation_claim": False,
        "note": "Baseline supports comparison only; it is not a trained landmark model and is not clinical validation.",
    }
    dump(META / "idrid_localization_baseline.json", report)
    print(json.dumps({"development_images": len(records), "metrics": report["metrics"], "official_test_images_opened": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
