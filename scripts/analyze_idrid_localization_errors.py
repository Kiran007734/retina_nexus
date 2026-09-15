"""Create development-only localization error analysis from CV predictions."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.localization.idrid import LANDMARKS  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
VISUAL_DIR = ROOT / "ml" / "evaluation" / "idrid_localization" / "dev_visuals"


def main() -> int:
    candidate = sys.argv[1] if len(sys.argv) > 1 else "idrid-shared-heatmap-coord-512x352-v1"
    root = ROOT / "ml" / "weights" / "localization" / "idrid" / "cv" / candidate
    rows = []
    for fold_dir in sorted(root.glob("fold_*")):
        path = fold_dir / "validation_predictions.json"
        if path.is_file():
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) < 412:
        raise SystemExit(f"Expected 412 out-of-fold predictions, found {len(rows)}")
    dedup = {row["image_id"]: row for row in rows}
    scored = []
    for row in dedup.values():
        diagonal = max(float(np.hypot(row["image_width"], row["image_height"])), 1.0)
        errors = {landmark: float(np.linalg.norm(np.asarray(row["predicted"][index]) - np.asarray(row["ground_truth"][index]))) for index, landmark in enumerate(LANDMARKS)}
        scored.append({"image_id": row["image_id"], "errors_px": errors, "normalized_errors": {landmark: value / diagonal for landmark, value in errors.items()}, "confidence": row.get("confidence", {}), "ground_truth": row["ground_truth"], "predicted": row["predicted"], "image_width": row["image_width"], "image_height": row["image_height"]})
    overall = sorted(scored, key=lambda row: sum(row["normalized_errors"].values()), reverse=True)
    by_landmark = {landmark: sorted(scored, key=lambda row: row["normalized_errors"][landmark], reverse=True)[:10] for landmark in LANDMARKS}
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    visual_files = []
    for row in overall[:10]:
        record = next(item for item in json.loads((META / "idrid_localization_manifest.json").read_text(encoding="utf-8"))["records"] if item["image_id"] == row["image_id"])
        with Image.open(ROOT / record["image_path"]) as source:
            image = source.convert("RGB")
            image.thumbnail((1200, 800), Image.Resampling.LANCZOS)
            sx, sy = image.width / row["image_width"], image.height / row["image_height"]
            draw = ImageDraw.Draw(image)
            for index, landmark in enumerate(LANDMARKS):
                gt = row["ground_truth"][index]
                pred = row["predicted"][index]
                gx, gy = gt[0] * sx, gt[1] * sy
                px, py = pred[0] * sx, pred[1] * sy
                draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), outline=(0, 255, 0), width=3)
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=(255, 0, 0), width=3)
                draw.line((gx, gy, px, py), fill=(255, 255, 0), width=2)
                draw.text((px + 6, py + 6), f"{landmark} pred", fill=(255, 0, 0))
            output = VISUAL_DIR / f"worst_{row['image_id']}.png"
            image.save(output)
            visual_files.append(str(output.relative_to(ROOT)).replace("\\", "/"))
    payload = {"schema_version": "idrid-localization-error-analysis-1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "candidate": candidate, "development_images": len(dedup), "official_test_images_opened": 0, "worst_overall": overall[:20], "worst_by_landmark": by_landmark, "visual_files": visual_files, "production_promoted": False, "clinical_validation_claim": False}
    (META / "idrid_localization_error_analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": candidate, "development_images": len(dedup), "worst_overall": [item["image_id"] for item in overall[:5]], "official_test_images_opened": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
