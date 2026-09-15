"""Measure the preserved external lesion model on IDRiD development images.

This is a development-only comparison. It reads only the 54 official IDRiD
segmentation training images and does not modify or replace the external model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.lesions.idrid import LESION_CLASSES, load_training_sample, segmentation_metrics  # noqa: E402
from app.ml.evidence.lesion_model import DEFAULT_MODEL_PATH, PretrainedRetinalLesionAdapter  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MANIFEST = META / "idrid_lesion_manifest.json"
OUTPUT = META / "idrid_lesion_external_baseline.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not DEFAULT_MODEL_PATH.is_file():
        raise SystemExit(f"Preserved external lesion checkpoint is missing: {DEFAULT_MODEL_PATH}")
    records = [record for record in json.loads(MANIFEST.read_text(encoding="utf-8"))["records"] if record["split"] == "train"]
    if len(records) != 54:
        raise SystemExit(f"Expected 54 development images, found {len(records)}")
    adapter = PretrainedRetinalLesionAdapter(model_path=DEFAULT_MODEL_PATH, device=args.device, threshold=0.5)
    probabilities, targets, availability, per_image = [], [], [], []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        image, masks, available = load_training_sample(record, args.size, augment=False)
        source_probs = adapter._predict(image)  # noqa: SLF001 - this is a read-only benchmark of the verified adapter.
        probs = source_probs[[4, 3, 2, 1]]
        probabilities.append(probs)
        targets.append(masks)
        availability.append(available)
        metrics = segmentation_metrics(probs[None, ...], masks[None, ...], available[None, ...], 0.5)
        per_image.append({"image_id": record["image_id"], "metrics": metrics})
        print(f"external baseline {index}/{len(records)} {record['image_id']}", flush=True)
    arrays = {"probabilities": np.stack(probabilities), "targets": np.stack(targets), "availability": np.stack(availability)}
    report = {
        "schema_version": "idrid-lesion-external-baseline-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "fundus-lesions-unet-seresnext50-all-v1",
        "checkpoint": str(DEFAULT_MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(DEFAULT_MODEL_PATH),
        "evaluation_split": "IDRiD official segmentation training split used for development comparison only",
        "image_count": len(records),
        "input_size": args.size,
        "threshold": 0.5,
        "class_mapping": {"microaneurysms": 4, "haemorrhages": 3, "hard_exudates": 2, "soft_exudates": 1},
        "metrics": segmentation_metrics(arrays["probabilities"], arrays["targets"], arrays["availability"], 0.5),
        "per_image_metrics": per_image,
        "seconds": round(time.perf_counter() - started, 3),
        "official_test_images_opened": 0,
        "production_promoted": False,
        "clinical_validation_claim": False,
    }
    dump(OUTPUT, report)
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"), "metrics": report["metrics"], "official_test_images_opened": 0}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
