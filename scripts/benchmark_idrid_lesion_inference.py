"""Measure frozen IDRiD lesion adapter latency on development images only."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.evidence.idrid_lesion_model import IDRiDLesionAdapter  # noqa: E402
from ml.lesions.idrid import image_path, load_rgb  # noqa: E402

META = ROOT / "ml" / "datasets" / "metadata" / "idrid"
MODEL = ROOT / "ml" / "weights" / "lesions" / "idrid" / "checkpoint_best.pt"
OUTPUT = META / "idrid_lesion_inference_benchmark.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ids = ["IDRiD_01", "IDRiD_02", "IDRiD_03", "IDRiD_04", "IDRiD_05"]
    adapter = IDRiDLesionAdapter()
    images = [load_rgb(image_path(image_id, "train")) for image_id in ids]
    cold_start = time.perf_counter()
    adapter._predict(images[0])  # noqa: SLF001 - benchmark the real adapter.
    cold_ms = (time.perf_counter() - cold_start) * 1000.0
    warm_ms = []
    for image in images[1:] + [images[0]]:
        started = time.perf_counter()
        adapter._predict(image)  # noqa: SLF001
        warm_ms.append((time.perf_counter() - started) * 1000.0)
    report = {
        "schema_version": "idrid-lesion-inference-benchmark-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": adapter.version,
        "checkpoint_sha256": sha256(MODEL),
        "device": adapter.device_name,
        "development_image_ids": ids,
        "cold_start_ms": round(cold_ms, 3),
        "warm_inference_ms": [round(value, 3) for value in warm_ms],
        "warm_mean_ms": round(statistics.mean(warm_ms), 3),
        "warm_median_ms": round(statistics.median(warm_ms), 3),
        "warm_p95_ms": round(float(np.percentile(warm_ms, 95)), 3),
        "official_test_images_opened": 0,
        "production_promoted": False,
        "clinical_validation_claim": False,
        "note": "Engineering latency sample on CPU development images; not a clinical service-level guarantee.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
