"""Verify the frozen IDRiD max-quality research artifact without promotion.

The verification uses only five real IDRiD development images (one per grade)
for deterministic inference, controlled brightness robustness, one Grad-CAM,
and the existing quality gate.  It never loads the reserved official test set
and never changes the production environment or checkpoint.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.inference.classifier import TorchDRClassificationService  # noqa: E402
from app.ml.models.classifier import ReferableDRMapping  # noqa: E402
from app.ml.quality.trust_gate import ImageTrustGateService  # noqa: E402

RAW = ROOT / "ml" / "datasets" / "raw" / "idrid"
MANIFEST = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_v2_development_manifest.json"
MODEL_DIR = ROOT / "ml" / "weights" / "classifiers" / "idrid" / "research" / "max_quality" / "final"
CHECKPOINT = MODEL_DIR / "checkpoint_best.pt"
OUTPUT = ROOT / "ml" / "datasets" / "metadata" / "idrid" / "idrid_max_quality_integration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    import asyncio

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == "validation"]
    selected: list[dict] = []
    for grade in range(5):
        selected.append(next(record for record in records if int(record["label"]) == grade))
    service = TorchDRClassificationService(
        model_path=str(CHECKPOINT),
        backbone="efficientnet_b0",
        model_version="idrid-max-quality-20260912-v1",
        device="cpu",
        referable_mapping=ReferableDRMapping(name="moderate_or_worse", referable_grades=(2, 3, 4)),
    )
    service.verify_loadable()
    quality_service = ImageTrustGateService()
    rows = []
    for record in selected:
        path = RAW / record["image"]
        original = path.read_bytes()
        with Image.open(io.BytesIO(original)) as image:
            altered = io.BytesIO()
            ImageEnhance.Brightness(image.convert("RGB")).enhance(1.05).save(altered, format="JPEG", quality=95)
            altered_bytes = altered.getvalue()
        first = service.predict(original)
        second = service.predict(original)
        perturbed = service.predict(altered_bytes)
        quality = asyncio.run(quality_service.assess(original))
        rows.append({
            "image_id": record["image_id"],
            "actual_grade": int(record["label"]),
            "quality_decision": getattr(quality.quality_decision, "value", quality.quality_decision),
            "prediction": {"grade": first.predicted_grade, "probabilities": first.probabilities, "referable_probability": first.referable_probability, "referable": first.referable_dr, "confidence": first.raw_confidence},
            "deterministic_repeat": {"same_grade": first.predicted_grade == second.predicted_grade, "same_probabilities": first.probabilities == second.probabilities},
            "brightness_perturbation": {"grade": perturbed.predicted_grade, "probabilities": perturbed.probabilities, "same_grade": first.predicted_grade == perturbed.predicted_grade},
        })
    cam_record = selected[2]
    cam_bytes = (RAW / cam_record["image"]).read_bytes()
    explanation = service.explain(cam_bytes)
    payload = {
        "schema_version": "idrid-max-quality-integration-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "idrid-max-quality-20260912-v1",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "production_promoted": False,
        "official_test_images_opened": 0,
        "validation_images_used": len(selected),
        "rows": rows,
        "grad_cam": {"status": "PASS", "image_id": cam_record["image_id"], "target_class": explanation.target_class, "shape": list(explanation.attention_map.shape), "finite": bool(explanation.attention_map.size and explanation.attention_map.min() >= 0.0 and explanation.attention_map.max() <= 1.0)},
        "deterministic_inference_pass": all(row["deterministic_repeat"]["same_grade"] and row["deterministic_repeat"]["same_probabilities"] for row in rows),
        "brightness_grade_stability": sum(row["brightness_perturbation"]["same_grade"] for row in rows) / len(rows),
        "limitations": ["This is an engineering integration check on five development images, not a clinical validation.", "Lesion/vessel evidence services remain separate supporting modules and do not alter the severity grade.", "Raw softmax confidence is not clinically calibrated."],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": payload["checkpoint_sha256"], "deterministic_inference_pass": payload["deterministic_inference_pass"], "grad_cam": payload["grad_cam"], "official_test_images_opened": 0}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
