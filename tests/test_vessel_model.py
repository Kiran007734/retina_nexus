import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.evidence.vessel_model import DEFAULT_MODEL_PATH, PretrainedRetinalVesselAdapter  # noqa: E402

ARTIFACT_ROOT = DEFAULT_MODEL_PATH.parent
REAL_IMAGE = ROOT / "ml" / "datasets" / "raw" / "aptos2019" / "train_images" / "0d0b8fc9ab5c.png"
HAS_ARTIFACT = all((ARTIFACT_ROOT / filename).is_file() for filename in ("bv.safetensors", "bv_config.json", "model.py", "preprocessing.py"))


def _decode_dimensions(data_uri: str) -> tuple[int, int]:
    with Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))) as image:
        return image.size


def test_vessel_artifact_manifest_has_expected_contract():
    if not HAS_ARTIFACT:
        pytest.skip("real R2-V2 vessel artifact is not installed")
    config = json.loads((ARTIFACT_ROOT / "bv_config.json").read_text(encoding="utf-8"))
    assert config["model"] == "RRWNet"
    assert config["in_channels"] == 6
    assert config["out_channels"] == 3
    assert config["num_iterations"] == 5


@pytest.mark.skipif(not (HAS_ARTIFACT and REAL_IMAGE.is_file()), reason="real R2-V2 artifact or real retinal image is not installed")
def test_real_vessel_inference_returns_probability_mask_and_overlay():
    image = np.asarray(Image.open(REAL_IMAGE).convert("RGB"), dtype=np.uint8)
    result = PretrainedRetinalVesselAdapter(device="cpu").analyze(image, {"working_width": image.shape[1], "working_height": image.shape[0]})
    assert result.status == "model_inference"
    assert result.supported is True
    assert result.metadata["checkpoint_sha256"]
    assert result.metadata["measurement_status"] == "ENGINEERING_ESTIMATE"
    assert _decode_dimensions(result.mask_data_uri) == (image.shape[1], image.shape[0])
    assert _decode_dimensions(result.probability_map_data_uri) == (image.shape[1], image.shape[0])
    assert _decode_dimensions(result.overlay_data_uri) == (image.shape[1], image.shape[0])
    assert 0.0 <= result.confidence <= 1.0
