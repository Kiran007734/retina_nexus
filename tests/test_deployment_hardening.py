from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings  # noqa: E402
from app.core.safe_errors import safe_error_message  # noqa: E402
from app.ml.quality.trust_gate import ImageTrustGateError, ImageTrustGateService  # noqa: E402
from app.services.runtime import verify_models  # noqa: E402


def _png(width: int = 256, height: int = 256) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), (80, 100, 120)).save(output, format="PNG")
    return output.getvalue()


def test_pixel_limit_rejects_large_decoded_workload_before_processing():
    service = ImageTrustGateService(max_image_pixels=(256 * 256) - 1)
    with pytest.raises(ImageTrustGateError, match="too many pixels"):
        service.validate_input(_png())


def test_runtime_verification_fails_required_model_without_leaking_paths():
    result = verify_models(Settings(_env_file=None, classifier_model_path=None), load_models=False, verify_optional_checksums=False)
    assert result["status"] == "NOT_READY"
    assert result["models"]["classifier"]["status"] == "REQUIRED_MODEL_UNAVAILABLE"
    assert all("artifact_path" not in model for model in result["models"].values())


def test_exception_messages_with_paths_are_replaced_by_safe_fallbacks():
    message = safe_error_message(OSError(r"C:\private\weights\checkpoint.pt"), "The model artifact is unavailable.")
    assert message == "The model artifact is unavailable."
