import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.evidence.lesion_model import DEFAULT_MODEL_PATH, PretrainedRetinalLesionAdapter  # noqa: E402


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason="real pretrained lesion artifact is not installed")
def test_pretrained_lesion_checkpoint_loads_with_exact_architecture():
    adapter = PretrainedRetinalLesionAdapter(device="cpu")
    model = adapter._get_model()
    assert model is not None
    assert adapter.health()["load_error"] is None
    assert adapter.health()["classes"][4] == "microaneurysm"
