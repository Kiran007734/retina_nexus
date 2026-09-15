"""Opt-in adapter for the frozen IDRiD research lesion segmentor.

The adapter is deliberately separate from DR classification. It emits
supporting lesion evidence only. The default container continues to use the
preserved external lesion model unless this adapter is explicitly enabled.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.ml.evidence.interfaces import EvidenceModuleResult
from app.ml.evidence.lesion_model import MODEL_COLOURS, _png_data_uri, _regions

try:  # Optional ML runtime; API startup remains usable without it.
    import torch
except Exception:  # pragma: no cover
    torch = None

try:  # pragma: no cover - availability depends on the deployment image
    import torchseg  # noqa: F401
except Exception:
    torchseg = None


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = ROOT / "ml" / "weights" / "lesions" / "idrid" / "checkpoint_best.pt"
DEFAULT_MANIFEST_PATH = DEFAULT_MODEL_PATH.with_name("model_manifest.json")
IDRID_CLASSES = ("microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates")
IDRID_CLASS_TO_MODULE = {
    "microaneurysms": "microaneurysm_detection",
    "haemorrhages": "hemorrhage_detection",
    "hard_exudates": "exudate_segmentation",
    "soft_exudates": "cotton_wool_spot_detection",
}
IDRID_CLASS_COLOURS = {
    "microaneurysms": MODEL_COLOURS["microaneurysm"],
    "haemorrhages": MODEL_COLOURS["hemorrhage"],
    "hard_exudates": MODEL_COLOURS["exudate"],
    "soft_exudates": MODEL_COLOURS["cotton_wool_spot"],
}
MODEL_NAME = "idrid-unet-seresnext50"
MODEL_VERSION = "idrid-lesion-unet-seresnext50-768-focaldice-20260913-v2"


class IDRiDLesionAdapter:
    """Lazy-loading four-channel multi-label IDRiD evidence adapter."""

    module = "idrid_lesion_segmentation"
    name = MODEL_NAME

    def __init__(self, model_path: str | Path | None = None, device: str = "auto", threshold: float = 0.7, version: str | None = None, expected_sha256: str | None = None):
        self.model_path = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.is_absolute():
            self.model_path = (ROOT / self.model_path).resolve()
        self.manifest_path = self.model_path.with_name("model_manifest.json")
        self.device_name = self._resolve_device(device)
        self.threshold = float(threshold)
        self.version = version or MODEL_VERSION
        self.expected_sha256 = expected_sha256
        self._model: Any = None
        self._load_error: str | None = None
        self._checksum: str | None = None
        self._lock = threading.Lock()
        self._cache_key: str | None = None
        self._cache_probabilities: np.ndarray | None = None
        self._input_size = 768
        if self.manifest_path.is_file():
            try:
                manifest = __import__("json").loads(self.manifest_path.read_text(encoding="utf-8"))
                self._input_size = int(manifest.get("input_size", self._input_size))
                self.expected_sha256 = self.expected_sha256 or manifest.get("checkpoint_sha256")
            except Exception as exc:
                self._load_error = f"Invalid IDRiD lesion model manifest: {exc}"

    @property
    def is_configured(self) -> bool:
        return self.model_path.is_file() and self.manifest_path.is_file()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def health(self) -> dict[str, Any]:
        return {
            "model_version": self.version,
            "model_path": str(self.model_path),
            "artifact_present": self.model_path.is_file(),
            "manifest_present": self.manifest_path.is_file(),
            "runtime_available": torch is not None and torchseg is not None,
            "loaded": self._model is not None,
            "load_error": self._load_error,
            "device": self.device_name,
            "classes": {str(index): name for index, name in enumerate(IDRID_CLASSES)},
            "threshold": self.threshold,
            "production_promoted": False,
            "clinical_validation_claim": False,
        }

    def verify_loadable(self) -> None:
        self._get_model()

    def analyze(self, image_rgb: Any, context: dict[str, Any]) -> EvidenceModuleResult:
        module = str(context.get("requested_module", ""))
        class_name = next((name for name, mapped_module in IDRID_CLASS_TO_MODULE.items() if mapped_module == module), None)
        if class_name is None:
            return EvidenceModuleResult(module=module or self.module, category="lesion_detection", status="unsupported", supported=False, implementation=self.name, issues=[{"type": "unsupported", "message": f"IDRiD adapter has no class mapping for '{module}'."}])
        try:
            probabilities = self._predict(np.asarray(image_rgb, dtype=np.uint8))
            class_index = IDRID_CLASSES.index(class_name)
            probability = probabilities[class_index]
            mask = probability >= self.threshold
            regions = _regions(mask, probability, minimum_area=2 if class_name == "microaneurysms" else 4)
            score = float(np.mean(probability[mask])) if np.any(mask) else float(np.max(probability))
            return EvidenceModuleResult(
                module=module,
                category="segmentation" if class_name == "hard_exudates" else "lesion_detection",
                status="model_inference",
                supported=True,
                implementation=self.name,
                confidence=round(float(np.clip(score, 0.0, 1.0)), 4),
                count=len(regions),
                mask_data_uri=_png_data_uri(mask, IDRID_CLASS_COLOURS[class_name], probability),
                bounding_regions=regions,
                metadata={
                    "model_version": self.version,
                    "architecture": "U-Net with SE-ResNeXt-50 32x4d encoder",
                    "class_name": class_name,
                    "class_index": class_index,
                    "checkpoint_sha256": self._checksum,
                    "input_resolution": self._input_size,
                    "threshold": self.threshold,
                    "device": self.device_name,
                    "pixel_count": int(mask.sum()),
                    "coverage_ratio": round(float(mask.mean()), 8),
                    "mean_probability": round(float(np.mean(probability)), 6),
                    "max_probability": round(float(np.max(probability)), 6),
                    "production_promoted": False,
                    "clinical_validation_claim": False,
                },
                issues=[{"type": "research_model", "message": "Frozen IDRiD model is supporting evidence only; it is not a clinical diagnosis or production-promoted model."}],
            )
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return EvidenceModuleResult(module=module, category="segmentation" if class_name == "hard_exudates" else "lesion_detection", status="unsupported", supported=False, implementation=self.name, issues=[{"type": "model_unavailable", "message": f"IDRiD lesion model failed safely: {exc}"}], metadata=self.health())

    def _predict(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape HxWx3")
        cache_key = hashlib.sha1(image_rgb.tobytes()).hexdigest()
        with self._lock:
            if cache_key == self._cache_key and self._cache_probabilities is not None:
                return self._cache_probabilities
            model = self._get_model()
            fitted = np.asarray(Image.fromarray(image_rgb, mode="RGB").resize((self._input_size, self._input_size), Image.Resampling.LANCZOS), dtype=np.uint8)
            tensor = torch.from_numpy(fitted.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
            tensor = ((tensor - mean) / std).to(self.device_name)
            with torch.inference_mode():
                logits = model(tensor)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            restored = np.stack([np.asarray(Image.fromarray(channel.astype(np.float32), mode="F").resize((image_rgb.shape[1], image_rgb.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) for channel in probabilities])
            self._cache_key = cache_key
            self._cache_probabilities = np.clip(restored, 0.0, 1.0)
            return self._cache_probabilities

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise FileNotFoundError(f"IDRiD lesion checkpoint is missing at {self.model_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"IDRiD lesion model manifest is missing at {self.manifest_path}")
        if torch is None or torchseg is None:
            raise RuntimeError("IDRiD lesion runtime is unavailable; install torch and torchseg")
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if self.expected_sha256 and digest != self.expected_sha256:
            raise RuntimeError(f"IDRiD lesion checkpoint SHA-256 mismatch: expected {self.expected_sha256}, got {digest}")
        from ml.lesions.idrid import build_idrid_model

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        model, _transfer = build_idrid_model(None)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self._model = model.to(self.device_name)
        self._model.eval()
        self._checksum = digest
        self._load_error = None
        return self._model

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device != "auto":
            return device
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
