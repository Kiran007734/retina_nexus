"""Adapter for the verified pretrained retinal lesion segmentation model.

The model is published by the authors of ``fundus-lesions-toolkit`` as
``ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d``.  It is a
five-class semantic segmentor (background plus four lesion classes) trained
on IDRiD, DDR, FGADR, Messidor, and RETLES.  This adapter is intentionally
separate from DR grading: its output is supporting evidence, not a diagnosis.

Only safetensors artifacts are loaded.  If the optional TorchSeg runtime or
the explicitly configured artifact is unavailable, callers receive an
unsupported result with the reason; no heuristic result is substituted for a
configured model failure.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.ml.evidence.interfaces import EvidenceModuleResult

try:  # Optional ML serving dependency; the API shell remains installable without it.
    import torch
    import torchseg
    from safetensors.torch import load_file
except Exception:  # pragma: no cover - exercised through the unavailable-runtime path
    torch = None
    torchseg = None
    load_file = None

try:
    import cv2
except Exception:  # pragma: no cover - the backend normally installs OpenCV
    cv2 = None


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = ROOT / "ml" / "weights" / "lesion_segmentation" / "fundus-lesions-unet-seresnext50-all-v1" / "model.safetensors"
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_PATH.with_name("config.json")
MODEL_REPOSITORY = "ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d"
MODEL_VERSION = "fundus-lesions-unet-seresnext50-all-v1"
MODEL_SOURCE_URL = f"https://huggingface.co/{MODEL_REPOSITORY}"
MODEL_CODE_URL = "https://github.com/ClementPla/fundus-lesions-toolkit"
MODEL_ARCHITECTURE = "U-Net with SE-ResNeXt-50 32x4d encoder"
MODEL_CLASSES = {
    0: "background",
    1: "cotton_wool_spot",
    2: "exudate",
    3: "hemorrhage",
    4: "microaneurysm",
}
MODEL_CLASS_TO_MODULE = {
    "cotton_wool_spot": "cotton_wool_spot_detection",
    "exudate": "exudate_segmentation",
    "hemorrhage": "hemorrhage_detection",
    "microaneurysm": "microaneurysm_detection",
}
MODEL_COLOURS = {
    "cotton_wool_spot": (236, 166, 63),
    "exudate": (140, 241, 142),
    "hemorrhage": (68, 152, 240),
    "microaneurysm": (20, 20, 136),
}


def _png_data_uri(mask: np.ndarray, colour: tuple[int, int, int], alpha: np.ndarray | None = None) -> str:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    present = mask > 0
    rgba[present, 0] = colour[0]
    rgba[present, 1] = colour[1]
    rgba[present, 2] = colour[2]
    if alpha is None:
        rgba[present, 3] = 165
    else:
        rgba[present, 3] = np.clip(alpha[present] * 205.0, 80, 220).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _regions(mask: np.ndarray, probability: np.ndarray, minimum_area: int = 2, limit: int = 64) -> list[dict[str, Any]]:
    """Extract image-space connected regions from a model mask."""
    if cv2 is None:
        return []
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    result: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        if area < minimum_area:
            continue
        component = mask[y : y + height, x : x + width]
        values = probability[y : y + height, x : x + width][component]
        result.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "center_x": round(float(centroids[index][0]), 2),
                "center_y": round(float(centroids[index][1]), 2),
                "score": round(float(np.mean(values)) if values.size else 0.0, 4),
            }
        )
    result.sort(key=lambda item: (item["score"], item["area"]), reverse=True)
    return result[:limit]


class PretrainedRetinalLesionAdapter:
    """Lazy-loading, multi-class adapter backed by the published checkpoint."""

    module = "retinal_lesion_segmentation"
    name = "fundus-lesions-toolkit-unet-seresnext50"

    def __init__(self, model_path: str | Path | None = None, device: str = "auto", threshold: float = 0.5, version: str | None = None):
        self.model_path = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.is_absolute():
            from_cwd = (Path.cwd() / self.model_path).resolve()
            from_root = (ROOT / self.model_path).resolve()
            self.model_path = from_cwd if from_cwd.is_file() else from_root
        self.config_path = self.model_path.with_name("config.json")
        self.device_name = self._resolve_device(device)
        self.threshold = float(threshold)
        self.version = version or MODEL_VERSION
        self._model: Any = None
        self._load_error: str | None = None
        self._cache_key: str | None = None
        self._cache_probabilities: np.ndarray | None = None
        self._checkpoint_checksum: str | None = None
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return self.model_path.is_file()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def health(self) -> dict[str, Any]:
        return {
            "model_version": self.version,
            "model_path": str(self.model_path),
            "artifact_present": self.model_path.is_file(),
            "runtime_available": torch is not None and torchseg is not None and load_file is not None,
            "loaded": self._model is not None,
            "load_error": self._load_error,
            "device": self.device_name,
            "classes": MODEL_CLASSES,
            "source": MODEL_SOURCE_URL,
            "license": "MIT (model repository and model card declaration)",
            "clinical_validation_claim": False,
        }

    def verify_loadable(self) -> None:
        """Load the optional artifact when deployment preflight requests it."""
        self._get_model()

    def analyze(self, image_rgb: Any, context: dict[str, Any]) -> EvidenceModuleResult:
        module = str(context.get("requested_module", ""))
        class_name = next((name for name, item in MODEL_CLASS_TO_MODULE.items() if item == module), None)
        if class_name is None:
            return EvidenceModuleResult(
                module=module or self.module,
                category="lesion_detection",
                status="unsupported",
                supported=False,
                implementation=self.name,
                issues=[{"type": "unsupported", "message": f"The pretrained model does not expose a class mapping for '{module}'."}],
            )
        try:
            probabilities = self._predict(image_rgb)
            class_index = next(index for index, name in MODEL_CLASSES.items() if name == class_name)
            probability = probabilities[class_index]
            mask = probability >= self.threshold
            regions = _regions(mask, probability, minimum_area=2 if class_name == "microaneurysm" else 4)
            score = float(np.mean(probability[mask])) if np.any(mask) else float(np.max(probability))
            return EvidenceModuleResult(
                module=module,
                category="segmentation" if class_name == "exudate" else "lesion_detection",
                status="model_inference",
                supported=True,
                implementation=self.name,
                confidence=round(max(0.0, min(1.0, score)), 4),
                count=len(regions),
                mask_data_uri=_png_data_uri(mask, MODEL_COLOURS[class_name], probability),
                bounding_regions=regions,
                metadata=self._metadata(class_name, context, probability, mask),
                issues=[
                    {
                        "type": "research_model",
                        "message": "Pretrained research model output is supporting evidence; it is not a clinical diagnosis or validated production claim.",
                    }
                ],
            )
        except Exception as exc:  # Safely expose model failures to the orchestration layer.
            self._load_error = f"{type(exc).__name__}: {exc}"
            return EvidenceModuleResult(
                module=module,
                category="segmentation" if class_name == "exudate" else "lesion_detection",
                status="unsupported",
                supported=False,
                implementation=self.name,
                issues=[{"type": "model_unavailable", "message": f"Pretrained lesion model failed safely: {exc}"}],
                metadata=self.health(),
            )

    def _predict(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape HxWx3")
        cache_key = hashlib.sha1(image_rgb.tobytes()).hexdigest()
        with self._lock:
            if self._cache_key == cache_key and self._cache_probabilities is not None:
                return self._cache_probabilities
            model = self._get_model()
            fitted, transform = self._fit_image(image_rgb, 1024)
            tensor = torch.from_numpy(fitted.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
            tensor = ((tensor - mean) / std).to(self.device_name)
            with torch.inference_mode():
                logits = model(tensor)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            restored = np.stack([self._restore_mask(channel, transform, image_rgb.shape[:2]) for channel in probabilities], axis=0)
            self._cache_key = cache_key
            self._cache_probabilities = np.clip(restored.astype(np.float32), 0.0, 1.0)
            return self._cache_probabilities

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Lesion model weights are missing at {self.model_path}. Run 'python scripts/acquire_lesion_model.py' after reviewing the source and license."
            )
        if torch is None or torchseg is None or load_file is None:
            raise RuntimeError("Lesion model runtime is unavailable. Install backend/requirements-ml.txt including torchseg and safetensors.")
        config = self._read_config()
        if config.get("arch") != "unet" or config.get("n_classes") != 5:
            raise ValueError("Lesion model config does not match the required U-Net five-class contract")
        model = torchseg.create_model(
            arch="unet",
            encoder_name="se_resnext50_32x4d",
            encoder_weights=None,
            in_channels=3,
            classes=5,
        )
        state = load_file(str(self.model_path), device="cpu")
        target_keys = set(model.state_dict())
        mapped: dict[str, Any] = {}
        for key, value in state.items():
            normalized = key[6:] if key.startswith("model.") else key
            if normalized.startswith("encoder.model."):
                normalized = "encoder." + normalized[len("encoder.model.") :]
            normalized = normalized.replace(".se.", ".se_module.")
            if normalized.startswith("encoder.conv1."):
                normalized = "encoder.layer0." + normalized[len("encoder.") :]
            elif normalized.startswith("encoder.bn1."):
                normalized = "encoder.layer0." + normalized[len("encoder.") :]
            if normalized in target_keys:
                mapped[normalized] = value
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Lesion checkpoint architecture mismatch: missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}")
        self._model = model.to(self.device_name)
        self._model.eval()
        self._load_error = None
        return self._model

    def _read_config(self) -> dict[str, Any]:
        path = self.config_path if self.config_path.is_file() else DEFAULT_CONFIG_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Lesion model config is missing beside the weights: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _fit_image(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, dict[str, int]]:
        height, width = image_rgb.shape[:2]
        if cv2 is None:
            raise RuntimeError("OpenCV is required for model-compatible retinal image fitting")
        max_value = float(np.max(image_rgb))
        visible = np.max(image_rgb, axis=2) > 0.05 * max_value if max_value else np.zeros((height, width), dtype=bool)
        points = cv2.findNonZero(visible.astype(np.uint8))
        if points is None:
            x0, y0, x1, y1 = 0, 0, width, height
        else:
            x0, y0, x2, y2 = cv2.boundingRect(points)
            x1, y1 = x0 + max(1, x2), y0 + max(1, y2)
        crop = image_rgb[y0:y1, x0:x1]
        scale = size / max(crop.shape[0], crop.shape[1])
        resized_width = max(1, round(crop.shape[1] * scale))
        resized_height = max(1, round(crop.shape[0] * scale))
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        pad_top = (size - resized_height) // 2
        pad_left = (size - resized_width) // 2
        canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
        return canvas, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "pad_top": pad_top, "pad_left": pad_left, "resized_width": resized_width, "resized_height": resized_height}

    @staticmethod
    def _restore_mask(mask: np.ndarray, transform: dict[str, int], output_shape: tuple[int, int]) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for model mask restoration")
        y0, y1 = transform["pad_top"], transform["pad_top"] + transform["resized_height"]
        x0, x1 = transform["pad_left"], transform["pad_left"] + transform["resized_width"]
        cropped = mask[y0:y1, x0:x1]
        restored = cv2.resize(cropped, (transform["x1"] - transform["x0"], transform["y1"] - transform["y0"]), interpolation=cv2.INTER_LINEAR)
        output = np.zeros(output_shape, dtype=np.float32)
        output[transform["y0"] : transform["y1"], transform["x0"] : transform["x1"]] = restored
        return output

    def _metadata(self, class_name: str, context: dict[str, Any], probability: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        if self._checkpoint_checksum is None and self.model_path.is_file():
            self._checkpoint_checksum = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        return {
            "model_version": self.version,
            "model_name": self.name,
            "model_repository": MODEL_REPOSITORY,
            "model_source": MODEL_SOURCE_URL,
            "source_code": MODEL_CODE_URL,
            "license": "MIT",
            "architecture": MODEL_ARCHITECTURE,
            "trained_on": ["IDRiD", "DDR", "FGADR", "MESSIDOR", "RETLES"],
            "class_name": class_name,
            "class_index": next(index for index, name in MODEL_CLASSES.items() if name == class_name),
            "checkpoint_sha256": self._checkpoint_checksum,
            "input_resolution": 1024,
            "threshold": self.threshold,
            "device": self.device_name,
            "pixel_count": int(mask.sum()),
            "coverage_ratio": round(float(mask.mean()), 8),
            "mean_probability": round(float(np.mean(probability)), 6),
            "max_probability": round(float(np.max(probability)), 6),
            "coarse_to_fine": "global fundus resized/cropped to 1024; class masks restored to the evidence working image",
            "clinical_validation_claim": False,
            "context": {key: value for key, value in context.items() if key in {"working_width", "working_height", "proposal_count", "patch_count"}},
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device != "auto":
            return device
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
