"""Adapter for the published R2-V2 retinal vessel model.

R2-V2 ``bv`` is a real RRWNet vessel segmentor published with a
``safetensors`` checkpoint and its inference source/configuration.  The
adapter loads that source implementation from the verified local artifact
directory, follows the published preprocessing path, and exposes channel 2
(``blood vessels``) as a probability map.  It is supporting engineering
evidence, not a clinical biomarker or diagnosis.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.ml.evidence.interfaces import EvidenceModuleResult

try:  # Optional serving dependency; the API remains importable without ML extras.
    import torch
    from safetensors.torch import load_model
except Exception:  # pragma: no cover - exercised by the unavailable-runtime path
    torch = None
    load_model = None

try:  # The published preprocessing uses scikit-image and scipy.
    from skimage.transform import resize as skimage_resize
except Exception:  # pragma: no cover - dependency availability is environment-specific
    skimage_resize = None

try:
    import cv2
except Exception:  # pragma: no cover - the backend normally installs OpenCV
    cv2 = None


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = ROOT / "ml" / "weights" / "vessel_segmentation" / "r2-v2-bv-2025" / "bv.safetensors"
MODEL_VERSION = "r2-v2-bv-2025"
MODEL_REPOSITORY = "j-morano/R2-V2"
MODEL_SOURCE_URL = f"https://huggingface.co/{MODEL_REPOSITORY}"
MODEL_CODE_URL = "https://github.com/j-morano/R2-V2"
MODEL_ARCHITECTURE = "RRWNet (R2-V2 bv variant)"
MODEL_CLASSES = {0: "artery", 1: "vein", 2: "blood_vessels"}
VESSEL_CHANNEL = 2
PUBLISHED_INPUT_WIDTH = 1408
PUBLISHED_INPUT_CHANNELS = 6
PUBLISHED_OUTPUT_CHANNELS = 3


class VesselArtifactInvalidError(RuntimeError):
    """Raised when the downloaded model/configuration contract is invalid."""


def _data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    output = io.BytesIO()
    image.save(output, format=fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _probability_data_uri(probability: np.ndarray) -> str:
    values = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    return _data_uri(Image.fromarray(values, mode="L"))


def _mask_data_uri(mask: np.ndarray, colour: tuple[int, int, int] = (0, 220, 120), alpha: int = 165) -> str:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    present = mask.astype(bool)
    rgba[present, :3] = colour
    rgba[present, 3] = alpha
    return _data_uri(Image.fromarray(rgba, mode="RGBA"))


def _overlay_data_uri(image_rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int] = (0, 220, 120), alpha: int = 145) -> str:
    base = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").convert("RGBA")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    present = mask.astype(bool)
    rgba[present, :3] = colour
    rgba[present, 3] = alpha
    return _data_uri(Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")))


def _connected_components(mask: np.ndarray) -> int:
    if cv2 is None:
        return 0
    count, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return max(0, int(count) - 1)


class PretrainedRetinalVesselAdapter:
    """Lazy-loading adapter for R2-V2's published ``bv`` model."""

    module = "vessel_segmentation"
    name = "R2-V2 RRWNet bv vessel segmentor"

    def __init__(self, model_path: str | Path | None = None, device: str = "auto", threshold: float = 0.5, version: str | None = None):
        self.model_path = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.is_absolute():
            from_cwd = (Path.cwd() / self.model_path).resolve()
            from_root = (ROOT / self.model_path).resolve()
            self.model_path = from_cwd if from_cwd.is_file() else from_root
        self.config_path = self.model_path.with_name("bv_config.json")
        self.source_model_path = self.model_path.with_name("model.py")
        self.source_preprocessing_path = self.model_path.with_name("preprocessing.py")
        self.device_name = self._resolve_device(device)
        self.threshold = float(threshold)
        self.version = version or MODEL_VERSION
        self._model: Any = None
        self._load_error: str | None = None
        self._artifact_status = "MODEL_AVAILABLE" if self.model_path.is_file() else "MODEL_MISSING"
        self._checkpoint_checksum: str | None = None
        self._cache_key: str | None = None
        self._cache_probability: np.ndarray | None = None
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
            "model_available": self.model_path.is_file() and self._load_error is None,
            "model_loaded": self._model is not None,
            "artifact_status": self._artifact_status,
            "runtime_available": torch is not None and load_model is not None and skimage_resize is not None,
            "load_error": self._load_error,
            "device": self.device_name,
            "architecture": MODEL_ARCHITECTURE,
            "source": MODEL_SOURCE_URL,
            "source_code": MODEL_CODE_URL,
            "license": "CC BY 4.0 (model repository/model card declaration)",
            "classes": MODEL_CLASSES,
            "vessel_channel": VESSEL_CHANNEL,
            "published_input_width": PUBLISHED_INPUT_WIDTH,
            "published_input_channels": PUBLISHED_INPUT_CHANNELS,
            "published_output_channels": PUBLISHED_OUTPUT_CHANNELS,
            "clinical_validation_claim": False,
        }

    def analyze(self, image_rgb: Any, context: dict[str, Any]) -> EvidenceModuleResult:
        try:
            probability = self._predict(np.asarray(image_rgb, dtype=np.uint8))
            mask = probability >= self.threshold
            coverage = float(mask.mean())
            inside = probability[mask] if np.any(mask) else probability.reshape(-1)
            metadata = self._metadata(context, probability, mask)
            return EvidenceModuleResult(
                module="vessel_segmentation",
                category="segmentation",
                status="model_inference",
                supported=True,
                implementation=self.name,
                confidence=round(float(np.mean(inside)) if inside.size else 0.0, 4),
                count=_connected_components(mask),
                mask_data_uri=_mask_data_uri(mask),
                probability_map_data_uri=_probability_data_uri(probability),
                overlay_data_uri=_overlay_data_uri(np.asarray(image_rgb, dtype=np.uint8), mask),
                metadata={**metadata, "coverage_ratio": round(coverage, 8)},
                issues=[
                    {
                        "type": "engineering_evidence",
                        "message": "Real neural-network vessel segmentation output; measurements are engineering estimates and are not clinically validated biomarkers.",
                    }
                ],
            )
        except Exception as exc:  # Fail explicitly; never substitute a classical mask.
            self._load_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, VesselArtifactInvalidError):
                self._artifact_status = "MODEL_INVALID"
            elif not isinstance(exc, ValueError):
                self._artifact_status = "MODEL_LOAD_FAILED"
            return EvidenceModuleResult(
                module="vessel_segmentation",
                category="segmentation",
                status="unsupported",
                supported=False,
                implementation=self.name,
                issues=[{"type": "invalid_input" if isinstance(exc, ValueError) else "model_unavailable", "message": f"Pretrained vessel model failed safely: {exc}"}],
                metadata=self.health(),
            )

    def predict_probability(self, image_rgb: Any) -> np.ndarray:
        """Return the raw blood-vessel probability map for evaluation tooling."""
        return self._predict(np.asarray(image_rgb, dtype=np.uint8)).copy()

    def _predict(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape HxWx3")
        cache_key = hashlib.sha1(image_rgb.tobytes()).hexdigest()
        with self._lock:
            if self._cache_key == cache_key and self._cache_probability is not None:
                return self._cache_probability
            model, preprocessing = self._get_model()
            original_shape = image_rgb.shape[:2]
            cfp = np.clip(image_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
            if cfp.shape[1] != PUBLISHED_INPUT_WIDTH:
                new_height = max(1, int(cfp.shape[0] * (PUBLISHED_INPUT_WIDTH / cfp.shape[1])))
                if skimage_resize is None:
                    raise RuntimeError("scikit-image is required for the published R2-V2 resize path")
                cfp = skimage_resize(cfp, (new_height, PUBLISHED_INPUT_WIDTH), anti_aliasing=True, preserve_range=True).astype(np.float32)
            fov = (cfp.sum(axis=2) > 0.01).astype(np.float32)
            enhanced, enhanced_mask = preprocessing.preprocess_img(cfp, fov)
            if not np.isfinite(enhanced).all():
                raise ValueError("Published vessel preprocessing produced non-finite values")
            model_input = np.concatenate([enhanced.astype(np.float32), cfp.astype(np.float32)], axis=-1)
            padded, padding = self._pad_for_unet(model_input)
            tensor = torch.from_numpy(padded.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(self.device_name)
            with torch.inference_mode():
                output = model(tensor)
                if isinstance(output, (list, tuple)):
                    output = output[-1]
                if output.ndim != 4 or output.shape[1] != PUBLISHED_OUTPUT_CHANNELS:
                    raise VesselArtifactInvalidError(f"Unexpected R2-V2 output shape: {tuple(output.shape)}")
                probabilities = torch.sigmoid(output)[0, VESSEL_CHANNEL].detach().cpu().numpy()
            top, bottom, left, right = padding
            probability = probabilities[top:probabilities.shape[0] - bottom, left:probabilities.shape[1] - right]
            probability = np.where(np.asarray(enhanced_mask) > 0.5, probability, 0.0)
            if cv2 is None:
                raise RuntimeError("OpenCV is required to restore the vessel probability map")
            probability = cv2.resize(probability, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            probability = np.clip(probability.astype(np.float32), 0.0, 1.0)
            self._cache_key = cache_key
            self._cache_probability = probability
            return probability

    def _get_model(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._load_preprocessing()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Vessel model weights are missing at {self.model_path}. Run 'python scripts/acquire_vessel_model.py' after reviewing the source and license."
            )
        if torch is None or load_model is None or skimage_resize is None:
            raise RuntimeError("Vessel model runtime is unavailable. Install backend/requirements-ml.txt including scikit-image and safetensors.")
        if not self.config_path.is_file() or not self.source_model_path.is_file() or not self.source_preprocessing_path.is_file():
            raise FileNotFoundError("R2-V2 weights require bv_config.json, model.py, and preprocessing.py beside bv.safetensors; reacquire the complete artifact.")
        manifest_path = self.model_path.with_name("model_manifest.json")
        expected_checksum = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_checksum = manifest.get("checkpoint_sha256")
        actual_checksum = self._checksum()
        if expected_checksum and expected_checksum != actual_checksum:
            raise VesselArtifactInvalidError(f"Vessel checkpoint checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        expected = {"model": "RRWNet", "in_channels": 6, "out_channels": 3, "base_channels": 64, "num_iterations": 5}
        for key, value in expected.items():
            if config.get(key) != value:
                raise VesselArtifactInvalidError(f"R2-V2 configuration mismatch for {key}: expected {value!r}, got {config.get(key)!r}")
        external_model = self._load_external_module(self.source_model_path, "r2_v2_model")
        model = external_model.RRWNet(config["in_channels"], config["out_channels"], config["base_channels"], config["num_iterations"])
        load_model(model, str(self.model_path))
        model = model.to(self.device_name)
        model.eval()
        self._model = model
        self._load_error = None
        self._artifact_status = "MODEL_AVAILABLE"
        return model, self._load_preprocessing()

    def _load_preprocessing(self) -> Any:
        return self._load_external_module(self.source_preprocessing_path, "r2_v2_preprocessing")

    @staticmethod
    def _load_external_module(path: Path, name: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load published R2-V2 source module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _pad_for_unet(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        # Match the published get_unet_padding_np behavior, including its
        # one-block pad when a dimension is already divisible by 32.
        h_pad = 32 - image.shape[0] % 32
        w_pad = 32 - image.shape[1] % 32
        top, bottom = h_pad // 2, h_pad - h_pad // 2
        left, right = w_pad // 2, w_pad - w_pad // 2
        return np.pad(image, ((top, bottom), (left, right), (0, 0))), (top, bottom, left, right)

    def _checksum(self) -> str:
        if self._checkpoint_checksum is None:
            digest = hashlib.sha256()
            with self.model_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._checkpoint_checksum = digest.hexdigest()
        return self._checkpoint_checksum

    def _metadata(self, context: dict[str, Any], probability: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        inside = probability[mask] if np.any(mask) else probability.reshape(-1)
        manifest_path = self.model_path.with_name("model_manifest.json")
        revision = None
        if manifest_path.is_file():
            try:
                revision = json.loads(manifest_path.read_text(encoding="utf-8")).get("revision")
            except (OSError, json.JSONDecodeError):
                revision = None
        return {
            "model_version": self.version,
            "model_name": self.name,
            "model_repository": MODEL_REPOSITORY,
            "model_source": MODEL_SOURCE_URL,
            "source_code": MODEL_CODE_URL,
            "source_revision": revision,
            "license": "CC BY 4.0",
            "architecture": MODEL_ARCHITECTURE,
            "training_dataset_provenance": "Unified_Fundus (as stated in the published R2-V2 bv_config.json)",
            "input_configuration": {"width": PUBLISHED_INPUT_WIDTH, "channels": PUBLISHED_INPUT_CHANNELS, "preprocessing": "published R2-V2 enhance_image + CLAHE, RGB fundus concatenated with enhanced RGB"},
            "output_configuration": {"channels": MODEL_CLASSES, "vessel_channel": VESSEL_CHANNEL, "probability_activation": "sigmoid"},
            "checkpoint_sha256": self._checksum(),
            "threshold": self.threshold,
            "device": self.device_name,
            "pixel_count": int(mask.sum()),
            "mean_probability_inside_mask": round(float(np.mean(inside)) if inside.size else 0.0, 6),
            "max_probability": round(float(np.max(probability)), 6),
            "connected_components": _connected_components(mask),
            "measurement_status": "ENGINEERING_ESTIMATE",
            "clinical_validation_claim": False,
            "context": {key: value for key, value in context.items() if key in {"working_width", "working_height", "retina_area_ratio"}},
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device != "auto":
            if device == "cuda" and (torch is None or not torch.cuda.is_available()):
                raise RuntimeError("CUDA was requested for the vessel model but is unavailable; use VESSEL_MODEL_DEVICE=cpu or install CUDA PyTorch.")
            return device
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
