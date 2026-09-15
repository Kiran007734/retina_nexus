"""Opt-in research adapter for the newly trained DRIVE vessel candidate.

The protected R2-V2 adapter remains the default production vessel module.  This
adapter is enabled only with ``DRIVE_VESSEL_MODEL_ENABLED=true`` and always
labels its outputs as research engineering evidence.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.ml.evidence.interfaces import EvidenceModuleResult
from app.ml.evidence.vessel_model import _connected_components, _mask_data_uri, _overlay_data_uri, _probability_data_uri
from ml.vessels.drive import normalize_input

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = ROOT / "ml" / "weights" / "vessels" / "drive" / "checkpoint_best.pt"
DEFAULT_MANIFEST_PATH = DEFAULT_MODEL_PATH.with_name("model_manifest.json")


class DriveResearchVesselAdapter:
    """Lazy-loaded lightweight U-Net adapter for research-only DRIVE output."""

    module = "vessel_segmentation"
    name = "DRIVE research lightweight U-Net vessel segmentor"

    def __init__(self, model_path: str | Path | None = None, device: str = "auto", threshold: float = 0.3, version: str | None = None, expected_sha256: str | None = None):
        self.model_path = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.is_absolute():
            candidates = ((Path.cwd() / self.model_path).resolve(), (ROOT / self.model_path).resolve())
            self.model_path = next((path for path in candidates if path.is_file()), candidates[-1])
        self.manifest_path = self.model_path.with_name("model_manifest.json")
        self.device_name = self._resolve_device(device)
        self.threshold = float(threshold)
        self.version = version or "drive-vessel-scratch-green-focal-dice-512-20260913-v1"
        self.expected_sha256 = expected_sha256
        self.input_size = 512
        self.preprocessing = "green"
        self._model: Any = None
        self._checksum: str | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._read_manifest()

    def _read_manifest(self) -> None:
        if not self.manifest_path.is_file():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.version = self.version if self.version != "drive-vessel-scratch-green-focal-dice-512-20260913-v1" else manifest.get("model_version", self.version)
            self.expected_sha256 = self.expected_sha256 or manifest.get("checkpoint_sha256")
            self.input_size = int(manifest.get("input_size", self.input_size))
            self.preprocessing = str(manifest.get("preprocessing", {}).get("mode", self.preprocessing))
            self.threshold = float(manifest.get("threshold", self.threshold))
        except Exception as exc:
            self._load_error = f"Invalid DRIVE research model manifest: {exc}"

    @property
    def is_configured(self) -> bool:
        return self.model_path.is_file() and self.manifest_path.is_file()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def health(self) -> dict[str, Any]:
        return {"model_version": self.version, "model_path": str(self.model_path), "manifest_path": str(self.manifest_path), "artifact_present": self.model_path.is_file(), "manifest_present": self.manifest_path.is_file(), "runtime_available": torch is not None, "loaded": self._model is not None, "load_error": self._load_error, "device": self.device_name, "architecture": "lightweight U-Net", "input_size": self.input_size, "preprocessing": self.preprocessing, "threshold": self.threshold, "checkpoint_sha256": self._checksum or self.expected_sha256, "production_promoted": False, "clinical_validation_claim": False}

    def verify_loadable(self) -> None:
        self._get_model()

    def analyze(self, image_rgb: Any, context: dict[str, Any]) -> EvidenceModuleResult:
        try:
            image = np.asarray(image_rgb, dtype=np.uint8)
            probability = self._predict(image)
            mask = probability >= self.threshold
            fov = np.any(image > 8, axis=2)
            count_mask = mask & fov
            inside = probability[fov] if np.any(fov) else probability.reshape(-1)
            confidence = float(np.mean(inside)) if inside.size else 0.0
            return EvidenceModuleResult(module=self.module, category="segmentation", status="model_inference", supported=True, implementation=self.name, confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4), count=_connected_components(count_mask), mask_data_uri=_mask_data_uri(count_mask), probability_map_data_uri=_probability_data_uri(probability), overlay_data_uri=_overlay_data_uri(image, count_mask), metadata={"model_version": self.version, "architecture": "lightweight U-Net", "checkpoint_sha256": self._checksum, "input_size": self.input_size, "preprocessing": self.preprocessing, "threshold": self.threshold, "device": self.device_name, "pixel_count": int(count_mask.sum()), "density_within_fov": float(count_mask.sum() / max(1, fov.sum())), "research_only": True, "production_promoted": False, "clinical_validation_claim": False}, issues=[{"type": "research_model", "message": "DRIVE-trained vessel output is experimental supporting evidence only; it is not a clinically validated biomarker."}])
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return EvidenceModuleResult(module=self.module, category="segmentation", status="unsupported", supported=False, implementation=self.name, issues=[{"type": "model_unavailable", "message": f"DRIVE research vessel model failed safely: {exc}"}], metadata=self.health())

    def predict_probability(self, image_rgb: Any) -> np.ndarray:
        return self._predict(np.asarray(image_rgb, dtype=np.uint8)).copy()

    def _predict(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape HxWx3")
        model = self._get_model()
        original_height, original_width = image.shape[:2]
        resized = np.asarray(Image.fromarray(image, mode="RGB").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR), dtype=np.uint8)
        values = normalize_input(resized, self.preprocessing)
        tensor = torch.from_numpy(values.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device_name)
        with torch.inference_mode():
            output = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
        probability = np.asarray(Image.fromarray(np.clip(output * 255.0, 0, 255).astype(np.uint8), mode="L").resize((original_width, original_height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return np.clip(probability, 0.0, 1.0)

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if torch is None:
                raise RuntimeError("PyTorch is unavailable for the DRIVE research vessel model")
            if not self.model_path.is_file():
                raise FileNotFoundError(f"DRIVE research vessel checkpoint is missing at {self.model_path}")
            if not self.manifest_path.is_file():
                raise FileNotFoundError(f"DRIVE research vessel manifest is missing at {self.manifest_path}")
            digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            if self.expected_sha256 and digest != self.expected_sha256:
                raise RuntimeError(f"DRIVE research vessel checkpoint SHA-256 mismatch: expected {self.expected_sha256}, got {digest}")
            from app.ml.models.evidence import build_vessel_segmentation_model

            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise RuntimeError("DRIVE research vessel checkpoint does not contain state_dict")
            model = build_vessel_segmentation_model()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            self._model = model.to(self.device_name)
            self._model.eval()
            self._checksum = digest
            self._load_error = None
            return self._model

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device and device != "auto":
            if device == "cuda" and (torch is None or not torch.cuda.is_available()):
                raise RuntimeError("CUDA was requested for the DRIVE research vessel model but is unavailable")
            return device
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
