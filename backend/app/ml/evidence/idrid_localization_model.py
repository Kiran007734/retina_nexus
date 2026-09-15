"""Opt-in IDRiD optic-disc/fovea localization evidence adapter.

This adapter is intentionally separate from DR classification and from the
RetinaGuard decision engine.  It emits anatomical supporting evidence only.
The default runtime remains unchanged unless the adapter is explicitly
enabled and a frozen model manifest/checkpoint are available.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.evidence.interfaces import EvidenceModuleResult

try:  # Optional ML runtime; the API can still start without this feature.
    import torch
except Exception:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = ROOT / "ml" / "weights" / "localization" / "idrid" / "checkpoint_best.pt"
LANDMARKS = ("optic_disc", "fovea")
MODULES = {
    "optic_disc": "optic_disc_localization",
    "fovea": "fovea_localization",
}
MODEL_VERSION = "idrid-localization-frozen"


class IDRiDLocalizationAdapter:
    """Lazy-loading two-landmark heatmap/coordinate model adapter."""

    name = "idrid-shared-heatmap-coord"
    module = "idrid_anatomical_localization"

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        version: str | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.is_absolute():
            self.model_path = (ROOT / self.model_path).resolve()
        self.manifest_path = self.model_path.with_name("model_manifest.json")
        self.device_name = self._resolve_device(device)
        self.version = version or MODEL_VERSION
        self.expected_sha256 = expected_sha256
        self._model: Any = None
        self._checksum: str | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self.input_width = 512
        self.input_height = 352
        if self.manifest_path.is_file():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self.version = version or manifest.get("model_version", self.version)
                self.expected_sha256 = self.expected_sha256 or manifest.get("checkpoint_sha256")
                self.input_width = int(manifest.get("input_resolution", [self.input_width, self.input_height])[0])
                self.input_height = int(manifest.get("input_resolution", [self.input_width, self.input_height])[1])
            except Exception as exc:
                self._load_error = f"Invalid IDRiD localization model manifest: {exc}"

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
            "manifest_path": str(self.manifest_path),
            "artifact_present": self.model_path.is_file(),
            "manifest_present": self.manifest_path.is_file(),
            "runtime_available": torch is not None,
            "loaded": self._model is not None,
            "load_error": self._load_error,
            "device": self.device_name,
            "landmarks": list(LANDMARKS),
            "checkpoint_sha256": self._checksum or self.expected_sha256,
            "production_promoted": False,
            "clinical_validation_claim": False,
        }

    def verify_loadable(self) -> None:
        self._get_model()

    def analyze(self, image_rgb: Any, context: dict[str, Any]) -> EvidenceModuleResult:
        requested_module = str(context.get("requested_module", ""))
        landmark = next((name for name, module in MODULES.items() if module == requested_module), None)
        category = "landmark"
        if landmark is None:
            return EvidenceModuleResult(
                module=requested_module or self.module,
                category=category,
                status="unsupported",
                supported=False,
                implementation=self.name,
                issues=[{"type": "unsupported", "message": f"Unknown localization module '{requested_module}'."}],
            )
        try:
            prediction = self._predict(np.asarray(image_rgb, dtype=np.uint8))
            landmark_index = LANDMARKS.index(landmark)
            point = prediction["points"][landmark_index]
            confidence = float(prediction["confidence"][landmark_index])
            width = int(prediction["image_width"])
            height = int(prediction["image_height"])
            item = {
                "landmark_type": landmark,
                "status": "model_inference",
                "method": self.name,
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
                "x_normalized": round(float(point[0]) / max(width, 1), 6),
                "y_normalized": round(float(point[1]) / max(height, 1), 6),
                "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
                "eye_assumption": context.get("eye") or "unspecified",
            }
            return EvidenceModuleResult(
                module=requested_module,
                category=category,
                status="model_inference",
                supported=True,
                implementation=self.name,
                confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
                landmarks=[item],
                metadata={
                    "model_version": self.version,
                    "architecture": "shared compact heatmap network with auxiliary coordinate regression",
                    "landmark": landmark,
                    "checkpoint_sha256": self._checksum,
                    "input_resolution": [self.input_width, self.input_height],
                    "preprocessing": "RGB aspect-ratio-preserving letterbox; ImageNet normalization",
                    "prediction_method": "heatmap spatial soft-argmax",
                    "device": self.device_name,
                    "production_promoted": False,
                    "clinical_validation_claim": False,
                },
                issues=[{"type": "research_model", "message": "IDRiD localization is supporting evidence only; it is not a clinical diagnosis or production-promoted model."}],
            )
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return EvidenceModuleResult(
                module=requested_module,
                category=category,
                status="unsupported",
                supported=False,
                implementation=self.name,
                issues=[{"type": "model_unavailable", "message": f"IDRiD localization model failed safely: {exc}"}],
                metadata=self.health(),
            )

    def _predict(self, image_rgb: np.ndarray) -> dict[str, Any]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape HxWx3")
        from ml.localization.idrid import inverse_points, letterbox, localization_tensor, decode_heatmaps

        height, width = image_rgb.shape[:2]
        dummy_points = np.zeros((2, 2), dtype=np.float32)
        canvas, _dummy, transform = letterbox(image_rgb, dummy_points, self.input_width, self.input_height)
        tensor = localization_tensor(canvas).unsqueeze(0).to(self.device_name)
        with torch.inference_mode():
            outputs = self._get_model()(tensor)
            heatmap_points, confidence, _probabilities = decode_heatmaps(outputs["heatmaps"])
            points = inverse_points(heatmap_points[0].detach().cpu().numpy(), transform)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        return {"points": points, "confidence": confidence[0].detach().cpu().numpy(), "image_width": width, "image_height": height}

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise FileNotFoundError(f"IDRiD localization checkpoint is missing at {self.model_path}")
            if not self.manifest_path.is_file():
                raise FileNotFoundError(f"IDRiD localization model manifest is missing at {self.manifest_path}")
            if torch is None:
                raise RuntimeError("IDRiD localization runtime is unavailable; install torch")
            digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            if self.expected_sha256 and digest != self.expected_sha256:
                raise RuntimeError(f"IDRiD localization checkpoint SHA-256 mismatch: expected {self.expected_sha256}, got {digest}")
            from ml.localization.idrid import SharedLandmarkHeatmapNet

            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            model = SharedLandmarkHeatmapNet.build()
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
