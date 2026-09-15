"""Research-gated referable-risk fusion for the existing screening pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.ml.inference.classifier import DRPrediction


FUSION_THRESHOLD = 0.40
RETGUARD_THRESHOLD = 0.204983
RETGUARD_MODEL_VERSION = "retguard-dr-v1.0.0"
RETGUARD_MODEL_SHA256 = "f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b"


@dataclass(frozen=True)
class ReferableFusionResult:
    status: str
    primary_probability: float | None
    primary_referable: bool | None
    verifier_probability: float | None
    verifier_referable: bool | None
    fused_probability: float | None
    fused_referable: bool | None
    disagreement: bool | None
    review_recommended: bool
    fusion_rule: str
    fusion_threshold: float = FUSION_THRESHOLD
    verifier_model_version: str | None = None
    verifier_model_sha256: str | None = None
    verifier_ood: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "primary_probability": self.primary_probability,
            "primary_referable": self.primary_referable,
            "verifier_probability": self.verifier_probability,
            "verifier_referable": self.verifier_referable,
            "fused_probability": self.fused_probability,
            "fused_referable": self.fused_referable,
            "disagreement": self.disagreement,
            "review_recommended": self.review_recommended,
            "fusion_rule": self.fusion_rule,
            "fusion_threshold": self.fusion_threshold,
            "verifier_threshold": RETGUARD_THRESHOLD,
            "verifier_model_version": self.verifier_model_version,
            "verifier_model_sha256": self.verifier_model_sha256,
            "verifier_ood": self.verifier_ood,
            "error": self.error,
            "clinical_validation_claim": False,
        }


class RetguardVerifier:
    """Lazy official RETGUARD DR adapter; it is never loaded unless enabled."""

    def __init__(self, model_path: str | Path | None, expected_sha256: str | None = RETGUARD_MODEL_SHA256, model_version: str = RETGUARD_MODEL_VERSION):
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.expected_sha256 = expected_sha256
        self.model_version = model_version
        self._session = None
        self._gate = None
        self._preprocess = None
        self._gap = None
        self._actual_sha256: str | None = None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load(self) -> None:
        if self._session is not None:
            return
        if self.model_path is None:
            raise RuntimeError("No RETGUARD verifier checkpoint is configured.")
        if not self.model_path.is_file():
            raise RuntimeError(f"RETGUARD verifier checkpoint is unavailable: {self.model_path}")
        actual_sha = self._sha256(self.model_path)
        self._actual_sha256 = actual_sha
        if self.expected_sha256 and actual_sha.lower() != self.expected_sha256.lower():
            raise RuntimeError(f"RETGUARD verifier SHA-256 mismatch: expected {self.expected_sha256}, got {actual_sha}")
        try:
            import onnx
            import onnxruntime as ort
            from retguard.ood import MahalanobisGate
            from retguard.predictor import CAM_SPATIAL_NODE, GAP_OUTPUT_NODE
        except Exception as exc:
            raise RuntimeError("RETGUARD dependencies are unavailable; install onnxruntime, onnx, and the authorized retguard package.") from exc
        gate_path = self.model_path.with_name("ood_gate_dr_v1.0.0.npz")
        if not gate_path.is_file():
            raise RuntimeError(f"RETGUARD OOD gate artifact is unavailable: {gate_path}")
        model = onnx.load(str(self.model_path))
        self._gap = GAP_OUTPUT_NODE["dr"]
        spatial = CAM_SPATIAL_NODE["dr"]
        produced = {tensor for node in model.graph.node for tensor in node.output}
        for tensor_name, shape in ((self._gap, ["batch_size", 1280]), (spatial, ["batch_size", 1280, "h", "w"])):
            if tensor_name not in produced:
                raise RuntimeError(f"RETGUARD graph is missing required internal tensor {tensor_name}")
            model.graph.output.append(onnx.helper.make_tensor_value_info(tensor_name, onnx.TensorProto.FLOAT, shape))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
        self._gate = MahalanobisGate.from_npz(gate_path)
        self._preprocess = __import__("retguard.preprocess", fromlist=["preprocess_dr"]).preprocess_dr

    @staticmethod
    def _views(tensor: Any) -> Any:
        import numpy as np

        horizontal_flip = np.flip(tensor, axis=2)
        vertical_flip = np.flip(tensor, axis=1)
        return np.ascontiguousarray(np.stack([
            tensor,
            np.rot90(tensor, k=1, axes=(1, 2)),
            np.rot90(tensor, k=2, axes=(1, 2)),
            np.rot90(tensor, k=3, axes=(1, 2)),
            horizontal_flip,
            vertical_flip,
            np.rot90(horizontal_flip, k=1, axes=(1, 2)),
            np.rot90(vertical_flip, k=1, axes=(1, 2)),
        ], axis=0))

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        self._load()
        import numpy as np
        from retguard.calibrate import apply_temperature, sigmoid
        from retguard.constants import DR_TEMPERATURE

        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = np.asarray(image.convert("RGB"))
        tensor = self._preprocess(rgb)
        views = self._views(tensor)
        logits, features = self._session.run(["logit", self._gap], {"input": views})
        probability = float(sigmoid(apply_temperature(float(logits.mean(axis=0)), DR_TEMPERATURE)))
        ood_score = float(self._gate.score(features[:1])[0])
        return {
            "probability": probability,
            "referable": probability >= RETGUARD_THRESHOLD,
            "model_version": self.model_version,
            "model_sha256": self._actual_sha256,
            "ood": {"status": "SHIFTED" if ood_score > self._gate.threshold else "IN_DISTRIBUTION", "score": ood_score, "threshold": float(self._gate.threshold), "method": "RETGUARD Mahalanobis gate"},
        }


class ReferableFusionService:
    """Apply the frozen research rule while preserving primary severity."""

    def __init__(self, enabled: bool = False, verifier: Any | None = None, fusion_threshold: float = FUSION_THRESHOLD):
        if not 0.0 < float(fusion_threshold) < 1.0:
            raise ValueError("Referable fusion threshold must be between 0 and 1")
        self.enabled = bool(enabled)
        self.fusion_threshold = float(fusion_threshold)
        self.verifier = verifier

    @staticmethod
    def disabled() -> "ReferableFusionService":
        return ReferableFusionService(enabled=False)

    async def evaluate(self, image_bytes: bytes | None, primary: DRPrediction | None) -> ReferableFusionResult:
        primary_probability = float(primary.referable_probability) if primary is not None else None
        primary_referable = bool(primary.referable_dr) if primary is not None else None
        if not self.enabled:
            return self._primary_fallback(primary_probability, primary_referable, "DISABLED")
        verifier_result: dict[str, Any] | None = None
        try:
            if image_bytes is None or self.verifier is None:
                raise RuntimeError("Independent verifier is unavailable for this request.")
            verifier_result = await asyncio.to_thread(self.verifier.predict, image_bytes)
        except Exception as exc:
            if primary is not None:
                result = self._primary_fallback(primary_probability, primary_referable, "VERIFIER_UNAVAILABLE")
                return ReferableFusionResult(**{**result.__dict__, "review_recommended": True, "error": str(exc)})
            return ReferableFusionResult(
                status="INSUFFICIENT_EVIDENCE", primary_probability=None, primary_referable=None,
                verifier_probability=None, verifier_referable=None, fused_probability=None, fused_referable=None,
                disagreement=None, review_recommended=True, fusion_rule=self._rule(), fusion_threshold=self.fusion_threshold,
                error=str(exc),
            )
        verifier_probability = float(verifier_result["probability"])
        verifier_referable = bool(verifier_result.get("referable", verifier_probability >= RETGUARD_THRESHOLD))
        if primary is None:
            return ReferableFusionResult(
                status="VERIFIER_ONLY_FALLBACK", primary_probability=None, primary_referable=None,
                verifier_probability=verifier_probability, verifier_referable=verifier_referable,
                fused_probability=verifier_probability, fused_referable=verifier_probability >= self.fusion_threshold,
                disagreement=None, review_recommended=True, fusion_rule=self._rule(), fusion_threshold=self.fusion_threshold,
                verifier_model_version=verifier_result.get("model_version"), verifier_model_sha256=verifier_result.get("model_sha256"), verifier_ood=verifier_result.get("ood"),
            )
        fused_probability = max(primary_probability, verifier_probability)
        disagreement = primary_referable != verifier_referable
        return ReferableFusionResult(
            status="COMPLETED", primary_probability=primary_probability, primary_referable=primary_referable,
            verifier_probability=verifier_probability, verifier_referable=verifier_referable,
            fused_probability=fused_probability, fused_referable=fused_probability >= self.fusion_threshold,
            disagreement=disagreement, review_recommended=bool(disagreement or (verifier_result.get("ood") or {}).get("status") == "SHIFTED"),
            fusion_rule=self._rule(), fusion_threshold=self.fusion_threshold,
            verifier_model_version=verifier_result.get("model_version"), verifier_model_sha256=verifier_result.get("model_sha256"), verifier_ood=verifier_result.get("ood"),
        )

    def _primary_fallback(self, probability: float | None, referable: bool | None, status: str) -> ReferableFusionResult:
        return ReferableFusionResult(
            status=status, primary_probability=probability, primary_referable=referable,
            verifier_probability=None, verifier_referable=None, fused_probability=probability,
            fused_referable=referable, disagreement=None, review_recommended=False,
            fusion_rule=self._rule(), fusion_threshold=self.fusion_threshold, error=None,
        )

    def _rule(self) -> str:
        return f"fused_referable = max(primary_referable_probability, verifier_referable_probability) >= {self.fusion_threshold:g}; severity_grade = argmax(primary P0..P4)"
