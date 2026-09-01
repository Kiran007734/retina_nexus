"""Runtime classifier adapter loaded from a registered training artifact."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.ml.models.classifier import ReferableDRMapping, build_classifier, severity_probabilities


GRADE_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


class ClassifierNotConfiguredError(RuntimeError):
    """Raised when no usable trained model artifact is configured."""


@dataclass
class DRPrediction:
    predicted_grade: int
    predicted_grade_label: str
    probabilities: dict[str, float]
    referable_dr: bool
    referable_probability: float
    raw_confidence: float
    model_name: str
    model_version: str
    backbone: str
    referable_mapping: dict[str, Any]
    hierarchical_probabilities: dict[str, dict[str, float]]
    ordinal_mode: bool
    severity_logits: list[float] | None = None


@dataclass
class DRExplanation:
    """A model-linked attention artifact for one DR prediction."""

    prediction: DRPrediction
    attention_map: Any
    target_class: int
    input_width: int
    input_height: int
    target_layer: str


class TorchDRClassificationService:
    """Loads a checkpoint lazily and performs CPU/GPU inference.

    An absent artifact is an explicit configuration error; this service never
    returns a fabricated grade or confidence.
    """

    def __init__(self, model_path: str | None, backbone: str, model_version: str | None = None, device: str = "auto", referable_mapping: ReferableDRMapping | None = None):
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.configured_backbone = backbone
        self.configured_model_version = model_version
        self.device_name = device
        self.mapping = referable_mapping or ReferableDRMapping()
        self._model = None
        self._torch = None
        self._transform = None
        self._artifact_config: dict[str, Any] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision import transforms
        except Exception as exc:
            raise ClassifierNotConfiguredError("PyTorch and torchvision are not installed or cannot be imported. Install backend/requirements-ml.txt before classification.") from exc
        if self.model_path is None:
            raise ClassifierNotConfiguredError("No classifier checkpoint is configured. Set CLASSIFIER_MODEL_PATH after training and registering a model artifact.")
        if not self.model_path.is_file():
            raise ClassifierNotConfiguredError("The configured classifier checkpoint is unavailable. Run scripts/verify_models.py and set CLASSIFIER_MODEL_PATH to a valid registered artifact.")
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ClassifierNotConfiguredError(f"The configured classifier checkpoint could not be loaded ({type(exc).__name__}). Verify the model manifest and installed ML runtime.") from exc
        model_config = checkpoint.get("model_config", {})
        backbone = model_config.get("backbone", self.configured_backbone)
        ordinal_mode = bool(model_config.get("ordinal_mode", False))
        input_size = int(model_config.get("input_size", 224))
        try:
            model = build_classifier(backbone=backbone, num_classes=5, pretrained=False, ordinal_mode=ordinal_mode)
            model.load_state_dict(checkpoint["state_dict"])
        except Exception as exc:
            raise ClassifierNotConfiguredError(f"The classifier artifact is incompatible with its registered model configuration ({type(exc).__name__}).") from exc
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise ClassifierNotConfiguredError("CLASSIFIER_DEVICE=cuda was requested but CUDA is unavailable. Use CLASSIFIER_DEVICE=cpu or install a CUDA-enabled PyTorch build.")
        if self.device_name == "cuda" or (self.device_name == "auto" and torch.cuda.is_available()):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device)
        model.eval()
        self._torch = torch
        self._model = model
        self._device = device
        self._ordinal_mode = ordinal_mode
        self._artifact_config = {**model_config, **checkpoint.get("artifact", {})}
        self._transform = transforms.Compose([
            transforms.Resize((input_size, input_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_bytes: bytes) -> DRPrediction:
        self._load()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ClassifierNotConfiguredError(f"Classifier could not decode the stored image: {exc}") from exc
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(tensor)
        return self._prediction_from_outputs(outputs)

    def verify_loadable(self) -> None:
        """Load the configured artifact for startup/readiness verification."""
        self._load()

    def _prediction_from_outputs(self, outputs: dict[str, Any]) -> DRPrediction:
        probabilities_tensor = severity_probabilities(outputs, self._ordinal_mode)[0].detach().cpu().tolist()
        stage1 = self._torch.softmax(outputs["stage1_logits"], dim=1)[0].detach().cpu().tolist()
        stage2 = self._torch.softmax(outputs["stage2_logits"], dim=1)[0].detach().cpu().tolist()
        grade = int(max(range(len(probabilities_tensor)), key=probabilities_tensor.__getitem__))
        probabilities = {GRADE_LABELS[index]: round(float(value), 6) for index, value in enumerate(probabilities_tensor)}
        model_version = self.configured_model_version or self._artifact_config.get("model_version") or self._artifact_config.get("version") or "unversioned"
        model_name = self._artifact_config.get("model_name", "RETINA-NEXUS DR classifier")
        mapping_probability = self.mapping.probability(probabilities_tensor)
        return DRPrediction(
            predicted_grade=grade, predicted_grade_label=GRADE_LABELS[grade], probabilities=probabilities,
            referable_dr=self.mapping.is_referable(grade), referable_probability=round(float(mapping_probability), 6),
            raw_confidence=round(float(max(probabilities_tensor)), 6), model_name=model_name,
            model_version=model_version, backbone=self._artifact_config.get("backbone", self.configured_backbone),
            referable_mapping=self.mapping.to_dict(),
            hierarchical_probabilities={"dr_vs_no_dr": {"non_dr": round(float(stage1[0]), 6), "dr": round(float(stage1[1]), 6)}, "referable_vs_non_referable": {"non_referable": round(float(stage2[0]), 6), "referable": round(float(stage2[1]), 6)}}, ordinal_mode=self._ordinal_mode,
            severity_logits=outputs.get("severity_logits")[0].detach().cpu().tolist() if outputs.get("severity_logits") is not None else None,
        )

    def explain(self, image_bytes: bytes, target_class: int | None = None) -> DRExplanation:
        """Generate Grad-CAM from the registered classifier's final feature map.

        This is intentionally coupled to the actual classifier artifact. When
        no artifact is configured, the caller receives an explicit setup error
        instead of a synthetic heatmap.
        """
        self._load()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ClassifierNotConfiguredError(f"Classifier could not decode the stored image: {exc}") from exc
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        tensor.requires_grad_(True)
        activations: dict[str, Any] = {}

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            activations["value"] = output
            if hasattr(output, "retain_grad") and output.requires_grad:
                output.retain_grad()

        handle = self._model.feature_extractor.register_forward_hook(capture)
        try:
            self._model.zero_grad(set_to_none=True)
            with self._torch.enable_grad():
                outputs = self._model(tensor)
                prediction = self._prediction_from_outputs(outputs)
                selected_class = prediction.predicted_grade if target_class is None else int(target_class)
                selected_class = max(0, min(4, selected_class))
                target_score = self._target_score(outputs, selected_class)
                target_score.sum().backward()
            activation = activations.get("value")
            gradients = getattr(activation, "grad", None)
            if activation is None or gradients is None or activation.ndim != 4:
                raise ClassifierNotConfiguredError("The classifier backbone did not expose a differentiable spatial feature map for Grad-CAM.")
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = self._torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = self._torch.nn.functional.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
            attention = cam[0, 0].detach().cpu().numpy().astype("float32")
            low, high = float(attention.min()), float(attention.max())
            attention = ((attention - low) / max(1e-8, high - low)).clip(0.0, 1.0)
        finally:
            handle.remove()
            self._model.zero_grad(set_to_none=True)
        return DRExplanation(
            prediction=prediction, attention_map=attention, target_class=selected_class,
            input_width=image.width, input_height=image.height,
            target_layer="feature_extractor_output",
        )

    def _target_score(self, outputs: dict[str, Any], target_class: int):
        if not self._ordinal_mode:
            return outputs["severity_logits"][:, target_class]
        logits = outputs["ordinal_logits"]
        if target_class == 0:
            return -logits[:, 0]
        if target_class == 4:
            return logits[:, -1]
        return logits[:, target_class - 1] - logits[:, target_class]

    async def classify(self, image_bytes: bytes) -> DRPrediction:
        return await asyncio.to_thread(self.predict, image_bytes)

    async def explain_async(self, image_bytes: bytes, target_class: int | None = None) -> DRExplanation:
        return await asyncio.to_thread(self.explain, image_bytes, target_class)

    def predict_with_dropout_sync(self, image_bytes: bytes, samples: int = 8) -> list[list[float]]:
        """Experimental MC-dropout samples with normalization layers kept in eval mode."""
        self._load()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ClassifierNotConfiguredError(f"Classifier could not decode the stored image: {exc}") from exc
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        dropout_modules = []
        for module in self._model.modules():
            if isinstance(module, self._torch.nn.Dropout):
                dropout_modules.append(module)
        try:
            self._model.eval()
            for module in dropout_modules:
                module.train()
            results: list[list[float]] = []
            with self._torch.inference_mode():
                for _ in range(max(2, min(30, int(samples)))):
                    outputs = self._model(tensor)
                    results.append([float(value) for value in severity_probabilities(outputs, self._ordinal_mode)[0].detach().cpu().tolist()])
            return results
        finally:
            self._model.eval()

    async def predict_with_dropout(self, image_bytes: bytes, samples: int = 8) -> list[list[float]]:
        return await asyncio.to_thread(self.predict_with_dropout_sync, image_bytes, samples)
