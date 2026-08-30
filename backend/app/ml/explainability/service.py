"""Evidence-linked explainability for the RETINA-NEXUS screening path.

Grad-CAM is generated from the registered DR classifier artifact. Lesion
agreement is computed against the separate retinal-evidence output; neither
result changes the classifier prediction or establishes clinical causality.
"""

from __future__ import annotations

import asyncio
import base64
import io
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.ml.evidence.service import RetinalEvidenceAnalysis
from app.ml.explainability.interfaces import ExplainableClassifier


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return round(float(max(lower, min(upper, value))), 4)


def _data_uri(payload: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def _png(array: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGBA" if array.shape[-1] == 4 else "RGB").save(output, format="PNG", optimize=True)
    return _data_uri(output.getvalue())


def _rgb_png(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()


def _heatmap_rgba(attention: np.ndarray, alpha: int = 205) -> np.ndarray:
    values = np.clip(attention.astype(np.float32), 0.0, 1.0)
    # A deterministic blue-to-yellow-to-red map avoids depending on a plotting
    # library in the API process.
    red = np.clip(255.0 * (values * 1.8), 0, 255)
    green = np.clip(255.0 * (1.0 - np.abs(values - 0.5) * 2.0), 0, 255)
    blue = np.clip(255.0 * (1.0 - values * 1.8), 0, 255)
    rgba = np.stack([red, green, blue, np.full_like(values, alpha)], axis=-1)
    rgba[values < 0.04, 3] = 0
    return rgba.astype(np.uint8)


def _overlay(rgb: np.ndarray, attention: np.ndarray) -> str:
    heat = _heatmap_rgba(attention, alpha=255)
    weight = (0.12 + 0.58 * np.clip(attention, 0.0, 1.0))[..., None]
    colour = heat[..., :3].astype(np.float32)
    blended = np.clip(rgb.astype(np.float32) * (1.0 - weight) + colour * weight, 0, 255)
    return _png(blended.astype(np.uint8))


def _decode_mask(data_uri: str | None, shape: tuple[int, int]) -> np.ndarray:
    if not data_uri or "," not in data_uri:
        return np.zeros(shape, dtype=bool)
    try:
        payload = base64.b64decode(data_uri.split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as image:
            rgba = np.asarray(image.convert("RGBA"))
        if rgba.shape[:2] != shape:
            rgba = np.asarray(Image.fromarray(rgba, mode="RGBA").resize((shape[1], shape[0]), Image.Resampling.NEAREST))
        return rgba[:, :, 3] > 0
    except Exception:
        return np.zeros(shape, dtype=bool)


def _mask_from_regions(regions: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for region in regions:
        try:
            x, y = max(0, int(region["x"])), max(0, int(region["y"]))
            right = min(width, x + max(1, int(region["width"])))
            bottom = min(height, y + max(1, int(region["height"])))
            if right > x and bottom > y:
                mask[y:bottom, x:right] = True
        except (KeyError, TypeError, ValueError):
            continue
    return mask


def _lesion_mask(evidence: RetinalEvidenceAnalysis, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for name, module in evidence.modules.items():
        if not module.get("supported"):
            continue
        is_lesion = module.get("category") == "lesion_detection" or name == "exudate_segmentation"
        if not is_lesion:
            continue
        mask |= _decode_mask(module.get("mask_data_uri"), shape)
        if not mask.any():
            mask |= _mask_from_regions(module.get("bounding_regions") or [], shape)
    return mask


def _attention_mask(attention: np.ndarray) -> np.ndarray:
    if not attention.size:
        return np.zeros(attention.shape, dtype=bool)
    threshold = max(0.5, float(np.percentile(attention, 75)))
    return attention >= threshold


def _agreement(attention: np.ndarray, lesion: np.ndarray) -> dict[str, Any]:
    if not lesion.any():
        return {
            "status": "UNAVAILABLE", "score": None,
            "metrics": {"attention_coverage": round(float(_attention_mask(attention).mean()), 6), "lesion_coverage": 0.0, "intersection_over_union": None, "dice": None, "attention_in_lesion": None, "lesion_in_attention": None},
            "reason": "No supported lesion region was available for comparison; this is not evidence of disagreement.",
            "note": "Engineering explainability metric only; overlap does not prove clinical causality.",
        }
    attention_regions = _attention_mask(attention)
    intersection = int(np.logical_and(attention_regions, lesion).sum())
    union = int(np.logical_or(attention_regions, lesion).sum())
    attention_count = int(attention_regions.sum())
    lesion_count = int(lesion.sum())
    iou = intersection / max(1, union)
    dice = (2.0 * intersection) / max(1, attention_count + lesion_count)
    attention_in_lesion = intersection / max(1, attention_count)
    lesion_in_attention = intersection / max(1, lesion_count)
    score = 0.5 * dice + 0.5 * attention_in_lesion
    status = "HIGH AGREEMENT" if score >= 0.6 else "MODERATE AGREEMENT" if score >= 0.3 else "LOW AGREEMENT"
    return {
        "status": status, "score": _clamp(score),
        "metrics": {"attention_coverage": round(attention_count / attention_regions.size, 6), "lesion_coverage": round(lesion_count / lesion.size, 6), "intersection_over_union": _clamp(iou), "dice": _clamp(dice), "attention_in_lesion": _clamp(attention_in_lesion), "lesion_in_attention": _clamp(lesion_in_attention)},
        "reason": "Attention and supported lesion regions were compared at the working image resolution.",
        "note": "Engineering explainability metric only; overlap does not prove clinical causality.",
    }


@dataclass
class ExplainabilityAnalysis:
    image_id: str
    screening_session_id: str
    predicted_class: int
    predicted_class_label: str
    model_version: str
    classification: dict[str, Any]
    grad_cam: dict[str, Any]
    lesion_evidence_map_data_uri: str | None
    attention_lesion_agreement: dict[str, Any]
    explanation_stability: dict[str, Any]
    counterfactual: dict[str, Any]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "screening_session_id": self.screening_session_id,
            "predicted_class": self.predicted_class,
            "predicted_class_label": self.predicted_class_label,
            "model_version": self.model_version,
            "classification": self.classification,
            "grad_cam": self.grad_cam,
            "lesion_evidence_map_data_uri": self.lesion_evidence_map_data_uri,
            "attention_lesion_agreement": self.attention_lesion_agreement,
            "explanation_stability": self.explanation_stability,
            "counterfactual": self.counterfactual,
            "note": self.note,
        }


class ExplainabilityService:
    """Generate model-linked explanations without altering screening output."""

    def __init__(self, classifier: ExplainableClassifier, stability_enabled: bool = False, counterfactual_enabled: bool = False, max_stability_variants: int = 3):
        self.classifier = classifier
        self.stability_enabled = stability_enabled
        self.counterfactual_enabled = counterfactual_enabled
        self.max_stability_variants = max(1, min(5, max_stability_variants))

    async def analyze(
        self,
        image_bytes: bytes,
        image_id: str,
        screening_session_id: str,
        evidence: RetinalEvidenceAnalysis,
        run_stability: bool | None = None,
        run_counterfactual: bool | None = None,
    ) -> ExplainabilityAnalysis:
        base = await self._explain(image_bytes)
        attention = np.asarray(base.attention_map, dtype=np.float32).clip(0.0, 1.0)
        with Image.open(io.BytesIO(image_bytes)) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        if attention.shape != rgb.shape[:2]:
            attention = np.asarray(Image.fromarray(attention, mode="F").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32).clip(0.0, 1.0)
        lesion = _lesion_mask(evidence, attention.shape)
        agreement = _agreement(attention, lesion)
        classification = {
            "predicted_grade": base.prediction.predicted_grade,
            "predicted_grade_label": base.prediction.predicted_grade_label,
            "probabilities": base.prediction.probabilities,
            "referable_dr": base.prediction.referable_dr,
            "referable_probability": base.prediction.referable_probability,
            "raw_confidence": base.prediction.raw_confidence,
            "model_name": base.prediction.model_name,
            "model_version": base.prediction.model_version,
            "backbone": base.prediction.backbone,
        }
        stability = await self._stability(image_bytes, base, run_stability)
        counterfactual = await self._counterfactual(image_bytes, base, attention, lesion, run_counterfactual)
        return ExplainabilityAnalysis(
            image_id=image_id, screening_session_id=screening_session_id,
            predicted_class=base.prediction.predicted_grade,
            predicted_class_label=base.prediction.predicted_grade_label,
            model_version=base.prediction.model_version, classification=classification,
            grad_cam={
                "heatmap_data_uri": _png(_heatmap_rgba(attention)),
                "overlay_data_uri": _overlay(rgb, attention),
                "normalized_attention_map_data_uri": _png(_heatmap_rgba(attention, alpha=230)),
                "target_class": base.target_class, "target_layer": base.target_layer,
                "map_width": int(attention.shape[1]), "map_height": int(attention.shape[0]),
            },
            lesion_evidence_map_data_uri=evidence.evidence_map_data_uri,
            attention_lesion_agreement=agreement,
            explanation_stability=stability,
            counterfactual=counterfactual,
            note="Grad-CAM is linked to the registered classifier and compared with the separate evidence layer. Explainability metrics are engineering diagnostics, not clinical causality or a trust guarantee.",
        )

    async def _stability(self, image_bytes: bytes, base: Any, requested: bool | None) -> dict[str, Any]:
        enabled = self.stability_enabled if requested is None else requested
        if not enabled:
            return {"status": "SKIPPED", "reason": "Stability perturbations are disabled for real-time mode. Set run_stability=true or enable EXPLAINABILITY_STABILITY_ENABLED.", "prediction_stability": None, "grad_cam_stability": None, "variants": []}
        variants = self._variants(image_bytes)[: self.max_stability_variants]
        records: list[dict[str, Any]] = []
        cam_differences: list[float] = []
        same_predictions = 0
        for name, variant in variants:
            try:
                explanation = await self._explain(variant, target_class=base.prediction.predicted_grade)
                variant_attention = np.asarray(explanation.attention_map, dtype=np.float32)
                base_attention = np.asarray(base.attention_map, dtype=np.float32)
                if variant_attention.shape != base_attention.shape:
                    variant_attention = np.asarray(Image.fromarray(variant_attention, mode="F").resize((base_attention.shape[1], base_attention.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
                difference = float(np.mean(np.abs(base_attention - variant_attention)))
                cam_differences.append(difference)
                same = explanation.prediction.predicted_grade == base.prediction.predicted_grade
                same_predictions += int(same)
                records.append({"name": name, "predicted_grade": explanation.prediction.predicted_grade, "raw_confidence": explanation.prediction.raw_confidence, "prediction_unchanged": same, "grad_cam_mean_absolute_difference": round(difference, 6)})
            except Exception as exc:
                records.append({"name": name, "status": "failed", "error": str(exc)})
        completed = len(cam_differences)
        return {
            "status": "COMPLETED" if completed else "FAILED",
            "variant_count": len(variants), "completed_variant_count": completed,
            "prediction_stability": _clamp(same_predictions / max(1, completed)) if completed else None,
            "grad_cam_stability": _clamp(1.0 - float(np.mean(cam_differences))) if completed else None,
            "variants": records,
            "note": "Controlled perturbation diagnostic; not a clinical validation result.",
        }

    async def _explain(self, image_bytes: bytes, target_class: int | None = None) -> Any:
        method = getattr(self.classifier, "explain_async", None) or getattr(self.classifier, "explain", None)
        if method is None:
            raise RuntimeError("The configured classifier does not expose a Grad-CAM explanation interface.")
        result = method(image_bytes, target_class=target_class)
        return await result if inspect.isawaitable(result) else result

    async def _counterfactual(self, image_bytes: bytes, base: Any, attention: np.ndarray, lesion: np.ndarray, requested: bool | None) -> dict[str, Any]:
        enabled = self.counterfactual_enabled if requested is None else requested
        if not enabled:
            return {"status": "SKIPPED", "reason": "Counterfactual analysis is disabled by default because it adds another inference pass.", "experimental": True}
        selected = lesion if lesion.any() else _attention_mask(attention)
        if not selected.any():
            return {"status": "UNAVAILABLE", "reason": "No suspicious region was available to mask.", "experimental": True}
        with Image.open(io.BytesIO(image_bytes)) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        blurred = np.asarray(Image.fromarray(rgb, mode="RGB").filter(ImageFilter.GaussianBlur(radius=9)), dtype=np.uint8)
        counterfactual_rgb = rgb.copy()
        counterfactual_rgb[selected] = blurred[selected]
        counterfactual_bytes = _rgb_png(counterfactual_rgb)
        try:
            prediction = await self.classifier.classify(counterfactual_bytes)
        except Exception as exc:
            return {"status": "FAILED", "reason": str(exc), "experimental": True}
        original_probability = float(base.prediction.probabilities.get(base.prediction.predicted_grade_label, 0.0))
        changed_probability = float(prediction.probabilities.get(base.prediction.predicted_grade_label, 0.0))
        return {
            "status": "COMPLETED", "experimental": True,
            "selected_region": "lesion_evidence" if lesion.any() else "top_attention",
            "masked_region_data_uri": _png(np.where(selected[..., None], np.array([245, 180, 40, 180], dtype=np.uint8), np.zeros((*selected.shape, 4), dtype=np.uint8))),
            "original_predicted_grade": base.prediction.predicted_grade,
            "counterfactual_predicted_grade": prediction.predicted_grade,
            "predicted_grade_changed": prediction.predicted_grade != base.prediction.predicted_grade,
            "original_target_probability": round(original_probability, 6),
            "counterfactual_target_probability": round(changed_probability, 6),
            "target_probability_delta": round(changed_probability - original_probability, 6),
            "note": "Experimental region-masking diagnostic. A prediction change does not establish lesion causality.",
        }

    @staticmethod
    def _variants(image_bytes: bytes) -> list[tuple[str, bytes]]:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
        brightness_low = ImageEnhance.Brightness(image).enhance(0.95)
        brightness_high = ImageEnhance.Brightness(image).enhance(1.05)
        rotated = image.rotate(2.0, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        rng = np.random.default_rng(17)
        noise = np.clip(np.asarray(image, dtype=np.int16) + rng.normal(0, 2.0, np.asarray(image).shape), 0, 255).astype(np.uint8)
        candidates = [("brightness_minus_5pct", brightness_low), ("brightness_plus_5pct", brightness_high), ("rotation_plus_2deg", rotated), ("minor_noise", Image.fromarray(noise, mode="RGB"))]
        encoded: list[tuple[str, bytes]] = []
        for name, variant in candidates:
            output = io.BytesIO()
            variant.save(output, format="PNG", optimize=True)
            encoded.append((name, output.getvalue()))
        return encoded
