"""Coarse-to-fine retinal evidence analysis.

This module is deliberately separate from DR classification. Its default
baselines are transparent image-processing heuristics for development and
visualization only. They are not clinically validated detectors.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from app.core.safe_errors import safe_error_message
from app.ml.evidence.dataset_support import evidence_dataset_support
from app.ml.evidence.interfaces import EvidenceModelAdapter, EvidenceModuleResult
from app.ml.quality.trust_gate import ImageTrustGateService

try:
    import cv2
except Exception:  # pragma: no cover - optional fallback is covered instead
    cv2 = None


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return round(float(max(lower, min(upper, value))), 4)


def _normalise(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
    return np.clip((values.astype(np.float32) - low) / max(1e-6, high - low), 0.0, 1.0)


def _blur(channel: np.ndarray, radius: float = 7.0) -> np.ndarray:
    if cv2 is not None:
        return cv2.GaussianBlur(channel.astype(np.float32), (0, 0), sigmaX=radius)
    image = Image.fromarray(np.clip(channel, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)


def _components(mask: np.ndarray, minimum: int = 1, maximum: int | None = None) -> list[dict[str, Any]]:
    """Return connected components without requiring OpenCV."""
    mask_uint8 = mask.astype(np.uint8)
    components: list[dict[str, Any]] = []
    if cv2 is not None:
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        for index in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[index]]
            if area < minimum or (maximum is not None and area > maximum):
                continue
            components.append({
                "x": x, "y": y, "width": width, "height": height, "area": area,
                "center_x": round(float(centroids[index][0]), 2),
                "center_y": round(float(centroids[index][1]), 2),
            })
        return components

    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for start_y, start_x in zip(*np.where(mask)):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            current_y, current_x = queue.popleft()
            pixels.append((current_y, current_x))
            for offset_y, offset_x in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                next_y, next_x = current_y + offset_y, current_x + offset_x
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not visited[next_y, next_x]:
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        area = len(pixels)
        if area < minimum or (maximum is not None and area > maximum):
            continue
        ys = [point[0] for point in pixels]
        xs = [point[1] for point in pixels]
        components.append({
            "x": min(xs), "y": min(ys), "width": max(xs) - min(xs) + 1,
            "height": max(ys) - min(ys) + 1, "area": area,
            "center_x": round(float(np.mean(xs)), 2), "center_y": round(float(np.mean(ys)), 2),
        })
    return components


def _regions(mask: np.ndarray, response: np.ndarray, minimum: int = 1, maximum: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    regions = _components(mask, minimum=minimum, maximum=maximum)
    for region in regions:
        x, y = region["x"], region["y"]
        crop = response[y:y + region["height"], x:x + region["width"]]
        region["score"] = _clamp(float(np.mean(crop)) if crop.size else 0.0)
    regions.sort(key=lambda region: (region["score"], region["area"]), reverse=True)
    return regions[:limit] if limit else regions


def _png_data_uri(array: np.ndarray, colour: tuple[int, int, int] = (0, 210, 180), alpha: int = 150) -> str:
    rgba = np.zeros((*array.shape[:2], 4), dtype=np.uint8)
    if array.ndim == 2:
        present = array > 0
    else:
        present = array[:, :, 3] > 0
    rgba[present, 0] = colour[0]
    rgba[present, 1] = colour[1]
    rgba[present, 2] = colour[2]
    rgba[present, 3] = array[present] if array.ndim == 2 and array.dtype != bool else alpha
    output = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _evidence_overlay(layers: list[tuple[np.ndarray, tuple[int, int, int], int]], shape: tuple[int, int]) -> str:
    canvas = np.zeros((*shape, 4), dtype=np.uint8)
    for mask, colour, alpha in layers:
        present = mask.astype(bool)
        canvas[present, 0] = colour[0]
        canvas[present, 1] = colour[1]
        canvas[present, 2] = colour[2]
        canvas[present, 3] = np.maximum(canvas[present, 3], alpha)
    return _png_data_uri(canvas)


def _mask_from_data_uri(data_uri: str | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Decode an adapter's transparent mask for the combined evidence map."""
    if not data_uri or not data_uri.startswith("data:image/"):
        return None
    try:
        encoded = data_uri.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            rgba = np.asarray(image.convert("RGBA"))
        if rgba.shape[:2] != shape:
            return None
        return rgba[:, :, 3] > 0
    except (ValueError, OSError, IndexError):
        return None


def _unsupported(module: str, category: str, reason: str, implementation: str = "none") -> EvidenceModuleResult:
    return EvidenceModuleResult(
        module=module, category=category, status="unsupported", supported=False,
        implementation=implementation, issues=[{"type": "unsupported", "message": reason}],
    )


@dataclass
class RetinalEvidenceAnalysis:
    image_id: str
    screening_session_id: str
    status: str
    image_metadata: dict[str, Any]
    coarse_to_fine: dict[str, Any]
    modules: dict[str, dict[str, Any]]
    anatomical_landmarks: list[dict[str, Any]]
    evidence_map_data_uri: str | None
    dataset_support: dict[str, Any]
    note: str
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "screening_session_id": self.screening_session_id,
            "status": self.status,
            "image_metadata": self.image_metadata,
            "coarse_to_fine": self.coarse_to_fine,
            "modules": self.modules,
            "anatomical_landmarks": self.anatomical_landmarks,
            "evidence_map_data_uri": self.evidence_map_data_uri,
            "dataset_support": self.dataset_support,
            "note": self.note,
            "stage_timings_ms": self.stage_timings_ms,
        }


class RetinalEvidenceService:
    """Run global context, region proposals, patches, and local evidence modules."""

    def __init__(self, max_dimension: int = 768, enable_heuristics: bool = True, model_adapters: dict[str, EvidenceModelAdapter] | None = None, enable_vessel_baseline: bool = False):
        self.max_dimension = max_dimension
        self.enable_heuristics = enable_heuristics
        self.model_adapters = model_adapters or {}
        self.enable_vessel_baseline = enable_vessel_baseline
        self.input_validator = ImageTrustGateService()

    async def analyze(self, image_bytes: bytes, image_id: str, screening_session_id: str, eye: str | None = None) -> RetinalEvidenceAnalysis:
        return await asyncio.to_thread(self.analyze_sync, image_bytes, image_id, screening_session_id, eye)

    def analyze_sync(self, image_bytes: bytes, image_id: str, screening_session_id: str, eye: str | None = None) -> RetinalEvidenceAnalysis:
        metadata = self.input_validator.validate_input(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as source:
            original_width, original_height = source.size
            scale = min(1.0, self.max_dimension / max(source.size))
            if scale < 1:
                source = source.resize((max(1, round(source.width * scale)), max(1, round(source.height * scale))), Image.Resampling.LANCZOS)
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        gray = np.asarray(np.dot(rgb[..., :3], [0.299, 0.587, 0.114]), dtype=np.float32)
        retina_mask = self._retina_mask(gray)
        context = {
            "working_width": width, "working_height": height,
            "original_width": original_width, "original_height": original_height,
            "retina_area_ratio": round(float(np.mean(retina_mask)), 4),
            "mean_intensity": round(float(np.mean(gray[retina_mask])) if np.any(retina_mask) else 0.0, 4),
            "green_mean": round(float(np.mean(rgb[:, :, 1][retina_mask])) if np.any(retina_mask) else 0.0, 4),
            "contrast_stddev": round(float(np.std(gray[retina_mask])) if np.any(retina_mask) else 0.0, 4),
        }
        dark_response = np.maximum(_blur(rgb[:, :, 0], 9) - rgb[:, :, 0].astype(np.float32), 0) * retina_mask
        bright_response = np.maximum(gray - _blur(gray, 15), 0) * retina_mask
        proposal_strength = np.maximum(_normalise(dark_response), _normalise(bright_response)) * retina_mask
        proposal_threshold = float(np.percentile(proposal_strength[retina_mask], 98)) if np.any(retina_mask) else 1.0
        proposal_mask = (proposal_strength >= max(0.25, proposal_threshold)) & retina_mask
        proposals = _regions(proposal_mask, proposal_strength, minimum=max(3, (width * height) // 100000), maximum=(width * height) // 5, limit=16)
        patches = []
        patch_size = max(64, min(192, round(min(height, width) * 0.16)))
        for proposal in proposals:
            center_x, center_y = int(proposal["center_x"]), int(proposal["center_y"])
            x = max(0, min(width - patch_size, center_x - patch_size // 2))
            y = max(0, min(height - patch_size, center_y - patch_size // 2))
            patches.append({"x": x, "y": y, "width": patch_size, "height": patch_size, "proposal_score": proposal["score"]})
        context["proposal_count"] = len(proposals)
        context["patch_count"] = len(patches)

        layers: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
        modules: dict[str, dict[str, Any]] = {}
        landmarks: list[dict[str, Any]] = []
        stage_timings: dict[str, float] = {}

        vessel_baseline = None
        if self.enable_vessel_baseline and self.enable_heuristics:
            vessel_mask = self._vessel_mask(rgb, retina_mask)
            vessel_baseline = EvidenceModuleResult(
                module="vessel_segmentation", category="segmentation", status="experimental_heuristic",
                supported=True, implementation="classical_cv_baseline", confidence=_clamp(0.18 + float(np.mean(vessel_mask)) * 2.0),
                mask_data_uri=_png_data_uri(vessel_mask, (0, 210, 120), 145),
                metadata={"pixel_count": int(vessel_mask.sum()), "coverage_ratio": round(float(vessel_mask.mean()), 6), "model_interface": "SegmentationModel", "measurement_status": "ENGINEERING_ESTIMATE", "clinical_validation_claim": False},
                issues=[{"type": "experimental_baseline", "message": "EXPERIMENTAL BASELINE — NOT MODEL-BACKED; classical vessel enhancement is not clinically validated."}],
            )
        vessel_started = time.perf_counter()
        vessel = self._run_adapter("vessel_segmentation", "segmentation", rgb, context)
        stage_timings["vessel_inference_ms"] = round((time.perf_counter() - vessel_started) * 1000, 3)
        if vessel is None:
            vessel = vessel_baseline or _unsupported(
                "vessel_segmentation", "segmentation",
                "No verified model-backed vessel segmentor is configured. The classical-CV baseline is disabled by default; run the documented acquisition command or explicitly enable the experimental baseline.",
                "none",
            )
        modules[vessel.module] = vessel.to_dict()
        if vessel.supported:
            vessel_layer = vessel_mask if vessel.status == "experimental_heuristic" else _mask_from_data_uri(vessel.mask_data_uri, (height, width))
            if vessel_layer is not None and vessel_layer.any():
                layers.append((vessel_layer, (0, 210, 120), 125))

        structure_started = time.perf_counter()
        disc_mask, disc_landmark = self._optic_disc(gray, retina_mask)
        optic_adapter = self._run_adapter("optic_disc_localization", "landmark", rgb, context)
        if optic_adapter is not None:
            optic_disc = optic_adapter
            disc_landmark = optic_adapter.landmarks[0] if optic_adapter.landmarks else None
            if disc_landmark is not None and "x" in disc_landmark and "y" in disc_landmark:
                disc_landmark.setdefault("radius", 0.0)
                disc_landmark.setdefault("brightness_score", 0.5)
                landmarks.extend(optic_adapter.landmarks)
        elif not self.enable_heuristics:
            optic_disc = _unsupported("optic_disc_localization", "landmark", "Heuristic anatomical localization is disabled and no trained landmark adapter is configured.")
            disc_landmark = None
        elif disc_landmark is None:
            optic_disc = _unsupported("optic_disc_localization", "landmark", "No reliable bright anatomical candidate was found in this image.", "classical_cv_baseline")
        else:
            disc_landmark["landmark_type"] = "optic_disc"
            disc_landmark["status"] = "experimental_heuristic"
            disc_landmark["method"] = "bright-region-localization"
            disc_landmark["confidence"] = _clamp(0.2 + disc_landmark["brightness_score"] * 0.5)
            landmarks.append(disc_landmark)
            optic_disc = EvidenceModuleResult(
                module="optic_disc_localization", category="landmark", status="experimental_heuristic",
                supported=True, implementation="classical_cv_baseline", confidence=disc_landmark["confidence"],
                mask_data_uri=_png_data_uri(disc_mask, (50, 150, 255), 110), landmarks=[disc_landmark],
                issues=[{"type": "experimental_baseline", "message": "Bright-region localization baseline; not clinically validated."}],
            )
            layers.append((disc_mask, (50, 150, 255), 100))
        modules[optic_disc.module] = optic_disc.to_dict()

        fovea_adapter = self._run_adapter("fovea_localization", "landmark", rgb, context)
        if fovea_adapter is not None:
            fovea = fovea_adapter
            if fovea_adapter.landmarks:
                landmarks.extend(fovea_adapter.landmarks)
        elif not self.enable_heuristics:
            fovea = _unsupported("fovea_localization", "landmark", "Heuristic anatomical localization is disabled and no trained landmark adapter is configured.")
        elif disc_landmark is None:
            fovea = _unsupported("fovea_localization", "landmark", "Fovea localization requires a usable optic-disc reference; no approximation was emitted.")
        else:
            sign = -1 if eye == "left" else 1
            fovea_x = max(0.0, min(width - 1.0, disc_landmark["x"] + sign * disc_landmark["radius"] * 2.2))
            fovea_y = max(0.0, min(height - 1.0, disc_landmark["y"]))
            fovea_landmark = {"landmark_type": "fovea", "status": "approximate", "method": "optic-disc-relative-anatomical-approximation", "x": round(fovea_x, 2), "y": round(fovea_y, 2), "x_normalized": round(fovea_x / width, 6), "y_normalized": round(fovea_y / height, 6), "radius": round(max(3.0, disc_landmark["radius"] * 0.35), 2), "confidence": 0.1, "eye_assumption": eye or "unspecified"}
            landmarks.append(fovea_landmark)
            fovea = EvidenceModuleResult(
                module="fovea_localization", category="landmark", status="approximate", supported=True,
                implementation="optic_disc_relative_heuristic", confidence=0.1, landmarks=[fovea_landmark],
                issues=[{"type": "approximation_only", "message": "Approximate anatomical position only; not a validated fovea detector."}],
            )
        modules[fovea.module] = fovea.to_dict()
        stage_timings["structure_analysis_ms"] = round((time.perf_counter() - structure_started) * 1000, 3)

        lesion_specs = [
            ("cotton_wool_spot_detection", "lesion_detection", bright_response, (2, max(8, (width * height) // 50000)), (236, 166, 63), "bright cotton-wool-spot candidate regions"),
            ("microaneurysm_detection", "lesion_detection", dark_response, (2, max(8, (width * height) // 60000)), (190, 30, 70), "small dark-red candidate regions"),
            ("hemorrhage_detection", "lesion_detection", dark_response, (max(8, (width * height) // 50000), max(30, (width * height) // 300)), (190, 20, 35), "larger dark candidate regions"),
            ("exudate_segmentation", "segmentation", bright_response, (2, max(8, (width * height) // 50000)), (245, 190, 30), "bright candidate regions"),
        ]
        lesion_started = time.perf_counter()
        for module_name, category, response, area_range, colour, description in lesion_specs:
            adapter_result = self._run_adapter(module_name, category, rgb, context)
            if adapter_result is not None:
                result = adapter_result
            elif not self.enable_heuristics:
                result = _unsupported(module_name, category, "Heuristic lesion baselines are disabled and no trained lesion adapter is configured.")
            else:
                threshold = float(np.percentile(response[retina_mask], 96 if module_name == "hemorrhage_detection" else 98)) if np.any(retina_mask) else 1.0
                mask = (response >= max(3.0, threshold)) & retina_mask
                regions = _regions(mask, _normalise(response), minimum=area_range[0], maximum=area_range[1], limit=32)
                filtered = np.zeros(mask.shape, dtype=bool)
                for region in regions:
                    filtered[region["y"]:region["y"] + region["height"], region["x"]:region["x"] + region["width"]] |= mask[region["y"]:region["y"] + region["height"], region["x"]:region["x"] + region["width"]]
                result = EvidenceModuleResult(
                    module=module_name, category=category, status="experimental_heuristic", supported=True,
                    implementation="coarse_to_fine_classical_cv_baseline", confidence=_clamp(0.1 + min(0.25, len(regions) / 100)),
                    count=len(regions), mask_data_uri=_png_data_uri(filtered, colour, 130), bounding_regions=regions,
                    metadata={"proposal_count": len(proposals), "patch_count": len(patches), "description": description, "model_interface": "PatchDetector" if category == "lesion_detection" else "SegmentationModel", "clinical_validation_claim": False},
                    issues=[{"type": "experimental_baseline", "message": f"{description.capitalize()} from image-processing heuristics; not clinically validated."}],
                )
                if filtered.any():
                    layers.append((filtered, colour, 125))
            modules[result.module] = result.to_dict()
        stage_timings["lesion_inference_ms"] = round((time.perf_counter() - lesion_started) * 1000, 3)

        modules["neovascularization_detection"] = _unsupported(
            "neovascularization_detection", "lesion_detection",
            "Advanced experimental module interface is present, but no validated model artifact or compatible annotation is configured.",
            "experimental_interface_only",
        ).to_dict()

        return RetinalEvidenceAnalysis(
            image_id=image_id, screening_session_id=screening_session_id, status="completed",
            image_metadata={"width": metadata.width, "height": metadata.height, "channels": metadata.channels, "format": metadata.format, "working_width": width, "working_height": height},
            coarse_to_fine={
                "global_context": context,
                "suspicious_region_proposals": proposals,
                "high_resolution_patch_extraction": {"patch_size": patch_size, "patches": patches},
                "local_analysis": {"modules": list(modules), "note": "Configured model-backed modules return real inference; unconfigured baselines and approximations remain explicitly experimental."},
            },
            modules=modules, anatomical_landmarks=landmarks,
            evidence_map_data_uri=_evidence_overlay(layers, (height, width)) if layers else None,
            dataset_support=evidence_dataset_support(),
            note="Clinical evidence layer is distinct from DR classification. Model-backed vessel/lesion outputs are supporting engineering evidence; experimental baselines and approximations are explicitly labelled and are not clinical findings.",
            stage_timings_ms=stage_timings,
        )

    def _module_or_adapter(self, module: str, category: str, image_rgb: np.ndarray, context: dict[str, Any], baseline: EvidenceModuleResult) -> EvidenceModuleResult:
        result = self._run_adapter(module, category, image_rgb, context)
        if result is None:
            return baseline
        return result

    def _run_adapter(self, module: str, category: str, image_rgb: np.ndarray, context: dict[str, Any]) -> EvidenceModuleResult | None:
        adapter = self.model_adapters.get(module)
        if adapter is None:
            return None
        try:
            result = adapter.analyze(image_rgb, {**context, "requested_module": module})
            result.metadata = {**result.metadata, "adapter_name": adapter.name, "adapter_version": adapter.version}
            return result
        except Exception as exc:
            return _unsupported(
                module,
                category,
                "Configured adapter failed safely: " + safe_error_message(exc, "the adapter returned an internal error"),
                "configured_adapter",
            )

    @staticmethod
    def _retina_mask(gray: np.ndarray) -> np.ndarray:
        threshold = max(8.0, float(np.percentile(gray, 1)))
        mask = gray > threshold
        if cv2 is not None:
            kernel = np.ones((9, 9), dtype=np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            if count > 1:
                index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                mask = labels == index
        return mask

    @staticmethod
    def _vessel_mask(rgb: np.ndarray, retina_mask: np.ndarray) -> np.ndarray:
        green = rgb[:, :, 1].astype(np.float32)
        response = np.maximum(_blur(green, 5) - green, 0) * retina_mask
        threshold = float(np.percentile(response[retina_mask], 88)) if np.any(retina_mask) else 255.0
        mask = (response > max(4.0, threshold)) & retina_mask
        if cv2 is not None:
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        return mask

    @staticmethod
    def _optic_disc(gray: np.ndarray, retina_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None]:
        if not np.any(retina_mask):
            return np.zeros(gray.shape, dtype=bool), None
        threshold = float(np.percentile(gray[retina_mask], 99.5))
        candidate = (gray >= threshold) & retina_mask
        regions = _regions(candidate, _normalise(gray), minimum=max(20, gray.size // 10000), maximum=max(100, gray.size // 4), limit=1)
        if not regions:
            return np.zeros(gray.shape, dtype=bool), None
        region = regions[0]
        mask = np.zeros(gray.shape, dtype=bool)
        mask[region["y"]:region["y"] + region["height"], region["x"]:region["x"] + region["width"]] = candidate[region["y"]:region["y"] + region["height"], region["x"]:region["x"] + region["width"]]
        radius = max(region["width"], region["height"]) / 2
        return mask, {
            "x": region["center_x"], "y": region["center_y"], "x_normalized": round(region["center_x"] / gray.shape[1], 6),
            "y_normalized": round(region["center_y"] / gray.shape[0], 6), "radius": round(radius, 2),
            "brightness_score": _clamp(float(np.mean(gray[mask])) / 255.0),
        }
