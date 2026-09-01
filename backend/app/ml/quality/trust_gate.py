"""Measurable image quality assessment and controlled enhancement.

This module is intentionally a quality gate, not a disease classifier. Its
decision only describes whether an image is suitable for a later pipeline stage.
"""

from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal local environments
    cv2 = None
import numpy as np
from PIL import ExifTags, Image, ImageEnhance, ImageFilter, UnidentifiedImageError
from PIL.Image import DecompressionBombError


class ImageTrustGateError(ValueError):
    """Raised when an image cannot safely enter the trust gate."""


class TrustGateDecision:
    GRADABLE = "GRADABLE"
    BORDERLINE = "BORDERLINE"
    UNGRADABLE = "UNGRADABLE"


@dataclass
class ImageInputMetadata:
    width: int
    height: int
    channels: int
    mode: str
    format: str
    camera_metadata: dict[str, str]


@dataclass
class QualityIssue:
    type: str
    severity: str
    message: str
    recommendation: str


@dataclass
class QualityAssessment:
    quality_decision: str
    quality_score: float
    component_scores: dict[str, float]
    metrics: dict[str, float]
    issues: list[QualityIssue]
    recommended_action: str
    next_action: str
    input_metadata: dict[str, Any]
    feature_vector: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["issues"] = [asdict(issue) for issue in self.issues]
        return value


@dataclass
class TrustGateOutcome:
    initial: QualityAssessment
    final: QualityAssessment
    enhancement_applied: bool = False
    enhancement_passes: int = 0


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return round(max(lower, min(upper, float(value))), 4)


def _log_score(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return _clamp((math.log1p(value) - math.log1p(low)) / (math.log1p(high) - math.log1p(low)))


def _camera_metadata(image: Image.Image) -> dict[str, str]:
    try:
        exif = image.getexif()
    except Exception:
        return {}
    tags = {ExifTags.TAGS.get(tag, str(tag)): value for tag, value in exif.items()}
    return {key: str(tags[key]) for key in ("Make", "Model", "DateTimeOriginal", "Software") if key in tags}


class ImageTrustGateService:
    """Quality gate using independent focus, exposure, field, and artifact signals."""

    min_dimension = 256
    max_dimension = 12000

    def __init__(self, max_image_pixels: int = 50_000_000):
        self.max_image_pixels = int(max_image_pixels)

    async def extract_features(self, image_bytes: bytes) -> dict[str, float]:
        """Return normalized quality features for future OOD calibration."""
        # Reusing the assessed feature vector keeps feature extraction aligned
        # with the quality decision until a validated embedding model is added.
        return (await self.assess(image_bytes)).feature_vector

    def validate_input(self, image_bytes: bytes) -> ImageInputMetadata:
        if not image_bytes:
            raise ImageTrustGateError("Image file is empty")
        try:
            with Image.open(io.BytesIO(image_bytes)) as probe:
                image_format = (probe.format or "").upper()
                if image_format not in {"JPEG", "PNG"}:
                    raise ImageTrustGateError("Only JPEG and PNG fundus images are supported")
                probe.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                channels = len(image.getbands())
                if width < self.min_dimension or height < self.min_dimension:
                    raise ImageTrustGateError(f"Image dimensions {width}x{height} are below the minimum {self.min_dimension}x{self.min_dimension}")
                if width > self.max_dimension or height > self.max_dimension:
                    raise ImageTrustGateError(f"Image dimensions {width}x{height} exceed the maximum supported size")
                if width * height > self.max_image_pixels:
                    raise ImageTrustGateError("Image contains too many pixels for safe processing")
                image.load()
                if image.mode not in {"RGB", "RGBA"} or channels not in {3, 4}:
                    raise ImageTrustGateError(f"Unsupported color channels: mode {image.mode}; RGB or RGBA is required")
                metadata = _camera_metadata(image)
                return ImageInputMetadata(width, height, channels, image.mode, image_format, metadata)
        except ImageTrustGateError:
            raise
        except DecompressionBombError as exc:
            raise ImageTrustGateError("Image dimensions exceed the safe decoding limit") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            # Pillow's exception text may contain implementation details or
            # object representations. Keep the API message stable and safe.
            raise ImageTrustGateError("Image decoding or integrity check failed") from exc

    async def assess(self, image_bytes: bytes, camera_metadata: dict[str, str] | None = None) -> QualityAssessment:
        input_metadata = self.validate_input(image_bytes)
        if cv2 is not None:
            decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ImageTrustGateError("OpenCV could not decode the image")
            gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(decoded, cv2.COLOR_BGR2HSV)
        else:
            with Image.open(io.BytesIO(image_bytes)) as fallback_image:
                rgb = np.asarray(fallback_image.convert("RGB"), dtype=np.uint8)
            decoded = rgb[:, :, ::-1]
            gray = np.asarray(np.dot(rgb[..., :3], [0.299, 0.587, 0.114]), dtype=np.uint8)
            max_channel = rgb.max(axis=2).astype(np.float32)
            min_channel = rgb.min(axis=2).astype(np.float32)
            saturation = np.where(max_channel == 0, 0, (max_channel - min_channel) / np.maximum(max_channel, 1) * 255)
            hsv = np.stack((np.zeros_like(gray), saturation.astype(np.uint8), max_channel.astype(np.uint8)), axis=2)

        # Fundus captures commonly have a circular retinal field surrounded by
        # a black mask. Compute focus/exposure/contrast on the retinal region;
        # otherwise an expected capture boundary dominates the measurements.
        global_percentiles = np.percentile(gray, [1, 5, 50, 95, 99])
        dark_threshold = max(8, float(global_percentiles[1]) * 0.25)
        retinal_mask = gray > dark_threshold
        roi_mask = retinal_mask
        if cv2 is not None and int(retinal_mask.sum()) > 1000:
            eroded = cv2.erode(retinal_mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            if int(eroded.sum()) > 1000:
                roi_mask = eroded
        if int(roi_mask.sum()) < 1000:
            roi_mask = np.ones_like(gray, dtype=bool)

        if cv2 is not None:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        else:
            padded = np.pad(gray.astype(np.float32), 1, mode="edge")
            laplacian = (padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1] + padded[2:, 1:-1] - 4 * padded[1:-1, 1:-1])
        laplacian_variance = float(laplacian[roi_mask].var())
        # The focus range is calibrated for decoded fundus captures after
        # masking the non-retinal field. It remains a heuristic and must be
        # recalibrated for a new camera population.
        focus_score = _log_score(laplacian_variance, 5, 300)
        roi_gray = gray[roi_mask]
        percentiles = np.percentile(roi_gray, [1, 5, 50, 95, 99])
        mean_intensity = float(np.mean(roi_gray))
        std_intensity = float(np.std(roi_gray))
        low_clip_ratio = float(np.mean(roi_gray <= 8))
        high_clip_ratio = float(np.mean(roi_gray >= 247))
        illumination_score = _clamp(1 - abs(mean_intensity - 128) / 260 - (low_clip_ratio + high_clip_ratio) * 1.5)
        contrast_spread = float(percentiles[3] - percentiles[1])
        contrast_score = _clamp(0.55 * _clamp(contrast_spread / 125) + 0.45 * _clamp(std_intensity / 65))

        retinal_coverage = float(np.mean(gray > dark_threshold))
        if retinal_coverage < 0.30:
            field_score = _clamp(retinal_coverage / 0.30)
        elif retinal_coverage > 0.98:
            field_score = _clamp((1 - retinal_coverage) / 0.02)
        else:
            field_score = 1.0

        clipped_ratio = low_clip_ratio + high_clip_ratio
        exposure_score = _clamp(1 - clipped_ratio * 8)
        border_pixels = np.concatenate([
            gray[: max(1, gray.shape[0] // 20), :].ravel(),
            gray[-max(1, gray.shape[0] // 20):, :].ravel(),
            gray[:, : max(1, gray.shape[1] // 20)].ravel(),
            gray[:, -max(1, gray.shape[1] // 20):].ravel(),
        ])
        dark_border_ratio = float(np.mean(border_pixels <= 8))
        bright_border_ratio = float(np.mean(border_pixels >= 247))
        # A dark circular field boundary is expected. Only treat a bright
        # boundary as suspicious when no expected dark boundary is present.
        border_ratio = bright_border_ratio if dark_border_ratio < 0.30 else 0.0
        saturated_ratio = float(np.mean((hsv[:, :, 1] > 245) & (hsv[:, :, 2] > 235)))
        artifact_burden = _clamp(border_ratio * 1.6 + saturated_ratio * 0.7)
        artifact_score = _clamp(1 - artifact_burden)

        component_scores = {
            "focus": focus_score, "illumination": illumination_score, "contrast": contrast_score,
            "field_of_view": field_score, "exposure": exposure_score, "artifacts": artifact_score,
        }
        metrics = {
            "laplacian_variance": round(laplacian_variance, 4), "mean_intensity": round(mean_intensity, 4),
            "intensity_stddev": round(std_intensity, 4), "p1_intensity": round(float(percentiles[0]), 4),
            "p5_intensity": round(float(percentiles[1]), 4), "p50_intensity": round(float(percentiles[2]), 4),
            "p95_intensity": round(float(percentiles[3]), 4), "p99_intensity": round(float(percentiles[4]), 4),
            "retinal_coverage_ratio": round(retinal_coverage, 4), "low_clip_ratio": round(low_clip_ratio, 4),
            "high_clip_ratio": round(high_clip_ratio, 4), "border_artifact_ratio": round(border_ratio, 4),
            "saturated_artifact_ratio": round(saturated_ratio, 4),
        }
        weighted_score = sum(component_scores[name] * weight for name, weight in {"focus": .25, "illumination": .15, "contrast": .15, "field_of_view": .20, "exposure": .15, "artifacts": .10}.items())
        issues = self._issues(component_scores, metrics)
        if weighted_score < 0.45 or any(issue.severity == "severe" for issue in issues):
            decision = TrustGateDecision.UNGRADABLE
        elif weighted_score < 0.75 or issues:
            decision = TrustGateDecision.BORDERLINE
        else:
            decision = TrustGateDecision.GRADABLE
        camera = camera_metadata or input_metadata.camera_metadata
        features = {**component_scores, "mean_intensity": _clamp(mean_intensity / 255), "contrast_spread": _clamp(contrast_spread / 255), "retinal_coverage": _clamp(retinal_coverage)}
        return QualityAssessment(
            quality_decision=decision, quality_score=_clamp(weighted_score), component_scores=component_scores,
            metrics=metrics, issues=issues, recommended_action=self._recommended_action(decision),
            next_action=self._next_action(decision), input_metadata={**asdict(input_metadata), "camera_metadata": camera},
            feature_vector=features,
        )

    def _issues(self, scores: dict[str, float], metrics: dict[str, float]) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if scores["focus"] < 0.30:
            severity = "severe" if scores["focus"] < 0.10 else "moderate"
            issues.append(QualityIssue("severe_blur" if severity == "severe" else "low_focus", severity, "The retinal image has insufficient focus detail.", "Stabilize the camera and refocus before recapturing."))
        if scores["illumination"] < 0.45:
            issue_type = "underexposure" if metrics["mean_intensity"] < 85 else "overexposure" if metrics["mean_intensity"] > 190 else "uneven_illumination"
            issues.append(QualityIssue(issue_type, "severe" if scores["illumination"] < 0.22 else "moderate", "Illumination is outside the preferred range for review.", "Adjust the light source and avoid glare before recapturing."))
        if scores["contrast"] < 0.45:
            issues.append(QualityIssue("low_contrast", "moderate", "Retinal structures have limited tonal separation.", "Improve focus and illumination; a controlled enhancement pass may help."))
        if scores["field_of_view"] < 0.45:
            issues.append(QualityIssue("insufficient_field_of_view", "severe" if scores["field_of_view"] < 0.22 else "moderate", "Too little of the retinal field is visible.", "Re-center the retina and recapture the image."))
        if scores["exposure"] < 0.45:
            issues.append(QualityIssue("clipped_exposure", "moderate", "Highlights or shadows are clipped.", "Reduce glare or adjust exposure before recapturing."))
        if scores["artifacts"] < 0.45:
            issues.append(QualityIssue("image_artifacts", "moderate", "Borders, saturation, or capture artifacts may obscure retinal evidence.", "Clean the lens, remove obstruction, and recapture a centered image."))
        return issues

    @staticmethod
    def _recommended_action(decision: str) -> str:
        return {TrustGateDecision.GRADABLE: "Proceed to clinical AI analysis.", TrustGateDecision.BORDERLINE: "Apply one controlled enhancement pass, then reassess.", TrustGateDecision.UNGRADABLE: "Recapture the image using the guidance below."}[decision]

    @staticmethod
    def _next_action(decision: str) -> str:
        return {TrustGateDecision.GRADABLE: "CONTINUE_SCREENING", TrustGateDecision.BORDERLINE: "ENHANCE_AND_REASSESS", TrustGateDecision.UNGRADABLE: "RECAPTURE_IMAGE"}[decision]

    def enhance(self, image_bytes: bytes) -> bytes:
        self.validate_input(image_bytes)
        if cv2 is None:
            with Image.open(io.BytesIO(image_bytes)) as source:
                enhanced = ImageEnhance.Contrast(source.convert("RGB")).enhance(1.18)
                enhanced = ImageEnhance.Color(enhanced).enhance(1.04)
                enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
                output = io.BytesIO()
                enhanced.save(output, format="PNG", compress_level=3)
                return output.getvalue()
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ImageTrustGateError("OpenCV could not decode the image for enhancement")
        lab = cv2.cvtColor(decoded, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_lightness = clahe.apply(lightness)
        illumination = cv2.GaussianBlur(enhanced_lightness, (0, 0), sigmaX=15)
        normalized = cv2.divide(enhanced_lightness, illumination, scale=128)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        normalized_lab = cv2.merge((normalized.astype(np.uint8), a_channel, b_channel))
        normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(normalized_bgr, None, 3, 3, 7, 21)
        success, encoded = cv2.imencode(".png", denoised, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            raise ImageTrustGateError("Could not encode enhanced image")
        return encoded.tobytes()

    async def assess_with_controlled_enhancement(self, image_bytes: bytes) -> TrustGateOutcome:
        initial = await self.assess(image_bytes)
        if initial.quality_decision != TrustGateDecision.BORDERLINE:
            return TrustGateOutcome(initial, initial)
        enhanced = self.enhance(image_bytes)
        recheck = await self.assess(enhanced)
        return TrustGateOutcome(initial, recheck, enhancement_applied=True, enhancement_passes=1)
