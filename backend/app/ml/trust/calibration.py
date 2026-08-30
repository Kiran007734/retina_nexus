"""Configurable confidence calibration utilities.

Temperature must be fitted on held-out calibration data before it is used as a
calibrated clinical probability. The runtime default of 1.0 is an explicit
identity transform, not a claim that the model is calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    probabilities: dict[str, float]
    calibrated_confidence: float
    method: str
    temperature: float
    version: str
    fitted: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities,
            "calibrated_confidence": self.calibrated_confidence,
            "method": self.method,
            "temperature": self.temperature,
            "version": self.version,
            "fitted": self.fitted,
            "note": self.note,
        }


class TemperatureScaler:
    """Apply temperature scaling to classifier logits or a probability proxy."""

    def __init__(self, temperature: float = 1.0, version: str = "temperature-scaling-unfitted", fitted: bool = False):
        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero")
        self.temperature = float(temperature)
        self.version = version
        self.fitted = fitted

    def calibrate(self, probabilities: dict[str, float], logits: list[float] | None = None) -> CalibrationResult:
        labels = list(probabilities)
        values = np.asarray([float(probabilities[label]) for label in labels], dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Calibration requires a non-empty finite probability vector")
        values = values / max(1e-12, float(values.sum()))
        if logits is not None and len(logits) == len(values):
            source = np.asarray(logits, dtype=np.float64)
        else:
            # A probability-only fallback is useful at API boundaries where
            # logits are not persisted; it is explicitly identified in note.
            source = np.log(np.clip(values, 1e-8, 1.0))
        scaled = source / self.temperature
        scaled -= float(np.max(scaled))
        calibrated = np.exp(scaled)
        calibrated /= max(1e-12, float(calibrated.sum()))
        output = {label: round(float(calibrated[index]), 6) for index, label in enumerate(labels)}
        note = "Fitted temperature scaling applied to logits." if logits is not None and len(logits) == len(values) else "Temperature scaling applied to log-probability proxy because classifier logits were unavailable."
        return CalibrationResult(output, round(float(calibrated.max()), 6), "temperature_scaling", round(self.temperature, 6), self.version, self.fitted, note)
