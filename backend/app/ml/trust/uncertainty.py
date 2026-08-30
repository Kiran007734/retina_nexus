"""Transparent predictive uncertainty estimators."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class MCDropoutProvider(Protocol):
    async def predict_with_dropout(self, image_bytes: bytes, samples: int) -> list[list[float]]: ...


def _clamp(value: float) -> float:
    return round(float(max(0.0, min(1.0, value))), 6)


class UncertaintyEstimator:
    """Combine normalized predictive entropy and probability-margin uncertainty."""

    def __init__(self, entropy_weight: float = 0.6, margin_weight: float = 0.4):
        if entropy_weight < 0 or margin_weight < 0 or entropy_weight + margin_weight <= 0:
            raise ValueError("Uncertainty component weights must be non-negative and non-zero")
        total = entropy_weight + margin_weight
        self.entropy_weight = entropy_weight / total
        self.margin_weight = margin_weight / total

    def estimate(self, probabilities: dict[str, float], mc_probabilities: list[list[float]] | None = None, mc_error: str | None = None) -> dict[str, Any]:
        values = np.asarray(list(probabilities.values()), dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Uncertainty requires a non-empty finite probability vector")
        values /= max(1e-12, float(values.sum()))
        entropy = float(-(values * np.log(np.clip(values, 1e-12, 1.0))).sum() / max(1e-12, np.log(len(values)))) if len(values) > 1 else 0.0
        ordered = np.sort(values)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        margin_uncertainty = 1.0 - margin
        predictive = self.entropy_weight * entropy + self.margin_weight * margin_uncertainty
        mc_result: dict[str, Any]
        if mc_error:
            mc_result = {"status": "FAILED", "reason": mc_error}
        elif mc_probabilities:
            samples = np.asarray(mc_probabilities, dtype=np.float64)
            if samples.ndim == 2 and samples.shape[1] == len(values) and np.all(np.isfinite(samples)):
                samples = samples / np.maximum(samples.sum(axis=1, keepdims=True), 1e-12)
                variance = float(np.mean(np.var(samples, axis=0)))
                mc_uncertainty = _clamp(variance * len(values))
                predictive = _clamp(0.7 * predictive + 0.3 * mc_uncertainty)
                mc_result = {"status": "COMPLETED", "samples": int(samples.shape[0]), "mean_probability_variance": round(variance, 8), "uncertainty": mc_uncertainty}
            else:
                mc_result = {"status": "FAILED", "reason": "MC-dropout samples did not match the classifier probability vector."}
        else:
            mc_result = {"status": "SKIPPED", "reason": "Monte Carlo dropout is disabled by default for real-time inference."}
        return {
            "score": _clamp(predictive),
            "predictive_entropy": _clamp(entropy),
            "probability_margin": round(margin, 6),
            "margin_uncertainty": _clamp(margin_uncertainty),
            "component_weights": {"entropy": round(self.entropy_weight, 6), "margin": round(self.margin_weight, 6)},
            "method": "predictive_entropy+probability_margin" + ("+mc_dropout" if mc_result["status"] == "COMPLETED" else ""),
            "mc_dropout": mc_result,
            "note": "Uncertainty is an engineering estimate; it is not a calibrated probability of model error.",
        }
