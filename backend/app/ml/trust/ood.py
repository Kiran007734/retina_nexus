"""Feature-distribution monitoring for experimental OOD preparation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class FeatureDistributionMonitor:
    """Compare quality features to a configured reference via robust z-scores."""

    def __init__(self, reference_path: str | None = None, threshold: float = 3.0):
        self.reference_path = Path(reference_path).expanduser() if reference_path else None
        self.threshold = float(threshold)

    def evaluate(self, feature_vector: dict[str, float] | None) -> dict[str, Any]:
        reference = self._load_reference()
        if not reference:
            return {"status": "UNAVAILABLE", "score": None, "distance": None, "threshold": self.threshold, "per_feature_z": {}, "compared_feature_count": 0, "method": "robust_feature_z_score", "reason": "No authorized reference distribution is configured; OOD detection is not claimed."}
        vector = feature_vector or {}
        z_scores: dict[str, float] = {}
        for feature, stats in reference.items():
            if feature not in vector:
                continue
            mean = float(stats.get("mean", 0.0))
            std = max(1e-6, float(stats.get("std", 0.0)))
            z_scores[feature] = round(abs(float(vector[feature]) - mean) / std, 6)
        if not z_scores:
            return {"status": "UNAVAILABLE", "score": None, "distance": None, "threshold": self.threshold, "per_feature_z": {}, "compared_feature_count": 0, "method": "robust_feature_z_score", "reason": "The configured reference shares no features with this image quality vector."}
        distance = math.sqrt(sum(value * value for value in z_scores.values()) / len(z_scores))
        max_z = max(z_scores.values())
        shifted = max_z >= self.threshold or distance >= self.threshold
        score = math.exp(-0.5 * distance * distance)
        return {"status": "SHIFTED" if shifted else "IN_DISTRIBUTION", "score": round(max(0.0, min(1.0, score)), 6), "distance": round(distance, 6), "max_feature_z": round(max_z, 6), "threshold": self.threshold, "per_feature_z": z_scores, "compared_feature_count": len(z_scores), "method": "robust_feature_z_score", "reason": "Input features were compared with the configured reference distribution; this cannot guarantee detection of every unfamiliar medical image."}

    def _load_reference(self) -> dict[str, dict[str, float]]:
        if self.reference_path is None or not self.reference_path.is_file():
            return {}
        try:
            payload = json.loads(self.reference_path.read_text(encoding="utf-8"))
            features = payload.get("features", payload)
            return {str(key): value for key, value in features.items() if isinstance(value, dict) and "mean" in value and "std" in value}
        except (OSError, ValueError, TypeError):
            return {}
