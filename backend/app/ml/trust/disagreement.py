"""Model ensemble disagreement metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Any


def calculate_model_disagreement(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [item for item in predictions if item.get("predicted_grade") is not None]
    if len(usable) < 2:
        return {"status": "UNAVAILABLE", "model_count": len(usable), "disagreement": None, "agreement": None, "severity": "not_applicable", "reason": "At least two model predictions are required to measure disagreement."}
    grades = [int(item["predicted_grade"]) for item in usable]
    counts = {grade: grades.count(grade) for grade in set(grades)}
    majority_fraction = max(counts.values()) / len(grades)
    pairs = list(combinations(grades, 2))
    pairwise_severity = sum(abs(left - right) / 4.0 for left, right in pairs) / max(1, len(pairs))
    class_disagreement = 1.0 - majority_fraction
    disagreement = 0.6 * pairwise_severity + 0.4 * class_disagreement
    severity = "high" if disagreement >= 0.5 else "moderate" if disagreement >= 0.2 else "low"
    return {
        "status": "COMPLETED", "model_count": len(usable),
        "predictions": [{key: value for key, value in item.items() if key in {"model_version", "predicted_grade", "predicted_grade_label"}} for item in usable],
        "majority_grade": max(counts, key=counts.get), "majority_fraction": round(majority_fraction, 6),
        "pairwise_severity": round(pairwise_severity, 6), "class_disagreement": round(class_disagreement, 6),
        "disagreement": round(disagreement, 6), "agreement": round(1.0 - disagreement, 6), "severity": severity,
        "method": "weighted_pairwise_grade_distance_and_majority_disagreement",
        "note": "Disagreement increases review priority; it does not identify which model is clinically correct.",
    }
