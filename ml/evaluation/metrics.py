"""Classification metrics used by training and benchmark reports."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    return {
        "sensitivity": recall, "specificity": specificity, "precision": precision,
        "recall": recall, "f1": _safe_divide(2 * precision * recall, precision + recall),
        # ``actual`` is one-vs-rest encoded; support is the number of true
        # positives, not the total evaluation-set size.
        "support": int(np.sum(actual == 1)), "true_positive": true_positive, "true_negative": true_negative,
        "false_positive": false_positive, "false_negative": false_negative,
    }


def _multiclass_confusion(actual: np.ndarray, predicted: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for truth, guess in zip(actual.tolist(), predicted.tolist()):
        if 0 <= truth < num_classes and 0 <= guess < num_classes:
            matrix[int(truth), int(guess)] += 1
    return matrix


def _roc_auc(actual: np.ndarray, probabilities: np.ndarray, num_classes: int) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(actual)) < 2:
            return None
        one_hot = np.eye(num_classes, dtype=float)[actual]
        return float(roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro"))
    except (ImportError, ValueError):
        return None


def _quadratic_weighted_kappa(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import cohen_kappa_score
        if len(np.unique(actual)) < 2 or len(np.unique(predicted)) < 2:
            return None
        return float(cohen_kappa_score(actual, predicted, weights="quadratic"))
    except (ImportError, ValueError):
        return None


def _binary_roc_auc(actual: np.ndarray, probability: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(actual)) < 2:
            return None
        return float(roc_auc_score(actual, probability))
    except (ImportError, ValueError):
        return None


def classification_metrics(actual_labels: list[int] | np.ndarray, probabilities: list[list[float]] | np.ndarray, referable_grades: tuple[int, ...] = (2, 3, 4)) -> dict[str, Any]:
    actual = np.asarray(actual_labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    if probabilities_array.ndim != 2 or probabilities_array.shape[1] != 5:
        raise ValueError("Expected a probability matrix with five DR grade columns")
    predicted = probabilities_array.argmax(axis=1)
    matrix = _multiclass_confusion(actual, predicted, 5)
    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(GRADE_LABELS):
        binary_actual = (actual == index).astype(int)
        binary_predicted = (predicted == index).astype(int)
        metrics = _binary_metrics(binary_actual, binary_predicted)
        per_class[label] = {key: value for key, value in metrics.items() if key in {"sensitivity", "specificity", "precision", "recall", "f1", "support"}}
    referable_actual = np.isin(actual, referable_grades).astype(int)
    referable_probability = probabilities_array[:, list(referable_grades)].sum(axis=1)
    referable_predicted = (referable_probability >= 0.5).astype(int)
    referable = _binary_metrics(referable_actual, referable_predicted)
    referable["roc_auc"] = _binary_roc_auc(referable_actual, referable_probability)
    return {
        "accuracy": float(np.mean(actual == predicted)) if len(actual) else 0.0,
        "sensitivity": float(np.mean([float(value["sensitivity"]) for value in per_class.values()])),
        "specificity": float(np.mean([float(value["specificity"]) for value in per_class.values()])),
        "precision": float(np.mean([float(value["precision"]) for value in per_class.values()])),
        "recall": float(np.mean([float(value["recall"]) for value in per_class.values()])),
        "f1": float(np.mean([float(value["f1"]) for value in per_class.values()])),
        "roc_auc_ovr_macro": _roc_auc(actual, probabilities_array, 5),
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(actual, predicted),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "referable_dr": {"referable_grades": list(referable_grades), **referable},
        "sample_count": int(len(actual)),
        "class_distribution": {str(label): int(count) for label, count in sorted(Counter(actual.tolist()).items())},
    }
