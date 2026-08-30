"""Shared DRIVE vessel training/evaluation utilities."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def split_pairs(pairs: list[tuple[Any, Any]], validation_ratio: float, seed: int) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation ratio must be between 0 and 1")
    ordered = list(pairs)
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, round(len(ordered) * validation_ratio))
    validation_count = min(len(ordered) - 1, validation_count)
    return ordered[validation_count:], ordered[:validation_count]


def vessel_metrics(targets: list[np.ndarray], probabilities: list[np.ndarray], threshold: float = 0.5) -> dict[str, float | int]:
    if not targets:
        return {"sample_count": 0, "dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    actual = np.concatenate([target.reshape(-1) >= 0.5 for target in targets])
    predicted = np.concatenate([probability.reshape(-1) >= threshold for probability in probabilities])
    true_positive = int(np.sum(actual & predicted))
    true_negative = int(np.sum(~actual & ~predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    intersection = true_positive
    dice = (2 * intersection) / max(1, int(actual.sum()) + int(predicted.sum()))
    iou = intersection / max(1, int(np.sum(actual | predicted)))
    return {
        "sample_count": len(targets), "dice": float(dice), "iou": float(iou),
        "sensitivity": float(true_positive / max(1, true_positive + false_negative)),
        "specificity": float(true_negative / max(1, true_negative + false_positive)),
    }
