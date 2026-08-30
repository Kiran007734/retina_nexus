"""Benchmark result structures and model comparison helpers."""

from __future__ import annotations

from typing import Any


def compare_benchmark_results(results: list[dict[str, Any]], metric: str = "f1") -> list[dict[str, Any]]:
    """Sort models for reporting; this does not declare a universally best model."""
    return sorted(results, key=lambda result: float(result.get("metrics", {}).get(metric) or 0), reverse=True)


def benchmark_matrix(backbones: tuple[str, ...] = ("efficientnet_b0", "resnet18", "mobilenet_v3_small")) -> dict[str, Any]:
    return {
        "selection_policy": "compare_on_the_same_dataset_split_and_training_budget; do not assume a winner",
        "backbones": list(backbones),
        "families": {"efficientnet": ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"], "resnet": ["resnet18", "resnet50"], "lightweight": ["mobilenet_v3_small"]},
    }
