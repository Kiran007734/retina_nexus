"""Configurable losses for hierarchical and ordinal DR experiments."""

from __future__ import annotations

from collections import Counter


def build_class_weights(labels, num_classes: int = 5):
    """Build inverse-frequency weights without changing the target distribution."""
    import torch

    counts = Counter(int(label) for label in labels)
    total = max(1, len(labels))
    weights = [total / (num_classes * max(1, counts.get(index, 0))) for index in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels):
    import torch
    from torch.utils.data import WeightedRandomSampler

    counts = Counter(int(label) for label in labels)
    sample_weights = [1.0 / max(1, counts[int(label)]) for label in labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_focal_loss(gamma: float = 2.0, alpha=None):
    import torch
    from torch import nn

    class FocalLoss(nn.Module):
        def __init__(self):
            super().__init__()
            if alpha is None:
                self.alpha = None
            elif isinstance(alpha, torch.Tensor):
                self.register_buffer("alpha", alpha.float())
            else:
                self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

        def forward(self, logits, targets):
            log_probability = torch.nn.functional.log_softmax(logits, dim=1)
            probability = log_probability.exp()
            target_log_probability = log_probability.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_probability = probability.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -((1 - target_probability) ** gamma) * target_log_probability
            if self.alpha is not None:
                loss = loss * self.alpha.to(logits.device).gather(0, targets)
            return loss.mean()

    return FocalLoss()


def build_severity_loss(strategy: str, class_weights):
    import torch

    if strategy == "plain":
        return torch.nn.CrossEntropyLoss()
    if strategy == "weighted_loss":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    if strategy == "focal_loss":
        return build_focal_loss(alpha=class_weights)
    if strategy == "weighted_sampler":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss strategy: {strategy}")


def build_ordinal_loss(strategy: str, labels):
    """Build a threshold loss for ordinal mode using the same imbalance policy."""
    import torch

    if strategy in {"plain", "weighted_sampler"}:
        return torch.nn.BCEWithLogitsLoss()
    threshold_weights = []
    total = max(1, len(labels))
    for threshold in range(4):
        positive = max(1, sum(int(label) > threshold for label in labels))
        negative = max(1, total - positive)
        threshold_weights.append(negative / positive)
    pos_weight = torch.tensor(threshold_weights, dtype=torch.float32)
    if strategy == "weighted_loss":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if strategy == "focal_loss":
        gamma = 2.0

        def focal_ordinal_loss(logits, targets):
            probability = torch.sigmoid(logits)
            cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            target_probability = torch.where(targets == 1, probability, 1 - probability)
            return (((1 - target_probability) ** gamma) * cross_entropy).mean()

        return focal_ordinal_loss
    raise ValueError(f"Unknown loss strategy: {strategy}")


def hierarchical_loss(outputs, labels, mapping, severity_loss, ordinal_mode: bool = False, stage_weights=(0.25, 0.25, 1.0)):
    """Combine the configured fine-grade loss with the two workflow heads."""
    import torch

    stage1_targets = (labels > 0).long()
    stage2_targets = torch.tensor(
        [1 if mapping.is_referable(int(label)) else 0 for label in labels.tolist()],
        device=labels.device,
    )
    stage1_loss = torch.nn.functional.cross_entropy(outputs["stage1_logits"], stage1_targets)
    stage2_loss = torch.nn.functional.cross_entropy(outputs["stage2_logits"], stage2_targets)
    if ordinal_mode:
        ordinal_targets = torch.stack([(labels > threshold).float() for threshold in range(4)], dim=1)
        severity_loss_value = severity_loss(outputs["ordinal_logits"], ordinal_targets)
    else:
        severity_loss_value = severity_loss(outputs["severity_logits"], labels)
    total = stage_weights[0] * stage1_loss + stage_weights[1] * stage2_loss + stage_weights[2] * severity_loss_value
    return total, {"stage1": float(stage1_loss.detach().cpu()), "stage2": float(stage2_loss.detach().cpu()), "severity": float(severity_loss_value.detach().cpu())}
