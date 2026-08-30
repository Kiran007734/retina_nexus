"""Transfer-learning DR classifier architectures.

The factory deliberately exposes multiple baselines. No backbone is assumed to
be superior; benchmark results should drive model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
    from torchvision import models as tv_models
except Exception:  # Keep the API importable without optional ML packages or broken binary installs.
    torch = None
    nn = None
    tv_models = None


SUPPORTED_BACKBONES = (
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "resnet18", "resnet50", "mobilenet_v3_small",
)


@dataclass(frozen=True)
class ReferableDRMapping:
    """Configurable engineering mapping from grade to referable DR.

    Default: moderate-or-worse (grades 2, 3, 4) is referable. This is a
    configurable implementation choice, not a universal clinical rule.
    """

    name: str = "moderate_or_worse"
    referable_grades: tuple[int, ...] = (2, 3, 4)

    def is_referable(self, grade: int) -> bool:
        return grade in self.referable_grades

    def probability(self, probabilities: list[float]) -> float:
        return sum(probabilities[index] for index in self.referable_grades if 0 <= index < len(probabilities))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "referable_grades": list(self.referable_grades)}


def _require_torch() -> None:
    if torch is None or nn is None or tv_models is None:
        raise RuntimeError("PyTorch and torchvision are not installed. Install backend/requirements-ml.txt before training or inference.")


def build_backbone(name: str, pretrained: bool = True):
    _require_torch()
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{name}'. Choose one of: {', '.join(SUPPORTED_BACKBONES)}")
    try:
        if name == "efficientnet_b0":
            network = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            return network.features, network.classifier[-1].in_features
        if name == "efficientnet_b1":
            network = tv_models.efficientnet_b1(weights=tv_models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
            return network.features, network.classifier[-1].in_features
        if name == "efficientnet_b2":
            network = tv_models.efficientnet_b2(weights=tv_models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
            return network.features, network.classifier[-1].in_features
        if name == "resnet18":
            network = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
            return nn.Sequential(*list(network.children())[:-2]), network.fc.in_features
        if name == "resnet50":
            network = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
            return nn.Sequential(*list(network.children())[:-2]), network.fc.in_features
        network = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        return network.features, network.classifier[-1].in_features
    except Exception as exc:
        if pretrained:
            raise RuntimeError(f"Could not load pretrained {name} weights through torchvision. Check network access or provide a local/cacheable weight source; no fallback weights were assumed. Original error: {exc}") from exc
        raise


if nn is not None:
    class HierarchicalDRClassifier(nn.Module):
        def __init__(self, backbone: str, num_classes: int = 5, pretrained: bool = True, ordinal_mode: bool = False):
            super().__init__()
            self.backbone_name = backbone
            self.num_classes = num_classes
            self.ordinal_mode = ordinal_mode
            self.feature_extractor, feature_dim = build_backbone(backbone, pretrained=pretrained)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(p=0.2)
            self.stage1_head = nn.Linear(feature_dim, 2)
            self.stage2_head = nn.Linear(feature_dim, 2)
            if ordinal_mode:
                self.ordinal_head = nn.Linear(feature_dim, num_classes - 1)
                self.severity_head = None
            else:
                self.severity_head = nn.Linear(feature_dim, num_classes)
                self.ordinal_head = None

        def forward(self, inputs):
            features = self.pool(self.feature_extractor(inputs)).flatten(1)
            features = self.dropout(features)
            result = {
                "stage1_logits": self.stage1_head(features),
                "stage2_logits": self.stage2_head(features),
                "features": features,
            }
            if self.ordinal_mode:
                result["ordinal_logits"] = self.ordinal_head(features)
            else:
                result["severity_logits"] = self.severity_head(features)
            return result
else:
    class HierarchicalDRClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


def ordinal_probabilities(ordinal_logits):
    """Decode cumulative P(y > threshold) logits into five class probabilities."""
    _require_torch()
    cumulative = torch.sigmoid(ordinal_logits)
    pieces = [1 - cumulative[:, 0]]
    pieces.extend(cumulative[:, index - 1] - cumulative[:, index] for index in range(1, cumulative.shape[1]))
    pieces.append(cumulative[:, -1])
    probabilities = torch.stack(pieces, dim=1).clamp_min(0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def severity_probabilities(outputs: dict[str, Any], ordinal_mode: bool):
    _require_torch()
    return ordinal_probabilities(outputs["ordinal_logits"]) if ordinal_mode else torch.softmax(outputs["severity_logits"], dim=1)


def hierarchical_targets(labels, mapping: ReferableDRMapping):
    _require_torch()
    stage1 = (labels > 0).long()
    stage2 = torch.tensor([1 if mapping.is_referable(int(label)) else 0 for label in labels.tolist()], device=labels.device)
    return stage1, stage2


def ordinal_targets(labels, num_classes: int = 5):
    _require_torch()
    return torch.stack([(labels > threshold).float() for threshold in range(num_classes - 1)], dim=1)


def build_classifier(backbone: str, num_classes: int = 5, pretrained: bool = True, ordinal_mode: bool = False):
    _require_torch()
    return HierarchicalDRClassifier(backbone=backbone, num_classes=num_classes, pretrained=pretrained, ordinal_mode=ordinal_mode)
