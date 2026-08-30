"""Small configurable segmentation baseline for retinal structure experiments."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # Keep API imports safe when optional ML dependencies are absent.
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not installed. Install backend/requirements-ml.txt for evidence model training/evaluation.")


if nn is not None:
    class _ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs):
            return self.block(inputs)


    class LightweightUNet(nn.Module):
        """A compact U-Net suitable for a DRIVE vessel-mask baseline."""

        def __init__(self, input_channels: int = 3, output_channels: int = 1, width: int = 16):
            super().__init__()
            self.encoder1 = _ConvBlock(input_channels, width)
            self.encoder2 = _ConvBlock(width, width * 2)
            self.bottleneck = _ConvBlock(width * 2, width * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
            self.decoder2 = _ConvBlock(width * 4, width * 2)
            self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
            self.decoder1 = _ConvBlock(width * 2, width)
            self.head = nn.Conv2d(width, output_channels, 1)

        def forward(self, inputs):
            first = self.encoder1(inputs)
            second = self.encoder2(self.pool(first))
            bridge = self.bottleneck(self.pool(second))
            decoded_second = self.up2(bridge)
            if decoded_second.shape[-2:] != second.shape[-2:]:
                decoded_second = torch.nn.functional.interpolate(decoded_second, size=second.shape[-2:], mode="bilinear", align_corners=False)
            decoded_second = self.decoder2(torch.cat([decoded_second, second], dim=1))
            decoded_first = self.up1(decoded_second)
            if decoded_first.shape[-2:] != first.shape[-2:]:
                decoded_first = torch.nn.functional.interpolate(decoded_first, size=first.shape[-2:], mode="bilinear", align_corners=False)
            decoded_first = self.decoder1(torch.cat([decoded_first, first], dim=1))
            return self.head(decoded_first)
else:
    class LightweightUNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch()


def build_vessel_segmentation_model() -> LightweightUNet:
    require_torch()
    return LightweightUNet()
