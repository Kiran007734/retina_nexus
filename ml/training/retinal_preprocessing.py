"""Shared retinal-field preprocessing used by research artifacts.

The crop removes only very dark outer acquisition borders.  It is opt-in via
the registered model artifact; the APTOS production artifact does not enable
it, so production preprocessing remains unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class RetinalFieldCrop:
    """Remove dark outer borders while preserving the visible retinal field."""

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        mask = gray > 0.035
        ys, xs = np.where(mask)
        if len(xs) < 256:
            return image.convert("RGB")
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
        width, height = right - left + 1, bottom - top + 1
        if width * height >= 0.92 * 256 * 256:
            return image.convert("RGB")
        margin_x = max(2, int(width * 0.02))
        margin_y = max(2, int(height * 0.02))
        scale_x = image.width / 256.0
        scale_y = image.height / 256.0
        box = (
            max(0, int((left - margin_x) * scale_x)),
            max(0, int((top - margin_y) * scale_y)),
            min(image.width, int((right + margin_x + 1) * scale_x)),
            min(image.height, int((bottom + margin_y + 1) * scale_y)),
        )
        if box[2] - box[0] < 32 or box[3] - box[1] < 32:
            return image.convert("RGB")
        return image.convert("RGB").crop(box)


def build_inference_transform(input_size: int, retinal_field_crop: bool = False) -> Any:
    """Build the deterministic transform recorded by a model artifact."""
    from torchvision import transforms

    steps: list[Any] = []
    if retinal_field_crop:
        steps.append(RetinalFieldCrop())
    steps.extend([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)
