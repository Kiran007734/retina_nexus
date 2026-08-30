"""DRIVE image/vessel-mask pairing and tensor dataset utilities."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MASK_EXTENSIONS = IMAGE_EXTENSIONS | {".gif"}


def _key(path: Path) -> str:
    value = path.stem.lower()
    value = re.sub(r"(?:_)?(?:training|test|manual1|manual|1st|2nd|mask|image)$", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def find_drive_pairs(raw_root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(raw_root).expanduser().resolve()
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() not in MASK_EXTENSIONS:
            continue
        is_mask = any(token in name for token in ("manual", "mask", "1st", "2nd"))
        target = masks if is_mask else images
        target.setdefault(_key(path), path)
    return [(images[key], masks[key]) for key in sorted(images.keys() & masks.keys())]


class DriveVesselDataset:
    def __init__(self, pairs: list[tuple[Path, Path]], input_size: int = 512):
        self.pairs = pairs
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import torch

        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image_source:
            image = image_source.convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        with Image.open(mask_path) as mask_source:
            mask = mask_source.convert("L").resize((self.input_size, self.input_size), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) >= 128).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_array[None, ...]).float()
        return image_tensor, mask_tensor
