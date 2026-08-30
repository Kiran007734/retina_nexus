"""Dataset loader for the governance split manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


class FundusClassificationDataset:
    def __init__(self, manifest_path: str | Path, raw_root: str | Path, split: str, transform=None):
        import torch
        from torch.utils.data import Dataset

        self._dataset_base = Dataset
        self.manifest_path = Path(manifest_path)
        self.raw_root = Path(raw_root)
        self.split = split
        self.transform = transform
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.records = [record for record in payload.get("records", []) if record.get("split") == split and record.get("label") not in (None, "")]
        if not self.records:
            raise ValueError(f"No labeled records found for split '{split}' in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch
        record = self.records[index]
        path = (self.raw_root / record["image"]).resolve()
        path.relative_to(self.raw_root.resolve())
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform else image
        label = int(record["label"])
        return tensor, torch.tensor(label, dtype=torch.long), record
