"""CASIA2 dataset over processed split CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resolve_processed_root(root: str | Path) -> Path:
    """Resolve a processed-data path even when the notebook cwd is ``ipynb/``."""
    root = Path(root)
    candidates: list[Path] = [root]
    if not root.is_absolute():
        cwd = Path.cwd().resolve()
        candidates.append(cwd / root)
        if cwd.name == "ipynb":
            candidates.append(cwd.parent / root)
        for parent in cwd.parents:
            candidates.append(parent / root)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and (resolved / "train.csv").is_file():
            return resolved
    return root.resolve()


class CASIA2Dataset(Dataset):
    """Load preprocessed CASIA2 RGB images, binary masks, and labels.

    Expects the layout produced by ``utils/prepare_casia2.py``::

        root/
          train.csv
          images/{split}/<stem>.png
          masks/{split}/<stem>.png
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test, got {split!r}")
        self.root = _resolve_processed_root(root)
        csv_path = self.root / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        self.frame = pd.read_csv(csv_path, keep_default_na=False)
        required = {"stem", "label", "image_path", "mask_path"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        mask = Image.open(self.root / row["mask_path"]).convert("L")

        image_t = torch.from_numpy(
            _hwc_to_chw(_pil_to_float(image))
        ).float()
        mask_t = torch.from_numpy(_pil_mask_to_float(mask)).unsqueeze(0).float()
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {
            "image": image_t,
            "mask": mask_t,
            "label": label,
            "stem": str(row["stem"]),
        }


def _pil_to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype="float32") / 255.0


def _hwc_to_chw(array: np.ndarray) -> np.ndarray:
    return array.transpose(2, 0, 1)


def _pil_mask_to_float(mask: Image.Image) -> np.ndarray:
    array = np.asarray(mask, dtype="float32")
    return (array > 127).astype("float32")
