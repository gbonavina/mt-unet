from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SKIP_NAMES = {"_list.txt", "thumbs.db"}
DEFAULT_DATA_ROOT = ROOT / "data" / "CASIA2"


def _iter_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing image folder: {folder}")
    for path in folder.iterdir():
        if not path.is_file() or path.name.lower() in SKIP_NAMES:
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return sorted(files)


def collect_casia2_samples(root: str | Path | None = None) -> list[tuple[Path, int]]:
    root = Path(root) if root is not None else DEFAULT_DATA_ROOT
    authentic = [(path, 0) for path in _iter_images(root / "Au")]
    tampered = [(path, 1) for path in _iter_images(root / "Tp")]
    samples = authentic + tampered
    if not samples:
        raise FileNotFoundError(f"No images found under {root}")
    return samples


def stratified_train_val_test_split(
    labels: Sequence[int],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {total}")

    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    rest_ratio = val_ratio + test_ratio
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=rest_ratio,
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_ratio / rest_ratio,
        stratify=labels[rest_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def _build_transform(image_size: int, train: bool) -> transforms.Compose:
    ops: list[object] = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class CASIA2ImageDataset(Dataset):
    """RGB (or ELA) classification samples. Returns ``(image, label)``."""

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        image_size: int = 224,
        train: bool = False,
        use_ela: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.use_ela = use_ela
        self.transform = _build_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> list[int]:
        return [label for _path, label in self.samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        if self.use_ela:
            from utils.ela import compute_ela

            image = compute_ela(path)
        else:
            image = Image.open(path).convert("RGB")
        image_t = self.transform(image)
        label_t = torch.tensor(label, dtype=torch.long)
        return image_t, label_t


class StratifiedBatchSampler(Sampler[list[int]]):
    """Yield batches whose class counts follow the dataset class ratios."""

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        classes = np.unique(self.labels)
        self.class_indices = [np.flatnonzero(self.labels == c).tolist() for c in classes]
        class_sizes = [len(idx) for idx in self.class_indices]
        self.per_class = _per_class_quota(class_sizes, batch_size)
        self.n_samples = len(self.labels)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size
        return math.ceil(self.n_samples / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(
            None if self.seed is None else self.seed + self._epoch
        )
        self._epoch += 1

        pools = [idx.copy() for idx in self.class_indices]
        if self.shuffle:
            for pool in pools:
                rng.shuffle(pool)

        pointers = [0] * len(pools)
        batches: list[list[int]] = []

        while True:
            batch: list[int] = []
            can_fill = all(
                pointers[c] + n_c <= len(pools[c])
                for c, n_c in enumerate(self.per_class)
            )
            if not can_fill:
                leftover: list[int] = []
                for c, pool in enumerate(pools):
                    leftover.extend(pool[pointers[c] :])
                if leftover and not self.drop_last:
                    if self.shuffle:
                        rng.shuffle(leftover)
                    for start in range(0, len(leftover), self.batch_size):
                        chunk = leftover[start : start + self.batch_size]
                        if chunk:
                            batches.append(chunk)
                break

            for c, n_c in enumerate(self.per_class):
                start = pointers[c]
                end = start + n_c
                batch.extend(pools[c][start:end])
                pointers[c] = end
            if self.shuffle:
                rng.shuffle(batch)
            batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def _per_class_quota(class_sizes: Sequence[int], batch_size: int) -> list[int]:
    total = sum(class_sizes)
    raw = [batch_size * size / total for size in class_sizes]
    counts = [int(math.floor(value)) for value in raw]
    leftover = batch_size - sum(counts)
    frac_order = np.argsort([-(value - math.floor(value)) for value in raw])
    for i in range(leftover):
        counts[int(frac_order[i])] += 1

    if batch_size >= len(class_sizes):
        for i, count in enumerate(counts):
            if count == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[i] += 1
    return counts


def _split_counts(samples: Sequence[tuple[Path, int]]) -> dict[str, int]:
    labels = Counter(label for _path, label in samples)
    return {
        "n": len(samples),
        "authentic": labels.get(0, 0),
        "tampered": labels.get(1, 0),
    }


def get_dataloaders(
    root: str | Path | None = None,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
    use_ela: bool = False,
    num_workers: int = 4,
    drop_last: bool = False,
) -> dict[str, DataLoader]:
    """Build stratified 80/10/10 CASIA2 loaders with stratified batches."""
    samples = collect_casia2_samples(root)
    labels = [label for _path, label in samples]
    train_idx, val_idx, test_idx = stratified_train_val_test_split(labels, seed=seed)

    split_samples = {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }
    datasets = {
        split: CASIA2ImageDataset(
            split_samples[split],
            image_size=image_size,
            train=(split == "train"),
            use_ela=use_ela,
        )
        for split in ("train", "val", "test")
    }

    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        sampler = StratifiedBatchSampler(
            dataset.labels,
            batch_size=batch_size,
            drop_last=drop_last and split == "train",
            shuffle=(split == "train"),
            seed=seed,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


if __name__ == "__main__":
    samples = collect_casia2_samples()
    labels = [label for _path, label in samples]
    train_idx, val_idx, test_idx = stratified_train_val_test_split(labels)
    grouped = {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }
    print("total", _split_counts(samples))
    for split, items in grouped.items():
        print(split, _split_counts(items))
    sampler = StratifiedBatchSampler(
        [label for _path, label in grouped["train"]],
        batch_size=32,
        shuffle=False,
        seed=42,
    )
    first = next(iter(sampler))
    batch_labels = [grouped["train"][i][1] for i in first]
    print("first train batch size", len(first), "counts", dict(Counter(batch_labels)))
