"""Dataset discovery, stratified splitting and transforms for the hip-implant X-rays.

The Kaggle database (tawsifurrahman/aseptic-loose-hip-implant-xray-database)
stores 331x331 grayscale PNGs in two directories::

    <root>/Control/control (N).png   -> label 0 (well fixed)
    <root>/Loose/loose (N).png       -> label 1 (aseptically loose)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ("Control", "Loose")
# ImageNet statistics: the grayscale scan is replicated to 3 channels so the
# pretrained backbone sees the distribution it was trained on.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def find_data_root(start: Path) -> Path:
    """Return the directory that directly contains ``Control/`` and ``Loose/``."""
    start = start.expanduser().resolve()
    if not start.exists():
        raise SystemExit(f"Data path does not exist: {start}")
    if _is_data_root(start):
        return start
    candidates = sorted(p for p in start.rglob("*") if p.is_dir() and _is_data_root(p))
    if not candidates:
        raise SystemExit(
            f"Could not find a directory containing 'Control' and 'Loose' under {start}.\n"
            "Pass --data-root pointing at the folder that holds both class directories."
        )
    if len(candidates) > 1:
        print(f"[data] several candidate roots found, using {candidates[0]}")
    return candidates[0]


def _is_data_root(path: Path) -> bool:
    return all((path / name).is_dir() for name in CLASS_NAMES)


def list_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for label, name in enumerate(CLASS_NAMES):
        files = sorted(
            p
            for p in (root / name).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise SystemExit(f"No images found in {root / name}")
        samples.extend(Sample(path=p, label=label) for p in files)
    return samples


def stratified_indices(labels: list[int], seed: int) -> dict[int, list[int]]:
    """Shuffle indices once per class so every split stays class balanced."""
    rng = random.Random(seed)
    per_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        per_class.setdefault(label, []).append(index)
    for indices in per_class.values():
        rng.shuffle(indices)
    return per_class


def holdout_split(
    labels: list[int], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Stratified train/val/test split with no overlap between the three sets."""
    train, val, test = [], [], []
    for indices in stratified_indices(labels, seed).values():
        n = len(indices)
        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)
        if n - n_test - n_val < 1:
            raise SystemExit("--val-ratio/--test-ratio leave no training images for a class")
        test.extend(indices[:n_test])
        val.extend(indices[n_test : n_test + n_val])
        train.extend(indices[n_test + n_val :])
    return sorted(train), sorted(val), sorted(test)


def stratified_folds(labels: list[int], n_splits: int, seed: int) -> list[list[int]]:
    """Return ``n_splits`` index lists, each usable as a validation fold."""
    if n_splits < 2:
        raise SystemExit("--folds must be >= 2 for cross-validation")
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for indices in stratified_indices(labels, seed).values():
        for position, index in enumerate(indices):
            folds[position % n_splits].append(index)
    return [sorted(fold) for fold in folds]


class XrayDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        # X-rays are single channel; ``convert("L")`` also normalises the odd
        # palette/RGBA PNGs some of the exports contain.
        image = Image.open(sample.path).convert("L")
        return self.transform(image), torch.tensor(float(sample.label))


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Augmentations kept anatomically plausible: no vertical flip, mild rotation.

    A horizontal flip turns a left hip into a right hip, which is a real and
    common variation in the database, so it is safe. Upside-down pelvises are
    not, and neither are strong shears that would fake implant subsidence.
    """
    if train:
        geometry = [
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.11)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=12, translate=(0.05, 0.05))], p=0.7
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
        ]
    else:
        geometry = [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
        ]
    return transforms.Compose(
        [
            *geometry,
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            *([transforms.RandomErasing(p=0.25, scale=(0.02, 0.12))] if train else []),
        ]
    )
