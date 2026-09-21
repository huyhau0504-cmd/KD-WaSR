"""MaSTr1325 dataset utilities.

Supports the official naming convention (``0001.jpg`` and ``0001m.png``) as
well as masks that share the image stem. Labels are returned as class indices:
0=obstacle/environment, 1=water, 2=sky, 4=ignore.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MASK_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff")
VALID_LABELS = {0, 1, 2, 4}


def decode_mask(mask: Image.Image) -> np.ndarray:
    """Decode grayscale IDs or the official RGB one-hot-style MaSTr masks."""
    array = np.asarray(mask)
    if array.ndim == 2:
        decoded = array.astype(np.int64)
    elif array.ndim == 3:
        rgb = array[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(
            rgb[..., 1], rgb[..., 2]
        ):
            decoded = rgb[..., 0].astype(np.int64)
        else:
            active = rgb.max(axis=-1) > 0
            decoded = rgb.argmax(axis=-1).astype(np.int64)
            decoded[~active] = 4
    else:
        raise ValueError(f"Unsupported mask shape: {array.shape}")

    labels = set(np.unique(decoded).tolist())
    if not labels.issubset(VALID_LABELS):
        raise ValueError(f"Unexpected mask labels {sorted(labels)}; expected {sorted(VALID_LABELS)}")
    return decoded


def discover_image_stems(image_dir: Path) -> list[str]:
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(p.stem for p in files)


def _find_existing(directory: Path, stems: Iterable[str], extensions: Sequence[str]) -> Path:
    for stem in stems:
        for extension in extensions:
            candidate = directory / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No matching file in {directory} for stems={list(stems)} and extensions={extensions}"
    )


def read_split(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as stream:
        return [Path(line.strip()).stem for line in stream if line.strip() and not line.startswith("#")]


class JointTransform:
    def __init__(
        self,
        size: tuple[int, int] = (384, 512),
        train: bool = False,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        self.size = size
        self.train = train
        self.horizontal_flip_probability = horizontal_flip_probability
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)

    def __call__(
        self, image: Image.Image, mask: Image.Image, imu: Optional[Image.Image] = None
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        if imu is not None:
            imu = TF.resize(imu, self.size, interpolation=InterpolationMode.NEAREST)

        if self.train and random.random() < self.horizontal_flip_probability:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            if imu is not None:
                imu = TF.hflip(imu)
        if self.train:
            image = self.color_jitter(image)

        image_tensor = TF.normalize(
            TF.to_tensor(image),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        mask_tensor = torch.from_numpy(decode_mask(mask)).long()
        imu_tensor = None
        if imu is not None:
            imu_tensor = torch.from_numpy(np.asarray(imu) > 0).float()
        return image_tensor, mask_tensor, imu_tensor


class MaSTr1325Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        train: bool = False,
        size: tuple[int, int] = (384, 512),
        require_imu: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.imu_dir = self.root / "imus"
        self.require_imu = require_imu

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.image_dir} and {self.mask_dir}. See README.md for the layout."
            )

        self.stems = read_split(Path(split_file)) if split_file else discover_image_stems(self.image_dir)
        if not self.stems:
            raise RuntimeError(
                f"No images found in {self.image_dir}. Download and extract MaSTr1325 first."
            )
        self.transform = JointTransform(size=size, train=train)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        stem = self.stems[index]
        image_path = _find_existing(self.image_dir, (stem,), IMAGE_EXTENSIONS)
        mask_path = _find_existing(self.mask_dir, (f"{stem}m", stem), MASK_EXTENSIONS)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        original_size = (image.height, image.width)

        imu = None
        if self.imu_dir.is_dir():
            try:
                imu_path = _find_existing(self.imu_dir, (stem,), MASK_EXTENSIONS)
                imu = Image.open(imu_path).convert("L")
            except FileNotFoundError:
                if self.require_imu:
                    raise
        elif self.require_imu:
            raise FileNotFoundError(f"IMU directory not found: {self.imu_dir}")

        image_tensor, mask_tensor, imu_tensor = self.transform(image, mask, imu)
        features = {"image": image_tensor}
        if imu_tensor is not None:
            features["imu_mask"] = imu_tensor
        target = {
            "mask": mask_tensor,
            "name": stem,
            "original_size": torch.tensor(original_size, dtype=torch.int64),
        }
        return features, target


def split_paths(split_dir: str | Path) -> dict[str, Path]:
    split_dir = Path(split_dir)
    paths = {name: split_dir / f"{name}.txt" for name in ("train", "val", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files: {missing}. Run prepare_splits.py before training."
        )
    return paths
