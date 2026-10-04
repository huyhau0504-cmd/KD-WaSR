"""MaSTr1325 dataset utilities.

Supports the official naming convention (``0001.jpg`` and ``0001m.png``) as
well as masks that share the image stem. Labels are returned as class indices:
0=obstacle/environment, 1=water, 2=sky, 4=ignore.
"""

from __future__ import annotations

from io import BytesIO
import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MASK_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff")
VALID_LABELS = {0, 1, 2, 4}
AUGMENTATION_PROFILES = {
    "baseline": {
        "horizontal_flip_probability": 0.5,
        "color_jitter": {
            "brightness": 0.20,
            "contrast": 0.20,
            "saturation": 0.15,
            "hue": 0.03,
        },
        "gamma": {"probability": 0.0, "range": [1.0, 1.0]},
        "gaussian_blur": {"probability": 0.0, "radius": [0.0, 0.0]},
        "jpeg": {"probability": 0.0, "quality": [100, 100]},
        "fog": {"probability": 0.0, "alpha": [0.0, 0.0]},
        "gaussian_noise": {"probability": 0.0, "sigma": [0.0, 0.0]},
    },
    "domain": {
        "horizontal_flip_probability": 0.5,
        "color_jitter": {
            "brightness": 0.30,
            "contrast": 0.30,
            "saturation": 0.20,
            "hue": 0.04,
        },
        "gamma": {"probability": 0.25, "range": [0.75, 1.35]},
        "gaussian_blur": {"probability": 0.20, "radius": [0.10, 1.20]},
        "jpeg": {"probability": 0.20, "quality": [55, 95]},
        "fog": {"probability": 0.15, "alpha": [0.04, 0.16]},
        "gaussian_noise": {"probability": 0.20, "sigma": [0.0, 0.025]},
    },
}


def augmentation_spec(profile: str) -> dict:
    if profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unknown augmentation profile '{profile}'. Expected {sorted(AUGMENTATION_PROFILES)}"
        )
    # The values are JSON-compatible; copying prevents callers from mutating
    # the pre-registered experiment definition.
    return {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in AUGMENTATION_PROFILES[profile].items()
    }


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
        augmentation_profile: str = "baseline",
    ) -> None:
        self.size = size
        self.train = train
        self.augmentation_profile = augmentation_profile
        self.spec = augmentation_spec(augmentation_profile)
        self.horizontal_flip_probability = self.spec["horizontal_flip_probability"]
        self.color_jitter = ColorJitter(**self.spec["color_jitter"])

    @staticmethod
    def _jpeg(image: Image.Image, quality: int) -> Image.Image:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")

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

            gamma = self.spec["gamma"]
            if random.random() < gamma["probability"]:
                image = TF.adjust_gamma(image, random.uniform(*gamma["range"]))

            blur = self.spec["gaussian_blur"]
            if random.random() < blur["probability"]:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(*blur["radius"])))

            jpeg = self.spec["jpeg"]
            if random.random() < jpeg["probability"]:
                image = self._jpeg(image, random.randint(*jpeg["quality"]))

        image_tensor = TF.to_tensor(image)
        if self.train:
            fog = self.spec["fog"]
            if random.random() < fog["probability"]:
                alpha = random.uniform(*fog["alpha"])
                image_tensor = image_tensor * (1.0 - alpha) + alpha

            noise = self.spec["gaussian_noise"]
            if random.random() < noise["probability"]:
                sigma = random.uniform(*noise["sigma"])
                image_tensor = (image_tensor + torch.randn_like(image_tensor) * sigma).clamp(0, 1)

        image_tensor = TF.normalize(
            image_tensor,
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
        augmentation_profile: str = "baseline",
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
        self.transform = JointTransform(
            size=size,
            train=train,
            augmentation_profile=augmentation_profile,
        )

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
