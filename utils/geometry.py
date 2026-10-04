"""Aspect-ratio-safe preprocessing and inverse mapping for segmentation.

All sizes are stored as ``(height, width)``.  Padding is applied *after*
ImageNet normalization with a value of zero, which is equivalent to padding
the RGB image with the ImageNet mean without introducing a black border.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from PIL import Image
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


GeometryMode = Literal["stretch", "letterbox", "center_crop"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
UNKNOWN_LABEL = 255


@dataclass(frozen=True)
class GeometryMetadata:
    mode: GeometryMode
    original_size: tuple[int, int]
    target_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)  # left, top, right, bottom
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)  # left, top, right, bottom

    def to_dict(self) -> dict:
        return asdict(self)


def _scaled_size(
    original_size: tuple[int, int],
    target_size: tuple[int, int],
    fit: Literal["inside", "cover"],
) -> tuple[int, int]:
    original_height, original_width = original_size
    target_height, target_width = target_size
    ratios = (target_height / original_height, target_width / original_width)
    scale = min(ratios) if fit == "inside" else max(ratios)
    height = max(1, int(round(original_height * scale)))
    width = max(1, int(round(original_width * scale)))
    return height, width


def preprocess_image(
    image: Image.Image,
    target_size: tuple[int, int] = (384, 512),
    mode: GeometryMode = "stretch",
) -> tuple[Tensor, GeometryMetadata]:
    """Return a normalized CHW tensor and reversible geometry metadata."""
    source = image.convert("RGB")
    original_size = (source.height, source.width)
    target_height, target_width = target_size

    if mode == "stretch":
        resized_size = target_size
        transformed = TF.resize(source, target_size, interpolation=InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(transformed)
        metadata = GeometryMetadata(mode, original_size, target_size, resized_size)
    elif mode == "letterbox":
        resized_size = _scaled_size(original_size, target_size, "inside")
        transformed = TF.resize(source, resized_size, interpolation=InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(transformed)
        resized_height, resized_width = resized_size
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        right = target_width - resized_width - left
        bottom = target_height - resized_height - top
        metadata = GeometryMetadata(
            mode,
            original_size,
            target_size,
            resized_size,
            padding=(left, top, right, bottom),
        )
    elif mode == "center_crop":
        resized_size = _scaled_size(original_size, target_size, "cover")
        transformed = TF.resize(source, resized_size, interpolation=InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(transformed)
        resized_height, resized_width = resized_size
        left = (resized_width - target_width) // 2
        top = (resized_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        tensor = tensor[:, top:bottom, left:right]
        metadata = GeometryMetadata(
            mode,
            original_size,
            target_size,
            resized_size,
            crop=(left, top, right, bottom),
        )
    else:
        raise ValueError(f"Unknown geometry mode: {mode}")

    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if mode == "letterbox":
        left, top, right, bottom = metadata.padding
        tensor = F.pad(tensor, (left, right, top, bottom), mode="constant", value=0.0)
    if tuple(tensor.shape[-2:]) != target_size:
        raise RuntimeError(
            f"Geometry transform returned {tuple(tensor.shape[-2:])}, expected {target_size}"
        )
    return tensor, metadata


def restore_logits(logits: Tensor, metadata: GeometryMetadata) -> tuple[Tensor, Tensor]:
    """Map BCHW/CHW logits back to the source image.

    The returned boolean valid mask is all true for stretch/letterbox.  For
    center crop it explicitly marks pixels that were visible to the model;
    missing pixels must remain ``unknown`` in any derived class mask.
    """
    squeeze = logits.ndim == 3
    if squeeze:
        logits = logits.unsqueeze(0)
    if logits.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW logits, got shape {tuple(logits.shape)}")

    target_size = metadata.target_size
    if tuple(logits.shape[-2:]) != target_size:
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

    if metadata.mode == "stretch":
        restored = F.interpolate(
            logits, size=metadata.original_size, mode="bilinear", align_corners=False
        )
        valid = torch.ones(
            (logits.shape[0], *metadata.original_size), dtype=torch.bool, device=logits.device
        )
    elif metadata.mode == "letterbox":
        left, top, right, bottom = metadata.padding
        height, width = target_size
        content = logits[:, :, top : height - bottom, left : width - right]
        if tuple(content.shape[-2:]) != metadata.resized_size:
            raise RuntimeError("Letterbox metadata does not match the transformed logits")
        restored = F.interpolate(
            content, size=metadata.original_size, mode="bilinear", align_corners=False
        )
        valid = torch.ones(
            (logits.shape[0], *metadata.original_size), dtype=torch.bool, device=logits.device
        )
    elif metadata.mode == "center_crop":
        left, top, right, bottom = metadata.crop
        resized_height, resized_width = metadata.resized_size
        full = torch.zeros(
            (logits.shape[0], logits.shape[1], resized_height, resized_width),
            dtype=logits.dtype,
            device=logits.device,
        )
        visible = F.interpolate(
            logits, size=(bottom - top, right - left), mode="bilinear", align_corners=False
        )
        full[:, :, top:bottom, left:right] = visible
        valid_resized = torch.zeros(
            (logits.shape[0], 1, resized_height, resized_width),
            dtype=logits.dtype,
            device=logits.device,
        )
        valid_resized[:, :, top:bottom, left:right] = 1.0
        restored = F.interpolate(
            full, size=metadata.original_size, mode="bilinear", align_corners=False
        )
        valid = F.interpolate(
            valid_resized, size=metadata.original_size, mode="nearest"
        )[:, 0].bool()
    else:
        raise ValueError(f"Unknown geometry mode: {metadata.mode}")

    return (restored[0] if squeeze else restored), (valid[0] if squeeze else valid)


def logits_to_mask(
    logits: Tensor,
    metadata: GeometryMetadata,
    unknown_label: int = UNKNOWN_LABEL,
) -> Tensor:
    restored, valid = restore_logits(logits, metadata)
    if restored.ndim == 3:
        prediction = restored.argmax(dim=0)
    else:
        prediction = restored.argmax(dim=1)
    prediction = prediction.to(torch.uint8)
    return torch.where(valid, prediction, torch.full_like(prediction, unknown_label))
