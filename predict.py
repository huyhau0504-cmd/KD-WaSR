"""Predict MaSTr-style segmentation masks for an image or directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from models import load_checkpoint_model
from utils import resolve_device


COLORS = np.array(
    [
        [255, 210, 0],  # obstacle/environment
        [0, 210, 255],  # water
        [55, 70, 150],  # sky
    ],
    dtype=np.uint8,
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    return parser.parse_args()


def preprocess(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    image = TF.resize(image.convert("RGB"), size, interpolation=InterpolationMode.BILINEAR)
    return TF.normalize(
        TF.to_tensor(image),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def save_prediction(
    source: Image.Image,
    prediction: np.ndarray,
    output_stem: Path,
    alpha: float,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    raw = Image.fromarray(prediction.astype(np.uint8), mode="L")
    raw.save(output_stem.with_name(output_stem.name + "_mask.png"))
    color = Image.fromarray(COLORS[prediction], mode="RGB")
    source_resized = source.convert("RGB").resize(color.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(source_resized, color, alpha=alpha)
    overlay.save(output_stem.with_name(output_stem.name + "_overlay.jpg"), quality=92)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_checkpoint_model(str(args.checkpoint), device=device)
    model.eval()
    paths = image_paths(args.input)
    if not paths:
        raise RuntimeError(f"No images found under {args.input}")
    base = args.input if args.input.is_dir() else args.input.parent

    with torch.inference_mode():
        for path in tqdm(paths, desc="predict"):
            image = Image.open(path).convert("RGB")
            tensor = preprocess(image, (args.height, args.width)).unsqueeze(0).to(device)
            prediction = model(tensor)["out"].argmax(dim=1)[0].cpu().numpy()
            relative = path.relative_to(base)
            save_prediction(
                image,
                prediction,
                args.output_dir / relative.parent / relative.stem,
                args.overlay_alpha,
            )


if __name__ == "__main__":
    main()
