"""Render a deterministic montage to approve augmentation severity before training."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as TF

from datasets.mastr import (
    IMAGE_EXTENSIONS,
    MASK_EXTENSIONS,
    JointTransform,
    augmentation_spec,
    read_split,
)
from utils import save_json, seed_everything


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
COLORS = np.array([[255, 210, 0], [0, 210, 255], [55, 70, 150]], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/MaSTr1325"))
    parser.add_argument("--split-file", type=Path, default=Path("data/grouped_splits/train.txt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/augmentation_montage.jpg"))
    parser.add_argument("--profile", choices=("baseline", "domain"), default="domain")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_file(directory: Path, stem: str, extensions: tuple[str, ...]) -> Path:
    stems = (f"{stem}m", stem) if directory.name == "masks" else (stem,)
    for candidate_stem in stems:
        for extension in extensions:
            candidate = directory / f"{candidate_stem}{extension}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(stem)


def denormalize(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return TF.to_pil_image((tensor.cpu() * std + mean).clamp(0, 1))


def render_cell(image: Image.Image, mask: torch.Tensor, label: str) -> Image.Image:
    image = image.convert("RGB")
    mask_array = mask.cpu().numpy()
    color = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    valid = mask_array < 3
    color[valid] = COLORS[mask_array[valid]]
    overlay = Image.blend(image, Image.fromarray(color), alpha=0.30)
    canvas = Image.new("RGB", (image.width, image.height * 2 + 26), (25, 25, 25))
    canvas.paste(image, (0, 26))
    canvas.paste(overlay, (0, 26 + image.height))
    ImageDraw.Draw(canvas).text((8, 7), label, fill=(255, 255, 255))
    return canvas


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.variants < 1:
        raise ValueError("--samples and --variants must be positive")
    stems = read_split(args.split_file)[: args.samples]
    transform = JointTransform(
        size=(384, 512), train=True, augmentation_profile=args.profile
    )
    rows = []
    manifest = []
    for sample_index, stem in enumerate(stems):
        source = Image.open(
            find_file(args.data_root / "images", stem, IMAGE_EXTENSIONS)
        ).convert("RGB")
        mask = Image.open(find_file(args.data_root / "masks", stem, MASK_EXTENSIONS))
        cells = []
        seeds = []
        for variant in range(args.variants):
            variant_seed = args.seed + sample_index * 1000 + variant
            seed_everything(variant_seed)
            random.seed(variant_seed)
            image_tensor, mask_tensor, _ = transform(source, mask)
            cells.append(render_cell(denormalize(image_tensor), mask_tensor, f"{stem} seed={variant_seed}"))
            seeds.append(variant_seed)
        row = Image.new("RGB", (sum(cell.width for cell in cells), cells[0].height))
        left = 0
        for cell in cells:
            row.paste(cell, (left, 0))
            left += cell.width
        rows.append(row)
        manifest.append({"image": stem, "variant_seeds": seeds})

    montage = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)))
    top = 0
    for row in rows:
        montage.paste(row, (0, top))
        top += row.height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output, quality=92)
    save_json(
        args.output.with_suffix(".json"),
        {
            "profile": args.profile,
            "augmentation_config": augmentation_spec(args.profile),
            "samples": manifest,
        },
    )
    print(f"Saved augmentation montage to {args.output}")


if __name__ == "__main__":
    main()
