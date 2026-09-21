"""Generate class-ID masks while preserving the MODS sequence directory layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import load_checkpoint_model
from predict import IMAGE_EXTENSIONS, preprocess
from utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mods-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_checkpoint_model(str(args.checkpoint), device=device)
    model.eval()
    images = sorted(
        path for path in args.mods_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No MODS images found under {args.mods_root}")

    with torch.inference_mode():
        for path in tqdm(images, desc="MODS inference"):
            image = Image.open(path).convert("RGB")
            tensor = preprocess(image, (args.height, args.width)).unsqueeze(0).to(device)
            prediction = model(tensor)["out"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            output = args.output_dir / path.relative_to(args.mods_root)
            output = output.with_suffix(".png")
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction, mode="L").save(output)


if __name__ == "__main__":
    main()
