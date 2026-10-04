"""Create a fixed qualitative geometry comparison without cherry-picking images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from tqdm import tqdm

from models import load_checkpoint_model
from predict import image_paths, predict_image, save_prediction
from utils import resolve_device, save_json


MODES = ("stretch", "letterbox", "center_crop")
LABELS = ("Original", "S0 stretch", "S0 letterbox", "S0 center-crop", "Official eWaSR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry_comparison"))
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rows-per-page", type=int, default=10)
    return parser.parse_args()


def panel(image: Image.Image | None, label: str, size: tuple[int, int] = (320, 240)) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", (width, height + 28), (30, 30, 30))
    if image is not None:
        fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
        left = (width - fitted.width) // 2
        top = 28 + (height - fitted.height) // 2
        canvas.paste(fitted, (left, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill=(255, 255, 255))
    return canvas


def main() -> None:
    args = parse_args()
    if args.rows_per_page < 1:
        raise ValueError("--rows-per-page must be positive")
    paths = image_paths(args.input)
    if not paths:
        raise RuntimeError(f"No images found under {args.input}")
    device = resolve_device(args.device)
    model = load_checkpoint_model(str(args.checkpoint), device=device).eval()
    base = args.input if args.input.is_dir() else args.input.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    rows: list[Image.Image] = []
    page = 1

    with torch.inference_mode():
        for path in tqdm(paths, desc="geometry comparison"):
            source = Image.open(path).convert("RGB")
            relative = path.relative_to(base)
            predicted_overlays: list[Image.Image] = []
            metadata_by_mode = {}
            for mode in MODES:
                prediction, metadata = predict_image(
                    model, source, (args.height, args.width), mode, device
                )
                output_stem = args.output_dir / mode / relative.parent / relative.stem
                save_prediction(source, prediction, output_stem, alpha=0.55)
                predicted_overlays.append(
                    Image.open(output_stem.with_name(output_stem.name + "_overlay.jpg")).copy()
                )
                metadata_by_mode[mode] = metadata

            official_mask_path = (
                args.official_dir / relative.parent / f"{relative.stem}_mask.png"
            )
            official = None
            if official_mask_path.exists():
                official_mask = np.asarray(Image.open(official_mask_path), dtype=np.uint8)
                official_stem = (
                    args.output_dir / "official_common_style" / relative.parent / relative.stem
                )
                save_prediction(source, official_mask, official_stem, alpha=0.55)
                official = Image.open(
                    official_stem.with_name(official_stem.name + "_overlay.jpg")
                ).copy()
            cells = [source, *predicted_overlays, official]
            rendered = [panel(image, label) for image, label in zip(cells, LABELS)]
            row = Image.new("RGB", (sum(item.width for item in rendered), rendered[0].height))
            offset = 0
            for item in rendered:
                row.paste(item, (offset, 0))
                offset += item.width
            rows.append(row)
            records.append(
                {
                    "image": relative.as_posix(),
                    "official_mask_available": official is not None,
                    "geometry": metadata_by_mode,
                }
            )

            if len(rows) == args.rows_per_page or path == paths[-1]:
                sheet = Image.new("RGB", (rows[0].width, sum(item.height for item in rows)))
                top = 0
                for item in rows:
                    sheet.paste(item, (0, top))
                    top += item.height
                sheet.save(args.output_dir / f"contact_sheet_{page:02d}.jpg", quality=92)
                rows.clear()
                page += 1

    save_json(
        args.output_dir / "comparison_manifest.json",
        {
            "checkpoint": str(args.checkpoint),
            "input": str(args.input),
            "official_dir": str(args.official_dir),
            "rendering": {
                "student_and_official_palette": "predict.COLORS",
                "overlay_alpha": 0.55,
            },
            "ordered_images": [path.relative_to(base).as_posix() for path in paths],
            "records": records,
        },
    )
    print(f"Saved {len(paths)} fixed-order comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
