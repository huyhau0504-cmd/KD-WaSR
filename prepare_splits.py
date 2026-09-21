"""Create deterministic MaSTr1325 train/validation/test split files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets.mastr import discover_image_stems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/MaSTr1325"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_split(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must be positive and train_ratio + val_ratio must be below 1.")

    stems = discover_image_stems(args.data_root / "images")
    if not stems:
        raise RuntimeError(f"No images found in {args.data_root / 'images'}")

    random.Random(args.seed).shuffle(stems)
    train_end = round(len(stems) * args.train_ratio)
    val_end = train_end + round(len(stems) * args.val_ratio)
    splits = {
        "train": sorted(stems[:train_end]),
        "val": sorted(stems[train_end:val_end]),
        "test": sorted(stems[val_end:]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        write_split(args.output_dir / f"{name}.txt", values)
        print(f"{name}: {len(values)}")
    print(
        "Warning: this is an image-level deterministic split. Replace it with a session-grouped "
        "split before paper submission if capture-session metadata is available."
    )


if __name__ == "__main__":
    main()
