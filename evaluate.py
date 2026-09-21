"""Evaluate a trained checkpoint on a locked MaSTr1325 split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MaSTr1325Dataset, split_paths
from models import load_checkpoint_model
from utils import SegmentationMetrics, move_features, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/MaSTr1325"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    path = split_paths(args.split_dir)[args.split]
    dataset = MaSTr1325Dataset(
        args.data_root, path, train=False, size=(args.height, args.width)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_checkpoint_model(str(args.checkpoint), device=device)
    model.eval()
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for features, target in tqdm(loader, desc=f"evaluate:{args.split}"):
            features = move_features(features, device)
            mask = target["mask"].to(device, non_blocking=True)
            metrics.update(model(features)["out"], mask)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "samples": len(dataset),
        **metrics.compute(),
    }
    for key, value in result.items():
        print(f"{key}: {value}")
    output = args.output or args.checkpoint.with_name(f"metrics_{args.split}.json")
    save_json(output, result)


if __name__ == "__main__":
    main()
