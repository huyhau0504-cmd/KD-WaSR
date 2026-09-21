"""Static INT8 post-training quantization using train-split calibration images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from torch.utils.data import DataLoader

from datasets import MaSTr1325Dataset, split_paths


class MaSTrCalibrationReader(CalibrationDataReader):
    def __init__(self, loader: DataLoader, maximum_samples: int) -> None:
        self.iterator = iter(loader)
        self.maximum_samples = maximum_samples
        self.seen = 0

    def get_next(self):
        if self.seen >= self.maximum_samples:
            return None
        try:
            features, _ = next(self.iterator)
        except StopIteration:
            return None
        self.seen += features["image"].shape[0]
        return {"image": features["image"].numpy().astype(np.float32)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/MaSTr1325"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--samples", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = split_paths(args.split_dir)["train"]
    dataset = MaSTr1325Dataset(
        args.data_root, train_path, train=False, size=(args.height, args.width)
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    reader = MaSTrCalibrationReader(loader, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.model),
        str(args.output),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    print(f"Quantized {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
