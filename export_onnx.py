"""Export a trained segmentation checkpoint to fixed-resolution ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from models import load_checkpoint_model


class OutputWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["out"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 64 or args.width % 64:
        raise ValueError(
            "For reliable legacy ONNX export, height and width must be divisible by 64 "
            "(for example 384x512)."
        )
    model = OutputWrapper(load_checkpoint_model(str(args.checkpoint), device="cpu")).eval()
    example = torch.randn(1, 3, args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {"image": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            str(args.output),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"Exported {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
