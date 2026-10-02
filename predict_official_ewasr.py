"""Run the official pretrained eWaSR ResNet-18 ONNX model.

The model artifact is published by the eWaSR authors under Apache-2.0:
https://github.com/tersekmatija/eWaSR/releases/tag/0.1.0
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


OFFICIAL_MODEL_URL = (
    "https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/"
    "ewasr_resnet18.onnx"
)
OFFICIAL_MODEL_SHA256 = "15e520fc5f7e910c9367556db9a9c1107bdc9c74e7982b3bf89c7050509411d7"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
COLORS = np.array(
    [
        [247, 195, 37],  # obstacle/environment
        [41, 167, 224],  # water
        [90, 75, 164],  # sky
    ],
    dtype=np.uint8,
)
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("pretrained/ewasr_resnet18.onnx"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/official_ewasr_predictions"),
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_official_model(path: Path, download: bool) -> Path:
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Official model not found at {path}. Pass --download to fetch it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading official eWaSR model to {path} ...")
        with urlopen(OFFICIAL_MODEL_URL) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)

    actual_hash = sha256(path)
    if actual_hash != OFFICIAL_MODEL_SHA256:
        raise RuntimeError(
            "Official model checksum mismatch: "
            f"expected {OFFICIAL_MODEL_SHA256}, got {actual_hash}"
        )
    return path


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def preprocess(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize((512, 384), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[None].astype(np.float32)


def logits_to_mask(logits: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    # The official ONNX output is 96x128. The original training code first
    # bilinearly upsamples logits to 384x512 and only then takes argmax.
    tensor = torch.from_numpy(logits)
    tensor = F.interpolate(tensor, size=(384, 512), mode="bilinear", align_corners=False)
    mask = tensor.argmax(dim=1)[0].numpy().astype(np.uint8)
    return np.asarray(
        Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def save_prediction(
    source: Image.Image,
    mask: np.ndarray,
    output_stem: Path,
    alpha: float,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(
        output_stem.with_name(output_stem.name + "_mask.png")
    )
    color = Image.fromarray(COLORS[mask], mode="RGB")
    color.save(output_stem.with_name(output_stem.name + "_color.png"))
    overlay = Image.blend(source.convert("RGB"), color, alpha=alpha)
    overlay.save(output_stem.with_name(output_stem.name + "_overlay.jpg"), quality=92)


def main() -> None:
    args = parse_args()
    model_path = ensure_official_model(args.model_path, args.download)
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_meta = session.get_inputs()
    if len(input_meta) != 1 or input_meta[0].name != "image":
        raise RuntimeError(f"Unexpected official model inputs: {input_meta}")

    paths = image_paths(args.input)
    if not paths:
        raise RuntimeError(f"No images found under {args.input}")
    base = args.input if args.input.is_dir() else args.input.parent
    print(f"providers={session.get_providers()} images={len(paths)}")

    for path in tqdm(paths, desc="official-eWaSR"):
        with Image.open(path) as opened:
            source = opened.convert("RGB")
        logits = session.run(None, {"image": preprocess(source)})[0]
        mask = logits_to_mask(logits, source.size)
        relative = path.relative_to(base)
        save_prediction(
            source,
            mask,
            args.output_dir / relative.parent / relative.stem,
            args.overlay_alpha,
        )


if __name__ == "__main__":
    main()
