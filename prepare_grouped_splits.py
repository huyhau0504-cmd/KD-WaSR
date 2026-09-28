"""Audit MaSTr1325 near-duplicates and create deterministic grouped splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from datasets.mastr import IMAGE_EXTENSIONS, MASK_EXTENSIONS, decode_mask, discover_image_stems
from utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/MaSTr1325"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grouped_splits"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--embedding-threshold", type=float, default=0.985)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def find_file(directory: Path, stems: tuple[str, ...], extensions: set[str]) -> Path:
    for stem in stems:
        for extension in sorted(extensions):
            path = directory / f"{stem}{extension}"
            if path.exists():
                return path
    raise FileNotFoundError(f"No file for {stems} under {directory}")


def dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float32)
    frequencies = positions[:, None]
    matrix = np.cos(math.pi * (positions + 0.5) * frequencies / size)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix.astype(np.float32)


_DCT32 = dct_matrix(32)


def perceptual_hash(path: Path) -> int:
    """Return a deterministic 64-bit pHash using an 8x8 low-frequency DCT block."""
    image = Image.open(path).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    coefficients = _DCT32 @ pixels @ _DCT32.T
    low = coefficients[:8, :8].reshape(-1)
    threshold = float(np.median(low[1:]))
    bits = low > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


@torch.inference_mode()
def extract_embeddings(
    paths: list[Path], device: torch.device, batch_size: int, num_workers: int
) -> np.ndarray:
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    encoder = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
    dataset = ImagePathDataset(paths, weights.transforms())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    outputs = []
    for batch in tqdm(loader, desc="embeddings"):
        features = encoder(batch.to(device, non_blocking=True)).flatten(1)
        features = torch.nn.functional.normalize(features, dim=1)
        outputs.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def near_duplicate_groups(
    hashes: list[int], embeddings: np.ndarray, phash_threshold: int, embedding_threshold: float
) -> tuple[list[list[int]], int, int, np.ndarray]:
    count = len(hashes)
    sets = DisjointSet(count)
    phash_edges = 0
    for left in range(count):
        for right in range(left + 1, count):
            if (hashes[left] ^ hashes[right]).bit_count() <= phash_threshold:
                sets.union(left, right)
                phash_edges += 1

    similarities = embeddings @ embeddings.T
    rows, columns = np.where(np.triu(similarities, 1) >= embedding_threshold)
    for left, right in zip(rows.tolist(), columns.tolist()):
        sets.union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(sets.find(index), []).append(index)
    groups = sorted(grouped.values(), key=lambda values: (-len(values), values[0]))
    return groups, phash_edges, len(rows), similarities


def mask_histogram(data_root: Path, stem: str) -> np.ndarray:
    mask_path = find_file(data_root / "masks", (f"{stem}m", stem), MASK_EXTENSIONS)
    with Image.open(mask_path) as mask:
        labels = decode_mask(mask)
    return np.bincount(labels.reshape(-1), minlength=5)[[0, 1, 2]].astype(np.int64)


def mask_histogram_from_path(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as mask:
        labels = decode_mask(mask)
    return np.bincount(labels.reshape(-1), minlength=5)[[0, 1, 2]].astype(np.int64)


@dataclass
class SplitState:
    name: str
    target: float
    groups: list[list[int]]
    image_count: int = 0
    class_counts: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.class_counts is None:
            self.class_counts = np.zeros(3, dtype=np.int64)


def assign_groups(
    groups: list[list[int]], histograms: np.ndarray, ratios: tuple[float, float, float], seed: int
) -> dict[str, list[int]]:
    total_images = sum(len(group) for group in groups)
    total_classes = histograms.sum(axis=0).clip(min=1)
    states = [
        SplitState("train", ratios[0] * total_images, []),
        SplitState("val", ratios[1] * total_images, []),
        SplitState("test", ratios[2] * total_images, []),
    ]
    rng = random.Random(seed)
    ordered = [(rng.random(), group) for group in groups]
    ordered.sort(key=lambda item: (-len(item[1]), item[0]))

    for _, group in ordered:
        group_histogram = histograms[group].sum(axis=0)
        best_state = None
        best_score = None
        for state, ratio in zip(states, ratios):
            new_count = state.image_count + len(group)
            fill_ratio = new_count / max(state.target, 1.0)
            new_classes = state.class_counts + group_histogram
            desired_classes = total_classes * ratio
            class_error = float(np.mean(((new_classes - desired_classes) / desired_classes) ** 2))
            overflow = max(new_count - state.target, 0.0) / max(state.target, 1.0)
            # Fill each split in proportion to its target. Comparing squared
            # distance to the final target would incorrectly favor the smaller
            # validation/test bins at the beginning of the assignment.
            score = fill_ratio + 0.02 * class_error + 5.0 * overflow
            if best_score is None or score < best_score:
                best_score, best_state = score, state
        assert best_state is not None
        best_state.groups.append(group)
        best_state.image_count += len(group)
        best_state.class_counts += group_histogram

    return {
        state.name: sorted(index for group in state.groups for index in group) for state in states
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def dataset_manifest_sha256(
    stems: list[str], image_paths: list[Path], mask_paths: list[Path]
) -> str:
    digest = hashlib.sha256()
    for stem, image_path, mask_path in zip(stems, image_paths, mask_paths):
        digest.update(stem.encode("utf-8"))
        digest.update(b"\0image\0")
        digest.update(sha256_file(image_path).encode("ascii"))
        digest.update(b"\0mask\0")
        digest.update(sha256_file(mask_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_split(path: Path, stems: list[str]) -> None:
    path.write_text("\n".join(sorted(stems)) + "\n", encoding="utf-8")


def top_cross_split_pairs(
    similarities: np.ndarray, split_by_index: list[str], stems: list[str], limit: int = 20
) -> list[dict[str, object]]:
    candidates: list[tuple[float, int, int]] = []
    for left in range(len(stems)):
        for right in range(left + 1, len(stems)):
            if split_by_index[left] != split_by_index[right]:
                candidates.append((float(similarities[left, right]), left, right))
    candidates.sort(reverse=True)
    return [
        {
            "left": stems[left],
            "right": stems[right],
            "left_split": split_by_index[left],
            "right_split": split_by_index[right],
            "cosine_similarity": similarity,
        }
        for similarity, left, right in candidates[:limit]
    ]


def cross_split_audit(
    hashes: list[int],
    similarities: np.ndarray,
    split_by_index: list[str],
    phash_threshold: int,
    embedding_threshold: float,
) -> dict[str, int | float]:
    minimum_hamming = 64
    maximum_cosine = -1.0
    phash_violations = 0
    embedding_violations = 0
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if split_by_index[left] == split_by_index[right]:
                continue
            hamming = (hashes[left] ^ hashes[right]).bit_count()
            cosine = float(similarities[left, right])
            minimum_hamming = min(minimum_hamming, hamming)
            maximum_cosine = max(maximum_cosine, cosine)
            phash_violations += int(hamming <= phash_threshold)
            embedding_violations += int(cosine >= embedding_threshold)
    return {
        "cross_split_phash_violations": phash_violations,
        "minimum_cross_split_phash_distance": minimum_hamming,
        "cross_split_embedding_violations": embedding_violations,
        "maximum_cross_split_cosine": maximum_cosine,
    }


def main() -> None:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if min(args.train_ratio, args.val_ratio, test_ratio) <= 0:
        raise ValueError("train/val/test ratios must all be positive")

    stems = discover_image_stems(args.data_root / "images")
    paths = [find_file(args.data_root / "images", (stem,), IMAGE_EXTENSIONS) for stem in stems]
    mask_paths = [
        find_file(args.data_root / "masks", (f"{stem}m", stem), MASK_EXTENSIONS)
        for stem in stems
    ]
    if not paths:
        raise RuntimeError("No images found")

    device = resolve_device(args.device)
    print(f"images={len(paths)} device={device}")
    hashes = [perceptual_hash(path) for path in tqdm(paths, desc="pHash")]
    embeddings = extract_embeddings(paths, device, args.batch_size, args.num_workers)
    groups, phash_edges, embedding_edges, similarities = near_duplicate_groups(
        hashes, embeddings, args.phash_threshold, args.embedding_threshold
    )
    histograms = np.stack(
        [mask_histogram_from_path(path) for path in tqdm(mask_paths, desc="masks")]
    )
    assignments = assign_groups(
        groups, histograms, (args.train_ratio, args.val_ratio, test_ratio), args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_by_index = [""] * len(stems)
    split_reports: dict[str, object] = {}
    for split_name, indices in assignments.items():
        split_stems = [stems[index] for index in indices]
        path = args.output_dir / f"{split_name}.txt"
        write_split(path, split_stems)
        for index in indices:
            split_by_index[index] = split_name
        counts = histograms[indices].sum(axis=0)
        split_reports[split_name] = {
            "images": len(indices),
            "class_pixels": {
                "obstacle": int(counts[0]),
                "water": int(counts[1]),
                "sky": int(counts[2]),
            },
            "sha256": sha256_file(path),
        }

    group_split_violations = 0
    for group in groups:
        if len({split_by_index[index] for index in group}) != 1:
            group_split_violations += 1

    audit = cross_split_audit(
        hashes,
        similarities,
        split_by_index,
        args.phash_threshold,
        args.embedding_threshold,
    )
    weights = ResNet18_Weights.DEFAULT
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    report = {
        "protocol": "pre-registered grouped near-duplicate split v1",
        "command": [sys.executable, *sys.argv],
        "configuration": configuration,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "provenance": {
            "git_head": git_head(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "embedding_weights_url": weights.url,
            "embedding_weights_sha256": sha256_file(weights_path),
            "dataset_manifest_sha256": dataset_manifest_sha256(stems, paths, mask_paths),
        },
        "thresholds": {
            "phash_bits": 64,
            "phash_hamming_max": args.phash_threshold,
            "embedding_model": "torchvision ResNet18 ImageNet DEFAULT penultimate",
            "embedding_cosine_min": args.embedding_threshold,
        },
        "dataset": {
            "images": len(stems),
            "groups": len(groups),
            "multi_image_groups": sum(len(group) > 1 for group in groups),
            "largest_group": max(map(len, groups)),
            "phash_edges": phash_edges,
            "embedding_edges": embedding_edges,
        },
        "splits": split_reports,
        "group_split_violations": group_split_violations,
        **audit,
        "top_cross_split_embedding_pairs": top_cross_split_pairs(
            similarities, split_by_index, stems
        ),
        "groups": [[stems[index] for index in group] for group in groups],
    }
    report_path = args.output_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": report["dataset"], "splits": split_reports}, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
