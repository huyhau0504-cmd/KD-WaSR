"""Audit grouped-split balance beyond the existing leakage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.mastr import MASK_EXTENSIONS, decode_mask, discover_image_stems, read_split
from utils import dataset_manifest_sha256, save_json, sha256_file
from utils.metrics import COMPONENT_BINS, _component_bin, _connected_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/MaSTr1325"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/grouped_splits"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def mask_path(mask_dir: Path, stem: str) -> Path:
    for candidate_stem in (f"{stem}m", stem):
        for suffix in MASK_EXTENSIONS:
            candidate = mask_dir / f"{candidate_stem}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No mask found for {stem}")


def main() -> None:
    args = parse_args()
    output = args.output or args.split_dir / "balance_report.json"
    source_report_path = args.split_dir / "split_report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    split_stems = {
        name: read_split(args.split_dir / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    split_checksums = {
        name: sha256_file(args.split_dir / f"{name}.txt") for name in split_stems
    }
    for name, stems in split_stems.items():
        if len(stems) != len(set(stems)):
            raise RuntimeError(f"Duplicate stems within {name}.txt")
        expected_checksum = source_report["splits"][name]["sha256"]
        if split_checksums[name] != expected_checksum:
            raise RuntimeError(
                f"Stale split report: {name}.txt checksum is {split_checksums[name]}, "
                f"expected {expected_checksum}"
            )

    split_sets = {name: set(stems) for name, stems in split_stems.items()}
    names = tuple(split_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise RuntimeError(
                    f"Split overlap between {left} and {right}: {sorted(overlap)[:10]}"
                )
    dataset_stems = set(discover_image_stems(args.data_root / "images"))
    split_union = set().union(*split_sets.values())
    missing = dataset_stems - split_union
    extra = split_union - dataset_stems
    if missing or extra:
        raise RuntimeError(
            f"Split completeness failure: missing={sorted(missing)[:10]} "
            f"extra={sorted(extra)[:10]}"
        )
    current_manifest = dataset_manifest_sha256(args.data_root)
    expected_manifest = source_report["provenance"]["dataset_manifest_sha256"]
    if current_manifest != expected_manifest:
        raise RuntimeError(
            f"Dataset manifest mismatch: current={current_manifest} expected={expected_manifest}"
        )

    owner = {stem: name for name, stems in split_stems.items() for stem in stems}
    groups = source_report.get("groups", [])
    group_counts = {name: 0 for name in split_stems}
    group_sizes = {name: [] for name in split_stems}
    for group in groups:
        owners = {owner[stem] for stem in group if stem in owner}
        if len(owners) != 1:
            raise RuntimeError(f"Grouped split violation during balance audit: {group[:5]}")
        split = next(iter(owners))
        group_counts[split] += 1
        group_sizes[split].append(len(group))

    split_results = {}
    for name, stems in split_stems.items():
        class_pixels = np.zeros(3, dtype=np.int64)
        component_counts = {bin_name: 0 for bin_name, _, _ in COMPONENT_BINS}
        image_sizes: dict[str, int] = {}
        for stem in tqdm(stems, desc=f"audit:{name}"):
            with Image.open(mask_path(args.data_root / "masks", stem)) as opened:
                decoded = decode_mask(opened)
            for class_index in range(3):
                class_pixels[class_index] += int((decoded == class_index).sum())
            _, areas = _connected_components(decoded == 0)
            for area in areas:
                component_counts[_component_bin(int(area))] += 1
            size_key = f"{decoded.shape[1]}x{decoded.shape[0]}"
            image_sizes[size_key] = image_sizes.get(size_key, 0) + 1

        valid_pixels = int(class_pixels.sum())
        split_results[name] = {
            "images": len(stems),
            "groups": group_counts[name],
            "group_size": {
                "minimum": min(group_sizes[name]),
                "maximum": max(group_sizes[name]),
                "mean": float(np.mean(group_sizes[name])),
            },
            "class_pixels": {
                "obstacle": int(class_pixels[0]),
                "water": int(class_pixels[1]),
                "sky": int(class_pixels[2]),
            },
            "class_ratio": {
                "obstacle": float(class_pixels[0] / max(valid_pixels, 1)),
                "water": float(class_pixels[1] / max(valid_pixels, 1)),
                "sky": float(class_pixels[2] / max(valid_pixels, 1)),
            },
            "component_count": component_counts,
            "image_sizes": image_sizes,
        }

    largest_group = max(groups, key=len)
    largest_owners = {owner[stem] for stem in largest_group if stem in owner}
    report = {
        "source_report": str(source_report_path),
        "source_protocol": source_report.get("protocol"),
        "verified_current_state": {
            "split_sha256": split_checksums,
            "dataset_manifest_sha256": current_manifest,
            "dataset_images": len(dataset_stems),
            "split_union_images": len(split_union),
            "splits_disjoint": True,
            "splits_complete": True,
        },
        "leakage_audit": {
            "group_split_violations": source_report.get("group_split_violations"),
            "cross_split_phash_violations": source_report.get(
                "cross_split_phash_violations"
            ),
            "cross_split_embedding_violations": source_report.get(
                "cross_split_embedding_violations"
            ),
        },
        "splits": split_results,
        "largest_group": {
            "size": len(largest_group),
            "split": next(iter(largest_owners)),
            "members": largest_group,
        },
    }
    save_json(output, report)
    print(f"Saved grouped-split balance audit to {output}")


if __name__ == "__main__":
    main()
