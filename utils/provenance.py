"""Reproducibility metadata for training artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision

from datasets.mastr import IMAGE_EXTENSIONS, MASK_EXTENSIONS, discover_image_stems


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _find_file(directory: Path, stems: tuple[str, ...], extensions: set[str]) -> Path:
    for stem in stems:
        for extension in sorted(extensions):
            candidate = directory / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No file for {stems} under {directory}")


def dataset_manifest_sha256(data_root: str | Path) -> str:
    root = Path(data_root)
    stems = discover_image_stems(root / "images")
    digest = hashlib.sha256()
    for stem in stems:
        image_path = _find_file(root / "images", (stem,), IMAGE_EXTENSIONS)
        mask_path = _find_file(root / "masks", (f"{stem}m", stem), MASK_EXTENSIONS)
        digest.update(stem.encode("utf-8"))
        digest.update(b"\0image\0")
        digest.update(sha256_file(image_path).encode("ascii"))
        digest.update(b"\0mask\0")
        digest.update(sha256_file(mask_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_provenance() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], capture_output=True, check=False
    )
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": dirty,
        "status_porcelain": status.stdout.splitlines() if status.returncode == 0 else None,
        "tracked_diff_sha256": (
            hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
        ),
    }


def environment_provenance(device: torch.device) -> dict:
    cuda_device = None
    if device.type == "cuda":
        cuda_device = torch.cuda.get_device_name(device)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "cuda_device": cuda_device,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def training_provenance(
    data_root: str | Path,
    split_dir: str | Path,
    device: torch.device,
) -> dict:
    split_dir = Path(split_dir)
    split_checksums = {
        name: sha256_file(split_dir / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    report_path = split_dir / "split_report.json"
    manifest = dataset_manifest_sha256(data_root)
    registration = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_splits = {
            name: report["splits"][name]["sha256"] for name in split_checksums
        }
        if split_checksums != expected_splits:
            raise RuntimeError(
                f"Current split checksums do not match split_report.json: "
                f"current={split_checksums} expected={expected_splits}"
            )
        expected_manifest = report["provenance"]["dataset_manifest_sha256"]
        if manifest != expected_manifest:
            raise RuntimeError(
                f"Current dataset manifest does not match split_report.json: "
                f"current={manifest} expected={expected_manifest}"
            )
        registration = {
            "split_checksums_match": True,
            "dataset_manifest_matches": True,
        }
    return {
        "command": [sys.executable, *sys.argv],
        "git": git_provenance(),
        "environment": environment_provenance(device),
        "split_sha256": split_checksums,
        "split_report_sha256": sha256_file(report_path) if report_path.exists() else None,
        "dataset_manifest_sha256": manifest,
        "registration_check": registration,
    }
