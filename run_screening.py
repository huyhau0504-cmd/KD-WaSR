"""Run the pre-registered validation-only R1/R2 screening matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch

from utils import save_json


RUNS = {
    "R1": {},
    "R1-B": {"boundary_weight": 1.0},
    "R1-O": {"sparse_obstacle_weight": 0.25},
    "R2": {
        "boundary_weight": 1.0,
        "sparse_obstacle_weight": 0.25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("data/grouped_splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/screening"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def best_validation(checkpoint_path: Path) -> dict:
    """Read the exact checkpoint selected by train_student's min-delta rule."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best = {"epoch": checkpoint["epoch"], **checkpoint["metrics"]}
    keys = (
        "epoch",
        "safety_score",
        "mean_iou",
        "obstacle_f1",
        "small_component_recall",
        "boundary_f1_tol3",
        "false_positive_components_per_100_images",
    )
    return {key: best[key] for key in keys}


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "classification": "exploratory validation-only screening",
        "test_evaluation_allowed": False,
        "shared": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "seed": args.seed,
            "augmentation_profile": "domain",
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 1e-4,
        },
        "runs": RUNS,
    }
    save_json(args.output_dir / "screening_protocol.json", protocol)

    summary = {}
    for name, overrides in RUNS.items():
        run_dir = args.output_dir / name
        command = [
            sys.executable,
            "train_student.py",
            "--data-root",
            str(args.data_root),
            "--split-dir",
            str(args.split_dir),
            "--output-dir",
            str(run_dir),
            "--experiment-id",
            name,
            "--model",
            "ewasr_resnet18",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            "1e-4",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
            "--augmentation-profile",
            "domain",
            "--early-stopping-patience",
            "10",
            "--early-stopping-min-delta",
            "1e-4",
        ]
        if "boundary_weight" in overrides:
            command.extend(["--boundary-weight", str(overrides["boundary_weight"])])
        if "sparse_obstacle_weight" in overrides:
            command.extend(
                ["--sparse-obstacle-weight", str(overrides["sparse_obstacle_weight"])]
            )
        print(f"\n=== {name} ===\n{' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
        summary[name] = best_validation(run_dir / "best.pt")
        save_json(args.output_dir / "screening_summary.json", summary)

    winner = max(summary, key=lambda name: summary[name]["safety_score"])
    save_json(
        args.output_dir / "screening_summary.json",
        {"classification": "exploratory", "winner_by_validation": winner, "runs": summary},
    )
    print(f"Screening winner by validation safety_score: {winner}")


if __name__ == "__main__":
    main()
