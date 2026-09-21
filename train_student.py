"""Train WaSR/eWaSR baselines and safety-aware KD variants."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MaSTr1325Dataset, split_paths
from models import build_model, load_checkpoint_model
from utils import (
    LossWeights,
    MaritimeObjective,
    SegmentationMetrics,
    move_features,
    resolve_device,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/MaSTr1325"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ewasr_fp32"))
    parser.add_argument(
        "--model", choices=("ewasr_resnet18", "wasr_resnet101"), default="ewasr_resnet18"
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--sparse-obstacle-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def make_loaders(args: argparse.Namespace, device: torch.device):
    paths = split_paths(args.split_dir)
    size = (args.height, args.width)
    train_dataset = MaSTr1325Dataset(args.data_root, paths["train"], train=True, size=size)
    val_dataset = MaSTr1325Dataset(args.data_root, paths["val"], train=False, size=size)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader


def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for features, target in tqdm(loader, desc="validate", leave=False):
            features = move_features(features, device)
            mask = target["mask"].to(device, non_blocking=True)
            logits = model(features)["out"]
            metrics.update(logits, mask)
    return metrics.compute()


def serializable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> None:
    args = parse_args()
    if args.kd_weight > 0 and args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required when --kd-weight is positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "config.json", serializable_args(args))

    train_loader, val_loader = make_loaders(args, device)
    model = build_model(
        args.model,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    teacher = None
    if args.teacher_checkpoint:
        teacher = load_checkpoint_model(str(args.teacher_checkpoint), device=device)
        teacher.eval()
        teacher.requires_grad_(False)

    objective = MaritimeObjective(
        LossWeights(
            kd=args.kd_weight,
            boundary=args.boundary_weight,
            sparse_obstacle=args.sparse_obstacle_weight,
        ),
        temperature=args.temperature,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_f1 = -1.0
    print(
        f"device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        start = time.perf_counter()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for features, target in progress:
            features = move_features(features, device)
            mask = target["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                student_logits = model(features)["out"]
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(features)["out"]
                loss, terms = objective(student_logits, mask, teacher_logits)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        scheduler.step()
        val_metrics = validate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start,
            **val_metrics,
        }
        history.append(record)
        save_json(args.output_dir / "history.json", history)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": val_metrics,
            "args": serializable_args(args),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["obstacle_f1"] > best_f1:
            best_f1 = val_metrics["obstacle_f1"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        print(
            f"epoch={epoch} loss={record['train_loss']:.4f} "
            f"mIoU={val_metrics['mean_iou']:.4f} obstacle_F1={val_metrics['obstacle_f1']:.4f} "
            f"boundary_F1={val_metrics['boundary_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
