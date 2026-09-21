"""Streaming metrics for three-class maritime segmentation."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F


CLASS_NAMES = ("obstacle", "water", "sky")


def binary_boundary(mask: Tensor, radius: int = 1) -> Tensor:
    """Return a morphological boundary map for a BxHxW binary tensor."""
    x = mask.float().unsqueeze(1)
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(x, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-x, kernel, stride=1, padding=radius)
    return (dilated - eroded > 0).squeeze(1)


class SegmentationMetrics:
    def __init__(self, num_classes: int = 3, ignore_index: int = 4) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.boundary_tp = 0
        self.boundary_fp = 0
        self.boundary_fn = 0
        self.edge_abs_sum = 0.0
        self.edge_sq_sum = 0.0
        self.edge_count = 0

    @torch.no_grad()
    def update(self, logits_or_prediction: Tensor, target: Tensor) -> None:
        prediction = (
            logits_or_prediction.argmax(dim=1)
            if logits_or_prediction.ndim == 4
            else logits_or_prediction
        )
        prediction = prediction.detach().cpu().long()
        target = target.detach().cpu().long()
        valid = target != self.ignore_index
        encoded = target[valid] * self.num_classes + prediction[valid]
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

        pred_boundary = binary_boundary(prediction == 0)
        target_boundary = binary_boundary(target == 0)
        valid_boundary = valid
        self.boundary_tp += int((pred_boundary & target_boundary & valid_boundary).sum())
        self.boundary_fp += int((pred_boundary & ~target_boundary & valid_boundary).sum())
        self.boundary_fn += int((~pred_boundary & target_boundary & valid_boundary).sum())
        self._update_water_edge(prediction, target)

    def _update_water_edge(self, prediction: Tensor, target: Tensor) -> None:
        # Internal diagnostic, not a replacement for the official MODS water-edge metric.
        for pred_item, target_item in zip(prediction, target):
            height, width = target_item.shape
            rows = torch.arange(height).view(height, 1).expand(height, width)
            missing = torch.full_like(rows, height)
            pred_rows = torch.where(pred_item == 1, rows, missing).amin(dim=0)
            target_rows = torch.where(target_item == 1, rows, missing).amin(dim=0)
            valid = (pred_rows < height) & (target_rows < height)
            if valid.any():
                difference = (pred_rows[valid] - target_rows[valid]).float()
                self.edge_abs_sum += float(difference.abs().sum())
                self.edge_sq_sum += float(difference.square().sum())
                self.edge_count += int(valid.sum())

    def compute(self) -> Dict[str, float]:
        matrix = self.confusion.float()
        true_positive = matrix.diag()
        false_positive = matrix.sum(dim=0) - true_positive
        false_negative = matrix.sum(dim=1) - true_positive
        union = true_positive + false_positive + false_negative
        iou = true_positive / union.clamp_min(1)
        precision = true_positive / (true_positive + false_positive).clamp_min(1)
        recall = true_positive / (true_positive + false_negative).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        accuracy = true_positive.sum() / matrix.sum().clamp_min(1)

        boundary_precision = self.boundary_tp / max(self.boundary_tp + self.boundary_fp, 1)
        boundary_recall = self.boundary_tp / max(self.boundary_tp + self.boundary_fn, 1)
        boundary_f1 = (
            2 * boundary_precision * boundary_recall / max(boundary_precision + boundary_recall, 1e-12)
        )

        result: Dict[str, float] = {
            "pixel_accuracy": float(accuracy),
            "mean_iou": float(iou.mean()),
            "obstacle_precision": float(precision[0]),
            "obstacle_recall": float(recall[0]),
            "obstacle_f1": float(f1[0]),
            "boundary_f1": float(boundary_f1),
            "water_edge_mae_px": self.edge_abs_sum / max(self.edge_count, 1),
            "water_edge_rmse_px": math.sqrt(self.edge_sq_sum / max(self.edge_count, 1)),
        }
        for index, name in enumerate(CLASS_NAMES[: self.num_classes]):
            result[f"iou_{name}"] = float(iou[index])
        return result
