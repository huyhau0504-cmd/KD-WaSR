"""Supervised and safety-aware knowledge-distillation losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .metrics import binary_boundary


def sparse_obstacle_loss(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = 4,
    maximum_weight: float = 8.0,
) -> Tensor:
    """Upweight obstacle pixels when they occupy a small fraction of an image."""
    per_pixel = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    valid = target != ignore_index
    obstacle = target == 0
    valid_count = valid.flatten(1).sum(dim=1).clamp_min(1).float()
    obstacle_count = obstacle.flatten(1).sum(dim=1).clamp_min(1).float()
    weights = torch.sqrt(valid_count / obstacle_count).clamp(max=maximum_weight)
    weight_map = torch.ones_like(per_pixel)
    weight_map = torch.where(obstacle, weights[:, None, None], weight_map)
    return (per_pixel * weight_map * valid).sum() / valid.sum().clamp_min(1)


def boundary_band_cross_entropy(
    logits: Tensor,
    target: Tensor,
    alpha: float = 1.0,
    radius: int = 3,
    ignore_index: int = 4,
) -> Tensor:
    """Cross entropy with extra weight in a GT obstacle-boundary band.

    Unlike the previous boundary BCE, this objective keeps the original class
    target everywhere.  Consequently, obstacle interiors are still trained as
    obstacle instead of being incorrectly trained as non-boundary pixels.
    """
    if alpha < 0:
        raise ValueError("boundary alpha must be non-negative")
    if radius < 0:
        raise ValueError("boundary radius must be non-negative")

    valid = target != ignore_index
    per_pixel = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    band = binary_boundary((target == 0) & valid, radius=radius) & valid
    weights = 1.0 + alpha * band.to(per_pixel.dtype)
    valid_weights = weights * valid.to(per_pixel.dtype)
    return (per_pixel * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)


def logit_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target: Tensor,
    temperature: float = 4.0,
    ignore_index: int = 4,
) -> Tensor:
    valid = (target != ignore_index).float()
    student_log_probability = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probability = F.softmax(teacher_logits.detach() / temperature, dim=1)
    per_pixel = F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="none",
    ).sum(dim=1)
    return temperature**2 * (per_pixel * valid).sum() / valid.sum().clamp_min(1)


@dataclass
class LossWeights:
    kd: float = 0.0
    boundary: float = 0.0
    sparse_obstacle: float = 0.0


class MaritimeObjective(nn.Module):
    def __init__(
        self,
        weights: LossWeights,
        temperature: float = 4.0,
        ignore_index: int = 4,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: Tensor,
        target: Tensor,
        teacher_logits: Optional[Tensor] = None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        if self.weights.boundary > 0:
            segmentation = boundary_band_cross_entropy(
                student_logits,
                target,
                alpha=self.weights.boundary,
                radius=3,
                ignore_index=self.ignore_index,
            )
        else:
            segmentation = F.cross_entropy(
                student_logits, target, ignore_index=self.ignore_index
            )
        terms: Dict[str, Tensor] = {"segmentation": segmentation}
        total = terms["segmentation"]

        if self.weights.kd > 0:
            if teacher_logits is None:
                raise ValueError("teacher_logits are required when kd weight is positive")
            terms["kd"] = logit_distillation_loss(
                student_logits,
                teacher_logits,
                target,
                temperature=self.temperature,
                ignore_index=self.ignore_index,
            )
            total = total + self.weights.kd * terms["kd"]
        if self.weights.boundary > 0:
            # Kept as a named term for logging/backward compatibility. It is
            # the segmentation objective itself and must not be added twice.
            terms["boundary_band_ce"] = segmentation
        if self.weights.sparse_obstacle > 0:
            terms["sparse_obstacle"] = sparse_obstacle_loss(
                student_logits, target, self.ignore_index
            )
            total = total + self.weights.sparse_obstacle * terms["sparse_obstacle"]
        terms["total"] = total
        return total, terms
