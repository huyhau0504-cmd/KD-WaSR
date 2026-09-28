"""Streaming metrics for three-class maritime segmentation."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


CLASS_NAMES = ("obstacle", "water", "sky")
COMPONENT_BINS = (
    ("tiny", 1, 64),
    ("very_small", 65, 256),
    ("small", 257, 1024),
    ("medium_plus", 1025, None),
)


def binary_boundary(mask: Tensor, radius: int = 1) -> Tensor:
    """Return a morphological boundary map for a BxHxW binary tensor."""
    x = mask.float().unsqueeze(1)
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(x, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-x, kernel, stride=1, padding=radius)
    return (dilated - eroded > 0).squeeze(1)


def _binary_boundary_numpy(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """NumPy equivalent used by CPU-side evaluation metrics."""
    if radius == 0:
        return np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    outside_zero = np.pad(mask, radius, constant_values=False)
    outside_one = np.pad(mask, radius, constant_values=True)
    dilated = np.zeros_like(mask, dtype=bool)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dilated |= outside_zero[dy : dy + height, dx : dx + width]
            eroded &= outside_one[dy : dy + height, dx : dx + width]
    return dilated & ~eroded


def _binary_dilate_numpy(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate an already extracted binary boundary by a pixel tolerance."""
    if radius < 0:
        raise ValueError("dilation radius must be non-negative")
    if radius == 0:
        return mask.astype(bool, copy=True)
    height, width = mask.shape
    padded = np.pad(mask.astype(bool), radius, constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dilated |= padded[dy : dy + height, dx : dx + width]
    return dilated


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label a binary mask with 8-connectivity using row runs.

    Run-length union-find avoids a Python operation for every foreground pixel,
    which keeps validation practical without adding an OpenCV/SciPy dependency.
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = [0]

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: int, right: int) -> int:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
        return left_root

    # Each run is (row, inclusive_start, inclusive_end, provisional_label).
    all_runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []
    for row_index in range(height):
        row = np.asarray(mask[row_index], dtype=bool)
        padded = np.pad(row.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        current: list[tuple[int, int, int]] = []
        previous_start = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            while previous_start < len(previous) and previous[previous_start][1] < start - 1:
                previous_start += 1
            overlap_labels: list[int] = []
            scan = previous_start
            while scan < len(previous) and previous[scan][0] <= end + 1:
                overlap_labels.append(previous[scan][2])
                scan += 1
            if overlap_labels:
                run_label = find(overlap_labels[0])
                for other in overlap_labels[1:]:
                    run_label = union(run_label, other)
            else:
                run_label = len(parent)
                parent.append(run_label)
            current.append((start, end, run_label))
            all_runs.append((row_index, start, end, run_label))
        previous = current

    root_to_label: dict[int, int] = {}
    areas: list[int] = []
    for row_index, start, end, provisional in all_runs:
        root = find(provisional)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label) + 1
            areas.append(0)
        final_label = root_to_label[root]
        labels[row_index, start : end + 1] = final_label
        areas[final_label - 1] += end - start + 1
    return labels, np.asarray(areas, dtype=np.int64)


def _linear_sum_assignment_max(scores: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight rectangular Hungarian assignment without SciPy."""
    if scores.size == 0:
        return []
    cost = -np.asarray(scores, dtype=np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    rows, columns = cost.shape
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    p = np.zeros(columns + 1, dtype=np.int64)
    way = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        p[0] = row
        min_values = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = [(int(p[column] - 1), column - 1) for column in range(1, columns + 1) if p[column]]
    return [(column, row) for row, column in pairs] if transposed else pairs


def _component_bin(area: int) -> str:
    for name, minimum, maximum in COMPONENT_BINS:
        if area >= minimum and (maximum is None or area <= maximum):
            return name
    raise ValueError(f"Invalid component area: {area}")


class SegmentationMetrics:
    def __init__(self, num_classes: int = 3, ignore_index: int = 4) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.boundary_counts = {
            tolerance: {"tp_precision": 0, "predicted": 0, "tp_recall": 0, "target": 0}
            for tolerance in (1, 3, 5)
        }
        self.component_total = {name: 0 for name, _, _ in COMPONENT_BINS}
        self.component_recalled = {name: 0 for name, _, _ in COMPONENT_BINS}
        self.false_positive_components = 0
        self.image_count = 0
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

        for pred_item, target_item, valid_item in zip(prediction, target, valid):
            pred_obstacle = (pred_item == 0).numpy()
            target_obstacle = (target_item == 0).numpy()
            valid_array = valid_item.numpy()
            pred_boundary = _binary_boundary_numpy(pred_obstacle) & valid_array
            target_boundary = _binary_boundary_numpy(target_obstacle) & valid_array
            for tolerance, counts in self.boundary_counts.items():
                pred_near_target = _binary_dilate_numpy(target_boundary, tolerance) & valid_array
                target_near_pred = _binary_dilate_numpy(pred_boundary, tolerance) & valid_array
                counts["tp_precision"] += int((pred_boundary & pred_near_target).sum())
                counts["predicted"] += int(pred_boundary.sum())
                counts["tp_recall"] += int((target_boundary & target_near_pred).sum())
                counts["target"] += int(target_boundary.sum())
        self._update_components(prediction, target, valid)
        self._update_water_edge(prediction, target)

    def _update_components(self, prediction: Tensor, target: Tensor, valid: Tensor) -> None:
        for pred_item, target_item, valid_item in zip(prediction, target, valid):
            pred_mask = ((pred_item == 0) & valid_item).numpy()
            target_mask = ((target_item == 0) & valid_item).numpy()
            pred_labels, pred_areas = _connected_components(pred_mask)
            target_labels, target_areas = _connected_components(target_mask)

            for area in target_areas:
                self.component_total[_component_bin(int(area))] += 1

            accepted_pred: set[int] = set()
            if len(pred_areas) and len(target_areas):
                combined = target_labels.ravel() * (len(pred_areas) + 1) + pred_labels.ravel()
                intersections = np.bincount(
                    combined,
                    minlength=(len(target_areas) + 1) * (len(pred_areas) + 1),
                ).reshape(len(target_areas) + 1, len(pred_areas) + 1)[1:, 1:]
                unions = target_areas[:, None] + pred_areas[None, :] - intersections
                iou = intersections / np.maximum(unions, 1)
                for gt_index, pred_index in _linear_sum_assignment_max(iou):
                    coverage = intersections[gt_index, pred_index] / max(target_areas[gt_index], 1)
                    if coverage >= 0.5:
                        name = _component_bin(int(target_areas[gt_index]))
                        self.component_recalled[name] += 1
                        accepted_pred.add(pred_index)

            self.false_positive_components += sum(
                int(area >= 16) for index, area in enumerate(pred_areas) if index not in accepted_pred
            )
            self.image_count += 1

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

        non_obstacle = matrix[1:, :].sum()
        obstacle_pixel_fpr = matrix[1:, 0].sum() / non_obstacle.clamp_min(1)

        boundary_results: Dict[int, float] = {}
        for tolerance, counts in self.boundary_counts.items():
            boundary_precision = counts["tp_precision"] / max(counts["predicted"], 1)
            boundary_recall = counts["tp_recall"] / max(counts["target"], 1)
            boundary_results[tolerance] = (
                2
                * boundary_precision
                * boundary_recall
                / max(boundary_precision + boundary_recall, 1e-12)
            )

        small_total = sum(self.component_total[name] for name in ("tiny", "very_small", "small"))
        small_recalled = sum(
            self.component_recalled[name] for name in ("tiny", "very_small", "small")
        )
        small_component_recall = small_recalled / max(small_total, 1)

        result: Dict[str, float] = {
            "pixel_accuracy": float(accuracy),
            "mean_iou": float(iou.mean()),
            "obstacle_precision": float(precision[0]),
            "obstacle_recall": float(recall[0]),
            "obstacle_f1": float(f1[0]),
            "obstacle_pixel_fpr": float(obstacle_pixel_fpr),
            "boundary_f1": float(boundary_results[3]),
            "boundary_f1_tol1": float(boundary_results[1]),
            "boundary_f1_tol3": float(boundary_results[3]),
            "boundary_f1_tol5": float(boundary_results[5]),
            "small_component_recall": float(small_component_recall),
            "false_positive_components_per_100_images": (
                100.0 * self.false_positive_components / max(self.image_count, 1)
            ),
            "water_edge_mae_px": self.edge_abs_sum / max(self.edge_count, 1),
            "water_edge_rmse_px": math.sqrt(self.edge_sq_sum / max(self.edge_count, 1)),
        }
        for name, _, _ in COMPONENT_BINS:
            result[f"component_recall_{name}"] = self.component_recalled[name] / max(
                self.component_total[name], 1
            )
            result[f"component_count_{name}"] = float(self.component_total[name])
        result["safety_score"] = (
            0.40 * result["obstacle_f1"]
            + 0.30 * result["small_component_recall"]
            + 0.20 * result["boundary_f1_tol3"]
            + 0.10 * (1.0 - result["obstacle_pixel_fpr"])
        )
        for index, name in enumerate(CLASS_NAMES[: self.num_classes]):
            result[f"iou_{name}"] = float(iou[index])
        return result
