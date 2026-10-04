from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.mastr import JointTransform, MaSTr1325Dataset, augmentation_spec, decode_mask
from models import build_model
from utils.losses import LossWeights, MaritimeObjective, boundary_band_cross_entropy
from utils.metrics import (
    SegmentationMetrics,
    _binary_boundary_numpy,
    _binary_dilate_numpy,
    _linear_sum_assignment_max,
    binary_boundary,
)
from prepare_grouped_splits import assign_groups, cross_split_audit, near_duplicate_groups
from predict_official_ewasr import logits_to_mask, preprocess as preprocess_official
from utils import SAFETY_SCORE_SPEC, seed_everything


class BaselineSmokeTest(unittest.TestCase):
    def test_domain_augmentation_is_locked_and_reproducible(self):
        spec = augmentation_spec("domain")
        self.assertEqual(spec["gamma"]["range"], [0.75, 1.35])
        self.assertEqual(spec["jpeg"]["quality"], [55, 95])
        image = Image.fromarray(np.full((64, 96, 3), 128, dtype=np.uint8))
        mask = Image.fromarray(np.ones((64, 96), dtype=np.uint8))
        transform = JointTransform(
            size=(64, 96), train=True, augmentation_profile="domain"
        )
        seed_everything(123)
        first_image, first_mask, _ = transform(image, mask)
        seed_everything(123)
        second_image, second_mask, _ = transform(image, mask)
        self.assertTrue(torch.equal(first_image, second_image))
        self.assertTrue(torch.equal(first_mask, second_mask))

    def test_safety_score_protocol_is_locked(self):
        self.assertEqual(SAFETY_SCORE_SPEC["direction"], "maximize")
        self.assertAlmostEqual(sum(SAFETY_SCORE_SPEC["terms"].values()), 1.0)

    def test_official_ewasr_preprocess_and_logit_resize(self):
        image = Image.fromarray(np.full((640, 640, 3), 128, dtype=np.uint8))
        tensor = preprocess_official(image)
        self.assertEqual(tensor.shape, (1, 3, 384, 512))
        self.assertEqual(tensor.dtype, np.float32)

        logits = np.zeros((1, 3, 96, 128), dtype=np.float32)
        logits[:, 1] = 1.0
        mask = logits_to_mask(logits, image.size)
        self.assertEqual(mask.shape, (640, 640))
        self.assertTrue(np.all(mask == 1))

    def test_mask_decoding(self):
        grayscale = Image.fromarray(np.array([[0, 1], [2, 4]], dtype=np.uint8))
        np.testing.assert_array_equal(decode_mask(grayscale), np.array([[0, 1], [2, 4]]))

        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0, 0] = 255
        rgb[0, 1, 1] = 255
        rgb[1, 0, 2] = 255
        np.testing.assert_array_equal(decode_mask(Image.fromarray(rgb)), [[0, 1], [2, 4]])

    def test_dataset_model_loss_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            for index in range(2):
                image = np.zeros((64, 96, 3), dtype=np.uint8)
                image[:32, :, 2] = 180
                image[32:, :, 1] = 130
                mask = np.ones((64, 96), dtype=np.uint8)
                mask[:28] = 2
                mask[40:50, 40 + index : 50 + index] = 0
                Image.fromarray(image).save(root / "images" / f"{index:04d}.jpg")
                Image.fromarray(mask).save(root / "masks" / f"{index:04d}m.png")

            dataset = MaSTr1325Dataset(root, train=False, size=(64, 96))
            features = torch.stack([dataset[0][0]["image"], dataset[1][0]["image"]])
            target = torch.stack([dataset[0][1]["mask"], dataset[1][1]["mask"]])

            model = build_model("ewasr_resnet18", pretrained_backbone=False).eval()
            with torch.inference_mode():
                logits = model(features)["out"]
            self.assertEqual(tuple(logits.shape), (2, 3, 64, 96))

            trainable_logits = logits.detach().clone().requires_grad_(True)
            objective = MaritimeObjective(
                LossWeights(boundary=0.1, sparse_obstacle=0.1)
            )
            loss, terms = objective(trainable_logits, target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("boundary_band_ce", terms)

            metrics = SegmentationMetrics()
            metrics.update(logits, target)
            result = metrics.compute()
            self.assertIn("obstacle_f1", result)
            self.assertTrue(0.0 <= result["mean_iou"] <= 1.0)

    def test_ewasr_training_forward_supports_batch_size_one(self):
        model = build_model("ewasr_resnet18", pretrained_backbone=False).train()
        image = torch.randn(1, 3, 64, 96)
        logits = model(image)["out"]
        self.assertEqual(tuple(logits.shape), (1, 3, 64, 96))
        logits.mean().backward()

    def test_grouped_split_keeps_near_duplicates_together(self):
        hashes = [0b0, 0b1, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFE]
        embeddings = np.eye(4, dtype=np.float32)
        groups, _, _, _ = near_duplicate_groups(hashes, embeddings, 1, 0.999)
        self.assertEqual(sorted(map(len, groups)), [2, 2])

        histograms = np.array(
            [[10, 20, 30], [10, 20, 30], [30, 20, 10], [30, 20, 10]],
            dtype=np.int64,
        )
        assignments = assign_groups(groups, histograms, (0.5, 0.25, 0.25), seed=42)
        owner = {index: split for split, indices in assignments.items() for index in indices}
        for group in groups:
            self.assertEqual(len({owner[index] for index in group}), 1)

    def test_grouped_split_tracks_requested_ratios(self):
        groups = [[index] for index in range(100)]
        histograms = np.tile(np.array([[10, 20, 30]], dtype=np.int64), (100, 1))
        assignments = assign_groups(groups, histograms, (0.7, 0.15, 0.15), seed=42)
        self.assertLessEqual(abs(len(assignments["train"]) - 70), 1)
        self.assertLessEqual(abs(len(assignments["val"]) - 15), 1)
        self.assertLessEqual(abs(len(assignments["test"]) - 15), 1)

    def test_near_duplicate_grouping_is_transitive(self):
        hashes = [0b000, 0b001, 0b011]
        embeddings = np.eye(3, dtype=np.float32)
        groups, _, _, _ = near_duplicate_groups(hashes, embeddings, 1, 1.1)
        self.assertEqual(groups, [[0, 1, 2]])

    def test_embedding_edge_groups_images_when_phash_does_not(self):
        hashes = [0, 0xFFFFFFFFFFFFFFFF]
        embeddings = np.array([[1.0, 0.0], [0.999, 0.02]], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        groups, _, embedding_edges, _ = near_duplicate_groups(hashes, embeddings, 0, 0.99)
        self.assertEqual(groups, [[0, 1]])
        self.assertEqual(embedding_edges, 1)

    def test_cross_split_audit_reports_no_grouping_edge(self):
        hashes = [0, 1, 0xFF]
        similarities = np.array(
            [[1.0, 0.99, 0.20], [0.99, 1.0, 0.30], [0.20, 0.30, 1.0]],
            dtype=np.float32,
        )
        audit = cross_split_audit(hashes, similarities, ["train", "train", "test"], 1, 0.95)
        self.assertEqual(audit["cross_split_phash_violations"], 0)
        self.assertEqual(audit["cross_split_embedding_violations"], 0)
        self.assertEqual(audit["minimum_cross_split_phash_distance"], 7)

    def test_boundary_band_ce_preserves_obstacle_interior(self):
        target = torch.ones((1, 17, 17), dtype=torch.long)
        target[:, 4:13, 4:13] = 0
        correct = torch.full((1, 3, 17, 17), -5.0)
        correct.scatter_(1, target.unsqueeze(1), 5.0)

        boundary_only_labels = torch.ones_like(target)
        obstacle_boundary = (
            torch.nn.functional.max_pool2d(
                (target == 0).float().unsqueeze(1), 3, stride=1, padding=1
            )
            - (-torch.nn.functional.max_pool2d(
                -(target == 0).float().unsqueeze(1), 3, stride=1, padding=1
            ))
            > 0
        ).squeeze(1) & (target == 0)
        boundary_only_labels[obstacle_boundary] = 0
        boundary_only = torch.full_like(correct, -5.0)
        boundary_only.scatter_(1, boundary_only_labels.unsqueeze(1), 5.0)

        correct_loss = boundary_band_cross_entropy(correct, target, alpha=2.0)
        boundary_only_loss = boundary_band_cross_entropy(boundary_only, target, alpha=2.0)
        self.assertLess(float(correct_loss), float(boundary_only_loss))

    def test_boundary_band_error_is_weighted_and_ignore_has_zero_gradient(self):
        target = torch.ones((1, 17, 17), dtype=torch.long)
        target[:, 3:14, 3:14] = 0
        logits = torch.full((1, 3, 17, 17), -4.0)
        logits.scatter_(1, target.unsqueeze(1), 4.0)

        boundary_error = logits.clone()
        boundary_error[:, 0, 3, 8] = -4.0
        boundary_error[:, 1, 3, 8] = 4.0
        interior_error = logits.clone()
        interior_error[:, 0, 8, 8] = -4.0
        interior_error[:, 1, 8, 8] = 4.0
        self.assertGreater(
            float(boundary_band_cross_entropy(boundary_error, target, alpha=4.0)),
            float(boundary_band_cross_entropy(interior_error, target, alpha=4.0)),
        )

        ignored_target = target.clone()
        ignored_target[:, 0, 0] = 4
        differentiable = logits.clone().requires_grad_(True)
        boundary_band_cross_entropy(differentiable, ignored_target, alpha=2.0).backward()
        self.assertEqual(float(differentiable.grad[:, :, 0, 0].abs().sum()), 0.0)

    def test_boundary_tolerance_responds_to_pixel_shifts(self):
        # A straight boundary far from image edges makes the tolerance semantics
        # exact: translations <=3 px must be fully accepted at tolerance 3.
        target = torch.ones((1, 32, 40), dtype=torch.long)
        target[:, :, :16] = 0

        def shifted_metrics(offset: int) -> dict[str, float]:
            prediction = torch.ones_like(target)
            prediction[:, :, : 16 + offset] = 0
            metrics = SegmentationMetrics()
            metrics.update(prediction, target)
            return metrics.compute()

        shift1 = shifted_metrics(1)
        shift2 = shifted_metrics(2)
        shift3 = shifted_metrics(3)
        shift4 = shifted_metrics(4)
        self.assertAlmostEqual(shift1["boundary_f1_tol3"], 1.0)
        self.assertAlmostEqual(shift2["boundary_f1_tol3"], 1.0)
        self.assertAlmostEqual(shift3["boundary_f1_tol3"], 1.0)
        self.assertLess(shift4["boundary_f1_tol3"], 1.0)
        self.assertGreaterEqual(shift3["boundary_f1_tol5"], shift3["boundary_f1_tol3"])

    def test_numpy_boundary_matches_training_boundary(self):
        generator = np.random.default_rng(42)
        mask = generator.random((19, 23)) > 0.7
        tensor = torch.from_numpy(mask).unsqueeze(0)
        for radius in (1, 3, 5):
            np.testing.assert_array_equal(
                _binary_boundary_numpy(mask, radius),
                binary_boundary(tensor, radius).squeeze(0).numpy(),
            )

        single = np.zeros((9, 11), dtype=bool)
        single[4, 5] = True
        dilated = _binary_dilate_numpy(single, 3)
        self.assertTrue(dilated[4, 2])
        self.assertTrue(dilated[4, 8])
        self.assertFalse(dilated[4, 1])

    def test_component_recall_is_one_to_one_and_fp_threshold_is_16_pixels(self):
        target = torch.ones((1, 32, 32), dtype=torch.long)
        target[:, 8:12, 4:8] = 0
        target[:, 8:12, 12:16] = 0
        prediction = torch.ones_like(target)
        prediction[:, 8:12, 4:16] = 0  # one merged prediction for two GT objects
        prediction[:, 20:24, 20:24] = 0  # 16 px: counted as false positive
        prediction[:, 26:29, 20:25] = 0  # 15 px: ignored as noise

        metrics = SegmentationMetrics()
        metrics.update(prediction, target)
        result = metrics.compute()
        self.assertEqual(result["component_count_tiny"], 2.0)
        self.assertEqual(result["component_recall_tiny"], 0.5)
        self.assertEqual(result["small_component_recall"], 0.5)
        self.assertEqual(result["false_positive_components_per_100_images"], 100.0)

    def test_component_connectivity_is_eight_neighbour(self):
        target = torch.ones((1, 16, 16), dtype=torch.long)
        target[:, 5, 5] = 0
        target[:, 6, 6] = 0
        metrics = SegmentationMetrics()
        metrics.update(target.clone(), target)
        result = metrics.compute()
        self.assertEqual(result["component_count_tiny"], 1.0)
        self.assertEqual(result["component_recall_tiny"], 1.0)

    def test_hungarian_matching_maximizes_global_iou(self):
        scores = np.array([[0.9, 0.8], [0.85, 0.1]], dtype=np.float64)
        pairs = _linear_sum_assignment_max(scores)
        self.assertEqual(set(pairs), {(0, 1), (1, 0)})
        self.assertAlmostEqual(sum(scores[row, column] for row, column in pairs), 1.65)


if __name__ == "__main__":
    unittest.main()
