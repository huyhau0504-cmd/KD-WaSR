from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.mastr import MaSTr1325Dataset, decode_mask
from models import build_model
from utils.losses import LossWeights, MaritimeObjective
from utils.metrics import SegmentationMetrics
from prepare_grouped_splits import assign_groups, cross_split_audit, near_duplicate_groups


class BaselineSmokeTest(unittest.TestCase):
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
            self.assertIn("boundary", terms)

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


if __name__ == "__main__":
    unittest.main()
