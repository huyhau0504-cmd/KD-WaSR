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


if __name__ == "__main__":
    unittest.main()
