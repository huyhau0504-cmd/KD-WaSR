from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.geometry import (
    UNKNOWN_LABEL,
    logits_to_mask,
    preprocess_image,
    restore_logits,
)


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.square = Image.fromarray(np.full((640, 640, 3), 128, dtype=np.uint8))

    def test_square_letterbox_uses_zero_after_normalization(self):
        tensor, metadata = preprocess_image(
            self.square, target_size=(384, 512), mode="letterbox"
        )
        self.assertEqual(tuple(tensor.shape), (3, 384, 512))
        self.assertEqual(metadata.resized_size, (384, 384))
        self.assertEqual(metadata.padding, (64, 0, 64, 0))
        self.assertTrue(torch.equal(tensor[:, :, :64], torch.zeros_like(tensor[:, :, :64])))
        self.assertTrue(torch.equal(tensor[:, :, -64:], torch.zeros_like(tensor[:, :, -64:])))

    def test_letterbox_inverse_restores_original_size(self):
        _, metadata = preprocess_image(
            self.square, target_size=(384, 512), mode="letterbox"
        )
        logits = torch.zeros((1, 3, 96, 128))
        logits[:, 1] = 3.0
        restored, valid = restore_logits(logits, metadata)
        self.assertEqual(tuple(restored.shape), (1, 3, 640, 640))
        self.assertEqual(tuple(valid.shape), (1, 640, 640))
        self.assertTrue(valid.all())
        mask = logits_to_mask(logits, metadata)
        self.assertTrue(torch.all(mask == 1))

    def test_letterbox_spatial_alignment_with_asymmetric_content(self):
        image = Image.fromarray(np.zeros((333, 517, 3), dtype=np.uint8))
        _, metadata = preprocess_image(
            image, target_size=(385, 513), mode="letterbox"
        )
        source = torch.ones((333, 517), dtype=torch.long)
        source[40:170, 70:210] = 0
        source[210:300, 330:480] = 2
        resized = F.interpolate(
            source[None, None].float(), size=metadata.resized_size, mode="nearest"
        )[0, 0].long()
        logits = F.one_hot(resized, num_classes=3).permute(2, 0, 1).float() * 20.0
        left, top, right, bottom = metadata.padding
        logits = F.pad(logits, (left, right, top, bottom))
        restored = logits_to_mask(logits, metadata)
        agreement = (restored == source).float().mean()
        self.assertGreater(float(agreement), 0.994)
        # Interior probe points detect even a small translation of the inverse map.
        self.assertEqual(int(restored[100, 100]), 0)
        self.assertEqual(int(restored[250, 400]), 2)
        self.assertEqual(int(restored[20, 20]), 1)

    def test_odd_letterbox_metadata_is_exact(self):
        odd = Image.fromarray(np.zeros((333, 517, 3), dtype=np.uint8))
        tensor, metadata = preprocess_image(
            odd, target_size=(385, 513), mode="letterbox"
        )
        left, top, right, bottom = metadata.padding
        resized_height, resized_width = metadata.resized_size
        self.assertEqual(resized_width + left + right, 513)
        self.assertEqual(resized_height + top + bottom, 385)
        self.assertEqual(tuple(tensor.shape[-2:]), (385, 513))

    def test_center_crop_marks_missing_pixels_unknown(self):
        tensor, metadata = preprocess_image(
            self.square, target_size=(384, 512), mode="center_crop"
        )
        self.assertEqual(tuple(tensor.shape), (3, 384, 512))
        self.assertEqual(metadata.resized_size, (512, 512))
        self.assertEqual(metadata.crop, (0, 64, 512, 448))
        logits = torch.zeros((3, 384, 512))
        logits[2] = 1.0
        mask = logits_to_mask(logits, metadata)
        labels = set(torch.unique(mask).tolist())
        self.assertEqual(labels, {2, UNKNOWN_LABEL})
        self.assertTrue(torch.all(mask[:80] == UNKNOWN_LABEL))
        self.assertTrue(torch.all(mask[-80:] == UNKNOWN_LABEL))
        self.assertTrue(torch.all(mask[100:-100] == 2))

    def test_odd_center_crop_has_precise_unknown_boundaries(self):
        odd = Image.fromarray(np.zeros((333, 517, 3), dtype=np.uint8))
        _, metadata = preprocess_image(
            odd, target_size=(385, 513), mode="center_crop"
        )
        logits = torch.zeros((3, 385, 513))
        logits[1] = 1.0
        mask = logits_to_mask(logits, metadata)
        valid = mask != UNKNOWN_LABEL
        rows, columns = torch.where(valid)
        self.assertEqual((int(rows.min()), int(rows.max())), (0, 332))
        # The 4:3 crop removes unequal left/right margins after odd rounding.
        self.assertEqual((int(columns.min()), int(columns.max())), (37, 479))
        self.assertTrue(torch.all(mask[:, :37] == UNKNOWN_LABEL))
        self.assertTrue(torch.all(mask[:, 480:] == UNKNOWN_LABEL))

    def test_stretch_round_trip_batch_one(self):
        _, metadata = preprocess_image(
            self.square, target_size=(384, 512), mode="stretch"
        )
        logits = torch.randn(1, 3, 48, 64)
        restored, valid = restore_logits(logits, metadata)
        self.assertEqual(tuple(restored.shape), (1, 3, 640, 640))
        self.assertTrue(valid.all())


if __name__ == "__main__":
    unittest.main()
