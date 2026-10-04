from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from run_screening import best_validation
from utils import sha256_split_file


class ScreeningTest(unittest.TestCase):
    def test_split_checksum_is_line_ending_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            lf = Path(directory) / "lf.txt"
            crlf = Path(directory) / "crlf.txt"
            lf.write_bytes(b"0001\n0002\n")
            crlf.write_bytes(b"0001\r\n0002\r\n")
            self.assertEqual(sha256_split_file(lf), sha256_split_file(crlf))

    def test_summary_reads_exact_selected_checkpoint(self):
        metrics = {
            "safety_score": 0.6123,
            "mean_iou": 0.70,
            "obstacle_f1": 0.55,
            "small_component_recall": 0.42,
            "boundary_f1_tol3": 0.64,
            "false_positive_components_per_100_images": 3.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"epoch": 7, "metrics": metrics}, path)
            selected = best_validation(path)

        self.assertEqual(selected["epoch"], 7)
        self.assertEqual(selected["safety_score"], metrics["safety_score"])
        self.assertEqual(
            selected["false_positive_components_per_100_images"], 3.0
        )


if __name__ == "__main__":
    unittest.main()
