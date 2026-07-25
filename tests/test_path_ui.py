from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackflow_vision.path_ui import ForestNode, extract_mask_supported_edges


class PathMaskGraphTests(unittest.TestCase):
    def test_independent_path_mask_is_split_into_undirected_edges(self) -> None:
        mask = np.zeros((720, 1280), dtype=np.uint8)
        cv2.line(mask, (200, 300), (600, 300), 255, 9)
        nodes = tuple(
            ForestNode(
                id=f"forest_{index}",
                center=center,
                radius=9,
                confidence=0.9,
                evidence=("fixture",),
            )
            for index, center in enumerate(
                ((200, 300), (400, 300), (600, 300))
            )
        )

        edges = extract_mask_supported_edges(mask, nodes)

        self.assertEqual(
            {(edge.first, edge.second) for edge in edges},
            {
                ("forest_0", "forest_1"),
                ("forest_1", "forest_2"),
            },
        )
        self.assertTrue(all(edge.confidence > 0.7 for edge in edges))


if __name__ == "__main__":
    unittest.main()
