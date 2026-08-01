from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.build_route_planner import (
    _node_icon_data_uris,
    _portable_graph,
)


class RoutePlannerBuilderTests(unittest.TestCase):
    def test_label_mapping_overrides_outdated_coarse_kind(self) -> None:
        graph = {
            "nodes": [
                {
                    "node_id": "node_r0c0",
                    "center": [100, 100],
                    "label": "先行一步",
                    "kind": "event",
                }
            ],
            "edges": [],
        }
        portable = _portable_graph(graph, {"先行一步": "scout"})
        self.assertEqual(portable["nodes"][0]["kind"], "scout")

    def test_local_icon_crops_skip_forest_nodes(self) -> None:
        graph = {
            "nodes": [
                {
                    "node_id": "node_r0c0",
                    "center": [200, 200],
                    "label": "作战",
                    "kind": "combat",
                },
                {
                    "node_id": "node_r0c1",
                    "center": [300, 200],
                    "label": "林中节点",
                    "kind": "forest",
                },
            ]
        }
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(image, (200, 200), 24, (255, 255, 255), thickness=-1)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "map.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            icons = _node_icon_data_uris(image_path, graph)
        self.assertEqual(set(icons), {"node_r0c0"})
        self.assertTrue(icons["node_r0c0"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
