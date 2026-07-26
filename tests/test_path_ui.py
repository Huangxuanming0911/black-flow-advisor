from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackflow_vision.path_ui import (
    DirectPathUiRecognizer,
    ForestNode,
    _path_corridor_occupancy,
    extract_mask_supported_edges,
)
from blackflow_vision.screen import normalize_pc_frame


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


class RealPathUiFeedbackRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = ROOT / "data" / "private" / "raw" / "2026-07-26"
        cls.annotations_path = (
            ROOT
            / "data"
            / "private"
            / "annotations"
            / "2026-07-26"
            / "recognized-scenes.json"
        )
        if not cls.raw.exists() or not cls.annotations_path.exists():
            raise unittest.SkipTest("private feedback fixtures unavailable")
        payload = json.loads(
            cls.annotations_path.read_text(encoding="utf-8")
        )
        cls.frames = {frame["id"]: frame for frame in payload["frames"]}

    def _mask_and_nodes(
        self, frame_id: str, filename: str
    ) -> tuple[np.ndarray, dict[str, dict]]:
        image = cv2.imread(str(self.raw / filename), cv2.IMREAD_COLOR)
        normalized, _ = normalize_pc_frame(image)
        nodes = {
            node["id"]: node for node in self.frames[frame_id]["nodes"]
        }
        protected = tuple(
            (
                int(node["center"][0]),
                int(node["center"][1]),
                12 if node["kind"] == "forest" else 34,
            )
            for node in nodes.values()
        )
        _, mask, _ = DirectPathUiRecognizer().analyze(
            normalized,
            protected_regions=protected,
        )
        return mask, nodes

    def test_reported_false_edges_have_no_path_band_support(self) -> None:
        cases = {
            ("layer02-map-full", "layer02_map_normal_full.png"): (
                ("l2-r0-c0", "l2-r1-c0"),
                ("l2-r1-c1", "l2-r1-c2"),
                ("l2-r1-c1", "l2-r3-c1"),
                ("l2-r2-c2", "l2-r2-c3"),
            ),
            ("layer03-map-full", "layer03_map_normal_full.png"): (
                ("l3-rm1-c3", "l3-rm1-c4"),
                ("l3-r0-c1", "l3-r1-c1"),
                ("l3-r2-c1", "l3-r2-c2"),
                ("l3-r2-c2", "l3-r2-c3"),
                ("l3-r3-c5", "l3-r3-c6"),
            ),
            (
                "layer03-map-partial",
                "layer03_map_normal_partial_pan.png",
            ): (("p-l3-r2-c1", "p-l3-r2-c2"),),
        }
        for (frame_id, filename), false_edges in cases.items():
            mask, nodes = self._mask_and_nodes(frame_id, filename)
            for first, second in false_edges:
                with self.subTest(frame=frame_id, edge=(first, second)):
                    start = tuple(nodes[first]["center"])
                    end = tuple(nodes[second]["center"])
                    horizontal = abs(start[1] - end[1]) <= 14
                    occupancy, _ = _path_corridor_occupancy(
                        mask,
                        start,
                        end,
                        horizontal,
                        endpoint_margin=35,
                    )
                    self.assertLess(
                        occupancy,
                        0.35,
                        f"false edge retained with occupancy {occupancy:.3f}",
                    )

    def test_reported_missing_vertical_edge_has_path_band_support(self) -> None:
        mask, nodes = self._mask_and_nodes(
            "layer03-map-partial",
            "layer03_map_normal_partial_pan.png",
        )
        first = tuple(nodes["p-l3-r1-c3"]["center"])
        second = tuple(nodes["p-l3-r2-c3"]["center"])
        occupancy, _ = _path_corridor_occupancy(
            mask,
            first,
            second,
            horizontal=False,
            endpoint_margin=35,
        )
        self.assertGreaterEqual(occupancy, 0.35)

    def test_false_partial_node_is_not_a_forest_circle(self) -> None:
        image = cv2.imread(
            str(self.raw / "layer03_map_normal_partial_pan.png"),
            cv2.IMREAD_COLOR,
        )
        normalized, _ = normalize_pc_frame(image)
        result, _, _ = DirectPathUiRecognizer().analyze(normalized)
        self.assertTrue(
            all(
                np.hypot(
                    node.center[0] - 563,
                    node.center[1] - 230,
                )
                > 30
                for node in result.forest_nodes
            )
        )

    def test_all_visible_node_geometry_matches_reviewed_maps(self) -> None:
        cases = (
            ("layer01-map-full", "layer01_map_normal_full.png"),
            ("layer02-map-full", "layer02_map_normal_full.png"),
            ("layer03-map-full", "layer03_map_normal_full.png"),
            (
                "layer03-map-partial",
                "layer03_map_normal_partial_pan.png",
            ),
        )
        for frame_id, filename in cases:
            image = cv2.imread(
                str(self.raw / filename),
                cv2.IMREAD_COLOR,
            )
            normalized, _ = normalize_pc_frame(image)
            result, _, _ = DirectPathUiRecognizer().analyze(normalized)
            reviewed = self.frames[frame_id]["nodes"]

            with self.subTest(frame=frame_id):
                self.assertEqual(len(result.nodes), len(reviewed))
                for expected in reviewed:
                    nearest = min(
                        result.nodes,
                        key=lambda node: np.hypot(
                            node.center[0] - expected["center"][0],
                            node.center[1] - expected["center"][1],
                        ),
                    )
                    self.assertLessEqual(
                        np.hypot(
                            nearest.center[0] - expected["center"][0],
                            nearest.center[1] - expected["center"][1],
                        ),
                        6,
                    )
                    expected_kind = (
                        "forest"
                        if expected["kind"] == "forest"
                        else (
                            "current"
                            if expected["kind"] == "current"
                            else "semantic_unknown"
                        )
                    )
                    self.assertEqual(nearest.kind, expected_kind)

    def test_fast_bank_detector_recovers_reviewed_full_map_edges(self) -> None:
        cases = (
            ("layer01-map-full", "layer01_map_normal_full.png"),
            ("layer02-map-full", "layer02_map_normal_full.png"),
            ("layer03-map-full", "layer03_map_normal_full.png"),
        )
        true_positive = 0
        expected_total = 0
        false_positive = 0
        for frame_id, filename in cases:
            mask, nodes_by_id = self._mask_and_nodes(frame_id, filename)
            graph_nodes = tuple(
                ForestNode(
                    id=node["id"],
                    center=tuple(node["center"]),
                    radius=12 if node["kind"] == "forest" else 34,
                    confidence=1.0,
                    evidence=("reviewed_fixture",),
                )
                for node in nodes_by_id.values()
            )
            predicted = {
                (edge.first, edge.second)
                for edge in extract_mask_supported_edges(mask, graph_nodes)
            }
            expected = {
                tuple(sorted(edge))
                for edge in self.frames[frame_id]["edges"]
            }
            true_positive += len(predicted & expected)
            expected_total += len(expected)
            false_positive += len(predicted - expected)

        self.assertGreaterEqual(true_positive / expected_total, 0.95)
        self.assertLessEqual(false_positive, 2)

    def test_source_resolution_does_not_change_normalized_path_graph(self) -> None:
        frame_id = "layer02-map-full"
        raw = cv2.imread(
            str(self.raw / "layer02_map_normal_full.png"),
            cv2.IMREAD_COLOR,
        )
        nodes_by_id = {
            node["id"]: node for node in self.frames[frame_id]["nodes"]
        }
        graph_nodes = tuple(
            ForestNode(
                id=node["id"],
                center=tuple(node["center"]),
                radius=12 if node["kind"] == "forest" else 34,
                confidence=1.0,
                evidence=("resolution_fixture",),
            )
            for node in nodes_by_id.values()
        )
        protected = tuple(
            (node.center[0], node.center[1], node.radius)
            for node in graph_nodes
        )

        observed = []
        for scale in (0.75, 1.0, 1.25):
            resized = cv2.resize(
                raw,
                None,
                fx=scale,
                fy=scale,
                interpolation=(
                    cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                ),
            )
            normalized, _ = normalize_pc_frame(resized)
            _, mask, _ = DirectPathUiRecognizer().analyze(
                normalized,
                protected_regions=protected,
            )
            observed.append(
                {
                    (edge.first, edge.second)
                    for edge in extract_mask_supported_edges(
                        mask, graph_nodes
                    )
                }
            )

        self.assertEqual(observed[0], observed[1])
        self.assertEqual(observed[1], observed[2])


if __name__ == "__main__":
    unittest.main()
