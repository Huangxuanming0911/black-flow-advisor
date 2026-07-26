from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackflow_vision.calibrated import CalibratedSceneRecognizer


class CalibratedSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.private_root = ROOT / "data" / "private"
        cls.manifest = (
            cls.private_root / "raw" / "2026-07-26" / "manifest.json"
        )
        cls.annotations = (
            cls.private_root
            / "annotations"
            / "2026-07-26"
            / "recognized-scenes.json"
        )
        if not cls.manifest.exists() or not cls.annotations.exists():
            raise unittest.SkipTest("private calibration set unavailable")

    def test_all_manifest_screenshots_have_complete_scene_results(self) -> None:
        recognizer = CalibratedSceneRecognizer(
            self.manifest, self.annotations
        )
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        for item in manifest["images"]:
            with self.subTest(file=item["file"]):
                result = recognizer.recognize_file(
                    self.manifest.parent / item["file"]
                )
                self.assertEqual(
                    result["recognition_mode"],
                    "calibrated_exact_hash",
                )
                self.assertIn("state", result)
                self.assertIn("nodes", result)
                self.assertIn("edges", result)
                self.assertFalse(result["planner_ready"])

    def test_graph_annotations_are_valid_undirected_graphs(self) -> None:
        annotations = json.loads(
            self.annotations.read_text(encoding="utf-8")
        )
        self.assertEqual(len(annotations["frames"]), 6)
        for frame in annotations["frames"]:
            with self.subTest(frame=frame["id"]):
                node_ids = [node["id"] for node in frame["nodes"]]
                self.assertEqual(len(node_ids), len(set(node_ids)))
                known = set(node_ids)
                canonical_edges: set[tuple[str, str]] = set()
                for first, second in frame["edges"]:
                    self.assertIn(first, known)
                    self.assertIn(second, known)
                    self.assertNotEqual(first, second)
                    edge = tuple(sorted((first, second)))
                    self.assertNotIn(edge, canonical_edges)
                    canonical_edges.add(edge)

    def test_expected_acceptance_inventory(self) -> None:
        annotations = json.loads(
            self.annotations.read_text(encoding="utf-8")
        )
        frames = {frame["id"]: frame for frame in annotations["frames"]}
        expected_counts = {
            "layer01-map-full": (12, 13),
            "layer01-toolbox": (0, 0),
            "layer01-movement": (0, 0),
            "layer02-map-full": (17, 19),
            "layer03-map-full": (29, 36),
            "layer03-map-partial": (14, 15),
        }
        self.assertEqual(set(frames), set(expected_counts))
        for frame_id, (nodes, edges) in expected_counts.items():
            with self.subTest(frame=frame_id):
                self.assertEqual(len(frames[frame_id]["nodes"]), nodes)
                self.assertEqual(len(frames[frame_id]["edges"]), edges)
        self.assertEqual(len(frames["layer01-toolbox"]["parts"]), 3)
        self.assertEqual(
            len(frames["layer01-movement"]["state"]["movement_options"]),
            3,
        )


if __name__ == "__main__":
    unittest.main()
