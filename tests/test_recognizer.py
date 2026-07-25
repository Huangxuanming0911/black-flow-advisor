from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackflow_vision.config import RecognitionConfig
from blackflow_vision.parts import PartRecognitionConfig, PartRecognizer
from blackflow_vision.recognizer import MapRecognizer, RecognitionError
from blackflow_vision.synthetic import build_synthetic_map, build_synthetic_parts


class MapRecognizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RecognitionConfig.load(
            ROOT / "config" / "recognition.default.json"
        )

    def test_synthetic_map_recovers_nodes_and_edges(self) -> None:
        result, _ = MapRecognizer(self.config).analyze(build_synthetic_map())

        self.assertEqual(len(result.nodes), 8)
        self.assertEqual(len(result.edges), 9)
        self.assertEqual(result.grid.columns, (240, 390, 540))
        self.assertEqual(result.grid.rows, (170, 320, 470))
        self.assertFalse(result.planner_ready)
        self.assertIn("human_verification_required", result.issues)

    def test_rejects_wrong_resolution(self) -> None:
        image = build_synthetic_map()[0:600, 0:1000]
        with self.assertRaises(RecognitionError):
            MapRecognizer(self.config).analyze(image)

    def test_part_slots_are_classified_from_local_templates(self) -> None:
        config = PartRecognitionConfig.load(
            ROOT / "config" / "parts.default.json"
        )
        image = build_synthetic_parts()
        probe = PartRecognizer(config, None)
        boxes = probe.slot_boxes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = ("wheel", "spring", "engine")
            for label, box in zip(labels, boxes):
                label_dir = root / label
                label_dir.mkdir()
                crop = image[
                    box.y : box.y + box.height,
                    box.x : box.x + box.width,
                ]
                cv2.imwrite(str(label_dir / "sample.png"), crop)

            result = PartRecognizer(config, root).analyze(image)

        self.assertEqual(
            [part.part_id for part in result.parts],
            ["wheel", "spring", "engine"],
        )
        self.assertTrue(all(part.confidence > 0.99 for part in result.parts))
        self.assertFalse(result.planner_ready)


if __name__ == "__main__":
    unittest.main()
