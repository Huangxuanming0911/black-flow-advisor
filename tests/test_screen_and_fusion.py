from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackflow_vision.fusion import GraphObservation, WorldGraph
from blackflow_vision.models import NodeKind
from blackflow_vision.screen import (
    ScreenState,
    StableFrameGate,
    classify_screen_state,
    normalize_pc_frame,
)


class RealScreenshotCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = ROOT / "data" / "private" / "raw" / "2026-07-26"
        if not cls.data_root.exists():
            raise unittest.SkipTest("private calibration images are unavailable")
        cls.manifest = json.loads(
            (cls.data_root / "manifest.json").read_text(encoding="utf-8")
        )

    def test_normalization_and_page_state_on_seed_images(self) -> None:
        for item in self.manifest["images"]:
            with self.subTest(file=item["file"]):
                raw = cv2.imread(str(self.data_root / item["file"]))
                normalized, transform = normalize_pc_frame(raw)
                self.assertEqual(normalized.shape, (720, 1280, 3))
                self.assertGreater(transform.client_top, 20)
                observed = classify_screen_state(normalized)
                self.assertEqual(observed.state, ScreenState(item["screen_state"]))

    def test_stable_gate_emits_once_until_pixels_change(self) -> None:
        raw = cv2.imread(str(self.data_root / "layer01_map_normal_full.png"))
        frame, _ = normalize_pc_frame(raw)
        gate = StableFrameGate(required_frames=3)
        self.assertFalse(gate.offer(frame, ScreenState.MAP))
        self.assertFalse(gate.offer(frame, ScreenState.MAP))
        self.assertTrue(gate.offer(frame, ScreenState.MAP))
        self.assertFalse(gate.offer(frame, ScreenState.MAP))


class GraphFusionTests(unittest.TestCase):
    def test_overlapping_partial_views_are_merged_by_grid_translation(self) -> None:
        first = GraphObservation(
            layer=3,
            nodes={
                (0, 0): NodeKind.COMBAT,
                (0, 1): NodeKind.UNKNOWN,
                (1, 1): NodeKind.EVENT,
            },
            edges=frozenset(
                {
                    ((0, 0), (0, 1)),
                    ((0, 1), (1, 1)),
                }
            ),
            clipped_sides=frozenset({"right"}),
            source_id="left_view",
        )
        second = GraphObservation(
            layer=3,
            nodes={
                (0, 0): NodeKind.UNKNOWN,
                (1, 0): NodeKind.EVENT,
                (1, 1): NodeKind.SHOP,
            },
            edges=frozenset(
                {
                    ((0, 0), (1, 0)),
                    ((1, 0), (1, 1)),
                }
            ),
            clipped_sides=frozenset({"left"}),
            source_id="right_view",
        )
        world = WorldGraph(layer=3)
        world.merge(first)
        alignment = world.merge(second)

        self.assertEqual(
            (alignment.row_offset, alignment.column_offset),
            (0, 1),
        )
        self.assertEqual(len(world.nodes), 4)
        self.assertEqual(world.nodes[(1, 2)], NodeKind.SHOP)
        self.assertEqual(len(world.edges), 3)


if __name__ == "__main__":
    unittest.main()
