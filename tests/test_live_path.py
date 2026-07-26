from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackflow_vision.live_path import (
    LatestPathSnapshotWriter,
    RealtimePathProcessor,
    _vote_edges,
)
from blackflow_vision.path_ui import UndirectedPathEdge
from blackflow_vision.runtime import StableCapture
from blackflow_vision.screen import (
    ScreenState,
    ScreenStateResult,
    normalize_pc_frame,
)


class RealtimePathProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = (
            ROOT
            / "data"
            / "private"
            / "raw"
            / "2026-07-26"
            / "layer01_map_normal_full.png"
        )
        if not cls.raw.exists():
            raise unittest.SkipTest("private live-path fixture unavailable")

    def _capture(self, state: ScreenState) -> StableCapture:
        raw = cv2.imread(str(self.raw), cv2.IMREAD_COLOR)
        frame, transform = normalize_pc_frame(raw)
        return StableCapture(
            frame=frame,
            state=ScreenStateResult(
                state=state,
                confidence=0.9,
                evidence=("fixture",),
            ),
            transform=transform,
        )

    def test_stable_map_capture_writes_atomic_live_artifacts(self) -> None:
        snapshot = RealtimePathProcessor().process(
            self._capture(ScreenState.MAP)
        )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertGreater(snapshot.recognition_ms, 0)
        self.assertFalse(snapshot.result.planner_ready)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            LatestPathSnapshotWriter(output).write(
                snapshot,
                "明日方舟",
            )
            expected = {
                "latest-path-state.json",
                "latest-path-mask.png",
                "latest-path-skeleton.png",
                "latest-path-annotated.png",
            }
            self.assertEqual(
                {path.name for path in output.iterdir()},
                expected,
            )
            payload = json.loads(
                (output / "latest-path-state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["window_title"], "明日方舟")
            self.assertEqual(
                payload["graph_scope"],
                "all_visible_nodes_geometry",
            )
            self.assertFalse(payload["planner_ready"])
            self.assertEqual(payload["edge_vote_samples"], 1)
            self.assertIn("path_ui", payload)

    def test_non_map_capture_is_not_sent_to_path_recognition(self) -> None:
        snapshot = RealtimePathProcessor().process(
            self._capture(ScreenState.TOOLBOX_DETAIL)
        )
        self.assertIsNone(snapshot)

    def test_edge_vote_requires_a_temporal_majority(self) -> None:
        stable = UndirectedPathEdge("a", "b", 0.9, 100)
        flicker = UndirectedPathEdge("b", "c", 0.9, 100)
        results = (
            SimpleNamespace(edges=(stable, flicker)),
            SimpleNamespace(edges=(stable,)),
            SimpleNamespace(edges=(stable,)),
        )

        voted = _vote_edges(results, {"a", "b", "c"})

        self.assertEqual(
            {(edge.first, edge.second) for edge in voted},
            {("a", "b")},
        )


if __name__ == "__main__":
    unittest.main()
