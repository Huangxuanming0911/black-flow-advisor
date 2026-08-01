from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from blackflow_vision.runtime import RealtimeRecognitionLoop


class _FrameSource:
    def capture(self) -> np.ndarray:
        return np.zeros((720, 1280, 3), dtype=np.uint8)


class AdaptiveCapturePolicyTests(unittest.TestCase):
    def test_polling_slows_when_idle_and_wakes_after_visual_change(self) -> None:
        with patch("blackflow_vision.runtime.time.monotonic", return_value=10.0):
            loop = RealtimeRecognitionLoop(
                _FrameSource(),
                lambda capture: None,
                active_window_seconds=1.75,
            )
            self.assertEqual(loop.recommended_interval(0.25, 1.0), 0.25)

        with patch("blackflow_vision.runtime.time.monotonic", return_value=12.0):
            self.assertEqual(loop.recommended_interval(0.25, 1.0), 1.0)
            loop._note_frame_activity(np.zeros((720, 1280, 3), dtype=np.uint8))
            self.assertEqual(loop.recommended_interval(0.25, 1.0), 0.25)

        changed = np.full((720, 1280, 3), 255, dtype=np.uint8)
        with patch("blackflow_vision.runtime.time.monotonic", return_value=14.0):
            self.assertEqual(loop.recommended_interval(0.25, 1.0), 1.0)
            loop._note_frame_activity(changed)
            self.assertEqual(loop.recommended_interval(0.25, 1.0), 0.25)


if __name__ == "__main__":
    unittest.main()
