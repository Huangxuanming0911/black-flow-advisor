from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Protocol

import cv2
import numpy as np

from .screen import (
    ScreenState,
    ScreenStateResult,
    StableFrameGate,
    ViewportTransform,
    classify_screen_state,
    normalize_pc_frame,
)


class FrameSource(Protocol):
    """A read-only PC-window capture source."""

    def capture(self) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class StableCapture:
    frame: np.ndarray
    state: ScreenStateResult
    transform: ViewportTransform
    temporal_samples: int = 1
    temporal_frames: tuple[np.ndarray, ...] = ()


class RealtimeRecognitionLoop:
    """State-aware capture loop; input/control automation is intentionally absent."""

    def __init__(
        self,
        source: FrameSource,
        on_capture: Callable[[StableCapture], None],
        required_stable_frames: int = 3,
        active_window_seconds: float = 1.75,
    ) -> None:
        self.source = source
        self.on_capture = on_capture
        self.gate = StableFrameGate(required_frames=required_stable_frames)
        self._frame_window: deque[np.ndarray] = deque(
            maxlen=required_stable_frames
        )
        self._window_state: ScreenState | None = None
        self._activity_signature: np.ndarray | None = None
        self._active_window_seconds = max(0.25, active_window_seconds)
        self._active_until = time.monotonic() + self._active_window_seconds

    def recommended_interval(
        self,
        active_interval: float = 0.25,
        idle_interval: float = 1.0,
    ) -> float:
        """Use fast polling briefly after motion, then drop to an idle rate."""
        active = max(0.05, active_interval)
        idle = max(active, idle_interval)
        return active if time.monotonic() < self._active_until else idle

    def _note_frame_activity(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        signature = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
        if self._activity_signature is None:
            changed = True
        else:
            changed = float(
                cv2.absdiff(signature, self._activity_signature).mean()
            ) > 1.8
        self._activity_signature = signature
        if changed:
            self._active_until = time.monotonic() + self._active_window_seconds

    def poll_once(self) -> StableCapture | None:
        raw = self.source.capture()
        frame, transform = normalize_pc_frame(raw)
        self._note_frame_activity(frame)
        state = classify_screen_state(frame)
        if state.state == ScreenState.UNKNOWN:
            self._frame_window.clear()
            self._window_state = None
            return None
        if state.state != self._window_state:
            self._frame_window.clear()
            self._window_state = state.state
        self._frame_window.append(frame)
        if not self.gate.offer(frame, state.state):
            return None
        temporal_frame = np.median(
            np.stack(tuple(self._frame_window), axis=0),
            axis=0,
        ).astype(np.uint8)
        capture = StableCapture(
            temporal_frame,
            state,
            transform,
            temporal_samples=len(self._frame_window),
            temporal_frames=tuple(self._frame_window),
        )
        self.on_capture(capture)
        return capture
