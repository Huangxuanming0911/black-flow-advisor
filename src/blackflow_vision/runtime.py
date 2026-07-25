from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

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


class RealtimeRecognitionLoop:
    """State-aware capture loop; input/control automation is intentionally absent."""

    def __init__(
        self,
        source: FrameSource,
        on_capture: Callable[[StableCapture], None],
        required_stable_frames: int = 3,
    ) -> None:
        self.source = source
        self.on_capture = on_capture
        self.gate = StableFrameGate(required_frames=required_stable_frames)

    def poll_once(self) -> StableCapture | None:
        raw = self.source.capture()
        frame, transform = normalize_pc_frame(raw)
        state = classify_screen_state(frame)
        if state.state == ScreenState.UNKNOWN:
            return None
        if not self.gate.offer(frame, state.state):
            return None
        capture = StableCapture(frame, state, transform)
        self.on_capture(capture)
        return capture
