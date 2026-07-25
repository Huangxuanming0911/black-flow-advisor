from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np


class ScreenState(StrEnum):
    MAP = "map"
    TOOLBOX_DETAIL = "toolbox_detail"
    MOVEMENT_SELECTOR = "movement_selector"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ViewportTransform:
    source_width: int
    source_height: int
    client_top: int
    client_height: int
    normalized_width: int = 1280
    normalized_height: int = 720
    padded_bottom: int = 0

    def to_source(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        source_x = round(x * self.source_width / self.normalized_width)
        usable_height = self.client_height - self.padded_bottom
        source_y = self.client_top + round(
            y * usable_height / self.normalized_height
        )
        return source_x, min(source_y, self.source_height - 1)


@dataclass(frozen=True, slots=True)
class ScreenStateResult:
    state: ScreenState
    confidence: float
    evidence: tuple[str, ...]


def normalize_pc_frame(
    image: np.ndarray,
    target_size: tuple[int, int] = (1280, 720),
) -> tuple[np.ndarray, ViewportTransform]:
    """Remove a light Windows title bar and normalize the game client to 16:9."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a three-channel BGR image")

    source_height, source_width = image.shape[:2]
    row_mean = image.mean(axis=(1, 2))
    top_is_light_chrome = float(np.median(row_mean[:20])) > 200
    candidates = np.flatnonzero(
        (np.arange(source_height) > 20) & (row_mean < 100)
    )
    client_top = (
        int(candidates[0])
        if top_is_light_chrome and candidates.size
        else 0
    )
    expected_height = round(source_width * 9 / 16)
    available_height = source_height - client_top
    crop_height = min(expected_height, available_height)
    client = image[client_top : client_top + crop_height]
    padded_bottom = max(0, expected_height - crop_height)
    if padded_bottom:
        client = cv2.copyMakeBorder(
            client,
            0,
            padded_bottom,
            0,
            0,
            cv2.BORDER_REPLICATE,
        )
    normalized = cv2.resize(client, target_size, interpolation=cv2.INTER_AREA)
    transform = ViewportTransform(
        source_width=source_width,
        source_height=source_height,
        client_top=client_top,
        client_height=expected_height,
        normalized_width=target_size[0],
        normalized_height=target_size[1],
        padded_bottom=padded_bottom,
    )
    return normalized, transform


def classify_screen_state(image: np.ndarray) -> ScreenStateResult:
    """Classify the three UI states needed by the capture state machine.

    This is deliberately a conservative structural baseline. It does not OCR
    game text and returns UNKNOWN around thresholds.
    """
    if image.shape[:2] != (720, 1280):
        image, _ = normalize_pc_frame(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)

    right_mid = edges[170:570, 930:1250]
    map_body = edges[100:620, 100:900]
    right_density = float(np.count_nonzero(right_mid) / right_mid.size)
    map_density = float(np.count_nonzero(map_body) / map_body.size)

    if right_density >= 0.060:
        confidence = min(0.99, 0.70 + (right_density - 0.060) * 4)
        return ScreenStateResult(
            ScreenState.MOVEMENT_SELECTOR,
            round(confidence, 4),
            (f"right_panel_edge_density:{right_density:.4f}",),
        )
    if map_density >= 0.045 and right_density <= 0.018:
        confidence = min(0.95, 0.65 + (map_density - 0.045) * 5)
        return ScreenStateResult(
            ScreenState.TOOLBOX_DETAIL,
            round(confidence, 4),
            (
                f"card_area_edge_density:{map_density:.4f}",
                f"right_empty_density:{right_density:.4f}",
            ),
        )
    if right_density < 0.050 and map_density < 0.043:
        distance = min(0.050 - right_density, 0.043 - map_density)
        return ScreenStateResult(
            ScreenState.MAP,
            round(min(0.92, 0.62 + max(distance, 0) * 5), 4),
            (
                f"map_area_edge_density:{map_density:.4f}",
                f"right_area_edge_density:{right_density:.4f}",
            ),
        )
    return ScreenStateResult(
        ScreenState.UNKNOWN,
        0.4,
        (
            f"map_area_edge_density:{map_density:.4f}",
            f"right_area_edge_density:{right_density:.4f}",
        ),
    )


class StableFrameGate:
    """Emit a frame after the UI state and pixels remain stable."""

    def __init__(
        self,
        required_frames: int = 3,
        max_mean_difference: float = 2.5,
    ) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        self.required_frames = required_frames
        self.max_mean_difference = max_mean_difference
        self._last_signature: np.ndarray | None = None
        self._last_state: ScreenState | None = None
        self._stable_count = 0
        self._emitted_signature: np.ndarray | None = None

    @staticmethod
    def _signature(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)

    def offer(self, image: np.ndarray, state: ScreenState) -> bool:
        signature = self._signature(image)
        if self._last_signature is None or state != self._last_state:
            self._stable_count = 1
        else:
            difference = float(
                cv2.absdiff(signature, self._last_signature).mean()
            )
            self._stable_count = (
                self._stable_count + 1
                if difference <= self.max_mean_difference
                else 1
            )
        self._last_signature = signature
        self._last_state = state
        if self._stable_count < self.required_frames:
            return False
        if self._emitted_signature is not None:
            duplicate_difference = float(
                cv2.absdiff(signature, self._emitted_signature).mean()
            )
            if duplicate_difference <= self.max_mean_difference:
                return False
        self._emitted_signature = signature.copy()
        return True
