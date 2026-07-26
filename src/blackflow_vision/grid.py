from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import RecognitionConfig
from .models import EdgeObservation, GridSpec, NodeObservation


@dataclass(frozen=True, slots=True)
class CircleEvidence:
    x: int
    y: int
    radius: int
    confidence: float


def detect_circles(
    image: np.ndarray, config: RecognitionConfig
) -> tuple[CircleEvidence, ...]:
    x, y, width, height = config.map_roi
    roi = image[y : y + height, x : x + width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=config.hough_dp,
        minDist=config.node_min_distance,
        param1=config.hough_param1,
        param2=config.hough_param2,
        minRadius=config.node_radius_min,
        maxRadius=config.node_radius_max,
    )
    if circles is None:
        return ()

    results: list[CircleEvidence] = []
    for cx, cy, radius in np.round(circles[0]).astype(int):
        if radius <= 0:
            continue
        global_x, global_y = int(cx + x), int(cy + y)
        confidence = _circle_confidence(gray, int(cx), int(cy), int(radius))
        results.append(
            CircleEvidence(global_x, global_y, int(radius), confidence)
        )
    return tuple(sorted(results, key=lambda item: (item.y, item.x)))


def _circle_confidence(gray: np.ndarray, x: int, y: int, radius: int) -> float:
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), radius, 255, 2)
    edges = cv2.Canny(gray, 60, 140)
    ring_pixels = edges[mask > 0]
    if ring_pixels.size == 0:
        return 0.0
    return min(1.0, 0.55 + float(np.mean(ring_pixels > 0)) * 0.8)


def _cluster_axis(values: list[int], tolerance: int) -> tuple[int, ...]:
    if not values:
        return ()
    groups: list[list[int]] = [[value] for value in sorted(values)]
    changed = True
    while changed:
        changed = False
        merged: list[list[int]] = []
        for group in groups:
            if not merged:
                merged.append(group)
                continue
            left_center = sum(merged[-1]) / len(merged[-1])
            right_center = sum(group) / len(group)
            if right_center - left_center <= tolerance:
                merged[-1].extend(group)
                changed = True
            else:
                merged.append(group)
        groups = merged
    return tuple(int(round(sum(group) / len(group))) for group in groups)


def fit_grid(
    circles: tuple[CircleEvidence, ...], tolerance: int
) -> GridSpec | None:
    if len(circles) < 2:
        return None
    columns = _cluster_axis([circle.x for circle in circles], tolerance)
    rows = _cluster_axis([circle.y for circle in circles], tolerance)
    if not columns or not rows:
        return None
    return GridSpec(columns=columns, rows=rows)


def nearest_axis(value: int, axes: tuple[int, ...]) -> tuple[int, int]:
    indexed = min(enumerate(axes), key=lambda pair: abs(pair[1] - value))
    return indexed[0], indexed[1]


def road_mask(image: np.ndarray, config: RecognitionConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, config.road_value_min], dtype=np.uint8)
    upper = np.array(
        [179, config.road_saturation_max, 255], dtype=np.uint8
    )
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def reconstruct_edges(
    nodes: tuple[NodeObservation, ...],
    mask: np.ndarray,
    config: RecognitionConfig,
) -> tuple[EdgeObservation, ...]:
    nodes_by_cell = {(node.row, node.column): node for node in nodes}
    edges: list[EdgeObservation] = []
    for node in nodes:
        for delta_row, delta_column in ((0, 1), (1, 0)):
            other = nodes_by_cell.get(
                (node.row + delta_row, node.column + delta_column)
            )
            if other is None:
                continue
            occupancy = corridor_occupancy(
                mask,
                node.center,
                other.center,
                config.road_half_width,
                max(
                    config.road_endpoint_margin,
                    node.radius + 8,
                    other.radius + 8,
                ),
            )
            if occupancy < config.road_occupancy_threshold:
                continue
            confidence = min(
                1.0,
                0.5
                + 0.5
                * (
                    occupancy - config.road_occupancy_threshold
                )
                / max(1e-6, 1 - config.road_occupancy_threshold),
            )
            edges.append(
                EdgeObservation(
                    from_node=node.id,
                    to_node=other.id,
                    confidence=round(confidence, 4),
                    occupancy=round(occupancy, 4),
                )
            )
    return tuple(edges)


def corridor_occupancy(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    half_width: int,
    endpoint_margin: int,
) -> float:
    x1, y1 = start
    x2, y2 = end
    distance = float(np.hypot(x2 - x1, y2 - y1))
    if distance <= endpoint_margin * 2:
        return 0.0
    ratio = endpoint_margin / distance
    sx = int(round(x1 + (x2 - x1) * ratio))
    sy = int(round(y1 + (y2 - y1) * ratio))
    ex = int(round(x2 - (x2 - x1) * ratio))
    ey = int(round(y2 - (y2 - y1) * ratio))

    corridor = np.zeros_like(mask)
    cv2.line(
        corridor,
        (sx, sy),
        (ex, ey),
        255,
        thickness=half_width * 2 + 1,
    )
    pixels = mask[corridor > 0]
    if pixels.size == 0:
        return 0.0
    return float(np.mean(pixels > 0))
