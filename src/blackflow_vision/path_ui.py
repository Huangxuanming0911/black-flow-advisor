from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .models import Box


@dataclass(frozen=True, slots=True)
class ForestNode:
    id: str
    center: tuple[int, int]
    radius: int
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UndirectedPathEdge:
    first: str
    second: str
    confidence: float
    component_pixels: int


@dataclass(frozen=True, slots=True)
class PathUiResult:
    map_roi: Box
    forest_nodes: tuple[ForestNode, ...]
    edges: tuple[UndirectedPathEdge, ...]
    ambiguous_components: tuple[tuple[str, ...], ...]
    line_evidence_count: int
    planner_ready: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class DirectPathUiRecognizer:
    """Directly recognize path UI pixels and small forest-node UI elements.

    This is a transparent classical-CV seed for producing inspectable masks.
    Its output is not planning-safe until replaced/calibrated with labelled
    path segmentation data.
    """

    def __init__(
        self,
        map_roi: tuple[int, int, int, int] = (100, 120, 1040, 500),
    ) -> None:
        self.map_roi = map_roi

    def analyze(
        self, image: np.ndarray
    ) -> tuple[PathUiResult, np.ndarray, np.ndarray]:
        if image.shape[:2] != (720, 1280):
            raise ValueError("path UI recognition expects normalized 1280x720")
        path_mask, line_count = self._segment_paths(image)
        forest_nodes = self._detect_forest_nodes(image, path_mask)
        skeleton = _skeletonize(path_mask)
        component_edges, ambiguous = _extract_undirected_edges(
            skeleton, forest_nodes
        )
        mask_edges = extract_mask_supported_edges(path_mask, forest_nodes)
        edge_by_key = {
            (edge.first, edge.second): edge
            for edge in (*component_edges, *mask_edges)
        }
        edges = tuple(edge_by_key[key] for key in sorted(edge_by_key))
        x, y, width, height = self.map_roi
        result = PathUiResult(
            map_roi=Box(x, y, width, height),
            forest_nodes=forest_nodes,
            edges=edges,
            ambiguous_components=ambiguous,
            line_evidence_count=line_count,
            planner_ready=False,
        )
        return result, path_mask, skeleton

    def _segment_paths(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        x, y, width, height = self.map_roi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = gray[y : y + height, x : x + width]
        edges = cv2.Canny(roi, 20, 60)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=25,
            minLineLength=35,
            maxLineGap=18,
        )

        horizontal = np.zeros_like(gray)
        vertical = np.zeros_like(gray)
        accepted = 0
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx = abs(int(x2) - int(x1))
                dy = abs(int(y2) - int(y1))
                start = (int(x1 + x), int(y1 + y))
                end = (int(x2 + x), int(y2 + y))
                if dx >= 5 * max(dy, 1):
                    cv2.line(horizontal, start, end, 255, 5, cv2.LINE_AA)
                    accepted += 1
                elif dy >= 5 * max(dx, 1):
                    cv2.line(vertical, start, end, 255, 5, cv2.LINE_AA)
                    accepted += 1

        horizontal = cv2.morphologyEx(
            horizontal,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)),
        )
        vertical = cv2.morphologyEx(
            vertical,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)),
        )
        mask = cv2.max(horizontal, vertical)
        roi_guard = np.zeros_like(mask)
        roi_guard[y : y + height, x : x + width] = 255
        return cv2.bitwise_and(mask, roi_guard), accepted

    def _detect_forest_nodes(
        self, image: np.ndarray, path_mask: np.ndarray
    ) -> tuple[ForestNode, ...]:
        x, y, width, height = self.map_roi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        edge_map = cv2.Canny(gray, 30, 80)
        roi = cv2.GaussianBlur(
            gray[y : y + height, x : x + width],
            (5, 5),
            1.2,
        )
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=28,
            param1=100,
            param2=18,
            minRadius=5,
            maxRadius=12,
        )
        if circles is None:
            return ()
        large_circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=42,
            param1=100,
            param2=24,
            minRadius=16,
            maxRadius=38,
        )
        large = (
            []
            if large_circles is None
            else [
                (int(cx + x), int(cy + y), int(radius))
                for cx, cy, radius in np.round(
                    large_circles[0]
                ).astype(int)
            ]
        )

        candidates: list[tuple[int, int, int, float, tuple[str, ...]]] = []
        for local_x, local_y, radius in np.round(circles[0]).astype(int):
            center_x = int(local_x + x)
            center_y = int(local_y + y)
            center_mean, ring_mean, saturation = _circle_features(
                gray, hsv, center_x, center_y, int(radius)
            )
            contrast = center_mean - ring_mean
            if not (
                35 <= center_mean <= 145
                and contrast >= 8
                and saturation <= 75
            ):
                continue
            if any(
                (center_x - large_x) ** 2 + (center_y - large_y) ** 2
                <= max(14, large_radius * 0.55) ** 2
                for large_x, large_y, large_radius in large
            ):
                continue
            arms = _path_arm_support(
                path_mask, center_x, center_y, int(radius)
            )
            supported_arms = sum(value >= 0.08 for value in arms)
            if supported_arms < 2:
                continue
            outer_edge_density = _outer_edge_density(
                edge_map, center_x, center_y
            )
            if outer_edge_density > 0.13:
                continue
            confidence = min(
                0.94,
                0.45
                + min(contrast, 45) / 100
                + max(0, 75 - saturation) / 500
                + min(supported_arms, 4) * 0.025,
            )
            evidence = (
                f"center_ring_contrast:{contrast:.1f}",
                f"mean_saturation:{saturation:.1f}",
                f"outer_edge_density:{outer_edge_density:.3f}",
                "path_arms:" + ",".join(f"{value:.2f}" for value in arms),
                "small_circle_ui",
            )
            candidates.append(
                (
                    center_x,
                    center_y,
                    int(radius),
                    round(confidence, 4),
                    evidence,
                )
            )

        candidates.sort(key=lambda item: (item[1], item[0]))
        return tuple(
            ForestNode(
                id=f"forest_{index}",
                center=(item[0], item[1]),
                radius=item[2],
                confidence=item[3],
                evidence=item[4],
            )
            for index, item in enumerate(candidates)
        )


def _circle_features(
    gray: np.ndarray,
    hsv: np.ndarray,
    x: int,
    y: int,
    radius: int,
) -> tuple[float, float, float]:
    y0, y1 = max(0, y - radius * 2), min(gray.shape[0], y + radius * 2 + 1)
    x0, x1 = max(0, x - radius * 2), min(gray.shape[1], x + radius * 2 + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    center = distance_sq <= (radius * 0.45) ** 2
    ring = (distance_sq >= (radius * 0.72) ** 2) & (
        distance_sq <= (radius * 1.25) ** 2
    )
    disk = distance_sq <= radius**2
    gray_crop = gray[y0:y1, x0:x1]
    saturation_crop = hsv[y0:y1, x0:x1, 1]
    return (
        float(gray_crop[center].mean()),
        float(gray_crop[ring].mean()),
        float(saturation_crop[disk].mean()),
    )


def _path_arm_support(
    mask: np.ndarray,
    x: int,
    y: int,
    radius: int,
    reach: int = 45,
    half_width: int = 5,
) -> tuple[float, float, float, float]:
    height, width = mask.shape
    gap = radius + 3
    regions = (
        mask[
            max(0, y - half_width) : min(height, y + half_width + 1),
            max(0, x - reach) : max(0, x - gap),
        ],
        mask[
            max(0, y - half_width) : min(height, y + half_width + 1),
            min(width, x + gap) : min(width, x + reach),
        ],
        mask[
            max(0, y - reach) : max(0, y - gap),
            max(0, x - half_width) : min(width, x + half_width + 1),
        ],
        mask[
            min(height, y + gap) : min(height, y + reach),
            max(0, x - half_width) : min(width, x + half_width + 1),
        ],
    )
    return tuple(
        float(np.count_nonzero(region) / region.size)
        if region.size
        else 0.0
        for region in regions
    )


def _outer_edge_density(
    edge_map: np.ndarray,
    x: int,
    y: int,
    inner_radius: int = 14,
    outer_radius: int = 28,
) -> float:
    y0, y1 = max(0, y - outer_radius), min(
        edge_map.shape[0], y + outer_radius + 1
    )
    x0, x1 = max(0, x - outer_radius), min(
        edge_map.shape[1], x + outer_radius + 1
    )
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    annulus = (distance_sq >= inner_radius**2) & (
        distance_sq <= outer_radius**2
    )
    return float(np.count_nonzero(edge_map[y0:y1, x0:x1][annulus]) / np.count_nonzero(annulus))


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(binary):
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(
            skeleton, cv2.subtract(binary, opened)
        )
        binary = cv2.erode(binary, element)
    return skeleton


def _extract_undirected_edges(
    skeleton: np.ndarray,
    nodes: tuple[ForestNode, ...],
) -> tuple[tuple[UndirectedPathEdge, ...], tuple[tuple[str, ...], ...]]:
    if not nodes:
        return (), ()
    cut = skeleton.copy()
    for node in nodes:
        cv2.circle(
            cut,
            node.center,
            node.radius + 7,
            0,
            thickness=-1,
        )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.where(cut > 0, 255, 0).astype(np.uint8),
        connectivity=8,
    )
    edges: dict[tuple[str, str], UndirectedPathEdge] = {}
    ambiguous: list[tuple[str, ...]] = []
    for component in range(1, count):
        pixels = int(stats[component, cv2.CC_STAT_AREA])
        if pixels < 12:
            continue
        component_mask = np.where(labels == component, 255, 0).astype(np.uint8)
        expanded = cv2.dilate(
            component_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        )
        touching = []
        for node in nodes:
            cx, cy = node.center
            radius = node.radius + 8
            y0, y1 = max(0, cy - radius), min(labels.shape[0], cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(labels.shape[1], cx + radius + 1)
            if np.any(expanded[y0:y1, x0:x1]):
                touching.append(node.id)
        touching = sorted(set(touching))
        if len(touching) == 2:
            key = (touching[0], touching[1])
            confidence = min(0.9, 0.45 + pixels / 500)
            previous = edges.get(key)
            candidate = UndirectedPathEdge(
                key[0], key[1], round(confidence, 4), pixels
            )
            if previous is None or candidate.component_pixels > previous.component_pixels:
                edges[key] = candidate
        elif len(touching) > 2:
            ambiguous.append(tuple(touching))
    return (
        tuple(edges[key] for key in sorted(edges)),
        tuple(sorted(set(ambiguous))),
    )


def extract_mask_supported_edges(
    path_mask: np.ndarray,
    nodes: tuple[ForestNode, ...],
    axis_tolerance: int = 14,
    minimum_occupancy: float = 0.20,
) -> tuple[UndirectedPathEdge, ...]:
    """Split an independently recognized path mask at detected UI nodes."""
    candidates: list[tuple[ForestNode, ForestNode]] = []
    for index, first in enumerate(nodes):
        for second in nodes[index + 1 :]:
            dx = abs(first.center[0] - second.center[0])
            dy = abs(first.center[1] - second.center[1])
            if dx <= axis_tolerance or dy <= axis_tolerance:
                candidates.append((first, second))

    accepted: dict[tuple[str, str], UndirectedPathEdge] = {}
    for first, second in candidates:
        horizontal = (
            abs(first.center[1] - second.center[1]) <= axis_tolerance
        )
        if _node_between(first, second, nodes, horizontal, axis_tolerance):
            continue
        occupancy, pixels = _path_corridor_occupancy(
            path_mask,
            first.center,
            second.center,
            horizontal,
            endpoint_margin=max(first.radius, second.radius) + 5,
        )
        if occupancy < minimum_occupancy:
            continue
        key = tuple(sorted((first.id, second.id)))
        accepted[key] = UndirectedPathEdge(
            first=key[0],
            second=key[1],
            confidence=round(min(0.94, 0.45 + occupancy * 0.5), 4),
            component_pixels=pixels,
        )
    return tuple(accepted[key] for key in sorted(accepted))


def _node_between(
    first: ForestNode,
    second: ForestNode,
    nodes: tuple[ForestNode, ...],
    horizontal: bool,
    tolerance: int,
) -> bool:
    first_axis = first.center[1] if horizontal else first.center[0]
    low = min(
        first.center[0] if horizontal else first.center[1],
        second.center[0] if horizontal else second.center[1],
    )
    high = max(
        first.center[0] if horizontal else first.center[1],
        second.center[0] if horizontal else second.center[1],
    )
    for other in nodes:
        if other.id in {first.id, second.id}:
            continue
        other_axis = other.center[1] if horizontal else other.center[0]
        other_progress = other.center[0] if horizontal else other.center[1]
        if (
            abs(other_axis - first_axis) <= tolerance
            and low < other_progress < high
        ):
            return True
    return False


def _path_corridor_occupancy(
    path_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    horizontal: bool,
    endpoint_margin: int,
    half_width: int = 7,
) -> tuple[float, int]:
    x1, y1 = start
    x2, y2 = end
    if horizontal:
        low, high = sorted((x1, x2))
        low += endpoint_margin
        high -= endpoint_margin
        center = round((y1 + y2) / 2)
        region = path_mask[
            max(0, center - half_width) : min(
                path_mask.shape[0], center + half_width + 1
            ),
            max(0, low) : min(path_mask.shape[1], high + 1),
        ]
    else:
        low, high = sorted((y1, y2))
        low += endpoint_margin
        high -= endpoint_margin
        center = round((x1 + x2) / 2)
        region = path_mask[
            max(0, low) : min(path_mask.shape[0], high + 1),
            max(0, center - half_width) : min(
                path_mask.shape[1], center + half_width + 1
            ),
        ]
    if region.size == 0:
        return 0.0, 0
    occupied = int(np.count_nonzero(region))
    return float(occupied / region.size), occupied


def annotate_path_ui(
    image: np.ndarray,
    result: PathUiResult,
    path_mask: np.ndarray,
    skeleton: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    cyan = np.zeros_like(canvas)
    cyan[:, :, 0] = 255
    cyan[:, :, 1] = 190
    visible = path_mask > 0
    canvas[visible] = cv2.addWeighted(
        canvas, 0.40, cyan, 0.60, 0
    )[visible]
    canvas[skeleton > 0] = (255, 255, 255)
    by_id = {node.id: node for node in result.forest_nodes}
    for edge in result.edges:
        cv2.line(
            canvas,
            by_id[edge.first].center,
            by_id[edge.second].center,
            (60, 255, 60),
            2,
            cv2.LINE_AA,
        )
    for node in result.forest_nodes:
        cv2.circle(
            canvas,
            node.center,
            node.radius + 5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{node.id} {node.confidence:.2f}",
            (node.center[0] + 10, node.center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "PATH UI PROTOTYPE - VERIFY REQUIRED",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 30, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas
