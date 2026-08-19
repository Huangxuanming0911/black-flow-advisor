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
class MapUiNode:
    id: str
    center: tuple[int, int]
    radius: int
    kind: str
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NodeLattice:
    spacing: int
    columns: tuple[int, ...]
    rows: tuple[int, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class UndirectedPathEdge:
    first: str
    second: str
    confidence: float
    component_pixels: int


@dataclass(frozen=True, slots=True)
class PathUiResult:
    map_roi: Box
    lattice: NodeLattice | None
    nodes: tuple[MapUiNode, ...]
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
        self.debug_layers: dict[str, np.ndarray] = {}

    def analyze(
        self,
        image: np.ndarray,
        protected_regions: tuple[tuple[int, int, int], ...] = (),
    ) -> tuple[PathUiResult, np.ndarray, np.ndarray]:
        if image.shape[:2] != (720, 1280):
            raise ValueError("path UI recognition expects normalized 1280x720")
        path_mask, line_count = self._segment_paths(
            image, protected_regions
        )
        forest_seeds = self._detect_forest_nodes(image, path_mask)
        lattice, nodes = self._detect_map_nodes(
            image,
            path_mask,
            forest_seeds,
        )
        forest_nodes = tuple(
            ForestNode(
                id=node.id,
                center=node.center,
                radius=node.radius,
                confidence=node.confidence,
                evidence=node.evidence,
            )
            for node in nodes
            if node.kind == "forest"
        )
        clean_path_mask = path_mask.copy()
        for node in nodes:
            cv2.circle(
                clean_path_mask,
                node.center,
                node.radius + 8,
                0,
                thickness=-1,
            )
        self.debug_layers["path_mask"] = clean_path_mask.copy()
        path_mask = clean_path_mask
        skeleton = _skeletonize(path_mask)
        component_edges, ambiguous = _extract_undirected_edges(
            skeleton, nodes
        )
        mask_edges = extract_mask_supported_edges(path_mask, nodes)
        edge_by_key = {
            (edge.first, edge.second): edge
            for edge in (*component_edges, *mask_edges)
        }
        edges = tuple(edge_by_key[key] for key in sorted(edge_by_key))
        x, y, width, height = self.map_roi
        result = PathUiResult(
            map_roi=Box(x, y, width, height),
            lattice=lattice,
            nodes=nodes,
            forest_nodes=forest_nodes,
            edges=edges,
            ambiguous_components=ambiguous,
            line_evidence_count=line_count,
            planner_ready=False,
        )
        return result, path_mask, skeleton

    def _detect_map_nodes(
        self,
        image: np.ndarray,
        path_mask: np.ndarray,
        forest_seeds: tuple[ForestNode, ...],
    ) -> tuple[NodeLattice | None, tuple[MapUiNode, ...]]:
        lattice = _fit_node_lattice(forest_seeds, self.map_roi)
        if lattice is None:
            return None, ()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        edge_map = cv2.Canny(gray, 40, 100)
        spacing = lattice.spacing
        semantic_edge_threshold = 0.06 if spacing > 140 else 0.099
        circle_radius = max(7, min(11, round(spacing * 0.07)))
        nodes: list[MapUiNode] = []

        for row, center_y in enumerate(lattice.rows):
            for column, center_x in enumerate(lattice.columns):
                small_edges, middle_edges = _node_edge_bands(
                    edge_map,
                    center_x,
                    center_y,
                    spacing,
                )
                center_mean, ring_mean, saturation = _circle_features(
                    gray,
                    hsv,
                    center_x,
                    center_y,
                    circle_radius,
                )
                circle_contrast = center_mean - ring_mean
                is_semantic = middle_edges >= semantic_edge_threshold
                is_forest = (
                    not is_semantic
                    and small_edges >= 0.075
                    and circle_contrast >= 2
                    and saturation <= 80
                )
                if not (is_semantic or is_forest):
                    continue

                kind = "forest" if is_forest else "semantic_unknown"
                if is_semantic and _looks_like_current_marker(
                    hsv,
                    center_x,
                    center_y,
                ):
                    kind = "current"
                radius = (
                    circle_radius
                    if kind == "forest"
                    else min(34, max(22, round(spacing * 0.25)))
                )
                arms = _path_arm_support(
                    path_mask,
                    center_x,
                    center_y,
                    circle_radius,
                    reach=max(45, round(spacing * 0.34)),
                )
                supported_arms = sum(value >= 0.08 for value in arms)
                if kind == "forest":
                    margin = min(
                        small_edges - 0.075,
                        circle_contrast / 100,
                        (80 - saturation) / 200,
                    )
                else:
                    margin = middle_edges - semantic_edge_threshold
                confidence = min(
                    0.98,
                    0.72
                    + max(0.0, margin) * 1.4
                    + min(supported_arms, 4) * 0.025,
                )
                evidence = (
                    f"lattice_spacing:{spacing}",
                    f"small_edge_density:{small_edges:.3f}",
                    f"middle_edge_density:{middle_edges:.3f}",
                    f"center_ring_contrast:{circle_contrast:.1f}",
                    f"center_saturation:{saturation:.1f}",
                    "path_arms:" + ",".join(
                        f"{value:.2f}" for value in arms
                    ),
                    "node_icon_pixels",
                )
                nodes.append(
                    MapUiNode(
                        id=f"node_r{row}c{column}",
                        center=(center_x, center_y),
                        radius=radius,
                        kind=kind,
                        confidence=round(confidence, 4),
                        evidence=evidence,
                    )
                )

        return lattice, tuple(nodes)

    def _segment_paths(
        self,
        image: np.ndarray,
        protected_regions: tuple[tuple[int, int, int], ...] = (),
    ) -> tuple[np.ndarray, int]:
        """Segment translucent paths from their paired pale banks.

        A Black Flow path is not a solid bright stroke: its centre frequently
        inherits the map background while two pale banks remain visible.  For
        every possible centre pixel we therefore look for opposite luminance
        gradients at several plausible half-widths.  The midpoint between a
        matched gradient pair receives path probability.  Directional
        hysteresis then keeps weak pixels only when they continue a strong,
        long horizontal or vertical path.

        This stage does not use node adjacency.  ``protected_regions`` merely
        remove icon/label pixels that could imitate paired banks.
        """
        x, y, width, height = self.map_roi
        # The topology is much larger than individual pixels. Processing the
        # map crop at half resolution cuts the cost of every following filter
        # by roughly four while preserving 50 px node spacing and 3--8 px
        # path-bank separation in the normalized source.
        map_crop = image[y : y + height, x : x + width]
        working = cv2.resize(
            map_crop,
            (width // 2, height // 2),
            interpolation=cv2.INTER_AREA,
        )
        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        lightness = cv2.GaussianBlur(
            lab[:, :, 0].astype(np.float32), (0, 0), 0.5
        )
        chroma = np.linalg.norm(
            lab[:, :, 1:].astype(np.float32) - 128.0,
            axis=2,
        )
        pair_horizontal, pair_vertical = _paired_bank_scores(
            lightness,
            chroma,
            half_widths=(1, 2, 3, 4),
        )
        bank_horizontal, bank_vertical = _double_bank_scores(
            lightness,
            half_widths=(1, 2, 3, 4),
            outside_offset=2,
        )
        horizontal_score = np.sqrt(
            pair_horizontal * bank_horizontal * 0.5
        )
        vertical_score = np.sqrt(pair_vertical * bank_vertical * 0.5)

        horizontal_small = _directional_hysteresis(
            horizontal_score,
            direction="horizontal",
        )
        vertical_small = _directional_hysteresis(
            vertical_score,
            direction="vertical",
        )
        horizontal = np.zeros(image.shape[:2], dtype=np.uint8)
        vertical = np.zeros_like(horizontal)
        horizontal[y : y + height, x : x + width] = cv2.resize(
            horizontal_small,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        vertical[y : y + height, x : x + width] = cv2.resize(
            vertical_small,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        # A real path provides a locally continuous tangent. Short paired
        # edges from stars, lettering and background speckles do not. This is
        # strictly a local visual filter: it neither joins components nor
        # requires the resulting graph to be globally connected.
        horizontal = cv2.morphologyEx(
            horizontal,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        )
        horizontal = cv2.morphologyEx(
            horizontal,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),
        )
        vertical = cv2.morphologyEx(
            vertical,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
        )
        vertical = cv2.morphologyEx(
            vertical,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)),
        )
        mask = cv2.max(horizontal, vertical)
        for center_x, center_y, radius in protected_regions:
            guard_radius = radius + 8
            cv2.circle(
                mask,
                (center_x, center_y),
                guard_radius,
                0,
                thickness=-1,
            )
        score_horizontal_full = np.zeros_like(horizontal)
        score_vertical_full = np.zeros_like(vertical)
        score_horizontal_full[y : y + height, x : x + width] = cv2.resize(
            _score_to_u8(horizontal_score),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        score_vertical_full[y : y + height, x : x + width] = cv2.resize(
            _score_to_u8(vertical_score),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        self.debug_layers = {
            "horizontal_bank_score": score_horizontal_full,
            "vertical_bank_score": score_vertical_full,
            "horizontal_path_mask": horizontal.copy(),
            "vertical_path_mask": vertical.copy(),
            "path_mask": mask.copy(),
        }
        return mask, _directional_component_count(horizontal, vertical)

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


def _fit_node_lattice(
    seeds: tuple[ForestNode, ...],
    map_roi: tuple[int, int, int, int],
) -> NodeLattice | None:
    """Fit the visible UI lattice without inventing occupied nodes.

    Seed circles determine only the grid scale and phase. Whether a grid cell
    contains a node is decided later from pixels at that cell.
    """
    if len(seeds) < 2:
        return None
    points = [node.center for node in seeds]
    candidates: list[float] = []
    for index, (first_x, first_y) in enumerate(points):
        for second_x, second_y in points[index + 1 :]:
            differences: list[int] = []
            if abs(first_y - second_y) <= 18:
                differences.append(abs(first_x - second_x))
            if abs(first_x - second_x) <= 18:
                differences.append(abs(first_y - second_y))
            for difference in differences:
                if difference < 70:
                    continue
                for divisor in range(1, 7):
                    spacing = difference / divisor
                    if 90 <= spacing <= 190:
                        candidates.append(spacing)
    if not candidates:
        return None

    best: tuple[tuple[int, int, float], int, float, float] | None = None
    for candidate in candidates:
        spacing = round(candidate)
        tolerance = max(12.0, spacing * 0.08)
        x_fit = _fit_lattice_phase(
            [point[0] for point in points],
            spacing,
            tolerance,
        )
        y_fit = _fit_lattice_phase(
            [point[1] for point in points],
            spacing,
            tolerance,
        )
        inliers = sum(
            _axis_residual(point[0], x_fit[2], spacing) <= tolerance
            and _axis_residual(point[1], y_fit[2], spacing) <= tolerance
            for point in points
        )
        rank = (
            inliers,
            x_fit[0] + y_fit[0],
            x_fit[1] + y_fit[1],
        )
        item = (rank, spacing, x_fit[2], y_fit[2])
        if best is None or item[0] > best[0]:
            best = item

    assert best is not None
    rank, spacing, x_phase, y_phase = best
    if rank[0] < 2:
        return None
    x, y, width, height = map_roi
    columns = _lattice_axes(x_phase, spacing, x, x + width)
    rows = _lattice_axes(y_phase, spacing, y, y + height)
    confidence = min(
        0.98,
        0.55 + 0.4 * rank[0] / max(1, len(points)),
    )
    return NodeLattice(
        spacing=spacing,
        columns=columns,
        rows=rows,
        confidence=round(confidence, 4),
    )


def _fit_lattice_phase(
    values: list[int],
    spacing: int,
    tolerance: float,
) -> tuple[int, float, float]:
    best: tuple[int, float, float] | None = None
    for seed in values:
        aligned = [
            value - round((value - seed) / spacing) * spacing
            for value in values
        ]
        phase = float(np.median(aligned))
        residuals = [
            _axis_residual(value, phase, spacing)
            for value in values
        ]
        item = (
            sum(residual <= tolerance for residual in residuals),
            -sum(min(residual, tolerance * 2) for residual in residuals),
            phase,
        )
        if best is None or item[:2] > best[:2]:
            best = item
    assert best is not None
    return best


def _axis_residual(value: int, phase: float, spacing: int) -> float:
    snapped = phase + round((value - phase) / spacing) * spacing
    return abs(value - snapped)


def _lattice_axes(
    phase: float,
    spacing: int,
    lower: int,
    upper: int,
) -> tuple[int, ...]:
    axes = {
        round(phase + index * spacing)
        for index in range(-20, 21)
        if lower <= phase + index * spacing <= upper
    }
    return tuple(sorted(axes))


def _node_edge_bands(
    edge_map: np.ndarray,
    x: int,
    y: int,
    spacing: int,
) -> tuple[float, float]:
    outer_radius = max(18, round(spacing * 0.28))
    inner_radius = max(9, round(spacing * 0.13))
    y0, y1 = max(0, y - outer_radius), min(
        edge_map.shape[0], y + outer_radius + 1
    )
    x0, x1 = max(0, x - outer_radius), min(
        edge_map.shape[1], x + outer_radius + 1
    )
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    small = distance_sq <= inner_radius**2
    middle = (distance_sq > inner_radius**2) & (
        distance_sq <= outer_radius**2
    )
    crop = edge_map[y0:y1, x0:x1]
    return (
        float(np.count_nonzero(crop[small]) / np.count_nonzero(small)),
        float(np.count_nonzero(crop[middle]) / np.count_nonzero(middle)),
    )


def _looks_like_current_marker(
    hsv: np.ndarray,
    x: int,
    y: int,
    radius: int = 45,
) -> bool:
    crop = hsv[
        max(0, y - radius) : min(hsv.shape[0], y + radius + 1),
        max(0, x - radius) : min(hsv.shape[1], x + radius + 1),
    ]
    if crop.size == 0:
        return False
    saturation = crop[:, :, 1]
    value = crop[:, :, 2]
    hue = crop[:, :, 0]
    white_density = float(
        np.mean((saturation < 45) & (value > 180))
    )
    yellow_density = float(
        np.mean(
            (hue > 15)
            & (hue < 40)
            & (saturation > 90)
            & (value > 150)
        )
    )
    return white_density > 0.03 and yellow_density > 0.003


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
    nodes: tuple[ForestNode | MapUiNode, ...],
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
    # Index components from small neighbourhoods around nodes. The previous
    # implementation allocated and dilated a full-frame mask once per
    # component; textured maps can contain hundreds of fragments, making this
    # step slower than all preceding filters combined.
    touching_by_component: dict[int, set[str]] = {}
    for node in nodes:
        cx, cy = node.center
        radius = node.radius + 15
        y0, y1 = max(0, cy - radius), min(
            labels.shape[0], cy + radius + 1
        )
        x0, x1 = max(0, cx - radius), min(
            labels.shape[1], cx + radius + 1
        )
        for component in np.unique(labels[y0:y1, x0:x1]):
            component = int(component)
            if (
                component > 0
                and int(stats[component, cv2.CC_STAT_AREA]) >= 12
            ):
                touching_by_component.setdefault(component, set()).add(
                    node.id
                )

    for component, node_ids in touching_by_component.items():
        pixels = int(stats[component, cv2.CC_STAT_AREA])
        touching = sorted(node_ids)
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
    nodes: tuple[ForestNode | MapUiNode, ...],
    axis_tolerance: int = 14,
    minimum_occupancy: float = 0.35,
) -> tuple[UndirectedPathEdge, ...]:
    """Split an independently recognized path mask at detected UI nodes."""
    candidates: list[
        tuple[ForestNode | MapUiNode, ForestNode | MapUiNode]
    ] = []
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
            endpoint_margin=max(first.radius, second.radius) + 10,
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


def _paired_bank_scores(
    lightness: np.ndarray,
    chroma: np.ndarray | None = None,
    half_widths: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal/vertical path-centre scores from paired banks.

    Opposite Sobel gradients separated by a plausible path width vote for
    their midpoint. Both bright-inside and dark-inside polarities are
    accepted because the translucent centre changes with the map background.
    The response is divided by local gradient energy so thresholds transfer
    between the bright first layer and the dim third layer.
    """
    gradient_x = cv2.Sobel(
        lightness, cv2.CV_32F, 1, 0, ksize=3, scale=0.25
    )
    gradient_y = cv2.Sobel(
        lightness, cv2.CV_32F, 0, 1, ksize=3, scale=0.25
    )
    horizontal = np.zeros_like(lightness, dtype=np.float32)
    vertical = np.zeros_like(lightness, dtype=np.float32)

    for half_width in half_widths:
        upper = np.roll(gradient_y, half_width, axis=0)
        lower = np.roll(gradient_y, -half_width, axis=0)
        bright_pair = np.minimum(np.maximum(upper, 0), np.maximum(-lower, 0))
        dark_pair = np.minimum(np.maximum(-upper, 0), np.maximum(lower, 0))
        horizontal_pair = np.maximum(bright_pair, dark_pair)
        if chroma is not None:
            bank_chroma = (
                np.roll(chroma, half_width, axis=0)
                + np.roll(chroma, -half_width, axis=0)
            ) * 0.5
            horizontal_pair *= np.clip(
                (70.0 - bank_chroma) / 45.0, 0.35, 1.0
            )
        horizontal = np.maximum(horizontal, horizontal_pair)

        left = np.roll(gradient_x, half_width, axis=1)
        right = np.roll(gradient_x, -half_width, axis=1)
        bright_pair = np.minimum(np.maximum(left, 0), np.maximum(-right, 0))
        dark_pair = np.minimum(np.maximum(-left, 0), np.maximum(right, 0))
        vertical_pair = np.maximum(bright_pair, dark_pair)
        if chroma is not None:
            bank_chroma = (
                np.roll(chroma, half_width, axis=1)
                + np.roll(chroma, -half_width, axis=1)
            ) * 0.5
            vertical_pair *= np.clip(
                (70.0 - bank_chroma) / 45.0, 0.35, 1.0
            )
        vertical = np.maximum(vertical, vertical_pair)

    horizontal[:10, :] = 0
    horizontal[-10:, :] = 0
    vertical[:, :10] = 0
    vertical[:, -10:] = 0

    local_energy_y = cv2.GaussianBlur(
        np.abs(gradient_y), (0, 0), 5.0
    )
    local_energy_x = cv2.GaussianBlur(
        np.abs(gradient_x), (0, 0), 5.0
    )
    horizontal = horizontal / (local_energy_y + 1.5)
    vertical = vertical / (local_energy_x + 1.5)
    return horizontal, vertical


def _double_bank_scores(
    lightness: np.ndarray,
    half_widths: tuple[int, ...],
    outside_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score the pale-bank / translucent-centre / pale-bank profile."""
    center = cv2.GaussianBlur(lightness, (0, 0), 0.8)
    horizontal = np.zeros_like(lightness, dtype=np.float32)
    vertical = np.zeros_like(lightness, dtype=np.float32)
    for half_width in half_widths:
        upper = np.roll(lightness, half_width, axis=0)
        lower = np.roll(lightness, -half_width, axis=0)
        banks = (upper + lower) * 0.5
        outside = (
            np.roll(lightness, half_width + outside_offset, axis=0)
            + np.roll(lightness, -half_width - outside_offset, axis=0)
        ) * 0.5
        horizontal = np.maximum(
            horizontal,
            np.minimum(banks - center, banks - outside),
        )

        left = np.roll(lightness, half_width, axis=1)
        right = np.roll(lightness, -half_width, axis=1)
        banks = (left + right) * 0.5
        outside = (
            np.roll(lightness, half_width + outside_offset, axis=1)
            + np.roll(lightness, -half_width - outside_offset, axis=1)
        ) * 0.5
        vertical = np.maximum(
            vertical,
            np.minimum(banks - center, banks - outside),
        )
    return np.maximum(horizontal, 0), np.maximum(vertical, 0)


def _directional_hysteresis(
    score: np.ndarray,
    direction: str,
    point_threshold: float = 0.70,
    continuity_threshold: float = 0.12,
) -> np.ndarray:
    """Keep paired banks only when their direction has long local support."""
    if direction == "horizontal":
        averaging_size = (9, 3)
        bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (5, 3)
        )
    elif direction == "vertical":
        averaging_size = (3, 9)
        bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 5)
        )
    else:
        raise ValueError(f"unsupported direction: {direction}")

    directional_mean = cv2.blur(score, averaging_size)
    connected = np.where(
        (score >= point_threshold)
        & (directional_mean >= continuity_threshold),
        255,
        0,
    ).astype(np.uint8)
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, bridge_kernel)
    return cv2.dilate(
        connected,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def _score_to_u8(score: np.ndarray) -> np.ndarray:
    return np.clip(score * 80.0, 0, 255).astype(np.uint8)


def _directional_component_count(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    minimum_pixels: int = 30,
) -> int:
    count = 0
    for mask in (horizontal, vertical):
        components, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        count += sum(
            int(stats[index, cv2.CC_STAT_AREA]) >= minimum_pixels
            for index in range(1, components)
        )
    return count


def _node_between(
    first: ForestNode | MapUiNode,
    second: ForestNode | MapUiNode,
    nodes: tuple[ForestNode | MapUiNode, ...],
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
    by_id = {node.id: node for node in result.nodes}
    for edge in result.edges:
        cv2.line(
            canvas,
            by_id[edge.first].center,
            by_id[edge.second].center,
            (60, 255, 60),
            2,
            cv2.LINE_AA,
        )
    colors = {
        "forest": (0, 255, 255),
        "semantic_unknown": (0, 140, 255),
        "current": (90, 255, 90),
    }
    for index, node in enumerate(result.nodes, start=1):
        color = colors.get(node.kind, (255, 180, 0))
        cv2.circle(
            canvas,
            node.center,
            node.radius + 5,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"N{index:02d}",
            (node.center[0] - 11, node.center[1] - node.radius - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    counts = {
        kind: sum(node.kind == kind for node in result.nodes)
        for kind in colors
    }
    summary = (
        f"NODES {len(result.nodes)}  "
        f"FOREST {counts['forest']}  "
        f"SEMANTIC {counts['semantic_unknown']}  "
        f"CURRENT {counts['current']}  "
        f"EDGES {len(result.edges)}"
    )
    cv2.rectangle(canvas, (15, 72), (585, 108), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        summary,
        (24, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return canvas
