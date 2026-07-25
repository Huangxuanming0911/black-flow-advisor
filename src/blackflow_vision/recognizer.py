from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import RecognitionConfig
from .grid import (
    detect_circles,
    fit_grid,
    nearest_axis,
    reconstruct_edges,
    road_mask,
)
from .models import (
    Box,
    NodeKind,
    NodeObservation,
    RecognitionResult,
)
from .templates import NodeTemplateLibrary


class RecognitionError(RuntimeError):
    pass


class MapRecognizer:
    def __init__(
        self,
        config: RecognitionConfig,
        template_root: str | Path | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.templates = NodeTemplateLibrary(
            template_root, config.node_crop_size
        )

    def analyze_file(self, path: str | Path) -> tuple[RecognitionResult, np.ndarray]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RecognitionError(f"unable to read image: {path}")
        return self.analyze(image)

    def analyze(self, image: np.ndarray) -> tuple[RecognitionResult, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise RecognitionError("expected a three-channel BGR image")
        height, width = image.shape[:2]
        if self.config.strict_size and (
            width != self.config.target_width
            or height != self.config.target_height
        ):
            raise RecognitionError(
                f"expected {self.config.target_width}x"
                f"{self.config.target_height}, got {width}x{height}"
            )

        circles = detect_circles(image, self.config)
        grid = fit_grid(circles, self.config.axis_cluster_tolerance)
        issues: list[str] = []
        if not circles:
            issues.append("no_node_candidates")
        if grid is None:
            issues.append("grid_fit_failed")

        nodes: list[NodeObservation] = []
        occupied_cells: set[tuple[int, int]] = set()
        if grid is not None:
            for circle in circles:
                column, snapped_x = nearest_axis(circle.x, grid.columns)
                row, snapped_y = nearest_axis(circle.y, grid.rows)
                cell = (row, column)
                if cell in occupied_cells:
                    issues.append(f"duplicate_grid_cell:r{row}c{column}")
                    continue
                occupied_cells.add(cell)
                crop = self._crop_square(
                    image, (circle.x, circle.y), self.config.node_crop_size
                )
                matched = self.templates.classify(crop)
                if (
                    matched is not None
                    and matched.score >= self.config.node_template_threshold
                ):
                    kind = matched.kind
                    confidence = min(circle.confidence, matched.score)
                    evidence = (
                        "hough_circle",
                        f"template:{matched.template_name}",
                    )
                else:
                    kind = NodeKind.UNKNOWN
                    confidence = circle.confidence
                    evidence = ("hough_circle",)

                radius = circle.radius
                nodes.append(
                    NodeObservation(
                        id=f"r{row}c{column}",
                        row=row,
                        column=column,
                        center=(snapped_x, snapped_y),
                        radius=radius,
                        kind=kind,
                        confidence=round(float(confidence), 4),
                        bbox=Box(
                            snapped_x - radius,
                            snapped_y - radius,
                            radius * 2,
                            radius * 2,
                        ),
                        evidence=evidence,
                    )
                )

        nodes_tuple = tuple(sorted(nodes, key=lambda node: (node.row, node.column)))
        mask = road_mask(image, self.config)
        edges = reconstruct_edges(nodes_tuple, mask, self.config)

        current_nodes = [node for node in nodes_tuple if node.kind == NodeKind.CURRENT]
        if self.templates.available and len(current_nodes) != 1:
            issues.append(f"expected_one_current_node:found_{len(current_nodes)}")
        if any(
            node.confidence < self.config.minimum_node_confidence
            for node in nodes_tuple
        ):
            issues.append("low_confidence_node")
        if any(
            edge.confidence < self.config.minimum_edge_confidence
            for edge in edges
        ):
            issues.append("low_confidence_edge")
        if nodes_tuple and not edges:
            issues.append("no_roads_detected")
        issues.append("human_verification_required")

        x, y, roi_width, roi_height = self.config.map_roi
        result = RecognitionResult(
            image_size=(width, height),
            map_roi=Box(x, y, roi_width, roi_height),
            grid=grid,
            nodes=nodes_tuple,
            edges=edges,
            issues=tuple(dict.fromkeys(issues)),
            planner_ready=False,
            metadata={
                "node_candidate_count": len(circles),
                "template_library_available": self.templates.available,
                "recognizer_version": "0.1.0",
            },
        )
        return result, mask

    @staticmethod
    def _crop_square(
        image: np.ndarray, center: tuple[int, int], size: int
    ) -> np.ndarray:
        x, y = center
        half = size // 2
        padded = cv2.copyMakeBorder(
            image, half, half, half, half, cv2.BORDER_REPLICATE
        )
        return padded[y : y + size, x : x + size]

