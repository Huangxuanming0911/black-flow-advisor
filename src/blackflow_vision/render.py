from __future__ import annotations

import cv2
import numpy as np

from .models import NodeKind, RecognitionResult


COLORS = {
    NodeKind.UNKNOWN: (0, 190, 255),
    NodeKind.CURRENT: (90, 255, 90),
    NodeKind.EMPTY: (180, 180, 180),
    NodeKind.EXIT: (90, 220, 90),
    NodeKind.HIDDEN_EXIT: (0, 210, 210),
    NodeKind.BOSS_EXIT: (90, 90, 255),
}


def annotate(image: np.ndarray, result: RecognitionResult) -> np.ndarray:
    canvas = image.copy()
    node_by_id = {node.id: node for node in result.nodes}
    for edge in result.edges:
        start = node_by_id[edge.from_node].center
        end = node_by_id[edge.to_node].center
        cv2.line(canvas, start, end, (255, 150, 40), 3, cv2.LINE_AA)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(
            canvas,
            f"{edge.occupancy:.2f}",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 220, 160),
            1,
            cv2.LINE_AA,
        )

    for node in result.nodes:
        color = COLORS.get(node.kind, (255, 190, 0))
        cv2.circle(canvas, node.center, node.radius + 4, color, 2, cv2.LINE_AA)
        label = f"{node.id} {node.kind} {node.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (node.center[0] - node.radius, node.center[1] - node.radius - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    status = "PLANNER READY" if result.planner_ready else "VERIFY REQUIRED"
    cv2.putText(
        canvas,
        status,
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (40, 40, 255) if not result.planner_ready else (50, 220, 50),
        2,
        cv2.LINE_AA,
    )
    return canvas

