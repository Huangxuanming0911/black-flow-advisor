"""Import-safe boundary between MaaFramework and the recognition core.

The official Maa Python runtime is deliberately optional. This module contains
no AgentServer registration because callback signatures must be pinned to the
exact MaaFramework release used for packaging. A future launcher should call
`MaaMapAdapter.analyze_bgr` from its registered Custom Recognition callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from blackflow_vision.config import RecognitionConfig
from blackflow_vision.recognizer import MapRecognizer


class MaaMapAdapter:
    def __init__(
        self,
        project_root: str | Path,
        template_root: str | Path | None = None,
    ) -> None:
        root = Path(project_root)
        config = RecognitionConfig.load(
            root / "config" / "recognition.default.json"
        )
        self._recognizer = MapRecognizer(config, template_root)

    def analyze_bgr(self, image: np.ndarray) -> dict[str, Any]:
        result, _ = self._recognizer.analyze(image)
        return {
            "box": self._primary_box(result),
            "detail": result.to_dict(),
        }

    @staticmethod
    def _primary_box(result: Any) -> list[int]:
        if result.nodes:
            node = result.nodes[0]
            return [
                node.bbox.x,
                node.bbox.y,
                node.bbox.width,
                node.bbox.height,
            ]
        roi = result.map_roi
        return [roi.x, roi.y, roi.width, roi.height]
