from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import NodeKind


@dataclass(frozen=True, slots=True)
class TemplateMatchResult:
    kind: NodeKind
    score: float
    template_name: str


class NodeTemplateLibrary:
    """Local-only templates grouped as `<root>/<node_kind>/*.png`."""

    def __init__(self, root: str | Path | None, crop_size: int) -> None:
        self._templates: list[tuple[NodeKind, str, np.ndarray]] = []
        self._crop_size = crop_size
        if root is None:
            return
        root_path = Path(root)
        if not root_path.exists():
            return
        for kind in NodeKind:
            kind_dir = root_path / kind.value
            if not kind_dir.exists():
                continue
            for path in sorted(kind_dir.glob("*.png")):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                normalized = cv2.resize(
                    image, (crop_size, crop_size), interpolation=cv2.INTER_AREA
                )
                self._templates.append((kind, path.name, normalized))

    @property
    def available(self) -> bool:
        return bool(self._templates)

    def classify(self, crop: np.ndarray) -> TemplateMatchResult | None:
        if not self._templates:
            return None
        normalized = cv2.resize(
            crop, (self._crop_size, self._crop_size), interpolation=cv2.INTER_AREA
        )
        best: TemplateMatchResult | None = None
        for kind, name, template in self._templates:
            score = float(
                cv2.matchTemplate(
                    normalized, template, cv2.TM_CCOEFF_NORMED
                ).max()
            )
            candidate = TemplateMatchResult(kind, score, name)
            if best is None or candidate.score > best.score:
                best = candidate
        return best

