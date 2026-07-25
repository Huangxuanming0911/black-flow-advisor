from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import Box, PartObservation, PartRecognitionResult
from .recognizer import RecognitionError


@dataclass(frozen=True, slots=True)
class PartRecognitionConfig:
    target_width: int = 1280
    target_height: int = 720
    strict_size: bool = True
    panel_roi: tuple[int, int, int, int] = (820, 90, 380, 540)
    rows: int = 5
    columns: int = 2
    slot_padding: int = 12
    template_size: int = 96
    template_threshold: float = 0.82
    empty_stddev_threshold: float = 8.0

    @classmethod
    def load(cls, path: str | Path) -> "PartRecognitionConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["panel_roi"] = tuple(data["panel_roi"])
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("part grid rows and columns must be positive")
        x, y, width, height = self.panel_roi
        if min(x, y) < 0 or min(width, height) <= 0:
            raise ValueError("panel_roi must be a positive rectangle")
        if x + width > self.target_width or y + height > self.target_height:
            raise ValueError("panel_roi extends beyond target dimensions")
        if not 0 < self.template_threshold <= 1:
            raise ValueError("template_threshold must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class _PartTemplate:
    part_id: str
    name: str
    image: np.ndarray


class PartTemplateLibrary:
    """Loads local templates from `<root>/<part-id>/*.png`."""

    def __init__(self, root: str | Path | None, size: int) -> None:
        self._templates: list[_PartTemplate] = []
        self._size = size
        if root is None:
            return
        root_path = Path(root)
        if not root_path.exists():
            return
        for part_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            for path in sorted(part_dir.glob("*.png")):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                self._templates.append(
                    _PartTemplate(
                        part_id=part_dir.name,
                        name=path.name,
                        image=self._normalize(image),
                    )
                )

    @property
    def available(self) -> bool:
        return bool(self._templates)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image, (self._size, self._size), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def classify(self, crop: np.ndarray) -> tuple[str, float, str] | None:
        if not self._templates:
            return None
        normalized = self._normalize(crop)
        best: tuple[str, float, str] | None = None
        for template in self._templates:
            score = float(
                cv2.matchTemplate(
                    normalized, template.image, cv2.TM_CCOEFF_NORMED
                ).max()
            )
            candidate = (template.part_id, score, template.name)
            if best is None or candidate[1] > best[1]:
                best = candidate
        return best


class PartRecognizer:
    def __init__(
        self,
        config: PartRecognitionConfig,
        template_root: str | Path | None,
    ) -> None:
        config.validate()
        self.config = config
        self.templates = PartTemplateLibrary(template_root, config.template_size)

    def analyze(self, image: np.ndarray) -> PartRecognitionResult:
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

        observations: list[PartObservation] = []
        issues: list[str] = []
        for slot, box in enumerate(self.slot_boxes()):
            crop = image[
                box.y : box.y + box.height,
                box.x : box.x + box.width,
            ]
            if float(np.std(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))) < (
                self.config.empty_stddev_threshold
            ):
                continue
            matched = self.templates.classify(crop)
            if matched is None:
                observations.append(
                    PartObservation(
                        slot=slot,
                        part_id="unknown",
                        confidence=0.0,
                        bbox=box,
                        evidence=("occupied_slot",),
                    )
                )
                issues.append(f"unknown_part:slot_{slot}")
                continue
            part_id, score, name = matched
            if score < self.config.template_threshold:
                part_id = "unknown"
                issues.append(f"low_confidence_part:slot_{slot}")
            observations.append(
                PartObservation(
                    slot=slot,
                    part_id=part_id,
                    confidence=round(score, 4),
                    bbox=box,
                    evidence=("occupied_slot", f"template:{name}"),
                )
            )

        if not self.templates.available:
            issues.append("part_template_library_missing")
        issues.append("human_verification_required")
        x, y, roi_width, roi_height = self.config.panel_roi
        return PartRecognitionResult(
            image_size=(width, height),
            panel_roi=Box(x, y, roi_width, roi_height),
            parts=tuple(observations),
            issues=tuple(dict.fromkeys(issues)),
            planner_ready=False,
            metadata={
                "slot_count": self.config.rows * self.config.columns,
                "template_library_available": self.templates.available,
                "recognizer_version": "0.1.0",
            },
        )

    def slot_boxes(self) -> tuple[Box, ...]:
        x, y, width, height = self.config.panel_roi
        cell_width = width / self.config.columns
        cell_height = height / self.config.rows
        boxes: list[Box] = []
        for row in range(self.config.rows):
            for column in range(self.config.columns):
                left = int(round(x + column * cell_width)) + self.config.slot_padding
                top = int(round(y + row * cell_height)) + self.config.slot_padding
                right = (
                    int(round(x + (column + 1) * cell_width))
                    - self.config.slot_padding
                )
                bottom = (
                    int(round(y + (row + 1) * cell_height))
                    - self.config.slot_padding
                )
                boxes.append(Box(left, top, right - left, bottom - top))
        return tuple(boxes)


def annotate_parts(
    image: np.ndarray, result: PartRecognitionResult
) -> np.ndarray:
    canvas = image.copy()
    for part in result.parts:
        color = (70, 220, 70) if part.part_id != "unknown" else (0, 80, 255)
        box = part.bbox
        cv2.rectangle(
            canvas,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            color,
            2,
        )
        cv2.putText(
            canvas,
            f"{part.slot}:{part.part_id} {part.confidence:.2f}",
            (box.x, box.y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "VERIFY REQUIRED",
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (40, 40, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas

