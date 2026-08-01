from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .path_ui import MapUiNode
from .screen import normalize_pc_frame


NODE_LABEL_KINDS: dict[str, str] = {
    "曲折密道": "tunnel",
    "未知的诡秘": "unknown_event",
    "秘境行商": "secret_trader",
    "不期而遇": "event",
    "作战": "combat",
    "狭路相逢": "encounter",
    "紧急作战": "emergency_combat",
    "险路恶敌": "enemy",
    "诡意行商": "rogue_trader",
    "先行一步": "scout",
    "失与得": "lost_and_found",
    "未知的凶戾": "unknown_combat",
    "得偿所愿": "wish",
    "安全的角落": "safe_house",
    "命运所指": "fate",
    "险路尽头": "exit_end",
    "险路小径": "exit_path",
    "应急助力": "emergency_support",
    "羽瞰点": "overlook",
    "“居民”据点": "resident_base",
    "流窜“居民”占领": "resident_occupied",
    "误入奇境": "portal",
}


@dataclass(frozen=True, slots=True)
class IconMatch:
    kind: str
    confidence: float
    template_name: str


@dataclass(frozen=True, slots=True)
class NodeSemanticObservation:
    node_id: str
    center: tuple[int, int]
    label: str
    kind: str
    raw_text: str | None
    ocr_confidence: float | None
    text_bbox: tuple[tuple[int, int], ...] | None
    icon_kind: str | None
    icon_confidence: float | None
    cross_validation: str
    confidence: float
    needs_review: bool
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NodeSemanticResult:
    nodes: tuple[NodeSemanticObservation, ...]
    ocr_elapsed_ms: float
    template_library_available: bool
    planner_ready: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class LocalIconTemplateLibrary:
    """Optional local icon templates derived from private reviewed captures."""

    def __init__(
        self,
        templates: tuple[tuple[str, str, np.ndarray], ...] = (),
        crop_size: int = 64,
    ) -> None:
        self.templates = templates
        self.crop_size = crop_size

    @property
    def available(self) -> bool:
        return bool(self.templates)

    @classmethod
    def from_reviewed_annotations(
        cls,
        annotations_path: str | Path,
        crop_size: int = 64,
    ) -> "LocalIconTemplateLibrary":
        path = Path(annotations_path)
        if not path.exists():
            return cls(crop_size=crop_size)
        payload = json.loads(path.read_text(encoding="utf-8"))
        templates: list[tuple[str, str, np.ndarray]] = []
        for frame in payload.get("frames", []):
            if frame.get("screen", {}).get("kind") != "map":
                continue
            image_path = (path.parent / frame["image"]).resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            normalized, _ = normalize_pc_frame(image)
            for node in frame.get("nodes", []):
                kind = str(node.get("kind", "unknown"))
                if kind in {"forest", "current", "unknown"}:
                    continue
                center = tuple(int(value) for value in node["center"])
                crop = _crop_square(normalized, center, crop_size)
                templates.append(
                    (
                        kind,
                        f"{frame['id']}:{node['id']}",
                        _icon_feature(crop, crop_size),
                    )
                )
        return cls(tuple(templates), crop_size)

    def classify(
        self,
        image: np.ndarray,
        center: tuple[int, int],
    ) -> IconMatch | None:
        if not self.templates:
            return None
        crop = _crop_square(image, center, self.crop_size)
        feature = _icon_feature(crop, self.crop_size)
        best: IconMatch | None = None
        for kind, name, template in self.templates:
            score = _feature_similarity(feature, template)
            candidate = IconMatch(
                kind=kind,
                confidence=round(score, 4),
                template_name=name,
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        return best


class NodeSemanticRecognizer:
    """Recognize node labels with OCR first and icon matching as validation."""

    def __init__(
        self,
        icon_templates: LocalIconTemplateLibrary | None = None,
        ocr_engine: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        self.icon_templates = icon_templates or LocalIconTemplateLibrary()
        self._ocr_engine = ocr_engine

    def analyze(
        self,
        image: np.ndarray,
        nodes: tuple[MapUiNode, ...],
    ) -> NodeSemanticResult:
        raw_result, elapsed = self._run_ocr(image)
        text_items = _parse_ocr_items(raw_result)
        assigned = _assign_text_to_nodes(text_items, nodes)
        observations: list[NodeSemanticObservation] = []

        for node in nodes:
            if node.kind == "forest":
                observations.append(
                    _fixed_observation(node, "林中节点", "forest")
                )
                continue
            if node.kind == "current":
                observations.append(
                    _fixed_observation(node, "当前位置", "current")
                )
                continue

            text_item = assigned.get(node.id)
            raw_text = text_item[1] if text_item else None
            ocr_confidence = text_item[2] if text_item else None
            bbox = text_item[0] if text_item else None
            corrected, text_kind, lexical_similarity = _correct_label(
                raw_text
            )
            icon = self.icon_templates.classify(image, node.center)
            icon_kind = icon.kind if icon else None
            icon_confidence = icon.confidence if icon else None

            if text_kind is None:
                validation = (
                    "icon_only" if icon is not None else "unavailable"
                )
                kind = "semantic_unknown"
                label = corrected or "未识别"
                needs_review = True
                confidence = (
                    min(0.69, icon.confidence)
                    if icon is not None
                    else 0.0
                )
            else:
                kind = text_kind
                label = corrected or raw_text or "未识别"
                strong_icon = (
                    icon is not None and icon.confidence >= 0.45
                )
                if not strong_icon:
                    validation = "text_only"
                    needs_review = False
                elif icon.kind == text_kind:
                    validation = "agree"
                    needs_review = False
                else:
                    validation = "conflict_text_kept"
                    needs_review = True
                text_score = float(ocr_confidence or 0.0)
                confidence = text_score * max(0.75, lexical_similarity)
                if validation == "agree":
                    confidence = min(
                        0.995,
                        confidence * 0.85 + icon.confidence * 0.15,
                    )
                elif validation == "conflict_text_kept":
                    confidence *= 0.82

            evidence = [
                "ocr_primary",
                f"lexical_similarity:{lexical_similarity:.3f}",
            ]
            if raw_text is not None:
                evidence.append(f"ocr_raw:{raw_text}")
            if icon is not None:
                evidence.extend(
                    (
                        f"icon_template:{icon.template_name}",
                        f"icon_score:{icon.confidence:.3f}",
                    )
                )
            observations.append(
                NodeSemanticObservation(
                    node_id=node.id,
                    center=node.center,
                    label=label,
                    kind=kind,
                    raw_text=raw_text,
                    ocr_confidence=(
                        round(float(ocr_confidence), 4)
                        if ocr_confidence is not None
                        else None
                    ),
                    text_bbox=bbox,
                    icon_kind=icon_kind,
                    icon_confidence=icon_confidence,
                    cross_validation=validation,
                    confidence=round(float(confidence), 4),
                    needs_review=needs_review,
                    evidence=tuple(evidence),
                )
            )

        return NodeSemanticResult(
            nodes=tuple(observations),
            ocr_elapsed_ms=round(_elapsed_ms(elapsed), 2),
            template_library_available=self.icon_templates.available,
        )

    def _run_ocr(self, image: np.ndarray) -> tuple[Any, Any]:
        if self._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise RuntimeError(
                    "node OCR requires the optional 'ocr' dependency"
                ) from exc
            self._ocr_engine = RapidOCR()
        output = self._ocr_engine(image)
        if (
            isinstance(output, tuple)
            and len(output) == 2
        ):
            return output
        return output, 0.0


def annotate_node_semantics(
    image: np.ndarray,
    result: NodeSemanticResult,
) -> np.ndarray:
    canvas = image.copy()
    labels: list[tuple[tuple[int, int], str, tuple[int, int, int]]] = []
    for index, node in enumerate(result.nodes, start=1):
        color = (
            (40, 40, 255)
            if node.needs_review
            else (
                (0, 255, 255)
                if node.kind == "forest"
                else (
                    (90, 255, 90)
                    if node.kind == "current"
                    else (0, 170, 255)
                )
            )
        )
        cv2.circle(canvas, node.center, 30, color, 2, cv2.LINE_AA)
        display_label = "林中" if node.kind == "forest" else node.label
        labels.append(
            (
                (node.center[0], max(112, node.center[1] - 39)),
                f"N{index:02d} {display_label}",
                (color[2], color[1], color[0]),
            )
        )
    review_count = sum(node.needs_review for node in result.nodes)
    summary = (
        f"NODE SEMANTICS {len(result.nodes)}  "
        f"REVIEW {review_count}  OCR {result.ocr_elapsed_ms:.0f}ms"
    )
    cv2.rectangle(canvas, (15, 72), (560, 108), (0, 0, 0), -1)
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
    pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = _load_unicode_font(14)
    for center, text, color in labels:
        box = draw.textbbox(center, text, font=font, anchor="mm")
        padded = (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2)
        draw.rounded_rectangle(padded, radius=3, fill=(0, 0, 0))
        draw.text(
            center,
            text,
            font=font,
            fill=color,
            anchor="mm",
        )
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _fixed_observation(
    node: MapUiNode,
    label: str,
    kind: str,
) -> NodeSemanticObservation:
    return NodeSemanticObservation(
        node_id=node.id,
        center=node.center,
        label=label,
        kind=kind,
        raw_text=None,
        ocr_confidence=None,
        text_bbox=None,
        icon_kind=None,
        icon_confidence=None,
        cross_validation="geometry",
        confidence=node.confidence,
        needs_review=False,
        evidence=("geometry_node_kind",),
    )


def _load_unicode_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _parse_ocr_items(
    result: Any,
) -> tuple[tuple[tuple[tuple[int, int], ...], str, float], ...]:
    parsed = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box, text, confidence = item[:3]
        if not text:
            continue
        parsed_box = tuple(
            (round(float(point[0])), round(float(point[1])))
            for point in box
        )
        parsed.append(
            (parsed_box, str(text).strip(), float(confidence))
        )
    return tuple(parsed)


def _assign_text_to_nodes(
    text_items: tuple[
        tuple[tuple[tuple[int, int], ...], str, float], ...
    ],
    nodes: tuple[MapUiNode, ...],
) -> dict[
    str,
    tuple[tuple[tuple[int, int], ...], str, float],
]:
    assigned: dict[
        str,
        tuple[tuple[tuple[int, int], ...], str, float],
    ] = {}
    assigned_rank: dict[str, tuple[float, float]] = {}
    semantic_nodes = [
        node
        for node in nodes
        if node.kind not in {"forest", "current"}
    ]
    for item in text_items:
        box, text, confidence = item
        center_x = sum(point[0] for point in box) / len(box)
        center_y = sum(point[1] for point in box) / len(box)
        candidates = []
        for node in semantic_nodes:
            delta_x = abs(center_x - node.center[0])
            delta_y = center_y - node.center[1]
            if delta_x <= 55 and 10 <= delta_y <= 62:
                candidates.append(
                    (
                        delta_x + abs(delta_y - 27) * 0.35,
                        node,
                    )
                )
        if not candidates:
            continue
        distance, node = min(candidates, key=lambda value: value[0])
        rank = (confidence, -distance)
        if rank > assigned_rank.get(node.id, (-1.0, -999.0)):
            assigned[node.id] = item
            assigned_rank[node.id] = rank
    return assigned


def _correct_label(
    raw_text: str | None,
) -> tuple[str | None, str | None, float]:
    if raw_text is None:
        return None, None, 0.0
    normalized = "".join(
        character
        for character in raw_text
        if character.isalnum() or "\u4e00" <= character <= "\u9fff"
    )
    if normalized in NODE_LABEL_KINDS:
        return normalized, NODE_LABEL_KINDS[normalized], 1.0
    best_label = ""
    best_similarity = 0.0
    for label in NODE_LABEL_KINDS:
        if abs(len(label) - len(normalized)) > 2:
            continue
        similarity = SequenceMatcher(
            None,
            normalized,
            label,
        ).ratio()
        if similarity > best_similarity:
            best_label = label
            best_similarity = similarity
    if best_similarity >= 0.72:
        return (
            best_label,
            NODE_LABEL_KINDS[best_label],
            best_similarity,
        )
    return normalized or raw_text, None, best_similarity


def _crop_square(
    image: np.ndarray,
    center: tuple[int, int],
    size: int,
) -> np.ndarray:
    x, y = center
    half = size // 2
    padded = cv2.copyMakeBorder(
        image,
        half,
        half,
        half,
        half,
        cv2.BORDER_REPLICATE,
    )
    return padded[y : y + size, x : x + size]


def _icon_feature(crop: np.ndarray, size: int) -> np.ndarray:
    normalized = cv2.resize(
        crop,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    lightness = cv2.equalizeHist(lab[:, :, 0])
    saturation = cv2.cvtColor(
        normalized,
        cv2.COLOR_BGR2HSV,
    )[:, :, 1]
    edges = cv2.Canny(lightness, 40, 100)
    return np.dstack((lightness, saturation, edges)).astype(np.float32)


def _feature_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    height, width = first.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    mask = (
        (xx - width / 2) ** 2 + (yy - height / 2) ** 2
        <= (min(width, height) * 0.46) ** 2
    )
    scores = []
    for channel in range(first.shape[2]):
        left = first[:, :, channel][mask].astype(np.float32)
        right = second[:, :, channel][mask].astype(np.float32)
        left -= left.mean()
        right -= right.mean()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        scores.append(
            float(np.dot(left, right) / denominator)
            if denominator > 1e-6
            else 0.0
        )
    score = scores[0] * 0.45 + scores[1] * 0.2 + scores[2] * 0.35
    return round(max(0.0, min(1.0, (score + 1) / 2)), 4)


def _elapsed_ms(elapsed: Any) -> float:
    if isinstance(elapsed, (list, tuple)):
        return sum(float(value) for value in elapsed) * 1000
    return float(elapsed or 0.0) * 1000
