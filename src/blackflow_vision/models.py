from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class NodeKind(StrEnum):
    UNKNOWN = "unknown"
    CURRENT = "current"
    EMPTY = "empty"
    UNKNOWN_COMBAT = "unknown_combat"
    UNKNOWN_EVENT = "unknown_event"
    COMBAT = "combat"
    EVENT = "event"
    SHOP = "shop"
    LOOKOUT = "lookout"
    TUNNEL = "tunnel"
    EXIT = "exit"
    HIDDEN_EXIT = "hidden_exit"
    BOSS_EXIT = "boss_exit"
    DESTINY = "destiny"


def _to_jsonable_dataclass(value: Any) -> dict[str, Any]:
    def convert(item: Any) -> Any:
        if isinstance(item, StrEnum):
            return str(item)
        if isinstance(item, tuple):
            return [convert(child) for child in item]
        if isinstance(item, list):
            return [convert(child) for child in item]
        if isinstance(item, dict):
            return {key: convert(child) for key, child in item.items()}
        return item

    return convert(asdict(value))


@dataclass(frozen=True, slots=True)
class Box:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class GridSpec:
    columns: tuple[int, ...]
    rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NodeObservation:
    id: str
    row: int
    column: int
    center: tuple[int, int]
    radius: int
    kind: NodeKind
    confidence: float
    bbox: Box
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EdgeObservation:
    from_node: str
    to_node: str
    confidence: float
    occupancy: float


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    image_size: tuple[int, int]
    map_roi: Box
    grid: GridSpec | None
    nodes: tuple[NodeObservation, ...]
    edges: tuple[EdgeObservation, ...]
    issues: tuple[str, ...] = ()
    planner_ready: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable_dataclass(self)


@dataclass(frozen=True, slots=True)
class PartObservation:
    slot: int
    part_id: str
    confidence: float
    bbox: Box
    remaining_uses: int | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PartRecognitionResult:
    image_size: tuple[int, int]
    panel_roi: Box
    parts: tuple[PartObservation, ...]
    issues: tuple[str, ...] = ()
    planner_ready: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable_dataclass(self)
