from __future__ import annotations

from dataclasses import dataclass, field

from .models import EdgeObservation, NodeKind, NodeObservation, RecognitionResult


GridPoint = tuple[int, int]
GridEdge = tuple[GridPoint, GridPoint]


def _ordered_edge(first: GridPoint, second: GridPoint) -> GridEdge:
    return (first, second) if first <= second else (second, first)


@dataclass(frozen=True, slots=True)
class GraphObservation:
    layer: int
    nodes: dict[GridPoint, NodeKind]
    edges: frozenset[GridEdge]
    clipped_sides: frozenset[str] = frozenset()
    source_id: str = ""

    @classmethod
    def from_recognition(
        cls,
        layer: int,
        result: RecognitionResult,
        clipped_sides: frozenset[str] = frozenset(),
        source_id: str = "",
    ) -> "GraphObservation":
        nodes = {(node.row, node.column): node.kind for node in result.nodes}
        by_id = {node.id: (node.row, node.column) for node in result.nodes}
        edges = frozenset(
            _ordered_edge(by_id[edge.from_node], by_id[edge.to_node])
            for edge in result.edges
            if edge.from_node in by_id and edge.to_node in by_id
        )
        return cls(layer, nodes, edges, clipped_sides, source_id)


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    row_offset: int
    column_offset: int
    overlapping_nodes: int
    score: float


@dataclass(slots=True)
class WorldGraph:
    layer: int
    nodes: dict[GridPoint, NodeKind] = field(default_factory=dict)
    edges: set[GridEdge] = field(default_factory=set)
    observations: list[str] = field(default_factory=list)
    unresolved_boundaries: set[tuple[str, str]] = field(default_factory=set)

    @staticmethod
    def _compatible(first: NodeKind, second: NodeKind) -> bool:
        return (
            first == second
            or first == NodeKind.UNKNOWN
            or second == NodeKind.UNKNOWN
        )

    def align(
        self,
        observation: GraphObservation,
        minimum_overlap: int = 2,
    ) -> AlignmentResult | None:
        if observation.layer != self.layer:
            return None
        if not self.nodes:
            return AlignmentResult(0, 0, 0, 0.0)

        candidates: set[GridPoint] = set()
        for local_point, local_kind in observation.nodes.items():
            for world_point, world_kind in self.nodes.items():
                if self._compatible(local_kind, world_kind):
                    candidates.add(
                        (
                            world_point[0] - local_point[0],
                            world_point[1] - local_point[1],
                        )
                    )

        best: AlignmentResult | None = None
        for row_offset, column_offset in candidates:
            overlap = 0
            score = 0.0
            conflict = False
            for local_point, local_kind in observation.nodes.items():
                world_point = (
                    local_point[0] + row_offset,
                    local_point[1] + column_offset,
                )
                if world_point not in self.nodes:
                    continue
                world_kind = self.nodes[world_point]
                if not self._compatible(local_kind, world_kind):
                    conflict = True
                    break
                overlap += 1
                score += (
                    3.0
                    if local_kind == world_kind and local_kind != NodeKind.UNKNOWN
                    else 1.0
                )
            if conflict or overlap < minimum_overlap:
                continue
            translated_edges = {
                _ordered_edge(
                    (first[0] + row_offset, first[1] + column_offset),
                    (second[0] + row_offset, second[1] + column_offset),
                )
                for first, second in observation.edges
            }
            score += 2.0 * len(translated_edges & self.edges)
            candidate = AlignmentResult(
                row_offset, column_offset, overlap, score
            )
            if best is None or (
                candidate.score,
                candidate.overlapping_nodes,
                -abs(candidate.row_offset) - abs(candidate.column_offset),
            ) > (
                best.score,
                best.overlapping_nodes,
                -abs(best.row_offset) - abs(best.column_offset),
            ):
                best = candidate
        return best

    def merge(
        self,
        observation: GraphObservation,
        minimum_overlap: int = 2,
    ) -> AlignmentResult:
        alignment = self.align(observation, minimum_overlap)
        if alignment is None:
            raise ValueError("observation cannot be aligned safely")
        offset = (alignment.row_offset, alignment.column_offset)
        for local_point, kind in observation.nodes.items():
            world_point = (
                local_point[0] + offset[0],
                local_point[1] + offset[1],
            )
            previous = self.nodes.get(world_point)
            if previous is None or previous == NodeKind.UNKNOWN:
                self.nodes[world_point] = kind
        for first, second in observation.edges:
            self.edges.add(
                _ordered_edge(
                    (first[0] + offset[0], first[1] + offset[1]),
                    (second[0] + offset[0], second[1] + offset[1]),
                )
            )
        if observation.source_id:
            self.observations.append(observation.source_id)
        self.unresolved_boundaries.difference_update(
            {
                (observation.source_id, side)
                for side in {"top", "right", "bottom", "left"}
            }
        )
        self.unresolved_boundaries.update(
            (observation.source_id, side)
            for side in observation.clipped_sides
        )
        return alignment

