from __future__ import annotations

from dataclasses import asdict, dataclass

from .node_semantics import NodeSemanticResult
from .path_ui import PathUiResult


@dataclass(frozen=True, slots=True)
class UnifiedGraphNode:
    node_id: str
    center: tuple[int, int]
    label: str
    kind: str
    confidence: float
    geometry_confidence: float
    semantic_confidence: float
    needs_review: bool
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnifiedGraphEdge:
    edge_id: str
    first: str
    second: str
    confidence: float
    component_pixels: int
    evidence: tuple[str, ...] = ("direct_ui_path_segmentation",)


@dataclass(frozen=True, slots=True)
class UnifiedMapGraph:
    nodes: tuple[UnifiedGraphNode, ...]
    edges: tuple[UnifiedGraphEdge, ...]
    adjacency: dict[str, tuple[str, ...]]
    connected_components: tuple[tuple[str, ...], ...]
    isolated_nodes: tuple[str, ...]
    ambiguous_components: tuple[tuple[str, ...], ...]
    dangling_edges: tuple[tuple[str, str], ...]
    complete_node_semantics: bool
    graph_scope: str = "all_visible_nodes_and_directly_observed_paths"
    schema_version: str = "0.2.0"
    planner_ready: bool = False
    read_only: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def build_unified_map_graph(
    paths: PathUiResult,
    semantics: NodeSemanticResult,
) -> UnifiedMapGraph:
    """Join direct path evidence and node semantics without inventing edges.

    Both recognizers operate on the same ``MapUiNode`` sequence. The join is
    nevertheless performed by stable node ID so callers can serialize or run
    the two recognition stages independently.
    """

    semantic_by_id = {node.node_id: node for node in semantics.nodes}
    nodes: list[UnifiedGraphNode] = []
    complete_node_semantics = True

    for geometry in paths.nodes:
        semantic = semantic_by_id.get(geometry.id)
        if semantic is None:
            complete_node_semantics = False
            nodes.append(
                UnifiedGraphNode(
                    node_id=geometry.id,
                    center=geometry.center,
                    label="未识别",
                    kind=geometry.kind,
                    confidence=round(geometry.confidence * 0.5, 4),
                    geometry_confidence=round(geometry.confidence, 4),
                    semantic_confidence=0.0,
                    needs_review=True,
                    evidence=(
                        *geometry.evidence,
                        "missing_semantic_observation",
                    ),
                )
            )
            continue
        nodes.append(
            UnifiedGraphNode(
                node_id=geometry.id,
                center=geometry.center,
                label=semantic.label,
                kind=semantic.kind,
                confidence=round(
                    min(geometry.confidence, semantic.confidence),
                    4,
                ),
                geometry_confidence=round(geometry.confidence, 4),
                semantic_confidence=round(semantic.confidence, 4),
                needs_review=semantic.needs_review,
                evidence=(
                    *geometry.evidence,
                    *semantic.evidence,
                ),
            )
        )

    node_ids = {node.node_id for node in nodes}
    edge_by_key: dict[tuple[str, str], UnifiedGraphEdge] = {}
    dangling_edges: list[tuple[str, str]] = []
    for edge in paths.edges:
        first, second = sorted((edge.first, edge.second))
        if first == second:
            continue
        if first not in node_ids or second not in node_ids:
            dangling_edges.append((first, second))
            continue
        candidate = UnifiedGraphEdge(
            edge_id=f"{first}--{second}",
            first=first,
            second=second,
            confidence=round(edge.confidence, 4),
            component_pixels=edge.component_pixels,
        )
        previous = edge_by_key.get((first, second))
        if previous is None or candidate.confidence > previous.confidence:
            edge_by_key[(first, second)] = candidate

    edges = tuple(edge_by_key[key] for key in sorted(edge_by_key))
    adjacency_sets = {node_id: set() for node_id in node_ids}
    for edge in edges:
        adjacency_sets[edge.first].add(edge.second)
        adjacency_sets[edge.second].add(edge.first)
    adjacency = {
        node_id: tuple(sorted(neighbours))
        for node_id, neighbours in sorted(adjacency_sets.items())
    }
    connected_components = _connected_components(adjacency)
    isolated_nodes = tuple(
        node_id for node_id, neighbours in adjacency.items() if not neighbours
    )

    planner_ready = (
        paths.planner_ready
        and semantics.planner_ready
        and complete_node_semantics
        and not dangling_edges
        and not any(node.needs_review for node in nodes)
    )
    return UnifiedMapGraph(
        nodes=tuple(nodes),
        edges=edges,
        adjacency=adjacency,
        connected_components=connected_components,
        isolated_nodes=isolated_nodes,
        ambiguous_components=paths.ambiguous_components,
        dangling_edges=tuple(sorted(set(dangling_edges))),
        complete_node_semantics=complete_node_semantics,
        planner_ready=planner_ready,
    )


def _connected_components(
    adjacency: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    remaining = set(adjacency)
    components: list[tuple[str, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in component:
                continue
            component.add(node_id)
            stack.extend(adjacency[node_id])
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(
        sorted(
            components,
            key=lambda component: (-len(component), component),
        )
    )
