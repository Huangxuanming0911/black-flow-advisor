from __future__ import annotations

import unittest

from blackflow_vision.models import Box
from blackflow_vision.node_semantics import (
    NodeSemanticObservation,
    NodeSemanticResult,
)
from blackflow_vision.path_ui import (
    MapUiNode,
    PathUiResult,
    UndirectedPathEdge,
)
from blackflow_vision.scene_graph import build_unified_map_graph


def _geometry(node_id: str, x: int) -> MapUiNode:
    return MapUiNode(
        id=node_id,
        center=(x, 100),
        radius=10,
        kind="semantic",
        confidence=0.9,
        evidence=("visible_node_geometry",),
    )


def _semantic(node_id: str, x: int, label: str) -> NodeSemanticObservation:
    return NodeSemanticObservation(
        node_id=node_id,
        center=(x, 100),
        label=label,
        kind="event",
        raw_text=label,
        ocr_confidence=0.95,
        text_bbox=None,
        icon_kind="event",
        icon_confidence=0.9,
        cross_validation="agree",
        confidence=0.93,
        needs_review=False,
        evidence=("ocr_primary",),
    )


class UnifiedSceneGraphTests(unittest.TestCase):
    def test_joins_semantics_edges_and_adjacency_by_node_id(self) -> None:
        geometries = tuple(
            _geometry(node_id, x)
            for node_id, x in (("a", 100), ("b", 200), ("c", 300))
        )
        paths = PathUiResult(
            map_roi=Box(0, 0, 400, 200),
            lattice=None,
            nodes=geometries,
            forest_nodes=(),
            edges=(
                UndirectedPathEdge("b", "a", 0.88, 42),
                UndirectedPathEdge("b", "c", 0.91, 51),
            ),
            ambiguous_components=(),
            line_evidence_count=2,
        )
        semantics = NodeSemanticResult(
            nodes=(
                _semantic("c", 300, "作战"),
                _semantic("a", 100, "不期而遇"),
                _semantic("b", 200, "秘境行商"),
            ),
            ocr_elapsed_ms=10.0,
            template_library_available=True,
        )

        graph = build_unified_map_graph(paths, semantics)

        self.assertEqual([node.node_id for node in graph.nodes], ["a", "b", "c"])
        self.assertEqual([node.label for node in graph.nodes], [
            "不期而遇",
            "秘境行商",
            "作战",
        ])
        self.assertEqual(
            {(edge.first, edge.second) for edge in graph.edges},
            {("a", "b"), ("b", "c")},
        )
        self.assertEqual(graph.adjacency["b"], ("a", "c"))
        self.assertEqual(graph.connected_components, (("a", "b", "c"),))
        self.assertEqual(graph.isolated_nodes, ())
        self.assertFalse(graph.planner_ready)

    def test_reports_missing_semantics_without_inventing_edges(self) -> None:
        geometries = (_geometry("a", 100), _geometry("b", 200))
        paths = PathUiResult(
            map_roi=Box(0, 0, 300, 200),
            lattice=None,
            nodes=geometries,
            forest_nodes=(),
            edges=(),
            ambiguous_components=(),
            line_evidence_count=0,
        )
        semantics = NodeSemanticResult(
            nodes=(_semantic("a", 100, "作战"),),
            ocr_elapsed_ms=10.0,
            template_library_available=False,
        )

        graph = build_unified_map_graph(paths, semantics)

        self.assertFalse(graph.complete_node_semantics)
        self.assertEqual(graph.isolated_nodes, ("a", "b"))
        self.assertEqual(graph.connected_components, (("a",), ("b",)))
        self.assertEqual(len(graph.edges), 0)
        missing = next(node for node in graph.nodes if node.node_id == "b")
        self.assertTrue(missing.needs_review)
        self.assertEqual(missing.label, "未识别")


if __name__ == "__main__":
    unittest.main()
