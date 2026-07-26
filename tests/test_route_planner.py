from __future__ import annotations

import unittest

from blackflow_vision.route_planner import (
    MovementRule,
    PartState,
    PlannerGraph,
    RouteAction,
    pair_tunnels,
    simulate_route,
)


def _graph() -> PlannerGraph:
    return PlannerGraph.from_unified_dict(
        {
            "nodes": [
                {
                    "node_id": "node_r0c0",
                    "kind": "current",
                    "label": "当前位置",
                    "center": [0, 0],
                },
                {
                    "node_id": "node_r0c1",
                    "kind": "overlook",
                    "label": "羽瞰点",
                    "center": [100, 0],
                },
                {
                    "node_id": "node_r0c2",
                    "kind": "resident_base",
                    "label": "“居民”据点",
                    "center": [200, 0],
                },
                {
                    "node_id": "node_r1c0",
                    "kind": "tunnel",
                    "label": "曲折密道",
                    "center": [0, 100],
                },
                {
                    "node_id": "node_r2c2",
                    "kind": "tunnel",
                    "label": "曲折密道",
                    "center": [200, 200],
                },
            ],
            "edges": [
                {"first": "node_r0c0", "second": "node_r0c1"},
                {"first": "node_r0c1", "second": "node_r0c2"},
                {"first": "node_r0c0", "second": "node_r1c0"},
            ],
        }
    )


RULES = {
    "walk": MovementRule(
        mode_id="walk",
        part_id=None,
        target_type="path_reachable",
        maximum=None,
        ap_cost=1,
    ),
    "structural_principle": MovementRule(
        mode_id="structural_principle",
        part_id="structural_principle",
        target_type="any_node",
        maximum=None,
        ap_cost=1,
    ),
}


class RoutePlannerTests(unittest.TestCase):
    def test_overlook_offsets_walk_cost_on_every_visit(self) -> None:
        result = simulate_route(
            _graph(),
            "node_r0c0",
            (
                RouteAction("node_r0c1"),
                RouteAction("node_r0c0"),
                RouteAction("node_r0c1"),
            ),
            RULES,
            initial_action_points=3,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.action_points, 2)
        self.assertEqual(result.reward_opportunities["羽瞰点行动力"], 2)

    def test_arrival_at_tunnel_forces_zero_ap_transfer(self) -> None:
        graph = _graph()
        pairs = pair_tunnels(graph)
        self.assertEqual(pairs["node_r1c0"], "node_r2c2")
        result = simulate_route(
            graph,
            "node_r0c0",
            (RouteAction("node_r1c0"),),
            RULES,
            initial_action_points=2,
            tunnel_pairs=pairs,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.action_points, 1)
        self.assertEqual(result.current_node, "node_r2c2")
        self.assertEqual(result.steps[0].selected_target, "node_r1c0")
        self.assertEqual(result.steps[0].target, "node_r2c2")
        self.assertTrue(result.steps[0].auto_teleport)

    def test_tunnel_transfer_cannot_be_selected_as_a_move(self) -> None:
        rules = {
            **RULES,
            "tunnel_transfer": MovementRule(
                mode_id="tunnel_transfer",
                part_id=None,
                target_type="paired_node",
                maximum=None,
                ap_cost=0,
            ),
        }
        result = simulate_route(
            _graph(),
            "node_r0c0",
            (RouteAction("node_r2c2", mode_id="tunnel_transfer"),),
            rules,
            initial_action_points=2,
        )
        self.assertFalse(result.valid)
        self.assertIn("不是可选移动方式", result.errors[0])

    def test_resident_base_is_a_combat_and_cleanup_opportunity(self) -> None:
        result = simulate_route(
            _graph(),
            "node_r0c0",
            (
                RouteAction("node_r0c1"),
                RouteAction("node_r0c2"),
            ),
            RULES,
            initial_action_points=2,
        )
        self.assertTrue(result.valid)
        self.assertEqual(
            result.reward_opportunities["居民作战随机奖励"],
            1,
        )
        self.assertEqual(
            result.reward_opportunities["驱逐本层流窜居民"],
            1,
        )

    def test_last_part_use_removes_its_estimated_value(self) -> None:
        part = PartState(
            instance_id="p1",
            part_id="structural_principle",
            remaining_uses=1,
            estimated_value=7,
        )
        result = simulate_route(
            _graph(),
            "node_r0c0",
            (
                RouteAction(
                    "node_r2c2",
                    mode_id="structural_principle",
                    part_instance_id="p1",
                ),
            ),
            RULES,
            initial_action_points=2,
            parts=(part,),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.parts["p1"].remaining_uses, 0)
        self.assertEqual((result.part_value_min, result.part_value_max), (0, 0))

    def test_invalid_walk_does_not_invent_an_edge(self) -> None:
        result = simulate_route(
            _graph(),
            "node_r0c0",
            (RouteAction("node_r2c2"),),
            RULES,
            initial_action_points=3,
        )
        self.assertFalse(result.valid)
        self.assertIn("徒步必须逐段选择", result.errors[0])

    def test_natural_part_value_is_calculated_separately(self) -> None:
        result = simulate_route(
            _graph(),
            "node_r0c0",
            (
                RouteAction("node_r0c1"),
                RouteAction("node_r0c2"),
            ),
            RULES,
            initial_action_points=2,
            parts=(
                PartState("blood", "blood_mushroom", None, 3),
                PartState("fruit", "homesick_fruit", None, 9),
                PartState("wave", "wave", None, 5),
            ),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.part_value_min, 10)
        self.assertEqual(result.part_value_max, 37)


if __name__ == "__main__":
    unittest.main()
