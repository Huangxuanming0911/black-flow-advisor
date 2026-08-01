from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.validate_knowledge_base import validate


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = (
    ROOT / "data" / "knowledge" / "black-flow-rules.v0.1.json"
)


class KnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))

    def test_schema_and_references_are_valid(self) -> None:
        self.assertEqual(validate(self.payload), [])

    def test_all_processed_parts_have_movement_models(self) -> None:
        modes = {
            mode["id"]: mode for mode in self.payload["movement_modes"]
        }
        processed = [
            part
            for part in self.payload["parts"]["items"]
            if part["category"] == "processed"
        ]
        self.assertEqual(len(processed), 12)
        for part in processed:
            self.assertIn(part["movement_mode"], modes)

    def test_endings_have_machine_readable_boss_stages(self) -> None:
        endings = {ending["id"]: ending for ending in self.payload["endings"]}
        self.assertEqual(endings["ending_1"]["boss_stage"], [
            "永无安宁",
            "痛苦将息",
        ])
        self.assertEqual(
            endings["ending_2"]["boss_stage"],
            ["混沌源阶理论"],
        )
        self.assertEqual(endings["ending_3"]["boss_stage"], ["畸症"])

    def test_vision_labels_bridge_to_planning_rules(self) -> None:
        mapping = self.payload["vision_bridge"]["node_name_to_rule_id"]
        self.assertEqual(mapping["作战"], "combat")
        self.assertEqual(mapping["秘境行商"], "secret_trader")
        self.assertEqual(mapping["诡意行商"], "rogue_trader")
        self.assertEqual(mapping["先行一步"], "scout")

    def test_resource_lifecycle_spends_more_near_the_end(self) -> None:
        policy = self.payload["resource_lifecycle_policy"]
        floors = policy["floors"]
        self.assertEqual([item["floor"] for item in floors], list(range(1, 7)))
        self.assertGreater(
            floors[0]["reserve_ratio"],
            floors[-1]["reserve_ratio"],
        )
        self.assertGreater(
            floors[0]["spend_multiplier"],
            floors[-1]["spend_multiplier"],
        )
        self.assertGreater(
            floors[0]["ingot_reserve"],
            floors[-1]["ingot_reserve"],
        )
        self.assertEqual(policy["status"], "planner_heuristic")

    def test_intrinsic_part_value_and_user_preferences_are_configured(self) -> None:
        policy = self.payload["resource_lifecycle_policy"]
        model = policy["part_intrinsic_value_model"]
        self.assertEqual(model["status"], "planner_heuristic")
        self.assertGreater(model["target_rule_value"]["any_node"], 0)
        self.assertGreater(model["pursuit_insurance"]["zero_ap_move"], 0)
        self.assertLess(model["expiring_use_cost_fraction"], 1)
        groups = policy["node_preference_groups"]
        self.assertGreater(policy["node_preference_score_per_level"], 0)
        ids = {group["id"] for group in groups}
        self.assertEqual(len(ids), len(groups))
        self.assertIn("merchant", ids)
        self.assertIn("portal", ids)


if __name__ == "__main__":
    unittest.main()
