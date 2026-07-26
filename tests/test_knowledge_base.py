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


if __name__ == "__main__":
    unittest.main()
