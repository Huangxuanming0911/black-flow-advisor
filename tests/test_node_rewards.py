from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "node-rewards.v0.1.json"
)


class NodeRewardKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(REWARD_PATH.read_text(encoding="utf-8"))

    def test_six_requested_dimensions_are_present(self) -> None:
        self.assertEqual(
            [item["id"] for item in self.payload["dimensions"]],
            [
                "hope",
                "originium_ingots",
                "command_xp",
                "collectibles",
                "recruitment_tickets",
                "parts",
            ],
        )

    def test_node_ids_are_unique_and_reward_dimensions_are_known(self) -> None:
        dimensions = {
            item["id"] for item in self.payload["dimensions"]
        }
        ids = [
            item["id"] for item in self.payload["node_rewards"]
        ]
        self.assertEqual(len(ids), len(set(ids)))
        for node in self.payload["node_rewards"]:
            self.assertLessEqual(set(node["rewards"]), dimensions)
        self.assertNotIn("pursuit", ids)

    def test_pursuit_is_a_forced_encounter_not_a_node(self) -> None:
        forced = {
            item["id"]: item
            for item in self.payload["forced_encounters"]
        }
        pursuit = forced["pursuit_on_ap_exhaustion"]
        self.assertEqual(pursuit["trigger"]["action_points"], 0)
        self.assertEqual(
            pursuit["variants"]["normal"]["rewards"][
                "recruitment_tickets"
            ]["value"],
            1,
        )

    def test_combat_has_a_partial_known_expectation(self) -> None:
        catalog = {
            item["id"]: item for item in self.payload["node_rewards"]
        }
        combat = catalog["combat"]["rewards"]
        self.assertAlmostEqual(
            combat["originium_ingots"]["known_distribution"][
                "expected"
            ],
            1.028846,
        )
        self.assertAlmostEqual(
            combat["parts"]["known_distribution"]["expected"],
            0.067308,
        )

    def test_key_black_flow_rewards_are_structured(self) -> None:
        catalog = {
            item["id"]: item for item in self.payload["node_rewards"]
        }
        self.assertEqual(
            catalog["wish"]["rewards"]["collectibles"]["value"],
            1,
        )
        self.assertEqual(
            catalog["encounter"]["rewards"][
                "recruitment_tickets"
            ]["value"],
            2,
        )
        self.assertEqual(
            catalog["exit_end"]["rewards"]["parts"]["value"],
            1,
        )
        self.assertEqual(
            catalog["scout"]["rewards"]["hope"]["maximum"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
