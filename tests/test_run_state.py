from __future__ import annotations

import unittest

from blackflow_vision.run_state import (
    add_checkpoint,
    aggregate_run_effects,
    apply_manual_corrections,
    merge_observation,
    new_run_state,
)


class RunStateTests(unittest.TestCase):
    def test_manual_value_survives_ocr_until_checkpoint(self) -> None:
        state = new_run_state()
        state = apply_manual_corrections(
            state, {"resources": {"hope": 9}}
        )
        state = merge_observation(state, {"resources": {"hope": 6, "shield": 2}})
        self.assertEqual(state["resources"]["hope"], 9)
        self.assertEqual(state["resources"]["shield"], 2)
        state = add_checkpoint(state)
        state = merge_observation(state, {"resources": {"hope": 7}})
        self.assertEqual(state["resources"]["hope"], 7)

    def test_collectible_and_status_effects_stack_by_metric(self) -> None:
        catalog = {
            "collectibles": [
                {
                    "id": "relic",
                    "planner_effects": [
                        {"metric": "node_score", "target": "combat", "value": 2},
                        {"metric": "combat_risk", "value": -1},
                    ],
                }
            ],
            "statuses": [
                {
                    "id": "weather",
                    "planner_effects": [
                        {"metric": "merchant_value_multiplier", "value": 0.8}
                    ],
                    "planner_effects_by_phase": {
                        "middle": [{"metric": "combat_risk", "value": 3}]
                    },
                }
            ],
        }
        state = new_run_state(floor=3)
        state["collectibles"] = [{"id": "relic", "active": True}]
        state["statuses"] = [{"id": "weather", "active": True}]
        effects = aggregate_run_effects(state, catalog)
        self.assertEqual(effects.node_scores["combat"], 2)
        self.assertEqual(effects.combat_risk, 2)
        self.assertEqual(effects.merchant_value_multiplier, 0.8)

    def test_checkpoint_records_stage_recruitment_and_resource_delta(self) -> None:
        state = new_run_state()
        state = add_checkpoint(state)
        state["resources"]["hope"] = 4
        state["operators"] = [
            {"name": "测试干员", "class": "近卫", "promoted": True}
        ]
        state["retained_tickets"] = [{"type": "医疗", "count": 2}]
        state = add_checkpoint(state)
        delta = state["checkpoints"][-1]["delta"]
        self.assertEqual(delta["resources"]["hope"], -2)
        self.assertEqual(delta["operators_added"], ["测试干员"])
        self.assertEqual(delta["retained_tickets"]["医疗"], 2)


if __name__ == "__main__":
    unittest.main()
