from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "tools" / "build_empirical_rewards.py"
SPEC = importlib.util.spec_from_file_location("build_empirical_rewards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(sample_id: str, floor: str, context: str, **overrides: object) -> dict:
    payload = {
        "sample_id": sample_id,
        "difficulty": 6,
        "difficulty_source": "manual",
        "source_floor": floor,
        "location_context": "main_map",
        "combat_context": context,
        "stage_name": "测试关卡",
        "command_xp": 15,
        "normal_reward_ingots": 2,
        "collectibles": 0,
        "recruitment_tickets": 1,
        "parts": 0,
        "target_life": None,
        "command_xp_multiplier": 1.2,
        "bonus_source": "none",
        "review_status": "confirmed",
        "eligible_for_base_statistics": True,
    }
    payload.update(overrides)
    return payload


class EmpiricalRewardBuilderTests(unittest.TestCase):
    def test_checked_in_snapshot_matches_current_reviewed_baseline(self) -> None:
        payload = json.loads(
            (
                PROJECT_ROOT
                / "data"
                / "knowledge"
                / "empirical-node-rewards.v0.1.json"
            ).read_text(encoding="utf-8"),
        )
        self.assertEqual(payload["sample_policy"]["included"], 31)
        floor_three = next(
            profile
            for profile in payload["profiles"]
            if profile["id"] == "difficulty-6:floor-3:main_map:combat"
        )
        self.assertEqual(floor_three["sample_count"], 3)
        self.assertEqual(
            floor_three["rewards"]["command_xp"]["expected"],
            15,
        )

    def test_filters_bonus_and_incomplete_samples(self) -> None:
        records = [
            _record("clean", "3", "combat"),
            _record("bonus", "3", "combat", bonus_source="chest"),
            _record("incomplete", "3", "combat", command_xp=None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rewards.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records),
                encoding="utf-8",
            )
            payload = MODULE.build_payload(path)
        self.assertEqual(payload["sample_policy"]["included"], 1)
        self.assertEqual(payload["sample_policy"]["excluded"], 2)
        profile = payload["profiles"][0]
        self.assertEqual(
            profile["id"],
            "difficulty-6:floor-3:main_map:combat",
        )
        self.assertEqual(profile["rewards"]["command_xp"]["expected"], 15)

    def test_builds_cross_floor_encounter_fallback(self) -> None:
        records = [
            _record("floor-4", "4", "encounter", command_xp=19),
            _record("floor-5", "5", "encounter", command_xp=19),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rewards.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records),
                encoding="utf-8",
            )
            payload = MODULE.build_payload(path)
        fallback = next(
            profile
            for profile in payload["profiles"]
            if profile["id"]
            == "difficulty-6:floor-all:main_map:encounter"
        )
        self.assertEqual(fallback["sample_count"], 2)
        self.assertEqual(fallback["confidence_weight"], 0.4875)

    def test_separates_difficulties_and_builds_same_floor_prior(self) -> None:
        records = [
            _record("n6", "3", "combat", difficulty=6, command_xp=15),
            _record("n11", "3", "combat", difficulty=11, command_xp=30),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rewards.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records),
                encoding="utf-8",
            )
            payload = MODULE.build_payload(path)
        n6 = next(
            profile
            for profile in payload["profiles"]
            if profile["id"] == "difficulty-6:floor-3:main_map:combat"
        )
        n11 = next(
            profile
            for profile in payload["profiles"]
            if profile["id"] == "difficulty-11:floor-3:main_map:combat"
        )
        same_floor = next(
            profile
            for profile in payload["profiles"]
            if profile["id"]
            == "difficulty-all:floor-3:main_map:combat"
        )
        self.assertEqual(n6["rewards"]["command_xp"]["expected"], 15)
        self.assertEqual(n11["rewards"]["command_xp"]["expected"], 30)
        self.assertEqual(
            same_floor["rewards"]["command_xp"]["expected"],
            22.5,
        )
        self.assertEqual(
            same_floor["fallback_scope"],
            "same_floor_cross_difficulty",
        )


if __name__ == "__main__":
    unittest.main()
