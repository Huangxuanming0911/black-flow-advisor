from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.update_run_state_catalog import (
    parse_collectibles,
    parse_statuses,
)


ROOT = Path(__file__).resolve().parents[1]


class RunStateCatalogTests(unittest.TestCase):
    def test_seed_catalog_has_unique_route_relevant_entries(self) -> None:
        payload = json.loads(
            (ROOT / "data" / "knowledge" / "run-state.v0.1.json").read_text(
                encoding="utf-8"
            )
        )
        collectible_ids = [item["id"] for item in payload["collectibles"]]
        status_ids = [item["id"] for item in payload["statuses"]]
        self.assertEqual(len(collectible_ids), len(set(collectible_ids)))
        self.assertEqual(len(status_ids), len(set(status_ids)))
        self.assertGreaterEqual(len(collectible_ids), 10)
        self.assertIn("rogue6_relic_cargo_14", collectible_ids)
        self.assertIn("rogue6_weather_5", status_ids)
        self.assertIn("detection_vanguard", status_ids)

    def test_wiki_collectible_template_is_parsed(self) -> None:
        text = """{{肉鸽收藏品
|relicId=rogue6_relic_test
|编号=999
|藏品名=测试藏品
|描述=描述
|效果=【结构化】希望+1<br/>【半结构化】希望+2
|价格=
}}
"""
        items = parse_collectibles(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name_zh"], "测试藏品")
        self.assertEqual(
            items[0]["effect_by_structure"]["semi_structured"],
            "希望+2",
        )

    def test_wiki_status_template_is_parsed(self) -> None:
        text = (
            "{{理想域|图标=weather_test|名称=测试天气|类型=实托邦|"
            "早期效果=敌方攻击+10%，生命+10%|"
            "中期效果=敌方攻击+20%，生命+20%|"
            "晚期效果=敌方攻击+30%，生命+30%}}"
        )
        items = parse_statuses(text)
        self.assertEqual(items[0]["name_zh"], "测试天气")
        self.assertGreater(
            items[0]["planner_effects_by_phase"]["late"][0]["value"],
            items[0]["planner_effects_by_phase"]["early"][0]["value"],
        )


if __name__ == "__main__":
    unittest.main()
