from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "companion_app", ROOT / "packaging" / "companion_app.py"
)
assert SPEC and SPEC.loader
companion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(companion)


class CompanionAppTests(unittest.TestCase):
    def test_provider_endpoints_do_not_duplicate_suffix(self) -> None:
        self.assertEqual(
            companion._normalize_endpoint("https://api.openai.com/v1", "/responses"),
            "https://api.openai.com/v1/responses",
        )
        self.assertEqual(
            companion._normalize_endpoint("https://host/v1/responses", "/responses"),
            "https://host/v1/responses",
        )

    def test_provider_parsers(self) -> None:
        self.assertEqual(
            companion._openai_output(
                {"output": [{"content": [{"type": "output_text", "text": "建议保留零件"}]}]}
            ),
            "建议保留零件",
        )
        self.assertEqual(
            companion._compatible_output({"choices": [{"message": {"content": "向上走"}}]}),
            "向上走",
        )
        self.assertEqual(
            companion._anthropic_output({"content": [{"type": "text", "text": "先去羽瞰点"}]}),
            "先去羽瞰点",
        )

    def test_knowledge_search_uses_chinese_terms(self) -> None:
        index = companion.KnowledgeIndex(ROOT / "data" / "knowledge")
        self.assertGreater(len(index.chunks), 50)
        results = index.search("行动力 零件 移动", 5)
        self.assertTrue(results)
        self.assertTrue(any("行动力" in item["text"] or "移动" in item["text"] for item in results))

    def test_settings_never_persist_api_key(self) -> None:
        api = companion.DesktopApi()
        with tempfile.TemporaryDirectory() as directory:
            api.settings_path = Path(directory) / "settings.json"
            api.save_settings(
                {
                    "provider": "openai",
                    "models": {"openai": "example"},
                    "api_key": "must-not-be-written",
                }
            )
            saved = json.loads(api.settings_path.read_text(encoding="utf-8"))
        self.assertNotIn("api_key", saved)
        self.assertNotIn("must-not-be-written", json.dumps(saved))

    def test_prompt_labels_evidence_boundaries(self) -> None:
        prompt = companion._system_prompt("advisor", "", "样本内容", {"floor": 3})
        for label in ("本地规则", "实测样本", "外部资料", "模型推断"):
            self.assertIn(label, prompt)


if __name__ == "__main__":
    unittest.main()
