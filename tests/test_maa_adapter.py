from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integration.maafw.agent_adapter import MaaMapAdapter
from blackflow_vision.synthetic import build_synthetic_map


class MaaAdapterTests(unittest.TestCase):
    def test_pure_adapter_returns_box_and_non_ready_detail(self) -> None:
        result = MaaMapAdapter(ROOT).analyze_bgr(build_synthetic_map())

        self.assertEqual(len(result["box"]), 4)
        self.assertEqual(len(result["detail"]["nodes"]), 8)
        self.assertFalse(result["detail"]["planner_ready"])


if __name__ == "__main__":
    unittest.main()
