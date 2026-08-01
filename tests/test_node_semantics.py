from __future__ import annotations

from collections import Counter
import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackflow_vision.node_semantics import (
    IconMatch,
    LocalIconTemplateLibrary,
    NodeSemanticRecognizer,
    _correct_label,
)
from blackflow_vision.path_ui import (
    DirectPathUiRecognizer,
    MapUiNode,
)
from blackflow_vision.screen import normalize_pc_frame


class _ConflictingIconLibrary:
    available = True

    @staticmethod
    def classify(image, center):
        return IconMatch("shop", 0.95, "fixture")


class NodeSemanticTests(unittest.TestCase):
    def test_vocabulary_repairs_a_nearby_ocr_character(self) -> None:
        label, kind, similarity = _correct_label("未知的凶房")

        self.assertEqual(label, "未知的凶戾")
        self.assertEqual(kind, "unknown_combat")
        self.assertGreaterEqual(similarity, 0.8)

    def test_specific_event_labels_keep_planner_semantics(self) -> None:
        expected = {
            "先行一步": "scout",
            "失与得": "lost_and_found",
            "得偿所愿": "wish",
            "秘境行商": "secret_trader",
            "诡意行商": "rogue_trader",
        }
        for label, kind in expected.items():
            corrected, observed_kind, similarity = _correct_label(label)
            self.assertEqual(corrected, label)
            self.assertEqual(observed_kind, kind)
            self.assertEqual(similarity, 1.0)

    def test_text_remains_primary_when_icon_validation_conflicts(self) -> None:
        node = MapUiNode(
            id="node_r0c0",
            center=(200, 200),
            radius=25,
            kind="semantic_unknown",
            confidence=0.9,
            evidence=("fixture",),
        )
        ocr_result = (
            [
                [
                    [[175, 220], [225, 220], [225, 240], [175, 240]],
                    "紧急作战",
                    0.99,
                ]
            ],
            0.01,
        )
        recognizer = NodeSemanticRecognizer(
            _ConflictingIconLibrary(),
            ocr_engine=lambda image: ocr_result,
        )

        result = recognizer.analyze(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            (node,),
        )

        observed = result.nodes[0]
        self.assertEqual(observed.label, "紧急作战")
        self.assertEqual(observed.kind, "emergency_combat")
        self.assertEqual(
            observed.cross_validation,
            "conflict_text_kept",
        )
        self.assertTrue(observed.needs_review)


class RealNodeSemanticRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("rapidocr_onnxruntime") is None:
            raise unittest.SkipTest("optional OCR dependency unavailable")
        cls.annotations = (
            ROOT
            / "data"
            / "private"
            / "annotations"
            / "2026-07-26"
            / "recognized-scenes.json"
        )
        cls.image_path = (
            ROOT
            / "data"
            / "private"
            / "raw"
            / "2026-07-26"
            / "layer03_map_normal_full.png"
        )
        if not cls.annotations.exists() or not cls.image_path.exists():
            raise unittest.SkipTest("private node fixtures unavailable")

    def test_layer_three_text_and_icons_agree(self) -> None:
        raw = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        image, _ = normalize_pc_frame(raw)
        graph, _, _ = DirectPathUiRecognizer().analyze(image)
        templates = LocalIconTemplateLibrary.from_reviewed_annotations(
            self.annotations
        )

        result = NodeSemanticRecognizer(templates).analyze(
            image,
            graph.nodes,
        )

        expected = Counter(
            {
                "曲折密道": 2,
                "未知的诡秘": 1,
                "秘境行商": 1,
                "不期而遇": 1,
                "作战": 2,
                "狭路相逢": 1,
                "紧急作战": 1,
                "险路恶敌": 1,
                "诡意行商": 1,
                "先行一步": 1,
                "失与得": 1,
                "未知的凶戾": 2,
                "得偿所愿": 1,
            }
        )
        observed = Counter(
            node.label
            for node in result.nodes
            if node.kind not in {"forest", "current"}
        )
        self.assertEqual(observed, expected)
        self.assertTrue(
            all(
                node.cross_validation == "agree"
                for node in result.nodes
                if node.kind not in {"forest", "current"}
            )
        )
        self.assertFalse(any(node.needs_review for node in result.nodes))


if __name__ == "__main__":
    unittest.main()
