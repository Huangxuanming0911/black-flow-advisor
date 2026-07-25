"""MaaFramework 5.12.2 AgentServer launcher for read-only recognition."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition

from blackflow_vision.config import RecognitionConfig
from blackflow_vision.parts import PartRecognitionConfig, PartRecognizer
from blackflow_vision.recognizer import MapRecognizer, RecognitionError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private"


def _primary_box(result: Any) -> list[int]:
    if result.nodes:
        node = result.nodes[0]
        return [
            node.bbox.x,
            node.bbox.y,
            node.bbox.width,
            node.bbox.height,
        ]
    roi = result.map_roi
    return [roi.x, roi.y, roi.width, roi.height]


def _error_result(exc: Exception) -> CustomRecognition.AnalyzeResult:
    return CustomRecognition.AnalyzeResult(
        box=None,
        detail={
            "planner_ready": False,
            "error": type(exc).__name__,
            "message": str(exc),
        },
    )


@AgentServer.custom_recognition("BlackFlow.MapRecognize")
class BlackFlowMapRecognition(CustomRecognition):
    def __init__(self) -> None:
        super().__init__()
        config = RecognitionConfig.load(
            PROJECT_ROOT / "config" / "recognition.default.json"
        )
        self._recognizer = MapRecognizer(
            config,
            PRIVATE_ROOT / "node-templates",
        )

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        del context
        try:
            result, _ = self._recognizer.analyze(argv.image)
            detail = result.to_dict()
            detail["source"] = "maafw-custom-recognition"
            detail["custom_param"] = _safe_json(argv.custom_recognition_param)
            return CustomRecognition.AnalyzeResult(
                box=_primary_box(result),
                detail=detail,
            )
        except (RecognitionError, ValueError) as exc:
            return _error_result(exc)


@AgentServer.custom_recognition("BlackFlow.PartRecognize")
class BlackFlowPartRecognition(CustomRecognition):
    def __init__(self) -> None:
        super().__init__()
        config = PartRecognitionConfig.load(
            PROJECT_ROOT / "config" / "parts.default.json"
        )
        self._recognizer = PartRecognizer(
            config,
            PRIVATE_ROOT / "part-templates",
        )

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        del context
        try:
            result = self._recognizer.analyze(argv.image)
            detail = result.to_dict()
            detail["source"] = "maafw-custom-recognition"
            detail["custom_param"] = _safe_json(argv.custom_recognition_param)
            roi = result.panel_roi
            return CustomRecognition.AnalyzeResult(
                box=[roi.x, roi.y, roi.width, roi.height],
                detail=detail,
            )
        except (RecognitionError, ValueError) as exc:
            return _error_result(exc)


def _safe_json(value: str) -> Any:
    try:
        return json.loads(value) if value else None
    except json.JSONDecodeError:
        return value


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: agent_main.py <agent-socket-identifier>")
        return 2
    if not AgentServer.start_up(sys.argv[1]):
        print("failed to start MaaFramework AgentServer")
        return 1
    AgentServer.join()
    AgentServer.shut_down()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
