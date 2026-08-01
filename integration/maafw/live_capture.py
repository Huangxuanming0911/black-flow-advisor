"""Read-only Win32 capture loop backed by official MaaFramework."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import maa
from maa.controller import Win32Controller
from maa.define import (
    MaaWin32InputMethodEnum,
    MaaWin32ScreencapMethodEnum,
)
from maa.library import Library
from maa.toolkit import Toolkit

from blackflow_vision.live_path import (
    LatestPathSnapshotWriter,
    RealtimePathProcessor,
)
from blackflow_vision.runtime import RealtimeRecognitionLoop, StableCapture
from blackflow_vision.screen import ScreenState


class MaaControllerFrameSource:
    def __init__(self, controller: Win32Controller) -> None:
        self.controller = controller

    def capture(self):
        job = self.controller.post_screencap().wait()
        if not job.succeeded:
            raise RuntimeError("MaaFramework screenshot failed")
        return job.get()


def _open_maa() -> None:
    package_root = Path(maa.__file__).resolve().parent
    Library.open(package_root / "bin")


def _find_window(title: str) -> Any:
    exact = [
        window
        for window in Toolkit.find_desktop_windows()
        if window.window_name == title
    ]
    if len(exact) == 1:
        return exact[0]
    partial = [
        window
        for window in Toolkit.find_desktop_windows()
        if title.casefold() in window.window_name.casefold()
    ]
    if len(partial) != 1:
        choices = [window.window_name for window in partial]
        raise RuntimeError(
            f"expected one window matching {title!r}, found {choices!r}"
        )
    return partial[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blackflow-live-capture")
    parser.add_argument("--window-title", default="明日方舟")
    parser.add_argument("--output", type=Path, default=Path("data/output/live"))
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument(
        "--idle-interval",
        type=float,
        default=1.0,
        help="capture interval after the game image remains unchanged",
    )
    parser.add_argument(
        "--active-window",
        type=float,
        default=1.75,
        help="seconds of fast polling after a visible frame change",
    )
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="with --once, wait until the stable screen is a map",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    _open_maa()
    try:
        window = _find_window(args.window_title)
    except RuntimeError as exc:
        print(
            json.dumps(
                {
                    "connected": False,
                    "error": str(exc),
                    "hint": "open the PC client and retry",
                },
                ensure_ascii=False,
            )
        )
        return 2
    controller = Win32Controller(
        window.hwnd,
        screencap_method=MaaWin32ScreencapMethodEnum.Background,
        mouse_method=MaaWin32InputMethodEnum.Seize,
        keyboard_method=MaaWin32InputMethodEnum.Seize,
    )
    controller.set_screenshot_target_long_side(1280)
    connection = controller.post_connection().wait()
    if not connection.succeeded:
        raise RuntimeError("MaaFramework could not connect to the game window")

    args.output.mkdir(parents=True, exist_ok=True)
    path_processor = RealtimePathProcessor()
    path_writer = LatestPathSnapshotWriter(args.output)

    def on_capture(capture: StableCapture) -> None:
        cv2.imwrite(str(args.output / "latest.png"), capture.frame)
        path_snapshot = path_processor.process(capture)
        if path_snapshot is not None:
            path_writer.write(path_snapshot, window.window_name)
            path_recognition = {
                "status": "recognized",
                "recognition_ms": round(
                    path_snapshot.recognition_ms,
                    2,
                ),
                "forest_nodes": len(
                    path_snapshot.result.forest_nodes
                ),
                "visible_nodes": len(path_snapshot.result.nodes),
                "semantic_nodes": sum(
                    node.kind != "forest"
                    for node in path_snapshot.result.nodes
                ),
                "candidate_edges": len(path_snapshot.result.edges),
                "temporal_samples": capture.temporal_samples,
                "edge_vote_samples": path_snapshot.edge_vote_samples,
                "planner_ready": False,
                "graph_scope": "all_visible_nodes_geometry",
            }
        else:
            path_recognition = {
                "status": "skipped_non_map",
                "planner_ready": False,
            }
        state = {
            "screen_state": str(capture.state.state),
            "confidence": capture.state.confidence,
            "evidence": capture.state.evidence,
            "captured_unix_ms": round(time.time() * 1000),
            "window_title": window.window_name,
            "read_only": True,
            "capture_policy": {
                "mode": "adaptive",
                "active_interval_ms": round(max(0.05, args.interval) * 1000),
                "idle_interval_ms": round(
                    max(args.interval, args.idle_interval) * 1000
                ),
            },
            "path_recognition": path_recognition,
        }
        (args.output / "latest-state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(state, ensure_ascii=False), flush=True)

    loop = RealtimeRecognitionLoop(
        MaaControllerFrameSource(controller),
        on_capture,
        required_stable_frames=args.stable_frames,
        active_window_seconds=args.active_window,
    )
    while True:
        capture = loop.poll_once()
        if args.once and capture is not None:
            if (
                not args.map_only
                or capture.state.state == ScreenState.MAP
            ):
                return 0
        time.sleep(
            loop.recommended_interval(
                active_interval=args.interval,
                idle_interval=args.idle_interval,
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
