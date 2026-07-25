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

from blackflow_vision.runtime import RealtimeRecognitionLoop, StableCapture


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
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    _open_maa()
    window = _find_window(args.window_title)
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

    def on_capture(capture: StableCapture) -> None:
        cv2.imwrite(str(args.output / "latest.png"), capture.frame)
        state = {
            "screen_state": str(capture.state.state),
            "confidence": capture.state.confidence,
            "evidence": capture.state.evidence,
            "captured_unix_ms": round(time.time() * 1000),
            "window_title": window.window_name,
            "read_only": True,
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
    )
    while True:
        capture = loop.poll_once()
        if args.once and capture is not None:
            return 0
        time.sleep(max(0.05, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
