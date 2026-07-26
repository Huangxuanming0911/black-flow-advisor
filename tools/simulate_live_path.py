from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blackflow_vision.live_path import (
    LatestPathSnapshotWriter,
    RealtimePathProcessor,
)
from blackflow_vision.runtime import RealtimeRecognitionLoop, StableCapture


class StaticFrameSource:
    def __init__(self, image) -> None:
        self.image = image

    def capture(self):
        return self.image.copy()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-frames", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    raw = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"unable to read {args.image}")
    processor = RealtimePathProcessor()
    writer = LatestPathSnapshotWriter(args.output)
    published = []

    def on_capture(capture: StableCapture) -> None:
        snapshot = processor.process(capture)
        if snapshot is None:
            return
        writer.write(snapshot, "offline-live-simulation")
        cv2.imwrite(str(args.output / "latest.png"), capture.frame)
        published.append(snapshot)

    loop = RealtimeRecognitionLoop(
        StaticFrameSource(raw),
        on_capture,
        required_stable_frames=args.stable_frames,
    )
    for _ in range(args.stable_frames):
        loop.poll_once()
    if not published:
        print(
            json.dumps(
                {
                    "published": False,
                    "reason": "frame did not stabilize as a map",
                },
                ensure_ascii=False,
            )
        )
        return 1
    snapshot = published[-1]
    print(
        json.dumps(
            {
                "published": True,
                "recognition_ms": round(snapshot.recognition_ms, 2),
                "forest_nodes": len(snapshot.result.forest_nodes),
                "visible_nodes": len(snapshot.result.nodes),
                "candidate_edges": len(snapshot.result.edges),
                "planner_ready": snapshot.result.planner_ready,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
