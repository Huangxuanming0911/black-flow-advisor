from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .path_ui import (
    DirectPathUiRecognizer,
    PathUiResult,
    UndirectedPathEdge,
    annotate_path_ui,
)
from .runtime import StableCapture
from .screen import ScreenState


@dataclass(frozen=True, slots=True)
class LivePathSnapshot:
    result: PathUiResult
    path_mask: np.ndarray
    skeleton: np.ndarray
    annotated: np.ndarray
    recognition_ms: float
    captured_unix_ms: int
    source_transform: dict[str, int]
    temporal_samples: int
    edge_vote_samples: int

    def to_dict(self) -> dict:
        return {
            "schema_version": "0.1.0",
            "captured_unix_ms": self.captured_unix_ms,
            "recognition_ms": round(self.recognition_ms, 2),
            "read_only": True,
            "planner_ready": False,
            "graph_scope": "all_visible_nodes_geometry",
            "temporal_samples": self.temporal_samples,
            "edge_vote_samples": self.edge_vote_samples,
            "source_transform": self.source_transform,
            "path_ui": self.result.to_dict(),
        }


class RealtimePathProcessor:
    """Run the direct path recognizer only on stable map captures."""

    def __init__(
        self,
        recognizer: DirectPathUiRecognizer | None = None,
    ) -> None:
        self.recognizer = recognizer or DirectPathUiRecognizer()

    def process(
        self,
        capture: StableCapture,
    ) -> LivePathSnapshot | None:
        if capture.state.state != ScreenState.MAP:
            return None
        started = time.perf_counter()
        temporal_frames = capture.temporal_frames
        if len(temporal_frames) >= 2:
            recognizers = [self.recognizer] + [
                DirectPathUiRecognizer(map_roi=self.recognizer.map_roi)
                for _ in temporal_frames
            ]
            frames = [capture.frame, *temporal_frames]
            with ThreadPoolExecutor(max_workers=len(frames)) as executor:
                outputs = list(
                    executor.map(
                        lambda pair: pair[0].analyze(pair[1]),
                        zip(recognizers, frames),
                    )
                )
            result, path_mask, skeleton = outputs[0]
            voted_edges = _vote_edges(
                tuple(output[0] for output in outputs[1:]),
                {node.id for node in result.nodes},
            )
            result = replace(result, edges=voted_edges)
            edge_vote_samples = len(temporal_frames)
        else:
            result, path_mask, skeleton = self.recognizer.analyze(
                capture.frame
            )
            edge_vote_samples = 1
        elapsed_ms = (time.perf_counter() - started) * 1000
        transform = capture.transform
        return LivePathSnapshot(
            result=result,
            path_mask=path_mask,
            skeleton=skeleton,
            annotated=annotate_path_ui(
                capture.frame,
                result,
                path_mask,
                skeleton,
            ),
            recognition_ms=elapsed_ms,
            captured_unix_ms=round(time.time() * 1000),
            source_transform={
                "source_width": transform.source_width,
                "source_height": transform.source_height,
                "client_top": transform.client_top,
                "client_height": transform.client_height,
                "padded_bottom": transform.padded_bottom,
            },
            temporal_samples=capture.temporal_samples,
            edge_vote_samples=edge_vote_samples,
        )


class LatestPathSnapshotWriter:
    """Atomically publish the latest live path artifacts for a UI consumer."""

    def __init__(self, output: str | Path) -> None:
        self.output = Path(output)

    def write(
        self,
        snapshot: LivePathSnapshot,
        window_title: str | None = None,
    ) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        payload = snapshot.to_dict()
        if window_title is not None:
            payload["window_title"] = window_title
        _atomic_write_image(
            self.output / "latest-path-mask.png",
            snapshot.path_mask,
        )
        _atomic_write_image(
            self.output / "latest-path-skeleton.png",
            snapshot.skeleton,
        )
        _atomic_write_image(
            self.output / "latest-path-annotated.png",
            snapshot.annotated,
        )
        # Publish metadata last so readers never observe a new state pointing
        # at images that have not been replaced yet.
        _atomic_write_json(
            self.output / "latest-path-state.json",
            payload,
        )


def _vote_edges(
    results: tuple[PathUiResult, ...],
    allowed_node_ids: set[str],
) -> tuple[UndirectedPathEdge, ...]:
    if not results:
        return ()
    required_votes = len(results) // 2 + 1
    votes: Counter[tuple[str, str]] = Counter()
    observations: dict[
        tuple[str, str], list[UndirectedPathEdge]
    ] = defaultdict(list)
    for result in results:
        seen: set[tuple[str, str]] = set()
        for edge in result.edges:
            key = tuple(sorted((edge.first, edge.second)))
            if (
                key[0] not in allowed_node_ids
                or key[1] not in allowed_node_ids
                or key in seen
            ):
                continue
            seen.add(key)
            votes[key] += 1
            observations[key].append(edge)

    accepted: list[UndirectedPathEdge] = []
    for key in sorted(votes):
        if votes[key] < required_votes:
            continue
        samples = observations[key]
        vote_ratio = votes[key] / len(results)
        confidence = (
            sum(edge.confidence for edge in samples) / len(samples)
        )
        accepted.append(
            UndirectedPathEdge(
                first=key[0],
                second=key[1],
                confidence=round(
                    min(0.98, confidence * (0.8 + 0.2 * vote_ratio)),
                    4,
                ),
                component_pixels=round(
                    sum(edge.component_pixels for edge in samples)
                    / len(samples)
                ),
            )
        )
    return tuple(accepted)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_image(path: Path, image: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"unable to write live artifact: {temporary}")
    temporary.replace(path)
