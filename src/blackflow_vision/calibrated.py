from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class UnknownCalibrationImageError(ValueError):
    pass


class CalibratedSceneRecognizer:
    """Exact-hash recognizer for reviewed calibration screenshots.

    This provides deterministic regression fixtures and acceptance data. It is
    intentionally separate from the general CV recognizer.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        annotations_path: str | Path,
    ) -> None:
        manifest_file = Path(manifest_path)
        annotations_file = Path(annotations_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        annotations = json.loads(
            annotations_file.read_text(encoding="utf-8")
        )
        scenes_by_file = {
            Path(scene["image"]).name: scene
            for scene in annotations["frames"]
        }
        self._scenes_by_hash: dict[str, dict[str, Any]] = {}
        for item in manifest["images"]:
            scene = scenes_by_file.get(item["file"])
            if scene is not None:
                self._scenes_by_hash[item["sha256"].lower()] = scene

    def recognize_file(self, path: str | Path) -> dict[str, Any]:
        image_path = Path(path)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        scene = self._scenes_by_hash.get(digest)
        if scene is None:
            raise UnknownCalibrationImageError(
                f"image hash is not in the calibrated set: {digest}"
            )
        return {
            **scene,
            "source_sha256": digest,
            "recognition_mode": "calibrated_exact_hash",
            "planner_ready": False,
            "acceptance_required": True,
        }
