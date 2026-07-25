from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RecognitionConfig:
    target_width: int = 1280
    target_height: int = 720
    strict_size: bool = True
    map_roi: tuple[int, int, int, int] = (80, 70, 1000, 570)
    node_radius_min: int = 13
    node_radius_max: int = 34
    node_min_distance: int = 42
    hough_dp: float = 1.2
    hough_param1: float = 100
    hough_param2: float = 24
    axis_cluster_tolerance: int = 14
    node_crop_size: int = 56
    node_template_threshold: float = 0.82
    road_value_min: int = 145
    road_saturation_max: int = 105
    road_half_width: int = 5
    road_endpoint_margin: int = 24
    road_occupancy_threshold: float = 0.42
    minimum_node_confidence: float = 0.70
    minimum_edge_confidence: float = 0.80

    @classmethod
    def load(cls, path: str | Path) -> "RecognitionConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "map_roi" in data:
            data["map_roi"] = tuple(data["map_roi"])
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        x, y, width, height = self.map_roi
        if min(x, y) < 0 or min(width, height) <= 0:
            raise ValueError("map_roi must be a positive rectangle")
        if x + width > self.target_width or y + height > self.target_height:
            raise ValueError("map_roi extends beyond target dimensions")
        if not 0 < self.node_template_threshold <= 1:
            raise ValueError("node_template_threshold must be in (0, 1]")
        if not 0 < self.road_occupancy_threshold <= 1:
            raise ValueError("road_occupancy_threshold must be in (0, 1]")

