from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PARTS = [
    {
        "part_id": "scrap_wheel",
        "uses": 1,
        "estimated_value": 1,
    },
    {
        "part_id": "structural_principle",
        "uses": 3,
        "estimated_value": 1,
    },
    {
        "part_id": "heavy_spring",
        "uses": 1,
        "estimated_value": 2,
    },
    {
        "part_id": "blood_mushroom",
        "uses": None,
        "estimated_value": 3,
    },
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local interactive route simulation report.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "output"
        / "node-semantics"
        / "unified-map-graph.json",
    )
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "knowledge"
        / "black-flow-rules.v0.1.json",
    )
    parser.add_argument(
        "--reward-knowledge",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "knowledge"
        / "node-rewards.v0.1.json",
    )
    parser.add_argument(
        "--empirical-rewards",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "knowledge"
        / "empirical-node-rewards.v0.1.json",
        help="Reviewed post-battle reward samples used as route priors.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Override the source screenshot recorded in the graph JSON.",
    )
    parser.add_argument(
        "--initial-action-points",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "route-planner",
    )
    parser.add_argument(
        "--no-sample-parts",
        action="store_true",
        help="Start with an empty part box instead of demo inventory.",
    )
    return parser


def _image_data_uri(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read source image: {path}")
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 76],
    )
    if not success:
        raise RuntimeError(f"unable to encode source image: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _node_icon_data_uris(path: Path, graph: dict) -> dict[str, str]:
    """Crop compact game-native node icons from the recognized source frame.

    The crops are embedded only in the generated local planner. Raw screenshots
    and derived icon files are never added to the repository.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read source image: {path}")
    height, width = image.shape[:2]
    scale_x = width / 1280
    scale_y = height / 720
    nodes = graph.get("nodes", ())
    distances: list[float] = []
    for index, node in enumerate(nodes):
        x1, y1 = map(float, node["center"])
        for other in nodes[index + 1 :]:
            x2, y2 = map(float, other["center"])
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if 40 <= distance <= 180:
                distances.append(distance)
    spacing = float(np.median(distances)) if distances else 100.0
    radius = max(24.0, min(30.0, spacing * 0.26))
    icons: dict[str, str] = {}
    for node in nodes:
        if node.get("kind") == "forest":
            continue
        center_x = float(node["center"][0]) * scale_x
        center_y = float(node["center"][1]) * scale_y
        radius_x = radius * scale_x
        radius_y = radius * scale_y
        left = max(0, round(center_x - radius_x))
        right = min(width, round(center_x + radius_x))
        top = max(0, round(center_y - radius_y))
        bottom = min(height, round(center_y + radius_y))
        if right - left < 8 or bottom - top < 8:
            continue
        crop = cv2.resize(
            image[top:bottom, left:right],
            (72, 72),
            interpolation=cv2.INTER_AREA,
        )
        alpha = np.zeros((72, 72), dtype=np.uint8)
        # Shift the visible mask upward so the node caption below the icon is
        # excluded while the original circular artwork remains recognizable.
        cv2.ellipse(alpha, (36, 32), (30, 27), 0, 0, 360, 255, thickness=-1)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        icon = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        icon[:, :, 3] = alpha
        success, encoded = cv2.imencode(".png", icon)
        if not success:
            continue
        icons[node["node_id"]] = (
            "data:image/png;base64,"
            + base64.b64encode(encoded).decode("ascii")
        )
    return icons


def _resolve_source_image(
    graph: dict,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    raw = graph.get("source_image")
    if not raw:
        raise RuntimeError(
            "graph has no source_image; pass --image explicitly",
        )
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _portable_graph(graph: dict, label_kinds: dict[str, str]) -> dict:
    return {
        "schema_version": graph.get("schema_version"),
        "nodes": [
            {
                "node_id": node["node_id"],
                "center": node["center"],
                "label": node.get("label", node["node_id"]),
                "kind": label_kinds.get(
                    str(node.get("label", "")).strip(),
                    node.get("kind", "unknown"),
                ),
                "confidence": node.get("confidence"),
            }
            for node in graph.get("nodes", ())
        ],
        "edges": [
            {
                "first": edge["first"],
                "second": edge["second"],
                "confidence": edge.get("confidence"),
            }
            for edge in graph.get("edges", ())
        ],
    }


def _infer_floor(graph: dict, source_image: Path) -> int:
    raw_floor = graph.get("floor") or graph.get("source_floor")
    if raw_floor is not None:
        return max(1, min(6, int(raw_floor)))
    search_text = " ".join(
        (
            str(graph.get("source_image", "")),
            source_image.name,
        ),
    )
    match = re.search(
        r"(?:layer|floor|层)[-_ ]*0?([1-6])",
        search_text,
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else 3


def main() -> int:
    args = _parser().parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    knowledge = json.loads(args.knowledge.read_text(encoding="utf-8"))
    reward_knowledge = json.loads(
        args.reward_knowledge.read_text(encoding="utf-8"),
    )
    empirical_rewards = json.loads(
        args.empirical_rewards.read_text(encoding="utf-8"),
    )
    source_image = _resolve_source_image(graph, args.image)
    label_kinds = knowledge.get("vision_bridge", {}).get(
        "node_name_to_rule_id",
        {},
    )
    template_path = PROJECT_ROOT / "web" / "route-planner.html"
    template = template_path.read_text(encoding="utf-8")

    data = {
        "graph": _portable_graph(graph, label_kinds),
        "movement_modes": knowledge["movement_modes"],
        "parts": knowledge["parts"]["items"],
        "reward_knowledge": reward_knowledge,
        "empirical_rewards": empirical_rewards,
        "resource_lifecycle_policy": knowledge.get(
            "resource_lifecycle_policy",
            {},
        ),
        "floor": _infer_floor(graph, source_image),
        "location_context": "main_map",
        "sample_parts": [] if args.no_sample_parts else SAMPLE_PARTS,
        "initial_action_points": max(0, args.initial_action_points),
        "source": {
            "graph": str(args.graph),
            "image": str(source_image),
            "knowledge": str(args.knowledge),
            "reward_knowledge": str(args.reward_knowledge),
            "empirical_rewards": str(args.empirical_rewards),
        },
        "image_data": _image_data_uri(source_image),
        "node_icons": _node_icon_data_uris(source_image, graph),
    }
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    if template.count("__BLACKFLOW_DATA__") != 1:
        raise RuntimeError("route planner template placeholder is invalid")
    html = template.replace("__BLACKFLOW_DATA__", encoded)

    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "index.html"
    output.write_text(html, encoding="utf-8")
    portable = dict(data)
    portable.pop("image_data")
    portable.pop("node_icons")
    (args.output / "planner-data.json").write_text(
        json.dumps(portable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
