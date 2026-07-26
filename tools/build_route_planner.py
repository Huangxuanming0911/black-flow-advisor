from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _portable_graph(graph: dict) -> dict:
    return {
        "schema_version": graph.get("schema_version"),
        "nodes": [
            {
                "node_id": node["node_id"],
                "center": node["center"],
                "label": node.get("label", node["node_id"]),
                "kind": node.get("kind", "unknown"),
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


def main() -> int:
    args = _parser().parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    knowledge = json.loads(args.knowledge.read_text(encoding="utf-8"))
    source_image = _resolve_source_image(graph, args.image)
    template_path = PROJECT_ROOT / "web" / "route-planner.html"
    template = template_path.read_text(encoding="utf-8")

    data = {
        "graph": _portable_graph(graph),
        "movement_modes": knowledge["movement_modes"],
        "parts": knowledge["parts"]["items"],
        "initial_action_points": max(0, args.initial_action_points),
        "source": {
            "graph": str(args.graph),
            "image": str(source_image),
            "knowledge": str(args.knowledge),
        },
        "image_data": _image_data_uri(source_image),
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
    (args.output / "planner-data.json").write_text(
        json.dumps(portable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
