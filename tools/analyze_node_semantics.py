from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackflow_vision.node_semantics import (  # noqa: E402
    LocalIconTemplateLibrary,
    NodeSemanticRecognizer,
    annotate_node_semantics,
)
from blackflow_vision.path_ui import (  # noqa: E402
    DirectPathUiRecognizer,
    annotate_path_ui,
)
from blackflow_vision.screen import normalize_pc_frame  # noqa: E402
from blackflow_vision.scene_graph import build_unified_map_graph  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze-node-semantics",
        description="Run offline node OCR and icon cross-validation.",
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=Path("data/output/live-full-node/latest.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/node-semantics"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "data/private/annotations/2026-07-26/"
            "recognized-scenes.json"
        ),
        help="optional private reviewed dataset used only for icon validation",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"unable to read image: {args.image}")
    normalized, transform = normalize_pc_frame(image)
    path_result, path_mask, skeleton = DirectPathUiRecognizer().analyze(
        normalized
    )
    templates = LocalIconTemplateLibrary.from_reviewed_annotations(
        args.annotations
    )
    semantics = NodeSemanticRecognizer(templates).analyze(
        normalized,
        path_result.nodes,
    )
    unified_graph = build_unified_map_graph(path_result, semantics)

    args.output.mkdir(parents=True, exist_ok=True)
    clean = annotate_node_semantics(normalized, semantics)
    graph = annotate_path_ui(
        normalized,
        path_result,
        path_mask,
        skeleton,
    )
    graph = annotate_node_semantics(graph, semantics)
    clean_graph = annotate_path_ui(
        normalized,
        path_result,
        np.zeros_like(path_mask),
        np.zeros_like(skeleton),
    )
    clean_graph = annotate_node_semantics(clean_graph, semantics)
    cv2.rectangle(clean_graph, (590, 72), (1025, 108), (0, 0, 0), -1)
    cv2.putText(
        clean_graph,
        (
            f"EDGES {len(unified_graph.edges)}  "
            f"COMPONENTS {len(unified_graph.connected_components)}  "
            f"ISOLATED {len(unified_graph.isolated_nodes)}"
        ),
        (600, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(
        str(args.output / "node-semantics-annotated.png"),
        clean,
    )
    cv2.imwrite(
        str(args.output / "node-semantics-graph.png"),
        graph,
    )
    cv2.imwrite(
        str(args.output / "unified-map-graph.png"),
        clean_graph,
    )
    payload = semantics.to_dict()
    payload.update(
        {
            "schema_version": "0.1.0",
            "source_image": str(args.image),
            "source_transform": {
                "source_width": transform.source_width,
                "source_height": transform.source_height,
                "client_top": transform.client_top,
                "client_height": transform.client_height,
                "padded_bottom": transform.padded_bottom,
            },
            "visible_edges": len(path_result.edges),
            "read_only": True,
        }
    )
    (args.output / "node-semantics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "unified-map-graph.json").write_text(
        json.dumps(
            {
                **unified_graph.to_dict(),
                "source_image": str(args.image),
                "source_transform": payload["source_transform"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "nodes": len(semantics.nodes),
                "recognized_labels": sum(
                    node.label not in {"林中节点", "当前位置", "未识别"}
                    for node in semantics.nodes
                ),
                "needs_review": sum(
                    node.needs_review for node in semantics.nodes
                ),
                "edges": len(unified_graph.edges),
                "connected_components": len(
                    unified_graph.connected_components
                ),
                "isolated_nodes": len(unified_graph.isolated_nodes),
                "ocr_elapsed_ms": semantics.ocr_elapsed_ms,
                "icon_templates": templates.available,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
