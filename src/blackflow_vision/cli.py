from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .config import RecognitionConfig
from .calibrated import (
    CalibratedSceneRecognizer,
    UnknownCalibrationImageError,
)
from .parts import (
    PartRecognitionConfig,
    PartRecognizer,
    annotate_parts,
)
from .path_ui import DirectPathUiRecognizer, annotate_path_ui
from .recognizer import MapRecognizer, RecognitionError
from .render import annotate
from .screen import classify_screen_state, normalize_pc_frame
from .synthetic import build_synthetic_map, build_synthetic_parts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blackflow-recognize")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthesize = subparsers.add_parser("synthesize")
    synthesize.add_argument("output", type=Path)

    synthesize_parts = subparsers.add_parser("synthesize-parts")
    synthesize_parts.add_argument("output", type=Path)
    synthesize_parts.add_argument("--templates", type=Path)

    recognize = subparsers.add_parser("recognize-map")
    recognize.add_argument("image", type=Path)
    recognize.add_argument("--config", type=Path, required=True)
    recognize.add_argument("--output", type=Path, required=True)
    recognize.add_argument("--templates", type=Path)
    recognize.add_argument(
        "--pc-frame",
        action="store_true",
        help="crop Windows chrome and normalize the game client to 1280x720",
    )

    recognize_parts = subparsers.add_parser("recognize-parts")
    recognize_parts.add_argument("image", type=Path)
    recognize_parts.add_argument("--config", type=Path, required=True)
    recognize_parts.add_argument("--output", type=Path, required=True)
    recognize_parts.add_argument("--templates", type=Path, required=True)

    recognize_path_ui = subparsers.add_parser("recognize-path-ui")
    recognize_path_ui.add_argument("image", type=Path)
    recognize_path_ui.add_argument("--output", type=Path, required=True)

    calibrated = subparsers.add_parser("recognize-calibrated")
    calibrated.add_argument("image", type=Path)
    calibrated.add_argument("--manifest", type=Path, required=True)
    calibrated.add_argument("--annotations", type=Path, required=True)
    calibrated.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "synthesize":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), build_synthetic_map()):
            raise SystemExit(f"failed to write {args.output}")
        print(args.output)
        return 0
    if args.command == "synthesize-parts":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image = build_synthetic_parts()
        if not cv2.imwrite(str(args.output), image):
            raise SystemExit(f"failed to write {args.output}")
        if args.templates:
            config_path = (
                Path(__file__).resolve().parents[2]
                / "config"
                / "parts.default.json"
            )
            config = PartRecognitionConfig.load(config_path)
            boxes = PartRecognizer(config, None).slot_boxes()
            for part_id, box in zip(("wheel", "spring", "engine"), boxes):
                part_dir = args.templates / part_id
                part_dir.mkdir(parents=True, exist_ok=True)
                crop = image[
                    box.y : box.y + box.height,
                    box.x : box.x + box.width,
                ]
                cv2.imwrite(str(part_dir / "synthetic.png"), crop)
        print(args.output)
        return 0

    if args.command == "recognize-parts":
        try:
            config = PartRecognitionConfig.load(args.config)
            image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image is None:
                raise RecognitionError(f"unable to read image: {args.image}")
            result = PartRecognizer(config, args.templates).analyze(image)
        except (OSError, ValueError, RecognitionError) as exc:
            print(f"recognition failed: {exc}")
            return 2
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "parts-state.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cv2.imwrite(
            str(args.output / "parts-annotated.png"),
            annotate_parts(image, result),
        )
        print(
            json.dumps(
                {
                    "parts": len(result.parts),
                    "planner_ready": result.planner_ready,
                    "issues": result.issues,
                    "output": str(args.output),
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.parts else 1

    if args.command == "recognize-path-ui":
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            print(f"recognition failed: unable to read image: {args.image}")
            return 2
        normalized, transform = normalize_pc_frame(image)
        state = classify_screen_state(normalized)
        if str(state.state) != "map":
            print(
                f"recognition failed: expected map screen, got {state.state}"
            )
            return 2
        result, path_mask, skeleton = DirectPathUiRecognizer().analyze(
            normalized
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "path-ui-state.json").write_text(
            json.dumps(
                {
                    **result.to_dict(),
                    "screen_state": str(state.state),
                    "screen_state_confidence": state.confidence,
                    "source_transform": {
                        "source_width": transform.source_width,
                        "source_height": transform.source_height,
                        "client_top": transform.client_top,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        cv2.imwrite(str(args.output / "path-mask.png"), path_mask)
        cv2.imwrite(str(args.output / "path-skeleton.png"), skeleton)
        cv2.imwrite(
            str(args.output / "path-ui-annotated.png"),
            annotate_path_ui(normalized, result, path_mask, skeleton),
        )
        print(
            json.dumps(
                {
                    "forest_nodes": len(result.forest_nodes),
                    "undirected_edges": len(result.edges),
                    "ambiguous_components": len(
                        result.ambiguous_components
                    ),
                    "line_evidence": result.line_evidence_count,
                    "planner_ready": result.planner_ready,
                    "output": str(args.output),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "recognize-calibrated":
        try:
            result = CalibratedSceneRecognizer(
                args.manifest,
                args.annotations,
            ).recognize_file(args.image)
        except (OSError, ValueError, UnknownCalibrationImageError) as exc:
            print(f"recognition failed: {exc}")
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "frame": result["id"],
                    "screen": result["screen"]["kind"],
                    "nodes": len(result.get("nodes", [])),
                    "edges": len(result.get("edges", [])),
                    "output": str(args.output),
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        config = RecognitionConfig.load(args.config)
        recognizer = MapRecognizer(config, args.templates)
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise RecognitionError(f"unable to read image: {args.image}")
        transform = None
        state = None
        if args.pc_frame:
            image, transform = normalize_pc_frame(image)
            state = classify_screen_state(image)
        result, mask = recognizer.analyze(image)
    except (OSError, ValueError, RecognitionError) as exc:
        print(f"recognition failed: {exc}")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "map-state.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cv2.imwrite(str(args.output / "annotated.png"), annotate(image, result))
    cv2.imwrite(str(args.output / "road-mask.png"), mask)
    if args.pc_frame:
        cv2.imwrite(str(args.output / "normalized.png"), image)
        (args.output / "capture-state.json").write_text(
            json.dumps(
                {
                    "screen_state": str(state.state),
                    "confidence": state.confidence,
                    "evidence": state.evidence,
                    "transform": {
                        "source_width": transform.source_width,
                        "source_height": transform.source_height,
                        "client_top": transform.client_top,
                        "client_height": transform.client_height,
                        "padded_bottom": transform.padded_bottom,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "nodes": len(result.nodes),
                "edges": len(result.edges),
                "planner_ready": result.planner_ready,
                "issues": result.issues,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.nodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
