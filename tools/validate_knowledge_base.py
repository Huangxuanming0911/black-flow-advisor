from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PATH = Path("data/knowledge/black-flow-rules.v0.1.json")


def validate(payload: dict) -> list[str]:
    issues: list[str] = []
    source_ids = {source["id"] for source in payload.get("sources", [])}
    if len(source_ids) != len(payload.get("sources", [])):
        issues.append("duplicate_source_id")

    node_types = payload.get("node_types", [])
    node_ids = {node["id"] for node in node_types}
    if len(node_ids) != len(node_types):
        issues.append("duplicate_node_id")
    bridge = payload.get("vision_bridge", {})
    allowed_bridge_ids = {*node_ids, "current_position"}
    for name, rule_id in bridge.get(
        "node_name_to_rule_id",
        {},
    ).items():
        if rule_id not in allowed_bridge_ids:
            issues.append(f"unknown_bridge_node:{name}:{rule_id}")
    for kind, candidates in bridge.get(
        "vision_kind_candidates",
        {},
    ).items():
        for rule_id in candidates:
            if rule_id not in allowed_bridge_ids:
                issues.append(
                    f"unknown_bridge_candidate:{kind}:{rule_id}"
                )

    parts = payload.get("parts", {}).get("items", [])
    part_ids = {part["id"] for part in parts}
    if len(part_ids) != len(parts):
        issues.append("duplicate_part_id")

    movement_modes = payload.get("movement_modes", [])
    movement_ids = {mode["id"] for mode in movement_modes}
    if len(movement_ids) != len(movement_modes):
        issues.append("duplicate_movement_id")

    for part in parts:
        movement = part.get("movement_mode")
        if movement and movement not in movement_ids:
            issues.append(
                f"unknown_movement_mode:{part['id']}:{movement}"
            )

    endings = payload.get("endings", [])
    ending_ids = {ending["id"] for ending in endings}
    if len(ending_ids) != len(endings):
        issues.append("duplicate_ending_id")

    source_users: list[tuple[str, list[str]]] = []
    source_users.extend(
        (f"node:{node['id']}", node.get("sources", []))
        for node in node_types
    )
    source_users.extend(
        (f"ending:{ending['id']}", ending.get("sources", []))
        for ending in endings
    )
    for owner, references in source_users:
        for source_id in references:
            if source_id not in source_ids:
                issues.append(f"unknown_source:{owner}:{source_id}")

    normal_stages = payload.get("combat_catalog", {}).get(
        "normal_and_emergency_by_region",
        {},
    )
    flat_stages = [
        stage for stages in normal_stages.values() for stage in stages
    ]
    if len(flat_stages) != 31:
        issues.append(
            f"unexpected_normal_stage_count:{len(flat_stages)}"
        )
    if len(set(flat_stages)) != len(flat_stages):
        issues.append("duplicate_normal_stage")

    if len(parts) != 30:
        issues.append(f"unexpected_part_count:{len(parts)}")
    if len(node_types) != 20:
        issues.append(f"unexpected_node_type_count:{len(node_types)}")
    if len(endings) != 3:
        issues.append(f"unexpected_ending_count:{len(endings)}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_PATH)
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    issues = validate(payload)
    summary = {
        "valid": not issues,
        "issues": issues,
        "node_types": len(payload["node_types"]),
        "parts": len(payload["parts"]["items"]),
        "movement_modes": len(payload["movement_modes"]),
        "endings": len(payload["endings"]),
        "normal_stages": sum(
            len(stages)
            for stages in payload["combat_catalog"][
                "normal_and_emergency_by_region"
            ].values()
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
