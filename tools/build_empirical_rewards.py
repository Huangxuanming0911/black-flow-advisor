from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = (
    PROJECT_ROOT.parent
    / "black-flow-reward-collector"
    / "data"
    / "records"
    / "rewards.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "empirical-node-rewards.v0.1.json"
)

REWARD_FIELDS = {
    "originium_ingots": "normal_reward_ingots",
    "command_xp": "command_xp",
    "collectibles": "collectibles",
    "recruitment_tickets": "recruitment_tickets",
    "parts": "parts",
    "target_life": "target_life",
}
NODE_KIND_ALIASES = {
    "boss": "enemy",
}
FLOOR_FALLBACK_KINDS = {"encounter"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build route-planner reward priors from reviewed collector records."
        ),
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_clean_base_sample(record: dict[str, Any]) -> bool:
    return (
        record.get("review_status") == "confirmed"
        and record.get("eligible_for_base_statistics") is True
        and record.get("bonus_source") == "none"
        and record.get("command_xp") is not None
        and bool(str(record.get("stage_name", "")).strip())
    )


def _confidence(sample_count: int) -> tuple[str, str, float]:
    if sample_count >= 5:
        return "moderate", "中等", 0.9
    if sample_count >= 3:
        return "preliminary_stable", "初步稳定", 0.78
    if sample_count == 2:
        return "preliminary", "初步", 0.65
    return "anecdotal", "单例", 0.45


def _reward_stats(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    rewards: dict[str, dict[str, int | float]] = {}
    for dimension, field in REWARD_FIELDS.items():
        values = [
            float(record[field])
            for record in records
            if record.get(field) is not None
        ]
        if not values:
            continue
        mean = fmean(values)
        rewards[dimension] = {
            "minimum": int(min(values)),
            "maximum": int(max(values)),
            "expected": round(mean, 4),
            "observations": len(values),
        }
    return rewards


def _profile(
    records: list[dict[str, Any]],
    *,
    floor: int | None,
    location_context: str,
    node_kind: str,
    fallback: bool = False,
) -> dict[str, Any]:
    confidence, confidence_zh, weight = _confidence(len(records))
    floor_id = "all" if floor is None else str(floor)
    return {
        "id": f"floor-{floor_id}:{location_context}:{node_kind}",
        "floor": floor,
        "location_context": location_context,
        "node_kind": node_kind,
        "sample_count": len(records),
        "confidence": confidence,
        "confidence_zh": confidence_zh,
        "confidence_weight": weight,
        "cross_floor_fallback": fallback,
        "command_xp_multipliers": sorted(
            {
                float(record.get("command_xp_multiplier", 1.0))
                for record in records
            },
        ),
        "stage_names": sorted(
            {str(record["stage_name"]) for record in records},
        ),
        "sample_ids": [str(record["sample_id"]) for record in records],
        "rewards": _reward_stats(records),
    }


def build_payload(records_path: Path) -> dict[str, Any]:
    records = _load_records(records_path)
    clean = [record for record in records if _is_clean_base_sample(record)]
    excluded = [
        str(record.get("sample_id", ""))
        for record in records
        if not _is_clean_base_sample(record)
    ]
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in clean:
        floor = int(record["source_floor"])
        context = str(record.get("location_context", "main_map"))
        node_kind = NODE_KIND_ALIASES.get(
            str(record["combat_context"]),
            str(record["combat_context"]),
        )
        groups[(floor, context, node_kind)].append(record)

    profiles = [
        _profile(
            rows,
            floor=floor,
            location_context=context,
            node_kind=node_kind,
        )
        for (floor, context, node_kind), rows in sorted(groups.items())
    ]

    for node_kind in sorted(FLOOR_FALLBACK_KINDS):
        rows = [
            record
            for record in clean
            if NODE_KIND_ALIASES.get(
                str(record["combat_context"]),
                str(record["combat_context"]),
            ) == node_kind
            and record.get("location_context", "main_map") == "main_map"
        ]
        if len(rows) >= 2:
            profiles.append(
                _profile(
                    rows,
                    floor=None,
                    location_context="main_map",
                    node_kind=node_kind,
                    fallback=True,
                ),
            )

    return {
        "schema_version": "0.1.0",
        "generated_at": date.today().isoformat(),
        "source": {
            "repository": "black-flow-reward-collector",
            "path": "data/records/rewards.jsonl",
            "snapshot_note": str(records_path.name),
        },
        "scope": (
            "人工确认且无额外奖励来源的战后基础结算；源石锭使用正常奖励栏，"
            "零件不含藏品附带零件，希望暂不纳入。"
        ),
        "sample_policy": {
            "included": len(clean),
            "excluded": len(excluded),
            "excluded_sample_ids": excluded,
            "minimum_stable_samples": 5,
            "confidence_note": (
                "样本不足5条时只作为推荐先验；显示原始均值，排序时按置信权重折扣。"
            ),
        },
        "profiles": sorted(profiles, key=lambda item: item["id"]),
    }


def main() -> int:
    args = _parser().parse_args()
    payload = build_payload(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\r\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"wrote {len(payload['profiles'])} profiles from "
        f"{payload['sample_policy']['included']} clean samples to {args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
