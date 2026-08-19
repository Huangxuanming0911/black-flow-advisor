from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
from statistics import fmean, pstdev
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
LEGACY_DEFAULT_DIFFICULTY = 6
MIN_FALLBACK_SAMPLES = 2


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


def _record_difficulty(record: dict[str, Any]) -> int:
    raw = record.get("difficulty")
    if raw in (None, ""):
        return LEGACY_DEFAULT_DIFFICULTY
    return int(raw)


def _is_clean_base_sample(record: dict[str, Any]) -> bool:
    return (
        record.get("review_status") == "confirmed"
        and record.get("eligible_for_base_statistics") is True
        and record.get("bonus_source") == "none"
        and record.get("command_xp") is not None
        and bool(str(record.get("stage_name", "")).strip())
    )


def _confidence(sample_count: int) -> tuple[str, str, float]:
    if sample_count >= 10:
        return "moderate_high", "中高", 0.95
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
            "standard_deviation": round(pstdev(values), 4),
            "nonzero_rate": round(
                sum(value > 0 for value in values) / len(values),
                4,
            ),
            "observations": len(values),
        }
    return rewards


def _profile(
    records: list[dict[str, Any]],
    *,
    difficulty: int | None,
    floor: int | None,
    location_context: str,
    node_kind: str,
    fallback_scope: str = "exact",
) -> dict[str, Any]:
    confidence, confidence_zh, base_weight = _confidence(len(records))
    scope_weight = {
        "exact": 1.0,
        "same_floor_cross_difficulty": 0.85,
        "same_difficulty_cross_floor": 0.75,
        "global": 0.6,
    }[fallback_scope]
    weight = round(base_weight * scope_weight, 4)
    difficulty_id = "all" if difficulty is None else str(difficulty)
    floor_id = "all" if floor is None else str(floor)
    return {
        "id": (
            f"difficulty-{difficulty_id}:floor-{floor_id}:"
            f"{location_context}:{node_kind}"
        ),
        "difficulty": difficulty,
        "floor": floor,
        "location_context": location_context,
        "node_kind": node_kind,
        "sample_count": len(records),
        "confidence": confidence,
        "confidence_zh": confidence_zh,
        "confidence_weight": weight,
        "fallback_scope": fallback_scope,
        "cross_floor_fallback": floor is None,
        "cross_difficulty_fallback": difficulty is None,
        "difficulty_source_counts": {
            source: sum(
                1
                for record in records
                if str(record.get("difficulty_source") or "legacy_default_n6")
                == source
            )
            for source in sorted(
                {
                    str(
                        record.get("difficulty_source")
                        or "legacy_default_n6"
                    )
                    for record in records
                },
            )
        },
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
    groups: dict[
        tuple[int, int, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for record in clean:
        difficulty = _record_difficulty(record)
        floor = int(record["source_floor"])
        context = str(record.get("location_context", "main_map"))
        node_kind = NODE_KIND_ALIASES.get(
            str(record["combat_context"]),
            str(record["combat_context"]),
        )
        groups[(difficulty, floor, context, node_kind)].append(record)

    profiles = [
        _profile(
            rows,
            difficulty=difficulty,
            floor=floor,
            location_context=context,
            node_kind=node_kind,
        )
        for (
            difficulty,
            floor,
            context,
            node_kind,
        ), rows in sorted(groups.items())
    ]

    normalized: list[tuple[dict[str, Any], int, int, str, str]] = []
    for record in clean:
        normalized.append(
            (
                record,
                _record_difficulty(record),
                int(record["source_floor"]),
                str(record.get("location_context", "main_map")),
                NODE_KIND_ALIASES.get(
                    str(record["combat_context"]),
                    str(record["combat_context"]),
                ),
            ),
        )

    fallback_groups: dict[
        tuple[int | None, int | None, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for record, difficulty, floor, context, node_kind in normalized:
        fallback_groups[
            (None, floor, context, node_kind, "same_floor_cross_difficulty")
        ].append(record)
        fallback_groups[
            (difficulty, None, context, node_kind, "same_difficulty_cross_floor")
        ].append(record)
        fallback_groups[(None, None, context, node_kind, "global")].append(
            record,
        )
    for (
        difficulty,
        floor,
        context,
        node_kind,
        fallback_scope,
    ), rows in sorted(
        fallback_groups.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        if len(rows) < MIN_FALLBACK_SAMPLES:
            continue
        profiles.append(
            _profile(
                rows,
                difficulty=difficulty,
                floor=floor,
                location_context=context,
                node_kind=node_kind,
                fallback_scope=fallback_scope,
            ),
        )

    return {
        "schema_version": "0.2.0",
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
            "legacy_default_difficulty": LEGACY_DEFAULT_DIFFICULTY,
            "legacy_default_sample_count": sum(
                record.get("difficulty") in (None, "") for record in clean
            ),
            "observed_difficulties": sorted(
                {
                    _record_difficulty(record)
                    for record in clean
                },
            ),
            "confidence_note": (
                "样本不足5条时只作为推荐先验；按难度和层数优先匹配，"
                "稀疏组向同层跨难度或同难度跨层样本收缩。"
            ),
        },
        "profiles": sorted(profiles, key=lambda item: item["id"]),
    }


def main() -> int:
    args = _parser().parse_args()
    payload = build_payload(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"wrote {len(payload['profiles'])} profiles from "
        f"{payload['sample_policy']['included']} clean samples to {args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
