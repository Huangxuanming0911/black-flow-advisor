from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


RESOURCE_FIELDS = (
    "target_life",
    "target_life_cap",
    "shield",
    "hope",
    "hope_cap",
    "originium_ingots",
    "command_level",
    "command_xp",
    "action_points",
    "operator_capacity",
    "deployment_capacity",
    "part_box_capacity",
)


@dataclass(slots=True)
class AggregatedRunEffects:
    """Planner-facing effects after all inventory and status stacking rules."""

    node_scores: dict[str, float] = field(default_factory=dict)
    resource_multipliers: dict[str, float] = field(default_factory=dict)
    combat_risk: float = 0.0
    merchant_value_multiplier: float = 1.0
    bonus_parts_per_combat: float = 0.0
    information_risk: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_scores": self.node_scores,
            "resource_multipliers": self.resource_multipliers,
            "combat_risk": self.combat_risk,
            "merchant_value_multiplier": self.merchant_value_multiplier,
            "bonus_parts_per_combat": self.bonus_parts_per_combat,
            "information_risk": self.information_risk,
            "notes": self.notes,
        }


def new_run_state(*, floor: int = 1, action_points: int = 6) -> dict[str, Any]:
    """Create the stable exchange object shared by recognition and planning."""

    return {
        "schema_version": "0.1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": {"floor": floor, "checkpoint": "map"},
        "difficulty": {"confidentiality_level": 0},
        "resources": {
            "target_life": 8,
            "target_life_cap": 8,
            "shield": 0,
            "hope": 6,
            "hope_cap": None,
            "originium_ingots": 6,
            "command_level": 1,
            "command_xp": 0,
            "action_points": action_points,
            "operator_capacity": 6,
            "deployment_capacity": None,
            "part_box_capacity": 10,
        },
        "operators": [],
        "retained_tickets": [],
        "collectibles": [],
        "statuses": [],
        "observed_counts": {"collectibles_total": 0},
        "field_provenance": {},
        "checkpoints": [],
    }


def merge_observation(
    state: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    source: str = "ui_recognition",
) -> dict[str, Any]:
    """Merge one UI observation without overwriting explicit user corrections.

    A field corrected by the user stays authoritative until the next checkpoint.
    This makes frequent OCR safe while still allowing a small manual adjustment.
    """

    merged = deepcopy(dict(state))
    provenance = merged.setdefault("field_provenance", {})
    for group in ("phase", "difficulty", "resources"):
        incoming = observation.get(group)
        if not isinstance(incoming, Mapping):
            continue
        target = merged.setdefault(group, {})
        for key, value in incoming.items():
            path = f"{group}.{key}"
            if provenance.get(path, {}).get("source") == "manual":
                continue
            target[key] = value
            provenance[path] = {"source": source}
    for key in ("operators", "retained_tickets", "collectibles", "statuses"):
        incoming = observation.get(key)
        if isinstance(incoming, list) and provenance.get(key, {}).get(
            "source"
        ) != "manual":
            merged[key] = deepcopy(incoming)
            provenance[key] = {"source": source}
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    return merged


def apply_manual_corrections(
    state: Mapping[str, Any],
    corrections: Mapping[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(dict(state))
    provenance = merged.setdefault("field_provenance", {})
    for group in ("phase", "difficulty", "resources"):
        incoming = corrections.get(group)
        if not isinstance(incoming, Mapping):
            continue
        target = merged.setdefault(group, {})
        for key, value in incoming.items():
            target[key] = value
            provenance[f"{group}.{key}"] = {"source": "manual"}
    for key in ("operators", "retained_tickets", "collectibles", "statuses"):
        if key in corrections:
            merged[key] = deepcopy(corrections[key])
            provenance[key] = {"source": "manual"}
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    return merged


def add_checkpoint(
    state: Mapping[str, Any],
    *,
    note: str = "",
) -> dict[str, Any]:
    merged = deepcopy(dict(state))
    operator_names = [
        str(item.get("name", "")) for item in merged.get("operators", ())
    ]
    ticket_counts = _counts_by(merged.get("retained_tickets", ()), "type")
    collectible_ids = [
        str(item.get("id", "")) for item in merged.get("collectibles", ())
    ]
    status_ids = [
        str(item.get("id", "")) for item in merged.get("statuses", ())
    ]
    checkpoint = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": deepcopy(merged.get("phase", {})),
        "resources": deepcopy(merged.get("resources", {})),
        "operator_count": len(merged.get("operators", ())),
        "retained_ticket_count": sum(
            int(item.get("count", 1))
            for item in merged.get("retained_tickets", ())
        ),
        "collectible_count": len(merged.get("collectibles", ())),
        "status_count": len(merged.get("statuses", ())),
        "operator_names": operator_names,
        "retained_tickets": ticket_counts,
        "collectible_ids": collectible_ids,
        "status_ids": status_ids,
        "note": note,
    }
    checkpoints = merged.setdefault("checkpoints", [])
    if checkpoints:
        previous = checkpoints[-1]
        checkpoint["delta"] = {
            "resources": {
                key: _numeric(checkpoint["resources"].get(key))
                - _numeric(previous.get("resources", {}).get(key))
                for key in RESOURCE_FIELDS
                if checkpoint["resources"].get(key) is not None
                and previous.get("resources", {}).get(key) is not None
            },
            "operators_added": sorted(
                set(operator_names) - set(previous.get("operator_names", ()))
            ),
            "retained_tickets": {
                key: ticket_counts.get(key, 0)
                - previous.get("retained_tickets", {}).get(key, 0)
                for key in set(ticket_counts).union(
                    previous.get("retained_tickets", {})
                )
            },
            "collectibles_added": sorted(
                set(collectible_ids)
                - set(previous.get("collectible_ids", ()))
            ),
            "statuses_added": sorted(
                set(status_ids) - set(previous.get("status_ids", ()))
            ),
            "statuses_removed": sorted(
                set(previous.get("status_ids", ())) - set(status_ids)
            ),
        }
    checkpoints.append(checkpoint)
    # Manual locks are intentionally checkpoint-local. The next map/reward
    # observation may update them again and can then be corrected once.
    merged["field_provenance"] = {}
    return merged


def _numeric(value: Any) -> float:
    return float(value or 0)


def _counts_by(items: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        name = str(item.get(key, ""))
        if name:
            result[name] = result.get(name, 0) + int(item.get("count", 1))
    return result


def aggregate_run_effects(
    state: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> AggregatedRunEffects:
    """Combine route-relevant effects without replaying acquisition rewards."""

    result = AggregatedRunEffects()
    collectible_catalog = {
        item["id"]: item for item in catalog.get("collectibles", ())
    }
    status_catalog = {item["id"]: item for item in catalog.get("statuses", ())}
    effects: list[Mapping[str, Any]] = []
    for owned in state.get("collectibles", ()):
        if not owned.get("active", True):
            continue
        definition = collectible_catalog.get(owned.get("id"), {})
        effects.extend(definition.get("planner_effects", ()))
    for active in state.get("statuses", ()):
        if not active.get("active", True):
            continue
        definition = status_catalog.get(active.get("id"), {})
        effects.extend(_status_effects(definition, state))
    _apply_effects(result, effects)
    return result


def _status_effects(
    definition: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    phase = int(state.get("phase", {}).get("floor", 1))
    tier = "early" if phase <= 2 else "middle" if phase <= 4 else "late"
    tiered = definition.get("planner_effects_by_phase", {})
    return (*definition.get("planner_effects", ()), *tiered.get(tier, ()))


def _apply_effects(
    result: AggregatedRunEffects,
    effects: Iterable[Mapping[str, Any]],
) -> None:
    for effect in effects:
        metric = effect.get("metric")
        value = float(effect.get("value", 0))
        if metric == "node_score":
            target = str(effect.get("target", ""))
            result.node_scores[target] = result.node_scores.get(target, 0) + value
        elif metric == "resource_multiplier":
            target = str(effect.get("target", ""))
            result.resource_multipliers[target] = (
                result.resource_multipliers.get(target, 1.0) * value
            )
        elif metric == "combat_risk":
            result.combat_risk += value
        elif metric == "merchant_value_multiplier":
            result.merchant_value_multiplier *= value
        elif metric == "bonus_parts_per_combat":
            result.bonus_parts_per_combat += value
        elif metric == "information_risk":
            result.information_risk += value
        note = str(effect.get("note", "")).strip()
        if note and note not in result.notes:
            result.notes.append(note)
