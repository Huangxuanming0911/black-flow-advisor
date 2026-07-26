from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping


_GRID_ID = re.compile(r"^node_r(-?\d+)c(-?\d+)$")
_COMBAT_KINDS = {
    "combat",
    "emergency_combat",
    "enemy",
    "resident_base",
    "resident_occupied",
}
_ONE_SHOT_KINDS = {
    "combat",
    "emergency_combat",
    "enemy",
    "unknown_combat",
    "unknown_event",
    "event",
    "safe_house",
    "fate",
    "wish",
    "encounter",
    "lost_and_found",
    "emergency_support",
    "resident_base",
    "resident_occupied",
    "portal",
    "scout",
    "exit_end",
    "exit_path",
}


@dataclass(frozen=True, slots=True)
class PlannerNode:
    node_id: str
    kind: str
    label: str = ""
    center: tuple[int, int] = (0, 0)

    @property
    def grid_position(self) -> tuple[int, int] | None:
        match = _GRID_ID.match(self.node_id)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True, slots=True)
class PlannerGraph:
    nodes: Mapping[str, PlannerNode]
    adjacency: Mapping[str, frozenset[str]]

    @classmethod
    def from_unified_dict(cls, payload: Mapping[str, Any]) -> "PlannerGraph":
        nodes = {
            str(raw["node_id"]): PlannerNode(
                node_id=str(raw["node_id"]),
                kind=str(raw.get("kind", "unknown")),
                label=str(raw.get("label", "")),
                center=tuple(int(value) for value in raw.get("center", (0, 0))),
            )
            for raw in payload.get("nodes", ())
        }
        adjacency = {
            node_id: set() for node_id in nodes
        }
        for edge in payload.get("edges", ()):
            first = str(edge["first"])
            second = str(edge["second"])
            if first in nodes and second in nodes and first != second:
                adjacency[first].add(second)
                adjacency[second].add(first)
        return cls(
            nodes=nodes,
            adjacency={
                node_id: frozenset(neighbours)
                for node_id, neighbours in adjacency.items()
            },
        )


@dataclass(frozen=True, slots=True)
class MovementRule:
    mode_id: str
    part_id: str | None
    target_type: str
    maximum: int | None
    ap_cost: int
    node_filter: str | None = None
    post_effects: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_knowledge(cls, raw: Mapping[str, Any]) -> "MovementRule":
        target = raw.get("target_rule", {})
        cost = raw.get("action_point_cost", {})
        return cls(
            mode_id=str(raw["id"]),
            part_id=raw.get("part_id"),
            target_type=str(target.get("type", "any_node")),
            maximum=(
                int(target["max"])
                if target.get("max") is not None
                else None
            ),
            ap_cost=int(cost.get("value", 0)),
            node_filter=target.get("node_filter"),
            post_effects=tuple(
                (str(effect["resource"]), int(effect["delta"]))
                for effect in raw.get("post_effects", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class PartState:
    instance_id: str
    part_id: str
    remaining_uses: int | None
    estimated_value: int


@dataclass(frozen=True, slots=True)
class RouteAction:
    target: str
    mode_id: str = "walk"
    part_instance_id: str | None = None


@dataclass(frozen=True, slots=True)
class RouteStep:
    index: int
    source: str
    selected_target: str
    target: str
    mode_id: str
    ap_cost: int
    ap_gain: int
    ap_after: int
    node_kind: str
    first_completion: bool
    rewards: tuple[str, ...]
    warnings: tuple[str, ...]
    auto_teleport: bool = False


@dataclass(slots=True)
class ResourceEstimate:
    minimum: int = 0
    maximum: int = 0
    expected: float = 0.0
    pending: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RouteSimulation:
    valid: bool
    start_node: str
    current_node: str
    initial_action_points: int
    action_points: int
    hope: int = 0
    originium_ingots: int = 0
    guaranteed_collectibles: int = 0
    completed_nodes: set[str] = field(default_factory=set)
    reward_opportunities: dict[str, int] = field(default_factory=dict)
    resource_estimates: dict[str, ResourceEstimate] = field(default_factory=dict)
    forced_encounters: list[str] = field(default_factory=list)
    parts: dict[str, PartState] = field(default_factory=dict)
    part_value_min: int = 0
    part_value_max: int = 0
    steps: list[RouteStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def pair_tunnels(
    graph: PlannerGraph,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    overrides = overrides or {}
    tunnels = sorted(
        node_id
        for node_id, node in graph.nodes.items()
        if overrides.get(node_id, node.kind) == "tunnel"
    )
    if len(tunnels) != 2:
        return {}
    return {tunnels[0]: tunnels[1], tunnels[1]: tunnels[0]}


def is_legal_target(
    graph: PlannerGraph,
    source: str,
    target: str,
    movement: MovementRule,
    *,
    overrides: Mapping[str, str] | None = None,
    tunnel_pairs: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    overrides = overrides or {}
    tunnel_pairs = tunnel_pairs or {}
    if source not in graph.nodes or target not in graph.nodes:
        return False, "起点或终点不在当前图中"
    if source == target:
        return False, "本版路线试算不处理原地重复进入"

    if movement.mode_id == "walk":
        if target in graph.adjacency.get(source, frozenset()):
            return True, ""
        return False, "徒步必须逐段选择识别图中的相邻节点"

    if movement.mode_id == "tunnel_transfer":
        return False, "曲折密道不是可选移动方式；抵达入口后会自动传送"

    source_position = graph.nodes[source].grid_position
    target_position = graph.nodes[target].grid_position
    if source_position is None or target_position is None:
        return False, "零件移动需要带行列坐标的稳定节点 ID"
    row_delta = abs(source_position[0] - target_position[0])
    column_delta = abs(source_position[1] - target_position[1])
    maximum = movement.maximum or 0

    legal = False
    if movement.target_type == "orthogonal_distance":
        legal = (
            (row_delta == 0 or column_delta == 0)
            and row_delta + column_delta <= maximum
        )
    elif movement.target_type == "chebyshev_radius":
        legal = max(row_delta, column_delta) <= maximum
    elif movement.target_type == "manhattan_radius":
        legal = row_delta + column_delta <= maximum
    elif movement.target_type in {"any_node", "random_any"}:
        legal = True
    elif movement.target_type == "paired_node":
        legal = tunnel_pairs.get(source) == target

    if not legal:
        return False, "目标不满足该零件的距离或方向限制"

    target_kind = overrides.get(target, graph.nodes[target].kind)
    if movement.node_filter == "non_combat" and target_kind in (
        _COMBAT_KINDS | {"unknown_combat"}
    ):
        return False, "该零件不能以作战节点为目标"
    if movement.node_filter == "merchant" and target_kind not in {
        "shop",
        "secret_trader",
        "special_shop",
        "rogue_trader",
    }:
        return False, "该零件只能以行商节点为目标"
    return True, ""


def simulate_route(
    graph: PlannerGraph,
    start_node: str,
    actions: Iterable[RouteAction],
    movement_rules: Mapping[str, MovementRule],
    *,
    initial_action_points: int,
    parts: Iterable[PartState] = (),
    overrides: Mapping[str, str] | None = None,
    tunnel_pairs: Mapping[str, str] | None = None,
    reward_knowledge: Mapping[str, Any] | None = None,
) -> RouteSimulation:
    overrides = overrides or {}
    tunnel_pairs = tunnel_pairs or pair_tunnels(graph, overrides)
    part_states = {part.instance_id: part for part in parts}
    initial_value = sum(max(0, part.estimated_value) for part in part_states.values())
    result = RouteSimulation(
        valid=True,
        start_node=start_node,
        current_node=start_node,
        initial_action_points=initial_action_points,
        action_points=initial_action_points,
        parts=part_states,
        part_value_min=initial_value,
        part_value_max=initial_value,
        resource_estimates={
            str(dimension["id"]): ResourceEstimate()
            for dimension in (reward_knowledge or {}).get("dimensions", ())
        },
    )
    movement_count = 0
    combat_count = 0

    for index, action in enumerate(actions, start=1):
        movement = movement_rules.get(action.mode_id)
        if movement is None:
            result.valid = False
            result.errors.append(f"第{index}步使用了未知移动方式 {action.mode_id}")
            break

        if movement.part_id is not None:
            part = result.parts.get(action.part_instance_id or "")
            if part is None or part.part_id != movement.part_id:
                result.valid = False
                result.errors.append(f"第{index}步没有可用的对应零件")
                break
            if part.remaining_uses is not None and part.remaining_uses <= 0:
                result.valid = False
                result.errors.append(f"第{index}步所选零件已经损毁")
                break

        legal, reason = is_legal_target(
            graph,
            result.current_node,
            action.target,
            movement,
            overrides=overrides,
            tunnel_pairs=tunnel_pairs,
        )
        if not legal:
            result.valid = False
            result.errors.append(f"第{index}步无效：{reason}")
            break

        if result.action_points < movement.ap_cost:
            result.valid = False
            result.errors.append(f"第{index}步行动力不足")
            break

        source = result.current_node
        result.action_points -= movement.ap_cost
        movement_count += 1
        step_warnings: list[str] = []
        if movement.target_type == "random_any":
            step_warnings.append("该零件目标随机；当前终点只用于情景试算")

        if movement.part_id is not None:
            part = result.parts[action.part_instance_id or ""]
            if part.remaining_uses is not None:
                remaining = part.remaining_uses - 1
                result.parts[part.instance_id] = PartState(
                    instance_id=part.instance_id,
                    part_id=part.part_id,
                    remaining_uses=remaining,
                    estimated_value=part.estimated_value,
                )
                if remaining == 0:
                    result.part_value_min -= max(0, part.estimated_value)
                    result.part_value_max -= max(0, part.estimated_value)

        for resource, delta in movement.post_effects:
            if resource == "action_points":
                result.action_points += delta
            elif resource == "hope":
                result.hope += delta
                _add_exact_resource(
                    result,
                    "hope",
                    delta,
                    "移动零件效果",
                )
            elif resource == "originium_ingots":
                result.originium_ingots += delta
                _add_exact_resource(
                    result,
                    "originium_ingots",
                    delta,
                    "移动零件效果",
                )

        node = graph.nodes[action.target]
        kind = overrides.get(action.target, node.kind)
        first_completion = (
            action.target not in result.completed_nodes
            and kind in _ONE_SHOT_KINDS
        )
        rewards, ap_gain, combat_delta = _settle_node(
            kind,
            first_completion=first_completion,
        )
        result.action_points += ap_gain
        if first_completion:
            _settle_reward_catalog(
                result,
                kind,
                reward_knowledge=reward_knowledge,
            )
        combat_count += combat_delta
        if first_completion:
            result.completed_nodes.add(action.target)
        if kind == "tunnel":
            paired = tunnel_pairs.get(action.target)
            if paired:
                result.completed_nodes.update((action.target, paired))
        for reward in rewards:
            result.reward_opportunities[reward] = (
                result.reward_opportunities.get(reward, 0) + 1
            )
        if (
            kind == "wish"
            and first_completion
            and not reward_knowledge
        ):
            result.guaranteed_collectibles += 1

        destination = (
            tunnel_pairs[action.target]
            if kind == "tunnel" and action.target in tunnel_pairs
            else action.target
        )
        auto_teleport = destination != action.target
        if kind == "tunnel" and not auto_teleport:
            step_warnings.append("曲折密道尚未成功配对，无法自动传送")
        result.current_node = destination
        result.steps.append(
            RouteStep(
                index=index,
                source=source,
                selected_target=action.target,
                target=destination,
                mode_id=action.mode_id,
                ap_cost=movement.ap_cost,
                ap_gain=ap_gain + sum(
                    delta
                    for resource, delta in movement.post_effects
                    if resource == "action_points"
                ),
                ap_after=result.action_points,
                node_kind=kind,
                first_completion=first_completion,
                rewards=tuple(rewards),
                warnings=tuple(step_warnings),
                auto_teleport=auto_teleport,
            )
        )
        result.warnings.extend(step_warnings)
        if (
            result.action_points == 0
            and kind not in {"exit_end", "exit_path"}
            and not result.forced_encounters
        ):
            forced_combat = _settle_forced_encounter(
                result,
                reward_knowledge=reward_knowledge,
            )
            combat_count += forced_combat

    _settle_dynamic_part_values(
        result,
        movement_count=movement_count,
        combat_count=combat_count,
    )
    return result


def _add_exact_resource(
    result: RouteSimulation,
    resource: str,
    value: int,
    note: str,
) -> None:
    estimate = result.resource_estimates.get(resource)
    if estimate is None:
        return
    estimate.minimum += value
    estimate.maximum += value
    estimate.expected += value
    if note and note not in estimate.notes:
        estimate.notes.append(note)


def _settle_reward_catalog(
    result: RouteSimulation,
    kind: str,
    *,
    reward_knowledge: Mapping[str, Any] | None,
) -> None:
    if not reward_knowledge:
        return
    aliases = reward_knowledge.get("kind_aliases", {})
    canonical_kind = str(aliases.get(kind, kind))
    catalog = {
        str(item["id"]): item
        for item in reward_knowledge.get("node_rewards", ())
    }
    node_rule = catalog.get(canonical_kind)
    if node_rule is None:
        return
    for resource, raw in node_rule.get("rewards", {}).items():
        _settle_reward_entry(
            result,
            str(resource),
            raw,
        )


def _settle_reward_entry(
    result: RouteSimulation,
    resource: str,
    raw: Mapping[str, Any],
) -> None:
    estimate = result.resource_estimates.get(resource)
    if estimate is None:
        return
    rule_kind = str(raw.get("kind", "possible"))
    if rule_kind == "exact":
        value = int(raw.get("value", 0))
        estimate.minimum += value
        estimate.maximum += value
        estimate.expected += value
        if resource == "hope":
            result.hope += value
        elif resource == "originium_ingots":
            result.originium_ingots += value
        elif resource == "collectibles":
            result.guaranteed_collectibles += value
    elif rule_kind == "conditional_exact":
        estimate.maximum += int(raw.get("value", 0))
        estimate.pending += 1
    elif rule_kind in {"choice", "conditional"}:
        estimate.minimum += int(raw.get("minimum", 0))
        maximum = raw.get("maximum")
        if maximum is None:
            estimate.pending += 1
        else:
            estimate.maximum += int(maximum)
    elif rule_kind == "remaining_ap":
        value = max(0, result.action_points)
        estimate.minimum += value
        estimate.maximum += value
        estimate.expected += value
    elif rule_kind in {"possible", "variable", "indirect"}:
        estimate.pending += 1

    distribution = raw.get("known_distribution")
    if isinstance(distribution, Mapping):
        estimate.minimum += int(distribution.get("minimum", 0))
        estimate.maximum += int(distribution.get("maximum", 0))
        estimate.expected += float(distribution.get("expected", 0))
        distribution_note = str(distribution.get("summary", "")).strip()
        if distribution_note and distribution_note not in estimate.notes:
            estimate.notes.append(distribution_note)
    note = str(raw.get("summary", "")).strip()
    if note and note not in estimate.notes:
        estimate.notes.append(note)


def _settle_forced_encounter(
    result: RouteSimulation,
    *,
    reward_knowledge: Mapping[str, Any] | None,
) -> int:
    if not reward_knowledge:
        return 0
    rules = reward_knowledge.get("forced_encounters", ())
    rule = next(
        (
            item
            for item in rules
            if item.get("id") == "pursuit_on_ap_exhaustion"
        ),
        None,
    )
    if rule is None:
        return 0
    variant_id = str(rule.get("default_variant", "normal"))
    variant = rule.get("variants", {}).get(variant_id, {})
    label = str(variant.get("name_zh", rule.get("name_zh", "追猎")))
    result.forced_encounters.append(label)
    result.reward_opportunities[
        f"强制遭遇：{label}"
    ] = result.reward_opportunities.get(f"强制遭遇：{label}", 0) + 1
    for resource, raw in variant.get("rewards", {}).items():
        _settle_reward_entry(result, str(resource), raw)
    for effect in variant.get("additional_effects", ()):
        note = str(effect.get("summary", "")).strip()
        if note and note not in result.warnings:
            result.warnings.append(note)
    return 1


def _settle_node(
    kind: str,
    *,
    first_completion: bool,
) -> tuple[list[str], int, int]:
    if kind == "overlook":
        return ["羽瞰点行动力"], 1, 0
    if not first_completion and kind in _ONE_SHOT_KINDS:
        return [], 0, 0
    if kind == "combat":
        return ["普通作战随机奖励", "战后零件掉落机会"], 0, 1
    if kind == "emergency_combat":
        return ["紧急作战随机奖励", "战后零件掉落机会"], 0, 1
    if kind == "enemy":
        return ["领袖作战奖励", "战后零件掉落机会"], 0, 1
    if kind in {"resident_base", "resident_occupied"}:
        rewards = ["居民作战随机奖励", "战后零件掉落机会"]
        if kind == "resident_base":
            rewards.append("驱逐本层流窜居民")
        else:
            rewards.append("当前节点战后变为林间空地")
        return rewards, 0, 1
    if kind == "unknown_combat":
        return ["未揭示作战奖励机会"], 0, 0
    if kind == "unknown_event":
        return ["未揭示事件机会"], 0, 0
    if kind == "event":
        return ["随机事件机会"], 0, 0
    if kind == "encounter":
        return ["狭路相逢自选奖励机会"], 0, 0
    if kind == "safe_house":
        return ["安全角落三选一增益"], 0, 0
    if kind == "wish":
        return ["免费收藏品"], 0, 0
    if kind == "fate":
        return ["结局路线抉择"], 0, 0
    if kind == "scout":
        return ["先行一步选择收益"], 0, 0
    return [], 0, 0


def _settle_dynamic_part_values(
    result: RouteSimulation,
    *,
    movement_count: int,
    combat_count: int,
) -> None:
    for part in result.parts.values():
        if part.remaining_uses == 0:
            continue
        if part.part_id == "blood_mushroom":
            delta = combat_count * 2
            result.part_value_min += delta
            result.part_value_max += delta
        elif part.part_id == "homesick_fruit":
            loss = min(part.estimated_value, movement_count * 2)
            result.part_value_min -= loss
            result.part_value_max -= loss
        elif part.part_id == "wave":
            result.part_value_min -= min(
                part.estimated_value,
                movement_count * 8,
            )
            result.part_value_max += movement_count * 11
