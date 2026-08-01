from __future__ import annotations

import argparse
from datetime import date
import html
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API = "https://wiki.biligame.com/arknights/api.php"
COLLECTIBLE_PAGE = "沉沦者的黑流树海/拟造物质编目"
STATUS_PAGE = "沉沦者的黑流树海/乌托邦幸福论"
SOURCE_URLS = {
    "collectibles": "https://wiki.biligame.com/arknights/沉沦者的黑流树海/拟造物质编目",
    "statuses": "https://wiki.biligame.com/arknights/沉沦者的黑流树海/乌托邦幸福论",
}


def fetch_wikitext(page: str) -> str:
    query = urlencode(
        {"action": "parse", "format": "json", "prop": "wikitext", "page": page}
    )
    request = Request(
        f"{API}?{query}",
        headers={"User-Agent": "black-flow-advisor/0.1 (catalog updater)"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload["parse"]["wikitext"]["*"]


def clean_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    value = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", value)
    return html.unescape(value).strip()


def parse_collectibles(wikitext: str) -> list[dict]:
    entries: list[dict] = []
    for block in re.findall(r"\{\{肉鸽收藏品\s*\n(.*?)\n\}\}", wikitext, re.DOTALL):
        fields = dict(
            re.findall(r"^\|([^=\n]+)=(.*?)(?=\n\||\Z)", block, re.MULTILINE | re.DOTALL)
        )
        relic_id = fields.get("relicId", "").strip()
        name = clean_text(fields.get("藏品名", ""))
        effect = clean_text(fields.get("效果", ""))
        if not relic_id or not name:
            continue
        item = {
            "id": relic_id,
            "number": fields.get("编号", "").strip(),
            "name_zh": name,
            "effect_text": effect,
            "effect_by_structure": split_structure_effect(effect),
            "planner_effects": infer_collectible_planner_effects(name, effect),
            "source": "bwiki_collectibles",
        }
        entries.append(item)
    return entries


def split_structure_effect(effect: str) -> dict[str, str]:
    labels = {
        "结构化": "structured",
        "半结构化": "semi_structured",
        "非结构化": "unstructured",
        "混沌化": "chaotic",
    }
    result: dict[str, str] = {}
    matches = list(re.finditer(r"【(结构化|半结构化|非结构化|混沌化)】", effect))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(effect)
        result[labels[match.group(1)]] = effect[match.end() : end].strip()
    return result


def infer_collectible_planner_effects(name: str, effect: str) -> list[dict]:
    """Only model persistent route consequences, never one-time resource gains."""

    structured = split_structure_effect(effect)
    if structured:
        effect = structured.get("structured", next(iter(structured.values())))
    effects: list[dict] = []
    if "作战后获得收藏品的概率提升" in effect:
        effects.extend(
            {"metric": "node_score", "target": kind, "value": 2.0}
            for kind in ("combat", "emergency_combat", "resident_base", "enemy")
        )
    if "战斗胜利后获得随机零件" in effect or "作战胜利后获得随机零件" in effect:
        effects.append({"metric": "bonus_parts_per_combat", "value": 1.0})
    if "行商" in effect and any(word in effect for word in ("商品", "价格", "投资")):
        effects.append(
            {"metric": "node_score", "target": "secret_trader", "value": 2.0}
        )
        effects.append(
            {"metric": "node_score", "target": "rogue_trader", "value": 2.0}
        )
    if "失与得" in effect:
        effects.append(
            {"metric": "node_score", "target": "lost_and_found", "value": 3.0}
        )
    if "得偿所愿" in effect:
        effects.append({"metric": "node_score", "target": "wish", "value": 3.0})
    if "不期而遇" in effect:
        effects.append({"metric": "node_score", "target": "event", "value": 2.0})
    # Combat power is deliberately coarse. It changes risk preference, not reward totals.
    combat_terms = sum(
        effect.count(term)
        for term in ("攻击力", "防御力", "生命值", "攻击速度", "技力", "屏障")
    )
    if combat_terms:
        effects.append(
            {
                "metric": "combat_risk",
                "value": -min(2.5, combat_terms * 0.25),
                "note": f"{name}提供作战增益，降低战斗路线风险估值",
            }
        )
    return effects


def parse_statuses(wikitext: str) -> list[dict]:
    entries: list[dict] = []
    for index, body in enumerate(re.findall(r"\{\{理想域\|(.*?)\}\}", wikitext, re.DOTALL), 1):
        fields = {}
        for part in body.split("|"):
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key.strip()] = clean_text(value)
        name = fields.get("名称", "").strip("“”\"")
        if not name:
            continue
        item = {
            "id": fields.get("图标", f"ideal_domain_{index}"),
            "name_zh": name,
            "kind": fields.get("类型", "理想域"),
            "positive": fields.get("正面") == "是",
            "description": fields.get("描述", ""),
            "effect_text": fields.get("效果", ""),
            "effect_by_phase": {
                key: value
                for key, value in {
                    "early": fields.get("早期效果", ""),
                    "middle": fields.get("中期效果", ""),
                    "late": fields.get("晚期效果", ""),
                }.items()
                if value
            },
            "planner_effects": infer_status_effects(fields.get("效果", "")),
            "planner_effects_by_phase": {
                phase: infer_status_effects(fields.get(key, ""))
                for phase, key in (
                    ("early", "早期效果"),
                    ("middle", "中期效果"),
                    ("late", "晚期效果"),
                )
                if fields.get(key)
            },
            "source": "bwiki_ideal_domains",
        }
        entries.append(item)
    entries.append(
        {
            "id": "detection_vanguard",
            "name_zh": "探测先锋",
            "kind": "追猎后遗物",
            "positive": False,
            "effect_text": "下一层初始行动力-1",
            "planner_effects": [
                {
                    "metric": "information_risk",
                    "value": 1.0,
                    "note": "下一层行动力降低，当前层应提高离层与行动力保留价值",
                }
            ],
            "planner_effects_by_phase": {},
            "source": "bwiki_main",
        }
    )
    return entries


def infer_status_effects(text: str) -> list[dict]:
    effects: list[dict] = []
    if not text:
        return effects
    if "战斗胜利后获得随机零件" in text:
        effects.append({"metric": "bonus_parts_per_combat", "value": 1.0})
        effects.append({"metric": "combat_risk", "value": -1.0})
    enemy_bonus = max([int(value) for value in re.findall(r"敌方[^，。%]*\+(\d+)%", text)] or [0])
    friendly_penalty = max([int(value) for value in re.findall(r"我方[^，。%]*(?:降低|提高|损失)(\d+)%", text)] or [0])
    if enemy_bonus or friendly_penalty or any(term in text for term in ("冻结", "寒冷", "费用自然回复效率-")):
        effects.append(
            {
                "metric": "combat_risk",
                "value": max(1.0, (enemy_bonus + friendly_penalty) / 20),
            }
        )
    sale = re.search(r"零件售出价值-(\d+)%", text)
    if sale:
        effects.append(
            {"metric": "merchant_value_multiplier", "value": 1 - int(sale.group(1)) / 100}
        )
    if "无法揭示已连通的节点信息" in text:
        effects.append({"metric": "information_risk", "value": 2.0})
    if "途经" in text and any(term in text for term in ("源石锭-", "护盾-", "生命-")):
        effects.append(
            {
                "metric": "information_risk",
                "value": 1.5,
                "note": "理想域途经代价尚未标注到单个节点，当前按全局保守估值",
            }
        )
    return effects


def build_catalog(collectible_text: str, status_text: str) -> dict:
    return {
        "schema_version": "0.1.0",
        "retrieved": date.today().isoformat(),
        "sources": [
            {"id": "bwiki_collectibles", "url": SOURCE_URLS["collectibles"]},
            {"id": "bwiki_ideal_domains", "url": SOURCE_URLS["statuses"]},
        ],
        "stacking_rules": {
            "acquisition_resources": "已由当前资源 UI 体现，不重复加到路线收益",
            "node_score": "add",
            "combat_risk": "add",
            "resource_multiplier": "multiply",
            "merchant_value_multiplier": "multiply",
            "bonus_parts_per_combat": "add",
        },
        "collectibles": parse_collectibles(collectible_text),
        "statuses": parse_statuses(status_text),
        "ticket_types": ["先锋", "近卫", "重装", "狙击", "术师", "医疗", "辅助", "特种", "高级资深"],
        "operator_classes": ["先锋", "近卫", "重装", "狙击", "术师", "医疗", "辅助", "特种"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update run-state effect catalog from BWIKI.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "knowledge" / "run-state.v0.1.json",
    )
    args = parser.parse_args()
    catalog = build_catalog(fetch_wikitext(COLLECTIBLE_PAGE), fetch_wikitext(STATUS_PAGE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {len(catalog['collectibles'])} collectibles and "
        f"{len(catalog['statuses'])} statuses to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
