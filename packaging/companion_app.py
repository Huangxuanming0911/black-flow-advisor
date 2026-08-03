from __future__ import annotations

import ctypes
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import traceback
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


APP_MUTEX_NAME = "Local\\BlackFlowCompanion"
ERROR_ALREADY_EXISTS = 183
APP_VERSION = "0.2.0"

PROVIDERS = {
    "openai": {
        "label": "OpenAI Responses",
        "base_url": "https://api.openai.com/v1",
        "model": "",
        "key_required": True,
        "hint": "使用 Responses API；模型名称由你填写，避免版本更新后失效。",
    },
    "compatible": {
        "label": "OpenAI-compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "",
        "key_required": False,
        "hint": "适用于 Ollama、LM Studio 以及提供 Chat Completions 的服务。",
    },
    "anthropic": {
        "label": "Anthropic Messages",
        "base_url": "https://api.anthropic.com/v1",
        "model": "",
        "key_required": True,
        "hint": "使用 Messages API；模型名称由你填写。",
    },
}

PERSONAS = {
    "advisor": {
        "label": "冷静参谋",
        "description": "结论优先，说明代价、风险与依据。",
        "prompt": "像冷静的作战参谋。先给可执行结论，再用简短理由说明收益、代价和风险。",
    },
    "terminal": {
        "label": "罗德岛终端",
        "description": "克制的终端播报感，不扮演官方角色。",
        "prompt": "使用克制、清晰的行动终端语气，可有少量世界观氛围，但不要冒充任何官方角色。",
    },
    "partner": {
        "label": "轻松搭档",
        "description": "自然、友好，少用术语。",
        "prompt": "像熟悉游戏的轻松搭档，自然友好，少用术语，不要过度卖萌。",
    },
    "coach": {
        "label": "严格教练",
        "description": "直接指出风险与不划算的决策。",
        "prompt": "像严格但公正的教练。明确指出高风险、低收益和信息不足，不用委婉话掩盖判断。",
    },
}


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[1]


def shell_html() -> Path:
    if getattr(sys, "frozen", False):
        return resource_root() / "app" / "bot-shell.html"
    return resource_root() / "web" / "bot-shell.html"


def planner_html() -> Path:
    if getattr(sys, "frozen", False):
        return resource_root() / "planner" / "index.html"
    return resource_root() / "data" / "output" / "route-planner" / "index.html"


def knowledge_root() -> Path:
    if getattr(sys, "frozen", False):
        return resource_root() / "knowledge"
    return resource_root() / "data" / "knowledge"


def _message(title: str, message: str, *, error: bool = False) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, title, 0x10 if error else 0x40)


def _enable_dpi_awareness() -> None:
    """Keep WebView CSS pixels aligned with window coordinates on scaled displays."""
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


def _acquire_single_instance() -> object | None:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, APP_MUTEX_NAME)
    if not handle:
        raise ctypes.WinError()
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return handle


def _write_crash_log(root: Path) -> Path:
    log_dir = root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"companion-crash-{datetime.now():%Y%m%d-%H%M%S}.log"
    path.write_text(traceback.format_exc(), encoding="utf-8")
    return path


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    words = set(re.findall(r"[a-z0-9_\-]{2,}|[\u4e00-\u9fff]", normalized))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    words.update(chinese[i : i + 2] for i in range(max(0, len(chinese) - 1)))
    return {token for token in words if token}


def _display_json(value: Any, limit: int = 1500) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return text if len(text) <= limit else text[: limit - 1] + "…"


class KnowledgeIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.chunks: list[dict[str, Any]] = []
        self.sources: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for source in data.get("sources", []) if isinstance(data, dict) else []:
                if isinstance(source, dict) and source.get("url"):
                    key = str(source.get("id") or source["url"])
                    self.sources[key] = {
                        "title": str(source.get("title") or key),
                        "url": str(source["url"]),
                    }
            self._walk(data, path.name, "root")

    def _walk(self, value: Any, filename: str, pointer: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                self._walk(item, filename, f"{pointer}[{index}]")
            return
        if not isinstance(value, dict):
            return
        scalar = {key: val for key, val in value.items() if not isinstance(val, (dict, list))}
        if len(scalar) >= 2 or any(key in scalar for key in ("name_zh", "effect_text", "node_kind", "id")):
            text = _display_json(scalar)
            self.chunks.append(
                {
                    "source": filename,
                    "path": pointer,
                    "text": text,
                    "tokens": _tokens(text),
                }
            )
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                self._walk(item, filename, f"{pointer}.{key}")

    def search(self, query: str, limit: int = 6) -> list[dict[str, str]]:
        wanted = _tokens(query)
        if not wanted:
            return []
        ranked: list[tuple[float, dict[str, Any]]] = []
        for chunk in self.chunks:
            overlap = wanted & chunk["tokens"]
            if not overlap:
                continue
            exact_bonus = 4 if query.strip().lower() in chunk["text"].lower() else 0
            score = sum(2 if len(token) > 1 else 0.35 for token in overlap) + exact_bonus
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            {"source": item[1]["source"], "path": item[1]["path"], "text": item[1]["text"]}
            for item in ranked[:limit]
        ]

    def source_list(self) -> list[dict[str, str]]:
        return list(self.sources.values())


def _normalize_endpoint(base_url: str, suffix: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("API 地址不能为空")
    if not urlparse(base).scheme in {"http", "https"}:
        raise ValueError("API 地址必须以 http:// 或 https:// 开头")
    return base if base.endswith(suffix) else base + suffix


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 45) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:700]
        raise RuntimeError(f"模型服务返回 HTTP {exc.code}：{detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接模型服务：{exc.reason}") from exc


def _openai_output(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()
    parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts).strip()


def _compatible_output(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("模型返回中没有 choices[0].message.content") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict)).strip()
    return str(content).strip()


def _anthropic_output(data: dict[str, Any]) -> str:
    return "\n".join(
        str(item.get("text", ""))
        for item in data.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ).strip()


def _system_prompt(persona: str, custom_style: str, context: str, state: dict[str, Any]) -> str:
    style = PERSONAS.get(persona, PERSONAS["advisor"])["prompt"]
    if custom_style.strip():
        style += f"\n用户补充的语言要求：{custom_style.strip()[:500]}"
    return f"""你是“黑流树海陪跑助手”，在玩家进行《明日方舟》集成战略时提供短而可执行的建议。
{style}

规则：
1. 严格区分【本地规则】【实测样本】【外部资料】【模型推断】；不要把推断说成确定事实。
2. 资料不足时直接说明缺什么，并给出玩家现在能做的检查。
3. 路线建议必须考虑当前位置、行动力、零件、楼层、风险、队伍与用户倾向；不要默认把零件全部花完。
4. 不自动操作游戏，不声称看到了未提供的画面。
5. 默认用简体中文，回答适合玩家在对局中快速阅读。

当前局状态：
{_display_json(state, 2400)}

按相关度检索到的资料（可能为空）：
{context or '没有检索到直接相关资料。'}"""


class DesktopApi:
    def __init__(self) -> None:
        self.window = None
        self.index = KnowledgeIndex(knowledge_root())
        self.session_keys: dict[str, str] = {}
        self.settings_path = application_root() / "data" / "settings.json"

    def _load_settings(self) -> dict[str, Any]:
        defaults = {
            "provider": "openai",
            "base_urls": {key: val["base_url"] for key, val in PROVIDERS.items()},
            "models": {key: val["model"] for key, val in PROVIDERS.items()},
            "persona": "advisor",
            "custom_style": "",
            "knowledge_enabled": True,
        }
        try:
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for key in defaults:
                    if key in saved and key != "api_key":
                        defaults[key] = saved[key]
        except (OSError, json.JSONDecodeError):
            pass
        return defaults

    def bootstrap(self) -> dict[str, Any]:
        return {
            "version": APP_VERSION,
            "settings": self._load_settings(),
            "providers": PROVIDERS,
            "personas": PERSONAS,
            "knowledge": {"chunks": len(self.index.chunks), "sources": self.index.source_list()},
            "planner_url": planner_html().as_uri(),
        }

    def set_session_key(self, provider: str, api_key: str) -> dict[str, Any]:
        if provider not in PROVIDERS:
            raise ValueError("未知模型服务")
        value = api_key.strip()
        if value:
            self.session_keys[provider] = value
        else:
            self.session_keys.pop(provider, None)
        return {"saved": bool(value), "storage": "memory_only"}

    def save_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        allowed = {"provider", "base_urls", "models", "persona", "custom_style", "knowledge_enabled"}
        safe = {key: value for key, value in settings.items() if key in allowed}
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"saved": True, "path": str(self.settings_path), "secret_saved": False}

    def search_knowledge(self, query: str) -> dict[str, Any]:
        query = str(query).strip()
        return {
            "results": self.index.search(query, 8),
            "external": [
                {
                    "title": f"在 PRTS 搜索：{query}",
                    "url": "https://prts.wiki/index.php?search=" + quote(query),
                    "status": "按需打开；PRTS API 当前拒绝自动访问",
                }
            ] if query else [],
        }

    def open_external(self, url: str) -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("只允许打开 http/https 链接")
        __import__("webbrowser").open(url)
        return {"opened": True}

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(payload.get("provider", "openai"))
        if provider not in PROVIDERS:
            raise ValueError("未知模型服务")
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("没有可发送的消息")
        user_text = next(
            (str(item.get("content", "")) for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        results = self.index.search(user_text, 6) if payload.get("knowledge_enabled", True) else []
        context = "\n".join(
            f"- 【{item['source']} {item['path']}】{item['text']}" for item in results
        )[:7000]
        system = _system_prompt(
            str(payload.get("persona", "advisor")),
            str(payload.get("custom_style", "")),
            context,
            payload.get("state", {}) if isinstance(payload.get("state"), dict) else {},
        )
        model = str(payload.get("model", "")).strip()
        if not model:
            raise ValueError("请先填写模型名称")
        base_url = str(payload.get("base_url") or PROVIDERS[provider]["base_url"])
        key = self.session_keys.get(provider, "")
        if PROVIDERS[provider]["key_required"] and not key:
            raise ValueError("请先输入 API Key；Key 只保存在本次运行的内存中")
        clean_messages = [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))[:8000]}
            for item in messages[-16:]
            if item.get("role") in {"user", "assistant"}
        ]
        if provider == "openai":
            data = _post_json(
                _normalize_endpoint(base_url, "/responses"),
                {"Authorization": f"Bearer {key}"},
                {"model": model, "instructions": system, "input": clean_messages},
            )
            answer = _openai_output(data)
        elif provider == "compatible":
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            data = _post_json(
                _normalize_endpoint(base_url, "/chat/completions"),
                headers,
                {"model": model, "messages": [{"role": "system", "content": system}, *clean_messages], "temperature": 0.35},
            )
            answer = _compatible_output(data)
        else:
            data = _post_json(
                _normalize_endpoint(base_url, "/messages"),
                {"x-api-key": key, "anthropic-version": "2023-06-01"},
                {"model": model, "max_tokens": 1200, "system": system, "messages": clean_messages},
            )
            answer = _anthropic_output(data)
        if not answer:
            raise RuntimeError("模型没有返回可显示文本")
        external_search = "https://prts.wiki/index.php?search=" + quote(user_text)
        return {
            "answer": answer,
            "sources": [
                {"title": f"{item['source']} · {item['path']}", "kind": "local"} for item in results
            ] + ([{"title": "PRTS 外部检索", "kind": "external", "url": external_search}] if results else []),
            "external_search": external_search,
        }


def _run_self_test(root: Path) -> int:
    log_dir = root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "companion-self-test.json"
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "executable": str(Path(sys.executable).resolve()),
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    try:
        import webview

        shell = shell_html()
        planner = planner_html()
        index = KnowledgeIndex(knowledge_root())
        assert shell.is_file() and planner.is_file()
        assert "黑流树海陪跑助手" in shell.read_text(encoding="utf-8")
        assert index.chunks
        report.update(
            {
                "status": "ok",
                "shell_html": str(shell),
                "planner_html": str(planner),
                "knowledge_chunks": len(index.chunks),
                "webview_module": str(Path(webview.__file__).resolve()),
            }
        )
        result = 0
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        result = 2
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    root = application_root()
    if "--self-test" in sys.argv[1:]:
        return _run_self_test(root)
    handle = _acquire_single_instance()
    if handle is None:
        _message("黑流树海陪跑助手", "陪跑助手已经在运行。")
        return 0
    try:
        _enable_dpi_awareness()
        import webview

        html_path = shell_html()
        if not html_path.is_file():
            raise FileNotFoundError(f"找不到助手页面：{html_path}")
        storage = root / "data" / "webview-companion"
        storage.mkdir(parents=True, exist_ok=True)
        api = DesktopApi()
        webview.settings["ALLOW_DOWNLOADS"] = True
        api.window = webview.create_window(
            "黑流树海 · 陪跑助手",
            url=html_path.as_uri(),
            js_api=api,
            width=1500,
            height=920,
            min_size=(1120, 720),
            resizable=True,
            background_color="#091115",
            text_select=True,
            zoomable=True,
        )
        webview.start(gui="edgechromium", debug=False, private_mode=False, storage_path=str(storage))
        return 0
    except Exception:
        log_path = _write_crash_log(root)
        _message("黑流树海陪跑助手启动失败", f"错误详情已保存到：\n{log_path}", error=True)
        return 1
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


if __name__ == "__main__":
    raise SystemExit(main())
