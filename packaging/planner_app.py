from __future__ import annotations

import ctypes
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import traceback


APP_MUTEX_NAME = "Local\\BlackFlowRoutePlanner"
ERROR_ALREADY_EXISTS = 183


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[1]


def planner_html() -> Path:
    if getattr(sys, "frozen", False):
        return resource_root() / "planner" / "index.html"
    return resource_root() / "data" / "output" / "route-planner" / "index.html"


def _message(title: str, message: str, *, error: bool = False) -> None:
    flags = 0x10 if error else 0x40
    ctypes.windll.user32.MessageBoxW(None, message, title, flags)


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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"crash-{timestamp}.log"
    path.write_text(traceback.format_exc(), encoding="utf-8")
    return path


class DesktopApi:
    def __init__(self) -> None:
        self.window = None

    def save_text(self, suggested_name: str, content: str) -> dict[str, object]:
        import webview

        if self.window is None:
            raise RuntimeError("桌面窗口尚未就绪")
        safe_name = re.sub(r"[^0-9A-Za-z._-]+", "-", suggested_name)
        selected = self.window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename=safe_name or "blackflow-run-state.json",
            file_types=("JSON 文件 (*.json)", "所有文件 (*.*)"),
        )
        if not selected:
            return {"saved": False}
        path = Path(selected[0])
        path.write_text(content, encoding="utf-8")
        return {"saved": True, "path": str(path)}


def _run_self_test(root: Path) -> int:
    log_dir = root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "self-test.json"
    report: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "executable": str(Path(sys.executable).resolve()),
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    try:
        import webview

        html_path = planner_html()
        html = html_path.read_text(encoding="utf-8")
        report.update(
            {
                "status": "ok",
                "planner_html": str(html_path),
                "planner_bytes": html_path.stat().st_size,
                "bootstrap_embedded": (
                    "const BOOTSTRAP={" in html
                    and "__BLACKFLOW_DATA__" not in html
                ),
                "webview_module": str(Path(webview.__file__).resolve()),
            }
        )
        if not report["bootstrap_embedded"]:
            raise RuntimeError("规划器页面没有嵌入有效的规划数据")
        result = 0
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        result = 2
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    root = application_root()
    if "--self-test" in sys.argv[1:]:
        return _run_self_test(root)

    handle = _acquire_single_instance()
    if handle is None:
        _message("黑流树海路径规划器", "路径规划器已经在运行。")
        return 0

    try:
        import webview

        html_path = planner_html()
        if not html_path.is_file():
            raise FileNotFoundError(f"找不到规划器页面：{html_path}")
        storage = root / "data" / "webview"
        storage.mkdir(parents=True, exist_ok=True)
        api = DesktopApi()
        webview.settings["ALLOW_DOWNLOADS"] = True
        api.window = webview.create_window(
            "黑流树海 · 路径规划器",
            url=html_path.as_uri(),
            js_api=api,
            width=1440,
            height=900,
            min_size=(1100, 700),
            resizable=True,
            background_color="#081018",
            text_select=True,
            zoomable=True,
        )
        webview.start(
            gui="edgechromium",
            debug=False,
            private_mode=False,
            storage_path=str(storage),
        )
        return 0
    except Exception:
        log_path = _write_crash_log(root)
        _message(
            "黑流树海路径规划器启动失败",
            f"错误详情已保存到：\n{log_path}",
            error=True,
        )
        return 1
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


if __name__ == "__main__":
    raise SystemExit(main())
