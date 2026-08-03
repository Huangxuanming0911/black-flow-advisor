from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


project_root = Path(SPECPATH).parent
shell_html = project_root / "web" / "bot-shell.html"
planner_html = project_root / "data" / "output" / "route-planner" / "index.html"
knowledge_dir = project_root / "data" / "knowledge"

datas, binaries, hiddenimports = collect_all("webview")
hiddenimports += collect_submodules("webview.platforms")

analysis = Analysis(
    [str(project_root / "packaging" / "companion_app.py")],
    pathex=[str(project_root / "packaging")],
    binaries=binaries,
    datas=datas + [
        (str(shell_html), "app"),
        (str(planner_html), "planner"),
        (str(knowledge_dir), "knowledge"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["numpy", "cv2", "PIL", "pytest"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="BlackFlowCompanion",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BlackFlowCompanion",
)
