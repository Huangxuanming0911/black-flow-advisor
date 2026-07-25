# 黑流树海路线助手

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个面向《明日方舟》集成战略“黑流树海”地图阶段的独立、只读视觉识别原型。

本项目不会复制、Fork 或依赖现有的黑流树海路线规划项目。官方
MaaFramework 仅作为可选的截图与任务编排宿主。

## 当前能力

- 从不同尺寸的 PC 截图中移除 Windows 标题栏，并转换到可逆的
  1280×720 游戏客户区坐标系。
- 区分正常地图、零件箱详情和移动方式选择界面。
- 等待画面稳定后再截图，并过滤未变化的重复帧。
- 直接识别半透明路径 UI，输出路径掩膜和单像素拓扑骨架。
- 单独识别林中节点，并利用已识别的路径像素生成临时无向边。
- 识别圆形地图节点、拟合离散行列，并支持本地模板分类。
- 识别固定零件面板中的已占用栏位。
- 合并存在可靠重叠的局部地图观测。
- 输出 JSON、调试标注图、置信度和验证问题。
- 在结果经过人工验证前，始终保持 `planner_ready: false`。

当前版本不包含路线规划、游戏点击、ADB 操作、账号自动化或随仓库发布的
游戏素材。真实截图只保存在被 Git 忽略的 `data/private/` 中。

## 本地运行

项目需要 Python 3.11 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[maa]"
$env:PYTHONPATH = "src"
```

生成并识别合成地图：

```powershell
python -m blackflow_vision.cli synthesize examples/synthetic-map.png
python -m blackflow_vision.cli recognize-map `
  examples/synthetic-map.png `
  --config config/recognition.default.json `
  --output data/output/synthetic
```

识别 PC 地图截图：

```powershell
python -m blackflow_vision.cli recognize-map `
  data/private/raw/2026-07-26/layer01_map_normal_full.png `
  --config config/recognition.default.json `
  --pc-frame `
  --output data/output/layer01
```

直接识别路径 UI 和林中节点：

```powershell
python -m blackflow_vision.cli recognize-path-ui `
  data/private/raw/2026-07-26/layer03_map_normal_full.png `
  --output data/output/path-ui/layer03
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 输出文件

- `map-state.json`：地图节点与边的机器可读结果。
- `annotated.png`：普通节点识别调试图。
- `road-mask.png`：旧版道路像素基线。
- `path-mask.png`：直接从 UI 提取的路径证据。
- `path-skeleton.png`：从路径掩膜生成的单像素拓扑。
- `path-ui-annotated.png`：青色路径、黄色林中节点和绿色临时无向边。
- `path-ui-state.json`：路径 UI、林中节点和临时无向图的 JSON 结果。

## 零件识别

零件识别需要用户在本地提供模板：

```powershell
python -m blackflow_vision.cli recognize-parts `
  examples/parts.png `
  --config config/parts.default.json `
  --templates data/private/part-templates `
  --output data/output/parts
```

## PC 后台只读截图

实时入口通过窗口标题找到游戏，并使用官方 MaaFramework 的 Win32 后台截图
控制器。FramePool 与 PrintWindow 会同时提供给框架，由框架选择可用方式。

```powershell
$env:PYTHONPATH = "src;."
python -m integration.maafw.live_capture `
  --window-title "明日方舟" `
  --output data/output/live
```

程序默认每 250 毫秒采样一次，等待连续三帧稳定后才提交识别。它只保存最近
一次被接受的截图和状态 JSON，不会发送鼠标或键盘输入。

## 当前限制

- 游戏客户区应为 16:9。
- 当前只支持接近水平或垂直的地图拓扑。
- 被弹窗遮挡或无法判断的界面不会进入地图识别。
- 伸出画面边界的路径会保持“未解析”，直到其他视角提供重叠证据。
- 传统路径 UI 原型仍会把部分节点图标和文字边缘识别为路径。
- 大型语义节点尚未完整加入路径切分，因此临时边可能跨过作战或事件节点。
- 所有结果仅用于辅助判断，不能直接驱动自动操作。

更详细的效果与问题记录见
[路径 UI 原型报告](docs/path-ui-prototype-2026-07-26.md)。

## MaaFramework 边界

`maafw_project/resource/pipeline/recognition.json` 定义了只读的自定义识别节点，
动作类型为 `DoNothing`。`integration/maafw/agent_main.py` 注册了
MaaFramework 5.12.2 Python Agent 回调。

当前没有注册自定义操作，也不会通过 Controller 发送游戏输入。视觉识别核心
可以在完全不启动 MaaFramework 的情况下独立运行和测试。

## 私有校准数据

用户提供的真实截图及其衍生裁剪都应放入 `data/private/`。该目录已被
`.gitignore` 排除，不应将游戏截图或用户数据上传到公开仓库。

当前本地校准集覆盖：

- 第一、二、三层地图；
- 零件箱详情界面；
- 移动方式选择界面；
- 第三层完整视野和拖动后的局部视野。

这些图片只用于早期校准，不能作为正式测试集。

## 后续需要的数据

- 每层地图向不同方向拖动时的短截图序列；
- 30–50 张不同层级和缩放状态的完整地图；
- 10–20 张零件箱与移动选择界面；
- 每种常见节点和零件至少 5 个样本；
- 独立的训练、调参和最终留出测试数据。
