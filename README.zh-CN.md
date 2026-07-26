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

## 六张截图的验收

当前六张校准截图已有完整的人工核对基线，覆盖可见 HUD 状态、界面类型、
零件与移动方式，以及地图节点和无向边。运行：

```powershell
python tools/build_acceptance_report.py
start data/output/acceptance/index.html
```

验收页可以逐张切换原图，独立显示或隐藏路径、节点和标签，并列出结构化
状态与图数据。每张图可标记“正确”或“需修改”，填写备注后导出验收 JSON。

也可以对校准集中的原图执行确定性识别：

```powershell
python -m blackflow_vision.cli recognize-calibrated `
  data/private/raw/2026-07-26/layer03_map_normal_full.png `
  --manifest data/private/raw/2026-07-26/manifest.json `
  --annotations data/private/annotations/2026-07-26/recognized-scenes.json `
  --output data/output/acceptance/scenes/layer03-map-full.json
```

这个入口通过 SHA-256 精确匹配已标注截图，用于回归测试和人工验收，不代表
模型已经能泛化到任意新截图。用户完成验收前，结果仍为
`planner_ready: false`。

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
  --output data/output/live `
  --map-only
```

程序默认每 250 毫秒采样一次，等待连续三帧稳定后才提交识别。稳定地图帧会
运行新版路径识别，并原子更新最近一次截图、状态 JSON、路径 JSON、路径掩膜、
骨架和标注图。当前实时图包含可见林中节点、大型语义节点和当前位置，但语义与
路径仍需验收，因此标记为
`graph_scope: all_visible_nodes_geometry` 和 `planner_ready: false`。程序不会发送
鼠标或键盘输入。

## 当前限制

- 游戏客户区应为 16:9。
- 当前只支持接近水平或垂直的地图拓扑。
- 被弹窗遮挡或无法判断的界面不会进入地图识别。
- 伸出画面边界的路径会保持“未解析”，直到其他视角提供重叠证据。
- 快速路径识别以半分辨率检测“淡白边缘—透明中心—淡白边缘”结构，完整
  地图热身后的路径分割约为几十毫秒；验收页可分别查看双边缘响应、方向
  候选、掩膜和骨架。
- 极淡路径和长条背景纹理之间仍可能产生少量漏判或误判，局部视角必须与
  稳定的完整地图观测融合。
- 不把“整张图必须连通”作为识别规则；局部方向连续性只用于排除文字、粒子
  和短背景纹理，不会为了连通性凭空补边。
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

## 离线节点文字识别

截图与分析是分开的。`latest.png` 是标准化后的只读原图；路径掩膜、图结构、
OCR 结果和语义标注图都是根据原图生成的派生产物。因此，可以直接反复处理已有
截图，不需要重新打开或捕获游戏窗口。

安装可选的离线 OCR 依赖，然后分析保存的截图：

```powershell
python -m pip install -e ".[ocr]"
$env:PYTHONPATH = "src;."
python tools/analyze_node_semantics.py `
  data/output/live-full-node/latest.png `
  --output data/output/node-semantics
```

识别器会对整幅画面运行一次中文 OCR，把文字框关联到最近的节点，根据节点词典
修正常见单字误识别，再使用私有人工校准集提取的图标模板进行交叉验证。文字是
主证据：如果文字和图标发生强冲突，程序保留文字结果，并输出
`conflict_text_kept` 和 `needs_review: true`，不会让图标静默覆盖文字。

命令会生成：

- `node-semantics-annotated.png`：干净的节点文字与事件类型标注；
- `unified-map-graph.png`：路径、稳定节点 ID 和事件文字的统一验收图；
- `node-semantics.json`：OCR 与图标交叉验证细节；
- `unified-map-graph.json`：供规划器使用的节点、无向边、邻接表、连通分量、
  孤立点和歧义诊断。

统一图通过稳定节点 ID 合并两个识别阶段。图中的边依然只来自路径 UI 的直接
像素证据；节点文字、网格距离和“应当连通”的假设都不会创建路径。连通性仅作为
诊断信息输出而非强约束，因此局部截图可以合理地包含多个连通分量。原始截图和
本地提取的图标模板继续保存在 `data/private/`，不会上传到公开仓库。
