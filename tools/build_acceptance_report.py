from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blackflow_vision.screen import normalize_pc_frame
from blackflow_vision.path_ui import (
    DirectPathUiRecognizer,
    ForestNode,
    extract_mask_supported_edges,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "private"
        / "annotations"
        / "2026-07-26"
        / "recognized-scenes.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "acceptance",
    )
    return parser


def _image_data_uri(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read {path}")
    normalized, _ = normalize_pc_frame(image)
    success, encoded = cv2.imencode(
        ".jpg",
        normalized,
        [cv2.IMWRITE_JPEG_QUALITY, 68],
    )
    if not success:
        raise RuntimeError(f"unable to encode {path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _path_mask_data(
    path: Path,
    nodes: list[dict],
) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read {path}")
    normalized, _ = normalize_pc_frame(image)
    protected_regions = tuple(
        (
            int(node["center"][0]),
            int(node["center"][1]),
            12 if node["kind"] == "forest" else 34,
        )
        for node in nodes
    )
    recognizer = DirectPathUiRecognizer()
    started = time.perf_counter()
    result, mask, skeleton = recognizer.analyze(
        normalized,
        protected_regions=protected_regions,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    graph_nodes = tuple(
        ForestNode(
            id=node["id"],
            center=(int(node["center"][0]), int(node["center"][1])),
            radius=12 if node["kind"] == "forest" else 34,
            confidence=1.0,
            evidence=("node_icon_protection_region",),
        )
        for node in nodes
    )
    method_edges = [
        [edge.first, edge.second]
        for edge in extract_mask_supported_edges(mask, graph_nodes)
    ]
    debug_layers = {
        "路径掩膜": mask,
        "路径骨架": skeleton,
        "横向双边缘响应": recognizer.debug_layers[
            "horizontal_bank_score"
        ],
        "纵向双边缘响应": recognizer.debug_layers[
            "vertical_bank_score"
        ],
        "横向候选": recognizer.debug_layers["horizontal_path_mask"],
        "纵向候选": recognizer.debug_layers["vertical_path_mask"],
    }
    encoded_layers = {}
    for label, layer in debug_layers.items():
        if "响应" in label:
            color = cv2.applyColorMap(layer, cv2.COLORMAP_TURBO)
            alpha = np.where(layer > 8, 185, 0).astype(np.uint8)
            overlay = cv2.cvtColor(
                np.dstack((color, alpha)),
                cv2.COLOR_BGRA2RGBA,
            )
        else:
            overlay = cv2.cvtColor(
                cv2.merge(
                    (
                        np.full_like(layer, 255),
                        np.full_like(layer, 190),
                        np.zeros_like(layer),
                        np.where(layer > 0, 165, 0).astype(np.uint8),
                    )
                ),
                cv2.COLOR_BGRA2RGBA,
            )
        success, encoded = cv2.imencode(".png", overlay)
        if not success:
            raise RuntimeError(f"unable to encode {label} for {path}")
        encoded_layers[label] = (
            "data:image/png;base64,"
            + base64.b64encode(encoded).decode("ascii")
        )
    return {
        "method_debug_layers": encoded_layers,
        "method_elapsed_ms": round(elapsed_ms, 2),
        "method_line_evidence_count": result.line_evidence_count,
        "method_edges": method_edges,
    }


def _build_html(dataset: dict) -> str:
    encoded = json.dumps(dataset, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>黑流树海识别验收</title>
<style>
:root {{
  color-scheme: dark;
  --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3;
  --muted:#8b949e; --accent:#2f81f7; --good:#3fb950; --bad:#f85149;
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif }}
header {{ padding:16px 20px; border-bottom:1px solid var(--line) }}
h1,h2,h3,p {{ margin:0 }}
h1 {{ font-size:20px; font-weight:600 }}
.sub {{ color:var(--muted); margin-top:4px }}
.controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center;
  padding:12px 20px; border-bottom:1px solid var(--line) }}
select,button,textarea {{ color:var(--text); background:var(--panel);
  border:1px solid var(--line); border-radius:6px; padding:7px 9px }}
button {{ cursor:pointer }}
button.active-good {{ border-color:var(--good); color:var(--good) }}
button.active-bad {{ border-color:var(--bad); color:var(--bad) }}
label {{ display:flex; gap:6px; align-items:center }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 330px; gap:16px;
  padding:16px; align-items:start }}
.stage {{ position:relative; aspect-ratio:16/9; background:#000;
  overflow:hidden; border:1px solid var(--line); border-radius:8px }}
.stage img,.stage svg {{ position:absolute; inset:0; width:100%; height:100% }}
.stage svg {{ pointer-events:none }}
.stage svg.editing {{ pointer-events:auto }}
.stage svg.editing .node {{ cursor:pointer }}
.stage svg.editing .edge.calibrated {{ cursor:pointer; pointer-events:stroke }}
.node.selected {{ fill:#fff17688; stroke:#fff176 }}
.edge {{ stroke:#41d1ff; stroke-width:3; vector-effect:non-scaling-stroke;
  filter:drop-shadow(0 0 2px #000) }}
.edge.method {{ stroke:#3fb950; stroke-width:4 }}
.edge.calibrated {{ stroke:#41d1ff; stroke-dasharray:7 5 }}
.edge.ref {{ stroke-dasharray:7 5; opacity:.65 }}
.node {{ stroke-width:3; vector-effect:non-scaling-stroke; fill:#0009 }}
.node.forest {{ stroke:#ffe066 }}
.node.current {{ stroke:#58d68d; fill:#58d68d55 }}
.node.semantic {{ stroke:#ff8f5b }}
.node-label {{ fill:#fff; stroke:#000; stroke-width:3; paint-order:stroke;
  font-size:15px; font-weight:600 }}
.side {{ display:grid; gap:12px }}
.panel {{ background:var(--panel); border:1px solid var(--line);
  border-radius:8px; padding:12px }}
.panel h2 {{ font-size:15px; margin-bottom:9px }}
.kv {{ display:grid; grid-template-columns:1fr auto; gap:5px 12px }}
.kv dt {{ color:var(--muted) }}
.kv dd {{ margin:0; text-align:right }}
.list {{ margin:0; padding-left:18px; max-height:230px; overflow:auto }}
.badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px;
  padding:2px 7px; color:var(--muted); margin-right:5px }}
textarea {{ width:100%; min-height:76px; resize:vertical; margin-top:8px }}
.review-actions {{ display:flex; gap:8px; margin-top:8px }}
.review-actions button {{ flex:1 }}
.counts {{ color:var(--muted); margin-bottom:8px }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr }} }}
</style>
</head>
<body>
<header>
  <h1>黑流树海识别验收</h1>
  <p class="sub">逐张检查游戏状态、节点类型、路径连接和遮挡边界；验收记录保存在浏览器本地。</p>
</header>
<div class="controls">
  <label>截图 <select id="frame-select"></select></label>
  <label><input id="show-method-edges" type="checkbox" checked>算法判定边</label>
  <label>算法中间层
    <select id="method-layer"><option value="">关闭</option></select>
  </label>
  <label><input id="show-edges" type="checkbox">人工校准边</label>
  <label><input id="show-nodes" type="checkbox" checked>节点</label>
  <label><input id="show-labels" type="checkbox" checked>标签</label>
  <label><input id="edit-edges" type="checkbox">编辑人工边</label>
  <button id="reset-edges">重置人工边</button>
  <button id="export-review">导出验收结果</button>
</div>
<div class="controls">
  <span class="sub">编辑人工边：点击已有虚线可删除；依次点击两个节点可添加或取消一条边。算法边不会随人工修订改变。</span>
</div>
<main>
  <div class="stage">
    <img id="scene-image" alt="待验收游戏截图">
    <img id="method-mask" alt="算法路径识别中间层">
    <svg id="overlay" viewBox="0 0 1280 720" role="img"
      aria-label="识别出的节点与路径"></svg>
  </div>
  <div class="side">
    <section class="panel">
      <h2 id="scene-title"></h2>
      <div id="screen-badges"></div>
      <p class="counts" id="counts"></p>
      <dl class="kv" id="state"></dl>
    </section>
    <section class="panel" id="parts-panel">
      <h2>零件 / 移动方式</h2>
      <ul class="list" id="parts"></ul>
    </section>
    <section class="panel">
      <h2>节点清单</h2>
      <ul class="list" id="nodes"></ul>
    </section>
    <section class="panel">
      <h2>本张验收</h2>
      <textarea id="review-note" placeholder="记录错误节点、漏路或状态字段问题"></textarea>
      <div class="review-actions">
        <button id="accept">正确</button>
        <button id="reject">需修改</button>
      </div>
      <textarea id="export-output" hidden readonly aria-label="验收结果 JSON"></textarea>
    </section>
  </div>
</main>
<script>
const DATA={encoded};
const byId=Object.fromEntries(DATA.frames.map(f=>[f.id,f]));
const FIELD_LABELS={{
  map_name:"地图",layer:"层数",action_points:"行动力",target_life:"目标生命值",
  shield:"护盾",command_level:"指挥等级",command_exp:"指挥经验",hope:"希望",
  collectibles:"收藏品",total_part_value:"零件总估价",part_box:"零件箱",
  squad_size:"编队人数",movement_mode:"移动方式",selected_target:"选择目标",
  warning:"警告"
}};
const select=document.getElementById("frame-select");
const svg=document.getElementById("overlay");
const ns="http://www.w3.org/2000/svg";
const reviews=JSON.parse(localStorage.getItem("blackflow-acceptance-v4")||"{{}}");
let selectedNodeId=null;
for(const frame of DATA.frames){{
  const option=document.createElement("option");
  option.value=frame.id; option.textContent=frame.title; select.append(option);
}}
function effective(frame){{
  if(!frame.map_reference) return {{nodes:frame.nodes||[],edges:frame.edges||[],ref:false}};
  const source=byId[frame.map_reference];
  return {{nodes:source.nodes||[],edges:source.edges||[],ref:true}};
}}
function canonicalEdge(a,b){{ return a<b?[a,b]:[b,a]; }}
function reviewedEdges(frame,graph){{
  const corrected=(reviews[frame.id]||{{}}).corrected_edges;
  return corrected||graph.edges;
}}
function persistReview(frameId, patch){{
  reviews[frameId]={{
    ...(reviews[frameId]||{{}}),
    ...patch,
    reviewed_at:new Date().toISOString()
  }};
  localStorage.setItem("blackflow-acceptance-v4",JSON.stringify(reviews));
}}
function formatValue(value){{
  if(value===null) return "—";
  if(Array.isArray(value)) return value.map(formatValue).join("；");
  if(typeof value==="object"){{
    if("current" in value && "max" in value) return `${{value.current}}/${{value.max}}`;
    return JSON.stringify(value);
  }}
  return String(value);
}}
function kindClass(kind){{
  if(kind==="forest") return "forest";
  if(kind==="current") return "current";
  return "semantic";
}}
function render(){{
  const frame=byId[select.value||DATA.frames[0].id];
  const graph=effective(frame);
  const correctedEdges=reviewedEdges(frame,graph);
  const editing=document.getElementById("edit-edges").checked;
  svg.classList.toggle("editing",editing);
  document.getElementById("scene-image").src=frame.image_data;
  const methodMask=document.getElementById("method-mask");
  const methodLayer=document.getElementById("method-layer");
  const debugLayers=frame.method_debug_layers||{{}};
  const oldLayer=methodLayer.value;
  methodLayer.textContent="";
  const off=document.createElement("option");
  off.value=""; off.textContent="关闭"; methodLayer.append(off);
  for(const label of Object.keys(debugLayers)){{
    const option=document.createElement("option");
    option.value=label; option.textContent=label; methodLayer.append(option);
  }}
  methodLayer.value=oldLayer in debugLayers?oldLayer:"";
  methodMask.src=debugLayers[methodLayer.value]||"";
  methodMask.hidden=!methodLayer.value;
  document.getElementById("scene-title").textContent=frame.title;
  document.getElementById("screen-badges").innerHTML=
    `<span class="badge">${{frame.screen.kind}}</span>`+
    `<span class="badge">${{frame.screen.viewport}}</span>`+
    (frame.screen.occlusion!=="none"?`<span class="badge">${{frame.screen.occlusion}}</span>`:"");
  document.getElementById("counts").textContent=
    `${{graph.nodes.length}} 个节点 · ${{correctedEdges.length}} 条人工边`+
    ((reviews[frame.id]||{{}}).corrected_edges?"（已修订）":"")+
    (frame.method_edges?
      ` · 算法判定 ${{frame.method_edges.length}} 条`:"")+
    (frame.method_line_evidence_count!==undefined?
      ` · ${{frame.method_line_evidence_count}} 个算法路径组件`:"")+
    (frame.method_elapsed_ms!==undefined?
      ` · 本次完整识别 ${{frame.method_elapsed_ms.toFixed(1)}} ms`:"")+
    (graph.ref?" · 引用完整地图，遮挡区域需复核":"");
  const state=document.getElementById("state"); state.textContent="";
  for(const [key,value] of Object.entries(frame.state||{{}})){{
    if(["movement_options"].includes(key)) continue;
    const dt=document.createElement("dt"); dt.textContent=FIELD_LABELS[key]||key;
    const dd=document.createElement("dd"); dd.textContent=formatValue(value);
    state.append(dt,dd);
  }}
  const extras=[...(frame.parts||[]),...((frame.state||{{}}).movement_options||[])];
  const parts=document.getElementById("parts"); parts.textContent="";
  for(const item of extras){{
    const li=document.createElement("li");
    li.textContent=`${{item.name}}${{item.remaining_uses!==undefined&&item.remaining_uses!==null?` · 剩余${{item.remaining_uses}}次`:""}}${{item.effect?` · ${{item.effect}}`:""}}`;
    parts.append(li);
  }}
  document.getElementById("parts-panel").hidden=extras.length===0;
  const nodeList=document.getElementById("nodes"); nodeList.textContent="";
  for(const node of graph.nodes){{
    const li=document.createElement("li");
    li.textContent=`${{node.id}} · ${{node.label}} · (${{node.center.join(", ")}})`;
    nodeList.append(li);
  }}
  svg.textContent="";
  const nodeMap=Object.fromEntries(graph.nodes.map(n=>[n.id,n]));
  if(document.getElementById("show-method-edges").checked){{
    for(const [a,b] of frame.method_edges||[]){{
      if(!nodeMap[a]||!nodeMap[b]) continue;
      const line=document.createElementNS(ns,"line");
      line.setAttribute("x1",nodeMap[a].center[0]); line.setAttribute("y1",nodeMap[a].center[1]);
      line.setAttribute("x2",nodeMap[b].center[0]); line.setAttribute("y2",nodeMap[b].center[1]);
      line.setAttribute("class","edge method"); svg.append(line);
    }}
  }}
  if(document.getElementById("show-edges").checked){{
    for(const [a,b] of correctedEdges){{
      if(!nodeMap[a]||!nodeMap[b]) continue;
      const line=document.createElementNS(ns,"line");
      line.setAttribute("x1",nodeMap[a].center[0]); line.setAttribute("y1",nodeMap[a].center[1]);
      line.setAttribute("x2",nodeMap[b].center[0]); line.setAttribute("y2",nodeMap[b].center[1]);
      line.setAttribute("class","edge calibrated"+(graph.ref?" ref":""));
      line.addEventListener("click",event=>{{
        if(!document.getElementById("edit-edges").checked) return;
        event.stopPropagation();
        const key=canonicalEdge(a,b).join("|");
        const next=reviewedEdges(frame,graph).filter(
          edge=>canonicalEdge(edge[0],edge[1]).join("|")!==key
        );
        persistReview(frame.id,{{corrected_edges:next,status:"needs_changes"}});
        selectedNodeId=null; render();
      }});
      svg.append(line);
    }}
  }}
  if(document.getElementById("show-nodes").checked){{
    for(const node of graph.nodes){{
      const circle=document.createElementNS(ns,"circle");
      circle.setAttribute("cx",node.center[0]); circle.setAttribute("cy",node.center[1]);
      circle.setAttribute("r",node.kind==="forest"?11:18);
      circle.setAttribute(
        "class",
        `node ${{kindClass(node.kind)}}${{selectedNodeId===node.id?" selected":""}}`
      );
      circle.addEventListener("click",event=>{{
        if(!document.getElementById("edit-edges").checked) return;
        event.stopPropagation();
        if(selectedNodeId===null){{
          selectedNodeId=node.id; render(); return;
        }}
        if(selectedNodeId===node.id){{
          selectedNodeId=null; render(); return;
        }}
        const key=canonicalEdge(selectedNodeId,node.id);
        const keyText=key.join("|");
        const current=reviewedEdges(frame,graph).map(
          edge=>canonicalEdge(edge[0],edge[1])
        );
        const exists=current.some(edge=>edge.join("|")===keyText);
        const next=exists
          ? current.filter(edge=>edge.join("|")!==keyText)
          : [...current,key].sort((a,b)=>a.join("|").localeCompare(b.join("|")));
        persistReview(frame.id,{{corrected_edges:next,status:"needs_changes"}});
        selectedNodeId=null; render();
      }});
      svg.append(circle);
      if(document.getElementById("show-labels").checked){{
        const text=document.createElementNS(ns,"text");
        text.setAttribute("x",node.center[0]+14); text.setAttribute("y",node.center[1]-14);
        text.setAttribute("class","node-label"); text.textContent=node.label; svg.append(text);
      }}
    }}
  }}
  const review=reviews[frame.id]||{{status:"",note:""}};
  document.getElementById("review-note").value=review.note||"";
  document.getElementById("accept").className=review.status==="accepted"?"active-good":"";
  document.getElementById("reject").className=review.status==="needs_changes"?"active-bad":"";
}}
function saveReview(status){{
  persistReview(select.value,{{
    status,
    note:document.getElementById("review-note").value
  }});
  render();
}}
select.addEventListener("change",()=>{{selectedNodeId=null;render();}});
for(const id of ["show-method-edges","method-layer","show-edges","show-nodes","show-labels"]) document.getElementById(id).addEventListener("change",render);
document.getElementById("edit-edges").addEventListener("change",event=>{{
  selectedNodeId=null;
  if(event.target.checked){{
    document.getElementById("show-edges").checked=true;
    document.getElementById("show-nodes").checked=true;
  }}
  render();
}});
document.getElementById("reset-edges").addEventListener("click",()=>{{
  const current=reviews[select.value]||{{}};
  delete current.corrected_edges;
  reviews[select.value]=current;
  localStorage.setItem("blackflow-acceptance-v4",JSON.stringify(reviews));
  selectedNodeId=null; render();
}});
document.getElementById("accept").addEventListener("click",()=>saveReview("accepted"));
document.getElementById("reject").addEventListener("click",()=>saveReview("needs_changes"));
document.getElementById("review-note").addEventListener("change",()=>saveReview((reviews[select.value]||{{}}).status||"unreviewed"));
document.getElementById("export-review").addEventListener("click",()=>{{
  const output=JSON.stringify({{schema_version:DATA.schema_version,reviews}},null,2);
  const exportOutput=document.getElementById("export-output");
  exportOutput.hidden=false; exportOutput.value=output;
  const blob=new Blob([output],{{type:"application/json"}});
  const link=document.createElement("a"); link.href=URL.createObjectURL(blob);
  link.download="blackflow-acceptance-review.json";
  document.body.append(link); link.click(); link.remove();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000);
}});
select.value=DATA.frames[0].id; render();
</script>
</body>
</html>"""


def main() -> int:
    args = _parser().parse_args()
    annotations = args.annotations.resolve()
    dataset = json.loads(annotations.read_text(encoding="utf-8"))
    for frame in dataset["frames"]:
        image_path = (annotations.parent / frame["image"]).resolve()
        frame["image_data"] = _image_data_uri(image_path)
        if frame["screen"]["kind"] == "map":
            frame.update(_path_mask_data(image_path, frame.get("nodes", [])))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.html").write_text(
        _build_html(dataset),
        encoding="utf-8",
    )
    portable = {
        **dataset,
        "frames": [
            {
                key: value
                for key, value in frame.items()
                if key not in {"image_data", "method_debug_layers"}
            }
            for frame in dataset["frames"]
        ],
    }
    (args.output / "recognition-data.json").write_text(
        json.dumps(portable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
