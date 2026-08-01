# Black Flow Advisor

[English](README.md) | [简体中文](README.zh-CN.md)

Independent, read-only recognition baseline for the map phase of Arknights
Integrated Strategies: Black Flow.

This repository intentionally does **not** copy, fork, or depend on any existing
Black Flow route-planning project. MaaFramework is treated only as an optional
official capture/orchestration host.

## Current scope

- Normalize variable-size PC screenshots (including a Windows title bar) to a
  reversible 1280x720 client coordinate system.
- Classify map, toolbox-detail and movement-selector UI states.
- Wait for a stable frame and deduplicate unchanged captures.
- Detect circular map nodes inside a configured map ROI.
- Fit nodes to discrete rows and columns.
- Reconstruct orthogonal road edges using corridor evidence.
- Classify node crops when locally supplied templates are available.
- Inspect a fixed parts-panel grid and classify occupied slots from local
  templates.
- Emit JSON, an annotated PNG, confidence scores, and validation issues.
- Merge overlapping partial graph observations only when at least two
  compatible nodes establish a safe grid translation.
- Refuse to mark a result as planner-ready without human verification.

No planner, game input, ADB clicking, account automation, or bundled game
assets are included in this milestone. Real screenshots stay under ignored
`data/private/`.

## Run locally

The current workspace already has Python 3.12, NumPy, OpenCV and Pillow:

```powershell
$env:PYTHONPATH = "src"
python -m blackflow_vision.cli synthesize examples/synthetic-map.png
python -m blackflow_vision.cli recognize-map `
  data/private/raw/2026-07-26/layer01_map_normal_full.png `
  --config config/recognition.default.json `
  --pc-frame `
  --output data/output/layer01
python -m blackflow_vision.cli recognize-path-ui `
  data/private/raw/2026-07-26/layer03_map_normal_full.png `
  --output data/output/path-ui/layer03
python -m blackflow_vision.cli synthesize-parts `
  examples/synthetic-parts.png `
  --templates examples/synthetic-part-templates
python -m unittest discover -s tests -v
```

## Accept the six calibrated screenshots

The six current screenshots have a manually reviewed baseline covering visible
HUD state, screen type, parts/movement choices, map nodes, and undirected
edges. Build the local acceptance report with:

```powershell
python tools/build_acceptance_report.py
start data/output/acceptance/index.html
```

The report switches among source images, toggles paths/nodes/labels, lists the
structured result, and exports per-frame approval or correction notes as JSON.

For deterministic regression output on one of the calibrated source images:

```powershell
python -m blackflow_vision.cli recognize-calibrated `
  data/private/raw/2026-07-26/layer03_map_normal_full.png `
  --manifest data/private/raw/2026-07-26/manifest.json `
  --annotations data/private/annotations/2026-07-26/recognized-scenes.json `
  --output data/output/acceptance/scenes/layer03-map-full.json
```

This command uses an exact SHA-256 match against the annotated set. It is an
acceptance/regression path, not evidence that the CV model generalizes to new
screenshots. Results remain `planner_ready: false` until user acceptance.

Outputs:

- `map-state.json`: machine-readable recognition result.
- `annotated.png`: nodes, grid coordinates, edges and confidence overlay.
- `road-mask.png`: debug view of pixels considered road evidence.
- `path-mask.png`: independently recognized path UI evidence.
- `path-skeleton.png`: one-pixel topology derived from the path UI.
- `path-ui-annotated.png`: cyan path pixels, yellow forest candidates and
  green provisional undirected edges.

Part recognition uses:

```powershell
python -m blackflow_vision.cli recognize-parts `
  examples/parts.png `
  --config config/parts.default.json `
  --templates data/private/part-templates `
  --output data/output/parts
```

## Real-time read-only capture

The live entry point locates the PC window by title and uses the official
MaaFramework Win32 background screenshot controller. FramePool and PrintWindow
are offered together so the framework can select a compatible method.

```powershell
$env:PYTHONPATH = "src;."
python -m integration.maafw.live_capture `
  --window-title "明日方舟" `
  --output data/output/live `
  --map-only
```

The loop samples every 250 ms, waits for three stable frames, classifies the UI
state, and runs direct path recognition on stable map frames. It atomically
publishes the latest path JSON, mask, skeleton and annotated image alongside
the accepted screenshot. The live graph currently has
`graph_scope: all_visible_nodes_geometry` and remains
`planner_ready: false`. It does
not issue input events.

## Offline node semantics

Capture and analysis are separate. `latest.png` is the normalized read-only
capture; path masks, graph annotations, OCR results and semantic annotations
are derived artifacts. Existing captures can therefore be reprocessed without
opening or recapturing the game window.

Install the optional offline OCR dependency and analyze a saved frame:

```powershell
python -m pip install -e ".[ocr]"
$env:PYTHONPATH = "src;."
python tools/analyze_node_semantics.py `
  data/output/live-full-node/latest.png `
  --output data/output/node-semantics
```

The recognizer runs Chinese OCR once over the complete frame, associates text
boxes with node geometry, corrects close matches against a node-label
vocabulary, and optionally validates the resulting type against icon templates
derived from the private reviewed dataset. Text remains authoritative: a
strong text/icon conflict is reported as `conflict_text_kept` and
`needs_review: true`; the icon never silently overwrites readable text.

The command writes:

- `node-semantics-annotated.png`: clean node labels and semantic types.
- `unified-map-graph.png`: paths, stable node IDs and semantic labels together.
- `node-semantics.json`: OCR and icon cross-validation details.
- `unified-map-graph.json`: planner-facing nodes, undirected edges, adjacency
  lists, connected components, isolated nodes and ambiguity diagnostics.

The unified graph joins the two stages by stable node ID. Its edges still come
only from direct path-UI evidence; semantic labels, grid proximity and global
connectivity never create a path. Connectivity is reported as a diagnostic
rather than enforced, so partial screenshots may legitimately contain several
components. Private screenshots and extracted icon templates remain under
`data/private/` and are not published.

## Black Flow planning knowledge base

`data/knowledge/black-flow-rules.v0.1.json` contains planner-facing node
effects, all currently documented parts, movement models, the three ending
routes and stage-to-region assignments. Its `vision_bridge` maps recognized
Chinese node labels to rule IDs while preserving ambiguity for unrevealed
nodes. See `docs/black-flow-knowledge.zh-CN.md` for the Chinese review and
sources; run `python tools/validate_knowledge_base.py` to validate the data.

## Controlled V0 assumptions and hard stops

- The game client is 16:9 after removing desktop chrome.
- Map graph axes remain orthogonal at the supported zoom levels.
- An obscured or unknown page is never sent to map recognition.
- Cropped viewport boundaries remain unresolved until another overlapping
  observation supplies evidence.
- Recognition is advisory; uncertain fields require correction.
- The fast path detector runs paired-bank and translucent-centre filters on a
  half-resolution map crop. The acceptance page exposes its response maps,
  directional candidates, mask and skeleton. Extremely faint paths and long
  background structures can still be confused, especially in partial views,
  so `planner_ready` remains false.
- Global connectivity is not a recognition constraint. Local tangent
  continuity suppresses short visual interference but never invents an edge
  to join separate components.

Real game screenshots and derived crops belong under `data/private/`, which is
ignored by Git. Do not commit proprietary game assets or user screenshots.

## MaaFramework boundary

`maafw_project/resource/pipeline/recognition.json` describes read-only Custom
Recognition nodes with `DoNothing`. `integration/maafw/agent_adapter.py`
contains an import-safe adapter around the recognition core.
`integration/maafw/agent_main.py` registers the exact
MaaFramework 5.12.2 Python callback contract.

The local development environment pins official `MaaFw==5.12.2`. The Agent
returns recognition detail only; no custom action or controller input is
registered. The recognition core remains runnable and testable without
MaaFramework.

## Private calibration seed

The six user-supplied screenshots are stored locally under
`data/private/raw/2026-07-26/` with a SHA-256 manifest. They cover all three
layers, toolbox detail, movement selection, and a panned partial layer-three
view. They are calibration inputs, not a statistically valid test set.

## Next evidence needed

The next milestone needs lossless frame sequences, not just isolated images:

- 10-20 short sequences while panning each map direction, including overlap.
- 30-50 complete map screenshots across layers and zoom states.
- 10-20 toolbox and movement-selector screenshots.
- At least 5 examples for each common node/part class.
- Separate runs for train/tuning and final holdout evaluation.

## Interactive route simulation

The first controlled planner consumes the recognized
`unified-map-graph.json` without changing or inventing its edges. Build the
local review page with:

```powershell
$env:PYTHONPATH = "src;."
python tools/build_route_planner.py
start data/output/route-planner/index.html
```

Click nodes in order to construct a route. Each step can use walking or a
recognized processed part. Reaching a paired tunnel forces an immediate
zero-action-point transfer to its other end. The
simulator keeps three ledgers separate:

- exact action-point and deterministic resource changes;
- post-completion rewards across seven dimensions, separating exact values,
  known expectations, ranges, and unresolved components;
- current part-box valuation, including consumed movement parts and documented
  dynamic valuation rules.

Pursuit is modeled as a forced encounter rather than a map node. It is
triggered when action points reach zero away from an exit; the normal variant
adds its fixed recruitment-ticket reward, while the boss variant remains an
  explicit placeholder until the current zone endpoint is known. Normal and
  emergency combat now use floor-specific, manually reviewed clean samples as
  confidence-weighted recommendation priors. Chest/unowned-wealth rewards and
  collectible-granted parts remain separate from the base result.

The page proposes combat, conservative, balanced, and exploration routes.
Every strategy may use processed parts and obeys the same reserve, forced
tunnel-transfer, and portal-entry constraints. Run
`python tools/build_empirical_rewards.py` after collecting more reviewed runs,
then rebuild the planner to refresh the empirical snapshot.

`data/knowledge/node-rewards.v0.1.json` records the current Black Flow node
reward matrix. Exact, choice-based, conditional, transaction, and
region/stage-dependent rewards remain distinguishable. Older Integrated
Strategies data is used only to shape the schema, never to fill missing Black
Flow numbers.

The page also supports manual semantic corrections for a previously visited
overlook and resident-occupied nodes. Normal completed nodes are treated as
ordinary forest clearings; their previous event identity is not retained by
the planner.

The local demo starts with several editable sample parts. Pass
`--no-sample-parts` to start with an empty part box.
