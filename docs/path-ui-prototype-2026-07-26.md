# Path UI + forest-node prototype — 2026-07-26

## Recognition order

1. Normalize the PC client image to 1280x720.
2. Crop the map and process it at half resolution.
3. At several widths, find opposite LAB-lightness gradients that can be the
   two pale banks of a translucent path.
4. Confirm the full pale-bank / transparent-centre / pale-bank profile. The
   geometric mean of this response and the gradient-pair response becomes the
   path-centre probability.
5. Require long horizontal or vertical support, bridge only short gaps, and
   scale the mask back into normalized coordinates.
6. Detect nodes independently and remove only their icon protection circles.
   Labels are no longer erased because that used to cut downward paths.
7. Score a possible edge only on the remaining corridor outside both endpoint
   radii; node adjacency alone never creates an edge.
8. Split the recognized path mask at nodes and emit canonical
   undirected edges.

Nodes do not create path pixels. Every emitted edge must have occupancy in the
independently generated path mask.

Global graph connectivity is deliberately **not** a recognition constraint.
Multiple connected components may be valid game output. Connectivity can be
reported to the planner as a diagnostic, but it never creates, deletes or
bridges a visually recognized path.

Source screenshots are first normalized to 1280x720, then the map crop is
processed at half resolution over several possible bank widths. Regression
tests resize the same PC frame to 75%, 100% and 125% of its source resolution
and require an identical normalized path graph.

## Current reviewed-fixture result

| Frame | Reviewed edges recovered | Extra candidates |
|---|---:|---:|
| Layer 1 full | 13 / 13 | 0 |
| Layer 2 full | 18 / 19 | 0 |
| Layer 3 full | 35 / 36 | 2 |
| Layer 3 partial pan | 13 / 15 | 3 |

The cyan overlay is raw path UI evidence. Yellow circles are forest-node
candidates. Green lines are graph edges whose protected corridor occupancy is
at least 0.35.

On the current machine, after OpenCV warm-up, the layer-three full screenshot
takes about 25 ms for paired-bank segmentation, 18--26 ms for skeletonization,
and 20--22 ms for forest-node detection. These numbers are measurements, not
portable latency guarantees.

## User-feedback regression

The first acceptance pass exposed a systematic error: Canny/Hough was treating
large node icons, labels and glow as path pixels, and a 25-pixel close joined
those false fragments. The current implementation removes that global
edge-based stage.

The private regression suite now asserts that ten reported false edges remain
below the path threshold, the reported missing vertical edge remains above it,
the reported nonexistent partial-view node is not accepted as a forest circle,
and at least 95% of reviewed full-map edges are recovered with no more than two
extra candidates.

## Known limitations

- Long pale background structures can still imitate a double bank. The
  acceptance page exposes both score maps and directional masks so those
  failures can be separated from graph-conversion failures.
- A local tangent-continuity filter removes short paired edges from text,
  particles and background speckles. It does not require different path
  components to join.
- The semantic-node detector must provide a reliable center and radius. Missing
  a large icon can leave icon strokes in the raw debug mask.
- Only forest nodes are included in this experiment. Consequently an edge can
  span across an unrecognized encounter node. The full graph must add semantic
  node instances before it is planning-safe.
- The partial-pan screenshot deliberately contains incomplete path rendering.
  It remains less reliable than a stable full-map capture and must be fused
  with another viewport before planning.

All results remain `planner_ready: false`.
