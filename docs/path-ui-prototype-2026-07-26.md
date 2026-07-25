# Path UI + forest-node prototype — 2026-07-26

## Recognition order

1. Normalize the PC client image to 1280x720.
2. Detect horizontal and vertical path UI edge evidence directly from pixels.
3. Consolidate line evidence into a path mask and skeleton.
4. Detect small forest-node circles independently.
5. Reject candidates inside large circular UI, candidates without path arms,
   and high-texture icon/detail candidates.
6. Split the recognized path mask at forest nodes and emit canonical
   undirected edges.

Nodes do not create path pixels. Every emitted edge must have occupancy in the
independently generated path mask.

## Calibration output

| Frame | Forest candidates | Provisional undirected edges |
|---|---:|---:|
| Layer 1 full | 9 | 6 |
| Layer 2 full | 5 | 3 |
| Layer 3 full | 14 | 15 |
| Layer 3 partial pan | 7 | 7 |

The cyan overlay is direct path UI evidence. Yellow circles are forest-node
candidates. Green lines are graph edges supported by path-mask occupancy.

## Known limitations

- Encounter icons and their labels also contain horizontal/vertical UI edges,
  so the classical path mask includes some icon pixels.
- Some forest candidates inside icons or labels remain.
- Only forest nodes are included in this experiment. Consequently an edge can
  span across an unrecognized encounter node. The full graph must add semantic
  node instances before it is planning-safe.
- Hough line evidence is a calibration bootstrap, not the final segmentation
  model. The generated masks should be corrected into training labels for a
  topology-aware segmentation model.

All results remain `planner_ready: false`.
