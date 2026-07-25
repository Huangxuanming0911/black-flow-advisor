# Real screenshot calibration report — 2026-07-26

## Dataset

Six lossless, user-provided PC screenshots are stored in the ignored private
dataset directory. The SHA-256 manifest records original dimensions, layer,
page state, viewport state and visible action/part facts.

This is a calibration seed, not an evaluation set. No screenshot or derived
game asset should be committed to a public repository.

## What works on all six images

- Windows title-bar detection and client-area crop.
- Reversible coordinate transform to a 1280x720 recognition frame.
- Structural classification of:
  - normal map;
  - toolbox detail;
  - movement selector.
- Stable-frame gating and unchanged-frame deduplication.
- Explicit representation of clipped viewport sides.
- Safe grid-translation fusion for overlapping partial graph observations.

## Real-map baseline result

| Image | Candidates | Accepted roads | Planner ready |
|---|---:|---:|---|
| Layer 1 full | 5 | 0 | no |
| Layer 2 full | 10 | 0 | no |
| Layer 3 full | 10 | 0 | no |
| Layer 3 partial pan | 9 | 0 | no |

The original detector was tuned on synthetic fixtures. On real frames it
detects some large encounter icons, misses small junction beads, and sometimes
locks onto high-contrast detail inside an icon. The original road threshold
also assumes bright paths, whereas the real paths are dark, textured ridges.
These failures are preserved in `data/output/real-seed/`; every result contains
`planner_ready: false`.

## Consequence for the next CV milestone

Do not lower confidence thresholds to make the output look successful. Split
the problem into three evidence channels:

1. small junction detector;
2. encounter-icon detector/classifier;
3. road-segment detector conditioned on candidate endpoints.

Fuse those channels with lattice and topology constraints, then evaluate on a
holdout run. For partial views, a road touching the viewport boundary remains
an unresolved boundary port; it is never extrapolated from a single frame.

## Live validation still required

The read-only MaaFramework Win32 loop is implemented, but no window titled
`明日方舟` was running during this calibration. A real connection test requires
the game window to be open. The loop emits no mouse or keyboard input.
