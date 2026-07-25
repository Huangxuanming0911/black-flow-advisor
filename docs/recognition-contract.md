# Recognition contract

The recognition layer may observe screenshots but must not decide or execute a
route.

## Trust states

- `raw`: produced by detectors.
- `review_required`: detector output with confidence and issues.
- `human_verified`: a user has corrected and accepted all nodes, edges and
  resources.

Only `human_verified` state may be passed to a future planner.

The current CLI always emits `planner_ready: false`.

## MaaFramework process boundary

MaaFramework host and AgentServer checks run in separate processes. The Agent
binary intentionally does not implement host-only APIs such as `Resource`.
Loading a resource bundle from inside the Agent process is therefore invalid;
it is not an integration failure.

## Failure policy

Recognition must fail closed:

- Wrong resolution: reject.
- No nodes: reject for planning.
- Grid fit failure: reject for planning.
- Duplicate grid cells: require correction.
- Missing or ambiguous current position: require correction.
- Any low-confidence road: require correction.

The system must never invent hidden or occluded nodes.

## Dataset split

Split by complete game run, not individual screenshot. Frames from one run are
too visually similar and would leak into the holdout set.

Private screenshots, crops, manifests and annotations are not repository
content. Only synthetic fixtures and user-authored metadata may be committed.
