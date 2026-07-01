# Plan: Multi-Placement Board Support for Handeye Calibration

**Goal.** Allow the operator to record samples with the ChArUco board at multiple
named physical positions ("placements"), then solve for a single camera-to-EEF
transform X that is consistent across all placements.  Mirrors the pan_tilt
calibration workflow where `phase1_waypoints` (canonical park) and
`phase1_waypoints_custom` (alternate park) are combined in the polish step.

**Date:** 2026-06-28

---

## Background and key constraint

`T_eef_cam = X` is a rigid physical constant — it does not depend on where the
board sits.  What changes between placements is `T_base_board` (Tbb).

The AX = XB closed-form seeds compute *relative* EEF and board-in-camera motions
between pairs of poses.  Pairing samples **across** placements is invalid: if the
board moved, the relative B_ij includes board motion as well as camera motion,
corrupting the solve.  Pairs must be drawn **within** each placement.

The bundle adjust fixes this cleanly: each placement has its own Tbb in the
parameter vector; X is shared.  The per-placement independent solve seeds all
Tbbs; the best independent X seeds the shared X.

---

## Phase overview

| Phase | Scope | Files touched |
|---|---|---|
| 1 | Data model + node state | `handeye_web.py`, new `placement_state.py` |
| 2 | Session persistence | `handeye_sessions.py` |
| 3 | API layer | `handeye_web.py` (endpoints), `web_support.py` |
| 4 | Multi-placement solver | `handeye_solve.py` |
| 5 | Web UI | `webui/app.js`, `webui/index.html` |
| 6 | Tests | `test/` |

---

## Phase 1 — Data model and node state

### 1.1  `PlacementState` dataclass

Create `handeye_calib/placement_state.py` (pure Python, no ROS):

```python
@dataclass
class PlacementState:
    label: str                          # user-given name, e.g. "table_left"
    session: CaptureSession             # samples accumulated for this placement
    thumbs: dict                        # {int idx: bytes jpeg}
    sample_joints: dict                 # {int idx: list[float] | None}
    sample_ts: dict                     # {int idx: float monotonic}
    sample_reproj_px: dict              # {int idx: float}
    sample_area_frac: dict              # {int idx: float}
    sample_depth_source: dict           # {int idx: str}
    anchor_obs: list                    # list of 4x4 T_base_board from head
    tbb_head: np.ndarray | None         # averaged T_base_board from head
    anchor_scatter: dict | None         # {"trans_mm","rot_deg","n",...}
```

The `label` is displayed in the UI; a derived `id` (slugified label, collision-
deduped with a counter suffix) is the dict key and the path segment used on disk.

### 1.2  `HandeyeWebNode` state refactor

Replace the flat per-sample sidecar fields with:

```python
self._placements: dict[str, PlacementState]   # ordered, insertion order = creation order
self._active_placement_id: str                # key into _placements
```

Initialise with one placement (`id="default"`, `label="default"`) at startup.

**Backward-compatibility shim.**  All existing code that reads
`self.session`, `self._thumbs`, `self._sample_joints`, etc. is refactored
to delegate through `_active_placement`:

```python
@property
def _active_placement(self) -> PlacementState:
    return self._placements[self._active_placement_id]

# Aliases used throughout do_capture / do_delete_sample / get_state_dict:
@property
def session(self): return self._active_placement.session
@property
def _thumbs(self): return self._active_placement.thumbs
# etc.
```

The head anchor state (`_anchor_obs`, `_tbb_head`, `_anchor_scatter`) moves
from flat node fields into `PlacementState`; `do_anchor_board` and
`do_clear_anchor` operate on the active placement only.

### 1.3  Placement management helpers (in node)

```python
def do_add_placement(self, label: str) -> dict:
    """Create and activate a new empty placement.  Returns {ok, id, label, count}."""

def do_activate_placement(self, pid: str) -> dict:
    """Switch active placement.  Does NOT clear anything.  Returns {ok, id}."""

def do_rename_placement(self, pid: str, new_label: str) -> dict:
    """Rename a placement label in-place.  Returns {ok}."""

def do_delete_placement(self, pid: str) -> dict:
    """Remove a placement (must have ≥ 2 placements; cannot delete the only one).
    If the deleted placement was active, switch to the first remaining one.
    Returns {ok}."""
```

Placement IDs are slugified labels (`re.sub(r'[^a-z0-9_-]', '_', label.lower())`)
deduped with a monotone counter (`_1`, `_2`, …) when a collision occurs.
They never leave the node as URL path segments exposed to the internet, so
the validation requirement is just "non-empty, no slash/null-byte".

---

## Phase 2 — Session persistence

### 2.1  Session JSON schema v2

Bump `"schema"` from `"wrist_handeye_session/1"` to `"wrist_handeye_session/2"`.

New top-level structure:

```json
{
  "schema": "wrist_handeye_session/2",
  "timestamp": "20260628_140000",
  "robot": "tinker2",
  "calib_frame": "color",
  "board": { "squares_x": 5, "squares_y": 5, "square_len_m": 0.04, "aruco_dict": "DICT_5X5_100" },
  "K": [[...], ...],
  "D": [...],
  "active_placement": "table_left",
  "placements": [
    {
      "id": "table_left",
      "label": "Table — left side",
      "anchor_have": false,
      "Tbb_head": null,
      "samples": [
        { "idx": 0, "T_base_eef": [...], "T_cam_board": [...], ... }
      ],
      "result": null
    },
    {
      "id": "floor_center",
      "label": "Floor — centre",
      "anchor_have": true,
      "Tbb_head": [[...], ...],
      "samples": [ ... ],
      "result": { "status": "PASS", "X_eef_cam": [...], ... }
    }
  ],
  "combined_result": null
}
```

`combined_result` holds the result of the multi-placement joint solve (Phase 4).
Each `placement.result` holds the per-placement independent solve for reference.

### 2.2  `_build_session_dict` update

`_build_session_dict` (in `HandeyeWebNode`) iterates `self._placements` and
serialises each into a placement dict.  The `result` block inside a placement
dict records its last independent solve result.

### 2.3  v1 → v2 migration in `do_load_session`

When `read_session` returns a dict with `"schema": "wrist_handeye_session/1"`
(flat `"samples"` at the top level, no `"placements"` key), the load path
wraps the samples in a synthetic placement:

```python
if "placements" not in data:
    data = {**data, "placements": [{"id": "default", "label": "default",
                                    "samples": data.get("samples", []),
                                    "anchor_have": data.get("anchor_have", False),
                                    "Tbb_head": data.get("Tbb_head"), "result": None}],
            "active_placement": "default"}
```

### 2.4  Thumbnails on disk

Current layout: `thumbs/<i>.jpg` (flat, one placement implied).

New layout: `thumbs/<placement_id>/<i>.jpg`.

`handeye_sessions.py` adds:
```python
def write_thumb(name, placement_id, idx, jpg_bytes, base=None): ...
def rewrite_placement_thumbs(name, placement_id, thumbs_by_idx, base=None): ...
def thumb_path(name, placement_id, idx, base=None): ...
```

`rewrite_thumbs` (old single-placement signature) becomes a deprecated shim
that calls `rewrite_placement_thumbs` with `placement_id="default"`.

Old sessions on disk (no `thumbs/<id>/` subdirs) are still readable: the
v1-load path maps placement `"default"` → `thumbs/<i>.jpg` fallback.

### 2.5  `list_sessions` summary update

The summary dict gains:
```python
{
  ...,
  "n_placements": 2,
  "n_samples_total": 34,   # sum across placements
  "placements": [{"id": "table_left", "n_samples": 18, "has_solve": True, "status": "PASS"},
                 {"id": "floor_center", "n_samples": 16, "has_solve": False}],
  "has_combined_solve": True,
  "combined_status": "PASS",
}
```

---

## Phase 3 — API layer

### 3.1  New placement endpoints

```
GET  /api/placements
     → list[{id, label, n_samples, has_solve, status, n_rejected, is_active}]

POST /api/placements/new
     body: {label: str}
     → {ok, id, label, count}  (creates + activates new placement)

POST /api/placements/{id}/activate
     → {ok, id}

PATCH /api/placements/{id}
     body: {label: str}
     → {ok, id, label}

DELETE /api/placements/{id}
     → {ok, deleted}  (error if only placement)
```

### 3.2  Existing solve endpoint — new `scope` parameter

```
POST /api/solve
     body: {method?, reject_sigma?, scope?: "active"|"all"}
```

`scope="active"` (default): current behaviour — solve the active placement only.

`scope="all"`: multi-placement solve (Phase 4) — solve each placement
independently for its seed Tbb, then run the joint bundle adjust over all
placements sharing X.

### 3.3  Thumb endpoint — placement-aware

```
GET /api/sessions/{name}/samples/{i}/thumb.jpg
```

Kept unchanged for single-placement sessions.  For multi-placement sessions the
endpoint needs to know which placement:

```
GET /api/sessions/{name}/placements/{pid}/samples/{i}/thumb.jpg
```

The old single-path form (`/samples/{i}/thumb.jpg`) continues to work and
resolves against the first / only placement, maintaining backward compatibility
for the existing history browser render loop.

### 3.4  State push — `placements` summary

`get_state_dict()` adds to the WebSocket payload:

```python
payload["placements"] = [
    {"id": pid, "label": p.label, "n_samples": len(p.session.samples),
     "is_active": pid == self._active_placement_id}
    for pid, p in self._placements.items()
]
payload["active_placement_id"] = self._active_placement_id
```

---

## Phase 4 — Multi-placement solver

### 4.1  `bundle_adjust_multi` in `handeye_solve.py`

```python
def bundle_adjust_multi(
    placements_samples: list[list[Sample]],
    K, dist, board_pts,
    X0: np.ndarray,           # initial T_eef_cam (from best per-placement solve)
    Tbb0s: list[np.ndarray],  # initial T_base_board per placement
    depth_weight=0.0, depth_sigma_m=0.005, loss="soft_l1",
    xtol=None, ftol=None, gtol=None, max_nfev=None,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Joint bundle-adjust: X shared, one Tbb per placement.

    Parameter vector: [X_6dof | Tbb_0_6dof | Tbb_1_6dof | ... | Tbb_{n-1}_6dof]
    Returns (X, Tbbs, info).
    """
```

The residual function for the joint problem:

```python
def _residuals_multi(p, placements_samples, K, dist, board_pts, dw, ds):
    n = len(placements_samples)
    X   = tf.T_from_vec(p[:6])
    Tbbs = [tf.T_from_vec(p[6 + 6*i : 12 + 6*i]) for i in range(n)]
    r = []
    for samples, Tbb in zip(placements_samples, Tbbs):
        r.append(_residuals(np.concatenate([p[:6], p[6+...6+6]]),
                            samples, K, dist, board_pts, dw, ds))
    return np.concatenate(r)
```

(Can share the existing `_residuals` inner loop by passing sub-slices of `p`.)

### 4.2  `@dataclass MultiPlacementSolveResult`

```python
@dataclass
class MultiPlacementSolveResult:
    X: np.ndarray               # shared T_eef_cam
    placement_Tbbs: list        # [T_base_board] per placement
    placement_results: list     # [SolveResult] per-placement independent solve
    combined_metrics: dict      # aggregate over all placements using best X
    status: str                 # gate on combined_metrics
    seed_placement_id: str      # which placement's X seeded the joint solve
```

### 4.3  `solve_multi_placement` entry point

```python
def solve_multi_placement(
    placements: list[tuple[str, list[Sample]]],  # [(id, samples), ...]
    K, dist, board_pts,
    *,
    methods=None,
    reject_sigma=2.5,
    max_reject_frac=0.25,
    depth_weight=1.0,
    depth_sigma_m=0.005,
    anchor_Tbbs: dict | None = None,   # {placement_id: T_base_board} from head
    progress_cb=None,
) -> MultiPlacementSolveResult:
```

Algorithm:

1. **Per-placement independent solve.**  For each `(pid, samples)`, call
   `solve(samples, K, dist, board_pts, anchor_Tbb=anchor_Tbbs.get(pid), ...)`.
   Collect `SolveResult` list and per-placement Tbb seeds.

2. **Seed selection.**  The placement whose independent solve has the lowest
   `trans_rmse_m` seeds the shared X.  Log which placement "won" as
   `seed_placement_id`.

3. **Joint bundle adjust.**  Call `bundle_adjust_multi(placements_samples, K,
   dist, board_pts, X0=best_X, Tbb0s=[each placement's Tbb], ...)`.  This
   refines X against all placements simultaneously.

4. **Combined metrics.**  Evaluate X against all samples (using per-placement
   Tbb from step 3).  Gate using the same `_PASS`/`_WARN` thresholds.

5. **Return** `MultiPlacementSolveResult`.

**Why not simply average X estimates?**
A geodesic mean of SE(3) matrices is correct but discards the cross-placement
constraint structure.  The joint bundle adjust keeps the full Jacobian and finds
the X that best explains *all* observations simultaneously — it strictly
dominates per-placement averaging in both accuracy and in the validity of the
reported residuals.

### 4.4  Minimum sample requirement per placement

Each placement must supply ≥ 6 samples (same floor as the single-placement
`solve()`).  `solve_multi_placement` rejects any placement below this floor
with a clear error before proceeding, listing the offending placement IDs.

---

## Phase 5 — Web UI

### 5.1  Capture tab — placement switcher

Add a "Placements" bar above the waypoints sub-panel.  Layout:

```
[ Table — left side ▾ ]   [ + New ]   [ 🗑 ]
```

- Dropdown lists all placements; clicking one calls `POST /api/placements/{id}/activate`.
- `[ + New ]` opens a modal (`<dialog>`) with a text input for the label and
  `[Create]` / `[Cancel]` buttons.  On Create: `POST /api/placements/new`.
- `[ 🗑 ]` (trash icon, disabled when only one placement exists) opens a confirm
  dialog and calls `DELETE /api/placements/{id}`.
- The gallery below the switcher shows only the **active** placement's samples.
  The WS `state.placements` list drives the dropdown; `state.active_placement_id`
  sets the selected option.

### 5.2  Solve tab — scope selector

Add a radio group above the Solve button:

```
◉ Active placement only   ○ All placements (combined joint solve)
```

Default: "Active placement only" (preserves existing behaviour).

When "All placements" is selected and `POST /api/solve` fires with `scope="all"`:

- Show a per-placement results table:

  | Placement | N kept | Trans RMSE | Rot RMSE | Reproj | Status |
  |---|---|---|---|---|---|
  | Table — left | 16 | 1.3 mm | 0.2° | 0.8 px | PASS |
  | Floor — centre | 14 | 2.1 mm | 0.3° | 1.1 px | PASS |
  | **Combined** | **30** | **1.1 mm** | **0.2°** | **0.9 px** | **PASS** |

- The existing X/rpy + Promote controls bind to the **combined** X.
- When the combined status is PASS, the Promote tab operates normally.

### 5.3  History tab — placement count column

Add `N placements` and `N samples total` columns (or a sub-row) to each
history entry row using `state.n_placements` / `state.n_samples_total`.

---

## Phase 6 — Tests

### 6.1  `test_sessions.py` additions

- `test_write_read_multi_placement`: write a session dict with 2 placements,
  read it back, assert both placements present.
- `test_v1_migration`: read a v1-schema dict (flat `samples`), assert it is
  wrapped as a single `"default"` placement.
- `test_thumb_path_multi_placement`: assert `thumb_path(name, "table_left", 0)`
  returns `…/thumbs/table_left/0.jpg`.
- `test_safe_name` coverage remains unchanged.

### 6.2  `test_solve_multi_placement.py` (new file)

Reuse `synthetic.py` ground-truth scenario generation:

```python
def test_solve_multi_placement_two_boards():
    """Two independent board positions; solver must recover same X from both."""
    from handeye_calib.synthetic import make_scenario
    X_gt = make_random_X()
    p1 = make_scenario(X_gt, Tbb=Tbb1, n=14)
    p2 = make_scenario(X_gt, Tbb=Tbb2, n=14)
    res = solve_multi_placement([("p1", p1.samples), ("p2", p2.samples)], ...)
    assert np.allclose(res.X, X_gt, atol=1e-3)
    assert res.status in ("PASS", "WARN")
```

- Test with 2 placements, 3 placements.
- Test that mixing samples *across* placements in a single `solve()` call gives
  *worse* X recovery than `solve_multi_placement` (demonstrates the motivation).
- Test that a placement with < 6 samples raises a clear error before solve.

### 6.3  `test_sessions_web.py` additions

- `test_add_placement_endpoint`: POST `/api/placements/new`, assert 201 with
  `{ok, id, label, count}`.
- `test_activate_placement_endpoint`: POST activate, assert state push reflects
  new active placement.
- `test_delete_placement_last_fails`: DELETE the only placement, assert
  `{ok: False}`.
- `test_solve_scope_all`: mock two placements each with 8 synthetic samples,
  POST `/api/solve` with `scope="all"`, assert response contains
  `combined_result` with per-placement breakdown.

### 6.4  `test_rigid_and_progress.py` / existing suites

No regressions expected: all existing tests operate with a single-placement
session and the refactored node delegates the active placement via property
aliases.  Verify by running the existing suite against the refactored node
state after Phase 1.

---

## Implementation order and parallelism

Phases can be implemented in **three parallel streams** and merged/shipped in
one go.  File-level analysis shows zero overlap between streams A and B, and
between either of those and stream C.  Expected merge conflicts at the merge
gate: **none**.

```
Stream A ──── Phase 4 (handeye_solve.py + test_solve_multi_placement.py) ────────┐
Stream B ──── Phase 2/sessions.py (handeye_sessions.py + test_sessions.py) ──────┤ merge → ship
Stream C ──── Phase 1 → Phase 2/web → Phase 3 (handeye_web.py, sequential)       │
               └─ Phase 5 UI (app.js, index.html, drafted alongside Phase 3) ────┘
```

**Stream A** — `handeye_solve.py` only.  Adds `bundle_adjust_multi`,
`_residuals_multi`, `MultiPlacementSolveResult`, `solve_multi_placement`.
Imports only `transforms.py` and numpy/scipy; no file overlap with B or C.
Tests live in a new `test_solve_multi_placement.py`.

**Stream B** — `handeye_sessions.py` only.  Adds placement-aware thumb paths,
v1→v2 read shim, updated `list_sessions` summary.  No imports from any other
handeye module.  No file overlap with A or C.  Tests are additions to the
existing `test_sessions.py`.

**Stream C** — the only stream with an internal sequential constraint:
`placement_state.py` (new) → `handeye_web.py` Phase 1 refactor → Phase 2
web.py parts (`_build_session_dict`, `do_load_session`) → Phase 3 new
endpoints → Phase 5 UI.  The UI can be drafted against the documented API
contract while Phase 3 is still in progress and finalised last.

**Critical path** is Stream C (Phases 1→2→3 sequential in one file).
Streams A and B complete independently and wait at the merge gate.

---

## What does NOT change

- The `solve()` single-placement function signature and behaviour — it remains
  the building block called by `solve_multi_placement`.
- The Promote / apply_handeye workflow — it still operates on a single `X`
  (the combined one when multi-placement was used, the per-placement one
  otherwise).
- Waypoints — they remain per-robot, shared across placements.  The same
  waypoint set can be used for multiple placements (the arm visits the same
  poses; only the board has moved).
- The `calib_frame` setting (`color` / `ir`) — it is a node-level setting
  shared across all placements.  Mixing frames within a session remains
  disallowed (clearing still detaches from the on-disk session and resets
  all placements).
