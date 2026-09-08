# Person-Tracker Benchmark Harness — Implementation Plan

Scaffolding to validate the `vision_track` person tracker (`/track_person`,
`PersonTrackNode` → `YOLOTracker` ResNet50+color ReID) against its real
deployment domain (RoboCup@Home arena, Orbbec RGB-D, single-target follow).

**Production framing.** The benchmark we improve against is **Tier A**: a
self-recorded Tinker arena follow-regression suite (own Orbbec rosbags, scored
on the node's real output contract). Real recordings are weeks out, so this plan
builds the *tooling* now and proves it on synthetic fixtures. **Tier B** is
TPT-Bench, the one external smoke-test. Decision rationale lives in the session
memory `person-tracker-benchmark-strategy`.

Everything here is built so the **pure logic is unit-testable today** with
synthetic fixtures (fabricated GT JSON, numpy frames, fabricated prediction
streams, synthetic rosbags written via `rosbag2_py`). The thin ROS/model
integration shells are exercised later against real bags / TPT-Bench data.

---

## Repo layout (git root = `src/tk26_vision`)

```
benchmarks/person_tracker/
├── PLAN.md                      # this file
├── README.md                    # contract + usage  (Task 0 stubs, Task D finalizes)
├── pyproject.toml               # pytest config (pythonpath=["."]) + optional editable install
├── ptbench/
│   ├── __init__.py
│   ├── common/                  # ── Task 0 (the shared contract) ──
│   │   ├── __init__.py
│   │   ├── schema.py            # GT annotation dataclasses + JSON load/save/validate
│   │   ├── geometry.py          # bbox+depth → 3D centroid (mirrors node math)
│   │   ├── align.py             # align a prediction stream to GT frames by timestamp
│   │   ├── metrics.py           # the scoreboard metrics (pure functions)
│   │   └── scoreboard.py        # PASS/WARN/FAIL gates + table + JSON dump
│   ├── replay/                  # ── Task A ──
│   │   ├── __init__.py
│   │   ├── bag_io.py            # rosbag2 reader: synced (color, depth, info) iterator
│   │   ├── runner.py            # drive /track_person (action) OR offline tracker core
│   │   └── score_cli.py         # CLI: bag + GT.json → scoreboard
│   ├── labeler/                 # ── Task B ──
│   │   ├── __init__.py
│   │   └── label_cli.py         # cv2 UI: step bag frames → operator box + present → GT.json
│   └── tpt_bench/               # ── Task C ──
│       ├── __init__.py
│       ├── dataset.py           # TPT-Bench annotation layout loader
│       ├── metrics.py           # TPT metrics: Precision / Recall / F-score / AMR / AO
│       ├── runner.py            # feed TPT frames through tracker core (force-init frame 1)
│       └── score_cli.py         # CLI: TPT seq dir → TPT scoreboard
└── tests/
    ├── conftest.py
    ├── test_schema.py  test_geometry.py  test_align.py
    ├── test_metrics.py  test_scoreboard.py        # Task 0
    ├── test_bag_io.py                              # Task A (writes a synthetic bag)
    ├── test_tpt_dataset.py  test_tpt_metrics.py    # Task C
    └── ...
```

Run tests: `cd benchmarks/person_tracker && <venv>/bin/python -m pytest -q`
(`pyproject.toml` sets `pythonpath=["."]`, so `import ptbench...` works without install).
venv: `/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main`.

---

## THE CONTRACT (authoritative — every task references this)

### GT annotation schema (one JSON per clip) — `ptbench/common/schema.py`

```json
{
  "schema_version": "1.0",
  "clip_id": "cml_crossing_01",
  "bag_path": "bags/cml_crossing_01",
  "scenario": "cml_crossing",
  "color_topic": "/camera/color/image_raw",
  "depth_topic": "/camera/depth/image_raw",
  "camera_info_topic": "/camera/color/camera_info",
  "fps_hint": 30.0,
  "notes": "referees cross twice",
  "frames": [
    {
      "t_ns": 1234567890123,        // color-frame stamp (ns); the alignment key
      "present": true,              // operator visible/in-frame this frame
      "bbox": [x1, y1, x2, y2],     // operator box in color px, or null if absent
      "centroid_3d": [x, y, z]      // GT 3D in CAMERA optical frame (m), or null
    }
  ]
}
```

> Schema note (v1.1): the on-disk schema is now v1.1 with **dual centroids** —
> `centroid_field` (gated, mask + robust median; the gate scores this) and
> `centroid_track` (node-identical diagnostic). The single `centroid_3d` shown
> above remains accepted only as a legacy-1.0 *input* key (it maps onto both
> new fields on load); new GT is written with the two fields.

- Camera optical-frame convention (matches `PersonTrackNode._depth_image_to_points`):
  `x = (u-cx)·z/fx` (right), `y = (v-cy)·z/fy` (down), `z = depth` (forward).
  So **range = z**, **lateral = sqrt(x²+y²)**.
- `schema.py` provides `@dataclass GtFrame`, `@dataclass GtClip`, and
  `load_gt(path) -> GtClip`, `save_gt(clip, path)`, with validation that raises
  `GtSchemaError` on: bad version, non-monotonic `t_ns`, `present=true` with
  null bbox, bbox not 4 numbers / x2<=x1, centroid not 3 numbers.

### Prediction record (what the runner captures per node feedback) — `ptbench/common/align.py`

```python
@dataclass
class PredFrame:
    t_ns: int                 # PointStamped/feedback header stamp (ns)
    target_lost: bool
    target_track_id: int
    point_xyz: tuple|None     # camera-frame (x,y,z) m, or None when lost/no-transform
```

The node emits `TrackPerson.Feedback` every loop tick: `target_lost`,
`target_track_id`, `target_position` (PointStamped), `is_transformation_successful`.
**Score from feedback** (every frame), not from the published topic. Run the node
with `target_frame='none'` so `target_position` stays in the camera frame and
matches GT `centroid_3d`.

### Alignment — `align.py`
`align_pred_to_gt(preds, gt_frames, tol_ms=50) -> list[(GtFrame, PredFrame|None)]`:
for each GT frame, nearest pred by `|t_ns|` within `tol_ms`, else `None`
(treated as "no output that frame"). Pure function; fully unit-testable.

### Metrics — `ptbench/common/metrics.py`

Pure functions over the aligned `[(GtFrame, PredFrame|None)]` list + a
`MetricConfig`. A pred is a **correct lock** iff `pred` exists, `not target_lost`,
`point_xyz` is not None, GT `present`, and `dist3d(pred.point, gt.centroid_3d) <=
correct_radius_m`. A **wrong lock** iff `pred` exists, `not target_lost`,
`point_xyz` not None, GT `present`, and `dist3d > wrong_radius_m`.

| metric | definition |
|---|---|
| `correct_lock_rate` | correct-lock frames ÷ present frames |
| `wrong_lock_episodes` | # maximal wrong-lock runs lasting > `sustained_s` |
| `reacquire_latency_s` | per present-segment that starts after an absent gap or a lost run: seconds from segment start to first correct lock; report `{median, max, samples}` |
| `pos_error_lateral_m` / `pos_error_range_m` | over correct-lock frames: `{median, p95}` of lateral & range error |
| `false_target_rate` | (frames with `not present` but pred exists & `not target_lost`) ÷ absent frames |
| `throughput_hz` | from runner per-frame wall-clock (median); pass-through value, metrics just formats it |

`MetricConfig` defaults: `correct_radius_m=0.50`, `wrong_radius_m=0.75`,
`align_tol_ms=50`, `sustained_s=0.5`. Time from `t_ns` deltas (don't assume fps).
Empty/degenerate inputs return well-defined zeros/NaN, never raise.

### Gates — `ptbench/common/scoreboard.py`

`GateConfig` (defaults below); `score(metrics, gates) -> Scoreboard` with overall
verdict = worst per-metric verdict. Pretty ASCII table + `to_dict()` JSON.

| metric | PASS | WARN | else FAIL |
|---|---|---|---|
| wrong_lock_episodes | == 0 | — | ≥ 1 |
| correct_lock_rate | ≥ 0.92 | ≥ 0.85 | < 0.85 |
| reacquire_latency_s (median) | ≤ 1.0 | ≤ 2.0 | > 2.0 |
| pos_error_lateral_m (median) | ≤ 0.25 | ≤ 0.40 | > 0.40 |
| false_target_rate | ≤ 0.05 | ≤ 0.10 | > 0.10 |
| throughput_hz | ≥ 12 | ≥ 8 | < 8 |

---

## Tasks

### Task 0 — `common/` package + project skeleton  [SEQUENTIAL, FIRST]
Create `pyproject.toml`, `ptbench/__init__.py`, `ptbench/common/{schema,geometry,align,metrics,scoreboard}.py`,
a stub `README.md`, and full unit tests (`test_schema, test_geometry, test_align,
test_metrics, test_scoreboard`) with synthetic fixtures. TDD. This is the contract
that Tasks A/B/C import — it must land and pass review before they start.
`geometry.py`: `centroid_from_bbox_depth(depth_mm, K, bbox, mask=None, min_depth=0.1, max_depth=10.0)`
replicating `PersonTrackNode._depth_image_to_points` + `_calculate_centroid` (mean x/y, **median z**).

### Task A — `replay/`  [PARALLEL]
- `bag_io.py`: open a rosbag2 dir, yield time-synced `(color_bgr ndarray, depth_mm ndarray, CameraInfo, t_ns)` tuples (nearest depth within slop; color stamp is `t_ns`). Decode via `cv_bridge` or raw buffer.
- `runner.py`: two backends → list[`PredFrame`] + per-frame timing:
  (1) **action** — `rclpy` action client to `/track_person` (`target_frame='none'`), publish recorded frames onto the camera topics OR (simpler) document that the live server consumes the bag played back; capture feedback.
  (2) **offline** — drive `vision_track.track_yolo.YOLOTracker` directly on the color frames + `common.geometry` for 3D; deterministic, no ROS.
- `score_cli.py`: `python -m ptbench.replay.score_cli --bag DIR --gt GT.json [--backend offline|action] [--gates gates.yaml]` → align → metrics → scoreboard (stdout table + optional `--json out.json`).
- Tests: `test_bag_io.py` writes a tiny synthetic bag with `rosbag2_py` and reads it back. Keep model/server backends out of unit tests (no recordings/model in CI); unit-test only pure glue.

### Task B — `labeler/`  [PARALLEL]
- `label_cli.py`: `python -m ptbench.labeler.label_cli --bag DIR [--out GT.json] [--scenario NAME]`. Steps color frames (reuse `replay.bag_io` if present, else its own minimal reader — but DO NOT import unfinished replay internals; read the bag directly), cv2 window: draw/adjust operator bbox, keys for next/prev/toggle-present/absent/save/quit, copies box forward as default. On save, samples depth→3D via `common.geometry` and writes a `common.schema` GtClip. Headless `--auto-detect` assist optional (skip if it needs the model). Tests: pure helpers (box propagation, frame→GtFrame assembly) with synthetic; the cv2 UI loop is thin and excluded from unit tests.

### Task C — `tpt_bench/`  [PARALLEL]
- `dataset.py`: load a TPT-Bench sequence (LaSOT-style per-frame bbox + `absent`/visibility flags) into `[(frame_path, gt_bbox|None)]`. (Verify the exact TPT-Bench annotation format from https://medlartea.github.io/tpt-bench/ / arXiv 2505.07446; if unreachable, implement against the documented LaSOT-style `groundtruth.txt` + `absence.txt`/`out_of_view.txt` and note the assumption.)
- `metrics.py`: TPT metrics — Tracking Precision, Recall, **F-score**, **AMR** (avg max recall at 100% precision), **AO** (avg overlap on visible frames). Pure; unit-test with synthetic box/absent sequences.
- `runner.py`: force-init `YOLOTracker` on frame-1 GT box (`initialize_tracking(frame, target_bbox=...)`), loop `update` → per-frame pred bbox|None → `metrics`. Thin; not unit-tested (needs model).
- `score_cli.py`: `python -m ptbench.tpt_bench.score_cli --seq DIR [--json out.json]` → scoreboard. Plus a `DOWNLOAD.md` documenting how to fetch TPT-Bench (do NOT auto-download GBs).

### Task D — integration  [SEQUENTIAL, LAST]
- Add `follow_regression` subcommand to `scripts/tests/t4_hardware.sh` (reuse `lib.sh`): replay each bag under a configurable dir through `replay.score_cli`, print scoreboards, aggregate PASS/WARN/FAIL into the `summary`. SKIP cleanly when the bag dir is empty (the expected state until recordings exist).
- Finalize `README.md`: the contract, the scenario taxonomy (cml_crossing, occlusion_reentry, lookalike_distractors, back_to_camera, range_lighting), the labeling workflow, the scoring workflow, TPT-Bench workflow, and the gate table.

---

## Execution
Subagent-driven: Task 0 solo (impl → spec review → quality review), then A/B/C
**in parallel** (disjoint dirs; implementers create files only under their own
subdir + `tests/`, and do NOT run `git`/commit — the controller commits), then
per-component spec+quality review, then Task D, then final review.
