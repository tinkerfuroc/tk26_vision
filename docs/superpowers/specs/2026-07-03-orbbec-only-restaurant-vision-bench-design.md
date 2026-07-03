# Orbbec-Only Restaurant Vision Bench (Tiers 0–2) — Design Spec

- **Date:** 2026-07-03
- **Status:** Approved design, pending implementation
- **Hardware:** one Orbbec Femto Bolt on a tripod. No pan-tilt, no base, no arm, no mic.
- **Author:** Claude (with cindy)
- **Related:** `2026-07-03-restaurant-resolution-and-waving-vlm-concurrency-design.md`
  (this bench is the pre/post regression harness for that spec's implementation).

## 1. Context & Goal

The Restaurant BT's vision surface is exactly two live endpoints:
`detect_waving_persons` (srv `DetectWaving`, served by
`tk_vision_specialized/waving_person_server.py`) consumed by
`BtNode_ScanForWavingPerson`, and `follow_head_action` consumed by
`BtNode_MaintainEyeContact`. The second is intrinsically pan-tilt-bound (the
server aims via live servo state) and is **out of scope** here. Everything
else on the customer-detection path — YOLO11-seg + MediaPipe waving
classification, depth→3D centroids, closest-first ordering, the BT sweep /
dedup / blackboard contract — is testable with a bare Orbbec.

Goal: a repeatable three-tier ladder that validates that path standalone,
with written pass criteria, so detection regressions are caught on a desk
instead of at a competition, and so the pending resolution/VLM-concurrency
changes can be gated before/after.

### Non-goals

- `follow_head` / eye contact (needs the servo; covered later by a pan-tilt
  loopback stub — explicitly deferred).
- `BtNode_DetectTray` / `object_detection_generalist` (separate server;
  can be bolted on later as a Tier-1 scenario, not designed here).
- Navigation/approach behavior downstream of detection.

## 2. Bench rig

- Femto Bolt on tripod, **height matched to the robot's head-camera height**
  (measure on the robot; record in the session log), pitched level or the
  measured down-tilt. USB to the vision workstation.
- 8 m of clear line of sight; floor tape at **1 / 2 / 4 / 6 / 8 m** from the
  camera plane (8 m = `DETECT_WAVING_THRESHOLD_M` in the production tree).
- Pre-warm before any timed run: first YOLO/MediaPipe call silently
  downloads/loads models (~30 s). Run one throwaway detection first.
- If runtime behavior contradicts source, suspect a stale `install/` and
  rebuild — this workspace has been bitten before.

## 3. Tier 0 — offline, no hardware

**Proves:** the gesture predicate + centroid/VLM geometry math, against the
labelled corpus. Cheap enough to run before every change to
`waving_person_server.py`.

Existing pieces, no new code:

```bash
source src/tk26_vision/.venv-vision-main/bin/activate
# unit tests (geometry + VLM plumbing)
pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_geometry.py \
       src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py
# labelled-image regression over /home/tinker/tk25_ws/detect_waving_test/{waving,not_waving}/
python3 src/tk26_vision/scripts/tests/debug_waving_pipeline.py \
    --out-dir /tmp/wave_audit_$(date +%Y%m%d)
```

**Pass gates:**
- pytest green.
- `debug_waving_pipeline.py` accuracy ≥ the last recorded run (compare
  `results.csv`; keep the previous run's CSV). Any new FP/FN must be
  eyeballed via the emitted overlays and either accepted with a note or fixed.

**Corpus growth (process rule):** every Tier-1 session that produces an
interesting frame (miss, false positive, marginal distance) gets that frame
copied from the `vision_log/<session>/` dumps into
`detect_waving_test/{waving,not_waving}/`. The corpus is the long-term asset.

## 4. Tier 1 — live camera + waving server, no BT

**Proves:** the full service end-to-end on real sensor data — sync, depth
registration, 3D centroid accuracy, closest-first ordering, threshold gating,
debug overlay, VLM fallback arming.

### 4.1 Bringup (all existing)

```bash
# camera only — keeps the mandatory FastDDS SHM profile, drops pan-tilt + FFS
ros2 launch vision_bringup vision_driver.launch.py enable_pan_tilt:=false enable_ffs:=false
# sanity: ~30 Hz expected
ros2 topic hz /camera/color/image_raw
# waving server + rqt on /detect_waving_debug_image
ros2 launch tk_vision_specialized detect_waving.launch.py
```

For regression runs against the pending resolution spec, repeat the suite at
both resolutions: default 1280×720 and `color_width:=1920 color_height:=1080`.

### 4.2 TF shim (only for `target_frame != ""`)

`enable_pan_tilt:=false` also drops `robot_state_publisher`, so no TF above
`camera_link` exists, and the server **hard-fails** a request whose
`target_frame` can't be resolved (`_snapshot_latest_transform` → None →
error response). Camera-frame scenarios therefore use `target_frame: ""`.
For `base_link`/`map` scenarios, two static publishers stand in for the
robot (the Orbbec driver itself publishes
`camera_link→camera_color_optical_frame`; `publish_tf` defaults true):

```bash
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 \
    --frame-id map --child-frame-id base_link
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z <TRIPOD_H_M> \
    --pitch <DOWN_TILT_RAD> --frame-id base_link --child-frame-id camera_link
```

`TRIPOD_H_M` and `DOWN_TILT_RAD` (positive = down) are measured per session
and recorded in the session log.

### 4.3 New deliverable: `waving_bench` scenario client

The existing `waving_client` hardcodes threshold 5.0 m / camera frame / 1 Hz
and has no pass/fail notion. Per the established test-client convention
(YAML multi-scenario suite; rclpy-free logic + pure pytest — same shape as
the earlier xArm suite), add to `tk_vision_specialized`:

| File | Role |
|---|---|
| `tk_vision_specialized/waving_bench.py` | console script `waving_bench`. Loads the YAML, runs one named scenario (`--scenario range_ladder`) or `--all`. For each sub-case: prints the operator prompt, waits for Enter, fires N service calls, evaluates, prints a ✓/✗ table, appends JSONL to `--out-dir`. |
| `config/waving_bench.yaml` | defaults + named scenarios (installed via `data_files`; overridable with `--config`). |
| `tk_vision_specialized/_waving_bench_eval.py` | **rclpy-free**: scenario schema, per-call evaluation (expected waver count, expected-distance window, ordering monotonicity, expected status). Operates on plain dicts extracted from the response. |
| `test/test_waving_bench_eval.py` | pure pytest over the eval helper (no ROS). |

YAML sketch:

```yaml
defaults:
  threshold_meters: 8.0        # match production (config.py DETECT_WAVING_THRESHOLD_M)
  target_frame: ""             # camera optical frame unless overridden
  min_waving_persons: 0        # VLM off unless a scenario opts in
  calls_per_case: 5
  interval_sec: 1.0
  pass_ratio: 0.8              # >=4/5 calls must satisfy the case expectation

scenarios:
  smoke:
    - prompt: "one person, wave at 2 m"
      expect: {count: 1, z_range_m: [1.7, 2.3]}
  range_ladder:                # one sub-case per tape mark
    - {prompt: "wave at 1 m", expect: {count: 1, z_range_m: [0.7, 1.3]}}
    - {prompt: "wave at 2 m", expect: {count: 1, z_range_m: [1.7, 2.3]}}
    - {prompt: "wave at 4 m", expect: {count: 1, z_range_m: [3.7, 4.3]}}
    - {prompt: "wave at 6 m", expect: {count: 1, z_range_m: [5.6, 6.4]}}
    - {prompt: "wave at 8 m", expect: {count: 1, z_range_m: [7.5, 8.5]}, best_effort: true}
  gesture_matrix:
    - {prompt: "static raised hand (above head)", expect: {count: 1}}
    - {prompt: "hand raised above elbow, elbow at shoulder", expect: {count: 1}}
    - {prompt: "arms down, walk around", expect: {status: 1, count: 0}}   # negative
    - {prompt: "point sideways at shoulder height", expect: {status: 1, count: 0}}  # known-hard negative
  two_person_arbitration:
    - prompt: "A waves at 2 m, B waves at 4 m"
      expect: {count: 2, ordering: closest_first}
    - prompt: "A idle at 2 m, B waves at 4 m"
      expect: {count: 1, z_range_m: [3.7, 4.3]}
  threshold_gate:
    - prompt: "wave at 4 m"
      overrides: {threshold_meters: 3.0}
      expect: {status: 1, count: 0}          # beyond threshold → dropped
  frames:                                    # requires the §4.2 TF shim
    - prompt: "wave at 2 m"
      overrides: {target_frame: base_link}
      expect: {count: 1, frame_id: base_link}
  vlm_fallback:                              # requires DASHSCOPE/OPENROUTER key + network
    - prompt: "one person waves clearly, second sits waving small/far"
      overrides: {min_waving_persons: 2}
      expect: {count: 2}
```

Evaluation semantics (in `_waving_bench_eval.py`): a call passes its case iff
status matches, `len(waving_persons)` matches `count`, every reported point's
distance (camera frame: `point.z`; base_link/map: `hypot(x,y)`) falls inside
`z_range_m` when given, and `ordering: closest_first` holds. A **case**
passes iff ≥ `pass_ratio` of its calls pass; `best_effort: true` cases report
but never fail the suite.

**VLM note (forward-compat):** today the fallback arms via
`min_waving_persons > 0`. The approved concurrency spec replaces that trigger
with server params (`enable_vlm_fallback` + `vlm_skip_min_wavers`). The
`vlm_fallback` scenario stays valid across the migration; the eval helper must
not encode the trigger mechanism, only the expected outcome. For all *other*
scenarios, run the server with `-p enable_vlm_fallback:=false` (or no keys in
env) so CV-path timing is measured clean.

**Tier-1 pass gates:** `smoke`, `range_ladder` (≤6 m cases), `gesture_matrix`,
`two_person_arbitration`, `threshold_gate`, and `frames` all pass; 8 m case
recorded. Distance tolerance ±0.3 m at ≤4 m, ±0.5 m beyond.

## 5. Tier 2 — BT node level (`test_restaurant_scan`), Orbbec only

**Proves:** the decision-layer contract on top of a live service: real
`tinker_vision_msgs_26` messages (no silent-mock), blackboard writes
(`all_persons` / `closest_person` / crop paths), the FailureIsSuccess sweep
advancement, and `BtNode_PackWavingCustomers`' 0.3 m dedup gate.

### 5.1 Components

- Tier-1 stack running (camera + waving server).
- **TF shim from §4.2 is required** — `test_scan.scanAllPositions()`
  hardcodes `target_frame="base_link"`. (Only the `base_link→camera_link`
  publisher is strictly needed; `map` is unused at this tier.)
- **New file:** `src/tk25_decision/src/behavior_tree/config/vision_live_bench.json`
  — derived from `full_mock.json` (global mock ON, `auto_detect: false`,
  keyboard disabled so KEYPRESS auto-advances) with exactly one change in
  `mock_mode.subsystems.vision.nodes`:

  ```json
  "BtNode_ScanForWavingPerson": "NO_MOCK"
  ```

  Everything else stays mocked: `BtNode_TurnPanTilt` (KEYPRESS → instant
  SUCCESS, no 2 s no-feedback timeouts, no serial device) and
  `BtNode_Announce` (KEYPRESS → log line instead of TTS). `NO_MOCK` is the
  documented per-node force-real escape hatch under global mock
  (`config.py:is_node_mocked`). Header comment in the JSON must state it is
  derived from `full_mock.json` and list the delta.

- **One-line patch to `Restaurant/test_scan.py`** (bench script, not
  production): pass `bb_key_pictures="test_scan_waving_pictures"` into its
  `BtNode_ScanForWavingPerson` so the referee-crop path (`_crop_and_save` →
  `/tmp/restaurant_customer_*.png`) is exercised — today the bench skips it
  while the production tree relies on it.

### 5.2 Run

```bash
# no rebuild needed for config iteration — BT_MOCK_CONFIG takes a source path
BT_MOCK_CONFIG=/home/tinker/tk25_ws/src/tk25_decision/src/behavior_tree/config/vision_live_bench.json \
  ros2 run behavior_tree test_restaurant_scan
```

(The `test_scan.py` patch itself does need
`./tkbuild tk25_decision --packages-select behavior_tree` so the root
install the entry point runs from picks it up.)

Expected shape per sweep: 8 positions × (instant mocked TurnPanTilt +
2.5 s settle timer + live scan + mocked announce + pack) ≈ 25–35 s
CV-only. With a static camera, all 8 positions image the same scene — that is
the dedup test, not a defect.

### 5.3 Pass criteria

1. `tree.setup(timeout=15)` completes — proves service discovery **and**
   that real message types loaded (a silent-mock fallback would be visible as
   `ServiceHandler` force-mocking; workspace must be sourced).
2. One person waving → per-position scans report it; after a full sweep
   `customer_centroids` contains **exactly one** entry (0.3 m XY dedup
   collapsed 8 sightings), in `base_link`, at the taped distance ±0.5 m
   (through the static TF: `x ≈ distance`, `z ≈` chest height).
3. Two people ≥1 m apart, both waving → exactly two entries;
   `test_scan_waving_closest_person` is the nearer one.
4. Crop files exist and contain the correct person (spot-check the PNGs
   against the rqt debug overlay).
5. Positions where nobody waves log the scan FAILURE but the sweep advances
   (FailureIsSuccess) — the tree never wedges on an empty position.
6. Known limitation to record, not fix: a person who moves >0.3 m between
   positions is double-counted (the gate is XY distance only).

Timing note: leave the VLM fallback disabled for the standard Tier-2 run
(each `min_waving_persons=2` scan with <2 CV wavers otherwise blocks up to
`vlm_timeout_s`=20 s per position); a separate keyed run of the same command
is the VLM-in-the-tree check.

## 6. New-artifact summary (implementation checklist)

| # | Artifact | Where | Size |
|---|---|---|---|
| 1 | `waving_bench.py` console script | `tk_vision_specialized` | ~200 lines |
| 2 | `waving_bench.yaml` scenario suite | `tk_vision_specialized/config/` | ~60 lines |
| 3 | `_waving_bench_eval.py` (rclpy-free) | `tk_vision_specialized` | ~100 lines |
| 4 | `test_waving_bench_eval.py` | `tk_vision_specialized/test/` | ~80 lines |
| 5 | `vision_live_bench.json` | `behavior_tree/config/` | copy + 1-key delta |
| 6 | `bb_key_pictures` patch | `Restaurant/test_scan.py` | 1 line |
| 7 | Session-log template (rig height/pitch, resolution, scenario results) | `src/tk26_vision/scripts/tests/logs/` | markdown stub |

Tier 0 ships with zero new code. Builds: #1–4 via
`./src/tk26_vision/scripts/build.sh --packages-select tk_vision_specialized`
(+ setup.py entry point and data_files additions), #5 used from source path,
#6 via `./tkbuild tk25_decision --packages-select behavior_tree`.

## 7. Risks & gotchas (carry into the runbook)

- **FastDDS SHM profile**: never launch the vendored Orbbec launch bare;
  always go through `vision_driver.launch.py` (sets
  `FASTRTPS_DEFAULT_PROFILES_FILE`), else ~3 Hz and the 0.1 s sync window
  starts dropping pairs.
- **TF absence is a hard service failure**, not a fallback to camera frame —
  scenario frames must match the shim actually running.
- **Silent-mock trap**: run Tier 2 from a shell that sourced
  `install/setup.zsh`; if `tinker_vision_msgs_26` imports from
  `mock_messages`, the scan node force-mocks and the bench "passes" without
  touching the camera. The setup-completes gate plus a visible rqt detection
  is the guard.
- **First-call model download** (~30 s) — pre-warm.
- **VLM keys/network**: DashScope key is CN-region; Clash routing for
  huggingface/pytorch downloads has a known dead-group failure mode. Keep the
  standard suite key-free; isolate network dependence to the `vlm_fallback`
  scenario.
- `test_scan.py` sweeps pans including ±180/±120° — meaningless but harmless
  with a fixed camera; do not "fix" the angle list for the bench, it keeps
  the script identical to the pan-tilt-equipped use.
