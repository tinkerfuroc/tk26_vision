# handeye_calib

Eye-in-hand calibration for the wrist-mounted RealSense on the xArm flange (`link_eef`).
Solves `T_eef→color_optical`, verifies it, and writes it to `tinker_robot_config`.

Design: `../../docs/specs/2026-06-15-xarm-handeye-calibration-design.md`
Plan: `../../docs/plans/2026-06-15-xarm-handeye-calibration.md`

## How it works

The camera is rigidly bolted to `link_eef`, so there is one fixed unknown,
`T_eef→color_optical`. With a ChArUco board fixed in the world, the arm visits a
set of diverse poses; at each, the flange pose (`link_base→link_eef`, from TF) and
the board pose (ChArUco PnP) are recorded. `cv2.calibrateHandEye` gives a linear
seed (no prior needed — robust to a stale/moved mount), then a nonlinear
bundle-adjust jointly refines the transform and the board-in-base pose by minimizing
ChArUco corner reprojection. Held-out poses gate the result.

Module map:
- `transforms.py` — SE(3) helpers.
- `handeye_model.py` — ChArUco board geometry + pinhole reprojection.
- `synthetic.py` — synthetic ground-truth scenarios (+ `handeye_synthetic_check` CLI).
- `handeye_solve.py` — multi-method seed → bundle-adjust → held-out evaluate + gate.
- `gates.py` — settle/stability, pose-diversity, per-frame quality gates.
- `handeye_collect.py` — ROS node: drive arm, settle-gate, detect, accumulate.
- `handeye_web.py` — calib_web-style author / run / verify / promote tool.
- `apply_handeye.py` — compose to the URDF mount frame; write `hand_eye.yaml` / patch URDF.

## Board setup

Print the 5×5 / 40 mm ChArUco board, mount it **rigidly** (aluminium composite, not
foam) on a stand at ~table height inside the arm's reach, tilted ~30° toward the
camera. It must stay **fixed for the whole session** — if it shifts, restart.
Caliper-measure the square edge and confirm it matches the configured `square_len`.

## Calibrate (operator)

```bash
export ROBOT_NAME=tinker2
ros2 run handeye_calib handeye_web --ros-args -p bind:=127.0.0.1 -p port:=8766
```
Open `http://127.0.0.1:8766`, then: author/preview a pose set (validated against the
safety envelope), run collection + solve, read the PASS/WARN/FAIL banner and the live
predicted-corner overlay, and promote the result (diff-preview before any write).

Note: the camera's flexible support rings for ~1–2 s after each arm move, so capture
is gated by a settle delay **plus** a wait-until-stable check (consecutive frames must
agree within tolerance) — a pose that never settles is rejected, not captured mid-ring.

## Launch

```bash
ros2 launch handeye_calib handeye_web.launch.py port:=8766 robot_name:=tinker2
```

Brings up the `handeye_web` node (FastAPI UI + rclpy) with every ROS param exposed
as a launch arg (`bind`, `port`, `robot_name`, camera topics, `base_frame`/`eef_frame`,
`aruco_dict`, board geometry, `jointmove_action`, `mount_to_color_*`, `min_diversity_deg`).
The RealSense camera (`realsense2_camera`, `camera_name:=xarm_camera`) must be launched
**separately** — the UI shows `no camera` until color frames arrive. `ros2 launch`
teardown (SIGINT/SIGTERM) shuts the server down cleanly.

## Verify without hardware

The whole solver is provable on a laptop against synthetic ground truth:
```bash
python -m handeye_calib.synthetic     # prints recovered-X error + PASS
pytest test/                          # unit suite (source the workspace first for test_import)
```

## UI

The `handeye_web` tool is a single-page calibration UI with five tabs:

- **Info** — camera / TF / robot / ChArUco board / safety envelope status, plus the `T_base_eef` matrix.
- **Move** — joint editor (rad/deg toggle), Load-current / Zero / presets, with a live SafetyEnvelope verdict on the current EE pose before sending.
- **Capture** — stability-gated capture (three steady frames at < 0.0003 m / 0.1° drift), sample gallery with per-capture thumbnails + per-sample delete, diversity meter (max pairwise rotation° / 30° target).
- **Solve** — method picker (auto / TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS), per-method reprojection comparison table, residual histogram + scatter, sample-coverage canvas, PASS / WARN / FAIL gate pill with mm / deg / px units.
- **Promote** — side-by-side unified-diff preview for **both** `hand_eye.yaml` AND a per-robot `wrist_camera.xacro` override, ROBOT_NAME-scoped, confirm-before-apply, backup paths surfaced for each write. Reload-from-disk clears the cached solve so the operator can re-run.

Live state pushed via WebSocket at 5 Hz (no polling for state). The live camera feed polls `/api/frame.jpg` at ~3 Hz with an annotated / raw toggle and a resizable panel (width persisted in `localStorage`).

## Per-robot xacro override (one-time setup)

The wrist camera mount is defined in a **shared vendor xacro** at
`src/tk25_manipulation/src/xarm_ros2/xarm_description/urdf/camera/realsense_d435i.urdf.xacro`
(joint `camera_link_joint`). Patching the vendor file in place would
overwrite tinker1's calibration when tinker2 calibrates and vice versa.

Instead, `handeye_web` writes a **per-robot override** at:

```
src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/wrist_camera.xacro
```

For the override to actually take effect in the live URDF, the operator must
do a **one-time** include from the main robot xacro (e.g.
`src/tk25_basic/src/tinker_urdf/src/mobile_manipulator.urdf.xacro`):

```xml
<xacro:include filename="$(find tinker_robot_config)/robots/$(arg robot_name)/wrist_camera.xacro"/>
```

…and either remove the corresponding `<joint name="camera_link_joint">`
block from the vendor d435i xacro, or guard it with
`<xacro:unless value="$(arg use_handeye_override)">…</xacro:unless>`.
This wiring happens once per workspace; afterwards every `handeye_web`
calibration just overwrites the per-robot override file in place (with a
timestamped backup).

If `ROBOT_NAME` is unset when promoting, the UI offers **yaml-only
promote** — the `hand_eye.yaml` is still written, but the xacro half is
disabled with a banner explaining why. The promote endpoint will **refuse**
to write the shared vendor xacro under any circumstance (path-prefix
check; see `apply_promote` in `handeye_web.py`).

When the per-robot override file does not yet exist, the UI's xacro diff
shows the **seed** template (a complete one-joint `<robot>` body with a
header comment instructing the include + vendor-disable). When it does
exist, the UI shows a **patch** of the existing file's `<origin>` only,
preserving the rest of the override verbatim.

## Waypoints + auto-capture

The Capture tab supports an ordered list of arm waypoints and a one-click
auto-capture sweep:

1. Move the arm to a pose you want (xArm teach mode, or the Move tab's
   joint editor). Click **+ Add current joints** to append the live
   `xArm joints` to the waypoint list.
2. Repeat until you have 12-20 diverse poses (the diversity meter helps —
   you want > 30° of rotation spread).
3. Click **Save to disk** — the list persists to
   `src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/handeye_waypoints.yaml`
   (atomic write with timestamped backup). Re-runs of the web server
   reload the file on startup; you only record the sequence once per
   robot.
4. Click **Run dry** first to verify every waypoint is reachable
   (move + settle, NO capture). The arm sweeps through the list; any
   waypoint that times out on settle is logged and skipped. Cancel
   stops the run after the current step.
5. When the dry-run looks clean, click **Run sequence** for the real
   thing. Each settled pose feeds the manual capture path — same
   StabilityTracker gate, same SafetyEnvelope check, same sample
   gallery population. The Solve tab is reachable as soon as ≥ 6
   diverse samples are in the gallery.

The sequence runner is single-instance — `Run sequence` is disabled
while another run is in flight. **Cancel** is cooperative + immediate:
the in-flight JointMove goal is cancelled and the runner exits at the
next state transition (typically within 50 ms). Operator safety is on
the arm driver + SafetyEnvelope, not on this UI.

## Acceptance gate

Held-out poses must clear pan-tilt parity: translation < 3 mm, rotation < 0.5°,
reprojection < 1.5 px (PASS; within 2× = WARN). Note the held-out **rotation** metric
compares against single-shot PnP, so it also reflects observation noise; on hardware
the 10-frame consensus voter reduces that. The live overlay is the human-trustable
check: predicted board corners should track the real corners within a few px across
the workspace.

## Changelog
- 0.5.0 (2026-06-20): **Waypoint authoring + auto-capture sequence.** New backend module
  `waypoints.py` with `WaypointStore` + per-robot YAML persistence
  (`src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/handeye_waypoints.yaml`).
  Auto-loaded on startup; refuses save when `ROBOT_NAME` unset.
  - Capture tab gained a waypoints sub-panel: add-current / per-row load+delete /
    save+reload buttons, and a sequence controls strip below it:
    **Run sequence**, **Run dry**, **Cancel**, with a live progress line and
    bounded log.
  - New `CaptureSequenceRunner` daemon-thread state machine drives
    move → settle → capture per waypoint. Reuses the existing JointMove
    client + StabilityTracker + `do_capture` path — no fast path bypassing
    the SafetyEnvelope. Cancel is cooperative + immediate (in-flight goal
    cancelled, runner exits at next transition).
  - New endpoints: `GET/POST /api/waypoints`, `DELETE /api/waypoints/{idx}`,
    `POST /api/waypoints/save`, `POST /api/waypoints/reload`,
    `POST /api/sequence/start` (body `{dry_run: bool}`),
    `POST /api/sequence/cancel`. WS state extended with `state.waypoints`
    + `state.sequence`.
- 0.4.0 (2026-06-20): **handeye_web quality rewrite to pan_tilt parity.** The
  v1 inline ~30-line UI was replaced by a static `webui/` (index.html +
  style.css + app.js) bundle covering all five tabs. New surface:
    - **Frontend**: WebSocket state stream (5 Hz, no polling for state);
      tabbed layout (Info / Move / Capture / Solve / Promote); resizable
      live frame with annotated / raw toggle and detection badge
      (corners + RMS, colour-coded); per-tab content as listed under
      `## UI` above.
    - **Capture**: `StabilityTracker` is now the hard pre-capture gate
      (closes the v1 deferral); sample gallery with thumbnails + per-sample
      delete; diversity meter (max pairwise rotation° / 30° target).
    - **Solve tab**: method picker (auto / TSAI / PARK / HORAUD / ANDREFF /
      DANIILIDIS), per-method comparison table, residual histogram +
      scatter canvases, board-coverage canvas, mm/deg units, PASS/WARN/FAIL
      gate pill. New helpers `web_support.solve_payload_v2`,
      `seed_handeye(methods=)` / `solve(methods=)` kwargs,
      `HandeyeWebNode.do_solve(method=)`.
    - **Promote tab**: side-by-side unified-diff preview for **both**
      `hand_eye.yaml` AND a per-robot `wrist_camera.xacro` override
      (ROBOT_NAME-scoped; refuses to overwrite the shared vendor xacro);
      confirm-before-apply per half; backup paths surfaced. New
      `apply_handeye` helpers `resolve_robot_xacro_path`,
      `seed_handeye_override_xacro`; `write_with_backup` now returns the
      backup path and `os.makedirs(parent, exist_ok=True)`s its target.
      New `HandeyeWebNode.compute_promote_diff` / `apply_promote` /
      `reload_promote` accessors; the v1 `do_promote` becomes a back-compat
      shim onto `apply_promote(which='both')`.
    - **Endpoints**: `/ws`, `/api/samples/{idx}/thumb.jpg`,
      `DELETE /api/samples/{idx}`, `/api/promote/diff` (returns both
      yaml + xacro halves), `/api/promote/apply` (accepts
      `which ∈ {yaml,xacro,both}`), `/api/promote/reload`. `/api/solve`
      accepts `{method}` body; `/api/frame.jpg` accepts `?raw=1`.
    - **Per-robot xacro override** convention added (see
      `## Per-robot xacro override (one-time setup)` above).
- 0.3.0 (2026-06-15): handeye_web server implemented (live ChArUco overlay,
  capture/solve/promote) + launch file (handeye_web.launch.py).
- 0.2.0 (2026-06-15): math core (transforms/model/solver/gates), synthetic harness,
  collection node, calib_web-style web tool, yaml/URDF persistence.
- 0.1.0 (2026-06-15): package scaffold.
