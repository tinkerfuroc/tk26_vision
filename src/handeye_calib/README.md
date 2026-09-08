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
ChArUco corner reprojection **plus a FoundationStereo metric-depth residual at the
corners** (the depth pins the optical-axis/scale DOF that planar reprojection leaves
weak — see the 0.6.0 changelog entry). Held-out poses gate the result.

Module map:
- `transforms.py` — SE(3) helpers (incl. `rigid_3d_3d` Umeyama 3D-3D fit).
- `depth_sample.py` — sample FFS depth at ChArUco corners → metric camera-frame points.
- `handeye_model.py` — ChArUco board geometry + pinhole reprojection.
- `synthetic.py` — synthetic ground-truth scenarios (+ `handeye_synthetic_check` CLI).
- `handeye_solve.py` — multi-method seed → bundle-adjust (+ depth residual) → held-out evaluate + gate.
- `gates.py` — settle/stability, pose-diversity, per-frame quality gates.
- `handeye_collect.py` — ROS node: drive arm, settle-gate, detect, accumulate.
- `handeye_web.py` — calib_web-style author / run / verify / promote tool.
- `apply_handeye.py` — compose to the URDF mount frame; write `hand_eye.yaml` / the per-robot `wrist_camera.xacro` property-override.

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

Brings up the `handeye_web` node (FastAPI UI + rclpy). Most ROS params are exposed
as launch args (`bind`, `port`, `robot_name`, camera topics, `base_frame`/`eef_frame`,
`aruco_dict`, board geometry, `jointmove_action`, `mount_to_color_*`, `min_diversity_deg`,
and the FFS/depth knobs `use_ffs_depth`, `ffs_service`, `depth_weight`, `depth_sigma_m`,
`depth_win`, `depth_z_min`/`depth_z_max`, `depth_min_corners`, `ffs_wait_for_service_s`,
`ffs_call_timeout_s`, `ffs_depth_warn_after`). A few niche params (e.g. `settle_timeout_s`,
`settle_steady_ticks`, `settle_poll_s`, `capture_min_steady_frames`, `min_corners`,
`joint_states_topic`) are not declared in the launch file yet — set those
via `ros2 run … --ros-args -p name:=value`. `capture_min_steady_frames` (default 5) is the
settle floor: the board-stability verdict must HOLD for that many consecutive frames before
a capture is accepted (manual click and the auto sweep both honor it) — raise it for a longer
settle.
The RealSense camera (`realsense2_camera`, `camera_name:=xarm_camera`) must be launched
**separately** — the UI shows `no camera` until color frames arrive. `ros2 launch`
teardown (SIGINT/SIGTERM) shuts the server down cleanly.

### FoundationStereo depth (default ON)

Since 0.6.0 the solve fuses FFS metric depth (see the changelog). For it to
actually contribute on hardware you need **both**:
1. The `foundation_stereo` node running:
   `ros2 launch foundation_stereo foundation_stereo.launch.py`.
2. The wrist RealSense brought up **with the IR stereo streams FFS consumes** —
   the canonical color-only bringup in `CAMERA_BRINGUP.md` does *not* enable them,
   so add `enable_infra1:=true enable_infra2:=true` to the `rs_launch.py` line.
   Without the IR pair, FFS returns `status=1` ("no synced stereo frame") on every
   call, every capture records `depth_source='unavailable'`, and the solve silently
   falls back to monocular. The node logs a one-time WARN after
   `ffs_depth_warn_after` (default 5) depth-less captures, and the Solve log line
   shows `depth: N/M samples`. Set `use_ffs_depth:=false` to opt out entirely.

**Gate caveat (important).** The PASS/WARN/FAIL pill is scored on `trans_rmse_m`
(predicted board-in-cam vs the *monocular PnP* pose) and reprojection. When FFS
depth is active **and your intrinsics/board-scale are slightly off**, the depth
term intentionally pulls `X` away from the biased PnP estimate — which *inflates*
`trans_rmse_m`/reproj and can flip the pill to WARN/FAIL even though the
depth-grounded calibration is more accurate. In that case read
**`depth_point_rmse_mm`** (the held-out depth-vs-solve agreement, surfaced in the
Solve metrics + log) as the honest real-world error budget, and treat a large
gap between the two numbers as a signal to re-check camera intrinsics or FFS depth
calibration. If `depth_point_rmse_mm` and `trans_rmse_m` agree and are both small,
the calibration is trustworthy.

## Verify without hardware

The whole solver is provable on a laptop against synthetic ground truth:
```bash
python -m handeye_calib.synthetic     # prints recovered-X error + PASS
pytest test/                          # unit suite (source the workspace first for test_import)
```

## UI

The `handeye_web` tool is a single-page calibration UI with four tabs:

- **Info** — camera / TF / robot / ChArUco board / safety envelope status, plus the `T_base_eef` matrix.
- **Capture** — single-tab end-to-end authoring + capture workflow (mirrors `pan_tilt/calib_web`'s "xArm Waypoints" tab):
  - Joint editor (rad/deg toggle), Load-current / Zero / Move (joints) / presets, with a live SafetyEnvelope verdict on the current EE pose before sending.
  - Waypoints sub-panel: record an ordered list of arm poses, save/reload per-robot (see [Waypoints + auto-capture](#waypoints--auto-capture) below).
  - Auto-capture sequence controls: Run / Run dry / Cancel + live progress + bounded log.
  - Manual capture: stability-gated single-shot button — the board-pose StabilityTracker (last 5 poses within < 3 mm / 0.5°) must stay steady for ≥ `capture_min_steady_frames` consecutive frames (default 5) before the shot is accepted, so the button waits out the mount's post-move micro-settle instead of firing on the first steady frame. Sample gallery with per-capture thumbnails + per-sample delete, diversity meter (max pairwise rotation° / 30° target).
- **Solve** — method picker (auto / TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS), per-method reprojection comparison table, residual histogram + scatter, sample-coverage canvas, PASS / WARN / FAIL gate pill with mm / deg / px units.
- **Promote** — side-by-side unified-diff preview for **both** `hand_eye.yaml` AND a per-robot `wrist_camera.xacro` override, ROBOT_NAME-scoped, confirm-before-apply, backup paths surfaced for each write. Reload-from-disk clears the cached solve so the operator can re-run.

Live state pushed via WebSocket at 5 Hz (no polling for state). The live camera feed polls `/api/frame.jpg` at ~3 Hz with an annotated / raw toggle and a resizable panel (width persisted in `localStorage`).

## Per-robot xacro override (one-time setup)

The wrist camera mount is defined in a **shared vendor xacro** at
`src/tk25_manipulation/src/xarm_ros2/xarm_description/urdf/camera/realsense_d435i.urdf.xacro`.
That file declares `handeye_xyz` / `handeye_rpy` xacro **properties**
(factory-nominal defaults), then — when `ROBOT_NAME` is set — does
`<xacro:include filename="$(find tinker_robot_config)/robots/$(optenv
ROBOT_NAME)/wrist_camera.xacro"/>` before consuming those two properties in
its `camera_link_joint`'s `<origin>`. Patching the vendor file in place
would overwrite tinker1's calibration when tinker2 calibrates and vice
versa, so instead `handeye_web` writes a **per-robot override** at:

```
src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/wrist_camera.xacro
```

Since 2026-07-03, Promote **always** writes the full **property-redefinition
form** — a complete `<robot>` body that redefines `handeye_xyz` /
`handeye_rpy` (`<xacro:property name="handeye_xyz" value="..."/>` etc.),
matching exactly what the vendor `<xacro:include>` consumes. This is the
*only* form that actually takes effect: an earlier version of this writer
emitted a `<joint>` block instead, which was silently inert (the vendor
xacro already owns the one `camera_link_joint` and reads its origin from
the two properties — a sibling `<joint>` in the include never got used).
Every promote fully overwrites the per-robot file (with a timestamped
backup), whether it already exists or not — there is no more separate
"patch existing `<origin>`" mode.

For the override to actually take effect, `ROBOT_NAME` must be exported
before the URDF is parsed (see the workspace's per-robot env setup,
`src/tk25_basic/tools/robot-env.sh`) — no additional one-time xacro wiring
is required beyond what the vendor xacro already does.

If `ROBOT_NAME` is unset when promoting, the UI offers **yaml-only
promote** — the `hand_eye.yaml` is still written, but the xacro half is
disabled with a banner explaining why. The promote endpoint will **refuse**
to write the shared vendor xacro under any circumstance (path-prefix
check; see `apply_promote` in `handeye_web.py`).

**After a successful xacro promote, rebuild + relaunch before the new mount
pose is live** — the promote success message says so, but to spell it out:

```
tkbuild tk25_basic --packages-select tinker_robot_config
```

...then relaunch `robot_state_publisher` / the arm bringup so the freshly
built `tinker_robot_config` package (and the xacro include it now serves)
is actually re-parsed.

## Waypoints + auto-capture

The Capture tab supports an ordered list of arm waypoints and a one-click
auto-capture sweep:

1. Move the arm to a pose you want (xArm teach mode, or the Capture tab's
   built-in joint editor at the top). Click **+ Add current joints** to
   append the live `xArm joints` to the waypoint list.
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
- 0.13.0 (2026-07-03): **Fixed dead session/dump persistence (`self._robot_name` never
  existed) + robot-tagged session/dump names.**
  - **Root cause:** `_build_session_dict` read `self._robot_name`, an attribute the
    node never assigned (the real accessor is `_resolve_robot_name()`). Every call
    raised `AttributeError`, which `_persist_session`'s best-effort `except Exception`
    swallowed — so **every** capture/solve since this line was introduced silently
    failed to write `session.json` (returned `None`, logged only a WARN). Caught by 4
    pre-existing `test_sessions_web.py` failures. Fixed: `"robot":
    self._resolve_robot_name() or None`.
  - **Robot-tagged names.** `handeye_sessions.new_session_name(now_struct=None,
    robot='')` now optionally tags the session dir: `wrist_handeye_<robot>_<ts>`
    (legacy positional call with no `robot` kwarg is unchanged —
    `wrist_handeye_<ts>`). New `_safe_robot_tag(robot)` reuses the existing
    `_safe_name` path-traversal guard and **degrades to the untagged legacy name on
    any unsafe/empty tag rather than raising** — a raise here would silently kill
    persistence again, the exact bug class this release fixes. `_persist_session` /
    `_dump_solve` (`solve_<robot>_<ts>.json`) both pass `self._resolve_robot_name()`
    through this path.
  - **Fixed a second, related latent bug** this unmasked once real v2 (multi-placement)
    sessions started actually reaching disk: `_summary()`'s `"n_samples"` field always
    read the (v1-only) top-level `data["samples"]`, so every v2 session listed
    `n_samples: 0` in the History tab regardless of real content (`n_samples_total`
    was computed correctly but never used for the `n_samples` field itself — now it
    is). The `GET /api/sessions/{name}` detail endpoint and the
    `.../samples/{i}/thumb.jpg` route had the same v1-only assumption (flat
    `data["samples"]` / flat `thumbs/<idx>.jpg`, both nonexistent for a v2 session
    whose samples + thumbs are nested per placement) — since no v2 session had ever
    reached these routes with real content before, the History tab's detail view and
    thumbnail gallery were silently broken for every capture since the multi-placement
    migration. Fixed with two new pure-FS helpers in `handeye_sessions.py`:
    `flatten_samples(data)` (concatenates placement samples in order; passthrough for
    v1) and `flat_thumb_path(name, flat_idx, data=None, base=None)` (maps a flat
    gallery index to the owning placement's on-disk thumb, or the legacy flat path for
    v1).
  - `session.json`'s `"robot"` field is restored end-to-end (was always writing an
    `AttributeError`-crash before this fix, never a real value). **Pending operator
    verify:** first live capture on hardware confirming
    `wrist_handeye_sessions/wrist_handeye_<robot>_<ts>/session.json` has `"robot":
    "<robot>"` and schema `wrist_handeye_session/2` actually lands on disk (skipped
    here — needs the wrist RealSense, which may be in use by the live robot).
- 0.12.1 (2026-07-03): **Promote now writes the property-redefinition xacro form the
  vendor URDF actually consumes** (was silently inert).
  - `apply_handeye.seed_handeye_override_xacro(robot_name, xyz_str, rpy_str)` (the
    `joint_name` param is gone) now emits a `<robot>` body that redefines the
    `handeye_xyz` / `handeye_rpy` **xacro properties**
    (`<xacro:property name="handeye_xyz" value="..."/>`), matching exactly what
    `xarm_description/urdf/camera/realsense_d435i.urdf.xacro`'s
    `<xacro:include>` reads. The previous template wrote a standalone `<joint
    name="{joint_name}">` block — the vendor xacro already owns the one
    `camera_link_joint` and reads its `<origin>` from those two properties, so
    that sibling `<joint>` in the include was **never used**. This is why
    tinker2's deployed `wrist_camera.xacro` had to be hand-copied instead of
    Promoted (see that file's changelog comments, 2026-06-22 through
    2026-07-01) — Promote's own writer couldn't produce a file that worked.
  - `handeye_web.py`'s `_xacro_half_for_diff` **always** writes the full
    property-form file via `write_with_backup` now (`mode` is always `"seed"`);
    the old `patch_urdf_origin(existing_file, joint_name, ...)` branch — which
    tried to patch an existing override's inert `<joint>` `<origin>` in place —
    is deleted. `patch_urdf_origin` itself is unchanged and still covered by
    its direct tests (it has no other caller now; kept for the generic
    `<joint>`-origin-patch utility it is).
  - The Promote **apply** response for a successful xacro write now includes a
    `message` field (surfaced in the UI status line) reminding the operator to
    `tkbuild tk25_basic --packages-select tinker_robot_config` and relaunch the
    arm bringup — the on-disk xacro alone does nothing until the package is
    rebuilt and `robot_state_publisher` re-parses it.
  - No change to `hand_eye.yaml` promote, to `patch_urdf_origin`'s own
    behavior/tests, or to the vendor-xacro refusal guard.
- 0.12.0 (2026-06-28): **Longer, pan-tilt-style settle: capture waits for held steadiness.**
  - New param `capture_min_steady_frames` (default **5**). A capture is now accepted
    only after the `StabilityTracker` verdict has HELD for ≥ this many consecutive
    steady frames, instead of firing on the first steady verdict — mirroring the
    pan-tilt calibration's "held for a duration" image-stability gate. Gives the
    wrist mount extra frames to stop micro-settling after the 5-frame stability
    window first agrees. Raise it for an even longer settle.
  - Applies to BOTH capture paths: the manual `/api/capture` gate (was: any single
    steady frame) and the automated waypoint sweep's `_wait_for_settle` (now requires
    the poll hold AND the ≥N-frame floor, so a slow camera can't satisfy the poll
    count with too few real stable frames). Reject reason reads `not stable yet
    (k/N steady frames)`.
  - Does NOT change the static error floor (mount flex / FK compliance) — settle only
    addresses transient/ring-down error; see the rigid-mount check for the real gate.
- 0.11.1 (2026-06-28): **Test-pollution fix + MAD audit + honest rejection log.**
  - **MAD rejection audited and confirmed correct** (no over-/under-rejection):
    each drop is a genuine outlier (gross board-flip, or a translation/rotation/
    reprojection outlier past both the robust-z and the physical floor), kept
    borderline poses are typical of the dataset, deterministic per config. A
    clean MAD pass does NOT imply a good calibration — the rigid-mount check is
    the real gate.
  - **Captures were NOT failing to accumulate** — that was a test-hygiene bug:
    `do_capture` now persists each capture to `$HANDEYE_DUMP_DIR` (default
    `calibration_data/`), and the no-hardware capture tests didn't isolate that
    dir, so they wrote synthetic identity-matrix n=1 sessions into the operator's
    real `wrist_handeye_sessions/`. Added `test/conftest.py` with an **autouse
    fixture pinning `HANDEYE_DUMP_DIR` to a tmp dir** for every test, so no test
    can ever touch the real tree. (Removed 48 polluted sessions; real captures
    were unaffected.)
  - **Rejection-log residuals re-scored against the final clean fit** so the
    reported mm/deg/px are each measured against one consistent reference (the
    at-drop snapshot used a different intermediate fit per drop). z-scores stay
    as the at-drop trigger evidence.
- 0.11.0 (2026-06-28): **Web app made usable: live coverage, live MAD streaming,
  state-driven Solve tab, and an intrinsics-free rigid-mount go/no-go.** Fixes a
  cluster of UI/state-sync bugs found by a systematic debug pass + drives the
  calibration verdict home.
  - **Coverage now renders.** The state payload was missing `K` and per-sample
    `T_cam_board`, so the projector returned null for every point. Added both
    (`web_support.enriched_state_payload` gains `K`; `sample_metadata` gains
    `T_cam_board`; `get_state_dict` emits them). Coverage now draws **live as
    samples are captured**, not only after a Solve.
  - **MAD rejections stream live.** `/api/solve` ran `do_solve` *synchronously on
    the asyncio loop*, blocking every WebSocket push for the whole solve — so
    nothing updated mid-solve and `last_solve` was hardcoded `None` in the push.
    Now: the route offloads to an executor (`run_in_executor`); `hs.solve` takes
    a `progress_cb` that fires per rejection; `do_solve` streams it into
    `state.solve_progress` and caches the final JSON payload into
    `state.last_solve` (with `solve_ts`). The UI shows "rejecting… dropped N
    (kept …)" with the why-table filling in real time, and the Solve tab
    **rehydrates on reload/reconnect** instead of blanking.
  - **Rigid-mount check (`hs.rigid_closure_deg`).** An intrinsics-/scale-/depth-
    independent AX=XB rotation-conjugacy diagnostic: for a rigid mount the flange
    motion and camera motion have equal rotation angle, so the median mismatch is
    a pure mechanical-rigidity readout. Surfaced in the solve payload + a Solve-
    tab banner as a go/no-go: if it FAILs, **no** solve can pass — rigidify the
    mount and recapture (don't tune the solver).
  - **Cache-busting** (`?v=` on app.js/style.css) + defensive render guards so a
    stale browser cache / one missing field can't make the page look broken.
  - **Calibration verdict (no code change):** the 2026-06-28 captures fail because
    the camera-to-flange mount is **not rigid** (≈2.5° AX=XB closure floor,
    gravity-correlated). Recapture after rigidifying the mount; verify with the
    rigid-mount check before trusting any solve.
- 0.10.0 (2026-06-28): **Captures persisted to disk + historical-session browser
  (pan-tilt parity), and MAD rejections shown where the operator looks.**
  - **Every capture is persisted** to
    `<HANDEYE_DUMP_DIR|calibration_data>/wrist_handeye_sessions/<name>/`
    (`session.json` = meta + samples + optional solve result; `thumbs/<i>.jpg`).
    Written on each capture/delete/solve, so a restart or crash never loses a
    capture. (The flat `wrist_handeye_dumps/solve_*.json` replay archive stays.)
    A new module `handeye_sessions.py` holds the pure FS helpers; new node hooks
    `_persist_session` / `_build_session_dict`.
  - **History tab.** New endpoints `GET /api/sessions`, `GET /api/sessions/{name}`,
    `GET /api/sessions/{name}/samples/{i}/thumb.jpg`,
    `POST /api/sessions/{name}/load`, `DELETE /api/sessions/{name}`. The History
    tab lists past captures (date / #samples / #rejected / status), and opening
    one shows its gallery, solve verdict + per-sample residuals + why-rejected.
    **Load into live session** re-hydrates an old capture (samples + thumbs +
    intrinsics + board + head anchor) so it can be **re-solved with the current
    solver** — invaluable for re-running an old capture through the new all-sample
    MAD.
  - **MAD rejections visible in the Capture gallery**, not just the Solve tab:
    a solved-then-rejected sample gets a red outline + `REJECTED (MAD)` badge in
    the gallery where the operator actually looks (the previous UI surfaced
    rejections only as blanked histogram points, so it read as "nothing rejected"
    even when the backend had weeded several poses).
- 0.9.0 (2026-06-28): **Solve on ALL samples (no train/held-out split), MAD
  parity + visibility, mm/deg residuals.** Aligns the wrist solve with the
  pan-tilt calibration's model and surfaces its reasoning to the operator.
  - **No train/held-out split.** `solve()` fits X/Tbb on, and reports the
    residual over, *every* surviving sample (pan-tilt parity). Generalization is
    measured separately later with a freshly-recorded validation pose set.
    `SolveResult` now carries a single `metrics` block (was
    `train_metrics`/`heldout_metrics`); `solve()` dropped the `heldout_frac` /
    `rng_seed` params and `split_train_test`. The yaml metadata keys were renamed
    `heldout_* → residual_*`.
  - **MAD screens the whole set.** The previous loop ran on the train split only,
    so an outlier that landed in the held-out 20% was never weeded — on real data
    a 26 px pose survived in plain sight. Rejection now iterates over all samples
    (single-worst-drop-then-resolve, same per-axis floors), so an outlier is
    caught wherever it sits.
  - **Anchor (head) MAD.** `average_board_anchors` now runs the same per-axis MAD
    rejection on the head observations before averaging, so one bad pan/tilt read
    can't drag the warm-start; scatter reports `n` / `n_total` / `n_rejected` /
    `rejected`.
  - **Visibility (pan-tilt parity).** `SolveResult.rejection_log` records, per
    dropped sample, the residual (mm / deg / px) and the robust-z that triggered
    it; surfaced in the solve log, the JSON dump, and the UI (per-sample residual
    table + "rejected samples — why" table). The anchor button shows kept/total +
    which observations were rejected.
  - **Human-readable residuals.** Per-sample residuals are now reported in
    `per_sample_trans_mm` / `per_sample_rot_deg` alongside `per_sample_reproj_px`
    (pixels alone are hard to judge).
- 0.8.0 (2026-06-27): **Head-Orbbec warm-start + pan_tilt parity ports.** Two
  threads landed together (plan: `../../docs/plans/2026-06-27-handeye-headassist-parity.md`).
  - **Head warm-start (optional).** The already-calibrated pan-tilt head Orbbec
    observes the *same fixed board* and supplies `T_base_board`; that measured
    board pose becomes a basin-immune **seed** (`seed_from_board_anchor`,
    `solve(anchor_Tbb=...)`) fed into a **multi-start** bundle adjust alongside
    the existing `calibrateHandEye` seed (`_solve_once` keeps the lowest post-BA
    reprojection). `Tbb` stays a FREE bundle-adjust parameter, so the head's
    absolute bias only picks the convergence basin and is NOT injected into the
    final `X` — wrist reprojection + FFS depth still own sub-3 mm accuracy. New
    node surface: head camera subs (`head_image_topic` / `head_info_topic` /
    `head_optical_frame`), `do_anchor_board()` + `POST /api/anchor`
    (+ `/api/anchor/clear`), a Capture-tab "Anchor board (head)" button, and an
    `anchor` block in the WS state. The head TF is looked up relative to the arm
    `base_frame` (chains through `base_link`); anchors are averaged over repeated
    looks with a scatter readout (`average_board_anchors`). Honest ceiling:
    anchoring cuts the required pose count / rescues degenerate or wrong-basin
    seeds; it cannot beat the head's ~3 mm/0.5° (systematic) base-frame accuracy,
    so it is a seed + cross-check, never the promoted result.
  - **Consensus capture (pan_tilt parity).** `do_capture` averages the last N
    steady detections' corners (`consensus_corners`, per-corner median over a 60%
    frame quorum) and re-PnPs them with an IPPE-seed → ITERATIVE refine
    (`_pnp_ippe_refine`, planar-flip safe), replacing the single-shot pose; falls
    back to the single frame when quorum / the ≥6-corner PnP floor can't be met.
    Kills the single-frame PnP noise the 0.5° held-out rotation gate previously ate.
  - **Observability diagnostic.** `rotation_observability` flags AX=XB
    rank-deficiency (relative-rotation axes nearly collinear ⇒ `X` rotation
    unobservable) via the 2nd singular value of the stacked pairwise axes;
    surfaced in the Solve payload + a UI WARN.
  - **Outlier rejection ON by default.** `solve()` defaults `reject_sigma=2.5`,
    `max_reject_frac=0.25`; the loop scores per-sample **SE(3) chain error** with
    separate per-axis modified z-scores (catches a bad-FK / bad-depth sample a
    reprojection-only metric misses) and drops the single worst per round, then
    re-solves. A sample must be **both** a statistical outlier **and** beyond an
    absolute physical band (`reject_min_trans_m=0.01` = 10 mm,
    `reject_min_rot_rad=3.0°`) to be dropped, so clean right-skewed residuals at
    small N aren't trimmed (validated zero-reject across a 40-seed clean sweep;
    catches a 5 cm FK outlier on 10/10 seeds). The held-out split is never
    rejected. Override via `/api/solve` `reject_sigma` (omit → default 2.5;
    `null` → off; number → that). Floors are plain `solve()` kwargs (retunable
    against arena rosbags; not UI-exposed).
- 0.7.0 (2026-06-22): **Calibrate in color OR left-IR (`calib_frame`), runtime-
  adjustable in the UI; compose to the true `camera_link`.** The hand-eye unknown
  is the rigid camera *body* (`camera_link`); every optical frame is a fixed
  factory child of it. On a D435 **`camera_link` ≡ the left-IR / depth sensor**
  (vendor URDF left-IR joint is `0 0 0`; color is +15 mm). So observing the board
  in **left-IR** measures the body directly with **native un-warped FFS depth**,
  while color samples a point 15 mm away and back-composes through the factory
  offset. New `calib_frame` param (`color` default | `ir`) switches the observed
  frame; the stored artifact is **frame-agnostic** — always `T_eef→camera_link`
  (the `camera_link_joint` origin) + a `color_optical` reference, identical
  regardless of which frame was observed.
  - **Bug fix folded in:** the compose now uses the **real** D435 internal
    geometry (`mount_to_color` defaulted to identity before, silently writing
    `T_eef→color_optical` into the `camera_link` joint — the reason the deployed
    `tinker2` xacro needed a *manual* correction). Color-mode promote now writes
    the correct `camera_link` directly.
  - **IR-mode requirements:** FFS streaming native-IR depth
    (`stream_enabled:=true stream_align_to_color:=false`; or point
    `ffs_ir_depth_topic` at `/camera/xarm_camera/depth/image_rect_raw`), the wrist
    IR streams (`enable_infra1/2:=true`), and the **IR emitter OFF** (the dot
    pattern corrupts ChArUco; FFS is passive so depth survives — toggle it from
    the UI, which flips the driver's `depth_module.emitter_enabled`). Add ambient/
    IR flood light if the IR image is dark.
  - **Runtime-adjustable web UI** (Info tab → *Calibration settings*): `calib_frame`
    radio (switching discards the frame-specific samples, with a confirm), depth
    knobs (`depth_weight/sigma_m/win/min_corners`), `use_ffs_depth`, and the IR
    emitter toggle — all via `POST /api/config`, surfaced in `state.config`.
  - **A/B procedure:** flip to IR, recalibrate, and compare the held-out
    `depth_point_rmse_mm` (and FFS-vs-PnP agreement) against color — the
    better-conditioned frame on your hardware wins. New params: `calib_frame`,
    `ir_image_topic`, `ir_info_topic`, `ffs_ir_depth_topic`, `mount_to_ir_xyz/rpy`,
    `camera_node_name`. New `transforms.rigid_3d_3d` stays the standalone util.
- 0.6.0 (2026-06-22): **FoundationStereo (FFS) metric depth in the solve.** The
  per-view `T_cam_board` was pure monocular ChArUco PnP, whose optical-axis
  (depth/scale) translation is the weakest-constrained DOF — which directly
  limited how well the rigidly-mounted camera pose `X = T_eef→color` could be
  recovered. Now, at each capture the node calls the FFS `get_depth` service
  (color-aligned 32FC1 metres, same color intrinsics the solver already uses),
  samples depth at the detected corner pixels, deprojects to metric camera-frame
  points (`depth_sample.deproject_corners`, robust local-median + validity
  mask), and stores them on the `Sample`. The bundle adjust adds a sigma-weighted
  3D **depth residual** (`depth_weight=1.0`, `depth_sigma_m=0.005`) that pins the
  optical-axis DOF reprojection can't see, while sub-pixel reprojection keeps
  owning rotation. New **depth-grounded** held-out metric `depth_point_rmse_mm`
  (compares the solved chain against an *independent* metric measurement, not the
  PnP pose — read it as the real-world error budget; see the Gate caveat under
  §FoundationStereo depth).
  - **Adversarial-review hardening (2026-06-22):** `depth_weight` defaults to
    **1.0** (not 2.0) so a realistic systematic FFS scale bias can't out-vote the
    sub-pixel reprojection and drag a good calibration past the gate; the FFS
    client body is fully exception-guarded (any rclpy error → monocular, never a
    500 / sequence-abort); steadiness is re-checked after the blocking FFS call so
    a pose that drifts during it drops depth (`moved-during-ffs`); per-sample
    `depth_source` is surfaced in the gallery; a one-time WARN fires if FFS is
    enabled but never delivers depth (the IR-streams-not-enabled trap);
    `depth_z_min`/`depth_z_max`/`depth_min_corners` are exposed; the depth
    residual ANDs validity with finiteness so a NaN hole can't poison the solve.
  - **Default ON, degrades gracefully to monocular** when FFS is unavailable
    (service missing / timeout / non-zero status / shape mismatch / <3 valid
    corners) — depth is a refinement, never a capture-admission gate, so an FFS
    hiccup never blocks a pose. Requires the `foundation_stereo` node running
    (`ros2 launch foundation_stereo foundation_stereo.launch.py`) to take effect;
    otherwise the solve is identical to the prior monocular behaviour.
  - New params: `use_ffs_depth` (True), `ffs_service`
    (`/foundation_stereo/get_depth`), `ffs_wait_for_service_s` (1.0),
    `ffs_call_timeout_s` (10.0), `depth_weight` (1.0), `depth_sigma_m` (0.005),
    `depth_win` (2), `depth_z_min` (0.05), `depth_z_max` (2.0),
    `depth_min_corners` (3), `ffs_depth_warn_after` (5). New module
    `transforms.rigid_3d_3d` (Umeyama, standalone util) + `depth_sample.py`.
    Reuses the proven `object_seg_yolo._try_ffs_depth` client
    pattern. Back-compat: `Sample`/`CaptureSession`/`bundle_adjust` depth args
    default to off, so the monocular path is byte-identical when no depth is present.
- 0.5.1 (2026-06-20): **Move tab folded into Capture tab.** The standalone
  "Move" tab was redundant — its joint editor + presets + live SafetyEnvelope
  preview were the prerequisite for waypoint authoring, so the workflow felt
  fragmented across two tabs. Merged: the Capture tab now mirrors
  `pan_tilt/calib_web`'s "xArm Waypoints" tab layout — joint editor →
  presets → waypoints sub-panel → auto-capture sequence controls → manual
  capture + stability + gallery + diversity, all in one top-to-bottom scroll.
  Tab count drops from five (Info / Move / Capture / Solve / Promote) to
  four (Info / Capture / Solve / Promote). All DOM IDs (`move-*`,
  `waypoint-*`, `sequence-*`, `capture-*`) unchanged — pure layout move.
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
