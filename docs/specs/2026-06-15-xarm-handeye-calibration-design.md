# Eye-in-Hand Calibration for the Wrist RealSense — Design Spec

- **Date:** 2026-06-15
- **Status:** Approved design, pending implementation plan
- **Robot:** tinker1 / tinker2 (xArm7 + wrist-mounted RealSense)
- **Author:** Claude (brainstormed with cindy)

## 1. Problem & Goal

The RealSense is rigidly mounted on the xArm flange (`link_eef`). We need the fixed
transform **`T_eef→color_optical`** (end-effector → RGB optical frame), accurately,
quickly, and with a verifiable acceptance gate.

Today the URDF carries a hardcoded CAD guess for the camera mount
(`xarm_camera_link` attached to `link_eef` at `xyz="0.06746 -0.0175 0.0237"
rpy="π -π/2 0"` in `realsense_d435i.urdf.xacro`). The camera was recently
**translated forward** from that position, so the guess is now stale. Calibration
replaces the guess with a measured value and writes it to the robot config (and,
optionally, the URDF mount joint).

This is the classic **eye-in-hand** hand-eye problem: one rigid unknown, solved by
moving the arm while it observes a board that is **fixed in the world**.

## 2. Decisions (from brainstorm)

| Topic | Decision |
|---|---|
| Board | **ChArUco 5×5, 40 mm squares** (~200 mm board, DICT to match pan-tilt). Square edge length to be caliper-confirmed before first run. |
| Data collection | **calib_web-style:** author/preview poses in a browser tool, run, diff, promote. |
| Anchor frame | **Color optical frame** (`xarm_camera_color_optical_frame`). Detect ChArUco in RGB; depth tasks ride the RealSense factory color↔depth extrinsic. |
| Verification | **Held-out RMSE gate + live physical test.** |
| Solver | **Method B:** linear `calibrateHandEye` seed → nonlinear bundle-adjust (with multi-method seed selection as a robustness net). |

## 3. Why a stale / forward-translated mount is fine

Method B needs **no prior** on the camera position, by construction:

- The linear seed (`cv2.calibrateHandEye`) is closed-form. Its only inputs per pose
  are `A_i = T_base→eef` (pure **arm FK**, camera-independent) and
  `B_i = T_color_optical→board` (**PnP through the real camera**). Neither reads the
  URDF mount.
- The bundle-adjust refine is seeded by `calibrateHandEye`; its board-in-base
  initial guess is derived from data (`T_base→eef · T_eef→cam_seed · T_cam→board_obs`),
  not from the URDF.
- A pure translation is the *benign* case — accuracy is driven by **rotation
  diversity** in the pose set, not by the offset magnitude.
- **Internal** camera extrinsics (`camera_link→{color,depth,ir}_optical`) are
  unchanged by translating the whole camera, so the §9 compose-out to the URDF mount
  joint still holds — only `link_eef→xarm_camera_link` moved.

## 4. Constraints

- **Rigid but flexy mount.** The camera is re-secured, but its supporting structure
  **rings/settles for ~1–2 s after each arm move.** A sample captured mid-ring sees a
  transiently displaced camera and corrupts `T_eef→cam`. → §7 settle handling is
  mandatory, not optional.
- **Board fixed for the entire session.** If it shifts, the eye-in-hand invariant
  breaks and the session must restart.
- **Compute budget / dep direction.** Reuse the existing ROS-free calibration core;
  do not pull `tk25_basic` into a dependency on `tk26_vision`.
- **Frame names.** Arm base `link_base`; flange `link_eef`; camera mount
  `xarm_camera_link`; RGB optical `xarm_camera_color_optical_frame`.

## 5. Non-goals / scope

- Not calibrating intrinsics (use the RealSense factory intrinsics; they are stable
  and not the error source here). The gate still *reports* reprojection so a bad
  intrinsic would surface.
- Not re-calibrating the color↔depth extrinsic (factory value trusted, per the
  "anchor to color" decision). The optional depth cross-check (§8) can flag gross
  factory error.
- Not touching the pan-tilt head calibration.
- This spec covers design only; implementation follows a separate plan.

## 6. Architecture & package layout

New package **`src/tk26_vision/src/handeye_calib/`**, importing the existing
`pan_tilt` calibration core as a **library** (no duplication):

Reused from `src/tk26_vision/src/pan_tilt/pan_tilt/`:
- `calibration/aruco_detect.py` — ChArUco detect + PnP (IPPE disambiguation,
  consensus voting, reprojection gating).
- `calibration/optimize.py` — `solve_handeye()` (`cv2.calibrateHandEye`), scipy
  `least_squares` infrastructure to extend for bundle-adjust.
- `calibration/utils.py` — SE(3) / optical↔body / residual helpers.
- `calibration/safety.py` — `SafetyEnvelope` (z-floor + mast-cylinder exclusion).
- `calibration/charuco_generate.py` — board presets / PDF-PNG-JSON generation.
- `calib_web.py` — reference pattern for the web tool (live overlay, waypoint
  authoring, subprocess runner, diff/promote, atomic write + `.old-<ts>` backup).

New components in `handeye_calib`:
- `handeye_model.py` — eye-in-hand sample dataclass, board-in-base parameterization,
  reprojection residual for bundle-adjust.
- `handeye_solve.py` — multi-method seed selection + bundle-adjust refine + held-out
  split + metrics.
- `handeye_collect.py` — ROS node: drive the arm to authored poses, settle-gate,
  capture color + `link_base→link_eef` TF, detect, accept/reject.
- `handeye_web.py` — calib_web-style browser tool (author/preview/run/verify/promote).
- `apply_handeye.py` — write `hand_eye.yaml`; optional atomic URDF mount-joint patch.

Motion uses `tinker_arm_msgs` `JointMove` / `CartesianMove` actions
(`src/tk25_manipulation/src/tinker_arm_msgs/action/`). Output targets
`tinker_robot_config` (`src/tk25_basic/src/tinker_robot_config/`).

**Rejected alternative:** add a "handeye mode" to `calib_web`. Different DOF, flow,
and safety; a focused sibling keeps both tools single-purpose.

## 7. Data collection flow

Physical setup:
- Mount the 5×5/40 mm board **rigidly** (aluminum composite, not foam) on a stand at
  ~table height inside the reach envelope, tilted ~30° toward the camera. Fixed for
  the whole session.

Per session:
1. **Author pose set** in the browser (or load a shipped default), validated live
   against `SafetyEnvelope` (z-floor 0.25 m, mast-cylinder exclusion). Because the
   camera moved forward, the **default pose set is regenerated for the new framing**
   (standoff/FOV); the live overlay + board-area gate self-correct any remaining
   mismatch (author slightly larger standoff if the board overflows the frame).
2. **Move** to the waypoint via `JointMove`/`CartesianMove`; wait for the action to
   report done.
3. **Settle handling (mandatory):**
   - Fixed **settle delay** (default 2.0 s, configurable) after "done", then
   - **Wait-until-stable detection gate:** keep sampling color frames + detecting the
     board until the last *M* consecutive detections agree within tolerance
     (default: inter-frame rotation < 0.1°, translation < 0.3 mm), or a timeout
     (default 5 s) → pose rejected with reason "did not settle".
   This directly absorbs the variable 1–2 s mount ring instead of trusting a single
   fixed delay.
4. **Capture** once stable: K frames of color + the `link_base→link_eef` TF sampled
   at the capture instant (arm stationary, so no dynamic sync needed).
   Run consensus over the K frames → one averaged detection.
5. **Quality gate** per pose: ≥ N corners, reprojection RMS < 1.5 px, board-area
   fraction above a floor, and **≥ 30° rotation diversity vs already-accepted poses**
   (the #1 accuracy driver for AX=XB). Live overlay shows accept/reject + reason.
6. Target **~15–20 accepted poses** spanning standoff 0.3–0.6 m, varied
   azimuth/elevation, large wrist roll/pitch changes between poses, board landing in
   several image regions. **Auto-stop** on coverage + count satisfied.

## 8. Solver (Method B)

Per accepted pose `i`:
- `A_i = T_base→eef` from the captured TF (arm FK).
- `B_i = T_color_optical→board` from ChArUco PnP (optical→body via `utils`).

Steps:
1. **Multi-method seed:** run OpenCV `calibrateHandEye` with TSAI / PARK / HORAUD /
   ANDREFF / DANIILIDIS; pick the lowest AX=XB residual as `T_eef→cam_seed`.
2. **Board-in-base init:** `T_base→board_0 = A_i · T_eef→cam_seed · B_i⁻¹`
   (median over poses).
3. **Bundle-adjust** (`scipy.optimize.least_squares`, soft_l1): optimize
   `[T_eef→cam (6) , T_base→board (6)]` to minimize **summed ChArUco corner
   reprojection error** over all training poses. This is the accuracy lever that
   turns "few mm" into sub-mm / sub-0.5°.
4. Emit `T_eef→color_optical`. Compose the fixed internal URDF chain
   `T_color_optical→camera_link` to produce **`T_eef→xarm_camera_link`** for the
   URDF mount joint.

Optional **depth cross-check:** transform a detected board point into the depth
frame via the factory extrinsic and compare to the measured depth — flags gross
factory-extrinsic error without re-calibrating it.

## 9. Verification (acceptance gate)

**Held-out statistical gate** (~80/20 train/test split of accepted poses):
- Mean reprojection RMS (px) on held-out poses.
- Predicted-vs-observed board pose error (mm / deg) on held-out poses.
- **Thresholds (pan-tilt parity):** reproj < 1.5 px, trans < 3 mm, rot < 0.5° →
  **PASS**; within 2× → **WARN**; else **FAIL**. Banner + numeric readout.

**Live physical test** (in `handeye_web`):
- Estimate board-in-base from the solution; jog the arm to **fresh** poses and
  overlay **predicted** board corners on the live image. Operator watches predicted
  vs actual corners track within a few px across the workspace. Live pixel-error
  readout. This is the human-trustable "it's actually right" proof.

**Optional metric touch test** (off by default, collision-aware): marker at a known
board cell; command the TCP toward it; measure residual with a ruler.

## 10. Storage & integration

Primary output — populate the existing (currently empty) schema at
`src/tk25_basic/src/tinker_robot_config/robots/<robot>/hand_eye.yaml`:

```yaml
hand_eye:
  reference_frame: link_eef
  camera_frame: xarm_camera_link
  arm_to_camera_xyz: "<x> <y> <z>"        # meters, link_eef -> xarm_camera_link
  arm_to_camera_rpy: "<r> <p> <yaw>"      # radians
  # traceability
  color_optical_xyz: "<x> <y> <z>"        # raw solved link_eef -> color_optical
  color_optical_rpy: "<r> <p> <yaw>"
  calibration_date: "2026-06-15"
  calibration_method: "calibrateHandEye+BA"
  board: { type: charuco, squares: "5x5", square_len_m: 0.040 }
  num_poses: <n>
  heldout_trans_rmse_m: <v>
  heldout_rot_rmse_rad: <v>
  heldout_reproj_px: <v>
```

Optional **atomic URDF patch** of the camera mount-joint origin in
`realsense_d435i.urdf.xacro` (or its include site), behind a **diff-preview +
explicit "promote"**, writing a `.old-<ISO8601>` backup — mirroring
`apply_to_urdf.py` and the workspace's UI-first / preview-before-write /
persist-to-source convention. Promotion defaults to the **source tree**, not the
install tree, so a rebuild does not clobber it.

Package **README + append-only changelog**, updated in the same commit as code.

## 11. Acceptance gates (summary)

| Gate | Threshold |
|---|---|
| Per-frame detection | ≥ N corners, reproj < 1.5 px |
| Pose stability (settle) | last M frames agree: rot < 0.1°, trans < 0.3 mm |
| Pose diversity | ≥ 30° rotation vs accepted set |
| Pose count | ~15–20 accepted |
| Held-out translation | < 3 mm (PASS), < 6 mm (WARN) |
| Held-out rotation | < 0.5° (PASS), < 1° (WARN) |
| Held-out reprojection | < 1.5 px (PASS) |
| Physical overlay | predicted vs actual corners within a few px across workspace |

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Camera mount rings 1–2 s after a move → corrupted sample | Settle delay + wait-until-stable detection gate (§7.3) |
| Board moves mid-session → invariant broken | Bundle-adjust residual spike flags it; restart guidance |
| Poor rotation diversity → degenerate AX=XB | Live ≥30° diversity gate refuses redundant poses |
| RGB motion blur | Capture only when stationary + settled |
| Stale URDF mount value (forward translation) | Solver is prior-free (§3); default pose set regenerated |
| Frame confusion (`link_base` vs `base_link`) | FK uses arm base `link_base`; documented |
| IPPE glancing-view ambiguity | Reuse consensus voter; prefer fronto-parallel-ish views |
| Reach/collision with forward-protruding camera | Operator watches during authoring; envelope is flange-based |

## 13. Open items (confirm before first run)

- Caliper-measure the board square edge (40 mm assumed) and the ArUco dictionary —
  must match the printed board exactly.
- Confirm RealSense color stream name/topic and that the IR projector state does not
  contaminate the RGB ChArUco detection (RGB is unaffected; noted for completeness).
- Decide default K (frames/pose), N (min corners), and the exact settle tolerances on
  the bench during T1 bring-up.
