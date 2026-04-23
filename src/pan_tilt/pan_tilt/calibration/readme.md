# Pan-tilt head camera extrinsic calibration

End-to-end walkthrough for calibrating the transform chain from `base_link` through the pan-tilt head to the Orbbec camera's `camera_link` frame, using the xArm's forward kinematics as ground truth and a ChArUco board rigidly attached to the xArm end-effector as the observation target.

## What this calibrates

```
base_link  ──T_A (trans only)──►  pan_axis
                              │
                              └─► [pan rot] ─ L_pan ─► tilt_axis
                                                    │
                                                    └─► [tilt rot] ──T_B──► camera_link
```

Fitted parameters (13 DOF baseline):

| Block | DOF | Meaning |
|---|---|---|
| `T_A` translation | 3 | base_link → pan axis (rotation assumed identity) |
| `T_B` translation | 3 | tilt-end → camera_link |
| `T_B` rotation | 3 | mounted camera twist — this robot has ~90° about X |
| `T_ee_marker` | 6 | xArm EE flange → ChArUco origin (unknown mount on the EE) |
| `θ_t_offset` | 1 | firmware tilt zero → physical tilt = 0 (default −π/4 because firmware zero parks the camera 45° down) |

`θ_p_offset` (pan bias) is opt-in via `--fit-pan-offset`.

## Hardware prerequisites

- Working pan-tilt stack: `ros2 launch pan_tilt pan_tilt.launch.py` bringing up controller + state publisher + `robot_state_publisher`.
- xArm with `set_servo_angle` service reachable at `/xarm/set_servo_angle`.
- Orbbec camera streaming at ≥ 30 Hz on `/camera/color/image_raw` + `/camera/color/camera_info` (see `CAMERA_BRINGUP.md`).
- A flat surface to mount a ChArUco board rigidly on the xArm EE flange (3 mm aluminium composite recommended — foamcore will warp and ruin the calibration).

## Step-by-step walkthrough

### 1. Print a ChArUco board

Two pre-generated boards live in `pan_tilt/calibration/boards/` (also installed under `share/pan_tilt/calibration/boards/`). Pick the one that fits your EE mount:

| Preset  | Board size | Squares | Square / marker | Best at | A4 margins |
|---|---|---|---|---|---|
| `default` | 200 × 280 mm | 5 × 7 | 40 / 30 mm | 0.5 – 1.5 m range | ~5 / ~8 mm (tight) |
| `compact` | 100 × 100 mm | 5 × 5 | 20 / 15 mm | EE mounts limited to ~10 × 10 cm | comfortable |

Both PDFs are sized for A4 at exact physical scale. Print at **100%** (no "fit to page" or "scale to fit"). Verify with calipers — if the squares don't measure exactly the spec'd size, re-print with the printer's scaling adjusted.

If you need a non-standard size, regenerate:

```bash
source src/tk26_vision/.venv-vision-main/bin/activate
# Named preset
python -m pan_tilt.calibration.charuco_generate --preset compact --out ~/calib/board
# Or fully custom (e.g. 35mm squares for more A4 margin)
python -m pan_tilt.calibration.charuco_generate \
    --squares-x 5 --squares-y 7 \
    --square-len 0.035 --marker-len 0.026 \
    --out ~/calib/board
```

Each invocation produces:
- `<out>.pdf`  — A4 page, board centered at exact scale.
- `<out>.png`  — source image (300 DPI).
- `<out>.json` — machine-readable board spec. Reference this from `calibration.yaml`.

**Whichever board you use, update `calibration.yaml`'s `board:` section to match its dimensions** — the collector instantiates the detector from those values.

### 2. Mount the board and verify streams

1. Rigidly bolt the printed board to a flat carrier (3 mm aluminium composite).
2. Attach the carrier to the xArm EE flange. The exact attachment geometry doesn't matter — the calibration solves for `T_ee_marker` — but the attachment must be **rigid**. Any play between captures ruins the hand-eye.
3. Sanity-check that detection works:
   ```bash
   ros2 topic hz /camera/color/image_raw   # expect ~30 Hz
   ros2 run tf2_ros tf2_echo base_link link_eef   # xArm FK is publishing
   ros2 topic echo /pan_tilt_controller/state --once   # feedback_ok: true
   ```
4. Point the camera at the board and verify that the `aruco_detect` module sees it:
   ```bash
   python -c "
   import cv2, numpy as np
   from pan_tilt.calibration.aruco_detect import detect_pose, build_board
   img = cv2.imread('your_test_capture.png')
   K = np.array([[...]]).reshape(3,3); D = np.zeros(5)  # your intrinsics
   print(detect_pose(img, K, D, board=build_board()))
   "
   ```

### 3. Record xArm waypoints

The collector drives the xArm to pre-authored joint-angle waypoints (no MoveIt, no collision checking). You must pre-validate these in RViz.

Two lists are needed in `calibration.yaml`:

- **`phase1_waypoints`** — **12–15 poses**, pan-tilt frozen at `(0, 0)`, xArm varying so the board orientation spans ≥ 60° between consecutive poses. Keep poses compact (elbow up, wrist close to base) to avoid gravity sag of the EE-mounted board.
- **`phase2_waypoints`** — **2–3 poses** where the board stays in the camera's field of view across the **full** `(pan × tilt)` sweep (defaults: pan ∈ {−60°, −30°, 0°, 30°, 60°}, tilt ∈ {−25°, −10°, 0°, 15°, 35°}).

Recording workflow:

```bash
# Terminal 1: bring up RViz with the full robot URDF loaded.
ros2 launch <your robot bringup>
# Terminal 2: hand-guide or jog the xArm to each pose, then read the joint state:
ros2 topic echo /xarm/joint_states --once | grep position
```

Paste each pose into the `.yaml` as a list of floats in radians, in the order the xArm driver expects (typically J1..J6 from base to tool).

**Safety envelope.** The collector enforces a software Z-floor (default 0.25 m) and a cylindrical exclusion around the pan-tilt mast (default 12 cm radius around `(-0.275, -0.013)` in base_link, up to 1.70 m). If any of your recorded waypoints violates this, the collector will log an error and skip the sample. Adjust `safety:` in the config if your geometry differs, but consider the values load-bearing — they exist so a flipped joint angle doesn't drive the board into the mast.

### 4. Edit the collector config

```bash
$EDITOR $(ros2 pkg prefix pan_tilt)/share/pan_tilt/config/calibration.yaml
# ...or edit the source at src/pan_tilt/config/calibration.yaml and rebuild.
```

Fill in `phase1_waypoints`, `phase2_waypoints`, `sanity_xarm_angles_rad`. Check that topic names, the `xarm_service`, and frame names (`base_link`, `link_eef`) match your robot. If you changed the ChArUco board, update the `board:` section.

### 5. (Optional) Calibrate Orbbec intrinsics

Factory intrinsics are usually within 0.5 px; run this check only if Phase-1 residuals come back > 3 mm. Capture ~20 board shots from varied distances/angles (handheld is fine — the xArm is not involved here). Save as PNGs in a directory, then:

```bash
python -m pan_tilt.calibration.run_calibration intrinsic \
    ~/calib/intrinsic_shots \
    --board ~/calib/charuco_5x7.json \
    --out ~/calib_out
```

Gate: `rms_px < 0.5`. If it fails, recapture with the board filling more of the frame.

### 6. Launch the collection node

In one terminal, bring up the robot (camera + pan-tilt + xArm driver). In another:

```bash
source /home/tinker/tk25_ws/install/setup.bash
ros2 run pan_tilt calibrate_collect --ros-args \
    -p config:=$(ros2 pkg prefix pan_tilt)/share/pan_tilt/config/calibration.yaml \
    -p out_dir:=$HOME/calib_out \
    -p phase:=both
```

The node:
1. Waits for camera + pan-tilt streams (up to 15 s).
2. Captures a **sanity pose** before anything else (start-of-session reference).
3. Runs Phase 1: parks pan-tilt at `(0, 0)`, drives the xArm through each `phase1_waypoints` pose, averages 10 detections per pose.
4. Runs Phase 2: for each `phase2_waypoints` pose, sweeps the 5×5 pan/tilt grid with **overshoot-return** backlash mitigation.
5. Captures a sanity pose at the end.
6. Writes:
   - `phase1_handeye.json`
   - `phase2_chain.json`
   - `sanity.json`

Watch the log output — per-cell failures (tf skew, too few detections, envelope violation) print a WARN but the session continues. A run is worth keeping if Phase 1 ends with ≥ 10 samples and Phase 2 with ≥ 50.

To split the session (e.g. collect Phase 1, review, then collect Phase 2):

```bash
# First pass
ros2 run pan_tilt calibrate_collect --ros-args ... -p phase:=phase1
# Review + adjust waypoints if needed, then:
ros2 run pan_tilt calibrate_collect --ros-args ... -p phase:=phase2
```

### 7. Solve

```bash
cd $HOME/calib_out

# Phase 1: hand-eye solve.
python -m pan_tilt.calibration.run_calibration handeye \
    phase1_handeye.json --out .
# Prints: "Hand-eye trans RMSE: X mm  rot RMSE: Y deg"
# Gate: < 3 mm / 0.5 deg.

# Phase 2: chain fit (warm-starts T_B from Phase 1's Z_0).
python -m pan_tilt.calibration.run_calibration chain \
    phase2_chain.json --handeye handeye.json \
    --fit-pan-offset --verbose --out .
# Prints training + validation residuals.
# Gate: val_trans_rmse_m < 0.003, val_rot_rmse_rad < 0.007 (< 3 mm / 0.4 deg).

# (Optional) Phase 3: joint polish. Unlock T_B rotation now that Phase-1 data
# is in the mix to break the Y-rotation degeneracy.
python -m pan_tilt.calibration.run_calibration polish \
    phase1_handeye.json phase2_chain.json \
    --seed chain.json \
    --unlock-tb-rotation --fit-pan-offset --out .

# Check all gates in one go.
python -m pan_tilt.calibration.run_calibration validate .
```

Only run polish if Phase-2 residuals show *structured* (non-random) pattern — e.g., rotation error that grows with |tilt|. If Phase-2 residuals already meet the gate, polish adds marginal value and can find worse local minima.

### 8. Review the URDF patch

The solver writes `chain.json` (and optionally `polish.json`) with the fitted parameters. Generate a unified diff against the current URDF:

```bash
python -m pan_tilt.calibration.apply_to_urdf \
    --results chain.json \
    --xacro src/tk26_vision/src/pan_tilt/urdf/pan_tilt.urdf.xacro
```

This prints (but does not apply) a diff updating:
- `pan_joint` origin `xyz` → fitted `t_a` (rpy forced to `0 0 0`).
- `camera_mount_joint` origin `xyz` → fitted `t_b_trans`, `rpy` → fitted `t_b_rotvec` (converted to XYZ Euler), or `0 0 0` if you're using `chain.json` (which froze T_B rotation).

**The URDF is never auto-modified.** Sanity-check the diff — the new values should be within a cm and a few degrees of the current URDF — then apply manually:

```bash
python -m pan_tilt.calibration.apply_to_urdf \
    --results chain.json \
    --xacro src/tk26_vision/src/pan_tilt/urdf/pan_tilt.urdf.xacro \
    --out /tmp/patched.urdf.xacro

# Inspect...
diff src/tk26_vision/src/pan_tilt/urdf/pan_tilt.urdf.xacro /tmp/patched.urdf.xacro

# Apply if satisfied.
cp /tmp/patched.urdf.xacro src/tk26_vision/src/pan_tilt/urdf/pan_tilt.urdf.xacro
```

Rebuild: `./src/tk26_vision/scripts/build.sh --packages-select pan_tilt`, restart the stack.

**Important:** if you used `polish.json` (T_B rotation fit), don't mentally compare the new `rpy` to the *old* URDF's `rpy` — the old value was a known artifact from a different calibration convention. Compare against the physical mount direction instead.

### 9. Verify

1. Launch the updated stack: `ros2 launch pan_tilt pan_tilt.launch.py device:=/dev/ttyUSB0`.
2. Visual check in RViz: at `pan=0, tilt=0` the camera frustum should match what you see — tilted ~45° down on this robot.
3. Back-project test with the ChArUco still on the EE: pick an xArm pose that wasn't in the collection set, query the expected camera view of the board vs what's actually detected. Disagreement < 5 mm on-image is expected.
4. Follow-head sanity: `ros2 run pan_tilt follow_head` and have a person walk across the camera's FoV. No systematic lag or directional bias should be visible.

## Session-drift sanity

`sanity.json` contains two captures of the same pose (start and end of session). If they disagree by > 2 mm / 0.2°, something drifted mid-session — thermal, servo slip, mount loosening — and the fit is suspect. Re-run.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Phase 1 rejects many samples for "image-vs-state skew" | Camera and pan-tilt state are on different clocks or slow topics | Check `ros2 topic hz` on both; raise `sample_stamp_skew_max_ms` if both are slow but stable |
| Phase 1 rejects many samples for "too few detections" | Board too small in frame, or motion blur | Move xArm closer; lock RGB exposure < 10 ms; verify board isn't warped |
| Phase 1 hand-eye RMSE > 5 mm | Insufficient orientation diversity, or flexing mount | Recapture with ≥ 60° orientation change between consecutive poses; re-tighten board mount |
| Phase 2 chain RMSE much worse than Phase 1 | Backlash, servo zero drift, or tilt-range too narrow | Check `tilt_grid_deg` spans ≥ ±25°; ensure each grid cell uses overshoot-return (default); run a sanity bracket |
| Chain fit `theta_t_offset` differs from −π/4 by >> 1° | Zero-set (`T:502`) was not where you thought it was | This is normal and what `theta_t_offset` is for. Fitted value is authoritative |
| Chain fit `t_b_rotvec` norm ≈ 1.57 rad | Expected: the ~90° physical mount | Not a problem. The warm-start put it there on purpose |
| Polish residuals worse than chain residuals | Local minimum in the joint fit; T_B rotation unlock found an alias | Re-run polish with `--fit-pan-offset` off; or trust `chain.json` and skip polish |
| Envelope violation errors | A recorded xArm waypoint puts the EE inside the mast exclusion or below the floor | Re-record that waypoint in RViz; or widen `safety.z_floor_m` / `safety.mast_radius_m` if the envelope is too tight for your geometry |

## Running the synthetic regression tests

Before touching `optimize.py`, `pan_tilt_model.py`, or `utils.py`, run:

```bash
cd src/tk26_vision/src/pan_tilt
python -m pytest test/test_calibration.py -v
```

These 5 tests fabricate samples from known ground-truth parameter blocks (including one with a 90° T_B mount) and assert that the solvers recover the truth inside tolerance. If any test fails after an edit, the fit will fail on real data too.

## File layout

```
src/pan_tilt/
├── config/
│   └── calibration.yaml            # waypoints + grid + safety envelope + board spec
├── pan_tilt/
│   ├── calibrate_collect.py        # ROS2 node: drives pan-tilt + xArm, writes JSON
│   └── calibration/
│       ├── aruco_detect.py         # ChArUco detection + MAD averaging (no ROS)
│       ├── apply_to_urdf.py        # emits a unified diff from fitted params
│       ├── charuco_generate.py     # PDF/PNG generator (no external deps)
│       ├── optimize.py             # fit_chain / fit_joint / solve_handeye / warm_start
│       ├── pan_tilt_model.py       # forward_kinematics + PanTiltParams
│       ├── run_calibration.py      # CLI: intrinsic | handeye | chain | polish | validate
│       ├── utils.py                # SE(3) log residual, optical↔body preprocess, I/O
│       ├── data/
│       │   └── _legacy_measurements.json  # historical reference only
│       └── readme.md               # this file
└── test/
    └── test_calibration.py         # synthetic regressions (run before edits!)
```

## Sample JSON schema (for reference)

`phase1_handeye.json` and `phase2_chain.json` share the same format:

```json
{
  "samples": [
    {
      "theta_pan_rad": 0.0,
      "theta_tilt_rad": 0.0,
      "t_base_ee": {
        "translation": [0.12, -0.05, 1.08],
        "rotation":    [0.001, 0.707, 0.002, 0.707]
      },
      "t_cam_marker_body": {
        "translation": [-0.01, 0.02, 0.62],
        "rotation":    [0.005, 0.004, 0.012, 0.999]
      },
      "image_stamp_ns": 1745432812345678900,
      "state_stamp_ns": 1745432812347123456,
      "detection_quality": 24,
      "reprojection_rms_px": 0.31,
      "label": "phase1/7"
    }
  ]
}
```

All quaternions are `[x, y, z, w]`. `t_cam_marker_body` is already preprocessed to the body-convention camera frame (x-forward, y-left, z-up); the optical-frame rotation `(−π/2, 0, −π/2)` was applied at collection time.

## See also

- `src/tk26_vision/CLAUDE.md` § "Pan-tilt / head camera extrinsic calibration" — the calibration procedure entry point at the repo level.
- `src/pan_tilt/urdf/pan_tilt.urdf.xacro` — the URDF that gets patched in step 8.
- The plan document that prompted this implementation: `~/.claude-wjy-paid/plans/shimmying-fluttering-koala.md`.
