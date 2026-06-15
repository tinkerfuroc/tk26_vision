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

## Verify without hardware

The whole solver is provable on a laptop against synthetic ground truth:
```bash
python -m handeye_calib.synthetic     # prints recovered-X error + PASS
pytest test/                          # unit suite (source the workspace first for test_import)
```

## Acceptance gate

Held-out poses must clear pan-tilt parity: translation < 3 mm, rotation < 0.5°,
reprojection < 1.5 px (PASS; within 2× = WARN). Note the held-out **rotation** metric
compares against single-shot PnP, so it also reflects observation noise; on hardware
the 10-frame consensus voter reduces that. The live overlay is the human-trustable
check: predicted board corners should track the real corners within a few px across
the workspace.

## Changelog
- 0.2.0 (2026-06-15): math core (transforms/model/solver/gates), synthetic harness,
  collection node, calib_web-style web tool, yaml/URDF persistence.
- 0.1.0 (2026-06-15): package scaffold.
