# vision_bringup

Composed launch files for the tk26 vision stack, split into a sensor/driver
layer and a BT-facing perception layer. The perception nodes are selected by
auditing what the behavior_tree production tasks actually call.

Full rationale (node selection, FFS/SHM fencing, cross-launch contract) lives in
[`docs/vision-bringup-design.md`](docs/vision-bringup-design.md). **All
launch-related documentation belongs under `docs/`.**

## Quick start

```bash
source install/setup.zsh   # re-source after building; this is a package

# 1) sensor layer: pan-tilt + Orbbec + FoundationStereo (streaming)
ros2 launch vision_bringup vision_driver.launch.py

# 2) perception layer: always-on core (generalist + door) only
ros2 launch vision_bringup vision_bringup.launch.py

# opt into one task (flags default OFF):
ros2 launch vision_bringup vision_bringup.launch.py enable_hri:=true
ros2 launch vision_bringup vision_bringup.launch.py enable_gpsr:=true
ros2 launch vision_bringup vision_bringup.launch.py enable_restaurant:=true
```

`enable_hri` covers **HRI + Follow** (one task). `enable_pick_place` gates no
extra node — its vision deps are the always-on core.

## What comes up

| Launch / flag | Nodes |
|---|---|
| `vision_driver` | pan-tilt, Orbbec, FoundationStereo (streaming, non-aligned, under SHM) |
| `vision_bringup` (bare) | generalist_node, door_detection |
| `enable_hri:=true` | + yolo_seg, person_track, waving, feature_recognition, feature_matching, seat_recommend_bbox, follow_head |
| `enable_gpsr:=true` | + yolo_seg, person_track, waving, feature_recognition, get_image |
| `enable_restaurant:=true` | + waving, follow_head |

## Build

```bash
WS_ROOT=/home/tinker/tk25_ws ./src/tk26_vision/scripts/build.sh \
    --packages-select vision_bringup
```

`scripts/build.sh` defaults `WS_ROOT` to the tk26_vision repo root; pass
`WS_ROOT=/home/tinker/tk25_ws` to install into the live workspace tree your
shell sources.

## Notes

- The kimi_api nodes need `OPENROUTER_API_KEY` / `DASHSCOPE_API_KEY` in a
  workspace-root `.env`, else they raise at init. Launch from the workspace root.
- FoundationStereo's streamed depth is non-empty only when the manipulation
  launch (which owns the RealSense IR pair) is up. Use `enable_ffs:=false` for
  vision-only bench runs.
- Alongside grasp_bringup, pass `launch_robot_state_publisher:=false` to the
  driver so only the xArm RSP owns `/robot_description`.

## Changelog

### 0.2.0 — 2026-06-23
- Rewrote both launches around a BT-driven node selection (tasks HRI+Follow,
  GPSR, Restaurant, PickAndPlace).
- `vision_driver`: removed RealSense (manipulation owns `xarm_camera`); moved
  FoundationStereo here in streaming mode. FFS runs under the blanket SHM
  profile (its RealSense IR pair needs the 20 MB segment; the earlier
  "SHM corrupts voxels" fencing was experimentally refuted and removed).
- `vision_bringup`: always-on core (generalist + door) + per-task flags
  (default OFF); dropped 9 unused nodes; OR-gated shared nodes.
- Trimmed `package.xml` exec_depends (`realsense2_camera`, `monocular_depth`).
- Added `docs/vision-bringup-design.md` as the canonical launch-docs home.

### 0.1.0
- Initial package: driver + perception launches with subsystem-level enable_*
  groups.
