# pan_tilt

ROS 2 pan-tilt stack for `tk26_vision`.

This package was cleanly refactored away from the old monolithic
`pan_tilt/ctrl` runtime. The current stack separates serial control, ROS state
publication, and TF generation.

## Runtime Stack

- `ros2 run pan_tilt controller`
  Owns `/dev/ttyUSB*`, sends firmware commands, and publishes low-level state.
- `ros2 run pan_tilt state_publisher`
  Converts `PanTiltState` into a `JointState` stream on
  `/pan_tilt/joint_states` (private, to avoid clashing with the main-robot
  `/joint_states` aggregator; override via the `joint_state_topic` parameter).
- `ros2 run pan_tilt follow_head`
  YOLO-based head following. Publishes native `PanTiltCommand` messages and
  still exposes `/follow_head_service` and `/follow_head_action`.
- `ros2 launch pan_tilt pan_tilt.launch.py`
  Canonical bringup. Starts `controller`, `state_publisher`, and
  `robot_state_publisher`.

## Public Runtime Interfaces

### Low-level

- Topic: `/pan_tilt_controller/cmd`
  Type: `tinker_vision_msgs_26/msg/PanTiltCommand`
- Topic: `/pan_tilt_controller/state`
  Type: `tinker_vision_msgs_26/msg/PanTiltState`
- Topic: `/pan_tilt/joint_states`
  Produced by `state_publisher` from `/pan_tilt_controller/state`. Private
  so it does not collide with the main-robot `/joint_states` aggregator.
- Topic: `/pan_tilt/robot_description`
  Published by `robot_state_publisher` in the bringup launch. Private so it
  does not collide with the main-robot `/robot_description` latched by
  `grasp_bringup` / `xarm_description`.
- TF frames (global `/tf`, `/tf_static`): `base_link -> pan_link ->
  tilt_link -> head_camera_link`. `head_camera_link` was renamed away from
  `camera_link` specifically so that the TF tree does not conflict with the
  xArm URDF's `link_eef -> camera_link` edge when both bringups run
  together.
- Service: `/pan_tilt_controller/set_torque`
  Type: `tinker_vision_msgs_26/srv/SetTorque`
- Service: `/pan_tilt_controller/set_zero`
  Type: `tinker_vision_msgs_26/srv/SetZero`

### High-level

- Action: `/follow_head_action`
  Type: `tinker_vision_msgs_26/action/FollowHeadAction`
- Service: `/follow_head_service`
  Type: `tinker_vision_msgs_26/srv/FollowHead`

The high-level `follow_head` entrypoints were kept stable. The low-level
control surface was intentionally broken cleanly.

## Bringup

Canonical:

```bash
ros2 launch pan_tilt pan_tilt.launch.py device:=/dev/ttyUSB0
```

Manual low-level bringup (mirrors what `pan_tilt.launch.py` does, with
explicit topic remaps so it can coexist with `grasp_bringup`):

```bash
ros2 run pan_tilt controller --ros-args -p device:=/dev/ttyUSB0
ros2 run pan_tilt state_publisher --ros-args \
  -p joint_state_topic:=/pan_tilt/joint_states
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(xacro $(ros2 pkg prefix tinker_urdf)/share/tinker_urdf/src/pan_tilt_standalone.urdf.xacro)"
```

Running alongside `grasp_bringup` (combined xArm + pan-tilt): the MoveIt
pipeline in `grasp_bringup.launch.py` already publishes the merged
`mobile_manipulator` URDF (which now contains the pan-tilt chain), so
`pan_tilt.launch.py` must not start a second `robot_state_publisher`. Pass
`launch_robot_state_publisher:=false`:

```bash
ros2 launch mobile_bringup grasp_bringup.launch.py   # terminal 1
ros2 launch pan_tilt pan_tilt.launch.py device:=/dev/ttyUSB0 \
    launch_robot_state_publisher:=false              # terminal 2
```

Native command example:

```bash
ros2 topic pub --once /pan_tilt_controller/cmd \
  tinker_vision_msgs_26/msg/PanTiltCommand \
  '{mode: 1, pan_rad: 0.2, tilt_rad: -0.1, speed_raw: 0, accel_raw: 0}'
```

## Runtime Configuration

- Runtime parameters live in [config/pan_tilt.yaml](./config/pan_tilt.yaml).
- Runtime geometry lives in `tinker_urdf`:
  `src/tk25_basic/src/tinker_urdf/src/pan_tilt.urdf.xacro` is a
  `pan_tilt_macro` (parent / prefix / attach_xyz / attach_rpy /
  camera_mount_xyz / camera_mount_rpy);
  `pan_tilt_standalone.urdf.xacro` is the standalone wrapper this launch
  loads, and `tracer_mini_manipulator.urdf.xacro` includes the same macro
  with `parent="base_link"` for the combined `mobile_manipulator` URDF.
  The geometry lives in `tinker_urdf` so the robot-description package does
  not gain a runtime dependency on `pan_tilt`; `pan_tilt` depends on
  `tinker_urdf` instead.
- Per-robot URDF overrides come from
  `tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/urdf_overrides.yaml`
  (`attach_xyz`/`attach_rpy`/`camera_mount_xyz`/`camera_mount_rpy`).
  `pan_tilt.launch.py` includes `tinker_robot_config`'s
  `robot_description.launch.py` wrapper, which flattens that sub-tree
  into xacro `--mappings`. The macro defaults still apply for manual
  `xacro …` invocations or `tracer_mini_manipulator`. (P6.2)
- `config/specs.json` is retained as historical calibration/reference data only.
  The runtime stack does not load it anymore.

## Calibration

The pan-tilt / head-camera extrinsic calibration yaml lives in
`tinker_robot_config` under the per-robot tree:

```
src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/calibration.yaml
```

`calib_web` and `calibrate_collect` resolve this file by default via the
`tinker_robot_config` resolver (which keys off `$ROBOT_NAME`). Operators can
override with `-p config:=<path>` to point at a custom file (e.g. a pruned
sidecar produced by `calib_web`'s prune-apply endpoint).

```bash
# Default — uses $ROBOT_NAME to pick robots/<ROBOT_NAME>/pan_tilt/calibration.yaml
ROBOT_NAME=tinker2 ros2 run pan_tilt calibrate_web --ros-args -p bind:=127.0.0.1 -p port:=8765
ROBOT_NAME=tinker2 ros2 run pan_tilt calibrate_collect --ros-args -p phase:=both -p out_dir:=$PWD/calib_out

# Override
ros2 run pan_tilt calibrate_collect --ros-args -p config:=/path/to/custom.yaml -p phase:=both
```

`calib_web`'s write paths (`save_waypoints_to_config`,
`_overwrite_source_with_prune`) write back to the canonical source-tree
file under `tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/`,
not the install share — backup `*.yaml.old-<ts>` files land alongside the
source.

There is no in-tree `config/calibration.yaml` in this package anymore (it
was retired in P5a). Full calibration procedure docs live alongside the
code at [`pan_tilt/calibration/readme.md`](./pan_tilt/calibration/readme.md).

## Firmware Assumptions

The current controller implementation expects the firmware behavior documented
in [about_pantilt.md](./about_pantilt.md):

- serial device at `115200`
- streaming `{"T":1001,"X":...,"Y":...}` feedback
- motion command `{"T":133,...}`
- torque command `{"T":210,...}`
- zero-set command `{"T":502,...}`

Hardware assumptions from the firmware notes:

- pan motor ID: `2`
- tilt motor ID: `1`
- pan range: `-180` to `180`
- tilt range: `-30` to `90`

## Breaking Changes From The Clean Refactor

These are intentional breakages. If old instructions still mention them, those
instructions are stale.

- `ros2 run pan_tilt ctrl` was removed.
- `/pan_tilt_ctrl` and `/pan_tilt_ctrl_modify` were removed.
- `tinker_vision_msgs_26/msg/PanTiltCtrl` is no longer consumed by the
  `pan_tilt` runtime.
- TF is no longer broadcast directly by the serial driver.
  Use `pan_tilt.launch.py`, or run `controller` + `state_publisher` +
  `robot_state_publisher` together.
- `config/specs.json` is no longer a runtime dependency.
- The low-level API is now radians-first and ROS-native:
  `PanTiltCommand`, `PanTiltState`, `SetTorque`, `SetZero`.

## Places Still Easy To Misread

- [about_pantilt.md](./about_pantilt.md) is firmware/bench documentation, not
  runtime integration documentation.
- [draft_refractor_plan.md](./draft_refractor_plan.md) is reference-only and
  predates the final implementation. It still uses outdated names such as
  `pan_tilt_state_tf_node` and a separate `pan_tilt_msgs` package.
- `scripts/tests/*.sh` still default `SERVO_DEVICE` to `/dev/ttyUSB1`.
  Current hardware validation in this worktree was run on `/dev/ttyUSB0`, so
  export `SERVO_DEVICE=/dev/ttyUSB0` when using the test harness on this host.

## Hardware Verification

Verified in this worktree on `2026-04-23` against `/dev/ttyUSB0`:

- controller opens the real serial device and receives `T:1001` feedback
- `/pan_tilt_controller/state` reports `connected: true` and `feedback_ok: true`
- `/pan_tilt/joint_states` tracks the hardware state
- `base_link -> head_camera_link` resolves through the launch stack
- relative and absolute commands both move the hardware
- the launch stack now exits cleanly on `SIGINT`
