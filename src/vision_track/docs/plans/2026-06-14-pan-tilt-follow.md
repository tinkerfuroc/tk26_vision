# Pan-Tilt Person-Follow (head centering) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the tracked person horizontally centered with the head pan-tilt —
tilt fixed at 40°, pan driven in ABSOLUTE mode from the tracker's per-frame bbox —
run inside `person_track_node`'s ~30 Hz loop (not the BT, which ticks too slowly).

**Architecture:** A pure, ROS-free `PanFollower` core encodes the control law
(horizontal angular error `θ = atan2(u − cx, fx)`; ABSOLUTE target =
`current_pan + sign·θ`, recomputed from live `PanTiltState` + live pixel error
every command so error never accumulates; error-vs-state deadband + anti-chatter
min-change + min command interval + EMA + clamp). Thin ROS glue in
`person_track_node` subscribes to `/pan_tilt_controller/state`, publishes
`PanTiltCommand` to `/pan_tilt_controller/cmd`, and calls the follower once per
loop iteration. The whole feature is param-gated by `enable_pan_tilt_follow`
(default **False** → zero overhead, no servo traffic). The follow bringup gains a
`pan_tilt:=true` arg that launches the servo and flips the tracker flag.

**Modes (per the agreed behavior):**
- **CENTER** — a bbox is visible this frame: pan toward `atan2(u − cx, fx)`.
- **HOLD** — normal lost / PASSIVE coast: no command (servo holds; tilt stays 40°).
- **RECENTER** — NEEDS_HELP (reacq_state == 2), no bbox: pan back to 0 (forward
  search pose, so the operator re-enters the forward view the wave-reseed expects).

**Tech Stack:** Python 3.10, rclpy, `tinker_vision_msgs_26.msg.{PanTiltCommand,
PanTiltState}` (radians; `PanTiltCommand.ABSOLUTE`), pan-tilt controller on
`/pan_tilt_controller/{cmd,state}`. Reference: `pan_tilt/pan_tilt/follow_head.py`.

**Pan sign (derived, not guessed):** `pan_sign = +1.0`, taken from `follow_head`'s
kinematics (`follow_head.py:1496-1516`): camera optical frame +x = right, the
person's horizontal offset is `pan_offset = atan2(x_cam, z_cam)` (= our
`atan2(u-cx, fx)`), and the absolute aim is `world_pan = cur_pan + pan_offset`
(the root→angles round-trip is identity). The URDF pan axis `0 0 -1` makes
positive pan "turn right", so a right-of-axis person (`θ>0`) → `target =
cur_pan + θ` → head turns right, **toward** them. Same servo + URDF as the tracker,
so `+1` is correct here too; it stays a param only so a different mount can flip it.

**Other hardware caveats (params; verify on the robot):**
- `fixed_tilt_deg = 40` is the **joint-frame** tilt (the controller's invert/trim
  maps to firmware). Per the calibration note (joint tilt 45° ≈ horizontal), 40° ≈
  slightly below horizontal. Adjust if the head pitches wrong.
- `pan_min_deg`/`pan_max_deg` must match the controller's mechanical limits.

---

## Task 1: `PanFollower` core control law (pure, unit-tested)

**Files:**
- Create: `src/tk26_vision/src/vision_track/vision_track/core/pan_follow.py`
- Test:   `src/tk26_vision/src/vision_track/test/test_pan_follow.py`
- Modify (changelog, same commit): `src/tk26_vision/src/vision_track/readme.md`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pan_follow.py
import math
import pytest
from vision_track.core.pan_follow import PanFollower


def _f(**kw):
    # Permissive throttle so single-call tests exercise the control law, not gating.
    base = dict(pan_sign=1.0, deadband_rad=math.radians(1.0),
                min_change_rad=0.0, min_interval_s=0.0, ema_alpha=1.0,
                pan_min_rad=math.radians(-90.0), pan_max_rad=math.radians(90.0))
    base.update(kw)
    return PanFollower(**base)


def test_center_person_right_of_axis_turns_head_toward_them():
    # Person to the right of the optical axis (u > cx): theta > 0. follow_head's
    # convention is world_pan = cur_pan + atan2(x_cam, z_cam) with +x=right and the
    # URDF pan axis "0 0 -1" (positive pan = turn right), i.e. pan_sign=+1 -> the
    # head turns RIGHT (target_pan > current_pan), TOWARD the person.
    foll = _f(pan_sign=1.0)
    fx, cx, u = 600.0, 320.0, 320.0 + 600.0  # atan2(600,600)=45deg
    out = foll.center(u=u, cx=cx, fx=fx, current_pan=0.0, now=1.0)
    assert out == pytest.approx(math.radians(45.0), abs=1e-6)


def test_center_is_absolute_no_accumulation():
    # Same pixel error from a different current_pan yields target = current_pan +
    # sign*theta — it tracks live state, it does NOT integrate.
    foll = _f(pan_sign=1.0)
    fx, cx, u = 600.0, 320.0, 320.0 + 600.0
    out = foll.center(u=u, cx=cx, fx=fx, current_pan=math.radians(10.0), now=1.0)
    assert out == pytest.approx(math.radians(10.0) + math.radians(45.0), abs=1e-6)


def test_center_within_deadband_holds():
    foll = _f(deadband_rad=math.radians(5.0))
    # u == cx -> theta 0 -> target == current_pan -> within deadband -> None.
    assert foll.center(u=320.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.0) is None


def test_center_requires_current_pan_for_absolute():
    foll = _f()
    assert foll.center(u=900.0, cx=320.0, fx=600.0, current_pan=None, now=1.0) is None


def test_center_clamps_to_limits():
    foll = _f(pan_sign=1.0, pan_max_rad=math.radians(30.0))
    # Big positive theta would exceed +30deg; clamp.
    out = foll.center(u=320.0 + 6000.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.0)
    assert out == pytest.approx(math.radians(30.0), abs=1e-6)


def test_recenter_targets_zero_for_needs_help():
    foll = _f(deadband_rad=math.radians(1.0))
    out = foll.recenter(current_pan=math.radians(40.0), now=1.0)
    assert out == pytest.approx(0.0, abs=1e-6)


def test_min_interval_throttles_commands():
    foll = _f(min_interval_s=1.0)
    a = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=10.0)
    assert a is not None                      # first command issues
    b = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=10.5)
    assert b is None                          # within 1s -> throttled
    c = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=11.1)
    assert c is not None                      # after the interval -> issues


def test_min_change_suppresses_micro_commands():
    foll = _f(min_change_rad=math.radians(5.0), min_interval_s=0.0)
    first = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.0)
    assert first is not None
    # A target within 5deg of the last command is suppressed.
    again = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=2.0)
    assert again is None


def test_reset_clears_throttle_and_ema():
    foll = _f(min_interval_s=100.0)
    foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.0)
    foll.reset()
    # After reset the interval clock no longer blocks the next command.
    assert foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.5) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd src/tk26_vision && source .venv-vision-main/bin/activate && cd src/vision_track && python -m pytest test/test_pan_follow.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision_track.core.pan_follow'`.

- [ ] **Step 3: Implement `PanFollower`**

```python
# vision_track/core/pan_follow.py
"""Pure horizontal pan-to-center controller for the head pan-tilt.

ROS-free + numpy-free so the control law is unit-testable with synthetic inputs.
Keeps the tracked person centered horizontally by commanding the pan servo in
ABSOLUTE mode: target = current_pan + pan_sign * atan2(u - cx, fx). Because the
target is recomputed from the live PanTiltState pan AND the live pixel error on
every command, error never accumulates (the explicit reason ABSOLUTE is used
instead of RELATIVE). Tilt is held fixed by the caller; this class owns pan only.

Modes:
  - center(...)   CENTER:   a bbox is visible -> pan toward atan2(u - cx, fx).
  - recenter(...) RECENTER: NEEDS_HELP, no bbox -> pan back to 0 (forward pose).
  (HOLD = the caller simply does not call either -> no command.)

Every command passes a common gate: EMA-smooth the target, clamp to limits, then
suppress it if (a) the servo already points within deadband_rad of it, (b) it is
within min_change_rad of the last command (anti-chatter), or (c) fewer than
min_interval_s have passed since the last command (rate-limit the 30 Hz loop).
"""
from __future__ import annotations

import math
from typing import Optional


class PanFollower:
    def __init__(
        self,
        *,
        pan_sign: float = 1.0,   # +1 matches follow_head (cur_pan + atan2(x_cam,z_cam))
        deadband_rad: float = math.radians(3.0),
        min_change_rad: float = math.radians(1.0),
        min_interval_s: float = 0.15,
        ema_alpha: float = 0.5,
        pan_min_rad: float = math.radians(-90.0),
        pan_max_rad: float = math.radians(90.0),
    ) -> None:
        self.pan_sign = float(pan_sign)
        self.deadband_rad = float(deadband_rad)
        self.min_change_rad = float(min_change_rad)
        self.min_interval_s = float(min_interval_s)
        self.ema_alpha = float(ema_alpha)
        self.pan_min_rad = float(pan_min_rad)
        self.pan_max_rad = float(pan_max_rad)
        self._ema_target: Optional[float] = None
        self._last_cmd_pan: Optional[float] = None
        self._last_cmd_t: float = -1e9

    def reset(self) -> None:
        """Drop EMA + throttle state (call on goal start/end)."""
        self._ema_target = None
        self._last_cmd_pan = None
        self._last_cmd_t = -1e9

    def center(self, u, cx, fx, current_pan, now) -> Optional[float]:
        """CENTER: pan toward the bbox center-x. None if it must hold."""
        if current_pan is None or fx is None or float(fx) == 0.0:
            return None
        theta = math.atan2(float(u) - float(cx), float(fx))
        raw_target = float(current_pan) + self.pan_sign * theta
        return self._gate(raw_target, current_pan, now)

    def recenter(self, current_pan, now) -> Optional[float]:
        """RECENTER: pan back to 0 (NEEDS_HELP forward search pose)."""
        return self._gate(0.0, current_pan, now)

    def _clamp(self, p: float) -> float:
        return max(self.pan_min_rad, min(self.pan_max_rad, p))

    def _gate(self, raw_target, current_pan, now) -> Optional[float]:
        if self._ema_target is None:
            self._ema_target = raw_target
        else:
            a = self.ema_alpha
            self._ema_target = a * raw_target + (1.0 - a) * self._ema_target
        target = self._clamp(self._ema_target)

        if current_pan is not None and abs(target - float(current_pan)) < self.deadband_rad:
            return None
        if (self._last_cmd_pan is not None
                and abs(target - self._last_cmd_pan) < self.min_change_rad):
            return None
        if (now - self._last_cmd_t) < self.min_interval_s:
            return None

        self._last_cmd_pan = target
        self._last_cmd_t = now
        return target
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest test/test_pan_follow.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Add the changelog entry** to `src/vision_track/readme.md` (top of
`## Changelog`):

```markdown
- **2026-06-14** — pan-tilt person-follow core (`core/pan_follow.py`): pure
  horizontal pan-to-center control law — ABSOLUTE target `current_pan +
  sign*atan2(u-cx, fx)` (recomputed from live state + live pixel error, so error
  never accumulates), with error-vs-state deadband, anti-chatter min-change, min
  command interval, EMA + clamp. CENTER/RECENTER(NEEDS_HELP)/HOLD modes. Wired
  into the tracker in a follow-up commit. Tests: `test/test_pan_follow.py` (9).
```

- [ ] **Step 6: Commit**

```bash
git -C src/tk26_vision add \
  src/vision_track/vision_track/core/pan_follow.py \
  src/vision_track/test/test_pan_follow.py \
  src/vision_track/readme.md
git -C src/tk26_vision commit -m "feat(vision_track): PanFollower pan-to-center control core" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire `PanFollower` into `person_track_node`

**Files:**
- Modify: `src/tk26_vision/src/vision_track/vision_track/person_track_node.py`
- Modify (changelog, same commit): `src/tk26_vision/src/vision_track/readme.md`
- Modify: `src/tk26_vision/src/vision_track/config/default.yaml` (param defaults)

Integration points in `person_track_node.py` (verify exact lines on read — the
file moves under concurrent edits):
- imports: `from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState`,
  `from vision_track.core.pan_follow import PanFollower`, `import math`,
  `from threading import Lock` (a lock already exists; reuse the import).
- `__init__` param declarations (near the other `declare_parameter` block ~line 405).
- `_init_*` / setup: build the follower + pub/sub when enabled.
- camera intrinsics: `self.camera_intrinsic.k[0]` = fx, `.k[2]` = cx (CameraInfo).
- the tracking loop (~line 1112–1124): reset `_last_target_u` each iteration, then
  call `self._pan_follow_tick()` after `_publish_debug_outputs(...)`.
- `_handle_tracked_frame` (`track_result` not None): set `_last_target_u` to the
  bbox center-x.
- goal cleanup (~line 1860, where `_help_latched`/`_last_reacq_state` reset): call
  `self._pan_follower.reset()` and clear `_pan_tilt_initialized`.

- [ ] **Step 1: Declare the parameters** (in `__init__`, with the other params)

```python
# --- Pan-tilt head follow (default OFF; needs pan_tilt controller running) ---
# When enabled the tracker keeps the locked person horizontally centered by
# commanding the head pan servo in ABSOLUTE mode (tilt held at fixed_tilt_deg).
# Runs in this ~30 Hz loop, NOT the BT (which ticks too slowly to center a head).
self.declare_parameter('enable_pan_tilt_follow', False)
self.declare_parameter('pan_tilt_command_topic', '/pan_tilt_controller/cmd')
self.declare_parameter('pan_tilt_state_topic', '/pan_tilt_controller/state')
self.declare_parameter('fixed_tilt_deg', 40.0)
# pan_sign: +1 derived from follow_head (world_pan = cur_pan + atan2(x_cam,z_cam),
# +x=right, URDF pan axis "0 0 -1" => positive pan turns right toward a right-side
# person). Param only so a different mount can flip it.
self.declare_parameter('pan_sign', 1.0)
self.declare_parameter('pan_deadband_deg', 3.0)
self.declare_parameter('pan_min_command_change_deg', 1.0)
self.declare_parameter('pan_min_command_interval_sec', 0.15)
self.declare_parameter('pan_ema_alpha', 0.5)
self.declare_parameter('pan_min_deg', -90.0)
self.declare_parameter('pan_max_deg', 90.0)
self.declare_parameter('pan_command_speed_raw', 0)   # 0 -> controller default
self.declare_parameter('pan_command_accel_raw', 0)
```

- [ ] **Step 2: Build the follower + pub/sub** (in setup, after the tracker/pubs
exist; gate on the flag so a disabled tracker creates nothing)

```python
self.enable_pan_tilt_follow = bool(
    self.get_parameter('enable_pan_tilt_follow').value)
self._current_pan_rad = None
self._pan_state_lock = Lock()
self._pan_tilt_initialized = False
self._last_target_u = None
self._pan_follower = None
self._pan_cmd_pub = None
self._fixed_tilt_rad = math.radians(
    float(self.get_parameter('fixed_tilt_deg').value))
self._pan_cmd_speed = int(self.get_parameter('pan_command_speed_raw').value)
self._pan_cmd_accel = int(self.get_parameter('pan_command_accel_raw').value)
if self.enable_pan_tilt_follow:
    self._pan_follower = PanFollower(
        pan_sign=float(self.get_parameter('pan_sign').value),
        deadband_rad=math.radians(float(self.get_parameter('pan_deadband_deg').value)),
        min_change_rad=math.radians(
            float(self.get_parameter('pan_min_command_change_deg').value)),
        min_interval_s=float(self.get_parameter('pan_min_command_interval_sec').value),
        ema_alpha=float(self.get_parameter('pan_ema_alpha').value),
        pan_min_rad=math.radians(float(self.get_parameter('pan_min_deg').value)),
        pan_max_rad=math.radians(float(self.get_parameter('pan_max_deg').value)),
    )
    cmd_topic = self.get_parameter('pan_tilt_command_topic').value
    state_topic = self.get_parameter('pan_tilt_state_topic').value
    self._pan_cmd_pub = self.create_publisher(PanTiltCommand, cmd_topic, 1)
    self.create_subscription(
        PanTiltState, state_topic, self._pan_state_cb, 10)
    self.get_logger().info(
        f'Pan-tilt follow ENABLED: cmd={cmd_topic} state={state_topic} '
        f'tilt={float(self.get_parameter("fixed_tilt_deg").value)} deg')
```

- [ ] **Step 3: Add the callbacks + per-frame tick + publish helper**

```python
def _pan_state_cb(self, msg: PanTiltState):
    with self._pan_state_lock:
        self._current_pan_rad = float(msg.pan_rad)

def _publish_pan_tilt(self, pan_rad: float):
    cmd = PanTiltCommand()
    cmd.header.stamp = self.get_clock().now().to_msg()
    cmd.mode = PanTiltCommand.ABSOLUTE          # ABSOLUTE only — no accumulation
    cmd.pan_rad = float(pan_rad)
    cmd.tilt_rad = float(self._fixed_tilt_rad)  # tilt held at fixed_tilt_deg
    cmd.speed_raw = int(self._pan_cmd_speed)
    cmd.accel_raw = int(self._pan_cmd_accel)
    self._pan_cmd_pub.publish(cmd)

def _pan_follow_tick(self):
    """Center the head on the tracked person; called once per loop iteration."""
    if not self.enable_pan_tilt_follow or self._pan_cmd_pub is None:
        return
    with self._pan_state_lock:
        current_pan = self._current_pan_rad
    if current_pan is None:
        # No PanTiltState yet -> can't do ABSOLUTE centering; wait (state_publisher
        # publishes continuously once the controller is up).
        self.get_logger().warn('pan-tilt follow: awaiting PanTiltState',
                               throttle_duration_sec=5.0)
        return
    now = time.monotonic()
    # One-time: pitch the head to the fixed tilt (and hold current pan) so the
    # head reaches 40 deg even before the first centering command.
    if not self._pan_tilt_initialized:
        self._publish_pan_tilt(current_pan)
        self._pan_tilt_initialized = True
        return
    target = None
    if self._last_target_u is not None and self.camera_intrinsic is not None:
        fx = float(self.camera_intrinsic.k[0])
        cx = float(self.camera_intrinsic.k[2])
        target = self._pan_follower.center(
            self._last_target_u, cx, fx, current_pan, now)   # CENTER
    elif int(self._last_reacq_state) == 2:                   # REACQ_NEEDS_HELP
        target = self._pan_follower.recenter(current_pan, now)  # RECENTER -> 0
    # else: HOLD (normal lost / passive) -> no command.
    if target is not None:
        self._publish_pan_tilt(target)
```

- [ ] **Step 4: Feed the bbox + call the tick from the loop**

In `_handle_tracked_frame`, where `track_result.bbox` is in scope, set the
centre-x (bbox is `x1, y1, x2, y2`):

```python
if track_result.bbox is not None:
    self._last_target_u = 0.5 * (float(track_result.bbox[0]) + float(track_result.bbox[2]))
```

In the tracking loop, reset `_last_target_u` at the top of each iteration (just
before `self.tracker.in_needs_help = ...` / `self.tracker.update(...)`):

```python
self._last_target_u = None
```

and after `self._publish_debug_outputs(...)`:

```python
self._pan_follow_tick()
```

In goal cleanup (where `_help_latched`/`_last_reacq_state` reset), add:

```python
if self._pan_follower is not None:
    self._pan_follower.reset()
self._pan_tilt_initialized = False
```

- [ ] **Step 5: Mirror the defaults into `config/default.yaml`** (under
`person_track_node`/the tracker params block) so a yaml-driven bringup matches:

```yaml
    enable_pan_tilt_follow: false
    pan_tilt_command_topic: /pan_tilt_controller/cmd
    pan_tilt_state_topic: /pan_tilt_controller/state
    fixed_tilt_deg: 40.0
    pan_sign: 1.0
    pan_deadband_deg: 3.0
    pan_min_command_change_deg: 1.0
    pan_min_command_interval_sec: 0.15
    pan_ema_alpha: 0.5
    pan_min_deg: -90.0
    pan_max_deg: 90.0
    pan_command_speed_raw: 0
    pan_command_accel_raw: 0
```

- [ ] **Step 6: Build into the live install tree + import-smoke**

Run:
```bash
cd /home/tinker/tk25_ws && tkbuild tk26_vision --packages-select vision_track
source install/setup.bash
python -c "from vision_track.core.pan_follow import PanFollower; print('ok')"
python -m pytest src/tk26_vision/src/vision_track/test/ -q -k "pan_follow or needs_help_reacq or track_web_app"
```
Expected: build clean; import ok; pan_follow tests pass; no regressions in the
sampled suites. (`enable_pan_tilt_follow` defaults False, so an existing tracker
bringup is byte-for-byte unchanged — no pub/sub created.)

- [ ] **Step 7: Changelog entry** (top of `## Changelog`):

```markdown
- **2026-06-14** — tracker now drives the head pan-tilt to keep the locked person
  centered (`enable_pan_tilt_follow`, default OFF). Runs in the ~30 Hz tracking
  loop (the BT ticks too slowly for head centering): per locked frame it pans in
  ABSOLUTE mode toward `current_pan + sign*atan2(u-cx, fx)` from the live
  `/pan_tilt_controller/state`, tilt held at `fixed_tilt_deg` (40°). HOLD on
  normal loss, RECENTER to 0 in NEEDS_HELP. Bring up the servo via
  `ros2 launch pan_tilt pan_tilt.launch.py device:=/dev/ttyUSB0
  launch_robot_state_publisher:=false` and `-p enable_pan_tilt_follow:=true`.
  Hardware: flip `pan_sign` if the head turns away instead of toward the person.
```

- [ ] **Step 8: Commit**

```bash
git -C src/tk26_vision add \
  src/vision_track/vision_track/person_track_node.py \
  src/vision_track/config/default.yaml \
  src/vision_track/readme.md
git -C src/tk26_vision commit -m "feat(vision_track): drive head pan-tilt to center the tracked person" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Start the pan-tilt with the follow bringup

**Files:**
- Modify: `src/tk26_navigation/src/navigation_bringup/launch/follow_bringup.launch.py`

> **Concurrent-committer caution (tk26_navigation main):** another session commits
> to this repo in the same checkout. Stage ONLY this launch file (pathspec), commit
> new (never `--amend`/rebase), and verify HEAD before/after.

- [ ] **Step 1: Add a `pan_tilt` launch arg + include + tracker flag**

Add near the other `LaunchConfiguration`s:
```python
pan_tilt = LaunchConfiguration("pan_tilt")
pan_tilt_device = LaunchConfiguration("pan_tilt_device")
```

Include the pan-tilt launch (real robot only — sim has no servo) and pass the
tracker the enable flag. The real tracker becomes:
```python
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

real_tracker = Node(
    package="vision_track", executable="person_track_server", output="screen",
    condition=UnlessCondition(sim),
    parameters=[{"enable_pan_tilt_follow": ParameterValue(
        PythonExpression(["'", pan_tilt, "' == 'true'"]), value_type=bool)}])

pan_tilt_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory("pan_tilt"), "launch", "pan_tilt.launch.py")),
    launch_arguments={
        "device": pan_tilt_device,
        # The main robot RSP owns the pan/tilt TF; don't start a second one.
        "launch_robot_state_publisher": "false",
    }.items(),
    condition=IfCondition(PythonExpression(
        ["'", sim, "' == 'false' and '", pan_tilt, "' == 'true'"])))
```

Declare the args and add `pan_tilt_launch` to the `LaunchDescription` list:
```python
DeclareLaunchArgument("pan_tilt", default_value="false",
                      description="start the head pan-tilt (real robot only) and "
                                  "have the tracker center the person with it"),
DeclareLaunchArgument("pan_tilt_device", default_value="/dev/ttyUSB0"),
...
follow_server, real_tracker, waving_server, pan_tilt_launch, dummy_person,
sim_tracker, seed_amcl, bt, observer, observer_teardown,
```

- [ ] **Step 2: Verify the launch parses (no hardware needed)**

Run:
```bash
cd /home/tinker/tk25_ws && tkbuild tk26_navigation --packages-select navigation_bringup
source install/setup.bash
ros2 launch navigation_bringup follow_bringup.launch.py --show-args
```
Expected: build clean; `--show-args` lists `pan_tilt` + `pan_tilt_device`. (Do not
fully launch without the base/servo up — `--show-args` is the static check.)

- [ ] **Step 3: Commit (verify HEAD first)**

```bash
git -C src/tk26_navigation rev-parse --short HEAD          # note it
git -C src/tk26_navigation add src/navigation_bringup/launch/follow_bringup.launch.py
git -C src/tk26_navigation commit -m "feat(follow_bringup): pan_tilt:=true starts the head servo + tracker head-follow" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git -C src/tk26_navigation log --oneline -1                # verify it landed on top
```

---

## On-robot verification (operator-in-the-loop, after Task 2 build)

1. Bring up the head: `ros2 launch pan_tilt pan_tilt.launch.py device:=/dev/ttyUSB0
   launch_robot_state_publisher:=false`. Confirm `/pan_tilt_controller/state` and
   that the head pitches to ~40° on the first tracker command.
2. Start a `/track_person` goal with `-p enable_pan_tilt_follow:=true`. Stand to one
   side: the head should turn **toward** you and center you. If it turns away, flip
   `pan_sign`.
3. Walk left/right: the head tracks and holds you centered; no chatter when centered
   (deadband), no runaway (ABSOLUTE, clamped).
4. Step out of view briefly (PASSIVE): head **holds**. Stay out until NEEDS_HELP:
   head **recenters to 0** (forward), ready for the wave-reseed.

---

## Self-Review

- **Spec coverage:** tilt fixed 40° (Task 2 `_fixed_tilt_rad`, every command);
  pan-to-center (Task 1 `center` + Task 2 tick); ABSOLUTE only / no accumulation
  (`PanTiltCommand.ABSOLUTE`, target recomputed from live state — Task 1 tests
  `test_center_is_absolute_no_accumulation`); inside the tracker not the BT (Task 2
  loop call); launchable via the given command (Task 3 + the `-p` flag).
- **On-lost behavior:** HOLD on normal/PASSIVE loss (tick leaves `target=None`),
  RECENTER to 0 on NEEDS_HELP (`reacq_state == 2` branch) — matches the agreed
  decision.
- **Default-off safety:** `enable_pan_tilt_follow=False` creates no pub/sub and
  no commands — existing tracker bringups unchanged.
- **Type consistency:** `center`/`recenter` both return `Optional[float]` (pan_rad);
  `_publish_pan_tilt` always sets `tilt_rad=_fixed_tilt_rad`; intrinsics via
  `camera_intrinsic.k[0]`/`k[2]`.
- **No placeholders:** all steps carry real code + run commands + expected output.
```
