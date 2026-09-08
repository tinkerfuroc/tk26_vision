# restaurant_nav_test_web

A browser dashboard to run and observe the **restaurant pure-navigation test**:
the robot scans for a waving person, then approaches the closest one. Instead of
juggling a fistful of `ros2 launch` terminals, the operator drives the whole
test from one web page — bring the prerequisites up with a single click, watch
a readiness panel turn green, start the test, and follow the behavior tree's
progress (phase, detected wavers, chosen target, live distance, result)
alongside a live MJPEG camera feed. Prerequisite bring-up is operator-driven and
restricted to a fixed allowlist of commands.

## Prerequisites

Source the robot environment and set the FastDDS SHM profile **before**
launching. Both are required:

```bash
# Sets ROBOT_NAME (needed by navigation + pan_tilt per-robot config).
source /home/tinker/tk25_ws/src/tk25_basic/tools/robot-env.sh

# 30 Hz camera: the SHM profile must be set on this subscriber too, or the
# color topic drops to ~3 Hz and the MJPEG feed stutters.
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/src/tk26_vision/config/fastdds_shm.xml
```

> The launch file also sets `FASTRTPS_DEFAULT_PROFILES_FILE` for the node (it
> respects an already-exported value), but exporting it in your shell first
> keeps any tools you run in the same terminal at 30 Hz too.

`fastapi` and `uvicorn` live in the vision venv
(`src/tk26_vision/.venv-vision-main`), so build with the vision build wrapper
rather than a bare `colcon build`:

```bash
./src/tk26_vision/scripts/build.sh --packages-select restaurant_nav_test_web
```

## Launch

```bash
ros2 launch restaurant_nav_test_web restaurant_nav_test_web.launch.py
```

Then open the dashboard in a browser:

```
http://<host>:8768
```

Launch arguments:

| Arg            | Default                      | Purpose                                                  |
| -------------- | ---------------------------- | -------------------------------------------------------- |
| `bind`         | `0.0.0.0`                    | Interface the web server binds to.                       |
| `port`         | `8768`                       | HTTP port for the dashboard.                             |
| `camera_topic` | `/camera/color/image_raw`    | Color image topic subscribed for the MJPEG feed.         |

Example with overrides:

```bash
ros2 launch restaurant_nav_test_web restaurant_nav_test_web.launch.py \
    port:=8800 camera_topic:=/camera/xarm_camera/color/image_raw
```

## Workflow

1. **Start all prerequisites.** Click *Start all prerequisites*. The dashboard
   spawns the camera, pan_tilt, waving detector, and navigation stack via the
   allowlisted `ProcessManager` (see the allowlist note below). Each command is
   launched on a short stagger so the stack comes up cleanly.
2. **Watch the readiness dots.** The Readiness panel shows four dots —
   `camera`, `pan_tilt`, `waving`, `goto` — that turn green as the corresponding
   topic/service/feed becomes live (`camera` also requires a fresh frame). Wait
   for all four to go green.
3. **Start the test.** Click *Start test*. The behavior tree sweeps the pan-tilt
   through `[0, -60, +60]°`, detects waving people, and navigates to the closest
   one.
4. **Follow the result.** The Status panel shows the current phase, the detected
   wavers, the chosen target, the result, and the live distance to the target.
   The camera feed updates alongside it.

The **`mock` checkbox** runs the behavior tree in `BT_MOCK_MODE` (the dashboard
sets the `BT_MOCK_MODE` environment variable for the test process), letting you
exercise the UI and the test flow without real hardware in the loop.

## camera_topic note

The default `/camera/color/image_raw` matches the **femto** camera (the default
in the prerequisite allowlist). If you bring up the **realsense** camera
instead, its color topic is `/camera/xarm_camera/color/image_raw` — pass it
through:

```bash
ros2 launch restaurant_nav_test_web restaurant_nav_test_web.launch.py \
    camera_topic:=/camera/xarm_camera/color/image_raw
```

## Allowlist note

The dashboard can **only** spawn the named commands defined in
`config/processes.yaml`. Request input never becomes part of a command line — a
button maps to a registry key, and the `ProcessManager` runs the fixed argv list
for that key. This is a deliberate security boundary: nothing the browser sends
is interpreted as a shell command. To change which camera, navigation map, or
launch arguments are used per robot, edit `config/processes.yaml`.

## Changelog

- **2026-06-14** — Initial restaurant pure-nav test dashboard (scan-waving → approach; prerequisite bring-up; status + camera + readiness).
