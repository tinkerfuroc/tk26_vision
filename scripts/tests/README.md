# tk26_vision integration tests

Shell-based smoke suite for the migrated tk26_vision nodes. Each tier is gated
by the previous — run T0 first, then T1, then T2, etc.

Logs are captured to `./logs/` (per-node, per-case). Inspect on any failure.

## Prerequisites

| Prereq | Required by | How to set up |
|---|---|---|
| Workspace built via `build.sh` | all | `./scripts/build.sh` |
| Venv exists at `.venv-vision-main` or the original checkout's shared venv | all | see `README.md` |
| `$WS_ROOT/.env` with real `OPENROUTER_API_KEY` | T2.7–T2.9, T3.1 | `cp src/kimi_api/.env.example .env && $EDITOR .env` |
| Orbbec camera running | T2–T4 | `ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true depth_registration:=true` |
| RealSense camera running | T2.2, T2.6 | `ros2 launch realsense2_camera rs_launch.py camera_name:=xarm_camera align_depth.enable:=true` |
| Servo at `/dev/ttyUSB0` or exported `SERVO_DEVICE` | T1.5 positive, T2.10, T3.3, T4.1–T4.2 | plug in pan-tilt; note the scripts still default to `/dev/ttyUSB1`, so export `SERVO_DEVICE=/dev/ttyUSB0` on current hardware |
| Physical shelf scene | T4.3 | 2–3 objects at two heights |
| Operator in frame | T4.4 | someone walks into orbbec view |

## Running

```bash
cd /path/to/tk26_vision
./scripts/tests/t0_static.sh       # <30 s, no cameras
./scripts/tests/t1_startup.sh      # ~2 min, no cameras
# Launch cameras in separate terminals first
./scripts/tests/t2_live.sh         # ~3 min
./scripts/tests/t3_interaction.sh  # ~1 min
./scripts/tests/t4_hardware.sh [servo_motion|servo_tracking|shelf_scene|person|person_phase2|follow_regression|all]
```

All scripts source the venv + ROS setup internally. You don't need to pre-source.

## Environment variable overrides

| Var | Default | Purpose |
|---|---|---|
| `WS_ROOT` | repo root | workspace root |
| `ROS_SETUP` | `/opt/ros/humble/setup.bash` | ROS setup file |
| `ENV_FILE` | `$WS_ROOT/.env` | dotenv file with `OPENROUTER_API_KEY` |
| `LOG_DIR` | `$WS_ROOT/scripts/tests/logs` | where per-node logs go |
| `SERVO_DEVICE` | `/dev/ttyUSB1` | pan-tilt serial device; stale default from before the clean pan-tilt refactor, override to `/dev/ttyUSB0` on current hardware |

## Exit codes

- `0` — all passes (skips are OK)
- `1` — one or more failures (failure summary printed at end)
- `2` — invalid invocation (T4 only)

## Known skip conditions

- T0.6, T1.7–T1.9 positive, T2.7–T2.9, T3.1, T2.11: skip if `ENV_FILE` is missing or contains placeholder key.
- T1.5 positive, T2.10, T3.3, T4.1/T4.2: skip if `$SERVO_DEVICE` is not a character device.
- T4.3, T4.4: require a human-staged scene; skipping = just don't run those subcommands.

## Debugging a failure

Per-node stdout/stderr is tee'd to `logs/<tag>.log`. The failure line prints a short tail; read the full file for details. Common issues:

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError` in a log | shebang not fixed — run `./scripts/fix_venv_shebangs.sh` |
| `OPENROUTER_API_KEY is not set` | edit `$ENV_FILE` with a real key |
| `SerialException` in the pan-tilt controller log | wrong `SERVO_DEVICE` or no servo plugged in |
| `/camera/color/image_raw not at 5 Hz` (T2 precheck) | orbbec launch not running |
| YOLO weights download hangs T1.1 | run once manually to let ultralytics cache the model |

Pan-tilt specific note:

- The test harness now talks to `/pan_tilt_controller/cmd` with
  `PanTiltCommand`. If you still see notes about `pan_tilt/ctrl`,
  `/pan_tilt_ctrl`, or `PanTiltCtrl`, those notes predate the clean refactor.

## Reverting the test suite

`rm -rf scripts/tests/` — no other artifacts. Doesn't touch package sources, tk23, or the venv.
