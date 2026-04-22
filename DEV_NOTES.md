# tk26_vision developer notes — on-robot verification log

Running notes on what has been exercised on the physical robot, what was fixed while getting there, and what still needs operator-in-the-loop checks. Meant to be appended to after each substantive run; treat older entries as historical.

This file is distinct from `CLAUDE.md` (which describes the *design*) and `README.md` (which is user-facing). Here we capture **what's been proven** on a specific workstation, what we had to patch along the way, and what remains unverified.

## Test matrix

| Category | Covered by | Status |
|---|---|---|
| Static / build-env (shebangs, venv imports, ROS interfaces) | `scripts/tests/t0_static.sh` | ✅ passing |
| Node startup, interface advertisement, clean SIGTERM | `scripts/tests/t1_startup.sh` | ✅ passing |
| Live-camera single-call per node (empty scene) | `scripts/tests/t2_live.sh` | ✅ passing (with skips — see below) |
| Cross-node interaction (client ↔ server) | `scripts/tests/t3_interaction.sh` | ✅ passing |
| Hardware-in-the-loop with staged scenes | `scripts/tests/t4_hardware.sh` | ⏳ **not yet run** (needs operator) |

---

## Verification run — 2026-04-22

**Workstation:**
- GPU: NVIDIA GeForce RTX 5070 Ti (driver 570.211.01, CUDA 12.8)
- Cameras: Orbbec Femto Bolt + Intel RealSense (xarm-mounted), both live on USB
- Pan-tilt servo: attached at `/dev/ttyUSB1` with 0777 perms. **Note**: depending on boot order / which USB device enumerates first, this may be `/dev/ttyUSB0` on other machines — always override via `--ros-args -p device:=/dev/ttyUSBX`. The default in `pan_tilt_ctrl.py` is `/dev/ttyUSB0`.
- OpenRouter creds: `/home/tinker/tk25_ws/.env` populated with real `OPENROUTER_API_KEY`.

### What was exercised and passed

- **T0 static** — 16/16 pass. Confirms shebangs point at the venv python, ROS interface definitions built, and every migrated entry-point imports cleanly under the venv.
- **T1 startup** — 13 pass / 3 skip. The 3 skips are T1.7/T1.8/T1.9 *negative* sub-cases (node must `RuntimeError` without an API key) — unreachable while `/home/tinker/tk25_ws/.env` exists with a real key, since `python-dotenv` loads it from CWD upward. The *positive* sub-cases passed for all three kimi_api nodes. Positive `pan_tilt/ctrl` case also passed (real servo on `/dev/ttyUSB1`, TF chain `base_link→pan_link→tilt_link→camera_link` resolved via `tf2_echo`).
- **T2 live** — 13 pass / 2 skip. All per-node service/action calls returned structurally valid responses on an empty scene:
  - `/object_detection` (default YOLO) + `/object_detection_yolo` (custom YOLO) — both cameras
  - `/door_detection_srv` (status=0, is_open either value)
  - `/feature_extraction_service` + `/seat_recommend_service` — real OpenRouter responses, e.g. `feature='There is no person visible in the center of the image.'` and a full seat recommendation
  - `/feature_matching_service` — propagates empty-scene status=1 with `centroids=[]`
  - `/follow_head_action` — goal accepted, servo holds position
  - `/grocery_categorize`, `/spot_on_shelf`, `/track_person` — actions accepted goals and terminated cleanly
- **T3 interaction** — 4/4 pass. `feature_matching` talked to `yolo_seg_default_node`; `spot_on_shelf` talked to `yolo_seg_node`; `pan_tilt_ctrl` TF chain remained intact with `follow_head` running, no error spam.

### Fixes applied during the run

1. **`transforms3d` missing from venv.** `pan_tilt/pan_tilt_ctrl.py` imports `tf_transformations` which imports `transforms3d`. Installing `ros-humble-tf-transformations` via apt doesn't populate the venv. Fix: `transforms3d>=0.4` pinned in `src/pan_tilt/requirements.txt` with an inline comment.
2. **`torch 2.11.0+cu130` vs. driver CUDA 12.8.** System NVIDIA driver (570.x) supports CUDA 12.8; the `+cu130` wheel triggered `UserWarning: NVIDIA driver on your system is too old (found version 12080)` and silently fell back to CPU. Fix: `pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision` into the venv. Verified `torch.cuda.is_available()=True` and YOLO `model.device=cuda:0` after reinstall.

### What still awaits testing

These are **not regressions** — just scenarios the automated tiers cannot prove without a human or a staged scene.

| Item | Tier | What it needs | Notes |
|---|---|---|---|
| Servo commanded motion | T4.1 | Operator runs `./scripts/tests/t4_hardware.sh servo_motion`, watches the head move to commanded pan angle | Confirms `/pan_tilt_ctrl_modify` publishes reach the servo + TF reflects commanded pose |
| Head-follow tracking | T4.2 | Operator waves a hand in front of Orbbec for ~15 s | Confirms `follow_head` + `ctrl` cooperation with a real subject |
| Shelf categorization w/ populated scene | T4.3 | 2–3 distinct objects at two heights in Orbbec view | Confirms `spot_on_shelf` returns non-empty `item_height_grids` / `item_horizontal_grids` |
| Person tracking (ReID persistence) | T4.4 | Operator walks into frame, occludes briefly, re-emerges | Confirms ReID keeps `target_track_id` stable across occlusion |
| `get_point_cloud` at healthy camera rates | T2.5 / T2.6 | Orbbec + RealSense both publishing color and depth at ≥ 10 Hz | Was skipping in the 2026-04-22 T2 run because cameras were at 2–4 Hz — root cause since found and fixed (see `2026-04-22 — Camera bringup performance fix` below). Re-run expected to pass once the fix is adopted. |
| First-boot YOLO weight caching | T1 cold | First run with no cached weights | Ultralytics auto-downloads `yolo11{n,m,s}-seg.pt` and `yolov8s-seg.pt` to CWD on first use. Accounted for — documented for anyone running on a fresh venv. |

### Known non-issues (expected behavior, worth remembering so you don't chase them)

- **`ObjectDetection` returns `status=1, objects=[]` on an empty scene.** `status=0` means detections exist; `status=1` means none. Neither is an error. Service callers that treat `status != 0` as failure (`feature_matching` does — logs `Detection failed (status 1): .`) are propagating the empty-scene signal, not reporting a bug.
- **Shutdown traceback `RCLError: failed to shutdown: rcl_shutdown already called`** appears at the tail of most node logs after SIGTERM. Cosmetic: SIGTERM triggers one shutdown; `main()`'s `rclpy.shutdown()` then runs again. All init completes well before shutdown; ignore.
- **kimi_api T1 negative sub-cases skip** when `/home/tinker/tk25_ws/.env` is populated. `python-dotenv` finds the file from any CWD, so we can't exercise the "no key" branch without moving the file aside. If you need to exercise the negative path for some reason:
    ```bash
    mv /home/tinker/tk25_ws/.env /tmp/.env.bak
    ./src/tk26_vision/scripts/tests/t1_startup.sh 2>&1 | grep -A1 T1.7
    mv /tmp/.env.bak /home/tinker/tk25_ws/.env
    ```

### Reproducing this run

```bash
cd /home/tinker/tk25_ws
./src/tk26_vision/scripts/tests/t0_static.sh
./src/tk26_vision/scripts/tests/t1_startup.sh
# Launch cameras in separate terminals before T2:
#   ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true depth_registration:=true
#   ros2 launch realsense2_camera rs_launch.py camera_name:=xarm_camera align_depth.enable:=true
MIN_HZ=1 ./src/tk26_vision/scripts/tests/t2_live.sh        # MIN_HZ lowered because camera rates were slow that day
./src/tk26_vision/scripts/tests/t3_interaction.sh
# T4 subcommands — operator required
```

If the servo isn't at `/dev/ttyUSB1` on your workstation, export `SERVO_DEVICE=/dev/ttyUSBX` before running the suite.

---

## 2026-04-22 — Camera bringup performance fix (3 Hz → 30 Hz)

The "cameras at 2–4 Hz" footnote attached to the T2 run (above) was not a quirk of that session — it was a reproducible config problem. Diagnosed and fixed the same day.

### Symptom

Running the documented camera launches

```bash
ros2 launch orbbec_camera femto_bolt.launch.py depth_registration:=true
ros2 launch realsense2_camera rs_launch.py camera_name:=xarm_camera align_depth.enable:=true
```

left every `/*/color/image_raw` and `/*/depth/image_raw` topic at **~3 Hz** as seen by `ros2 topic hz`, even though the driver logs reported 30 fps capture and no errors. `get_point_cloud`'s `ApproximateTimeSynchronizer(slop=0.05)` correspondingly never paired stamps and the service returned `No camera data for …`.

### Three layered root causes (each independently throttles rates; all had to be fixed)

1. **RealSense on USB 2.0 port.** `lsusb -v -d 8086:0b07` showed `bcdUSB 2.10` — the D435 had been plugged into a USB 2.0 port (or through a USB-2-only cable). USB 2.0 High-Speed practical throughput (~35–45 MB/s) is below the driver's default 848×480 color + depth @ 30 fps bandwidth (~62 MB/s). **Resolved by moving to a USB 3.0 port;** verify with `lsusb -v -d 8086:0b07 | grep bcdUSB` showing `3.10` or higher. Both the D435 and the Femto Bolt now share Bus 04 (5 Gbps root hub) on this workstation.

2. **realsense-ros publishes images with `Durability: TRANSIENT_LOCAL`.** `thirdparty/realsense-ros/realsense2_camera/include/constants.h:83` defaults `IMAGE_QOS = "SYSTEM_DEFAULT"`, which FastDDS resolves to RELIABLE + TRANSIENT_LOCAL — a profile meant for latched topics (static TF, maps), not 1.2 MB frames at 30 fps. The driver exposes runtime `*_qos` parameters but they are **not** in `configurable_parameters` in `rs_launch.py`, so `color_qos:=DEFAULT` on the command line is silently dropped. Must be supplied via `config_file:=…` and the YAML must be flat key/value (the launch loads it with plain `yaml.SafeLoader` and bypasses the normal ROS2 `/**: ros__parameters:` resolver).

3. **Kernel UDP receive buffer is 208 KB by default (Ubuntu 22.04).** ROS2 Humble's default RMW (rmw_fastrtps_cpp, FastDDS 2.6.11) fragments 1.2 MB image messages into many UDP datagrams. A 208 KB socket buffer overflows; `grep ^Udp: /proc/net/snmp` confirmed `RcvbufErrors` accumulating at ~1.2 k/s *per* camera-subscriber pair while running. This reproduces even when only one camera is running (so it is not USB contention), and also explains the strange earlier symptom where RealSense aligned-depth (smaller frames) was received at 15 Hz while color was at 3 Hz.

Secondary Orbbec-only knobs matter too:

- `enable_frame_sync:=true` (`thirdparty/OrbbecSDK_ROS2/orbbec_camera/launch/femto_bolt.launch.py:80`) ties color to the slowest stream in the SDK — any depth stall dragged color down with it. Disable it; color and depth still carry hardware capture timestamps so `ApproximateTimeSynchronizer(slop=0.05)` pairs them fine (measured median |Δ| = 1 ms, p95 = 2 ms, max = 2.4 ms across 300 paired frames).
- `enable_ir:=true` (line 52) is on by default. No tk26_vision node subscribes to IR; dropping it saves USB bandwidth and one decode thread.
- `align_mode:=HW` is **not usable** with the default 1280×720 MJPG color + 640×576 Y16 depth profile — driver logs `Failed to start pipeline: Current stream profile is not support hardware d2c process` and resets. Leave it at the default `SW`.

### Applied fix

Config checked into `src/tk26_vision/config/`:

- `fastdds_shm.xml` — FastDDS profile: SHM-preferred transport (`useBuiltinTransports=false`, SHM first, UDP as fallback). Removes the UDP-buffer failure mode for any producer + consumer that both set `FASTRTPS_DEFAULT_PROFILES_FILE`.
- `realsense_qos.yaml` — flat yaml overriding `color_qos`, `depth_qos`, `infra{1,2}_qos`, and the `*_info_qos` siblings to `DEFAULT` (= RELIABLE + VOLATILE + KEEP_LAST(10)).

Canonical launch sequence:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/src/tk26_vision/config/fastdds_shm.xml

# terminal 1 — RealSense
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=xarm_camera \
    align_depth.enable:=true \
    config_file:=$(pwd)/src/tk26_vision/config/realsense_qos.yaml

# terminal 2 — Orbbec Femto Bolt
ros2 launch orbbec_camera femto_bolt.launch.py \
    depth_registration:=true \
    enable_ir:=false \
    enable_frame_sync:=false
```

Any downstream node (`ros2 topic hz`, detection / tracking, `get_point_cloud`, etc.) also needs `FASTRTPS_DEFAULT_PROFILES_FILE` set in its shell — easiest is to export it from `~/.bashrc` / `~/.zshrc`.

### Verification (measured 60-second soak on this workstation)

| Topic | Mean rate | σ | Max interval |
|---|---|---|---|
| `/camera/xarm_camera/color/image_raw` | 29.97 Hz | 3.0 ms | 139 ms |
| `/camera/xarm_camera/aligned_depth_to_color/image_raw` | 29.63 Hz | 4.5 ms | 167 ms |
| `/camera/color/image_raw` (Orbbec) | 30.17 Hz | 1.2 ms | 43 ms |
| `/camera/depth/image_raw` (Orbbec) | 30.17 Hz | 1.2 ms | 42 ms |

UDP `InErrors` delta over the 60 s window: 487, down from >20 k in the same interval before the fix. The remaining errors are FastDDS discovery chatter, not image data.

Orbbec color ↔ depth header-stamp drift with `enable_frame_sync:=false`: **median 1.0 ms, p95 2.0 ms, max 2.4 ms** over 301/302 paired frames — 100% of pairs fall within the `ApproximateTimeSynchronizer(slop=0.05)` window `get_point_cloud` uses. `enable_frame_sync` only controls SDK-side frame *pairing* before publish, not the stamps; every frame still carries its own hardware capture time.

### Optional follow-ups

- **Raise kernel UDP buffers system-wide** (the ROS2-official fix — needs sudo):
  ```bash
  sudo tee /etc/sysctl.d/60-ros2-udp.conf <<'EOF'
  net.core.rmem_max=8388608
  net.core.rmem_default=8388608
  net.core.wmem_max=8388608
  net.core.wmem_default=8388608
  EOF
  sudo sysctl --system
  ```
  With this, the SHM profile becomes a perf choice rather than a correctness requirement — stock FastDDS nodes started without the env var will also behave reasonably.

- **Bundle a wrapper launch file** under `src/tk26_vision/launch/cameras_bringup.launch.py` that sets the env var and composes both camera launches with the overrides. Would let teammates run one command instead of remembering the two-launch incantation.

- **Switch RMW to CycloneDDS** (`apt install ros-humble-rmw-cyclonedds-cpp`, `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`). Handles large messages without the XML profile. Not installed on this box yet; noting for completeness.

### Files touched by this fix

| Path | Change |
|---|---|
| `src/tk26_vision/config/fastdds_shm.xml` | New — SHM-preferred FastDDS profile |
| `src/tk26_vision/config/realsense_qos.yaml` | New — flat-format QoS overrides consumed by `rs_launch.py:config_file` |
| `src/tk26_vision/CLAUDE.md` | New §"Camera bringup" + invariant update |
| `src/tk26_vision/README.md` | §"Camera Setup" rewritten with the env-var + launch-with-config commands |
| `src/tk26_vision/DEV_NOTES.md` | This entry + row update on `get_point_cloud` pending-test |
| `/home/tinker/tk25_ws/CLAUDE.md` | §"Running Launch Files" camera block updated to the new commands |

No source files (Python or C++) were modified. No vendored thirdparty drivers were patched.

