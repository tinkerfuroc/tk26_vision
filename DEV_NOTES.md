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

## 2026-04-23 — `follow_head` rewritten for the new pan_tilt stack

The `ce3abec` refactor gave us `PanTiltCommand` (radians, ABSOLUTE/RELATIVE, speed/accel), `PanTiltState` (20 Hz hardware feedback), and `SetTorque`/`SetZero` services — but `follow_head` was still running the old pattern: 1 Hz detection, 1 s fixed settle, camera-frame relative deltas, closest-by-depth person selection, open-loop. This pass rewrites `follow_head.py` to exploit the new interface plus adds a small local tracker.

### Architecture changes

- **Pan-tilt-rooted absolute targeting (not TF / not base_link).** The original plan was to transform detections into `base_link` via TF, but the URDF's `camera_mount_joint` rpy has a ~175° yaw that didn't match the physical install — base-frame math sent the head 143° the wrong direction on the first live test. New approach: compute `target_angles = current_pan/tilt + camera-frame offset` directly from the servo's own feedback, no TF lookup needed. `_camera_to_pan_tilt_root` packs the candidate into a Cartesian frame rooted at the servo so the tracker + EMA have a servo-invariant distance metric; `_pan_tilt_root_to_angles` is the inverse. No dependency on `base_link` / URDF / `robot_state_publisher`.
- **Feedback-gated settle replaces the 1 s time gate.** `PanTiltState` arrives at 20 Hz; we keep a 4-deep ring buffer. `_classify_settle_state()` returns `go` iff `|state − last_commanded| < ε` AND `|Δstate/dt| < ω_eps` for N consecutive samples. Fallbacks: `stale_feedback` falls back to the Laplacian blur gate; a `max_settle_timeout_sec` watchdog advances after the deadline to avoid deadlocks.
- **Detection runs during settling.** Previously the settle gate held off the entire `follow_head_logic` including YOLO, so effective detection was ~0.67 Hz instead of 5 Hz. Moved the gate to sit *between* detection and publish, so the tracker + EMA keep consuming fresh frames even while the servo is mid-motion; the next command fires on the latest smoothed target.
- **Anti-chatter on the command side.** `min_command_change_deg` skips the publish when the new target is within ~0.5° of the last commanded angle, so tiny detection jitter no longer produces a stream of near-identical commands that the Waveshare firmware interrupts each in turn.
- **`PersonTracker` + `WorldTargetEMA`** in a new `pan_tilt/head_tracking_helpers.py`. Sticky nearest-neighbor lock with a `reassoc_dist_m` hysteresis + TTL (fixes "closest-by-depth jumps between similarly-distant people"); exponential moving average on the pan-tilt-root xyz. Both reset on goal start and on cancel.
- **`FollowHeadAction.Feedback` enriched** with `person_visible, current_pan, current_tilt, target_pan, target_tilt, error_deg`. Wire-compatible — `BtNode_MaintainEyeContact` already accesses `pan`/`tilt` via `getattr`.

### Tuning pass — live servo test 2026-04-23

First live test surfaced two symptoms that the static smoke tests missed:

1. **Slow convergence.** `command_speed_raw_small=60` cut motor speed in half for any error under 10°, giving ~2.7 °/s — a 4° correction took 1.5 s and tripped the watchdog. Removed the handicap (`command_speed_raw_small=0` → controller default) and tightened `small_error_deg` so speed scaling almost never kicks in.
2. **Glitchy / twitchy motion.** Command chatter from the servo being interrupted by new commands at the detection rate, plus the synchronous `cv2.imwrite` from the vision logger blocking the action loop ~30-40 ms per tick. Fix: `vision_logging_enabled: false` by default in `pan_tilt.yaml`; `min_command_change_deg` added; deadbands and speed/accel bumped.

Operator-requested final tuning is biased **responsiveness > smoothness** ("fast and glitchy is better than slow"):
- `default_speed_raw: 240` (was 120) + `default_accel_raw: 40` in the controller
- `min_detection_interval_sec: 0.1` (10 Hz YOLO)
- `max_settle_timeout_sec: 0.3`, `steady_{pan,tilt}_eps_deg: 3.0`, `steady_velocity_eps_deg_per_sec: 60.0`, `steady_sample_count: 1`
- `ema_alpha: 0.7` (was 0.4 — fresher over smoother)
- `{pan,tilt}_deadband_deg: 1.5`, `min_command_change_deg: 0.5`

If a later use case wants calmer motion, lower `ema_alpha` first, then tighten the `steady_*_eps_deg`.

### Files touched

| File | Change |
|---|---|
| `src/pan_tilt/pan_tilt/follow_head.py` | near-rewrite — pan-tilt-rooted targeting, feedback-gated settle, free-running detection, PersonTracker + EMA wiring, enriched feedback, anti-chatter, motion profile |
| `src/pan_tilt/pan_tilt/head_tracking_helpers.py` | **new** — `PersonTracker` (sticky NN + hysteresis + TTL), `WorldTargetEMA` (α smoothing + TTL) |
| `src/pan_tilt/config/pan_tilt.yaml` | bumped controller `default_speed_raw/accel_raw`; full follow_head param surface (~25 keys); `vision_logging_enabled: false` default |
| `src/tinker_vision_msgs_26/action/FollowHeadAction.action` | appended Feedback fields (backward-compat with getattr-based clients) |
| `src/pan_tilt/package.xml` | no new deps after reverting the base_link approach — TF listener removed |
| `src/tk26_vision/CLAUDE.md` | updated follow_head param list + vision-logging note |

### What still awaits testing

- **T4.2 servo tracking convergence CSV.** Plan file at `/home/tinker/.claude-wjy-paid/plans/pan-tilt-has-been-modular-lecun.md` § Verification describes extending `t4_hardware.sh servo_tracking` to parse feedback into a CSV + assert monotone error decrease and < 3° residual. Not yet wired in.
- **Auto-succeed-on-convergence.** `BtNode_MaintainEyeContact` docstring says "Server returns success after a single gaze lock" but the current server runs until canceled. Left as-is to preserve Receptionist + HRI parallel-with-speech semantics. Reopen if a caller needs the one-shot behavior.

---

## 2026-04-22 — Vision logging unification

Generalized the `debug_log_overlays` hook already present on `YOLOSegmentationNode` into a shared `VisionLogger` (in `vision_util`) and wired it into every vision node whose service/action output carries a bbox, segmentation mask, or centroid:

- Default flipped **on**. Old param `debug_log_overlays` is gone; the new boolean is `vision_logging_enabled` (default `True`). Existing `vision_log_folder` param stays, but its default is now `'vision_log'` (not `tmp/vision_log<ts>`) — the logger appends a `<YYYYmmdd_HHMMSS>/` subfolder on first write, so each process gets an isolated run directory without baking the timestamp into the param default.
- Coverage added: `person_track_node` (only on `TRACKING→LOST` and `LOST→TRACKING` transitions — last-good frame + current frame at the transition; no steady-state logging), `waving_person_server` (one artifact per service call; also removed the ad-hoc `person_roi<ts>.png` CWD writeout from `is_waving`), `follow_head` (piggybacks on the existing 1 Hz YOLO tick; tags the chosen centroid in JSON).
- Artifact schema additions: JSON payload now carries a `detections` list (bbox / cls_name / conf / centroid per entry, mask stripped) in addition to the existing `branch` / `request` / `n_detections`. Overlay draws a red dot at 2-D centroids when supplied.
- To opt out per-run: `ros2 run <pkg> <node> --ros-args -p vision_logging_enabled:=false`. To redirect: `-p vision_log_folder:=/some/abs/path`.

### Files touched

| File | Change |
|---|---|
| `src/vision_util/vision_util/vision_logging.py` | **new** — `VisionLogger` class, lazy run-dir creation, shared overlay/JSON writer |
| `src/object_detection_new/object_detection_new/object_seg_yolo.py` | param rename + default flip; `_write_debug_artifacts` now delegates to `VisionLogger` |
| `src/object_detection_generalist/object_detection_generalist/generalist_node.py` | attribute rename (`self.debug_log_overlays` → `self._vision_logger.enabled`) |
| `src/vision_track/vision_track/person_track_node.py` | params + logger; lost/reclaim transition hooks in `_handle_{tracked,lost}_frame`; reset in `_cleanup_tracking` |
| `src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py` | params + logger; drop `person_roi<ts>.png` imwrite |
| `src/pan_tilt/pan_tilt/follow_head.py` | params + logger at the existing 1 Hz detection site |
| five `package.xml` files | `<exec_depend>vision_util</exec_depend>` |
| `src/tk26_vision/CLAUDE.md` + `src/tk_vision_specialized/README.md` | docs |

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

---

## 2026-04-22 — Migration audit, three small fixes, + generalist VLM+SAM service

One longer session tracked two threads:
1. Re-audit of the tk23→tk26 vision migration to catch leftover regressions.
2. Adding a new open-vocabulary detection service so a retrained, person-free specialist model can coexist with a pretrained fallback.

Live-camera verification was done on this workstation; both cameras were up via the canonical bringup from the previous entry.

### What shipped

**Audit fixes (merged the same session):**

1. **`DetectWaving` namespace collision.** `tk25_decision/GPSR/custom_nodes.py:3` imports `DetectWaving` from `tinker_vision_msgs` (tk23), but `tk_vision_specialized/waving_person_server.py:3` advertised `/detect_waving_persons` under `tinker_vision_msgs_26.srv.DetectWaving`. The `.srv` bodies were byte-identical, but ROS service type identity is `package/srv/Type` — the two sides could never match. Removed `srv/DetectWaving.srv` from `tinker_vision_msgs_26` (CMakeLists + file), repointed the server + client at the tk23 type. `ros2 interface list` now shows the single `tinker_vision_msgs/srv/DetectWaving`.
2. **`img_sync_thres=0.00` default starved detection.** `object_seg_yolo.py:112` declared the threshold at zero, so `(call_time − recent_time) > 0` was true for every frame and the service logged `Skipping detection: no recent data within sync threshold` on healthy 30 Hz cameras. Raised the default to `0.20` (2× the ApproximateTimeSynchronizer slop of `0.1`).
3. **`tracking_active` race in `vision_track/person_track_node.py`.** `_goal_callback` checked `self.tracking_active` without a lock while `_execute_callback` wrote it under none. Under the MultiThreadedExecutor two concurrent goals could both observe `False` and both get accepted. Added `lock_lifecycle` around read+test-and-set in `_goal_callback` and the matching writes in `_execute_callback` / `_cleanup_tracking`.

**New generalist detection service:**

- New ament_python package `object_detection_generalist/` — subclasses `YOLOSegmentationNode` so camera sync, TF, depth projection, 3D centroid, and sort-mode logic are reused (not duplicated).
- New srv `tinker_vision_msgs_26/srv/ObjectDetection.srv` with typed boolean flags replacing the old string-parsed `flags` / `category`. Adds `force_vlm_sam`, `use_vlm_sam_fallback`, and a `detection_source` response tag. `Object[]` references `tinker_vision_msgs/Object` by rosidl dependency, so there is no duplicate type namespace (the pattern that caused the DetectWaving split is avoided).
- VLM client = Gemini 2.5 Pro via OpenRouter (`vlm_bbox.py`). Strict JSON schema, `[y0,x0,y1,x1]` 0–1000-normalized decode, retry loop mirroring `feature_matching.py`, **lazy** `OPENROUTER_API_KEY` check so the node starts without a key (unlike kimi_api which fails at `__init__`).
- SAM = `FastSAM-s.pt` via ultralytics' built-in class (`sam_mask.py`). Loaded once at node init on `self.device` (GPU if available).
- Specialist change: added `excluded_classes` ROS param to `YOLOSegmentationNode`; new `object_seg_yolo_specialist.py` wrapper sets it to `['person']` and `setup.py`'s `yolo_seg_node` entry now points at the wrapper. The default node and the generalist inherit `[]` so existing callers of `/object_detection` (e.g. `kimi_api.feature_matching` with `prompt='person'`) are unaffected.
- `scripts/fix_venv_shebangs.sh` — added `object_detection_generalist` to the default package list; one-time shebang repair was needed since the build wrapper defers to that script.

### Live-camera verification (RealSense path)

| Case | Expectation | Result |
|---|---|---|
| `/object_detection` prompt=`chair` / `tv` (pretrained YOLO) | status=0, populated | ✅ chair (1.06,0.26,-0.01), tv (1.14,-0.24,0.41) |
| `/object_detection_yolo` startup | log "`Excluded classes: ['person']`" | ✅ |
| `/object_detection_yolo` routine prompts (chair, tv) | routes through YOLO, filter untouched when the class isn't excluded | ✅ model detected tv + suitcases; filter did not alter behavior for non-excluded prompts |
| `/object_detection_generalist` prompt=`chair`, auto-fallback=on | YOLO returns [] → VLM+SAM | ✅ Gemini bbox → FastSAM mask → centroid (2.76, 0.47, 0.38); `detection_source=vlm_sam` |
| `/object_detection_generalist` prompt=`"monitor screen"`, `use_vlm_sam_fallback=true` | open-vocab direct to VLM+SAM | ✅ centroid (2.53, -2.00, -0.08); `detection_source=vlm_sam` |
| Gemini empty response handling | status=1 with meaningful `error_msg` | ✅ `"VLM+SAM produced no detections for \"person\""` |

Measured round-trip: YOLO ≈ 90 ms, Gemini 2.5 Pro 9–14 s, FastSAM bbox-prompt ≈ 100 ms.

### What still awaits testing (needs either a staged scene, a human subject, or Orbbec in a 5-float cloud layout)

| Item | Tier / How | Notes |
|---|---|---|
| Specialist `excluded_classes` filter firing on a real YOLO `'person'` detection | T4 | No person walked in front of the camera during the session. Startup log proves the param is loaded; a positive observation needs someone in frame. |
| Orbbec camera path for all three detection nodes | T2 | Blocked by a pre-existing `_pointcloud_to_array` bug (see Follow-ups #1). Tested on RealSense instead. |
| Extending the automated test tiers to cover the new generalist | T0 / T1 / T2 additions | Called out in the plan, deferred. See Follow-ups #3. |
| Manual Gemini-bbox-decode fixture that eyeballs the `[y0,x0,y1,x1]` convention against a saved image | manual | The "monitor screen" live call showed a plausible centroid, but a saved bbox overlay was not produced. See Follow-ups #6. |

### Follow-ups — ordered roughly by impact

1. **Pre-existing bug in `_pointcloud_to_array` (orthogonal to this session).** `object_seg_yolo.py:399-401` hardcodes 5 floats per point, but the canonical Femto Bolt launch publishes `/camera/depth/points` with `point_step=16` (xyz only, no rgb). Every orbbec call after the first few fails with `cannot reshape array of size X into shape (Y,5)`. Two fixes possible; neither done here:
    - *Launch-side*: add `enable_colored_point_cloud:=true` to the canonical orbbec invocation in `CAMERA_BRINGUP.md` so the cloud is xyzrgb (matches the existing 5-float assumption).
    - *Code-side*: derive `floats_per_point = pc_msg.point_step // 4` inside `_pointcloud_to_array` — one-line change that also makes the node robust to future layout swaps.
2. **Scan tk25_decision + tk25_manipulation for legacy `ObjectDetection` use and migrate open-vocabulary callers to `/object_detection_generalist`.** Known call sites from the session's earlier audit:
    - `src/tk25_decision/src/behavior_tree/behavior_tree/TemplateNodes/Vision.py:81` `BtNode_ScanFor` (currently `/object_detection_yolo`)
    - `src/tk25_decision/src/behavior_tree/behavior_tree/TemplateNodes/Vision.py:210` `BtNode_TrackPerson` (currently `/object_detection`, prompt=`person`) — should target the generalist once callers are ready
    - `src/tk25_decision/src/behavior_tree/behavior_tree/TemplateNodes/Vision.py:356` `BtNode_FindObj` (currently `/object_detection`) — arena items, should point at `/object_detection_yolo`
    - `src/tk25_decision/src/behavior_tree/behavior_tree/StoringGroceries/customNodes.py:24` `BtNode_FindObjTable` (currently `/object_detection`) — arena items, should point at `/object_detection_yolo`
    - `src/tk25_decision/src/behavior_tree/behavior_tree/GPSR/custom_nodes.py:549` `BtNode_ScanForWavingPerson` — open-vocab, good generalist target
    - `src/tk25_decision/src/behavior_tree/behavior_tree/Restaurant/custumNodes.py:15` `BtNode_DetectCallingCustomer` — open-vocab, good generalist target
    - `src/tk25_decision/src/behavior_tree/behavior_tree/HelpMeCarry/customNodes.py:22` `BtNode_FindPointedLuggage` — arena items, `/object_detection_yolo`
    - `src/tk25_manipulation/arm_api/grasp_demo.py:19` and `grasp_demo_place.py:21` — already on `/object_detection_yolo`; no change, just sanity check
    - `src/tk25_manipulation/arm_api/anygrasp_test.py:30` — open-vocab grasping, good generalist candidate
    Each retarget requires either a `detection_service` param flip at launch (for callers that have one) or a one-line default change in the BT node. Migrating `feature_matching` and `grocery_categorize` (both already parameterized) is the lowest-risk first step.
3. **kimi_api srv migration**: `feature_matching` and `grocery_categorize` import from `tinker_vision_msgs.srv` today. They should optionally target the new `tinker_vision_msgs_26/srv/ObjectDetection` so open-vocab prompts work end-to-end. Requires a parallel `ObjectDetection` client type alongside the existing one (or a launch-time pick via an env var). See the srv field mapping in `tinker_vision_msgs_26/README.md` for what needs to change in the call site.
4. **Test tier extensions** (`src/tk26_vision/scripts/tests/`):
    - T0: `ros2 interface show tinker_vision_msgs_26/srv/ObjectDetection` returns a doc containing `force_vlm_sam` + `detection_source`; `python -c "from object_detection_generalist.generalist_node import GeneralistDetectionNode; from ultralytics import FastSAM"` in the venv exits 0.
    - T1: spawn `generalist_node` with and without `OPENROUTER_API_KEY`, assert it advertises the service in both cases and SIGTERMs cleanly. Negative: service call with `prompt='unicorn'`, `use_vlm_sam_fallback=true` under no-key → status=1, `error_msg` mentions `OPENROUTER_API_KEY`.
    - T2: one live call with `prompt='bottle'` (YOLO branch) and one with `prompt='spatula'`, `use_vlm_sam_fallback=true` (VLM+SAM branch), and one against the specialist with `prompt='person'` on a scene with a person in frame (excluded_classes positive case).
5. **Log seen images + overlays for user verification.** Today the generalist returns only `(cls, conf, centroid)` and optionally the raw rgb/segments — no overlay. Auditing whether Gemini's bbox and FastSAM's mask look right is tedious. Add a `vision_log_folder` dump of:
    - `req_{ts}.json` — request params + the branch taken + VLM raw JSON.
    - `orig_{ts}.jpg` — the RGB frame used.
    - `overlay_{ts}.jpg` — same frame with YOLO/VLM bboxes drawn + SAM/YOLO masks tinted.
    Gate behind a `debug_log_overlays` ROS param so it's off in production. `object_seg_yolo.py` already has a `visualization` param but it uses `cv2.imshow`, which requires a display — the new behavior should write PNGs instead. Consider extending to the specialist + default nodes as well so all three services produce comparable audit artifacts.
6. **Manual Gemini-bbox-decode fixture** (`scripts/tests/manual/gemini_bbox_decode.py`). Loads a saved image, calls `vlm_bbox.request_bboxes(img, 'spatula', model='google/gemini-2.5-pro')`, draws the returned xyxy on the image, writes an overlay PNG + the raw JSON response. Running it once confirms the `[y0,x0,y1,x1]` 0-1000 order matches what Gemini actually emits. Cheap; worth shipping before we rely on the VLM path in a demo.
7. **Specialist model training.** `yolo_seg_node`'s whole purpose is a custom-trained YOLOv11-seg on competition items only (no `'person'` class — the `excluded_classes=['person']` filter is belt-and-suspenders). That model does not exist yet; the specialist currently serves `yolo11m-seg.pt` (pretrained COCO) which has none of the arena-specific classes. Parking the training task here so it doesn't get lost.
8. **VLM latency**. Gemini 2.5 Pro at 9–14 s per call is fine for occasional open-vocab fallback but pushes total detection latency to ~15 s in the worst case. If the robot needs sub-second open-vocab detection, options are: cheaper model (`google/gemini-2.5-flash` — faster, worse bboxes), pre-cache common classes in YOLO via a few-shot fine-tune, or run a local open-vocab detector (GroundingDINO / OWL-ViT).
9. **Double camera subscription.** Running specialist + generalist + default together means every color frame + depth stream is decoded three times into three `YOLOSegmentationNode` instances. Fine at 30 Hz but wasteful. Consolidation options: factor the camera-input half of `YOLOSegmentationNode` out as a shared `Node` that the three services *use* rather than subclass; or run one node that advertises all three services. Neither is urgent.
10. **Per-VLM-detection confidence.** `generalist_node._build_vlm_sam_objects` hardcodes `obj.conf = 1.0` because Gemini's JSON shape doesn't carry a probability. If downstream callers start filtering on confidence, either lift a `score` from the VLM prompt ("also emit a 0-1 confidence per detection") or flag this as a known-constant so callers don't rely on it.

### Files touched by this session

| Path | Change |
|---|---|
| `src/tk26_vision/src/object_detection_new/object_detection_new/object_seg_yolo.py` | `excluded_classes` param + class filter; `img_sync_thres` default 0.00 → 0.20 |
| `src/tk26_vision/src/object_detection_new/object_detection_new/object_seg_yolo_specialist.py` | New — wrapper sets `excluded_classes=['person']` |
| `src/tk26_vision/src/object_detection_new/setup.py` | Repointed `yolo_seg_node` entry at the specialist wrapper |
| `src/tk26_vision/src/vision_track/vision_track/person_track_node.py` | `lock_lifecycle` around `tracking_active` |
| `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py` | Import `DetectWaving` from tk23 package |
| `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/waving_client.py` | Same |
| `src/tk26_vision/src/tk_vision_specialized/README.md` | Reflect tk23 srv namespace |
| `src/tk26_vision/src/tinker_vision_msgs_26/srv/ObjectDetection.srv` | New — boolean-flag srv |
| `src/tk26_vision/src/tinker_vision_msgs_26/srv/DetectWaving.srv` | Deleted (duplicate of tk23 type) |
| `src/tk26_vision/src/tinker_vision_msgs_26/CMakeLists.txt` | Register new srv, `find_package(tinker_vision_msgs)`, drop DetectWaving |
| `src/tk26_vision/src/tinker_vision_msgs_26/package.xml` | `<depend>tinker_vision_msgs</depend>` |
| `src/tk26_vision/src/tinker_vision_msgs_26/README.md` | Rewritten — services table + tk23→tk26 field mapping |
| `src/tk26_vision/src/object_detection_generalist/**` | New package (generalist_node, vlm_bbox, sam_mask, README, packaging) |
| `src/tk26_vision/scripts/fix_venv_shebangs.sh` | Add `object_detection_generalist` to default package list |
| `src/tk26_vision/CLAUDE.md` + `src/tk26_vision/README.md` | Architecture / running-nodes updates + `excluded_classes` gotcha |
| `CLAUDE.md` (workspace) | `object_detection_generalist` row in the source tree |

---

## 2026-04-22 — Follow-up wave

Executed the plan at `plans/plan-for-the-follow-keen-platypus.md`. Closed follow-ups #1–#5 and #9 from the prior session's list. Remaining open items (specialist training, VLM latency, triple-sub, plus a newly-surfaced TrackPerson/ScanForWavingPerson rearchitect) moved to the top of `CLAUDE.md § Known follow-ups`.

### What shipped

**Wave 1 — quick wins**
- `object_seg_yolo.py:399-403` — `_pointcloud_to_array` now derives `floats_per_point = pc_msg.point_step // 4` instead of hardcoding 5. Unblocks the Orbbec path for all three detection nodes (specialist, default, generalist) without needing `enable_colored_point_cloud:=true` on the canonical bringup.
- `generalist_node.py:_build_vlm_sam_objects` — docstring now states the `conf = 1.0` uniformity explicitly and points callers at `detection_source` for branch-level filtering.

**Wave 2 — downstream migration**
- New `ObjectDetectionGeneralist` shim in `behavior_tree/messages.py` + mock in `mock_messages.py`, following the existing `TrackPerson` try/except pattern. Real type resolves to `tinker_vision_msgs_26/srv/ObjectDetection`; falls back to a mock when tk26 isn't installed.
- Arena-item BT callers flipped to the specialist (service name only, tk23 srv preserved):
  - `behavior_tree/TemplateNodes/Vision.py:356` `BtNode_FindObj` → `/object_detection_yolo`
  - `behavior_tree/StoringGroceries/customNodes.py:24` `BtNode_FindObjTable` → `/object_detection_yolo`
  - `behavior_tree/HelpMeCarry/customNodes.py:22` `BtNode_FindPointedLuggage` → `/object_detection_yolo`
- Open-vocabulary BT callers migrated to the generalist (service name + srv type + request field mapping):
  - `behavior_tree/TemplateNodes/Vision.py:210` `BtNode_TrackPerson`
  - `behavior_tree/GPSR/custom_nodes.py:549` `BtNode_ScanForWavingPerson`
  - `behavior_tree/Restaurant/custumNodes.py:11` `BtNode_DetectCallingCustomer`
  - `tk25_manipulation/.../arm_api/anygrasp_test.py` (import, client, request)
- `kimi_api/feature_matching.py` + `kimi_api/grocery_categorize.py` — switched to `tinker_vision_msgs_26/srv/ObjectDetection`, boolean flag fields (`return_rgb_image`, `return_segments`, `use_vlm_sam_fallback`), default `detection_service` flipped to `'object_detection_generalist'`. `kimi_api/package.xml` gained `<depend>tinker_vision_msgs_26</depend>`. `tk25_manipulation/src/arm_api/package.xml` likewise.

  Important: the tk26 detection nodes' `request.flags` parser only honors `sort_closest|sort_highest|sort_none` substrings. The legacy strings `"register_person"`, `"find_for_grasp"`, `"find_waving_person"`, `"detect_gesture"`, `"find_pointed_object"`, `"find_pointed_object"`, and `"scan"` are all no-ops — they've been silently eaten by every caller since the tk23→tk26 swap. Dropping them on migration is behavior-preserving, **not** a semantic change.

  Caveat (see open follow-up #4 in CLAUDE.md): `BtNode_TrackPerson` still reads `result.person_id` and `BtNode_ScanForWavingPerson` still filters on `Object.being_pointed == 3`. Neither is populated by the generalist (or by tk26 detection nodes in general), so these two nodes are mechanically migrated but need a proper rearchitect.

**Wave 3 — observability + coverage**
- `debug_log_overlays` ROS param (default `False`) on `YOLOSegmentationNode`. When set, each service call dumps `orig_{ts}.jpg`, `overlay_{ts}.jpg`, `req_{ts}.json` under `vision_log_folder/`. Generalist overrides to produce equivalent artifacts on both the YOLO branch and the VLM+SAM branch (VLM raw bboxes included in the JSON payload).
- `scripts/tests/manual/gemini_bbox_decode.py` — standalone fixture that loads a JPEG, calls `request_bboxes` with a given prompt + model, and writes `<stem>_overlay.png` + `<stem>_raw.json`. Use it to sanity-check the `[y0,x0,y1,x1]` 0-1000 decode convention after model bumps or `_SYSTEM_PROMPT` edits.
- Test tier extensions:
  - T0: generalist added to the shebang sweep, the venv-deps check (`FastSAM` import), a dedicated T0.3b module-import check, the interface-show list (`tinker_vision_msgs_26/srv/ObjectDetection`), and the entry-point import sweep. `object_detection_generalist` gets a smoke `OPENROUTER_API_KEY` during `--help` import just like kimi_api.
  - T1.12: generalist advertises `/object_detection_generalist` both with and without `OPENROUTER_API_KEY` (the generalist checks the key lazily on the VLM branch — this is the contract that distinguishes it from kimi_api's fail-at-init).
  - T2.14: live YOLO-branch call (prompt `'bottle'`) + live VLM+SAM-branch call (prompt `'spatula'`, `use_vlm_sam_fallback=true`). VLM case skips cleanly without a real key.
  - T2.15: startup-log sanity on the specialist's `excluded_classes=['person']` param (live positive-case with a person in frame remains a T4 operator check).
  - T3.4: `feature_matching ↔ generalist_node` pairing, mirroring the existing T3.1 pair against the default YOLO.

### Verification (this workstation)

- `./scripts/build.sh --packages-select object_detection_new object_detection_generalist tinker_vision_msgs_26 kimi_api` — clean, shebangs patched.
- `colcon build --packages-select behavior_tree arm_api` — clean.
- `./scripts/tests/t0_static.sh` — **18 pass / 0 fail / 0 skip**.
- `./scripts/tests/t1_startup.sh` — **15 pass / 0 fail / 3 skip**. Skips are pre-existing (kimi_api no-key branches unreachable while `/home/tinker/tk25_ws/.env` carries a real key).
- T2/T3 with live cameras not rerun in this session; reproduction steps unchanged from the 2026-04-22 entry above.

### Files touched

| Path | Change |
|---|---|
| `src/tk26_vision/src/object_detection_new/object_detection_new/object_seg_yolo.py` | `point_step`-derived reshape; `debug_log_overlays` param + `_write_debug_artifacts` helper; stash `_last_detection_info` / `_last_rgb_img` for the service callback |
| `src/tk26_vision/src/object_detection_generalist/object_detection_generalist/generalist_node.py` | VLM-conf docstring; `debug_log_overlays` wiring for both YOLO and VLM+SAM branches |
| `src/tk26_vision/src/kimi_api/kimi_api/feature_matching.py`, `grocery_categorize.py` | tk26 srv import, boolean flag fields, default service → generalist |
| `src/tk26_vision/src/kimi_api/package.xml` | `+<depend>tinker_vision_msgs_26</depend>` |
| `src/tk25_decision/src/behavior_tree/behavior_tree/messages.py`, `mock_messages.py` | Add `ObjectDetectionGeneralist` shim + mock |
| `src/tk25_decision/.../TemplateNodes/Vision.py`, `GPSR/custom_nodes.py`, `Restaurant/custumNodes.py`, `StoringGroceries/customNodes.py`, `HelpMeCarry/customNodes.py` | Service-name + srv-type updates per plan table |
| `src/tk25_manipulation/src/arm_api/arm_api/anygrasp_test.py` + `package.xml` | tk26 srv import, service name, request fields, msgs dep |
| `src/tk26_vision/scripts/tests/t0_static.sh`, `t1_startup.sh`, `t2_live.sh`, `t3_interaction.sh` | Generalist coverage per Wave 3.2 |
| `src/tk26_vision/scripts/tests/manual/gemini_bbox_decode.py` | New manual VLM-decode fixture |
| `src/tk26_vision/CLAUDE.md` | Known-follow-ups trimmed to the 4 that remain open |

### Not done (deferred, see CLAUDE.md § Known follow-ups)

- Live T2/T3 regression pass and T4 operator scenarios (person-in-frame specialist positive case, servo motion, person ReID, shelf categorization with populated scene) — unchanged from the prior entry; still waiting on operator.
- `BtNode_TrackPerson` + `BtNode_ScanForWavingPerson` semantic rearchitect (generalist doesn't populate `person_id` or `being_pointed`). Mechanical migration landed; proper fix needs routing TrackPerson at `/track_person` (person_track_server action) and replacing ScanForWavingPerson with the sibling `ScanForWavingPersonNew` that uses `DetectWaving`.

