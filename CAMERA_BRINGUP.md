# Camera bringup — RealSense + Orbbec Femto Bolt

Canonical multi-camera bringup for tk26_vision. Covers the required config files, the exact launch commands, the three compounding misconfigurations the vendored launches ship with, and the gotchas found while tuning them. If you just want to run the cameras, jump to [§Canonical launch sequence](#canonical-launch-sequence). If the cameras are publishing slower than ~20 Hz, read [§Three compounding root causes](#three-compounding-root-causes).

## TL;DR

The vendored launch files by themselves publish image topics at **~3 Hz** when both cameras run together, even over USB 3. Three independent misconfigurations compound and all three have to be addressed; this doc describes the applied fix.

## Config under version control

- [`config/fastdds_shm.xml`](./config/fastdds_shm.xml) — FastDDS profile that prefers SHM over UDP for same-host traffic.
- [`config/realsense_qos.yaml`](./config/realsense_qos.yaml) — flat-format QoS overrides consumed by `rs_launch.py`'s `config_file` arg.

## Canonical launch sequence

```bash
# Put this in ~/.bashrc / ~/.zshrc so every shell — including downstream
# consumers (ros2 topic hz, detection / tracking nodes, get_point_cloud) —
# inherits it. Both publisher and subscriber must have it set to negotiate SHM.
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/tinker/tk25_ws/src/tk26_vision/config/fastdds_shm.xml

# terminal 1 — RealSense
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=xarm_camera \
    align_depth.enable:=true \
    config_file:=/home/tinker/tk25_ws/src/tk26_vision/config/realsense_qos.yaml

# terminal 2 — Orbbec Femto Bolt
ros2 launch orbbec_camera femto_bolt.launch.py \
    depth_registration:=true \
    enable_ir:=false \
    enable_frame_sync:=false
```

Expected rates on `/camera/xarm_camera/color/image_raw`, `/camera/color/image_raw`, and their `depth` siblings: **~30 Hz with σ ≤ 5 ms**. Verify with `ros2 topic hz`.

## Three compounding root causes

Each independently throttles rates; all had to be fixed.

### 1. RealSense D435 must enumerate as USB 3

Check:
```bash
lsusb -v -d 8086:0b07 | grep bcdUSB
# want: bcdUSB 3.10 (or higher)
# bad:  bcdUSB 2.10  ← USB 2 High-Speed, 480 Mbps
```

USB 2 High-Speed practical throughput (~35–45 MB/s) is below the driver's default 848×480 color + depth @ 30 fps bandwidth (~62 MB/s). Causes are usually (a) plugged into a USB 2 port, (b) USB 2-only cable (common with generic USB-C ↔ USB-A leads), or (c) bad contact.

**Fix:** move to a blue / SS USB 3 port; use the SuperSpeed cable that ships with the D435. Verify `lsusb -t` shows the D435 at `5000M` (or higher). On this workstation, both cameras now share Bus 04's 5 Gbps root hub.

### 2. realsense-ros publishes images with `Durability: TRANSIENT_LOCAL`

`thirdparty/realsense-ros/realsense2_camera/include/constants.h:83` defaults:

```cpp
const std::string IMAGE_QOS = "SYSTEM_DEFAULT";
```

which FastDDS resolves to RELIABLE + TRANSIENT_LOCAL + KEEP_LAST(1) — a profile meant for latched topics (static TF, maps), not 1.2 MB frames at 30 fps. The driver exposes runtime `color_qos`, `depth_qos`, `infra{1,2}_qos`, `color_info_qos`, `depth_info_qos` parameters but they are **not** in `configurable_parameters` in `rs_launch.py` — so `color_qos:=DEFAULT` on the command line is silently dropped.

**Fix:** supply the overrides via `config_file:=config/realsense_qos.yaml`. That yaml must be flat key/value (see gotchas below).

Verify with:
```bash
ros2 topic info -v /camera/xarm_camera/color/image_raw | grep Durability
# want: Durability: VOLATILE
```

### 3. Kernel UDP receive buffer is 208 KB by default

```bash
sysctl net.core.rmem_max
# Ubuntu 22.04 ships: net.core.rmem_max = 212992   ← 208 KB
```

ROS2 Humble's default RMW (`rmw_fastrtps_cpp`, FastDDS 2.6.11) fragments 1.2 MB image messages into many UDP datagrams. A 208 KB socket buffer overflows; `grep ^Udp: /proc/net/snmp` confirmed `RcvbufErrors` accumulating at ~1.2 k/s per camera-subscriber pair while running. This reproduces even with only one camera running (not a USB contention issue), and also explains the strange symptom where RealSense aligned-depth (smaller frames) was received at 15 Hz while color was at 3 Hz.

Two mutually-compatible fixes:

- **No sudo (what we ship):** FastDDS profile at `config/fastdds_shm.xml` that uses SHM (shared memory) transport for same-host traffic, bypassing the UDP socket path entirely. Activated by `export FASTRTPS_DEFAULT_PROFILES_FILE=…`. Both producer and consumer must have this set to negotiate SHM.
- **With sudo (the ROS2-official fix):** raise the kernel UDP buffer system-wide. One-time setup:
  ```bash
  sudo tee /etc/sysctl.d/60-ros2-udp.conf <<'EOF'
  net.core.rmem_max=8388608
  net.core.rmem_default=8388608
  net.core.wmem_max=8388608
  net.core.wmem_default=8388608
  EOF
  sudo sysctl --system
  ```
  With this, stock FastDDS (no XML profile) also behaves. The SHM profile becomes a perf choice rather than a correctness requirement.

## Secondary Orbbec-only knobs

In the vendored `thirdparty/OrbbecSDK_ROS2/orbbec_camera/launch/femto_bolt.launch.py`:

| Arg | Default | Recommended | Why |
|---|---|---|---|
| `enable_frame_sync` | `true` (line 80) | `false` | Ties color to the slowest stream inside the SDK — any depth stall dragged color down with it. See [§Is color/depth still synced without `enable_frame_sync`?](#is-colordepth-still-synced-without-enable_frame_sync) |
| `enable_ir` | `true` (line 52) | `false` | No tk26_vision node subscribes to IR; dropping it saves USB bandwidth and a decode thread. |
| `align_mode` | `SW` (line 89) | `SW` (don't touch) | `HW` is **not** usable with the default 1280×720 MJPG color + 640×576 Y16 depth profile — driver logs `Failed to start pipeline: Current stream profile is not support hardware d2c process` and resets. Stick with software alignment. |

## Is color/depth still synced without `enable_frame_sync`?

Yes. `enable_frame_sync` only controls SDK-side frame *pairing* before publish — it does **not** assign the stamps. Every frame carries its hardware capture time regardless, and `ApproximateTimeSynchronizer(slop=0.05)` (what `get_point_cloud` uses) pairs stamps post-hoc.

Measured on this workstation with `enable_frame_sync:=false`, 301 color frames ↔ 302 depth frames over 10 s:

| `|Δ stamp|` | value |
|---|---|
| median | 1.0 ms |
| p95 | 2.0 ms |
| max | 2.4 ms |
| within ±50 ms (ApproxTimeSync slop) | 100% |

Drift is a tiny constant offset (color ~1 ms before depth), not drift over time. Every consumer that does time-synchronization still works correctly.

## Gotchas

- `rs_launch.py` **silently ignores** arbitrary `<name>:=<value>` command-line args like `color_qos:=DEFAULT` because they're not in its `configurable_parameters` list. The yaml `config_file` route is the only one that works.
- `rs_launch.py` loads the `config_file` yaml with plain `yaml.SafeLoader` and passes the resulting dict directly to `Node(parameters=[...])`. The yaml must be **flat key/value**, not the usual `/**: ros__parameters:` form — the shipped `realsense_qos.yaml` is already in the correct format; don't reformat it.
- Both publisher and subscriber processes need `FASTRTPS_DEFAULT_PROFILES_FILE` set to negotiate SHM. If you set it on the camera launches but not on `ros2 topic hz`, the hz tool falls back to UDP and reports the pre-fix rate — check the env before suspecting the config.

## Verification

60-second soak, both cameras running with the config above:

| Topic | Mean rate | σ | Max interval |
|---|---|---|---|
| `/camera/xarm_camera/color/image_raw` | 29.97 Hz | 3.0 ms | 139 ms |
| `/camera/xarm_camera/aligned_depth_to_color/image_raw` | 29.63 Hz | 4.5 ms | 167 ms |
| `/camera/color/image_raw` (Orbbec) | 30.17 Hz | 1.2 ms | 43 ms |
| `/camera/depth/image_raw` (Orbbec) | 30.17 Hz | 1.2 ms | 42 ms |

UDP `InErrors` delta over the 60 s window: 487, down from >20 k in the same interval before the fix. Remaining errors are FastDDS discovery chatter, not image data.

## Follow-ups (not required for the fix)

- **Wrapper launch file** under `src/tk26_vision/launch/cameras_bringup.launch.py` that sets the env var and composes both camera launches with the overrides. Would let teammates run one command.
- **CycloneDDS** (`apt install ros-humble-rmw-cyclonedds-cpp`, `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`) — handles large messages without the XML profile. Not installed on this box.

## References

- `thirdparty/realsense-ros/realsense2_camera/include/constants.h:83` — `IMAGE_QOS = "SYSTEM_DEFAULT"` (the TRANSIENT_LOCAL source).
- `thirdparty/realsense-ros/realsense2_camera/src/profile_manager.cpp:298` — registers per-sensor `*_qos` runtime parameter.
- `thirdparty/realsense-ros/realsense2_camera/launch/rs_launch.py:99-119` — how `config_file` yaml is loaded (flat dict only).
- `thirdparty/OrbbecSDK_ROS2/orbbec_camera/launch/femto_bolt.launch.py:52,80,89` — `enable_ir`, `enable_frame_sync`, `align_mode` defaults.
- `DEV_NOTES.md` §"2026-04-22 — Camera bringup performance fix" — the diagnostic narrative behind this doc.
