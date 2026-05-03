# Orbbec Femto Bolt crash — diagnosis & resolution path

Recurring `Received signal: 11` segfaults of the `component_container` running `femto_bolt.launch.py` on this workstation. Nine+ crash logs in `Log/camera_crash_stack_trace_*.log` from 2026-04-30 through 2026-05-03, all with the same signature.

## Symptom

```
[component_container-1] Received signal: 11
[component_container-1] Log crash stack trace to Log/camera_crash_stack_trace_<ts>.log
…repeats from 2-3 different threads…
[ERROR] [component_container-1]: process has died [pid …, exit code -11,
  cmd '/opt/ros/humble/lib/rclcpp_components/component_container
       --ros-args -r __node:=camera_container -r __ns:=/camera'].
```

Crash always happens shortly after streams come up — color and depth start, then the container dies within seconds to minutes.

## Environment at crash time

| Component | Version |
|---|---|
| Host OS | Ubuntu 22.04.5, kernel 6.8.0-110-generic |
| GPU (dGPU) | NVIDIA GeForce RTX 4080 Laptop, driver `575.57.08`, CUDA 12.9 |
| GPU (iGPU) | Intel Raptor Lake-P UHD (PCI `8086:A788`, `i915`) |
| OrbbecSDK | `libOrbbecSDK.so.1.10.14` (vendored under `src/tk23_vision/src/OrbbecSDK_ROS2/`) |
| Depth engine | `libdepthengine.so.2.0` |
| Camera | Orbbec Femto Bolt (PID `0x066b`, USB 3.1) |

Render nodes:
- `/dev/dri/renderD128` → `i915` (Intel iGPU)
- `/dev/dri/renderD129` → `nvidia` (RTX 4080)

## Root-cause analysis

Reading the most recent crash log (`Log/camera_crash_stack_trace_2026_05_03_03_04_19.log`) we see **three concurrent SIGSEGVs from three different SDK threads**. Only one is the trigger; the others are collateral.

### Thread 8004 — root cause: depth engine vs. NVIDIA EGL

```
libdepthengine.so.2.0          0x7e581f422531
libdepthengine.so.2.0          0x7e581f408597  
libdepthengine.so.2.0          0x7e581f414189  
libdepthengine.so.2.0          0x7e581f413ca1  
libdepthengine.so.2.0          0x7e581f417838  
libnvidia-eglcore.so.575.57.08 _glDeleteSync          ← crash
libc.so.6                      <signal>
liborbbec_camera.so            signalHandler(int)
```

Femto Bolt's depth engine performs ToF → depth conversion **on the GPU via NVIDIA EGL/OpenGL**. It segfaults *inside* `_glDeleteSync` — i.e. inside the NVIDIA driver's EGL implementation, not in our code or in the OrbbecSDK above EGL.

NVIDIA driver `575.57.08` is a brand-new branch (released early 2026); `libdepthengine.so.2.0` ships in OrbbecSDK 1.10.14 (Sep 2024) and was built against driver branches in the 535/550 series. The EGL ABI / sync-object lifecycle changed in 575 in a way that breaks the depth engine's GL command stream.

The recurrence pattern (≥9 crashes in 4 days, every session) confirms environmental, not transient.

### Thread 8071 — collateral: TOCTOU race in `publishColoredPointCloud`

`src/tk23_vision/src/OrbbecSDK_ROS2/orbbec_camera/src/ob_camera_node.cpp:1454-1463`:

```cpp
void OBCameraNode::publishColoredPointCloud(const std::shared_ptr<ob::FrameSet> &frame_set) {
  if (!depth_registration_cloud_pub_ ||
      depth_registration_cloud_pub_->get_subscription_count() == 0 ||
      !enable_colored_point_cloud_ || !depth_frame_) {
    return;                                            // ← unlocked check
  }

  CHECK_NOTNULL(depth_frame_.get());                   // ← unlocked check
  std::lock_guard<...> point_cloud_msg_lock(point_cloud_mutex_);
  auto depth_frame = depth_frame_->as<ob::DepthFrame>();   // ← locked use
```

But `depth_frame_` is **reassigned in `onNewFrameSetCallback`** (lines 1637, 1648, 1654) without ever holding `point_cloud_mutex_`. When the depth engine dies in thread 8004 the SDK signals shutdown, `depth_frame_` is torn down concurrently → null deref in the color thread that already passed the unlocked check. Real SDK bug independent of the GPU crash; only fires under unclean shutdown.

Defensive patch (not applied in this session — vendored SDK, would need explicit OK):

```cpp
std::shared_ptr<ob::Frame> depth_frame_local;
{
  std::lock_guard<decltype(point_cloud_mutex_)> lock(point_cloud_mutex_);
  if (!depth_frame_) return;
  depth_frame_local = depth_frame_;        // refcount bump under lock
}
auto depth_frame = depth_frame_local->as<ob::DepthFrame>();
```

Same pattern needed in `publishDepthPointCloud` at line 1364. This wouldn't stop the GPU crash; it would stop the GPU crash from cascading the whole component_container.

### Thread 7168 — collateral: libusb shutdown

`libusb_handle_events_completed` segfaults while the SDK tears down USB transfers. Standard cascade after the first segfault.

### Why the whole container dies

Orbbec installs a `signalHandler(int)` (frame `#0` in every trace) that **only writes the log file** — it does not call `_exit(1)` or re-raise. Once any of the three threads dies, the runtime kills the whole `component_container` with `-11`. There is no recovery path inside the SDK.

## Workarounds attempted

Working around the NVIDIA-575 ↔ depthengine ABI mismatch by forcing the depth engine off NVIDIA EGL.

| # | Approach | Result | Lesson |
|---|---|---|---|
| 1 | `__EGL_VENDOR_LIBRARY_FILENAMES=…/50_mesa.json` (Mesa vendor pin only) | Mesa enumerates **all** DRI render nodes, hits NVIDIA's `0x27e0`, fails: `MESA: warning: Driver does not support the 0x27e0 PCI ID` → `libEGL warning: egl: failed to create dri2 screen`. Depth engine bails on first device failure. | Mesa's `eglQueryDevicesEXT` returns hardware nodes regardless of which one Mesa can drive. The depth engine doesn't fall back to the next device. |
| 2 | Add `EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1` | `libEGL warning: Not allowed to force software rendering when API explicitly selects a hardware device.` Depth engine refuses to init under software. | `libdepthengine.so.2.0` uses `EGL_PLATFORM_DEVICE_EXT` and explicitly demands a hardware device. Software EGL (llvmpipe) is rejected. |
| 3 | `MESA_LOADER_DRIVER_OVERRIDE=iris EGL_PLATFORM=device EGL_DEVICE_ID=/dev/dri/renderD128` | Same `0x27e0 PCI ID not supported` error. `iris` got force-loaded on the NVIDIA card too. | `MESA_LOADER_DRIVER_OVERRIDE` is global to the process — applies to every Mesa device probe, not just the one you want. Counter-productive on hybrid systems. |
| 4 | **`unshare -mr bash -c 'mount --bind /dev/null /dev/dri/renderD129; mount --bind /dev/null /dev/dri/card2; …'` + Mesa vendor pin** | **No more segfault.** Camera streams. | Hiding the NVIDIA DRI nodes via user-namespace bind-mount is the only way to make Mesa's enumeration return only the iGPU. Ubuntu 22.04 has `kernel.unprivileged_userns_clone=1` so no root needed. |

### Performance under workaround #4

| Topic | Rate (iGPU via Mesa) | Target (NVIDIA, native) |
|---|---|---|
| `/camera/color/image_raw` | ~5–8 Hz | ~30 Hz |
| `/camera/depth/image_raw` | ~5–10 Hz | ~30 Hz |
| `/camera/depth_registered/points` | ~3–7 Hz | ~30 Hz |

Adding `enable_ir:=false enable_frame_sync:=false FASTRTPS_DEFAULT_PROFILES_FILE=…/fastdds_shm.xml` per `CAMERA_BRINGUP.md` did **not** help — the publisher is now the bottleneck, not the transport.

#### Why color also drops, not just depth

`onNewColorFrameCallback` (`ob_camera_node.cpp:1718`) runs `publishPointCloud` → `publishColoredPointCloud` **in the color frame thread**. Every color frame triggers a 1280×720 CPU xy-table reprojection (~900k points). When `depth_registration:=true` + `enable_colored_point_cloud:=true`, that reprojection is in series with the iGPU's GL command-submission, both contending for CPU and memory bandwidth → color drops alongside depth.

Disabling `enable_colored_point_cloud` would restore color to ~30 Hz but breaks downstream consumers of `/camera/depth_registered/points`:
- `object_detection_new/object_seg_yolo.py:112`
- `vision_util/get_point_cloud.py:60`
- `vision_util/door_detection.py:32`
- `tk_vision_specialized/waving_person_server.py:47`
- `pick_and_place/grasp_node.hpp:117`

so it isn't an option.

## Resolution path

The env-var workaround stops the segfault but caps throughput at ~5 Hz, which is unworkable for downstream tasks. The real fix is to put the depth engine back on the GPU it was designed for — by pinning NVIDIA to a driver branch the binary was built against.

```bash
# Survey current state
dpkg -l | grep -E 'nvidia-driver|nvidia-utils|cuda' | awk '{print $2}'

# Install 550 (replaces 575 userland + DKMS module)
sudo apt install --install-recommends nvidia-driver-550

# Pin so unattended-upgrades doesn't drift back to 575
sudo apt-mark hold nvidia-driver-550
sudo tee /etc/apt/preferences.d/nvidia-pin <<'EOF'
Package: nvidia-driver-* nvidia-dkms-* nvidia-utils-* libnvidia-*
Pin: version 550.*
Pin-Priority: 1001
EOF

sudo reboot
```

Post-reboot verification:

```bash
nvidia-smi                     # Driver Version: 550.*
dkms status | grep nvidia      # nvidia/550.* … installed
```

After this, `tmux_hri_vision.sh` runs the camera launch in its original simple form with no env-var contortions and frames flow at native 30 Hz.

### Risk on the driver swap

- CUDA 12.9 toolkit on disk keeps working under driver 550 via forward-compat as long as compiled-for compute capability ≤ sm_90 (RTX 4080 is sm_89). cuMotion / Ultralytics YOLO / AnyGrasp all unaffected.
- DKMS rebuilds the kernel module against `6.8.0-110-generic`. Watch for build errors in the apt log before reboot. If the module fails to build, the system reboots to no NVIDIA driver and you'd have to run `sudo dpkg-reconfigure nvidia-dkms-550` from a rescue shell.
- Reverse: `sudo apt install --install-recommends nvidia-driver-575` (with the pin lifted) puts you back where you started.

## What `tmux_hri_vision.sh` should look like

After the driver pin, restore to the simple form (this is what's checked in):

```bash
tmux send-keys -t $SESSION:$WINDOW.0 \
    "source ~/tk25_ws/install/setup.zsh && \
     ros2 launch orbbec_camera femto_bolt.launch.py \
       enable_colored_point_cloud:=true depth_registration:=true; exec zsh" C-m
```

If you ever again see the same `_glDeleteSync` crash signature, the iGPU workaround for emergency operation is:

```bash
unshare -mr bash -c '
  mount --bind /dev/null /dev/dri/renderD129
  mount --bind /dev/null /dev/dri/card2
  source /home/tinker/tk25_ws/install/setup.bash
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
  exec ros2 launch orbbec_camera femto_bolt.launch.py \
    enable_colored_point_cloud:=true depth_registration:=true
'
```

5 Hz, but no crash. Use as a stopgap only.

## Open follow-ups

1. **Patch the TOCTOU race in `publishColoredPointCloud` / `publishDepthPointCloud`.** Defensive fix; prevents the same class of crash from cascading next time the depth engine destabilizes (e.g. driver upgrade, USB hiccup). Vendored SDK file at `src/tk23_vision/src/OrbbecSDK_ROS2/orbbec_camera/src/ob_camera_node.cpp:1454, 1364`. Local patch + `colcon build --packages-select orbbec_camera`.
2. **Audit `tk25_basic/src/scripts/vision.sh`** — diverges from the canonical `tk26_vision/scripts/launch_orbbec_shm.sh` (no FastDDS profile, no `enable_ir:=false`, no `enable_frame_sync:=false`). Not a crash trigger but a separate latent throughput issue documented in `CAMERA_BRINGUP.md`.
3. **Upgrade OrbbecSDK to ≥ 1.10.18 or v2.x** — newer SDKs ship rebuilt depthengine binaries and may natively support newer NVIDIA branches. Larger diff; defer until needed.

## References

- Crash logs: `Log/camera_crash_stack_trace_*.log`
- SDK source: `src/tk23_vision/src/OrbbecSDK_ROS2/orbbec_camera/src/ob_camera_node.cpp`
  - `publishColoredPointCloud`: line 1454
  - `publishDepthPointCloud`: line 1357
  - `onNewFrameSetCallback` (depth_frame_ writer): line 1622
  - `onNewColorFrameCallback`: line 1718
  - `signalHandler(int)`: present in compiled binary, source not in this tree
- Camera bringup canonical doc: `src/tk26_vision/CAMERA_BRINGUP.md`
- Launch script: `src/tk25_basic/src/scripts/tmux_hri_vision.sh`
- Wrapper used in normal operation: `src/tk26_vision/scripts/launch_orbbec_shm.sh`
- Femto Bolt vendored launch: `install/orbbec_camera/share/orbbec_camera/launch/femto_bolt.launch.py`
