# FoundationStereo integration into tk26_vision — design

**Status**: spec
**Date**: 2026-05-24
**Author**: Cindy + Claude
**Scope**: New ROS2 package `foundation_stereo` that integrates the existing
`dualrRGB-foundationStereo` runtime into `tk26_vision`, with two modes —
on-request (service + action) and continuous streaming (publishes depth like
a depth camera).

Reference setup lives at `/home/tinker/projects/vision_tests/dualrRGB-foundationStereo/`.
This spec adapts its `StereoRunner` and `LiveInferenceWorker` patterns into a
single ROS2 node, drops the webapp/Open3D/Plotly dependencies, and adds
topic-driven input plus optional D435 color alignment.

---

## 1. Goals & non-goals

**Goals**
- Run FoundationStereo (any of `vitl`, `vits`, `fast_fp32`, `fast_fp16`, `fast_trt`)
  inside the tk26_vision ROS2 graph.
- On-request mode: service + action sharing one `StereoRunner`, with per-call
  overrides for model, scale, iters, TRT variant, and target frame.
- Streaming mode: continuous depth publication that mimics the realsense
  driver's `aligned_depth_to_color/image_rect_raw` topic shape (16UC1 mm by
  default, `SensorDataQoS`, camera_info alongside).
- Topic-driven input (no direct camera ownership) restricted to the only two
  cameras this node will see in practice: **D435** and **D405**.
- Optional D435 reprojection of FS depth from the IR1 grid into the color
  camera grid, so downstream consumers can treat it as a drop-in alternative
  to the driver's native `aligned_depth_to_color/image_rect_raw`.
- All model / resolution / iters / variant choices configurable via launch
  parameters; per-call srv/action fields override the node defaults.

**Non-goals**
- D405 webapp parity (no MJPEG streams, no `/capture`, no PLY/NPY artifacts in
  the action result — debug JPEG only when explicitly requested).
- Camera ownership: the node never opens a camera. It subscribes to image
  topics published by `realsense2_camera` (or any compatible source).
- Bridging the IR1 / left_ir TF-name mismatch automatically. Documented +
  helper static_transform_publisher snippet provided in the README.
- Cross-camera generalisation beyond D435 + D405.

---

## 2. Architecture

New ROS2 package at `src/tk26_vision/src/foundation_stereo/`. Mirrors the
shape of `monocular_depth`: own package, own dedicated venv
(`src/tk26_vision/.venv-fs/`), own build wrapper
(`scripts/build_foundation_stereo.sh`).

Upstream model code is vendored under
`src/tk26_vision/thirdparty/foundation_stereo/`, containing the
`FoundationStereo/` and `Fast-FoundationStereo/` trees with `.git` and
`.venv/` stripped — mirroring the DA3 vendor pattern. Model weights and TRT
engines stay at a configurable `weights_root` (defaulting to
`/home/tinker/projects/vision_tests/dualrRGB-foundationStereo`) so the
workspace doesn't carry the ~3 GB of weights.

```
src/tk26_vision/
├── thirdparty/foundation_stereo/
│   ├── FoundationStereo/            # vendored upstream, no .git/.venv/weights
│   └── Fast-FoundationStereo/       # vendored upstream, no .git/weights
├── src/foundation_stereo/
│   ├── foundation_stereo/
│   │   ├── __init__.py
│   │   ├── stereo_runner.py         # lifted from webapp/stereo_runner.py
│   │   │                            #   - Open3D / Plotly removed
│   │   │                            #   - namespace-swap, variant discovery, lock retained
│   │   │                            #   - InferResult slimmed (disp, depth, optional debug JPEG, timing)
│   │   ├── color_align.py           # IR1-grid depth → color-grid depth (numpy)
│   │   ├── foundation_stereo_node.py # the ROS2 node (entry point)
│   │   └── _logging.py              # tk26 vision_log session hook
│   ├── launch/foundation_stereo.launch.py
│   ├── config/foundation_stereo.yaml
│   ├── requirements.txt
│   ├── package.xml / setup.py / resource/foundation_stereo
│   └── test/                        # T0 import-shape checks
├── .venv-fs/                        # provisioned mirror of FS's torch 2.8 + cu128 + TRT 10.16
└── scripts/build_foundation_stereo.sh
```

Inside the node, one `StereoRunner` is shared by three consumers, serialized
through the runner's existing `_lock`:

1. **Service** `/foundation_stereo/get_depth` — synchronous; reads the latest
   cached synced stereo pair.
2. **Action** `/foundation_stereo/infer_depth` — same shape, plus per-stage
   feedback and cancellation; suited to slow PyTorch backends (~1 s).
3. **Streaming worker** (optional, daemon thread; enabled via launch param) —
   loops `snapshot → infer(live=True) → optionally align → publish`.

Input plumbing: `message_filters.ApproximateTimeSynchronizer` subscribes to a
left + right + left-camera-info triple. The latest synced triple is cached
under a lock and consumed by the three callbacks above. This mirrors the
`get_point_cloud` precedent for sync handling.

---

## 3. Interfaces

### Service — `tinker_vision_msgs_26/srv/FoundationStereoDepth.srv`

```
# Request — per-call overrides (empty string / 0 means use node default)
string model_kind                    # "vitl"|"vits"|"fast_fp32"|"fast_fp16"|"fast_trt"|""
string trt_variant                   # Fast-FoundationStereo/<dir> basename; fast_trt only
float32 scale                        # 0.05..1.0; ignored for fast_trt; 0 = default
int32 iters                          # 0 = backend default
float32 z_far                        # clamp; 0 = default
bool want_pointcloud                 # include PointCloud2 in response
bool want_debug_jpeg                 # include JPEG vis of disparity
bool align_to_color                  # publish/return in color frame instead of IR1 frame
---
# Response
int32 status                         # 0=ok, 1=no_sync, 2=weights_missing, 3=infer_error
string error_msg
sensor_msgs/Image depth_image        # 32FC1 m, full source resolution
sensor_msgs/CameraInfo camera_info   # mirrors input (or color when align_to_color)
sensor_msgs/PointCloud2 pointcloud   # only if want_pointcloud
sensor_msgs/CompressedImage debug_jpeg  # only if want_debug_jpeg
float32 forward_ms                   # GPU-event time of just model.forward
float32 load_s                       # model-load time, 0 if cached
float32 end_to_end_s                 # wall-clock for the whole call
string model_used                    # effective model_kind
string trt_variant_used              # effective TRT variant
```

### Action — `tinker_vision_msgs_26/action/FoundationStereoDepth.action`

Goal and Result mirror the srv request/response field for field. Feedback:

```
string current_stage   # "loading_model" | "running_forward" | "aligning_to_color" | "building_pointcloud" | "encoding_debug"
float32 elapsed_s
```

### Streaming topics (no new message types)

| Topic | Type | Notes |
|---|---|---|
| `~/depth/image_rect_raw` *(raw mode)* | `sensor_msgs/Image` | 16UC1 mm by default; 32FC1 m if `stream_dtype=32FC1_m`. Frame = IR1 optical. |
| `~/depth/camera_info` *(raw mode)* | `sensor_msgs/CameraInfo` | Echo of input left IR `CameraInfo`. |
| `~/aligned_depth_to_color/image_rect_raw` *(align mode)* | `sensor_msgs/Image` | Reprojected into color grid. Frame = color optical. |
| `~/aligned_depth_to_color/camera_info` *(align mode)* | `sensor_msgs/CameraInfo` | Echo of input color `CameraInfo`. |
| `~/debug/disparity/compressed` | `sensor_msgs/CompressedImage` | JPEG vis; only if `stream_publish_vis=true`. |

QoS: `SensorDataQoS` on all output topics — matches the realsense /
orbbec driver convention.

---

## 4. Configuration parameters

### Paths & model defaults

| Param | Default | Notes |
|---|---|---|
| `weights_root` | `/home/tinker/projects/vision_tests/dualrRGB-foundationStereo` | Root of `FoundationStereo/pretrained_models/` + `Fast-FoundationStereo/`. |
| `default_model_kind` | `fast_trt` | One of `vitl|vits|fast_fp32|fast_fp16|fast_trt`. |
| `default_trt_variant` | `output_two_stage` | Auto-discovers any sibling dir with `feature_runner.engine` + `post_runner.engine` + `onnx.yaml`. |
| `default_scale` | `0.5` | Ignored for `fast_trt`. |
| `default_iters` | `0` | 0 = backend default (32 / 32 / 8 / 8 / N/A). |
| `default_z_far` | `10.0` | Meters. |

### Camera-profile preset

| Param | Default | Notes |
|---|---|---|
| `camera_profile` | `d435` | `d435|d405`. Sets default topic strings + baseline; individual topic params override per-key. |

Topic + baseline defaults derived from `camera_profile`:

| Profile | `left_topic` | `right_topic` | `left_info_topic` | `color_info_topic` | `extrinsics_topic` | `baseline_m` |
|---|---|---|---|---|---|---|
| `d435` | `/camera/xarm_camera/infra1/image_rect_raw` | `/camera/xarm_camera/infra2/image_rect_raw` | `/camera/xarm_camera/infra1/camera_info` | `/camera/xarm_camera/color/camera_info` | `/camera/xarm_camera/extrinsics/depth_to_color` | `0.050` |
| `d405` | `/camera/camera/infra1/image_rect_raw` | `/camera/camera/infra2/image_rect_raw` | `/camera/camera/infra1/camera_info` | `/camera/camera/color/camera_info` | `/camera/camera/extrinsics/depth_to_color` | `0.018` |

Baseline source: `baseline_m` if non-zero; else parsed from the latched
extrinsics topic's translation `[0]` (same path `webapp/live_worker.py:101`
uses). For D435 the depth and IR1 profiles are co-located, so
`depth_to_color` ≈ `infra1_to_color`.

### Streaming-mode

| Param | Default | Notes |
|---|---|---|
| `stream_enabled` | `false` | When false, only srv + action are advertised. |
| `sync_slop_sec` | `0.05` | `ApproximateTimeSynchronizer` slop. |
| `sync_queue_size` | `5` | |
| `stream_align_to_color` | `true` | Reproject FS depth from IR1 grid into color grid using IR1→Color extrinsics. False = raw IR1-grid output. |
| `stream_depth_topic` | `~/depth/image_rect_raw` (raw) or `~/aligned_depth_to_color/image_rect_raw` (align) | Auto-flips with `stream_align_to_color`; overridable. |
| `stream_info_topic` | matched to depth topic | Same auto-flip. |
| `stream_dtype` | `16UC1_mm` | Or `32FC1_m`. |
| `output_frame_id` | `''` | Empty = forward input's `frame_id`. Use this to rename `infra1_optical_frame` → `left_ir_optical_frame` etc. |
| `stream_publish_vis` | `false` | If true, publish `~/debug/disparity/compressed` (JPEG SensorDataQoS). |
| `stream_max_fps` | `0.0` | 0 = uncapped. |
| `extrinsics_warmup_timeout_sec` | `5.0` | If `stream_align_to_color=true` and the latched extrinsics topic hasn't arrived within this window, refuse to start the publisher and log a clear error. **Do not silently fall back to identity** — that produces 15 mm-scale wrong depth on D435. |

### Logging

| Param | Default | Notes |
|---|---|---|
| `vision_logging_enabled` | `false` | Default off — synchronous disk IO at fast_trt rates stalls the publisher loop (same rationale as `follow_head`). Flip on for one-shot debug. |
| `vision_log_folder` | `vision_log` | Shared session-dir resolution: `$TINKER_VISION_SESSION_TS` → newest sibling subdir by mtime → fresh `strftime`. |

---

## 5. Color alignment (D435 path)

When `stream_align_to_color=true` (default for D435) or
`align_to_color=true` is set in a srv/action request, the node reprojects FS
depth from the IR1 grid into the color grid. Pure numpy/cv2, no extra deps.

```
Inputs:  depth_ir1     (H_ir, W_ir) float32 meters
         K_ir1         3×3  from left_info_topic
         K_color       3×3  from color_info_topic
         R_ir1_color   3×3  from extrinsics_topic, after librealsense optical→ROS axis fix
         T_ir1_color   3    meters

Steps:   1. Backproject every valid pixel of depth_ir1 → (X, Y, Z) in IR1 frame.
         2. Transform to color frame: P_c = R · P_ir1 + T.
         3. Project: (u_c, v_c) = (fx·X_c/Z_c + cx, fy·Y_c/Z_c + cy).
         4. Round to color pixel grid. Write Z_c into depth_color using
            np.minimum.at — handles occlusion (nearer point wins on collision).
         5. Holes (no IR ray hit) stay zero. No interpolation fill; downstream
            can median-blur or dilate if they want.

Output:  depth_color   (H_color, W_color) float32 m
         frame_id      forwarded from color_info_topic's CameraInfo.frame_id
```

For D405 the IR1→Color translation is ~60 µm so the same code path is a
near-identity transform; correct but mostly redundant. Useful when you want
the color-frame name on the published topic anyway (avoids the URDF name
mismatch — see §6).

`stream_dtype` conversion happens after alignment: `(depth_color * 1000.0).astype(np.uint16)`
for `16UC1_mm`, raw for `32FC1_m`.

---

## 6. Frame IDs, the URDF mismatch, and recommended config

The realsense driver builds frame IDs via `<camera_name>_<stream>_optical_frame`
(`realsense-ros/include/base_realsense_node.h:89`). With `camera_name:=xarm_camera`:

- Header.frame_id on `/camera/xarm_camera/infra1/image_rect_raw`: `xarm_camera_infra1_optical_frame`
- Header.frame_id on `/camera/xarm_camera/infra2/image_rect_raw`: `xarm_camera_infra2_optical_frame`
- Header.frame_id on `/camera/xarm_camera/color/image_raw`: `xarm_camera_color_optical_frame`

The driver also publishes its own self-contained static TF subtree
(`realsense-ros/src/tfs.cpp:210`) connecting these.

The xarm URDF (`xarm_description/urdf/camera/realsense_d435i.urdf.xacro:32`)
declares the IR frames under different names:

- `xarm_camera_left_ir_optical_frame`   ← URDF, left IR
- `xarm_camera_right_ir_optical_frame`  ← URDF, right IR
- `xarm_camera_color_optical_frame`     ← URDF, color (matches driver by coincidence)

**The color name agrees; the IR names do not.** No `static_transform_publisher`
bridges them anywhere in the workspace (verified via grep across `src/`).

**Design consequence:** the FS node does not rewrite frame IDs internally —
it forwards whatever arrives on the input `CameraInfo`, modulo `output_frame_id`.
Two clean ways for users to make TF lookups work against the robot URDF tree:

1. **Recommended: stream aligned-to-color (default).** Output frame becomes
   `xarm_camera_color_optical_frame`, which agrees between driver and URDF —
   downstream TF lookups just work, no bridge required.
2. **If you need raw IR1-grid depth**, either:
   a. Set `output_frame_id:=xarm_camera_left_ir_optical_frame` to rename to
      the URDF convention. Geometrically equivalent — same physical sensor.
   b. Run a static_transform_publisher bridging the two:
      ```bash
      ros2 run tf2_ros static_transform_publisher \
          0 0 0  0 0 0 \
          xarm_camera_left_ir_optical_frame \
          xarm_camera_infra1_optical_frame
      ```
   The README documents both. Aligned-to-color is the recommended default.

---

## 7. Error handling

| Condition | Service / action behavior | Streaming behavior |
|---|---|---|
| No synced stereo frame yet | `status=1, error_msg="no synced stereo frame"` | Log `WARN` once, keep waiting; resume when frames arrive. |
| Stale synced frame (> N×`sync_slop`) | Same as above. | Log `WARN` once per stale window. |
| `weights_root` missing requested model | `status=2, error_msg="weights missing: <path>"` | Log `ERROR`, shut down publisher thread; node stays alive so srv/action can still return clean errors. |
| Inference exception | `status=3, error_msg=<exception>` | Log `EXCEPTION`, skip the frame, continue. |
| `stream_align_to_color=true` but no extrinsics topic within `extrinsics_warmup_timeout_sec` | If action/srv requests `align_to_color`: `status=3, error_msg="extrinsics not available"`. | Refuse to start; log `ERROR`. **No silent identity fallback.** |
| Concurrent calls | All three consumers serialize through `StereoRunner._lock`. Worst case = added latency. | Same. |
| Cold model load on first call | Reported via `load_s` field. | First inference after start absorbs the load; FPS catches up afterward. |

Out-of-scope failure modes (not handled by this node):

- Camera USB enumeration (covered by `CAMERA_BRINGUP.md`).
- Driver dropping below ~10 Hz (sync would stall — same gotcha as
  `get_point_cloud`).

---

## 8. Vendor / venv plan

`thirdparty/foundation_stereo/` mirrors the reference layout minus `.git`,
`.venv`, captures, and large weights. The vendored trees retain `core/`,
`Utils.py`, `scripts/run_demo.py`, the cfg files, and `Fast-FoundationStereo/`'s
ONNX export helpers — everything needed for the runner.

`stereo_runner.py`'s namespace-swap logic (`_swap_namespace`) is carried over
verbatim; it's required because both `FoundationStereo/` and
`Fast-FoundationStereo/` ship a top-level `core/` package and `Utils.py` with
the same module names but different classes.

### `.venv-fs/` provisioning (one-time)

```bash
cd src/tk26_vision
python3.10 -m venv .venv-fs --system-site-packages --symlinks
source .venv-fs/bin/activate
pip install --upgrade pip wheel
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r src/foundation_stereo/requirements.txt
# TensorRT 10.16 — match the reference workstation's TRT version
pip install tensorrt==10.16.1.11
pip freeze > .venv-fs/freeze.lock.txt
```

`src/foundation_stereo/requirements.txt` pins (initial draft, refined during
implementation):

```
numpy==1.26.4               # FS reference venv pin; tk26_vision-main uses 2.2 so isolate
omegaconf
opencv-python-headless
imageio
einops
safetensors
huggingface_hub
pillow
addict
# no open3d, no flask, no plotly — webapp-only deps dropped
```

### Build wrapper

`scripts/build_foundation_stereo.sh` clones the shape of
`scripts/build_monocular_depth.sh`:

1. Source `.venv-fs/bin/activate` + `/opt/ros/humble/setup.bash`.
2. Default to `--packages-select foundation_stereo` if no args given.
3. `colcon build --symlink-install`.
4. Re-shebang `install/foundation_stereo/lib/foundation_stereo/*` to point at
   `.venv-fs/bin/python3` so `ros2 run` resolves FS imports.

---

## 9. Testing

Adds rows to the existing tk26_vision integration suite at `scripts/tests/`:

| Tier | Check | Notes |
|---|---|---|
| T0 | Imports clean under `.venv-fs`; `from foundation_stereo.stereo_runner import StereoRunner` succeeds. | No GPU required. |
| T0 | Vendored `thirdparty/foundation_stereo/` resolves `core.foundation_stereo.FoundationStereo` and `core.foundation_stereo.TrtRunner` after `_swap_namespace`. | |
| T0 | New srv + action build in `tinker_vision_msgs_26` without touching other interfaces. | |
| T1 | Node startup with `stream_enabled=false` advertises both srv and action; clean SIGTERM. | Default mode. |
| T1 | Node startup with `stream_enabled=true` against a recorded bag opens publishers and processes ≥1 frame. | Bag-based, no live cameras. |
| T1 | Missing-`weights_root` → srv returns `status=2`; streaming worker exits with `ERROR` but node stays alive. | Negative path. |
| T2 | Against live D435: srv returns a non-empty depth image in IR1 frame. | Requires `CAMERA_BRINGUP.md` setup. |
| T2 | Live D435 + `stream_align_to_color=true`: `~/aligned_depth_to_color/image_rect_raw` publishes at ≥10 Hz on `fast_trt`. | |
| T3 | Compare FS aligned depth to driver's `/camera/xarm_camera/aligned_depth_to_color/image_raw` on the same scene; expect per-pixel MAE ≤ a few cm in non-occluded regions. | Loose threshold — different stereo algorithms. |
| T3 | Per-call srv override (`model_kind`, `scale`) takes effect — response `model_used` / `trt_variant_used` echoes the chosen values, `forward_ms` differs from the cached default. | |
| Synthetic | Color-align unit test: synthesize a depth image + known extrinsics, verify reprojected depth lands at the expected color pixels. | `pytest src/foundation_stereo/test/`. |

---

## 10. Out-of-scope / follow-ups

- TRT engine compilation (`make_onnx.py` + `trtexec`). Use the engines already
  built in the reference setup. Adding a build helper is a follow-up.
- Multi-camera support (one node per camera profile is fine; a second
  launch-file invocation suffices).
- Reprojection to arbitrary frames (only color frame is supported; arbitrary
  TF-frame projection is downstream's job via `depth_image_proc`).
- Auto-bridge static TF publisher for the `infra1_optical_frame` ↔
  `left_ir_optical_frame` mismatch. Documented + recommended-default-config
  workaround instead. If the URDF is fixed, this becomes moot.
- A REST/HTTP surface like the reference webapp. The ROS2 srv/action covers
  the same use cases inside the robot graph.
