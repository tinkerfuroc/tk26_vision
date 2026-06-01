# foundation_stereo

ROS2 node that serves NVIDIA **Fast-FoundationStereo** (TensorRT engines) over
ROS service + action + streaming interfaces. Takes a synchronised IR1/IR2
stereo pair from a RealSense (D435 / D405) and produces depth, optionally
aligned to the colour grid.

The node serves **only `fast_trt`** model_kind. PyTorch (`vitl`, `vits`,
`fast_fp32`, `fast_fp16`) are loaded by `StereoRunner` internally but
intentionally rejected at the request layer — see [§ TRT-only enforcement](#trt-only-enforcement).

| Mode | Triggered by | Use case |
|---|---|---|
| Service `/foundation_stereo/get_depth` | a client call | one-shot depth, lowest-latency way to get a single result |
| Action `/foundation_stereo/infer_depth` | a client goal | one-shot depth + per-stage feedback + cancellation |
| Streaming publisher | `stream_enabled:=true` | continuous depth at the IR frame rate, mimics realsense `aligned_depth_to_color/image_rect_raw` |

## Why a separate venv

`torch==2.8.0+cu128` + `tensorrt==10.16.1.11` conflict with the versions in
the shared `.venv-vision-main`. This package builds and runs under
`src/tk26_vision/.venv-fs/`. The `scripts/build_foundation_stereo.sh`
wrapper sources `.venv-fs` and re-shebangs the install entry-point to
that venv's interpreter so `ros2 run`/`ros2 launch` picks the right Python
automatically.

### Provisioning `.venv-fs` (one-time)

```bash
cd src/tk26_vision
python3.10 -m venv .venv-fs --system-site-packages --symlinks
source .venv-fs/bin/activate
pip install --upgrade pip wheel
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r src/foundation_stereo/requirements.txt
pip install tensorrt==10.16.1.11
pip freeze > .venv-fs/freeze.lock.txt
```

Critical: `.venv-fs/lib/python3.10/site-packages/numpy` **must be 1.x**
(currently `1.26.4`). The system `cv_bridge.boost` is compiled against
NumPy 1.x and segfaults on import if NumPy 2.x is present. The lock file
is the source of truth; diff against it before any further `pip install`.

## Vendored tree resolution (both build modes)

The TRT/PyTorch model code lives in the vendored tree at
`src/tk26_vision/thirdparty/foundation_stereo/{FoundationStereo,Fast-FoundationStereo}/`,
which is **outside** this ROS package and is **never copied into the install
tree**. `stereo_runner.py` locates it at runtime via three anchors, in order:

1. `$FOUNDATION_STEREO_VENDOR_ROOT` — explicit override (must contain
   `Fast-FoundationStereo/` and `FoundationStereo/`). Use this for any
   non-standard layout.
2. `__file__`-relative walk (`../../../thirdparty/foundation_stereo`) — hits
   when running from source or from a `--symlink-install` tree (the egg-link
   makes `__file__` resolve back into `src/`).
3. Ancestor scan for `<ws>/src/tk26_vision/thirdparty/foundation_stereo` —
   the fallback for a **copied** (non-symlink) install, where anchor #2 lands
   inside `install/` and finds nothing.

This is why the package works under **both** build paths:

* `scripts/build_foundation_stereo.sh` — `--symlink-install` (anchor #2).
* `tkbuild tk26_vision --packages-select foundation_stereo` — `tkbuild`
  strips `--symlink-install`, producing a copied install resolved by anchor
  #3. `foundation_stereo` is mapped to `.venv-fs` via tkbuild's
  `PER_PKG_VENV_BY_WS`, so the entry-point shebang is patched correctly.

If you ever see `warmup FAILED … No module named 'core'`, anchor resolution
failed — set `FOUNDATION_STEREO_VENDOR_ROOT` or rebuild with one of the two
wrappers above (a bare `colcon build` from an unexpected CWD can defeat the
ancestor scan).

## Build & launch

```bash
# Either wrapper works; tkbuild is the workspace-wide convention.
./src/tk26_vision/scripts/build_foundation_stereo.sh
# or:
tkbuild tk26_vision --packages-select foundation_stereo
source install/setup.bash

# Default: service + action only, no streaming, warmup at launch.
ros2 launch foundation_stereo foundation_stereo.launch.py

# Streaming, aligned to colour (recommended for D435).
ros2 launch foundation_stereo foundation_stereo.launch.py \
    stream_enabled:=true stream_align_to_color:=true

# Faster dev iteration — skip the ~3-5 s warmup.
ros2 launch foundation_stereo foundation_stereo.launch.py \
    -p warmup_on_launch:=false
```

The "foundation_stereo ready: ..." log line is only emitted **after** the
default TRT variant has been loaded onto GPU and exercised once, so
subsequent requests start from warm state (~25 ms forward, `load_s=0`).
See [§ Startup warmup](#startup-warmup).

## Architecture

```
src/foundation_stereo/foundation_stereo/
├── foundation_stereo_node.py   # The ROS2 node — params, subs, srv, action, stream loop.
├── stereo_runner.py            # StereoRunner: loads + caches TRT engines; runs inference.
├── color_align_rs2.py          # RealsenseAligner: IR1→colour alignment via librealsense
│                               # rs.align + software_device. PRIMARY alignment path.
├── color_align_legacy.py       # Naive forward-warp `reproject_ir_to_color`. Kept for
│                               # non-RealSense sources or as a debugging fallback.
└── _logging.py                 # Shared vision_log session-dir resolution.

src/foundation_stereo/
├── config/foundation_stereo.yaml   # Default param values.
├── launch/foundation_stereo.launch.py  # Sources config yaml + accepts CLI overrides.
└── README.md                    # This file.
```

## Subscribed topics (under the camera profile's namespace)

| Topic | Type | Purpose |
|---|---|---|
| `infra1/image_rect_raw` | `sensor_msgs/Image` (Y8) | Left IR. `message_filters` syncs with infra2 + camera_info. |
| `infra2/image_rect_raw` | `sensor_msgs/Image` (Y8) | Right IR. |
| `infra1/camera_info` | `sensor_msgs/CameraInfo` | K_ir + distortion `D` (used by `RealsenseAligner`). |
| `color/camera_info` | `sensor_msgs/CameraInfo` | K_color + distortion `D` (used by `RealsenseAligner`). Latched-style subscriber. |
| `extrinsics/depth_to_color` | `realsense2_camera_msgs/Extrinsics` | Latched (RELIABLE + TRANSIENT_LOCAL). `R, T` from IR1 (=depth on D435/D405) to colour. |

The triple `(infra1, infra2, infra1/camera_info)` is gated by an
`ApproximateTimeSynchronizer` with the params `sync_slop_sec` and
`sync_queue_size`. Default 50 ms slop + queue=5, comfortable at 30 Hz.

## Service: `/foundation_stereo/get_depth`

Interface: `tinker_vision_msgs_26/srv/FoundationStereoDepth`

### Request

| Field | Type | Default behaviour | Notes |
|---|---|---|---|
| `model_kind` | `string` | empty = `"fast_trt"` | **Only `"fast_trt"` is accepted.** Anything else → status=4 immediately, no inference. |
| `trt_variant` | `string` | empty = `default_trt_variant` param | Per-call override of which compiled engine to use (e.g. `"output_trt_20-30-48_576x960_iters4"`). Per-call overrides pay cold-load cost on first use. |
| `scale` | `float32` | 0 = `default_scale` param | Pre-resize factor before the model. Ignored by `fast_trt` (engine input shape is baked into the engine). |
| `iters` | `int32` | 0 = `default_iters` param | Per-backend iteration count override. Ignored by `fast_trt`. |
| `z_far` | `float32` | 0 = `default_z_far` param | Depth clamp in metres. Pixels with greater Z are written as 0 (invalid). |
| `want_pointcloud` | `bool` | false | Emit `sensor_msgs/PointCloud2` alongside the depth image. |
| `want_debug_jpeg` | `bool` | false | Emit a `sensor_msgs/CompressedImage` with a turbo-colourised disparity preview. |
| `align_to_color` | `bool` | false | If true, output depth is in the colour optical frame via `RealsenseAligner`. If false, output depth is in the IR1 optical frame at the engine's scaled resolution. |

### Response

| Field | Type | Notes |
|---|---|---|
| `status` | `int32` | 0 = success. See [§ Status codes](#status-codes). |
| `error_msg` | `string` | Human-readable failure reason when `status != 0`. Empty on success. |
| `depth_image` | `sensor_msgs/Image` | `32FC1` metres. Frame_id depends on `align_to_color`. Resolution = colour (if aligned) or IR1×scale (if not). Zero pixels = invalid (hole, clipped past z_far, or out of bounds). |
| `camera_info` | `sensor_msgs/CameraInfo` | Matches `depth_image` (i.e. colour camera_info when aligned, IR1's at the engine's scale otherwise). |
| `pointcloud` | `sensor_msgs/PointCloud2` | Populated only when `want_pointcloud=true`. Same frame_id as `depth_image`. |
| `debug_jpeg` | `sensor_msgs/CompressedImage` | Populated only when `want_debug_jpeg=true`. |
| `forward_ms` | `float32` | CUDA-events-measured model forward pass duration. ~22 ms warm on D435. |
| `load_s` | `float32` | Wall-clock cost of loading the requested TRT variant. `0.0` if cached (the default after warmup). |
| `end_to_end_s` | `float32` | Wall-clock from request received to response built. Includes cv_bridge, inference, alignment, point-cloud build. |
| `model_used` | `string` | Always `"fast_trt"` on success. |
| `trt_variant_used` | `string` | Which engine actually ran (echoes the requested or default variant). |

### Status codes

| `status` | Meaning |
|---|---|
| 0 | Success. |
| 1 | No synced stereo frame available yet (no IR1+IR2 published, or sync hasn't fired). |
| 3 | Internal error during preprocessing (cv_bridge failure, or `align_to_color=true` requested but `extrinsics` / `color_info` haven't arrived yet). |
| 4 | Invalid request — `model_kind` is non-empty and not `"fast_trt"`. |

## Action: `/foundation_stereo/infer_depth`

Interface: `tinker_vision_msgs_26/action/FoundationStereoDepth`

Goal + Result fields are **identical** to the service request + response.
The action additionally provides:

* **Feedback** (`current_stage`, `elapsed_s`): emitted at each phase
  boundary — `"running_forward"`, `"aligning_to_color"`,
  `"building_pointcloud"`, etc. Use for progress UI.
* **Cancellation**: a goal can be cancelled before inference completes;
  the goal handler checks `goal_handle.is_cancel_requested` before
  starting.

Use this instead of the service when you want progress visibility or
the ability to abort a long request.

## Streaming mode

When `stream_enabled:=true`, the node spawns a worker thread that, on
every new synced stereo frame:

1. Runs inference (always with the node's default model kind / variant /
   scale / iters / z_far).
2. Optionally aligns to colour via `RealsenseAligner` (gated by
   `stream_align_to_color`).
3. Publishes the depth image + camera_info on `stream_depth_topic` /
   `stream_info_topic`. QoS reliability is set by `stream_qos_reliability`
   (default `reliable`); see the note below.

> **QoS gotcha.** The stream defaults to **RELIABLE** so it's a drop-in for
> realsense `aligned_depth_to_color` and appears in default-QoS RViz. If you
> set `stream_qos_reliability:=best_effort`, remember a **RELIABLE subscriber
> cannot receive from a BEST_EFFORT publisher** — RViz/rqt will then show
> nothing unless you switch the display's Reliability Policy to *Best Effort*
> (and `ros2 topic echo` needs `--qos-profile sensor_data`). Durability is
> VOLATILE either way (camera_info is republished every frame, so no latching).

### Published topics (streaming only)

| Param | Default topic | Type | Notes |
|---|---|---|---|
| `stream_depth_topic` | `~/aligned_depth_to_color/image_rect_raw` | `sensor_msgs/Image` | Encoding controlled by `stream_dtype`. |
| `stream_info_topic` | `~/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | |
| (vis, if `stream_publish_vis=true`) | `~/aligned_depth_to_color/debug_jpeg` | `sensor_msgs/CompressedImage` | Turbo-colourised disparity preview. |

### Stream parameters

| Param | Default | Effect |
|---|---|---|
| `stream_enabled` | `false` | Master switch. |
| `stream_align_to_color` | `true` | Route output through `RealsenseAligner`. When false, output is in IR1 frame at engine-scaled resolution. |
| `stream_depth_topic`, `stream_info_topic` | empty (= node default) | Override the published topic names. |
| `stream_dtype` | `"16UC1_mm"` | Either `"16UC1_mm"` (millimetre Z16, matches realsense convention) or `"32FC1"` (metres, more precision). |
| `stream_qos_reliability` | `"reliable"` | QoS reliability for the depth + camera_info publishers: `reliable` (drop-in for realsense, default RViz-visible) or `best_effort` (lower-overhead sensor stream). See the QoS gotcha above. |
| `output_frame_id` | empty | Override the `frame_id` on `depth_image.header`. Useful for the D435 URDF-vs-driver frame-name mismatch; see [§ D435 frame-name caveat](#d435-frame-name-caveat). |
| `stream_publish_vis` | `false` | Also publish a colourised JPEG for human consumption. |
| `stream_max_fps` | `15.0` | Cap the publish rate to bound GPU usage. `0` = uncapped (matches the IR sync rate). |
| `extrinsics_warmup_timeout_sec` | `5.0` | When `stream_align_to_color=true`, the streaming worker waits this long for the latched `extrinsics` + `color_info` to arrive before erroring out. |
| `stream_measure_forward_ms` | `false` | Skip the per-stream-frame CUDA-event timing to shave a hundred microseconds. |

## Camera profile parameters

The `camera_profile` param picks a bundle of topic names + the nominal
baseline. Profile defaults can be overridden individually by setting
`left_topic`, `right_topic`, `left_info_topic`, `color_info_topic`,
`extrinsics_topic`, or `baseline_m` directly.

| `camera_profile` | Topics | Baseline |
|---|---|---|
| `d435` (default) | `/camera/xarm_camera/infra1`, `infra2`, `color/camera_info`, `extrinsics/depth_to_color` | 0.050 m (50 mm) |
| `d405` | `/camera/camera/infra1`, `infra2`, `color/camera_info`, `extrinsics/depth_to_color` | 0.018 m (18 mm) |

To run on the D405 mounted in a different namespace (e.g. `head_camera`), set:

```bash
ros2 launch foundation_stereo foundation_stereo.launch.py \
    camera_profile:=d405 \
    -p left_topic:=/camera/head_camera/infra1/image_rect_raw \
    -p right_topic:=/camera/head_camera/infra2/image_rect_raw \
    -p left_info_topic:=/camera/head_camera/infra1/camera_info \
    -p color_info_topic:=/camera/head_camera/color/camera_info \
    -p extrinsics_topic:=/camera/head_camera/extrinsics/depth_to_color
```

The TRT engines were compiled for the D435's baseline. Running them
against D405 IR pairs spatially aligns fine (the algorithm is
scene-agnostic) but depth values will be off by the baseline ratio
(D405's 18 mm vs D435's 50 mm ≈ 2.8× over-estimate of Z). To get
geometrically correct depth on D405, recompile the TRT engines with
the D405 baseline baked in.

## All ROS parameters

| Param | Default | Notes |
|---|---|---|
| `weights_root` | `/home/tinker/projects/vision_tests/dualrRGB-foundationStereo` | Root containing `FoundationStereo/` and `Fast-FoundationStereo/` sub-trees. |
| `camera_profile` | `"d435"` | Picks defaults for the topic params + baseline. See [§ Camera profile parameters](#camera-profile-parameters). |
| `default_model_kind` | `"fast_trt"` | **Ignored.** Kept for backwards compatibility; the node only serves `fast_trt`. |
| `default_trt_variant` | `"output_two_stage"` | Which compiled engine to use as default. Engines are auto-discovered under `weights_root/Fast-FoundationStereo/`. |
| `warmup_on_launch` | `true` | Load + run one forward through the default TRT engine at startup. See [§ Startup warmup](#startup-warmup). |
| `default_scale` | `0.5` | Default pre-resize factor (ignored by `fast_trt` — engine input shape is baked). |
| `default_iters` | `0` | Default iteration count override (ignored by `fast_trt`). |
| `default_z_far` | `10.0` | Default depth clamp (m). |
| `left_topic`, `right_topic`, `left_info_topic`, `color_info_topic`, `extrinsics_topic` | empty | Topic overrides; empty means use the camera_profile default. |
| `baseline_m` | `0.0` | Override the camera_profile baseline. 0 = use profile default. |
| `sync_slop_sec` | `0.05` | `ApproximateTimeSynchronizer` slop. |
| `sync_queue_size` | `5` | Sync queue depth. |
| `measure_forward_ms` | `true` | Per-request CUDA event timing. ~100 µs sync cost. |
| `stream_*` | various | See [§ Streaming mode](#streaming-mode). |
| `vision_logging_enabled` | `false` | Write per-call evidence images + JSON to `vision_log/<YYYYmmdd_HHMMSS>/`. Off by default because the synchronous disk IO can stall the streaming loop. |
| `vision_log_folder` | `"vision_log"` | Base dir for the session subdir. |

## TRT-only enforcement

The node serves **only** `model_kind="fast_trt"`. The rationale:

1. PyTorch backends (`vitl`, `vits`, `fast_fp32`, `fast_fp16`) are ~10×
   slower than the TRT engines and have no production use here.
2. Each backend has its own load cost; serving them all would force the
   node to thrash through GPU memory on every kind-switch.
3. The TRT engines are the only path tested + tuned for live robot use.

A request with non-empty `model_kind != "fast_trt"` returns immediately
with `status=4` and an explanatory `error_msg`. An empty
`model_kind` field defaults to `"fast_trt"`.

`StereoRunner.infer` itself still supports the other kinds (for offline
analysis / debugging via the underlying API), but the ROS layer hides
them.

## Startup warmup

Cold TRT-engine load + first execute takes ~2-5 s (engine deserialise,
CUDA context init, first-execute buffer alloc, kernel JIT for the
resize ops). To keep the first real request fast, the node pre-runs one
dummy forward at startup:

* `warmup_on_launch=true` (default).
* Dummy 480×848 zero IR pair fed through `StereoRunner.infer(..., live=True)`.
  `live=True` skips the depth-math and point-cloud build — we only care
  about engine + buffer warmup, not the resulting depth values.
* Total wall-clock cost: ~3-4 s on the workstation.
* Failure mode: logged as ERROR, node continues. Real requests will then
  pay cold-load cost and may surface the same error.

The "foundation_stereo ready: ..." log line is printed **after** warmup
completes, so a downstream supervisor can wait on that line to consider
the node operational.

To skip warmup for fast dev iteration:

```bash
ros2 launch foundation_stereo foundation_stereo.launch.py \
    -p warmup_on_launch:=false
```

Only the **default** variant is warmed. A request that overrides
`trt_variant` to a different compiled engine will pay one-time cold-load
cost on first use, then cache.

## Color alignment (`align_to_color=true`)

When the request asks for color-aligned output, the node routes the
IR1-grid depth through `color_align_rs2.RealsenseAligner` — a wrapper
around librealsense's `rs.align(rs.stream.color)` driven by an
`rs.software_device`. The wrapper is constructed once per
(K_color, K_ir, R, T, output-shape, D_color, D_ir) tuple and cached on
the node.

Why this path instead of the obvious forward-projection:

* The naive forward warp in `color_align_legacy.reproject_ir_to_color`
  produces sparse output (~11 % colour-pixel coverage on D435: each IR1
  pixel projects to exactly one colour pixel, leaving holes between
  projections). When that depth is consumed by a Sobel-style edge
  detector or rendered as a colour overlay, the per-pixel gradient
  between every valid projection and its adjacent zero-hole appears as
  "salt-and-pepper" depth edges everywhere — historically perceived as
  "background depth bleeding into the right side of foreground objects."
* `rs.align`'s C++ inner loop does sub-pixel splatting and proper
  occlusion handling, giving dense output (>95 % coverage) with clean
  depth discontinuities that follow colour silhouettes.

Per-call cost is comparable to the legacy forward warp (~3-5 ms warm,
~12 ms first call). The aligner is rebuilt only when intrinsics,
extrinsics, or output shape change — which in practice happens once at
startup.

See `debug_renders/2026-05-25-fs-vs-native-alignment/TRIAGE_FINDINGS.md`
for the full diagnosis, including:

* Why the published `extrinsics/depth_to_color` topic *does* match the
  firmware-internal calibration (verified via direct `pyrealsense2`
  device queries).
* Why the offset isn't fully zero against the ASIC's
  `aligned_depth_to_color` (the ASIC and rs.align both produce
  internally-consistent output; the residual is sensor-firmware
  internal-calibration trickery that's not exposed by the public API).
* Why the *visual* "bleed" the user perceived was the legacy sparsity
  artefact, not a geometric mis-alignment.

The legacy `color_align_legacy.reproject_ir_to_color` is kept for
non-RealSense sources (e.g. an Orbbec camera) or as a debugging
fallback. Consumers of the legacy path must hole-fill (`cv2.medianBlur`
or 3-pass dilate) before any per-pixel processing.

## D435 frame-name caveat

`realsense2_camera` publishes the left IR optical frame as
`xarm_camera_infra1_optical_frame`. The xarm URDF declares it as
`xarm_camera_left_ir_optical_frame`. No `static_transform_publisher`
bridges them out of the box. Three options when consuming this node's
raw-mode (non-aligned) depth:

1. **Stream aligned-to-colour** (the default). Output frame becomes
   `xarm_camera_color_optical_frame`, which agrees between driver and URDF.
2. **Override the published frame_id** to the URDF name:
   ```bash
   ros2 launch foundation_stereo foundation_stereo.launch.py \
       -p output_frame_id:=xarm_camera_left_ir_optical_frame
   ```
   Geometrically equivalent — same physical sensor.
3. **Bridge with a static TF at bringup**:
   ```bash
   ros2 run tf2_ros static_transform_publisher \
       0 0 0  0 0 0 \
       xarm_camera_left_ir_optical_frame \
       xarm_camera_infra1_optical_frame
   ```

## Logging (vision_log)

When `vision_logging_enabled:=true`, the node writes evidence images +
metadata to a per-session directory `<vision_log_folder>/<YYYYmmdd_HHMMSS>/`.

Session-dir resolution order (in `_logging.py`):

1. `$TINKER_VISION_SESSION_TS` env var (`YYYYmmdd_HHMMSS`) — exported
   by `master_*.sh` / `tmux_*.sh` so all vision nodes in one robot run
   share a single session dir.
2. Newest existing `<folder>/<YYYYmmdd_HHMMSS>/` subdir by mtime —
   lets a late-spawned standalone node join the active session.
3. Fresh `strftime` cold-start.

Off by default because the synchronous disk IO at the streaming rate
can stall the worker loop. Turn on selectively for debugging.

## Performance reference (D435, workstation: Ada RTX, CUDA 12.8)

| Operation | Latency |
|---|---|
| Cold startup → "ready" (warmup on) | ~3.4 s |
| Cold startup → "ready" (warmup off) | < 0.5 s |
| `StereoRunner.infer` forward, `fast_trt`, `output_two_stage`, warm | ~22 ms |
| `RealsenseAligner.align`, warm | ~3-5 ms |
| Service end-to-end (warm, no pointcloud, align_to_color=true) | ~28 ms |
| `align_to_color=true` output coverage | ~95 % of colour pixels |
| Forward-warp legacy output coverage | ~11 % (with ~89 % holes) |

## Status codes (full reference)

| `status` | Where set | Meaning |
|---|---|---|
| 0 | success path | Inference + alignment + serialisation OK. |
| 1 | `_run_inference` early | No synced (IR1, IR2, info) triple cached yet. |
| 3 | `_run_inference` mid-path | Either `cv_bridge` failed, or `align_to_color=true` was requested but `_color_info` or `_extrinsics` are still `None`. |
| 4 | `_run_inference` request guard | `model_kind` non-empty and != `"fast_trt"`. |

## Known follow-ups

* **Recompile TRT engines for D405**. The shipped engines were compiled
  for the D435's 50 mm baseline. Running them against D405 IR pairs
  produces spatially-aligned but geometrically-incorrect depth (Z is
  off by the baseline ratio). For D405 production use, the engines
  need re-export with the D405 baseline baked in.
* **Empirical per-camera calibration**. The rs.align output is sub-px
  consistent against the colour image silhouettes; absolute Z accuracy
  beyond that depends on the camera's factory calibration plus any
  drift over time. If sub-mm Z is needed, add an empirical
  correction layer per camera.
* **Streaming mode + alignment cache**. The streaming worker rebuilds
  the `RealsenseAligner` only on calibration change, so steady-state
  per-frame cost is the ~3-5 ms align. If `K_ir`/`K_color` ever
  *flicker* (e.g. the realsense node restarts), the aligner will be
  rebuilt mid-stream — usually invisible but logged on each rebuild.

## Spec + plan + triage

* Design spec: [`docs/superpowers/specs/2026-05-24-foundation-stereo-design.md`](../../docs/superpowers/specs/2026-05-24-foundation-stereo-design.md)
* Implementation plan: [`docs/superpowers/plans/2026-05-24-foundation-stereo.md`](../../docs/superpowers/plans/2026-05-24-foundation-stereo.md)
* Bleed triage + rationale for `rs.align`: [`debug_renders/2026-05-25-fs-vs-native-alignment/TRIAGE_FINDINGS.md`](../../debug_renders/2026-05-25-fs-vs-native-alignment/TRIAGE_FINDINGS.md)
