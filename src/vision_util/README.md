# vision_util

Utility ROS2 services for tk26 vision. Lightweight nodes that don't fit in
any of the detection / tracking / kimi_api packages.

## Nodes

| Entry point | Service / topic | Description |
|---|---|---|
| `door_detection` | `/door_detection_srv` | Heuristic open/closed door check from the Orbbec depth centre patch (mean of 20×20 pixels < 1.5 m ⇒ closed). |
| `get_point_cloud` | `/get_point_cloud_service` | Latest-frame relay. Caches synced color+depth (RealSense) or color+`/camera/depth_registered/points` (Orbbec) and returns the cached cloud on demand. Pure passthrough. |
| `get_orbbec_pc` | `/get_orbbec_pc` | CUDA-deprojected Orbbec point cloud. Subscribes `/camera/depth/image_raw` + `/camera/color/image_raw` + `/camera/color/camera_info`, deprojects on NVIDIA GPU via PyTorch, and returns a fresh `PointCloud2` (XYZ or XYZRGB). Sidesteps the SDK's CPU colored-PC reprojection that bottlenecks the camera under the iGPU bind-mount workaround in [`../../orbbec_diagnosis.md`](../../orbbec_diagnosis.md). Hard-fails at init if no CUDA device is visible. |
| `get_image` | (legacy) | Retained for parity with the tk23 service shape; no live caller. |

## Run

```bash
source install/setup.zsh
ros2 run vision_util door_detection
ros2 run vision_util get_point_cloud
ros2 run vision_util get_orbbec_pc
```

## `get_orbbec_pc` — service contract

Service type: `tinker_vision_msgs_26/srv/GetOrbbecPC`.

**Request**

| Field | Type | Notes |
|---|---|---|
| `stride` | `uint32` | Pixel stride. `0` or `1` = full resolution. `2` keeps every 2nd pixel ⇒ 4× fewer points. |
| `include_color` | `bool` | `true` ⇒ XYZRGB cloud (point_step=16, `rgb` packed FLOAT32). `false` ⇒ XYZ cloud (point_step=12). |

**Response**

| Field | Type | Notes |
|---|---|---|
| `status` | `int32` | `0` = OK, `1` = error (see `error_msg`). |
| `error_msg` | `string` | Populated when `status != 0`. |
| `points` | `sensor_msgs/PointCloud2` | `is_dense=true`. `header.frame_id` defaults to the source depth frame; override via the `output_frame_id` ROS param. |

**Failure modes**

- No CUDA device at startup ⇒ `RuntimeError`, node refuses to start.
- No depth/intrinsics observed yet ⇒ `status=1, error_msg="No Orbbec depth/intrinsics yet."`
- `include_color=true` with no synced color frame yet ⇒ `status=1, error_msg="No synced color frame for include_color=true."`
- Depth or color shape mismatches the cached `CameraInfo` ⇒ `status=1` with a shape-mismatch message.

**Assumptions**

- Orbbec is launched with `depth_registration:=true` so depth is already in the color frame and the color CameraInfo carries the right intrinsics. The node does not consume `/camera/depth/camera_info`.
- Depth is `16UC1`/`mono16` in millimetres (Femto Bolt default `depth_format:=Y16`) or `32FC1` in metres. Both are converted to metres before deprojection.

**Quick smoke test**

```bash
ros2 service call /get_orbbec_pc tinker_vision_msgs_26/srv/GetOrbbecPC \
  "{stride: 1, include_color: false}"
```

## Build

```bash
tkbuild tk26_vision --packages-select tinker_vision_msgs_26 vision_util
```

## Changelog

- **2026-06-09** — Fix a ~14 GB-per-write host-RAM leak in `vision_logging`. `VisionLogger.write` blended the seg mask with `overlay[mask]`, but `mask` arrives as a uint8 0/1 array (from `result_parser`), so this was **integer fancy-indexing** along axis 0 — allocating an `(H, W, W, 3)` float32 transient (~14 GB at 720p, vs ~5.5 MB for a correct boolean mask) on every masked log write, and corrupting the overlay (re-indexing rows 0/1). Off-loop on an unbounded single-worker queue, writes backed up and ratcheted RSS 2.4 → 28 GB → swap-thrash → ssh drops + Orbbec depthengine SIGSEGV (proven via tracemalloc: `vision_logging:243` = 26.4 GiB; leak vanished with `vision_logging_enabled=false`). Now coerces the mask to boolean and shape-guards in a shared `_apply_mask_overlay` helper — central sink, so all mask-logging nodes (`person_track_node`, `yolo_seg`, `generalist_node`, `waving_person_server`, `object_match_server`, `placing_location_server`) benefit. Orange overlay still renders correctly on the masked pixels.
- **2026-05-03** — Add `get_orbbec_pc` service node (CUDA deprojection of Orbbec depth → PointCloud2). Adds `srv/GetOrbbecPC.srv` to `tinker_vision_msgs_26`. Bypasses the SDK colored-PC bottleneck under the NVIDIA-575 ↔ depthengine workaround documented in `orbbec_diagnosis.md`.
