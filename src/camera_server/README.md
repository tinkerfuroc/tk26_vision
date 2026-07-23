# camera_server

Per-camera C++ servers that own the only streaming subscriptions to the wrist
RealSense / head Orbbec topics and serve frames, point clouds, and
time-correct transforms **on demand**:

- `~/get_snapshot` (`tinker_vision_msgs_26/srv/GetCameraSnapshot`) — latest
  synced color+depth pair + camera infos + transforms at the conservative
  pair stamp (`min(color_stamp, depth_stamp)`), with `max_age` /
  `captured_after` freshness semantics applying to both images.
- `~/get_point_cloud` (`GetCameraPointCloud`) — CPU deprojection of registered
  depth from the cached pair (stride / XYZ or XYZRGB / optional target frame
  at the depth image stamp).
- `~/get_transform` (`GetTransform`) — lookup against the server's warm 180 s
  TF buffer, for on-demand consumers with cold local buffers.
- `~/status` (`CameraServerStatus`, 1 Hz) — stream ages, sync fps, pair seq.

Two instances: `wrist_camera_server` (launched by the manipulation bringup
that owns the RealSense) and `head_camera_server` (launched by
`vision_bringup/vision_driver.launch.py`). A separate `camera_compat_bridge`
executable serves the legacy `get_image_service` / `get_point_cloud_service` /
`get_orbbec_pc` names by forwarding to the servers — param-gated, OFF by
default (the Python utility nodes keep those names until cutover).

Design: `../../docs/specs/2026-07-13-camera-server-design.md`.
Consumers are NOT migrated by this package's introduction (Appendix A of the
spec maps the deferred migration).

## Build

    tkbuild tk26_vision --packages-select camera_server

## Runtime contract

Image synchronization and both CameraInfo subscriptions run in a node-owned
callback group/executor thread. This is also true when the node is loaded as a
component: blocking `captured_after` service calls cannot prevent ingestion of
the pair that releases them. The standalone service executor uses
`num_executor_threads` (default 4, minimum 2).

The two images must be color-aligned, have compatible nonempty frame IDs, and
share one response optical frame (the depth frame when present, otherwise the
color frame). Head-camera defaults use `/camera/depth/image_raw` with
`/camera/color/camera_info`, matching the registered-depth path. Missing
CameraInfo is best-effort for snapshots and is reported diagnostically.

Raw registered images are supported. Deprojection caches distortion-aware rays
for `plumb_bob`, `rational_polynomial`, and `equidistant` CameraInfo models;
unsupported or internally inconsistent models fail closed.

## Changelog

- 2026-07-13: package scaffold + thread-safe FrameStore with captured_after wait.
- 2026-07-23: hardened CPU Deprojector for registered depth — validated
  16UC1/mono16/32FC1 input, deterministic little-endian XYZ[RGB], cached
  xy-table, stride, and rigid-transform support.
- 2026-07-23: added starvation-proof CameraServerNode ingestion, conservative
  two-image freshness, bounded snapshot/TF services, accurate status telemetry,
  distortion-aware raw-image rays, and focused ROS integration coverage.
- 2026-07-23: implemented `~/get_point_cloud` with depth-stamp TF/header
  semantics, stride/XYZ[RGB] deprojection, fail-closed TF and camera-data
  validation, plus bounded no-data/stale/timeout responses.
- 2026-07-23: added the opt-in zero-subscription legacy compatibility bridge
  with a dedicated client executor, bounded forwarding deadlines, and cleanup
  of timed-out requests.
- 2026-07-23: added standalone gated launch (wrist/head/bridge) and wired the
  head server into `vision_driver.launch.py` behind `enable_camera_server`.
