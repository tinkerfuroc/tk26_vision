# camera_server

Per-camera C++ servers that own the only streaming subscriptions to the wrist
RealSense / head Orbbec topics and serve frames, point clouds, and
time-correct transforms **on demand**:

- `~/get_snapshot` (`tinker_vision_msgs_26/srv/GetCameraSnapshot`) — latest
  synced color+depth pair + camera infos + transforms at the pair stamp, with
  `max_age` / `captured_after` freshness semantics.
- `~/get_point_cloud` (`GetCameraPointCloud`) — CPU deprojection of the cached
  pair (stride / XYZ or XYZRGB / optional target frame at pair stamp).
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

## Changelog

- 2026-07-13: package scaffold + thread-safe FrameStore with captured_after wait.
