Service name offered is `object_detection_yolo`.

Servive request `prompt` should match EXACTLY with trained yolo class.

Put model in `models/` directory (under `object_detection`, siblings with `config` folder)

Start node using 
```
ros2 run object_detection_new yolo_seg_node --ros-args -p model_path:="<pt_name>" -p visualization:=true -p sort_mode:="highest"
```

After starting, wait for `YOLO Segmentation Node initialized successfully` before starting task

## Changelog

- **2026-05-28** — FFS-first depth with native fallback (`prefer_ffs`). Six new ROS params added to `yolo_seg_node` / `yolo_seg_default_node` (and inherited by `generalist_node`): `prefer_ffs` (default `true`), `ffs_service` (default `'~/get_depth'`), `ffs_wait_for_service_s` (0.2), `ffs_call_timeout_s` (8.0), `ffs_align_to_color` (true), `ffs_fallback_log_period_s` (30.0). When `prefer_ffs=true` (default), realsense detection calls try the FoundationStereo on-demand `~/get_depth` service first; on any failure (unavailable, timeout, non-zero status, decode error) the node falls back silently to the native realsense aligned depth. A throttled warning fires at most once per `ffs_fallback_log_period_s` seconds. Vision_log sidecar JSON gains `depth_source ∈ {'ffs', 'native'}` per call. No srv/msg schema changes — zero impact on existing callers. Orbbec branch unchanged. Rollback: `ros2 param set /<node> prefer_ffs false` (takes effect on the next call, no restart).
- **2026-05-02** — `_sort_objects_and_segments` `'closest'` mode now sorts by Euclidean distance `sqrt(x²+y²+z²)` on `Object.centroid` instead of single-axis (`centroid.x` for realsense, `centroid.z` for orbbec). Single-axis ignored lateral / vertical offset, and orbbec centroids are TF-transformed to `target_frame` before sort, so `.z` no longer reliably meant "forward" anyway. The `'highest'` mode's no-map-frame fallback (which falls back to `'closest'`) was updated to match. Affects every node that inherits `YOLOSegmentationNode` (`yolo_seg_node`, `yolo_seg_default_node`, `generalist_node`).