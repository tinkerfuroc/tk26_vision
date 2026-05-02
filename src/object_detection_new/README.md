Service name offered is `object_detection_yolo`.

Servive request `prompt` should match EXACTLY with trained yolo class.

Put model in `models/` directory (under `object_detection`, siblings with `config` folder)

Start node using 
```
ros2 run object_detection_new yolo_seg_node --ros-args -p model_path:="<pt_name>" -p visualization:=true -p sort_mode:="highest"
```

After starting, wait for `YOLO Segmentation Node initialized successfully` before starting task

## Changelog

- **2026-05-02** — `_sort_objects_and_segments` `'closest'` mode now sorts by Euclidean distance `sqrt(x²+y²+z²)` on `Object.centroid` instead of single-axis (`centroid.x` for realsense, `centroid.z` for orbbec). Single-axis ignored lateral / vertical offset, and orbbec centroids are TF-transformed to `target_frame` before sort, so `.z` no longer reliably meant "forward" anyway. The `'highest'` mode's no-map-frame fallback (which falls back to `'closest'`) was updated to match. Affects every node that inherits `YOLOSegmentationNode` (`yolo_seg_node`, `yolo_seg_default_node`, `generalist_node`).