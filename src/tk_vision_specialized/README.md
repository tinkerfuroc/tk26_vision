# tk_vision_specialized

Task-specific vision servers. Each node wraps a narrow detection task and exposes a single ROS 2 action or service.

## Nodes

| Executable | Type | Interface | Description |
|---|---|---|---|
| `spot_on_shelf_server` | action | `tinker_vision_msgs_26/action/SpotOnShelf` | Detect objects on a shelf and bucket them into vertical layers + horizontal grids. Delegates detection to `object_detection_yolo`. |
| `waving_person_server` | service | `tinker_vision_msgs/srv/DetectWaving` | Find all persons raising a hand / waving in the current Orbbec frame. YOLOv8 person boxes + MediaPipe Pose on each ROI. |
| `waving_client` | — | — | Example client: calls `/detect_waving_persons` once per second, prints results. Useful for camera-alignment sanity before demos. |
| `check_waving_inference` | — | — | Offline tester. Subscribes to `/camera/color/image_raw`, runs N YOLO + MediaPipe passes on a timer, dumps `XX_raw.jpg`, `XX_annotated.jpg`, `XX_result.json` into a timestamped folder. Has no ROS service dependency. |

## SpotOnShelf

Goal fields: `shelf_left`, `shelf_right` (PoseStamped in `map`), `shelf_heights` (float[]), `item_ids` (string[]). Feedback: `status`, `message`. Result: `status`, `error_msg`, `item_height_grids[]`, `item_horizontal_grids[]`.

Depends on `object_detection_yolo` (from `object_detection_new`) running. Start both:

```bash
ros2 run object_detection_new yolo_seg_node --ros-args -p model_path:=yolo11m-seg.pt
ros2 run tk_vision_specialized spot_on_shelf_server
```

## DetectWaving

Request: `threshold_meters` (float, ≤0 = no limit), `target_frame` (string, e.g. `"map"` or `"base_link"`). Response: `status` (0=found, 1=none, -1=error), `error_msg`, `waving_persons[]` (PointStamped, sorted closest-first). `rgb_image`, `depth_image`, `segments[]` are declared but not populated by the current server.

Pipeline:
1. Grab latest synchronized RGB + `PointCloud2` + `CameraInfo`.
2. YOLOv8 (`yolov8s.pt`) → person boxes.
3. For each person ROI, MediaPipe Pose keypoints.
4. Single-frame heuristic on landmark y-coords (hand above shoulder, or hand above elbow with elbow above shoulder). No temporal motion tracking — a raised static hand counts.
5. Centroid = mean of valid 3-D depth points within the bbox; gated by `threshold_meters` on z; TF-transformed to `target_frame` if given.

Start server + example client:

```bash
ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true depth_registration:=true
ros2 run tk_vision_specialized waving_person_server
# separate shell:
ros2 run tk_vision_specialized waving_client
# or a single-shot call:
ros2 service call /detect_waving_persons tinker_vision_msgs/srv/DetectWaving \
  "{threshold_meters: 3.0, target_frame: 'map'}"
```

### Known issues (filed for follow-up, not fixed during migration)

- `waving_person_server.py:99` — `right_elbow.y <= right_shoulder.y + int(img_h + 0.1)` almost certainly meant `img_h * 0.1` (mirroring line 100). The right-arm branch is effectively always true.
- `waving_person_server.py:118` — writes `person_roi<ts>.png` to CWD on every call; unbounded disk fill. Gate behind a debug flag and route to `vision_log_folder`.

## check_waving_inference

Standalone diagnostic node — no `DetectWaving` dependency. Default: 10 runs, 1 s interval, reads `/camera/color/image_raw`, writes artifacts under `~/tk25_ws/src/tk26_vision/log_vision/waving_check_<timestamp>/`.

```bash
ros2 run tk_vision_specialized check_waving_inference \
  --ros-args -- --runs 20 --interval 0.5 --model yolov8s.pt
```

Flags: `--runs`, `--interval`, `--image-topic`, `--model`, `--output-root`, `--node-name`.

## Dependencies

ROS (via `package.xml`): `rclpy`, `geometry_msgs`, `std_msgs`, `sensor_msgs`, `tinker_vision_msgs`, `tinker_vision_msgs_26`, `tf2_ros`, `tf2_geometry_msgs`, `cv_bridge`, `message_filters`.

Python (via `requirements.txt`, pip-installed): `ultralytics>=8.0.0`, `mediapipe>=0.10.0`, `opencv-python>=4.5.0`, `numpy>=1.21.0`.
