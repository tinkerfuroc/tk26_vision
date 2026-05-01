# tk_vision_specialized

Task-specific vision servers. Each node wraps a narrow detection task and exposes a single ROS 2 action or service.

## Nodes

| Executable | Type | Interface | Description |
|---|---|---|---|
| `spot_on_shelf_server` | action | `tinker_vision_msgs_26/action/SpotOnShelf` | Detect objects on a shelf and bucket them into vertical layers + horizontal grids. Delegates detection to `object_detection_yolo`. |
| `waving_person_server` | service | `tinker_vision_msgs_26/srv/DetectWaving` | Find all persons raising a hand / waving in the current Orbbec frame. YOLOv8 person boxes + MediaPipe Pose on each ROI. |
| `waving_client` | — | — | Example client: calls `/detect_waving_persons` once per second, prints results. Useful for camera-alignment sanity before demos. |
| `check_waving_inference` | — | — | Offline tester. Subscribes to `/camera/color/image_raw`, runs N YOLO + MediaPipe passes on a timer, dumps `XX_raw.jpg`, `XX_annotated.jpg`, `XX_result.json` into a timestamped folder. Has no ROS service dependency. |
| `placing_location_server` | service | `tinker_vision_msgs_26/srv/PlacingLocation` | VLM-only tabletop placing-location finder. Asks Gemini 2.5 Pro for ranked empty regions on the visible desktop, projects each region's bbox centroid to 3D via the active camera's depth, optionally TF-transforms to a target frame. Subclasses `YOLOSegmentationNode` to reuse camera sync + intrinsics + depth-to-3D projection. |

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
ros2 service call /detect_waving_persons tinker_vision_msgs_26/srv/DetectWaving \
  "{threshold_meters: 3.0, target_frame: 'map'}"
```

### Known issues (filed for follow-up, not fixed during migration)

- `waving_person_server.py:99` — `right_elbow.y <= right_shoulder.y + int(img_h + 0.1)` almost certainly meant `img_h * 0.1` (mirroring line 100). The right-arm branch is effectively always true.

## check_waving_inference

Standalone diagnostic node — no `DetectWaving` dependency. Default: 10 runs, 1 s interval, reads `/camera/color/image_raw`, writes artifacts under `~/tk25_ws/src/tk26_vision/log_vision/waving_check_<timestamp>/`.

```bash
ros2 run tk_vision_specialized check_waving_inference \
  --ros-args -- --runs 20 --interval 0.5 --model yolov8s.pt
```

Flags: `--runs`, `--interval`, `--image-topic`, `--model`, `--output-root`, `--node-name`.

## PlacingLocation

Request: `camera` (`'realsense'`/`'orbbec'`), `item_description` (free text — what the robot is about to place, e.g. `'a coke can ~6 cm wide'`), `target_frame` (TF target, empty = camera frame), `max_candidates` (`0` = server default of 5; otherwise clamped to `[1, 10]`), `return_rgb_image`, `return_debug_overlay`. Response: `status` (`0`=ok, `1`=no candidates, `-1`=hard error — see `error_msg`), `candidate_points` (`PointStamped[]`, ordered best→worst), `candidate_bboxes` (`BoundingBox[]`, parallel array of pixel-space xyxy), and the optional images.

Pipeline (single VLM call, no fallback chain):

1. Wait for synchronized RGB+depth from the requested camera (parent class state).
2. Project depth to a `[H, W, 3]` point grid + valid mask.
3. Call Gemini via OpenRouter (`google/gemini-2.5-pro` by default) with a placement-tuned system prompt: enumerate clear, flat regions large enough for the described item, ranked best to worst, return up to `max_candidates` bboxes in normalized `[ymin, xmin, ymax, xmax]` 0–1000 form.
4. Build a synthetic rectangular mask over each bbox interior; reuse `YOLOSegmentationNode._calculate_centroid` (depth median over valid pixels) to get a 3D point.
5. TF-transform each point to `target_frame` if requested; drop on TF failure.
6. Return `PointStamped[]` in the VLM's rank order.

Compute budget: 8 s VLM timeout, 1 retry — fits the project-wide 10 s/call ceiling. Requires `OPENROUTER_API_KEY` per `src/tk26_vision/CLAUDE.md § Environment` (`.env` at workspace root). Node startup succeeds without the key; the service call returns `status=-1, error_msg='VLM unavailable: …'` instead of crashing.

```bash
ros2 launch orbbec_camera femto_bolt.launch.py depth_registration:=true
ros2 run tk_vision_specialized placing_location_server
ros2 service call /placing_location tinker_vision_msgs_26/srv/PlacingLocation \
  "{camera: 'orbbec', item_description: 'a coke can ~6 cm wide', target_frame: '', max_candidates: 3, return_debug_overlay: true}"
```

ROS parameters (in addition to those inherited from `YOLOSegmentationNode`): `vlm_model` (default `google/gemini-2.5-pro`), `vlm_timeout_s` (default `8.0`), `vlm_max_retries` (default `1`), `default_max_candidates` (default `5`).

## Dependencies

ROS (via `package.xml`): `rclpy`, `geometry_msgs`, `std_msgs`, `sensor_msgs`, `tinker_vision_msgs_26`, `tf2_ros`, `tf2_geometry_msgs`, `cv_bridge`, `message_filters`. `placing_location_server` adds runtime deps on `object_detection_new` (parent node class) and `kimi_api` (`._env` for OpenRouter key/base URL).

Python (via `requirements.txt`, pip-installed): `ultralytics>=8.0.0`, `mediapipe>=0.10.0`, `opencv-python>=4.5.0`, `numpy>=1.21.0`, `openai>=1.0`, `python-dotenv>=1.0`.

## Changelog

- **2026-04-30** — Add `placing_location_server` (VLM-only tabletop placing-location service) + `tinker_vision_msgs_26/srv/PlacingLocation`. Subclasses `YOLOSegmentationNode` for camera I/O reuse; calls Gemini 2.5 Pro via `kimi_api._env`; returns `PointStamped[]` ranked best→worst. No automated tests added — VLM round-trip is non-deterministic.
