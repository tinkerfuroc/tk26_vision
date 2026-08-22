# tk_vision_specialized

Task-specific vision servers. Each node wraps a narrow detection task and exposes a single ROS 2 action or service.

## Nodes

| Executable | Type | Interface | Description |
|---|---|---|---|
| `spot_on_shelf_server` | action | `tinker_vision_msgs_26/action/SpotOnShelf` | Detect objects on a shelf and bucket them into vertical layers + horizontal grids. Delegates detection to `object_detection_yolo`. |
| `waving_person_server` | action | `tinker_vision_msgs_26/action/DetectWaving` | Find all persons raising a hand / waving in the current Orbbec frame. |
| `waving_client` | — | — | Example client: sends `/detect_waving_persons` goals once per second and prints results. Useful for camera-alignment sanity before demos. |
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

Goal: `threshold_meters` (float, ≤0 = no limit), `target_frame` (string, e.g. `"map"` or `"base_link"`), `min_waving_persons` (int32, default 0). Result: `status` (0=found, 1=none, -1=error), `error_msg`, `waving_persons[]` (PointStamped, sorted closest-first). `rgb_image`, `depth_image`, `segments[]` are declared but not populated by the current server. Goals execute FIFO and can be canceled while queued or during camera, TF, inference, and VLM stages.

`min_waving_persons` is an explicit per-caller opt-in to a concurrent VLM (Gemini/Qwen) fallback: if `> 0`, the server launches the VLM call on a background thread as soon as the frame is available (parallel with the YOLO+MediaPipe pass above) and blocks on it only if the CV pass alone found fewer than `min_waving_persons` wavers — otherwise the VLM call is abandoned (left running, result discarded) without adding latency. Leaving it at the default `0` (every caller except Restaurant's `BtNode_ScanForWavingPerson`) keeps the fast, VLM-free path unchanged. `enable_vlm_fallback` (node param, default `true`) is the master kill-switch; `vlm_timeout_s` (default 20.0) bounds the wait.

Pipeline:
1. Wait for a fresh synchronized RGB + registered depth frame and `CameraInfo`.
2. YOLOv8 (`yolov8s.pt`) → person boxes.
3. For each person ROI, MediaPipe Tasks PoseLandmarker keypoints (`_pose_backend.py`).
4. Single-frame heuristic on landmark y-coords (hand above shoulder, or hand above elbow with elbow above shoulder). No temporal motion tracking — a raised static hand counts.
5. Centroid = mean of valid 3-D depth points within the bbox; gated by `threshold_meters` on z; TF-transformed to `target_frame` if given.

Start server + example client:

```bash
ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true depth_registration:=true
ros2 run tk_vision_specialized waving_person_server
# separate shell:
ros2 run tk_vision_specialized waving_client
# or a single-shot call:
ros2 action send_goal /detect_waving_persons tinker_vision_msgs_26/action/DetectWaving \
  "{threshold_meters: 3.0, target_frame: 'map'}"
```

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

Python (via `requirements.txt`, pip-installed): `ultralytics>=8.0.0`, `mediapipe>=1.0`, `opencv-python>=4.5.0`, `numpy>=1.21.0`, `openai>=1.0`, `python-dotenv>=1.0`.

## Changelog

- **2026-08-22** — Ported the waving pose pass from the legacy MediaPipe **Solutions** API (`mp.solutions.pose`, pinned `mediapipe==0.10.9`) to the **Tasks API** `PoseLandmarker` on `mediapipe==1.0.1` (`_pose_backend.py`), because Solutions no longer exists upstream from 0.10.30. New node params `pose_model_path` (default `'pose_landmarker_full.task'`, resolved via `weights_cache.find_cached` — no auto-download, stage with `scripts/download_models.py`) and `pose_delegate` (`'gpu'` default, GPU delegate ~8 ms/person vs ~60 ms/person CPU, falls back to CPU automatically with a WARN). `is_waving` semantics and the default `'vlm'` `waving_detector` mode are unchanged. Parity with 0.10.9 verdicts/landmarks is enforced by `test/test_pose_parity.py` against a frozen fixture in `test/fixtures/pose_parity/`.
- **2026-07-04** — Waving detection is now **VLM-only by default**. New `waving_detector` param (`'vlm'` default | `'hybrid'` | `'mediapipe'`): in `'vlm'` mode the VLM is the sole waver source and MediaPipe pose is skipped (YOLO still runs for the person mask → 3D centroid). `'hybrid'` reproduces the 2026-07-03 MediaPipe+VLM-augment behavior; `'mediapipe'` is CV-only. `'vlm'`/`'hybrid'` **auto-degrade to MediaPipe** at call time when no provider key is configured (or `enable_vlm_fallback=false`), so offline/no-key boxes are unchanged. The VLM prompt gained a **live-person clause**: a waver must be a real, physically-present human — figures printed/displayed on a wall mural, advertisement, poster, screen, photo, etc. are rejected even when their pose looks like a wave. Prompt-only enforcement (no schema change). Tradeoff: every call in `'vlm'` mode waits ~5–20 s for the VLM (vs ~100–300 ms MediaPipe); set `waving_detector:=hybrid`/`mediapipe` for a fast path. Design: `docs/superpowers/specs/2026-07-04-waving-vlm-only-live-person-design.md`.
- **2026-07-03** — `DetectWaving.min_waving_persons` now gates a concurrent VLM waving-detection fallback (was previously declared but unused — the trigger condition it fed was never satisfied by any real caller). VLM call launches on a background thread as soon as the frame is available, in parallel with the YOLO+MediaPipe pass; abandoned without waiting if CV alone already found enough wavers. Restaurant's BT node is the only caller that opts in (`min_waving_persons=2`); GPSR/EGPSR and the `track_web` bench tool never set it, so they keep the pre-existing fast-only behavior unchanged.
- **2026-04-30** — Add `placing_location_server` (VLM-only tabletop placing-location service) + `tinker_vision_msgs_26/srv/PlacingLocation`. Subclasses `YOLOSegmentationNode` for camera I/O reuse; calls Gemini 2.5 Pro via `kimi_api._env`; returns `PointStamped[]` ranked best→worst. No automated tests added — VLM round-trip is non-deterministic.
