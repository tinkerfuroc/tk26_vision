# CLAUDE.md

Guidance for Claude Code when working in `src/tk26_vision/`.

## Project Overview

ROS2 Humble vision module for Tinker 2026: object detection, person tracking with ReID, pan-tilt head control, LLM-backed person features / grocery categorization (OpenRouter), and utility services (door detection, point-cloud relay). Python 3.10 venv at `src/tk26_vision/.venv-vision-main/`.

## Build

Use the wrapper — it sources the venv + ROS, runs `colcon build --symlink-install`, and patches install-tree shebangs so `ros2 run` picks up the venv python (openai, dotenv, ultralytics, pyserial live only in the venv):

```bash
./src/tk26_vision/scripts/build.sh [colcon args...]
```

Plain `colcon build` produces `#!/usr/bin/python3` shebangs that can't see the venv. If you must run it manually, follow up with `./src/tk26_vision/scripts/fix_venv_shebangs.sh` (idempotent; covers all tk26 packages).

If a build errors on stale symlinks, `rm -rf build/<pkg> install/<pkg>` and rebuild that package.

## Environment

### Python deps

Per-package `requirements.txt` install into the shared venv:

```bash
source src/tk26_vision/.venv-vision-main/bin/activate
pip install -r src/tk26_vision/src/kimi_api/requirements.txt
pip install -r src/tk26_vision/src/pan_tilt/requirements.txt
pip install -r src/tk26_vision/src/vision_track/requirements.txt
```

### OpenRouter API key (kimi_api)

`feature_recognition`, `feature_matching`, `grocery_categorize` require `OPENROUTER_API_KEY`. Copy `src/tk26_vision/src/kimi_api/.env.example` to the workspace-root `.env` and fill in the key — `python-dotenv` auto-loads `.env` from CWD upward at node startup. Missing key ⇒ `RuntimeError` at node init.

Optional: `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`), `LLM_MODEL` (default `openai/gpt-4.1`, also `-p llm_model:=…`).

## Running Nodes

```bash
source install/setup.bash

# Object detection
ros2 run object_detection_new yolo_seg_node          # /object_detection_yolo (custom yolo11m-seg)
ros2 run object_detection_new yolo_seg_default_node  # /object_detection (pretrained yolo11n-seg)

# Tracking / shelves
ros2 run vision_track person_track_server            # action /track_person
ros2 run tk_vision_specialized spot_on_shelf_server  # action /spot_on_shelf

# Pan-tilt (servo on /dev/ttyUSB0; override via -p device:=…)
ros2 run pan_tilt ctrl                               # serial driver + TF
ros2 run pan_tilt follow_head                        # /follow_head_action + /follow_head_service

# LLM-backed (kimi_api)
ros2 run kimi_api feature_recognition                # /feature_extraction_service, /seat_recommend_service
ros2 run kimi_api feature_matching                   # /feature_matching_service
ros2 run kimi_api grocery_categorize                 # action /grocery_categorize

# Utilities
ros2 run vision_util door_detection                  # /door_detection_srv
ros2 run vision_util get_point_cloud                 # /get_point_cloud_service
```

## Architecture

```
src/tk26_vision/src/
├── tinker_vision_msgs_26/    # action/TrackPerson, action/SpotOnShelf
├── object_detection_new/     # YOLO-seg: yolo_seg_node + yolo_seg_default_node (same class, different params)
├── vision_track/             # ByteTrack + ResNet50 ReID (custom) or YOLO BoT-SORT (native)
├── tk_vision_specialized/    # SpotOnShelf action server
├── pan_tilt/                 # ctrl (serial + TF) + follow_head (YOLO@1Hz w/ blur gate)
├── kimi_api/                 # OpenRouter LLM services; _env.py centralizes key loading
└── vision_util/              # door_detection (Orbbec 20x20 depth heuristic), get_point_cloud (cached relay)
```

**Notes:**
- `kimi_api` calls `object_detection` (generalist) via the `detection_service` ROS param — retargetable without rebuilding.
- All migrated nodes import from `tinker_vision_msgs` (tk23's package), not `tinker_vision_msgs_26`. Intentional — `tk25_decision/messages.py` also imports from it. Consolidation deferred.
- Cameras: RealSense = aligned depth-to-color Image + pinhole intrinsics (ROS convention: x=fwd, y=left, z=up); Orbbec = PointCloud2 reprojected to image grid (standard ROS convention from the cloud).

## Models

YOLO `.pt` files pre-bundled under `object_detection_new/models/` / `vision_track/` or auto-downloaded by Ultralytics. Supported: `yolo11{n,s,m,l,x}-seg.pt`, `yolov8{n,s,m,l,x}-seg.pt`.

## Configuration

Key ROS2 parameters:
- `object_detection_new`: `service_name`, `model_path`, camera topics, `sort_mode`
- `pan_tilt/ctrl`: `device`, `specs_path`
- `pan_tilt/follow_head`: `yolo_model`
- `kimi_api/*`: `llm_model`, `detection_service`, `log_prompts`

## Third-party drivers

Vendored under `src/tk26_vision/thirdparty/` (plain clones, no submodules, `.git` stripped):
- `librealsense` v2.57.7 — built from source, installed system-wide (`sudo cmake --install build`)
- `realsense-ros` 4.57.7 — finds librealsense via `find_package`, builds in colcon
- `OrbbecSDK_ROS2` v2.7.6 — bundles OrbbecSDK v2, builds in colcon

See `thirdparty/README.md` for build/udev/kernel-patch steps.

## Camera bringup

**Do not run the vendored camera launches bare** — they drop to ~3 Hz together on this workstation due to three compounding misconfigurations (RealSense USB enumeration, realsense-ros `TRANSIENT_LOCAL` QoS default, kernel UDP buffer limit). [`CAMERA_BRINGUP.md`](./CAMERA_BRINGUP.md) has the canonical launch sequence, the config files under `config/`, the required `FASTRTPS_DEFAULT_PROFILES_FILE` env var, and the full root-cause writeup. Rates expected after the fix: ~30 Hz ±5 ms on every color/depth topic.

## Testing

Integration smoke suite at `scripts/tests/`, four tiers each gated by the previous:

| Tier | Script | Scope | Needs |
|---|---|---|---|
| T0 | `t0_static.sh` | shebangs, venv deps, ROS interfaces, entry-point imports, `.env` sanity | venv only |
| T1 | `t1_startup.sh` | all 11 nodes start + advertise + SIGTERM clean; pan_tilt serial pos/neg; kimi_api key pos/neg | venv + (opt) servo |
| T2 | `t2_live.sh` | one call per node with live cameras (empty scene OK) | orbbec + realsense running |
| T3 | `t3_interaction.sh` | cross-node: feature_matching↔yolo, spot_on_shelf↔yolo, ctrl↔follow_head TF | T2 + servo |
| T4 | `t4_hardware.sh {servo_motion\|servo_tracking\|shelf_scene\|person\|all}` | hardware-in-the-loop, staged scenes | operator |

See `scripts/tests/README.md` for env vars and skip conditions. Logs in `scripts/tests/logs/`.

**Suite invariants:**
- `ObjectDetection`: `status=1, objects=[]` on empty scene is expected, not a failure.
- `FeatureMatching` propagates `ObjectDetection` status — `status=1, centroids=[]` is the empty-scene response.
- `get_point_cloud` uses `ApproximateTimeSynchronizer(queue_size=3, slop=0.05)`. Below ~10 Hz camera rate, color+depth can drift past 50 ms and the sync won't fire ⇒ `status=1, error_msg='No camera data for …'`. Not a node bug — check `ros2 topic hz` first, then see [`CAMERA_BRINGUP.md`](./CAMERA_BRINGUP.md) for the canonical launch that sustains ~30 Hz.
- kimi_api loads `.env` via `load_dotenv()` from CWD upward at startup. Negative "no key" tests must move `.env` aside.

Per-run results and operator-in-the-loop matrix in [`DEV_NOTES.md`](./DEV_NOTES.md).
