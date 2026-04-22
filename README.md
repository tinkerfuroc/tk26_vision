# tk26_vision

Vision module for Tinker 2026 based on ROS2 Humble.

## Features

- **Object Detection**: YOLO11-seg based detection and segmentation, plus a default-YOLO variant for generalist queries
- **Person Tracking**: YOLO + ByteTrack + ReID for persistent identity tracking
- **Pan-tilt Head Control**: Serial servo driver with TF broadcasting + YOLO head-following
- **LLM-backed Services (OpenRouter)**: Person feature extraction, description-to-image matching, and grocery-shelf categorization
- **Utility Services**: Door detection (Orbbec depth heuristic), point-cloud relay
- **Specialized Vision Tasks**: Spot-on-shelf action server

## Packages

| Package | Description |
|---------|-------------|
| `tinker_vision_msgs_26` | Custom message and action definitions (`TrackPerson`, `SpotOnShelf`) |
| `object_detection_new` | YOLO segmentation detection — two service variants (`object_detection_yolo` custom, `object_detection` default) |
| `vision_track` | Person tracking action server with ReID |
| `tk_vision_specialized` | `SpotOnShelf` action server |
| `pan_tilt` | Pan-tilt servo control + YOLO head follow (migrated from tk23_vision) |
| `kimi_api` | OpenRouter LLM-backed feature extraction / matching / grocery categorization (migrated from tk23_vision) |
| `vision_util` | `door_detection` + `get_point_cloud` utility services (migrated from tk23_vision `util`) |

Corresponding tk23 packages are `COLCON_IGNORE`'d — see [CLAUDE.md § Reverting](./CLAUDE.md#reverting) for the mapping and revert procedure.

## Build Instructions

### Prerequisites

- ROS2 Humble installed
- Python 3.10
- CUDA-capable GPU (recommended for real-time performance)
- Serial device at `/dev/ttyUSB0` if using `pan_tilt/ctrl` (override via `-p device:=…`)

### 1. Create Virtual Environment

```bash
cd /path/to/tk26_vision
python3.10 -m venv .venv-vision-main --symlinks
```

### 2. Add ROS2 Python Packages

Create `.venv-vision-main/lib/python3.10/site-packages/ros2_packages.pth`:

```python
import sys, os; _ros='/opt/ros/humble/lib/python3.10/site-packages'; _local='/opt/ros/humble/local/lib/python3.10/dist-packages'; sys.path.extend([_ros, _local]) if _ros not in sys.path else None; (print('\n⚠️  ROS2 Python packages added to Python path:\n   ' + _ros + '\n   ' + _local + '\n   Note: pip in this venv can see but CANNOT manage these packages.\n   Set ROS2_PTH_WARNED=1 to suppress.\n'), os.environ.__setitem__('ROS2_PTH_WARNED', '1')) if not os.environ.get('ROS2_PTH_WARNED') else None
```

### 3. Install Dependencies

```bash
source .venv-vision-main/bin/activate
pip install --upgrade pip wheel setuptools

# Core (cv_bridge needs numpy<2)
ROS2_PTH_WARNED=1 pip install "numpy<2" "opencv-python<4.10" ultralytics netifaces pydot typeguard

# Per-package extras (installed the first time the migration was applied; leave here for reproducibility)
ROS2_PTH_WARNED=1 pip install -r src/kimi_api/requirements.txt      # openai, python-dotenv, scipy
ROS2_PTH_WARNED=1 pip install -r src/pan_tilt/requirements.txt      # pyserial, transforms3d (ultralytics/opencv already present)
ROS2_PTH_WARNED=1 pip install -r src/vision_track/requirements.txt  # torch, etc.
```

### 4. Configure OpenRouter (only if using kimi_api)

```bash
cp src/kimi_api/.env.example <workspace-root>/.env
# then edit .env and set OPENROUTER_API_KEY
```

`python-dotenv` auto-loads `.env` from CWD or any ancestor when each kimi_api node starts. Optional overrides: `OPENROUTER_BASE_URL`, `LLM_MODEL`.

### 5. Build — use the wrapper

Use `src/tk26_vision/scripts/build.sh`, which sources venv + ROS, runs `colcon build --symlink-install`, and patches install-tree shebangs so `ros2 run` invokes the venv python. Plain `colcon build` produces `#!/usr/bin/python3` shebangs that can't import from the venv.

```bash
# From workspace root (~/tk25_ws)
./src/tk26_vision/scripts/build.sh
./src/tk26_vision/scripts/build.sh --packages-select pan_tilt kimi_api
./src/tk26_vision/scripts/build.sh --packages-up-to object_detection_new
```

If you build with plain `colcon`, follow up with:
```bash
./src/tk26_vision/scripts/fix_venv_shebangs.sh
```
(idempotent; covers `object_detection_new`, `vision_util`, `pan_tilt`, `kimi_api`, `vision_track`, `tk_vision_specialized`).

Build `tinker_vision_msgs_26` (CMake) as usual — shebang fix doesn't apply. Clean + rebuild on symlink errors:
```bash
rm -rf build/tinker_vision_msgs_26 install/tinker_vision_msgs_26
colcon build --packages-select tinker_vision_msgs_26
```

## Running Nodes

```bash
source .venv-vision-main/bin/activate
source install/setup.bash

# Detection
ros2 run object_detection_new yolo_seg_node           # /object_detection_yolo (custom model)
ros2 run object_detection_new yolo_seg_default_node   # /object_detection (default model)

# Person tracking
ros2 run vision_track person_track_server
ros2 run vision_track person_track_test_client

# Specialized
ros2 run tk_vision_specialized spot_on_shelf_server

# Pan-tilt
ros2 run pan_tilt ctrl --ros-args -p device:=/dev/ttyUSB0
ros2 run pan_tilt follow_head

# LLM-backed (requires OPENROUTER_API_KEY in env or .env)
ros2 run kimi_api feature_recognition
ros2 run kimi_api feature_matching
ros2 run kimi_api grocery_categorize

# Utilities
ros2 run vision_util door_detection
ros2 run vision_util get_point_cloud
```

## Camera Setup

**Don't run the camera launches bare** — the vendored defaults drop to ~3 Hz together due to a USB QoS / UDP-buffer compound bug. Use the canonical launch sequence from [CAMERA_BRINGUP.md](./CAMERA_BRINGUP.md):

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/src/tk26_vision/config/fastdds_shm.xml

# RealSense
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=xarm_camera \
    align_depth.enable:=true \
    config_file:=$(pwd)/src/tk26_vision/config/realsense_qos.yaml

# Orbbec Femto Bolt
ros2 launch orbbec_camera femto_bolt.launch.py \
    depth_registration:=true \
    enable_ir:=false \
    enable_frame_sync:=false
```

Expected: ~30 Hz on every color / depth topic. See [CAMERA_BRINGUP.md](./CAMERA_BRINGUP.md) for the three compounding root causes, verification commands, and sudo-optional kernel-buffer alternative.

## Model Files

YOLO models are stored in `object_detection_new/models/` or downloaded automatically by Ultralytics.

Supported models:
- `yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`, `yolo11l-seg.pt`, `yolo11x-seg.pt`
- `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'openai'` (or `dotenv`, `ultralytics`, `serial`) when running `ros2 run …` | Shebangs were built with `/usr/bin/python3`. Run `./src/tk26_vision/scripts/fix_venv_shebangs.sh` or rebuild via `build.sh`. |
| `RuntimeError: OPENROUTER_API_KEY is not set` | Copy `src/kimi_api/.env.example` to a `.env` at workspace root (or ancestor of CWD), fill in the key. |
| `bad interpreter: ../../.venv-vision-main/bin/python` (on `vision_track`) | Relative shebang from a stale build; rerun the shebang fixer (or `build.sh`). |
| `numpy` version conflict with `cv_bridge` | Use `numpy<2` (cv_bridge compiled against NumPy 1.x). |
| CUDA out of memory | Use smaller model (yolo11n-seg.pt) or reduce `inference_size`. |
| Build fails for `tinker_vision_msgs_26` | `rm -rf build/tinker_vision_msgs_26 install/tinker_vision_msgs_26 && colcon build --packages-select tinker_vision_msgs_26` |
| `serial.serialutil.SerialException` opening `/dev/ttyUSB0` | No servo attached, or override with `--ros-args -p device:=/dev/ttyUSB1`. |
| kimi_api service client hangs on `object_detection` | Make sure `ros2 run object_detection_new yolo_seg_default_node` is running (it serves the generalist `object_detection` service). |

## Architecture

```
src/tk26_vision/src/
├── tinker_vision_msgs_26/
│   └── action/{TrackPerson,SpotOnShelf}.action
├── object_detection_new/
│   └── object_detection_new/
│       ├── object_seg_yolo.py          # Parameterized YOLO seg node
│       └── object_seg_yolo_default.py  # Wrapper: default model + service name
├── vision_track/
│   └── vision_track/{person_track_node.py, yolo_tracker.py, reid/, core/}
├── tk_vision_specialized/
│   └── tk_vision_specialized/spot_on_shelf_server.py
├── pan_tilt/
│   ├── pan_tilt/{pan_tilt_ctrl.py, follow_head.py, calibration/}
│   └── config/specs.json
├── kimi_api/
│   ├── kimi_api/{feature_recognition,feature_matching,grocery_categorize,_env}.py
│   └── .env.example
├── vision_util/
│   └── vision_util/{door_detection,get_point_cloud}.py
└── scripts/
    ├── build.sh              # Wrapper: venv + ROS + colcon + shebang fix
    └── fix_venv_shebangs.sh  # Idempotent shebang rewriter
```

## Reverting the migration

See [CLAUDE.md § Reverting](./CLAUDE.md#reverting) for step-by-step revert procedures covering:
- Shebang fix only (`rm -rf src/tk26_vision/scripts/`)
- tk23→tk26 migration only (removes new packages, un-hides tk23)
- Full revert
- Optional cleanup of Python deps installed into the venv

## Related Documentation

- [CLAUDE.md](./CLAUDE.md) — detailed architecture, env setup, running nodes, revert procedure
- Workspace root: `/home/tinker/tk25_ws/CLAUDE.md`
