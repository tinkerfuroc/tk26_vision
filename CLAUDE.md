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
ros2 run object_detection_new yolo_seg_node                 # /object_detection_yolo (specialist, excludes 'person')
ros2 run object_detection_new yolo_seg_default_node         # /object_detection (pretrained COCO, backward-compat)
ros2 run object_detection_generalist generalist_node        # /object_detection_generalist (pretrained YOLO + Gemini/FastSAM fallback)

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
├── tinker_vision_msgs_26/         # action/TrackPerson, action/SpotOnShelf, srv/ObjectDetection (new generalist)
├── object_detection_new/          # YOLO-seg: specialist (yolo_seg_node, excludes 'person') + default (yolo_seg_default_node, pretrained COCO)
├── object_detection_generalist/   # Pretrained YOLO + Gemini 2.5 Pro bbox + FastSAM mask fallback, tk26 srv
├── vision_track/                  # ByteTrack + ResNet50 ReID (custom) or YOLO BoT-SORT (native)
├── tk_vision_specialized/         # SpotOnShelf action + waving detector
├── pan_tilt/                      # ctrl (serial + TF) + follow_head (YOLO@1Hz w/ blur gate)
├── kimi_api/                      # OpenRouter LLM services; _env.py centralizes key loading
└── vision_util/                   # door_detection (Orbbec 20x20 depth heuristic), get_point_cloud (cached relay)
```

**Notes:**
- Three object-detection services now coexist. Canonical targets going forward:
  - `/object_detection_yolo` (specialist, custom-trained model, `excluded_classes=['person']`) — arena/competition items.
  - `/object_detection_generalist` (new, tk26 srv with boolean flags) — the recommended target for open-vocabulary / non-arena callers (person detection, seat-rec helpers, any class YOLO doesn't recognize).
  - `/object_detection` (pretrained COCO, tk23 srv, `excluded_classes=[]`) — kept for backward compatibility with tk25_decision BT nodes that still hard-code this name. Prefer the generalist for new code.
- The specialist (`yolo_seg_node`) now silently drops the `'person'` class regardless of what the (future) custom model emits. To detect people, use the generalist or the default node.
- `kimi_api` calls `object_detection` (generalist) via the `detection_service` ROS param — retargetable without rebuilding.
- All migrated nodes import from `tinker_vision_msgs` (tk23's package), not `tinker_vision_msgs_26`. Intentional — `tk25_decision/messages.py` also imports from it. Consolidation deferred.
- The new `tinker_vision_msgs_26/srv/ObjectDetection` references `tinker_vision_msgs/Object` by rosidl dependency, so there is no duplicate type namespace (same pattern that avoided the `DetectWaving` split).
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

## Known follow-ups

Actionable work that remains open. Full context, rationale, and prioritization in [`DEV_NOTES.md § Follow-ups`](./DEV_NOTES.md#follow-ups--ordered-roughly-by-impact). Items 1–5 and 9 from the previous follow-up list were addressed in the 2026-04-22 "Follow-up wave" session — see that DEV_NOTES entry.

1. **Specialist model training.** The custom-trained competition YOLO does not exist yet — `yolo_seg_node` currently serves pretrained `yolo11m-seg.pt`. `excluded_classes=['person']` is belt-and-suspenders for the future retrain.
2. **VLM latency** (9–14 s/call on Gemini 2.5 Pro) is the dominant cost on the fallback path. Options: lighter model (`gemini-2.5-flash`), few-shot-tune YOLO on the open-vocab classes you care about, or swap in a local open-vocab detector.
3. **Triple-subscription of camera streams** when specialist + default + generalist all run together. Not urgent, but worth factoring the input half of `YOLOSegmentationNode` into a shared node if we keep the three-service split.
4. **`BtNode_TrackPerson` / `BtNode_ScanForWavingPerson` / `BtNode_FindPointedLuggage` rearchitect** — these BT nodes were migrated to the tk26 generalist srv mechanically in Wave 2.1, but their *semantics* depend on tk23-only response fields the tk26 detection nodes never populate (`result.person_id`, `Object.being_pointed`). The full catalog of broken nodes, live-task blast radius, and recommended fix per node lives alongside the code in [`src/tk25_decision/CLAUDE.md § Known issues & broken nodes`](../../src/tk25_decision/CLAUDE.md#known-issues--broken-nodes).
