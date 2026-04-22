# tk26_vision

Vision stack for the Tinker 2026 robot (ROS 2 Humble, Python 3.10). Current codebase — `tk23_vision/` is legacy/reference.

## Packages

| Package | Kind | Purpose |
|---|---|---|
| `tinker_vision_msgs_26` | interface | Actions + services for tk26 vision nodes |
| `object_detection_new` | service | Generic YOLO11-seg object detection (RealSense or Orbbec) |
| `tk_vision_specialized` | action + service | Task-specific detectors: shelf-slot action, waving-person service |
| `vision_track` | action | Person tracking with ReID (Orbbec) |

Each package has its own README with usage details.

## Interfaces shipped

From `tinker_vision_msgs_26`:
- `action/SpotOnShelf.action` — shelf slot categorization
- `action/TrackPerson.action` — person tracking with ReID
- `srv/DetectWaving.srv` — detect all waving/calling persons in view

Still consumed from legacy `tinker_vision_msgs` (tk23):
- `srv/ObjectDetection.srv` — used by `object_detection_new` and by `spot_on_shelf_server`

## Camera assumptions

- **Orbbec Femto Bolt** (primary): `/camera/color/image_raw`, `/camera/depth_registered/points`, `/camera/color/camera_info`.
- **RealSense** (secondary, `object_detection_new` only): `/camera/xarm_camera/*`.

Depth reprojection to image grid is hardcoded at **720×1280** in `object_detection_new/object_seg_yolo.py` and in `tk_vision_specialized/waving_person_server.py`.

## Build

From workspace root (`/home/tinker/tk25_ws`):

```bash
colcon build --packages-up-to tinker_vision_msgs_26 object_detection_new tk_vision_specialized vision_track
source install/setup.zsh
```

Model files (`yolo11m-seg.pt`, `yolov8s.pt`, …) live at the workspace root and/or `object_detection_new/models/`. First-run Ultralytics download can block for ~30 s silently — pre-warm before demos.

## Python dependencies

ROS build deps are declared in each `package.xml`. ML runtime deps (`ultralytics`, `mediapipe`, `torch`, `torchvision`) are pip-installed — see each package's `requirements.txt` and the workspace's `src/tk25_basic/src/requirements.txt`.

## tk23 → tk26 migration status

| tk23 artifact | tk26 replacement |
|---|---|
| `tinker_vision_msgs/ObjectDetection` | still used (not yet ported) |
| `tinker_vision_msgs/DetectWaving` | `tinker_vision_msgs_26/srv/DetectWaving` |
| `tinker_vision_msgs/Categorize` (action) | `tinker_vision_msgs_26/action/SpotOnShelf` |
| `tinker_vision_msgs/FollowHeadAction`, `HumanFollowing` | `tinker_vision_msgs_26/action/TrackPerson` |
| `object_detection/waving_person` | `tk_vision_specialized/waving_person_server` |
| `object_detection/waving_client` | `tk_vision_specialized/waving_client` |
| `object_detection/check_waving_inference` | `tk_vision_specialized/check_waving_inference` |
| `object_detection/seg_yolov8`, `seg_langsam`, `tracking_yolo` | `object_detection_new/yolo_seg_node` |
| `body_track`, `deepsort_body_track`, `FollowHeadAction` | `vision_track/person_track_server` |

Still live from tk23 (not yet ported): `service_sam` (SAM + AnyGrasp), `face_recognition_arcface`, `point_direction_service`.
