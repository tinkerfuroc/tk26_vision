# tinker_vision_msgs_26

Canonical ROS 2 interface package for tk26 vision. Holds **all** messages, services, and actions used by the vision stack. Consumers (and any producer outside this package) should `<depend>tinker_vision_msgs_26</depend>` rather than putting `.action`/`.srv` files next to code.

As of the tk23→tk26 migration, the tk23 `tinker_vision_msgs` package has been retired and `src/tk23_vision/` is fully `COLCON_IGNORE`d. Every type previously defined there now lives here.

## Interfaces

### Messages

`BoundingBox`, `Face`, `FaceResult`, `Object`, `Objects`, `PanTiltCtrl`.

### Actions

| File | Server | Purpose |
|---|---|---|
| `action/SpotOnShelf.action` | `tk_vision_specialized/spot_on_shelf_server` | Shelf slot categorization — maps detected items to (layer, horizontal-grid). |
| `action/TrackPerson.action` | `vision_track/person_track_server` | Person tracking with ReID (Orbbec). |
| `action/Categorize.action` | `kimi_api/grocery_categorize` | Grocery categorization action. |
| `action/FollowHeadAction.action` | `pan_tilt/follow_head` | Pan-tilt head following. |
| `action/HumanFollowing.action` | (legacy) | Retained from tk23 for back-compat. |

### Services

Two `ObjectDetection`-shaped services coexist in this package, deliberately named to disambiguate:

| File | Server | Purpose |
|---|---|---|
| `srv/ObjectDetection.srv` | `object_detection_new/yolo_seg_node` (specialist) + `yolo_seg_default_node` (pretrained COCO) | Legacy string-flag schema inherited from tk23. Kept for back-compat with tk25_decision BTs that hard-code `/object_detection`. |
| `srv/ObjectDetectionGeneralist.srv` | `object_detection_generalist/generalist_node` | Clean YOLO + optional VLM+SAM open-vocabulary detection with typed boolean flags. |

Additional services: `DetectWaving`, `DoorDetection`, `FaceRegister`, `FeatureExtraction`, `FeatureMatching`, `FollowHead`, `GetImage`, `GetPointCloud`, `ObjectDetectionImage`, `PointDirection`, `SeatRecommendation`.

#### Generalist vs. legacy `ObjectDetection` — field mapping

| Legacy `ObjectDetection.srv` | Generalist `ObjectDetectionGeneralist.srv` | Notes |
|---|---|---|
| `string flags` (substring-parsed: `'sort_closest'`, `'sort_highest'`, `'sort_none'`, `'request_image'`, `'request_segments'`) | `bool sort_closest`, `bool sort_highest`, `bool return_rgb_image`, `bool return_depth_image`, `bool return_segments` | Typed booleans — no string parsing. |
| `string category` | *(dropped)* | Unused. |
| — | `bool force_vlm_sam`, `bool use_vlm_sam_fallback` | Generalist-only: opt in/out of the VLM+SAM path. |
| — | `string detection_source` (response) | `'yolo'` / `'vlm_sam'` / `'none'` — which branch answered. |
| `Object[] objects` | `Object[] objects` | Both reference the `Object.msg` in this package. |

## Build dependencies

Declared in `package.xml` / `CMakeLists.txt`: `geometry_msgs`, `std_msgs`, `sensor_msgs`, `rosidl_default_generators`, `rosidl_default_runtime`.

## Adding a new interface

1. Drop the `.action` / `.srv` / `.msg` file into the matching folder.
2. Add its path to `rosidl_generate_interfaces` in `CMakeLists.txt`. Update `DEPENDENCIES` if the new file references a package that's not already listed.
3. If the new file uses types from a new package, add `<depend>…</depend>` to `package.xml`.
4. `colcon build --packages-select tinker_vision_msgs_26` and re-source.
5. Append `build/tinker_vision_msgs_26/rosidl_generator_py` to `.vscode/settings.json` `python.analysis.extraPaths` (already listed — no change needed as of this README).

## Shape reference

### `SpotOnShelf.action`

```
# Goal
geometry_msgs/PoseStamped shelf_left
geometry_msgs/PoseStamped shelf_right
float32[] shelf_heights
string[] item_ids
---
# Result
int32 status
string error_msg
int32[] item_height_grids
int32[] item_horizontal_grids
---
# Feedback
int32 status
string message
```

### `TrackPerson.action`

```
# Goal
string target_frame          # TF target frame for target_position; "" = raw camera frame
string target_point_topic    # topic on which the server publishes PointStamped
                             # at its native tracking rate; "" → server param default
                             # (/target_points) — matches tk26_nav's tracking_server
bool return_rgb_img
bool return_depth_img
bool return_segment
bool debug
---
# Result
int32 status
string message
---
# Feedback
bool target_lost
int32 track_id
bool transformation_success
geometry_msgs/PointStamped target_position
# (optional images if requested)
```

The server publishes `target_position` on `target_point_topic` at the native
tracking rate (independent of BT tick cadence), but **only** when the target
is not lost and the TF transform to `target_frame` succeeded. This gating
means `tk26_nav/tracking_server`'s own LOST timer engages cleanly when the
stream goes silent.

