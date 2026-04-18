# tinker_vision_msgs_26

ROS 2 interface package for tk26 vision. Actions and services only — no messages. Consumers (and any producer outside this package) should `<depend>tinker_vision_msgs_26</depend>` rather than putting `.action`/`.srv` files next to code.

## Interfaces

### Actions

| File | Server | Purpose |
|---|---|---|
| `action/SpotOnShelf.action` | `tk_vision_specialized/spot_on_shelf_server` | Shelf slot categorization — maps detected items to (layer, horizontal-grid). |
| `action/TrackPerson.action` | `vision_track/person_track_server` | Person tracking with ReID (Orbbec). |

### Services

| File | Server | Purpose |
|---|---|---|
| `srv/DetectWaving.srv` | `tk_vision_specialized/waving_person_server` | Return all currently-waving persons as PointStamped centroids. |

## Build dependencies

Declared in `package.xml` / `CMakeLists.txt`: `geometry_msgs`, `std_msgs`, `sensor_msgs`, `rosidl_default_generators`, `rosidl_default_runtime`.

## Adding a new interface

1. Drop the `.action` / `.srv` / `.msg` file into the matching folder.
2. Add its path to `rosidl_generate_interfaces` in `CMakeLists.txt`. Update `DEPENDENCIES` if the new file references a package that's not already listed.
3. If the new file uses types from a new package, add `<depend>…</depend>` to `package.xml`.
4. `colcon build --packages-select tinker_vision_msgs_26` and re-source.
5. Append `build/tinker_vision_msgs_26/rosidl_generator_py` to `.vscode/settings.json` `python.analysis.extraPaths` (already listed — no change needed as of this README).

## Shape reference

### `DetectWaving.srv`

```
float32 threshold_meters     # z-distance cutoff in metres; ≤0 = no limit
string  target_frame         # TF target frame for output points (empty = raw camera frame)
---
int32   status               # 0=found, 1=none, -1=error
string  error_msg
geometry_msgs/PointStamped[] waving_persons   # sorted closest-first on z

sensor_msgs/Image  rgb_image     # reserved; current server does not populate
sensor_msgs/Image  depth_image   # reserved; current server does not populate
sensor_msgs/Image[] segments     # reserved; current server does not populate
```

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

## Legacy note

Several types still come from the tk23 package `tinker_vision_msgs` (e.g. `ObjectDetection.srv`, `PanTiltCtrl.msg`). Those have not been ported to tk26 yet — consumers import from both packages for now.
