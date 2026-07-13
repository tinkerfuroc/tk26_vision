# Camera Server Design — per-camera C++ snapshot/point-cloud/TF servers

- **Date:** 2026-07-13
- **Status:** Draft — pending user review
- **Scope of this effort:** servers only. Consumer migration is deferred (consumer
  packages are under active refactor) and documented as a map in Appendix A.

## 1. Motivation

A workspace-wide survey (2026-07-13) of subscriptions to the wrist RealSense
(`/camera/xarm_camera/*`) and head Orbbec Femto Bolt (`/camera/*`) topics found:

- **~25 nodes hold always-on RGB-D subscriptions** (most with a private
  `ApproximateTimeSynchronizer`) purely to cache a "latest frame" that is only
  read when a service/action request arrives. On `/camera/color/image_raw`
  alone, ~10–13 concurrent Python subscribers each deserialize 30 Hz 1080p
  frames they mostly discard.
- All camera consumption is concentrated in `tk26_vision` and
  `tk25_manipulation`. The behavior tree is already fully service-driven;
  navigation consumes only derived `PointStamped` targets; `tk26_sim` is a
  publisher.
- The genuinely continuous consumers are few and already pipelined:
  FoundationStereo → depth_decimator → robot_segmenter → nvblox (cuMotion
  collision backbone), `person_track_server`, `follow_head`.
- QoS is inconsistent across consumers (explicit BEST_EFFORT vs accidental
  RELIABLE defaults), and every consumer runs its **own** `tf2` buffer +
  listener with scattered cache times (10 s / 30 s / 60 s / 180 s). Several
  nodes look up transforms at latest-time where frame-stamp would be correct
  (e.g. `pick_and_place` transforms clouds at `TimePointZero`).

**Goal:** one C++ server per camera holds the only streaming subscriptions and
a continuously warm TF buffer; everything else obtains synchronized frames,
point clouds, and time-correct transforms on demand via services.

## 2. Scope

### In scope (this effort)

1. New ament_cmake package `camera_server` under `src/tk26_vision/src/`.
2. One executable (`camera_server_node`) launched twice:
   `wrist_camera_server` (RealSense) and `head_camera_server` (Orbbec).
3. New service interfaces in `tinker_vision_msgs_26`:
   `GetCameraSnapshot`, `GetCameraPointCloud`, `GetTransform`, plus status msg
   `CameraServerStatus`.
4. A separate lightweight `camera_compat_bridge` executable serving the legacy
   service names (`get_image_service`, `get_point_cloud_service`,
   `get_orbbec_pc`) verbatim — **param-gated, OFF by default**.
5. Launch wiring: head server added to `vision_bringup/launch/vision_driver.launch.py`;
   a standalone dev launch in `camera_server`. (Wrist-server inclusion in the
   manipulation bringup is a separate later commit in that repo — concurrent
   committer caution applies there.)
6. Tests per the tk26_vision T0/T1/T2 tier conventions, including parity
   harnesses against the existing Python services.

### Out of scope (deferred, see Appendix A)

- All consumer conversions (~21 nodes) — their packages are mid-refactor.
- Retiring the Python utility nodes (`get_image`, `get_point_cloud`,
  `get_orbbec_pc`) — they keep serving the legacy names until the bridge is
  enabled in bringup (cutover, §11).
- The continuous pipeline (FFS/decimator/segmenter/nvblox), `person_track_server`,
  `follow_head` — streaming by design, untouched.
- Calibration tools (`handeye_*`, `calibrate_*`, `capture_intrinsic`),
  legacy fold variants, diagnostics (`check_waving_inference`,
  `restaurant_nav_test_web`, `track_web`, `depth_colorizer`).
- IR / extrinsics streams (owned by the FFS path).

## 3. Architecture

```
RealSense driver ──color/aligned_depth/infos──▶ wrist_camera_server ──┐
                                                (rclcpp, standalone)  │  services:
Orbbec driver ────color/reg_depth/infos───────▶ head_camera_server ───┤  ~/get_snapshot
                                                (rclcpp, standalone)  │  ~/get_point_cloud
/tf, /tf_static ─────────────────────────────▶ (both, warm buffers)   │  ~/get_transform
                                                                      │
legacy callers (BT, pick_and_place) ─▶ camera_compat_bridge ──client──┘
                                       (no subscriptions, gated OFF)
```

- **Standalone processes** (decided): each server is its own executable with
  its own `MultiThreadedExecutor`. Exactly one DDS deserialization per stream
  system-wide once consumers migrate. The class is registered as an rclcpp
  component too (`RCLCPP_COMPONENTS_REGISTER_NODE`), so in-container
  composition next to a driver remains a later, measured optimization — not a
  launch-blocking bet.
- **One binary, per-instance parameters.** Node name, topics, and frames are
  parameters; services live under the node namespace
  (`/wrist_camera_server/get_snapshot`, `/head_camera_server/get_snapshot`, …).

### Per-instance stream wiring (defaults)

| Param | wrist_camera_server | head_camera_server |
|---|---|---|
| `color_topic` | `/camera/xarm_camera/color/image_raw` | `/camera/color/image_raw` |
| `depth_topic` | `/camera/xarm_camera/aligned_depth_to_color/image_raw` | `/camera/depth/image_raw` (registered to color) |
| `color_info_topic` | `/camera/xarm_camera/color/camera_info` | `/camera/color/camera_info` |
| `depth_info_topic` | `/camera/xarm_camera/aligned_depth_to_color/camera_info` | `/camera/depth/camera_info` |

Both cameras therefore serve **color-aligned depth**; deprojection uses
`depth_info` intrinsics and packs RGB from the synced color frame (same frame).

## 4. New interfaces (`tinker_vision_msgs_26`)

Status convention matches the existing ecosystem: `int32 status` (0 = OK,
non-zero = failure class below) + `string error_msg`.

### 4.1 `srv/GetCameraSnapshot.srv`

```
# Streams to include (payload control; server caches everything regardless).
bool want_color true
bool want_depth true
bool want_camera_info true

# TF: for each entry, include transform target_frames[i] <- (returned frame_id)
# looked up at the returned pair stamp.
string[] target_frames

# Freshness. All zero => newest cached pair, no waiting.
float32 max_age_sec        # >0: fail with STALE if cached pair older than this
builtin_interfaces/Time captured_after   # non-zero: wait for pair stamped >= this
float32 wait_timeout_sec   # bound on the captured_after wait (default 0 => server default)
---
int32 status               # 0 OK; 1 NO_DATA; 2 STALE; 3 WAIT_TIMEOUT; 5 BAD_REQUEST
string error_msg
builtin_interfaces/Time stamp   # stamp of the synced (color, depth) pair
string frame_id                 # optical frame of the returned data
sensor_msgs/Image color
sensor_msgs/Image depth
sensor_msgs/CameraInfo color_info
sensor_msgs/CameraInfo depth_info
geometry_msgs/TransformStamped[] transforms   # index-aligned with target_frames
bool[] transforms_ok                          # per-transform success; failures detailed in error_msg
```

Semantics:

- `WAIT_TIMEOUT` still populates the newest available pair (stamped in
  `stamp`) so the caller can decide; `error_msg` states the shortfall.
- TF failure never fails the snapshot: frames are returned,
  `transforms_ok[i]=false` for the failed pair(s).
- `captured_after` + `wait_timeout_sec` replaces today's ad-hoc
  capture-at-pose patterns (e.g. scan_and_place reading a possibly pre-motion
  "latest" frame) with an explicit, correct primitive.

### 4.2 `srv/GetCameraPointCloud.srv`

```
uint32 stride              # pixel stride; 0 or 1 = full resolution
bool include_color         # true => XYZRGB, false => XYZ
string target_frame        # empty => native optical frame; else transformed at pair stamp
float32 max_age_sec
builtin_interfaces/Time captured_after
float32 wait_timeout_sec
---
int32 status               # 0 OK; 1 NO_DATA; 2 STALE; 3 WAIT_TIMEOUT; 4 TF_FAIL; 5 BAD_REQUEST
string error_msg
builtin_interfaces/Time stamp
sensor_msgs/PointCloud2 points
```

Deprojected **in C++, on demand** from the cached depth + intrinsics
(precomputed per-intrinsics xy-table, invalidated on intrinsics change) —
this absorbs `get_orbbec_pc` (CUDA Python) and `get_point_cloud` (Python).
`target_frame` transforms during packing with the frame-stamped transform —
fixing, for future migrated callers, the latest-time cloud transforms in
`pick_and_place::pc_proc`. If TF at stamp fails: `TF_FAIL`, no cloud.

### 4.3 `srv/GetTransform.srv`

```
string target_frame
string source_frame
builtin_interfaces/Time time    # zero => latest available
float32 timeout_sec             # capped by server param (default cap 2.0)
---
int32 status               # 0 OK; 1 UNAVAILABLE; 5 BAD_REQUEST
string error_msg
geometry_msgs/TransformStamped transform
```

Serves the warm long-cache buffer to on-demand consumers so a cold node can do
time-correct lookups without running its own listener (the user-requested TF
requirement). Callers that need TF at times other than a frame stamp (VLA EE
pose, post-VLM lookups) use this directly.

### 4.4 `msg/CameraServerStatus.msg` (observability, §8)

```
builtin_interfaces/Time last_pair_stamp
float32 color_age_sec
float32 depth_age_sec
float32 pair_age_sec
float32 sync_fps           # synced-pair rate over the last window
uint64 pair_seq
```

## 5. Server internals

### 5.1 Subscriptions and QoS

| Stream | QoS (subscribe) | Rationale |
|---|---|---|
| color, depth | BEST_EFFORT, VOLATILE, KEEP_LAST(5) | Compatible with both the Orbbec BEST_EFFORT publisher and the RealSense RELIABLE publisher (RELIABLE pub → BEST_EFFORT sub is a valid DDS match). One explicit choice replaces today's per-node inconsistency. |
| color_info, depth_info | RELIABLE, VOLATILE, KEEP_LAST(10) | Both drivers publish info RELIABLE. |

Sync: `message_filters::Synchronizer<ApproximateTime<Image, Image>>`,
`queue_size=10`, `slop` param default **0.1 s** (the dominant existing value;
the 0.05 s variants are documented to stop firing below ~10 Hz camera rate).

### 5.2 Latest-frame store

- Stores `ConstSharedPtr`s: last synced `(color, depth)` pair + pair stamp +
  monotonic `pair_seq`, plus latest `color_info`/`depth_info`.
- Single mutex; readers copy the shared_ptrs out under the lock (no image
  copies until response serialization).
- A `condition_variable` signaled on every synced pair implements the
  `captured_after` wait.

### 5.3 Threading model

- `MultiThreadedExecutor`, `num_threads` param default **4**.
- Subscription + sync callbacks: one `MutuallyExclusive` callback group
  (cheap store-only work, keeps ordering).
- Each service: `Reentrant` group, so a handler blocked in a
  `captured_after` wait cannot starve the subscription callbacks or other
  service calls. Waiting handlers occupy an executor thread; with 4 threads
  and the tk26_vision compute-budget pattern (few concurrent callers) this is
  ample. `wait_timeout_sec` is capped by param (`max_wait_sec`, default 2.0).

### 5.4 TF

- `tf2_ros::Buffer` with `cache_time` param default **180 s** (matches the
  largest existing consumer need: `seat_recommend_bbox`'s VLM round-trip),
  `tf2_ros::TransformListener` on its own dedicated thread/node so TF
  ingestion never competes with service load.
- Snapshot/cloud lookups happen at the **pair stamp** with per-lookup timeout
  param (`tf_lookup_timeout_sec`, default 0.1).

### 5.5 Deprojection

- CPU, single pass over the depth image: `z = depth(u,v)` (`16UC1` mm → m, or
  `32FC1` m — both encodings handled, as anygrasp does today),
  `x = (u-cx)/fx * z`, `y = (v-cy)/fy * z` via the cached xy-table; optional
  RGB pack from the synced color image; optional stride; optional in-pass
  transform by the frame-stamped Eigen isometry.
- Output: unorganized `PointCloud2`, `x,y,z[,rgb]` float32 fields, invalid
  (z==0/NaN) pixels dropped — matching `get_orbbec_pc` output shape; parity
  verified in T2.
- Estimated cost: head 640×576 ≈ 0.37 M px, single-digit ms; wrist 1280×720 ≈
  0.92 M px, ~10–25 ms. Well inside the 10 s vision-call budget.

## 6. Compat bridge (`camera_compat_bridge`)

The legacy names are **cross-camera** (`camera: 'realsense'|'orbbec'` request
field), so they cannot live inside either per-camera server without coupling
the two. A separate forwarder keeps both servers clean and gives the legacy
surface a clean deletion story.

- **No subscriptions.** Pure service-client forwarder; gated by launch/param
  `enable_legacy_services` (default **false** — the Python utility nodes keep
  owning these names until cutover, avoiding duplicate-service collisions).
- Mapping (exact request/response types and semantics preserved, including
  `status=1` + `Unsupported camera: …` / `No camera data for …` message
  shapes):

| Legacy service | Type | Forwarded to |
|---|---|---|
| `get_image_service` | `GetImage` | `<server(camera)>/get_snapshot` (`want_depth = request.depth`) |
| `get_point_cloud_service` | `GetPointCloud` | `<server(camera)>/get_point_cloud` (`include_color=true`, full res) |
| `get_orbbec_pc` | `GetOrbbecPC` | `/head_camera_server/get_point_cloud` (`stride`, `include_color` passed through) |

- `camera` → server namespace is a parameter map
  (`realsense: /wrist_camera_server`, `orbbec: /head_camera_server`).
- Async forwarding with a bounded deadline; unreachable server ⇒ `status=1`
  with a distinguishable `error_msg`.

## 7. Error handling

- Camera silent / never seen: `NO_DATA` with the age of whatever exists in
  `error_msg`. Server startup does not block on cameras (T1 requirement).
- Stale pair vs `max_age_sec`: `STALE`, no wait (waiting is opt-in via
  `captured_after`).
- `captured_after` unmet within the wait bound: `WAIT_TIMEOUT`, newest pair
  still returned (§4.1).
- TF misses: snapshot degrades per-transform (`transforms_ok`); point cloud
  with `target_frame` fails closed (`TF_FAIL`) — a silently untransformed
  cloud is worse than an error.
- Intrinsics not yet received but frames present: `NO_DATA` for point cloud,
  snapshot succeeds with `want_camera_info` honored best-effort (empty info +
  note in `error_msg`).

## 8. Observability

- 1 Hz `~/status` publisher (`CameraServerStatus`): stream ages, synced-pair
  rate, seq. Cheap, and gives T2 and ops a one-topic health view.
- Throttled (10 s) WARN logs on stream starvation (no synced pair for >2 s
  while at least one input stream is alive) and on sync-partner skew.

## 9. Launch & deployment

- `camera_server/launch/camera_server.launch.py` — standalone dev launch
  (either instance by argument).
- `vision_driver.launch.py` gains `head_camera_server` behind
  `enable_camera_server` (default **true** — its service names are new, so it
  is additive and collision-free; it inherits the vision stack's FastDDS
  profile environment like every other head-camera consumer).
- Wrist instance: launched next to the RealSense driver in the manipulation
  bringup (which owns that camera). That wiring lands as its own later commit
  in the manipulation repo. The manip side does **not** use the fastdds_shm
  profile (workspace rule); the server makes no transport assumptions.
- Dependency direction: `camera_server` lives in tk26_vision;
  manipulation launching a tk26_vision node is consistent with the existing
  FFS arrangement (manip→vision allowed; reverse for tk25_basic is not).
- New package ships `README.md` with an append-only Changelog, updated in the
  same commits as code (workspace policy).
- Build via `tkbuild tk26_vision` (never raw colcon). C++ is unaffected by the
  venv shebang concerns.

## 10. Testing

Follows the tk26_vision tier conventions (`scripts/tests/`):

- **T0 (static):** package + srv/msg generation build clean under tkbuild;
  lint; interface files present in the install tree (stale-overlay check per
  workspace history).
- **T1 (node startup, no cameras):** both instances start and stay alive;
  `get_snapshot`/`get_point_cloud` return `NO_DATA` (not hang/crash);
  `get_transform` answers from `/tf_static`; `~/status` publishes.
- **T2 (live cameras):**
  - snapshot freshness: `pair_age` bounded; color/depth stamp skew ≤ slop;
  - `captured_after` semantics: request stamped "now" returns a strictly
    newer pair within the wait bound;
  - point-cloud parity: XYZ[RGB] cloud vs (a) existing `get_orbbec_pc`
    response and (b) driver `/camera/depth_registered/points` — point count
    and centroid within tolerance;
  - TF: snapshot `transforms` match a reference `lookup_transform` at the
    same stamp; `get_transform` at a 60–170 s-old stamp still answers
    (long-cache verification);
  - compat bridge (enabled in a test namespace): byte-level response-shape
    parity against the live Python services for identical requests;
  - light concurrency soak: N parallel snapshot + cloud calls, no starvation
    of the store (status topic keeps updating).
- Per-run results recorded in `DEV_NOTES.md` per repo practice.

## 11. Cutover plan (future phases, not this effort)

1. Servers land + verified live (T2) alongside the untouched existing stack.
2. Flip: enable `camera_compat_bridge`, disable the three Python utility
   nodes in `vision_bringup` — zero-change for BT / pick_and_place callers.
3. Consumers migrate per-package to the new services as their refactors land
   (Appendix A is the map); each migration deletes that node's subscriptions,
   sync machinery, and private TF listener.
4. Delete the bridge + retire the Python utility nodes once no caller uses
   the legacy names.
5. Follow-on optimizations unlocked: disable the Orbbec driver's colored
   point cloud (`enable_colored_point_cloud:=false`) once nothing subscribes
   `/camera/depth_registered/points` / `/camera/depth/points` directly;
   optionally compose servers into driver containers (component registration
   already in place).

## 12. Risks / notes

- **Payload sizes:** snapshot responses are multi-MB (wrist RGB ~2.7 MB +
  depth ~1.8 MB). The existing Python services already return identical
  payloads over the same transport, so this is proven in-tree; call rates are
  low (on-demand).
- **Transitional load:** until consumers migrate, the servers add one more
  subscriber per stream (net −13 later). Head-side /dev/shm segment history
  (fastrtps leak, 2026-07-04) is noted; the server adds standard subscribers
  under the same profile as existing vision nodes.
- **Single point of failure:** a server crash takes out on-demand vision for
  that camera. Mitigations: tiny store-only hot path, T1 no-camera
  resilience, `~/status` heartbeat for fast detection. (Consumers currently
  each fail independently, but also all fail together when a *driver* dies —
  the failure domain barely changes.)
- **Concurrent committers:** tk26_vision `tinker2-net` has in-flight work
  (kimi_api, tk_vision_specialized). This effort adds only new files + one
  guarded block in `vision_driver.launch.py`; commit-new-only, no amends.

## Appendix A — deferred consumer migration map

| Consumer (node) | Today | Future call |
|---|---|---|
| `yolo_seg_node`, `yolo_seg_default_node`, `generalist_node` | own RS+OB RGB-D syncs + infos + 60 s TF buffer | `get_snapshot(camera by request, target_frames=[request.target_frame])` |
| `object_match_server`, `placing_location_server` (YOLO subclasses) | inherited subs + TF | same as above |
| `object_match_all_server` (`camera_data_source`) | RS RGB-D sync + OB color+`depth_registered/points` sync + TF | `get_snapshot` (RS) / `get_point_cloud(include_color=true)` (OB) |
| `waving_person_server` | OB RGB-D sync, latest-time TF snapshot | `get_snapshot(target_frames=[…])` (gets time-correct TF for free) |
| `feature_recognition` | OB color + info | `get_snapshot(want_depth=false)` |
| `seat_recommend_bbox` | OB RGB-D sync + 180 s TF buffer | `get_snapshot` + `get_transform` (post-VLM lookups) |
| `grocery_categorize` | OB `/camera/depth/points` (2 Hz cache) + TF | `get_point_cloud(include_color=false)` + `get_transform` |
| `object_scan` | OB+RS color | `get_snapshot(want_depth=false)` per camera |
| `door_detection` | OB depth + color info | `get_snapshot(want_color=false)` |
| `get_image`, `get_point_cloud`, `get_orbbec_pc` (nodes) | Python service nodes with own subs | **retired**; names served by compat bridge until callers migrate |
| `monocular_depth_pc` | RS+OB RGB-D syncs | `get_snapshot` per camera |
| `scan_and_place_server` | wrist RGB-D+info subs + 60 s TF | `get_snapshot(captured_after=arrival_time, target_frames=[base_link])` |
| `fold/fold_clothing_server` | wrist RGB-D+info subs + TF | `get_snapshot(target_frames=[base_link])` |
| `openpi_bridge` | head+wrist color subs + 30 s TF | `get_snapshot(want_depth=false)` per tick + `get_transform` (EE pose) |
| `vla_action` | head color sub + 30 s TF | same pattern |
| `anygrasp_ros2` | wrist aligned-depth camera_info sub | `get_snapshot(want_color=false, want_depth=false, want_camera_info=true)` |
| `pick_and_place` | dormant `/camera/depth_registered/points` sub | delete sub; clouds via `get_point_cloud(target_frame=base_link)` — fixes the `TimePointZero` transform |

Not migrating (by decision): continuous pipeline, calibration tools, legacy
fold variants, diagnostics — see §2.

## Appendix B — survey provenance

Four parallel code surveys on 2026-07-13 (tk26_vision; tk25_manipulation +
nvblox/cuMotion/FFS; remainder of workspace; TF usage across all camera
consumers). Full per-file findings (file:line, topics as-written vs resolved,
QoS, sync, callback behavior, TF frame pairs and lookup times) are in the
session transcript; headline numbers are in §1.
