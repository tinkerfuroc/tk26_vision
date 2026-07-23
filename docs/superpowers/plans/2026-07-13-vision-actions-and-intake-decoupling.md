# Vision Actions + Intake/TF Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the six long-running VLM-backed bringup services to ROS 2
actions, migrate all in-repo callers, and factor repeated camera intake,
depth-source, depth-reprojection, and TF lookup plumbing into `vision_util`,
per `docs/superpowers/specs/2026-07-13-vision-actions-and-intake-decoupling-design.md`.

**Architecture:** Keep quick one-shot services as services. Add six action
interfaces whose goals/results mirror the existing service request/response
fields. Each converted node exposes only the action name, queues concurrent
goals FIFO for service-parity, supports cooperative cancel, and uses
`result_timeout=0`. Shared intake helpers land in `vision_util` before node
adoption. BT caller changes live in the separate `tk25_decision` repo.

**Tech Stack:** ROS 2 Humble, rclpy, message_filters, cv_bridge, tf2_ros,
tf2_geometry_msgs, pytest, `tkbuild` (`/home/tinker/tk25_ws/tkbuild`).

**Spec:** `docs/superpowers/specs/2026-07-13-vision-actions-and-intake-decoupling-design.md`

## Global Constraints

- Primary repo: `/home/tinker/tk25_ws/src/tk26_vision`, branch `tinker2-net`.
  There is concurrent WIP in this repo. Never amend/rebase/reset. Re-check
  `git status --short --branch` before every commit and stage only the exact
  files for the current task.
- BT repo: `/home/tinker/tk25_ws/src/tk25_decision`. It is a separate git repo
  and must be committed separately from `tk26_vision`.
- **One task = one git commit** unless a task explicitly says it is read-only.
  Do not bundle Wave 2 node conversions together; each converted node must cut
  over atomically in its own task.
- Build via `tkbuild tk26_vision --packages-select <pkg>`. Do not use raw
  `colcon`. Source `/home/tinker/tk25_ws/install/setup.zsh` before `ros2`
  interface/action/service checks.
- Interface rule: action Goal = old srv request fields verbatim; action Result
  = old srv response fields verbatim; action Feedback = canonical BT protocol
  fields: `int32 status`, `float32 delay_limit`, `string stage`,
  `string message`.
- Action server rule: use accept-and-queue service-parity, not goal rejection
  and not concurrent VLM execution. ActionServer callback group does not
  serialize execute callbacks; the queue does.
- Every ActionServer must set `result_timeout=0`.
- Cancellation rule: `CancelResponse.ACCEPT`; queued goals cancel without
  executing; active goals check cancellation at every stage boundary and inside
  provider retry/provider loops via `should_abort`.
- OpenAI SDK rule: every in-repo provider-chain helper touched by this work
  must construct OpenAI clients with `max_retries=0`; the chain already owns
  retry policy.
- Do not start the `waving_person_server` action conversion until both
  in-flight waving workstreams have landed and the current uncommitted waving
  files are reconciled. Waving is intentionally last.

---

## File Structure (end state)

```
src/tinker_vision_msgs_26/action/
  DetectWaving.action
  FeatureExtraction.action
  SeatRecommendation.action
  FeatureMatching.action
  SeatRecommendBbox.action
  ObjectScan.action

src/vision_util/vision_util/
  action_queue.py
  camera_intake.py
  tf_lookup.py
  depth_source.py
  depth_reproject.py          (extended)

src/vision_util/test/
  test_action_queue.py
  test_camera_intake.py
  test_tf_lookup.py
  test_depth_source.py
  test_depth_reproject_variants.py
```

BT-side end state in `/home/tinker/tk25_ws/src/tk25_decision`:

```
src/behavior_tree/behavior_tree/messages.py
src/behavior_tree/behavior_tree/mock_messages.py
src/behavior_tree/behavior_tree/TemplateNodes/ActionBase.py
src/behavior_tree/behavior_tree/TemplateNodes/Vision.py
src/behavior_tree/behavior_tree/GPSR/custom_nodes.py
src/behavior_tree/behavior_tree/Restaurant/custumNodes.py
src/behavior_tree/behavior_tree/FollowPerson/wave_reseed_cycle.py
src/behavior_tree/behavior_tree/scripts/verify_task_endpoints.py
```

---

## Wave 0: Interfaces and Shared Helpers

### Task 1: Six action interfaces in `tinker_vision_msgs_26`

**Files:**
- Create: `src/tinker_vision_msgs_26/action/DetectWaving.action`
- Create: `src/tinker_vision_msgs_26/action/FeatureExtraction.action`
- Create: `src/tinker_vision_msgs_26/action/SeatRecommendation.action`
- Create: `src/tinker_vision_msgs_26/action/FeatureMatching.action`
- Create: `src/tinker_vision_msgs_26/action/SeatRecommendBbox.action`
- Create: `src/tinker_vision_msgs_26/action/ObjectScan.action`
- Modify: `src/tinker_vision_msgs_26/CMakeLists.txt`

**Interfaces:**
- Consumes: existing `.srv` files with the same base names.
- Produces: six generated action classes used by Wave 2 server/client tasks.

- [ ] **Step 1: Write the action files**

Use the exact request/result fields from the matching `.srv`, with the
canonical feedback block appended after the second `---`.

`FeatureExtraction.action`:

```
string camera
---
int32 status
string error_msg
string feature
sensor_msgs/Image comparison_image
---
int32 status
float32 delay_limit
string stage
string message
```

`SeatRecommendation.action`:

```
string camera
string[] names
string[] features
---
int32 status
string error_msg
string recommendation
---
int32 status
float32 delay_limit
string stage
string message
```

`FeatureMatching.action`:

```
string camera
string[] features
sensor_msgs/Image[] comparison_images
float32 max_distance
string target_frame
---
int32 status
string error_msg
geometry_msgs/PointStamped[] centroids
---
int32 status
float32 delay_limit
string stage
string message
```

`SeatRecommendBbox.action`:

```
# Recommend where a new guest should sit. Returns the human-readable
# sentence (same semantics as SeatRecommendation.srv) plus the 2D
# bounding box and 3D centroid of the recommended empty seat.
string camera
string[] names
string[] features
string target_frame
# Optional pre-known seat labels. When non-empty, the VLM is instructed
# to choose `recommendation` from this list (or return "none"); the
# server rejects out-of-catalog labels with status=1. Empty preserves
# open-vocabulary behavior.
string[] known_seats
---
int32 status
string error_msg
string recommendation
tinker_vision_msgs_26/BoundingBox bbox
geometry_msgs/PointStamped centroid
---
int32 status
float32 delay_limit
string stage
string message
```

`ObjectScan.action`:

```
# Direct-VLM, labels-only scene scan over a candidate vocabulary.
#
# Splits `vocabulary` into batches, runs one vision-LLM call per batch (all
# batches in parallel; Gemini primary -> Qwen fallback), and returns the subset
# of the vocabulary actually visible in the scene. Labels only -- no bounding
# boxes, masks, depth, or centroids.

# camera in ['orbbec', 'realsense'] (substring match; defaults to orbbec).
string camera

# Candidate class names to look for.
string[] vocabulary
---
std_msgs/Header header
# 0 = ok (found_labels may be empty on a genuinely empty scene).
# 1 = failure (no camera frame / empty vocabulary / every VLM batch failed).
int32 status
string error_msg
# Subset of `vocabulary` present in the scene. Deduped, vocabulary order.
string[] found_labels
---
int32 status
float32 delay_limit
string stage
string message
```

`DetectWaving.action`:

```
float32 threshold_meters
string target_frame
int32 min_waving_persons
---
int32 status
string error_msg
geometry_msgs/PointStamped[] waving_persons
sensor_msgs/RegionOfInterest[] waving_boxes  # 1:1 with waving_persons; image-space boxes for re-seed

sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image
sensor_msgs/Image[] segments
---
int32 status
float32 delay_limit
string stage
string message
```

- [ ] **Step 2: Register actions in `CMakeLists.txt`**

Add the six action file entries in `rosidl_generate_interfaces`, next to the
existing action entries. Do not remove the legacy `.srv` entries.

- [ ] **Step 3: Build and inspect generated interfaces**

```bash
tkbuild tk26_vision --packages-select tinker_vision_msgs_26
source /home/tinker/tk25_ws/install/setup.zsh
ros2 interface show tinker_vision_msgs_26/action/FeatureMatching
ros2 interface show tinker_vision_msgs_26/action/DetectWaving
```

Expected: build succeeds; shown interfaces contain the exact goal/result fields
and feedback has `status`, `delay_limit`, `stage`, `message`.

- [ ] **Step 4: Commit**

```bash
git status --short --branch
git add src/tinker_vision_msgs_26/action/DetectWaving.action \
        src/tinker_vision_msgs_26/action/FeatureExtraction.action \
        src/tinker_vision_msgs_26/action/SeatRecommendation.action \
        src/tinker_vision_msgs_26/action/FeatureMatching.action \
        src/tinker_vision_msgs_26/action/SeatRecommendBbox.action \
        src/tinker_vision_msgs_26/action/ObjectScan.action \
        src/tinker_vision_msgs_26/CMakeLists.txt
git commit -m "feat(tinker_vision_msgs_26): add VLM query action interfaces"
```

---

### Task 2: Shared queued-action execution helper

**Files:**
- Create: `src/vision_util/vision_util/action_queue.py`
- Create: `src/vision_util/test/test_action_queue.py`

**Interfaces:**
- Produces: a small helper used by Wave 2 nodes to serialize accepted goals
  FIFO while preserving rclpy action semantics.

- [ ] **Step 1: Implement the helper**

Implement a minimal `QueuedActionGate` that can be used as an ActionServer
`handle_accepted_callback`:

- `accept(goal_handle)`: append to FIFO; if no active goal, start the next
  handle by calling `goal_handle.execute()`.
- `notify_finished(goal_handle)`: called in each execute callback `finally`
  block; clears the active goal and starts the next non-canceled queued handle.
- `cancel_queued(goal_handle) -> bool`: marks queued goals canceled before
  execution when cancel is accepted.
- The helper must not reject goals and must not start more than one goal at a
  time.

Keep the helper rclpy-light: tests can use fake goal handles with `execute()`,
`is_cancel_requested`, and identity comparison.

- [ ] **Step 2: Add tests**

Cover:
- first accepted handle executes immediately;
- second/third accepted handles do not execute until prior `notify_finished`;
- order is FIFO;
- cancel-before-execute skips execution and allows the next queued handle to
  run;
- double `notify_finished` is harmless and logged or ignored.

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=src/vision_util python3 -m pytest src/vision_util/test/test_action_queue.py -v
```

- [ ] **Step 4: Commit**

```bash
git status --short --branch
git add src/vision_util/vision_util/action_queue.py src/vision_util/test/test_action_queue.py
git commit -m "feat(vision_util): add FIFO action goal gate"
```

---

### Task 3: Shared intake, TF, depth-source, and reprojection helpers

**Files:**
- Create: `src/vision_util/vision_util/camera_intake.py`
- Create: `src/vision_util/vision_util/tf_lookup.py`
- Create: `src/vision_util/vision_util/depth_source.py`
- Modify: `src/vision_util/vision_util/depth_reproject.py`
- Modify: `src/vision_util/package.xml`
- Create: `src/vision_util/test/test_camera_intake.py`
- Create: `src/vision_util/test/test_tf_lookup.py`
- Create: `src/vision_util/test/test_depth_source.py`
- Create: `src/vision_util/test/test_depth_reproject_variants.py`

**Interfaces:**
- Produces `CameraIntake`, `IntakeConfig`, `StreamSpec`, `FrameBundle`,
  `TransformHelper`, and `FfsPreferredDepthSource`.

- [ ] **Step 1: Implement `camera_intake.py`**

Required behavior:
- per-stream `StreamSpec(topic, best_effort=True, qos_depth=5)`;
- `age_source='recv'|'stamp'`;
- `latest(max_age_s=None)`;
- `wait_fresh(max_age_s, timeout_s, poll_s=0.05, on_timeout='fail'|'stale')`;
- `latest_new(last_seq)` with tri-state result: bundle, `NO_NEW_FRAME`, or
  `None` for no data yet;
- lazy `color_bgr()`, `depth_m()`, and `points_xyz(roi=None, valid_band=...)`;
- read-only numpy outputs;
- decode failure drops the bad bundle and keeps the previous good bundle.

- [ ] **Step 2: Implement `tf_lookup.py`**

`TransformHelper` must provide `try_lookup`, `wait_lookup`, `transform_point`,
and a public `buffer` escape hatch. Add `tf2_ros` and `tf2_geometry_msgs` to
`vision_util/package.xml`.

- [ ] **Step 3: Implement `depth_source.py`**

Move the FFS-preferred depth acquisition policy out of `object_seg_yolo.py`
without changing behavior. The helper API is
`acquire(align_to_color: bool) -> (depth_m, source_tag)`; it does not accept a
`FrameBundle` because FFS captures its own stereo pair.

- [ ] **Step 4: Extend `depth_reproject.py`**

Add explicit valid-band/clip parameters and named variants needed by section 4.4 of
the spec. Preserve the RealSense body-axes variant exactly and name it as a
bug-compatible convention, not canonical pinhole math.

- [ ] **Step 5: Add unit tests**

Cover the test matrix from spec section 6:
- sync pairing, both age sources, both timeout modes, tri-state consume,
  decode-failure drop, rgb8/bgr8 normalization, read-only outputs;
- static TF lookup success/failure and wait timeout;
- FFS prefer/fallback/timeout paths with mocked service client;
- golden equivalence for each depth-reprojection variant against the current
  node-local math.

- [ ] **Step 6: Run tests**

```bash
PYTHONPATH=src/vision_util python3 -m pytest \
  src/vision_util/test/test_camera_intake.py \
  src/vision_util/test/test_tf_lookup.py \
  src/vision_util/test/test_depth_source.py \
  src/vision_util/test/test_depth_reproject_variants.py -v
```

- [ ] **Step 7: Commit**

```bash
git status --short --branch
git add src/vision_util/vision_util/camera_intake.py \
        src/vision_util/vision_util/tf_lookup.py \
        src/vision_util/vision_util/depth_source.py \
        src/vision_util/vision_util/depth_reproject.py \
        src/vision_util/package.xml \
        src/vision_util/test/test_camera_intake.py \
        src/vision_util/test/test_tf_lookup.py \
        src/vision_util/test/test_depth_source.py \
        src/vision_util/test/test_depth_reproject_variants.py
git commit -m "feat(vision_util): shared camera intake, TF lookup, and depth-source helpers"
```

---

## Wave 1: Intake Adoption for Stay-Service Nodes

### Task 4: Adopt intake/depth-source helpers in detection base nodes

**Files:**
- Modify: `src/object_detection_new/object_detection_new/object_seg_yolo.py`
- Modify: `src/object_detection_new/object_detection_new/generalist_node.py`
- Add/modify focused tests under `src/object_detection_new/test/` if available.

**Interfaces:**
- Consumes: `CameraIntake`, `FfsPreferredDepthSource`, `TransformHelper`.
- Produces: unchanged services `/object_detection_yolo`,
  `/object_detection`, and `/object_detection_generalist`.

- [ ] Replace duplicated subscriber/sync/cache setup with one
  `CameraIntake` per camera. Use `age_source='recv'`,
  `on_timeout='fail'`, and preserve existing per-stream QoS.
- [ ] Move `_acquire_depth` behavior to `FfsPreferredDepthSource`. The
  RealSense path must still bypass `FrameBundle.depth_m()` when FFS is used.
- [ ] Replace private TF buffer/lookup with `TransformHelper` while keeping
  `_frame_supports_tf_transform` as caller policy.
- [ ] Preserve the RealSense body-axes reprojection behavior exactly.
- [ ] Run the package tests or at minimum import/pytest tests that exercise
  the modified helpers.
- [ ] Build:

```bash
tkbuild tk26_vision --packages-select vision_util object_detection_new
```

- [ ] Commit only this package's files.

---

### Task 5: Adopt intake helpers in `door_detection`

**Files:**
- Modify: `src/vision_util/vision_util/door_detection.py`
- Modify/add: `src/vision_util/test/test_door_detection.py`

**Interfaces:**
- Consumes: `CameraIntake`.
- Produces: unchanged `door_detection_srv`.

- [ ] Replace raw depth/camera_info subscriptions with a depth+info
  `CameraIntake` instance. No color stream, no ATS.
- [ ] Preserve Orbbec-only request handling and status/error semantics.
- [ ] Run focused tests:

```bash
PYTHONPATH=src/vision_util python3 -m pytest src/vision_util/test/test_door_detection.py -v
```

- [ ] Build `vision_util` and commit.

---

### Task 6: Adopt intake helpers in `get_image`

**Files:**
- Modify: `src/vision_util/vision_util/get_image.py`
- Add/modify focused tests under `src/vision_util/test/`.

**Interfaces:**
- Consumes: `CameraIntake`.
- Produces: unchanged `get_image_service`.

- [ ] Replace the two ATS/cache blocks with two color+depth `CameraIntake`
  instances, preserving params, defaults, and staleness-free relay semantics.
- [ ] Preserve `depth` request handling: missing depth only fails when depth
  was requested.
- [ ] Build `vision_util` and commit.

---

## Wave 2: Action Conversion

### Task 7: BT action base and mock-message readiness (`tk25_decision`)

**Repo:** `/home/tinker/tk25_ws/src/tk25_decision`

**Files:**
- Modify: `src/behavior_tree/behavior_tree/TemplateNodes/ActionBase.py`
- Modify: `src/behavior_tree/behavior_tree/messages.py`
- Modify: `src/behavior_tree/behavior_tree/mock_messages.py`

**Interfaces:**
- Produces BT support needed by every converted node.

- [ ] In `ActionBase.py`, send cancel on action timeout before returning
  FAILURE.
- [ ] In `terminate()`, cancel on `RUNNING -> FAILURE` as well as
  `RUNNING -> INVALID`.
- [ ] In `setup()`, add ServiceHandler-style mock-type parity: if the action
  type comes from `mock_messages`, force mock mode instead of constructing an
  `rclpy.action.ActionClient` with a mock class.
- [ ] In `messages.py`, import action types with aliases, keeping srv imports
  alive. Example: `DetectWaving as DetectWavingAction`.
- [ ] In `mock_messages.py`, add `MockAction` classes for
  `DetectWavingAction`, `FeatureExtractionAction`,
  `SeatRecommendationAction`, `FeatureMatchingAction`,
  `SeatRecommendBboxAction`, and `ObjectScanAction`, with Goal/Result fields
  matching the generated actions.
- [ ] Run BT mock/import tests available in the repo.
- [ ] Commit in `tk25_decision` only.

---

### Task 8: Convert `feature_matching_service` to an action

**Files in `tk26_vision`:**
- Modify: `src/kimi_api/kimi_api/feature_matching.py`
- Modify: `src/kimi_api/kimi_api/_match_vlm.py`
- Add/modify tests under `src/kimi_api/test/` if present.

**Files in `tk25_decision`:**
- Modify: `src/behavior_tree/behavior_tree/TemplateNodes/Vision.py`
  (`BtNode_FeatureMatching`)

**Interfaces:**
- Consumes: `FeatureMatching.action`, `QueuedActionGate`, `TransformHelper`.
- Produces: action server named `feature_matching_service`.

- [ ] Add `ActionServer` with `result_timeout=0`, `CancelResponse.ACCEPT`,
  FIFO handle-accepted queue, and staged feedback.
- [ ] Keep service result semantics in the action result. Remove or stop
  creating the legacy service server in the cutover commit.
- [ ] Thread `should_abort` through `_match_vlm.py` provider/retry loops and
  set OpenAI `max_retries=0`.
- [ ] Replace private TF lookup block with `TransformHelper`.
- [ ] Upgrade `main()` to `MultiThreadedExecutor`.
- [ ] Migrate `BtNode_FeatureMatching` from `ServiceHandler` to
  `ActionHandler` using `FeatureMatchingAction`.
- [ ] Run focused Python tests plus `tkbuild` for `tinker_vision_msgs_26`,
  `vision_util`, and `kimi_api`.
- [ ] Commit `tk26_vision` and `tk25_decision` separately if both repos are
  touched.

---

### Task 9: Convert `feature_extraction_service` and `seat_recommend_service`

**Files in `tk26_vision`:**
- Modify: `src/kimi_api/kimi_api/feature_recognition.py`
- Modify: `src/kimi_api/kimi_api/_feature_vlm.py`

**Files in `tk25_decision`:**
- Modify: `src/behavior_tree/behavior_tree/TemplateNodes/Vision.py`
  (`BtNode_FeatureExtraction`, `BtNode_SeatRecommend`)

**Interfaces:**
- Consumes: `FeatureExtraction.action`, `SeatRecommendation.action`,
  `QueuedActionGate`, `CameraIntake`.
- Produces: action servers named `feature_extraction_service` and
  `seat_recommend_service`.

- [ ] Add two ActionServers sharing one per-node FIFO worker/gate.
- [ ] Give the seat path and camera subscriptions explicit callback groups;
  do not leave them on the node default group.
- [ ] Upgrade `main()` to `MultiThreadedExecutor`.
- [ ] Adopt `CameraIntake` for the seat-path color frame cache.
- [ ] Thread `should_abort` through `_feature_vlm.py` and set OpenAI
  `max_retries=0`.
- [ ] Add a small lock around `VisionLogger._ensure_run_dir` if concurrent
  first-use is now possible.
- [ ] Migrate both BT nodes to `ActionHandler` using aliased action types.
- [ ] Run focused tests/build and commit per repo.

---

### Task 10: Convert `seat_recommend_bbox_service`

**Files in `tk26_vision`:**
- Modify: `src/kimi_api/kimi_api/seat_recommend_bbox.py`
- Modify: `src/kimi_api/kimi_api/_seat_bbox_vlm.py`

**Files in `tk25_decision`:**
- Modify: `src/behavior_tree/behavior_tree/TemplateNodes/Vision.py`
  (`BtNode_SeatRecommendBbox`)

**Interfaces:**
- Consumes: `SeatRecommendBbox.action`, `CameraIntake`, `TransformHelper`.
- Produces: action server named `seat_recommend_bbox_service`.

- [ ] Add queued ActionServer with staged feedback and `result_timeout=0`.
- [ ] Adopt `CameraIntake` for color+depth, preserving QoS and pre-VLM
  snapshot policy.
- [ ] Replace private TF block with `TransformHelper`.
- [ ] Thread `should_abort` through `_seat_bbox_vlm.py` and set OpenAI
  `max_retries=0`.
- [ ] Migrate BT node to `ActionHandler`.
- [ ] Run focused tests/build and commit per repo.

---

### Task 11: Convert `object_scan`

**Files in `tk26_vision`:**
- Modify: `src/kimi_api/kimi_api/object_scan.py`
- Modify: `src/kimi_api/kimi_api/_scan_vlm.py`

**Files in `tk25_decision`:**
- Modify: `src/behavior_tree/behavior_tree/TemplateNodes/Vision.py`
  (`BtNode_ObjectScan`)

**Interfaces:**
- Consumes: `ObjectScan.action`, `CameraIntake`.
- Produces: action server named `object_scan`.

- [ ] Add queued ActionServer with feedback per batch (`batch i/N`) and
  `result_timeout=0`.
- [ ] Adopt two color-only `CameraIntake` instances. Use
  `latest(max_age_s=1.0)` with node-clock receive age, as flagged in the spec.
- [ ] Thread `should_abort` through `_scan_vlm.py`; cancel should stop before
  launching remaining batches and skip remaining provider retries.
- [ ] Set OpenAI `max_retries=0`.
- [ ] Migrate BT node to `ActionHandler`.
- [ ] Add/adjust T1/T2 checks because object_scan had no existing action
  coverage.
- [ ] Run focused tests/build and commit per repo.

---

### Task 12: Convert `detect_waving_persons` last

**Files in `tk26_vision`:**
- Modify: `src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/waving_bench.py`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/waving_client.py`
- Modify: `src/tk_vision_specialized/README.md`
- Modify: `src/vision_track/vision_track/track_web.py`
- Modify: `src/vision_track/vision_track/webui/app.js`
- Modify: `src/restaurant_nav_test_web/restaurant_nav_test_web.py`

**Files in `tk25_decision`:**
- Modify: `src/behavior_tree/behavior_tree/TemplateNodes/Vision.py`
- Modify: `src/behavior_tree/behavior_tree/GPSR/custom_nodes.py`
- Modify: `src/behavior_tree/behavior_tree/Restaurant/custumNodes.py`
- Modify: `src/behavior_tree/behavior_tree/FollowPerson/wave_reseed_cycle.py`
- Modify owner wiring/tests for `wave_reseed_cycle` as needed.

**Interfaces:**
- Consumes: `DetectWaving.action`, `CameraIntake`, `TransformHelper`.
- Produces: action server named `detect_waving_persons`.

- [ ] Confirm both 2026-07-04 waving workstreams have landed and reconcile
  current uncommitted waving files before editing.
- [ ] Add queued ActionServer with `result_timeout=0`.
- [ ] Adopt `CameraIntake.wait_fresh(age_source='stamp', on_timeout='stale')`.
- [ ] Replace `_snapshot_latest_transform` with `TransformHelper.wait_lookup`.
- [ ] Replace local waving depth reprojection with canonical helper using the
  same valid-band/clip semantics.
- [ ] Thread `should_abort` into `_waving_vlm.py`; do not merely abandon
  futures. Set OpenAI `max_retries=0`.
- [ ] Update all BT callers, `_WaveReseedBridge`, track_web button,
  waving_bench, waving_client, restaurant readiness probe, and docs.
- [ ] Run package tests, BT mock tests, and a live/recorded parity check
  before commit.
- [ ] Commit per repo.

---

## Wave 2 Integration Sweep

### Task 13: T-suite, endpoint checks, and docs sweep

**Files:**
- Modify: `scripts/tests/` T0/T1/T2/T3 scripts as needed.
- Modify: `CLAUDE.md`
- Modify: `src/kimi_api/README.md`
- Modify: `src/vision_bringup/docs/vision-bringup-design.md`
- Modify docs in `tk25_decision` and `tk26_sim` only if those repos are in
  scope for the current worker.

- [ ] T0 interface checks include the six actions while retaining legacy srvs.
- [ ] T1 advert checks use `wait_for_action` for the six converted names.
- [ ] Add missing checks for detect_waving, object_scan, and
  seat_recommend_bbox.
- [ ] T2/T3 callers use action clients.
- [ ] `behavior_tree/scripts/verify_task_endpoints.py` moves converted entries
  from `services` to `actions`.
- [ ] Run the endpoint checker and the relevant manual shell scripts.
- [ ] Commit docs/test changes per repo.

---

## Wave 3: Tuned Trackers Intake Adoption

### Task 14: Adopt `CameraIntake` in `person_track_server`

**Files:**
- Modify: `src/vision_track/vision_track/person_track_node.py`
- Add/modify focused tests under `src/vision_track/test/`.

- [ ] Use `latest_new` tri-state with QoS 5 and the existing Reentrant group.
- [ ] Preserve seq-consume behavior, stall watchdog, EMA/FSM, and current TF.
- [ ] Preserve read-only/copy-on-write image contracts.
- [ ] Run tracker tests/build and commit.

---

### Task 15: Adopt `CameraIntake` in `follow_head`

**Files:**
- Modify: `src/pan_tilt/pan_tilt/follow_head.py`
- Add/modify focused tests under `src/pan_tilt/test/`.

- [ ] Replace header dedup with `FrameBundle.seq` dedup.
- [ ] Preserve the 5 Hz cap branch's consume-without-processing behavior.
- [ ] Keep analytic servo frame behavior; no TF helper is added here.
- [ ] Run pan_tilt tests/build and commit.

---

## Final Verification

- [ ] Full interface build:

```bash
tkbuild tk26_vision --packages-select tinker_vision_msgs_26 vision_util kimi_api tk_vision_specialized object_detection_new vision_track pan_tilt
```

- [ ] Source install and check actions/services:

```bash
source /home/tinker/tk25_ws/install/setup.zsh
ros2 interface show tinker_vision_msgs_26/action/DetectWaving
ros2 interface show tinker_vision_msgs_26/action/ObjectScan
```

- [ ] Run all focused pytest suites added by this plan.
- [ ] Run T1 advertise checks on bringup: the six converted names appear as
  actions, not services.
- [ ] For each converted node, run one recorded/live parity test comparing
  action result fields against the previous service response semantics.
- [ ] Measure DDS endpoint/topic count and bringup timing before/after Wave 2
  on the robot; record the numbers in the task's final commit or follow-up
  note.
