# FFS-first depth in detection nodes via `prefer_ffs` param

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make object-detection nodes that consume RealSense depth (specialist `object_seg_yolo`, default `yolo_seg_default_node`, and `object_detection_generalist`) prefer `foundation_stereo/get_depth` over the camera's native aligned depth, with automatic fallback to native when FFS is unavailable. Zero change to the detection service request/response schemas — the policy lives entirely inside the nodes, gated by a new per-node `prefer_ffs` ROS param.

**Motivation:** FFS streaming depth at 30 Hz burns the GPU for detection services that fire once every 10–30 s. The FFS node already exposes `~/get_depth` on-demand (`stream_enabled` defaults to `false` in `foundation_stereo.yaml:31`), and the realsense detection nodes already block on a sync'd `(color, depth)` pair before they do anything. Wiring the on-demand FFS call into the depth-acquisition step gets FFS-quality depth without keeping the GPU saturated, and falling back to native depth on FFS-down keeps detection working unconditionally.

**Architecture:** Single `_acquire_depth(...)` seam per detection node replaces the inline `bridge.imgmsg_to_cv2(depth_msg, "passthrough")` call. When `prefer_ffs=True` and `camera=='realsense'`, the helper calls `/foundation_stereo/get_depth` with a short service-discovery deadline (200 ms `wait_for_service`) and call timeout (8 s, generous for cold TRT engine); on any of {service-unavailable, timeout, non-zero status, future failure} it falls back to the cached native realsense depth that the existing `ApproximateTimeSynchronizer` callback has already pulled. Branch on response encoding (`32FC1` meters from FFS vs `16UC1` mm from realsense) so the downstream pinhole-backproject math is unchanged. Orbbec branch is untouched (FFS only consumes realsense IR stereo).

**Tech Stack:** Python 3.10, ROS2 Humble (`rclpy`, `message_filters`, `cv_bridge`), existing `.venv-vision-main` (no new deps). FFS service contract: `tinker_vision_msgs_26/srv/FoundationStereoDepth` at `/foundation_stereo/get_depth`.

**Spec:** N/A — design was iterated in the discord:1509181974290628650 channel on 2026-05-28 (message ids `1509520152150933635` → `1509521988052324353`). This plan file is the canonical artifact.

---

## File map

**Modify:**
- `src/object_detection_new/object_detection_new/object_seg_yolo.py` (add params, `_acquire_depth` helper, FFS client, throttled warn; route realsense path through helper)
- `src/object_detection_generalist/object_detection_generalist/generalist_node.py` (verify the inherited `_process_realsense_data` picks up the new helper for free; if `generalist_node` overrides depth acquisition, mirror the change)
- `src/object_detection_new/config/default.yaml` (declare `prefer_ffs` + sibling params)
- `src/object_detection_new/object_detection_new/yolo_seg_default_node.py` (mirror change if it doesn't inherit the helper from `object_seg_yolo`)

**Create:**
- `src/object_detection_generalist/config/default.yaml` (currently no yaml; mirror the new param block here)

**Verify (no change expected, but read to confirm assumptions):**
- `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py:154,590` — service name + signature
- `src/tinker_vision_msgs_26/srv/FoundationStereoDepth.srv` — request/response shape

---

## Task 1: Confirm FFS service contract and current depth-acquisition seams

Pure read-only verification before writing code. The Discord-drafted plan was based on grep, not full reads of every file. Confirm the assumptions on disk.

- [ ] **Step 1: Read FFS service + msg definition**

  Read:
  - `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py:1-60` (imports, node name) and `:580-630` (service callback)
  - `src/tinker_vision_msgs_26/srv/FoundationStereoDepth.srv` (request + response fields)

  Confirm:
  - Absolute service path is `/foundation_stereo/get_depth` (node name `foundation_stereo` + relative srv `get_depth`)
  - Request fields are optional (no required stereo-pair input from caller)
  - Response carries `sensor_msgs/Image depth_image` in `32FC1` (meters, float) plus `sensor_msgs/CameraInfo` and `int8 status` (0 = ok)
  - Response header frame_id is the color optical frame when `align_to_color=true`

- [ ] **Step 2: Map current depth-acquisition seams in detection nodes**

  Read in full:
  - `src/object_detection_new/object_detection_new/object_seg_yolo.py` lines around `_process_realsense_data` (~`:400-440`) and the `_declare_parameters` / `__init__` blocks (~`:90-150`)
  - `src/object_detection_new/object_detection_new/yolo_seg_default_node.py` (full file — confirm whether it subclasses `YOLOSegmentationNode` and inherits the depth path, or duplicates it)
  - `src/object_detection_generalist/object_detection_generalist/generalist_node.py` lines around `_process_realsense_data` (~`:280-300`) and `_declare_parameters` (~`:160-210`)
  - `src/object_detection_new/config/default.yaml` (full)

  Record (in scratch notes for the next tasks):
  - Exact line numbers of the `bridge.imgmsg_to_cv2(depth_msg, "passthrough")` (or equivalent) calls
  - Whether `yolo_seg_default_node` and `generalist_node` inherit from `YOLOSegmentationNode` (if yes, the helper added there flows through for free)
  - The callback-group type used for the detection service callback (MutuallyExclusive vs Reentrant) — needed for Task 3
  - Whether `__main__` uses `MultiThreadedExecutor` or `SingleThreadedExecutor`

- [ ] **Step 3: Read the side-thread executor pattern this seam will reuse**

  Read `src/vision_util/vision_util/get_orbbec_pc.py` (or `src/kimi_api/kimi_api/feature_matching.py`) for the existing pattern that does `call_async` from inside another service callback. Note the exact shape: separate `SingleThreadedExecutor` on a side thread, or `ReentrantCallbackGroup` + `Event`-driven response wait. Whichever the codebase already does, mirror it here — don't invent a new variant.

---

## Task 2: Add `prefer_ffs` param schema to detection nodes

Per-node param decls (matches the existing yaml-or-CLI pattern; no shared yaml). Default `prefer_ffs=true` so this is a behavior change on next deploy — but the fallback path keeps the system working even when FFS is down, so the change is safe.

- [ ] **Step 1: Add param decls to `object_seg_yolo.py`**

  In `_declare_parameters` (around `object_seg_yolo.py:93`), add:

  ```python
  ('prefer_ffs', True),
  ('ffs_service', '/foundation_stereo/get_depth'),
  ('ffs_wait_for_service_s', 0.2),
  ('ffs_call_timeout_s', 8.0),
  ('ffs_align_to_color', True),
  ('ffs_fallback_log_period_s', 30.0),
  ```

  Match the surrounding declaration style (the existing decls there use `declare_parameter(name, default)` one-per-line, or a `declare_parameters(namespace, list-of-tuples)` block — read what's there and match).

  Read these out into `self.prefer_ffs`, `self.ffs_service`, etc. — read **per-call** (or via `get_parameter` at call time), not cached at init, so `ros2 param set` flips take effect on the next request without restart.

- [ ] **Step 2: Mirror in `generalist_node.py` if it does not inherit**

  If Task 1 Step 2 found `generalist_node` does not inherit `_declare_parameters` from `YOLOSegmentationNode`, copy the same 6 params into its `_declare_parameters` (around `:164`). If it does inherit, skip — the new params come along for free.

- [ ] **Step 3: Mirror in `yolo_seg_default_node.py` if it does not inherit**

  Same check as Step 2 for `yolo_seg_default_node`.

- [ ] **Step 4: Add `prefer_ffs: true` to existing yaml**

  Edit `src/object_detection_new/config/default.yaml` — add the 6 params under whichever node namespace it already uses (likely `/object_detection_yolo`, `/object_detection`, or both). Match the existing yaml indentation + key ordering.

- [ ] **Step 5: Create yaml for generalist**

  Create `src/object_detection_generalist/config/default.yaml` mirroring `object_detection_new/config/default.yaml`'s structure for the 6 new params under `/object_detection_generalist`. Only the FFS-related params need to be there for this PR — don't backfill other generalist params unless they're already missing from a launch.

---

## Task 3: Implement `_acquire_depth` + `_try_ffs_depth` helpers

This is the meat of the change. One helper replaces the inline `passthrough` conversion; one helper isolates the FFS call + fallback decision.

- [ ] **Step 1: Add imports + member init**

  In `object_seg_yolo.py`:
  - Top of file: `from tinker_vision_msgs_26.srv import FoundationStereoDepth`
  - In `__init__` (after the param decls): `self._ffs_cli = None` (lazy-init on first use to avoid paying the discovery cost at startup when FFS may not be up yet)
  - In `__init__`: `self._ffs_fallback_last_warn = 0.0` (for the rate-limited warn)
  - If the side-thread executor pattern from Task 1 Step 3 needs a `ReentrantCallbackGroup`, create it: `self._ffs_cb_group = ReentrantCallbackGroup()` and pass it when creating the client.

- [ ] **Step 2: Implement `_try_ffs_depth(self) -> Optional[np.ndarray]`**

  ```python
  def _try_ffs_depth(self):
      """Return float32 depth in meters from FFS, or None on any failure."""
      if self._ffs_cli is None:
          self._ffs_cli = self.create_client(
              FoundationStereoDepth,
              self.get_parameter('ffs_service').value,
              callback_group=self._ffs_cb_group,  # only if you added one in Step 1
          )
      if not self._ffs_cli.wait_for_service(
          timeout_sec=self.get_parameter('ffs_wait_for_service_s').value
      ):
          return None
      req = FoundationStereoDepth.Request()
      req.align_to_color = self.get_parameter('ffs_align_to_color').value
      fut = self._ffs_cli.call_async(req)
      # Use whichever wait mechanism Task 1 Step 3 identified — Event-driven
      # response callback OR side-thread SingleThreadedExecutor. Do NOT
      # rclpy.spin_until_future_complete here — we're already inside a
      # service callback on the main executor.
      resp = _await_future(fut, timeout_s=self.get_parameter('ffs_call_timeout_s').value)
      if resp is None or resp.status != 0:
          return None
      return self.bridge.imgmsg_to_cv2(resp.depth_image, 'passthrough').astype('float32')  # already 32FC1 meters
  ```

  `_await_future` is whatever the existing pattern uses — reuse it (don't redefine).

- [ ] **Step 3: Implement rate-limited fallback warn**

  ```python
  def _warn_fallback_throttled(self):
      now = self.get_clock().now().nanoseconds * 1e-9
      period = self.get_parameter('ffs_fallback_log_period_s').value
      if now - self._ffs_fallback_last_warn >= period:
          self.get_logger().warn('FFS unavailable, falling back to native depth')
          self._ffs_fallback_last_warn = now
  ```

  Note: `rclpy.Logger` has a `throttle_duration_sec` kwarg on `warn(...)` in some Humble builds — prefer it if available; otherwise the manual gate above is fine.

- [ ] **Step 4: Implement `_acquire_depth` and route the realsense path through it**

  ```python
  def _acquire_depth(self, camera: str, depth_msg) -> Tuple[np.ndarray, str]:
      """Returns (depth_in_meters_float32, source). source ∈ {'ffs','native'}."""
      if camera == 'realsense' and self.get_parameter('prefer_ffs').value:
          d = self._try_ffs_depth()
          if d is not None:
              return d, 'ffs'
          self._warn_fallback_throttled()
      # Native path: depth_msg is 16UC1 mm; convert to float32 m
      native = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
      if native.dtype != np.float32:
          native = native.astype('float32') / 1000.0  # mm → m
      return native, 'native'
  ```

  Replace the inline `bridge.imgmsg_to_cv2(depth_msg, "passthrough")` call in `_process_realsense_data` (`object_seg_yolo.py` around `:407-431` — confirm from Task 1 Step 2) with:

  ```python
  depth_m, depth_source = self._acquire_depth('realsense', depth_msg)
  ```

  Audit the rest of `_process_realsense_data` for any code that assumed `int16 mm` units (look for `* 1000`, `/ 1000`, `dtype` checks, raw `depth_array[y, x]` index into integer mm). The helper guarantees `float32 m` output now, so any downstream math that was implicitly `mm` needs to be `m`.

- [ ] **Step 5: Verify generalist + default-node paths pick up the helper**

  If they inherit `_process_realsense_data` from `YOLOSegmentationNode`, no further code change. If either overrides it, port the same one-liner replacement in those overrides.

- [ ] **Step 6: Hook `depth_source` into the existing vision_log sidecar JSON**

  Find where the per-call vision_log JSON is written (existing per-node helper — likely in the request handler near the response build). Add `depth_source` (the value returned from `_acquire_depth`) to the sidecar dict so post-hoc audits can tell which source served each call.

---

## Task 4: Verify build + smoke tests

- [ ] **Step 1: Build**

  ```bash
  ./src/tk26_vision/scripts/build.sh --packages-select object_detection_new object_detection_generalist tinker_vision_msgs_26
  ```

  Build must succeed clean — no new warnings about missing srv imports or unused vars.

- [ ] **Step 2: T0 static**

  ```bash
  ./src/tk26_vision/scripts/tests/t0_static.sh
  ```

  Existing test must still pass — no new fields to register at the static level (no msg/srv changes).

- [ ] **Step 3: T1 startup with FFS node DOWN**

  Run only the detection nodes (no `foundation_stereo` launch). Then:

  ```bash
  source install/setup.bash
  ros2 run object_detection_new yolo_seg_node &
  ros2 service call /object_detection_yolo tinker_vision_msgs_26/srv/ObjectDetection "{}"  # adapt to actual srv shape
  ```

  Expect:
  - Service returns `status=1, objects=[]` (empty scene OK)
  - Exactly one warn line per 30 s in stderr: `FFS unavailable, falling back to native depth`
  - Sidecar JSON in `vision_log/<session>/yolo_seg_node_*_req_*.json` shows `"depth_source": "native"`

- [ ] **Step 4: T2 with FFS node UP**

  ```bash
  ros2 launch foundation_stereo foundation_stereo.launch.py &
  # repeat the detection service call
  ```

  Expect:
  - Service returns the same shape as before
  - Sidecar JSON now shows `"depth_source": "ffs"`
  - On a flat target at ~1 m: depth values agree with native to within ~5%
  - No warn lines on stderr

- [ ] **Step 5: T3 mid-session FFS kill**

  With both running, `kill` the FFS node mid-session. The next detection call should:
  - Succeed (returns native)
  - Log the throttled warn exactly once
  - Sidecar JSON shows `"depth_source": "native"`

- [ ] **Step 6: Param toggle**

  With FFS up, `ros2 param set /object_detection_yolo prefer_ffs false`. Next call's sidecar should show `"depth_source": "native"` immediately (no restart). Flip back to `true`, next call should be `"ffs"` again.

---

## Task 5: Documentation + commit

- [ ] **Step 1: Update package README + changelog**

  Per the workspace's README+changelog discipline (memory: `feedback_readme_changelog`), append a Changelog entry to:
  - `src/object_detection_new/README.md` (or create the Changelog section if missing)
  - `src/object_detection_generalist/README.md` (same)

  Entry shape: date, the new `prefer_ffs` param + the 5 siblings, default `true`, fallback semantics, observability (vision_log sidecar `depth_source` field).

- [ ] **Step 2: Update `src/tk26_vision/CLAUDE.md` § Configuration**

  Add the new params to the `object_detection_new` line in the Configuration section. Mirror style of existing entries.

- [ ] **Step 3: Verify before commit**

  Run T0 + T1-FFS-down + T2-FFS-up + T3-mid-session-kill + param-toggle one more time. All must pass on a clean rebuild.

- [ ] **Step 4: Commit (single commit for the whole phase, per memory `feedback_phase_per_commit`)**

  Commit message: `feat(vision): FFS-first depth with native fallback in detection nodes`

  Body should reference:
  - Why (GPU cost of streaming FFS for service-driven detectors)
  - Surface impact (none — no srv changes)
  - Rollback (`ros2 param set ... prefer_ffs false`)

---

## Risk + rollback

- **Smallest reversion:** `ros2 param set /<node> prefer_ffs false` — restores prior behavior instantly, no rebuild. Same flag in yaml for permanent disable.
- **Bigger reversion:** revert the commit; `_acquire_depth` is the single seam, so all native-path code stays intact behind the `False` branch.
- **Concurrency hazard:** `call_async` from inside a service callback that already runs on `MutuallyExclusiveCallbackGroup` (yolo:~306, generalist:~258) will not progress the FFS response on the same group. Task 1 Step 3 picks the right resolution (ReentrantCallbackGroup OR side-thread executor — match the existing codebase pattern). Don't invent a third pattern.
- **Latency budget:** FFS fast_trt ~600 ms steady-state; the workspace 10 s/call vision budget (memory: `feedback_compute_budget`) absorbs it. `wait_for_service` is 200 ms so cold-miss → fallback is bounded.
- **Frame-id parity:** FFS aligned-to-color stamps with the color CameraInfo's frame_id; native realsense path uses the depth_msg frame_id. Both should be `xarm_camera_color_optical_frame` on tinker2; **verify on tinker1** before relying on TF — flag in `DEV_NOTES.md` if they diverge.

---

## Optional follow-ups (separate PR — not in this plan)

- `/depth_source_used` `std_msgs/String` topic (latched, last-source-per-call) — cheap observability for BT/recorder.
- Per-call override: extend `ObjectDetection` / `ObjectDetectionGeneralist` srv with `int8 depth_source` (0=auto, 1=ffs, 2=native). Holding off because it touches canonical srv schemas in `tinker_vision_msgs_26` and this plan deliberately keeps the surface unchanged.
- Cache the FFS depth for ~50 ms keyed on stereo-pair stamp so back-to-back generalist calls (e.g. race_world_vlm) don't double-pay inference.

---

## File:line index (from the design-pass grep)

- FFS service: `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py:590` (definition), `:592-619` (callback), `:30-31` (msg import)
- Srv schema: `src/tinker_vision_msgs_26/srv/FoundationStereoDepth.srv:1-22`
- Yolo realsense depth consumption: `src/object_detection_new/object_detection_new/object_seg_yolo.py:407-431`, callback dispatch `:1079-1112`
- Generalist realsense depth consumption: `src/object_detection_generalist/object_detection_generalist/generalist_node.py:290-298` (calls inherited `_process_realsense_data`)
- Param-decl sites: yolo `:93-145`, generalist `:164-205`
- Detection yaml: `src/object_detection_new/config/default.yaml` (generalist has no yaml yet — add one)
- FFS launch: `src/foundation_stereo/launch/foundation_stereo.launch.py`
