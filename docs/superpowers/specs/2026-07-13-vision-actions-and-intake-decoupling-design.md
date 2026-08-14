# Vision bringup: service→action conversion + intake/TF decoupling — design

- Date: 2026-07-13 (rev 2, same day — after 4-lens adversarial review; see §9)
- Scope: the perception-layer nodes launched by
  `src/vision_bringup/launch/vision_bringup.launch.py`
- Related: `src/vision_bringup/docs/vision-bringup-design.md` (node selection),
  `docs/superpowers/specs/2026-07-04-waving-vlm-only-live-person-design.md` and
  `docs/superpowers/specs/2026-07-04-waving-window-raise-on-update-design.md`
  (both in-flight waving workstreams this must sequence around)

## 1. Goal

Two coupled refactors over the BT-facing vision nodes:

1. **Service→action conversion — only where an action genuinely fits.** The
   long-running VLM-backed services (10–60 s blocking inside a service
   callback) become ROS2 actions with staged feedback and cooperative
   cancellation. Quick synchronous services (frame relays, ms-scale CV/YOLO
   queries, config setters) stay services — converting them adds churn and
   round-trip overhead with no benefit. (Per operator directive 2026-07-13:
   "the pure-service ones do not need to be converted".)
2. **Decouple image/depth intake and TF lookup.** Today every node re-implements
   camera subscription + sync + caching + staleness and its own TF
   buffer/poll/transform block (and there are 5 private depth→3D
   reprojection variants — NOT all the same math, see §4.4). Factor the
   genuinely-common parts into shared, composable helpers in `vision_util`
   so a node's camera source, sync policy, depth source, and TF policy can
   be changed in one place without touching detection/VLM logic.

Out of scope: the driver layer (`vision_driver.launch.py` — pan-tilt controller
services, FFS `get_depth`, vendored camera drivers), nodes not in the bringup
(`spot_on_shelf`, `object_match_all`, `placing_location`, `get_point_cloud`,
`get_orbbec_pc`, `monocular_depth`), and the known-broken BT semantics of
`BtNode_TrackPerson` (tk25_decision follow-up #4).

## 2. Service classification

Audit of every server on the 11 bringup nodes:

### 2.1 Convert to action — 6 services on 5 nodes

| Service | Node | Why an action fits |
|---|---|---|
| `detect_waving_persons` (`DetectWaving.srv`) | `waving_person_server` | Callback blocks up to ~27 s: freshness poll ≤2 s + TF snapshot poll ≤5 s + VLM `future.result(timeout=20)` (`waving_person_server.py:698-730,767-770,944`). Natural stages for feedback; VLM future is already abandonable. |
| `feature_extraction_service` (`FeatureExtraction.srv`) | `feature_recognition` | Awaits generalist detection then blocks on Gemini→Qwen chain (`feature_recognition.py:300-302,425-433`); chain worst case ≈2 providers × 3 retries × 20 s (more under SDK connect-retries, see §3.2). |
| `seat_recommend_service` (`SeatRecommendation.srv`) | `feature_recognition` | Same node, same blocking VLM chain (`feature_recognition.py:513-520`). |
| `feature_matching_service` (`FeatureMatching.srv`) | `feature_matching` | Detection await + blocking VLM chain (`feature_matching.py:295-297,410-418`); its own TF buffer is sized 180 s specifically because the handler can run ~125 s worst case (`feature_matching.py:137-149`). |
| `seat_recommend_bbox_service` (`SeatRecommendBbox.srv`) | `seat_recommend_bbox` | VLM calls measured 10–25 s, `vlm_timeout_s=25` (`seat_recommend_bbox.py:51-59,248`); blocking sync handler. |
| `object_scan` (`ObjectScan.srv`) | `object_scan` | Fans out parallel VLM batch calls and blocks until the slowest returns (`object_scan.py:186-203`); batch progress is an obvious feedback stream and per-batch abort is cheap. |

Shared traits: the caller (behavior tree) currently cannot cancel these — a
preempted BT node abandons the future but the server keeps burning its
callback slot for up to minutes, and queued calls stall behind the abandoned
one (all six are serialized: five on explicit MutuallyExclusive groups, the
seat_recommend server on the node **default** group — which it shares with
that node's camera subscriptions, a pre-existing hazard §3.2 fixes). Actions
add cancellation and staged progress **while keeping today's queue-and-wait
semantics for concurrent callers** (§3.2 goal policy).

### 2.2 Stay a service — with rationale

| Service | Node | Why it stays |
|---|---|---|
| `/object_detection_generalist` | `generalist_node` | Default path is YOLO/YOLO-World (ms–hundreds of ms). The slow VLM path is opt-in (`use_vlm_sam_fallback`/`force_vlm_sam`) and already has internal race + abandon cancellation (`generalist_node.py:693-827`). Decisive: it has three **in-process synchronous service clients** — `feature_recognition`, `feature_matching`, `grocery_categorize` all `await` it mid-handler — plus 6+ BT call sites. Converting it forces an action client into every one of those or a dual srv+action facade; blast radius outweighs benefit. Revisit only if BT preemption of the VLM fallback becomes a need. |
| `/object_detection_yolo` | `yolo_seg_node` | Pure YOLO snapshot query, ms-scale; the FFS sub-call is bounded (8 s worst, typically fast) and internal. (`yolo_seg_default_node` / `/object_detection` is not in the bringup, but shares the same base class, so intake adoption covers it for free.) |
| `door_detection_srv` | `door_detection` | ~20-line depth heuristic, <1 s. |
| `get_image_service` | `get_image` | Cache relay, <<1 s. |
| `~/reseed_target` | `person_track_server` | Single-frame re-lock, quick, tightly coupled to the tracking loop's locks. |
| `follow_head_service` | `follow_head` | One bounded tick by design; the continuous mode already exists as `follow_head_action`. |

Already actions (no change): `track_person`, `follow_head_action`,
`grocery_categorize`, plus non-bringup `spot_on_shelf`, `monocular_depth_pc`.

## 3. Action interface design

### 3.1 New `.action` definitions in `tinker_vision_msgs_26`

Six new files, named after their srv counterparts (srv/ and action/ are
separate type namespaces; `FoundationStereoDepth` already coexists in both —
`CMakeLists.txt:31,50`):

`action/DetectWaving.action`, `action/FeatureExtraction.action`,
`action/SeatRecommendation.action`, `action/FeatureMatching.action`,
`action/SeatRecommendBbox.action`, `action/ObjectScan.action`.

Shape rule: **Goal = the old request fields verbatim; Result = the old
response fields verbatim; Feedback = the standard block below.** No field
renames or semantic changes in this wave — the conversion must be mechanically
verifiable against the srv definitions.

Standard feedback block. **Constraint discovered in review:** the BT's
default `ActionHandler.feedback_callback` (`ActionBase.py:602-616`)
unconditionally reads `feedback.delay_limit`, `feedback.status`, and
`feedback.stage` — a feedback message without those fields raises
`AttributeError` inside a subscription callback on the first feedback. The
block therefore **conforms to the canonical BT feedback protocol** (shape
documented at `HRI/follow.py:160`; exact field types to be pinned during
implementation against the canonical actions the default callback was
written for):

```
# feedback — canonical BT protocol fields first (read by the default
# ActionHandler.feedback_callback; delay_limit drives its goal watchdog)
int32 status          # 0 = normal; nonzero = abnormal (watchdog trip)
float32 delay_limit   # server's promise: max seconds until next feedback/result
string stage          # machine token: acquiring_frame | detecting | vlm_call |
                      #   vlm_retry | vlm_fallback | judging | transforming |
                      #   input_frozen
string message        # human-readable detail (provider/model, batch i/N, ...)
bool input_frozen     # false until all camera/TF inputs are goal-owned;
                      # true on input_frozen and every later stage
```

`delay_limit` is the one field with a live client-side consumer (the
`goal_timeout` watchdog, `ActionBase.py:538-542`); servers must set it
honestly per stage (e.g. `vlm_timeout_s` + margin when entering `vlm_call`).
`input_frozen` is the typed motion-release signal. Servers publish exactly one
`input_frozen` stage after acquiring every image/depth/intrinsics/point-cloud
and capture-stamped transform required by the remaining work, then keep the
boolean true on all later feedback. `stage`/`message` remain useful for
`ros2 action` CLI debugging.

The old `.srv` files stay in `tinker_vision_msgs_26` untouched for one
release (interface package is shared; deleting them is deferred cleanup —
note `vision_track/test/test_active_reid_interfaces.py:60` asserts srv
fields and breaks at that deferred deletion, not now).

### 3.2 Server pattern

Corrected after review — two widely-copied in-repo beliefs about rclpy are
wrong, and the pattern below does not rely on them:

- **Fact 1 (verified empirically on this box, rclpy 3.3.21):** an action's
  execute callback runs as a **group-less executor task**
  (`rclpy/action/server.py:539-547`) — it never holds the ActionServer's
  callback-group mutex. Cancel requests are serviced regardless of whether
  the server sits on a MutuallyExclusive or Reentrant group, **provided a
  MultiThreadedExecutor with free threads exists**. follow_head's
  "Reentrant-or-deadlock" comment (`follow_head.py:288-297`) is a
  misdiagnosis; its thread-pool-exhaustion comment (`:257-265`) names the
  real constraint.
- **Fact 2 (verified empirically):** rclpy's default goal/handle-accepted
  callbacks accept **and execute** every goal immediately — two goals run
  two concurrent execute callbacks, on any group type. (follow_head's
  "rclpy enforces single-active-goal by default" comment at `:298-300` is
  false — pre-existing latent bug there, out of scope.) Serialization is
  therefore **mandatory structure, not policy hardening**.

The pattern:

- **Goal policy: accept-and-queue (service-parity).** All goals are
  ACCEPTED; a per-node FIFO + worker serializes execution, reproducing
  today's semantics where concurrent service calls queue behind the MutEx
  group and everybody eventually gets an answer. **Rejection is explicitly
  ruled out**: the BT `ActionHandler` maps a rejected goal to immediate
  FAILURE with no backoff (`ActionBase.py:532-535`), which review showed
  converts today's bounded ≤27 s stalls into task-level failures — GPSR's
  `Retry(scan, 5)` (`egpsr.py:97`) exhausts in ~2.5 s against a 20 s VLM
  call, and Restaurant's 14-position scan sequence
  (`wire_with_navigation.py:300-333`) hard-fails on one zombie goal. A
  cancelled-while-queued goal returns CANCELED without running user/VLM work.
  Its rclpy execute callback still runs once at its FIFO turn so the action
  server resolves the client's result future; skipping `goal_handle.execute()`
  leaves that future pending indefinitely on ROS 2 Humble.
  **The queue is per-node, not per-server**: `feature_recognition`'s two
  action servers share one worker (they contend for one VLM quota, one
  detection client, one camera cache — parallel execution buys nothing and
  risks plenty).
- **`result_timeout=0` on every ActionServer.** rclpy's default retains
  every terminal goal's result for **900 s**
  (`rclpy/action/server.py:200`); our results carry images
  (`FeatureExtraction.comparison_image` is actively filled,
  `feature_recognition.py:410-412`; `DetectWaving` carries
  rgb/depth/segments fields). 15 minutes of per-goal image retention on a
  box with a documented shared-memory/swap incident history is not
  acceptable.
- **Callback groups / executor:** ActionServer on its own
  MutuallyExclusive group (cheap, serializes only the short internal
  goal/cancel/result servicing — per Fact 1 the group does NOT serialize
  executes; the queue does). **`MultiThreadedExecutor` everywhere** — this
  is the load-bearing requirement for cancel servicing and intake liveness.
  `feature_recognition` and `feature_matching` currently run plain
  `rclpy.spin()` and must upgrade; `feature_recognition`'s seat server and
  camera subscriptions currently share the node **default** group
  (`feature_recognition.py:208-240`) and must each get explicit groups so a
  running handler cannot starve intake.
- **`async def execute_callback`** where the node awaits the generalist
  detection client (feature_extraction, feature_matching), keeping the
  existing two-callback-group client split. Plain sync execute elsewhere.
- **Cancellation is cooperative, with an honest latency bound.**
  `CancelResponse.ACCEPT`; the execute path checks
  `goal_handle.is_cancel_requested` at every stage boundary; the kimi_api
  provider-chain helpers gain an optional `should_abort:
  Callable[[], bool]` checked between retries and providers (all chain
  loops are in-repo — `_scan_vlm.py:132-163`, `_feature_vlm.py:80,133`,
  `_match_vlm.py:141,215`, `_seat_bbox_vlm.py:383,447`,
  `_waving_vlm.py:259,316` — precedent: `vlm_bbox.py`'s `abandon_event`).
  **`max_retries=0` on the OpenAI client constructions in those helpers** —
  the SDK default of 2 internal HTTP retries makes one "attempt" up to ~3×
  `vlm_timeout_s` (verified against openai 2.32.0), tripling the promised
  cancel latency; the chains already own retry policy, the SDK must not
  duplicate it. Waving: `should_abort` is threaded into the chain running
  on its VLM ThreadPoolExecutor (not just future-abandon — abandoned chains
  keep burning the 2-worker pool and provider quota), and the pool size is
  revisited for cancel-heavy action use. With these, worst-case cancel
  latency = the remainder of one in-flight HTTP call, ≤ one
  `vlm_timeout_s`.
- **Terminal semantics:** `succeed()` for any legitimate answer **including
  empty/none** (chain convention preserved); `abort()` only for internal
  errors; cancel → `goal_handle.canceled()` returning a result whose
  `status`/`error_msg` follow **that node's existing convention** (waving
  uses `-1` for errors, kimi nodes use `1` — no cross-node cancel code is
  imposed; clients must branch on `GoalStatus.STATUS_CANCELED`, which is
  what `ActionHandler.process_result` sees, not on result payload). Degrade
  modes (waving VLM→mediapipe auto-degrade, fallback-provider drop on
  missing key) are untouched.
- Server names keep their current strings (`feature_extraction_service`
  etc.) even where the `_service` suffix reads oddly — zero churn in launch
  scripts, params, and test greps.

### 3.3 Client migration

#### BT nodes (`ServiceHandler` → `ActionHandler`)

| Caller | File |
|---|---|
| `BtNode_ScanForWavingPerson` | `TemplateNodes/Vision.py:1713` |
| `BtNode_ScanForWavingPersonNew` | `GPSR/custom_nodes.py:880` (the *other* `BtNode_ScanForWavingPerson` at `custom_nodes.py:796` calls the generalist — correctly untouched) |
| `BtNode_DetectCallingCustomer` | `Restaurant/custumNodes.py:14` |
| `BtNode_FeatureExtraction` | `TemplateNodes/Vision.py:699` |
| `BtNode_SeatRecommend` | `TemplateNodes/Vision.py:868` |
| `BtNode_SeatRecommendBbox` | `TemplateNodes/Vision.py:954` |
| `BtNode_FeatureMatching` | `TemplateNodes/Vision.py:1059` |
| `BtNode_ObjectScan` | `TemplateNodes/Vision.py:345` |

Migration rules discovered in review (violating any of these ships a bug):

- **Do NOT wire the legacy dead `timeout` ctor params** (e.g.
  `BtNode_DetectCallingCustomer`'s ignored `timeout: float = 10.0/5.0`,
  `Restaurant/custumNodes.py:28`) into `action_timeout_ticks` — 5–10 s
  budgets against 10–27 s VLM calls would trip the watchdog on every call.
- **`ActionBase.py` base-class fixes land in the same wave** (they are
  currently latent because few action nodes exist):
  1. the timeout/FAILURE paths never cancel the goal — the literal
     `# TODO: abort the action here` at `ActionBase.py:524`; without this,
     every BT-side timeout leaves a zombie goal;
  2. cancel on `RUNNING→FAILURE` terminate, not only `RUNNING→INVALID`;
  3. mock-type parity: `ServiceHandler.setup` force-mocks when the type
     comes from `mock_messages` (`BaseBehaviors.py:202-213`);
     `ActionHandler.setup` lacks the equivalent and would hand a mock class
     to `rclpy.action.ActionClient` — add the same guard.
- **`messages.py` name collision:** the six srv types are imported flat
  (`messages.py:12-13`); the same-named action types must be aliased on
  import (e.g. `DetectWavingAction`) since the srv imports stay while the
  `.srv` files live.
- **`mock_messages.py`:** the six exist only as `MockService`
  (`:119,133,141,146,174,204`); six `MockAction` (Goal/Result/Feedback)
  classes must be added to the roster at `:664-706`.

#### Non-BT callers (found in review; all in-repo)

| Caller | File | Change |
|---|---|---|
| `_WaveReseedBridge` raw srv client | `FollowPerson/wave_reseed_cycle.py:33` + owner wiring `FollowPerson/nodes.py:44-46,229,248,311` | Action client behind the same future-shaped facade, or facade-preserving wrapper; its contract test `test/test_wave_reseed_cycle.py` pins the facade — keep it green or change both together |
| track_web dashboard 👋 button | `vision_track/vision_track/track_web.py:35,67,126,267-268` (+ `webui/app.js:221`) | Action client |
| waving_bench harness | `tk_vision_specialized/tk_vision_specialized/waving_bench.py:28,39,103,132,197` | Action client (judging half `_waving_bench_eval.py` unaffected) |
| waving_client example | `tk_vision_specialized/tk_vision_specialized/waving_client.py` | Action client |
| restaurant_nav_test_web readiness probe | `restaurant_nav_test_web/restaurant_nav_test_web.py:112` (`in services` membership) | Probe `/detect_waving_persons/_action/send_goal` (pattern already at `:113`) |
| BT endpoint checker | `behavior_tree/scripts/verify_task_endpoints.py:44-45` | Move the two pinned entries from `services` to `actions` |

Docs/examples that go stale (update greps, low risk):
`tk_vision_specialized/README.md:47`, `tk26_vision/CLAUDE.md`,
`kimi_api/README.md:9-12`, `vision_bringup/docs/vision-bringup-design.md:123-133`,
`Restaurant/README.md`, `behavior_tree/README.md:268`,
`tk26_sim/PORT_NOTES.md:33-38,60`, `tk25_decision/CLAUDE.md:161`.

## 4. Intake / TF decoupling

### 4.1 The duplication being removed

- **Subscription+sync+cache**: ≥10 private implementations. Subscription
  setup at: `object_seg_yolo.py:244-311` (dual-camera ATS; its staleness
  retry loop is in the service callback at `:1186-1211`),
  `generalist_node.py:112-132` (extra unsynced depth sub, QoS depth 1),
  `get_image.py:56-122`, `door_detection.py:30-66` (unsynced),
  `waving_person_server.py:83-91` (+ freshness barrier `:698-730`),
  `seat_recommend_bbox.py:227-236`, `object_scan.py:92-136` (staleness
  check at `:147-155`), `feature_recognition.py:209-220`,
  `person_track_node.py:697-730` (seq-consume at `:815-827,1271`, stall
  watchdog at `:162-167,390-399`), `follow_head.py:252-286` (header dedup
  at `:478-503`, including consume-without-processing in the 5 Hz-cap
  branch at `:495`).
- **Depth decode/reprojection**: 5 variants (see §4.4 — they are NOT
  equivalent): canonical `vision_util/depth_reproject.py`, local variants
  in `waving_person_server.py:37-59`, `person_track_node.py:829-872`,
  `follow_head.py:1123-1160`, `object_seg_yolo.py:536-550` (RealSense,
  deliberately non-optical axes), `seat_recommend_bbox.py:606-613`
  (mm-decode only).
- **TF**: four buffer configurations (10 s / 60 s / 180 s caches), three
  lookup idioms (single-try `object_seg_yolo.py:1035-1068`,
  poll-until-deadline `waving_person_server.py:384-424`,
  stamped-with-fallback `feature_matching.py:194-227`).

### 4.2 New shared modules (in `vision_util`)

`vision_util` is already the de-facto shared library and every affected
package already declares the dependency (verified in each `package.xml`);
its module-level imports are numpy/cv2/cv_bridge only, so it is importable
in every runtime interpreter in play, including follow_head's
`/usr/bin/python3`. **One new dependency edge:** `tf_lookup` needs
`tf2_ros` + `tf2_geometry_msgs` added to `vision_util/package.xml`
(non-circular).

**`vision_util/camera_intake.py`**

```python
@dataclass
class StreamSpec:
    topic: str
    best_effort: bool = True      # per-stream: camera_info is RELIABLE today
    qos_depth: int = 5

@dataclass
class IntakeConfig:
    camera: str                    # label: 'orbbec' | 'realsense'
    color: StreamSpec | None       # any stream may be omitted
    depth: StreamSpec | None       # aligned/raw depth Image
    camera_info: StreamSpec | None
    sync_queue: int = 10           # ATS used iff both color+depth configured
    sync_slop_s: float = 0.1
    age_source: str = 'recv'       # 'recv' (node clock at delivery) |
                                   # 'stamp' (header.stamp) — see below

class FrameBundle:
    # camera, seq (monotonic), header, recv_time (node clock), K
    # color_msg / depth_msg — configured subset; msgs immutable after store
    def color_bgr(self): ...   # lazy+memoized CvBridge; Orbbec rgb8/bgr8
                               # normalize; READ-ONLY array (writers copy)
    def depth_m(self): ...     # decode via depth_reproject; READ-ONLY
    def points_xyz(self, roi=None, valid_band=(1e-6, 10.0)): ...

class CameraIntake:
    def __init__(self, node, cfg, callback_group=None): ...
    def latest(self, max_age_s=None) -> FrameBundle | None: ...
    def wait_fresh(self, max_age_s, timeout_s, poll_s=0.05,
                   on_timeout='fail') -> FrameBundle | None: ...
        # on_timeout='fail'   → None (yolo semantics, caller returns status=1)
        # on_timeout='stale'  → newest available + throttled warn
        #                       (waving's proceed-anyway barrier)
    def latest_new(self, last_seq): ...
        # tri-state: FrameBundle | NO_NEW_FRAME | None(no data yet) —
        # person_track's watchdog distinguishes "nothing new" from "nothing"
    def intrinsics(self): ...
    @staticmethod
    def declare_params(node, camera, defaults) -> IntakeConfig: ...
```

Contracts pinned after review:

- **Staleness has two axes and both are config, not accident.**
  `age_source='recv'` reproduces yolo/object_scan (receive-clock age;
  note: object_scan moves from `time.time()` wall clock to node clock —
  an intentional, flagged behavior change that makes it sim-time/rosbag
  correct); `age_source='stamp'` reproduces waving (header-stamp age,
  camera-clock skew visible). `on_timeout` selects fail-vs-proceed-anyway.
- **Decode failures drop at intake**: the sync callback decodes nothing,
  but a bundle whose lazy decode fails is discarded and the previous good
  bundle remains served (waving's last-good-frame semantics; a poison frame
  must not brick one-shot `latest()` callers).
- **Read-only outputs, copy-on-write callers** — the contract
  `person_track_node.py:1295-1298` already documents. This is what lets the
  current deepcopies (`object_seg_yolo.py:1200`, `get_image.py:137-138`,
  `door_detection.py:90-91`) be dropped safely.
- One lock per intake; lazy decodes memoized under it. Per-stream QoS via
  `StreamSpec` — uniform defaults would silently flip waving /
  seat_recommend_bbox / feature_recognition subscriptions from their
  current default-RELIABLE to BEST_EFFORT; any intentional QoS change ships
  in its own commit with a table, not smuggled inside "refactor".
- `wait_fresh` polls with `time.sleep` and therefore **requires a
  MultiThreadedExecutor** (or an executor thread the intake callbacks don't
  share). Adoption in the two plain-`spin()` kimi nodes is sequenced after
  their executor upgrade (§7).
- One `CameraIntake` per camera per node (generalist's extra raw depth sub
  = a second depth-only instance); per-request camera selection stays
  caller-side; person_track keeps image subs on its Reentrant group via
  `callback_group` while camera_info may share (harmless — documented).

**`vision_util/tf_lookup.py`**

```python
class TransformHelper:
    def __init__(self, node, cache_time_s=180.0): ...   # VLM-safe default
    def try_lookup(self, target, source, stamp=None, timeout_s=0.1): ...
    def wait_lookup(self, target, source, deadline_s, latest=True,
                    poll_s=0.02): ...
    def transform_point(self, pt, transform_or_target): ...
    buffer: tf2_ros.Buffer  # escape hatch
```

Failures return `None`; warning emission is caller-configurable (waving
logs hard ERROR + specific error_msg, yolo warns-and-aborts-batch —
policy stays with callers). `wait_lookup(latest=True, deadline_s=5.0)`
reproduces waving's `_snapshot_latest_transform` exactly;
`try_lookup(stamp=..., timeout_s=0.1)` reproduces feature_matching's
stamped-with-degrade. The RealSense frame-convention gate
(`_frame_supports_tf_transform`, `object_seg_yolo.py:1030-1033`) is caller
policy and stays in the detection node. Buffer-widening (10 s/60 s → 180 s)
is strict and its memory cost at realistic /tf rates is a few MB/node.

**`vision_util/depth_source.py`**

The FFS-vs-native selection currently buried in
`object_seg_yolo.py:416-534` (`prefer_ffs`, lazy client on a Reentrant
group, Event-blocking call, throttled fallback warning, source tag) moves
here as `FfsPreferredDepthSource(node)`. Signature honesty (from review):
the FFS request carries **no image** — the server captures its own stereo
pair, so the returned depth is *temporally decoupled* from any frame the
caller holds. The API is therefore `acquire(align_to_color: bool) ->
(depth_m, source_tag)` — it does NOT take a FrameBundle, and on the
realsense path the caller **bypasses** `FrameBundle.depth_m()/points_xyz()`
entirely (FFS depth may not even match the bundle's color resolution).
Native-fallback dtype stays float64 (bug-compatible; the change to float32
is a flagged follow-up, not part of this refactor).

### 4.3 Per-node adoption map

| Node | Intake | TF |
|---|---|---|
| `yolo_seg_node` / default / `generalist_node` (shared base) | 2× `CameraIntake` (color+depth ATS, `age_source='recv'`, `on_timeout='fail'`); generalist adds a depth-only instance; `_acquire_depth` → `FfsPreferredDepthSource` | `TransformHelper` (60→180 s widening) |
| `door_detection` | depth+info instance (no color, no ATS) | — |
| `get_image` | 2× color+depth instances | — |
| `waving_person_server` | color+depth instance; `wait_fresh(age_source='stamp', on_timeout='stale')` replaces freshness barrier; local reproject deleted (→ canonical, §4.4) | `wait_lookup` replaces `_snapshot_latest_transform` |
| `feature_recognition` | color-only instance for the seat path, explicit group (extraction path's "intake" is the generalist client — unchanged) | — |
| `feature_matching` | none (detection-client based) | `TransformHelper` replaces its private block |
| `seat_recommend_bbox` | color+depth instance | `TransformHelper` (keeps pre-VLM snapshot policy) |
| `object_scan` | 2× color-only instances; `latest(max_age_s=1.0)` (recv-clock — flagged sim-time behavior change) | — |
| `person_track_server` | instance with `latest_new` tri-state, QoS 5, Reentrant group; watchdog/EMA/FSM untouched | unchanged |
| `follow_head` | instance; header dedup → seq dedup (must keep the consume-without-processing behavior of the 5 Hz-cap branch) | none by design (analytic servo frame) — stays |

### 4.4 Depth decode/reprojection consolidation — the honest version

Review refuted "all five copies delete." The five variants differ in
load-bearing ways:

- **The yolo RealSense variant is intentionally non-optical.** It pairs
  rows↔(cx,fx) / cols↔(cy,fy) (`object_seg_yolo.py:537-541`) and its
  centroids feed a matching hand-rolled body-axis convention that the
  **grasp service consumes**, with TF deliberately skipped
  (`object_seg_yolo.py:1026-1033`). It is preserved verbatim as a named,
  documented function (`realsense_body_axes_points`, marked
  bug-compatible-do-not-"fix") — not merged into the canonical math.
- **Valid-band/clip semantics differ per node** (waving `(1e-6,10)+clip`,
  follow_head `(1e-3,10)`, person_track `(0.1,10)`, yolo-orbbec
  `min_depth=-10.0` default — a latent bug that admits z=0 invalid pixels;
  **preserved as-is** in this refactor, logged as a follow-up). The
  canonical function takes an explicit `valid_band`/`clip` parameter; each
  node's adoption passes its current values.
- **Golden tests are per-variant**: each adoption pins the *new call* against
  the *old local function's* output on recorded frames — equivalence to the
  node's own previous behavior, never cross-node equivalence.

## 5. Approaches considered

**Conversion strategy** — (a) clean cutover per node: srv server removed,
action server added, all in-repo callers migrated in the same wave
*(chosen — all callers are in-repo, and dual-serving means maintaining two
code paths through VLM handlers)*; (b) transitional dual srv+action per
node — only worth it if out-of-repo callers surface; (c) generic
"long-running service" wrapper action — rejected, loses typed goals/results.

**Goal policy** — (a) accept-and-queue *(chosen — service-parity, see
§3.2)*; (b) single-goal reject — rejected after review (task-level failures
in GPSR/Restaurant trees, selector thrash in
`restaurant_simplified._createDetectAndAddOne`'s `memory=False` selector);
(c) accept-and-preempt (Nav2 style) — rejected for one-shot query actions:
preempting a goal another subtree is awaiting is queue-jumping with extra
steps.

**Decoupling placement** — (a) modules in `vision_util` *(chosen)*; (b) new
`vision_intake` package — cleaner nominally, but adds a package + rebuild
churn for no isolation gain; (c) a central camera-relay node that owns all
subscriptions (kills the triple-subscription follow-up #3 for real) —
out of scope here, and `CameraIntake` is the stepping stone to it.
**Option (c) is now a concurrent, separate effort**:
`docs/specs/2026-07-13-camera-server-design.md` (per-camera C++
snapshot/PC/TF servers, committed 2026-07-13 by a concurrent session). Its
§13 records the composition contract: once nodes here adopt
`CameraIntake`/`TransformHelper`, migrating them to the camera servers is a
backend swap *inside* the two helpers. Alignment obligations on this spec:
any semantics `CameraIntake` grows during Waves 1–3 must be reflectable in
`GetCameraSnapshot` — as of rev 2 that means `on_timeout` (both modes map
onto its `WAIT_TIMEOUT`-returns-newest) and `age_source`, where
`'stamp'` maps to the server's pair-stamp freshness but **`'recv'`
(local receive-clock) has no remote equivalent** — nodes needing exact
recv-clock semantics keep the subscription backend after the swap.
Continuous consumers (person_track, follow_head) keep subscriptions
regardless (both specs agree).

## 6. Testing

- **Unit**: pytest for `camera_intake` (sync pairing, both `age_source`
  modes, both `on_timeout` modes, tri-state `latest_new`, decode-failure
  drop, rgb8/bgr8, read-only outputs), `tf_lookup` (static broadcaster
  fixtures), `depth_source` (mock FFS client: prefer/fallback/timeout).
  Per-variant golden tests for depth reprojection (§4.4).
- **Action semantics** (new, per converted node): queue behavior (2nd goal
  while busy → queued, both complete, order preserved), cancel-latency
  bound (cancel during `vlm_call` stage → CANCELED within one
  `vlm_timeout_s`), cancel-while-queued (→ CANCELED, never executes),
  feedback conformance (fields present, `delay_limit` honest).
- **Integration suite**: T0 interface list gains the six actions (its srv
  array today covers only 3 of the 6 nodes' types); T1 has existing checks
  for only feature_extraction/seat_recommend (T1.7) and feature_matching
  (T1.8) — those switch to the existing `wait_for_action` helper
  (`lib.sh:113-125`, already used for the five current actions), and **new
  T1/T2 checks are added** for detect_waving, object_scan, and
  seat_recommend_bbox, which have none today. T2/T3 callers switch to
  action clients. Empty-scene invariants restated for results.
- **BT**: mock-mode tests for each migrated node (after the `MockAction`
  additions); one preemption test per converted node (terminate mid-goal →
  server observes cancel); `verify_task_endpoints.py` updated and run.
- **Behavioral parity gate**: for each converted node, one recorded
  live-scene run (T2 tier) comparing action result fields against the
  pre-conversion service response on the same scene class.
- **Discovery sanity**: each action adds ~5 DDS endpoints per side vs a
  service's 2; after Wave 2, `ros2 topic list | wc -l` + bringup timing
  compared against pre-conversion baseline on the robot (this workspace is
  demonstrably discovery-sensitive — interfaceWhiteList, SHM tuning).

## 7. Phasing

1. **Wave 0 — helpers**: `camera_intake` / `tf_lookup` / `depth_source` /
   `depth_reproject` extensions + unit tests. No node behavior change.
2. **Wave 1 — intake adoption, stay-service nodes with MTE already**:
   detection base class (yolo/generalist), door_detection, get_image. T2
   regression after each node. (object_scan's intake moved to its Wave 2
   sub-wave — touching it twice across waves buys nothing.)
3. **Wave 2 — action conversions**, one sub-wave per node, each =
   interfaces + server + BT/base-class work + non-BT callers + tests.
   Order (unblocked first, contended file last):
   `feature_matching` → `feature_recognition` (incl. executor upgrade +
   group fixes) → `seat_recommend_bbox` → `object_scan` (+its intake) →
   **`waving_person_server` last**, gated on BOTH in-flight waving
   workstreams landing (VLM-only 2026-07-04 spec + window-raise spec; the
   file has uncommitted changes on `tinker2-net` right now). The
   `ActionBase.py` fixes (§3.3) land with the first sub-wave.
4. **Wave 3 — tuned trackers**: person_track + follow_head intake adoption,
   last, behind extra care (arena-tuned QoS/dedup behavior).

Each wave is independently shippable; an arena freeze can stop after any
wave (and after any Wave-2 sub-wave — each node cuts over atomically).

## 8. Risks

- **Waving file contention**: two live workstreams touch
  `waving_person_server.py`; its sub-wave is last and explicitly gated.
- **BT preemption now has teeth**: `terminate()` cancels server work; trees
  relying on fire-and-forget side effects (vision-log artifacts of
  abandoned calls) will see fewer artifacts. Audit during Wave 2, together
  with a sweep for `memory=False` selectors/parallels over converted nodes
  (the review's restaurant thrash scenario is neutralized by
  accept-and-queue, but the audit is cheap insurance).
- **Executor upgrade side effects** (`feature_recognition`): two servers on
  one node now genuinely interleave with intake; explicit groups (§3.2)
  plus the per-node queue bound the concurrency. `VisionLogger`'s lazy
  `_ensure_run_dir` is unlocked (`vision_logging.py:176-181`) — add a lock
  when the executor upgrade lands.
- **DDS endpoint growth** (~+6 endpoints × 6 conversions × both sides):
  measured, not assumed — §6 discovery sanity check.
- **Sim-time semantics change in object_scan staleness** (wall→node clock):
  intentional, flagged; affects rosbag replay expectations.
- **`_service`-suffixed action names** are permanent cosmetic debt, chosen
  deliberately to avoid launch/test churn.

## 9. Adversarial review record (2026-07-13)

Four independent review lenses (technical claims, intake API, migration
blast radius, operational semantics) ran against rev 1. Material findings
folded into rev 2:

- rev 1's feedback block crashed the default BT `feedback_callback`
  (missing `delay_limit`/`status`) → §3.1 conformance.
- rev 1's single-goal **reject** policy caused task-level failures in real
  GPSR/Restaurant trees → §3.2 accept-and-queue + §3.3 `ActionBase` fixes.
- rev 1's Reentrant-group rationale was a misdiagnosis (execute callbacks
  are group-less tasks; serialization is mandatory, groups don't provide
  it) → §3.2 Facts 1–2.
- rclpy `result_timeout=900` retains image results 15 min → §3.2.
- OpenAI SDK `max_retries=2` tripled the promised cancel latency → §3.2.
- "All five reprojection copies delete" was false (RealSense body-axes
  variant is load-bearing) → §4.4.
- Seven missed callers + `messages.py` collision + missing `MockAction`s →
  §3.3; T-suite "switch" was really "add 3, switch 3" → §6.
- Staleness/QoS/decode-failure/read-only contracts under-specified → §4.2.
- Phasing double-touched object_scan and gated the whole wave on the
  contended waving file → §7 reorder.
