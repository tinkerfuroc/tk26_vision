# Vision bringup: service→action conversion + intake/TF decoupling — design

- Date: 2026-07-13
- Scope: the perception-layer nodes launched by
  `src/vision_bringup/launch/vision_bringup.launch.py`
- Related: `src/vision_bringup/docs/vision-bringup-design.md` (node selection),
  `docs/superpowers/specs/2026-07-04-waving-vlm-only-live-person-design.md`
  (in-flight waving work this builds on)

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
   buffer/poll/transform block (and there are ≥5 private copies of the same
   depth→3D reprojection). Factor these into shared, composable helpers in
   `vision_util` so a node's camera source, sync policy, depth source, and TF
   policy can be changed in one place without touching detection/VLM logic.

Out of scope: the driver layer (`vision_driver.launch.py` — pan-tilt controller
services, FFS `get_depth`, vendored camera drivers), nodes not in the bringup
(`spot_on_shelf`, `object_match_all`, `placing_location`, `get_point_cloud`,
`get_orbbec_pc`, `monocular_depth`), and the known-broken BT semantics of
`BtNode_TrackPerson` (tk25_decision follow-up #4).

## 2. Service classification

Audit of every server on the 11 bringup nodes (all file:line references
verified 2026-07-13):

### 2.1 Convert to action — 6 services on 5 nodes

| Service | Node | Why an action fits |
|---|---|---|
| `detect_waving_persons` (`DetectWaving.srv`) | `waving_person_server` | Callback blocks up to ~27 s: freshness poll ≤2 s + TF snapshot poll ≤5 s + VLM `future.result(timeout=20)` (`waving_person_server.py:698-730,767-770,944`). Natural stages for feedback; VLM future is already abandonable → real cancellation. |
| `feature_extraction_service` (`FeatureExtraction.srv`) | `feature_recognition` | Awaits generalist detection then blocks on Gemini→Qwen chain (`feature_recognition.py:300-302,425-433`); chain worst case ≈2 providers × 3 retries × 20 s. |
| `seat_recommend_service` (`SeatRecommendation.srv`) | `feature_recognition` | Same node, same blocking VLM chain (`feature_recognition.py:513-520`). |
| `feature_matching_service` (`FeatureMatching.srv`) | `feature_matching` | Detection await + blocking VLM chain (`feature_matching.py:295-297,410-418`); its own TF buffer is sized 180 s specifically because the handler can run ~125 s worst case (`feature_matching.py:137-149`). |
| `seat_recommend_bbox_service` (`SeatRecommendBbox.srv`) | `seat_recommend_bbox` | VLM calls measured 10–25 s, `vlm_timeout_s=25` (`seat_recommend_bbox.py:51-59,248`); blocking sync handler. |
| `object_scan` (`ObjectScan.srv`) | `object_scan` | Fans out parallel VLM batch calls and blocks until the slowest returns (`object_scan.py:187-203`); batch progress is an obvious feedback stream and per-batch abort is cheap. |

Shared traits: the caller (behavior tree) currently cannot cancel these — a
preempted BT node abandons the future but the server keeps burning its
callback slot for up to minutes (all six serve from a MutuallyExclusive
group, so queued calls stall behind the abandoned one). Actions fix
cancellation, give the BT staged progress, and remove the temptation to tune
service timeouts around VLM p90s.

### 2.2 Stay a service — with rationale

| Service | Node | Why it stays |
|---|---|---|
| `/object_detection_generalist` | `generalist_node` | Default path is YOLO/YOLO-World (ms–hundreds of ms). The slow VLM path is opt-in (`use_vlm_sam_fallback`/`force_vlm_sam`) and already has internal race + abandon cancellation (`generalist_node.py:693-842`). Decisive: it has three **in-process synchronous service clients** — `feature_recognition`, `feature_matching`, `grocery_categorize` all `await` it mid-handler — plus 6+ BT call sites. Converting it forces an action client into every one of those or a dual srv+action facade; blast radius outweighs benefit. Revisit only if BT preemption of the VLM fallback becomes a need. |
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
separate type namespaces; `FoundationStereoDepth` already coexists in both):

`action/DetectWaving.action`, `action/FeatureExtraction.action`,
`action/SeatRecommendation.action`, `action/FeatureMatching.action`,
`action/SeatRecommendBbox.action`, `action/ObjectScan.action`.

Shape rule: **Goal = the old request fields verbatim; Result = the old
response fields verbatim; Feedback = the standard block below.** No field
renames or semantic changes in this wave — the conversion must be mechanically
verifiable against the srv definitions.

Standard feedback block (duplicated into each `.action`; no shared msg type,
so each interface stays self-describing):

```
# feedback
string stage       # machine token, per-node vocabulary, e.g.:
                   #   acquiring_frame | detecting | vlm_call | vlm_retry |
                   #   vlm_fallback | judging | transforming
string message     # human-readable detail (provider/model, batch i/N, ...)
float32 elapsed_s  # seconds since goal acceptance
```

The old `.srv` files stay in `tinker_vision_msgs_26` untouched for one
release (interface package is shared; deleting them is deferred cleanup).

### 3.2 Server pattern

Modeled on the in-repo precedents, combining the parts each got right:

- **ActionServer on a `ReentrantCallbackGroup`** — follow_head's documented
  fix for the "cancel can't be processed while execute blocks" deadlock
  (`follow_head.py:288-300`). grocery_categorize's MutEx server group is NOT
  copied.
- **Single in-flight goal, atomic test-and-set in `goal_callback`** —
  person_track's pattern (`person_track_node.py:1028-1053`). Concurrent goals
  are rejected, not preempted: these handlers contend for VLM quota and
  camera frames; queueing them helps nobody.
- **`async def execute_callback`** where the node awaits the generalist
  detection client (feature_extraction, feature_matching), keeping the
  existing two-callback-group client split. Plain sync execute elsewhere.
- **`MultiThreadedExecutor` everywhere.** `feature_recognition` and
  `feature_matching` currently run plain `rclpy.spin()` and must upgrade;
  their existing MutEx groups already make that safe.
- **Cancellation is cooperative**: `CancelResponse.ACCEPT`; the execute loop
  checks `goal_handle.is_cancel_requested` at every stage boundary; the
  kimi_api provider-chain helpers gain an optional `should_abort:
  Callable[[], bool]` checked between retries and providers (no mid-HTTP
  abort — bounded by one `vlm_timeout_s`); waving abandons its VLM future via
  the existing discard-callback path. On cancel: `goal_handle.canceled()`,
  result `status=1`, `error_msg='canceled'`.
- Terminal semantics: `succeed()` for any legitimate answer **including
  empty/none** (chain convention preserved); `abort()` only for internal
  errors. Degrade modes (waving VLM→mediapipe auto-degrade, fallback-provider
  drop on missing key) are untouched.
- Server names keep their current strings (`feature_extraction_service` etc.)
  even where the `_service` suffix reads oddly — zero churn in launch
  scripts, params, and test greps.

### 3.3 Client migration (tk25_decision + in-repo callers)

The `ActionHandler` base (`TemplateNodes/ActionBase.py`) already gives the
polling recipe: ctor with action type/name, override `send_goal()`, inherited
polling `update()`/`process_result()`, auto-cancel in `terminate()` — exactly
what these BT nodes need (preemption now actually cancels server work).
Call sites to migrate `ServiceHandler`→`ActionHandler`:

| Caller | File |
|---|---|
| `BtNode_ScanForWavingPerson` | `TemplateNodes/Vision.py:1713` |
| `BtNode_ScanForWavingPersonNew` | `GPSR/custom_nodes.py:880` |
| `BtNode_DetectCallingCustomer` | `Restaurant/custumNodes.py:14` |
| `BtNode_FeatureExtraction` | `TemplateNodes/Vision.py:699` |
| `BtNode_SeatRecommend` | `TemplateNodes/Vision.py:868` |
| `BtNode_SeatRecommendBbox` | `TemplateNodes/Vision.py:954` |
| `BtNode_FeatureMatching` | `TemplateNodes/Vision.py:1059` |
| `BtNode_ObjectScan` | `TemplateNodes/Vision.py:345` |
| `WaveReseedBridge` (raw rclpy client, not a BT node) | `FollowPerson/wave_reseed_cycle.py:33` → action client, still future-driven |

Also: `behavior_tree/messages.py` mock imports, and the tk26 integration
suite (`scripts/tests/` T1 advertise checks + T2/T3 call scripts) switch from
`ros2 service call`/service clients to action clients for the six names.

## 4. Intake / TF decoupling

### 4.1 The duplication being removed

- **Subscription+sync+cache**: ≥10 private implementations —
  `object_seg_yolo.py:244-311` (dual-camera ATS + staleness retry loop),
  `generalist_node.py:112-132` (extra unsynced depth sub),
  `get_image.py:56-122`, `get_point_cloud.py:30-106` (PC2 variant),
  `door_detection.py:30-66` (unsynced), `waving_person_server.py:83-91,698-730`
  (ATS + freshness barrier), `seat_recommend_bbox.py:227-236`,
  `object_scan.py:92-136` (staleness-checked plain subs),
  `feature_recognition.py:209-220`, `person_track_node.py:697-730` (seq
  consume + stall watchdog), `follow_head.py:252-283` (header dedup).
- **Depth→metres / 2D→3D reprojection**: 5 copies —
  `vision_util/depth_reproject.py` (canonical), local clones in
  `waving_person_server.py:37-59`, `person_track_node.py:829-872`,
  `follow_head.py:1123-1160`, hand-rolled RealSense math in
  `object_seg_yolo.py:536-550`, Orbbec mm-decode in
  `seat_recommend_bbox.py:606-613`.
- **TF**: four buffer configurations (10 s / 60 s / 180 s caches), three
  lookup idioms (single-try `object_seg_yolo.py:1035-1068`, poll-until-deadline
  `waving_person_server.py:384-424`, stamped-with-fallback
  `feature_matching.py:194-227`).

### 4.2 New shared modules (in `vision_util`)

`vision_util` is already the de-facto shared library (`depth_reproject`,
`mask_utils`, `_pc_utils`, `vision_logging` are imported across packages), so
the helpers land there — no new package, no dependency churn.

**`vision_util/camera_intake.py`**

```python
@dataclass
class IntakeConfig:
    camera: str                    # label: 'orbbec' | 'realsense'
    color_topic: str | None        # any stream may be omitted
    depth_topic: str | None        # aligned/raw depth Image
    points_topic: str | None       # PointCloud2 (exclusive with depth_topic)
    camera_info_topic: str | None
    sync_queue: int = 10           # ATS used iff ≥2 image streams configured
    sync_slop_s: float = 0.1
    qos_depth: int = 5
    best_effort: bool = True

class FrameBundle:
    # camera, seq (monotonic), header, stamp, recv_time (node clock)
    # color_msg / depth_msg / points_msg / K — configured subset
    def color_bgr(self): ...   # lazy CvBridge; normalizes Orbbec rgb8 vs bgr8
    def depth_m(self): ...     # 16UC1 mm→m / passthrough via depth_reproject
    def points_xyz(self, roi=None): ...  # pinhole deprojection, optional ROI

class CameraIntake:
    def __init__(self, node, cfg, callback_group=None): ...
    def latest(self, max_age_s=None) -> FrameBundle | None: ...
    def wait_fresh(self, max_age_s, timeout_s, poll_s=0.05) -> FrameBundle | None: ...
    def latest_new(self, last_seq) -> FrameBundle | None: ...  # consume semantics
    def intrinsics(self): ...
    @staticmethod
    def declare_params(node, camera, defaults) -> IntakeConfig: ...  # uniform
        # <camera>_color_topic / <camera>_depth_topic / camera_info_topic /
        # sync_slop_s / staleness_s param names across all nodes
```

Design points: one `CameraIntake` per camera per node (generalist's extra raw
depth sub becomes a second depth-only instance); callers keep their loop
policy (person_track's stall watchdog and follow_head's dedup stay local,
driven by `seq`/`recv_time`); `wait_fresh` replaces both the yolo staleness
retry loop and waving's freshness barrier; QoS depth and callback group are
injectable so the tuned tracker settings survive.

**`vision_util/tf_lookup.py`**

```python
class TransformHelper:
    def __init__(self, node, cache_time_s=180.0): ...   # VLM-safe default
    def try_lookup(self, target, source, stamp=None, timeout_s=0.1): ...
    def wait_lookup(self, target, source, deadline_s, latest=True): ...
    def transform_point(self, pt, transform_or_target): ...
    buffer: tf2_ros.Buffer  # escape hatch
```

Uniform exception policy: lookup failures return `None` with a throttled
warning; callers decide whether that aborts (seat_recommend_bbox pre-VLM
gate) or degrades (feature_matching's untransformed-point fallback). The
frame-convention gate for RealSense centroids
(`_frame_supports_tf_transform`, `object_seg_yolo.py:1030-1033`) is caller
policy and stays in the detection node.

**`vision_util/depth_source.py`**

The FFS-vs-native selection currently buried in
`object_seg_yolo.py:416-523` (`prefer_ffs`, lazy client on a Reentrant group,
Event-blocking call, throttled fallback warning, source tag) moves here as
`FfsPreferredDepthSource(node).acquire(frame) -> (depth_m, source_tag)`. This
is the single point to swap depth backends later (e.g. monocular fusion) —
the explicit modifiability goal of this refactor.

**`vision_util/depth_reproject.py`** absorbs the local reprojection clones
(mm→m decode variants, ROI-restricted deprojection, cached meshgrid) so all
five copies delete.

### 4.3 Per-node adoption map

| Node | Intake | TF |
|---|---|---|
| `yolo_seg_node` / default / `generalist_node` (shared base) | 2× `CameraIntake` (color+depth ATS) replace `_init_subscribers` + staleness loop; generalist adds a depth-only instance; `_acquire_depth` → `FfsPreferredDepthSource` | `TransformHelper` (60 s→default 180 s cache is a strict widening) |
| `door_detection` | depth+info instance (no color, no ATS) | — |
| `get_image` | 2× color+depth instances | — |
| `waving_person_server` | color+depth instance; `wait_fresh` replaces freshness barrier; local reproject deleted | `wait_lookup` replaces `_snapshot_latest_transform` |
| `feature_recognition` | color-only instance for the seat path (extraction path's "intake" is the generalist client — unchanged) | — |
| `feature_matching` | none (detection-client based) | `TransformHelper` replaces its private block |
| `seat_recommend_bbox` | color+depth instance | `TransformHelper` (keeps pre-VLM snapshot policy) |
| `object_scan` | 2× color-only instances; `latest(max_age_s=1.0)` replaces its staleness check | — |
| `person_track_server` | instance with `latest_new` consume, QoS 5, Reentrant group; watchdog/EMA/FSM untouched | unchanged (`try_lookup` is a drop-in if desired) |
| `follow_head` | instance; header dedup → seq dedup | none by design (analytic servo frame) — stays |

## 5. Approaches considered

**Conversion strategy** — (a) clean cutover per node: srv server removed,
action server added, all in-repo callers migrated in the same wave
*(chosen — all callers are in-repo, the ActionHandler recipe is established,
and dual-serving means maintaining two code paths through VLM handlers)*;
(b) transitional dual srv+action per node — only worth it if out-of-repo
callers surface; (c) generic "long-running service" wrapper action — rejected,
loses typed goals/results.

**Decoupling placement** — (a) modules in `vision_util` *(chosen — already
the shared library, flows with existing dependency direction)*; (b) new
`vision_intake` package — cleaner nominally, but adds a package + rebuild
churn for no isolation gain; (c) a central camera-relay node that owns all
subscriptions (kills the triple-subscription follow-up #3 for real) —
attractive later, but a runtime-topology change with its own failure modes;
explicitly out of scope, and `CameraIntake` is the stepping stone to it.

## 6. Testing

- **Unit**: new pytest for `camera_intake` (synthetic Image/CameraInfo msgs:
  sync pairing, staleness windows, seq consume, rgb8/bgr8 normalize, mm→m),
  `tf_lookup` (static broadcaster fixtures), `depth_source` (mock FFS client:
  prefer/fallback/timeout paths). Reprojection equivalence: golden-value test
  pinning `depth_reproject` outputs against the deleted local copies' math.
- **Integration suite updates**: T0 interface checks add the six actions; T1
  advertise checks switch to `ros2 action list`; T2/T3 callers switch to
  action clients; response invariants in `CLAUDE.md §Testing` re-stated for
  results (empty-scene `status=1, []` semantics unchanged).
- **BT**: `behavior_tree` mock-mode tests for each migrated node
  (ActionHandler mock path); one preemption test per converted node
  (terminate mid-goal → server observes cancel).
- **Behavioral parity gate**: for each converted node, one recorded
  live-scene run (T2 tier) comparing action result fields against the
  pre-conversion service response on the same scene class.

## 7. Phasing

1. **Wave 0 — helpers**: `camera_intake` / `tf_lookup` / `depth_source` /
   `depth_reproject` extensions + unit tests. No node behavior change.
2. **Wave 1 — intake adoption, stay-service nodes**: detection base class,
   door_detection, get_image, object_scan (intake only). T2 regression after
   each node.
3. **Wave 2 — action conversions**: per node (waving → feature_matching →
   feature_recognition → seat_recommend_bbox → object_scan), each sub-wave =
   interfaces + server + BT clients + tests. Waving goes first only after the
   in-flight VLM-only work (2026-07-04 spec) lands — it touches the same file.
4. **Wave 3 — tuned trackers**: person_track + follow_head intake adoption,
   last, behind extra care (these have arena-tuned QoS/dedup behavior).

Each wave is independently shippable; an arena freeze can stop after any wave.

## 8. Risks

- **Waving file contention**: `waving_person_server.py` has uncommitted
  in-flight changes on `tinker2-net`; Wave 2 must rebase on that landing.
- **Executor upgrade** (`feature_recognition`/`feature_matching` to
  MultiThreadedExecutor): existing MutEx groups keep handler serialization,
  but the seat/extraction services sharing one node means two goals can now
  interleave across the *two different* servers — acceptable (different
  resources), noted for review.
- **BT preemption now has teeth**: `terminate()` cancels server work; any BT
  tree that relied on fire-and-forget service side effects (vision logging of
  abandoned calls) will see fewer log artifacts. Audit during Wave 2.
- **`_service`-suffixed action names** are permanent cosmetic debt, chosen
  deliberately to avoid launch/test churn.
