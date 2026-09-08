# Restaurant Orbbec Resolution + Concurrent Waving VLM Fallback — Design Spec

- **Date:** 2026-07-03
- **Status:** Approved design, pending implementation plan
- **Robot:** tinker1 / tinker2 (Orbbec Femto Bolt, head-mounted)
- **Author:** Claude (brainstormed with cindy)
- **Prior work:** builds on `2026-07-03-orbbec-hri-resolution-bump-design.md` — reuses its
  shared depth-reprojection fix and its `color_width`/`color_height` launch-arg
  passthrough without modification.

## 1. Problem & Goal

Two independent improvements, both scoped to the Restaurant behavior-tree task:

1. **Resolution.** The HRI resolution-bump work established that the Orbbec can
   launch at 1920x1080 instead of the vendored 1280x720 default, with the
   resolution-hardcode bugs that would have blocked it already fixed in shared
   infrastructure (`object_seg_yolo.py`, `door_detection.py`). That work
   deliberately did not extend to Restaurant. This spec extends it, at the
   user's direction, despite Restaurant having no direct analog to HRI's
   face/feature-enrollment VLM call — the payoff here is via Part 2 below
   (more pixels on far/small waving targets), not a detail-reading VLM call.
2. **Waving detection recall.** `waving_person_server`'s VLM fallback for
   waving-person detection exists (`enable_vlm_fallback` defaults `True`) but
   is never actually triggered in production: the sole caller,
   `BtNode_ScanForWavingPerson` in the Restaurant BT trees, never sets
   `DetectWaving.Request.min_waving_persons`, so the fallback's trigger
   condition (`request.min_waving_persons > 0 and len(found) <
   request.min_waving_persons`) is always false. In practice this means
   MediaPipe/YOLO is the *only* detector running today, and it is known to
   miss waving customers who are far away or small in frame — exactly the
   case a VLM read could catch.

Goal: launch Restaurant's Orbbec at 1920x1080 (same launch-time-only pattern
as HRI), and restructure `waving_person_server` so the VLM fallback actually
runs — concurrently with the CV pass rather than sequentially after it, with
an early-exit so a scene where CV alone already found enough wavers isn't
penalized with VLM latency it doesn't need.

## 2. Decisions (from brainstorm)

| Topic | Decision |
|---|---|
| Restaurant resolution target | **1920x1080**, same as HRI. Depth stays 640x576@30fps. |
| Launch script | **`master_restaurant1.sh`** only — confirmed via its header comment and content as the script that routes through `vision_bringup`'s `vision_driver.launch.py`/`vision_bringup.launch.py`, same pattern as HRI's `master_hri.sh`. **`master_restaurant2.sh` is explicitly out of scope** (user: "ignore any shell scripts that end in 2") — it is mid-migration, uncommitted, and contains unrelated debris (stale HRI-copied comments, an `ise_nvblox`/`use_nvblox` typo) that this work does not touch or fix. |
| VLM fallback measured latency | **Empirically measured, not assumed**: 5 live calls against a real production sample image (`vision_log/20260424_201433/detect_waving_persons_node_detect_waving_orig_20260608_181624_908.jpg`) via `_waving_vlm.request_waving_persons` — qwen3-vl-plus and gemini-2.5-pro trials ranged 3.9s–7.5s (including one trial upsampled to 1920x1080, 7.50s — modest increase over 720p, not a multiplier). Well under the original design's assumed 5-20s ceiling. This measurement is what justifies moving from "sequential fallback" to "always launch concurrently." |
| Waving-detection call structure | **Concurrent, not sequential fallback.** The VLM chain call launches on a background thread as soon as the synced RGB frame is available, running in parallel with the MediaPipe/YOLO CV pass (which doesn't depend on the VLM result). |
| Early exit | **If CV alone already found ≥2 wavers, don't wait for the VLM thread** — respond immediately with CV-only results. The VLM call is left running in the background and its result is discarded via a no-op done-callback. This is "stop waiting and discard," not true network-level cancellation (see §4). |
| Threshold value | **2**, exposed as a new declared ROS param (`vlm_skip_min_wavers`, default `2`) rather than hardcoded, matching this codebase's convention of exposing tunable thresholds. |
| Trigger gate | Replaces the dead `request.min_waving_persons > 0` condition with the existing node-level `enable_vlm_fallback` param (already default `True`) plus an available provider (`self._vlm_chain` non-empty). No new BT-node constructor parameter, no `DetectWaving.srv` change. |
| Blast radius | **Restaurant only.** Confirmed `waving_person_server`'s `/detect_waving_persons` service has exactly two callers in the whole tree — `Restaurant/restaurants.py:255` (production) and `Restaurant/restaurants_fake.py:122` (test/mock tree, left untouched, keeps today's behavior since it never sets any of the params this change reads). GPSR does **not** call this service — its own `BtNode_ScanForWavingPerson` (`GPSR/custom_nodes.py:796`) is a separate, explicitly-deprecated class calling `object_detection_generalist` directly. |

## 3. Part 1: Restaurant resolution launch change

Identical mechanism to the HRI work, applied to a different file. `master_restaurant1.sh`
inlines its vision-window launch commands directly (unlike HRI, which delegates
to `tmux_hri_vision.sh`) — both pane-0 branches (`if [ -n "$DEV" ]` / `else`)
add `color_width:=1920 color_height:=1080` to their existing
`ros2 launch vision_bringup vision_driver.launch.py ...` command lines. No
other file changes — `vision_driver.launch.py`'s `color_width`/`color_height`
args (added in the HRI work) already default to `1280`/`720` and are reused
unmodified; the FastDDS SHM segment size (already raised 20MB→64MB globally in
the HRI work) already covers Restaurant's larger frames without further
change.

**Known residual risk, accepted by the user:** Restaurant's delivery phase
(`createDeliverAllItemsPhase`) drives cuMotion/gripper actions in a tight
per-item loop for much of the task runtime, so the system has less spare
CPU/GPU/bus headroom during Restaurant than during HRI even though both start
an identical driver stack. This makes a resolution bump modestly riskier here
than it was for HRI. No specific mitigation beyond the live verification in
§6 — flagged so it's not silently assumed away.

## 4. Part 2: Concurrent VLM with early-exit

### 4.1 Current structure (for reference)

`detect_waving_callback` (`waving_person_server.py:542-`) copies the synced
`rgb_image`/`depth_image`/`header`/`camera_k` under a lock (`:546-564`), does a
depth-to-points conversion and a full YOLO+MediaPipe person/waving loop
(`:595-729`), then conditionally calls `self._vlm_augment(...)` (`:731-740`) —
today dead code, since the gating condition is never satisfied. `_vlm_augment`
(`:434-483`) does two things in one method: makes the blocking VLM chain call
(`request_waving_persons_chain`, itself a synchronous HTTP call), then dedupes
its boxes against `waving_annotations` (IoU-based, `is_duplicate_box`) and
computes centroids for the non-duplicate ones via `centroid_from_box`.

### 4.2 New structure

Split `_vlm_augment` into two methods:

- **`_start_vlm_call(self, rgb_image) -> Optional[Future]`** — if
  `self.enable_vlm_fallback and self._vlm_chain`, submits
  `request_waving_persons_chain(rgb_image, provider_models=self._vlm_chain,
  timeout_s=self.vlm_timeout_s, max_retries=self.vlm_max_retries,
  logger=self.get_logger())` to a dedicated `ThreadPoolExecutor` (see §4.3) and
  returns the `Future`. Returns `None` if VLM fallback is disabled or no
  provider has a key — callers treat `None` the same as "nothing to wait for."
- **`_merge_vlm_result(self, vlm_result, points, validmask_points, header,
  request, person_records, waving_persons_centroids, waving_annotations,
  waving_masks, waving_sources) -> tuple[int, str]`** — exactly today's
  `_vlm_augment` dedup/centroid/append logic (`:451-482`), unchanged, but
  taking an already-computed `WavingVlmResult` instead of fetching it itself.

`detect_waving_callback` changes:

1. Right after the `rgb_image`/`depth_image`/`header`/`camera_k` copy
   completes (`:559-564`, before the depth-to-points conversion), call
   `vlm_future = self._start_vlm_call(rgb_image)`. This is the earliest point
   the VLM call's only input (`rgb_image`) is available, maximizing overlap
   with the depth conversion + YOLO + MediaPipe pass that follows.
2. Run the existing CV/MediaPipe loop unchanged (`:595-729`).
3. Replace the current gated `_vlm_augment` call (`:731-740`) with:
   - If `vlm_future is None`: nothing to do (VLM disabled/unavailable), same
     as today's behavior in that case.
   - Else if `len(waving_persons_centroids) >= self.vlm_skip_min_wavers`
     (default `2`): **do not block on `vlm_future`.** Attach
     `vlm_future.add_done_callback(self._log_discarded_vlm_result)` — a small
     method that logs at debug level and swallows any exception the call
     eventually raises (so an abandoned, later-failing call doesn't produce an
     unhandled-exception warning from the executor) — and proceed straight to
     building the response from CV-only results.
   - Else: call `vlm_result = vlm_future.result(timeout=self.vlm_timeout_s)`
     inside a `try/except (WavingVlmError, concurrent.futures.TimeoutError)`
     (mirroring today's `_vlm_augment`'s own try/except around the call,
     `:443-446`) and, on success, call `self._merge_vlm_result(vlm_result,
     ...)` exactly where `_vlm_augment` is called today.

### 4.3 Executor sizing

A single dedicated `concurrent.futures.ThreadPoolExecutor` is created once in
`__init__` and stored as `self._vlm_executor` (shutdown in the node's
destructor/`main()` cleanup, matching how other long-lived resources in this
node are torn down). **`max_workers=2`, not 1.** The service callback already
runs under a `MutuallyExclusiveCallbackGroup`, so only one
`detect_waving_callback` invocation executes at a time — but an *abandoned*
VLM call from a call that early-exited can still be running when the *next*
call starts. With `max_workers=1`, that next call's `_start_vlm_call` would
enqueue behind the still-running abandoned thread and might not actually
start until the old one finishes — silently reintroducing the exact wait this
change exists to avoid. `max_workers=2` gives enough headroom for one
abandoned call plus one active call without queuing.

### 4.4 What "terminate" means here — explicitly not true cancellation

The early exit does not abort the in-flight HTTP request. `request_waving_persons`
uses the synchronous `openai.OpenAI` client; true mid-flight cancellation would
require switching to `openai.AsyncOpenAI` + `asyncio` so a cancelled task
actually closes the socket — a real architectural shift (running an asyncio
event loop inside a plain-threaded `rclpy` callback) that this spec explicitly
rejects as disproportionate to the goal. The chosen behavior is "stop waiting,
discard the late result": the abandoned call keeps running until it finishes
naturally or hits `vlm_timeout_s`, and its result — even a valid one — is
simply never used. Accepted cost: on a scene where CV finds ≥2 wavers, one VLM
call still consumes provider quota/cost in the background for a result that
is thrown away. Given this is a once-per-customer-scan call (not a hot loop),
this is a deliberate, bounded tradeoff, not an oversight.

### 4.5 Latency profile after this change

| Scenario | Before | After |
|---|---|---|
| CV finds ≥2 wavers | <1s (VLM never ran) | <1s (VLM launched but abandoned) |
| CV finds 0-1 wavers, VLM fallback would help | <1s (fallback never triggers today — dead code) | ~4-8s (waits for VLM, occasionally up to `vlm_timeout_s`=20s on a slow provider response) |
| VLM fallback disabled (`enable_vlm_fallback:=false`) | <1s | <1s (unchanged — `_start_vlm_call` returns `None` immediately) |

The middle row is the actual behavior change: today those scenes silently get
no VLM help at all; after this change they trade a few seconds of latency for
a real chance at catching a waver CV missed.

## 5. Explicitly out of scope

- Any change to `master_restaurant2.sh` (per explicit user direction).
- True network-level VLM call cancellation (asyncio migration) — see §4.4.
- Any change to `DetectWaving.srv`'s `min_waving_persons` field — stays in the
  message, unused by this new logic, not worth a breaking contract change.
- Any change to GPSR's waving-detection path (confirmed unaffected — separate
  class, separate service).
- Any change to `restaurants_fake.py`'s call site — the test/mock tree keeps
  today's behavior.
- Re-litigating the HRI resolution-bump work's shared fixes (depth
  reprojection helper, launch-arg passthrough, SHM segment size) — reused
  as-is.

## 6. Testing

No automated test can exercise the Orbbec hardware path or make live VLM API
calls part of a CI-safe suite. The concurrent/early-exit *logic* itself,
however, is unit-testable without hardware or network access by mocking
`_start_vlm_call`'s executor submission and asserting: (a) a `Future` that
resolves before the CV loop finishes still gets its result merged when CV
found <2 wavers, (b) a `Future` is left unawaited (only a done-callback
attached, no `.result()` call) when CV found ≥2 wavers, (c) `max_workers=2`
is actually configured on the executor. Implementation plan should include
these as real unit tests against the split `_start_vlm_call`/`_merge_vlm_result`
methods.

Live verification (operator-in-the-loop, mirroring the HRI plan's Task 7
pattern):
- Launch Restaurant's driver via `master_restaurant1.sh`, confirm
  `/camera/color/camera_info` reports `1920x1080` and `ros2 topic hz
  /camera/color/image_raw` sustains ~30Hz.
- Call `/detect_waving_persons` against a scene with 2+ people waving —
  confirm the response returns in under ~1-2s (early-exit path) and check the
  node log shows the VLM future was launched-then-discarded, not awaited.
- Call it against a scene with 0-1 waving people, including at least one
  person far enough / small enough in frame that MediaPipe is expected to
  miss them — confirm the response takes several seconds (VLM path engaged)
  and check whether the VLM successfully recovers the missed waver.
- Toggle `enable_vlm_fallback:=false` and confirm the call stays fast
  regardless of scene content (fallback fully disabled, no behavior change
  from before this work).
