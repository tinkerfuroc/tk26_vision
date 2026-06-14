For testing:
```bash
ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true depth_registration:=true

ros2 run vision_track person_track_server --ros-args -p reid_mode:=custom 2>&1 | stdbuf -oL -eL tee -a person_track.log || true

ros2 run vision_track person_track_test_client -d
```

## Active re-ID (behaviour-tree contract)

The person tracker exposes an active re-acquisition loop: when the target is lost
long enough, `TrackPerson` feedback escalates `reacquisition_state` to
`REACQ_NEEDS_HELP`, the BT asks the operator to raise a hand, and re-seeds the
tracker on the waving box (`ReseedTarget` at `~/reseed_target`, gallery-preserving).
See [`docs/active_reid.md`](docs/active_reid.md) for the full consumer contract.

### `~/reacq_state` heartbeat (navigation follow executive)

A permanently-on, low-rate **`std_msgs/UInt8`** topic — `~/reacq_state`
(`/person_track_node/reacq_state`) published at **10 Hz** — that mirrors the
`TrackPerson` feedback reacquisition enum as a plain byte so the navigation
follow executive can consume it **without depending on `tinker_vision_msgs_26`**:

| Value | Meaning |
|---|---|
| `0` | `TRACKING` — target held this frame |
| `1` | `PASSIVE` — lost, within the passive-recovery window |
| `2` | `NEEDS_HELP` — lost past `help_after_sec`, awaiting active re-ID |
| `255` | `INACTIVE` — **no `TrackPerson` goal active** (idle / between goals / tracker shut down) |

It carries the last value pushed to the action feedback (updated at each feedback
publish) and reverts to `255` (`INACTIVE`) on goal teardown — success, abort,
cancel, or exception. **Consumer guidance:** treat `1`/`2` as a "wait, don't
abort" hold (the tracker is actively trying to recover the operator) and `255` as
"no follow target — tracker is not driving" (distinct from a real loss). Unlike
the `debug_*` telemetry below, this publisher has **no param gate** — it is always
on, since the consumer needs a continuous liveness signal. It is purely advisory
and reflects the tracker's published state; it changes no tracking logic.

## track_web dashboard

A browser-based **live tracking dashboard + active-reID test bench** for
`person_track_server`. It renders the tracker's debug image (MJPEG), the live
target/candidate state (WebSocket), and the re-ID gallery thumbnails, and it
lets a human stand in for the behaviour tree — clicking a person re-seeds the
tracker, so you can validate the Spec B active-reID loop without a real BT.

Run it:

```bash
ros2 run vision_track track_web --ros-args -p bind:=0.0.0.0 -p port:=8766
```

then open `http://<robot>:8766/`.

### Bench bringup (one command)

To start the whole bench stack — tracker (with telemetry ON), waving server, and
dashboard — in a single command:

```bash
ros2 launch vision_track track_web_bench.launch.py [bind:=… port:=… with_waving:=…]
```

It runs `person_track_server` with the three `debug_*` flags forced ON, the
`waving_person_server` (gated by `with_waving`, default `true`), and `track_web`
(`bind` default `0.0.0.0`, `port` default `8766`). Cameras are deliberately not
included — start them first per `src/tk26_vision/CAMERA_BRINGUP.md`.

With the bench launch up, the dashboard now shows the live camera image even
before any goal is sent (badge `IDLE`) and during the goal's init search (badge
`INITIALIZING`); the annotated tracker overlay replaces the raw frame once the
tracker locks. So you get a live preview to frame your shot / confirm the camera
is up without first starting a goal.

### Enabling the tracker-side telemetry

The dashboard consumes three publishers that the tracker only emits when
explicitly enabled. **All three default off — zero production impact** (no extra
encode/serialize work in a normal competition run). Turn them on for a
debugging/bench session:

```bash
ros2 run vision_track person_track_server --ros-args \
  -p debug_state_enabled:=true \
  -p gallery_keep_crops:=true \
  -p debug_image_enabled:=true
```

- `debug_state_enabled` → `~/debug_state` (target/candidate JSON + scores)
- `gallery_keep_crops` → `~/debug_gallery` (per-identity thumbnail crops; also
  what makes the gallery thumbs appear in the dashboard)
- `debug_image_enabled` → `~/debug_image` (the annotated frame, streamed as MJPEG)

### Bench mode vs observer mode

- **Bench mode** — when no behaviour tree holds the `track_person` goal, the
  dashboard's **Start / Stop** buttons send/cancel the action goal themselves, so
  you can drive the tracker standalone.
- **Observer mode** — when a BT already holds the goal, the dashboard
  automatically drops to read-only observation (no Start/Stop), so it never
  fights the real consumer for the action.

### Re-seeding the target (human-as-BT)

- **Click a person** in the live view → posts to the tracker's
  `~/reseed_target` service, gallery-preserving (same re-lock the BT uses).
- **👋 DetectWaving** → calls the waving service, draws the returned wave boxes,
  and clicking a wave box re-seeds on it. This closes the **human-as-BT loop**:
  a person raises a hand, you click their box, the tracker re-locks — exactly the
  active-reID recovery path the BT will automate, usable now for Spec B
  validation.

### Params

| Param | Default | Meaning |
|---|---|---|
| `bind` | `127.0.0.1` | HTTP/WS bind address (`0.0.0.0` to reach from another host) |
| `port` | `8766` | HTTP/WS port |
| `tracker_node_name` | `person_track_node` | node name (optionally `ns/node`) the bridge resolves the `debug_*` topics and `reseed_target` service against; the `track_person` action client stays relative and is unaffected |
| `waving_service` | `detect_waving_persons` | DetectWaving service the 👋 button calls |

## Changelog

- **2026-06-14** — pan-tilt person-follow core (`core/pan_follow.py`): pure
  horizontal pan-to-center control law — ABSOLUTE target `current_pan +
  sign*atan2(u-cx, fx)` (recomputed from live state + live pixel error, so error
  never accumulates), with error-vs-state deadband, anti-chatter min-change, min
  command interval, EMA + clamp. CENTER/RECENTER(NEEDS_HELP)/HOLD modes. Wired
  into the tracker in a follow-up commit. Tests: `test/test_pan_follow.py` (9).

- **2026-06-14** — **fix: passive re-ID now runs indefinitely under the NEEDS_HELP
  hold (operator could not be re-acquired after ~20 s lost).** The hold is
  indefinite (`active_help_timeout_sec=0`), but `reidentify_target` dead-ended once
  `frames_lost > max_frames_lost` (600 ≈ 20 s @ 30 fps): past the cap it returned
  None every frame and kept incrementing `frames_lost`, so `find_best_match_reid` /
  `_confirm_reid_candidate` never ran again — a fully-visible returning operator was
  never re-locked, and with wave→reseed not wired into the follow tree the follow
  sat alive but frozen. The active-help timer only escalates the *advisory*
  NEEDS_HELP state (so the robot can request help); it must not bound re-ID. Now
  the node sets `tracker.active_help_enabled = (active_help_after_sec > 0)` and
  `reidentify_target` keeps re-ID alive indefinitely while it is enabled; the cap
  bounds only the legacy active-help-disabled coast. No ReID/lock threshold
  changed. Fixed a stale "rate * lost_timeout" comment + init log that misdescribed
  the cap. Tests:
  `test/test_needs_help_reacq_integration.py::test_reacq_after_frames_lost_exceeds_cap_indefinite_hold`
  (+ `…_cap_still_applies_when_active_help_disabled` legacy guard).

- **2026-06-14** — track_web: static assets (`/`, `/app.js`, `/style.css`) now
  send `Cache-Control: no-cache` so the browser revalidates against the server's
  Last-Modified on every load. Without any cache directive a browser could keep a
  stale bundle across a rebuild/relaunch indefinitely — the cause of "I relaunched
  and the UI still shows the old thing" (e.g. the `dummy_nav` control, dropped in
  the 2026-06-13 integration, lingering in a cached pre-integration `app.js`).
  Clear an already-cached tab once with a hard refresh (Ctrl+Shift+R). Also
  removed a stale `dummy_nav` comment in `track_web.py`. Test:
  `test/test_track_web_app.py::test_webui_static_assets_are_no_cache`.

- sim_track_server: topic->action adapter mirroring /target_points + reacq_state into /track_person, the sim stand-in for person_track_server (F4 BT-in-the-loop).

- **2026-06-13** — track_web: follow-mode selector (vision+audio | with-nav) +
  Start/Stop + live Follow-state panel; dropped dead dummy_nav; allowlist now
  audio/follow_server/bt_vision/bt_nav.

- **2026-06-10** — **track_web: state panel + gallery now update at goal start
  (no more ~10 s freeze).** The video stream came up immediately but the state
  panel and gallery sat frozen until the first lock, because (1) `~/debug_gallery`
  was only ever published from the post-lock tracking branch, (2) the init
  `'initializing'` `~/debug_state` payload carried no fields the dashboard
  renders (so it looked dead), and (3) a post-goal-accept warmup window
  (idle-timer already off, CUDA warmup holding `lock_tracker`) published nothing.
  Fixes, all additive to the search phase: emit an empty `{"version":0,
  "thumbs":[]}` "searching" gallery once per goal; publish the `'initializing'`
  state (+ searching gallery) up front *before* the warmup block to close the
  blackout; add a `search_started_ts` anchor to the init payload and render an
  animated "Searching for target… Ns" banner (1 Hz elapsed timer) in the webui
  while `fsm_state === 'initializing'`. Post-lock tracking/gallery behaviour is
  unchanged. Verified: `debug_gallery` (v0) and `debug_state` (`initializing`,
  carrying `search_started_ts`) both arrive ~300 ms after goal-accept, before any
  lock.

- **2026-06-10** — **`~/reacq_state` UInt8 heartbeat for the navigation follow
  executive.** Added a permanently-on 10 Hz `std_msgs/UInt8` topic
  (`/person_track_node/reacq_state`) mirroring the `TrackPerson` feedback
  reacquisition enum (`0` TRACKING / `1` PASSIVE / `2` NEEDS_HELP), reporting
  `255` INACTIVE whenever no goal is active (idle / teardown). Lets the nav-side
  follow executive distinguish a "wait, don't abort" PASSIVE/NEEDS_HELP hold from
  tracker shutdown **without depending on `tinker_vision_msgs_26`**. Additive
  only — mirrors the value at the two feedback-publish sites and reverts to
  INACTIVE in `_cleanup_tracking`; no tracking logic changed. (P2 of the
  person-following navigation plan.)

- **2026-06-10** — **`track_web_control.launch.py`: follow-demo control surface
  (tracker + upgraded dashboard).** A focused one-command bringup of the person
  tracker (with the three `debug_*` telemetry flags forced ON, like the bench) +
  the upgraded `track_web` dashboard. The dashboard's new **Bringup panel** has
  per-component start/stop toggles for the follow demo's supporting stack —
  **audio** (`ros2 launch audio_pakage audio.launch.py`), **dummy_nav**
  (`ros2 run behavior_tree dummy-nav`), and the **follow-person BT**
  (`ros2 run behavior_tree follow-person`) — spawned on demand at runtime by a
  fixed-allowlist `ProcessManager` (NOT part of this launch). While the BT runs,
  the dashboard's manual goal-Start is disabled (the BT owns the
  `/track_person` goal). Args: `launch_tracker` (default `true`; set `false` to
  point the dashboard at an already-running tracker — the dashboard always
  starts), `with_waving` (default `false`, the 👋 button backend), `kill_stale`
  (default `true`, SIGTERMs stale tracker/dashboard `lib/<pkg>/` execs first),
  `bind` (`0.0.0.0`), `port` (`8766`), `perf_logging` (`false`).
  **Prerequisite:** build `behavior_tree` + `audio_pakage` into
  `tk25_ws/install` via `tkbuild tk25_decision` + `tkbuild tk_24_audio` so the
  Bringup-panel spawns resolve. Run:
  `ros2 launch vision_track track_web_control.launch.py` (cameras started first
  per `src/tk26_vision/CAMERA_BRINGUP.md`).

- **2026-06-10** — **fix: NEEDS_HELP passive re-acquisition near-impossible
  despite a clear, near-identical operator.** Root cause: the N-of-M re-ID
  confirm window (`reid_confirm_window`) was keyed on the raw ByteTrack
  `track_id` and **wiped on every id change** — but during a real loss +
  reappearance ByteTrack re-emits the returning operator under a *churning*
  `track_id` (a fresh id most/every frame), so `sum(window)` never reached the
  arm (3) or commit (12) count even at high similarity → the swap never
  committed. Fix (in `_confirm_reid_candidate`): while `in_help` (latched
  NEEDS_HELP **AND** exactly one person visible) an id change KEEPS the
  accumulated window and only retargets the id, so the lone returning operator
  re-locks regardless of id churn. Precision preserved — the `is_hit` bar
  (`>= single_person_commit_bar_help` 0.62) and the `num_candidates == 1` gate
  are unchanged (a different person below the bar still accumulates nothing), and
  OUTSIDE `in_help` the strict id-keyed window reset is byte-for-byte unchanged.
  Surfaced by an integration repro (stable id re-locks; same-similarity churning
  id did not, until fixed).

- **2026-06-10** — **fix: premature `/track_person` abort after loss
  (regression from the wall-clock escalation).** Moving the help-latch to 5 s
  wall-clock decoupled it from the lock-FSM recovery cap (`max_recovery_frames`,
  ~1.5 s @ 30 fps), which still set `hard_lost` and fired the abort branch in the
  `[~1.5 s, 5 s]` window — so the goal aborted (`'hard-lost (recovery cap)'`)
  before reaching the NEEDS_HELP hold. (Pre-change the frame-based latch coincided
  with the recovery cap and always shadowed it.) Fix: the abort decision is now a
  pure `lost_should_abort(...)` helper — the FSM recovery cap aborts ONLY when
  active help is DISABLED (`active_help_after_sec <= 0`); with active help enabled
  the goal coasts through the passive-recovery window into the indefinite
  NEEDS_HELP hold. The `lost_timeout` absolute ceiling and the legacy
  (help-disabled) abort are unchanged. `_handle_frame_stall` inherits the fix.

- **2026-06-10** — **tracker seg model upgraded to `yolo11m-seg` by default.**
  Benchmark (`scripts/bench_yolo_seg.py`, imgsz 736 fp16, RTX 5070 Ti): `s`=4.0 ms
  (250 fps), **`m`=5.5 ms (182 fps)**, `l`=7.0 ms (143 fps), `x`=9.5 ms (105 fps)
  — all far under the 33 ms/frame 30 Hz budget *in plain PyTorch*, so the seg
  model is nowhere near the bottleneck and a bigger model is fast enough **without
  TensorRT** (TRT is now an optional top-end, not a requirement). `m`-seg gives
  better and more-frequent masks, which directly feed the OSNet background
  neutralization and the new mask-fill gallery gate. `model_path` default
  `yolo11s-seg.pt` → `yolo11m-seg.pt` (node + `config/default.yaml`); override back
  to `s` on weaker/offline GPUs (the m-seg weight must be cached for offline
  arena), or point at a `.engine` for TRT (`scripts/export_yolo_trt.py`).

- **2026-06-10** — **gallery admission gated on BOTH bbox w/h ratio AND
  mask-fill (upright + clean).** The operator is always standing, so only
  upright, well-segmented views should enrich the gallery. Admission now requires
  BOTH: bbox width/height `<= gallery_max_aspect_ratio` (default **0.5** — upright
  is taller-than-wide ~0.4; a square/wide box rejects) AND mask-fill
  `> gallery_min_mask_fill` (default **0.35** = mask_px/bbox_area, strict `<=` —
  a merged/garbage/occlusion-inflated box has the operator mask as a thin slice →
  reject). `crop_quality_ok` ANDs the checks (reject if either fails); a
  None-coverage frame isn't rejected on fill but still must pass the aspect gate.
  Both are per-call overrides of the shared gate (assembled in `update_appearance`
  from tracker attrs; `DEFAULT_GATE` unchanged) and adjustable at launch.
  Admission-only — never affects the query/match path or the live lock; a rejected
  frame still refreshes motion/last-seen. (Supersedes the first-draft mask-fill-
  only / relaxed-aspect-2.0 approach — operator opted to keep the upright w/h gate
  too.)

- **2026-06-10** — **passive recovery: post-NEEDS_HELP auto re-lock + wall-clock
  escalation.** (1) While *latched* in NEEDS_HELP with exactly one person visible,
  the lone re-ID commit bar relaxes from `single_person_commit_bar` (0.72) to
  `single_person_commit_bar_help` (default **0.62**), requiring
  `needs_help_confirm_frames` (**12**) confirm hits within the last
  `needs_help_commit_window` (**16**) frames — so a returning operator
  auto-re-locks without a wave. Outside that gate (multi-person, or before
  escalation) behaviour is byte-for-byte unchanged (strict 0.72 lone /
  `reid_threshold` multi); the N-of-M sustained gate is preserved and
  strengthened. The relaxed commit produces a real `target_track_id` swap, so the
  help-latch clears on the genuine re-lock. (2) **NEEDS_HELP escalation is now
  wall-clock, not frame-count:** new param `active_help_after_sec` (default
  **5.0 s**) replaces `active_help_after_frames` (was 45) — frame rate is
  unreliable under tournament GPU contention, so a frame threshold gave an
  unpredictable real-time window. `reacq_state(tracked, time_since_lost,
  help_after_sec)` and `_is_awaiting_help` now key off seconds since the last
  CONFIRMED lock, anchored by `self._last_confirmed_time` (refreshed only on a
  true re-lock — never a provisional coast — preserving the abort-mid-reacquire
  protection). The debug-state telemetry field `active_help_after_frames` →
  `active_help_after_sec`.

- **2026-06-09** — throttled the tracker's `vision_log` EVENT writes to real
  acquire/lost transitions plus a ~5 s steady-state heartbeat, eliminating the
  per-frame churn flood. Previously the lost↔reclaim artifacts were keyed on
  `_was_lost`, which flips per-frame on `track_result` presence; in a churny
  scene (rapid loss/reacquire, id churn) the "transition" writes re-fired over
  and over, and there was **no** logging during steady tracking or steady lost.
  Now `PersonTrackNode._vision_log_due(state, now)` gates emission: it fires on a
  debounced state change (state changed **and** ≥ `vision_log_min_gap_s` since
  the last write) or on the steady-state heartbeat (≥ `vision_log_interval_s`
  since the last write), for both tracking and lost. New params
  `vision_log_interval_s` (default `5.0`) and `vision_log_min_gap_s`
  (default `1.0`); new states `self._vlog_last_state` / `self._vlog_last_time`.
  Acquire emits `event=acquired` (first lock), `reclaimed` (lost→tracked), or
  `tracking` (steady heartbeat); loss emits the `lost` + `lost_current` pair on a
  real transition or a single `lost_heartbeat` artifact on the steady-lost
  heartbeat. `_was_lost`/help-latch/point-EMA-reset logic is unchanged — only the
  log-emission decision moved to the throttle; gallery thumbs out of scope.

- **2026-06-09** — bounded the per-track-id state dicts to stop a slow host-RAM
  growth over long runs. `candidate_consistency` and `relative_positions` are
  keyed by ByteTrack track_id, which increases monotonically for the life of the
  process (a fresh id every time someone enters/leaves), so both dicts grew one
  entry per id forever — a slow leak feeding the long-run swap-thrash (the
  secondary cause behind the ssh-drop + Orbbec depth-engine SIGSEGV; the
  dominant orphan-process pileup was fixed separately). New
  `YOLOTracker._prune_track_state(current_ids)`, called once per frame from
  `core/tracking_pipeline.py update_tracker`, lazily evicts gone ids past
  `MAX_TRACK_STATE_IDS` (256): it only acts once a dict exceeds the cap and only
  removes ids NOT visible this frame, so a currently-relevant id (even one
  flickering through occlusion) is never dropped and scenes with ≤ cap distinct
  ids behave **identically** to before — no scoring threshold or window changed.
  `scene_center_history` was already bounded (capped to 3 in `update_scene_motion`)
  and `target_velocity_history` is declared/cleared but never appended (dead
  state, always empty), so both were left untouched.

- **2026-06-09** — idempotent `track_web_bench` startup: `kill_stale` guard
  (default `true`) SIGTERMs any stale `person_track_server`, `track_web`, and
  `waving_person_server` instances via narrow `lib/<pkg>/`-scoped patterns before
  the bench nodes start. Fixes orphan pileup across ungraceful bench restarts
  (terminal closed / SIGKILL): each orphaned `person_track_server` squatted ~700
  MiB GPU + growing host RAM until swap-thrash dropped ssh and SIGSEGV'd the
  Orbbec depth engine. Patterns scoped to installed `lib/<pkg>/` exec paths so
  editors, greps, and the parent `ros2 launch` process are never matched. SIGTERM
  only (never -9), letting each node release cameras/GPU cleanly. Launch arg
  `kill_stale:=false` disables the guard for CI/headless scenarios. Nodes start
  only after the cleanup `ExecuteProcess` exits (`OnProcessExit`). Same three
  patterns mirrored in `scripts/kill_stale_bench.sh` for manual cleanup.

- **2026-06-09** — do not give up on look-alikes during **passive** reacquisition
  (operator returns without a wave), implemented as Option B (pursue floor) +
  Option A (N-of-M) **without lowering any commit bar** (Option D — relaxing the
  color veto — was dropped; `reid/reid.py` and `DEEP_CONFIDENT_BYPASS` are
  untouched). A lone returner scoring ReID ~0.55-0.71 with occasional dips used
  to be dropped: the lone candidate hit the hard `0.72` `_single_candidate_guard`
  wall (so `find_best_match_reid` returned `None` every sub-0.72 frame) and any
  dip wiped the strict-consecutive confirmation streak, so 12 unbroken ≥0.72
  frames were effectively unreachable for a real returner.
  - **Pursue floor** (`reid_search._single_candidate_guard`, new param
    `single_person_pursue_floor`, default **0.55** = `reid_threshold`): a lone
    person whose similarity is in `[pursue_floor, 0.72)` is now KEPT IN PLAY
    (pursued, surfaced as `reidentifying` / `target_lost=True` → YELLOW) instead
    of discarded. Below the floor it is still discarded. Pursuit is **not** a
    lock.
  - **Commit bar held high** (`tracking_pipeline._confirm_reid_candidate`, new
    param `single_person_commit_bar`, default **0.72**): a frame counts as a
    *confirm hit* only when `match_similarity >= commit_bar`, where
    `commit_bar = single_person_commit_bar` when there is one candidate else
    `reid_threshold`. `num_candidates` is now computed in `reidentify_target`
    BEFORE the confirm call and passed in. **THE PRECISION INVARIANT:** lowering
    the lone *pursue* floor did NOT lower the lone *commit* bar — a lone candidate
    that never clears 0.72 is pursued but **never** committed (no wrong-person
    lock). `frames_lost` resets only on a confirm hit, so a pursued-but-not-hit
    lone frame leaves it growing and `NEEDS_HELP` still escalates (operator can
    wave). **Both commit paths respect the held-high lone bar:** the pre-confirm
    ramp ARMS `pending_reid_match` only after `reid_preconfirm_frames`
    **commit-bar hits** (`sum(reid_confirm_window)`), not after
    `reid_preconfirm_frames` `reid_threshold`-counted `reid_fit_streak` frames. A
    lone sub-0.72 candidate therefore never arms, so neither the Stage 2 N-of-M
    commit (`_confirm_reid_candidate`) NOR the Stage 1 by-id adoption
    (`track_by_id` → `_confirm_pending_reid`, which would otherwise lock the
    pending id by its ByteTrack id without re-checking the bar) can ever lock it.
    (Closes a precision leak found in Phase-3 spec-review: the original arming on
    `reid_fit_streak` let a lone 0.60 candidate arm pending and be locked via
    Stage 1, bypassing the 0.72 bar.) For the multi-candidate case
    `commit_bar == reid_threshold`, so arming is unchanged.
  - **N-of-M confirmation** (new param `provisional_commit_window`, default
    **18** = M; reuses `reid_confirmation_frames` = **12** = N): the strict-
    consecutive `consecutive_reid_frames` commit is replaced by a sliding window
    (`reid_confirm_window`) — commit when there are **≥ N confirm hits within the
    last M frames**. A non-hit frame (sim in `[pursue_floor, commit_bar)`) KEEPS
    the pending alive (does not zero the window), so dips are tolerated; the good
    ≥0.72 frames accumulate across dips and eventually reach the commit (→ GREEN).
    N consecutive hits still commit (a 12-of-12 within an 18-window), so the old
    behaviour is a strict subset and existing consecutive-confirmation tests stay
    green. The same windowed counter replaces the FSM's strict `_provisional_streak`
    reset in `LockStateMachine.step()` (new `provisional_commit_window` ctor arg);
    the per-frame bar (`high_bar`, depth, distinctiveness) is unchanged.
  - **Untouched precision guards:** `high_bar=0.72`, deep-ratio `0.92`,
    distinctiveness `0.10/0.15`, `MIN_REID_SIMILARITY_RAW=0.40`, color vetoes, and
    `DEEP_CONFIDENT_BYPASS` — all unchanged. `NEEDS_HELP` fires at
    `time_since_lost >= active_help_after_sec` (wall-clock; see 2026-06-10);
    the help-hold latch (`_help_latched`)
    still clears only on a true re-lock (pursuit frames are `target_lost=True`).
    Multi-person commit bar stays `reid_threshold` (N-of-M only adds dip-tolerant
    recall there, never lowers the bar). All four new knobs are ROS params on
    `person_track_server` with tracker defaults so unit tests work without the node.

- **2026-06-09** — gate the reseed re-lock behind a short ReID confirmation
  (manual dashboard click AND waving auto-reseed — both share the
  `~/reseed_target` service). Previously `_apply_reseed` matched the requested
  bbox by IoU only and immediately set `state=TRACKING` + `lock_state_machine.start()`,
  so a reseed re-locked on a single geometric frame with no appearance check — a
  box overlapping a bystander, or a slightly-off click, could lock the wrong
  person. Now a reseed enters a **probation** (new param
  `reseed_confirmation_frames`, default **5**): `_apply_reseed` still re-locks the
  ids and appends the fresh crop to the gallery, but sets `state=REIDENTIFYING`
  and re-arms the FSM via the new `LockStateMachine.start_probation()` (enters
  `reidentifying`, not `tracking`; also lifts a terminal `lost`). A per-frame gate
  (`tracking_pipeline._step_reseed_probation`, run BEFORE `track_by_id` so
  ByteTrack can't instant-lock the seeded id) requires the seeded id to be
  **present** (matched by ByteTrack) AND **ReID-confirmed** (`sim >= reid_threshold`,
  scored against the gallery that now includes the fresh crop) for 5 **consecutive**
  frames before committing the lock (`target_lost` flips False → GREEN). A
  present-but-unconfirmed frame **resets** the streak; an **absent** seeded-id
  frame **abandons** probation (falls back to normal recovery). Selection stays
  geometric (IoU); the gate only **adds** an appearance confirmation — strictly
  stricter than before, no threshold lowered. During probation the tracker reports
  `target_lost=True` (YELLOW via the prior viz fix), so the help-hold latch
  (`_help_latched`, clears only on a true re-lock) correctly persists until a real
  commit. The reseed service still returns the seeded tid — it now means
  "accepted, confirming", not "locked".

- **2026-06-09** — fix stuck-yellow bounding box after ReID reacquire. After any
  reacquisition the live ByteTrack id (`target_track_id`) diverges from the frozen
  `original_track_id` stored in `target_result.track_id`. The old color decision
  compared `track_id == target_result.track_id`, so the matched detection failed the
  green test and fell through to yellow. The fix: refactor the per-box color decision
  into `_target_box_color_kind()` which gates on the LIVE id (`target_track_id`) AND
  the FSM state (`last_lock_decision.state == 'tracking'`). A fully-locked target is
  now drawn GREEN regardless of id-space divergence; yellow is reserved for the
  reidentifying coast; blue for unrelated detections. Visualization only — no
  behavior/threshold/feedback changes.

- **2026-06-09** — fix the laggy web feed introduced by the deep-crop
  segmentation. `_segment_crop_for_reid` ran `cv2.GaussianBlur` on full-resolution
  crops ~2× per person per frame, dropping the tracking loop (and thus the
  dashboard MJPEG) to ~13 Hz (pipeline 56 ms). Two fixes: (1) do the
  dilate/blur at the fixed OSNet input size (128×256) — the model resizes there
  anyway, so it's equivalent output at constant ~0.3 ms cost; (2) stop the batch
  path from running a redundant per-detection deep forward whose result was
  discarded (`extract_features(..., compute_reid=False)` — the batch forward is
  the single pass it was always meant to be). Pipeline 56 ms → ~15 ms; loop back
  to camera-bound. Row-equivalence + reacquisition re-verified on the bag.
- **2026-06-09** — transparent (person-only) gallery crops. Gallery view
  thumbnails are now built as BGRA with the seg mask as alpha and tight-cropped
  to the person (transparent background), published as PNG to the dashboard
  (was opaque JPEG) and written to the `vision_log` run dir as transparent PNGs
  when crops + logging are enabled. Reinforces the deep-crop segmentation: the
  operator sees exactly the person the gallery stores, with no bystander/
  background. Visualization/telemetry only (gated by `gallery_keep_crops`,
  default off); does not feed scoring.
- **2026-06-09** — deep-gate the hard color veto (ReID was too strict). The
  body/upper/lower color-histogram floors (0.40) used to `return 0.0` outright,
  force-zeroing even a 0.9 deep match the instant a color score dipped — the top
  cause of a correct (yellow) target never re-locking (going green) under
  lighting/pose change. The color veto now bypasses when the deep cosine is
  confident (`DEEP_CONFIDENT_BYPASS = 0.70`, well above the bystander deep band
  ~0.47–0.57). Bystanders still have low deep, so the veto still rejects them;
  the raw-deep floor, distinctiveness margin, and deep-ratio gates are unchanged.
- **2026-06-09** — person-segment the deep ReID crop. The deep OSNet embedding
  was extracted from the raw bounding-box crop, so the gallery baked in
  background and any bystander sharing the box — which both let the tracker lock
  onto another person and depressed the same-person cosine when the scene
  changed (forcing over-strict thresholds). The deep crop is now segmented:
  dilate the YOLO-seg mask, tight-crop to it (evicts a co-bbox bystander +
  re-centers the person for the 128×256 resize), and soft-blur the background
  (OSNet takes RGB, so this is the equivalent of a transparent background).
  Applied identically on the single + batched deep paths (row-equivalence
  preserved); no thresholds changed.
- **2026-06-09** — latch the help-hold so auto-reclaim isn't aborted
  mid-reappearance. Found while bag-testing the passive-reacquire fix: when the
  operator reappears, the pre-commit re-ID resets `frames_lost` to 0 on every
  confirmation frame (before the ~12-frame re-lock commits). The hold/abort
  decision keyed on the instantaneous `frames_lost`, so that reset flipped
  `_is_awaiting_help` False and the hard-lost abort fired — killing the goal
  before the reclaim could complete. `_is_awaiting_help` now **latches** the
  escalation (`_help_latched`); the latch clears only on a **true re-lock**
  (`target_lost` False) or goal cleanup — NOT on a provisional/partial recovery
  frame, which surfaces with `target_lost` still True while the FSM is `'lost'`
  (clearing it there would re-open the abort). So each loss→reclaim cycle gets a
  fresh hold and repeated reclaims work. Bounded-timeout mode still honors its
  time limit.
- **2026-06-08** — fix single-waver auto-reseed failing with "no camera frame
  available". `_reseed_callback` fetched the frame with the consuming
  `_get_latest_data()`, which returns the `False` dedup sentinel when the
  concurrent tracking loop has already claimed the current frame-seq (a race on
  the shared `last_processed_seq`) — so the reseed was rejected despite a cached
  frame. `_get_latest_data` now takes `consume=False`; the reseed reads the
  latest cached frame non-consuming (mirrors the idle telemetry tick) and never
  races the loop. The 👋 wave→reseed loop now re-locks reliably.
- **2026-06-08** — passive reacquisition is now always-on during the help hold.
  Previously, once the lock FSM latched terminal `'lost'` (operator gone >
  `max_recovery_frames`), a reappearing operator could only re-lock via a wave —
  the FSM short-circuited `'lost'` and squashed even a committed passive ReID
  re-lock to `target_lost=True`. Now a committed passive swap re-arms the FSM
  (`fsm.start`, mirroring reseed), so the operator walking back into frame
  auto-re-locks after the normal 12-frame ReID confirmation, no wave needed.
  No thresholds changed; wave/reseed remains the manual fallback.
- **2026-06-08** — harden the RGB+depth synchronizer (reduce stall frequency).
  The two `message_filters` subscribers now use QoS history `depth=5` (was 1) and
  a dedicated `ReentrantCallbackGroup` so RGB and depth are delivered
  concurrently instead of serialized on the node default group — both shrink the
  window in which one half of a pair is dropped and the `ApproximateTimeSynchronizer`
  stalls. `BEST_EFFORT` is kept (the camera publishes BEST_EFFORT). Pairs with the
  frame-starvation watchdog below as defense-in-depth.
- **2026-06-08** — frame-starvation watchdog. The tracking loop's only frame
  source is the RGB+depth `ApproximateTimeSynchronizer`; when it stops emitting
  matched pairs (a sync stall — root cause of "stopped getting new camera frames"
  while the camera still publishes at 30 Hz) the loop used to busy-wait forever
  with a frozen dashboard, no loss handling, and no log. Now, after
  `frame_stall_warn_sec` (0.5 s) it warns + keeps the dashboard alive
  (`camera_stalled` banner + last good frame); after `frame_stall_lost_sec`
  (1.5 s) it engages the existing loss/recovery FSM (forever-hold + wave/reseed)
  so a camera stall recovers like a person-lost.
- **2026-06-07** — `track_web` dashboard + active-reID test bench shipped: live
  MJPEG/WebSocket tracking view, re-ID gallery thumbnails, bench/observer modes,
  click-to-reseed (incl. the 👋 DetectWaving human-as-BT loop). Backed by
  param-gated tracker telemetry (`debug_state_enabled`, `gallery_keep_crops`,
  `debug_image_enabled`) — **all default off, no production-run impact**.
- **2026-06-07** — added `track_web_bench.launch.py`: one-command bench bringup
  (tracker w/ telemetry ON + waving server + dashboard), `bind`/`port`/
  `with_waving` launch args; cameras separate per CAMERA_BRINGUP.md.
- **2026-06-07** — idle/init camera preview in the dashboard (`IDLE` /
  `INITIALIZING` badge; live raw frame before a goal + during init search,
  replaced by the overlay on lock) + `rgb8`→BGR color normalization at the
  tracker's single decode point. The live Orbbec publishes `rgb8`, not the
  assumed `bgr8`, so colors were red/blue-swapped before — including the
  tracker feed, gallery thumbs, debug overlay/feedback, and the on-disk
  `vision_log` images.