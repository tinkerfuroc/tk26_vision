# `track_web` — live tracking dashboard + active-reID test bench (design)

**Date:** 2026-06-07 · **Status:** approved (brainstorm w/ user; architecture A, layout 1)
**Depends on:** Spec A (multi-view gallery) + Spec B (active re-ID interface), both merged on `dev`.

## Goal

A web dashboard that visualizes the `vision_track` person-tracker's live state and
doubles as the **human-in-the-loop test bench** for Spec B's deferred on-robot
validation: an operator can start/stop tracking, watch the lock/reacquisition
state, and play the BT's role in the call-out → raise-hand → re-seed loop by
clicking a person — with **no behaviour-tree code required**.

## Decisions (from brainstorm)

| Question | Decision |
|---|---|
| Purpose | **Live dashboard** (offline replay is a non-goal for v1) |
| Interactivity | **Full test bench**: start/stop `TrackPerson` goal, click-to-reseed, `DetectWaving` trigger |
| State depth | **Internals + gallery view crops** (param-gated tracker instrumentation) |
| Architecture | **House pattern**: `track_web` ROS2 node + FastAPI, mirroring `pan_tilt/calib_web.py` |
| Layout | **Video-dominant + right rail** |

## §1 Tracker instrumentation (`vision_track`, all default-OFF, zero production impact)

`person_track_node` gains three param-gated debug outputs so the dashboard has
**one data path regardless of who holds the goal** (our bench or a real BT):

1. **`~/debug_state`** (`std_msgs/String`, JSON, every loop frame while a goal is
   active) — gated by `debug_state_enabled` (bool, default `false`). Built by a
   **pure function** `build_debug_state(...)` (new `core/debug_state.py`),
   unit-testable with a bare tracker. Keys:
   `ts`, `fsm_state` (`last_lock_decision.state` or `null`), `target_lost`,
   `reacquisition_state` (0/1/2), `frames_lost`, `time_since_seen`,
   `awaiting_help` (hold active), `active_help_after_frames`,
   `active_help_timeout_sec` (so the UI can render the hold countdown),
   `target_track_id`, `original_track_id`,
   `candidates` (`[{id, bbox:[x1,y1,x2,y2], score}]` from `last_results` persons),
   `best_sim` / `second_sim`, `gallery_len`, `gallery_version`.
   - **Score source (pinned):** the tracker stashes the most recent
     per-candidate similarity map as `self.last_debug_scores: dict[int, float]`
     (written where similarities are already computed — `reid_search`
     `_score_candidates` and the periodic verification path; plain assignment,
     no extra compute). `build_debug_state` joins it on `track_id`; `score`,
     `best_sim`, `second_sim` are `null` on frames where no similarity was
     computed.
2. **`~/debug_gallery`** (`std_msgs/String`, JSON, published **only when
   `gallery_version` changes**) — gated by `gallery_keep_crops` (bool, default
   `false`). Payload: `{version, thumbs: [base64 JPEG, …]}` (index-aligned with
   gallery views; anchor first).
   - `ReIDGallery.maybe_add()` gains an optional **`thumb=None`** kwarg. Thumbs
     are stored/evicted in **lockstep** with views; the payload is **opaque** to
     `ReIDGallery` (pure module stays cv2-free). A `version` counter increments
     on every accepted add/eviction/clear.
   - `appearance_manager` passes a crop (from `frame` + `result.bbox`, resized to
     height ≤192) only when the param is on; the node JPEG/base64-encodes at
     publish time. Memory: ≤ K=6 small crops (~1 MB).
3. **`~/debug_image`** (`sensor_msgs/Image`, bgr8, annotated via the existing
   `_draw_debug_info`) — gated by `debug_image_enabled` (bool, default `false`)
   **and** `count_subscribers > 0` (no drawing cost when nobody watches).
   Independent of the goal's `debug` flag.

All three params mirror into `config/default.yaml` (commented, default off).
Topic names resolve under the tracker node (default
`/person_track_node/debug_state` etc.).

## §2 `track_web` server (new `vision_track/vision_track/track_web.py` + `vision_track/webui/`)

One ROS2 node; `rclpy.spin` in the main thread, uvicorn in a worker thread,
shared state behind a lock — the `calib_web` threading model. Params: `bind`
(default `127.0.0.1`), `port` (default `8766`), `tracker_node_name` (default
`person_track_node`), `waving_service` (default `detect_waving_persons`).

ROS side: subscribers to the three debug topics; `TrackPerson` **action client**
(`track_person`) used *only* to start/stop a bench goal (image-return flags
`false` — video comes from `~/debug_image`); service clients for
`<tracker>/reseed_target` (`ReseedTarget`) and `detect_waving_persons`
(`DetectWaving`).

HTTP surface:

| Endpoint | Method | Behaviour |
|---|---|---|
| `/` + static | GET | `webui/` (installed via `data_files` glob, pan_tilt pattern) |
| `/api/status` | GET | snapshot: data freshness, goal ownership, last state, gallery version |
| `/ws/state` | WS | pushes each `debug_state` JSON + gallery payloads on change + server events |
| `/stream.mjpg` | GET | MJPEG (`multipart/x-mixed-replace`) of latest `~/debug_image`, ~15 fps cap, JPEG q80 |
| `/api/goal/start` | POST | send bench `TrackPerson` goal; surfaces REJECT (goal held elsewhere) |
| `/api/goal/stop` | POST | cancel the bench goal (no-op + message if we don't hold one) |
| `/api/reseed` | POST | `{bbox:[x1,y1,x2,y2]}` → `ReseedTarget` (frame_id from last `~/debug_image` header); returns the service response |
| `/api/wave` | POST | `DetectWaving` call → `{waving_persons, waving_boxes}` for UI overlay |

**Observer mode:** fresh `debug_state` arriving while we hold no goal ⇒ another
client (BT) is tracking — the UI shows everything, disables Start, and labels
the session "observer". A start attempt that gets REJECTed reports the same.

## §3 Web UI (vanilla JS, layout 1 — video-dominant + right rail)

- **Video pane (left):** `<img src=/stream.mjpg>` with a click handler. A click
  is hit-tested against the **current `candidates` boxes** (scaled to display
  size); the smallest-area box containing the point wins (overlap rule) →
  `POST /api/reseed`. After `POST /api/wave`, returned `waving_boxes` render as
  clickable overlays (distinct color); clicking one re-seeds with that box —
  the full Spec-B loop with a human as the BT.
- **Right rail:** color-coded `reacquisition_state` badge (TRACKING green /
  PASSIVE amber / NEEDS_HELP red), FSM state, target/original ids,
  `frames_lost`, **hold countdown** (`active_help_timeout − time_since_seen`
  while NEEDS_HELP), best/2nd similarity, gallery thumb strip (with version).
- **Bottom bar:** Start / Stop / Wave buttons (+ clear-overlays); event log
  derived client-side from transitions (`target_lost` edges, reacq changes,
  reseed/wave responses, goal results), newest first.
- No build step; `index.html` + `app.js` + `style.css`, mirroring `pan_tilt/webui`.

## §4 Error handling

- **Stale banner** when no `debug_state` for >1 s (shows last-message age).
- Goal REJECT, `reseed success=false`, `DetectWaving status≠0` → toast + log
  entry (never silent).
- WS reconnect with backoff; MJPEG `<img>` retried on error.
- Every ROS interaction in HTTP handlers is try/except'd into a JSON error
  response — the server must never crash (node-side reseed is already
  exception-safe from Spec B).

## §5 Testing

- **Pure units:** `ReIDGallery` thumb lockstep (add/evict/clear + version);
  `build_debug_state` with a bare tracker (pattern: `test_reseed_target.py`).
- **Server:** FastAPI `TestClient` against a faked ROS layer — endpoints,
  observer-mode logic, MJPEG header shape (precedent: `test_calib_web_prune.py`).
- **Static import checks** (sourced workspace), like Spec B Tasks 4/5.
- **Live tiers:** camera-in-the-loop (T2) and the on-robot bench loop
  (start → wave → click → reseed → TRACKING) are deferred to operator sessions
  and recorded in `DEV_NOTES.md` — same discipline as Spec B.

## §6 Deployment

- New console script: `track_web = vision_track.track_web:main`.
- Run: `ros2 run vision_track track_web --ros-args -p bind:=0.0.0.0 -p port:=8766`
  (bind non-loopback for an operator laptop on the robot LAN).
- Deps: `fastapi`, `uvicorn` (pan_tilt precedent; install into
  `.venv-vision-main`; add to vision_track `requirements.txt`).
- README + changelog entries for `vision_track` in the same commits as code.

## Non-goals (v1)

Offline replay/benchmark browser · authentication (LAN-trusted; default bind is
loopback) · recording/export · multi-robot views · any BT integration (the BT
remains the production consumer; this bench only emulates it).

## Acceptance criteria

1. With all debug params **off** (defaults), `person_track_node` behaviour and
   published topics are unchanged (production unaffected).
2. With params on + cameras (T2): dashboard shows live annotated video, state
   rail, and gallery strip; Start/Stop drives the goal; Wave overlays boxes;
   clicking a person/wave box calls `ReseedTarget` and the UI reflects the
   result; reacq badge transitions TRACKING → PASSIVE → NEEDS_HELP → (reseed) →
   TRACKING.
3. All pure-unit + server tests green; no NEW flake8 (99-char) errors.

## §7 Bench launch file (addendum, 2026-06-07)

`launch/track_web_bench.launch.py` — one-command bench bringup: ①
`person_track_server` with `parameters=[<share>/config/default.yaml,
{debug_state_enabled: True, gallery_keep_crops: True, debug_image_enabled:
True}]` (yaml first, bench overrides win); ② `waving_person_server`, gated by
`with_waving` (default true); ③ `track_web` with `bind`/`port` launch args
(defaults `0.0.0.0`/`8766`). Cameras deliberately excluded (CAMERA_BRINGUP.md
sequence). Launch-time resolution via `FindPackageShare` substitutions keeps
`generate_launch_description()` pure (unit-testable unsourced); `port` passes
through `ParameterValue(value_type=int)`. Installed via a `data_files` glob.
