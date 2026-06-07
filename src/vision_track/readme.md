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