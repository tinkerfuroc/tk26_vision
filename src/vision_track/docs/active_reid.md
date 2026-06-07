# Active re-ID — behaviour-tree consumer contract

How a `tk25_decision` behaviour tree (BT) wires the active re-acquisition loop on
top of `person_track_server`. This document defines the **consumer contract**;
the BT policy itself (when to call out, how to phrase the request, accepting the
RoboCup points penalty) lives in `tk25_decision` and is **out of scope** for
`vision_track`.

Spec: `docs/superpowers/specs/2026-06-06-active-reid-interface-design.md`
Plan: `docs/superpowers/plans/2026-06-07-active-reid-interface.md`

## Why active re-ID

Passive re-acquisition recovers the **easy half** of target loss — brief
occlusion, the operator stepping out of frame — by holding the lock and matching
against the multi-view ReID gallery (Spec A). It deliberately will **not** lower
its match threshold to chase the **hard half** (long absence, genuine ambiguity,
appearance drift), because doing so risks locking onto the wrong person and
violates "precision over recall."

The active loop covers that hard half **precision-safely**: the robot asks the
operator to self-identify (raise a hand / wait), the waving detector localises
them, and the tracker re-seeds onto that confirmed box. Because the operator
self-identifies, the re-lock cannot silently grab a bystander the way an
automatic recall might — the BT trades a points penalty for a guaranteed-correct
re-lock.

## The `reacquisition_state` feedback signal

`TrackPerson` action feedback carries a `uint8 reacquisition_state` field with
three constants (defined in `action/TrackPerson.action`):

| Value | Constant | Meaning for the BT |
|---|---|---|
| `0` | `REACQ_TRACKING` | Target held this frame. Normal operation; no action. |
| `1` | `REACQ_PASSIVE` | Target lost, but within the passive-recovery window. The tracker is still trying to re-lock on its own — **wait, do not call out yet.** |
| `2` | `REACQ_NEEDS_HELP` | Lost for `>= active_help_after_frames` consecutive frames. Passive recovery has been given its budget and failed — escalation is now warranted. |

The state is pure hysteresis over `(tracked?, consecutive frames lost)` —
see `vision_track/core/reacq_state.py`. The tracker is the publish authority;
`active_help_after_frames` (default **45**, `config/default.yaml`) is the
**escalation debounce**: how many lost frames to spend on passive recovery before
advertising `NEEDS_HELP`. It exists so the BT does not call out (and incur the
penalty) on every momentary occlusion. Tune it up for a more patient robot, down
for a more eager one.

`reacquisition_state` is published from both `_handle_tracked_frame`
(→ `REACQ_TRACKING`) and `_handle_lost_frame` (→ `REACQ_PASSIVE` /
`REACQ_NEEDS_HELP`) on every feedback tick, so the BT reads it from the normal
`TrackPerson` feedback stream — no extra subscription.

## The loop

1. **Watch feedback.** The BT consumes `TrackPerson` feedback. While
   `reacquisition_state` is `REACQ_TRACKING` (0) or `REACQ_PASSIVE` (1), do
   nothing — the tracker is coping.
2. **Escalate on `NEEDS_HELP`.** When `reacquisition_state == REACQ_NEEDS_HELP`
   (2), the target has been lost for `>= active_help_after_frames` (default 45)
   frames. The BT decides escalation is worth the penalty (task-dependent),
   speaks a call-out ("please raise your hand / wait"), and accepts the RoboCup
   points penalty for the assist.
3. **Find the raised hand.** The BT calls the waving detector
   (`DetectWaving` via `waving_person_server`). The response carries
   `geometry_msgs/PointStamped[] waving_persons` (3D points) **1:1** with
   `sensor_msgs/RegionOfInterest[] waving_boxes` (image-space ROIs) — the box is
   the seam the re-seed needs.
4. **Re-seed.** The BT picks the best `waving_boxes[i]` (the self-identified
   operator) and calls the person-track node's `ReseedTarget` service
   (`~/reseed_target`) with `bbox = waving_boxes[i]` and `frame_id` = the color
   frame the box is expressed in. On success the response returns
   `success = true` and `target_track_id` = the (re)locked track id
   (`-1` on failure, with a human-readable `message`).
5. **Resume.** The node re-locks the tracker on that box **preserving the
   multi-view gallery** (Spec A) — same operator, so the accumulated appearance
   is kept and the fresh confirmed view is **appended to the deep gallery**
   rather than reset (see `yolo_tracker._apply_reseed` / `reseed_target`).
   Tracking resumes and `reacquisition_state` returns to `REACQ_TRACKING` (0)
   on the next held frame.

   **Re-seed is gallery-additive only (precision-first).** The re-seed match is
   geometric (IoU) — there is no appearance verification of the supplied box, by
   design (the operator self-identified). For that reason the re-seed appends the
   fresh view to the deep ReID gallery but **does not overwrite the colour /
   identity anchors**: promoting an IoU-only-matched crop to the anchor would let
   a wrong-overlap box poison identity, which "precision is sacred" forbids. The
   consequence to know: under **heavy appearance drift** (e.g. the operator
   removed a jacket) the colour reject-floors keyed on the *old* anchors may
   re-drop the operator shortly after a successful re-seed. The deep gallery's
   max-over-views partially covers this, but full drift recovery is a deliberate
   non-goal here — a repeat call-out/re-seed is the fallback.

```
TrackPerson feedback ──▶ reacquisition_state
   0 TRACKING / 1 PASSIVE ─▶ (wait)
   2 NEEDS_HELP ─▶ BT call-out (penalty) ─▶ DetectWaving
                         │
                         ▼
            waving_boxes[i]  (best self-identified operator)
                         │
                         ▼
            ReseedTarget(~/reseed_target, bbox, frame_id)
                         │
                         ▼
            gallery-preserving re-lock ─▶ back to TRACKING
```

## The hold — keeping the tracker alive for the handoff

The call-out → raise-hand → `DetectWaving` → `ReseedTarget` round-trip takes
*seconds*; the passive-recovery cap (`max_recovery_frames`, default 45 ≈
1.5–3 s) is far too short to span it. Without intervention the action would
hard-abort and `tracker.reset()` would **wipe the gallery** ~1 frame after
`NEEDS_HELP` first appears, so the re-seed would have nothing left to preserve.

The node therefore **holds**: once `reacquisition_state == REACQ_NEEDS_HELP`, it
keeps coasting (publishing `NEEDS_HELP` feedback every frame, tracker + gallery
intact, **no abort/reset**) for up to `active_help_timeout_sec`
(default **20.0 s**, `config/default.yaml`). A successful `ReseedTarget` within
that window re-locks and returns to `REACQ_TRACKING`; if the window expires with
no re-seed, the action aborts (gives up) as before. Set
`active_help_timeout_sec: 0.0` to disable the hold entirely (legacy fast-abort).

> **Give-up timing changed for *all* `TrackPerson` callers, not just active-reID.**
> Because `active_help_after_frames` (45) equals `max_recovery_frames` (45), the
> hold engages on **every** hard loss. A caller that is *not* running an
> active-reID BT (the target genuinely left) now sees the action stay alive up to
> `active_help_timeout_sec` (~20 s) before it aborts, versus ~1.5–3 s previously.
> If a caller needs the old fast failure, lower `active_help_timeout_sec` or set
> it to `0.0` for that node. (Note: a *provisional* re-match resets the
> lost-frame clock, so in a crowd the effective hold can extend toward the
> `lost_timeout` ceiling, default 300 s, before final give-up.)

## Interfaces (exact)

### `TrackPerson.action` feedback (subset)
```
uint8 reacquisition_state
uint8 REACQ_TRACKING=0
uint8 REACQ_PASSIVE=1
uint8 REACQ_NEEDS_HELP=2
```

### `ReseedTarget.srv`
```
sensor_msgs/RegionOfInterest bbox   # target box in the current color frame
string frame_id                     # color frame the bbox is expressed in (sanity/logging)
---
bool success
int32 target_track_id               # the (re)locked track id, -1 on failure
string message
```

### `DetectWaving.srv` response (subset)
```
geometry_msgs/PointStamped[] waving_persons
sensor_msgs/RegionOfInterest[] waving_boxes  # 1:1 with waving_persons; image-space boxes for re-seed
```

## Resolved service name caveat

The re-seed service is registered as `~/reseed_target` — a **private** name, so
it resolves **under the person-track node's namespace/name**, not at the global
root. With the default launch (node name `person_track_node`, no namespace) the
fully-qualified name is `/person_track_node/reseed_target`. If the node is
launched under a namespace or remapped, the BT's service client must target the
correspondingly-resolved name. Confirm the live name with
`ros2 service list | grep reseed_target` before wiring the BT client.
