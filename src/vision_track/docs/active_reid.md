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
   is kept and the fresh confirmed view is appended rather than reset (see
   `yolo_tracker._apply_reseed`). Tracking resumes and `reacquisition_state`
   returns to `REACQ_TRACKING` (0) on the next held frame.

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
