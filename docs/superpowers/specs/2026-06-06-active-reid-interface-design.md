# Active Re-ID Interface (vision-side) — Design (Spec B)

**Status:** approved design, pre-plan
**Date:** 2026-06-06
**Branch:** `feat/person-tracker-overhaul`
**Depends on:** [Spec A — Multi-View ReID Gallery](2026-06-06-multiview-reid-gallery-design.md) (the re-seed refreshes the gallery).
**Related:** memory `vision-track-tracker-gap-analysis`.

## Goal

Give the vision stack the **capability** for *active* re-identification: when
passive reacquisition cannot safely re-lock, the robot can ask the operator to
raise a hand / wait (a RoboCup@Home action that costs points), and the tracker
re-locks on that self-identified person — **precision-safe by construction**.
This spec covers the **vision-side capability only**; the behaviour-tree policy
that decides *when* to call out and accepts the penalty is a separate
tk25_decision effort, defined here as a consumer contract.

## Background & problem

Spec A's multi-view gallery recovers the "easy half" of the recall gap (operator
visible but drifted) silently and precision-safely. The "hard half" — long
absences, genuine ambiguity, appearance drift past any safe threshold — cannot be
recovered passively without lowering thresholds, which violates "precision is
sacred." RoboCup@Home permits an explicit fallback: the robot **calls out** to
the operator to raise a hand or wait, accepting a points penalty, to
re-identify. Because the operator self-identifies, the re-lock is precision-safe
by construction. The vision stack must expose (a) a signal that passive recovery
has failed and active help is warranted, and (b) a trusted re-seed path that
re-locks on the confirmed person without discarding identity continuity.

## Approach

Three vision-side pieces plus a documented BT contract:

1. A **reacquisition-state signal** on the `TrackPerson` action feedback so a
   consumer knows when to escalate.
2. A **gallery-preserving re-seed service** (`ReseedTarget`) that re-locks the
   running tracker on an externally-confirmed target.
3. A **`DetectWaving` → re-seed seam** so the raise-hand detector's output can
   drive the re-seed (resolving the 3D-point vs image-bbox mismatch).
4. A **BT consumer contract** (out of scope to implement) tying it together.

### 1. Reacquisition-state feedback

Add `uint8 reacquisition_state` to `TrackPerson.action` feedback with constants
`TRACKING=0`, `PASSIVE_REACQUIRING=1`, `NEEDS_ACTIVE_HELP=2`.

- The node derives it from the existing `frames_lost` / FSM `target_lost` plus
  hysteresis: enter `PASSIVE_REACQUIRING` on loss; escalate to
  `NEEDS_ACTIVE_HELP` only after `active_help_after_frames` (default
  conservative, penalty-aware — e.g. ~3 s worth of frames) of continuous loss
  with no confident passive candidate; return to `TRACKING` on re-lock.
- This is *advisory*: the tracker never calls out itself. It replaces the silent
  "stay lost forever" with an explicit, debounced request the BT can act on.
- Backward compatible: existing `target_lost` field is unchanged; consumers that
  ignore the new field behave as today.

### 2. `ReseedTarget` service (gallery-preserving re-lock)

New service `tinker_vision_msgs_26/srv/ReseedTarget`:

```
# request
sensor_msgs/RegionOfInterest bbox   # target box in the current color frame
string frame_id                     # color frame the bbox is expressed in (sanity)
---
# response
bool success
int32 target_track_id               # the (re)locked track id, -1 on failure
string message
```

Node handler:

- Match `bbox` to a current YOLO detection (max IoU above a small floor).
- Re-establish the target on that detection: set `target_track_id` /
  `original_track_id`, position, `state = TRACKING`, `frames_lost = 0`.
- **Preserve** the Spec-A multi-view gallery and the person registry (same
  operator → keep accumulated identity), and **append the fresh confirmed view**
  to the gallery (high-trust enrichment). Contrast with `initialize_tracking()`,
  which calls `reset()` and wipes the gallery.
- Re-arm the lock FSM if attached (`lock_state_machine.start(original_track_id)`).
- Idempotent and allowed at any time (also usable as an operator correction
  mid-track, not only after a loss).

### 3. `DetectWaving` → re-seed seam

`DetectWaving.srv` currently returns `waving_persons` as 3D `PointStamped[]` plus
`segments`, but `ReseedTarget` needs an image-space bbox. Add
`sensor_msgs/RegionOfInterest[] waving_boxes` (1:1 with `waving_persons`) to the
`DetectWaving` response so the BT can pass a bbox straight through. Minimal,
additive change; existing fields untouched.

### 4. BT consumer contract (out of scope — tk25_decision)

Documented so the interface is complete; **not built here**:

```
feedback.reacquisition_state == NEEDS_ACTIVE_HELP
  └▶ BT decides escalation is worth the penalty (task-dependent)
       └▶ speak "please raise your hand / please wait"
            └▶ call waving_person_server (DetectWaving)
                 └▶ pick best waving_box
                      └▶ call ReseedTarget(bbox)  ──success──▶ tracking resumes
```

## Data flow

```
lost ─▶ node: frames_lost++ ─▶ feedback.reacquisition_state escalates with hysteresis
                                          │ (NEEDS_ACTIVE_HELP)
                                BT (out of scope): call-out + DetectWaving
                                          │ waving_box
                                          ▼
                                ReseedTarget(bbox) ─▶ match detection ─▶ re-lock,
                                   keep gallery+registry, append fresh view, re-arm FSM
                                          │ success
                                          ▼
                                feedback.reacquisition_state = TRACKING
```

## Error handling & edge cases

- **bbox matches no current detection** (operator moved between detect and
  reseed): return `success=false`, tracker stays in its current state; BT may
  retry. No partial/garbage lock.
- **Re-seed while still tracking** (operator correction): allowed — re-points the
  target cleanly, gallery preserved.
- **Multiple waving persons:** disambiguation is the BT's job (it picks the
  box); the service trusts the single bbox it receives.
- **No gallery yet (Spec A disabled):** re-seed still works; it just seeds the
  appearance from the fresh view, as initialize does.
- **Interface/version:** adding the feedback field + new srv + `DetectWaving`
  field requires rebuilding `tinker_vision_msgs_26` and dependents.

## Testing

- **Unit (pure / node-level with stubs):** `reacquisition_state` hysteresis
  transitions (TRACKING→PASSIVE→NEEDS_ACTIVE_HELP→TRACKING with the frame
  thresholds); `ReseedTarget` handler — bbox→detection matching, gallery &
  registry preserved, fresh view appended, FSM re-armed, failure path on no
  match.
- **Offline (no operator):** drive a recorded/synthetic frame stream, force a
  loss, call `ReseedTarget` with the known operator bbox, assert tracking
  restored and identity continuity (same `original_track_id`, gallery retained).
- **Active end-to-end (deferred to on-robot acceptance):** call-out → raise hand
  → DetectWaving → ReseedTarget loop with a live operator + BT. Documented as the
  acceptance gate; not run this cycle (no robot time).

## Scope

**In:** `reacquisition_state` feedback field + node hysteresis; `ReseedTarget`
srv + handler (gallery-preserving); `DetectWaving` `waving_boxes` field;
unit/offline tests; the BT contract documentation.

**Out:** the BT policy (when to escalate / accept penalty), the audio request,
and on-robot validation — all tk25_decision / hardware, separate effort. Also
out: depth-assisted reacquisition (separate, bag-gated).

## Sequencing

Build **after Spec A** (the re-seed's gallery-refresh and continuity depend on
the gallery existing). Spec A is independently shippable and LaSOT-validated;
Spec B layers the active backstop on top.
