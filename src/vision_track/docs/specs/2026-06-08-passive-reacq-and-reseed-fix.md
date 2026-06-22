# Always-on passive reacquisition + single-waver reseed fix — design

**Date:** 2026-06-08
**Scope:** `vision_track` person tracker — two independent fixes from a live bench session.
**Constraint:** precision is sacred — **no ReID/lock thresholds change.** Both fixes
remove a *dead-end*; they do not relax any matching gate.

---

## Issue 1 — passive reacquisition must always be on (even while awaiting help)

### Observed
While the tracker is in the indefinite NEEDS_HELP hold (target lost > 45 frames,
`active_help_timeout_sec=0`), the operator walking back into frame does **not**
re-lock the tracker. It only re-locks via a wave → reseed.

### Root cause (verified)
The lock FSM (`core/lock_state_machine.py`) latches its terminal `'lost'` state
once `frames_since_loss > max_recovery_frames` (45) — dropping `committed_id`
(`:80-84`) — and from then on `step()` short-circuits on its first line for
every frame, ignoring all inputs:

```python
if self._state == "lost":
    return LockDecision(False, True, None, "lost")   # :70-71
```

The only exit from `'lost'` is `start()` (`:44-48`), which today is called **only**
by init (`_try_initialize`) and by reseed (`_apply_reseed`, `yolo_tracker.py:577-578`).

Meanwhile the appearance/gallery re-ID search **keeps running** the whole hold —
`reidentify_target` is gated by `frames_lost > max_frames_lost` (**600**), not 45
(`core/tracking_pipeline.py:348`; `max_frames_lost` set to `max(600, …)` in
`person_track_node.py`). So every frame the pipeline still:
- runs `find_best_match_reid` against the multi-view gallery, and
- accumulates `_confirm_reid_candidate` toward a committed id-swap.

When the operator reappears and is re-ID'd for the full confirmation window
(`reid_confirmation_frames` = 12, + post-shake extra), `_confirm_reid_candidate`
**commits the id-swap** (`tracking_pipeline.py:498-518`) and `reidentify_target`
sets `committed_swap=True` (`:411`). The committed-swap branch then steps the FSM
`present=True` (`:431-436`) — **but the FSM is in `'lost'`, so `:70-71` squashes
it to `target_lost=True`.** The node reads `feedback.target_lost = decision.target_lost`
(`person_track_node.py:1050`) → `True`, so the re-lock the pipeline already made
internally is never surfaced. The tracker is silently re-locked but reports lost,
and keeps waiting for a wave.

### Why reseed works but passive doesn't
`reseed_target` → `_apply_reseed` calls `lock_state_machine.start(id)`
(`yolo_tracker.py:577-578`), lifting the FSM out of `'lost'`. The passive
committed-swap path never does.

### Fix (surgical, no threshold change)
In `reidentify_target`'s committed-swap branch (`core/tracking_pipeline.py:427`),
**re-arm the FSM before stepping it** — mirror exactly what reseed does:

```python
if committed_swap:
    if fsm is not None:
        # A passive re-ID just re-locked the operator (id-swap committed this
        # frame after the full reid_confirmation_frames window). Re-arm the FSM
        # the same way reseed does (yolo_tracker._apply_reseed) so it leaves the
        # terminal 'lost' state and re-syncs committed_id to the new id —
        # otherwise step() short-circuits 'lost' and the re-lock never surfaces
        # (target_lost stays True forever, waiting for a wave).
        fsm.start(tracker.target_track_id)
        tracker.last_lock_decision = fsm.step(
            sim_score=float(best_similarity), present=True, frames_since_loss=0,
            num_candidates=num_cands, distinct_margin=margin,
            depth_consistent=depth_consistent,
        )
    return confirmed
```

**Precision:** unchanged. The re-lock still requires the existing 12-frame
`_confirm_reid_candidate` window at `reid_threshold` (+ pre-occlusion check + depth
gate + distinctiveness). This fix only lets a *confirmed* passive re-lock take
effect after frame 45 instead of being dead-ended. It also fixes a latent bug
(`committed_id` was never re-synced in the FSM after any id-swap).

**Behavior after fix:** operator reappears → re-ID'd for ~12 consecutive frames →
auto re-lock, no wave needed. Wave/reseed remains as the manual fallback.

### Not doing (deliberate)
We are **not** enabling FAST provisional publish from `'lost'` (the pre-commit
point publish during the 12-frame window). That would publish a target point
before the slow commit completes — a precision trade we don't need. A ~0.4–0.8 s
gap before re-lock surfaces is acceptable and matches the existing
`'reidentifying'` semantics.

---

## Issue 2 — single-waver auto-reseed fails ("no camera frame available")

### Observed
Wave → DetectWaving returns exactly one waver → auto-reseed → **fails**.

### Root cause (verified)
`_reseed_callback` (`person_track_node.py`) fetches the frame via
`self._get_latest_data()` and rejects on a falsy result with "no camera frame
available". But `_get_latest_data()` returns the **`False` sentinel** (not a
frame) whenever `frame_seq == last_processed_seq` — i.e. the current frame was
already consumed. The tracking loop runs concurrently (MultiThreadedExecutor,
both callbacks reentrant) and consumes essentially every frame within sub-ms via
the **shared** `last_processed_seq` token. So when the reseed service fires, the
loop has almost always already claimed the latest `frame_seq` → reseed gets
`False` → rejected, even though a perfectly good frame is cached in
`recent_sync_msg`. It's a race on `last_processed_seq`.

The bbox path itself is correct (same `/camera/color/image_raw`, same resolution,
x1y1x2y2; IoU of the waver box vs the last person box ≈ 0.36 > the 0.3 reseed
threshold) — the reseed never gets far enough to use it.

The author already solved this exact race for the idle telemetry tick, which
reads the cache **without** consuming (never touches `last_processed_seq`,
`person_track_node.py:130-134`). The reseed path was not given the same treatment.

### Fix (mirror the non-consuming pattern)
Add a `consume` flag to `_get_latest_data`; the reseed reads non-consuming:

```python
def _get_latest_data(self, consume: bool = True):
    with self.lock_msg:
        if self.recent_sync_msg is None:
            return None
        current_seq = self.frame_seq
        if consume:
            if current_seq == self.last_processed_seq:
                return False          # loop dedup — only when consuming
            self.last_processed_seq = current_seq
        rgb_msg, depth_msg = self.recent_sync_msg
    with self.lock_info:
        intrinsic = self.camera_intrinsic
    if intrinsic is None:
        return None
    rgb_img, err = decode_color_msg(rgb_msg)
    if rgb_img is None:
        self.get_logger().warn(f'color frame dropped: {err}', throttle_duration_sec=5.0)
        return None
    return rgb_img, rgb_msg, depth_msg, intrinsic
```

`_reseed_callback` calls `self._get_latest_data(consume=False)`. With
`consume=False` the method never returns `False` (no dedup, no token write), so it
never races the loop — it returns the latest cached frame, or `None` only when
there is genuinely no frame/intrinsic. The existing `if not data:` reject then
means what it says. The tracking loop keeps calling `_get_latest_data()`
(consume defaults True) — unchanged behavior.

**Precision/perf:** unchanged. Non-consuming read mirrors the proven idle-tick
pattern; reseed already runs under `lock_tracker`.

---

## Test plan (TDD, unit-level — both logic-isolated like existing tests)

1. **FSM re-arm on committed swap** (`test_passive_reacq.py`): drive a
   `LockStateMachine` into `'lost'` (>45 absent steps), then assert that after
   `start(new_id)` a `present=True` step returns `target_lost=False, state='tracking'`
   — i.e. `'lost'` is escapable via the same call the pipeline will now make.
   (Confirms the FSM contract the fix relies on.)
2. **Non-consuming fetch** (`test_get_latest_data_consume.py`): unbound-method +
   `SimpleNamespace` self (existing pattern). Assert: `consume=True` returns
   `False` on an unchanged `frame_seq` and advances `last_processed_seq`;
   `consume=False` returns the tuple on the **same** `frame_seq` and leaves
   `last_processed_seq` untouched (no race), and returns `None` when
   `recent_sync_msg is None`.
3. Full suite stays green (currently 179 passed / 1 skip, flake8 baseline 534).

## On-robot verification (operator, after build via `tkbuild tk26_vision`)
- Lose the operator > 2 s (NEEDS_HELP hold), have them walk back in → tracker
  auto re-locks within ~1 s **without** a wave.
- Lose the operator, raise hand once → single-waver auto-reseed re-locks (no
  "no camera frame available").

## Build/deploy note
Deploy with **`tkbuild tk26_vision --packages-select vision_track`** (installs to
`/home/tinker/tk25_ws/install`, the tree the bench loads). `scripts/build.sh`
targets `src/tk26_vision/install`, which is NOT on the live path.
