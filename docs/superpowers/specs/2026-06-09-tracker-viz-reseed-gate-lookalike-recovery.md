# Tracker: yellow-box fix, reseed confirmation gate, look-alike recovery — design

**Date:** 2026-06-09
**Package:** `src/vision_track`
**Branch:** `feat/track-web-idle-video`
**Status:** approved (user, 2026-06-09 — "Proceed, drop Option D")

Three independent changes to the `person_track_server` tracker, each root-caused
by a separate read-only investigation. They compose: after the fixes a reclaim
shows YELLOW (pursuing / confirming) and flips GREEN only on a real committed
lock.

---

## Issue 1 — yellow bbox persists during the TRACKING state (visualization only)

### Root cause
`_draw_debug_info` (`person_track_node.py:686-687`) decides GREEN with
`is_target = (target_result is not None and track_id == target_result.track_id)`.
But `target_result` is the value returned by `tracker.update()`, whose
`track_id` is rewritten to the **frozen `original_track_id`** by
`_with_original_id()` (`yolo_tracker.py:955-975`). The per-box `track_id`s in
`last_results` are the **live ByteTrack ids** (`core/result_parser.py:45`). After
any ReID reacquire the live id (`target_track_id`) diverges from
`original_track_id` (writers: `core/tracking_pipeline.py:294,513`), so the
genuinely-matched detection fails the green test and falls through to YELLOW
(`is_yolo_target`). The lock FSM state is never consulted in the draw.

The dashboard path is `_publish_debug_outputs` → `_draw_debug_info`
(`person_track_node.py:1447-1449`), feeding `/{tracker}/debug_image` →
`track_web`. Same bug at the `debug_mode` call site (`:1114-1119`). The lost-loop
call (`:1263-1268`) passes `target_result=None` and is already correct.

### Fix (pure visualization; no behavior change)
In `_draw_debug_info`, decide color from the **live** id + the FSM state:

```python
decision = getattr(self.tracker, 'last_lock_decision', None)
fsm_tracking = (getattr(decision, 'state', None) == 'tracking')
is_target = (
    target_result is not None
    and target_track_id is not None
    and track_id == target_track_id      # live id space, not original_track_id
    and fsm_tracking                      # GREEN only when fully committed
)
is_yolo_target = (track_id == target_track_id) and not is_target  # YELLOW: id, not committed
```

- GREEN = the locked target this frame (FSM `tracking`).
- YELLOW = it carries the target's id but is not committed this frame (passive
  coast / reidentifying / reseed probation).
- BLUE = other persons.
- `last_lock_decision` read-only via `getattr` with `None` fallback. No
  thresholds, no FSM transitions, no id writes.

### Invariants
- No change to any tracking decision, threshold, or published feedback.
- Lost-loop call site (`target_result=None`) stays all-yellow (unchanged).
- Side benefit: box color now agrees with the dashboard's textual FSM state
  (`debug_state.py:56` already reads `decision.state`).

---

## Issue 2 — gate the reseed re-lock with a short ReID confirmation

Applies to **both** reseed triggers, which share one service path
`~/reseed_target` → `_reseed_callback` → `reseed_target` → `_apply_reseed`:
- **manual click** (dashboard `/api/reseed`), and
- **waving auto-reseed** (`track_web.py:262-267`: an unambiguous single waver
  auto-reseeds via `self.reseed(boxes[0])`; multi-waver → human clicks → same
  path).

### Root cause
`_apply_reseed` (`yolo_tracker.py:555-580`) matches the requested bbox by **IoU
only** (`_find_best_match_iou`, ≥0.3 — no ReID check) and immediately sets
`state=TRACKING`, `frames_lost=0`, and calls `lock_state_machine.start()`
(instant jump to `tracking`). On the next loop frame the node's `target_present`
check (`person_track_node.py:1066-1077`) reports `target_lost=False`. So a reseed
re-locks on a **single geometric frame**, with no appearance confirmation — a box
that overlaps a bystander, or a slightly-off click, locks the wrong person.

### Fix — short reseed probation (default 5 frames, consecutive)
Reseed becomes "seed the candidate, then confirm over N frames before
committing". New param `reseed_confirmation_frames` (default **5**).

Mechanism:
1. `_apply_reseed` still sets `target_track_id`/`original_track_id`/class, clears
   occlusion bookkeeping, and appends the fresh crop to the gallery — but it does
   **NOT** set `state=TRACKING` and does **NOT** call
   `lock_state_machine.start()`. Instead it enters a **reseed-probation**:
   record `reseed_probation_id = target_track_id`,
   `reseed_probation_count = 0`, set `state=REIDENTIFYING`, and re-arm the FSM
   into a **probationary** state (see below) rather than `tracking`.
2. Each subsequent tracking-loop frame, while probation is active: the seeded id
   must be **present** (matched by ByteTrack) **and** ReID-confirmed
   (`similarity >= reid_threshold`, computed against the gallery that now
   includes the fresh crop). A qualifying frame increments
   `reseed_probation_count`; a non-qualifying frame (present but ReID < threshold)
   **resets** the count to 0. If the seeded id is **absent** for the frame,
   probation is abandoned (fall back to the normal lost/recovery path).
3. When `reseed_probation_count >= reseed_confirmation_frames`, commit: call
   `lock_state_machine.start(original_track_id)` (or step present=True),
   `state=TRACKING`, clear probation. Now `target_lost=False` (GREEN).
4. During probation, the tracker reports `target_lost=True` (YELLOW via Issue 1)
   and `reacq_state` stays PASSIVE/NEEDS_HELP — it has not re-locked yet.

### FSM support
Add `LockStateMachine.start_probation(committed_id)`:
sets `_committed_id`, `_state = "reidentifying"`, `_provisional_streak = 0`.
Unlike `start()` (which jumps to `tracking`), this leaves the machine in a
non-committed state so the node does not report a lock until the probation
commits. This also lifts the FSM out of a terminal `'lost'` so it can be stepped
(mirrors the `f76e6ad` passive-reacq re-arm, but probationary).

### Invariants / risks
- Reseed match is still **geometric (IoU)** for *selection*; the new gate adds
  **appearance** confirmation before the lock — strictly stricter than today.
- Consecutive (not N-of-M): a deliberate human action over a short window; a
  miss resets. Keep it short (5) so a real reclaim still feels responsive.
- Must NOT clear `_help_latched` during probation: it clears only on a true
  re-lock (`feedback.target_lost==False`, `person_track_node.py:1178-1179`).
  Route reseed through the same code path; probation frames surface
  `target_lost=True` so the latch correctly persists.
- If probation never completes (operator turns away / wrong box), the tracker
  stays in its prior lost/NEEDS_HELP state — no false lock. Acceptable: the
  human can re-wave / re-click.

---

## Issue 3 — do not give up on look-alikes during passive reacquisition

Scope: the **passive** recovery path only (operator returns without a wave).
Options **B + A** (Option D — relaxing the color veto — was explicitly dropped;
`reid/reid.py` vetoes and `DEEP_CONFIDENT_BYPASS=0.70` stay untouched).

### Root cause (dominant give-ups)
For a lone returning operator scoring ReID ~0.55–0.71 with occasional dips:
- **#3 single-candidate wall:** `_single_candidate_guard`
  (`reid/reid_search.py:422-438`) requires `>= 0.72` for a lone person, so
  `find_best_match_reid` returns `None` on every sub-0.72 frame — the candidate
  is discarded before it can accumulate anything.
- **#1 strict-consecutive reset:** the confirmation streak is wiped by any dip
  (the `find_best_match_reid → None` path resets
  `reid_fit_streak`/`pending_reid_match`/`consecutive_reid_frames`,
  `tracking_pipeline.py:384-392`; and the FSM resets `_provisional_streak=0`,
  `lock_state_machine.py:101`). 12 unbroken ≥0.72 frames are essentially never
  achievable for a real returner → the correct (yellow) person never goes green.

### CRITICAL precision invariant
Lowering the single-candidate *pursue* floor must **NOT** lower the lone-candidate
*commit* bar. Today, once `find_best_match_reid` returns a lone candidate,
`_confirm_reid_candidate` commits the id-swap after its ramp at merely
`reid_threshold` (0.55) and then force-arms the FSM present=True
(`tracking_pipeline.py:427-444`), bypassing the FSM's 0.72 gate. So naively
lowering `_single_candidate_guard` to 0.55 would silently let a **lone bystander
commit at 0.55**. The fix must keep the lone **commit** bar at the high bar
(0.72) while only relaxing *pursuit* and *dip tolerance*.

### Fix — Option B (pursue) + Option A (N-of-M), with the commit bar held high

**B — pursue floor (`reid_search.py`):** new param
`single_person_pursue_floor` (default **0.55** = `reid_threshold`).
`_single_candidate_guard` returns True (candidate kept) for a lone person whose
similarity is in `[single_person_pursue_floor, 0.72)`, instead of discarding it.
Below the pursue floor → still discard. This keeps the look-alike *in play* so
its good frames can accumulate; it does **not** by itself authorize a lock.

**Commit-bar safeguard (`tracking_pipeline.py:_confirm_reid_candidate`):**
introduce a per-frame **commit bar**:
`commit_bar = single_person_commit_bar (default 0.72) if num_candidates == 1
else reid_threshold`. A frame counts as a **confirm hit** only when
`match_similarity >= commit_bar`. `num_candidates` is passed in from
`reidentify_target` (it already computes `num_cands`). This preserves today's
effective bars: lone = 0.72, multi = `reid_threshold` (+ existing
distinctiveness/margin gates upstream in `find_best_match_reid`, unchanged).

**A — N-of-M confirmation (`tracking_pipeline.py`):** replace the
strict-consecutive `consecutive_reid_frames` commit with a sliding window: commit
when there are `>= reid_confirmation_frames` (N, default 12) confirm-hits within
the last `provisional_commit_window` (M, default **18**) frames in which the
pending id stayed the best candidate. A non-hit frame (sim in
`[pursue_floor, commit_bar)`) **keeps the pending alive** (does not zero it) but
does not count toward commit. The pending is abandoned only if the best
candidate's **id changes** or there are **no hits within the window** (stale).

Apply the same dip tolerance to the FSM provisional streak (Option A in the FSM):
`lock_state_machine.step()` commits the provisional when `>= commit_frames`
clear-bar frames occurred within the last `provisional_commit_window` frames,
instead of strict-consecutive (replace the `_provisional_streak = 0` reset at
`:101` with a windowed counter). Per-frame bar (`high_bar`, depth, distinctness)
unchanged.

### What this achieves
- The lone look-alike at 0.55–0.71 is **pursued** (surfaced as `reidentifying`,
  `target_lost=True` → YELLOW) instead of dropped — "do not give up".
- The frames where the returner **does** clear 0.72 accumulate across dips
  (N-of-M), so they eventually reach the commit and go GREEN.
- A lone candidate that **never** clears 0.72 is pursued but **never committed**
  → no wrong-person lock. Precision preserved.

### Invariants / untouched precision guards
- `high_bar=0.72`, deep-ratio `0.92`, distinctiveness margin `0.10`/`0.15`,
  `MIN_REID_SIMILARITY_RAW=0.40`, color vetoes, `DEEP_CONFIDENT_BYPASS=0.70` —
  ALL unchanged.
- `NEEDS_HELP` still fires at `frames_lost >= active_help_after_frames` (45)
  because pursuit keeps `target_lost=True` (`reacq_state` unchanged).
- Help-hold latch (`_help_latched`) still clears only on a true re-lock —
  pursuit frames are `target_lost=True`, so the latch persists.
- Multi-person commit bar stays `reid_threshold` (no behavior change there beyond
  N-of-M dip tolerance, which only *adds* recall, never lowers the bar).

---

## New ROS parameters (all on `person_track_server`)

| Param | Default | Meaning |
|---|---|---|
| `reseed_confirmation_frames` | `5` | consecutive present+ReID-confirmed frames a reseed (manual/waving) must hold before committing the re-lock |
| `single_person_pursue_floor` | `0.55` | lone-candidate similarity floor to *pursue* (not discard); below this still discarded |
| `single_person_commit_bar` | `0.72` | lone-candidate similarity bar to *commit* a re-lock (held high; was the hard-coded `_single_candidate_guard` 0.72) |
| `provisional_commit_window` | `18` | N-of-M window (M); commit needs `reid_confirmation_frames` (N) hits within the last M frames |

`reid_confirmation_frames` (existing, default 12) is reused as N. No existing
threshold is changed.

---

## Testing (TDD, pure-logic where possible)

- **Issue 1:** unit-test the color decision (extract the green/yellow/blue rule
  or test `_draw_debug_info` with a `SimpleNamespace` tracker): live-id-matched +
  `decision.state=='tracking'` → green; live-id-matched + `state=='reidentifying'`
  → yellow; `original_track_id != target_track_id` no longer forces yellow when
  committed. Extend/add near `test_debug_state.py`.
- **Issue 2:** extend `test_reseed_target.py` + `test_wave_auto_reseed.py`:
  reseed does not report `target_lost=False` until `reseed_confirmation_frames`
  present+confirmed frames; a sub-threshold frame resets; an absent frame
  abandons; `_help_latched` not cleared during probation. Add
  `start_probation()` unit tests in `test_lock_state_machine.py`.
- **Issue 3:** extend `test_lock_state_machine.py` (N-of-M window: 12-of-18
  commits, 11 hits + 7 misses does not; strict bystander 5-frame spike never
  commits) and add a pipeline test (lone candidate at 0.6 is pursued — returned,
  `target_lost=True` — but never commits; lone candidate hitting ≥0.72 on
  12-of-18 frames commits). Reuse the `SimpleNamespace` tracker pattern from
  `test_passive_reacq.py`.
- Full suite must stay green; flake8/pep257/copyright tests must not regress
  (pre-existing flake8 count is the baseline, do not add new violations in
  touched files).

## Files touched
- `vision_track/person_track_node.py` — Issue 1 draw; Issue 2 reseed callback /
  probation wiring + param declares; Issue 3 param declares + pass-through.
- `vision_track/yolo_tracker.py` — Issue 2 `_apply_reseed`/`reseed_target`
  probation; param plumbing.
- `vision_track/core/lock_state_machine.py` — `start_probation()`; N-of-M streak.
- `vision_track/core/tracking_pipeline.py` — Issue 2 probation step; Issue 3
  commit-bar + N-of-M in `_confirm_reid_candidate`; pass `num_candidates`.
- `vision_track/reid/reid_search.py` — Issue 3 pursue floor in
  `_single_candidate_guard`.
- `test/…` — new/extended tests as above.
- `readme.md` Changelog + `DEV_NOTES.md` — same-commit doc discipline.
