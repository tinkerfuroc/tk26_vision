# Tracker viz / reseed-gate / look-alike recovery — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Steps use checkbox (`- [ ]`) syntax. TDD throughout. Phase = one git commit.

**Goal:** (1) fix the yellow-box-during-TRACKING overlay; (2) gate both reseed
triggers (manual click + waving auto-reseed) behind a short ReID confirmation;
(3) stop giving up on look-alikes during passive reacquisition — without
lowering any commit bar.

**Spec:** `docs/superpowers/specs/2026-06-09-tracker-viz-reseed-gate-lookalike-recovery.md`

**Architecture:** ROS2 Humble `vision_track` person tracker. Pure-logic FSM
(`core/lock_state_machine.py`) + recovery pipeline (`core/tracking_pipeline.py`)
+ ReID search (`reid/reid_search.py`) + node (`person_track_node.py`) +
`yolo_tracker.py`. Tests are pytest with `SimpleNamespace` mock trackers.

**Tech stack:** Python 3.10, pytest, numpy. Build: `tkbuild tk26_vision
--packages-select vision_track`. Test (from repo root
`/home/tinker/tk25_ws/src/tk26_vision`, venv `.venv-vision-main`):
`source .venv-vision-main/bin/activate && python -m pytest src/vision_track/test/ -x -q`.

**Precision invariant (applies to all phases):** no existing threshold is
lowered. The lone-candidate **commit** bar stays 0.72; only *pursuit* and
*dip tolerance* are relaxed. Reseed becomes *stricter* (adds appearance
confirmation). All four guard thresholds (`high_bar`, deep-ratio `0.92`,
distinctiveness `0.10/0.15`, `MIN_REID_SIMILARITY_RAW 0.40`) and the color
vetoes / `DEEP_CONFIDENT_BYPASS=0.70` are untouched.

---

## Phase 1 — Issue 1: yellow-box visualization fix (one commit)

**Files:** Modify `src/vision_track/vision_track/person_track_node.py` (`_draw_debug_info`, ~686-687).
Test: `src/vision_track/test/test_debug_draw_color.py` (new). Doc: `src/vision_track/readme.md`.

- [ ] **Step 1 — failing test.** Create `test/test_debug_draw_color.py`. Test the
  green/yellow/blue decision. Two clean options: (a) call `_draw_debug_info` on a
  `SimpleNamespace`-backed node instance and assert pixel colors at box corners;
  (b) preferred — refactor the decision into a small pure helper
  `_target_box_color(track_id, target_result, target_track_id, decision)` returning
  `('target'|'yolo_target'|'other')` and unit-test that. Assertions:
  - live id match + `decision.state=='tracking'` → `'target'` (GREEN), even when
    `target_result.track_id (==original_track_id) != target_track_id`.
  - live id match + `decision.state=='reidentifying'` → `'yolo_target'` (YELLOW).
  - live id match + `decision is None` → `'yolo_target'` (YELLOW; strict).
  - non-matching id → `'other'` (BLUE).
  - `target_result is None` (lost loop) + id==target_track_id → `'yolo_target'`.

- [ ] **Step 2 — run, verify fail.** `python -m pytest src/vision_track/test/test_debug_draw_color.py -x -q` → FAIL (helper/behavior absent).

- [ ] **Step 3 — implement.** In `person_track_node.py` replace the color decision
  (lines ~685-694) per spec:
  ```python
  decision = getattr(self.tracker, 'last_lock_decision', None)
  fsm_tracking = (getattr(decision, 'state', None) == 'tracking')
  is_target = (
      target_result is not None
      and target_track_id is not None
      and track_id == target_track_id
      and fsm_tracking
  )
  is_yolo_target = (track_id == target_track_id) and not is_target
  ```
  If using the helper refactor, extract `_target_box_color(...)` and call it.
  Keep label text logic (`(TARGET)`/`(YOLO_TARGET)`) following `is_target`/`is_yolo_target`.

- [ ] **Step 4 — run, verify pass.** New test green; then full suite
  `python -m pytest src/vision_track/test/ -q` → no regressions (note the
  pre-existing flake8 baseline count; do not add new flake8 in touched lines).

- [ ] **Step 5 — README changelog** (same commit). Prepend a `readme.md`
  Changelog entry (most-recent-first) describing the id-space + FSM-state fix.

- [ ] **Step 6 — build + commit.**
  `tkbuild tk26_vision --packages-select vision_track` (verify clean), then
  commit ONLY the touched files (explicit paths; never `git add -A`):
  `git add src/vision_track/vision_track/person_track_node.py src/vision_track/test/test_debug_draw_color.py src/vision_track/readme.md`
  Message: `fix(vision_track): draw locked target green via live id + FSM state (kill stuck-yellow)`
  Trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Phase 2 — Issue 2: reseed confirmation gate (one commit)

**Files:** `core/lock_state_machine.py`, `yolo_tracker.py`,
`core/tracking_pipeline.py` (or node) for the per-frame probation step,
`person_track_node.py` (param declare + wiring). Tests:
`test_lock_state_machine.py` (extend), `test_reseed_target.py` (extend),
`test_wave_auto_reseed.py` (extend), new `test_reseed_probation.py`. Doc: `readme.md`.

- [ ] **Step 1 — FSM `start_probation` failing test.** In
  `test_lock_state_machine.py` add: after `start_probation(7)`, state is
  `'reidentifying'`, `_committed_id==7`, a `step(present=True,...)` returns
  `tracking`/`target_lost=False` (probation→commit happens via the pipeline, but
  a present-by-id frame still commits as today). Assert `start_probation` does
  NOT immediately yield `target_lost=False` without a present/step.

- [ ] **Step 2 — run fail; implement `start_probation`.**
  ```python
  def start_probation(self, committed_id: int) -> None:
      """Re-arm onto a candidate id WITHOUT committing (probationary reseed).

      Unlike start() (jumps to 'tracking'), this enters 'reidentifying' so the
      node does not report a lock until the reseed probation confirms. Also
      lifts the machine out of terminal 'lost' so it can be stepped.
      """
      self._committed_id = committed_id
      self._state = "reidentifying"
      self._provisional_streak = 0
  ```
  Run → green.

- [ ] **Step 3 — `_apply_reseed` probation: failing test.** In a new
  `test_reseed_probation.py` (SimpleNamespace tracker like `test_reseed_target.py`):
  - `reseed_target(...)` / `_apply_reseed(...)` sets `reseed_probation_id` and
    `reseed_probation_count=0`, `state=REIDENTIFYING`, and does NOT call
    `start()` / does NOT set `state=TRACKING`.
  - the FSM is re-armed via `start_probation` (so it's `reidentifying`).
  - gallery still gets the fresh crop.

- [ ] **Step 4 — run fail; implement.** In `yolo_tracker.py` `_apply_reseed`
  (~555-580): keep ids/class/occlusion-clear/gallery-add; REMOVE
  `self.state = TrackerState.TRACKING` and the `lock_state_machine.start(...)`
  call; instead set:
  ```python
  self.state = TrackerState.REIDENTIFYING
  self.reseed_probation_id = self.target_track_id
  self.reseed_probation_count = 0
  if self.lock_state_machine is not None and self.original_track_id is not None:
      self.lock_state_machine.start_probation(self.original_track_id)
  ```
  Initialize `self.reseed_probation_id = None` / `self.reseed_probation_count = 0`
  in `__init__`/state-init and clear them in `reset()` and on a true commit.
  Add `reseed_confirmation_frames` attribute (set from the node param; default 5).

- [ ] **Step 5 — per-frame probation step: failing test.** Add to
  `test_reseed_probation.py`: drive N frames through the recovery/track path.
  - present + ReID `sim >= reid_threshold` → `reseed_probation_count` increments;
    `target_lost` stays True until count reaches `reseed_confirmation_frames`,
    then commits (`start()`/present → `target_lost=False`, probation cleared).
  - present + `sim < reid_threshold` → count resets to 0, still `target_lost=True`.
  - seeded id absent → probation abandoned (fall back to normal recovery; count
    cleared); no commit.
  - `_help_latched` is NOT cleared during probation frames (they are
    `target_lost=True`).

- [ ] **Step 6 — run fail; implement the probation step.** Implement in the
  recovery path so it runs every loop frame while `reseed_probation_id is not
  None`. Cleanest location: at the top of `update_tracker`/`reidentify_target`
  in `core/tracking_pipeline.py`, before the normal Stage-1/Stage-2 logic:
  ```python
  if getattr(tracker, "reseed_probation_id", None) is not None:
      return _step_reseed_probation(tracker, frame, results)
  ```
  `_step_reseed_probation`:
  - find the detection with `track_id == tracker.reseed_probation_id`.
  - if absent → clear probation (`reseed_probation_id=None`, count=0), let the
    normal recovery run next frame (return None / fall through), step FSM coast.
  - if present → compute ReID similarity vs `tracker.target_appearance`
    (reuse `_get_or_extract_features` + `ReIDMatcher.compute_similarity`, the
    same call `_confirm_reid_candidate` uses). If `>= tracker.reid_threshold`:
    `reseed_probation_count += 1`, `frames_lost = 0`; else
    `reseed_probation_count = 0`.
  - if `reseed_probation_count >= tracker.reseed_confirmation_frames`: commit —
    `fsm.start(tracker.original_track_id)`, step present=True, `state=TRACKING`,
    clear probation, return `tracker._with_original_id(present_result)`.
  - else: still probationary — step FSM as a non-committing frame
    (`start_probation` already left it `reidentifying`; step present=False so
    `target_lost` stays True), return None (node routes to lost handler →
    YELLOW). Keep `last_frame_recovery=True` so the node defers to
    `last_lock_decision`.
  Ensure `reid_confirmation_frames`-style state and the help latch are untouched.

- [ ] **Step 7 — node param + wiring; extend existing reseed tests.** In
  `person_track_node.py` declare `reseed_confirmation_frames` (default 5) and set
  it on the tracker (near the FSM construction ~367-376). Update
  `test_reseed_target.py` / `test_wave_auto_reseed.py` so any assertion that
  reseed instantly reports locked now expects the probation (or drives the
  confirmation frames). The waving auto-reseed path (`track_web.py`) is
  unchanged — it calls the same service; assert it inherits the gate.

- [ ] **Step 8 — run full suite green; README changelog** (same commit):
  document the reseed probation gate (both manual + waving), default 5 frames,
  the precision rationale (IoU selection + appearance confirmation), and the
  help-latch interaction.

- [ ] **Step 9 — build + commit.** `tkbuild tk26_vision --packages-select
  vision_track`; commit explicit touched paths.
  Message: `feat(vision_track): gate reseed (manual + waving) behind a short ReID confirmation`

---

## Phase 3 — Issue 3: don't give up on look-alikes (one commit)

**Files:** `reid/reid_search.py` (`_single_candidate_guard`),
`core/tracking_pipeline.py` (`_confirm_reid_candidate` commit-bar + N-of-M,
pass `num_candidates`), `core/lock_state_machine.py` (N-of-M provisional streak),
`person_track_node.py` (param declares + pass-through). Tests:
`test_lock_state_machine.py` (extend), new `test_lookalike_pursuit.py`,
`test_single_candidate_pursue.py`. Doc: `readme.md`.

- [ ] **Step 1 — pursue-floor failing test.** New `test_single_candidate_pursue.py`:
  `_single_candidate_guard` with a single person:
  - `best_similarity = 0.60`, `single_person_pursue_floor=0.55` → returns True
    (pursued, not discarded).
  - `best_similarity = 0.50` → returns False (below pursue floor → discard).
  - `best_similarity = 0.80` → True (unchanged).
  Note: the signature must receive the pursue floor (pass `tracker` or the float).

- [ ] **Step 2 — run fail; implement.** In `reid_search.py` change
  `_single_candidate_guard` to take the pursue floor (thread `tracker` through
  from `find_best_match_reid`, default 0.55 if attr missing) and compare against
  `single_person_pursue_floor` instead of the hard-coded `0.72`. Keep the log.

- [ ] **Step 3 — commit-bar + N-of-M failing tests.** In `test_lookalike_pursuit.py`
  (SimpleNamespace tracker, drive `_confirm_reid_candidate` / `reidentify_target`):
  - **lone pursue, no commit:** lone candidate at sim 0.60 for 20 frames →
    pursued (returns non-None provisional OR routes provisional with
    `target_lost=True`) but NEVER commits the id-swap (`target_track_id`
    unchanged). Asserts the precision invariant.
  - **lone commit at high bar with dips (N-of-M):** lone candidate hitting
    `sim>=0.72` on 12 of the last 18 frames (with ≤6 dips interspersed) →
    commits (id-swap, `target_lost=False`). Strict-consecutive would have failed.
  - **bystander spike rejected:** a 5-frame ≥0.72 spike (then gone) within an
    18-window never reaches 12 → no commit.
  Add FSM N-of-M tests to `test_lock_state_machine.py`: 12-of-18 clear-bar frames
  commit; 11 hits never; a single dip no longer zeroes the accumulated count.

- [ ] **Step 4 — run fail; implement.**
  - `core/lock_state_machine.py`: add `provisional_commit_window` (M, default 18)
    to `__init__`; replace `_provisional_streak`'s reset-to-0 (line ~101) with a
    bounded windowed counter (e.g. a deque/ring of the last M `clears_bar`
    booleans, or a `(_window_hits, _window_len)` pair); commit when
    `hits >= commit_frames` within the last M frames. Per-frame bar
    (`high_bar`+depth+distinct) unchanged. Keep `start()`/`start_probation()`
    resetting the window.
  - `core/tracking_pipeline.py` `_confirm_reid_candidate`: accept
    `num_candidates`; compute
    `commit_bar = tracker.single_person_commit_bar if num_candidates == 1 else tracker.reid_threshold`.
    A frame is a confirm-hit only if `match_similarity >= commit_bar`. Convert
    `consecutive_reid_frames` (and the preconfirm ramp) to N-of-M over the last
    `provisional_commit_window` frames where the pending id stayed best: a non-hit
    frame keeps the pending alive (do not zero), abandon only on id change or no
    hits within the window. Commit when hits `>= reid_confirmation_frames`.
    Update the caller `reidentify_target` to pass `num_cands`.
  - Keep the existing post-shake-extra and switch-cooldown semantics.

- [ ] **Step 5 — node params + pass-through.** In `person_track_node.py` declare
  `single_person_pursue_floor` (0.55), `single_person_commit_bar` (0.72),
  `provisional_commit_window` (18); set them on the tracker and pass M into the
  `LockStateMachine(...)` constructor (~367-376).

- [ ] **Step 6 — run full suite green; README changelog** (same commit):
  document pursue-floor + commit-bar safeguard + N-of-M, the precision invariant
  (lone commit bar stays 0.72), and that NEEDS_HELP / help-latch are preserved.

- [ ] **Step 7 — build + commit.** `tkbuild tk26_vision --packages-select
  vision_track`; commit explicit touched paths.
  Message: `feat(vision_track): pursue look-alikes in passive reacq (soft floor + N-of-M, commit bar held high)`

---

## Final — DEV_NOTES + suite verification (one commit)

- [ ] Run the full suite once more; capture pass/skip counts + flake8 baseline.
- [ ] Append a `DEV_NOTES.md` entry (2026-06-09) summarizing all three fixes,
  the precision invariant, and the operator-in-the-loop checks still pending
  (yellow→green on a real reclaim; reseed needs ~5 confirmed frames; lone
  look-alike re-locks under lighting change without a wave; no bystander lock).
- [ ] Commit: `docs(tk26_vision): DEV_NOTES — viz/reseed-gate/look-alike recovery`

## Self-review checklist (controller, before Phase 1)
- Spec coverage: every spec section maps to a phase ✓ (1→P1, 2→P2, 3→P3).
- No threshold lowered; lone commit bar held at 0.72 ✓ (P3 commit-bar safeguard).
- Type/name consistency: `start_probation`, `reseed_probation_id/_count`,
  `single_person_pursue_floor`, `single_person_commit_bar`,
  `provisional_commit_window`, `reseed_confirmation_frames` used identically
  across spec + plan ✓.
