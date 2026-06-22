# Person Tracker — Phase 2: Recovery Policy + Geometry Robustness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `reacquire_latency_s` (median ≤ 1.0) and hold `false_target_rate`
(≤ 0.05) **without re-introducing wrong locks**, plus lift lateral position
accuracy (`pos_error_lateral_m` ≤ 0.25). Resolve the reacquire↔false-target
tension with **asymmetric hysteresis**: publish a provisional 3D point *fast* only
when a candidate clears a high single-candidate bar **and** a distinctiveness
margin, while keeping `target_lost=True` during any coast and bounding the coast
with a finite `max_recovery_frames`. Reject "crosser" candidates whose depth jumps
toward the camera. Smooth the published 3D point with torso-band sampling + EMA,
reset on loss.

**Architecture:** All decision logic that the gates depend on is extracted into
**pure, ROS-free, numpy-only modules under `vision_track/core/`** so it is unit
-testable with synthetic per-frame inputs:

- `core/lock_state_machine.py` — `LockStateMachine`, a pure finite-state machine.
  Input per frame: `(sim_score, present, frames_since_loss, num_candidates,
  distinct_margin, depth_consistent)`. Output: `LockDecision(publish, target_lost,
  committed_id, state)`. It encodes the asymmetric hysteresis (fast provisional
  publish above the HIGH bar; conservative commit; bounded coast). `tracking_pipeline.py`
  and `person_track_node.py` call into it; they keep no duplicate transition logic.
- `core/depth_gate.py` — `is_depth_consistent(candidate_depth, operator_depth,
  jump_threshold)`, a pure predicate rejecting toward-camera depth jumps, plus
  `roi_median_depth(...)` to derive a candidate's median depth from a depth ROI.
- `core/centroid_smooth.py` — `torso_band_mask(...)` (selects chest-band rows that
  feed Phase 0's robust median reduction, layered ON TOP of it) and `PointEMA`
  (EMA / constant-velocity smoothing with reset-on-loss).

The node (`person_track_node.py`) owns the only depth image, so it computes the
operator's last median depth and *plumbs it into the tracker* (a new
`tracker.operator_last_depth_m` attribute the tracker reads; the tracker never
imports ROS). The tracker wires the pure predicate into
`_verify_person_candidate` / occlusion handling, and the state machine into the
lock/recovery path. Geometry smoothing is applied in the node's
`_handle_tracked_frame` via the pure reducer + EMA.

**Tech Stack:** ROS2 Humble (`rclpy`, node layer only), `ultralytics` YOLO11s-seg,
custom ResNet50/OSNet ReID (Phase 1), numpy, OpenCV, pytest. Pure modules import
only `numpy` + stdlib — **no `rclpy`, no `torch`, no `cv2` at module top level**.

**Spec:** `docs/superpowers/specs/2026-06-03-person-tracker-overhaul-design.md` (§7,
Phase 2).

---

## Dependency on Phase 1 (read first)

**Phase 2 depends on Phase 1 (reliable identity) being merged.** The aggressive
recovery here is only safe because Phase 1 made identity trustworthy (real OSNet
weights via `reid_backbone`, gallery hygiene, Lowe-ratio identity gating). If
Phase 1 is *not* merged, the high-bar provisional publish will surface the same
wrong locks Phase 1 was built to kill. At plan-execution time, verify:

- Branch `feat/person-tracker-overhaul` contains Phase 0 + Phase 1 commits
  (`git log --oneline | grep -iE 'phase 0|phase 1|reid backbone|gallery|default.yaml'`).
- `src/vision_track/config/default.yaml` **exists** (created in Phase 1). Phase 2
  *appends* its params there; it does not create the file.
- `core/centroid_smooth.py`'s reducer is layered on Phase 0's robust median-x/y +
  z-outlier reduction in `_calculate_centroid` (`person_track_node.py:378-381`
  pre-Phase-0; Phase 0 replaces the `np.mean(...)` x/y with a robust median and
  adds z-outlier rejection). Do **not** redefine that reduction — call it / build
  on it.

**Names this plan must reuse verbatim from the SHARED CONTRACT (do NOT redefine):**
`centroid_field`, `centroid_track`, the robust median-x/y + z-outlier reduction,
`pos_error_range_m` gate, `perf_logging_enabled`, `reid_backbone`,
`yolo_track_conf`, `config/default.yaml`.

**Names this plan INTRODUCES:** `max_recovery_frames` (int, default 45),
`provisional_high_bar` / `provisional_distinct_margin` (the asymmetric-hysteresis
thresholds), `crosser_depth_jump_m` (float, default 0.6), `centroid_ema_alpha`
(float, default 0.5), `torso_band_lo` / `torso_band_hi` (float fracs, default
0.15 / 0.55).

---

## File Structure

```
src/vision_track/
├── vision_track/
│   ├── core/
│   │   ├── lock_state_machine.py     # NEW — pure asymmetric-hysteresis FSM (Task 1)
│   │   ├── depth_gate.py             # NEW — pure depth-consistency predicate (Task 2)
│   │   ├── centroid_smooth.py        # NEW — pure torso-band reducer + PointEMA (Task 3)
│   │   ├── tracking_pipeline.py      # EDIT — wire FSM + depth gate (Tasks 1,2)
│   │   └── tracking_types.py         # EDIT — add LockDecision dataclass (Task 1)
│   ├── yolo_tracker.py               # EDIT — operator_last_depth_m attr + FSM/gate state (Tasks 1,2)
│   └── person_track_node.py          # EDIT — params, depth plumb, smoothing, FSM-gated publish (all tasks)
├── config/
│   └── default.yaml                  # EDIT (Phase-1 file) — append Phase-2 params (all tasks)
└── test/
    ├── test_lock_state_machine.py    # NEW (Task 1)
    ├── test_depth_gate.py            # NEW (Task 2)
    └── test_centroid_smooth.py       # NEW (Task 3)
```

**Test invocation (exact, every task):**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_<name>.py -v
```

(`.venv-vision-main` carries pytest + numpy. The pure modules live under
`vision_track/core/`; importing `from vision_track.core.<mod> import ...` triggers
`vision_track/__init__.py`, which imports `yolo_tracker` → `torch`/`cv2`/`ultralytics`
— all present in the venv — but **not** `rclpy`, so the import succeeds outside a
ROS environment. No node import is required by any Task-1/2/3 unit test.)

---

### Task 1: Asymmetric-hysteresis lock-state machine (pure FSM + wiring)

Replace the effectively-infinite `allow_indefinite_recovery` coast with a bounded,
asymmetric policy: emit a provisional position fast **only** above the HIGH bar +
distinctiveness margin; keep `target_lost=True` during any coast; declare
hard-lost after `max_recovery_frames`.

**Files:**
- Create: `src/vision_track/vision_track/core/lock_state_machine.py`
- Create: `src/vision_track/test/test_lock_state_machine.py`
- Modify: `src/vision_track/vision_track/core/tracking_types.py` (add `LockDecision`
  dataclass after `TrackingResult`, ~line 43)
- Modify: `src/vision_track/vision_track/yolo_tracker.py`
  (`_init_reid_settings` ~122-133, `_init_temporal_consistency` ~165-174,
  `reset()` ~807-844)
- Modify: `src/vision_track/vision_track/core/tracking_pipeline.py`
  (`reidentify_target` ~277-314, `_confirm_reid_candidate` ~317-386)
- Modify: `src/vision_track/vision_track/person_track_node.py`
  (`_declare_parameters` ~117-146, `_load_parameters` ~148-176, `_init_tracker`
  ~178-225 — the `allow_indefinite_recovery` math at 186-191, publish gate
  680-685, `_handle_lost_frame` 736-794)
- Modify: `src/vision_track/config/default.yaml` (Phase-1 file)

- [ ] **Step 1: Failing test — `LockDecision` dataclass exists and is frozen-ish**

Create `test/test_lock_state_machine.py` with the first cases. Start with the
type contract and the trivial "absent → coast → hard-lost" path:

```python
"""Tests for the pure asymmetric-hysteresis lock-state machine."""
import pytest

from vision_track.core.lock_state_machine import LockStateMachine
from vision_track.core.tracking_types import LockDecision


def make_sm(**kw):
    """Build a state machine with test-friendly defaults."""
    params = dict(
        high_bar=0.72,
        distinct_margin=0.10,
        commit_frames=12,
        max_recovery_frames=45,
    )
    params.update(kw)
    return LockStateMachine(**params)


class TestLockDecisionType:
    def test_fields(self):
        d = LockDecision(publish=True, target_lost=False, committed_id=7, state="tracking")
        assert d.publish is True
        assert d.target_lost is False
        assert d.committed_id == 7
        assert d.state == "tracking"


class TestCoastAndHardLost:
    def test_absent_coasts_with_target_lost_true(self):
        sm = make_sm()
        sm.start(committed_id=3)
        # First few absent frames: coasting, target_lost stays True, no publish.
        for f in range(1, 5):
            d = sm.step(sim_score=0.0, present=False, frames_since_loss=f,
                        num_candidates=0, distinct_margin=0.0, depth_consistent=True)
            assert d.target_lost is True
            assert d.publish is False
            assert d.committed_id == 3  # identity retained while coasting
            assert d.state == "reidentifying"

    def test_hard_lost_after_max_recovery_frames(self):
        sm = make_sm(max_recovery_frames=10)
        sm.start(committed_id=3)
        last = None
        for f in range(1, 12):
            last = sm.step(sim_score=0.0, present=False, frames_since_loss=f,
                           num_candidates=0, distinct_margin=0.0, depth_consistent=True)
        # Past the cap → hard lost, identity dropped.
        assert last.state == "lost"
        assert last.target_lost is True
        assert last.publish is False
```

- [ ] **Step 2: Run to fail (module missing)**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_lock_state_machine.py -v
```

Expected: `ModuleNotFoundError: No module named 'vision_track.core.lock_state_machine'`
(collection error) plus `ImportError: cannot import name 'LockDecision'`.

- [ ] **Step 3: Add `LockDecision` to `tracking_types.py`**

Insert directly after the `TrackingResult` dataclass (after line 42, before
`@dataclass class TargetAppearance`):

```python
@dataclass
class LockDecision:
    """Output of LockStateMachine.step — what the node should do this frame.

    Attributes:
        publish: emit the 3D point this frame (provisional or committed).
        target_lost: feedback flag; True during any coast (asymmetric hysteresis).
        committed_id: stable original track id, or None once hard-lost.
        state: one of 'tracking' | 'reidentifying' | 'lost'.
    """

    publish: bool
    target_lost: bool
    committed_id: Optional[int]
    state: str
```

(`Optional` is already imported at `tracking_types.py:5`.)

- [ ] **Step 4: Implement the pure FSM (minimal, passes Step-1 cases)**

Create `core/lock_state_machine.py`:

```python
"""Pure asymmetric-hysteresis lock-state machine for person tracking.

ROS-free + numpy-free: encodes the recovery policy as a finite-state machine so
the reacquire↔false-target trade can be unit-tested with synthetic per-frame
inputs. The node and tracking_pipeline call into it; they hold no duplicate
transition logic.

States: 'tracking' (committed, present) → 'reidentifying' (coasting / probing) →
'lost' (hard-lost past max_recovery_frames).

Asymmetric hysteresis:
- FAST out (provisional publish): a single high-confidence candidate that clears
  the HIGH bar AND wins the distinctiveness margin AND is depth-consistent gets
  published immediately, but target_lost STAYS True until a full commit.
- SLOW commit: target_lost flips to False (full re-lock) only after the candidate
  holds for commit_frames consecutive frames.
- Bounded coast: after max_recovery_frames absent/unconfirmed frames, declare
  hard-lost and drop committed_id.
"""
from typing import Optional

from .tracking_types import LockDecision


class LockStateMachine:
    """Finite-state recovery policy with asymmetric hysteresis."""

    def __init__(
        self,
        high_bar: float = 0.72,
        distinct_margin: float = 0.10,
        commit_frames: int = 12,
        max_recovery_frames: int = 45,
    ) -> None:
        """Store policy thresholds; call start() before stepping."""
        self.high_bar = high_bar
        self.distinct_margin = distinct_margin
        self.commit_frames = commit_frames
        self.max_recovery_frames = max_recovery_frames
        self._committed_id: Optional[int] = None
        self._state = "lost"
        self._provisional_streak = 0

    def start(self, committed_id: int) -> None:
        """Lock onto an initial committed id (called once on init/commit)."""
        self._committed_id = committed_id
        self._state = "tracking"
        self._provisional_streak = 0

    def step(
        self,
        sim_score: float,
        present: bool,
        frames_since_loss: int,
        num_candidates: int,
        distinct_margin: float,
        depth_consistent: bool,
    ) -> LockDecision:
        """Advance one frame; return the publish/target_lost/committed_id decision.

        Args:
            sim_score: best candidate's ReID similarity this frame (0 if absent).
            present: the committed track id was matched by ByteTrack this frame.
            frames_since_loss: consecutive frames since the committed id was
                last directly present (0 while present).
            num_candidates: number of person candidates considered this frame.
            distinct_margin: best - second-best similarity (∞-ish if single).
            depth_consistent: depth gate verdict for the best candidate.
        """
        if self._state == "lost":
            return LockDecision(False, True, None, "lost")

        # Direct, present-by-id tracking: full lock, publish, not lost.
        if present:
            self._state = "tracking"
            self._provisional_streak = 0
            return LockDecision(True, False, self._committed_id, "tracking")

        # Absent: coasting. Bound the coast first.
        if frames_since_loss > self.max_recovery_frames:
            self._committed_id = None
            self._state = "lost"
            self._provisional_streak = 0
            return LockDecision(False, True, None, "lost")

        self._state = "reidentifying"

        # Provisional FAST-publish gate: HIGH bar + distinctiveness + depth + single
        # OR clearly-distinct multi. target_lost STAYS True (asymmetric).
        clears_bar = sim_score >= self.high_bar and depth_consistent
        distinct_ok = num_candidates <= 1 or distinct_margin >= self.distinct_margin
        if clears_bar and distinct_ok:
            self._provisional_streak += 1
            # SLOW commit: only after holding commit_frames do we drop target_lost.
            if self._provisional_streak >= self.commit_frames:
                self._state = "tracking"
                return LockDecision(True, False, self._committed_id, "tracking")
            return LockDecision(True, True, self._committed_id, "reidentifying")

        # Did not clear the bar: coast silently, keep identity, stay lost.
        self._provisional_streak = 0
        return LockDecision(False, True, self._committed_id, "reidentifying")
```

- [ ] **Step 5: Run to pass (Step-1 cases)**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_lock_state_machine.py -v
```

Expected: `TestLockDecisionType` + `TestCoastAndHardLost` PASS (3 passed).

- [ ] **Step 6: Failing test — the reacquire-vs-false-target trade explicitly**

Append to `test/test_lock_state_machine.py`:

```python
class TestAsymmetricHysteresis:
    def test_fast_provisional_publish_above_high_bar(self):
        """A single high-bar, depth-consistent candidate publishes immediately,
        but target_lost stays True until commit_frames are accumulated."""
        sm = make_sm(high_bar=0.72, commit_frames=12)
        sm.start(committed_id=5)
        d = sm.step(sim_score=0.80, present=False, frames_since_loss=1,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=True)
        assert d.publish is True          # FAST out
        assert d.target_lost is True      # but still coasting (asymmetric)
        assert d.committed_id == 5
        assert d.state == "reidentifying"

    def test_no_publish_below_high_bar(self):
        """Below the HIGH bar, no provisional publish even if 'matched-ish'."""
        sm = make_sm(high_bar=0.72)
        sm.start(committed_id=5)
        d = sm.step(sim_score=0.69, present=False, frames_since_loss=1,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=True)
        assert d.publish is False         # conservative: no false target
        assert d.target_lost is True

    def test_ambiguous_multi_candidate_blocks_provisional(self):
        """High score but tiny distinctiveness margin in a crowd → no publish
        (prevents locking a lookalike crosser)."""
        sm = make_sm(high_bar=0.72, distinct_margin=0.10)
        sm.start(committed_id=5)
        d = sm.step(sim_score=0.85, present=False, frames_since_loss=1,
                    num_candidates=3, distinct_margin=0.04, depth_consistent=True)
        assert d.publish is False
        assert d.target_lost is True

    def test_depth_inconsistent_blocks_provisional(self):
        """A toward-camera crosser (depth gate fails) is not published."""
        sm = make_sm(high_bar=0.72)
        sm.start(committed_id=5)
        d = sm.step(sim_score=0.90, present=False, frames_since_loss=1,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=False)
        assert d.publish is False
        assert d.target_lost is True

    def test_sustained_match_commits_and_clears_lost(self):
        """Holding the high bar for commit_frames flips target_lost → False."""
        sm = make_sm(high_bar=0.72, commit_frames=3)
        sm.start(committed_id=5)
        decisions = [
            sm.step(sim_score=0.80, present=False, frames_since_loss=f,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=True)
            for f in range(1, 4)
        ]
        # First two: provisional (publish True, lost True). Third: committed.
        assert decisions[0].target_lost is True and decisions[0].publish is True
        assert decisions[1].target_lost is True
        assert decisions[2].target_lost is False
        assert decisions[2].state == "tracking"

    def test_reacquire_latency_bound_in_frames(self):
        """At a high bar that is met from frame 1, full re-lock happens in
        exactly commit_frames frames (latency = commit_frames / rate)."""
        sm = make_sm(high_bar=0.72, commit_frames=12)
        sm.start(committed_id=5)
        committed_at = None
        for f in range(1, 30):
            d = sm.step(sim_score=0.80, present=False, frames_since_loss=f,
                        num_candidates=1, distinct_margin=999.0, depth_consistent=True)
            if d.target_lost is False:
                committed_at = f
                break
        assert committed_at == 12   # deterministic reacquire bound
```

- [ ] **Step 7: Run to fail (assert the trade)**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_lock_state_machine.py::TestAsymmetricHysteresis -v
```

Expected: all `TestAsymmetricHysteresis` cases PASS already against the Step-4
implementation (the impl was written to satisfy them). If any fail, fix the FSM in
`lock_state_machine.py` (not the test) — most likely the `distinct_ok` /
`clears_bar` conjunction or the `commit_frames` off-by-one. Re-run until green.

- [ ] **Step 8: Commit the pure FSM**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && \
  git add src/vision_track/vision_track/core/lock_state_machine.py \
          src/vision_track/vision_track/core/tracking_types.py \
          src/vision_track/test/test_lock_state_machine.py && \
  git commit -m "feat(vision_track): pure asymmetric-hysteresis lock-state machine

Bounded recovery (max_recovery_frames) + fast provisional publish above the
high single-candidate bar with target_lost held True during coast.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 9: Wire `max_recovery_frames` + thresholds into the tracker (state only)**

In `yolo_tracker.py` `_init_temporal_consistency` (~165-174), append the FSM
state holder + params (read by `_init_tracker` in the node):

```python
        # Phase 2: asymmetric-hysteresis recovery policy params (defaults;
        # the node overrides from ROS params). Pure FSM lives in core/.
        self.max_recovery_frames = 45
        self.provisional_high_bar = 0.72
        self.provisional_distinct_margin = 0.10
        self.lock_state_machine = None  # set by node after construction
```

In `reset()` (~807-844), after `self.reid_fit_id = None` (line 823), add:

```python
        if self.lock_state_machine is not None and self.original_track_id is not None:
            self.lock_state_machine.start(self.original_track_id)
```

(Do **not** import `LockStateMachine` into `yolo_tracker.py` at module top — the
node constructs and injects it, so the tracker stays free of the import while still
holding the reference. This keeps the FSM swappable from tests.)

- [ ] **Step 10: Replace `allow_indefinite_recovery` with `max_recovery_frames` in the node**

In `person_track_node.py` `_declare_parameters` (~126), replace:

```python
        self.declare_parameter('allow_indefinite_recovery', True)  # if True, never abort for long-term loss
```

with:

```python
        # Phase 2: bound recovery so the tracker eventually declares hard-lost.
        # Replaces the effectively-infinite allow_indefinite_recovery coast.
        self.declare_parameter('max_recovery_frames', 45)
        self.declare_parameter('provisional_high_bar', 0.72)
        self.declare_parameter('provisional_distinct_margin', 0.10)
```

In `_load_parameters` (~156), replace the `allow_indefinite_recovery` load with:

```python
        self.max_recovery_frames = self.get_parameter('max_recovery_frames').value
        self.provisional_high_bar = self.get_parameter('provisional_high_bar').value
        self.provisional_distinct_margin = self.get_parameter('provisional_distinct_margin').value
```

Remove the `self.get_logger().info(f'Allow indefinite recovery: ...')` line (~175)
and add `self.get_logger().info(f'Max recovery frames: {self.max_recovery_frames}')`.

In `_init_tracker` (~186-191), replace the infinite-frames math:

```python
            max_frames_allowed = (
                int(self.tracking_rate * self.lost_timeout)
                if not self.allow_indefinite_recovery
                else int(1e12)  # effectively infinite
            )
            max_frames_allowed = max(max_frames_allowed, int(self.max_frames_lost))
```

with:

```python
            # Bounded by max_recovery_frames; max_frames_lost remains the
            # ByteTrack buffer ceiling. The lock FSM owns hard-lost timing.
            max_frames_allowed = max(int(self.max_frames_lost), int(self.max_recovery_frames))
```

After the tracker is constructed (after `self.tracker.max_frames_lost =
max_frames_allowed`, both branches), inject the FSM:

```python
            from vision_track.core.lock_state_machine import LockStateMachine
            self.tracker.max_recovery_frames = int(self.max_recovery_frames)
            self.tracker.provisional_high_bar = float(self.provisional_high_bar)
            self.tracker.provisional_distinct_margin = float(self.provisional_distinct_margin)
            self.tracker.lock_state_machine = LockStateMachine(
                high_bar=self.tracker.provisional_high_bar,
                distinct_margin=self.tracker.provisional_distinct_margin,
                commit_frames=self.tracker.reid_confirmation_frames,
                max_recovery_frames=self.tracker.max_recovery_frames,
            )
```

- [ ] **Step 11: Gate node publish + `target_lost` on the FSM decision**

The pipeline already returns `None` when not tracking and a `TrackingResult` when
present/provisional. Phase 2 routes the recovery decision through the FSM. In
`person_track_node.py`, expose the FSM verdict to the node loop. Minimal wiring:
after `track_result = self.tracker.update(rgb_frame)` (~565), capture the FSM's
latest decision (the tracker stores it on each `update`; see Step 12):

In `_handle_tracked_frame`, change the publish gate (680-685) so that a provisional
frame (`feedback.target_lost == True`) does **not** publish to `/target_points`,
matching the asymmetric contract — only committed frames publish a live point:

```python
            if (
                not feedback.target_lost           # provisional coast does NOT publish
                and feedback.is_transformation_successful
                and self.target_point_pub is not None
            ):
                self.target_point_pub.publish(feedback.target_position)
```

and set `feedback.target_lost` from the FSM decision rather than hardcoding `False`
(655):

```python
            decision = getattr(self.tracker, 'last_lock_decision', None)
            feedback.target_lost = bool(decision.target_lost) if decision is not None else False
```

In `_handle_lost_frame`, replace the time-based abort (788-793) with an FSM
hard-lost check so the coast is bounded by frames, not only `lost_timeout`:

```python
        decision = getattr(self.tracker, 'last_lock_decision', None)
        hard_lost = decision is not None and decision.state == 'lost'
        if hard_lost or time_since_seen > self.lost_timeout:
            reason = 'hard-lost (recovery cap)' if hard_lost else f'lost for {time_since_seen:.1f}s'
            self.get_logger().warn(f'Target {reason}, aborting')
            goal_handle.abort()
            result.status = 1
            result.message = f'Target {reason}'
            return True
        return False
```

- [ ] **Step 12: Drive the FSM inside the pipeline recovery path**

In `tracking_pipeline.py` `reidentify_target` (~277-314) and
`_confirm_reid_candidate` (~317-386), call `tracker.lock_state_machine.step(...)`
once per frame and store the verdict on `tracker.last_lock_decision`. Concretely,
at the top of `reidentify_target`, after `tracker.state = TrackerState.REIDENTIFYING`,
compute the per-frame inputs and step the FSM when one exists:

```python
    fsm = getattr(tracker, "lock_state_machine", None)
    if fsm is not None:
        present = any(r.track_id == tracker.target_track_id for r in results)
        cands = [r for r in results if r.class_id == 0 and r.track_id >= 0]
        # sim_score / distinct_margin / depth_consistent are filled below once
        # find_best_match_reid runs; seed a coast decision first so an early
        # return (no match) still records target_lost.
        tracker.last_lock_decision = fsm.step(
            sim_score=0.0, present=present, frames_since_loss=tracker.frames_lost,
            num_candidates=len(cands), distinct_margin=0.0, depth_consistent=True,
        )
```

After `find_best_match_reid` returns a `(match_result, best_similarity)` and the
depth gate (Task 2) has run, re-step the FSM with the real `sim_score`,
`distinct_margin` (best minus second-best, from `candidate_scores` — surface it via
a `tracker.last_reid_margin` attribute set in `reid_search.find_best_match_reid`),
and `depth_consistent`. Store on `tracker.last_lock_decision`. Use
`decision.publish` to decide whether `reidentify_target` returns the provisional
`tracker._with_original_id(match_result)` or `None`.

> Keep the *existing* commit logic (`_confirm_reid_candidate`'s
> `reid_confirmation_frames` accumulation) as the mechanism that actually swaps
> `target_track_id`; the FSM mirrors that timing via `commit_frames =
> reid_confirmation_frames`. The FSM is the **publish/target_lost authority**; the
> pipeline remains the **identity-swap authority**. This avoids a double source of
> truth for the ByteTrack id while letting the node gate publishing on the FSM.

- [ ] **Step 13: Append Phase-2 recovery params to `config/default.yaml`**

Append under the existing `person_track_node:` `ros__parameters:` block (created in
Phase 1):

```yaml
    # --- Phase 2: recovery policy (asymmetric hysteresis) ---
    max_recovery_frames: 45          # coast cap; past this → hard-lost
    provisional_high_bar: 0.72       # single-candidate fast-publish floor (reid_search.py)
    provisional_distinct_margin: 0.10  # min best-vs-second margin to publish in a crowd
```

- [ ] **Step 14: Run the full pure suite to confirm no regression**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_lock_state_machine.py -v
```

Expected: all Task-1 cases PASS. (Node wiring is exercised by manual T3/T4 below;
it imports `rclpy` so it is not in the pure suite.)

- [ ] **Step 15: Commit the wiring**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && \
  git add src/vision_track/vision_track/yolo_tracker.py \
          src/vision_track/vision_track/core/tracking_pipeline.py \
          src/vision_track/vision_track/person_track_node.py \
          src/vision_track/config/default.yaml && \
  git commit -m "feat(vision_track): bound recovery via max_recovery_frames + FSM-gated publish

Replace allow_indefinite_recovery with a finite coast; route publish/target_lost
through the asymmetric-hysteresis FSM (provisional frames do not publish).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Depth-gated crosser rejection (pure predicate + wiring)

A person crossing between robot and operator is geometrically nearer — a cue
appearance cannot spoof. Plumb the node's last operator depth into the tracker and
reject candidates whose median depth jumps toward the camera beyond
`crosser_depth_jump_m`.

**Files:**
- Create: `src/vision_track/vision_track/core/depth_gate.py`
- Create: `src/vision_track/test/test_depth_gate.py`
- Modify: `src/vision_track/vision_track/yolo_tracker.py`
  (`_init_motion_tracking` ~135-154 — add `operator_last_depth_m`; `reset()`
  ~807-844)
- Modify: `src/vision_track/vision_track/core/tracking_pipeline.py`
  (`_verify_person_candidate` ~159-212, `_handle_occlusion_state` ~108-156)
- Modify: `src/vision_track/vision_track/person_track_node.py`
  (`_handle_tracked_frame` ~634-734 — set `self.tracker.operator_last_depth_m`;
  `_declare_parameters` / `_load_parameters`; pass candidate depth ROI in)
- Modify: `src/vision_track/config/default.yaml`

- [ ] **Step 1: Failing test — `is_depth_consistent` predicate**

Create `test/test_depth_gate.py`:

```python
"""Tests for the pure depth-consistency (crosser-rejection) predicate."""
import numpy as np
import pytest

from vision_track.core.depth_gate import is_depth_consistent, roi_median_depth


class TestIsDepthConsistent:
    def test_same_depth_consistent(self):
        assert is_depth_consistent(3.0, 3.0, jump_threshold=0.6) is True

    def test_small_toward_camera_jump_consistent(self):
        # operator at 3.0 m, candidate at 2.5 m → 0.5 m nearer < 0.6 threshold
        assert is_depth_consistent(2.5, 3.0, jump_threshold=0.6) is True

    def test_large_toward_camera_jump_rejected(self):
        # candidate 1.0 m vs operator 3.0 m → 2.0 m nearer, a crosser
        assert is_depth_consistent(1.0, 3.0, jump_threshold=0.6) is False

    def test_farther_candidate_always_consistent(self):
        # moving AWAY from the camera is never a crosser cue
        assert is_depth_consistent(5.0, 3.0, jump_threshold=0.6) is True

    def test_no_operator_depth_passes(self):
        # unknown operator depth → cannot gate → permissive
        assert is_depth_consistent(1.0, None, jump_threshold=0.6) is True

    def test_invalid_candidate_depth_passes(self):
        assert is_depth_consistent(0.0, 3.0, jump_threshold=0.6) is True
        assert is_depth_consistent(float("nan"), 3.0, jump_threshold=0.6) is True

    def test_threshold_boundary_inclusive(self):
        # exactly at the threshold is still consistent (reject only beyond)
        assert is_depth_consistent(2.4, 3.0, jump_threshold=0.6) is True
        assert is_depth_consistent(2.39, 3.0, jump_threshold=0.6) is False


class TestRoiMedianDepth:
    def _depth(self, H, W, val_m):
        return np.full((H, W), int(val_m * 1000), dtype=np.uint16)

    def test_constant_roi(self):
        d = self._depth(100, 100, 2.5)
        m = roi_median_depth(d, (10, 10, 60, 60), min_depth=0.1, max_depth=10.0)
        assert abs(m - 2.5) < 1e-3

    def test_excludes_zero_and_out_of_range(self):
        d = self._depth(100, 100, 2.5)
        d[10:20, 10:20] = 0          # invalid
        d[20:30, 10:20] = 11000      # out of range
        m = roi_median_depth(d, (10, 10, 60, 60), min_depth=0.1, max_depth=10.0)
        assert abs(m - 2.5) < 1e-3   # median over valid only

    def test_all_invalid_returns_none(self):
        d = np.zeros((100, 100), dtype=np.uint16)
        m = roi_median_depth(d, (10, 10, 60, 60), min_depth=0.1, max_depth=10.0)
        assert m is None
```

- [ ] **Step 2: Run to fail (module missing)**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_depth_gate.py -v
```

Expected: `ModuleNotFoundError: No module named 'vision_track.core.depth_gate'`.

- [ ] **Step 3: Implement the pure predicate**

Create `core/depth_gate.py`:

```python
"""Pure depth-consistency predicate for crosser rejection.

ROS-free + numpy-only. A candidate whose median depth jumps toward the camera
beyond a threshold (relative to the operator's last known depth) is geometrically
a crosser passing between robot and operator — a cue appearance cannot spoof.
"""
import math
from typing import Optional, Tuple

import numpy as np


def is_depth_consistent(
    candidate_depth: float,
    operator_depth: Optional[float],
    jump_threshold: float,
) -> bool:
    """Return True if the candidate is NOT a toward-camera crosser.

    Args:
        candidate_depth: candidate's median depth (m). 0/NaN ⇒ permissive (True).
        operator_depth: operator's last known depth (m). None ⇒ permissive.
        jump_threshold: max allowed toward-camera jump (m). A candidate nearer
            than ``operator_depth - jump_threshold`` is rejected.

    Moving farther than the operator is never a crosser cue (always consistent).
    """
    if operator_depth is None:
        return True
    if candidate_depth is None or candidate_depth <= 0.0 or math.isnan(candidate_depth):
        return True
    # Nearer to the camera by more than the threshold ⇒ crosser ⇒ inconsistent.
    return candidate_depth >= (operator_depth - jump_threshold)


def roi_median_depth(
    depth_mm,
    bbox: Tuple[int, int, int, int],
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Optional[float]:
    """Median valid depth (m) over a bbox ROI of a uint16/float mm depth image.

    Returns None if no pixel is in (min_depth, max_depth).
    """
    depth = np.asarray(depth_mm).astype(np.float32) * 0.001
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth[y1:y2, x1:x2]
    valid = roi[(roi > min_depth) & (roi < max_depth)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
```

- [ ] **Step 4: Run to pass**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_depth_gate.py -v
```

Expected: all `TestIsDepthConsistent` + `TestRoiMedianDepth` cases PASS.

- [ ] **Step 5: Commit the pure predicate**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && \
  git add src/vision_track/vision_track/core/depth_gate.py \
          src/vision_track/test/test_depth_gate.py && \
  git commit -m "feat(vision_track): pure depth-consistency crosser-rejection predicate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Failing test — the wiring decision (pure helper)**

The integration into `_verify_person_candidate` needs `rclpy`-free testability for
the *decision*, so add a thin pure helper `should_reject_candidate(...)` that
combines the predicate with the tracker's stored operator depth and the candidate's
depth. Add cases to `test/test_depth_gate.py`:

```python
from vision_track.core.depth_gate import should_reject_candidate


class TestShouldRejectCandidate:
    def test_rejects_toward_camera_crosser(self):
        # operator 3.0 m, candidate 1.0 m, threshold 0.6 → reject
        assert should_reject_candidate(
            candidate_depth=1.0, operator_depth=3.0, jump_threshold=0.6
        ) is True

    def test_keeps_consistent_candidate(self):
        assert should_reject_candidate(
            candidate_depth=2.8, operator_depth=3.0, jump_threshold=0.6
        ) is False

    def test_no_operator_depth_keeps(self):
        assert should_reject_candidate(
            candidate_depth=1.0, operator_depth=None, jump_threshold=0.6
        ) is False
```

- [ ] **Step 7: Run to fail (helper missing)**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_depth_gate.py::TestShouldRejectCandidate -v
```

Expected: `ImportError: cannot import name 'should_reject_candidate'`.

- [ ] **Step 8: Implement `should_reject_candidate`**

Append to `core/depth_gate.py`:

```python
def should_reject_candidate(
    candidate_depth: Optional[float],
    operator_depth: Optional[float],
    jump_threshold: float,
) -> bool:
    """Convenience inverse of is_depth_consistent for the call sites.

    Returns True when the candidate should be rejected as a crosser.
    """
    return not is_depth_consistent(candidate_depth, operator_depth, jump_threshold)
```

- [ ] **Step 9: Run to pass**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_depth_gate.py -v
```

Expected: all depth-gate cases PASS.

- [ ] **Step 10: Plumb operator depth + candidate depth into the tracker**

Add the `crosser_depth_jump_m` param. In `person_track_node.py`
`_declare_parameters` (~127, after the recovery params from Task 1):

```python
        self.declare_parameter('crosser_depth_jump_m', 0.6)
```

In `_load_parameters`:

```python
        self.crosser_depth_jump_m = self.get_parameter('crosser_depth_jump_m').value
```

In `_init_tracker`, after FSM injection, set:

```python
            self.tracker.crosser_depth_jump_m = float(self.crosser_depth_jump_m)
```

In `yolo_tracker.py` `_init_motion_tracking` (~135-154), add:

```python
        # Phase 2: operator's last known median depth (m), plumbed from the node
        # (only the node owns the depth image). None ⇒ depth gate permissive.
        self.operator_last_depth_m: Optional[float] = None
        self.crosser_depth_jump_m = 0.6
        # Per-frame map: track_id -> candidate median depth (m), set by the node
        # before each tracker.update so the pipeline can gate ReID candidates.
        self.candidate_depths_m: Dict[int, float] = {}
```

In `reset()` (~807-844), add `self.operator_last_depth_m = None` and
`self.candidate_depths_m = {}`.

In `person_track_node.py` `_handle_tracked_frame`, after `position` is computed
(~653), update the operator's last depth from the committed centroid's z:

```python
        if position is not None:
            self.tracker.operator_last_depth_m = float(position.z)
```

And, **before** calling `self.tracker.update(rgb_frame)` in `_run_tracking_loop`
(~565), populate per-candidate depths so the pipeline can gate ReID candidates.
Since the depth image is available in the loop only after `_get_latest_data`,
compute `candidate_depths_m` from `self.tracker.last_results` of the *previous*
frame's boxes against the *current* depth, or (simpler, deterministic) compute it
lazily inside `_handle_tracked_frame`/recovery using `depth_gate.roi_median_depth`
over each candidate bbox. Recommended minimal wiring: in `_run_tracking_loop`,
after `data` is unpacked and before `self.tracker.update`, build the map from the
last frame's results:

```python
            with self.lock_tracker:
                if self.tracker.last_results and depth_msg is not None:
                    self._refresh_candidate_depths(depth_msg, intrinsic)
                ...
                track_result = self.tracker.update(rgb_frame)
```

with a new node method:

```python
    def _refresh_candidate_depths(self, depth_msg, intrinsic):
        """Median-depth per visible person bbox, for the crosser gate."""
        from vision_track.core.depth_gate import roi_median_depth
        h, w = depth_msg.height, depth_msg.width
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(h, w)
        depths = {}
        for r in self.tracker.last_results:
            if r.class_id != 0 or r.track_id < 0:
                continue
            m = roi_median_depth(depth, r.bbox, self.min_depth, self.max_depth)
            if m is not None:
                depths[r.track_id] = m
        self.tracker.candidate_depths_m = depths
```

- [ ] **Step 11: Wire the gate into `_verify_person_candidate` + occlusion handling**

In `tracking_pipeline.py` `_verify_person_candidate` (~159-212), after `features`
are extracted and before/with the `similarity < 0.50` check, add the depth gate:

```python
    from .depth_gate import should_reject_candidate
    cand_depth = getattr(tracker, "candidate_depths_m", {}).get(result.track_id)
    op_depth = getattr(tracker, "operator_last_depth_m", None)
    jump = getattr(tracker, "crosser_depth_jump_m", 0.6)
    if should_reject_candidate(cand_depth, op_depth, jump):
        logger.warning(
            f"Depth gate reject: Track ID {result.track_id} candidate depth "
            f"{cand_depth} jumped toward camera vs operator {op_depth} "
            f"(jump>{jump} m); treating as crosser."
        )
        return None
```

In `_handle_occlusion_state` (~108-156), apply the same gate to the
occlusion-recovery candidate before `return tracker._with_original_id(result)` at
line 137 — a crosser causing the occlusion must not be adopted as the target:

```python
        from .depth_gate import should_reject_candidate
        cand_depth = getattr(tracker, "candidate_depths_m", {}).get(result.track_id)
        if should_reject_candidate(
            cand_depth, getattr(tracker, "operator_last_depth_m", None),
            getattr(tracker, "crosser_depth_jump_m", 0.6),
        ):
            logger.warning(
                f"Depth gate reject during occlusion: ID {result.track_id} is a crosser."
            )
            return None
        tracker.state = TrackerState.TRACKING
```

Also feed `depth_consistent` into the FSM step in `reidentify_target` (Task 1,
Step 12): compute `depth_consistent = not should_reject_candidate(
candidate_depths_m.get(match_result.track_id), operator_last_depth_m,
crosser_depth_jump_m)` for the chosen `match_result` and pass it.

- [ ] **Step 12: Append `crosser_depth_jump_m` to `config/default.yaml`**

```yaml
    crosser_depth_jump_m: 0.6        # reject candidates this much nearer than the operator
```

- [ ] **Step 13: Run the pure suite**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_depth_gate.py test/test_lock_state_machine.py -v
```

Expected: all Task-1 + Task-2 cases PASS.

- [ ] **Step 14: Commit the depth-gate wiring**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && \
  git add src/vision_track/vision_track/yolo_tracker.py \
          src/vision_track/vision_track/core/tracking_pipeline.py \
          src/vision_track/vision_track/person_track_node.py \
          src/vision_track/vision_track/core/depth_gate.py \
          src/vision_track/test/test_depth_gate.py \
          src/vision_track/config/default.yaml && \
  git commit -m "feat(vision_track): depth-gated crosser rejection

Plumb operator last depth + per-candidate median depth into the tracker; reject
toward-camera depth jumps in _verify_person_candidate and occlusion handling.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Geometry robustness — torso-band sampling + EMA smoothing

Layer torso-band sampling (chest-height rows of the mask) on top of Phase 0's
robust median-x/y + z-outlier reduction, then EMA/constant-velocity smooth the
published 3D point, resetting on loss.

**Files:**
- Create: `src/vision_track/vision_track/core/centroid_smooth.py`
- Create: `src/vision_track/test/test_centroid_smooth.py`
- Modify: `src/vision_track/vision_track/person_track_node.py`
  (`_calculate_centroid` ~324-388 — apply torso band; `_handle_tracked_frame`
  ~634-734 — EMA on the published point; `_cleanup_tracking` ~796-814 /
  `_handle_lost_frame` ~736-794 — reset EMA on loss; params)
- Modify: `src/vision_track/config/default.yaml`

- [ ] **Step 1: Failing test — torso-band reducer**

Create `test/test_centroid_smooth.py`:

```python
"""Tests for pure torso-band reducer + EMA point smoother."""
import numpy as np
import pytest

from vision_track.core.centroid_smooth import torso_band_mask, PointEMA


class TestTorsoBandMask:
    def test_band_selects_chest_rows(self):
        """A bbox of height 100 with band (0.15, 0.55) keeps rows 15..55."""
        bbox = (10, 0, 60, 100)  # x1,y1,x2,y2 — height 100
        m = torso_band_mask(bbox, lo=0.15, hi=0.55)
        y1_band, y2_band = m
        assert y1_band == 15
        assert y2_band == 55

    def test_band_clamped_to_bbox(self):
        bbox = (0, 40, 50, 60)  # height 20
        y1_band, y2_band = torso_band_mask(bbox, lo=0.15, hi=0.55)
        assert y1_band == 40 + 3   # 0.15 * 20 = 3
        assert y2_band == 40 + 11  # 0.55 * 20 = 11

    def test_degenerate_band_returns_full(self):
        """Tiny bbox where lo*h == hi*h → fall back to full bbox rows."""
        bbox = (0, 0, 50, 2)  # height 2 → 0.15*2=0, 0.55*2=1
        y1_band, y2_band = torso_band_mask(bbox, lo=0.15, hi=0.55, min_rows=4)
        # band too thin → returns full bbox y-range
        assert y1_band == 0
        assert y2_band == 2


class TestPointEMA:
    def test_first_sample_passes_through(self):
        ema = PointEMA(alpha=0.5)
        out = ema.update((1.0, 2.0, 3.0))
        assert out == (1.0, 2.0, 3.0)

    def test_ema_blends(self):
        ema = PointEMA(alpha=0.5)
        ema.update((0.0, 0.0, 0.0))
        out = ema.update((2.0, 4.0, 6.0))
        assert out == pytest.approx((1.0, 2.0, 3.0))

    def test_alpha_one_is_passthrough(self):
        ema = PointEMA(alpha=1.0)
        ema.update((0.0, 0.0, 0.0))
        out = ema.update((5.0, 5.0, 5.0))
        assert out == pytest.approx((5.0, 5.0, 5.0))

    def test_reset_clears_state(self):
        ema = PointEMA(alpha=0.5)
        ema.update((10.0, 10.0, 10.0))
        ema.reset()
        out = ema.update((1.0, 2.0, 3.0))
        assert out == (1.0, 2.0, 3.0)   # first-after-reset passes through

    def test_none_sample_does_not_corrupt_state(self):
        ema = PointEMA(alpha=0.5)
        ema.update((2.0, 2.0, 2.0))
        assert ema.update(None) is None
        out = ema.update((4.0, 4.0, 4.0))
        # state preserved across the None gap
        assert out == pytest.approx((3.0, 3.0, 3.0))
```

- [ ] **Step 2: Run to fail (module missing)**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_centroid_smooth.py -v
```

Expected: `ModuleNotFoundError: No module named 'vision_track.core.centroid_smooth'`.

- [ ] **Step 3: Implement the pure reducer + EMA**

Create `core/centroid_smooth.py`:

```python
"""Pure geometry helpers: torso-band row selection + EMA point smoothing.

ROS-free + numpy-only. Layers on TOP of Phase 0's robust median-x/y + z-outlier
reduction in PersonTrackNode._calculate_centroid: the band restricts which mask
rows feed that reduction, and PointEMA smooths the resulting 3D point across
frames (reset on loss).
"""
from typing import Optional, Tuple


def torso_band_mask(
    bbox: Tuple[int, int, int, int],
    lo: float = 0.15,
    hi: float = 0.55,
    min_rows: int = 4,
) -> Tuple[int, int]:
    """Return absolute (y1_band, y2_band) image rows for the chest band of a bbox.

    Args:
        bbox: (x1, y1, x2, y2). Only the y-range is used.
        lo, hi: band fractions of the bbox height (chest ≈ 0.15..0.55 from top).
        min_rows: if the band is thinner than this, fall back to the full bbox
            y-range (avoids starving the centroid on small/far people).
    """
    _, y1, _, y2 = bbox
    h = y2 - y1
    if h <= 0:
        return y1, y2
    band_y1 = y1 + int(lo * h)
    band_y2 = y1 + int(hi * h)
    if band_y2 - band_y1 < min_rows:
        return y1, y2
    return band_y1, band_y2


class PointEMA:
    """Exponential-moving-average smoother for a 3D point, with reset-on-loss."""

    def __init__(self, alpha: float = 0.5) -> None:
        """alpha in (0,1]: 1.0 = passthrough, lower = smoother/laggier."""
        self.alpha = alpha
        self._state: Optional[Tuple[float, float, float]] = None

    def reset(self) -> None:
        """Drop the smoothed state (call on target loss)."""
        self._state = None

    def update(
        self, point: Optional[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """Blend a new sample; first sample (or first after reset) passes through.

        A None sample returns None and leaves the stored state untouched.
        """
        if point is None:
            return None
        if self._state is None:
            self._state = (float(point[0]), float(point[1]), float(point[2]))
            return self._state
        a = self.alpha
        sx, sy, sz = self._state
        self._state = (
            a * point[0] + (1 - a) * sx,
            a * point[1] + (1 - a) * sy,
            a * point[2] + (1 - a) * sz,
        )
        return self._state
```

- [ ] **Step 4: Run to pass**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_centroid_smooth.py -v
```

Expected: all `TestTorsoBandMask` + `TestPointEMA` cases PASS. If
`test_degenerate_band_returns_full` fails on the `min_rows` boundary, the impl's
`< min_rows` check is correct — adjust the test bbox, not the impl, only if the
math genuinely disagrees; otherwise fix the impl.

- [ ] **Step 5: Commit the pure geometry helpers**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && \
  git add src/vision_track/vision_track/core/centroid_smooth.py \
          src/vision_track/test/test_centroid_smooth.py && \
  git commit -m "feat(vision_track): pure torso-band reducer + EMA point smoother

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Apply torso band inside `_calculate_centroid`**

In `person_track_node.py` `_calculate_centroid` (~324-388), restrict the mask/ROI
to the torso band **before** the existing Phase-0 robust reduction. After the bbox
clamp (line 348) and before extracting `roi_points` (line 354), add:

```python
        from vision_track.core.centroid_smooth import torso_band_mask
        if self.torso_band_enabled:
            yb1, yb2 = torso_band_mask((x1, y1, x2, y2),
                                       lo=self.torso_band_lo, hi=self.torso_band_hi)
            y1, y2 = yb1, yb2
            if y2 <= y1:
                return None
```

The downstream `roi_points = points[y1:y2, x1:x2]` etc. then operate on the band,
and the **Phase-0 robust median-x/y + z-outlier reduction** (replacing the
`np.mean`/`np.median` at 378-381) runs unchanged on the band's points. Do **not**
re-implement the reduction here — it is Phase 0's.

- [ ] **Step 7: Apply EMA on the published point**

In `__init__` (after `self._was_lost = False`, ~78), create the smoother:

```python
        from vision_track.core.centroid_smooth import PointEMA
        self._point_ema = PointEMA(alpha=self.centroid_ema_alpha)
```

In `_handle_tracked_frame`, after `position` is computed (~653) and before it is
written into `feedback.target_position.point`, smooth it:

```python
        if position is not None:
            sx, sy, sz = self._point_ema.update((position.x, position.y, position.z))
            position.x, position.y, position.z = float(sx), float(sy), float(sz)
            self.tracker.operator_last_depth_m = float(position.z)  # Task-2 plumb uses smoothed z
```

In `_handle_lost_frame` (first lost tick, ~765) and `_cleanup_tracking` (~802),
reset the smoother so a stale point does not bleed across a loss:

```python
        self._point_ema.reset()
```

- [ ] **Step 8: Declare/load the geometry params**

In `_declare_parameters`:

```python
        self.declare_parameter('centroid_ema_alpha', 0.5)
        self.declare_parameter('torso_band_enabled', True)
        self.declare_parameter('torso_band_lo', 0.15)
        self.declare_parameter('torso_band_hi', 0.55)
```

In `_load_parameters`:

```python
        self.centroid_ema_alpha = self.get_parameter('centroid_ema_alpha').value
        self.torso_band_enabled = self.get_parameter('torso_band_enabled').value
        self.torso_band_lo = self.get_parameter('torso_band_lo').value
        self.torso_band_hi = self.get_parameter('torso_band_hi').value
```

(Load these **before** `_init_tracker` so `__init__`'s `PointEMA(alpha=...)` sees
the value — `_load_parameters` already runs before `_init_tracker` at lines 67/107,
and the `PointEMA` is constructed at ~78 after `_load_parameters`. Verify order at
implementation time; if `_point_ema` is constructed before `centroid_ema_alpha` is
loaded, move its construction below `_load_parameters`.)

- [ ] **Step 9: Append geometry params to `config/default.yaml`**

```yaml
    # --- Phase 2: geometry robustness ---
    centroid_ema_alpha: 0.5          # 1.0 = no smoothing; lower = smoother
    torso_band_enabled: true
    torso_band_lo: 0.15              # chest band top, frac of bbox height
    torso_band_hi: 0.55              # chest band bottom, frac of bbox height
```

- [ ] **Step 10: Run the full pure suite**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest \
    test/test_centroid_smooth.py test/test_depth_gate.py test/test_lock_state_machine.py -v
```

Expected: all Task-1 + Task-2 + Task-3 cases PASS.

- [ ] **Step 11: Commit the geometry wiring**

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && \
  git add src/vision_track/vision_track/person_track_node.py \
          src/vision_track/config/default.yaml && \
  git commit -m "feat(vision_track): torso-band sampling + EMA on published 3D point

Restrict centroid to the chest band (layered on Phase-0 robust reduction); EMA
-smooth the published point, reset on loss.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Build + manual integration verification (T3/T4, operator-in-the-loop)

The node wiring imports `rclpy` and needs live cameras + a moving operator, so it
is **manual**, not in the pure suite.

**Files:** none (verification only).

- [ ] **Step 1: Build the package with the venv-aware wrapper**

```bash
cd /home/tinker/tk25_ws && \
  ./src/tk26_vision/scripts/build.sh --packages-select vision_track
```

Expected: build succeeds; `install/vision_track/.../person_track_node` shebang
points at `.venv-vision-main/bin/python`.

- [ ] **Step 2: T1 startup — node boots with Phase-2 params (manual)**

```bash
source /home/tinker/tk25_ws/install/setup.bash && \
  ros2 run vision_track person_track_server --ros-args \
    --params-file $(ros2 pkg prefix vision_track)/share/vision_track/config/default.yaml
```

Expected log lines: `Max recovery frames: 45`, no `Allow indefinite recovery`
line, `Person Track Node initialized successfully`. SIGTERM exits clean.

- [ ] **Step 3: T3 — provisional vs committed publish (manual, cameras + operator)**

With Orbbec + RealSense up (see `CAMERA_BRINGUP.md`), start tracking, then have the
operator step fully behind a pillar and re-emerge:

```bash
ros2 action send_goal /track_person tinker_vision_msgs_26/action/TrackPerson \
  "{return_rgb_img: false, debug: true, target_frame: ''}" --feedback
```

Expected observations:
- During occlusion: feedback `target_lost: true`; **no** new messages on
  `/target_points` (`ros2 topic echo /target_points` stays silent through the coast).
- On re-emergence: a brief provisional window (`target_lost` still true) then, after
  ~`commit_frames` frames, `target_lost: false` resumes and `/target_points`
  republishes. Re-lock should feel ≲ 1 s at ~12–15 Hz.
- A bystander crossing between robot and operator does **not** capture the lock
  (depth gate); the green TARGET box stays on the operator in the `debug` overlay.

- [ ] **Step 4: T4 — hard-lost bound (manual)**

Have the operator leave the scene entirely and not return. Expected: after
`max_recovery_frames` coast frames the action **aborts** with
`result.message` containing `hard-lost (recovery cap)` (previously it would coast
forever under `allow_indefinite_recovery`).

- [ ] **Step 5: T4 — lateral accuracy smoke (manual)**

Operator stands at a tape-measured lateral offset (e.g. 0.5 m left of optical
axis at 2.5 m range). `ros2 topic echo /target_points` x/y should track the
measured offset with visibly less frame-to-frame jitter than pre-Phase-2 (EMA),
and the torso band should keep the centroid off the legs/feet. Record the observed
value in `src/tk26_vision/DEV_NOTES.md` (manual, deferred to operator session).

---

## Acceptance

### Now-testable (this plan, pure unit tests under the venv)

- `test/test_lock_state_machine.py` — asymmetric hysteresis: fast provisional
  publish only above the HIGH bar + distinctiveness margin + depth-consistency;
  `target_lost=True` held through the coast; bounded hard-lost at
  `max_recovery_frames`; deterministic reacquire bound = `commit_frames`. The
  reacquire-vs-false-target trade is asserted with crafted sequences
  (`TestAsymmetricHysteresis`).
- `test/test_depth_gate.py` — `is_depth_consistent` / `should_reject_candidate`
  reject toward-camera jumps beyond `crosser_depth_jump_m`, pass farther/unknown
  /invalid depths; `roi_median_depth` excludes zero + out-of-range pixels.
- `test/test_centroid_smooth.py` — `torso_band_mask` selects/clamps the chest band
  with a full-bbox fallback; `PointEMA` blends, passes the first sample through,
  resets on loss, and survives a `None` gap without corrupting state.
- Build (`build.sh --packages-select vision_track`) succeeds; T1 startup confirms
  the node boots with the new params and no `allow_indefinite_recovery` line.

Run all three pure suites:

```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest \
    test/test_lock_state_machine.py test/test_depth_gate.py test/test_centroid_smooth.py -v
```

### Arena-deferred (until Orbbec arena recordings + ptbench bags exist)

- `reacquire_latency_s` median **≤ 1.0** on the `occlusion_reentry` scenario.
- `false_target_rate` **≤ 0.05** (no provisional publishes while the operator is
  absent — enforced by the FSM holding `target_lost=True` and the node gating
  `/target_points` on `not target_lost`).
- `pos_error_lateral_m` median **≤ 0.25** (torso band + EMA), scored against
  `centroid_field` per Phase 0.
- **No new `wrong_lock_episodes` vs Phase 1** — the aggressive recovery must not
  re-introduce wrong locks; this is the headline risk and the reason Phase 2
  depends on Phase 1 being merged. Validated on `cml_crossing` +
  `lookalike_distractors` bags.

These arena numbers cannot be confirmed without recordings; per
`person-tracker-benchmark-strategy`, academic ReID/MOT sets are tuning knobs, never
gates. The manual T3/T4 steps in Task 4 are the interim, operator-in-the-loop
confirmation.
