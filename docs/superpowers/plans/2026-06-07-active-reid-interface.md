# Active Re-ID Interface (Spec B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the vision stack the *capability* for active (call-out) re-identification — a reacquisition-state signal the behaviour tree can act on, and a gallery-preserving re-seed that re-locks the tracker on a self-identified (raise-hand) operator — precision-safe by construction.

**Architecture:** Three vision-side pieces + a documented BT contract. (1) A `uint8 reacquisition_state` field on `TrackPerson` action feedback, computed by a pure hysteresis helper. (2) A new `ReseedTarget` service on the person-track node that, under the existing `lock_tracker`, re-locks the running tracker on a provided bbox **without** wiping the multi-view gallery/registry (preserving identity + appending the fresh confirmed view). (3) A `waving_boxes` field added to `DetectWaving` so the raise-hand detector's output can drive the re-seed. BT policy (when to call out / accept the penalty) is out of scope.

**Tech Stack:** ROS2 Humble (rclpy, rosidl), Python 3.10, pytest. Spec: `docs/superpowers/specs/2026-06-06-active-reid-interface-design.md`. Depends on Spec A (multi-view gallery) — already merged on this branch.

**Conventions:**
- `VENV=/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python`.
- Pure-python unit tests: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/<file> -v`.
- Interface (msg/srv/action) changes require **rebuilding `tinker_vision_msgs_26`**: `./src/tk26_vision/scripts/build.sh --packages-select tinker_vision_msgs_26` from the workspace root `/home/tinker/tk25_ws`, then `source install/setup.bash`. NOTE: this regenerates types the whole workspace consumes — after it, the generated python types import as `from tinker_vision_msgs_26.srv import ReseedTarget`, `from tinker_vision_msgs_26.action import TrackPerson`.
- New source files: docstring-first, pep257-style docstrings, flake8-clean (repo-wide `test_flake8`/`test_pep257` are pre-existing red — only avoid NEW errors). New TEST files carry the `# Copyright 2026 Tinker` Apache-2.0 header (see `test/test_reid_batch.py`).
- Commit-message trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Interface definitions + msgs rebuild

**Files:**
- Modify: `src/tinker_vision_msgs_26/action/TrackPerson.action`
- Modify: `src/tinker_vision_msgs_26/srv/DetectWaving.srv`
- Create: `src/tinker_vision_msgs_26/srv/ReseedTarget.srv`
- Modify: `src/tinker_vision_msgs_26/CMakeLists.txt` (rosidl list)
- Test: `src/vision_track/test/test_active_reid_interfaces.py` (create)

- [ ] **Step 1: Add the feedback field to `TrackPerson.action`.** The file has three `---`-separated sections (goal / result / feedback). In the **feedback** section (currently starting `bool target_lost`), add at the end:

```
# Reacquisition state for active-reID escalation (Spec B):
#   0 TRACKING, 1 PASSIVE_REACQUIRING, 2 NEEDS_ACTIVE_HELP
uint8 reacquisition_state
uint8 REACQ_TRACKING=0
uint8 REACQ_PASSIVE=1
uint8 REACQ_NEEDS_HELP=2
```
(Constants in an `.action` feedback block are allowed; they generate as class attributes on the Feedback type.)

- [ ] **Step 2: Add `waving_boxes` to `DetectWaving.srv`.** In the response section (after `geometry_msgs/PointStamped[] waving_persons`), add:

```
sensor_msgs/RegionOfInterest[] waving_boxes  # 1:1 with waving_persons; image-space boxes for re-seed
```

- [ ] **Step 3: Create `src/tinker_vision_msgs_26/srv/ReseedTarget.srv`:**

```
# Re-lock the running person tracker on an externally-confirmed target
# (e.g. the raise-hand operator), preserving the multi-view gallery.
sensor_msgs/RegionOfInterest bbox   # target box in the current color frame
string frame_id                     # color frame the bbox is expressed in (sanity/logging)
---
bool success
int32 target_track_id               # the (re)locked track id, -1 on failure
string message
```

- [ ] **Step 4: Register the new srv in `CMakeLists.txt`.** In the `rosidl_generate_interfaces(${PROJECT_NAME}` block, add a line alphabetically near the other srvs:

```
  "srv/ReseedTarget.srv"
```
Also confirm `sensor_msgs` and `geometry_msgs` are in `find_package(...)` + `rosidl_generate_interfaces` deps (they are — `DetectWaving` already uses both). No dependency change needed.

- [ ] **Step 5: Build the msgs package.**

Run (from `/home/tinker/tk25_ws`): `./src/tk26_vision/scripts/build.sh --packages-select tinker_vision_msgs_26`
Expected: build succeeds. Then `source /home/tinker/tk25_ws/install/setup.bash`.

- [ ] **Step 6: Write + run an import test** (`src/vision_track/test/test_active_reid_interfaces.py`). This verifies the generated types exist and have the expected fields/constants:

```python
# <Apache-2.0 copyright header — copy from test/test_reid_batch.py>
"""Spec B interface smoke tests: generated types have the new fields/constants."""
import pytest


def test_trackperson_feedback_has_reacq_state():
    tp = pytest.importorskip("tinker_vision_msgs_26.action").TrackPerson
    fb = tp.Feedback()
    assert hasattr(fb, "reacquisition_state")
    assert tp.Feedback.REACQ_TRACKING == 0
    assert tp.Feedback.REACQ_PASSIVE == 1
    assert tp.Feedback.REACQ_NEEDS_HELP == 2


def test_reseed_target_srv_shape():
    srv = pytest.importorskip("tinker_vision_msgs_26.srv").ReseedTarget
    req, resp = srv.Request(), srv.Response()
    assert hasattr(req, "bbox") and hasattr(req, "frame_id")
    assert hasattr(resp, "success") and hasattr(resp, "target_track_id") and hasattr(resp, "message")


def test_detectwaving_has_waving_boxes():
    srv = pytest.importorskip("tinker_vision_msgs_26.srv").DetectWaving
    assert hasattr(srv.Response(), "waving_boxes")
```

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_active_reid_interfaces.py -v` (with `install/setup.bash` sourced so the generated types are importable).
Expected: 3 PASS (or skip cleanly if the workspace isn't sourced — but for this task it MUST be sourced and pass).

- [ ] **Step 7: Commit:**

```bash
git add src/tinker_vision_msgs_26/action/TrackPerson.action src/tinker_vision_msgs_26/srv/DetectWaving.srv src/tinker_vision_msgs_26/srv/ReseedTarget.srv src/tinker_vision_msgs_26/CMakeLists.txt src/vision_track/test/test_active_reid_interfaces.py
git commit -m "feat(msgs): TrackPerson reacquisition_state + ReseedTarget srv + DetectWaving waving_boxes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Pure reacquisition-state hysteresis helper

**Files:**
- Create: `src/vision_track/vision_track/core/reacq_state.py`
- Test: `src/vision_track/test/test_reacq_state.py`

- [ ] **Step 1: Write the failing test** (`test/test_reacq_state.py`):

```python
# <Apache-2.0 copyright header — copy from test/test_reid_batch.py>
"""Unit tests for the pure reacquisition-state hysteresis."""
from vision_track.core.reacq_state import (
    REACQ_TRACKING, REACQ_PASSIVE, REACQ_NEEDS_HELP, reacq_state,
)


def test_tracked_is_tracking():
    assert reacq_state(tracked=True, frames_lost=0, help_after=45) == REACQ_TRACKING
    assert reacq_state(tracked=True, frames_lost=999, help_after=45) == REACQ_TRACKING


def test_lost_within_window_is_passive():
    assert reacq_state(tracked=False, frames_lost=1, help_after=45) == REACQ_PASSIVE
    assert reacq_state(tracked=False, frames_lost=44, help_after=45) == REACQ_PASSIVE


def test_lost_past_window_needs_help():
    assert reacq_state(tracked=False, frames_lost=45, help_after=45) == REACQ_NEEDS_HELP
    assert reacq_state(tracked=False, frames_lost=200, help_after=45) == REACQ_NEEDS_HELP


def test_help_after_zero_escalates_immediately_when_lost():
    assert reacq_state(tracked=False, frames_lost=0, help_after=0) == REACQ_NEEDS_HELP
```

- [ ] **Step 2: Run, confirm FAIL** (ModuleNotFoundError). `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_reacq_state.py -v`.

- [ ] **Step 3: Implement** `src/vision_track/vision_track/core/reacq_state.py`:

```python
"""Pure reacquisition-state hysteresis for the active-reID escalation signal.

The tracker is the publish authority; this maps (tracked?, consecutive frames
lost) to an advisory state a consumer (behaviour tree) can act on. It NEVER
calls out itself — it only debounces "lost long enough that active help is
warranted" so the BT doesn't escalate (and incur a points penalty) prematurely.
"""
from __future__ import annotations

REACQ_TRACKING = 0
REACQ_PASSIVE = 1
REACQ_NEEDS_HELP = 2


def reacq_state(tracked: bool, frames_lost: int, help_after: int) -> int:
    """Map tracking status to a reacquisition state.

    Args:
        tracked: True if the target was matched/published this frame.
        frames_lost: consecutive frames since the target was last held.
        help_after: escalate to NEEDS_HELP once frames_lost reaches this.

    Returns:
        REACQ_TRACKING while held; REACQ_PASSIVE while lost but within the
        passive-recovery window; REACQ_NEEDS_HELP once lost >= help_after.
    """
    if tracked:
        return REACQ_TRACKING
    if frames_lost >= help_after:
        return REACQ_NEEDS_HELP
    return REACQ_PASSIVE
```

- [ ] **Step 4: Run, confirm PASS** (4 tests).

- [ ] **Step 5: Commit:**

```bash
git add src/vision_track/vision_track/core/reacq_state.py src/vision_track/test/test_reacq_state.py
git commit -m "feat(vision_track): pure reacquisition-state hysteresis helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Gallery-preserving re-seed on the tracker

**Files:**
- Modify: `src/vision_track/vision_track/yolo_tracker.py` (add `reseed_target` + a testable `_apply_reseed` core)
- Test: `src/vision_track/test/test_reseed_target.py`

**Context:** `initialize_tracking` (yolo_tracker.py ~417) RESETS `target_appearance=None` etc. then detects + selects the bbox match via `_find_best_match_iou(candidates, target_bbox)` and locks. The re-seed must do the same *selection* but PRESERVE `target_appearance`/`gallery`/`person_registry` (same operator, self-identified) and APPEND the fresh confirmed view, then re-arm the FSM. Split the YOLO-dependent detection from the testable state logic so the latter is unit-testable without a model.

- [ ] **Step 1: Write the failing test** (`test/test_reseed_target.py`) — exercises the pure `_apply_reseed` state logic with a fake detection + fake extractor, asserting the gallery is preserved + the fresh view appended + identity re-locked + FSM re-armed:

```python
# <Apache-2.0 copyright header — copy from test/test_reid_batch.py>
"""Gallery-preserving re-seed: _apply_reseed preserves gallery + re-locks."""
import numpy as np

from vision_track.core.tracking_types import TargetAppearance, TrackingResult, TrackerState
from vision_track.yolo_tracker import YOLOTracker


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


class _Lsm:
    def __init__(self): self.started = None
    def start(self, tid): self.started = tid


def _bare_tracker():
    t = YOLOTracker.__new__(YOLOTracker)          # bypass heavy __init__
    t.target_appearance = TargetAppearance(class_id=0, class_name="person")
    t.target_appearance.configure_gallery(enabled=True, size=6, novelty_max=0.99, score_mode="max")
    t.target_appearance.gallery.maybe_add(_v(1, 0))   # pre-existing identity view
    t.target_track_id = 3
    t.original_track_id = 3
    t.frames_lost = 40
    t.state = TrackerState.REIDENTIFYING
    t.lock_state_machine = _Lsm()
    return t


def test_apply_reseed_preserves_gallery_and_relocks():
    t = _bare_tracker()
    det = TrackingResult(track_id=9, bbox=(10, 10, 50, 120), mask=None,
                         confidence=0.9, class_id=0, class_name="person")
    fresh = _v(0, 1)                              # a new, distinct confirmed view
    tid = t._apply_reseed(det, fresh)
    assert tid == 9
    assert t.target_track_id == 9 and t.original_track_id == 9
    assert t.state == TrackerState.TRACKING
    assert t.frames_lost == 0
    # gallery preserved (still has the old view) AND the fresh view appended
    assert len(t.target_appearance.gallery) == 2
    # FSM re-armed on the new id
    assert t.lock_state_machine.started == 9


def test_apply_reseed_none_detection_fails():
    t = _bare_tracker()
    assert t._apply_reseed(None, _v(0, 1)) == -1
    assert t.target_track_id == 3                 # unchanged on failure
```

(Confirm `TrackingResult`/`TrackerState` import locations by reading `core/tracking_types.py` first; adjust the import if they live elsewhere.)

- [ ] **Step 2: Run, confirm FAIL** (`AttributeError: ... '_apply_reseed'`).

- [ ] **Step 3: Implement** in `yolo_tracker.py`:

```python
    def _apply_reseed(self, selected_result, fresh_reid_feature) -> int:
        """Re-lock onto an externally-confirmed detection, preserving identity.

        Unlike initialize_tracking (which resets appearance), this keeps the
        multi-view gallery + person registry (same operator, self-identified),
        appends the fresh confirmed view, re-locks the ids, clears the lost
        counter, and re-arms the lock FSM. Returns the locked track id, or -1
        if selected_result is None.
        """
        if selected_result is None:
            return -1
        self.target_track_id = selected_result.track_id
        self.original_track_id = selected_result.track_id
        self.target_class_id = selected_result.class_id
        self.frames_lost = 0
        self.state = TrackerState.TRACKING
        if self.target_appearance is not None and fresh_reid_feature is not None:
            self.target_appearance.gallery.maybe_add(fresh_reid_feature)
        if self.lock_state_machine is not None and self.original_track_id is not None:
            self.lock_state_machine.start(self.original_track_id)
        return self.target_track_id

    def reseed_target(self, frame, bbox, target_class: str = "person") -> int:
        """Detect on `frame`, match `bbox`, and re-lock preserving the gallery.

        Returns the locked track id, or -1 if no detection matches the bbox.
        """
        results = self.track(frame, persist=True)
        if not results:
            return -1
        candidates = [r for r in results
                      if target_class is None or r.class_name.lower() == target_class.lower()]
        if not candidates:
            candidates = results
        best = self._find_best_match_iou(candidates, bbox)
        if best is None:
            return -1
        fresh = None
        if self.appearance_extractor is not None:
            feats = self.appearance_extractor.extract_features_batch(
                frame, [best.bbox], [best.mask], [best.class_id])
            if feats and feats[0] and "reid" in feats[0]:
                fresh = feats[0]["reid"]
        return self._apply_reseed(best, fresh)
```

Confirm `_find_best_match_iou`, `self.appearance_extractor`, `TrackerState`, and `extract_features_batch` names/signatures by reading the file before editing; if any differs, adapt and note it.

- [ ] **Step 4: Run, confirm PASS** (2 tests). Also regression: `$VENV -m pytest test/test_reid_gallery.py test/test_target_deep_score.py -q`.

- [ ] **Step 5: Commit:**

```bash
git add src/vision_track/vision_track/yolo_tracker.py src/vision_track/test/test_reseed_target.py
git commit -m "feat(vision_track): gallery-preserving reseed_target (active re-ID re-lock)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Wire `reacquisition_state` into the action feedback

**Files:**
- Modify: `src/vision_track/vision_track/person_track_node.py` (`_declare_parameters`/`_load_parameters`, `_handle_tracked_frame`, `_handle_lost_frame`)
- Test: covered by the pure helper (Task 2) + spec review of the wiring (the node loop needs a live tracker/cameras, not unit-testable here).

- [ ] **Step 1: Add the ROS param.** In `_declare_parameters`, add `self.declare_parameter('active_help_after_frames', 45)`; in `_load_parameters`, read `self.active_help_after_frames = int(self.get_parameter('active_help_after_frames').value)`. (Match the existing declare/load pattern.)

- [ ] **Step 2: Import the helper** at the top of `person_track_node.py`:

```python
from vision_track.core.reacq_state import reacq_state
```

- [ ] **Step 3: Set the feedback field in the tracked path.** In `_handle_tracked_frame`, just before `goal_handle.publish_feedback(feedback)`, add:

```python
        feedback.reacquisition_state = reacq_state(
            tracked=True, frames_lost=0, help_after=self.active_help_after_frames)
```

- [ ] **Step 4: Set it in the lost path.** In `_handle_lost_frame`, just before its `goal_handle.publish_feedback(feedback)`, add:

```python
        feedback.reacquisition_state = reacq_state(
            tracked=False, frames_lost=int(getattr(self.tracker, 'frames_lost', 0)),
            help_after=self.active_help_after_frames)
```

- [ ] **Step 5: Verify import + param load don't break node construction.** Run the static import check: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -c "import vision_track.person_track_node"` (expect no error). The full node behaviour is verified at T1/T2 integration, not unit level.

- [ ] **Step 6: Commit:**

```bash
git add src/vision_track/vision_track/person_track_node.py
git commit -m "feat(vision_track): publish reacquisition_state in TrackPerson feedback

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `ReseedTarget` service on the node

**Files:**
- Modify: `src/vision_track/vision_track/person_track_node.py` (service creation + handler)
- Test: spec review of the handler (ROS service; needs runtime). A light logic test via a stub is added where feasible.

- [ ] **Step 1: Create the service.** In `_init_action_server` (or a new `_init_services` called from `__init__`), after the action server, add (import `ReseedTarget` + `ReentrantCallbackGroup` at top if not present):

```python
        from tinker_vision_msgs_26.srv import ReseedTarget
        self.reseed_srv = self.create_service(
            ReseedTarget, '~/reseed_target', self._reseed_callback,
            callback_group=ReentrantCallbackGroup())
```

- [ ] **Step 2: Implement the handler:**

```python
    def _reseed_callback(self, request, response):
        """Re-lock the tracker on request.bbox, preserving the gallery.

        Runs under lock_tracker (serialized with the tracking loop's
        tracker.update). Uses the latest cached color frame to match the bbox.
        """
        roi = request.bbox
        bbox = (int(roi.x_offset), int(roi.y_offset),
                int(roi.x_offset + roi.width), int(roi.y_offset + roi.height))
        data = self._get_latest_data()
        if not data or data is True:
            response.success = False
            response.target_track_id = -1
            response.message = 'no camera frame available'
            return response
        rgb_img = data[0]
        import cv2
        rgb_frame = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        with self.lock_tracker:
            tid = self.tracker.reseed_target(rgb_frame, bbox, target_class='person')
        response.success = tid >= 0
        response.target_track_id = int(tid)
        response.message = 'reseeded' if tid >= 0 else 'no detection matched bbox'
        return response
```

Confirm `_get_latest_data`'s return shape (it returns `(rgb_img, rgb_msg, depth_msg, intrinsic)` or `None`/`False`) before finalizing the guard; adapt the unpacking to match.

- [ ] **Step 3: Verify import/construction.** `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -c "import vision_track.person_track_node"` (with `install/setup.bash` sourced so `ReseedTarget` imports). Expect no error.

- [ ] **Step 4: Commit:**

```bash
git add src/vision_track/vision_track/person_track_node.py
git commit -m "feat(vision_track): ReseedTarget service (gallery-preserving active re-lock)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Populate `waving_boxes` in the waving detector

**Files:**
- Modify: `src/tk_vision_specialized/.../waving_person_server.py` (the `DetectWaving` handler)
- Test: spec review (the server needs MediaPipe/cameras at runtime).

- [ ] **Step 1: Locate the handler.** `grep -rn "waving_persons" src/tk_vision_specialized` to find where the response is built (it appends `PointStamped` per waver). Read that section.

- [ ] **Step 2: Populate `waving_boxes` 1:1.** Wherever a waver's `PointStamped` is appended to `response.waving_persons`, also append a `sensor_msgs/RegionOfInterest` for that waver's image-space bbox to `response.waving_boxes`. Each waver already has a bbox (MediaPipe person box or the YOLO/VLM box used for its centroid — read the code to find the existing bbox variable). Build:

```python
        from sensor_msgs.msg import RegionOfInterest
        roi = RegionOfInterest()
        roi.x_offset = int(x1); roi.y_offset = int(y1)
        roi.width = int(x2 - x1); roi.height = int(y2 - y1)
        roi.do_rectify = False
        response.waving_boxes.append(roi)
```
Ensure `waving_boxes` and `waving_persons` stay index-aligned on every code path that appends a waver (heuristic AND VLM-fallback paths).

- [ ] **Step 3: Verify import/construction.** `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -c "import sensor_msgs.msg"` and a static import of the server module (with workspace sourced). Expect no error.

- [ ] **Step 4: Commit:**

```bash
git add src/tk_vision_specialized
git commit -m "feat(tk_vision_specialized): populate DetectWaving waving_boxes for re-seed

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: BT consumer contract doc + DEV_NOTES

**Files:**
- Modify: `src/vision_track/README.md` (or create `src/vision_track/docs/active_reid.md`)
- Modify: `src/tk26_vision/DEV_NOTES.md`

- [ ] **Step 1: Document the BT contract** (out-of-scope-to-implement, defined for tk25_decision). Add a short section describing the loop: `feedback.reacquisition_state == NEEDS_ACTIVE_HELP` → BT speaks "please raise your hand / wait" + accepts the points penalty → calls `waving_person_server` (`DetectWaving`) → picks the best `waving_boxes[i]` → calls `~/reseed_target` (`ReseedTarget`) → tracking resumes (state returns to TRACKING). Note the precision-safety rationale (operator self-identifies) and that `active_help_after_frames` (default 45) is the escalation debounce.

- [ ] **Step 2: Add a DEV_NOTES entry** recording: Spec B vision-side capability shipped; active end-to-end (call-out → raise-hand → reseed) validation is **deferred to on-robot** (needs operator + BT); unit/import tests cover the pieces that don't need hardware.

- [ ] **Step 3: Commit:**

```bash
git add src/vision_track/README.md src/tk26_vision/DEV_NOTES.md
git commit -m "docs(vision_track): active re-ID BT contract + on-robot deferral note

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Validation reality (read before executing)

The **active path** (call-out → raise-hand → re-seed) cannot be validated this cycle — it needs an operator, the BT (out of scope), and the robot. What IS verified now: interface generation (Task 1 import test), the pure hysteresis (Task 2), the gallery-preserving re-seed state logic (Task 3 `_apply_reseed` unit test), and that the node + waving server still import/construct with the new wiring (Tasks 4–6 static checks). The `reseed_target` YOLO-detection path, the service under a live executor, and `waving_boxes` population are **spec-reviewed + import-checked here, behaviour-validated on-robot later** (documented in Task 7). This matches Spec B's stated scope (vision-side capability; BT policy + on-robot acceptance deferred).

## Self-Review

**Spec coverage** (`2026-06-06-active-reid-interface-design.md`):
- §1 reacquisition-state feedback (TRACKING/PASSIVE/NEEDS_HELP) + hysteresis → Task 1 (field) + Task 2 (helper) + Task 4 (wiring). ✓
- §2 `ReseedTarget` srv (gallery-preserving, bbox→detection, keep gallery+registry, append fresh view, re-arm FSM, idempotent) → Task 1 (srv) + Task 3 (tracker) + Task 5 (service). ✓
- §3 `DetectWaving` `waving_boxes` seam → Task 1 (field) + Task 6 (population). ✓
- §4 BT consumer contract (out of scope, documented) → Task 7. ✓
- Error handling (bbox matches no detection → success=false, tracker unchanged) → Task 3 (`_apply_reseed(None)→-1`, `reseed_target` no-match→-1) + Task 5 (response). ✓
- Interface/version rebuild flagged → Task 1 Step 5. ✓

**Placeholder scan:** none — each step has concrete code/commands. Where hardware blocks a unit test, that's stated explicitly with the static/spec-review substitute (not a hidden gap).

**Type consistency:** `reacq_state(tracked, frames_lost, help_after)` + constants `REACQ_TRACKING/PASSIVE/NEEDS_HELP` consistent across Tasks 2/4. `_apply_reseed(selected_result, fresh_reid_feature)->int` and `reseed_target(frame, bbox, target_class)->int` consistent across Tasks 3/5. `ReseedTarget` request `bbox`/`frame_id`, response `success`/`target_track_id`/`message` consistent across Tasks 1/5. `reacquisition_state` field + constants consistent across Tasks 1/4.

**Note for implementer:** Tasks build in order (Task 1's generated types are imported by Tasks 4–6). Confirm exact symbol locations (`TrackingResult`, `TrackerState`, `_find_best_match_iou`, `appearance_extractor`, `_get_latest_data` shape) by reading before editing; the snippets are accurate as of 2026-06-07 but adapt to the live code and report NEEDS_CONTEXT if materially different.
