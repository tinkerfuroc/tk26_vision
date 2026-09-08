# track_web idle/init camera preview + color-format normalization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The dashboard shows the live camera image while NO goal is running and during the init search — and every frame consumer in `person_track_node` handles the camera's REAL wire format (`rgb8` on the live Orbbec, verified 2026-06-07; previously assumed `bgr8`, i.e. the tracker ran channel-swapped on-robot).

**Architecture:** (1) A pure, duck-typed `decode_color_msg()` (`core/color_decode.py`) normalizes any supported color Image msg to BGR — TDD'd with fakes, no ROS import. (2) `person_track_node` uses it in `_get_latest_data`, gains a generalized phase state tick (`initializing` / `idle`), a raw-frame debug publisher used by the init branch, and a 10 Hz idle timer that non-consumingly reads the frame cache so the dashboard stays alive between goals. (3) The webui renders the `idle`/`initializing` phases with a neutral badge.

**Tech Stack:** numpy/cv2 (pure decode), rclpy timer, vanilla JS. No interface changes, no msgs rebuild.

**Branch:** `feat/track-web-idle-video` (from `dev` @ f62d7c9). Spec addendum lands as §8 of `docs/superpowers/specs/2026-06-07-track-web-dashboard-design.md` (Task 3).

**Verified live facts (2026-06-07):** Orbbec `/camera/color/image_raw` → `encoding: rgb8`, `step == width*3` (no padding), ~30 Hz with the CAMERA_BRINGUP profile. `person_track_node:832` comment "already bgr8 on the wire" is WRONG for this camera; the loop's `cvtColor(..., COLOR_BGR2RGB)` therefore fed the tracker actual-BGR. The chain after normalization: `rgb_img` (BGR by contract) → `rgb_frame = cvtColor(BGR2RGB)` (true RGB to tracker) → `_draw_debug_info`/`cv2_to_imgmsg(...,'bgr8')` (true BGR out) → track_web imencode (BGR in, correct JPEG) → gallery thumbs RGB→`COLOR_RGB2BGR`→imencode (correct).

**Conventions:** identical to `2026-06-07-track-web-dashboard.md` (explicit `git add` paths only — user WIP in tree; `VENV=/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python`; pure tests via `PYTHONPATH="$(pwd)"`; tkbuild; clean-env (`env -i`) sourced checks because of the stale etot_ws overlay; Apache header on new test files from `test/test_reid_batch.py:1-13`; flake8 99; commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`).

---

### Task 1: pure `decode_color_msg()` (TDD)

**Files:**
- Create: `src/vision_track/vision_track/core/color_decode.py`
- Test: `src/vision_track/test/test_color_decode.py` (create)

- [ ] **Step 1: failing test** (header from test_reid_batch.py lines 1-13, then):

```python
"""decode_color_msg: normalize rgb8/bgr8 color Image msgs to BGR ndarrays."""
from types import SimpleNamespace

import numpy as np

from vision_track.core.color_decode import decode_color_msg


def _msg(encoding, data, w=2, h=1, step=None):
    return SimpleNamespace(encoding=encoding, width=w, height=h,
                           step=w * 3 if step is None else step,
                           data=bytes(data))


def test_bgr8_passthrough():
    img, err = decode_color_msg(_msg("bgr8", [1, 2, 3, 4, 5, 6]))
    assert err is None
    assert img.shape == (1, 2, 3)
    assert img[0, 0].tolist() == [1, 2, 3]          # untouched


def test_rgb8_channel_swap():
    img, err = decode_color_msg(_msg("rgb8", [10, 20, 30, 40, 50, 60]))
    assert err is None
    assert img[0, 0].tolist() == [30, 20, 10]       # R<->B swapped to BGR
    assert img[0, 1].tolist() == [60, 50, 40]


def test_unsupported_encoding_rejected():
    img, err = decode_color_msg(_msg("yuv422", [0] * 6))
    assert img is None and "yuv422" in err


def test_padded_step_rejected():
    img, err = decode_color_msg(_msg("rgb8", [0] * 8, step=8))
    assert img is None and "step" in err


def test_short_buffer_rejected():
    img, err = decode_color_msg(_msg("bgr8", [1, 2, 3]))
    assert img is None and err
```

- [ ] **Step 2: run, confirm FAIL** (ModuleNotFoundError): `cd /home/tinker/tk25_ws/src/tk26_vision/src/vision_track && PYTHONPATH="$(pwd)" $VENV -m pytest test/test_color_decode.py -v`

- [ ] **Step 3: implement** `src/vision_track/vision_track/core/color_decode.py`:

```python
"""Normalize color Image messages to BGR ndarrays.

The Orbbec publishes ``rgb8`` (verified live 2026-06-07); other drivers publish
``bgr8``. Every cv2-side consumer in this package works in BGR, so decode +
normalize HERE, once. Duck-typed (needs only encoding/width/height/step/data)
so it unit-tests without ROS.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def decode_color_msg(msg) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Color Image msg -> (BGR HxWx3 uint8 array, None) or (None, reason)."""
    if msg.step != msg.width * 3:
        return None, f"unexpected step {msg.step} for width {msg.width}"
    try:
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
    except Exception as exc:
        return None, f"decode failed: {exc}"
    if msg.encoding == "bgr8":
        return buf, None
    if msg.encoding == "rgb8":
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR), None
    return None, f"unsupported color encoding {msg.encoding!r}"
```

- [ ] **Step 4: run, confirm PASS** (5 tests). flake8 (99) exit=0 on both files.
- [ ] **Step 5: commit:**
```bash
git add src/vision_track/vision_track/core/color_decode.py src/vision_track/test/test_color_decode.py
git commit -m "feat(vision_track): encoding-aware color decode (Orbbec is rgb8)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: node integration — normalized decode + idle/init preview

**Files:**
- Modify: `src/vision_track/vision_track/person_track_node.py`

READ each region first; anchors verified 2026-06-07.

- [ ] **Step 1: `_get_latest_data` uses the normalizer.** Replace the inline decode (~line 829-838, the `try: rgb_img = np.frombuffer(...)` block and its misleading "already bgr8 on the wire" comment) with:

```python
        # Normalize the wire format (Orbbec = rgb8, others = bgr8) to BGR once,
        # here — every downstream consumer (tracker feed via BGR2RGB, debug
        # draw/publish, vision logger) assumes BGR.
        rgb_img, err = decode_color_msg(rgb_msg)
        if rgb_img is None:
            self.get_logger().warn(f'color frame dropped: {err}',
                                   throttle_duration_sec=5.0)
            return None
```
  Import `from vision_track.core.color_decode import decode_color_msg` next to the other `vision_track.core` imports.

- [ ] **Step 2: generalize the phase state tick.** Rename `_publish_init_debug_state()` → `_publish_phase_debug_state(self, phase: str)`; same body but `state["fsm_state"] = phase` and additionally blank the per-frame fields that may be stale from a previous goal:

```python
            state["fsm_state"] = phase
            state["candidates"] = []      # may be stale from a previous goal;
            state["best_sim"] = None      # no live click targets outside the
            state["second_sim"] = None    # tracking loop
```
  Update its docstring ("'initializing' during goal init, 'idle' between goals"), the warn text (`f'{phase} debug state failed: ...'`), and the init-branch call site → `self._publish_phase_debug_state('initializing')`.

- [ ] **Step 3: raw-frame publisher + init-branch preview.** Add next to `_publish_phase_debug_state`:

```python
    def _publish_raw_debug_image(self, rgb_img):
        """Un-annotated BGR camera frame for the dashboard outside TRACKING."""
        if not (self.debug_image_enabled
                and self.debug_image_pub.get_subscription_count() > 0):
            return
        try:
            self.debug_image_pub.publish(
                self.bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8'))
        except Exception as exc:
            self.get_logger().warn(f'raw debug image failed: {exc}',
                                   throttle_duration_sec=5.0)
```
  In the init branch (where `_publish_phase_debug_state('initializing')` now sits), add `self._publish_raw_debug_image(rgb_img)` on the next line — the operator sees the scene while init hunts for a person.

- [ ] **Step 4: idle timer.** In `__init__` right after `self._last_gallery_version = -1`:

```python
        # Idle telemetry: between goals the tracking loop isn't running, so a
        # light timer keeps the dashboard alive (camera preview + 'idle' state)
        # when the debug params are on. The tick reads the frame cache WITHOUT
        # consuming it (never touches last_processed_seq) so it cannot race
        # the tracking loop.
        self._idle_last_seq = -1
        if self.debug_state_enabled or self.debug_image_enabled:
            self.idle_debug_timer = self.create_timer(0.1, self._idle_debug_tick)
```
  And the callback (after `_publish_raw_debug_image`):

```python
    def _idle_debug_tick(self):
        """Dashboard telemetry while NO goal is active (loop not running)."""
        if self.tracking_active:
            return  # the tracking loop owns telemetry during a goal
        self._publish_phase_debug_state('idle')
        if not (self.debug_image_enabled
                and self.debug_image_pub.get_subscription_count() > 0):
            return
        with self.lock_msg:
            pair = self.recent_sync_msg
            seq = self.frame_seq
        if pair is None or seq == self._idle_last_seq:
            return
        rgb_img, err = decode_color_msg(pair[0])
        if rgb_img is None:
            self.get_logger().warn(f'idle frame dropped: {err}',
                                   throttle_duration_sec=5.0)
            return
        self._idle_last_seq = seq
        self._publish_raw_debug_image(rgb_img)
```

- [ ] **Step 5: verify.** Clean-env sourced import check (`env -i ... import vision_track.person_track_node`) → `import OK`. Pure suite still green: `PYTHONPATH="$(pwd)" $VENV -m pytest test/ -q --ignore=test/test_flake8.py --ignore=test/test_pep257.py` → 157 passed (152 + 5 new), 4 skipped. flake8 (99): no NEW findings on edited ranges. `cd /home/tinker/tk25_ws && ./tkbuild tk26_vision --packages-select vision_track` → success.

- [ ] **Step 6: commit:**
```bash
git add src/vision_track/vision_track/person_track_node.py
git commit -m "feat(vision_track): idle/init camera preview + normalized color decode in node

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: webui phase badge + docs

**Files:**
- Modify: `src/vision_track/webui/app.js`
- Modify: `src/vision_track/readme.md`
- Modify: `docs/superpowers/specs/2026-06-07-track-web-dashboard-design.md` (append §8)
- Modify: `DEV_NOTES.md` (tk26_vision root)

- [ ] **Step 1: badge phase handling.** In `renderState` (app.js), replace the badge block:

```javascript
  const badge = $("reacq-badge");
  if (s.fsm_state === "idle" || s.fsm_state === "initializing") {
    badge.textContent = s.fsm_state.toUpperCase();
    badge.className = "reacq";                       // neutral gray
  } else {
    const [label, cls] = REACQ[s.reacquisition_state] || ["?", ""];
    badge.textContent = label;
    badge.className = "reacq " + cls;
  }
```
  (Keep the `const [label, cls] = ...` line OUT of the top of the function if it was there — the transition log lines below still reference `REACQ[...]` directly and are unchanged.) `node --check` after editing.

- [ ] **Step 2: docs.**
  - readme.md track_web section: one paragraph — the dashboard now shows the live camera image whenever the bench launch is up (badge `IDLE`), and during the init search (badge `INITIALIZING`); annotated overlay appears once tracking locks. Changelog line (2026-06-07, append-only).
  - Spec: append `## §8 Idle/init preview + color normalization (addendum, 2026-06-07)` — 5-8 lines: idle timer (non-consuming cache read) + init-branch raw frame + phase states `idle`/`initializing` (candidates blanked) + `decode_color_msg` normalizing rgb8/bgr8→BGR at the single decode point, motivated by the live `rgb8` finding.
  - DEV_NOTES: add an entry `## 2026-06-07 — Orbbec publishes rgb8 (NOT bgr8) — tracker ran channel-swapped on-robot; fixed at the decode point` recording: the live probe evidence (encoding rgb8, step==width*3, 30Hz after CAMERA_BRINGUP profile); that offline benchmarks used true RGB so on-robot now matches validated-offline behavior; that the dashboard/gallery/feedback images were red-blue swapped before the fix; and a **flagged follow-up**: audit the OTHER raw-color consumers (`waving_person_server` MediaPipe/VLM crops, `object_detection_*`, `kimi_api` feature crops, `follow_head`) for the same bgr8 assumption — out of scope here.

- [ ] **Step 3: final verify.** Full suite (157 passed, 4 skipped); `node --check`; tkbuild; clean-env `ros2 launch vision_track track_web_bench.launch.py -s` still exit 0.

- [ ] **Step 4: commit:**
```bash
git add src/vision_track/webui/app.js src/vision_track/readme.md docs/superpowers/specs/2026-06-07-track-web-dashboard-design.md DEV_NOTES.md
git commit -m "feat(vision_track): idle/initializing badge + rgb8 finding docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Validation reality

Pure decode is fully TDD'd (5 fakes incl. the channel-swap assertion). Node wiring + timer verified at import/build level; the LIVE check (idle preview visible before Start, correct colors — operator confirms reds are red) needs the user's running bench: restart `track_web_bench.launch.py` after the rebuild and look at the page. Colors can be verified objectively by holding a known-red object in view.

## Self-Review

- **Coverage:** user ask #1 (frame during init search) → Task 2 Step 3; #2 (image when action not started) → Task 2 Step 4 (+ Task 3 badge); #3 (verify rgb/bgr consistency) → live probe (rgb8 finding) + Task 1 normalizer + Task 2 Step 1 + chain doc in the header + DEV_NOTES follow-up flag for non-tracker consumers.
- **Placeholders:** none; all code complete.
- **Type consistency:** `decode_color_msg(msg) -> (ndarray|None, str|None)` used identically in Tasks 1/2; `_publish_phase_debug_state(phase)` call sites consistent ('initializing'/'idle'); JS reads `fsm_state` values exactly as the node sets them.
