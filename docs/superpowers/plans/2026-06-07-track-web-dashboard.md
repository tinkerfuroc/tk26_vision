# `track_web` Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live web dashboard (`ros2 run vision_track track_web`) that visualizes the person-tracker's state and acts as the human-in-the-loop test bench for the Spec-B active re-ID loop (start/stop goal, click-to-reseed, DetectWaving trigger).

**Architecture:** Three layers. (1) Param-gated debug outputs on `person_track_node` (`~/debug_state` + `~/debug_gallery` JSON Strings, `~/debug_image` annotated frames) fed by a pure `build_debug_state()` and a thumb-retaining `ReIDGallery`. (2) A `track_web` ROS2 node + FastAPI app (calib_web threading model) that bridges those topics + the `TrackPerson` action + `ReseedTarget`/`DetectWaving` services to HTTP/WS/MJPEG. (3) A vanilla-JS `webui/` (video-dominant + right rail).

**Tech Stack:** ROS2 Humble (rclpy), FastAPI + uvicorn, vanilla JS, pytest (+ httpx TestClient). **No msgs changes — no `tinker_vision_msgs_26` rebuild anywhere in this plan.**

**Spec:** `docs/superpowers/specs/2026-06-07-track-web-dashboard-design.md`

**Conventions (read first):**
- Work in the MAIN checkout `/home/tinker/tk25_ws/src/tk26_vision`, branch `feat/track-web-dashboard`. The working tree carries the user's unrelated WIP (`foundation_stereo`, `kimi_api`, `object_detection_generalist` files) — **NEVER `git add -A` / `git add .`; always add explicit paths.**
- `VENV=/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python`.
- Pure-python tests (no ROS sourcing): `cd /home/tinker/tk25_ws/src/tk26_vision/src/vision_track && PYTHONPATH="$(pwd)" $VENV -m pytest test/<file> -v` — single bash call (Bash state does not persist).
- Sourced import checks (node code importing rclpy/msgs): `source /home/tinker/tk25_ws/install/setup.bash && cd .../src/vision_track && PYTHONPATH="$(pwd):$PYTHONPATH" $VENV -c "..."` — APPEND to PYTHONPATH, never replace (replacing drops the ROS install path).
- Builds: `cd /home/tinker/tk25_ws && ./tkbuild tk26_vision --packages-select vision_track` (user-mandated wrapper; do NOT use plain colcon or scripts/build.sh).
- New TEST files carry the Apache-2.0 header copied verbatim from `src/vision_track/test/test_reid_batch.py` lines 1-13. New source files: docstring-first. flake8-clean at `--max-line-length=99` on files you touch (repo-wide lint is pre-existing red; only NEW errors matter).
- Commit per task, message trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `ReIDGallery` thumb retention + version counter (pure)

**Files:**
- Modify: `src/vision_track/vision_track/core/reid_gallery.py`
- Test: `src/vision_track/test/test_gallery_thumbs.py` (create)

Current internals (verified 2026-06-07): views live in `self._views: List[np.ndarray]`; `maybe_add(feature)` appends (anchor pinned at index 0), novelty-gates at `novelty_max`, and calls `_evict_most_redundant()` which `self._views.pop(drop)`s a non-anchor index; `clear()` empties.

- [ ] **Step 1: Write the failing test:**

```python
# <Apache-2.0 header — copy verbatim from test/test_reid_batch.py lines 1-13>
"""Gallery thumb retention stays in lockstep with views; version counts changes."""
import numpy as np

from vision_track.core.reid_gallery import ReIDGallery


def _v(i, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    a[i] = 1.0
    return a


def test_thumbs_lockstep_and_version():
    g = ReIDGallery(enabled=True, size=3, novelty_max=0.99)
    assert g.version == 0 and g.thumbs == []
    assert g.maybe_add(_v(0), thumb="t0")          # anchor
    assert g.maybe_add(_v(1), thumb="t1")
    assert g.maybe_add(_v(2), thumb="t2")
    assert g.version == 3
    assert g.thumbs == ["t0", "t1", "t2"] and len(g) == 3
    # 4th admit evicts a non-anchor view; thumbs must follow the same index
    assert g.maybe_add(_v(3), thumb="t3")
    assert g.version == 4
    assert len(g) == 3 and len(g.thumbs) == 3
    assert g.thumbs[0] == "t0"                      # anchor thumb pinned
    assert "t3" in g.thumbs                         # newcomer survived the evict


def test_rejected_add_changes_nothing():
    g = ReIDGallery(enabled=True, size=3, novelty_max=0.99)
    g.maybe_add(_v(0), thumb="t0")
    v_before = g.version
    assert not g.maybe_add(_v(0), thumb="dup")      # novelty reject (cos=1.0)
    assert g.version == v_before and g.thumbs == ["t0"]
    assert not g.maybe_add(None, thumb="bad")       # invalid feature
    assert g.version == v_before


def test_thumbless_add_and_clear():
    g = ReIDGallery(enabled=True, size=3, novelty_max=0.99)
    g.maybe_add(_v(0))                              # thumb defaults to None
    assert g.thumbs == [None]
    v = g.version
    g.clear()
    assert len(g) == 0 and g.thumbs == [] and g.version == v + 1
    g.clear()                                       # clearing empty: no bump
    assert g.version == v + 1
```

- [ ] **Step 2: Run, confirm FAIL** (`AttributeError: ... 'version'` / unexpected kwarg `thumb`):
`cd /home/tinker/tk25_ws/src/tk26_vision/src/vision_track && PYTHONPATH="$(pwd)" $VENV -m pytest test/test_gallery_thumbs.py -v`

- [ ] **Step 3: Implement.** In `reid_gallery.py`:
  - `__init__`: add `self._thumbs: List[object] = []` and `self.version: int = 0` after `self._views = []`.
  - Add a read-only property:
```python
    @property
    def thumbs(self) -> List[object]:
        """Per-view thumbnail payloads (opaque; index-aligned with views)."""
        return list(self._thumbs)
```
  - `clear()` becomes:
```python
    def clear(self) -> None:
        """Drop all views (e.g. on tracker reset)."""
        if self._views:
            self.version += 1
        self._views = []
        self._thumbs = []
```
  - `maybe_add` gains `thumb=None` and keeps `_thumbs` in lockstep (note the evict now returns the dropped index):
```python
    def maybe_add(self, feature: Optional[np.ndarray], thumb: object = None) -> bool:
        """Admit an (already quality-gated) feature if novel. Return admitted.

        ``thumb`` is an opaque per-view payload (e.g. an RGB crop) stored in
        lockstep with the view: same index, same eviction. ``version``
        increments on every accepted add / eviction / non-empty clear so
        publishers can cheaply detect change.
        """
        if feature is None or feature.ndim != 1 or not np.all(np.isfinite(feature)):
            return False
        f = _l2norm(feature.astype(np.float32))
        if not self._views:
            self._views.append(f)  # anchor, pinned at index 0
            self._thumbs.append(thumb)
            self.version += 1
            return True
        same = self._matching(f.shape[0])
        if same and max(_cos(f, v) for v in same) >= self.novelty_max:
            return False
        self._views.append(f)
        self._thumbs.append(thumb)
        if len(self._views) > self.size:
            drop = self._evict_most_redundant()
            if drop is not None:
                self._thumbs.pop(drop)
        self.version += 1
        return True
```
  - `_evict_most_redundant` returns the dropped index:
```python
    def _evict_most_redundant(self) -> Optional[int]:
        """Drop the most-redundant non-anchor view; return its index."""
        if len(self._views) <= 1:
            return None
        non_anchor = list(range(1, len(self._views)))

        def redundancy(idx: int) -> float:
            vi = self._views[idx]
            others = [v for j, v in enumerate(self._views)
                      if j != idx and v.shape[0] == vi.shape[0]]
            return float(np.mean([_cos(vi, o) for o in others])) if others else -1.0

        drop = max(non_anchor, key=redundancy)
        self._views.pop(drop)
        return drop
```

- [ ] **Step 4: Run, confirm PASS** (3 tests) + regression `PYTHONPATH="$(pwd)" $VENV -m pytest test/test_reid_gallery.py test/test_gallery_population.py test/test_reseed_target.py -q` (all green — `_apply_reseed` calls `maybe_add(fresh)` positionally, unaffected).

- [ ] **Step 5: Commit:**
```bash
git add src/vision_track/vision_track/core/reid_gallery.py src/vision_track/test/test_gallery_thumbs.py
git commit -m "feat(vision_track): gallery thumb retention + version counter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: pure `build_debug_state()` (core/debug_state.py)

**Files:**
- Create: `src/vision_track/vision_track/core/debug_state.py`
- Test: `src/vision_track/test/test_debug_state.py` (create)

- [ ] **Step 1: Write the failing test:**

```python
# <Apache-2.0 header — copy verbatim from test/test_reid_batch.py lines 1-13>
"""build_debug_state: pure, defensive snapshot of tracker state for the dashboard."""
from types import SimpleNamespace

from vision_track.core.debug_state import build_debug_state


def _bare(**over):
    t = SimpleNamespace(
        last_lock_decision=SimpleNamespace(state="reidentifying"),
        frames_lost=12,
        target_track_id=3,
        original_track_id=3,
        last_results=[
            SimpleNamespace(class_id=0, track_id=3, bbox=(10, 10, 50, 120)),
            SimpleNamespace(class_id=0, track_id=7, bbox=(60, 12, 110, 130)),
            SimpleNamespace(class_id=39, track_id=9, bbox=(0, 0, 5, 5)),   # not a person
            SimpleNamespace(class_id=0, track_id=-1, bbox=(0, 0, 9, 9)),   # untracked det
        ],
        last_debug_scores={3: 0.81, 7: 0.44},
        target_appearance=SimpleNamespace(
            gallery=SimpleNamespace(version=5, __len__=lambda s: 2)),
    )
    for k, v in over.items():
        setattr(t, k, v)
    return t


def _kw(**over):
    kw = dict(ts=123.0, target_lost=True, reacquisition_state=1,
              time_since_seen=0.8, awaiting_help=False,
              active_help_after_frames=45, active_help_timeout_sec=20.0)
    kw.update(over)
    return kw


def test_full_snapshot():
    d = build_debug_state(_bare(), **_kw())
    assert d["ts"] == 123.0 and d["target_lost"] is True
    assert d["fsm_state"] == "reidentifying"
    assert d["reacquisition_state"] == 1 and d["frames_lost"] == 12
    assert d["awaiting_help"] is False and d["active_help_timeout_sec"] == 20.0
    assert d["target_track_id"] == 3 and d["original_track_id"] == 3
    # persons with a real track id only; scores joined on id
    assert d["candidates"] == [
        {"id": 3, "bbox": [10, 10, 50, 120], "score": 0.81},
        {"id": 7, "bbox": [60, 12, 110, 130], "score": 0.44},
    ]
    assert d["best_sim"] == 0.81 and d["second_sim"] == 0.44
    assert d["gallery_len"] == 2 and d["gallery_version"] == 5


def test_defensive_on_bare_tracker():
    t = SimpleNamespace()  # nothing set at all
    d = build_debug_state(t, **_kw(target_lost=False, reacquisition_state=0))
    assert d["fsm_state"] is None and d["frames_lost"] == 0
    assert d["candidates"] == [] and d["best_sim"] is None and d["second_sim"] is None
    assert d["gallery_len"] == 0 and d["gallery_version"] == 0


def test_no_scores_yields_nulls():
    d = build_debug_state(_bare(last_debug_scores={}), **_kw())
    assert d["candidates"][0]["score"] is None
    assert d["best_sim"] is None and d["second_sim"] is None
```

- [ ] **Step 2: Run, confirm FAIL** (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/vision_track/vision_track/core/debug_state.py`:

```python
"""Pure snapshot of tracker state for the track_web dashboard.

JSON-serializable dict, built defensively (getattr everywhere) so a partially
initialized or bare tracker never raises. No ROS, no cv2 — unit-testable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _gallery(tracker) -> tuple:
    app = getattr(tracker, "target_appearance", None)
    gal = getattr(app, "gallery", None) if app is not None else None
    if gal is None:
        return 0, 0
    try:
        return len(gal), int(getattr(gal, "version", 0))
    except TypeError:
        return 0, int(getattr(gal, "version", 0))


def build_debug_state(
    tracker: Any,
    *,
    ts: float,
    target_lost: bool,
    reacquisition_state: int,
    time_since_seen: float,
    awaiting_help: bool,
    active_help_after_frames: int,
    active_help_timeout_sec: float,
) -> Dict[str, Any]:
    """Snapshot tracker + node loss-state into a JSON-serializable dict."""
    decision = getattr(tracker, "last_lock_decision", None)
    scores = getattr(tracker, "last_debug_scores", None) or {}

    candidates = []
    for r in getattr(tracker, "last_results", None) or []:
        tid = getattr(r, "track_id", None)
        if getattr(r, "class_id", None) != 0 or tid is None or tid < 0:
            continue
        bbox = getattr(r, "bbox", None)
        sc: Optional[float] = scores.get(tid)
        candidates.append({
            "id": int(tid),
            "bbox": [int(v) for v in bbox] if bbox is not None else None,
            "score": float(sc) if sc is not None else None,
        })

    ranked = sorted((s for s in scores.values() if s is not None), reverse=True)
    return {
        "ts": float(ts),
        "fsm_state": getattr(decision, "state", None),
        "target_lost": bool(target_lost),
        "reacquisition_state": int(reacquisition_state),
        "frames_lost": int(getattr(tracker, "frames_lost", 0) or 0),
        "time_since_seen": float(time_since_seen),
        "awaiting_help": bool(awaiting_help),
        "active_help_after_frames": int(active_help_after_frames),
        "active_help_timeout_sec": float(active_help_timeout_sec),
        "target_track_id": getattr(tracker, "target_track_id", None),
        "original_track_id": getattr(tracker, "original_track_id", None),
        "candidates": candidates,
        "best_sim": float(ranked[0]) if ranked else None,
        "second_sim": float(ranked[1]) if len(ranked) > 1 else None,
        "gallery_len": _gallery(tracker)[0],
        "gallery_version": _gallery(tracker)[1],
    }
```

- [ ] **Step 4: Run, confirm PASS** (3 tests).
- [ ] **Step 5: Commit:**
```bash
git add src/vision_track/vision_track/core/debug_state.py src/vision_track/test/test_debug_state.py
git commit -m "feat(vision_track): pure build_debug_state snapshot for track_web

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: tracker plumbing — score stash + thumb crops

**Files:**
- Modify: `src/vision_track/vision_track/yolo_tracker.py` (ctor param, attr init, reset)
- Modify: `src/vision_track/vision_track/reid/reid_search.py` (score stash)
- Modify: `src/vision_track/vision_track/core/tracking_pipeline.py` (periodic-validation stash)
- Modify: `src/vision_track/vision_track/reid/appearance_manager.py` (`_make_thumb` + thumb pass-through)
- Test: `src/vision_track/test/test_make_thumb.py` (create)

READ each file region before editing; line refs verified 2026-06-07 but confirm.

- [ ] **Step 1: Write the failing `_make_thumb` test:**

```python
# <Apache-2.0 header — copy verbatim from test/test_reid_batch.py lines 1-13>
"""_make_thumb: clamped, aspect-preserving gallery thumbnails."""
import numpy as np

from vision_track.reid.appearance_manager import _make_thumb


def test_resizes_tall_crop_to_max_height():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = _make_thumb(frame, (100, 50, 200, 450))      # 100x400 crop
    assert t is not None and t.shape[0] == 192
    assert abs(t.shape[1] - 48) <= 1                 # aspect preserved

def test_small_crop_kept_as_is():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = _make_thumb(frame, (10, 10, 60, 110))        # 50x100, under max
    assert t.shape[:2] == (100, 50)

def test_degenerate_bbox_returns_none():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert _make_thumb(frame, (700, 500, 800, 600)) is None   # fully off-frame
    assert _make_thumb(frame, (50, 50, 50, 120)) is None      # zero width
```

- [ ] **Step 2: Run, confirm FAIL** (ImportError `_make_thumb`).
- [ ] **Step 3: Implement.**

  In `reid/appearance_manager.py`, add after the module logger:
```python
def _make_thumb(frame, bbox, max_h: int = 192):
    """Clamped, aspect-preserving crop of ``bbox`` (RGB, same channel order as
    ``frame``) for gallery visualization; None when the bbox is degenerate."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] > max_h:
        new_w = max(1, round(crop.shape[1] * max_h / crop.shape[0]))
        crop = cv2.resize(crop, (new_w, max_h), interpolation=cv2.INTER_AREA)
    return crop.copy()
```

  In `update_appearance` (line ~85), build the thumb and pass it down (the frame here is RGB — keep it RGB; the node converts at encode time):
```python
    thumb = None
    if getattr(tracker, "keep_gallery_thumbs", False):
        thumb = _make_thumb(frame, result.bbox)
    _update_feature_history(tracker, features, similarity, current_time, refresh_allowed, thumb)
```
  and change `_update_feature_history(tracker, features, similarity, current_time, refresh_allowed)` → `..., refresh_allowed, thumb=None)`, with line 108 becoming `tracker.target_appearance.gallery.maybe_add(new_feature, thumb=thumb)`.

  In `yolo_tracker.py`:
  - ctor: add param `keep_gallery_thumbs: bool = False` (after `reid_gallery_score_mode`), store `self.keep_gallery_thumbs = keep_gallery_thumbs`, document in the docstring.
  - `_init_reid_settings`: add `self.last_debug_scores: dict = {}`.
  - `reset()`: add `self.last_debug_scores = {}`.

  In `reid/reid_search.py` `find_best_match_reid`, right after `candidate_scores = _score_candidates(...)` (line ~44):
```python
    # Dashboard telemetry: latest per-candidate similarities (plain assignment).
    tracker.last_debug_scores = {
        int(r.track_id): float(s) for r, s, _, _ in candidate_scores
    }
```

  In `core/tracking_pipeline.py`, find the periodic-validation similarity (the `ReIDMatcher.compute_similarity(...)` whose result drives the periodic re-check of the tracked target, ~line 227 — READ the surrounding function to confirm it is the periodic path, not occlusion verify) and stash after it:
```python
    tracker.last_debug_scores = {int(result.track_id): float(similarity)}
```
  (Adapt `result`/`similarity` to the real local names at that site.)

- [ ] **Step 4: Run, confirm PASS** + regression:
`PYTHONPATH="$(pwd)" $VENV -m pytest test/test_make_thumb.py test/test_reid_gallery.py test/test_gallery_thumbs.py test/test_reseed_target.py test/test_debug_state.py -q` → all green. Also full functional suite: `PYTHONPATH="$(pwd)" $VENV -m pytest test/ -q --ignore=test/test_flake8.py --ignore=test/test_pep257.py` → no new failures.

- [ ] **Step 5: Commit:**
```bash
git add src/vision_track/vision_track/yolo_tracker.py src/vision_track/vision_track/reid/reid_search.py src/vision_track/vision_track/core/tracking_pipeline.py src/vision_track/vision_track/reid/appearance_manager.py src/vision_track/test/test_make_thumb.py
git commit -m "feat(vision_track): score stash + gallery thumb plumbing for track_web

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: node debug publishers (`~/debug_state`, `~/debug_gallery`, `~/debug_image`)

**Files:**
- Modify: `src/vision_track/vision_track/person_track_node.py`
- Modify: `src/vision_track/config/default.yaml`
- Test: sourced import check (node loop needs cameras; pure logic already covered by Tasks 1-3)

READ the node first. Known anchors (verified 2026-06-07): params declared ~lines 143-156 / loaded ~209-220; tracker constructed with `reid_gallery_*` kwargs (find the `YOLOTracker(` call); `_run_tracking_loop` calls `_handle_tracked_frame` / `_handle_lost_frame` at ~739-746 with `feedback` + `last_seen_time` in scope; `_draw_debug_info(rgb_img, results, track_result, target_id)` exists; `self.bridge` is a CvBridge.

- [ ] **Step 1: Params.** Declare (next to `active_help_timeout_sec`): `debug_state_enabled` (False), `gallery_keep_crops` (False), `debug_image_enabled` (False). Load into `self.debug_state_enabled` / `self.gallery_keep_crops` / `self.debug_image_enabled` (bool). Thread `keep_gallery_thumbs=self.gallery_keep_crops` into the `YOLOTracker(...)` construction call.

- [ ] **Step 2: Publishers + helper.** Import `json`, `base64` (top, if absent) and `from std_msgs.msg import String`. In `__init__` (near other publishers):
```python
        self.debug_state_pub = self.create_publisher(String, '~/debug_state', 10)
        gallery_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.debug_gallery_pub = self.create_publisher(String, '~/debug_gallery', gallery_qos)
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 1)
        self._last_gallery_version = -1
```
  (Import `QoSProfile`, `ReliabilityPolicy`, `DurabilityPolicy` from `rclpy.qos` if absent; `Image` from `sensor_msgs.msg` is already imported. TRANSIENT_LOCAL so a late-started dashboard receives the current gallery immediately.)

  Add the method (place after `_handle_lost_frame`):
```python
    def _publish_debug_outputs(self, rgb_img, track_result, feedback, last_seen_time):
        """Param-gated dashboard telemetry; must never raise into the loop."""
        try:
            if self.debug_state_enabled:
                tss = time.time() - last_seen_time
                frames_lost = int(getattr(self.tracker, 'frames_lost', 0))
                awaiting = (self.active_help_timeout_sec > 0.0
                            and self.active_help_after_frames > 0
                            and frames_lost >= self.active_help_after_frames
                            and tss <= self.active_help_timeout_sec
                            and bool(feedback.target_lost))
                state = build_debug_state(
                    self.tracker, ts=time.time(),
                    target_lost=bool(feedback.target_lost),
                    reacquisition_state=int(feedback.reacquisition_state),
                    time_since_seen=tss, awaiting_help=awaiting,
                    active_help_after_frames=self.active_help_after_frames,
                    active_help_timeout_sec=self.active_help_timeout_sec)
                self.debug_state_pub.publish(String(data=json.dumps(state)))
                if self.gallery_keep_crops:
                    self._maybe_publish_gallery(state["gallery_version"])
            if self.debug_image_enabled and self.debug_image_pub.get_subscription_count() > 0:
                annotated = self._draw_debug_info(
                    rgb_img, self.tracker.last_results, track_result,
                    self.tracker.target_track_id)
                self.debug_image_pub.publish(
                    self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))
        except Exception as exc:  # telemetry must never kill tracking
            self.get_logger().warn(f'debug output failed: {exc}')

    def _maybe_publish_gallery(self, version: int):
        if version == self._last_gallery_version:
            return
        app = getattr(self.tracker, 'target_appearance', None)
        thumbs = list(getattr(getattr(app, 'gallery', None), 'thumbs', []) or []) if app else []
        encoded = []
        for t in thumbs:
            if t is None:
                encoded.append(None)
                continue
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(t, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            encoded.append(base64.b64encode(buf).decode('ascii') if ok else None)
        self.debug_gallery_pub.publish(String(data=json.dumps(
            {'version': version, 'thumbs': encoded})))
        self._last_gallery_version = version
```
  Import `from vision_track.core.debug_state import build_debug_state` with the other `vision_track.core` imports. (Thumbs are stored RGB by Task 3; encode converts RGB→BGR for `imencode`.)

- [ ] **Step 3: Call site.** In `_run_tracking_loop`, immediately after the tracked/lost `if/else` block (the lost branch `return result` path skips it — acceptable, that's the abort frame):
```python
            self._publish_debug_outputs(rgb_img, track_result, feedback, last_seen_time)
```

- [ ] **Step 4: Config.** In `config/default.yaml`, after `active_help_timeout_sec`:
```yaml
    # --- track_web dashboard telemetry (all default OFF; zero production impact) ---
    debug_state_enabled: false       # ~/debug_state JSON per frame
    gallery_keep_crops: false        # retain + publish gallery view thumbnails (~/debug_gallery)
    debug_image_enabled: false       # ~/debug_image annotated frames (only drawn when subscribed)
```

- [ ] **Step 5: Verify.** Sourced import check (single bash call):
```bash
source /home/tinker/tk25_ws/install/setup.bash && \
cd /home/tinker/tk25_ws/src/tk26_vision/src/vision_track && \
PYTHONPATH="$(pwd):$PYTHONPATH" /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python \
  -c "import vision_track.person_track_node; print('import OK')"
```
Expected `import OK`. Plus yaml parse: `$VENV -c "import yaml; yaml.safe_load(open('config/default.yaml'))"`. flake8 (99): no NEW errors on your added ranges.

- [ ] **Step 6: Commit:**
```bash
git add src/vision_track/vision_track/person_track_node.py src/vision_track/config/default.yaml
git commit -m "feat(vision_track): param-gated debug_state/gallery/image publishers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `track_web` app core (bridge protocol + FastAPI endpoints, ROS-free tests)

**Files:**
- Create: `src/vision_track/vision_track/track_web_app.py` (ROS-free: bridge protocol + `create_app`)
- Test: `src/vision_track/test/test_track_web_app.py` (create)

- [ ] **Step 0: Deps.** In `.venv-vision-main`: `pip show fastapi uvicorn httpx` — `pip install fastapi uvicorn httpx` for any missing (httpx is the TestClient transport). Append `fastapi`, `uvicorn`, `httpx` to `src/vision_track/requirements.txt` (create the lines; file exists).

- [ ] **Step 1: Write the failing tests:**

```python
# <Apache-2.0 header — copy verbatim from test/test_reid_batch.py lines 1-13>
"""track_web_app endpoints against a fake bridge (no ROS)."""
import json

from fastapi.testclient import TestClient

from vision_track.track_web_app import create_app


class FakeBridge:
    def __init__(self):
        self.calls = []
        self._state = {"ts": 1.0, "reacquisition_state": 2, "candidates": []}
        self._jpeg = b"\xff\xd8fakejpeg\xff\xd9"

    def snapshot(self):
        return {"state": self._state, "state_age_s": 0.1,
                "goal": {"held": False, "observer": False}, "gallery_version": 0}

    def latest_state(self):
        return 7, self._state

    def latest_gallery(self):
        return {"version": 0, "thumbs": []}

    def latest_jpeg(self):
        return 3, self._jpeg

    def start_goal(self):
        self.calls.append("start")
        return {"ok": True, "message": "goal sent"}

    def stop_goal(self):
        self.calls.append("stop")
        return {"ok": True, "message": "cancelled"}

    def reseed(self, bbox):
        self.calls.append(("reseed", tuple(bbox)))
        return {"success": True, "target_track_id": 9, "message": "reseeded"}

    def wave(self):
        self.calls.append("wave")
        return {"status": 0, "boxes": [[1, 2, 3, 4]], "points": [[0.5, 0.1, 2.0]]}


def _client():
    b = FakeBridge()
    return b, TestClient(create_app(b, webui_dir=None))


def test_status():
    b, c = _client()
    r = c.get("/api/status")
    assert r.status_code == 200 and r.json()["goal"] == {"held": False, "observer": False}


def test_goal_and_wave_roundtrip():
    b, c = _client()
    assert c.post("/api/goal/start").json()["ok"] is True
    assert c.post("/api/goal/stop").json()["ok"] is True
    assert c.post("/api/wave").json()["boxes"] == [[1, 2, 3, 4]]
    assert b.calls[:3] == ["start", "stop", "wave"]


def test_reseed_validates_bbox():
    b, c = _client()
    assert c.post("/api/reseed", json={"bbox": [1, 2, 30, 40]}).json()["success"] is True
    assert ("reseed", (1, 2, 30, 40)) in b.calls
    assert c.post("/api/reseed", json={"bbox": [1, 2]}).status_code == 422
    assert c.post("/api/reseed", json={"bbox": [30, 40, 1, 2]}).status_code == 422


def test_ws_pushes_state():
    b, c = _client()
    with c.websocket_connect("/ws/state") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "state" and msg["data"]["reacquisition_state"] == 2


def test_mjpeg_stream_headers_and_first_frame():
    b, c = _client()
    with c.stream("GET", "/stream.mjpg") as r:
        assert r.status_code == 200
        assert "multipart/x-mixed-replace" in r.headers["content-type"]
        chunk = next(r.iter_bytes())
        assert b"--frame" in chunk and b"fakejpeg" in chunk
```

- [ ] **Step 2: Run, confirm FAIL** (ModuleNotFoundError `track_web_app`).
`PYTHONPATH="$(pwd)" $VENV -m pytest test/test_track_web_app.py -v`

- [ ] **Step 3: Implement** `src/vision_track/vision_track/track_web_app.py`:

```python
"""ROS-free FastAPI app for the track_web dashboard.

``create_app(bridge, webui_dir)`` wires HTTP/WS/MJPEG endpoints to a bridge
object (the ROS node in production, a fake in tests). The bridge contract:

    snapshot() -> dict                      # /api/status payload
    latest_state() -> (seq:int, dict|None)  # newest ~/debug_state
    latest_gallery() -> dict|None           # newest ~/debug_gallery payload
    latest_jpeg() -> (seq:int, bytes|None)  # newest annotated frame as JPEG
    start_goal() / stop_goal() -> dict      # {ok: bool, message: str}
    reseed(bbox:[x1,y1,x2,y2]) -> dict      # ReseedTarget response fields
    wave() -> dict                          # DetectWaving boxes/points or error

All bridge methods must be thread-safe; handlers poll (no cross-thread asyncio).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

_STATE_POLL_S = 0.03   # ~33 Hz cap on WS pushes
_MJPEG_POLL_S = 1 / 15  # 15 fps cap on the stream


class ReseedRequest(BaseModel):
    bbox: List[int]

    @field_validator("bbox")
    @classmethod
    def _valid_box(cls, v):
        if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("bbox must be [x1,y1,x2,y2] with x2>x1, y2>y1")
        return v


def create_app(bridge, webui_dir: Optional[Path] = None) -> FastAPI:
    """Build the FastAPI app around a (real or fake) tracker bridge."""
    app = FastAPI(title="track_web")

    if webui_dir is not None and Path(webui_dir).exists():
        webui = Path(webui_dir)

        @app.get("/")
        def index():
            return FileResponse(webui / "index.html", media_type="text/html")

        @app.get("/style.css")
        def style():
            return FileResponse(webui / "style.css", media_type="text/css")

        @app.get("/app.js")
        def appjs():
            return FileResponse(webui / "app.js",
                                media_type="application/javascript")
    else:
        @app.get("/")
        def index_missing():
            return JSONResponse({"error": "webui dir not found"}, status_code=500)

    @app.get("/api/status")
    def status():
        return bridge.snapshot()

    @app.post("/api/goal/start")
    def goal_start():
        return bridge.start_goal()

    @app.post("/api/goal/stop")
    def goal_stop():
        return bridge.stop_goal()

    @app.post("/api/reseed")
    def reseed(req: ReseedRequest):
        return bridge.reseed(req.bbox)

    @app.post("/api/wave")
    def wave():
        return bridge.wave()

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        last_state_seq = -1
        last_gallery_version = -1
        try:
            while True:
                seq, state = bridge.latest_state()
                if state is not None and seq != last_state_seq:
                    last_state_seq = seq
                    await ws.send_text(json.dumps({"type": "state", "data": state}))
                    gal = bridge.latest_gallery()
                    if gal is not None and gal.get("version", -1) != last_gallery_version:
                        last_gallery_version = gal["version"]
                        await ws.send_text(json.dumps({"type": "gallery", "data": gal}))
                await asyncio.sleep(_STATE_POLL_S)
        except WebSocketDisconnect:
            return

    async def _mjpeg_gen():
        last_seq = -1
        while True:
            seq, jpeg = bridge.latest_jpeg()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            await asyncio.sleep(_MJPEG_POLL_S)

    @app.get("/stream.mjpg")
    def stream():
        return StreamingResponse(
            _mjpeg_gen(),
            media_type="multipart/x-mixed-replace; boundary=frame")

    return app
```

- [ ] **Step 4: Run, confirm PASS** (6 tests). flake8 (99) clean on both files.
- [ ] **Step 5: Commit:**
```bash
git add src/vision_track/vision_track/track_web_app.py src/vision_track/test/test_track_web_app.py src/vision_track/requirements.txt
git commit -m "feat(vision_track): track_web FastAPI app core (ROS-free, bridge-tested)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `track_web` ROS node (the real bridge) + entry point

**Files:**
- Create: `src/vision_track/vision_track/track_web.py`
- Modify: `src/vision_track/setup.py` (entry point + webui data_files)
- Test: sourced import check (live behaviour needs the tracker running)

Pattern source: `src/pan_tilt/pan_tilt/calib_web.py` (threading model + `_resolve_webui_dir` + main()). READ its `_resolve_webui_dir` and mirror it for `vision_track`.

- [ ] **Step 1: Implement** `src/vision_track/vision_track/track_web.py`:

```python
"""track_web — live tracking dashboard + active-reID test bench (ROS side).

Run:
    ros2 run vision_track track_web --ros-args -p bind:=0.0.0.0 -p port:=8766

Bridges the person tracker's debug topics + TrackPerson action +
ReseedTarget/DetectWaving services to the FastAPI app in track_web_app.py.
Threading: rclpy.spin in the main thread, uvicorn in a daemon thread, all
shared state behind self._lock (the calib_web model).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import numpy as np

from tinker_vision_msgs_26.action import TrackPerson
from tinker_vision_msgs_26.srv import DetectWaving, ReseedTarget

from vision_track.track_web_app import create_app

_STALE_S = 1.0


def _resolve_webui_dir() -> Path:
    """Installed share/vision_track/webui, falling back to the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory("vision_track")) / "webui"
        if p.exists():
            return p
    except Exception:
        pass
    return Path(__file__).resolve().parents[2] / "webui"


class TrackWebNode(Node):
    """ROS bridge implementing the track_web_app bridge contract."""

    def __init__(self):
        super().__init__("track_web")
        self.declare_parameter("bind", "127.0.0.1")
        self.declare_parameter("port", 8766)
        self.declare_parameter("tracker_node_name", "person_track_node")
        self.declare_parameter("waving_service", "detect_waving_persons")
        self.bind_host = str(self.get_parameter("bind").value)
        self.bind_port = int(self.get_parameter("port").value)
        tracker = str(self.get_parameter("tracker_node_name").value)
        waving = str(self.get_parameter("waving_service").value)

        self._lock = threading.Lock()
        self._state = None          # latest debug_state dict
        self._state_seq = 0
        self._state_ts = 0.0
        self._gallery = None        # latest debug_gallery dict
        self._jpeg = None           # latest annotated frame as JPEG bytes
        self._jpeg_seq = 0
        self._goal_handle = None    # our bench goal (None = not held by us)

        cb = ReentrantCallbackGroup()
        self.create_subscription(
            String, f"/{tracker}/debug_state", self._on_state, 10, callback_group=cb)
        from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy)
        gallery_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, f"/{tracker}/debug_gallery", self._on_gallery, gallery_qos,
            callback_group=cb)
        self.create_subscription(
            Image, f"/{tracker}/debug_image", self._on_image, 1, callback_group=cb)

        self._action = ActionClient(self, TrackPerson, "track_person",
                                    callback_group=cb)
        self._reseed_cli = self.create_client(
            ReseedTarget, f"/{tracker}/reseed_target", callback_group=cb)
        self._wave_cli = self.create_client(DetectWaving, waving,
                                            callback_group=cb)
        self.get_logger().info(
            f"track_web bridging tracker '{tracker}', waving '{waving}'")

    # ---- subscription callbacks -------------------------------------------
    def _on_state(self, msg: String):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._state = state
            self._state_seq += 1
            self._state_ts = time.time()

    def _on_gallery(self, msg: String):
        try:
            gal = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._gallery = gal

    def _on_image(self, msg: Image):
        # bgr8 on the wire; encode once here, serve many times.
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as exc:
            self.get_logger().warn(f"debug_image decode failed: {exc}")
            return
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
                self._jpeg_seq += 1

    # ---- bridge contract ---------------------------------------------------
    def snapshot(self):
        with self._lock:
            age = time.time() - self._state_ts if self._state is not None else None
            held = self._goal_handle is not None
            observer = (not held and self._state is not None
                        and age is not None and age < _STALE_S)
            return {"state": self._state, "state_age_s": age,
                    "goal": {"held": held, "observer": observer},
                    "gallery_version": (self._gallery or {}).get("version", -1)}

    def latest_state(self):
        with self._lock:
            return self._state_seq, self._state

    def latest_gallery(self):
        with self._lock:
            return self._gallery

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg_seq, self._jpeg

    def start_goal(self):
        with self._lock:
            if self._goal_handle is not None:
                return {"ok": False, "message": "bench goal already running"}
        if not self._action.wait_for_server(timeout_sec=2.0):
            return {"ok": False, "message": "track_person action server unavailable"}
        goal = TrackPerson.Goal()  # all image-return flags default False
        future = self._action.send_goal_async(goal)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            return {"ok": False, "message": "goal send timed out"}
        handle = future.result()
        if not handle.accepted:
            return {"ok": False,
                    "message": "goal REJECTED (another client is tracking?)"}
        with self._lock:
            self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_goal_done)
        return {"ok": True, "message": "tracking goal accepted"}

    def _on_goal_done(self, _future):
        with self._lock:
            self._goal_handle = None

    def stop_goal(self):
        with self._lock:
            handle = self._goal_handle
        if handle is None:
            return {"ok": False, "message": "no bench goal to stop"}
        handle.cancel_goal_async()
        return {"ok": True, "message": "cancel requested"}

    def _call(self, client, request, timeout=10.0, name="service"):
        if not client.wait_for_service(timeout_sec=2.0):
            return None, f"{name} unavailable"
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            return None, f"{name} timed out after {timeout:.0f}s"
        return future.result(), None

    def reseed(self, bbox):
        req = ReseedTarget.Request()
        req.bbox.x_offset = max(0, int(bbox[0]))
        req.bbox.y_offset = max(0, int(bbox[1]))
        req.bbox.width = max(0, int(bbox[2] - bbox[0]))
        req.bbox.height = max(0, int(bbox[3] - bbox[1]))
        req.frame_id = ""
        resp, err = self._call(self._reseed_cli, req, name="reseed_target")
        if err:
            return {"success": False, "target_track_id": -1, "message": err}
        return {"success": bool(resp.success),
                "target_track_id": int(resp.target_track_id),
                "message": str(resp.message)}

    def wave(self):
        resp, err = self._call(self._wave_cli, DetectWaving.Request(),
                               timeout=30.0, name="detect_waving_persons")
        if err:
            return {"status": -1, "boxes": [], "points": [], "error": err}
        boxes = [[int(b.x_offset), int(b.y_offset),
                  int(b.x_offset + b.width), int(b.y_offset + b.height)]
                 for b in resp.waving_boxes]
        points = [[float(p.point.x), float(p.point.y), float(p.point.z)]
                  for p in resp.waving_persons]
        return {"status": int(resp.status), "boxes": boxes, "points": points}


def main():
    # Mirror calib_web: avoid the SHM-discovery stall on a live robot.
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)

    rclpy.init()
    node = TrackWebNode()
    webui_dir = _resolve_webui_dir()
    node.get_logger().info(f"web UI static dir: {webui_dir}")
    app = create_app(node, webui_dir=webui_dir)

    import uvicorn
    config = uvicorn.Config(app, host=node.bind_host, port=node.bind_port,
                            log_level="info", access_log=False, loop="asyncio")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve())

    thread = threading.Thread(target=_serve, daemon=True, name="uvicorn")
    thread.start()
    node.get_logger().info(
        f"track_web listening on http://{node.bind_host}:{node.bind_port}")

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
```
  (Note: `start_goal`/`_call` poll futures with `time.sleep` — they run on uvicorn worker threads while the `MultiThreadedExecutor` spins callbacks on the main thread, so blocking here never deadlocks the executor. The reseed `frame_id` is sent empty — the tracker matches against its own frame and Spec B's handler only logs mismatches.)

- [ ] **Step 2: setup.py.** Add to `data_files`:
```python
        (os.path.join('share', package_name, 'webui'), glob('webui/*')),
```
  and to `console_scripts`:
```python
            'track_web = vision_track.track_web:main',
```
  Create a placeholder `src/vision_track/webui/.gitkeep`? **No** — Task 7 creates the real files first; `glob('webui/*')` of a missing dir yields `[]` (no build break), so order is safe either way.

- [ ] **Step 3: Verify.** Sourced import check (single bash call):
```bash
source /home/tinker/tk25_ws/install/setup.bash && \
cd /home/tinker/tk25_ws/src/tk26_vision/src/vision_track && \
PYTHONPATH="$(pwd):$PYTHONPATH" /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python \
  -c "import vision_track.track_web; print('import OK')"
```
Expected `import OK`. flake8 (99) clean on the new file.

- [ ] **Step 4: Commit:**
```bash
git add src/vision_track/vision_track/track_web.py src/vision_track/setup.py
git commit -m "feat(vision_track): track_web ROS bridge node + entry point

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: webui (layout 1 — video + right rail)

**Files:**
- Create: `src/vision_track/webui/index.html`
- Create: `src/vision_track/webui/style.css`
- Create: `src/vision_track/webui/app.js`
- Test: manual via FastAPI TestClient route check (Step 3) — interactive behaviour is browser/T2-tier

- [ ] **Step 1: `index.html`:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>track_web — Tinker person tracker</title>
<link rel="stylesheet" href="/style.css">
</head>
<body>
<header>
  <h1>track_web</h1>
  <span id="conn" class="badge off">connecting…</span>
  <span id="mode" class="badge off">—</span>
</header>
<main>
  <section id="video-pane">
    <div id="video-wrap">
      <img id="video" src="/stream.mjpg" alt="annotated tracker view">
      <div id="overlays"></div>
      <div id="stale-banner" class="hidden">NO DATA — tracker silent</div>
    </div>
    <div id="controls">
      <button id="btn-start">▶ Start goal</button>
      <button id="btn-stop">■ Stop</button>
      <button id="btn-wave">👋 DetectWaving</button>
      <button id="btn-clear">✕ Clear overlays</button>
    </div>
  </section>
  <aside id="rail">
    <div class="panel">
      <div id="reacq-badge" class="reacq">—</div>
      <table id="state-table">
        <tr><td>FSM</td><td id="fsm">—</td></tr>
        <tr><td>target_lost</td><td id="lost">—</td></tr>
        <tr><td>track id</td><td id="ids">—</td></tr>
        <tr><td>frames_lost</td><td id="frames">—</td></tr>
        <tr><td>hold left</td><td id="hold">—</td></tr>
        <tr><td>best / 2nd sim</td><td id="sims">—</td></tr>
      </table>
    </div>
    <div class="panel">
      <h2>Gallery <span id="gal-meta"></span></h2>
      <div id="gallery"></div>
    </div>
    <div class="panel">
      <h2>Events</h2>
      <ul id="log"></ul>
    </div>
  </aside>
</main>
<script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: `style.css`:**

```css
:root { --bg:#11141a; --panel:#1a1f29; --line:#2c3344; --fg:#cdd6f4; --dim:#8893ad;
        --green:#3fb950; --amber:#d29922; --red:#f85149; --blue:#58a6ff; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui,-apple-system,sans-serif; }
header { display:flex; align-items:center; gap:12px; padding:10px 16px;
         border-bottom:1px solid var(--line); }
header h1 { font-size:16px; font-weight:600; }
.badge { font-size:12px; padding:2px 10px; border-radius:10px; background:var(--line); }
.badge.on { background:var(--green); color:#04110a; }
.badge.off { background:var(--line); color:var(--dim); }
main { display:flex; gap:12px; padding:12px; align-items:flex-start; }
#video-pane { flex:3; min-width:0; }
#video-wrap { position:relative; background:#000; border:1px solid var(--line);
              border-radius:8px; overflow:hidden; }
#video { display:block; width:100%; cursor:crosshair; }
#overlays { position:absolute; inset:0; pointer-events:none; }
.wave-box { position:absolute; border:2px solid var(--blue); border-radius:4px;
            pointer-events:auto; cursor:pointer; background:rgba(88,166,255,.12); }
.wave-box:hover { background:rgba(88,166,255,.3); }
#stale-banner { position:absolute; inset:auto 0 0 0; padding:6px; text-align:center;
                background:rgba(248,81,73,.85); color:#fff; font-weight:600; }
.hidden { display:none; }
#controls { display:flex; gap:8px; margin-top:10px; }
#controls button { background:var(--panel); color:var(--fg); border:1px solid var(--line);
                   border-radius:6px; padding:8px 14px; cursor:pointer; }
#controls button:hover { border-color:var(--blue); }
#rail { flex:1; display:flex; flex-direction:column; gap:12px; min-width:260px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px;
         padding:12px; }
.panel h2 { font-size:13px; color:var(--dim); margin-bottom:8px; }
.reacq { text-align:center; font-weight:700; padding:10px; border-radius:6px;
         margin-bottom:10px; background:var(--line); }
.reacq.tracking { background:var(--green); color:#04110a; }
.reacq.passive  { background:var(--amber); color:#1a1200; }
.reacq.needs-help { background:var(--red); color:#fff; }
#state-table { width:100%; font-size:13px; border-collapse:collapse; }
#state-table td { padding:3px 4px; border-bottom:1px solid var(--line); }
#state-table td:first-child { color:var(--dim); }
#gallery { display:flex; flex-wrap:wrap; gap:6px; }
#gallery img { height:72px; border-radius:4px; border:1px solid var(--line); }
#gallery img.anchor { border-color:var(--blue); }
#log { list-style:none; font-size:12px; max-height:200px; overflow-y:auto; }
#log li { padding:2px 0; border-bottom:1px solid var(--line); color:var(--dim); }
#log li b { color:var(--fg); }
```

- [ ] **Step 3: `app.js`:**

```javascript
/* track_web dashboard client: WS state feed, MJPEG video, click-to-reseed. */
"use strict";

const REACQ = {0: ["TRACKING", "tracking"], 1: ["PASSIVE", "passive"],
               2: ["NEEDS HELP", "needs-help"]};
const $ = (id) => document.getElementById(id);
let lastState = null;
let lastStateAt = 0;
let waveBoxes = [];

function log(msg) {
  const li = document.createElement("li");
  li.innerHTML = `<b>${new Date().toLocaleTimeString()}</b> ${msg}`;
  $("log").prepend(li);
  while ($("log").children.length > 80) $("log").lastChild.remove();
}

function renderState(s) {
  const prev = lastState;
  lastState = s;
  lastStateAt = Date.now();
  const [label, cls] = REACQ[s.reacquisition_state] || ["?", ""];
  const badge = $("reacq-badge");
  badge.textContent = label;
  badge.className = "reacq " + cls;
  $("fsm").textContent = s.fsm_state ?? "—";
  $("lost").textContent = s.target_lost;
  $("ids").textContent = `${s.target_track_id ?? "—"} (orig ${s.original_track_id ?? "—"})`;
  $("frames").textContent = s.frames_lost;
  $("hold").textContent = s.awaiting_help
    ? `${Math.max(0, s.active_help_timeout_sec - s.time_since_seen).toFixed(1)}s`
    : "—";
  const f = (x) => (x == null ? "—" : x.toFixed(3));
  $("sims").textContent = `${f(s.best_sim)} / ${f(s.second_sim)}`;
  if (prev) {
    if (prev.target_lost !== s.target_lost)
      log(s.target_lost ? "target LOST" : "target reacquired");
    if (prev.reacquisition_state !== s.reacquisition_state)
      log(`reacq → ${(REACQ[s.reacquisition_state] || ["?"])[0]}`);
  }
}

function renderGallery(g) {
  $("gal-meta").textContent = `v${g.version} · ${g.thumbs.length} views`;
  const div = $("gallery");
  div.innerHTML = "";
  g.thumbs.forEach((b64, i) => {
    if (!b64) return;
    const img = document.createElement("img");
    img.src = "data:image/jpeg;base64," + b64;
    if (i === 0) img.classList.add("anchor");
    img.title = i === 0 ? "anchor view" : `view ${i}`;
    div.appendChild(img);
  });
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws/state`);
  ws.onopen = () => { $("conn").textContent = "live"; $("conn").className = "badge on"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") renderState(msg.data);
    if (msg.type === "gallery") renderGallery(msg.data);
  };
  ws.onclose = () => {
    $("conn").textContent = "reconnecting…";
    $("conn").className = "badge off";
    setTimeout(connectWS, 1500);
  };
}

/* Map a click on the displayed <img> to native pixel coords. */
function clickToNative(ev) {
  const img = $("video");
  const r = img.getBoundingClientRect();
  if (!img.naturalWidth) return null;
  return [(ev.clientX - r.left) * img.naturalWidth / r.width,
          (ev.clientY - r.top) * img.naturalHeight / r.height];
}

async function post(url, body) {
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: body ? {"Content-Type": "application/json"} : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    return await r.json();
  } catch (e) { return {message: `request failed: ${e}`}; }
}

async function reseed(bbox, label) {
  const r = await post("/api/reseed", {bbox: bbox.map(Math.round)});
  log(`reseed(${label}) → ${r.success ? "OK id=" + r.target_track_id : "FAIL"} (${r.message})`);
  clearOverlays();
}

$("video").addEventListener("click", (ev) => {
  const pt = clickToNative(ev);
  if (!pt || !lastState) return;
  const hits = (lastState.candidates || []).filter((c) => c.bbox &&
    pt[0] >= c.bbox[0] && pt[0] <= c.bbox[2] &&
    pt[1] >= c.bbox[1] && pt[1] <= c.bbox[3]);
  if (!hits.length) { log("click: no candidate box there"); return; }
  hits.sort((a, b) => (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
                    - (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]));
  reseed(hits[0].bbox, `candidate ${hits[0].id}`);
});

function clearOverlays() { waveBoxes = []; $("overlays").innerHTML = ""; }

function renderWaveBoxes() {
  const img = $("video");
  const ov = $("overlays");
  ov.innerHTML = "";
  if (!img.naturalWidth) return;
  const r = img.getBoundingClientRect();
  const sx = r.width / img.naturalWidth, sy = r.height / img.naturalHeight;
  waveBoxes.forEach((b, i) => {
    const d = document.createElement("div");
    d.className = "wave-box";
    d.style.left = b[0] * sx + "px";
    d.style.top = b[1] * sy + "px";
    d.style.width = (b[2] - b[0]) * sx + "px";
    d.style.height = (b[3] - b[1]) * sy + "px";
    d.title = `waving person ${i} — click to re-seed`;
    d.onclick = (e) => { e.stopPropagation(); reseed(b, `wave ${i}`); };
    ov.appendChild(d);
  });
}

$("btn-start").onclick = async () => log("start → " + (await post("/api/goal/start")).message);
$("btn-stop").onclick = async () => log("stop → " + (await post("/api/goal/stop")).message);
$("btn-clear").onclick = clearOverlays;
$("btn-wave").onclick = async () => {
  log("DetectWaving…");
  const r = await post("/api/wave");
  if (r.error || r.status !== 0) { log(`wave FAIL (${r.error || "status " + r.status})`); return; }
  waveBoxes = r.boxes;
  log(`wave → ${r.boxes.length} box(es); click one to re-seed`);
  renderWaveBoxes();
};

/* Stale banner + observer/bench mode chip. */
setInterval(async () => {
  $("stale-banner").classList.toggle("hidden", Date.now() - lastStateAt < 1000);
  const st = await (await fetch("/api/status")).json().catch(() => null);
  if (st) {
    const m = st.goal.held ? ["bench", "on"] : st.goal.observer ? ["observer", "on"] : ["idle", "off"];
    $("mode").textContent = m[0];
    $("mode").className = "badge " + m[1];
  }
}, 1000);

window.addEventListener("resize", renderWaveBoxes);
connectWS();
```

- [ ] **Step 4: Verify route serving.** Extend `test/test_track_web_app.py` with:
```python
def test_webui_served_from_dir(tmp_path):
    (tmp_path / "index.html").write_text("<html>ok</html>")
    (tmp_path / "style.css").write_text("body{}")
    (tmp_path / "app.js").write_text("'use strict';")
    b = FakeBridge()
    c = TestClient(create_app(b, webui_dir=tmp_path))
    assert c.get("/").status_code == 200
    assert "javascript" in c.get("/app.js").headers["content-type"]
```
Run the file → all green (7 tests).

- [ ] **Step 5: Commit:**
```bash
git add src/vision_track/webui/index.html src/vision_track/webui/style.css src/vision_track/webui/app.js src/vision_track/test/test_track_web_app.py
git commit -m "feat(vision_track): track_web webui (video + rail, click-to-reseed)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: build, docs, final verification

**Files:**
- Modify: `src/vision_track/readme.md` (usage section + changelog)
- Modify: `DEV_NOTES.md` (tk26_vision root)

- [ ] **Step 1: Build + install-tree check.**
```bash
cd /home/tinker/tk25_ws && ./tkbuild tk26_vision --packages-select vision_track
```
Expected: success. Then (single bash call):
```bash
source /home/tinker/tk25_ws/install/setup.bash && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "
import vision_track.track_web, vision_track.track_web_app
from vision_track.core.debug_state import build_debug_state
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
w = Path(get_package_share_directory('vision_track'))/'webui'
assert (w/'index.html').exists(), w
print('INSTALL OK; webui at', w)"
```
Expected `INSTALL OK`.

- [ ] **Step 2: Full test suite.**
```bash
cd /home/tinker/tk25_ws/src/tk26_vision/src/vision_track && PYTHONPATH="$(pwd)" /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/ -q --ignore=test/test_flake8.py --ignore=test/test_pep257.py
```
Expected: all pass (133 baseline + ~13 new), no new failures.

- [ ] **Step 3: Docs.** In `src/vision_track/readme.md` add a `## track_web dashboard` section: run command (`ros2 run vision_track track_web --ros-args -p bind:=0.0.0.0 -p port:=8766`), the three tracker debug params to enable (`-p debug_state_enabled:=true -p gallery_keep_crops:=true -p debug_image_enabled:=true` on `person_track_server`), bench vs observer mode, click-to-reseed + wave flow, and a changelog entry (append-only, dated 2026-06-07). In `DEV_NOTES.md` add an entry: track_web shipped; unit/TestClient-verified; live camera (T2) + on-robot bench loop deferred to an operator session (record results back).

- [ ] **Step 4: Commit:**
```bash
git add src/vision_track/readme.md DEV_NOTES.md
git commit -m "docs(vision_track): track_web usage + changelog + deferral notes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Validation reality (read before executing)

Unit/TestClient level verifies: gallery thumb lockstep + version, `build_debug_state`, `_make_thumb`, every HTTP/WS/MJPEG endpoint against a fake bridge, and webui route serving. Sourced import checks verify the node wiring constructs. **Not verifiable here:** real WS/MJPEG under a live tracker, click-to-reseed end-to-end, DetectWaving overlay flow, observer mode against a real BT — all need cameras/operator (T2+) and are recorded as deferred in Task 8's DEV_NOTES entry, matching the Spec-B discipline.

## Self-Review

- **Spec coverage:** §1 instrumentation → Tasks 1-4 (debug_state keys incl. `active_help_*`, score-stash sites, thumbs opaque-in-gallery + RGB→BGR at encode, TRANSIENT_LOCAL gallery topic, debug_image gated on param+subscribers). §2 server → Tasks 5-6 (all 8 endpoints, observer heuristic `held/observer`, params incl. `waving_service='detect_waving_persons'`). §3 UI → Task 7 (badge colors, hold countdown, smallest-area click rule, wave overlays, event log, reconnect). §4 errors → stale banner (app.js interval), toasts-as-log entries, try/except'd bridge calls, never-raise `_publish_debug_outputs`. §5 testing → Tasks 1,2,3,5,7 + import checks + deferral note. §6 deployment → Tasks 6 (entry point, data_files), 5 Step 0 (deps), 8 (docs). Acceptance-1 (defaults off ⇒ unchanged) → all three params default false and `_publish_debug_outputs` early-outs.
- **Placeholder scan:** none; every code step has complete code. The two "READ before editing" notes (periodic-validation stash site, `YOLOTracker(` call site) name the exact symbol + file and instruct adapting local names — discovery, not placeholders.
- **Type consistency:** bridge contract identical in `track_web_app.py` docstring, FakeBridge, and `TrackWebNode` (snapshot/latest_state/latest_gallery/latest_jpeg/start_goal/stop_goal/reseed/wave); `maybe_add(feature, thumb=None)` matches Task 1 ↔ Task 3 call; `build_debug_state` kwargs match Task 2 def ↔ Task 4 call; gallery JSON `{version, thumbs}` matches Task 4 ↔ app.js `renderGallery`.
