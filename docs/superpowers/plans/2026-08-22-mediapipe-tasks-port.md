# MediaPipe 1.0.1 Tasks-API Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `waving_person_server`'s per-ROI pose pass from the removed `mp.solutions.pose` API on mediapipe 0.10.9 to the Tasks `PoseLandmarker` on mediapipe 1.0.1 (GPU delegate, CPU fallback) with a frozen-fixture test proving identical `is_waving` verdicts.

**Architecture:** A new import-light adapter `tk_vision_specialized/_pose_backend.py` wraps `PoseLandmarker` and returns landmarks shaped like the legacy API so `is_waving` and the overlay code barely change. A fixture recorded under 0.10.9 *before* the venv is touched becomes a pytest that must stay green after the swap. The `.task` model is located by a new download-free `weights_cache.find_cached()`; the only downloader is `scripts/download_models.py`.

**Tech Stack:** Python 3.10, mediapipe 1.0.1 (Tasks API, ctypes over `libmediapipe.so`), numpy 1.26.4, OpenCV 4.9, ROS 2 Humble, pytest, uv (lock file).

**Spec:** `docs/superpowers/specs/2026-08-22-mediapipe-tasks-port-design.md`

## Global Constraints

- Only `.venv-vision-main` changes, and inside it only the `mediapipe` package: `0.10.9 → 1.0.1`. `pip freeze` before/after must differ in exactly that one line; `pip check` must be clean.
- `protobuf==3.20.3` stays (tensorboard 2.11.2 needs it). `.venv-da3`, `.venv-fs`, `.venv-calib` untouched. jax/jaxlib orphans left alone.
- Default `waving_detector` stays `'vlm'`. `is_waving` body unchanged. Default pose model is `pose_landmarker_full.task` (same weights as legacy `model_complexity=1`).
- The rgb8/bgr8 normalizer, lite/heavy tuning, and YOLO-pose are **out of scope**.
- Venvs are git-ignored and live only in the main checkout. Code is edited in the worktree; every command below uses absolute venv paths. Define once per shell:
  ```bash
  export WT=/home/tinker/tk25_ws/src/tk26_vision/.claude/worktrees/mediapipe-tasks-port
  export MAIN=/home/tinker/tk25_ws/src/tk26_vision
  export VENV=$MAIN/.venv-vision-main
  source /opt/ros/humble/setup.bash
  source /home/tinker/tk25_ws/install/setup.bash
  export ROS2_PTH_WARNED=1
  # worktree source must shadow the stale colcon install tree for vision_util
  export PYTHONPATH=$WT/src/vision_util:$WT/src/tk_vision_specialized:$PYTHONPATH
  cd $WT
  ```
- Test runner: `$VENV/bin/python -m pytest <path> -v` from `$WT` with the exports above.
- Commit after every task with the trailer:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01QtfVtdCLKh4HoFMVBd2TvR
  ```

---

## File structure

| Path | Responsibility |
|---|---|
| `src/vision_util/vision_util/weights_cache.py` (modify) | add `find_cached(name) -> Path \| None`: cache lookup without download |
| `scripts/tests/test_weights_cache.py` (modify) | tests for `find_cached` |
| `scripts/tests/record_pose_fixture.py` (create) | one-off: run 0.10.9 Solutions on crops, write expected JSON |
| `src/tk_vision_specialized/test/fixtures/pose_parity/` (create) | `*.png` crops, `expected_0.10.9.json`, `README.md` |
| `src/tk_vision_specialized/tk_vision_specialized/_pose_backend.py` (create) | `PoseLandmarkIdx`, `Landmark`, `POSE_CONNECTIONS`, `PoseBackend`, `draw_pose` |
| `src/tk_vision_specialized/test/test_pose_backend.py` (create) | fallback + drawing unit tests (mocked mediapipe) |
| `src/tk_vision_specialized/test/test_pose_parity.py` (create) | the regression gate |
| `src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py` (modify) | use the adapter; new params |
| `src/tk_vision_specialized/tk_vision_specialized/check_waving_inference.py` (modify) | use the adapter |
| `scripts/tests/debug_waving_pipeline.py` (modify) | use the adapter |
| `scripts/download_models.py` (modify) | fetch the `.task` file |
| `.venv-vision-main.uv-project/pyproject.toml`, `uv.lock`, `requirements.txt` (modify) | pin 1.0.1 |
| `CLAUDE.md`, `src/tk_vision_specialized/README.md`, `DEV_NOTES.md` (modify) | docs |

---

### Task 1: `weights_cache.find_cached`

**Files:**
- Modify: `src/vision_util/vision_util/weights_cache.py` (append after `resolve_weights`, ~line 141)
- Test: `scripts/tests/test_weights_cache.py`

**Interfaces:**
- Produces: `find_cached(name: str) -> Path | None`. Absolute existing path → itself; absolute missing → `None`; bare name → first hit in `_search_paths(name)` else `None`; name containing a separator → `ValueError` (same rule as `resolve_weights`). Never downloads.

- [ ] **Step 1: Write the failing tests** — append to `scripts/tests/test_weights_cache.py`:

```python
from vision_util.weights_cache import find_cached


def test_find_cached_returns_cache_hit(isolated_cache):
    path = _touch(isolated_cache, "pose_landmarker_full.task")
    assert find_cached("pose_landmarker_full.task") == path


def test_find_cached_returns_none_on_miss_without_downloading(isolated_cache, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("find_cached must never download")
    monkeypatch.setattr(weights_cache, "_download", _boom)
    assert find_cached("pose_landmarker_full.task") is None


def test_find_cached_honours_env_override(isolated_cache, monkeypatch, tmp_path):
    override = tmp_path / "override"
    path = _touch(override, "pose_landmarker_full.task")
    monkeypatch.setenv(weights_cache._ENV_VAR, str(override))
    assert find_cached("pose_landmarker_full.task") == path


def test_find_cached_absolute_path(tmp_path):
    path = _touch(tmp_path, "x.task")
    assert find_cached(str(path)) == path
    assert find_cached(str(tmp_path / "missing.task")) is None


def test_find_cached_rejects_relative_with_separator(isolated_cache):
    with pytest.raises(ValueError):
        find_cached("models/pose_landmarker_full.task")
```

- [ ] **Step 2: Run to verify failure**

Run: `$VENV/bin/python -m pytest scripts/tests/test_weights_cache.py -k find_cached -v`
Expected: 5 × FAIL / ERROR with `ImportError: cannot import name 'find_cached'`.

- [ ] **Step 3: Implement** — append to `weights_cache.py`:

```python
def find_cached(name: str) -> "Path | None":
    """Locate ``name`` in the weight cache **without** downloading.

    Same path rules as :func:`resolve_weights` (absolute path honoured,
    relative path with separators rejected, otherwise the
    ``$TK26_MODEL_CACHE`` / ``~/.cache/tk26_vision/weights`` search order),
    but a miss returns ``None`` instead of invoking the Ultralytics
    downloader. Use it for non-Ultralytics assets such as MediaPipe
    ``.task`` bundles, which ``scripts/download_models.py`` stages.
    """
    if not name:
        raise ValueError("find_cached: name must be non-empty")
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    if os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(
            f"find_cached: relative paths with separators are rejected "
            f"(got {name!r}) — pass a bare filename or an absolute path"
        )
    for path in _search_paths(name):
        if path.exists():
            return path
    return None
```

Also add `find_cached` to the module docstring's "Lookup order" paragraph: one line — "``find_cached`` performs steps 1–4 only (no download) for non-Ultralytics assets."

- [ ] **Step 4: Run tests**

Run: `$VENV/bin/python -m pytest scripts/tests/test_weights_cache.py -v`
Expected: all PASS (existing `resolve_weights` tests included).

- [ ] **Step 5: Commit**

```bash
git add src/vision_util/vision_util/weights_cache.py scripts/tests/test_weights_cache.py
git commit -m "feat(vision_util): weights_cache.find_cached — cache lookup without download"
```

---

### Task 2: Record the 0.10.9 parity fixture (BEFORE any venv change)

**Files:**
- Create: `scripts/tests/record_pose_fixture.py`
- Create: `src/tk_vision_specialized/test/fixtures/pose_parity/{NN.png, expected_0.10.9.json, README.md}`

**Interfaces:**
- Produces the fixture JSON schema consumed by Task 4:
  ```json
  {"mediapipe_version": "0.10.9",
   "solutions_options": {"static_image_mode": true, "min_detection_confidence": 0.5, "model_complexity": 1},
   "crops": [{"file": "00.png", "detected": true, "is_waving": false,
              "landmarks": [[x, y, z, visibility], ... 33 entries]} ]}
  ```
  `landmarks` is `null` when `detected` is false. Verdicts come from the node's own `is_waving` method, called unbound with a stub `self`.

- [ ] **Step 1: Confirm 0.10.9 is still installed** (this task is meaningless otherwise)

Run: `$VENV/bin/python -c "import mediapipe as mp; print(mp.__version__); mp.solutions.pose"`
Expected: `0.10.9`, no error. If it prints 1.0.1, STOP — the fixture cannot be recorded; reinstall 0.10.9 first (`$VENV/bin/pip install mediapipe==0.10.9`).

- [ ] **Step 2: Write the recorder** — `scripts/tests/record_pose_fixture.py`:

```python
#!/usr/bin/env python3
"""Record the legacy mediapipe 0.10.9 Solutions pose output on person crops.

One-off, run BEFORE upgrading mediapipe. Produces the fixture that
``test_pose_parity.py`` replays against the Tasks-API adapter.

    python scripts/tests/record_pose_fixture.py \
        --images <img.jpg ...> --out src/tk_vision_specialized/test/fixtures/pose_parity

Crops are produced by the node's YOLO11m-seg (conf 0.4, CPU) so they match
what waving_person_server feeds the pose model.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import mediapipe as mp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "vision_util"))
sys.path.insert(0, str(REPO / "src" / "tk_vision_specialized"))

from vision_util.weights_cache import resolve_weights  # noqa: E402
from tk_vision_specialized.waving_person_server import DetectWavingPersonsNode  # noqa: E402


class _Stub:
    """Minimal ``self`` so the node's is_waving runs without rclpy."""
    MIN_VISIBILITY = DetectWavingPersonsNode.MIN_VISIBILITY
    ELBOW_TOL_NORM = DetectWavingPersonsNode.ELBOW_TOL_NORM

    def get_logger(self):
        return logging.getLogger("record_pose_fixture")


def crop_persons(yolo, img, conf=0.4):
    res = yolo(img, conf=conf, verbose=False, device="cpu")[0]
    for box in res.boxes:
        if yolo.names[int(box.cls[0])] != "person":
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        roi = img[y1:y2, x1:x2]
        if roi.size:
            yield roi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11m-seg.pt")
    args = ap.parse_args()

    assert mp.__version__ == "0.10.9", f"fixture must be recorded on 0.10.9, got {mp.__version__}"
    from ultralytics import YOLO
    yolo = YOLO(str(resolve_weights(args.model)))
    opts = dict(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
    pose = mp.solutions.pose.Pose(**opts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stub = _Stub()
    crops = []
    n = 0
    for img_path in args.images:
        img = cv2.imread(img_path)
        assert img is not None, img_path
        for roi in crop_persons(yolo, img):
            name = f"{n:02d}.png"
            cv2.imwrite(str(out / name), roi)
            res = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            lm = res.pose_landmarks
            entry = {"file": name, "source": Path(img_path).name,
                     "detected": lm is not None, "landmarks": None,
                     "is_waving": DetectWavingPersonsNode.is_waving(stub, lm, roi)}
            if lm is not None:
                entry["landmarks"] = [[round(p.x, 6), round(p.y, 6), round(p.z, 6),
                                       round(p.visibility, 6)] for p in lm.landmark]
            crops.append(entry)
            print(f"{name}: detected={entry['detected']} waving={entry['is_waving']} ({img_path})")
            n += 1

    (out / "expected_0.10.9.json").write_text(json.dumps(
        {"mediapipe_version": mp.__version__, "solutions_options": opts, "crops": crops},
        indent=1))
    print(f"wrote {n} crops to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: (Optional but valuable) grab live frames.** If the Orbbec is running (`ros2 topic hz /camera/color/image_raw` shows ~30 Hz), have someone stand in view and run the existing dumper twice — once waving, once arms down:

```bash
ros2 run tk_vision_specialized check_waving_inference --ros-args -p max_runs:=3 -p interval_sec:=1.0 -p output_root:=/home/tinker/.claude/jobs/df430f11/tmp/live_wave
ros2 run tk_vision_specialized check_waving_inference --ros-args -p max_runs:=3 -p interval_sec:=1.0 -p output_root:=/home/tinker/.claude/jobs/df430f11/tmp/live_still
```
Collect the `*_raw.jpg` files it writes. If the camera is not up, skip this step and say so in the README (Step 5).

- [ ] **Step 4: Record**

```bash
ASSETS=$VENV/lib/python3.10/site-packages/ultralytics/assets
$VENV/bin/python scripts/tests/record_pose_fixture.py \
    --images $ASSETS/bus.jpg $ASSETS/zidane.jpg $(ls /home/tinker/.claude/jobs/df430f11/tmp/live_*/**/*_raw.jpg 2>/dev/null) \
    --out src/tk_vision_specialized/test/fixtures/pose_parity
```
Expected: one line per crop; the two stock images alone yield 6 crops (3 detected). Check `du -sh src/tk_vision_specialized/test/fixtures/pose_parity` is under ~1 MB; if live frames pushed it higher, drop duplicates (keep ≤ 20 crops) and re-run.

- [ ] **Step 5: Write `README.md` in the fixture dir**

```markdown
# pose_parity fixture

Person crops + the mediapipe **0.10.9** `mp.solutions.pose` output
(`static_image_mode=True, min_detection_confidence=0.5, model_complexity=1`)
recorded by `scripts/tests/record_pose_fixture.py` on 2026-08-22.

`test_pose_parity.py` replays these through `_pose_backend.PoseBackend`
(mediapipe ≥ 1.0, Tasks API, `pose_landmarker_full.task`) and requires
identical `is_waving` verdicts and near-identical landmarks.

Sources: ultralytics `bus.jpg`, `zidane.jpg` (crops 00–05); live Orbbec frames
<list them here, or write "none — camera was not running when recorded">.

Do not regenerate under a newer mediapipe; the value of this fixture is that
it encodes the legacy behaviour.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/tests/record_pose_fixture.py src/tk_vision_specialized/test/fixtures/pose_parity
git commit -m "test: freeze mediapipe 0.10.9 pose landmarks + is_waving verdicts as parity fixture"
```

---

### Task 3: `_pose_backend.py` adapter with unit tests (mocked mediapipe)

**Files:**
- Create: `src/tk_vision_specialized/tk_vision_specialized/_pose_backend.py`
- Test: `src/tk_vision_specialized/test/test_pose_backend.py`

**Interfaces:**
- Produces:
  - `class PoseLandmarkIdx(IntEnum)`: `NOSE=0, LEFT_SHOULDER=11, RIGHT_SHOULDER=12, LEFT_ELBOW=13, RIGHT_ELBOW=14, LEFT_WRIST=15, RIGHT_WRIST=16`
  - `POSE_CONNECTIONS: tuple[tuple[int, int], ...]` (35 edges)
  - `@dataclass Landmark(x: float, y: float, z: float, visibility: float)`
  - `class PoseBackend(model_path: str, delegate: str = 'gpu', min_detection_confidence: float = 0.5)` with attributes `active_delegate: str`, `fallback_reason: str | None`, methods `process(rgb: np.ndarray) -> list[Landmark] | None`, `close() -> None`
  - `draw_pose(bgr: np.ndarray, landmarks: list[Landmark], connections=POSE_CONNECTIONS) -> None` (in place)
  - `POSE_MODEL_URL: str` (the full-model download URL, used by Task 7)

- [ ] **Step 1: Write the failing unit tests** — `src/tk_vision_specialized/test/test_pose_backend.py`:

```python
"""Unit tests for _pose_backend that do not need a real model file.

The mediapipe Tasks landmarker is replaced with a fake so these run anywhere.
Real-model behaviour is covered by test_pose_parity.py.
"""
import numpy as np
import pytest

from tk_vision_specialized import _pose_backend as pb
from tk_vision_specialized._pose_backend import (
    Landmark, PoseBackend, PoseLandmarkIdx, POSE_CONNECTIONS, draw_pose,
)


class _FakeNormLm:
    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _FakeResult:
    def __init__(self, poses):
        self.pose_landmarks = poses


class _FakeLandmarker:
    created = []

    def __init__(self, delegate, poses):
        self.delegate = delegate
        self.poses = poses
        self.closed = False
        _FakeLandmarker.created.append(self)

    def detect(self, _image):
        return _FakeResult(self.poses)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_mp(monkeypatch):
    """Patch PoseBackend's factory so no model file / libmediapipe is needed."""
    calls = {"gpu_raises": False, "poses": [[_FakeNormLm(0.1, 0.2, 0.0, 0.9)] * 33]}

    def _create(model_path, delegate, min_conf):
        if delegate == "gpu" and calls["gpu_raises"]:
            raise RuntimeError("no EGL")
        return _FakeLandmarker(delegate, calls["poses"])

    monkeypatch.setattr(pb, "_create_landmarker", _create)
    monkeypatch.setattr(pb, "_to_mp_image", lambda rgb: rgb)
    _FakeLandmarker.created.clear()
    return calls


def test_enum_matches_blazepose_indices():
    assert PoseLandmarkIdx.NOSE == 0
    assert PoseLandmarkIdx.LEFT_SHOULDER == 11 and PoseLandmarkIdx.RIGHT_SHOULDER == 12
    assert PoseLandmarkIdx.LEFT_ELBOW == 13 and PoseLandmarkIdx.RIGHT_ELBOW == 14
    assert PoseLandmarkIdx.LEFT_WRIST == 15 and PoseLandmarkIdx.RIGHT_WRIST == 16
    assert len(POSE_CONNECTIONS) == 35
    assert all(0 <= a < 33 and 0 <= b < 33 for a, b in POSE_CONNECTIONS)


def test_gpu_first_success(fake_mp):
    be = PoseBackend("dummy.task", delegate="gpu")
    assert be.active_delegate == "gpu"
    assert be.fallback_reason is None


def test_gpu_failure_falls_back_to_cpu(fake_mp):
    fake_mp["gpu_raises"] = True
    be = PoseBackend("dummy.task", delegate="gpu")
    assert be.active_delegate == "cpu"
    assert "no EGL" in be.fallback_reason
    # the failed GPU attempt must not leak an open landmarker
    assert all(l.closed for l in _FakeLandmarker.created if l.delegate == "gpu")


def test_cpu_requested_never_tries_gpu(fake_mp):
    fake_mp["gpu_raises"] = True  # would raise if attempted
    be = PoseBackend("dummy.task", delegate="cpu")
    assert be.active_delegate == "cpu" and be.fallback_reason is None
    assert [l.delegate for l in _FakeLandmarker.created] == ["cpu"]


def test_invalid_delegate_rejected(fake_mp):
    with pytest.raises(ValueError):
        PoseBackend("dummy.task", delegate="tpu")


def test_process_returns_landmark_list_indexable_by_enum(fake_mp):
    be = PoseBackend("dummy.task", delegate="cpu")
    lms = be.process(np.zeros((64, 64, 3), np.uint8))
    assert len(lms) == 33
    lm = lms[PoseLandmarkIdx.RIGHT_WRIST]
    assert isinstance(lm, Landmark)
    assert (lm.x, lm.y, lm.visibility) == (0.1, 0.2, 0.9)


def test_process_returns_none_when_no_pose(fake_mp):
    fake_mp["poses"] = []
    be = PoseBackend("dummy.task", delegate="cpu")
    assert be.process(np.zeros((64, 64, 3), np.uint8)) is None


def test_process_rejects_non_rgb_uint8(fake_mp):
    be = PoseBackend("dummy.task", delegate="cpu")
    with pytest.raises(ValueError):
        be.process(np.zeros((64, 64), np.uint8))


def test_close_is_idempotent(fake_mp):
    be = PoseBackend("dummy.task", delegate="cpu")
    be.close(); be.close()
    assert _FakeLandmarker.created[-1].closed


def test_draw_pose_modifies_image_and_tolerates_none():
    img = np.zeros((100, 80, 3), np.uint8)
    lms = [Landmark(0.5, 0.5, 0.0, 1.0)] * 33
    draw_pose(img, lms)
    assert img.any()
    untouched = np.zeros((10, 10, 3), np.uint8)
    draw_pose(untouched, None)
    assert not untouched.any()
```

- [ ] **Step 2: Run to verify failure**

Run: `$VENV/bin/python -m pytest src/tk_vision_specialized/test/test_pose_backend.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'tk_vision_specialized._pose_backend'`.

- [ ] **Step 3: Implement** — `src/tk_vision_specialized/tk_vision_specialized/_pose_backend.py`:

```python
"""Pose-estimation backend for waving detection (MediaPipe Tasks API).

Wraps ``mediapipe.tasks.python.vision.PoseLandmarker`` (mediapipe >= 1.0)
behind the small surface ``waving_person_server`` needs, returning landmarks
shaped like the legacy ``mp.solutions.pose`` output so the ``is_waving``
heuristic is untouched.

Import-light on purpose: mediapipe, numpy, cv2 — no rclpy.

GPU delegate notes (Ubuntu only): creation takes 3–6 s once and mediapipe
prints an ``Unable to initialize EGL`` probe error even when it succeeds.
We verify with a warm-up ``detect`` and fall back to CPU on any failure.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

POSE_MODEL_FILENAME = "pose_landmarker_full.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


class PoseLandmarkIdx(enum.IntEnum):
    """BlazePose landmark indices used by ``is_waving`` (33-point topology)."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16


# BlazePose skeleton (same edge list as the legacy mp.solutions.pose.POSE_CONNECTIONS).
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


_VALID_DELEGATES = ("gpu", "cpu")


def _create_landmarker(model_path: str, delegate: str, min_conf: float):
    """Build a Tasks PoseLandmarker (IMAGE mode, one pose). Separated for tests."""
    from mediapipe.tasks.python import BaseOptions, vision
    deleg = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=deleg),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_conf,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def _to_mp_image(rgb: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class PoseBackend:
    """Single-person pose landmarks for one RGB crop at a time.

    ``delegate='gpu'`` tries the GPU delegate and silently rebuilds on CPU if
    creation or a warm-up inference fails; ``active_delegate`` /
    ``fallback_reason`` tell the caller what happened so it can log once.
    """

    def __init__(self, model_path: str, delegate: str = "gpu",
                 min_detection_confidence: float = 0.5):
        if delegate not in _VALID_DELEGATES:
            raise ValueError(f"pose delegate must be one of {_VALID_DELEGATES}, got {delegate!r}")
        self.model_path = model_path
        self.active_delegate: str = delegate
        self.fallback_reason: Optional[str] = None
        self._lm = None
        if delegate == "gpu":
            try:
                lm = _create_landmarker(model_path, "gpu", min_detection_confidence)
                try:
                    lm.detect(_to_mp_image(np.zeros((256, 256, 3), np.uint8)))  # warm-up / probe
                except Exception:
                    lm.close()
                    raise
                self._lm = lm
            except Exception as exc:  # noqa: BLE001 — any failure means "use CPU"
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                self.active_delegate = "cpu"
        if self._lm is None:
            self._lm = _create_landmarker(model_path, "cpu", min_detection_confidence)

    def process(self, rgb: np.ndarray) -> Optional[list]:
        """Return 33 normalized ``Landmark`` for the first pose, or ``None``."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"process expects HxWx3 uint8 RGB, got {rgb.shape} {rgb.dtype}")
        result = self._lm.detect(_to_mp_image(rgb))
        if not result.pose_landmarks:
            return None
        return [Landmark(p.x, p.y, p.z, float(p.visibility or 0.0))
                for p in result.pose_landmarks[0]]

    def close(self) -> None:
        if self._lm is not None:
            self._lm.close()
            self._lm = None

    def __del__(self):  # best effort; explicit close() preferred
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def draw_pose(bgr: np.ndarray, landmarks, connections=POSE_CONNECTIONS) -> None:
    """Draw joints + skeleton onto ``bgr`` in place (no-op for ``None``)."""
    if landmarks is None or bgr.size == 0:
        return
    h, w = bgr.shape[:2]
    pts = [(int(round(lm.x * w)), int(round(lm.y * h))) for lm in landmarks]
    for a, b in connections:
        cv2.line(bgr, pts[a], pts[b], (255, 255, 255), 2)
    for (x, y), lm in zip(pts, landmarks):
        color = (0, 255, 0) if lm.visibility >= 0.5 else (0, 0, 255)
        cv2.circle(bgr, (x, y), 3, color, -1)
```

- [ ] **Step 4: Run tests**

Run: `$VENV/bin/python -m pytest src/tk_vision_specialized/test/test_pose_backend.py -v`
Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tk_vision_specialized/tk_vision_specialized/_pose_backend.py src/tk_vision_specialized/test/test_pose_backend.py
git commit -m "feat(tk_vision_specialized): PoseBackend adapter over mediapipe Tasks PoseLandmarker"
```

---

### Task 4: Parity test (runs on 0.10.9's Tasks API now, and on 1.0.1 after Task 5)

**Files:**
- Test: `src/tk_vision_specialized/test/test_pose_parity.py`

**Interfaces:**
- Consumes: fixture from Task 2; `PoseBackend`, `PoseLandmarkIdx` from Task 3; `find_cached` from Task 1; `DetectWavingPersonsNode.is_waving` (unbound, stub self).

- [ ] **Step 1: Stage the model file** (the recorder needed no `.task`; this test does)

```bash
CACHE=${TK26_MODEL_CACHE:-$HOME/.cache/tk26_vision/weights}; mkdir -p $CACHE
curl -fsSL -o $CACHE/pose_landmarker_full.task.part \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task \
  && mv $CACHE/pose_landmarker_full.task.part $CACHE/pose_landmarker_full.task
ls -la $CACHE/pose_landmarker_full.task   # ~9.4 MB
```

- [ ] **Step 2: Write the test**

```python
"""Regression gate: the Tasks-API backend must reproduce the mediapipe 0.10.9
Solutions verdicts recorded in fixtures/pose_parity (see its README).

Needs the real model file in the weights cache (scripts/download_models.py);
skips — loudly — if it is absent so weight-less CI doesn't fail, but T0 on the
robot must have it so the skip never masks a regression.
"""
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_util.weights_cache import find_cached
from tk_vision_specialized._pose_backend import PoseBackend, PoseLandmarkIdx
from tk_vision_specialized.waving_person_server import DetectWavingPersonsNode

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pose_parity"
EXPECTED = json.loads((FIXTURE_DIR / "expected_0.10.9.json").read_text())
JOINTS = [PoseLandmarkIdx.NOSE, PoseLandmarkIdx.LEFT_SHOULDER, PoseLandmarkIdx.RIGHT_SHOULDER,
          PoseLandmarkIdx.LEFT_ELBOW, PoseLandmarkIdx.RIGHT_ELBOW,
          PoseLandmarkIdx.LEFT_WRIST, PoseLandmarkIdx.RIGHT_WRIST]

MODEL = find_cached("pose_landmarker_full.task")
pytestmark = pytest.mark.skipif(
    MODEL is None, reason="pose_landmarker_full.task not in weights cache — run scripts/download_models.py")


class _Stub:
    MIN_VISIBILITY = DetectWavingPersonsNode.MIN_VISIBILITY
    ELBOW_TOL_NORM = DetectWavingPersonsNode.ELBOW_TOL_NORM

    def get_logger(self):
        return logging.getLogger("parity")


def _verdict(landmarks, roi):
    return DetectWavingPersonsNode.is_waving(_Stub(), landmarks, roi)


def _crops():
    for entry in EXPECTED["crops"]:
        bgr = cv2.imread(str(FIXTURE_DIR / entry["file"]))
        assert bgr is not None, entry["file"]
        yield entry, bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def cpu_backend():
    be = PoseBackend(str(MODEL), delegate="cpu")
    yield be
    be.close()


@pytest.fixture(scope="module")
def gpu_backend():
    be = PoseBackend(str(MODEL), delegate="gpu")
    if be.active_delegate != "gpu":
        be.close()
        pytest.skip(f"GPU delegate unavailable: {be.fallback_reason}")
    yield be
    be.close()


def _check(backend, y_tol, vis_tol):
    mismatches = []
    for entry, bgr, rgb in _crops():
        lms = backend.process(rgb)
        detected = lms is not None
        if detected != entry["detected"]:
            mismatches.append(f"{entry['file']}: detected {detected} != {entry['detected']}")
            continue
        verdict = _verdict(lms, bgr)
        if verdict != entry["is_waving"]:
            mismatches.append(f"{entry['file']}: is_waving {verdict} != {entry['is_waving']}")
        if detected:
            for j in JOINTS:
                ex, ey, ez, ev = entry["landmarks"][int(j)]
                dy, dv = abs(lms[j].y - ey), abs(lms[j].visibility - ev)
                if dy > y_tol or dv > vis_tol:
                    mismatches.append(f"{entry['file']} {j.name}: dy={dy:.4f} dvis={dv:.4f}")
    assert not mismatches, "\n".join(mismatches)


def test_fixture_is_legacy():
    assert EXPECTED["mediapipe_version"] == "0.10.9"
    assert len(EXPECTED["crops"]) >= 6
    assert any(c["detected"] for c in EXPECTED["crops"])


def test_cpu_parity(cpu_backend):
    _check(cpu_backend, y_tol=0.01, vis_tol=0.05)


def test_gpu_parity(gpu_backend):
    # fp16 GPU path drifts more in coordinates; verdicts must still be identical
    _check(gpu_backend, y_tol=0.05, vis_tol=0.15)


def test_gpu_fallback_keeps_parity(monkeypatch):
    from tk_vision_specialized import _pose_backend as pb
    real = pb._create_landmarker

    def _gpu_breaks(model_path, delegate, min_conf):
        if delegate == "gpu":
            raise RuntimeError("forced GPU failure")
        return real(model_path, delegate, min_conf)

    monkeypatch.setattr(pb, "_create_landmarker", _gpu_breaks)
    be = PoseBackend(str(MODEL), delegate="gpu")
    try:
        assert be.active_delegate == "cpu" and "forced" in be.fallback_reason
        _check(be, y_tol=0.01, vis_tol=0.05)
    finally:
        be.close()
```

- [ ] **Step 3: Run on the still-installed 0.10.9** (its Tasks API exists; this is a free pre-upgrade check)

Run: `$VENV/bin/python -m pytest src/tk_vision_specialized/test/test_pose_parity.py -v 2>&1 | grep -v "^I0000\|^W0000\|^E0000\|^    @\|^Stack trace"`
Expected: `test_fixture_is_legacy`, `test_cpu_parity`, `test_gpu_fallback_keeps_parity` PASS; `test_gpu_parity` PASS or SKIPPED (0.10.9's delegate may not init — a skip here is fine, it must pass after Task 5). If `test_cpu_parity` fails, the adapter is wrong (not the upgrade) — fix before continuing; compare against `record_pose_fixture.py` for e.g. a BGR/RGB slip.

- [ ] **Step 4: Commit**

```bash
git add src/tk_vision_specialized/test/test_pose_parity.py
git commit -m "test: frozen-fixture parity gate for the Tasks-API pose backend"
```

---

### Task 5: Upgrade `.venv-vision-main` to mediapipe 1.0.1 and prove nothing else moved

**Files:**
- Modify: `.venv-vision-main.uv-project/pyproject.toml:57` (`"mediapipe==0.10.9"` → `"mediapipe==1.0.1"`)
- Modify: `.venv-vision-main.uv-project/uv.lock` (re-lock)
- Modify: `requirements.txt:135` (`mediapipe==0.10.9` → `mediapipe==1.0.1`)
- Refresh: `$VENV/freeze.lock.txt` (git-ignored, main checkout)

- [ ] **Step 1: Snapshot before**

```bash
$VENV/bin/pip freeze | sort > /home/tinker/.claude/jobs/df430f11/tmp/freeze_before.txt
$VENV/bin/pip check; echo "pip check exit=$?"
```
Expected: `pip check` exit 0 (if it is already non-zero, record the output — it must not get worse).

- [ ] **Step 2: Install exactly one package**

```bash
$VENV/bin/pip install --no-deps mediapipe==1.0.1
$VENV/bin/pip check; echo "pip check exit=$?"
$VENV/bin/pip freeze | sort > /home/tinker/.claude/jobs/df430f11/tmp/freeze_after.txt
diff /home/tinker/.claude/jobs/df430f11/tmp/freeze_before.txt /home/tinker/.claude/jobs/df430f11/tmp/freeze_after.txt
```
Expected: diff shows only `< mediapipe==0.10.9` / `> mediapipe==1.0.1`; `pip check` exit 0 (`--no-deps` is safe because every declared dep is already satisfied — `pip check` is what proves that). If `pip check` reports a missing/incompatible dep, install only that dep and re-run the diff; anything else moving is a STOP-and-report.

- [ ] **Step 3: Import smoke + parity again**

```bash
$VENV/bin/python -c "import mediapipe as mp; from mediapipe.tasks.python import vision; print(mp.__version__); assert not hasattr(mp,'solutions')"
$VENV/bin/python -m pytest src/tk_vision_specialized/test/test_pose_backend.py src/tk_vision_specialized/test/test_pose_parity.py -v 2>&1 | grep -v "^I0000\|^W0000\|^E0000\|^    @\|^Stack trace"
```
Expected: prints `1.0.1`; all parity tests PASS, including `test_gpu_parity` on the robot (record the outcome verbatim in the DEV_NOTES entry, Task 9).

- [ ] **Step 4: Pin files**

```bash
sed -i 's/"mediapipe==0.10.9"/"mediapipe==1.0.1"/' .venv-vision-main.uv-project/pyproject.toml
sed -i 's/^mediapipe==0.10.9$/mediapipe==1.0.1/' requirements.txt
grep -n "mediapipe==" .venv-vision-main.uv-project/pyproject.toml requirements.txt
(cd .venv-vision-main.uv-project && uv lock)   # re-lock; if uv is unavailable, note it and skip
$VENV/bin/pip freeze > $VENV/freeze.lock.txt
git diff --stat
```
Expected: both greps show `1.0.1`; `uv.lock` changes only in the mediapipe block (inspect `git diff .venv-vision-main.uv-project/uv.lock | grep '^[-+]name'` — only `mediapipe` should appear). If uv re-resolves other packages, revert the lock and record that `uv lock` needs a dedicated follow-up; the `pip` state is the source of truth on the robot.

- [ ] **Step 5: Stack-wide smoke (the node still imports `mp.solutions` at this point, so the waving node is EXPECTED to fail here; everything else must pass)**

```bash
bash scripts/tests/t0_static.sh 2>&1 | tail -40
```
Expected: every row passes except the `waving_person_server` / `check_waving_inference` entry-point import rows. Any other failure = the venv change broke something → STOP and report.

- [ ] **Step 6: Commit the pins**

```bash
git add .venv-vision-main.uv-project/pyproject.toml .venv-vision-main.uv-project/uv.lock requirements.txt
git commit -m "deps(vision-main): mediapipe 0.10.9 -> 1.0.1 (Tasks API only; Solutions removed upstream)"
```

---

### Task 6: Wire `waving_person_server` to the adapter

**Files:**
- Modify: `src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py` — lines 12 (import), 84–94 (init), 361–366 (`destroy_node`), 453–470 and 472–497 (`_annotate_frame`, `_annotate_all_persons`), 678–683 (`is_waving` header), 920–928 (call site)
- Test: `src/tk_vision_specialized/test/test_waving_action.py` (existing; must stay green) + `test_pose_parity.py`

**Interfaces:**
- Consumes: `PoseBackend`, `PoseLandmarkIdx`, `draw_pose`, `POSE_MODEL_FILENAME` (Task 3); `find_cached` (Task 1).
- Produces: ROS params `pose_model_path` (string, default `'pose_landmarker_full.task'`), `pose_delegate` (string, default `'gpu'`).

- [ ] **Step 1: Imports** — replace line 12 `import mediapipe as mp` with nothing, and after `from vision_util.weights_cache import resolve_weights` add:

```python
from vision_util.weights_cache import find_cached
from ._pose_backend import (
    PoseBackend,
    PoseLandmarkIdx,
    POSE_MODEL_FILENAME,
    draw_pose,
)
```

- [ ] **Step 2: Init** — replace lines 87–94 (`self.mp_pose = ...` through the `Pose(...)` call) with:

```python
        # Pose backend (MediaPipe Tasks PoseLandmarker). IMAGE mode with one
        # pose per call is the Tasks equivalent of the legacy
        # static_image_mode=True: each YOLO ROI is independent, no tracker
        # state leaks between crops.
        self.declare_parameter('pose_model_path', POSE_MODEL_FILENAME)
        self.declare_parameter('pose_delegate', 'gpu')
        pose_model_name = self.get_parameter('pose_model_path').get_parameter_value().string_value
        pose_delegate = self.get_parameter('pose_delegate').get_parameter_value().string_value
        pose_model = find_cached(pose_model_name)
        if pose_model is None:
            raise RuntimeError(
                f'Pose model {pose_model_name!r} not found in the weights cache; '
                f'run scripts/download_models.py (or set pose_model_path to an absolute path).')
        self.pose = PoseBackend(str(pose_model), delegate=pose_delegate,
                                min_detection_confidence=0.5)
        if self.pose.fallback_reason:
            self.get_logger().warning(
                f'pose delegate: requested {pose_delegate!r}, running on cpu '
                f'({self.pose.fallback_reason})')
        else:
            self.get_logger().info(f'pose delegate: {self.pose.active_delegate} ({pose_model})')
```

- [ ] **Step 3: destroy_node** — add `self.pose.close()` as the first line of `destroy_node` (line 362).

- [ ] **Step 4: Drawing sites** — in `_annotate_frame` (≈466) and `_annotate_all_persons` (≈493) replace

```python
                    self.mp_draw.draw_landmarks(
                        roi, landmarks, self.mp_pose.POSE_CONNECTIONS)
```
with
```python
                    draw_pose(roi, landmarks)
```
(both sites; `roi` is a view into `frame`, so in-place drawing lands on the frame exactly as before).

- [ ] **Step 5: `is_waving`** — replace lines 682–683

```python
        landmarks = pose_landmarks.landmark
        PL = mp.solutions.pose.PoseLandmark
```
with
```python
        landmarks = pose_landmarks
        PL = PoseLandmarkIdx
```
Nothing else in the method changes.

- [ ] **Step 6: Call site** — replace lines 923–925

```python
                        pose_results = self.pose.process(
                            cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                        landmarks = pose_results.pose_landmarks
```
with
```python
                        landmarks = self.pose.process(
                            cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
```

- [ ] **Step 7: Verify no legacy references remain**

Run: `grep -n "mp\.\|mp_pose\|mp_draw\|mediapipe" src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py`
Expected: only comment lines mentioning "MediaPipe" (mode docs); no `mp.`/`mp_pose`/`mp_draw` code.

- [ ] **Step 8: Tests + import smoke**

```bash
$VENV/bin/python -m pytest src/tk_vision_specialized/test/test_pose_parity.py src/tk_vision_specialized/test/test_waving_action.py src/tk_vision_specialized/test/test_waving_vlm.py -v 2>&1 | tail -15
$VENV/bin/python -c "import tk_vision_specialized.waving_person_server as w; print('import ok')"
```
Expected: all PASS; `import ok`.

- [ ] **Step 9: Build + start the node once**

```bash
./scripts/build.sh --packages-select vision_util tk_vision_specialized 2>&1 | tail -5
timeout 40 ros2 run tk_vision_specialized waving_person_server --ros-args -p show_window:=false 2>&1 | grep -i "pose delegate\|error\|Traceback" | head
```
Expected: a `pose delegate: gpu (...)` line (or `running on cpu (...)` with a reason, which is acceptable but must be reported); no Traceback. (The build runs from the worktree — `build.sh` resolves the workspace from its own location; if it insists on the main checkout, run the same command from `$MAIN` after merging, and note it.)

- [ ] **Step 10: Commit**

```bash
git add src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py
git commit -m "refactor(waving): pose pass via PoseBackend (mediapipe Tasks, GPU delegate w/ CPU fallback)"
```

---

### Task 7: `download_models.py` stages the `.task`; `check_waving_inference` + `debug_waving_pipeline` use the adapter

**Files:**
- Modify: `scripts/download_models.py` (docstring line 12; replace `warm_mediapipe` lines 69–81; flags lines 90, 101–102)
- Modify: `src/tk_vision_specialized/tk_vision_specialized/check_waving_inference.py` (lines 10, 35–40, 84–93, 188–228)
- Modify: `scripts/tests/debug_waving_pipeline.py` (lines 31–35, 40–78, 82–131, 167–177, 196)

**Interfaces:**
- Consumes: `POSE_MODEL_URL`, `POSE_MODEL_FILENAME`, `PoseBackend`, `PoseLandmarkIdx`, `draw_pose` (Task 3); `find_cached`, `_writable_cache` (Task 1 / existing).

- [ ] **Step 1: download_models.py** — replace `warm_mediapipe` with:

```python
POSE_TASK_NAME = "pose_landmarker_full.task"
POSE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_full/float16/latest/pose_landmarker_full.task")


def fetch_pose_landmarker() -> None:
    """Stage the MediaPipe Tasks pose bundle used by waving_person_server."""
    from urllib.request import urlopen
    from vision_util.weights_cache import _writable_cache, find_cached
    print("staging mediapipe pose landmarker (.task)…")
    existing = find_cached(POSE_TASK_NAME)
    if existing is not None:
        print(f"  ✓ {POSE_TASK_NAME:<26} {human_size(existing)}  ({existing})")
        return
    target = _writable_cache() / POSE_TASK_NAME
    part = target.with_suffix(target.suffix + ".part")
    with urlopen(POSE_TASK_URL, timeout=60) as resp, open(part, "wb") as fp:
        while chunk := resp.read(1 << 20):
            fp.write(chunk)
    part.replace(target)   # atomic: a partial file never satisfies find_cached
    print(f"  ✓ {POSE_TASK_NAME:<26} {human_size(target)}  ({target})")
```
Update the docstring bullet to `* MediaPipe Tasks pose landmarker bundle (waving detection)`. Replace the flag with
```python
    ap.add_argument("--skip-pose", "--skip-mediapipe", dest="skip_pose", action="store_true")
```
and the call with `if not args.skip_pose: fetch_pose_landmarker()`.

Verify: `mv $HOME/.cache/tk26_vision/weights/pose_landmarker_full.task /tmp/ptask.bak 2>/dev/null; $VENV/bin/python scripts/download_models.py --skip-ultralytics --skip-torchvision && $VENV/bin/python scripts/download_models.py --skip-ultralytics --skip-torchvision` — first run downloads (~9.4 MB), second run prints the ✓ hit without downloading; `rm -f /tmp/ptask.bak`.

- [ ] **Step 2: check_waving_inference.py** — line 10: drop `import mediapipe as mp`; add `from vision_util.weights_cache import find_cached` and `from ._pose_backend import PoseBackend, PoseLandmarkIdx, POSE_MODEL_FILENAME, draw_pose`. Replace lines 35–40 with:

```python
        pose_model = find_cached(POSE_MODEL_FILENAME)
        if pose_model is None:
            raise RuntimeError(f'{POSE_MODEL_FILENAME} missing; run scripts/download_models.py')
        self.pose = PoseBackend(str(pose_model), delegate='gpu')
        self.get_logger().info(f'pose delegate: {self.pose.active_delegate} '
                               f'{self.pose.fallback_reason or ""}')
```
In `_is_waving` (84–93) replace every `self.mp_pose.PoseLandmark.X` with `PoseLandmarkIdx.X`. At 188–189:
```python
                pose_lms = self.pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
```
then `'has_landmarks': pose_lms is not None`, `if pose_lms is not None: lm = pose_lms`, every `self.mp_pose.PoseLandmark.X` in the keypoints dict → `PoseLandmarkIdx.X`, and the draw block (224–228) → `draw_pose(roi_draw, pose_lms)`. Keep the `mediapipe_ms` JSON key name (the acceptance criterion reads it).

Verify: `grep -n "mp\.\|mp_pose\|mp_draw" src/tk_vision_specialized/tk_vision_specialized/check_waving_inference.py` → nothing; `$VENV/bin/python -c "import tk_vision_specialized.check_waving_inference"` → ok.

- [ ] **Step 3: debug_waving_pipeline.py** — `_lazy_imports` returns `YOLO, PoseBackend` (import `from tk_vision_specialized._pose_backend import PoseBackend, PoseLandmarkIdx, draw_pose` after inserting `REPO/src/tk_vision_specialized` and `REPO/src/vision_util` on `sys.path`). `is_waving_legacy` / `is_waving_fixed`: drop the `mp_mod` parameter, `PL = PoseLandmarkIdx`. `evaluate_image`: drop `mp_mod`; `pose_res = pose.process(rgb_roi)` → `lms`; `if lms is not None:` with `vis = [lms[k].visibility for k in ks]`, `verdict = predicate(lms, roi.shape[0])`, and `draw_landmarks_on_overlay(overlay, roi, lms, x1, y1)`. `draw_landmarks_on_overlay`: body becomes `roi_copy = roi.copy(); draw_pose(roi_copy, pose_landmarks); overlay[...] = roi_copy`. `main`: `pose = PoseBackend(str(find_cached("pose_landmarker_full.task")), delegate="gpu")` with a clear `SystemExit` if `find_cached` returns `None`.

Verify: `$VENV/bin/python scripts/tests/debug_waving_pipeline.py --help` works; `grep -n "mp_mod\|mp\.solutions" scripts/tests/debug_waving_pipeline.py` → nothing. (The labelled data dir no longer exists on this box; a full run is not expected.)

- [ ] **Step 4: Commit**

```bash
git add scripts/download_models.py src/tk_vision_specialized/tk_vision_specialized/check_waving_inference.py scripts/tests/debug_waving_pipeline.py
git commit -m "chore(waving): stage pose .task via download_models; move helpers to PoseBackend"
```

---

### Task 8: Stack-wide verification

**Files:** none modified (results are recorded in Task 9).

- [ ] **Step 1: Full unit suites**

```bash
$VENV/bin/python -m pytest src/tk_vision_specialized/test scripts/tests/test_weights_cache.py -q 2>&1 | tail -5
```
Expected: all pass or skip (pre-existing skips for live-VLM tests are fine); zero failures.

- [ ] **Step 2: T0 + T1**

```bash
bash scripts/tests/t0_static.sh 2>&1 | tail -30
bash scripts/tests/t1_startup.sh 2>&1 | tail -30
grep -i "pose delegate" scripts/tests/logs/*waving* 2>/dev/null | tail -2
```
Expected: all rows pass (T1's waving row must pass now); the log shows `pose delegate: gpu`.

- [ ] **Step 3: Live T2 in mediapipe mode (only if cameras are up)**

```bash
ros2 topic hz /camera/color/image_raw --window 30 2>&1 | head -3   # ~30 Hz expected
bash scripts/tests/t2_live.sh 2>&1 | tail -20
# then one explicit fast-path goal with someone waving in view:
ros2 run tk_vision_specialized waving_person_server --ros-args -p waving_detector:=mediapipe -p show_window:=false &
sleep 15; ros2 run tk_vision_specialized waving_client; kill %1
```
Expected: a waver detected within ~300 ms of the goal; if nobody is available to wave, an empty-scene `status=1, waving_persons=[]` is the accepted result. Also run `ros2 run tk_vision_specialized check_waving_inference --ros-args -p max_runs:=5` and read `mediapipe_ms` in the JSON: ≤ 20 ms per person on GPU.

- [ ] **Step 4: Record** every command's outcome verbatim (pass/fail/skip + reason) for Task 9. Nothing to commit.

---

### Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md:255` (waving entry), `src/tk_vision_specialized/README.md:35,90,94`, `DEV_NOTES.md` (new dated entry at the top of the log section)

- [ ] **Step 1: CLAUDE.md** — in the `waving_person_server` bullet append after the `vlm_dedup_iou` sentence:

> Pose backend (`hybrid`/`mediapipe` modes): `pose_model_path` (default `'pose_landmarker_full.task'`, resolved via `weights_cache.find_cached` — **no auto-download**, stage it with `scripts/download_models.py`) and `pose_delegate` (`'gpu'` default, `'cpu'` to force). MediaPipe **1.0.1 Tasks API** (`_pose_backend.py`); the legacy `mp.solutions` API no longer exists upstream. GPU delegate: ~8 ms/person, ~60 MiB VRAM, 3–6 s one-time init, falls back to CPU (~60 ms/person) with a WARN. Parity with the 0.10.9 behaviour is enforced by `test/test_pose_parity.py` against `test/fixtures/pose_parity/`.

and change the tradeoff sentence's `~100–300 ms (MediaPipe)` to `~10–50 ms (MediaPipe on GPU; ~100–300 ms on CPU)`.

- [ ] **Step 2: README.md** — line 35: `For each person ROI, MediaPipe Tasks PoseLandmarker keypoints (`_pose_backend.py`).`; line 90: `mediapipe>=1.0`; add a `- **2026-08-22**` changelog bullet summarising the port, the two params, and the parity test.

- [ ] **Step 3: DEV_NOTES.md** — add an entry `## 2026-08-22 — MediaPipe 0.10.9 → 1.0.1 Tasks-API port (waving)` containing: why (Solutions removed upstream from 0.10.30), the benchmark table from the spec (90 / 60 / 7.8 ms; 60 MiB VRAM), the fixture/parity gate description, the verbatim verification results from Task 8 (including whether `test_gpu_parity` ran or skipped and whether live T2 was possible), the `pip freeze` diff line, and the explicit non-goals (rgb8/bgr8 normalizer still open, default mode unchanged, protobuf pin kept, jax orphans).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md src/tk_vision_specialized/README.md DEV_NOTES.md
git commit -m "docs: mediapipe Tasks-API port — params, benchmarks, parity gate"
```

---

## Self-review against the spec

- §1 dependency change → Task 5 (freeze diff, pip check, pins, lock, T0). ✔
- §2 adapter (`PoseLandmarkIdx`, `Landmark`, `POSE_CONNECTIONS`, `PoseBackend` GPU→CPU with warm-up probe, `process`→`None`, `close`, `draw_pose`) → Task 3. ✔
- §2 `find_cached` without download → Task 1; `RuntimeError` naming `download_models.py` → Task 6 step 2. ✔
- §3 node params, logging of delegate, drawing sites, `is_waving` unchanged, `close()` in destroy → Task 6; helper scripts → Task 7; `download_models.py` temp-then-rename, `--skip-pose` alias → Task 7. ✔
- §4 fixture recorded before upgrade (Task 2), hard CPU verdict+landmark parity, soft-skip GPU with hard verdicts, fallback test, drawing smoke (Task 3 `test_draw_pose_*`, Task 4). ✔ Stack-wide checks → Tasks 5 & 8. ✔
- §5 rollout order: Tasks 2 → 3/4 → 5 → 6/7 → 8 matches. ✔
- §6 docs → Task 9. ✔
- Acceptance latency `≤ 20 ms` → Task 8 step 3. ✔
- Type consistency: `PoseBackend(model_path, delegate, min_detection_confidence)`, `process()->list[Landmark]|None`, `active_delegate`, `fallback_reason`, `_create_landmarker(model_path, delegate, min_conf)`, `_to_mp_image(rgb)` are used identically in Tasks 3, 4, 6, 7. ✔
