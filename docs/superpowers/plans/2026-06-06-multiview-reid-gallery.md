# Multi-View ReID Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise person-tracker reacquisition recall by scoring reappearance candidates against a curated bank of *diverse* operator views (max-over-views) instead of a single averaged feature — with no threshold changes and precision held flat.

**Architecture:** A new pure `ReIDGallery` (numpy-only, in `core/` to avoid the reid→core import cycle) holds up to K diverse, quality-gated, L2-normalized operator feature views. `TargetAppearance` owns one gallery and exposes `deep_score(candidate_reid)` that returns the gallery's max-over-views cosine when enabled+populated, else the legacy `max(avg, anchor)` cosine. The two deep-ReID call sites (`ReIDMatcher.compute_similarity` and `reid_search._score_candidates`) call `deep_score`; the existing gates (`reid_threshold`, distinctiveness, single-candidate guard, `MIN_REID_SIMILARITY_RAW`) are unchanged. The gallery is populated at the existing hygiene-gated appearance-update site.

**Tech Stack:** Python 3.10, numpy, ROS2 Humble (`vision_track`), pytest. Spec: `docs/superpowers/specs/2026-06-06-multiview-reid-gallery-design.md`.

**Conventions:**
- Repo root for paths below: `src/vision_track/` (the package dir; contains the `vision_track/` python package and `test/`).
- Run tests with the shared venv:
  `VENV=/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python`
  from `src/vision_track/`: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/<file> -v`
- New `.py` files MUST carry the same copyright header + module/class/function docstrings as existing `vision_track/core/*.py` (the `test/test_copyright.py`, `test/test_pep257.py`, `test/test_flake8.py` suites enforce this — open `vision_track/core/centroid.py` as the template).

---

### Task 1: `ReIDGallery` pure class

**Files:**
- Create: `src/vision_track/vision_track/core/reid_gallery.py`
- Test: `src/vision_track/test/test_reid_gallery.py`

- [ ] **Step 1: Write the failing tests**

```python
# src/vision_track/test/test_reid_gallery.py
"""Unit tests for the pure ReIDGallery (no torch / ROS)."""
import numpy as np

from vision_track.core.reid_gallery import ReIDGallery


def _vec(*vals, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        v[i] = x
    return v


def test_first_add_becomes_anchor_and_scores_one():
    g = ReIDGallery(size=6, novelty_max=0.85)
    assert g.maybe_add(_vec(1.0)) is True
    assert len(g) == 1
    # identical (normalized) view scores ~1.0
    assert g.score(_vec(2.0)) > 0.99  # same direction after L2 norm


def test_novelty_gate_rejects_near_duplicate():
    g = ReIDGallery(size=6, novelty_max=0.85)
    g.maybe_add(_vec(1.0, 0.0))
    # cosine ~0.997 to the existing view -> above novelty_max -> rejected
    assert g.maybe_add(_vec(1.0, 0.05)) is False
    assert len(g) == 1
    # a genuinely different direction is admitted
    assert g.maybe_add(_vec(0.0, 1.0)) is True
    assert len(g) == 2


def test_bounded_size_and_anchor_pinned():
    g = ReIDGallery(size=3, novelty_max=0.99)
    dirs = [_vec(1, 0, 0), _vec(0, 1, 0), _vec(0, 0, 1), _vec(1, 1, 0), _vec(1, 0, 1)]
    for d in dirs:
        g.maybe_add(d)
    assert len(g) == 3                      # capped
    # anchor (first ever view, direction e0) must still be retrievable at ~1.0
    assert g.score(_vec(5, 0, 0)) > 0.99


def test_score_is_max_over_views():
    g = ReIDGallery(size=6, novelty_max=0.99)
    g.maybe_add(_vec(1, 0, 0))
    g.maybe_add(_vec(0, 1, 0))
    # candidate aligned with the second view -> max cosine ~1.0, not the mean
    assert g.score(_vec(0, 3, 0)) > 0.99


def test_top2_mean_mode_is_stricter_than_max():
    g = ReIDGallery(size=6, novelty_max=0.99, score_mode="top2_mean")
    g.maybe_add(_vec(1, 0, 0))
    g.maybe_add(_vec(0, 1, 0))
    # candidate matches one view perfectly, the other at 0 -> top2 mean ~0.5
    s = g.score(_vec(0, 3, 0))
    assert 0.4 < s < 0.6


def test_empty_and_disabled_return_none():
    g = ReIDGallery(size=6)
    assert g.score(_vec(1.0)) is None       # empty
    g.maybe_add(_vec(1.0))
    g.enabled = False
    assert g.score(_vec(1.0)) is None       # disabled


def test_dim_mismatch_is_skipped_not_crash():
    g = ReIDGallery(size=6, novelty_max=0.99)
    g.maybe_add(np.array([1.0, 0.0], dtype=np.float32))
    # different-dim candidate -> no matching views -> None, no exception
    assert g.score(np.array([1.0, 0.0, 0.0], dtype=np.float32)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_reid_gallery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision_track.core.reid_gallery'`.

- [ ] **Step 3: Implement `ReIDGallery`**

```python
# src/vision_track/vision_track/core/reid_gallery.py
# <copyright header matching core/centroid.py>
"""Curated multi-view ReID gallery for precision-safe reacquisition.

Holds a bounded bank of diverse, high-quality operator feature views and scores
a candidate as the max cosine over them (or a stricter fallback mode). The
caller is responsible for passing only quality-gated features (the appearance
update path already applies ``crop_quality_ok`` before admission); the gallery
adds a novelty gate so the bank spans genuinely different views. Pure numpy — no
torch / ROS — so it lives in ``core`` and is unit-testable in isolation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _l2norm(v: np.ndarray) -> np.ndarray:
    """Return the L2-normalized vector (unchanged if near-zero norm)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of two already-L2-normalized vectors."""
    return float(np.dot(a, b))


class ReIDGallery:
    """Bounded bank of diverse operator feature views for reacquisition."""

    def __init__(self, enabled: bool = True, size: int = 6,
                 novelty_max: float = 0.85, score_mode: str = "max") -> None:
        """Store policy; the bank starts empty (index 0 is the pinned anchor)."""
        self.enabled = bool(enabled)
        self.size = max(1, int(size))
        self.novelty_max = float(novelty_max)
        self.score_mode = str(score_mode)
        self._views: List[np.ndarray] = []

    def configure(self, *, enabled: bool, size: int, novelty_max: float,
                  score_mode: str) -> None:
        """Apply runtime config (from ROS params) without dropping views."""
        self.enabled = bool(enabled)
        self.size = max(1, int(size))
        self.novelty_max = float(novelty_max)
        self.score_mode = str(score_mode)

    def __len__(self) -> int:
        """Number of stored views."""
        return len(self._views)

    def clear(self) -> None:
        """Drop all views (e.g. on tracker reset)."""
        self._views = []

    def _matching(self, dim: int) -> List[np.ndarray]:
        """Views whose dimension matches ``dim`` (guards backbone swaps)."""
        return [v for v in self._views if v.shape[0] == dim]

    def maybe_add(self, feature: Optional[np.ndarray]) -> bool:
        """Admit an (already quality-gated) feature if novel. Return admitted."""
        if feature is None or feature.ndim != 1:
            return False
        f = _l2norm(feature.astype(np.float32))
        if not self._views:
            self._views.append(f)            # anchor, pinned at index 0
            return True
        same = self._matching(f.shape[0])
        if same and max(_cos(f, v) for v in same) >= self.novelty_max:
            return False                     # too similar to an existing view
        self._views.append(f)
        if len(self._views) > self.size:
            self._evict_most_redundant()
        return True

    def _evict_most_redundant(self) -> None:
        """Drop the most-redundant non-anchor view (keep diversity)."""
        if len(self._views) <= 1:
            return
        non_anchor = list(range(1, len(self._views)))

        def redundancy(idx: int) -> float:
            vi = self._views[idx]
            others = [v for j, v in enumerate(self._views)
                      if j != idx and v.shape[0] == vi.shape[0]]
            return float(np.mean([_cos(vi, o) for o in others])) if others else -1.0

        drop = max(non_anchor, key=redundancy)
        self._views.pop(drop)

    def score(self, feature: Optional[np.ndarray]) -> Optional[float]:
        """Max cosine over matching views (or top2_mean). None if unusable."""
        if not self.enabled or feature is None or feature.ndim != 1:
            return None
        f = _l2norm(feature.astype(np.float32))
        sims = sorted((_cos(f, v) for v in self._matching(f.shape[0])), reverse=True)
        if not sims:
            return None
        if self.score_mode == "top2_mean":
            return float(np.mean(sims[:2]))
        return float(sims[0])
```

(Copy the exact copyright header block from `vision_track/core/centroid.py` to satisfy `test/test_copyright.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_reid_gallery.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vision_track/vision_track/core/reid_gallery.py src/vision_track/test/test_reid_gallery.py
git commit -m "feat(vision_track): pure ReIDGallery (multi-view reacquisition memory)"
```

---

### Task 2: `TargetAppearance` gallery field + `deep_score`

**Files:**
- Modify: `src/vision_track/vision_track/core/tracking_types.py` (class `TargetAppearance`, after `get_average_feature` ~line 107)
- Test: `src/vision_track/test/test_target_deep_score.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# src/vision_track/test/test_target_deep_score.py
"""deep_score: gallery max-over-views when enabled, legacy max(avg,anchor) else."""
import numpy as np

from vision_track.core.tracking_types import TargetAppearance


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


def test_legacy_fallback_when_gallery_empty():
    t = TargetAppearance(class_id=0, class_name="person")
    t.feature_history.append(_v(1, 0))         # avg == this
    t.configure_gallery(enabled=True, size=6, novelty_max=0.85, score_mode="max")
    # gallery empty -> fall back to cosine(avg, cand) ~1.0
    assert t.deep_score(_v(2, 0)) > 0.99


def test_gallery_used_when_populated():
    t = TargetAppearance(class_id=0, class_name="person")
    t.configure_gallery(enabled=True, size=6, novelty_max=0.99, score_mode="max")
    t.gallery.maybe_add(_v(1, 0))
    t.gallery.maybe_add(_v(0, 1))
    # candidate matches the SECOND view -> gallery max ~1.0 even though avg differs
    assert t.deep_score(_v(0, 5)) > 0.99


def test_disabled_gallery_is_legacy():
    t = TargetAppearance(class_id=0, class_name="person")
    t.feature_history.append(_v(1, 0))
    t.configure_gallery(enabled=False, size=6, novelty_max=0.85, score_mode="max")
    t.gallery.maybe_add(_v(0, 1))              # ignored for scoring when disabled
    # legacy path: cosine(avg=[1,0], cand=[0,1]) ~0.0
    assert abs(t.deep_score(_v(0, 1))) < 0.05


def test_none_when_no_feature():
    t = TargetAppearance(class_id=0, class_name="person")
    assert t.deep_score(_v(1, 0)) is None      # no history, no gallery, no anchor
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_target_deep_score.py -v`
Expected: FAIL — `AttributeError: 'TargetAppearance' object has no attribute 'configure_gallery'`.

- [ ] **Step 3: Implement the field, helper, and `deep_score`**

In `core/tracking_types.py`, add the import at the top:

```python
from .reid_gallery import ReIDGallery
```

Add a module-level cosine helper near the top of the file (after imports):

```python
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (0.0 on near-zero norm)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
```

Add the gallery field to the `TargetAppearance` dataclass (alongside the other
`field(default_factory=...)` declarations, e.g. after `feature_history`):

```python
    gallery: ReIDGallery = field(default_factory=ReIDGallery)
```

Add these methods to `TargetAppearance` (after `get_average_feature`):

```python
    def configure_gallery(self, *, enabled: bool, size: int,
                          novelty_max: float, score_mode: str) -> None:
        """Apply gallery config (from ROS params) to this target's gallery."""
        self.gallery.configure(enabled=enabled, size=size,
                               novelty_max=novelty_max, score_mode=score_mode)

    def deep_score(self, candidate_reid):
        """Deep-ReID similarity of a candidate to this target's appearance.

        Uses the multi-view gallery (max over diverse views) when enabled and
        populated, never doing worse than the pinned anchor; otherwise falls
        back to the legacy max(average, anchor) cosine. Returns a raw cosine in
        [-1, 1], or None when no usable target feature exists.
        """
        if candidate_reid is None:
            return None
        dim = candidate_reid.shape[0]
        if self.gallery.enabled and len(self.gallery) > 0:
            g = self.gallery.score(candidate_reid)
            if g is not None:
                if (self.anchor_feature is not None
                        and self.anchor_feature.shape[0] == dim):
                    return max(g, _cosine(self.anchor_feature, candidate_reid))
                return g
        best = None
        avg = self.get_average_feature()
        if avg is not None and avg.shape[0] == dim:
            best = _cosine(avg, candidate_reid)
        if self.anchor_feature is not None and self.anchor_feature.shape[0] == dim:
            a = _cosine(self.anchor_feature, candidate_reid)
            best = a if best is None else max(best, a)
        return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_target_deep_score.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vision_track/vision_track/core/tracking_types.py src/vision_track/test/test_target_deep_score.py
git commit -m "feat(vision_track): TargetAppearance.deep_score over multi-view gallery"
```

---

### Task 3: Route the two deep-ReID call sites through `deep_score`

**Files:**
- Modify: `src/vision_track/vision_track/reid/reid.py` (`compute_similarity`, the `'reid' in candidate_features` block ~lines 700–92)
- Modify: `src/vision_track/vision_track/reid/reid_search.py` (`_score_candidates` ~lines 44, and the `raw_cosine` block; remove the now-unused `target_reid` param)
- Test: `src/vision_track/test/test_deep_score_wiring.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# src/vision_track/test/test_deep_score_wiring.py
"""compute_similarity's deep term must come from TargetAppearance.deep_score."""
import numpy as np

from vision_track.core.tracking_types import TargetAppearance
from vision_track.reid.reid import ReIDMatcher


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


def test_compute_similarity_uses_gallery_max_view():
    t = TargetAppearance(class_id=0, class_name="person")
    t.last_seen_time = 0.0
    t.configure_gallery(enabled=True, size=6, novelty_max=0.99, score_mode="max")
    # Two distinct stored views; candidate matches the SECOND.
    t.gallery.maybe_add(_v(1, 0))
    t.gallery.maybe_add(_v(0, 1))
    t.anchor_feature = _v(1, 0)
    cand = {"reid": _v(0, 5)}                       # aligned with view 2
    sim = ReIDMatcher.compute_similarity(
        t, cand, candidate_bbox=(0, 0, 10, 20), current_time=1.0, is_person=True
    )
    # Deep term is high because the gallery max view matches -> not hard-rejected.
    assert sim > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_deep_score_wiring.py -v`
Expected: FAIL — with an empty `feature_history` and no `get_average_feature`, the legacy deep term is None so `reid_sim_raw` stays None and the candidate is scored without the gallery (sim 0.0 or hard-reject), proving the gallery isn't wired yet.

- [ ] **Step 3a: Edit `reid/reid.py` `compute_similarity`**

Replace the existing deep-term block (currently, ~lines 700–92):

```python
        if 'reid' in candidate_features:
            target_reid = target.get_average_feature()
            if target_reid is not None:
                candidate_reid = candidate_features['reid']
                if target_reid.shape[0] == candidate_reid.shape[0]:
                    reid_sim_raw = cls._cosine_similarity(target_reid, candidate_reid)
                    if target.anchor_feature is not None and target.anchor_feature.shape[0] == candidate_reid.shape[0]:
                        reid_anchor_sim = cls._cosine_similarity(target.anchor_feature, candidate_reid)
                        reid_sim_raw = max(reid_sim_raw, reid_anchor_sim)
                    if reid_sim_raw < cls.MIN_REID_SIMILARITY_RAW:
                        logger.info(f"HARD REJECT: ReID raw similarity {reid_sim_raw:.3f} < {cls.MIN_REID_SIMILARITY_RAW}")
                        return 0.0
                else:
                    logger.debug(f"ReID dimension mismatch: {target_reid.shape[0]} vs {candidate_reid.shape[0]}")
```

with (deep term now flows through `deep_score`, which itself unions gallery +
anchor; the hard-reject floor is preserved):

```python
        if 'reid' in candidate_features:
            candidate_reid = candidate_features['reid']
            reid_sim_raw = target.deep_score(candidate_reid)
            if reid_sim_raw is not None:
                if reid_sim_raw < cls.MIN_REID_SIMILARITY_RAW:
                    logger.info(f"HARD REJECT: ReID raw similarity {reid_sim_raw:.3f} < {cls.MIN_REID_SIMILARITY_RAW}")
                    return 0.0
            else:
                logger.debug("ReID deep term unavailable (no matching target feature)")
```

(Leave `reid_anchor_sim` initialization and all downstream `reid_sim_raw`
normalization/weighting untouched — `deep_score` already folds in the anchor.)

- [ ] **Step 3b: Edit `reid/reid_search.py` `_score_candidates`**

At the call site (~line 44–45), stop fetching `target_reid` and stop passing it:

```python
    candidate_scores = _score_candidates(tracker, frame, candidates, current_time, is_person)
```

In `_score_candidates`, drop the `target_reid` parameter from the signature and
replace the `raw_cosine` computation:

```python
        raw_cosine = 0.0
        if is_person and "reid" in features:
            ds = tracker.target_appearance.deep_score(features["reid"])
            if ds is not None:
                raw_cosine = ds
                logger.debug(f"ID {result.track_id}: gallery deep score={raw_cosine:.3f}")
```

(Remove the old `if is_person and "reid" in features and target_reid is not None:`
branch that called `ReIDMatcher._cosine_similarity(target_reid, ...)`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_deep_score_wiring.py test/test_reid_ratio_gate.py test/test_distinctiveness_margin.py test/test_fusion_weights.py -v`
Expected: PASS — new wiring test passes and the existing ReID gate/fusion tests still pass (no regression).

- [ ] **Step 5: Commit**

```bash
git add src/vision_track/vision_track/reid/reid.py src/vision_track/vision_track/reid/reid_search.py src/vision_track/test/test_deep_score_wiring.py
git commit -m "feat(vision_track): score reacquisition deep term via gallery deep_score"
```

---

### Task 4: Populate the gallery during confident tracking

**Files:**
- Modify: `src/vision_track/vision_track/reid/appearance_manager.py` (`_update_feature_history`, after `feature_history.append`)
- Test: `src/vision_track/test/test_gallery_population.py` (create)

**Context:** `_update_feature_history` is only reached after the upstream
`crop_quality_ok` gate passes (see `appearance_manager.py` ~line 61, which
returns early on low-quality crops). So features arriving here are already
quality-gated; the gallery's novelty gate handles diversity.

- [ ] **Step 1: Write the failing test**

```python
# src/vision_track/test/test_gallery_population.py
"""_update_feature_history must also feed the multi-view gallery (novelty-gated)."""
import numpy as np

from vision_track.core.tracking_types import TargetAppearance
from vision_track.reid.appearance_manager import _update_feature_history


class _Tracker:
    def __init__(self):
        self.target_appearance = TargetAppearance(class_id=0, class_name="person")
        self.target_appearance.configure_gallery(
            enabled=True, size=6, novelty_max=0.85, score_mode="max")
        self.original_track_id = None


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


def test_distinct_views_populate_gallery():
    trk = _Tracker()
    _update_feature_history(trk, {"reid": _v(1, 0)}, 0.9, 1.0, True)
    _update_feature_history(trk, {"reid": _v(0, 1)}, 0.9, 1.0, True)   # distinct
    assert len(trk.target_appearance.gallery) == 2


def test_near_duplicate_does_not_grow_gallery():
    trk = _Tracker()
    _update_feature_history(trk, {"reid": _v(1, 0.0)}, 0.9, 1.0, True)
    _update_feature_history(trk, {"reid": _v(1, 0.02)}, 0.9, 1.0, True)  # ~dup
    assert len(trk.target_appearance.gallery) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_gallery_population.py -v`
Expected: FAIL — gallery length stays 0 (population not wired).

- [ ] **Step 3: Wire the population**

In `_update_feature_history` (`appearance_manager.py`), immediately after the
existing `tracker.target_appearance.feature_history.append(new_feature)` line,
add:

```python
    tracker.target_appearance.gallery.maybe_add(new_feature)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_gallery_population.py test/test_appearance_quality_gate.py -v`
Expected: PASS — population works and the existing quality-gate test is unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/vision_track/vision_track/reid/appearance_manager.py src/vision_track/test/test_gallery_population.py
git commit -m "feat(vision_track): populate multi-view gallery at hygiene-gated append"
```

---

### Task 5: Config + tracker wiring (kill-switch and params)

**Files:**
- Modify: `src/vision_track/config/default.yaml`
- Modify: `src/vision_track/vision_track/yolo_tracker.py` (`__init__` params; configure gallery wherever `target_appearance` is created)
- Modify: `src/vision_track/vision_track/person_track_node.py` (declare + pass the four ROS params)
- Test: `src/vision_track/test/test_gallery_config.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# src/vision_track/test/test_gallery_config.py
"""YOLOTracker must apply gallery config to its target_appearance."""
from vision_track.track_yolo import YOLOTracker


def test_tracker_configures_gallery_disabled(monkeypatch):
    # Build a tracker without loading YOLO/ReID weights: only check that the
    # gallery-config plumbing applies to a freshly created TargetAppearance.
    trk = YOLOTracker.__new__(YOLOTracker)        # bypass heavy __init__
    trk.reid_gallery_enabled = False
    trk.reid_gallery_size = 4
    trk.reid_gallery_novelty_max = 0.8
    trk.reid_gallery_score_mode = "max"
    from vision_track.core.tracking_types import TargetAppearance
    ta = TargetAppearance(class_id=0, class_name="person")
    YOLOTracker._configure_gallery(trk, ta)
    assert ta.gallery.enabled is False
    assert ta.gallery.size == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_gallery_config.py -v`
Expected: FAIL — `AttributeError: type object 'YOLOTracker' has no attribute '_configure_gallery'`.

- [ ] **Step 3a: Add params + helper to `yolo_tracker.py`**

Add to `YOLOTracker.__init__` signature (with the other reid params):

```python
        reid_gallery_enabled: bool = True,
        reid_gallery_size: int = 6,
        reid_gallery_novelty_max: float = 0.85,
        reid_gallery_score_mode: str = "max",
```

Store them in `__init__` body (next to `self.reid_fp16 = reid_fp16`):

```python
        self.reid_gallery_enabled = reid_gallery_enabled
        self.reid_gallery_size = reid_gallery_size
        self.reid_gallery_novelty_max = reid_gallery_novelty_max
        self.reid_gallery_score_mode = reid_gallery_score_mode
```

Add the helper method:

```python
    def _configure_gallery(self, appearance) -> None:
        """Apply this tracker's gallery ROS params to a TargetAppearance."""
        appearance.configure_gallery(
            enabled=self.reid_gallery_enabled,
            size=self.reid_gallery_size,
            novelty_max=self.reid_gallery_novelty_max,
            score_mode=self.reid_gallery_score_mode,
        )
```

Call `self._configure_gallery(self.target_appearance)` at **every** site that
assigns `self.target_appearance = TargetAppearance(...)`. There are two:
- in `initialize_tracking` (after the target appearance is first created), and
- the lazy create in `reid/appearance_manager.py`:
  `if tracker.target_appearance is None: tracker.target_appearance = TargetAppearance(...)` →
  add `tracker._configure_gallery(tracker.target_appearance)` on the next line.

(Grep to confirm both sites: `grep -rn "TargetAppearance(class_id" src/vision_track/vision_track`.)

- [ ] **Step 3b: Declare ROS params in `person_track_node.py`**

Where the other `reid_*` params are declared/read (search `reid_fp16`), add the
four `reid_gallery_*` params with the same defaults and pass them into the
`YOLOTracker(...)` constructor call.

- [ ] **Step 3c: Add to `config/default.yaml`**

Under the same block as `reid_backbone` / `reid_fp16`:

```yaml
    reid_gallery_enabled: true        # multi-view reacquisition gallery (kill-switch)
    reid_gallery_size: 6              # K diverse views kept
    reid_gallery_novelty_max: 0.85    # admit a view only if < this cosine to existing
    reid_gallery_score_mode: 'max'    # 'max' | 'top2_mean' (precision fallback)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/test_gallery_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vision_track/vision_track/yolo_tracker.py src/vision_track/vision_track/person_track_node.py src/vision_track/config/default.yaml src/vision_track/test/test_gallery_config.py
git commit -m "feat(vision_track): gallery ROS params + kill-switch wiring"
```

---

### Task 6: Full-suite regression + LaSOT validation (acceptance gate)

**Files:**
- Run-only (no new code unless a guardrail trips).

- [ ] **Step 1: Full vision_track unit suite green**

Run: `cd src/vision_track && PYTHONPATH=$(pwd) $VENV -m pytest test/ -q`
Expected: all PASS (existing suite + the new gallery tests; flake8/pep257/copyright included).

- [ ] **Step 2: LaSOT baseline (gallery OFF) for comparison**

The committed runner builds `YOLOTracker` with constructor defaults
(`reid_gallery_enabled=True`). To get the OFF baseline, run with the env-style
override by temporarily passing `reid_gallery_enabled=False` — add a `--no-gallery`
flag to `benchmarks/person_tracker/demo/run_lasot_person_benchmark.py` that sets
`reid_gallery_enabled=False` on the tracker it constructs (the runner builds the
tracker in `_make_tracker`; thread the flag through), then:

Run: `$VENV benchmarks/person_tracker/demo/run_lasot_person_benchmark.py --no-gallery --json /tmp/lasot_gallery_off.json`
Record the mean P / R / F.

- [ ] **Step 3: LaSOT with gallery ON**

Run: `$VENV benchmarks/person_tracker/demo/run_lasot_person_benchmark.py --json /tmp/lasot_gallery_on.json`
Record the mean P / R / F and per-sequence values.

- [ ] **Step 4: Apply the acceptance guardrail**

Compare ON vs OFF:
- **PASS** iff mean recall(ON) > recall(OFF) AND mean precision(ON) ≥ precision(OFF) − 0.01 AND no single sequence's precision drops by > 0.02.
- If precision regresses beyond the guardrail: set `reid_gallery_score_mode: 'top2_mean'` in `config/default.yaml` (and the runner default) and re-run Steps 3–4. If `top2_mean` still regresses precision, STOP and report — do not merge; revisit the design (the gallery may need a stricter admission or a per-candidate require-mean rule).

- [ ] **Step 5: Record results + commit**

Append the ON/OFF table to `benchmarks/person_tracker/ptbench/tpt_bench/DOWNLOAD.md` under a new "Multi-view gallery (Spec A) result" subsection.

```bash
git add benchmarks/person_tracker/ptbench/tpt_bench/DOWNLOAD.md benchmarks/person_tracker/demo/run_lasot_person_benchmark.py
git commit -m "test(vision_track): LaSOT gallery on/off validation + --no-gallery flag"
```

---

## Self-Review

**Spec coverage:**
- Spec §"Component 1 ReIDGallery" → Task 1. ✓
- max-over-views scoring + fallback modes → Task 1 (`score`) + Task 6 guardrail. ✓
- Novelty + quality-gated admission, bounded K, anchor pinned, eviction → Task 1. ✓
- Integration into reacquisition (compute_similarity + _score_candidates) → Task 3. ✓
- Population at the hygiene-gated append → Task 4. ✓
- Config (`reid_gallery_enabled/size/novelty_max/score_mode`) + kill-switch → Task 5. ✓
- Reset clears gallery → covered: `gallery` is a `TargetAppearance` field; `tracker.reset()` already nulls `target_appearance`, and a fresh one starts with an empty gallery (no extra code). ✓
- Precision-flat guardrail on LaSOT → Task 6. ✓
- Out-of-scope items (active re-ID, terminal-LOST, depth) → not touched. ✓

**Placeholder scan:** none — every code step has complete code; commands have expected output.

**Type consistency:** `ReIDGallery(enabled, size, novelty_max, score_mode)` + `configure(...)` + `maybe_add(feature)->bool` + `score(feature)->Optional[float]` + `__len__` used consistently in Tasks 1–5. `TargetAppearance.configure_gallery(...)` and `deep_score(candidate_reid)->Optional[float]` consistent across Tasks 2–5. `_configure_gallery(self, appearance)` consistent in Task 5.

**Note for implementer:** confirm the exact line numbers in `reid/reid.py` / `reid/reid_search.py` by grep before editing (line numbers above are indicative from 2026-06-06); match the surrounding code style and the copyright header convention.
