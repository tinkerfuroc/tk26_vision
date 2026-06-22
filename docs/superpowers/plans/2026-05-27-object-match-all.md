# `object_match_all` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new ROS 2 service `/object_match_all` that scans every entry in `items_map.yaml` concurrently via batched VLM calls and returns a `ObjectDetection.srv`-shaped response, with per-conflict VLM-judge resolution and batched MobileSAM segmentation.

**Architecture:** Standalone node in the `tk_vision_specialized` package, no subclassing of `YOLOSegmentationNode`. Composed of a ROS-aware `CameraDataSource` (camera sync + intrinsics + TF + VisionLogger), provider-agnostic `MatchClient` / `JudgeClient` adapters (Qwen + Gemini backends), pure-Python `MatchPipeline` orchestrator (batches → within-cat NMS → cross-cat clusters → concurrent judge → batched SAM → centroid + TF), and the existing `SamPredictor` + `ItemsMapLoader`.

**Tech Stack:** ROS 2 Humble, Python 3.10 venv at `src/tk26_vision/.venv-vision-main/`, `openai` SDK (DashScope + OpenRouter compatible mode), Ultralytics SAM (MobileSAM), `rclpy`, `cv_bridge`, `tf2_ros`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-27-object-match-all-design.md` (in the same repo). Read it before starting.

---

## Conventions and one-time setup

**Working directory (every command):** `/home/tinker/tk25_ws` (the colcon workspace root). Don't `cd` away.

**Activate the venv and source ROS before any test or build command:**

```bash
source /opt/ros/humble/setup.bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
```

After Task 1 (interfaces build), additionally:

```bash
source /home/tinker/tk25_ws/src/tk26_vision/install/setup.bash
```

**Build commands** use the tk26_vision wrapper which patches install-tree shebangs:

```bash
/home/tinker/tk25_ws/src/tk26_vision/scripts/build.sh --packages-select <pkg>
```

**Unit test commands** run from the venv-activated shell. The plan uses `pytest <file> -v` directly; ament-style discovery via `colcon test` is also wired up (the test files live under the package's `test/` directory and follow pytest collection).

**Commit convention** (matches `git log` in this repo):

```
type(scope): short subject

Optional body paragraphs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

`type` ∈ `feat | fix | docs | refactor | test | chore`. `scope` is the package name (`tk_vision_specialized`, `tinker_vision_msgs_26`, `tk26_vision`).

**Branch:** all work lands on the `dev` branch. Don't create feature branches.

---

## Task 1 — Add `ObjectMatchAll.srv` and rebuild interfaces

**Files:**
- Create: `src/tk26_vision/src/tinker_vision_msgs_26/srv/ObjectMatchAll.srv`
- Modify: `src/tk26_vision/src/tinker_vision_msgs_26/CMakeLists.txt:38`

- [ ] **Step 1: Create the srv file**

Create `src/tk26_vision/src/tinker_vision_msgs_26/srv/ObjectMatchAll.srv` with this exact content:

```
# Camera identifier ("realsense", "orbbec", or substrings).
string camera

# Empty list = scan every entry in items_map.yaml. Non-empty list = scan only
# these dataset keys; unknown keys are dropped from the scan. If every key in
# the filter is unknown, the response is status=1.
string[] category_filter

# TF frame to express centroids in. Empty string = raw camera frame.
string target_frame

# Sort modes (mirrors ObjectDetectionGeneralist.srv conventions):
#   sort_closest  - by sqrt(x^2+y^2+z^2) ascending, camera frame
#   sort_highest  - by camera-frame Z ascending
# Both false (default) = confidence descending.
bool sort_closest
bool sort_highest

# Payload toggles.
bool return_rgb_image
bool return_depth_image
bool return_segments

---

# Response field set is the union of ObjectDetection.srv and
# ObjectDetectionGeneralist.srv. person_id kept for ABI parity with
# ObjectDetection.srv (always 0). detection_source mirrors the generalist's
# tag field. Callers written against ObjectDetection.srv can be retargeted
# at this service by swapping the srv import; this response's field set is
# a superset of ObjectDetection.srv's.
std_msgs/Header header
int32 status                     # 0 = ok with >=1 object, 1 = empty / failure (see error_msg)
string error_msg
int32 person_id                  # always 0; kept for ABI parity with ObjectDetection.srv
Object[] objects                 # cls / conf / centroid populated
string detection_source          # e.g. "vlm_match_all"

sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image
sensor_msgs/Image[] segments
```

- [ ] **Step 2: Register the srv in CMakeLists.txt**

Open `src/tk26_vision/src/tinker_vision_msgs_26/CMakeLists.txt` and add `"srv/ObjectMatchAll.srv"` to the `rosidl_generate_interfaces` block. Insert alphabetically after `"srv/ObjectMatch.srv"` (line 38):

```cmake
  "srv/ObjectMatch.srv"
  "srv/ObjectMatchAll.srv"
  "srv/PlacingLocation.srv"
```

- [ ] **Step 3: Build the interfaces package**

```bash
source /opt/ros/humble/setup.bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
/home/tinker/tk25_ws/src/tk26_vision/scripts/build.sh --packages-select tinker_vision_msgs_26
```

Expected: `Summary: 1 package finished` with no errors. If colcon errors on stale symlinks: `rm -rf build/tinker_vision_msgs_26 install/tinker_vision_msgs_26` and rebuild.

- [ ] **Step 4: Verify the generated Python import**

```bash
source /home/tinker/tk25_ws/src/tk26_vision/install/setup.bash
python -c "from tinker_vision_msgs_26.srv import ObjectMatchAll; r = ObjectMatchAll.Request(); r.category_filter = ['milk']; print(r)"
```

Expected: a `Request` instance prints with `category_filter=['milk']` and the other fields at their defaults. If `ImportError`: the build didn't generate Python bindings — re-run Step 3 with `--cmake-clean-cache`.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tinker_vision_msgs_26/srv/ObjectMatchAll.srv src/tinker_vision_msgs_26/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(tinker_vision_msgs_26): add ObjectMatchAll.srv

Sibling to ObjectMatch.srv. Request adds category_filter[], sort_closest,
sort_highest, and payload toggles. Response mirrors ObjectDetection.srv so
callers of /object_detection_yolo can be retargeted at the new service by
changing one parameter.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — `nms.py`: IoU + within-category NMS

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/nms.py`
- Create: `src/tk26_vision/src/tk_vision_specialized/test/test_nms.py`

- [ ] **Step 1: Write failing tests for IoU and within-category NMS**

Create `src/tk26_vision/src/tk_vision_specialized/test/test_nms.py`:

```python
"""Unit tests for nms.py — pure-function NMS and clustering helpers."""

from __future__ import annotations

import pytest

from tk_vision_specialized.nms import (
    iou,
    suppress_within_category,
    MatchRow,
)


def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # box A = 10x10, box B shifted right by 5 -> intersection 5x10=50, union 150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_zero_area_box_returns_zero():
    assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


def test_within_category_keeps_one_per_overlapping_pair():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(1, 1, 11, 11), conf=0.7),  # IoU > 0.5 with first
    ]
    kept = suppress_within_category(rows, iou_thresh=0.5)
    assert len(kept) == 1
    assert kept[0].conf == 0.9


def test_within_category_keeps_disjoint_same_label():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(50, 50, 60, 60), conf=0.5),  # disjoint
    ]
    kept = suppress_within_category(rows, iou_thresh=0.5)
    assert len(kept) == 2


def test_within_category_does_not_suppress_across_labels():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='cola', bbox=(0, 0, 10, 10), conf=0.8),  # same box, different label
    ]
    kept = suppress_within_category(rows, iou_thresh=0.5)
    assert len(kept) == 2  # cross-label overlap is not this function's job


def test_within_category_idempotent():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(1, 1, 11, 11), conf=0.7),
        MatchRow(label='milk', bbox=(50, 50, 60, 60), conf=0.5),
    ]
    once = suppress_within_category(rows, iou_thresh=0.5)
    twice = suppress_within_category(once, iou_thresh=0.5)
    assert once == twice


def test_within_category_empty_input():
    assert suppress_within_category([], iou_thresh=0.5) == []


def test_within_category_suppresses_at_threshold_equality():
    # Pin the strict `<` semantics: IoU == iou_thresh -> suppress.
    # A=(0,0,10,10) area=100, B=(0,0,10,5) area=50, intersection=50
    # -> IoU = 50 / (100 + 50 - 50) = 0.5.
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(0, 0, 10, 5), conf=0.5),
    ]
    kept = suppress_within_category(rows, iou_thresh=0.5)
    assert len(kept) == 1
    assert kept[0].conf == 0.9
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source /opt/ros/humble/setup.bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
cd /home/tinker/tk25_ws/src/tk26_vision/src/tk_vision_specialized
pytest test/test_nms.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: `ImportError: cannot import name 'iou'` (and friends) — `nms.py` doesn't exist yet.

- [ ] **Step 3: Implement `MatchRow`, `iou`, `suppress_within_category`**

Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/nms.py`:

```python
"""Pure-function NMS, clustering, and judge-payload helpers for
object_match_all.

No ROS imports here on purpose: this module is unit-testable from a plain
pytest run without sourcing the workspace. The shapes defined here are
reused by `match_pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


Bbox = tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


@dataclass(frozen=True)
class MatchRow:
    label: str
    bbox: Bbox
    conf: float


def iou(a: Bbox, b: Bbox) -> float:
    """Standard intersection-over-union on xyxy boxes. 0.0 on zero-area inputs."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    if a_area == 0 or b_area == 0:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(a_area + b_area - inter)


def suppress_within_category(
    rows: Sequence[MatchRow],
    iou_thresh: float,
) -> list[MatchRow]:
    """Greedy NMS, applied within each label independently.

    Same-label boxes that overlap above `iou_thresh` collapse to the higher
    confidence one. Different-label overlaps are preserved (resolved
    elsewhere by the cross-category clusterer + judge)."""

    by_label: dict[str, list[MatchRow]] = {}
    for r in rows:
        by_label.setdefault(r.label, []).append(r)

    kept: list[MatchRow] = []
    for _label, group in by_label.items():
        group.sort(key=lambda r: r.conf, reverse=True)
        survivors: list[MatchRow] = []
        for cand in group:
            if all(iou(cand.bbox, s.bbox) < iou_thresh for s in survivors):
                survivors.append(cand)
        kept.extend(survivors)
    return kept
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/test_nms.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/nms.py src/tk_vision_specialized/test/test_nms.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): nms.py IoU and within-category NMS

Pure-function helpers for object_match_all. No ROS deps so they're
unit-testable from a plain pytest run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — `nms.py`: cross-category clustering + judge payload builder

**Files:**
- Modify: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/nms.py`
- Modify: `src/tk26_vision/src/tk_vision_specialized/test/test_nms.py`

- [ ] **Step 1: Add failing tests for clustering and judge-payload**

Two edits to `src/tk26_vision/src/tk_vision_specialized/test/test_nms.py`:

(a) Extend the existing `from tk_vision_specialized.nms import ...` block at the top of the file to include the new symbols, and add `import numpy as np` to the top imports (do **not** append imports below the test functions — flake8 will flag E402). The updated top section should look like:

```python
"""Unit tests for nms.py — pure-function NMS and clustering helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tk_vision_specialized.nms import (
    Cluster,
    cluster_for_judge,
    build_judge_payload,
    iou,
    suppress_within_category,
    MatchRow,
)
```

(`JudgePayload` is constructed indirectly via `build_judge_payload` and accessed via attribute lookup, so it's not imported by name — that avoids the F401 unused-import warning.)

(b) Append the new test functions to the **bottom** of the file (after the existing tests):

```python
def test_cluster_singletons_when_disjoint():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='cola', bbox=(50, 50, 60, 60), conf=0.8),
    ]
    clusters = cluster_for_judge(rows, iou_thresh=0.5)
    assert len(clusters) == 2
    assert all(c.is_conflict() is False for c in clusters)


def test_cluster_groups_overlapping_cross_label():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='cola', bbox=(1, 1, 11, 11), conf=0.85),  # IoU > 0.5
    ]
    clusters = cluster_for_judge(rows, iou_thresh=0.5)
    assert len(clusters) == 1
    assert clusters[0].is_conflict() is True
    assert {r.label for r in clusters[0].rows} == {'milk', 'cola'}


def test_cluster_same_label_overlap_not_conflict():
    # After within-cat NMS this shouldn't happen, but defensively:
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(1, 1, 11, 11), conf=0.7),
    ]
    clusters = cluster_for_judge(rows, iou_thresh=0.5)
    assert len(clusters) == 1
    assert clusters[0].is_conflict() is False  # only one distinct label


def test_cluster_transitive_overlap_collapses_into_one():
    # A overlaps B, B overlaps C, A may not overlap C — still one cluster.
    rows = [
        MatchRow(label='milk',   bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='cola',   bbox=(5, 0, 15, 10), conf=0.8),
        MatchRow(label='sprite', bbox=(10, 0, 20, 10), conf=0.7),
    ]
    clusters = cluster_for_judge(rows, iou_thresh=0.3)
    assert len(clusters) == 1
    assert clusters[0].is_conflict() is True


def test_build_judge_payload_crops_with_margin_clamped_to_bounds():
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rows = [
        MatchRow(label='milk', bbox=(10, 10, 30, 30), conf=0.9),
        MatchRow(label='cola', bbox=(20, 20, 40, 40), conf=0.85),
    ]
    cluster = Cluster(rows=rows)
    items = {
        'milk': 'data:image/jpeg;base64,FAKE_MILK',
        'cola': 'data:image/jpeg;base64,FAKE_COLA',
    }
    payload = build_judge_payload(cluster, items, scene, margin_px=20)
    # Union bbox is (10,10,40,40); +20 margin -> (-10,-10,60,60) clamped to (0,0,60,60)
    assert payload.crop.shape == (60, 60, 3)
    competing_labels = {label for label, _url in payload.competing}
    assert competing_labels == {'milk', 'cola'}


def test_build_judge_payload_collapses_duplicate_labels():
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    # Cluster has two 'milk' rows (somehow survived within-cat NMS at this
    # IoU threshold) plus one 'cola' — competing list collapses duplicates.
    rows = [
        MatchRow(label='milk', bbox=(10, 10, 30, 30), conf=0.9),
        MatchRow(label='milk', bbox=(12, 12, 32, 32), conf=0.85),
        MatchRow(label='cola', bbox=(20, 20, 40, 40), conf=0.8),
    ]
    cluster = Cluster(rows=rows)
    items = {'milk': 'A', 'cola': 'B'}
    payload = build_judge_payload(cluster, items, scene, margin_px=0)
    assert len(payload.competing) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/test_nms.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 6 new tests fail with `ImportError` for `Cluster`, `JudgePayload`, `cluster_for_judge`, `build_judge_payload`.

- [ ] **Step 3: Implement clustering and judge-payload builder**

First, add the two new imports at the **top** of `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/nms.py` next to the existing ones (Task 2 deferred them since they were unused at that point):

```python
from itertools import combinations
```

and:

```python
import numpy as np
```

The full import block at the top of the file should now read:

```python
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
```

Then append the new types and helpers to the **bottom** of `nms.py`:

```python
@dataclass(frozen=True)
class Cluster:
    rows: list[MatchRow]

    def distinct_labels(self) -> list[str]:
        seen: list[str] = []
        for r in self.rows:
            if r.label not in seen:
                seen.append(r.label)
        return seen

    def is_conflict(self) -> bool:
        return len(self.rows) >= 2 and len(self.distinct_labels()) >= 2


@dataclass(frozen=True)
class JudgePayload:
    cluster: Cluster
    crop: np.ndarray
    crop_origin: tuple[int, int]                # (x_min, y_min) in scene coords
    competing: list[tuple[str, str]]            # (label, ref_data_url), deduped


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> list[list[int]]:
        gmap: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            gmap.setdefault(self.find(i), []).append(i)
        return list(gmap.values())


def cluster_for_judge(
    rows: Sequence[MatchRow],
    iou_thresh: float,
) -> list[Cluster]:
    """Greedy connected-components over the IoU graph.

    Two rows share an edge iff their IoU >= `iou_thresh`. Connected
    components become clusters. Singletons and same-label-only clusters are
    not conflicts; multi-label clusters of size >= 2 are."""

    rows = list(rows)
    n = len(rows)
    if n == 0:
        return []

    uf = _UnionFind(n)
    for i, j in combinations(range(n), 2):
        if iou(rows[i].bbox, rows[j].bbox) >= iou_thresh:
            uf.union(i, j)

    return [Cluster(rows=[rows[k] for k in members]) for members in uf.groups()]


def build_judge_payload(
    cluster: Cluster,
    items: dict[str, str],            # label -> ref_data_url
    scene_bgr: np.ndarray,
    margin_px: int,
) -> JudgePayload:
    """Compute the union bbox of cluster members, expand by `margin_px`,
    clamp to scene bounds, and produce the cropped image + the competing
    label/ref pairs (deduped by label)."""

    h, w = scene_bgr.shape[:2]
    x1 = min(r.bbox[0] for r in cluster.rows)
    y1 = min(r.bbox[1] for r in cluster.rows)
    x2 = max(r.bbox[2] for r in cluster.rows)
    y2 = max(r.bbox[3] for r in cluster.rows)

    x1c = max(0, x1 - margin_px)
    y1c = max(0, y1 - margin_px)
    x2c = min(w, x2 + margin_px)
    y2c = min(h, y2 + margin_px)
    crop = scene_bgr[y1c:y2c, x1c:x2c].copy()

    seen: set[str] = set()
    competing: list[tuple[str, str]] = []
    for r in cluster.rows:
        if r.label in seen:
            continue
        if r.label in items:
            competing.append((r.label, items[r.label]))
            seen.add(r.label)
    return JudgePayload(cluster=cluster, crop=crop, crop_origin=(x1c, y1c),
                        competing=competing)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/test_nms.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 16 passed (10 from Task 2 + 6 new).

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/nms.py src/tk_vision_specialized/test/test_nms.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): nms.py cross-category clustering + judge payload

Union-find clustering over the IoU graph plus a payload builder that crops
the scene around the cluster's union bbox (with margin) and dedupes
competing labels.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — `_vlm_common.py`: shared decode utilities

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_vlm_common.py`
- Create: `src/tk26_vision/src/tk_vision_specialized/test/test_vlm_common.py`

- [ ] **Step 1: Write failing tests**

Create `src/tk26_vision/src/tk_vision_specialized/test/test_vlm_common.py`:

```python
"""Unit tests for _vlm_common.py."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from tk_vision_specialized._vlm_common import (
    strip_fences,
    encode_data_url,
)


def test_strip_fences_passthrough_when_no_fence():
    raw = '{"a": 1}'
    assert strip_fences(raw) == '{"a": 1}'


def test_strip_fences_removes_json_fence():
    raw = '```json\n{"a": 1}\n```'
    out = strip_fences(raw)
    assert out.strip() == '{"a": 1}'


def test_strip_fences_removes_bare_fence():
    raw = '```\n{"a": 1}\n```'
    out = strip_fences(raw)
    assert out.strip() == '{"a": 1}'


def test_encode_data_url_round_trips_jpeg_bgr():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 2] = 128  # red channel in BGR
    url = encode_data_url(img)
    assert url.startswith('data:image/jpeg;base64,')
    payload = url.split(',', 1)[1]
    decoded = base64.b64decode(payload)
    # JPEG SOI marker
    assert decoded[:2] == b'\xff\xd8'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/test_vlm_common.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `_vlm_common.py`**

Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_vlm_common.py`:

```python
"""Shared utilities for the VLM match + judge clients.

Pure functions only (no ROS, no network) so they're trivially testable. The
clients build on top of these for prompt encoding and response decoding."""

from __future__ import annotations

import base64
import re

import cv2
import numpy as np


_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.MULTILINE)


def strip_fences(text: str) -> str:
    """Drop ```json ... ``` or ``` ... ``` fences that some VLM revisions
    emit despite explicit "no markdown" instructions in the system prompt."""
    if '```' not in text:
        return text
    return _FENCE_RE.sub('', text).strip()


def encode_data_url(rgb_bgr: np.ndarray) -> str:
    """Encode a BGR image (HxWx3 uint8) as a base64 JPEG data URL suitable
    for the OpenAI-compatible chat completions API."""
    ok, buf = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('cv2.imencode failed encoding scene image')
    return (
        'data:image/jpeg;base64,'
        + base64.b64encode(buf.tobytes()).decode('utf-8')
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/test_vlm_common.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/_vlm_common.py src/tk_vision_specialized/test/test_vlm_common.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): _vlm_common helpers for match/judge clients

Shared fence-strip and base64-JPEG encoding so the match and judge clients
don't duplicate this logic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — `vlm_match_client.py`: protocol, decoder, and Qwen backend

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/vlm_match_client.py`
- Create: `src/tk26_vision/src/tk_vision_specialized/test/test_vlm_match_client.py`

- [ ] **Step 1: Write failing tests for the decoder + Qwen backend**

Create `src/tk26_vision/src/tk_vision_specialized/test/test_vlm_match_client.py`:

```python
"""Unit tests for vlm_match_client.py — decoder logic and provider adapters.

Tests do not hit the network. The OpenAI client is monkeypatched per-test."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tk_vision_specialized.vlm_match_client import (
    MatchRow,
    QwenMatchClient,
    decode_qwen_response,
    build_match_client,
)


def _canned_completion(content: str):
    """Return a SimpleNamespace shaped like openai's completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content)
        )]
    )


def test_decode_qwen_normalized_box_scales_to_pixels():
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [100, 200, 500, 800],   # 0..1000 normalized
             'confidence': 0.92},
        ]
    })
    rows = decode_qwen_response(body, scene_w=200, scene_h=400,
                                allowed_labels={'milk'})
    assert len(rows) == 1
    r = rows[0]
    assert r.label == 'milk'
    # x1 = 100 * 200/1000 = 20; y1 = 200 * 400/1000 = 80
    # x2 = 500 * 200/1000 = 100; y2 = 800 * 400/1000 = 320
    assert r.bbox == (20, 80, 100, 320)
    assert r.conf == pytest.approx(0.92)


def test_decode_qwen_drops_hallucinated_label():
    body = json.dumps({
        'detections': [
            {'label': 'banana', 'box_2d': [0, 0, 100, 100], 'confidence': 0.99},
            {'label': 'milk',   'box_2d': [100, 100, 500, 500], 'confidence': 0.5},
        ]
    })
    rows = decode_qwen_response(body, scene_w=1000, scene_h=1000,
                                allowed_labels={'milk', 'cola'})
    assert len(rows) == 1
    assert rows[0].label == 'milk'


def test_decode_qwen_clamps_degenerate_box():
    body = json.dumps({
        'detections': [
            {'label': 'milk', 'box_2d': [500, 500, 500, 500], 'confidence': 0.9},
        ]
    })
    rows = decode_qwen_response(body, scene_w=100, scene_h=100,
                                allowed_labels={'milk'})
    assert rows == []


def test_decode_qwen_clamps_out_of_bounds():
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [-100, -100, 2000, 2000],
             'confidence': 0.9},
        ]
    })
    rows = decode_qwen_response(body, scene_w=100, scene_h=100,
                                allowed_labels={'milk'})
    assert len(rows) == 1
    assert rows[0].bbox == (0, 0, 99, 99)


def test_decode_qwen_handles_fenced_response():
    body = '```json\n{"detections": []}\n```'
    rows = decode_qwen_response(body, scene_w=100, scene_h=100, allowed_labels={'milk'})
    assert rows == []


def test_decode_qwen_swaps_inverted_coords():
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [500, 800, 100, 200],   # x2 < x1, y2 < y1
             'confidence': 0.9},
        ]
    })
    rows = decode_qwen_response(body, scene_w=1000, scene_h=1000,
                                allowed_labels={'milk'})
    assert len(rows) == 1
    x1, y1, x2, y2 = rows[0].bbox
    assert x1 < x2 and y1 < y2


def test_decode_qwen_clamps_confidence_to_unit_range():
    body = json.dumps({
        'detections': [
            {'label': 'milk', 'box_2d': [0, 0, 100, 100], 'confidence': 1.5},
            {'label': 'milk', 'box_2d': [200, 200, 300, 300], 'confidence': -0.2},
        ]
    })
    rows = decode_qwen_response(body, scene_w=1000, scene_h=1000,
                                allowed_labels={'milk'})
    assert {r.conf for r in rows} == {1.0, 0.0}


def test_qwen_client_resolves_dashcope_typo_first(monkeypatch):
    """The workspace .env historically carries DASHCOPE_API_KEY (typo);
    that should resolve first for backward compatibility."""
    monkeypatch.setenv('DASHCOPE_API_KEY', 'typo-key')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'official-key')
    client = QwenMatchClient(model='qwen3-vl-plus')
    assert client._api_key == 'typo-key'


def test_qwen_client_falls_back_to_dashscope_key(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'official-key')
    client = QwenMatchClient(model='qwen3-vl-plus')
    assert client._api_key == 'official-key'


def test_qwen_client_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='DashScope API key'):
        QwenMatchClient(model='qwen3-vl-plus')


def test_qwen_match_batch_end_to_end(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'fake')
    scene = np.zeros((400, 200, 3), dtype=np.uint8)
    body = json.dumps({
        'detections': [
            {'label': 'milk', 'box_2d': [0, 0, 500, 500], 'confidence': 0.9},
        ]
    })

    class FakeOpenAI:
        def __init__(self, *a, **kw):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: _canned_completion(body)
                )
            )
        def with_options(self, **kw):
            return self
        def close(self):
            pass

    with patch('tk_vision_specialized.vlm_match_client.OpenAI', FakeOpenAI):
        client = QwenMatchClient(model='qwen3-vl-plus')
        rows = client.match_batch(
            scene_bgr=scene,
            refs=[('milk', 'data:image/jpeg;base64,XXX')],
            timeout_s=5.0,
            max_retries=1,
        )

    assert len(rows) == 1
    assert rows[0].label == 'milk'


def test_build_match_client_unknown_provider_raises():
    with pytest.raises(ValueError, match='Unknown provider'):
        build_match_client('llama')
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/test_vlm_match_client.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the match client (Qwen path)**

Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/vlm_match_client.py`:

```python
"""Provider-agnostic match client for object_match_all.

The match client takes a scene BGR image plus a list of (label,
ref_data_url) pairs and asks the VLM to ground every reference in the
scene, returning a list of MatchRow. Two backends ship in this module:

- QwenMatchClient: DashScope Qwen3-VL, normalized 0..1000 coords
- GeminiMatchClient: OpenRouter Gemini, absolute pixel coords

`build_match_client(provider, **opts)` is the factory the node uses."""

from __future__ import annotations

import json
import os
from typing import Protocol, Sequence

import numpy as np
from dotenv import load_dotenv

from ._vlm_common import strip_fences, encode_data_url
from .nms import MatchRow, Bbox


_QWEN_DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
_QWEN_KEY_NAMES = ('DASHCOPE_API_KEY', 'DASHSCOPE_API_KEY')
_QWEN_DEFAULT_MODEL = 'qwen3-vl-plus'


class MatchClient(Protocol):
    def match_batch(
        self,
        scene_bgr: np.ndarray,
        refs: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> list[MatchRow]: ...


def _decode_bbox_normalized(
    box_2d, scene_w: int, scene_h: int,
) -> Bbox | None:
    """Decode a [x1, y1, x2, y2] 0..1000-normalized box to scene pixel xyxy."""
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(box_2d[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px1 = int(round(x1 * scene_w / 1000.0))
    py1 = int(round(y1 * scene_h / 1000.0))
    px2 = int(round(x2 * scene_w / 1000.0))
    py2 = int(round(y2 * scene_h / 1000.0))
    px1 = max(0, min(px1, scene_w - 1))
    px2 = max(0, min(px2, scene_w - 1))
    py1 = max(0, min(py1, scene_h - 1))
    py2 = max(0, min(py2, scene_h - 1))
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def decode_qwen_response(
    body: str, *, scene_w: int, scene_h: int, allowed_labels: set[str],
) -> list[MatchRow]:
    """Parse a Qwen3-VL match response body and return MatchRows.

    Drops rows whose `label` is not in `allowed_labels` (defensive against
    hallucinated labels). Clamps boxes to image bounds, drops degenerate
    ones, clamps confidence to [0, 1]."""
    try:
        parsed = json.loads(strip_fences(body))
    except (json.JSONDecodeError, ValueError):
        return []

    detections = parsed.get('detections') if isinstance(parsed, dict) else None
    if not isinstance(detections, list):
        return []

    rows: list[MatchRow] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = det.get('label')
        if not isinstance(label, str) or label not in allowed_labels:
            continue
        bbox = _decode_bbox_normalized(det.get('box_2d'), scene_w, scene_h)
        if bbox is None:
            continue
        try:
            conf = float(det.get('confidence', 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(conf, 1.0))
        rows.append(MatchRow(label=label, bbox=bbox, conf=conf))
    return rows


def _qwen_system_prompt(labels: list[str]) -> str:
    label_list = ', '.join(f'"{l}"' for l in labels)
    return (
        "You are a visual-grounding assistant for a service robot. The user "
        f"provides one SCENE image followed by {len(labels)} REFERENCE images, "
        f"each captioned with a label from this set: [{label_list}]. Find "
        "every visible instance of any reference item in the scene and return "
        "bounding boxes. Coordinates 'box_2d' are [x1, y1, x2, y2] normalized "
        "to 0-1000 over the SCENE image dimensions, where (0,0) is the top-left "
        "and (1000,1000) is the bottom-right. The 'label' field must be exactly "
        f"one of [{label_list}]. Confidence is a subjective match score in "
        "[0.0, 1.0]. If no reference item is visible, return detections=[]. "
        "Output JSON only, with no commentary or markdown fences."
    )


class QwenMatchClient:
    """Qwen3-VL match client (DashScope OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model: str = '',
        base_url: str = '',
    ):
        load_dotenv()
        self._api_key: str | None = None
        for name in _QWEN_KEY_NAMES:
            val = os.environ.get(name)
            if val:
                self._api_key = val
                break
        if not self._api_key:
            raise RuntimeError(
                f'DashScope API key not found in env (looked for {_QWEN_KEY_NAMES})'
            )
        self._model = model or _QWEN_DEFAULT_MODEL
        self._base_url = base_url or _QWEN_DEFAULT_BASE_URL

    def match_batch(
        self,
        scene_bgr: np.ndarray,
        refs: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> list[MatchRow]:
        if not refs:
            return []

        h, w = scene_bgr.shape[:2]
        labels = [label for label, _url in refs]
        allowed_labels = set(labels)

        scene_url = encode_data_url(scene_bgr)
        content: list[dict] = [{'type': 'image_url',
                                'image_url': {'url': scene_url}}]
        for label, url in refs:
            content.append({'type': 'image_url',
                            'image_url': {'url': url}})
        content.append({
            'type': 'text',
            'text': (
                'Image 1 is the scene. The remaining images are reference '
                'photos, in order: '
                + ', '.join(
                    f'image {i+2} = "{lbl}"'
                    for i, lbl in enumerate(labels)
                )
                + '. Return all visible instances grouped by label.'
            ),
        })

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        try:
            for attempt in range(max(1, max_retries)):
                try:
                    completion = client.with_options(
                        timeout=timeout_s,
                    ).chat.completions.create(
                        model=self._model,
                        messages=[
                            {'role': 'system',
                             'content': _qwen_system_prompt(labels)},
                            {'role': 'user', 'content': content},
                        ],
                        response_format={'type': 'json_object'},
                    )
                    raw = completion.choices[0].message.content or ''
                    rows = decode_qwen_response(
                        raw, scene_w=w, scene_h=h, allowed_labels=allowed_labels,
                    )
                    return rows
                except Exception as exc:    # noqa: BLE001
                    if logger is not None:
                        logger.warning(
                            f'Qwen match attempt {attempt+1}/{max_retries} '
                            f'failed: {exc}'
                        )
            return []
        finally:
            try:
                client.close()
            except Exception:    # noqa: BLE001
                pass


def build_match_client(provider: str, **opts) -> MatchClient:
    if provider == 'qwen':
        return QwenMatchClient(**opts)
    if provider == 'gemini':
        # Implemented in Task 6.
        from .vlm_match_client_gemini import GeminiMatchClient    # noqa: F401
        return GeminiMatchClient(**opts)
    raise ValueError(f'Unknown provider: {provider!r}')


try:
    from openai import OpenAI
except ImportError:    # pragma: no cover
    OpenAI = None    # type: ignore[assignment]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/test_vlm_match_client.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 12 passed (the Gemini-provider test in `test_build_match_client_unknown_provider_raises` only exercises the `Unknown provider` path, which doesn't need the Gemini module).

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/vlm_match_client.py src/tk_vision_specialized/test/test_vlm_match_client.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): vlm_match_client.py with Qwen3-VL backend

Defines the MatchClient protocol + Qwen3-VL adapter. Decoder normalizes
0..1000 coords to scene pixels, drops hallucinated labels, clamps degenerate
boxes. Gemini backend lands in the next task; build_match_client('gemini')
defers import to keep this commit self-contained.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — `vlm_match_client.py`: Gemini backend

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/vlm_match_client_gemini.py`
- Modify: `src/tk26_vision/src/tk_vision_specialized/test/test_vlm_match_client.py`

- [ ] **Step 1: Add failing tests for the Gemini decoder + client**

Two edits to `src/tk26_vision/src/tk_vision_specialized/test/test_vlm_match_client.py`:

(a) Add a new top-level import block **directly after** the existing `from tk_vision_specialized.vlm_match_client import (...)` block (do NOT append at the bottom — flake8 E402):

```python
from tk_vision_specialized.vlm_match_client_gemini import (
    GeminiMatchClient,
    decode_gemini_response,
)
```

(b) Append the new test functions to the **bottom** of the file:

```python
def test_decode_gemini_pixel_xyxy_passthrough():
    body = json.dumps({
        'detections': [
            {'label': 'cola', 'bbox_xyxy': [10, 20, 30, 40], 'confidence': 0.8},
        ]
    })
    rows = decode_gemini_response(body, scene_w=200, scene_h=200,
                                  allowed_labels={'cola'})
    assert len(rows) == 1
    assert rows[0].bbox == (10, 20, 30, 40)


def test_decode_gemini_clamps_out_of_bounds():
    body = json.dumps({
        'detections': [
            {'label': 'cola', 'bbox_xyxy': [-50, -10, 500, 1000], 'confidence': 0.7},
        ]
    })
    rows = decode_gemini_response(body, scene_w=100, scene_h=100,
                                  allowed_labels={'cola'})
    assert len(rows) == 1
    assert rows[0].bbox == (0, 0, 99, 99)


def test_decode_gemini_drops_hallucinated_label():
    body = json.dumps({
        'detections': [
            {'label': 'apple', 'bbox_xyxy': [0, 0, 50, 50], 'confidence': 0.9},
        ]
    })
    rows = decode_gemini_response(body, scene_w=100, scene_h=100,
                                  allowed_labels={'milk'})
    assert rows == []


def test_gemini_client_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='OPENROUTER_API_KEY'):
        GeminiMatchClient(model='google/gemini-2.5-pro')


def test_build_match_client_returns_gemini(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    client = build_match_client('gemini')
    assert isinstance(client, GeminiMatchClient)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/test_vlm_match_client.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 5 new tests fail with `ImportError` for `vlm_match_client_gemini`.

- [ ] **Step 3: Implement the Gemini backend**

Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/vlm_match_client_gemini.py`:

```python
"""Gemini match client (OpenRouter compatible endpoint)."""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
from dotenv import load_dotenv

from ._vlm_common import strip_fences, encode_data_url
from .nms import MatchRow, Bbox


# Load .env once at module-import time so that pytest's monkeypatch.delenv
# is authoritative after first construction (the workspace .env carries a
# real OPENROUTER_API_KEY which would otherwise repopulate after delete).
# Same pattern Task 5's QwenMatchClient uses; matches kimi_api/_env.py.
load_dotenv(override=False)


_GEMINI_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
_GEMINI_DEFAULT_MODEL = 'google/gemini-2.5-pro'


def _decode_bbox_pixels(
    bbox_xyxy, scene_w: int, scene_h: int,
) -> Bbox | None:
    if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(bbox_xyxy[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), scene_w - 1))
    y1 = max(0, min(int(round(y1)), scene_h - 1))
    x2 = max(0, min(int(round(x2)), scene_w - 1))
    y2 = max(0, min(int(round(y2)), scene_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def decode_gemini_response(
    body: str, *, scene_w: int, scene_h: int, allowed_labels: set[str],
) -> list[MatchRow]:
    try:
        parsed = json.loads(strip_fences(body))
    except (json.JSONDecodeError, ValueError):
        return []
    detections = parsed.get('detections') if isinstance(parsed, dict) else None
    if not isinstance(detections, list):
        return []

    rows: list[MatchRow] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = det.get('label')
        if not isinstance(label, str) or label not in allowed_labels:
            continue
        bbox = _decode_bbox_pixels(det.get('bbox_xyxy'), scene_w, scene_h)
        if bbox is None:
            continue
        try:
            conf = float(det.get('confidence', 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(conf, 1.0))
        rows.append(MatchRow(label=label, bbox=bbox, conf=conf))
    return rows


def _gemini_system_prompt(
    labels: list[str], scene_w: int, scene_h: int,
) -> str:
    label_list = ', '.join(f'"{lbl}"' for lbl in labels)
    return (
        "You are a visual-grounding assistant for a service robot. The user "
        f"provides one SCENE image of size {scene_w}x{scene_h} followed by "
        f"{len(labels)} REFERENCE images captioned with labels from this set: "
        f"[{label_list}]. Find every visible instance of any reference item in "
        "the scene and return bounding boxes. Coordinates 'bbox_xyxy' are "
        f"[x1, y1, x2, y2] in absolute pixels over the {scene_w}x{scene_h} "
        f"scene image. The 'label' field must be exactly one of [{label_list}]. "
        "Confidence is a subjective match score in [0.0, 1.0]. If no reference "
        "item is visible, return detections=[]. Output JSON only, with no "
        "commentary or markdown fences."
    )


class GeminiMatchClient:
    def __init__(
        self,
        model: str = '',
        base_url: str = '',
    ):
        # load_dotenv ran at module-import time; just read os.environ here
        self._api_key = os.environ.get('OPENROUTER_API_KEY', '')
        if not self._api_key:
            raise RuntimeError(
                'OPENROUTER_API_KEY not found in env '
                '(required for Gemini provider)'
            )
        self._model = model or _GEMINI_DEFAULT_MODEL
        self._base_url = base_url or _GEMINI_DEFAULT_BASE_URL

    def match_batch(
        self,
        scene_bgr: np.ndarray,
        refs: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> list[MatchRow]:
        if not refs:
            return []

        from openai import OpenAI

        h, w = scene_bgr.shape[:2]
        labels = [label for label, _url in refs]
        allowed_labels = set(labels)

        scene_url = encode_data_url(scene_bgr)
        content: list[dict] = [{'type': 'image_url',
                                'image_url': {'url': scene_url}}]
        for label, url in refs:
            content.append({'type': 'image_url',
                            'image_url': {'url': url}})
        content.append({
            'type': 'text',
            'text': (
                'Image 1 is the scene. The remaining images are reference '
                'photos, in order: '
                + ', '.join(
                    f'image {i+2} = "{lbl}"'
                    for i, lbl in enumerate(labels)
                )
                + '. Return all visible instances grouped by label.'
            ),
        })

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        try:
            for attempt in range(max(1, max_retries)):
                try:
                    completion = client.with_options(
                        timeout=timeout_s,
                    ).chat.completions.create(
                        model=self._model,
                        messages=[
                            {'role': 'system',
                             'content': _gemini_system_prompt(labels, w, h)},
                            {'role': 'user', 'content': content},
                        ],
                        response_format={'type': 'json_object'},
                    )
                    raw = completion.choices[0].message.content or ''
                    return decode_gemini_response(
                        raw, scene_w=w, scene_h=h, allowed_labels=allowed_labels,
                    )
                except Exception as exc:    # noqa: BLE001
                    if logger is not None:
                        logger.warning(
                            f'Gemini match attempt {attempt+1}/{max_retries} '
                            f'failed: {exc}'
                        )
            return []
        finally:
            try:
                client.close()
            except Exception:    # noqa: BLE001
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/test_vlm_match_client.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/vlm_match_client_gemini.py src/tk_vision_specialized/test/test_vlm_match_client.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): Gemini match client backend

OpenRouter-based Gemini 2.5 Pro adapter. Pixel xyxy convention (no
0..1000 normalize). Selected by build_match_client('gemini').

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — `vlm_judge_client.py`: full module

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/vlm_judge_client.py`
- Create: `src/tk26_vision/src/tk_vision_specialized/test/test_vlm_judge_client.py`

- [ ] **Step 1: Write failing tests for judge decode + factory**

Create `src/tk26_vision/src/tk_vision_specialized/test/test_vlm_judge_client.py`:

```python
"""Unit tests for vlm_judge_client.py."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tk_vision_specialized.vlm_judge_client import (
    JudgeChoice,
    decode_judge_response,
    QwenJudgeClient,
    GeminiJudgeClient,
    build_judge_client,
)


def test_decode_judge_winner():
    body = json.dumps({'label': 'milk', 'confidence': 0.95})
    choice = decode_judge_response(body, competing_labels={'milk', 'cola'})
    assert choice == JudgeChoice(label='milk', conf=0.95)


def test_decode_judge_abstain_via_null():
    body = json.dumps({'label': None, 'confidence': 0.0})
    choice = decode_judge_response(body, competing_labels={'milk', 'cola'})
    assert choice == JudgeChoice(label='', conf=0.0)


def test_decode_judge_abstain_via_empty_string():
    body = json.dumps({'label': '', 'confidence': 0.0})
    choice = decode_judge_response(body, competing_labels={'milk', 'cola'})
    assert choice == JudgeChoice(label='', conf=0.0)


def test_decode_judge_rejects_out_of_set_label():
    body = json.dumps({'label': 'banana', 'confidence': 0.9})
    choice = decode_judge_response(body, competing_labels={'milk', 'cola'})
    # Hallucinated label -> abstain.
    assert choice == JudgeChoice(label='', conf=0.0)


def test_decode_judge_bad_json_returns_none():
    body = 'not json'
    choice = decode_judge_response(body, competing_labels={'milk'})
    assert choice is None


def test_qwen_judge_client_init(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'fake')
    client = QwenJudgeClient(model='qwen3-vl-plus')
    assert client._api_key == 'fake'


def test_gemini_judge_client_init(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    client = GeminiJudgeClient(model='google/gemini-2.5-pro')
    assert client._api_key == 'fake'


def test_qwen_judge_end_to_end(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'fake')
    crop = np.zeros((40, 60, 3), dtype=np.uint8)
    body = json.dumps({'label': 'cola', 'confidence': 0.88})

    class FakeOpenAI:
        def __init__(self, *a, **kw):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content=body))])
                )
            )
        def with_options(self, **kw):
            return self
        def close(self):
            pass

    with patch('tk_vision_specialized.vlm_judge_client.OpenAI', FakeOpenAI):
        client = QwenJudgeClient(model='qwen3-vl-plus')
        choice = client.choose(
            crop_bgr=crop,
            competing=[('milk', 'data:image/jpeg;base64,M'),
                       ('cola', 'data:image/jpeg;base64,C')],
            timeout_s=5.0, max_retries=1,
        )

    assert choice is not None
    assert choice.label == 'cola'


def test_build_judge_client_unknown_provider_raises():
    with pytest.raises(ValueError, match='Unknown provider'):
        build_judge_client('llama')
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/test_vlm_judge_client.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the judge client**

Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/vlm_judge_client.py`:

```python
"""Provider-agnostic judge client for object_match_all conflict resolution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from dotenv import load_dotenv

from ._vlm_common import strip_fences, encode_data_url


# Load .env once at module-import time so pytest's monkeypatch.delenv is
# authoritative after first construction. Same defensive pattern Tasks 5
# and 6 use; matches kimi_api/_env.py.
load_dotenv(override=False)


_QWEN_DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
_QWEN_KEY_NAMES = ('DASHCOPE_API_KEY', 'DASHSCOPE_API_KEY')
_QWEN_DEFAULT_MODEL = 'qwen3-vl-plus'

_GEMINI_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
_GEMINI_DEFAULT_MODEL = 'google/gemini-2.5-pro'


@dataclass(frozen=True)
class JudgeChoice:
    label: str    # one of competing labels, or '' for abstain
    conf: float


class JudgeClient(Protocol):
    def choose(
        self,
        crop_bgr: np.ndarray,
        competing: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> JudgeChoice | None: ...


def decode_judge_response(
    body: str, *, competing_labels: set[str],
) -> JudgeChoice | None:
    """Parse a judge response body.

    Returns None on JSON parse failure (caller will fall back).
    Returns JudgeChoice(label='', conf=0.0) on abstain or hallucinated label."""
    try:
        parsed = json.loads(strip_fences(body))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None

    label = parsed.get('label')
    if label is None or label == '' or not isinstance(label, str):
        return JudgeChoice(label='', conf=0.0)
    if label not in competing_labels:
        return JudgeChoice(label='', conf=0.0)

    try:
        conf = float(parsed.get('confidence', 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(conf, 1.0))
    return JudgeChoice(label=label, conf=conf)


def _judge_system_prompt(labels: list[str]) -> str:
    label_list = ', '.join(f'"{lbl}"' for lbl in labels)
    return (
        "You are a tie-breaking visual-grounding assistant. The user provides "
        "one SCENE CROP image followed by N REFERENCE images, each captioned "
        f"with a label from this set: [{label_list}]. Choose the single label "
        "that best matches the object in the scene crop. If none of the "
        "references match what is in the crop, return label = null to abstain. "
        "The 'label' field must be exactly one of the input labels or null. "
        "Confidence is your match score in [0.0, 1.0]. Output JSON only, with "
        "no commentary or markdown fences."
    )


class _BaseJudgeClient:
    """Shared HTTP plumbing; subclasses set provider-specific config."""

    _api_key: str
    _model: str
    _base_url: str

    def choose(
        self,
        crop_bgr: np.ndarray,
        competing: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> JudgeChoice | None:
        if not competing:
            return None
        labels = [label for label, _url in competing]
        allowed = set(labels)

        crop_url = encode_data_url(crop_bgr)
        content: list[dict] = [{'type': 'image_url',
                                'image_url': {'url': crop_url}}]
        for label, url in competing:
            content.append({'type': 'image_url',
                            'image_url': {'url': url}})
        content.append({
            'type': 'text',
            'text': (
                'Image 1 is the scene crop. The remaining images are reference '
                'photos, in order: '
                + ', '.join(
                    f'image {i+2} = "{lbl}"'
                    for i, lbl in enumerate(labels)
                )
                + '. Choose the best matching label or return null to abstain.'
            ),
        })

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        try:
            for attempt in range(max(1, max_retries)):
                try:
                    completion = client.with_options(
                        timeout=timeout_s,
                    ).chat.completions.create(
                        model=self._model,
                        messages=[
                            {'role': 'system',
                             'content': _judge_system_prompt(labels)},
                            {'role': 'user', 'content': content},
                        ],
                        response_format={'type': 'json_object'},
                    )
                    raw = completion.choices[0].message.content or ''
                    return decode_judge_response(raw, competing_labels=allowed)
                except Exception as exc:    # noqa: BLE001
                    if logger is not None:
                        logger.warning(
                            f'Judge attempt {attempt+1}/{max_retries} '
                            f'failed: {exc}'
                        )
            return None
        finally:
            try:
                client.close()
            except Exception:    # noqa: BLE001
                pass


class QwenJudgeClient(_BaseJudgeClient):
    def __init__(self, model: str = '', base_url: str = ''):
        # load_dotenv ran at module-import time; read os.environ here
        self._api_key = ''
        for name in _QWEN_KEY_NAMES:
            val = os.environ.get(name)
            if val:
                self._api_key = val
                break
        if not self._api_key:
            raise RuntimeError(
                'DashScope API key not found in env '
                f'(looked for {_QWEN_KEY_NAMES})'
            )
        self._model = model or _QWEN_DEFAULT_MODEL
        self._base_url = base_url or _QWEN_DEFAULT_BASE_URL


class GeminiJudgeClient(_BaseJudgeClient):
    def __init__(self, model: str = '', base_url: str = ''):
        # load_dotenv ran at module-import time; read os.environ here
        self._api_key = os.environ.get('OPENROUTER_API_KEY', '')
        if not self._api_key:
            raise RuntimeError(
                'OPENROUTER_API_KEY not found in env '
                '(required for Gemini provider)'
            )
        self._model = model or _GEMINI_DEFAULT_MODEL
        self._base_url = base_url or _GEMINI_DEFAULT_BASE_URL


def build_judge_client(provider: str, **opts) -> JudgeClient:
    if provider == 'qwen':
        return QwenJudgeClient(**opts)
    if provider == 'gemini':
        return GeminiJudgeClient(**opts)
    raise ValueError(f'Unknown provider: {provider!r}')


try:
    from openai import OpenAI
except ImportError:    # pragma: no cover
    OpenAI = None    # type: ignore[assignment]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/test_vlm_judge_client.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/vlm_judge_client.py src/tk_vision_specialized/test/test_vlm_judge_client.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): vlm_judge_client.py with Qwen + Gemini backends

Per-conflict-cluster judge call. Returns JudgeChoice(label, conf) or None
on transport failure (caller falls back to highest-conf row). Empty/null
label is the abstain signal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — `match_pipeline.py`: orchestrator with fake clients

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/match_pipeline.py`
- Create: `src/tk26_vision/src/tk_vision_specialized/test/test_match_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Create `src/tk26_vision/src/tk_vision_specialized/test/test_match_pipeline.py`:

```python
"""Unit tests for match_pipeline.py.

The pipeline is pure-Python (no ROS, no network, no GPU). Tests drive it
with fake clients to cover the full failure matrix."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from tk_vision_specialized.match_pipeline import (
    MatchPipeline,
    PipelineParams,
    FinalRow,
)
from tk_vision_specialized.nms import MatchRow

# Touch FinalRow so the `from ... import FinalRow` isn't flagged unused
# (FinalRow is the documented return-row shape; future tests will assert
# against fields on it).
__all__ = ['FinalRow']


@dataclass
class FakeMatchClient:
    """Returns a deterministic per-batch result keyed by the labels in the
    batch. Raises if a batch is configured with `raise_on_call=True`."""
    per_batch: dict[frozenset[str], list[MatchRow]] = field(default_factory=dict)
    raise_for: set[frozenset[str]] = field(default_factory=set)
    sleep_for: dict[frozenset[str], float] = field(default_factory=dict)
    calls: list[frozenset[str]] = field(default_factory=list)

    def match_batch(self, scene_bgr, refs, *, timeout_s, max_retries, logger=None):
        key = frozenset(label for label, _url in refs)
        self.calls.append(key)
        if key in self.sleep_for:
            time.sleep(self.sleep_for[key])
        if key in self.raise_for:
            raise RuntimeError(f'fake match failure for {sorted(key)}')
        return list(self.per_batch.get(key, []))


@dataclass
class FakeJudgeClient:
    """Returns a deterministic choice per competing-label-set."""
    choices: dict[frozenset[str], object] = field(default_factory=dict)
    raise_for: set[frozenset[str]] = field(default_factory=set)
    calls: list[frozenset[str]] = field(default_factory=list)

    def choose(self, crop_bgr, competing, *, timeout_s, max_retries, logger=None):
        key = frozenset(label for label, _url in competing)
        self.calls.append(key)
        if key in self.raise_for:
            raise RuntimeError(f'fake judge failure for {sorted(key)}')
        return self.choices.get(key)


class FakeSam:
    """Returns one mask per bbox, drawn as the rectangle itself."""

    def __init__(self):
        self.calls = []

    def segment(self, rgb_bgr, bboxes):
        self.calls.append(list(bboxes))
        h, w = rgb_bgr.shape[:2]
        masks = []
        for x1, y1, x2, y2 in bboxes:
            m = np.zeros((h, w), dtype=bool)
            m[y1:y2, x1:x2] = True
            masks.append(m)
        return masks, 0.001


@dataclass
class FakeCameraData:
    """Surfaces only the centroid + TF lookup methods the pipeline calls."""
    centroid_value: object = None       # geometry_msgs.Point-like; None to fail
    tf_value: object = None             # transformed point; None to fail
    tf_support: bool = True

    def centroid_for(self, points, mask, valid_mask, bbox, camera):
        return self.centroid_value

    def transform_point(self, point, target, source, stamp):
        return self.tf_value

    def frame_supports_tf_transform(self, camera):
        return self.tf_support


def _items_map():
    return {
        'milk':  'data:image/jpeg;base64,M',
        'cola':  'data:image/jpeg;base64,C',
        'bread': 'data:image/jpeg;base64,B',
    }


def _make_scene():
    return np.zeros((400, 400, 3), dtype=np.uint8)


def _params(**overrides):
    base = dict(
        batch_size=2,
        max_workers=4,
        vlm_per_call_timeout_s=5.0,
        vlm_max_retries=1,
        stage1_timeout_s=10.0,
        stage2_timeout_s=10.0,
        nms_within_category_iou=0.5,
        cluster_iou=0.5,
        judge_crop_margin_px=10,
        min_valid_centroid_pixels=8,
    )
    base.update(overrides)
    return PipelineParams(**base)


class _Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def test_empty_scene_returns_no_rows():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [],
        frozenset({'bread'}): [],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0, 0, 1)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert final == []
    assert counters['batches_ok'] == 2
    assert counters['rows_in'] == 0


def test_single_hit_passes_through():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
        ],
        frozenset({'bread'}): [],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert len(final) == 1
    assert final[0].row.label == 'milk'


def test_conflict_resolved_by_judge():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.6),
            MatchRow(label='cola', bbox=(12, 12, 52, 52), conf=0.85),
        ],
        frozenset({'bread'}): [],
    })
    judge_client = FakeJudgeClient(choices={
        frozenset({'milk', 'cola'}):
            type('JC', (), {'label': 'milk', 'conf': 0.95})(),
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=judge_client,
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert len(final) == 1
    assert final[0].row.label == 'milk'
    assert final[0].row.conf == pytest.approx(0.95)   # judge's conf wins
    assert counters['judge_ok'] == 1


def test_judge_abstain_drops_cluster():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.6),
            MatchRow(label='cola', bbox=(12, 12, 52, 52), conf=0.65),
        ],
        frozenset({'bread'}): [],
    })
    judge_client = FakeJudgeClient(choices={
        frozenset({'milk', 'cola'}):
            type('JC', (), {'label': '', 'conf': 0.0})(),
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=judge_client,
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert final == []
    assert counters['judge_abstain'] == 1


def test_judge_failure_falls_back_to_top_conf():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.6),
            MatchRow(label='cola', bbox=(12, 12, 52, 52), conf=0.85),
        ],
        frozenset({'bread'}): [],
    })
    judge_client = FakeJudgeClient(raise_for={frozenset({'milk', 'cola'})})
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=judge_client,
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert len(final) == 1
    assert final[0].row.label == 'cola'   # top conf wins on judge failure
    assert counters['judge_fail'] == 1


def test_one_batch_fails_others_survive():
    match_client = FakeMatchClient(
        per_batch={
            frozenset({'milk', 'cola'}): [
                MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
            ],
            frozenset({'bread'}): [
                MatchRow(label='bread', bbox=(200, 200, 250, 250), conf=0.8),
            ],
        },
        raise_for={frozenset({'milk', 'cola'})},
    )
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert len(final) == 1
    assert final[0].row.label == 'bread'
    assert counters['batches_fail'] == 1
    assert counters['batches_ok'] == 1


def test_detection_dropped_when_no_valid_depth():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
        ],
        frozenset({'bread'}): [],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=None),    # always no depth
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert final == []
    assert counters['detections_dropped_no_depth'] == 1


def test_tf_failure_clears_results():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
        ],
        frozenset({'bread'}): [],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(
            centroid_value=_Point(0.1, 0.2, 0.5),
            tf_value=None,            # TF fails
            tf_support=True,
        ),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='base_link',
    )
    assert final == []
    assert counters['tf_failed'] == 1


def test_category_filter_restricts_scan():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
        ],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(batch_size=2),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=['milk'],
        target_frame='',
    )
    assert len(final) == 1
    assert match_client.calls == [frozenset({'milk'})]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/test_match_pipeline.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: `ImportError: cannot import name 'MatchPipeline'`.

- [ ] **Step 3: Implement the pipeline**

Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/match_pipeline.py`:

```python
"""Pure-Python orchestrator for object_match_all.

Knows nothing about ROS. Takes a captured scene + depth snapshot plus the
match/judge clients and returns the final list of FinalRow plus a
counters dict. The ROS service callback wraps this in the camera-sync,
TF, and response-packing layers."""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .nms import (
    Bbox, MatchRow, Cluster, JudgePayload,
    suppress_within_category, cluster_for_judge, build_judge_payload,
)


@dataclass(frozen=True)
class PipelineParams:
    batch_size: int
    max_workers: int
    vlm_per_call_timeout_s: float
    vlm_max_retries: int
    stage1_timeout_s: float
    stage2_timeout_s: float
    nms_within_category_iou: float
    cluster_iou: float
    judge_crop_margin_px: int
    min_valid_centroid_pixels: int


@dataclass
class FinalRow:
    row: MatchRow                    # final label + bbox + conf
    mask: np.ndarray                  # boolean HxW SAM mask
    point_camera: object              # geometry_msgs.Point in camera frame
    point_out: object                 # post-TF point (== point_camera if no TF)
    tf_failed: bool = False


def _chunks(seq: Sequence, n: int) -> Iterable[list]:
    for i in range(0, len(seq), max(1, n)):
        yield list(seq[i:i + n])


def _rect_mask(shape: tuple[int, int], bbox: Bbox) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = bbox
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return m


class MatchPipeline:
    def __init__(
        self, *,
        match_client,
        judge_client,
        sam,
        camera,
        items: dict[str, str],
        params: PipelineParams,
        logger=None,
    ):
        self.match_client = match_client
        self.judge_client = judge_client
        self.sam = sam
        self.camera = camera
        self.items = items
        self.params = params
        self.log = logger

    # ---------------- top-level entry point ---------------------------------
    def run(
        self,
        *,
        scene_bgr: np.ndarray,
        points_xyz: np.ndarray,
        valid_mask: np.ndarray,
        camera: str,
        category_filter: Sequence[str],
        target_frame: str,
        source_frame: str = '',
        header_stamp=None,
    ) -> tuple[list[FinalRow], dict]:
        counters: dict[str, int] = {
            'batches_ok': 0, 'batches_fail': 0,
            'rows_in': 0, 'after_nms': 0,
            'clusters_total': 0, 'clusters_conflict': 0,
            'judge_ok': 0, 'judge_abstain': 0, 'judge_fail': 0,
            'detections_dropped_no_depth': 0,
            'tf_failed': 0,
        }

        # [2] Resolve category filter
        keys = list(category_filter) if category_filter else list(self.items.keys())
        keys = [k for k in keys if k in self.items]
        if not keys:
            return [], counters

        refs = [(k, self.items[k]) for k in keys]

        # [3] Partition + [4] stage-1 concurrent VLM match
        rows: list[MatchRow] = []
        with ThreadPoolExecutor(max_workers=self.params.max_workers) as pool:
            futures = {
                pool.submit(
                    self.match_client.match_batch,
                    scene_bgr, batch,
                    timeout_s=self.params.vlm_per_call_timeout_s,
                    max_retries=self.params.vlm_max_retries,
                    logger=self.log,
                ): batch
                for batch in _chunks(refs, self.params.batch_size)
            }
            try:
                for fut in as_completed(futures, timeout=self.params.stage1_timeout_s):
                    try:
                        batch_rows = fut.result()
                        rows.extend(batch_rows)
                        counters['batches_ok'] += 1
                    except Exception as exc:    # noqa: BLE001
                        counters['batches_fail'] += 1
                        if self.log is not None:
                            self.log.warning(f'match batch failed: {exc}')
            except FutureTimeoutError:
                if self.log is not None:
                    self.log.warning('stage1 budget elapsed; cancelling stragglers')
                for fut in futures:
                    if not fut.done():
                        counters['batches_fail'] += 1
                        fut.cancel()

        counters['rows_in'] = len(rows)
        if not rows:
            return [], counters

        # [5] Within-category NMS
        rows = suppress_within_category(rows, iou_thresh=self.params.nms_within_category_iou)
        counters['after_nms'] = len(rows)

        # [6] Cross-category clustering
        clusters = cluster_for_judge(rows, iou_thresh=self.params.cluster_iou)
        counters['clusters_total'] = len(clusters)
        counters['clusters_conflict'] = sum(1 for c in clusters if c.is_conflict())

        # [7] Stage-2 concurrent judge
        survivors = self._resolve_conflicts(clusters, scene_bgr, counters)

        if not survivors:
            return [], counters

        # [9] Batched SAM
        bboxes = [r.bbox for r in survivors]
        masks, _sam_s = self.sam.segment(scene_bgr, bboxes)
        if len(masks) != len(survivors):
            # Defensive: pad with rect masks. SamPredictor contract says
            # 1:1, but a backend swap could change that.
            h, w = scene_bgr.shape[:2]
            while len(masks) < len(survivors):
                masks.append(_rect_mask((h, w), survivors[len(masks)].bbox))

        # [10] Centroids
        finals: list[FinalRow] = []
        for row, mask in zip(survivors, masks):
            pt = self.camera.centroid_for(points_xyz, mask, valid_mask, row.bbox, camera)
            if pt is None:
                rect = _rect_mask(scene_bgr.shape[:2], row.bbox)
                pt = self.camera.centroid_for(points_xyz, rect, valid_mask, row.bbox, camera)
            if pt is None:
                counters['detections_dropped_no_depth'] += 1
                if self.log is not None:
                    self.log.warning(
                        f'dropping {row.label}: no valid depth in bbox {row.bbox}')
                continue
            finals.append(FinalRow(row=row, mask=mask, point_camera=pt, point_out=pt))

        if not finals:
            return [], counters

        # [11] Optional TF
        if target_frame and self.camera.frame_supports_tf_transform(camera):
            for fr in finals:
                transformed = self.camera.transform_point(
                    fr.point_camera, target_frame, source_frame, header_stamp,
                )
                if transformed is None:
                    fr.tf_failed = True
                    counters['tf_failed'] += 1
                else:
                    fr.point_out = transformed
            if any(fr.tf_failed for fr in finals):
                # All-or-nothing per spec §11.
                return [], counters

        return finals, counters

    # ---------------- helpers ----------------------------------------------
    def _resolve_conflicts(
        self,
        clusters: list[Cluster],
        scene_bgr: np.ndarray,
        counters: dict,
    ) -> list[MatchRow]:
        survivors: list[MatchRow] = []
        conflict_payloads: list[JudgePayload] = []

        for cluster in clusters:
            if not cluster.is_conflict():
                survivors.append(
                    max(cluster.rows, key=lambda r: r.conf)
                    if len(cluster.rows) > 1 else cluster.rows[0]
                )
                continue
            payload = build_judge_payload(
                cluster, self.items, scene_bgr, self.params.judge_crop_margin_px,
            )
            conflict_payloads.append(payload)

        if not conflict_payloads:
            return survivors

        with ThreadPoolExecutor(max_workers=self.params.max_workers) as pool:
            futures = {
                pool.submit(
                    self.judge_client.choose,
                    p.crop, p.competing,
                    timeout_s=self.params.vlm_per_call_timeout_s,
                    max_retries=self.params.vlm_max_retries,
                    logger=self.log,
                ): p
                for p in conflict_payloads
            }
            try:
                for fut in as_completed(futures, timeout=self.params.stage2_timeout_s):
                    payload = futures[fut]
                    try:
                        choice = fut.result()
                    except Exception as exc:    # noqa: BLE001
                        if self.log is not None:
                            self.log.warning(f'judge call failed: {exc}')
                        choice = None
                    survivors.extend(self._row_from_choice(payload, choice, counters))
            except FutureTimeoutError:
                if self.log is not None:
                    self.log.warning('stage2 budget elapsed; falling back')
                for fut, payload in futures.items():
                    if fut.done():
                        continue
                    fut.cancel()
                    survivors.extend(self._row_from_choice(payload, None, counters))

        return survivors

    def _row_from_choice(
        self,
        payload: JudgePayload,
        choice,
        counters: dict,
    ) -> list[MatchRow]:
        cluster = payload.cluster
        if choice is None:
            counters['judge_fail'] += 1
            return [max(cluster.rows, key=lambda r: r.conf)]

        label = getattr(choice, 'label', '')
        conf = float(getattr(choice, 'conf', 0.0))
        if not label:
            counters['judge_abstain'] += 1
            return []

        cluster_labels = {r.label for r in cluster.rows}
        if label not in cluster_labels:
            counters['judge_fail'] += 1
            return [max(cluster.rows, key=lambda r: r.conf)]

        counters['judge_ok'] += 1
        chosen = max(
            (r for r in cluster.rows if r.label == label),
            key=lambda r: r.conf,
        )
        return [MatchRow(label=chosen.label, bbox=chosen.bbox, conf=conf)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/test_match_pipeline.py -v -p no:launch_testing_ros --ignore=test/test_spot_on_shelf.py
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/match_pipeline.py src/tk_vision_specialized/test/test_match_pipeline.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): match_pipeline orchestrator with full coverage

Pure-Python pipeline: batched match -> within-cat NMS -> cross-cat clusters
-> concurrent judge with abstain/timeout/fallback -> batched SAM -> centroid
-> all-or-nothing TF. Driven by fake clients in test/test_match_pipeline.py
to cover the failure matrix without network or GPU.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — `camera_data_source.py`: ROS-aware camera composable

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/camera_data_source.py`

This module composes camera sync + intrinsics + depth-to-3D + TF + VisionLogger from logic ported out of `YOLOSegmentationNode`. It does not subclass `Node`. The owning node passes itself as the `ros_node` argument so we can register subscribers/services through it.

Because this module holds ROS state, we test it via the integration suite (Task 11) rather than pytest. A minimal smoke import-check is added here.

- [ ] **Step 1: Implement `CameraDataSource`**

Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/camera_data_source.py`:

```python
"""ROS-aware camera composable for object_match_all.

Owns:
  - color + depth + camera_info subscribers for realsense and orbbec
  - ApproximateTimeSynchronizer per camera
  - TF buffer + listener
  - VisionLogger
  - depth-to-3D + centroid + TF helpers

Logic is lifted from object_detection_new.YOLOSegmentationNode but the
class is plain (no Node subclass)."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy.duration
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from tf2_ros import (
    Buffer, TransformListener,
    LookupException, ConnectivityException, ExtrapolationException,
)
from tf2_geometry_msgs import do_transform_point

from vision_util.vision_logging import VisionLogger


@dataclass
class CameraTopics:
    realsense_image: str
    realsense_depth: str
    realsense_camera_info: str
    orbbec_image: str
    orbbec_depth: str
    orbbec_camera_info: str


class CameraDataSource:
    """Camera-sync + intrinsics + depth-to-3D + TF + logging, composable.

    Construct from a Node so we can attach subscribers/TF listener to the
    same lifecycle. All public methods are thread-safe in the sense that
    they take their own locks; the underlying ros2 callback group should
    still be MutuallyExclusive on the service to serialise calls."""

    def __init__(self, ros_node, *, topics: CameraTopics, params, logger=None):
        self._node = ros_node
        self._log = logger or ros_node.get_logger()
        self._params = params

        self.bridge = CvBridge()

        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, ros_node)

        self.lock_msg = threading.Lock()
        self.lock_info = threading.Lock()
        self.camera_intrinsic: dict[str, dict | None] = {
            'realsense': None, 'orbbec': None,
        }
        self.recent_sync_msg: dict[str, tuple | None] = {
            'realsense': None, 'orbbec': None,
        }
        self.recent_publish_time: dict[str, object] = {
            'realsense': None, 'orbbec': None,
        }

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Realsense (color Image + aligned depth Image + CameraInfo)
        self._rs_color = Subscriber(ros_node, Image, topics.realsense_image, qos_profile=qos)
        self._rs_depth = Subscriber(ros_node, Image, topics.realsense_depth, qos_profile=qos)
        self._rs_info = ros_node.create_subscription(
            CameraInfo, topics.realsense_camera_info,
            lambda msg: self._set_intrinsic('realsense', msg),
            qos_profile=qos,
        )
        self._rs_sync = ApproximateTimeSynchronizer(
            [self._rs_color, self._rs_depth], queue_size=5, slop=0.05,
        )
        self._rs_sync.registerCallback(
            lambda c, d: self._on_sync('realsense', (c, d))
        )

        # Orbbec (color Image + PointCloud2 depth + CameraInfo)
        self._ob_color = Subscriber(ros_node, Image, topics.orbbec_image, qos_profile=qos)
        self._ob_depth = Subscriber(ros_node, PointCloud2, topics.orbbec_depth, qos_profile=qos)
        self._ob_info = ros_node.create_subscription(
            CameraInfo, topics.orbbec_camera_info,
            lambda msg: self._set_intrinsic('orbbec', msg),
            qos_profile=qos,
        )
        self._ob_sync = ApproximateTimeSynchronizer(
            [self._ob_color, self._ob_depth], queue_size=5, slop=0.05,
        )
        self._ob_sync.registerCallback(
            lambda c, d: self._on_sync('orbbec', (c, d))
        )

        # VisionLogger takes (node, enabled, base_folder) — we read the
        # corresponding ROS params (declared by the owning node, e.g. the
        # ObjectMatchAllServer in Task 10) and pass them in directly.
        try:
            log_enabled = bool(
                ros_node.get_parameter('vision_logging_enabled').value
            )
        except Exception:    # noqa: BLE001 — param may not be declared yet
            log_enabled = True
        try:
            log_folder = str(
                ros_node.get_parameter('vision_log_folder').value
            )
        except Exception:    # noqa: BLE001
            log_folder = 'vision_log'
        self.vision_logger = VisionLogger(
            ros_node,
            enabled=log_enabled,
            base_folder=log_folder,
        )

    # ---------------- subscriber callbacks ---------------------------------
    def _set_intrinsic(self, camera: str, msg: CameraInfo) -> None:
        with self.lock_info:
            self.camera_intrinsic[camera] = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'cx': msg.k[2], 'cy': msg.k[5],
                'width': msg.width, 'height': msg.height,
                'frame_id': msg.header.frame_id,
            }

    def _on_sync(self, camera: str, msg_pair: tuple) -> None:
        with self.lock_msg:
            self.recent_sync_msg[camera] = msg_pair
            self.recent_publish_time[camera] = self._node.get_clock().now()

    # ---------------- public API -------------------------------------------
    def snapshot(self, camera: str):
        """Wait briefly for a recent (color, depth) pair and return it
        alongside processed arrays. Returns None on timeout."""
        sync_thres_s = float(self._params.img_sync_thres_s)
        deadline = self._node.get_clock().now() + rclpy.duration.Duration(
            seconds=float(self._params.sync_wait_time_s),
        )
        while self._node.get_clock().now() < deadline:
            with self.lock_msg:
                pair = self.recent_sync_msg.get(camera)
                rt = self.recent_publish_time.get(camera)
            if pair is not None and rt is not None:
                age = (self._node.get_clock().now() - rt).nanoseconds / 1e9
                if age <= sync_thres_s:
                    with self.lock_info:
                        intrinsic = copy.deepcopy(self.camera_intrinsic.get(camera))
                    if intrinsic is None:
                        return None
                    color_msg, depth_msg = pair
                    if camera == 'realsense':
                        return self._process_realsense(color_msg, depth_msg, intrinsic)
                    return self._process_orbbec(color_msg, depth_msg, intrinsic)
            time.sleep(0.05)
        return None

    def _process_realsense(self, color_msg, depth_msg, intrinsic):
        rgb_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_m = depth_mm.astype(np.float32) / 1000.0
        h, w = depth_m.shape[:2]
        fx, fy = intrinsic['fx'], intrinsic['fy']
        cx, cy = intrinsic['cx'], intrinsic['cy']
        u = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        v = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_xyz = np.stack([x, y, z], axis=-1)
        valid = (z > self._params.min_depth_m) & (z < self._params.max_depth_m)
        header = Header(stamp=color_msg.header.stamp,
                        frame_id=color_msg.header.frame_id)
        return rgb_bgr, points_xyz, valid, header, color_msg

    def _process_orbbec(self, color_msg, points_msg, intrinsic):
        # Orbbec depth arrives as PointCloud2 already aligned to color.
        # Reproject into the image grid using the (cx, fx, cy, fy) intrinsic.
        rgb_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        h, w = rgb_bgr.shape[:2]
        from sensor_msgs_py import point_cloud2 as pc2
        pts = np.asarray(list(pc2.read_points(
            points_msg, field_names=('x', 'y', 'z'), skip_nans=False,
        )), dtype=np.float32)
        if pts.size == 0 or pts.shape[0] != h * w:
            return None
        pts = pts.reshape(h, w, 3)
        valid = np.isfinite(pts[:, :, 2]) & (pts[:, :, 2] > self._params.min_depth_m) \
            & (pts[:, :, 2] < self._params.max_depth_m)
        header = Header(stamp=color_msg.header.stamp,
                        frame_id=color_msg.header.frame_id)
        return rgb_bgr, pts, valid, header, color_msg

    def centroid_for(
        self,
        points_xyz: np.ndarray, mask: np.ndarray, valid_mask: np.ndarray,
        bbox, camera: str,
    ):
        x1, y1, x2, y2 = bbox
        h, w = points_xyz.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return None
        sub_pts = points_xyz[y1:y2, x1:x2]
        sub_mask = mask[y1:y2, x1:x2] & valid_mask[y1:y2, x1:x2]
        if not np.any(sub_mask):
            return None
        sel = sub_pts[sub_mask]
        if sel.shape[0] < int(self._params.min_valid_centroid_pixels):
            return None
        med = np.median(sel, axis=0)
        if not np.all(np.isfinite(med)):
            return None
        p = geometry_msgs.msg.Point()
        p.x, p.y, p.z = float(med[0]), float(med[1]), float(med[2])
        return p

    def frame_supports_tf_transform(self, camera: str) -> bool:
        # Both realsense + orbbec frames are published on /tf; the gate
        # exists for hypothetical synthetic/non-TF cameras.
        return True

    def transform_point(self, point, target_frame: str, source_frame: str, stamp):
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self._log.warning(f'TF {source_frame}->{target_frame} failed: {exc}')
            return None
        try:
            ps = PointStamped()
            ps.header = Header(stamp=stamp, frame_id=source_frame)
            ps.point = point
            return do_transform_point(ps, tf).point
        except Exception as exc:    # noqa: BLE001
            self._log.warning(f'do_transform_point failed: {exc}')
            return None

    def write(self, rgb_img, detections, *, request_ctx, branch, timings):
        try:
            self.vision_logger.write(
                rgb_img, detections,
                request_ctx=request_ctx, branch=branch, timings=timings,
            )
        except Exception as exc:    # noqa: BLE001
            self._log.warning(f'vision_logger.write failed: {exc}')
```

- [ ] **Step 2: Smoke-import check**

```bash
source /home/tinker/tk25_ws/src/tk26_vision/install/setup.bash
python -c "from tk_vision_specialized.camera_data_source import CameraDataSource, CameraTopics; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/camera_data_source.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): CameraDataSource composable

Port of camera-sync / intrinsics / depth-to-3D / TF / VisionLogger logic
out of YOLOSegmentationNode without subclassing. Plain class that takes
the owning ROS node as a constructor arg.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 — `object_match_all_server.py`: ROS node wiring + setup.py entry

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/object_match_all_server.py`
- Modify: `src/tk26_vision/src/tk_vision_specialized/setup.py:38`

- [ ] **Step 1: Implement the node**

Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/object_match_all_server.py`:

```python
"""ROS 2 service node /object_match_all.

Composes CameraDataSource, MatchClient, JudgeClient, SamPredictor,
ItemsMapLoader, and MatchPipeline. Single MutuallyExclusiveCallbackGroup
on the service so concurrent callers serialise at the node boundary."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from ament_index_python.packages import (
    get_package_share_directory, PackageNotFoundError,
)
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header

from object_detection_generalist.sam_mask import SamPredictor
from tinker_vision_msgs_26.msg import Object
from tinker_vision_msgs_26.srv import ObjectMatchAll
from vision_util.weights_cache import resolve_weights

from .camera_data_source import CameraDataSource, CameraTopics
from .items_map_loader import ItemsMapLoader
from .match_pipeline import MatchPipeline, PipelineParams
from .vlm_judge_client import build_judge_client
from .vlm_match_client import build_match_client


@dataclass
class NodeParams:
    # Pipeline params (a superset of PipelineParams plus camera/io knobs).
    batch_size: int
    max_workers: int
    vlm_per_call_timeout_s: float
    vlm_max_retries: int
    stage1_timeout_s: float
    stage2_timeout_s: float
    nms_within_category_iou: float
    cluster_iou: float
    judge_crop_margin_px: int
    min_valid_centroid_pixels: int
    # Camera io
    img_sync_thres_s: float
    sync_wait_time_s: float
    min_depth_m: float
    max_depth_m: float


class ObjectMatchAllServer(Node):
    def __init__(self):
        super().__init__('object_match_all_server')

        # Service / provider params
        self.declare_parameter('service_name', 'object_match_all')
        self.declare_parameter('vlm_provider', 'qwen')
        self.declare_parameter('judge_provider', '')
        self.declare_parameter('vlm_model', '')
        self.declare_parameter('judge_model', '')
        self.declare_parameter('vlm_base_url', '')
        # Camera topics
        self.declare_parameter('realsense_image_topic',
                               '/camera/xarm_camera/color/image_raw')
        self.declare_parameter('realsense_depth_topic',
                               '/camera/xarm_camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('realsense_camera_info_topic',
                               '/camera/xarm_camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('orbbec_image_topic', '/camera/color/image_raw')
        self.declare_parameter('orbbec_depth_topic', '/camera/depth_registered/points')
        self.declare_parameter('orbbec_camera_info_topic', '/camera/color/camera_info')
        # Items map + SAM
        self.declare_parameter('items_map_path', '')
        self.declare_parameter('sam_weights', 'mobile_sam.pt')
        self.declare_parameter('sam_device', '')
        # Pipeline / timeouts
        self.declare_parameter('batch_size', 3)
        self.declare_parameter('max_workers', 8)
        self.declare_parameter('vlm_per_call_timeout_s', 12.0)
        self.declare_parameter('vlm_max_retries', 1)
        self.declare_parameter('stage1_timeout_s', 15.0)
        self.declare_parameter('stage2_timeout_s', 10.0)
        self.declare_parameter('nms_within_category_iou', 0.5)
        self.declare_parameter('cluster_iou', 0.5)
        self.declare_parameter('judge_crop_margin_px', 20)
        self.declare_parameter('min_valid_centroid_pixels', 8)
        # Camera io
        self.declare_parameter('img_sync_thres_s', 0.5)
        self.declare_parameter('sync_wait_time_s', 1.5)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 8.0)
        # Logging
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self.declare_parameter('log_raw_vlm', False)

        self.params = NodeParams(
            batch_size=int(self.get_parameter('batch_size').value),
            max_workers=int(self.get_parameter('max_workers').value),
            vlm_per_call_timeout_s=float(
                self.get_parameter('vlm_per_call_timeout_s').value),
            vlm_max_retries=int(self.get_parameter('vlm_max_retries').value),
            stage1_timeout_s=float(self.get_parameter('stage1_timeout_s').value),
            stage2_timeout_s=float(self.get_parameter('stage2_timeout_s').value),
            nms_within_category_iou=float(
                self.get_parameter('nms_within_category_iou').value),
            cluster_iou=float(self.get_parameter('cluster_iou').value),
            judge_crop_margin_px=int(
                self.get_parameter('judge_crop_margin_px').value),
            min_valid_centroid_pixels=int(
                self.get_parameter('min_valid_centroid_pixels').value),
            img_sync_thres_s=float(self.get_parameter('img_sync_thres_s').value),
            sync_wait_time_s=float(self.get_parameter('sync_wait_time_s').value),
            min_depth_m=float(self.get_parameter('min_depth_m').value),
            max_depth_m=float(self.get_parameter('max_depth_m').value),
        )

        topics = CameraTopics(
            realsense_image=self.get_parameter('realsense_image_topic').value,
            realsense_depth=self.get_parameter('realsense_depth_topic').value,
            realsense_camera_info=self.get_parameter(
                'realsense_camera_info_topic').value,
            orbbec_image=self.get_parameter('orbbec_image_topic').value,
            orbbec_depth=self.get_parameter('orbbec_depth_topic').value,
            orbbec_camera_info=self.get_parameter(
                'orbbec_camera_info_topic').value,
        )

        self.bridge = CvBridge()
        self.camera = CameraDataSource(
            self, topics=topics, params=self.params, logger=self.get_logger(),
        )

        items_dir = self._resolve_items_dir()
        if not items_dir:
            raise RuntimeError(
                'Could not locate items_map directory; set the '
                'items_map_path parameter to an absolute path.'
            )
        self.items = ItemsMapLoader(items_dir, logger=self.get_logger())
        if len(self.items) == 0:
            self.get_logger().warning(
                f'items_map at {items_dir} is empty; every request will be 1.'
            )
        self.items_dict = {k: self.items.get_data_url(k) for k in self.items.keys()}

        provider = self.get_parameter('vlm_provider').value
        judge_provider = self.get_parameter('judge_provider').value or provider
        model = self.get_parameter('vlm_model').value or ''
        judge_model = self.get_parameter('judge_model').value or model
        base_url = self.get_parameter('vlm_base_url').value or ''

        self.match_client = build_match_client(provider, model=model, base_url=base_url)
        self.judge_client = build_judge_client(judge_provider, model=judge_model)

        sam_weights = resolve_weights(self.get_parameter('sam_weights').value)
        sam_device = self.get_parameter('sam_device').value or ''
        self.sam = SamPredictor(str(sam_weights), device=sam_device,
                                logger=self.get_logger())
        try:
            self.sam.segment(np.zeros((64, 64, 3), dtype=np.uint8), [(0, 0, 64, 64)])
        except Exception as exc:    # noqa: BLE001
            self.get_logger().warning(f'SAM warm-up failed: {exc}')

        self.pipeline = MatchPipeline(
            match_client=self.match_client,
            judge_client=self.judge_client,
            sam=self.sam,
            camera=self.camera,
            items=self.items_dict,
            params=PipelineParams(
                batch_size=self.params.batch_size,
                max_workers=self.params.max_workers,
                vlm_per_call_timeout_s=self.params.vlm_per_call_timeout_s,
                vlm_max_retries=self.params.vlm_max_retries,
                stage1_timeout_s=self.params.stage1_timeout_s,
                stage2_timeout_s=self.params.stage2_timeout_s,
                nms_within_category_iou=self.params.nms_within_category_iou,
                cluster_iou=self.params.cluster_iou,
                judge_crop_margin_px=self.params.judge_crop_margin_px,
                min_valid_centroid_pixels=self.params.min_valid_centroid_pixels,
            ),
            logger=self.get_logger(),
        )

        service_name = self.get_parameter('service_name').value
        self.srv = self.create_service(
            ObjectMatchAll, service_name, self._callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            f'object_match_all_server ready: service={service_name}, '
            f'items={len(self.items_dict)}, provider={provider}, '
            f'judge_provider={judge_provider}, batch_size={self.params.batch_size}'
        )

    # ---------------- callback ---------------------------------------------
    def _callback(self, req: ObjectMatchAll.Request, resp: ObjectMatchAll.Response):
        _t0 = time.perf_counter()
        resp.header = Header(stamp=self.get_clock().now().to_msg())
        resp.status = 1
        resp.error_msg = ''
        resp.person_id = 0
        resp.objects = []
        resp.detection_source = 'vlm_match_all'

        camera = self._select_camera(req.camera)
        snap = self.camera.snapshot(camera)
        if snap is None:
            resp.error_msg = f'No {camera} camera data within sync threshold'
            return resp
        rgb_bgr, points_xyz, valid_mask, header, raw_color_msg = snap
        resp.header = header

        # Resolve category filter and check for all-unknown.
        if req.category_filter:
            unknown = [k for k in req.category_filter if k not in self.items_dict]
            if unknown:
                self.get_logger().warning(
                    f'category_filter dropping unknown keys: {unknown}'
                )
            known = [k for k in req.category_filter if k in self.items_dict]
            if not known:
                resp.error_msg = f'Unknown items: {", ".join(unknown)}'
                return resp
        else:
            known = list(self.items_dict.keys())

        target_frame = req.target_frame.strip()
        source_frame = header.frame_id

        finals, counters = self.pipeline.run(
            scene_bgr=rgb_bgr,
            points_xyz=points_xyz,
            valid_mask=valid_mask,
            camera=camera,
            category_filter=known,
            target_frame=target_frame,
            source_frame=source_frame,
            header_stamp=header.stamp,
        )

        if not finals:
            resp.error_msg = self._error_msg_for_empty(counters, camera, target_frame)
            self._log_summary(counters, time.perf_counter() - _t0, resp.status)
            return resp

        finals = self._sort(finals, req)

        # Pack response.
        resp.header.frame_id = target_frame or source_frame
        for fr in finals:
            obj = Object()
            obj.cls = fr.row.label
            obj.conf = float(fr.row.conf)
            obj.id = 0
            obj.object_id = -1
            obj.similarity = 0.0
            obj.being_pointed = 0
            obj.centroid = fr.point_out
            resp.objects.append(obj)

        if req.return_rgb_image:
            resp.rgb_image = raw_color_msg
        if req.return_depth_image:
            depth_msg = self.bridge.cv2_to_imgmsg(
                points_xyz[:, :, 2].astype(np.float32), encoding='32FC1',
            )
            depth_msg.header = resp.header
            resp.depth_image = depth_msg
        if req.return_segments:
            seg_msgs = []
            for fr in finals:
                seg_msg = self.bridge.cv2_to_imgmsg(
                    (fr.mask.astype(np.uint8) * 255), encoding='8UC1',
                )
                seg_msg.header = resp.header
                seg_msgs.append(seg_msg)
            resp.segments = seg_msgs

        resp.status = 0
        self._log_summary(counters, time.perf_counter() - _t0, resp.status)
        return resp

    # ---------------- helpers ----------------------------------------------
    def _select_camera(self, request_camera: str) -> str:
        if 'realsense' in (request_camera or ''):
            return 'realsense'
        if 'orbbec' in (request_camera or ''):
            return 'orbbec'
        self.get_logger().warning(
            f'unknown camera "{request_camera}", defaulting to orbbec'
        )
        return 'orbbec'

    def _resolve_items_dir(self) -> str:
        override = self.get_parameter('items_map_path').value or ''
        if override:
            return override
        try:
            share_dir = get_package_share_directory('tk_vision_specialized')
        except PackageNotFoundError:
            share_dir = ''
        candidate = os.path.join(share_dir, 'items') if share_dir else ''
        if candidate and os.path.isfile(os.path.join(candidate, 'items_map.yaml')):
            return candidate
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(8):
            here = os.path.dirname(here)
            for guess in (
                os.path.join(here, 'src', 'items'),
                os.path.join(here, 'src', 'tk26_vision', 'src', 'items'),
            ):
                if os.path.isfile(os.path.join(guess, 'items_map.yaml')):
                    return guess
        return candidate or ''

    def _error_msg_for_empty(self, counters: dict, camera: str, target_frame: str) -> str:
        if counters.get('batches_ok', 0) == 0 and counters.get('batches_fail', 0) > 0:
            return (f'all VLM match batches failed: '
                    f'fail={counters["batches_fail"]}')
        if counters.get('tf_failed', 0) > 0:
            return f'TF -> {target_frame} unavailable for {counters["tf_failed"]} detections'
        if counters.get('detections_dropped_no_depth', 0) > 0:
            return 'no valid-depth pixels for any matched object'
        return 'no items matched'

    def _sort(self, finals, req: ObjectMatchAll.Request):
        if req.sort_closest:
            return sorted(finals, key=lambda fr: (fr.point_camera.x**2
                                                  + fr.point_camera.y**2
                                                  + fr.point_camera.z**2))
        if req.sort_highest:
            return sorted(finals, key=lambda fr: fr.point_camera.z)
        return sorted(finals, key=lambda fr: -fr.row.conf)

    def _log_summary(self, counters: dict, total_s: float, status: int) -> None:
        self.get_logger().info(
            f'match_all: status={status} '
            f'batches_ok={counters.get("batches_ok", 0)} '
            f'batches_fail={counters.get("batches_fail", 0)} '
            f'rows_in={counters.get("rows_in", 0)} '
            f'after_nms={counters.get("after_nms", 0)} '
            f'clusters_conflict={counters.get("clusters_conflict", 0)} '
            f'judge_ok={counters.get("judge_ok", 0)} '
            f'judge_abstain={counters.get("judge_abstain", 0)} '
            f'judge_fail={counters.get("judge_fail", 0)} '
            f'dropped_no_depth={counters.get("detections_dropped_no_depth", 0)} '
            f'tf_failed={counters.get("tf_failed", 0)} '
            f'total_s={total_s:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMatchAllServer()
    import multiprocessing
    num_threads = max(8, multiprocessing.cpu_count())
    executor = MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Add the entry point in `setup.py`**

Modify `src/tk26_vision/src/tk_vision_specialized/setup.py` — append one line inside the `console_scripts` list (after `object_match_server`):

```python
        'console_scripts': [
            'spot_on_shelf_server = tk_vision_specialized.spot_on_shelf_server:main',
            'waving_person_server = tk_vision_specialized.waving_person_server:main',
            'waving_client = tk_vision_specialized.waving_client:main',
            'check_waving_inference = tk_vision_specialized.check_waving_inference:main',
            'placing_location_server = tk_vision_specialized.placing_location_server:main',
            'object_match_server = tk_vision_specialized.object_match_server:main',
            'object_match_all_server = tk_vision_specialized.object_match_all_server:main',
        ],
```

- [ ] **Step 3: Build the package**

```bash
/home/tinker/tk25_ws/src/tk26_vision/scripts/build.sh --packages-select tk_vision_specialized
```

Expected: `Summary: 1 package finished` with no errors.

- [ ] **Step 4: Verify entry point exists**

```bash
source /home/tinker/tk25_ws/src/tk26_vision/install/setup.bash
ros2 pkg executables tk_vision_specialized | grep object_match_all_server
```

Expected: line `tk_vision_specialized object_match_all_server` prints.

- [ ] **Step 5: Smoke-start the node (will fail at runtime without VLM key + cameras, but it must reach service-advertise)**

This step verifies the wiring up to but not including a real request. If `OPENROUTER_API_KEY` / `DASHSCOPE_API_KEY` are missing it will fail at the right step (provider init) with a clear error.

```bash
DASHSCOPE_API_KEY=fake timeout 5s ros2 run tk_vision_specialized object_match_all_server 2>&1 | tee /tmp/oma_start.log
```

Expected: log contains `object_match_all_server ready: service=object_match_all, items=10`. The node exits when timeout kills it.

If you see `Could not locate items_map directory` instead, the install tree didn't ship `share/tk_vision_specialized/items/items_map.yaml` — re-run Step 3 with a clean build (`rm -rf build/tk_vision_specialized install/tk_vision_specialized` first).

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/object_match_all_server.py src/tk_vision_specialized/setup.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): object_match_all_server ROS node

Service /object_match_all wires CameraDataSource + ItemsMapLoader +
MatchPipeline + SAM + the provider-agnostic match/judge clients into a
MutuallyExclusive-serialized service callback. Mirrors the
ObjectDetection.srv response shape so callers of /object_detection_yolo
are drop-in retargetable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11 — Integration tier T0 + T1 extensions

**Files:**
- Modify: `src/tk26_vision/scripts/tests/t0_static.sh`
- Modify: `src/tk26_vision/scripts/tests/t1_startup.sh`

The existing test scripts have a per-node block. We add the same shape for the new node. Inspect them first to see the exact pattern:

```bash
sed -n '1,40p' /home/tinker/tk25_ws/src/tk26_vision/scripts/tests/t0_static.sh
sed -n '1,40p' /home/tinker/tk25_ws/src/tk26_vision/scripts/tests/t1_startup.sh
```

- [ ] **Step 1: T0 static — verify the new srv compiles + node imports**

Locate the section of `t0_static.sh` where existing srvs are imported (search for `from tinker_vision_msgs_26.srv import ObjectMatch`). Add `ObjectMatchAll` to the same import line, e.g.:

```python
from tinker_vision_msgs_26.srv import (
    ObjectDetection, ObjectDetectionGeneralist, ObjectMatch, ObjectMatchAll,
    # ... existing imports ...
)
```

Then locate the section where each entry-point module is import-checked (search for `from tk_vision_specialized.object_match_server import main`). Add:

```python
from tk_vision_specialized.object_match_all_server import main as _oma_main
```

If the script wraps these in a try/except, follow the existing pattern.

- [ ] **Step 2: T0 — run**

```bash
bash /home/tinker/tk25_ws/src/tk26_vision/scripts/tests/t0_static.sh
```

Expected: T0 ends with `PASS`.

- [ ] **Step 3: T1 startup — add a startup block for the new node**

Find the `object_match_server` block in `t1_startup.sh` and clone it for `object_match_all_server`. The block typically:
- launches the node in the background with `ros2 run`, capturing PID,
- waits ~3 s,
- runs `ros2 service list` and asserts `/object_match_all` appears,
- sends SIGTERM, waits, and asserts the PID is gone.

Use `DASHSCOPE_API_KEY=fake` so the Qwen provider init doesn't bail.

Add positive (`DASHSCOPE_API_KEY=fake`) and negative (key unset → RuntimeError) test variants identical to the existing `object_match_server` cases.

- [ ] **Step 4: T1 — run**

```bash
bash /home/tinker/tk25_ws/src/tk26_vision/scripts/tests/t1_startup.sh
```

Expected: T1 ends with `PASS`, with the new node startup + teardown + missing-key cases showing as passing.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add scripts/tests/t0_static.sh scripts/tests/t1_startup.sh
git commit -m "$(cat <<'EOF'
test(tk26_vision): T0/T1 coverage for object_match_all

T0 imports the new srv + node module. T1 covers startup with valid key
(advertises /object_match_all, clean SIGTERM) and refuses startup with
the API key unset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12 — Integration tier T2 + T3 extensions

**Files:**
- Modify: `src/tk26_vision/scripts/tests/t2_live.sh`
- Modify: `src/tk26_vision/scripts/tests/t3_interaction.sh`

- [ ] **Step 1: T2 — empty-scene case**

Find the existing `object_match_server` T2 invocation pattern and clone. The new block should:
- ensure cameras (RealSense + Orbbec) are running,
- `ros2 service call /object_match_all tinker_vision_msgs_26/srv/ObjectMatchAll "{camera: 'orbbec', category_filter: [], target_frame: '', sort_closest: false, sort_highest: false, return_rgb_image: false, return_depth_image: false, return_segments: false}"`,
- assert that the response includes `status: 1` and `objects: []` (empty-scene invariant — not a failure),
- assert the response includes `detection_source: vlm_match_all`.

Use the existing helper functions from t2_live.sh for service-call + response-parsing (they already exist for `object_match`). Time out the service call at 30 s to match the worst-case ceiling from the spec.

- [ ] **Step 2: T2 — run**

```bash
bash /home/tinker/tk25_ws/src/tk26_vision/scripts/tests/t2_live.sh
```

Expected: T2 ends with `PASS` (cameras must be running). If cameras are not running, the script should skip with a clear message — that's the existing behavior.

- [ ] **Step 3: T3 — response-shape parity check**

The spec's drop-in claim is at the **response shape**, not the request type. A real consumer migration requires swapping the srv import (`ObjectDetection` → `ObjectMatchAll`) plus the request-field changes. T3 verifies the response-shape parity in isolation: call both `/object_detection_yolo` and `/object_match_all` against the same scene and assert their response fields parse the same way.

Add a T3 case that:
- ensures both `yolo_seg_node` and `object_match_all_server` are running,
- captures one frame's worth of detections from each,
- asserts both responses include `header`, `status`, `error_msg`, `objects[]` (with `cls`, `conf`, `centroid`), `detection_source`, and the optional `rgb_image`/`depth_image`/`segments[]` fields when requested,
- asserts `objects[].centroid` from `/object_match_all` is parseable as `geometry_msgs/Point`.

It does **not** assert that the two services detect the same objects on the same scene — that's a T4 question.

- [ ] **Step 4: T3 — run**

```bash
bash /home/tinker/tk25_ws/src/tk26_vision/scripts/tests/t3_interaction.sh
```

Expected: T3 ends with `PASS`.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add scripts/tests/t2_live.sh scripts/tests/t3_interaction.sh
git commit -m "$(cat <<'EOF'
test(tk26_vision): T2/T3 coverage for object_match_all

T2 verifies the empty-scene invariant (status=1, objects=[]) against
live cameras. T3 retargets feature_matching at /object_match_all to
prove the response-shape drop-in claim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13 — Ground-truth generator script

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/scripts/produce_match_ground_truth.py`
- Create: `src/tk26_vision/src/tk_vision_specialized/scripts/README.md`

- [ ] **Step 1: Implement the GT generator**

Create `src/tk26_vision/src/tk_vision_specialized/scripts/produce_match_ground_truth.py`:

```python
#!/usr/bin/env python3
"""Generate ground truth for the object_match_all batch-size benchmark.

Runs the existing single-category VLM call (qwen_match_vlm.request_match_bboxes)
over every (scene, category) pair and writes the high-confidence predictions
to a JSON file that the benchmark scorer consumes.

This is "VLM ground truth," not human ground truth. It measures agreement
with the single-category /object_match service we trust in production.
See spec §8.3.1 for the rationale and caveat.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2

from tk_vision_specialized.qwen_match_vlm import request_match_bboxes
from tk_vision_specialized.items_map_loader import ItemsMapLoader
from tk_vision_specialized.nms import MatchRow, suppress_within_category


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes-dir', required=True, type=Path,
                   help='Directory containing scene_*.jpg files.')
    p.add_argument('--items-dir', required=True, type=Path,
                   help='Directory containing items_map.yaml + reference jpgs.')
    p.add_argument('--provider', default='qwen', choices=['qwen'],
                   help='Only qwen is supported here (production single-cat path).')
    p.add_argument('--vlm-model', default='qwen3-vl-plus')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--min-conf', type=float, default=0.6)
    p.add_argument('--timeout-s', type=float, default=12.0)
    p.add_argument('--out', required=True, type=Path,
                   help='Path to write the GT JSON.')
    return p.parse_args()


def main():
    args = _parse_args()
    items = ItemsMapLoader(str(args.items_dir))
    if len(items) == 0:
        print(f'No items found in {args.items_dir}', file=sys.stderr)
        return 1

    scenes = sorted(args.scenes_dir.glob('*.jpg')) + sorted(args.scenes_dir.glob('*.png'))
    if not scenes:
        print(f'No scenes found in {args.scenes_dir}', file=sys.stderr)
        return 1

    out: dict = {
        '_meta': {
            'provider': args.provider,
            'vlm_model': args.vlm_model,
            'top_k': args.top_k,
            'min_conf': args.min_conf,
            'items': sorted(items.keys()),
            'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
        },
    }

    for scene_path in scenes:
        rgb = cv2.imread(str(scene_path))
        if rgb is None:
            print(f'skip unreadable scene {scene_path}', file=sys.stderr)
            continue
        scene_gt: list[MatchRow] = []
        for category in items.keys():
            ref_url = items.get_data_url(category)
            boxes, confs, _labels, _elapsed = request_match_bboxes(
                rgb, ref_url, item_name=category, top_k=args.top_k,
                timeout_s=args.timeout_s, max_retries=1,
            )
            for bbox, conf in zip(boxes, confs):
                if conf >= args.min_conf:
                    scene_gt.append(MatchRow(label=category, bbox=tuple(bbox), conf=conf))
        scene_gt = suppress_within_category(scene_gt, iou_thresh=0.5)
        out[scene_path.name] = [
            {'category': r.label, 'bbox': list(r.bbox), 'conf': r.conf}
            for r in scene_gt
        ]
        print(f'{scene_path.name}: {len(scene_gt)} GT items', file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f'wrote {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Create the scripts README**

Create `src/tk26_vision/src/tk_vision_specialized/scripts/README.md`:

````markdown
# Benchmark scripts for `object_match_all`

Two sibling scripts:

## `produce_match_ground_truth.py`

Generates ground-truth detections for a directory of scenes by running the
single-category VLM (today's `/object_match` production path) for every
`(scene, category)` pair. The output JSON is what
`benchmark_match_batch_size.py` scores against.

This is **VLM ground truth**, not human ground truth — it measures
agreement with the single-category service we already trust. See the
design doc for the rationale (spec §8.3.1).

### Usage

```bash
source /opt/ros/humble/setup.bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
source /home/tinker/tk25_ws/src/tk26_vision/install/setup.bash

python -m tk_vision_specialized.scripts.produce_match_ground_truth \
    --scenes-dir /path/to/scenes \
    --items-dir  /home/tinker/tk25_ws/src/tk26_vision/src/items \
    --out        /tmp/gt_$(date +%Y%m%d_%H%M%S).json
```

Cost: `N_scenes * N_categories` single-category calls (~$1–3 per
regeneration on the default 10-scene × 10-item dataset).

Manual edits: the JSON is a plain dict; correct known-bad single-cat
predictions by editing the file before running the benchmark.

## `benchmark_match_batch_size.py`

Sweeps `batch_size` against the GT JSON and reports
precision/recall/F1/latency/token-cost per (provider, batch_size).

### Usage

```bash
python -m tk_vision_specialized.scripts.benchmark_match_batch_size \
    --scenes-dir /path/to/scenes \
    --items-dir  /home/tinker/tk25_ws/src/tk26_vision/src/items \
    --ground-truth /tmp/gt_TIMESTAMP.json \
    --batch-sizes 1 2 3 5 8 \
    --provider qwen \
    --repeats 3 \
    --out-prefix /tmp/bench_$(date +%Y%m%d_%H%M%S)
```

Output: a CSV (one row per `(scene, provider, batch_size, repeat)`) plus a
Markdown summary with the recommended `batch_size` default.

The recommendation is **advisory**. Update the `batch_size` ROS parameter in
your launch params after reviewing the summary. The choice doesn't
auto-update.

### When to re-run

- Items added to or removed from `items_map.yaml`.
- VLM provider switch.
- Accuracy regression observed in T3/T4 integration tests.
````

- [ ] **Step 3: Smoke-test the GT script (offline; no API key)**

The script will exit at the first VLM call without credentials. Verify the help and arg parsing:

```bash
source /home/tinker/tk25_ws/src/tk26_vision/install/setup.bash
python -m tk_vision_specialized.scripts.produce_match_ground_truth --help
```

Expected: help text prints, no traceback.

- [ ] **Step 4: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/scripts/produce_match_ground_truth.py src/tk_vision_specialized/scripts/README.md
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): ground-truth generator for batch-size benchmark

Single-category VLM call over every (scene, category) pair, NMS-collapsed,
written to a JSON the benchmark scorer consumes. Documented caveats:
this is "VLM ground truth" not human ground truth.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14 — Batch-size benchmark script

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/scripts/benchmark_match_batch_size.py`

- [ ] **Step 1: Implement the benchmark**

Create `src/tk26_vision/src/tk_vision_specialized/scripts/benchmark_match_batch_size.py`:

```python
#!/usr/bin/env python3
"""Sweep batch_size for object_match_all and report precision/recall/F1/latency.

Reads scenes + GT JSON, runs the configured provider's MatchClient with each
batch_size, scores against GT, writes a CSV and Markdown summary."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

from tk_vision_specialized.items_map_loader import ItemsMapLoader
from tk_vision_specialized.nms import MatchRow, iou
from tk_vision_specialized.vlm_match_client import build_match_client


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes-dir', required=True, type=Path)
    p.add_argument('--items-dir', required=True, type=Path)
    p.add_argument('--ground-truth', required=True, type=Path)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    p.add_argument('--provider', default='qwen', choices=['qwen', 'gemini', 'both'])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out-prefix', required=True, type=Path)
    p.add_argument('--timeout-s', type=float, default=12.0)
    p.add_argument('--iou-thresh', type=float, default=0.3,
                   help='IoU threshold for TP scoring.')
    return p.parse_args()


def _score(predictions, ground_truth, iou_thresh):
    """Standard set-matching: a prediction is TP if it has a same-label GT box
    with IoU >= thresh that hasn't been matched yet. Multi-prediction-to-one-
    GT is greedy by descending confidence."""
    preds = sorted(predictions, key=lambda r: -r.conf)
    gt_remaining = list(ground_truth)
    tp = 0
    for p in preds:
        for i, g in enumerate(gt_remaining):
            if g.label != p.label:
                continue
            if iou(p.bbox, g.bbox) >= iou_thresh:
                tp += 1
                gt_remaining.pop(i)
                break
    fp = len(preds) - tp
    fn = len(gt_remaining)
    return tp, fp, fn


def _provider_list(arg):
    if arg == 'both':
        return ['qwen', 'gemini']
    return [arg]


def main():
    args = _parse_args()
    items = ItemsMapLoader(str(args.items_dir))
    gt_raw = json.loads(args.ground_truth.read_text())
    meta = gt_raw.get('_meta', {})
    gt_items = set(meta.get('items', []))
    if gt_items and gt_items != set(items.keys()):
        print(f'GT items {sorted(gt_items)} differ from current items_map '
              f'{sorted(items.keys())}; regenerate GT.', file=sys.stderr)
        return 1

    refs_all = [(k, items.get_data_url(k)) for k in items.keys()]
    scenes = sorted(p for p in args.scenes_dir.iterdir()
                    if p.suffix.lower() in {'.jpg', '.png'})

    rows: list[dict] = []

    for provider in _provider_list(args.provider):
        client = build_match_client(provider)
        for B in args.batch_sizes:
            for scene_path in scenes:
                rgb = cv2.imread(str(scene_path))
                if rgb is None:
                    continue
                gt = [
                    MatchRow(label=e['category'], bbox=tuple(e['bbox']),
                             conf=float(e['conf']))
                    for e in gt_raw.get(scene_path.name, [])
                ]
                for r in range(args.repeats):
                    batches = [refs_all[i:i + B] for i in range(0, len(refs_all), B)]
                    t0 = time.perf_counter()
                    preds: list[MatchRow] = []
                    for batch in batches:
                        try:
                            preds.extend(client.match_batch(
                                rgb, batch,
                                timeout_s=args.timeout_s, max_retries=1,
                            ))
                        except Exception as exc:    # noqa: BLE001
                            print(f'batch fail provider={provider} B={B} '
                                  f'scene={scene_path.name}: {exc}',
                                  file=sys.stderr)
                    elapsed = time.perf_counter() - t0
                    tp, fp, fn = _score(preds, gt, args.iou_thresh)
                    rows.append({
                        'scene': scene_path.name,
                        'provider': provider,
                        'batch_size': B,
                        'repeat': r,
                        'n_calls': len(batches),
                        'elapsed_s': elapsed,
                        'tp': tp, 'fp': fp, 'fn': fn,
                        'n_pred': len(preds), 'n_gt': len(gt),
                    })
                    print(f'  {scene_path.name} provider={provider} B={B} r={r} '
                          f'tp={tp} fp={fp} fn={fn} {elapsed:.1f}s')

    csv_path = args.out_prefix.with_suffix('.csv')
    md_path = args.out_prefix.with_suffix('.md')
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open('w', newline='') as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Summary
    summary: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row['provider'], row['batch_size'])
        s = summary.setdefault(key, {'f1s': [], 'lats': [], 'precs': [], 'recs': []})
        prec = row['tp'] / max(1, row['tp'] + row['fp'])
        rec = row['tp'] / max(1, row['tp'] + row['fn'])
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        s['precs'].append(prec); s['recs'].append(rec); s['f1s'].append(f1)
        s['lats'].append(row['elapsed_s'])

    lines = ['# Batch-size benchmark summary',
             f'GT: {args.ground_truth.name}',
             f'Generated: {datetime.datetime.utcnow().isoformat()}Z',
             '',
             '| provider | batch_size | median F1 | median latency (s) | p95 latency (s) |',
             '|---|---|---|---|---|']
    for (provider, B), s in sorted(summary.items()):
        f1 = statistics.median(s['f1s'])
        lat_med = statistics.median(s['lats'])
        lat_p95 = sorted(s['lats'])[max(0, int(len(s['lats']) * 0.95) - 1)]
        lines.append(f'| {provider} | {B} | {f1:.3f} | {lat_med:.2f} | {lat_p95:.2f} |')

    # Recommendation per provider
    lines.append('')
    lines.append('## Recommended batch_size')
    for provider in _provider_list(args.provider):
        candidates = [(B, summary[(provider, B)]) for B in args.batch_sizes
                      if (provider, B) in summary]
        if not candidates:
            continue
        best_B, best_s = max(
            candidates,
            key=lambda kv: (statistics.median(kv[1]['f1s']),
                            -statistics.median(kv[1]['lats'])),
        )
        lines.append(
            f'- **{provider}**: `batch_size = {best_B}` '
            f'(median F1 {statistics.median(best_s["f1s"]):.3f}, '
            f'median latency {statistics.median(best_s["lats"]):.2f}s)'
        )

    md_path.write_text('\n'.join(lines))
    print(f'wrote {csv_path} and {md_path}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test arg parsing**

```bash
python -m tk_vision_specialized.scripts.benchmark_match_batch_size --help
```

Expected: help text prints.

- [ ] **Step 3: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/scripts/benchmark_match_batch_size.py
git commit -m "$(cat <<'EOF'
feat(tk_vision_specialized): batch-size benchmark for object_match_all

Sweeps batch_size against the GT JSON for a given provider; writes a CSV
plus a Markdown summary with the recommended batch_size per provider.
Recommendation is advisory: update the ROS parameter manually after
reviewing the summary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15 — Documentation updates

**Files:**
- Modify: `src/tk26_vision/CLAUDE.md`
- Modify: `src/tk26_vision/DEV_NOTES.md`
- Modify: `src/tk26_vision/README.md` (optional — only if there's a service-list section)

- [ ] **Step 1: Update `CLAUDE.md`**

In `src/tk26_vision/CLAUDE.md`, find the "Running Nodes" section's `tk_vision_specialized` block and add:

```bash
ros2 run tk_vision_specialized object_match_all_server   # /object_match_all (concurrent VLM scan over items_map)
```

In the "Architecture" tree under `tk_vision_specialized/`, add a line:

```
│   ├── object_match_all_server.py  # concurrent VLM matcher across all items_map entries (drop-in for /object_detection_yolo response shape)
```

In the "Configuration" section's per-package param list, add a `object_match_all` entry summarizing the key params (link to the spec for full detail):

```
- `object_match_all_server`: full param surface in `docs/superpowers/specs/2026-05-27-object-match-all-design.md`. Key knobs: `vlm_provider` (qwen|gemini), `judge_provider` (empty=inherit), `batch_size` (default 3, set from `scripts/benchmark_match_batch_size.py`), `stage1_timeout_s`/`stage2_timeout_s` (15s/10s), `cluster_iou` (0.5), `judge_crop_margin_px` (20).
```

- [ ] **Step 2: Update `DEV_NOTES.md`**

Append a new section under "Follow-ups" or in an appropriate "Verification history" section:

```markdown
## 2026-05-27: object_match_all node added

A new service `/object_match_all` answers the dual question to /object_match:
"given the items_map, where is each item in the camera frame?" Concurrent
batched VLM calls with per-conflict VLM-judge resolution, response shape
identical to ObjectDetection.srv.

- Spec: `docs/superpowers/specs/2026-05-27-object-match-all-design.md`
- Plan: `docs/superpowers/plans/2026-05-27-object-match-all.md`
- Scripts: `src/tk_vision_specialized/scripts/produce_match_ground_truth.py`
  + `benchmark_match_batch_size.py` (run before relying on the `batch_size`
  default).

Open items operator-side:
- Capture a 10-scene benchmark set and regenerate GT + sweep to pin
  `batch_size`.
- T4 hardware pass against `shelf_scene` to compare detection quality with
  /object_detection_yolo on the same scene.
```

- [ ] **Step 3: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add CLAUDE.md DEV_NOTES.md
git commit -m "$(cat <<'EOF'
docs(tk26_vision): mention object_match_all in CLAUDE.md + DEV_NOTES

Architecture tree, run commands, key params, and the follow-up checklist
for the operator-side benchmark + T4 pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done criteria

- 15 tasks committed on `dev`, each with passing tests or smoke output as described.
- New unit tests pass under `pytest src/tk_vision_specialized/test/ -v` (totals: 9 + 6 in `test_nms.py` + 4 in `test_vlm_common.py` + 12 + 5 in `test_vlm_match_client.py` + 9 in `test_vlm_judge_client.py` + 9 in `test_match_pipeline.py` = **54 tests**).
- Integration tiers T0–T3 pass with the new node included.
- `ros2 service list` shows `/object_match_all` after launching `object_match_all_server`.
- `feature_matching` retargeted at `/object_match_all` via param produces the same response shape and a meaningful result on a staged scene.
- The benchmark + GT scripts execute their `--help` and run end-to-end against a small captured scene set (operator-driven; not in CI).
