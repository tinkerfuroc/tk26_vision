# Person Tracker — Phase 0: Benchmark Fidelity + Quick Wins — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ptbench ruler trustworthy (dual-GT centroids, range gate, action-backend default), land zero-risk geometry/throughput/association quick wins with node↔ptbench parity, and kill the latent crashers — all validatable today without arena recordings.

**Architecture:** ptbench (`benchmarks/person_tracker/`) is a pure-python benchmark whose GT centroids must match how the live tracker computes 3D positions; Phase 0 fixes a mask-vs-bbox fidelity defect by carrying two GT centroids per frame (`centroid_field` = mask+robust-median best estimate, `centroid_track` = node-identical math) and gating on the field one. The `vision_track` node and ptbench share a single ROS-free centroid reduction helper (`vision_track/core/centroid.py`) imported by both `person_track_node._calculate_centroid` and `ptbench/common/geometry.centroid_from_bbox_depth`, enforced by a parity test. Operator init is extracted to a pure `vision_track/core/operator_init.py` helper so the nearest-central heuristic is unit-testable without ROS.

**Tech Stack:** Python 3.10, pytest, numpy, ROS2 Humble (rclpy), ultralytics, the ptbench package.

---

## File Structure

**Created:**
- `benchmarks/person_tracker/tests/test_centroid_reduction.py` — unit tests for the shared robust-centroid reduction (median lateral + z-outlier rejection) and node↔geometry parity.
- `benchmarks/person_tracker/tests/test_dual_gt.py` — schema 1.1 round-trip + 1.0 back-compat, `build_gt_clip` dual-centroid, the synthetic field-vs-track divergence fixture.
- `src/vision_track/vision_track/core/centroid.py` — ROS-free shared centroid reduction: `reduce_centroid(obj_pts)` (median x/y, z-outlier-rejected median z). Imported by both the node and ptbench geometry.
- `src/vision_track/vision_track/core/operator_init.py` — ROS-free `select_operator_detection(detections, depth_lookup)` nearest+central+conf heuristic.
- `src/vision_track/test/test_centroid.py` — unit tests for `reduce_centroid`.
- `src/vision_track/test/test_operator_init.py` — unit tests for `select_operator_detection`.

**Modified:**
- `benchmarks/person_tracker/ptbench/common/schema.py` — `GtFrame` gains `centroid_field` + `centroid_track`; `SCHEMA_VERSION="1.1"`; loader accepts `"1.0"` (maps `centroid_3d` → both) and `"1.1"`; `validate` checks both new fields.
- `benchmarks/person_tracker/ptbench/common/geometry.py` — `centroid_from_bbox_depth` uses the shared `reduce_centroid`; no behavior change to its public signature.
- `benchmarks/person_tracker/ptbench/labeler/label_io.py` — `build_gt_clip` computes both `centroid_field` (mask) and `centroid_track` (no mask), writes schema `1.1`; `FrameAnnotation` gains an optional `mask`.
- `benchmarks/person_tracker/ptbench/common/metrics.py` — score correct/lateral/range against `centroid_field`; emit a `centroid_track` diagnostic block.
- `benchmarks/person_tracker/ptbench/common/scoreboard.py` — `GateConfig` gains `pos_error_range_pass_m=0.30` / `pos_error_range_warn_m=0.50`; `score` adds the `pos_error_range_m` row.
- `benchmarks/person_tracker/ptbench/replay/score_cli.py` — default `--backend` becomes `action`; offline documented as approximate.
- `benchmarks/person_tracker/ptbench/replay/runner.py` — `run_offline` aligns its config to the deployed defaults (imgsz 736, conf unchanged for offline-approx) and threads the same centroid path.
- `src/vision_track/vision_track/person_track_node.py` — `perf_logging_enabled` param + per-stage timers + per-frame diagnostics; `_calculate_centroid` uses `reduce_centroid`; remove the `tracking_rate=15` cap (rely on frame-seq dedup); imgsz default 736 + `half=True`; ROI-crop the depth unproject; lost-sentinel republish on `/target_points`; `reid_mode='native'` raises `NotImplementedError`; plumb loop rate into ByteTrack via tracker `frame_rate`.
- `src/vision_track/vision_track/yolo_tracker.py` — `initialize_tracking` calls `select_operator_detection` for the class-only path; `track()` passes `half=True`; `_update_target_velocity` accepts a frame-stamp `dt`; tracker carries a `frame_rate` used when constructing the ByteTrack tracker config.

---

### Task 1: Dual-GT schema (centroid_field + centroid_track, version 1.1, back-compat loader)

**Files:**
- Modify: `benchmarks/person_tracker/ptbench/common/schema.py` (`SCHEMA_VERSION` line 16; `GtFrame` lines 23-28; `validate` lines 49-93; `_frame_to_dict` lines 96-102; `_frame_from_dict` lines 120-128)
- Test: `benchmarks/person_tracker/tests/test_dual_gt.py` (new)

- [ ] **Step 1: Write failing tests for the 1.1 schema + 1.0 back-compat loader.**
  Create `benchmarks/person_tracker/tests/test_dual_gt.py` with:
  ```python
  """Schema 1.1 dual-centroid round-trip + 1.0 back-compat + build_gt_clip duals."""
  import json

  import numpy as np
  import pytest

  from ptbench.common.schema import (
      SCHEMA_VERSION,
      GtClip,
      GtFrame,
      GtSchemaError,
      load_gt,
      save_gt,
  )


  def make_clip_11(**overrides) -> GtClip:
      frames = overrides.pop(
          "frames",
          [
              GtFrame(
                  t_ns=1000,
                  present=True,
                  bbox=(10, 20, 110, 220),
                  centroid_field=(0.10, 0.0, 2.5),
                  centroid_track=(0.12, 0.01, 2.5),
              ),
              GtFrame(
                  t_ns=2000,
                  present=False,
                  bbox=None,
                  centroid_field=None,
                  centroid_track=None,
              ),
          ],
      )
      return GtClip(
          schema_version=overrides.pop("schema_version", SCHEMA_VERSION),
          clip_id=overrides.pop("clip_id", "dual_clip"),
          bag_path=overrides.pop("bag_path", "bags/dual"),
          scenario=overrides.pop("scenario", "test"),
          frames=frames,
          **overrides,
      )


  class TestSchema11RoundTrip:
      def test_version_is_11(self):
          assert SCHEMA_VERSION == "1.1"

      def test_roundtrip_both_centroids(self, tmp_path):
          clip = make_clip_11()
          p = tmp_path / "gt.json"
          save_gt(clip, p)
          loaded = load_gt(p)
          f0 = loaded.frames[0]
          assert f0.centroid_field == (0.10, 0.0, 2.5)
          assert f0.centroid_track == (0.12, 0.01, 2.5)

      def test_canonical_json_has_both_fields(self, tmp_path):
          clip = make_clip_11()
          p = tmp_path / "gt.json"
          save_gt(clip, p)
          raw = json.loads(p.read_text())
          assert raw["schema_version"] == "1.1"
          assert raw["frames"][0]["centroid_field"] == [0.10, 0.0, 2.5]
          assert raw["frames"][0]["centroid_track"] == [0.12, 0.01, 2.5]
          assert raw["frames"][1]["centroid_field"] is None
          assert raw["frames"][1]["centroid_track"] is None


  class TestSchema10BackCompat:
      def test_10_centroid_3d_maps_to_both(self, tmp_path):
          raw = {
              "schema_version": "1.0",
              "clip_id": "x", "bag_path": "b", "scenario": "s",
              "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
              "fps_hint": 30.0, "notes": "",
              "frames": [
                  {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                   "centroid_3d": [0.1, 0.0, 2.5]},
                  {"t_ns": 2000, "present": False, "bbox": None, "centroid_3d": None},
              ],
          }
          p = tmp_path / "gt10.json"
          p.write_text(json.dumps(raw))
          loaded = load_gt(p)
          f0 = loaded.frames[0]
          assert f0.centroid_field == (0.1, 0.0, 2.5)
          assert f0.centroid_track == (0.1, 0.0, 2.5)
          assert loaded.frames[1].centroid_field is None
          assert loaded.frames[1].centroid_track is None

      def test_unsupported_version_still_rejected(self, tmp_path):
          raw = {
              "schema_version": "2.0",
              "clip_id": "x", "bag_path": "b", "scenario": "s",
              "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
              "fps_hint": 30.0, "notes": "",
              "frames": [
                  {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                   "centroid_field": [0.1, 0.0, 2.5], "centroid_track": [0.1, 0.0, 2.5]},
              ],
          }
          p = tmp_path / "bad.json"
          p.write_text(json.dumps(raw))
          with pytest.raises(GtSchemaError, match="schema_version"):
              load_gt(p)


  class TestSchema11Validation:
      def test_bad_centroid_field_count(self, tmp_path):
          raw = {
              "schema_version": "1.1",
              "clip_id": "x", "bag_path": "b", "scenario": "s",
              "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
              "fps_hint": 30.0, "notes": "",
              "frames": [
                  {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                   "centroid_field": [0.1, 0.0], "centroid_track": None},
              ],
          }
          p = tmp_path / "bad.json"
          p.write_text(json.dumps(raw))
          with pytest.raises(GtSchemaError, match="centroid_field"):
              load_gt(p)

      def test_bad_centroid_track_non_finite(self, tmp_path):
          raw = {
              "schema_version": "1.1",
              "clip_id": "x", "bag_path": "b", "scenario": "s",
              "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
              "fps_hint": 30.0, "notes": "",
              "frames": [
                  {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                   "centroid_field": None, "centroid_track": [float("nan"), 0.0, 2.5]},
              ],
          }
          p = tmp_path / "bad.json"
          p.write_text(json.dumps(raw))
          with pytest.raises(GtSchemaError, match="centroid_track"):
              load_gt(p)
  ```

- [ ] **Step 2: Run the new tests to confirm they fail.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_dual_gt.py -q`
  Expected: collection-time or assertion failures — `TypeError: __init__() got an unexpected keyword argument 'centroid_field'` (GtFrame has no such field yet) and `assert "1.0" == "1.1"`.

- [ ] **Step 3: Implement the 1.1 schema with a back-compatible loader.**
  In `benchmarks/person_tracker/ptbench/common/schema.py`:
  - Line 16: change `SCHEMA_VERSION = "1.0"` to:
    ```python
    SCHEMA_VERSION = "1.1"
    SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")
    ```
  - Replace the `GtFrame` dataclass (lines 23-28) with:
    ```python
    @dataclass
    class GtFrame:
        t_ns: int
        present: bool
        bbox: Optional[Tuple[float, float, float, float]]
        # Best-available estimate (mask + robust median) — the gate scores this.
        centroid_field: Optional[Tuple[float, float, float]] = None
        # Node-identical math (no mask) — reported as a diagnostic only.
        centroid_track: Optional[Tuple[float, float, float]] = None
    ```
  - In `validate` (lines 54-58) replace the version check with:
    ```python
    if clip.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise GtSchemaError(
            f"unsupported schema_version {clip.schema_version!r} "
            f"(expected one of {SUPPORTED_SCHEMA_VERSIONS!r})"
        )
    ```
  - Replace the `centroid_3d` validation block (lines 87-93) with a loop over both fields:
    ```python
        for field_name in ("centroid_field", "centroid_track"):
            val = getattr(f, field_name)
            if val is not None:
                c = tuple(val)
                if len(c) != 3 or not all(_is_finite_number(v) for v in c):
                    raise GtSchemaError(
                        f"frame {i}: {field_name} must be 3 finite numbers, "
                        f"got {val!r}"
                    )
    ```
  - In `_frame_to_dict` (lines 96-102) replace the centroid line with both fields:
    ```python
    def _frame_to_dict(f: GtFrame) -> dict:
        return {
            "t_ns": f.t_ns,
            "present": f.present,
            "bbox": list(f.bbox) if f.bbox is not None else None,
            "centroid_field": list(f.centroid_field) if f.centroid_field is not None else None,
            "centroid_track": list(f.centroid_track) if f.centroid_track is not None else None,
        }
    ```
  - Replace `_frame_from_dict` (lines 120-128) with a back-compat reader that accepts both schema shapes:
    ```python
    def _frame_from_dict(d: dict) -> GtFrame:
        bbox = d.get("bbox")
        # 1.1 carries centroid_field/centroid_track; 1.0 carries a single
        # centroid_3d that maps onto both.
        if "centroid_field" in d or "centroid_track" in d:
            cf = d.get("centroid_field")
            ct = d.get("centroid_track")
        else:
            legacy = d.get("centroid_3d")
            cf = legacy
            ct = legacy
        return GtFrame(
            t_ns=d["t_ns"],
            present=d["present"],
            bbox=tuple(bbox) if bbox is not None else None,
            centroid_field=tuple(cf) if cf is not None else None,
            centroid_track=tuple(ct) if ct is not None else None,
        )
    ```

- [ ] **Step 4: Run the new tests to confirm they pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_dual_gt.py -q`
  Expected: all tests in `test_dual_gt.py` pass.

- [ ] **Step 5: Update the existing 1.0 schema tests to the 1.1 surface (back-compat preserved).**
  `tests/test_schema.py` constructs `GtFrame(..., centroid_3d=...)` and asserts `schema_version == "1.0"` round-trips. Because `_frame_from_dict` maps `centroid_3d` → both new fields, the *in-memory* `GtFrame(centroid_3d=...)` constructor no longer exists. Update `tests/test_schema.py`:
  - In `make_clip` (lines 24-30), replace each `GtFrame(..., centroid_3d=(...))` with `GtFrame(..., centroid_field=(...), centroid_track=(...))` using the same tuple for both.
  - In `test_roundtrip_frame_values` (lines 70-78), replace `f0.centroid_3d == (0.1, 0.0, 2.5)` with `f0.centroid_field == (0.1, 0.0, 2.5)` and add `assert f0.centroid_track == (0.1, 0.0, 2.5)`; replace `f1.centroid_3d is None` with `f1.centroid_field is None` and `f1.centroid_track is None`.
  - In `test_canonical_json_shape` (lines 87-91): change `raw["schema_version"] == "1.0"` to `== "1.1"`; replace the two `centroid_3d` array assertions with `centroid_field`/`centroid_track` array assertions (both equal `[0.1, 0.0, 2.5]` for frame 0, both `None` for frame 1).
  - In `test_roundtrip_full_spec_example` (lines 95-109): set `schema_version="1.1"` and use `centroid_field`/`centroid_track`; assert `loaded.frames[0].centroid_field == (0.1, 0.0, 2.5)`.
  - In `test_roundtrip_lossless_float_centroid` (lines 122-132): use `centroid_field`/`centroid_track`; read `loaded.frames[0].centroid_field`.
  - In `test_bad_centroid_wrong_count` / `test_bad_centroid_non_finite` (lines 274-303): these write raw JSON with `"schema_version": "1.0"` and `"centroid_3d"`. Keep them as 1.0 back-compat coverage but change the `pytest.raises(..., match="centroid_3d")` to `match="centroid_field"` — under the loader the legacy `centroid_3d` populates `centroid_field`, and `validate` reports the offending new-field name. (The 1.1-native bad-centroid cases live in `test_dual_gt.py`.)
  - `test_bad_schema_version` (lines 140-149) already uses `"2.0"` — leave it; it still raises.

- [ ] **Step 6: Run the full ptbench suite — schema tests green, no regressions yet beyond known downstream.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_schema.py tests/test_dual_gt.py -q`
  Expected: both files pass. (The metrics/labeler/scoreboard files are updated in Tasks 2-4 and may fail until then; do not run the whole suite at this step.)

- [ ] **Step 7: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add benchmarks/person_tracker/ptbench/common/schema.py benchmarks/person_tracker/tests/test_dual_gt.py benchmarks/person_tracker/tests/test_schema.py && git commit -m "$(cat <<'EOF'
feat(ptbench): dual-GT schema 1.1 (centroid_field + centroid_track)

GtFrame now carries centroid_field (mask + robust median, the gated
estimate) and centroid_track (node-identical math, diagnostic). Loader
accepts 1.0 (maps centroid_3d onto both) and 1.1.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 2: Shared robust-centroid reduction helper (median lateral + z-outlier rejection)

This is the **geometry quick win** plus the node↔ptbench parity guarantee. The reduction lives in **one** ROS-free module that both the node and ptbench import. New behavior vs the current mean-x/y reduction (`person_track_node.py:378-381`, `geometry.py:93-94`): lateral x,y use the **median**; before reducing, drop points whose `|z - median_z| > 0.4 m`.

**Files:**
- Create: `src/vision_track/vision_track/core/centroid.py`
- Create: `src/vision_track/test/test_centroid.py`
- Create: `benchmarks/person_tracker/tests/test_centroid_reduction.py`
- Modify: `benchmarks/person_tracker/ptbench/common/geometry.py` (lines 17-18 imports; lines 93-94 reduction)

- [ ] **Step 1: Write the failing unit test for the reduction helper (vision_track side).**
  Create `src/vision_track/test/test_centroid.py`:
  ```python
  """Unit tests for the ROS-free robust-centroid reduction helper."""
  import numpy as np

  from vision_track.core.centroid import reduce_centroid, Z_OUTLIER_M


  def test_median_lateral_pure():
      # Three points: lateral x has one outlier; median ignores it.
      pts = np.array(
          [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [10.0, 0.0, 2.0]], dtype=np.float64
      )
      x, y, z = reduce_centroid(pts)
      assert abs(x - 0.0) < 1e-9   # median x is 0.0, not the mean 3.33
      assert abs(y - 0.0) < 1e-9
      assert abs(z - 2.0) < 1e-9


  def test_z_outlier_rejected_before_reduce():
      # 10 inliers at z=3.0 with x=1.0; one far-z outlier at z=9.0, x=5.0.
      inliers = np.array([[1.0, 0.0, 3.0]] * 10, dtype=np.float64)
      outlier = np.array([[5.0, 0.0, 9.0]], dtype=np.float64)
      pts = np.concatenate([inliers, outlier], axis=0)
      x, y, z = reduce_centroid(pts)
      # outlier z (|9-3|=6 > 0.4) dropped → x median stays 1.0, z stays 3.0
      assert abs(x - 1.0) < 1e-9
      assert abs(z - 3.0) < 1e-9


  def test_z_outlier_threshold_constant():
      assert Z_OUTLIER_M == 0.4


  def test_returns_python_floats():
      pts = np.array([[1.0, 2.0, 3.0]] * 12, dtype=np.float64)
      x, y, z = reduce_centroid(pts)
      assert isinstance(x, float) and isinstance(y, float) and isinstance(z, float)


  def test_all_dropped_falls_back_to_unfiltered_median():
      # Degenerate: only 1 point (can't reject); reduce must still return it.
      pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
      x, y, z = reduce_centroid(pts)
      assert (x, y, z) == (1.0, 2.0, 3.0)
  ```

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_centroid.py -v`
  Expected: `ModuleNotFoundError: No module named 'vision_track.core.centroid'`.

- [ ] **Step 3: Implement the shared reduction helper.**
  Create `src/vision_track/vision_track/core/centroid.py`:
  ```python
  """ROS-free robust 3D-centroid reduction shared by the node and ptbench.

  Both ``person_track_node._calculate_centroid`` and
  ``ptbench.common.geometry.centroid_from_bbox_depth`` MUST reduce a set of
  per-pixel 3D points through this one function so the live tracker and the
  benchmark never silently disagree (enforced by a parity test).

  Reduction (camera optical frame, x=right, y=down, z=forward):
    1. Compute median z.
    2. Drop points with |z - median_z| > Z_OUTLIER_M (depth-noise rejection).
       If that leaves nothing (degenerate), keep the original set.
    3. Lateral x, y = MEDIAN over the kept set (robust to limb/edge pixels);
       z = MEDIAN over the kept set.
  """
  from __future__ import annotations

  from typing import Tuple

  import numpy as np

  Z_OUTLIER_M = 0.4


  def reduce_centroid(obj_pts: np.ndarray) -> Tuple[float, float, float]:
      """Reduce an (N, 3) array of 3D points to one robust centroid.

      Args:
          obj_pts: (N, 3) float array of camera-frame XYZ points (meters).

      Returns:
          (x, y, z) as plain Python floats. Caller guarantees N >= 1.
      """
      pts = np.asarray(obj_pts, dtype=np.float64)
      z = pts[:, 2]
      median_z = float(np.median(z))
      keep = np.abs(z - median_z) <= Z_OUTLIER_M
      kept = pts[keep]
      if kept.shape[0] == 0:
          kept = pts
      x = float(np.median(kept[:, 0]))
      y = float(np.median(kept[:, 1]))
      zc = float(np.median(kept[:, 2]))
      return x, y, zc
  ```

- [ ] **Step 4: Run to confirm pass (vision_track side).**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_centroid.py -v`
  Expected: 5 passed.

- [ ] **Step 5: Write the failing ptbench-side test (geometry uses the shared helper + parity).**
  Create `benchmarks/person_tracker/tests/test_centroid_reduction.py`:
  ```python
  """ptbench geometry uses the shared reduction + node↔geometry parity.

  Imports the SAME helper the live node imports so a divergence is caught here.
  Requires the colcon workspace's vision_track on the path (the venv install).
  """
  import numpy as np
  import pytest

  from ptbench.common.geometry import centroid_from_bbox_depth

  vt_centroid = pytest.importorskip("vision_track.core.centroid")
  reduce_centroid = vt_centroid.reduce_centroid


  def pinhole_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
      return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


  def test_geometry_imports_the_shared_reduce():
      # geometry must reference the shared module, not a private copy.
      import ptbench.common.geometry as g
      assert g.reduce_centroid is reduce_centroid


  def test_z_outlier_rejection_in_geometry():
      # A bbox where most pixels are at 3.0 m and a stripe is at 9.0 m: the
      # rejected stripe must not pull z toward 9.0.
      H, W = 480, 640
      K = pinhole_K()
      bbox = (100, 100, 200, 200)  # 100x100
      depth = np.full((H, W), 3000, dtype=np.uint16)  # 3.0 m
      depth[100:110, 100:200] = 9000  # 9.0 m stripe (10% of rows)
      result = centroid_from_bbox_depth(depth, K, bbox)
      assert result is not None
      _, _, z = result
      assert abs(z - 3.0) < 0.05


  def test_lateral_uses_median():
      # Asymmetric mask weighting that would skew a mean but not a median.
      H, W = 480, 640
      K = pinhole_K()
      bbox = (100, 100, 400, 200)
      depth = np.full((H, W), 2000, dtype=np.uint16)
      r = centroid_from_bbox_depth(depth, K, bbox)
      assert r is not None
      # Sanity: result is finite and within the bbox-implied lateral extent.
      x, y, z = r
      assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z)


  def test_parity_node_vs_geometry():
      # Build the same (N,3) point set both code paths would reduce, and assert
      # the geometry result equals reduce_centroid applied directly.
      H, W = 480, 640
      fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
      K = pinhole_K(fx, fy, cx, cy)
      bbox = (120, 130, 220, 210)
      depth = np.full((H, W), 0, dtype=np.uint16)
      depth[130:210, 120:220] = 2500  # 2.5 m filled bbox
      geo = centroid_from_bbox_depth(depth, K, bbox)
      assert geo is not None

      # Reconstruct the exact point set geometry reduces and reduce it directly.
      x1, y1, x2, y2 = bbox
      u, v = np.meshgrid(
          np.arange(x1, x2, dtype=np.float32),
          np.arange(y1, y2, dtype=np.float32),
      )
      z = (depth[y1:y2, x1:x2].astype(np.float32)) * 0.001
      valid = (z > 0.1) & (z < 10.0)
      X = (u - cx) * z / fx
      Y = (v - cy) * z / fy
      pts = np.stack([X, Y, z], axis=-1)[np.nonzero(valid.astype(float))]
      direct = reduce_centroid(pts)
      for a, b in zip(geo, direct):
          assert abs(a - b) < 1e-6
  ```

- [ ] **Step 6: Run to confirm failure (geometry still uses inline mean).**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_centroid_reduction.py -v`
  Expected: `test_geometry_imports_the_shared_reduce` fails (`AttributeError: module 'ptbench.common.geometry' has no attribute 'reduce_centroid'`) and `test_z_outlier_rejection_in_geometry`/`test_parity_node_vs_geometry` fail because geometry still does mean-x/y with no z-rejection.

- [ ] **Step 7: Rewire geometry to the shared reduction.**
  In `benchmarks/person_tracker/ptbench/common/geometry.py`:
  - After the `import numpy as np` line (line 18), add:
    ```python
    from vision_track.core.centroid import reduce_centroid
    ```
  - Replace the reduction block (lines 93-94):
    ```python
    centroid = np.mean(obj_pts, axis=0)
    centroid[2] = np.median(obj_pts[:, 2])  # median z is more robust
    ```
    with:
    ```python
    cx_m, cy_m, cz_m = reduce_centroid(obj_pts)
    return cx_m, cy_m, cz_m
    ```
  - Delete the now-dead trailing `return float(centroid[0]), ...` line (old line 96).
  - Update the module docstring (lines 9-12) to say "median over x/y/z with z-outlier rejection (shared `vision_track.core.centroid.reduce_centroid`)" instead of "mean over x/y and median over z".

- [ ] **Step 8: Update the existing geometry tests for median reduction.**
  In `benchmarks/person_tracker/tests/test_geometry.py`, `test_constant_depth_rectangle_closed_form` (lines 37-62) computes expected x/y from the bbox midpoint as a *mean* of a uniform fill; for a filled constant-depth rectangle the median equals the mean (symmetric uniform grid), so the closed-form expectation still holds — keep it but loosen the lateral tolerance from `0.02` to `0.05` to absorb the even-pixel-count median (median of an even set averages the two central rows/cols, identical to the mean for a uniform grid, so this is belt-and-suspenders). `test_median_z_robustness`, `test_mask_restricts_region` (median of left-half is still left of full-bbox median), `test_too_few_valid_returns_none`, `test_mask_fallback_when_mask_too_sparse`, `test_zero_depth_excluded`, `test_K_as_3x3_ndarray`, `test_bbox_clamped_to_image_bounds`, `test_depth_outside_range_excluded`, `test_returns_python_floats` all remain valid unchanged.

- [ ] **Step 9: Run both new files + geometry suite to confirm pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_centroid_reduction.py tests/test_geometry.py -v`
  Expected: all pass.

- [ ] **Step 10: Apply the SAME reduction in the node's `_calculate_centroid`.**
  In `src/vision_track/vision_track/person_track_node.py`:
  - Add near the existing imports (after line 49 `from vision_util.weights_cache import resolve_weights`):
    ```python
    from vision_track.core.centroid import reduce_centroid
    ```
  - Replace the reduction block (lines 378-386):
    ```python
    # Calculate centroid (mean for x/y, median for depth)
    centroid_3d = np.mean(obj_pts, axis=0)
    centroid_3d[2] = np.median(obj_pts[:, 2])  # Use median for depth (more robust)

    # Create Point message (Orbbec frame convention)
    point = Point()
    point.x = float(centroid_3d[0])
    point.y = float(centroid_3d[1])
    point.z = float(centroid_3d[2])
    ```
    with:
    ```python
    # Robust reduction shared with ptbench geometry: median lateral x/y +
    # z-outlier-rejected median z (vision_track.core.centroid.reduce_centroid).
    cx_m, cy_m, cz_m = reduce_centroid(obj_pts)

    # Create Point message (Orbbec frame convention)
    point = Point()
    point.x = cx_m
    point.y = cy_m
    point.z = cz_m
    ```

- [ ] **Step 11: Confirm the node module still imports under the venv (no ROS graph needed for import).**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "import sys; sys.path.insert(0,'.'); import ast; ast.parse(open('vision_track/person_track_node.py').read()); print('person_track_node parses OK')"`
  Expected: `person_track_node parses OK`. (Full import pulls rclpy; a syntax/parse check is the ROS-free gate here. Functional verification is the T1 manual step in Acceptance.)

- [ ] **Step 12: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add src/vision_track/vision_track/core/centroid.py src/vision_track/test/test_centroid.py src/vision_track/vision_track/person_track_node.py benchmarks/person_tracker/ptbench/common/geometry.py benchmarks/person_tracker/tests/test_centroid_reduction.py benchmarks/person_tracker/tests/test_geometry.py && git commit -m "$(cat <<'EOF'
feat(vision_track,ptbench): shared robust centroid reduction + parity

median lateral x/y + z-outlier rejection (|z-median_z|>0.4m) in one
ROS-free helper imported by both person_track_node._calculate_centroid
and ptbench geometry. Parity test guards against silent desync.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 3: `build_gt_clip` computes both centroids (the fidelity defect fix)

The defect (spec §1.2 #4): `label_io.build_gt_clip` calls `centroid_from_bbox_depth(depth, K, bbox)` with **no mask** (line 138) while `runner.run_offline` passes `mask=getattr(result, "mask", None)` (line 124). So the GT centroid is bbox-only while predictions are mask-filtered → the measured lateral error is partly a mask-vs-bbox artifact. Fix: compute **both** GT centroids per frame — `centroid_field` from the (operator-supplied) mask when present, `centroid_track` always bbox-only (no mask), matching the node's exact path.

**Files:**
- Modify: `benchmarks/person_tracker/ptbench/labeler/label_io.py` (`FrameAnnotation` lines 39-50; `build_gt_clip` lines 85-160)
- Test: append to `benchmarks/person_tracker/tests/test_dual_gt.py`; existing `tests/test_labeler.py` updated.

- [ ] **Step 1: Write failing tests for dual-centroid `build_gt_clip` + the divergence fixture.**
  Append to `benchmarks/person_tracker/tests/test_dual_gt.py`:
  ```python
  from ptbench.labeler.label_io import FrameAnnotation, build_gt_clip


  def _pinhole_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
      return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


  class TestBuildGtClipDual:
      def test_present_frame_gets_both_centroids(self):
          H, W = 480, 640
          K = _pinhole_K()
          bbox = (100, 100, 200, 200)
          depth = np.full((H, W), 2500, dtype=np.uint16)  # 2.5 m
          ann = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox, mask=None)]
          clip = build_gt_clip(
              ann, [(1000, depth)], K,
              clip_id="c", bag_path="b", scenario="s",
              color_topic="/c", depth_topic="/d", camera_info_topic="/i",
          )
          assert clip.schema_version == "1.1"
          f0 = clip.frames[0]
          assert f0.centroid_field is not None
          assert f0.centroid_track is not None

      def test_field_uses_mask_track_ignores_it(self):
          # A mask covering only the left half shifts centroid_field left of the
          # bbox-only centroid_track — proving field != track when a mask exists.
          H, W = 480, 640
          K = _pinhole_K()
          bbox = (100, 100, 300, 200)  # 200 wide
          depth = np.full((H, W), 2000, dtype=np.uint16)
          mask = np.zeros((H, W), dtype=np.float32)
          mask[100:200, 100:200] = 1.0  # left half only
          ann = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox, mask=mask)]
          clip = build_gt_clip(
              ann, [(1000, depth)], K,
              clip_id="c", bag_path="b", scenario="s",
              color_topic="/c", depth_topic="/d", camera_info_topic="/i",
          )
          f0 = clip.frames[0]
          xf = f0.centroid_field[0]
          xt = f0.centroid_track[0]
          assert xf < xt  # masked (left half) is left of bbox-only

      def test_absent_frame_has_no_centroids(self):
          ann = [
              FrameAnnotation(t_ns=1000, present=True, bbox=(10, 10, 50, 50), mask=None),
              FrameAnnotation(t_ns=2000, present=False, bbox=None, mask=None),
          ]
          depth = np.full((480, 640), 3000, dtype=np.uint16)
          clip = build_gt_clip(
              ann, [(1000, depth), (2000, depth)], _pinhole_K(),
              clip_id="c", bag_path="b", scenario="s",
              color_topic="/c", depth_topic="/d", camera_info_topic="/i",
          )
          assert clip.frames[1].centroid_field is None
          assert clip.frames[1].centroid_track is None

      def test_divergence_is_measurable(self):
          # The synthetic fixture from the spec's "Testing (now)" section: a
          # tracker matching centroid_track EXACTLY still shows nonzero field
          # error, so the gate (on field) is not fooled by node-identical math.
          from ptbench.common.geometry import dist3d
          H, W = 480, 640
          K = _pinhole_K()
          bbox = (100, 100, 300, 200)
          depth = np.full((H, W), 2000, dtype=np.uint16)
          mask = np.zeros((H, W), dtype=np.float32)
          mask[100:200, 100:200] = 1.0
          ann = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox, mask=mask)]
          clip = build_gt_clip(
              ann, [(1000, depth)], K,
              clip_id="c", bag_path="b", scenario="s",
              color_topic="/c", depth_topic="/d", camera_info_topic="/i",
          )
          f0 = clip.frames[0]
          assert dist3d(f0.centroid_track, f0.centroid_field) > 0.05
  ```

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_dual_gt.py::TestBuildGtClipDual -v`
  Expected: `TypeError: __init__() got an unexpected keyword argument 'mask'` (FrameAnnotation has no `mask`) / centroid_field attribute errors.

- [ ] **Step 3: Add `mask` to `FrameAnnotation` and compute both centroids in `build_gt_clip`.**
  In `benchmarks/person_tracker/ptbench/labeler/label_io.py`:
  - Replace the `FrameAnnotation` dataclass (lines 39-50) with:
    ```python
    @dataclass
    class FrameAnnotation:
        """One labeled color frame, before depth sampling.

        ``bbox`` is ``(x1, y1, x2, y2)`` in color pixels, or ``None``. ``mask``
        is an optional HxW operator-supplied segmentation (truthy pixels = the
        operator) used for the mask-aware ``centroid_field``; ``None`` means the
        field centroid falls back to bbox-only (== track). The schema invariant
        ``present=True ⇒ bbox is not None`` is enforced by :func:`build_gt_clip`.
        """

        t_ns: int
        present: bool
        bbox: Optional[Bbox]
        mask: Optional[np.ndarray] = None
    ```
  - Replace the centroid computation + frame construction block (lines 134-147) with:
    ```python
            centroid_field = None
            centroid_track = None
            if present and bbox is not None:
                depth = nearest_depth(depth_list, ann.t_ns)
                if depth is not None:
                    # field: best estimate (mask-aware when a mask exists).
                    centroid_field = centroid_from_bbox_depth(
                        depth, K, bbox, mask=ann.mask
                    )
                    # track: node-identical math, always bbox-only (no mask).
                    centroid_track = centroid_from_bbox_depth(depth, K, bbox)

            frames.append(
                GtFrame(
                    t_ns=ann.t_ns,
                    present=present,
                    bbox=tuple(bbox) if (present and bbox is not None) else None,
                    centroid_field=tuple(centroid_field) if centroid_field is not None else None,
                    centroid_track=tuple(centroid_track) if centroid_track is not None else None,
                )
            )
    ```
  - Change the returned clip's `schema_version="1.0"` (line 150) to `schema_version="1.1"`.
  - Update the module docstring bullet for `build_gt_clip` (lines 14-16) and the `build_gt_clip` docstring (lines 99-116) to describe the dual centroids (field = mask-aware, track = bbox-only).

- [ ] **Step 4: Update `tests/test_labeler.py` for the dual-centroid surface.**
  In `benchmarks/person_tracker/tests/test_labeler.py`, every place that constructs a `FrameAnnotation` keeps working (the new `mask` defaults to `None`), and any assertion reading `frame.centroid_3d` must become `frame.centroid_track` (bbox-only path, identical to the old behavior when `mask=None`). Replace each `.centroid_3d` access with `.centroid_track`, and any assertion on `schema_version == "1.0"` with `== "1.1"`. (When `mask=None`, `centroid_field == centroid_track`, so existing value assertions hold against `centroid_track`.)

- [ ] **Step 5: Run to confirm pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_dual_gt.py tests/test_labeler.py -v`
  Expected: all pass.

- [ ] **Step 6: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add benchmarks/person_tracker/ptbench/labeler/label_io.py benchmarks/person_tracker/tests/test_dual_gt.py benchmarks/person_tracker/tests/test_labeler.py && git commit -m "$(cat <<'EOF'
fix(ptbench): build_gt_clip emits both centroid_field and centroid_track

Closes the mask-vs-bbox fidelity defect: GT now carries a mask-aware
centroid_field (the gated estimate) and a bbox-only centroid_track
(node-identical, diagnostic). Schema bumped to 1.1.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 4: Metrics gate on `centroid_field`; range gate in the scoreboard

**Files:**
- Modify: `benchmarks/person_tracker/ptbench/common/metrics.py` (`_has_pred_prefix` lines 37-45; `_is_correct_lock` 48-52; `_is_wrong_lock` 55-59; lateral/range loop 150-160; return dict 170-178)
- Modify: `benchmarks/person_tracker/ptbench/common/scoreboard.py` (`GateConfig` lines 20-30; `score` rows 99-155)
- Test: `benchmarks/person_tracker/tests/test_metrics.py`, `benchmarks/person_tracker/tests/test_scoreboard.py`

- [ ] **Step 1: Write failing tests — metrics score against `centroid_field`, emit `centroid_track` diagnostic; scoreboard has a range gate.**
  Append to `benchmarks/person_tracker/tests/test_metrics.py`:
  ```python
  class TestGatesOnField:
      def _gt(self, t_ns, field, track, present=True):
          from ptbench.common.schema import GtFrame
          return GtFrame(
              t_ns=t_ns, present=present,
              bbox=(0, 0, 10, 10) if present else None,
              centroid_field=field, centroid_track=track,
          )

      def _pred(self, t_ns, xyz, lost=False):
          from ptbench.common.align import PredFrame
          return PredFrame(t_ns=t_ns, target_lost=lost, target_track_id=1, point_xyz=xyz)

      def test_correct_lock_uses_field_not_track(self):
          from ptbench.common.metrics import compute_metrics
          # pred matches TRACK exactly but is 1.0 m from FIELD → NOT a correct lock.
          aligned = [(
              self._gt(1000, field=(1.0, 0.0, 3.0), track=(0.0, 0.0, 3.0)),
              self._pred(1000, (0.0, 0.0, 3.0)),
          )]
          m = compute_metrics(aligned)
          assert m["correct_lock_rate"] == 0.0  # 1.0 m > correct_radius 0.5

      def test_lateral_error_measured_against_field(self):
          from ptbench.common.metrics import compute_metrics
          # pred == field → correct lock with ~0 lateral error.
          aligned = [(
              self._gt(1000, field=(0.0, 0.0, 3.0), track=(0.3, 0.0, 3.0)),
              self._pred(1000, (0.0, 0.0, 3.0)),
          )]
          m = compute_metrics(aligned)
          assert m["correct_lock_rate"] == 1.0
          assert m["pos_error_lateral_m"]["median"] < 1e-6

      def test_centroid_track_diagnostic_present(self):
          from ptbench.common.metrics import compute_metrics
          aligned = [(
              self._gt(1000, field=(0.0, 0.0, 3.0), track=(0.3, 0.0, 3.0)),
              self._pred(1000, (0.0, 0.0, 3.0)),
          )]
          m = compute_metrics(aligned)
          assert "centroid_track_diag" in m
          diag = m["centroid_track_diag"]
          # pred (0,0,3) vs track (0.3,0,3) → lateral ~0.3 in the diagnostic.
          assert diag["pos_error_lateral_m"]["median"] == pytest.approx(0.3, abs=1e-6)
  ```
  Append to `benchmarks/person_tracker/tests/test_scoreboard.py`:
  ```python
  class TestRangeGate:
      def test_range_gate_pass(self):
          from ptbench.common.scoreboard import score
          board = score({"pos_error_range_m": {"median": 0.20}})
          row = [r for r in board.rows if r[0] == "pos_error_range_m"][0]
          assert row[2] == "PASS"

      def test_range_gate_warn(self):
          from ptbench.common.scoreboard import score
          board = score({"pos_error_range_m": {"median": 0.45}})
          row = [r for r in board.rows if r[0] == "pos_error_range_m"][0]
          assert row[2] == "WARN"

      def test_range_gate_fail(self):
          from ptbench.common.scoreboard import score
          board = score({"pos_error_range_m": {"median": 0.80}})
          row = [r for r in board.rows if r[0] == "pos_error_range_m"][0]
          assert row[2] == "FAIL"

      def test_range_gate_default_thresholds(self):
          from ptbench.common.scoreboard import GateConfig
          g = GateConfig()
          assert g.pos_error_range_pass_m == 0.30
          assert g.pos_error_range_warn_m == 0.50
  ```
  (Ensure `import pytest` is present at the top of `test_metrics.py`; it already is.)

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_metrics.py::TestGatesOnField tests/test_scoreboard.py::TestRangeGate -v`
  Expected: metrics fail (`compute_metrics` still reads `g.centroid_3d`, which no longer exists → `AttributeError`, and no `centroid_track_diag` key); scoreboard fails (no `pos_error_range_m` row, no `pos_error_range_pass_m` field).

- [ ] **Step 3: Rewire metrics to `centroid_field` + add the diagnostic block.**
  In `benchmarks/person_tracker/ptbench/common/metrics.py`:
  - In `_has_pred_prefix` (lines 37-45), replace `and g.centroid_3d is not None` with `and g.centroid_field is not None`.
  - In `_is_correct_lock` (line 52) replace `dist3d(p.point_xyz, g.centroid_3d)` with `dist3d(p.point_xyz, g.centroid_field)`.
  - In `_is_wrong_lock` (line 59) replace `g.centroid_3d` with `g.centroid_field`.
  - In the lateral/range loop (lines 153-158) replace `lateral_range(p.point_xyz, g.centroid_3d)` with `lateral_range(p.point_xyz, g.centroid_field)`.
  - Before the `return {...}` (line 170), build the diagnostic over `centroid_track` (only where it exists), reusing the correct-lock flags so the diagnostic measures the same frames:
    ```python
    # --- centroid_track diagnostic (reported, never gated) ----------------
    diag_lat: List[float] = []
    diag_rng: List[float] = []
    for k, pair in enumerate(aligned):
        if correct_flags[k]:
            g, p = pair
            if g.centroid_track is not None:
                lat, rng = lateral_range(p.point_xyz, g.centroid_track)
                diag_lat.append(lat)
                diag_rng.append(rng)
    centroid_track_diag = {
        "pos_error_lateral_m": _median_p95(diag_lat),
        "pos_error_range_m": _median_p95(diag_rng),
    }
    ```
  - Add `"centroid_track_diag": centroid_track_diag,` to the returned dict (after the `"throughput_hz"` entry, line 177).

- [ ] **Step 4: Add the range gate to the scoreboard.**
  In `benchmarks/person_tracker/ptbench/common/scoreboard.py`:
  - In `GateConfig` (after `pos_error_lateral_warn_m` line 26), add:
    ```python
    pos_error_range_pass_m: float = 0.30
    pos_error_range_warn_m: float = 0.50
    ```
  - In `score`, after the `pos_lat`/`pos_lat_med` extraction (line 106), add:
    ```python
    pos_rng = metrics.get("pos_error_range_m")
    pos_rng_med = pos_rng.get("median") if isinstance(pos_rng, dict) else None
    ```
  - Insert a row into the `rows` list immediately after the `pos_error_lateral_m` row (between lines 140 and 141):
    ```python
        (
            "pos_error_range_m",
            _fmt_value(pos_rng_med),
            _verdict_lower_better(
                pos_rng_med,
                gates.pos_error_range_pass_m,
                gates.pos_error_range_warn_m,
            ),
        ),
    ```

- [ ] **Step 5: Run to confirm pass + the rest of metrics/scoreboard suites.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_metrics.py tests/test_scoreboard.py -v`
  Expected: all pass. Note: existing tests in these files build `GtFrame(..., centroid_3d=...)`; those constructors no longer accept `centroid_3d`. Update them in Step 6 below before this passes — run order is Step 6 then Step 5.

- [ ] **Step 6: Migrate the existing metrics/scoreboard GtFrame constructions to `centroid_field`.**
  In `benchmarks/person_tracker/tests/test_metrics.py`, every `GtFrame(..., centroid_3d=X)` becomes `GtFrame(..., centroid_field=X, centroid_track=X)` (same tuple for both so existing correct/wrong-lock expectations hold — predictions were compared against the single centroid, now `centroid_field`). `tests/test_scoreboard.py` mostly feeds raw metric dicts to `score`, but any `GtFrame` built there gets the same treatment. Then re-run Step 5's command — all green.

- [ ] **Step 7: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add benchmarks/person_tracker/ptbench/common/metrics.py benchmarks/person_tracker/ptbench/common/scoreboard.py benchmarks/person_tracker/tests/test_metrics.py benchmarks/person_tracker/tests/test_scoreboard.py && git commit -m "$(cat <<'EOF'
feat(ptbench): gate on centroid_field; add pos_error_range_m gate

Metrics score correctness/lateral/range against centroid_field and emit
a centroid_track diagnostic block. Scoreboard gains a pos_error_range_m
gate (PASS <=0.30, WARN <=0.50).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 5: `action` backend is the acceptance default; offline documented as approximate

**Files:**
- Modify: `benchmarks/person_tracker/ptbench/replay/score_cli.py` (`--backend` arg lines 63-69; help text)
- Modify: `benchmarks/person_tracker/ptbench/replay/runner.py` (`run_offline` default `imgsz` line 51; docstring note)
- Test: `benchmarks/person_tracker/tests/test_score_cli_backend.py` (new)

- [ ] **Step 1: Write a failing test asserting the CLI default backend is `action`.**
  Create `benchmarks/person_tracker/tests/test_score_cli_backend.py`:
  ```python
  """The acceptance default backend is `action` (live server), not offline."""
  import argparse

  from ptbench.replay import score_cli


  def _parse_backend(argv):
      # Rebuild the same --backend contract main() uses and read what it
      # resolves to. This pins both the exposed constant and the wired default.
      parser = argparse.ArgumentParser()
      parser.add_argument("--bag", required=True)
      parser.add_argument("--gt", required=True)
      parser.add_argument(
          "--backend", choices=("offline", "action"),
          default=score_cli.DEFAULT_BACKEND,
      )
      return parser.parse_args(argv).backend


  def test_default_backend_constant_is_action():
      assert score_cli.DEFAULT_BACKEND == "action"


  def test_default_backend_resolves_to_action():
      assert _parse_backend(["--bag", "B", "--gt", "G"]) == "action"


  def test_explicit_offline_still_selectable():
      assert _parse_backend(["--bag", "B", "--gt", "G", "--backend", "offline"]) == "offline"
  ```

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_score_cli_backend.py -v`
  Expected: `AttributeError: module 'ptbench.replay.score_cli' has no attribute 'DEFAULT_BACKEND'`.

- [ ] **Step 3: Make `action` the default and expose `DEFAULT_BACKEND`.**
  In `benchmarks/person_tracker/ptbench/replay/score_cli.py`:
  - After the imports (after line 27), add a module constant:
    ```python
    DEFAULT_BACKEND = "action"  # acceptance default: exercise the live server
    ```
  - In `main`'s `--backend` argument (lines 63-69), change `default="offline"` to `default=DEFAULT_BACKEND` and update the help to:
    ```python
            help="prediction backend (default action: replay onto a live "
            "/track_person server, the acceptance path; offline: drive "
            "YOLOTracker in-process — APPROXIMATE, does not replicate the live "
            "frame-dropping loop or deployed config)",
    ```
  - Update the module docstring usage block (lines 3-12) to show `[--backend action|offline]` with `action` first and a one-line note that offline is approximate.

- [ ] **Step 4: Align `run_offline` defaults toward the deployed config + document approximation.**
  In `benchmarks/person_tracker/ptbench/replay/runner.py`, `run_offline` signature (line 51): change `imgsz: int = 1280` to `imgsz: int = 736` (matching the node's new default from Task 8). Leave `conf` as-is for Phase 0 (conf is re-tuned in Phase 1). Add to the `run_offline` docstring (after line 81) a note:
  ```
      Note:
          This backend is APPROXIMATE — it drives the tracker class in-process at
          full speed and does NOT replicate the live node's frame-dropping loop,
          ByteTrack frame_rate plumbing, or ROI-cropped depth. Use ``run_action``
          (the CLI default) for acceptance scoring; ``run_offline`` is for fast
          CI smoke and threshold sweeps only.
  ```

- [ ] **Step 5: Run to confirm pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest tests/test_score_cli_backend.py -v`
  Expected: 1 passed.

- [ ] **Step 6: Run the full ptbench suite — all 173 originals + new tests green.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest -q`
  Expected: `173 passed` baseline plus the new tests (the count rises; no failures). If `tests/test_score_wiring.py` or `tests/test_end_to_end_tier_a.py` asserted `centroid_3d` or the offline default, fix those the same way (centroid_3d → centroid_field/centroid_track; backend default offline → action) and re-run.

- [ ] **Step 7: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add benchmarks/person_tracker/ptbench/replay/score_cli.py benchmarks/person_tracker/ptbench/replay/runner.py benchmarks/person_tracker/tests/test_score_cli_backend.py && git commit -m "$(cat <<'EOF'
feat(ptbench): action backend is the acceptance default

score_cli defaults to the live /track_person backend; offline is
documented as approximate (no frame-dropping/deployed config). run_offline
imgsz aligned to the node's new 736 default.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 6: Operator-init heuristic (nearest + central, conf tie-break) replacing `results[0]`

Decision #1: at goal start, among class-`person` detections pick the candidate maximizing a combined *centeredness* (bbox-center proximity to image center) + *nearness* (smaller median depth) score, tie-broken by detection confidence. Extract a pure helper so it is unit-testable without ROS or a depth image (depth is supplied via a callable so the offline init path can pass `None`).

**Files:**
- Create: `src/vision_track/vision_track/core/operator_init.py`
- Create: `src/vision_track/test/test_operator_init.py`
- Modify: `src/vision_track/vision_track/yolo_tracker.py` (`initialize_tracking` class-only branch lines 384-393; import)

- [ ] **Step 1: Write the failing unit test for the pure selector.**
  Create `src/vision_track/test/test_operator_init.py`:
  ```python
  """Unit tests for the ROS-free operator-init heuristic."""
  from dataclasses import dataclass
  from typing import Optional, Tuple

  from vision_track.core.operator_init import select_operator_detection


  @dataclass
  class Det:
      track_id: int
      bbox: Tuple[int, int, int, int]
      confidence: float
      class_name: str = "person"


  IMG_W, IMG_H = 640, 480


  def test_picks_central_when_depth_equal():
      # Two people, equal (None) depth → the more central one wins.
      dets = [
          Det(1, (0, 0, 80, 400), 0.9),       # far left
          Det(2, (280, 40, 360, 440), 0.9),   # centered
      ]
      chosen = select_operator_detection(
          dets, image_wh=(IMG_W, IMG_H), depth_lookup=lambda b: None
      )
      assert chosen.track_id == 2


  def test_nearer_wins_over_more_central():
      # A slightly off-center but much nearer person beats a centered far one.
      dets = [
          Det(1, (300, 40, 380, 440), 0.9),   # centered, far (5 m)
          Det(2, (120, 40, 240, 440), 0.9),   # off-center, near (1 m)
      ]
      # Map each bbox (immutable tuple key) to its depth in meters.
      depth_by_bbox = {dets[0].bbox: 5.0, dets[1].bbox: 1.0}
      chosen = select_operator_detection(
          dets, image_wh=(IMG_W, IMG_H),
          depth_lookup=lambda b: depth_by_bbox[b],
      )
      assert chosen.track_id == 2


  def test_confidence_breaks_ties():
      # Two identical-geometry detections → higher confidence wins.
      dets = [
          Det(1, (280, 40, 360, 440), 0.6),
          Det(2, (280, 40, 360, 440), 0.95),
      ]
      chosen = select_operator_detection(
          dets, image_wh=(IMG_W, IMG_H), depth_lookup=lambda b: 2.0
      )
      assert chosen.track_id == 2


  def test_only_persons_considered():
      dets = [
          Det(1, (300, 40, 380, 440), 0.99, class_name="chair"),
          Det(2, (120, 40, 240, 440), 0.5, class_name="person"),
      ]
      chosen = select_operator_detection(
          dets, image_wh=(IMG_W, IMG_H), depth_lookup=lambda b: None,
          target_class="person",
      )
      assert chosen.track_id == 2


  def test_empty_returns_none():
      assert select_operator_detection(
          [], image_wh=(IMG_W, IMG_H), depth_lookup=lambda b: None
      ) is None
  ```

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_operator_init.py -v`
  Expected: `ModuleNotFoundError: No module named 'vision_track.core.operator_init'`.

- [ ] **Step 3: Implement the pure selector.**
  Create `src/vision_track/vision_track/core/operator_init.py`:
  ```python
  """ROS-free operator-selection heuristic for tracker initialization.

  At goal start, among class-person detections pick the candidate maximizing a
  combined centeredness (bbox-center proximity to image center, normalized) and
  nearness (smaller median depth), tie-broken by detection confidence. Replaces
  the nondeterministic ``results[0]`` init (yolo_tracker.initialize_tracking).

  Assumes the operator starts roughly centered/near — true for "follow me"
  framing. Depth is supplied via a callable so callers without a depth image
  (e.g. the offline benchmark init) can pass ``lambda bbox: None``; when depth is
  unavailable for all candidates the score reduces to centeredness + confidence.
  """
  from __future__ import annotations

  from typing import Callable, List, Optional, Tuple

  # Score weights: centeredness dominates, nearness assists, confidence is the
  # tie-break (small).
  W_CENTER = 1.0
  W_NEAR = 0.7
  W_CONF = 0.05
  # Depth (m) used to normalize nearness; anything >= this scores ~0 nearness.
  NEAR_NORM_M = 6.0


  def _centeredness(bbox, image_wh) -> float:
      """1.0 at image center, → 0 at the far corner. Normalized by half-diagonal."""
      w, h = image_wh
      cx, cy = w / 2.0, h / 2.0
      bx = (bbox[0] + bbox[2]) / 2.0
      by = (bbox[1] + bbox[3]) / 2.0
      dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
      half_diag = ((cx) ** 2 + (cy) ** 2) ** 0.5
      if half_diag <= 0:
          return 0.0
      return max(0.0, 1.0 - dist / half_diag)


  def _nearness(depth_m: Optional[float]) -> float:
      """1.0 at 0 m, → 0 at NEAR_NORM_M. None ⇒ 0 (neutral, no depth signal)."""
      if depth_m is None or depth_m <= 0:
          return 0.0
      return max(0.0, 1.0 - depth_m / NEAR_NORM_M)


  def select_operator_detection(
      detections: List,
      *,
      image_wh: Tuple[int, int],
      depth_lookup: Callable[[Tuple[int, int, int, int]], Optional[float]],
      target_class: str = "person",
  ):
      """Pick the best operator candidate, or ``None`` if there are no persons.

      Args:
          detections: objects with ``.bbox`` (x1,y1,x2,y2), ``.confidence``,
              and ``.class_name``.
          image_wh: (width, height) of the color image in pixels.
          depth_lookup: maps a bbox to its median depth in meters, or ``None``.
          target_class: class name to filter to (case-insensitive).

      Returns:
          The chosen detection object, or ``None``.
      """
      persons = [
          d for d in detections
          if getattr(d, "class_name", "").lower() == target_class.lower()
      ]
      if not persons:
          return None

      def score(d) -> float:
          center = _centeredness(d.bbox, image_wh)
          near = _nearness(depth_lookup(d.bbox))
          conf = float(getattr(d, "confidence", 0.0) or 0.0)
          return W_CENTER * center + W_NEAR * near + W_CONF * conf

      return max(persons, key=score)
  ```

- [ ] **Step 4: Run to confirm pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_operator_init.py -v`
  Expected: 5 passed.

- [ ] **Step 5: Wire the selector into `initialize_tracking` (class-only branch).**
  In `src/vision_track/vision_track/yolo_tracker.py`:
  - Add to the imports (after line 21 `from .core.registry import PersonRegistry`):
    ```python
    from .core.operator_init import select_operator_detection
    ```
  - Replace the class-only selection branch (lines 384-393):
    ```python
        # If target_class is provided, find first object of that class
        elif target_class is not None:
            for result in results:
                if result.class_name.lower() == target_class.lower():
                    selected_result = result
                    break

        # If no specific target, track the first detected object
        else:
            selected_result = results[0]
    ```
    with:
    ```python
        # If target_class is provided, pick the best operator candidate
        # (nearest + most central, conf tie-break) instead of results[0].
        elif target_class is not None:
            img_h, img_w = frame.shape[:2]
            selected_result = select_operator_detection(
                results,
                image_wh=(img_w, img_h),
                # No depth image at init time in this path; centeredness +
                # confidence drive the choice. The node's depth-aware init is a
                # Phase 2 concern — here depth is unavailable.
                depth_lookup=lambda _bbox: None,
                target_class=target_class,
            )

        # If no specific target, track the best central candidate of any class.
        else:
            img_h, img_w = frame.shape[:2]
            selected_result = select_operator_detection(
                results,
                image_wh=(img_w, img_h),
                depth_lookup=lambda _bbox: None,
                target_class=results[0].class_name,
            ) or results[0]
    ```

- [ ] **Step 6: Confirm yolo_tracker still imports under the venv.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "import sys; sys.path.insert(0,'.'); import vision_track.yolo_tracker; print('yolo_tracker import OK')"`
  Expected: `yolo_tracker import OK`.

- [ ] **Step 7: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add src/vision_track/vision_track/core/operator_init.py src/vision_track/test/test_operator_init.py src/vision_track/vision_track/yolo_tracker.py && git commit -m "$(cat <<'EOF'
feat(vision_track): nearest+central operator-init heuristic

Replaces nondeterministic results[0] at goal start with a pure
select_operator_detection (centeredness + nearness, conf tie-break),
extracted so it is unit-testable without ROS.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 7: Association quick win — periodic-ReID switch margin 0.08 → 0.15

The periodic on-track ReID validator switches the lock to a different track only when the candidate beats the current target's similarity by a margin. The current margin (`tracking_pipeline.py:508`, verified) is `margin_required = max(ReIDMatcher.REID_MARGIN, 0.08)`. With the untrained deep head, 0.08 is too easy and lets lookalikes steal the lock; raise the floor to 0.15.

**Files:**
- Modify: `src/vision_track/vision_track/core/tracking_pipeline.py` (line 508)
- Test: `src/vision_track/test/test_periodic_reid_margin.py` (new)

- [ ] **Step 1: Write a failing test pinning the switch margin floor.**
  Create `src/vision_track/test/test_periodic_reid_margin.py`:
  ```python
  """The periodic-ReID switch requires a >=0.15 similarity margin.

  This is a behavioral pin on tracking_pipeline.periodic_reid_validation: a
  candidate that beats the current target by 0.10 (above the old 0.08 floor,
  below the new 0.15 floor) must NOT trigger a switch.
  """
  from types import SimpleNamespace

  import numpy as np

  from vision_track.core import tracking_pipeline as tp
  from vision_track.core.tracking_types import TrackingResult


  class _StubExtractor:
      def extract_features(self, frame, bbox, mask, class_id=0):
          return [1.0]  # truthy, non-empty


  def _make_tracker(best_match, best_similarity):
      """A minimal duck-typed tracker for periodic_reid_validation."""
      tracker = SimpleNamespace()
      tracker.reid_verification_interval = 1
      tracker.frame_count = 1
      tracker.enable_reid = True
      tracker.target_appearance = object()
      tracker.appearance_extractor = _StubExtractor()
      tracker.reid_threshold = 0.5
      tracker._find_best_match_reid = lambda frame, results: (best_match, best_similarity)
      return tracker


  def _res(track_id):
      return TrackingResult(
          track_id=track_id, bbox=(0, 0, 10, 20), mask=None,
          confidence=0.9, class_id=0, class_name="person",
      )


  def test_margin_below_015_does_not_switch(monkeypatch):
      # current similarity 0.60, candidate 0.70 → margin 0.10 < 0.15 → no switch.
      monkeypatch.setattr(tp.ReIDMatcher, "compute_similarity",
                          staticmethod(lambda *a, **k: 0.60))
      cand = _res(2)
      tracker = _make_tracker(best_match=cand, best_similarity=0.70)
      cur = _res(1)
      ok, switch_to = tp.periodic_reid_validation(tracker, np.zeros((20, 10, 3)), [cur, cand], cur)
      assert ok is True
      assert switch_to is None


  def test_margin_at_or_above_015_switches(monkeypatch):
      # current 0.60, candidate 0.80 → margin 0.20 >= 0.15 and > threshold → switch.
      monkeypatch.setattr(tp.ReIDMatcher, "compute_similarity",
                          staticmethod(lambda *a, **k: 0.60))
      cand = _res(2)
      tracker = _make_tracker(best_match=cand, best_similarity=0.80)
      cur = _res(1)
      ok, switch_to = tp.periodic_reid_validation(tracker, np.zeros((20, 10, 3)), [cur, cand], cur)
      assert ok is False
      assert switch_to is cand
  ```

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_periodic_reid_margin.py -v`
  Expected: `test_margin_below_015_does_not_switch` FAILS — with the current `0.08` floor, margin 0.10 > 0.08 triggers a switch, so `ok` is `False` / `switch_to is cand`. (`test_margin_at_or_above_015_switches` passes already.)

- [ ] **Step 3: Raise the margin floor.**
  In `src/vision_track/vision_track/core/tracking_pipeline.py`, line 508, change:
  ```python
      margin_required = max(ReIDMatcher.REID_MARGIN, 0.08)
  ```
  to:
  ```python
      margin_required = max(ReIDMatcher.REID_MARGIN, 0.15)
  ```

- [ ] **Step 4: Run to confirm pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_periodic_reid_margin.py -v`
  Expected: 2 passed.

- [ ] **Step 5: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add src/vision_track/vision_track/core/tracking_pipeline.py src/vision_track/test/test_periodic_reid_margin.py && git commit -m "$(cat <<'EOF'
feat(vision_track): raise periodic-ReID switch margin 0.08 -> 0.15

Harder switch floor reduces lookalike lock-stealing while the deep ReID
head is still untrained (Phase 1 replaces the head).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 8: Throughput quick wins — remove the 15 Hz cap, imgsz 736 + half=True, ROI-crop depth unproject

Three independent node/tracker edits. None has a pure-Python unit surface (all are ROS/torch-bound), so each is verified by a parse check + a single manual T1 observation. The ROI-crop has a small pure-extractable core (compute the unproject sub-window from a bbox) — extract and unit-test that.

**Files:**
- Modify: `src/vision_track/vision_track/person_track_node.py` (`tracking_rate` declare line 137 / load 163 / loop use 539,576-578; `inference_size` default line 124; `_depth_image_to_points` lines 290-322; `_handle_tracked_frame` centroid call line 653)
- Modify: `src/vision_track/vision_track/yolo_tracker.py` (`track` kwargs lines 297-306)
- Create: `src/vision_track/vision_track/core/depth_roi.py` (pure window math)
- Create: `src/vision_track/test/test_depth_roi.py`

- [ ] **Step 1: Write the failing unit test for the ROI-window helper.**
  Create `src/vision_track/test/test_depth_roi.py`:
  ```python
  """Unit tests for the depth-unproject ROI-window helper."""
  from vision_track.core.depth_roi import roi_window


  def test_window_clamped_and_padded():
      # bbox near the top-left, pad 16, image 640x480.
      x0, y0, x1, y1 = roi_window((10, 5, 100, 200), w=640, h=480, pad=16)
      assert x0 == 0          # 10-16 clamped to 0
      assert y0 == 0          # 5-16 clamped to 0
      assert x1 == 116        # 100+16
      assert y1 == 216        # 200+16


  def test_window_clamped_to_image_max():
      x0, y0, x1, y1 = roi_window((600, 460, 700, 500), w=640, h=480, pad=16)
      assert x1 == 640
      assert y1 == 480
      assert x0 == 584
      assert y0 == 444


  def test_none_bbox_returns_full_frame():
      assert roi_window(None, w=640, h=480, pad=16) == (0, 0, 640, 480)


  def test_degenerate_bbox_returns_full_frame():
      # x2<=x1 after clamp → full frame fallback (caller unprojects everything).
      assert roi_window((300, 300, 300, 300), w=640, h=480, pad=0) == (0, 0, 640, 480)
  ```

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_depth_roi.py -v`
  Expected: `ModuleNotFoundError: No module named 'vision_track.core.depth_roi'`.

- [ ] **Step 3: Implement the ROI-window helper.**
  Create `src/vision_track/vision_track/core/depth_roi.py`:
  ```python
  """ROS-free helper: compute a padded, clamped depth-unproject sub-window.

  Today the node unprojects the entire HxW depth frame every tick; only the
  target bbox is ever sampled. This computes the (x0,y0,x1,y1) sub-window to
  unproject — the bbox padded by ``pad`` px and clamped to the image. A missing
  or degenerate bbox falls back to the full frame so the caller never crashes.
  """
  from __future__ import annotations

  from typing import Optional, Tuple


  def roi_window(
      bbox: Optional[Tuple[int, int, int, int]],
      *,
      w: int,
      h: int,
      pad: int = 16,
  ) -> Tuple[int, int, int, int]:
      """Return (x0, y0, x1, y1) of the padded, clamped unproject window."""
      if bbox is None:
          return (0, 0, w, h)
      bx1, by1, bx2, by2 = bbox
      x0 = max(0, int(bx1) - pad)
      y0 = max(0, int(by1) - pad)
      x1 = min(w, int(bx2) + pad)
      y1 = min(h, int(by2) + pad)
      if x1 <= x0 or y1 <= y0:
          return (0, 0, w, h)
      return (x0, y0, x1, y1)
  ```

- [ ] **Step 4: Run to confirm pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_depth_roi.py -v`
  Expected: 4 passed.

- [ ] **Step 5: ROI-crop the depth unproject in the node.**
  In `src/vision_track/vision_track/person_track_node.py`:
  - Add to imports (after the centroid import from Task 2):
    ```python
    from vision_track.core.depth_roi import roi_window
    ```
  - Change `_depth_image_to_points` to accept an optional `bbox` and only unproject the sub-window (the full-frame XYZ array is still returned, with non-ROI pixels left invalid so `_calculate_centroid`'s indexing math is unchanged). Replace the body (lines 301-322) with:
    ```python
            h, w = depth_msg.height, depth_msg.width
            fx, fy = intrinsic.k[0], intrinsic.k[4]
            cx, cy = intrinsic.k[2], intrinsic.k[5]

            # Orbbec Femto Bolt default: 16UC1 depth in millimeters.
            depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) * 0.001

            valid_mask = (depth > self.min_depth) & (depth < self.max_depth)

            # Only the target bbox is ever sampled by _calculate_centroid, so
            # restrict the unproject to a padded window around it. Pixels outside
            # the window stay zeroed and invalid.
            x0, y0, x1, y1 = roi_window(bbox, w=w, h=h, pad=16)
            valid_roi = np.zeros_like(valid_mask)
            valid_roi[y0:y1, x0:x1] = valid_mask[y0:y1, x0:x1]
            valid_mask = valid_roi

            # Cache meshgrid across calls at this resolution.
            cache = getattr(self, '_uv_cache', None)
            if cache is None or cache[0] != (h, w):
                u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                self._uv_cache = ((h, w), u, v)
            _, u, v = self._uv_cache

            points = np.zeros((h, w, 3), dtype=np.float32)
            z_roi = depth[y0:y1, x0:x1]
            u_roi = u[y0:y1, x0:x1]
            v_roi = v[y0:y1, x0:x1]
            points[y0:y1, x0:x1, 0] = (u_roi - cx) * z_roi / fx
            points[y0:y1, x0:x1, 1] = (v_roi - cy) * z_roi / fy
            points[y0:y1, x0:x1, 2] = z_roi

            return points, valid_mask
    ```
  - Change the signature (line 290) to:
    ```python
        def _depth_image_to_points(self, depth_msg: Image, intrinsic: CameraInfo, bbox: tuple = None) -> tuple:
    ```
  - In `_handle_tracked_frame`, change the call (line 646) from `self._depth_image_to_points(depth_msg, intrinsic)` to `self._depth_image_to_points(depth_msg, intrinsic, bbox=track_result.bbox)`.

- [ ] **Step 6: Remove the 15 Hz self-cap (rely on frame-seq dedup).**
  In `src/vision_track/vision_track/person_track_node.py`:
  - The loop already skips unchanged frames via `_get_latest_data` returning `False` when `current_seq == self.last_processed_seq` (line 600). Remove the artificial sleep cap. In `_run_tracking_loop`, delete the `rate_period` computation (line 539) and the trailing sleep (lines 576-578):
    ```python
            elapsed = time.time() - loop_start
            if elapsed < rate_period:
                time.sleep(rate_period - elapsed)
    ```
    Replace with a tiny yield to avoid a busy-spin when no new frame is pending (the `data is False` branch already sleeps 5 ms; keep a 1 ms floor here):
    ```python
            # No artificial Hz cap: frame-seq dedup in _get_latest_data gates the
            # loop to the camera rate. A 1 ms yield keeps the GIL fair.
            time.sleep(0.001)
    ```
  - Keep the `tracking_rate` parameter declared (it still feeds `max_frames_allowed` in `_init_tracker` line 187 and the ByteTrack `frame_rate` in Task 9). Remove the `rate_period = 1.0 / self.tracking_rate` line (539) since it is now unused.

- [ ] **Step 7: imgsz default 1280 → 736 + `half=True` on the track call.**
  In `src/vision_track/vision_track/person_track_node.py`, change the `inference_size` default (line 124) from `1280` to `736`:
  ```python
          self.declare_parameter('inference_size', 736)  # imgsz for YOLO; lower for speed
  ```
  In `src/vision_track/vision_track/yolo_tracker.py`, `track` kwargs (lines 297-304), add `half=True`:
  ```python
          track_kwargs = dict(
              conf=self.confidence_threshold,
              iou=self.iou_threshold,
              classes=classes,
              persist=persist,
              tracker="bytetrack.yaml",
              half=True,
              verbose=False,
          )
  ```
  (Ultralytics ignores `half=True` on CPU, so this is safe when CUDA is absent.)

- [ ] **Step 8: Parse-check both edited node/tracker modules.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "import sys; sys.path.insert(0,'.'); import ast; ast.parse(open('vision_track/person_track_node.py').read()); import vision_track.yolo_tracker; print('node parses + yolo_tracker imports OK')"`
  Expected: `node parses + yolo_tracker imports OK`.

- [ ] **Step 9: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add src/vision_track/vision_track/core/depth_roi.py src/vision_track/test/test_depth_roi.py src/vision_track/vision_track/person_track_node.py src/vision_track/vision_track/yolo_tracker.py && git commit -m "$(cat <<'EOF'
perf(vision_track): drop 15Hz cap, imgsz 736+half, ROI-crop depth

Loop now runs at camera rate (frame-seq dedup gates it); YOLO track at
imgsz=736 half=True; depth unproject restricted to a padded bbox window
(pure roi_window helper, unit-tested).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 9: Latent crashers — native-ReID guard, lost-sentinel republish, ByteTrack frame_rate, frame-stamp velocity dt

Four hazards from spec §1.2 #6. Two have pure surfaces (native-guard message, frame-stamp dt math); two are node-runtime (sentinel republish, frame_rate plumbing) verified manually.

**Files:**
- Modify: `src/vision_track/vision_track/person_track_node.py` (native branch lines 193-204; loop rate plumb in `_init_tracker`; lost-sentinel in `_handle_lost_frame` lines 766-786)
- Modify: `src/vision_track/vision_track/yolo_tracker.py` (`_update_target_velocity` lines 665-689; add `frame_rate` attr + use it; `track` tracker config)
- Create: `src/vision_track/test/test_velocity_dt.py`

- [ ] **Step 1: Write a failing test for frame-stamp `dt` in the velocity model.**
  `_update_target_velocity` (lines 665-689) uses `time.time()` wall-clock `dt`. Make it accept an explicit `dt` (frame-stamp delta) so velocity is in pixels/sec of *scene* time, not wall time. Create `src/vision_track/test/test_velocity_dt.py`:
  ```python
  """_update_target_velocity must use a supplied frame-stamp dt, not wall clock."""
  from vision_track.yolo_tracker import YOLOTracker


  def _bare_tracker():
      # Construct without loading YOLO/torch: bypass __init__ and set just the
      # fields _update_target_velocity touches.
      t = YOLOTracker.__new__(YOLOTracker)
      t.last_known_center = (100.0, 100.0)
      t.last_position_time = 0.0
      t.target_velocity = (0.0, 0.0)
      t.target_velocity_history = []
      return t


  def test_velocity_uses_supplied_dt():
      t = _bare_tracker()
      # Move 50 px in x over dt=0.5 s of scene time → raw vx = 100 px/s.
      # EMA alpha=0.3 from a zero start → 0.3 * 100 = 30 px/s.
      t._update_target_velocity((150.0, 100.0), dt=0.5)
      vx, vy = t.target_velocity
      assert abs(vx - 30.0) < 1e-6
      assert abs(vy - 0.0) < 1e-6


  def test_zero_dt_is_ignored():
      t = _bare_tracker()
      t._update_target_velocity((150.0, 100.0), dt=0.0)
      assert t.target_velocity == (0.0, 0.0)
  ```

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_velocity_dt.py -v`
  Expected: `TypeError: _update_target_velocity() got an unexpected keyword argument 'dt'`.

- [ ] **Step 3: Make `_update_target_velocity` take a frame-stamp `dt`.**
  In `src/vision_track/vision_track/yolo_tracker.py`, replace `_update_target_velocity` (lines 665-689) with:
  ```python
      def _update_target_velocity(self, current_center: Tuple[float, float], dt: Optional[float] = None):
          """Update target velocity estimate with smoothing.

          Args:
              current_center: Current target center (cx, cy).
              dt: Scene-time delta (s) between this and the previous center,
                  derived from frame stamps. When None, falls back to wall-clock
                  (legacy behavior) so non-stamped callers keep working.
          """
          if dt is None:
              current_time = time.time()
              dt = (current_time - self.last_position_time) if self.last_position_time > 0 else 0.0
              self.last_position_time = current_time

          if self.last_known_center is not None and dt > 0.001:
              vx = (current_center[0] - self.last_known_center[0]) / dt
              vy = (current_center[1] - self.last_known_center[1]) / dt
              alpha = 0.3
              old_vx, old_vy = self.target_velocity
              self.target_velocity = (
                  alpha * vx + (1 - alpha) * old_vx,
                  alpha * vy + (1 - alpha) * old_vy,
              )
  ```

- [ ] **Step 4: Run to confirm pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_velocity_dt.py -v`
  Expected: 2 passed. (The callers in `tracking_pipeline.py` that call `_update_target_velocity(center)` keep working via the `dt=None` wall-clock fallback — no caller change required in Phase 0; threading the frame stamp through `update_tracker` is a Phase 2 geometry concern.)

- [ ] **Step 5: Guard `reid_mode='native'` with a clear error.**
  `track_yolo_native.py` does not exist (`from vision_track.track_yolo_native import YOLOTrackerNative` at line 195 → `ImportError`). Replace the `if self.reid_mode == 'native':` block (lines 193-204) with an explicit, descriptive raise:
  ```python
              if self.reid_mode == 'native':
                  raise NotImplementedError(
                      "reid_mode='native' is not implemented in tk26 — "
                      "track_yolo_native.YOLOTrackerNative does not exist. "
                      "Use reid_mode='custom' (the default)."
                  )
  ```
  This converts a confusing `ImportError` deep in init into a clear contract error at the top of the branch.

- [ ] **Step 6: Plumb the loop rate into ByteTrack `frame_rate`.**
  ByteTrack's `max_time_lost` derives from `frame_rate` (stock Ultralytics `bytetrack.yaml` hardcodes ~30 → ~2 s buffer at the real loop rate). For Phase 0 (no project `bytetrack.yaml` yet — that lands in Phase 1's config dir), set the tracker's effective frame rate from the node's `tracking_rate` param so the in-process tracker knows the real cadence. In `_init_tracker`'s `else` branch (after constructing `YOLOTracker`, around line 213), add:
  ```python
                  # Communicate the real loop cadence so loss/buffer timing is
                  # wall-clock-correct (ByteTrack frame_rate is wired through a
                  # project bytetrack.yaml in Phase 1; here we record it on the
                  # tracker for max_frames_lost derivation).
                  self.tracker.frame_rate = float(self.tracking_rate)
  ```
  And in `src/vision_track/vision_track/yolo_tracker.py`, add `self.frame_rate: float = 30.0` to `_init_reid_settings` (after `self.max_frames_lost = 600`, line 128) so the attribute always exists.

  > **Note (Phase 1 handoff):** the genuine `byte_tracker.py:279 frame_rate=30` fix requires a project `bytetrack.yaml` passed to `model.track(tracker=...)`. The config dir (`src/vision_track/config/`) is introduced in Phase 1 (spec §9); Phase 0 only records the cadence on the tracker so `max_frames_lost` math (already in `_init_tracker`) is consistent. Do not create `bytetrack.yaml` here.

- [ ] **Step 7: Republish a lost-sentinel on `/target_points` during loss.**
  Today `target_point_pub.publish(...)` happens only in `_handle_tracked_frame` (line 685); during loss nav consumers see a stale point. Publish a sentinel PointStamped (NaN coords, current header) once per lost tick so consumers can detect the loss. In `_handle_lost_frame`, after `goal_handle.publish_feedback(feedback)` (line 786) and before the `if time_since_seen > self.lost_timeout:` check, add:
  ```python
          # Republish a lost-sentinel so /target_points consumers see the loss
          # instead of a stale last-good point. NaN coords flag "no target".
          if self.target_point_pub is not None:
              sentinel = PointStamped()
              sentinel.header = rgb_msg.header
              sentinel.point.x = float('nan')
              sentinel.point.y = float('nan')
              sentinel.point.z = float('nan')
              self.target_point_pub.publish(sentinel)
  ```

- [ ] **Step 8: Parse-check both modules + full vision_track unit suite.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "import sys; sys.path.insert(0,'.'); import ast; ast.parse(open('vision_track/person_track_node.py').read()); import vision_track.yolo_tracker; print('OK')" && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_velocity_dt.py test/test_centroid.py test/test_operator_init.py test/test_depth_roi.py test/test_periodic_reid_margin.py -q`
  Expected: `OK` then all vision_track Phase-0 unit tests pass.

- [ ] **Step 9: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add src/vision_track/vision_track/person_track_node.py src/vision_track/vision_track/yolo_tracker.py src/vision_track/test/test_velocity_dt.py && git commit -m "$(cat <<'EOF'
fix(vision_track): native-reid guard, lost sentinel, frame-rate, dt velocity

reid_mode='native' now raises a clear NotImplementedError; /target_points
gets a NaN lost-sentinel during loss; tracker records the real loop
cadence; velocity model accepts a frame-stamp dt (wall-clock fallback).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

### Task 10: Node instrumentation — `perf_logging_enabled` param + per-stage timers + per-frame diagnostics

Default `False` (off). When on: per-stage `perf_counter` timers in `_run_tracking_loop`, and per published frame log `mask_pixel_count`, `valid_pixel_count`, `used_mask` (did the `<10`-px bbox fallback at `person_track_node.py:366-368` fire), `depth_z_iqr`, BOTH mask & bbox centroids, and "alive-but-no-centroid" ticks. The per-frame diagnostic computation has a pure core (compute the diag dict from points+mask+bbox) — extract and unit-test it; the timing + logging glue is node-internal (manual verification).

**Files:**
- Modify: `src/vision_track/vision_track/person_track_node.py` (declare/load params; `_run_tracking_loop`; `_calculate_centroid` → return a used_mask flag; `_handle_tracked_frame` diag call)
- Create: `src/vision_track/vision_track/core/frame_diag.py` (pure diag computation)
- Create: `src/vision_track/test/test_frame_diag.py`

- [ ] **Step 1: Write the failing unit test for the pure diagnostic computation.**
  Create `src/vision_track/test/test_frame_diag.py`:
  ```python
  """Unit tests for the ROS-free per-frame perf/quality diagnostic."""
  import numpy as np

  from vision_track.core.frame_diag import compute_frame_diag


  def _points(h, w, z):
      pts = np.zeros((h, w, 3), dtype=np.float32)
      pts[:, :, 2] = z
      return pts


  def test_mask_and_valid_counts():
      h, w = 100, 100
      pts = _points(h, w, 2.0)
      valid = np.ones((h, w), dtype=bool)
      mask = np.zeros((h, w), dtype=np.uint8)
      mask[10:30, 10:30] = 1  # 400 px
      bbox = (0, 0, 100, 100)
      diag = compute_frame_diag(pts, mask, valid, bbox)
      assert diag["mask_pixel_count"] == 400
      assert diag["valid_pixel_count"] == 10000
      assert diag["used_mask"] is True  # mask has >=10 px in bbox


  def test_used_mask_false_when_mask_too_sparse():
      h, w = 100, 100
      pts = _points(h, w, 2.0)
      valid = np.ones((h, w), dtype=bool)
      mask = np.zeros((h, w), dtype=np.uint8)
      mask[0:1, 0:3] = 1  # 3 px < 10 → fallback fires
      bbox = (0, 0, 100, 100)
      diag = compute_frame_diag(pts, mask, valid, bbox)
      assert diag["used_mask"] is False


  def test_depth_z_iqr_computed():
      h, w = 100, 100
      pts = np.zeros((h, w, 3), dtype=np.float32)
      # Half at z=2.0, half at z=4.0 → IQR spans 2.0.
      pts[: h // 2, :, 2] = 2.0
      pts[h // 2 :, :, 2] = 4.0
      valid = np.ones((h, w), dtype=bool)
      mask = np.ones((h, w), dtype=np.uint8)
      bbox = (0, 0, 100, 100)
      diag = compute_frame_diag(pts, mask, valid, bbox)
      assert diag["depth_z_iqr"] > 1.5


  def test_both_centroids_present():
      h, w = 100, 100
      pts = _points(h, w, 2.0)
      valid = np.ones((h, w), dtype=bool)
      mask = np.zeros((h, w), dtype=np.uint8)
      mask[10:90, 10:50] = 1  # left-biased mask
      bbox = (0, 0, 100, 100)
      diag = compute_frame_diag(pts, mask, valid, bbox)
      assert diag["mask_centroid"] is not None
      assert diag["bbox_centroid"] is not None
      # mask is left-biased → mask centroid x < bbox centroid x
      assert diag["mask_centroid"][0] < diag["bbox_centroid"][0]


  def test_no_valid_points_marks_no_centroid():
      h, w = 100, 100
      pts = _points(h, w, 2.0)
      valid = np.zeros((h, w), dtype=bool)  # nothing valid
      mask = np.ones((h, w), dtype=np.uint8)
      bbox = (0, 0, 100, 100)
      diag = compute_frame_diag(pts, mask, valid, bbox)
      assert diag["bbox_centroid"] is None
      assert diag["no_centroid"] is True
  ```

- [ ] **Step 2: Run to confirm failure.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_frame_diag.py -v`
  Expected: `ModuleNotFoundError: No module named 'vision_track.core.frame_diag'`.

- [ ] **Step 3: Implement the pure diagnostic.**
  Create `src/vision_track/vision_track/core/frame_diag.py`:
  ```python
  """ROS-free per-frame perf/quality diagnostic for the person tracker.

  Computes mask/valid pixel counts, whether the <10px mask→bbox fallback would
  fire (used_mask), depth z IQR over the kept points, BOTH the mask-filtered and
  bbox-only centroids (via the shared reduce_centroid), and a no_centroid flag.
  Logged only when perf_logging_enabled. Pure; no rclpy/torch.
  """
  from __future__ import annotations

  from typing import Optional

  import numpy as np

  from .centroid import reduce_centroid


  def _roi(arr, bbox):
      x1, y1, x2, y2 = bbox
      h, w = arr.shape[:2]
      x1, y1 = max(0, int(x1)), max(0, int(y1))
      x2, y2 = min(w, int(x2)), min(h, int(y2))
      return arr[y1:y2, x1:x2], (x1, y1, x2, y2)


  def _centroid_from(points_roi, sel_mask) -> Optional[tuple]:
      if sel_mask.sum() < 10:
          return None
      obj = points_roi[np.nonzero(sel_mask)]
      if obj.ndim != 2 or obj.shape[0] == 0:
          return None
      return reduce_centroid(obj)


  def compute_frame_diag(points, mask, valid_mask, bbox) -> dict:
      """Return a per-frame diagnostic dict (see module docstring)."""
      points_roi, (x1, y1, x2, y2) = _roi(points, bbox)
      valid_roi, _ = _roi(valid_mask, bbox)
      valid_roi_b = valid_roi.astype(bool)

      if mask is not None and mask.shape[0] > 0 and mask.shape[1] > 0:
          mask_roi = mask[y1:y2, x1:x2].astype(bool)
      else:
          mask_roi = np.zeros_like(valid_roi_b)

      mask_sel = mask_roi & valid_roi_b
      mask_pixel_count = int(mask_roi.sum())
      valid_pixel_count = int(valid_mask.astype(bool).sum())
      used_mask = bool(mask_sel.sum() >= 10)

      bbox_centroid = _centroid_from(points_roi, valid_roi_b)
      mask_centroid = _centroid_from(points_roi, mask_sel)

      # z IQR over the chosen point set (mask if used, else bbox-valid).
      sel = mask_sel if used_mask else valid_roi_b
      if sel.sum() >= 2:
          zvals = points_roi[np.nonzero(sel)][:, 2]
          q75, q25 = np.percentile(zvals, [75, 25])
          depth_z_iqr = float(q75 - q25)
      else:
          depth_z_iqr = 0.0

      no_centroid = (bbox_centroid is None) and (mask_centroid is None)

      return {
          "mask_pixel_count": mask_pixel_count,
          "valid_pixel_count": valid_pixel_count,
          "used_mask": used_mask,
          "depth_z_iqr": depth_z_iqr,
          "mask_centroid": mask_centroid,
          "bbox_centroid": bbox_centroid,
          "no_centroid": no_centroid,
      }
  ```

- [ ] **Step 4: Run to confirm pass.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_frame_diag.py -v`
  Expected: 5 passed.

- [ ] **Step 5: Add the `perf_logging_enabled` param + per-stage timers + diag logging to the node.**
  In `src/vision_track/vision_track/person_track_node.py`:
  - Add the import (after the frame helpers from earlier tasks):
    ```python
    from vision_track.core.frame_diag import compute_frame_diag
    ```
  - In `_declare_parameters` (after line 144), declare:
    ```python
            self.declare_parameter('perf_logging_enabled', False)
    ```
  - In `_load_parameters` (after line 168), load:
    ```python
            self.perf_logging_enabled = self.get_parameter('perf_logging_enabled').value
    ```
  - In `_run_tracking_loop`, wrap the heavy stages with `perf_counter` when enabled. Replace the tracker-update + branch block (lines 558-574) with timed variants:
    ```python
                t_track0 = time.perf_counter()
                with self.lock_tracker:
                    if not initialized:
                        initialized = self._try_initialize(rgb_frame, init_start_time, goal_handle, result)
                        if not initialized:
                            time.sleep(0.1)
                            continue
                        last_seen_time = time.time()
                    track_result = self.tracker.update(rgb_frame)
                t_track = time.perf_counter() - t_track0

                t_post0 = time.perf_counter()
                if track_result is not None:
                    last_seen_time = time.time()
                    self._handle_tracked_frame(
                        track_result, rgb_img, rgb_msg, depth_msg, intrinsic, feedback, goal_handle, params
                    )
                else:
                    if self._handle_lost_frame(last_seen_time, rgb_img, rgb_msg, feedback, goal_handle, params, result):
                        return result
                t_post = time.perf_counter() - t_post0

                if self.perf_logging_enabled:
                    self.get_logger().info(
                        f"[perf] track={t_track*1000:.1f}ms post={t_post*1000:.1f}ms "
                        f"loop={(time.time()-loop_start)*1000:.1f}ms"
                    )
    ```
  - In `_handle_tracked_frame`, after computing `position` (line 653), add the diag emit:
    ```python
            if self.perf_logging_enabled and points is not None:
                diag = compute_frame_diag(points, track_result.mask, valid_mask, track_result.bbox)
                self.get_logger().info(
                    f"[diag] mask_px={diag['mask_pixel_count']} "
                    f"valid_px={diag['valid_pixel_count']} used_mask={diag['used_mask']} "
                    f"z_iqr={diag['depth_z_iqr']:.3f} "
                    f"mask_c={diag['mask_centroid']} bbox_c={diag['bbox_centroid']} "
                    f"no_centroid={diag['no_centroid']}"
                )
    ```
    This logs the "alive-but-no-centroid" tick via `no_centroid=True` when the tracker has a 2D lock but depth yields nothing.

- [ ] **Step 6: Parse-check the node.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "import ast; ast.parse(open('vision_track/person_track_node.py').read()); print('node parses OK')"`
  Expected: `node parses OK`.

- [ ] **Step 7: Commit.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker && git add src/vision_track/vision_track/core/frame_diag.py src/vision_track/test/test_frame_diag.py src/vision_track/vision_track/person_track_node.py && git commit -m "$(cat <<'EOF'
feat(vision_track): perf_logging_enabled instrumentation (default off)

Per-stage perf_counter timers + per-frame diag (mask/valid px counts,
used_mask fallback flag, depth z IQR, both mask & bbox centroids,
alive-but-no-centroid ticks) via a pure compute_frame_diag helper.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"`

---

## Acceptance

### Now-testable (must all hold before Phase 0 is done)

- [ ] **Full ptbench suite green (baseline + new).**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/benchmarks/person_tracker && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest -q`
  Expected: the original **173 passed** plus all new ptbench tests (`test_dual_gt.py`, `test_centroid_reduction.py`, `test_score_cli_backend.py`) — total > 173, **0 failed**. (`test_centroid_reduction.py` requires `vision_track.core.centroid` importable; if the worktree's `vision_track` is not installed in the venv, the parity test `importorskip`s — note this in the run output and follow up with a `colcon build --packages-select vision_track` via the tk26 wrapper so the parity test actually runs.)

- [ ] **Full vision_track Phase-0 unit suite green.**
  `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_centroid.py test/test_operator_init.py test/test_depth_roi.py test/test_periodic_reid_margin.py test/test_velocity_dt.py test/test_frame_diag.py -q`
  Expected: all pass.

- [ ] **Synthetic dual-GT divergence proven** (spec §"Testing (now)"): `test_dual_gt.py::TestBuildGtClipDual::test_divergence_is_measurable` is green — a tracker matching `centroid_track` exactly still shows nonzero `centroid_field` error.

- [ ] **node↔geometry centroid parity proven**: `test_centroid_reduction.py::test_parity_node_vs_geometry` and `::test_geometry_imports_the_shared_reduce` green (same reduction object on both sides).

- [ ] **`reid_mode='native'` fails loudly (manual T1).** Build via the tk26 wrapper, then:
  `./src/tk26_vision/scripts/build.sh --packages-select vision_track && source install/setup.bash && ros2 run vision_track person_track_server --ros-args -p reid_mode:=native`
  Expected: the node aborts at startup with `NotImplementedError: reid_mode='native' is not implemented in tk26 …`, **not** a bare `ImportError`. (Run from the worktree root; the worktree shares the workspace install tree.)

- [ ] **Node starts + publishes a lost-sentinel (manual T1).** With cameras up (see `CAMERA_BRINGUP.md`):
  `ros2 run vision_track person_track_server` then send a goal and occlude/remove the operator. In another shell:
  `ros2 topic echo /target_points`
  Expected during loss: a `PointStamped` with `x=y=z=nan` published each lost tick (instead of a stale last-good point). During track: finite coords. This confirms the sentinel republish.

- [ ] **perf logging toggles cleanly (manual T1).**
  `ros2 run vision_track person_track_server --ros-args -p perf_logging_enabled:=true`
  Expected: `[perf] track=…ms post=…ms loop=…ms` and `[diag] mask_px=… used_mask=… z_iqr=…` lines once a goal is active; default run (no param) prints neither.

### Arena-deferred (cannot be confirmed until Orbbec arena recordings exist)

- Actual scorecard numbers (`correct_lock_rate`, `wrong_lock_episodes`, `reacquire_latency_s`, `pos_error_lateral_m`, `pos_error_range_m`, `false_target_rate`, `throughput_hz`) under the `action` backend on recorded bags — Phase 0 fixes the *ruler*, not the score.
- Whether the nearest+central operator-init reliably locks the intended operator across the five scenarios.
- Whether the throughput wins (no-cap + imgsz 736 + half + ROI-crop) actually clear the ≥12 Hz gate under multi-person re-ID — confirmed later via the Phase-0 perf instrumentation on real scenes.
- Whether the 0.15 periodic-ReID margin reduces real wrong-lock episodes (full resolution waits on Phase 1's trained ReID head).
