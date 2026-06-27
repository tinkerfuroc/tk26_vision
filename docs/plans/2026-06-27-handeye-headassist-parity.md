# Head-Assisted Warm-Start + pan_tilt Parity Ports for hand-eye Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional pan-tilt-head **warm-start seed** to the wrist hand-eye solver (basin-immune, removes the rotation-diversity requirement for *seeding*, never injects head bias into the final X), then port the highest-value robustness features the proven `pan_tilt` calibration has and `handeye_calib` lacks (consensus PnP averaging, observability enforcement, default-on per-axis MAD rejection).

**Architecture:** The wrist eye-in-hand solve stays the backbone (fixed ChArUco board, arm sweeps, `cv2.calibrateHandEye` seed → bundle-adjust). The head Orbbec — already calibrated to ~3 mm/0.5° in `base_link` — observes the *same fixed board* and supplies `T_base_board` (the solver's nuisance parameter `Tbb`). That measured board pose becomes one **closed-form seed candidate** (`X_i = inv(A_i)·Tbb_head·inv(B_i)`, SE(3)-averaged) fed into a **multi-start** bundle-adjust alongside the existing closed-form seed; the lowest post-BA reprojection wins. `Tbb` stays a FREE bundle-adjust parameter, so the head's absolute bias is used only to pick the convergence basin, **not** to set the final accuracy (the wrist reprojection + FFS depth still own that). The parity ports are independent pure-logic additions to `handeye_solve.py` plus thin node wiring.

**Tech Stack:** Python 3.10, numpy, scipy (`least_squares`, `Rotation`), OpenCV (`cv2.aruco`, `solvePnP`), rclpy + tf2_ros + FastAPI (node layer), pytest. Reuses `pan_tilt.calibration.aruco_detect` (already imported by `handeye_web`).

## Global Constraints

- **Working directories & paths (authoritative):** the git repo root is `/home/tinker/tk25_ws/src/tk26_vision` (`tk25_ws` is NOT a git repo — it is the colcon workspace root). Run **build + pytest from `/home/tinker/tk25_ws`** (after sourcing the venv + `install/setup.bash`); run **all `git` commands from `/home/tinker/tk25_ws/src/tk26_vision`**. `Files:` bullets are paths relative to the git repo root (`src/handeye_calib/...`); command-line paths are absolute. `handeye_calib` is symlink-installed at `/home/tinker/tk25_ws/install/handeye_calib`.
- **Venv:** all Python runs under `src/tk26_vision/.venv-vision-main/`. Activate before any pytest/build.
- **Build wrapper:** `./src/tk26_vision/scripts/build.sh --packages-select handeye_calib` (plain `colcon build` writes `#!/usr/bin/python3` shebangs that can't see the venv). With `--symlink-install`, **edits to existing `.py` files and new `.py` files inside an already-built package are picked up live — no rebuild needed for pure-Python changes**; only rebuild once up front (Task 0) and again only if entry points change.
- **Pure modules stay ROS-free:** `handeye_solve.py`, `transforms.py`, `web_support.py`, `synthetic.py` must import only numpy/scipy/cv2 — no rclpy/fastapi (mirrors the existing import discipline so they unit-test under the plain venv).
- **Dependency direction:** do NOT add a `tk26_vision → tk25_basic` code dependency. The head warm-start reads the head pose via **TF at runtime** (`tf2_ros`), never by importing pan_tilt's URDF/config.
- **Frames:** arm base = `link_base` (the node's `self._base_frame`), flange = `link_eef` (`self._eef_frame`). The head pose is looked up relative to `self._base_frame` so TF chains through `base_link` automatically; **never** hard-code `base_link` in the lookup.
- **Back-compat:** every new solver parameter defaults to preserve current behavior on the monocular/no-anchor path, EXCEPT the two intentional default changes in Task 7 (`reject_sigma`, `max_reject_frac`) which are called out explicitly.
- **No head bias into X:** the head anchor is a SEED and a FREE-`Tbb` initial value only. Never hard-pin `Tbb` to the head value. (Verified rationale: the head's base-frame error is systematic ~1 cm at the real 1.1–1.4 m head→board slant; hard-anchoring would inject it into X.)
- **TDD, DRY, YAGNI, frequent commits.** Each task ends with a green test and a commit.

---

## File Structure

- `src/handeye_calib/handeye_calib/handeye_solve.py` — **all** new pure solver logic: `seed_from_board_anchor`, `average_board_anchors`, `_solve_once` (multi-start), `rotation_observability`, `consensus_corners`, `_per_sample_chain_errors`, `_modified_zscores`, and the rewritten `solve()` rejection loop.
- `src/handeye_calib/handeye_calib/synthetic.py` — add `rot_range` param to `make_scenario` (enables a low-diversity degenerate scenario for the warm-start test).
- `src/handeye_calib/handeye_calib/handeye_web.py` — node wiring only: head camera subscriptions, `do_anchor_board()`, anchor state, `/api/anchor` endpoint, consensus capture history, pass-through of `anchor_Tbb`/observability to `do_solve`.
- `src/handeye_calib/handeye_calib/webui/{index.html,app.js}` — one "Anchor board (head)" button + status line + observability badge.
- `src/handeye_calib/test/test_solve_anchor.py` — new: warm-start + multi-start + observability tests.
- `src/handeye_calib/test/test_consensus.py` — new: `consensus_corners` tests.
- `src/handeye_calib/test/test_solve_reject.py` — new: default-on per-axis MAD tests.
- `src/handeye_calib/test/test_web_app.py` — extend: `/api/anchor` graceful-degradation.
- `src/handeye_calib/README.md` — changelog entry (folded into the last task).

---

## Task 0: Baseline — build once and confirm the suite is green

**Files:** none (environment check).

- [ ] **Step 1: Activate venv + source workspace, build handeye_calib once**

Run:
```bash
cd /home/tinker/tk25_ws
source src/tk26_vision/.venv-vision-main/bin/activate
./src/tk26_vision/scripts/build.sh --packages-select handeye_calib
source install/setup.bash
```
Expected: build SUCCESS for `handeye_calib`.

- [ ] **Step 2: Run the existing hand-eye suite to confirm a clean baseline**

Run:
```bash
pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/ -q
```
Expected: all tests PASS (baseline before any change). If `test_import` fails, the workspace isn't sourced — re-run `source install/setup.bash`.

- [ ] **Step 3: Commit nothing — this is a checkpoint only.** Proceed to Task 1.

---

## Task 1: Pure warm-start primitives (`seed_from_board_anchor`, `average_board_anchors`) + `make_scenario(rot_range=...)`

**Files:**
- Modify: `src/handeye_calib/handeye_calib/handeye_solve.py` (add two functions after `seed_handeye`, ~line 75)
- Modify: `src/handeye_calib/handeye_calib/synthetic.py:31` (`make_scenario` signature + the rotation line ~61)
- Test: `src/handeye_calib/test/test_solve_anchor.py` (new)

**Interfaces:**
- Produces:
  - `handeye_solve.seed_from_board_anchor(samples, anchor_Tbb) -> (X_seed: np.ndarray(4,4), Tbb_seed: np.ndarray(4,4))`
  - `handeye_solve.average_board_anchors(anchors: list[np.ndarray]) -> (Tbb_mean: np.ndarray(4,4), scatter: dict{"trans_mm","rot_deg","n"})`
  - `synthetic.make_scenario(..., rot_range: float = 0.6)` — unchanged default behavior.
- Consumes: `transforms` (`tf.invert`, `tf.se3_average`, `tf.rotation_angle_deg`), already imported in `handeye_solve` as `tf`.

- [ ] **Step 1: Write the failing test**

Create `src/handeye_calib/test/test_solve_anchor.py`:
```python
import numpy as np
from handeye_calib import synthetic as syn, handeye_solve as hs, transforms as tf


def test_seed_from_board_anchor_recovers_X_with_zero_rotation_diversity():
    # Near pure-translation set: AX=XB is rank-deficient, but a known board
    # pose in base determines X from a single pose, so the anchor seed recovers
    # X even here.
    sc = syn.make_scenario(n_poses=8, pixel_noise=0.0, seed=3, rot_range=0.02)
    X_seed, Tbb_seed = hs.seed_from_board_anchor(sc.samples, sc.Tbb_true)
    dt = np.linalg.norm(X_seed[:3, 3] - sc.X_true[:3, 3]) * 1000.0
    dr = tf.rotation_angle_deg(X_seed[:3, :3], sc.X_true[:3, :3])
    assert dt < 1.0 and dr < 0.2          # sub-mm / sub-0.2deg from an exact anchor
    assert np.allclose(Tbb_seed, sc.Tbb_true)


def test_average_board_anchors_reports_scatter():
    sc = syn.make_scenario(n_poses=4, pixel_noise=0.0, seed=1)
    rng = np.random.default_rng(0)
    # Three noisy observations of the same true board pose.
    obs = []
    for _ in range(3):
        noise = tf.T_from_vec(np.concatenate([
            rng.normal(0, np.radians(0.3), 3), rng.normal(0, 0.004, 3)]))
        obs.append(sc.Tbb_true @ noise)
    mean, scatter = hs.average_board_anchors(obs)
    assert scatter["n"] == 3
    assert 0.0 < scatter["trans_mm"] < 20.0
    assert 0.0 < scatter["rot_deg"] < 2.0
    # mean is close to truth (noise averages partly down)
    assert np.linalg.norm(mean[:3, 3] - sc.Tbb_true[:3, 3]) * 1000.0 < 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py -v`
Expected: FAIL — `AttributeError: module 'handeye_calib.handeye_solve' has no attribute 'seed_from_board_anchor'` (and `make_scenario` rejecting `rot_range`).

- [ ] **Step 3: Add `rot_range` to `make_scenario`**

In `src/handeye_calib/handeye_calib/synthetic.py`, change the signature (line ~31) from:
```python
def make_scenario(n_poses=15, pixel_noise=0.3, seed=0,
                  squares_x=5, squares_y=5, square_len=0.04,
                  with_depth=False, depth_noise=0.0):
```
to:
```python
def make_scenario(n_poses=15, pixel_noise=0.3, seed=0,
                  squares_x=5, squares_y=5, square_len=0.04,
                  with_depth=False, depth_noise=0.0, rot_range=0.6):
```
and change the rotation-sampling line (~line 61) from:
```python
        rot = R.from_euler('xyz', rng.uniform(-0.6, 0.6, 3)).as_matrix()
```
to:
```python
        rot = R.from_euler('xyz', rng.uniform(-rot_range, rot_range, 3)).as_matrix()
```

- [ ] **Step 4: Add the two functions to `handeye_solve.py`**

Insert after `seed_handeye` (after line ~74), before `_residuals`:
```python
def seed_from_board_anchor(samples, anchor_Tbb):
    """Closed-form warm-start ``X = T_eef_cam`` from a KNOWN board pose in base.

    The board pose in the arm-base frame (``anchor_Tbb`` = T_base_board) is
    measured by an EXTERNAL, already-calibrated sensor (the pan-tilt head
    Orbbec, composed through TF into the arm base frame). Each sample then
    closes the kinematic loop directly::

        A_i @ X @ B_i = Tbb   =>   X_i = inv(A_i) @ Tbb @ inv(B_i)

    with A_i = T_base_eef (FK) and B_i = T_cam_board (wrist PnP). Unlike AX=XB
    this needs NO rotation diversity — a single pose determines X — so it is a
    basin-immune seed for the bundle adjust. Returns ``Tbb_seed = anchor_Tbb``;
    the bundle adjust keeps Tbb FREE, so the head's absolute bias is used only
    to choose the convergence basin and is NOT injected into the final X.
    """
    Xs = []
    for s in samples:
        A = np.asarray(s.T_base_eef, float)
        B = np.asarray(s.T_cam_board, float)
        Xs.append(tf.invert(A) @ np.asarray(anchor_Tbb, float) @ tf.invert(B))
    return tf.se3_average(Xs), np.asarray(anchor_Tbb, float)


def average_board_anchors(anchors):
    """SE(3)-average a list of board-in-base measurements and report scatter.

    ``anchors`` is a list of 4x4 ``T_base_board`` observations (e.g. the head
    at several pan/tilt poses). Returns ``(Tbb_mean, scatter)`` where scatter is
    ``{"trans_mm", "rot_deg", "n"}`` — the RMS deviation of the observations
    from their mean, a data-driven confidence readout (large scatter => the
    anchor is unreliable; widen the prior / re-check the head TF).
    """
    Ts = [np.asarray(T, float) for T in anchors]
    if not Ts:
        raise ValueError("average_board_anchors needs >=1 anchor")
    mean = tf.se3_average(Ts)
    inv_mean = tf.invert(mean)
    t_dev, r_dev = [], []
    for T in Ts:
        D = inv_mean @ T
        t_dev.append(float(np.linalg.norm(D[:3, 3])))
        r_dev.append(np.radians(tf.rotation_angle_deg(np.eye(3), D[:3, :3])))
    scatter = {
        "trans_mm": float(np.sqrt(np.mean(np.square(t_dev))) * 1000.0),
        "rot_deg": float(np.degrees(np.sqrt(np.mean(np.square(r_dev))))),
        "n": len(Ts),
    }
    return mean, scatter
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/handeye_solve.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/synthetic.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py
git commit -m "feat(handeye): board-anchor warm-start primitives + make_scenario rot_range"
```

---

## Task 2: Multi-start solve (`_solve_once`) + `solve(anchor_Tbb=...)`

**Files:**
- Modify: `src/handeye_calib/handeye_calib/handeye_solve.py` (add `_solve_once`; rewire `solve()` initial solve + add `anchor_Tbb` param at line ~223)
- Test: `src/handeye_calib/test/test_solve_anchor.py` (append)

**Interfaces:**
- Produces:
  - `handeye_solve._solve_once(samples, K, dist, board_pts, *, methods=None, depth_weight=0.0, depth_sigma_m=0.005, anchor_Tbb=None) -> (X, Tbb, per_method, seed_used: str)`
  - `handeye_solve.solve(..., anchor_Tbb=None)` — new keyword, defaults to current behavior.
- Consumes: `seed_handeye`, `seed_from_board_anchor` (Task 1), `bundle_adjust`.

- [ ] **Step 1: Write the failing test (append to `test_solve_anchor.py`)**

```python
def test_anchor_rescues_degenerate_solve():
    # Low rotation diversity => calibrateHandEye seed is poorly conditioned.
    sc = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=7, rot_range=0.05)
    plain = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25,
                     reject_sigma=None)
    # Simulate a realistic (slightly noisy) head anchor: ~5 mm / 0.3 deg off.
    rng = np.random.default_rng(2)
    anchor = sc.Tbb_true @ tf.T_from_vec(np.concatenate([
        rng.normal(0, np.radians(0.3), 3), rng.normal(0, 0.005, 3)]))
    assisted = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25,
                        reject_sigma=None, anchor_Tbb=anchor)
    err_plain = np.linalg.norm(plain.X[:3, 3] - sc.X_true[:3, 3])
    err_assisted = np.linalg.norm(assisted.X[:3, 3] - sc.X_true[:3, 3])
    # The anchor-assisted X is dramatically better on a degenerate set, and
    # within the head's ~1 cm floor (NOT necessarily the 3 mm gate).
    assert err_assisted < err_plain
    assert err_assisted < 0.012


def test_solve_default_no_anchor_is_unchanged():
    # anchor_Tbb=None must reproduce the historical clean-data PASS.
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25,
                   rng_seed=0, reject_sigma=None)
    assert res.status == "PASS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py::test_anchor_rescues_degenerate_solve -v`
Expected: FAIL — `solve() got an unexpected keyword argument 'anchor_Tbb'`.

- [ ] **Step 3: Add `_solve_once` and wire it into `solve()`**

Insert `_solve_once` immediately after `bundle_adjust` (after line ~134):
```python
def _solve_once(samples, K, dist, board_pts, *, methods=None,
                depth_weight=0.0, depth_sigma_m=0.005, anchor_Tbb=None):
    """Multi-start seed -> bundle-adjust; return (X, Tbb, per_method, seed_used).

    Candidate seeds: the best-of-5 closed-form ``calibrateHandEye`` seed, plus
    (when ``anchor_Tbb`` is given) the basin-immune board-anchor seed. Each is
    bundle-adjusted; the converged result with the lowest reprojection RMS wins.
    On degenerate (low-rotation) sets where calibrateHandEye returns a poor or
    flipped seed, the anchor branch rescues the solve.
    """
    X0, Tbb0, per_method = seed_handeye(samples, K, dist, board_pts, methods=methods)
    candidates = [("closed_form", X0, Tbb0)]
    if anchor_Tbb is not None:
        Xa, Tba = seed_from_board_anchor(samples, anchor_Tbb)
        candidates.append(("board_anchor", Xa, Tba))
    best = None
    for name, Xs, Tbs in candidates:
        X, Tbb, info = bundle_adjust(samples, K, dist, board_pts, Xs, Tbs,
                                     depth_weight=depth_weight,
                                     depth_sigma_m=depth_sigma_m)
        reproj = info["final_reproj_px"]
        if best is None or reproj < best[3]:
            best = (X, Tbb, name, reproj)
    return best[0], best[1], per_method, best[2]
```

In `solve()` (signature at line ~223), add the `anchor_Tbb` keyword:
```python
def solve(samples, K, dist, board_pts, heldout_frac=0.2, rng_seed=0, *,
          methods=None, reject_sigma=None, max_reject_frac=0.5,
          depth_weight=1.0, depth_sigma_m=0.005, anchor_Tbb=None):
```
Replace the initial seed+BA lines (currently lines ~243-245):
```python
    X0, Tbb0, per_method = seed_handeye(train, K, dist, board_pts, methods=methods)
    X, Tbb, _ = bundle_adjust(train, K, dist, board_pts, X0, Tbb0,
                              depth_weight=depth_weight, depth_sigma_m=depth_sigma_m)
```
with:
```python
    X, Tbb, per_method, _seed_used = _solve_once(
        train, K, dist, board_pts, methods=methods,
        depth_weight=depth_weight, depth_sigma_m=depth_sigma_m,
        anchor_Tbb=anchor_Tbb)
```
(The existing `reject_sigma` loop below stays as-is for now; Task 7 rewrites it. Leave it untouched in this task.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py -v`
Expected: all four PASS.

- [ ] **Step 5: Run the full suite — no regressions**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/ -q`
Expected: all PASS (the `reject_sigma` loop is unchanged; default solve path uses `_solve_once` with a single closed-form candidate == prior behavior).

- [ ] **Step 6: Commit**

```bash
git add /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/handeye_solve.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py
git commit -m "feat(handeye): multi-start solve with optional head board-anchor seed"
```

---

## Task 3: Node wiring — head camera observation + `do_anchor_board()` + `/api/anchor`

**Files:**
- Modify: `src/handeye_calib/handeye_calib/handeye_web.py` (`__init__` params/subscriptions ~line 420; new `_on_head_image`/`_on_head_info`/`do_anchor_board`/`do_clear_anchor`; `do_solve` pass-through ~line 1721; state surface)
- Modify: `src/handeye_calib/handeye_calib/webui/index.html` + `webui/app.js` (one button + status)
- Test: `src/handeye_calib/test/test_web_app.py` (append graceful-degradation test)

**Interfaces:**
- Consumes: `handeye_solve.average_board_anchors`, `handeye_solve.seed_from_board_anchor` (via `solve(anchor_Tbb=...)`); `pan_tilt.calibration.aruco_detect.detect_pose` (IPPE-seed PnP + disambiguation, already importable); `web_support.tf_to_matrix`.
- Produces: `HandeyeWebNode.do_anchor_board() -> dict`, `HandeyeWebNode.do_clear_anchor() -> dict`, `self._tbb_head: np.ndarray|None`, `self._anchor_obs: list[np.ndarray]`, `self._anchor_scatter: dict|None`; POST `/api/anchor`, POST `/api/anchor/clear`.

- [ ] **Step 1: Write the failing test (append to `test_web_app.py`)**

```python
def test_anchor_endpoint_graceful_without_head_camera():
    node, c = _client()
    try:
        r = c.post("/api/anchor")
        assert r.status_code == 200
        body = r.json()
        # No head frames have arrived -> ok:False with a clear reason, never 500.
        assert body["ok"] is False
        assert "head" in body["reason"].lower()
        assert body["n_anchor_obs"] == 0
    finally:
        node.destroy_node()


def test_anchor_clear_is_idempotent():
    node, c = _client()
    try:
        r = c.post("/api/anchor/clear")
        assert r.status_code == 200 and r.json()["ok"] is True
        assert r.json()["n_anchor_obs"] == 0
    finally:
        node.destroy_node()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_app.py::test_anchor_endpoint_graceful_without_head_camera -v`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Add head params + subscriptions + state in `__init__`**

In `HandeyeWebNode.__init__`, after the `_camera_node_name` block (~line 421), add:
```python
            # ---- pan-tilt HEAD Orbbec warm-start anchor ---------------------
            # The head is already calibrated (~3 mm/0.5deg in base_link). It
            # observes the SAME fixed board and supplies T_base_board, used ONLY
            # as a basin-immune SEED for the wrist solve (handeye_solve.solve
            # anchor_Tbb=...). Tbb stays FREE in the bundle adjust, so the head's
            # absolute bias never enters the final X. Disabled until the first
            # successful anchor; head camera defaults are the Orbbec /camera ns.
            self._head_image_topic = str(self._param("head_image_topic", "/camera/color/image_raw"))
            self._head_info_topic = str(self._param("head_info_topic", "/camera/color/camera_info"))
            self._head_optical_frame = str(self._param("head_optical_frame", "camera_color_optical_frame"))
            self._head_frame = None
            self._head_frame_stamp = None
            self._head_K = None
            self._head_D = None
            self._tbb_head = None          # 4x4 averaged T_base_board, or None
            self._anchor_obs = []          # list of 4x4 per-snap T_base_board
            self._anchor_scatter = None    # {"trans_mm","rot_deg","n"} or None
```
Then, where the color/info subscriptions are created (search for `self.create_subscription(` for the color image; add alongside, before the `_on_joint_state` sub or just after the active-frame subs), add:
```python
            self.create_subscription(
                Image, self._head_image_topic, self._on_head_image,
                qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, self._head_info_topic, self._on_head_info, 10)
```
(`Image`, `CameraInfo`, `qos_profile_sensor_data` are already imported/used by the existing wrist subscriptions — reuse them; confirm by grepping `qos_profile_sensor_data` near the wrist `create_subscription`.)

- [ ] **Step 4: Add the head callbacks + anchor methods**

Add as methods on `HandeyeWebNode` (place after `_on_ir_info`, ~line 786):
```python
        def _on_head_image(self, msg):
            """Cache the latest HEAD Orbbec color frame (warm-start anchor only;
            does NOT drive the wrist detection/stability path)."""
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"head cv_bridge failed ({exc})",
                                       throttle_duration_sec=10.0)
                return
            with self.lock:
                self._head_frame = bgr
                self._head_frame_stamp = self.get_clock().now().to_msg()

        def _on_head_info(self, msg):
            np = self._np
            K = np.array(msg.k, float).reshape(3, 3)
            D = np.array(msg.d, float).flatten() if len(msg.d) else np.zeros(5)
            with self.lock:
                self._head_K = K
                self._head_D = D

        def do_anchor_board(self):
            """Observe the fixed board with the HEAD Orbbec and record one
            T_base_board sample for the warm-start. Call multiple times (ideally
            from a few different pan/tilt head poses) to average down the head's
            pose-dependent bias; the running mean + scatter is stored on
            ``self._tbb_head`` / ``self._anchor_scatter``. Degrades to ok:False
            (never 500) when the head frame / intrinsics / TF / detection are
            missing."""
            from pan_tilt.calibration.aruco_detect import detect_pose
            with self.lock:
                bgr = None if self._head_frame is None else self._head_frame.copy()
                K = None if self._head_K is None else self._head_K.copy()
                D = None if self._head_D is None else self._head_D.copy()
                stamp = self._head_frame_stamp
                n_obs = len(self._anchor_obs)
            if bgr is None or K is None:
                return {"ok": False, "reason": "no head camera frame/intrinsics yet",
                        "n_anchor_obs": n_obs}
            det = detect_pose(bgr, K, D if D is not None else self._np.zeros(5),
                              board=self._board, detector=self._detector)
            if not det.success:
                return {"ok": False, "reason": "head saw no usable board",
                        "n_anchor_obs": n_obs}
            # detect_pose returns the board pose in the head OPTICAL frame.
            from rclpy.time import Time as _RclpyTime
            tf_time = (_RclpyTime.from_msg(stamp) if stamp is not None
                       else self._rclpy_time())
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self._base_frame, self._head_optical_frame, tf_time)
            except Exception:
                try:
                    tfm = self.tf_buffer.lookup_transform(
                        self._base_frame, self._head_optical_frame, self._rclpy_time())
                except Exception as exc2:
                    return {"ok": False,
                            "reason": (f"TF {self._base_frame}->"
                                       f"{self._head_optical_frame} unavailable: {exc2}"),
                            "n_anchor_obs": n_obs}
            T_base_headopt = ws.tf_to_matrix(
                [tfm.transform.translation.x, tfm.transform.translation.y,
                 tfm.transform.translation.z],
                [tfm.transform.rotation.x, tfm.transform.rotation.y,
                 tfm.transform.rotation.z, tfm.transform.rotation.w])
            Tbb_obs = T_base_headopt @ det.pose_optical
            with self.lock:
                self._anchor_obs.append(Tbb_obs)
                mean, scatter = hs.average_board_anchors(self._anchor_obs)
                self._tbb_head = mean
                self._anchor_scatter = scatter
                n_obs = len(self._anchor_obs)
            return {"ok": True, "n_anchor_obs": n_obs, "scatter": scatter,
                    "reproj_px": float(det.reprojection_rms_px)}

        def do_clear_anchor(self):
            with self.lock:
                self._anchor_obs = []
                self._tbb_head = None
                self._anchor_scatter = None
            return {"ok": True, "n_anchor_obs": 0}
```

- [ ] **Step 5: Pass the anchor into `do_solve`**

In `do_solve` (~line 1705), capture the anchor under the lock and forward it:
```python
            with self.lock:
                samples, K, D = list(self.session.samples), self._K, self._D
                anchor = None if self._tbb_head is None else self._tbb_head.copy()
```
and in the `hs.solve(...)` call (~line 1721) add the keyword:
```python
                res = hs.solve(samples, K, D, self._board_pts,
                               methods=methods_subset,
                               reject_sigma=(float(reject_sigma)
                                             if reject_sigma is not None else None),
                               depth_weight=self._depth_weight,
                               depth_sigma_m=self._depth_sigma_m,
                               anchor_Tbb=anchor)
```

- [ ] **Step 6: Register the endpoints**

In `make_app` near the existing `/api/solve` route (~line 2746), add:
```python
    @app.post("/api/anchor")
    async def anchor(request: Request):
        return JSONResponse(ws.json_safe(node.do_anchor_board()))

    @app.post("/api/anchor/clear")
    async def anchor_clear(request: Request):
        return JSONResponse(ws.json_safe(node.do_clear_anchor()))
```

- [ ] **Step 7: Surface anchor status in the WS state**

Find where `enriched_state_payload(...)` is called in `get_state_dict`/`_push_state` and add the anchor info to the dict the UI receives. Locate the call, then immediately after building the base payload add:
```python
            payload["anchor"] = {
                "have": self._tbb_head is not None,
                "n_obs": len(self._anchor_obs),
                "scatter": self._anchor_scatter,
            }
```
(If the state is assembled via `enriched_state_payload`'s return value `base`, mutate that dict before returning it; mirror how `rejected_indices` is attached in `do_solve`.)

- [ ] **Step 8: Add the UI button (best-effort; covered by manual check)**

In `webui/index.html`, in the Capture tab near the manual-capture button, add:
```html
      <div class="row" id="anchor-row">
        <button id="btn-anchor">Anchor board (head)</button>
        <button id="btn-anchor-clear">Clear anchor</button>
        <span id="anchor-status" class="status">no head anchor</span>
      </div>
```
In `webui/app.js`, near the other button handlers, add:
```javascript
  const btnAnchor = document.getElementById('btn-anchor');
  if (btnAnchor) btnAnchor.onclick = async () => {
    const r = await fetch('/api/anchor', {method: 'POST'});
    const j = await r.json();
    setStatus('anchor-status', j.ok
      ? `head anchor: ${j.n_anchor_obs} obs (scatter ${j.scatter ? j.scatter.trans_mm.toFixed(1) : '?'}mm)`
      : `anchor failed: ${j.reason}`);
  };
  const btnAnchorClear = document.getElementById('btn-anchor-clear');
  if (btnAnchorClear) btnAnchorClear.onclick = async () => {
    await fetch('/api/anchor/clear', {method: 'POST'});
    setStatus('anchor-status', 'no head anchor');
  };
```
(`setStatus` is the existing helper used by other buttons; if its name differs, grep `app.js` for the status-setter the capture button uses and match it.)

- [ ] **Step 9: Run the web tests**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_app.py -v`
Expected: existing tests still PASS; the two new anchor tests PASS.

- [ ] **Step 10: Commit**

```bash
git add /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/webui/index.html \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/webui/app.js \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_app.py
git commit -m "feat(handeye_web): head Orbbec board-anchor warm-start (capture + /api/anchor + solve wiring)"
```

- [ ] **Step 11: Manual hardware verification note (record outcome in DEV_NOTES.md)**

On the robot, with the wrist RealSense, head Orbbec, RSP + `/joint_states` all up and the board fixed and co-visible: confirm `/tf <base_frame> -> <head_optical_frame>` tracks the live head angle (guards the JSP zero-clobber bug), click **Anchor board (head)** from 2–3 head poses, then **Solve**, and confirm the solve log shows the board-anchor seed being used on a sparse set. **Pre-flight gotchas to check:** the deployed head body frame is named `camera_link` (same as the Orbbec driver's own) — confirm RSP owns `base_link->camera_link` and the driver only publishes optical children, or TF will have two parents; and confirm the head ChArUco origin matches the wrist board (same `DICT_5X5_100` / 5×5 / 40 mm). Defer to memory `handeye-head-orbbec-assist-verdict` for the prerequisites checklist.

---

## Task 4: Pure consensus PnP corner averaging (`consensus_corners`)

**Files:**
- Modify: `src/handeye_calib/handeye_calib/handeye_solve.py` (add `consensus_corners`)
- Test: `src/handeye_calib/test/test_consensus.py` (new)

**Interfaces:**
- Produces: `handeye_solve.consensus_corners(frames: list[tuple[ids, px]], *, min_frac=0.6) -> (ids: np.ndarray|None, px: np.ndarray(M,2)|None)`

- [ ] **Step 1: Write the failing test**

Create `src/handeye_calib/test/test_consensus.py`:
```python
import numpy as np
from handeye_calib import handeye_solve as hs


def test_consensus_corners_denoises_toward_truth():
    rng = np.random.default_rng(0)
    truth = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
    ids = np.array([0, 1, 2, 3])
    frames = [(ids, truth + rng.normal(0, 0.5, truth.shape)) for _ in range(10)]
    out_ids, out_px = hs.consensus_corners(frames)
    assert list(out_ids) == [0, 1, 2, 3]
    # consensus error well below any single frame's ~0.5 px noise
    assert np.max(np.linalg.norm(out_px - truth, axis=1)) < 0.3


def test_consensus_drops_below_quorum_corners():
    ids_full = np.array([0, 1, 2, 3])
    px = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    # corner id 3 appears in only 1 of 10 frames -> below 60% quorum, dropped.
    frames = [(np.array([0, 1, 2]), px[:3]) for _ in range(9)]
    frames.append((ids_full, px))
    out_ids, out_px = hs.consensus_corners(frames)
    assert list(out_ids) == [0, 1, 2]


def test_consensus_returns_none_when_too_few():
    out_ids, out_px = hs.consensus_corners([])
    assert out_ids is None and out_px is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_consensus.py -v`
Expected: FAIL — `module ... has no attribute 'consensus_corners'`.

- [ ] **Step 3: Add `consensus_corners`**

Append to `handeye_solve.py`:
```python
def consensus_corners(frames, *, min_frac=0.6):
    """Per-corner sub-pixel consensus across N steady frames.

    ``frames`` is a list of ``(ids, px)`` for one frame each: ``ids`` is an
    ``(M,)`` int array of ChArUco corner indices, ``px`` an ``(M,2)`` array of
    sub-pixel corner pixels. A corner id is kept only if it was detected in at
    least ``ceil(min_frac * N)`` frames; its consensus pixel is the per-corner
    MEDIAN over the frames that saw it (robust to the occasional mis-localized
    corner). Returns ``(ids, px)`` sorted by id, or ``(None, None)`` when fewer
    than 4 corners reach quorum (caller falls back to the single-frame pose).
    """
    n = len(frames)
    if n == 0:
        return None, None
    quorum = max(1, int(np.ceil(min_frac * n)))
    acc = {}
    for ids, px in frames:
        ids = np.asarray(ids).reshape(-1).astype(int)
        px = np.asarray(px, float).reshape(-1, 2)
        for cid, p in zip(ids, px):
            acc.setdefault(int(cid), []).append(p)
    kept_ids, kept_px = [], []
    for cid in sorted(acc):
        pts = np.asarray(acc[cid], float)
        if len(pts) >= quorum:
            kept_ids.append(cid)
            kept_px.append(np.median(pts, axis=0))
    if len(kept_ids) < 4:
        return None, None
    return np.asarray(kept_ids, int), np.asarray(kept_px, float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_consensus.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/handeye_solve.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_consensus.py
git commit -m "feat(handeye): per-corner consensus pixel averaging (consensus_corners)"
```

---

## Task 5: Wire consensus into capture (history ring + IPPE-seed re-PnP)

**Files:**
- Modify: `src/handeye_calib/handeye_calib/handeye_web.py` (`__init__` history deque; `_ingest_frame` push; `do_capture` consensus + IPPE-seed PnP; new `_pnp_ippe_refine` helper)
- Test: `src/handeye_calib/test/test_web_app.py` (append)

**Interfaces:**
- Consumes: `handeye_solve.consensus_corners` (Task 4); `self._board.matchImagePoints`; `cv2.solvePnP` with `SOLVEPNP_IPPE` then `SOLVEPNP_ITERATIVE`.
- Produces: `self._det_history: collections.deque`; `do_capture` stores a consensus-averaged `T_cam_board`/`obs_px` when history is available, falling back to single-frame `self._cap` otherwise; `_pnp_ippe_refine(obj_pts, img_pts, K, D) -> (T_cam_board, reproj_px) | (None, None)`.

- [ ] **Step 1: Write the failing test (append to `test_web_app.py`)**

```python
def test_pnp_ippe_refine_recovers_known_pose():
    import numpy as np
    node, c = _client()
    try:
        # Project the node's own board with a known pose, then recover it.
        from handeye_calib import handeye_model as hm
        bp = node._board_pts
        K = np.array([[615., 0, 320.], [0, 615., 240.], [0, 0, 1.]])
        T = np.eye(4); T[:3, 3] = [0.02, -0.01, 0.5]
        px = hm.project_corners(bp, T, K)
        T_rec, reproj = node._pnp_ippe_refine(bp.astype(float), px.astype(float), K, np.zeros(5))
        assert T_rec is not None
        assert np.linalg.norm(T_rec[:3, 3] - T[:3, 3]) < 1e-3
        assert reproj < 0.5
    finally:
        node.destroy_node()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_app.py::test_pnp_ippe_refine_recovers_known_pose -v`
Expected: FAIL — `'HandeyeWebNode' object has no attribute '_pnp_ippe_refine'`.

- [ ] **Step 3: Add the history deque in `__init__`**

After `self._cap = None` (~line 441) add:
```python
            # Rolling buffer of recent (ids, px) detections for multi-frame
            # consensus at capture time (pan_tilt parity: cluster_consensus).
            # Only pushed while a board pose is present; reset on lost detection.
            self._consensus_frames = int(self._param("consensus_frames", 10))
            self._consensus_min_frac = float(self._param("consensus_min_frac", 0.6))
            self._det_history = collections.deque(maxlen=self._consensus_frames)
```

- [ ] **Step 4: Push detections in `_ingest_frame`**

In `_ingest_frame`, inside the `if cap is not None:` branch (~line 693), record the per-frame corners; and reset on loss. Change:
```python
            if cap is not None:
                steady = self._stability.update(cap["T_cam_board"])
            else:
                self._stability.reset()
                steady = False
```
to:
```python
            if cap is not None:
                steady = self._stability.update(cap["T_cam_board"])
                self._det_history.append(
                    (np.asarray(cap["corner_idx"]).copy(),
                     np.asarray(cap["obs_px"], float).copy()))
            else:
                self._stability.reset()
                self._det_history.clear()
                steady = False
```
(`np` here is the module-level numpy alias used throughout `_ingest_frame`; it's available as `self._np` — use `self._np` if `np` is not in local scope. Grep the function to confirm; the file consistently uses `self._np` inside methods, so write `self._np.asarray(...)`.)

- [ ] **Step 5: Add `_pnp_ippe_refine` and use consensus in `do_capture`**

Add the helper method (place near `_detect`, ~line 1045):
```python
        def _pnp_ippe_refine(self, obj_pts, img_pts, K, D):
            """IPPE-seeded ITERATIVE PnP (planar two-fold-ambiguity safe).

            Returns (T_cam_board 4x4, reproj_px) or (None, None) on failure.
            Mirrors pan_tilt.aruco_detect._solve_iterative: IPPE seed picks the
            correct planar branch, ITERATIVE refines it."""
            np = self._np
            cv2 = self._cv2
            obj_pts = np.asarray(obj_pts, float).reshape(-1, 1, 3)
            img_pts = np.asarray(img_pts, float).reshape(-1, 1, 2)
            if len(obj_pts) < 6:
                return None, None
            try:
                n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE)
                if not n_sol:
                    return None, None
                rvec, tvec = rvecs[0], tvecs[0]
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, D, rvec, tvec, useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    return None, None
            except cv2.error:
                return None, None
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            reproj = float(np.sqrt(np.mean(np.sum(
                (proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2, axis=1))))
            T = np.eye(4)
            T[:3, :3] = cv2.Rodrigues(rvec)[0]
            T[:3, 3] = tvec.reshape(3)
            return T, reproj
```
In `do_capture`, after the steady gate and the snapshot block (after the `now_mono = self._last_frame_monotonic` line, ~line 1473) and BEFORE the FFS depth section, add the consensus replacement of `cap`:
```python
            # Multi-frame consensus (pan_tilt parity): average the last N steady
            # detections' corners and re-PnP, replacing the single-shot cap so
            # the stored obs_px AND T_cam_board are denoised. Falls back to the
            # single-frame cap when consensus can't reach quorum.
            with self.lock:
                hist = list(self._det_history)
            cons_ids, cons_px = hs.consensus_corners(
                hist, min_frac=self._consensus_min_frac)
            n_consensus = 0
            if cons_ids is not None and K is not None:
                try:
                    obj_pts, img_pts = self._board.matchImagePoints(
                        cons_px.reshape(-1, 1, 2).astype(np.float32),
                        cons_ids.reshape(-1, 1).astype(np.int32))
                except Exception:
                    obj_pts = None
                if obj_pts is not None and len(obj_pts) >= 6:
                    T_c, reproj_c = self._pnp_ippe_refine(
                        obj_pts, img_pts, K, (self._D if self._D is not None
                                              else np.zeros(5)))
                    if T_c is not None:
                        h, w = frame.shape[:2]
                        xs, ys = cons_px[:, 0], cons_px[:, 1]
                        area_frac = (float((xs.max() - xs.min()) *
                                           (ys.max() - ys.min())) / float(h * w))
                        cap = {"T_cam_board": T_c, "obs_px": cons_px,
                               "corner_idx": cons_ids, "reproj_px": reproj_c,
                               "area_frac": area_frac}
                        n_consensus = len(hist)
```
(`cap` is then used unchanged by the rest of `do_capture` — FFS deproject uses `cap["obs_px"]`, `session.try_add` uses `cap[...]`. The fallback keeps the original single-frame `cap`.) Add `"n_consensus_frames": n_consensus` to the final return dict:
```python
            return {"ok": ok, "reason": reason, "depth_source": depth_source,
                    "n_consensus_frames": n_consensus,
                    "num_samples": num}
```

- [ ] **Step 6: Run the web tests**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_app.py -v`
Expected: existing tests PASS; `test_pnp_ippe_refine_recovers_known_pose` PASS.

- [ ] **Step 7: Commit**

```bash
git add /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_app.py
git commit -m "feat(handeye_web): multi-frame consensus capture + IPPE-seeded re-PnP (pan_tilt parity)"
```

---

## Task 6: Observability diagnostic (`rotation_observability`) + surface in solve payload

**Files:**
- Modify: `src/handeye_calib/handeye_calib/handeye_solve.py` (add `rotation_observability`; import `Rotation`)
- Modify: `src/handeye_calib/handeye_calib/web_support.py` (`solve_payload_v2` adds an `observability` block)
- Modify: `src/handeye_calib/handeye_calib/handeye_web.py` (`do_solve` injects observability into the payload)
- Test: `src/handeye_calib/test/test_solve_anchor.py` (append) + `src/handeye_calib/test/test_web_support.py` (append)

**Interfaces:**
- Produces: `handeye_solve.rotation_observability(samples, *, min_singular=0.3) -> dict{"ok","n_axes","second_singular","detail"}`; `solve_payload_v2(...)` gains key `"observability"` when `samples` non-empty.
- Consumes: `scipy.spatial.transform.Rotation`.

- [ ] **Step 1: Write the failing tests**

Append to `test_solve_anchor.py`:
```python
def test_observability_flags_low_diversity_and_passes_diverse():
    diverse = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=4, rot_range=0.6)
    degen = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=4, rot_range=0.01)
    o_div = hs.rotation_observability(diverse.samples)
    o_deg = hs.rotation_observability(degen.samples)
    assert o_div["ok"] is True
    assert o_deg["ok"] is False
    assert o_deg["second_singular"] <= o_div["second_singular"]
```
Append to `test_web_support.py`:
```python
def test_solve_payload_v2_carries_observability():
    import numpy as np
    from handeye_calib import synthetic as syn, handeye_solve as hs, web_support as ws
    sc = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, reject_sigma=None)
    payload = ws.solve_payload_v2(res, sc.samples, sc.K, None, sc.board_pts)
    assert "observability" in payload
    assert "ok" in payload["observability"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py::test_observability_flags_low_diversity_and_passes_diverse \
       /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_support.py::test_solve_payload_v2_carries_observability -v
```
Expected: FAIL — missing `rotation_observability` / missing payload key.

- [ ] **Step 3: Add the import + function to `handeye_solve.py`**

At the top of `handeye_solve.py`, add after `from scipy.optimize import least_squares` (line 5):
```python
from scipy.spatial.transform import Rotation as _R
```
Append the function:
```python
def rotation_observability(samples, *, min_singular=0.3):
    """Diagnose AX=XB rotation observability of the accepted pose set.

    Eye-in-hand identifiability needs >= 2 non-parallel relative-rotation axes;
    a set whose flange rotations all share one axis (or are pure translation)
    leaves X's rotation unobservable and lets a rotation error in X hide in the
    board pose Tbb. We collect the unit axis of every pairwise relative rotation
    R_ij = R_j R_i^T (skipping pairs that rotate < 2 deg, which have no
    well-defined axis), stack them into a 3xK matrix, and SVD. The 2nd singular
    value measures how much the axes span a second dimension; below
    ``min_singular`` the set is effectively single-axis. Returns a JSON-safe
    dict; ``ok`` is the gate the UI shows as a WARN badge.
    """
    Rs = [np.asarray(s.T_base_eef, float)[:3, :3] for s in samples]
    axes = []
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            rv = _R.from_matrix(Rs[j] @ Rs[i].T).as_rotvec()
            ang = float(np.linalg.norm(rv))
            if np.degrees(ang) >= 2.0:
                axes.append(rv / ang)
    if len(axes) < 2:
        return {"ok": False, "n_axes": len(axes), "second_singular": 0.0,
                "detail": "fewer than 2 usable rotation axes — X rotation "
                          "unobservable; add poses that rotate the flange"}
    sv = np.linalg.svd(np.asarray(axes, float).T, compute_uv=False)
    second = float(sv[1]) if len(sv) >= 2 else 0.0
    ok = bool(second >= min_singular)
    return {"ok": ok, "n_axes": len(axes), "second_singular": second,
            "detail": ("rotation axes span >= 2 dimensions" if ok else
                       "rotation axes nearly collinear — add poses that rotate "
                       "the flange about a DIFFERENT axis")}
```

- [ ] **Step 4: Add the observability block to `solve_payload_v2`**

In `web_support.py`, inside `solve_payload_v2`, after computing `per_sample` (line ~129) and before `base.update({...})`, add:
```python
    observability = None
    if samples is not None and len(samples) > 0:
        from handeye_calib import handeye_solve as hs  # local import (ROS-free)
        observability = hs.rotation_observability(samples)
```
and add to the `base.update({...})` dict:
```python
        "observability": observability,
```

- [ ] **Step 5: (UI, best-effort) badge the observability WARN in `app.js`**

In `webui/app.js`, where the solve payload is rendered, add (concise):
```javascript
  if (data.observability && data.observability.ok === false) {
    setStatus('solve-status',
      'WARN: ' + data.observability.detail);
  }
```
(Match the actual solve-render function/status id by grepping `app.js` for where `heldout_metrics_mm_deg` is consumed.)

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py \
       /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_support.py -v
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/handeye_solve.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/web_support.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/webui/app.js \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_anchor.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_web_support.py
git commit -m "feat(handeye): AX=XB rotation-observability diagnostic surfaced in solve payload"
```

---

## Task 7: Default-on, per-axis MAD outlier rejection on the SE(3) chain error

**Files:**
- Modify: `src/handeye_calib/handeye_calib/handeye_solve.py` (add `_per_sample_chain_errors`, `_modified_zscores`; rewrite the rejection loop in `solve()`; change defaults)
- Modify: `src/handeye_calib/handeye_calib/handeye_web.py` (`do_solve` default sentinel so the operator can override but the default is active)
- Test: `src/handeye_calib/test/test_solve_reject.py` (new)

**Interfaces:**
- Produces: `handeye_solve._per_sample_chain_errors(X, Tbb, samples) -> (trans_m: np.ndarray, rot_rad: np.ndarray)`; `handeye_solve._modified_zscores(arr) -> np.ndarray`; `solve(..., reject_sigma=2.5, max_reject_frac=0.25)` new defaults.
- Consumes: `_solve_once` (Task 2).

- [ ] **Step 1: Write the failing test**

Create `src/handeye_calib/test/test_solve_reject.py`:
```python
import numpy as np
from handeye_calib import synthetic as syn, handeye_solve as hs, transforms as tf


def _corrupt_translation(sample):
    # Pure-translation FK corruption: shift T_base_eef by 4 cm, rotation intact.
    bad = sample.T_base_eef.copy()
    bad[:3, 3] = bad[:3, 3] + np.array([0.04, 0.0, 0.0])
    sample.T_base_eef = bad
    return sample


def test_default_rejection_is_on_and_catches_translation_outlier():
    sc = syn.make_scenario(n_poses=16, pixel_noise=0.3, seed=11)
    _corrupt_translation(sc.samples[3])
    # Default solve (reject_sigma defaults to 2.5) must flag the corrupted idx.
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.2, rng_seed=1)
    rejected = []
    for m in (res.per_method or []):
        if m.get("name") == "rejected_indices":
            rejected = m["rejected_indices"]
    assert len(rejected) >= 1  # the corrupted sample (or its train index) is caught


def test_clean_data_rejects_nothing_by_default():
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25, rng_seed=0)
    rejected = []
    for m in (res.per_method or []):
        if m.get("name") == "rejected_indices":
            rejected = m["rejected_indices"]
    assert rejected == []
    assert res.status == "PASS"


def test_per_axis_zscore_flags_translation_only_outlier():
    sc = syn.make_scenario(n_poses=12, pixel_noise=0.0, seed=2)
    X, Tbb, _, _ = hs._solve_once(sc.samples, sc.K, None, sc.board_pts)
    t_e, r_e = hs._per_sample_chain_errors(X, Tbb, sc.samples)
    t_e = t_e.copy(); t_e[5] += 0.05  # 5 cm translation spike
    zt = hs._modified_zscores(t_e)
    assert zt[5] > 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_reject.py -v`
Expected: FAIL — `_per_sample_chain_errors` missing AND default solve doesn't reject (current default `reject_sigma=None`).

- [ ] **Step 3: Add the helpers**

Append to `handeye_solve.py`:
```python
def _per_sample_chain_errors(X, Tbb, samples):
    """Per-sample (trans_m, rot_rad) of predicted-vs-observed board-in-camera.

    Predicted = inv(A_i @ X) @ Tbb; observed = s.T_cam_board (PnP). A bad FK
    (A_i) or a bad PnP shows up here as a large chain residual even when the
    pixel reprojection looks acceptable, which is why this scores rejection
    instead of reprojection alone."""
    t_e, r_e = [], []
    for s in samples:
        T_pred = tf.invert(s.T_base_eef @ X) @ Tbb
        T_obs = s.T_cam_board
        t_e.append(float(np.linalg.norm(T_pred[:3, 3] - T_obs[:3, 3])))
        r_e.append(np.radians(tf.rotation_angle_deg(T_pred[:3, :3], T_obs[:3, :3])))
    return np.asarray(t_e, float), np.asarray(r_e, float)


def _modified_zscores(arr):
    """Robust modified z-score |x - median| / (1.4826 * MAD), MAD floored."""
    arr = np.asarray(arr, float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return np.abs(arr - med) / (1.4826 * max(mad, 1e-9))
```

- [ ] **Step 4: Change `solve()` defaults and rewrite the rejection loop**

Change the `solve()` signature defaults (line ~223) to:
```python
def solve(samples, K, dist, board_pts, heldout_frac=0.2, rng_seed=0, *,
          methods=None, reject_sigma=2.5, max_reject_frac=0.25,
          depth_weight=1.0, depth_sigma_m=0.005, anchor_Tbb=None):
```
Replace the entire existing rejection block (the `rejected = []` through `train = [train[i] for i in active]` block, currently lines ~247-286) with this per-axis, `_solve_once`-based loop:
```python
    rejected = []
    if reject_sigma is not None and len(train) >= 6:
        # Iterative per-axis MAD rejection on the TRAIN set only (held-out stays
        # the honest evaluator). Translation and rotation chain errors are scored
        # as SEPARATE modified z-scores, so a pure-translation outlier that a
        # combined sqrt(t^2+r^2) metric would mask is still caught. Re-solves via
        # the multi-start _solve_once each round so the anchor seed (if any) is
        # reused.
        active = list(range(len(train)))
        n_orig = len(train)
        min_keep = max(6, int(np.ceil((1.0 - max_reject_frac) * n_orig)))
        for _ in range(20):
            sub = [train[i] for i in active]
            t_e, r_e = _per_sample_chain_errors(X, Tbb, sub)
            if len(sub) <= min_keep:
                break
            zt = _modified_zscores(t_e)
            zr = _modified_zscores(r_e)
            worst = np.maximum(zt, zr)
            k = int(np.argmax(worst))
            if worst[k] <= reject_sigma:
                break
            rejected.append(active.pop(k))
            sub = [train[i] for i in active]
            X, Tbb, _pm, _seed = _solve_once(
                sub, K, dist, board_pts, methods=methods,
                depth_weight=depth_weight, depth_sigma_m=depth_sigma_m,
                anchor_Tbb=anchor_Tbb)
        train = [train[i] for i in active]
```
(The `rejected_indices` attachment block just below — lines ~290-294 — stays unchanged; it already appends a `per_method` entry when `rejected` is non-empty. Confirm `rejected` indices are into the original `train` ordering, which they are since `active` starts as `range(len(train))`.)

- [ ] **Step 5: Update `do_solve` so the operator can still override**

In `handeye_web.py` `do_solve` (signature line ~1693), change so that not passing `reject_sigma` uses the solver default (2.5) rather than disabling it. Replace the signature + the call's `reject_sigma=` argument:
```python
        def do_solve(self, method: str = "auto", reject_sigma="default"):
```
and in the `hs.solve(...)` call replace the `reject_sigma=(...)` line with:
```python
                               reject_sigma=(2.5 if reject_sigma == "default"
                                             else (float(reject_sigma)
                                                   if reject_sigma is not None
                                                   else None)),
```
In the `/api/solve` route (~line 2746), the body already reads `reject_sigma` from JSON (`rs = body_d.get("reject_sigma")`); change the default so omission means "use solver default":
```python
        rs = body_d.get("reject_sigma", "default")
```
and pass `reject_sigma=rs` (unchanged call). So: omitted → "default" → 2.5; explicit `null` → disabled; explicit number → that value.

- [ ] **Step 6: Run the new tests + full suite**

Run:
```bash
pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_reject.py -v
pytest /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/ -q
```
Expected: new reject tests PASS; **all** existing tests PASS. If `test_solve_depth.py` or `test_solve_gate.py` now fail because the default changed, inspect: clean-data scenarios must reject nothing (MAD on clean residuals → no z-score over 2.5). If a test explicitly relied on `reject_sigma=None` behavior it already passes that; if a test called `solve(...)` bare and a borderline sample gets dropped, assert it still PASSes the gate (rejection only helps). Fix any test that asserted an exact `train` count to account for legitimate rejection, not by disabling the feature.

- [ ] **Step 7: Commit**

```bash
git add /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/handeye_solve.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py \
        /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/test/test_solve_reject.py
git commit -m "feat(handeye): default-on per-axis MAD outlier rejection on SE(3) chain error"
```

---

## Task 8: Docs — README changelog + DEV_NOTES hardware-verification stub

**Files:**
- Modify: `src/handeye_calib/README.md` (changelog entry)
- Modify: `/home/tinker/tk25_ws/src/tk26_vision/DEV_NOTES.md` (manual-verification matrix row)

- [ ] **Step 1: Add the changelog entry**

In `src/handeye_calib/README.md`, at the top of `## Changelog`, add a `0.8.0` entry summarizing: optional head-Orbbec **warm-start** (basin-immune board-anchor seed via `solve(anchor_Tbb=...)`, `Tbb` kept free so no head bias enters X; `/api/anchor` + Capture-tab button; head topics/frame params `head_image_topic`/`head_info_topic`/`head_optical_frame`); and the pan_tilt parity ports — multi-frame consensus capture + IPPE-seeded re-PnP, AX=XB rotation-observability diagnostic in the solve payload, and default-on per-axis MAD rejection on the SE(3) chain error (`reject_sigma` default 2.5, `max_reject_frac` 0.25). Note the honest ceiling: the head anchor only fixes the seed basin / reduces required pose count; the wrist reprojection + FFS depth still own sub-3 mm accuracy.

- [ ] **Step 2: Add a DEV_NOTES verification row**

In `/home/tinker/tk25_ws/src/tk26_vision/DEV_NOTES.md`, add a row to the operator-in-the-loop matrix for: "handeye head warm-start — anchor from 2–3 head poses on a degenerate (low-rotation) wrist set; confirm board-anchor seed selected + PASS/WARN; verify camera_link TF non-collision + live head angle (JSP patch)."

- [ ] **Step 3: Commit**

```bash
git add /home/tinker/tk25_ws/src/tk26_vision/src/handeye_calib/README.md /home/tinker/tk25_ws/src/tk26_vision/DEV_NOTES.md
git commit -m "docs(handeye): changelog 0.8.0 (head warm-start + parity ports) + DEV_NOTES row"
```

---

## Self-Review

**1. Spec coverage** (against the user's two decisions + the Part-1 review):
- Head warm-start → Tasks 1–3 (primitives, multi-start solve, node plumbing). ✓ `Tbb` stays free (Global Constraints + `seed_from_board_anchor` docstring) — no head bias into X. ✓
- Parity port #1 (consensus averaging) → Tasks 4–5. ✓ (also folds in the medium "IPPE-seed PnP" gap via `_pnp_ippe_refine`). ✓
- Parity port #2 (observability enforcement) → Task 6. ✓
- Parity port #4 (default-on, per-axis MAD on chain error) → Task 7. ✓
- **Deliberately deferred** (called out, not silently dropped): parity port #3 (RANSAC consensus pre-filter), #5 (independent head-validation gate — note: the head plumbing from Task 3 makes this a cheap follow-up and it is the single highest-value next step), session-drift start/end bracket, and hard timestamp-skew reject. These are independent, each worth its own plan; none is a prerequisite for Tasks 1–8.

**2. Placeholder scan:** every code step shows complete code; every test step shows the assertions; every run step shows the command + expected outcome. No "TBD"/"add error handling"/"similar to Task N". ✓

**3. Type consistency:**
- `seed_from_board_anchor(samples, anchor_Tbb) -> (X, Tbb)` defined Task 1, consumed by `_solve_once` Task 2. ✓
- `_solve_once(...) -> (X, Tbb, per_method, seed_used)` defined Task 2, consumed by the Task 7 rejection loop. ✓
- `solve(..., anchor_Tbb=None)` Task 2; `anchor_Tbb` passed from `do_solve` Task 3; defaults `reject_sigma=2.5`/`max_reject_frac=0.25` changed in Task 7. ✓
- `consensus_corners(frames, *, min_frac) -> (ids, px)` Task 4, consumed in `do_capture` Task 5. ✓
- `rotation_observability(samples, *, min_singular) -> dict` Task 6, consumed in `solve_payload_v2`. ✓
- `_per_sample_chain_errors`/`_modified_zscores` Task 7, consumed by the rejection loop. ✓
- `_pnp_ippe_refine(obj_pts, img_pts, K, D) -> (T, reproj)` Task 5, tested same task. ✓

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-06-27-handeye-headassist-parity.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?
