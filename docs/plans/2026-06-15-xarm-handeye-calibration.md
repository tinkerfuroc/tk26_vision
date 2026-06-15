# Wrist-RealSense Eye-in-Hand Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `handeye_calib` package that measures the fixed `T_eef→color_optical` transform of the wrist RealSense via calib_web-style pose authoring + `calibrateHandEye` seed + bundle-adjust refine, with held-out + physical verification, writing the result to `tinker_robot_config`.

**Architecture:** A ROS-free math core (`transforms`, `handeye_model`, `handeye_solve`, `gates`, `synthetic`) does all the calibration math and is fully unit-tested against synthetic ground truth. Two ROS nodes (`handeye_collect`, `handeye_web`) reuse the existing `pan_tilt.calibration` detection stack + safety envelope to drive the arm and capture data, then call the core. `apply_handeye` writes the result.

**Tech Stack:** Python 3.10, numpy, scipy (`least_squares`, `Rotation`), OpenCV (`cv2.aruco`, `cv2.calibrateHandEye`), rclpy, tf2_ros, cv_bridge, FastAPI/uvicorn, `tinker_arm_msgs` actions, reused `pan_tilt` modules.

**Spec:** `src/tk26_vision/docs/specs/2026-06-15-xarm-handeye-calibration-design.md`

---

## Frames & math contract (read once before coding)

All transforms are 4×4 homogeneous `T_a_b` = "pose of frame b expressed in frame a" = maps a point in b to a point in a (`p_a = T_a_b @ p_b`).

- `A_i = T_base_eef` — pose of flange in arm base (`link_base`←`link_eef`), from TF/FK. Known per pose.
- `X  = T_eef_cam`  — **unknown 1**: flange → color optical frame. Constant.
- `Tbb = T_base_board` — **unknown 2**: arm base → fixed board origin. Constant (board fixed in world).
- `B_i = T_cam_board` — pose of board in camera, from ChArUco PnP. Observed per pose.

Identities the code relies on:
```
T_base_cam_i = A_i @ X
B_i (= T_cam_board_i) = inv(A_i @ X) @ Tbb = inv(X) @ inv(A_i) @ Tbb
Tbb (estimated from a pose) = A_i @ X @ B_i
```

`cv2.calibrateHandEye` (eye-in-hand usage) returns `X = T_eef_cam` from:
- `R/t_gripper2base` ← `A_i`
- `R/t_target2cam`   ← `B_i`

Optical frame convention: x-right, y-down, z-forward. Pinhole projection of `p_cam=(x,y,z)`:
`u = fx*x/z + cx`, `v = fy*y/z + cy` (+ distortion for real camera; tests use zero distortion).

---

## File structure

Package `src/tk26_vision/src/handeye_calib/` (ament_python):

| File | Responsibility | ROS? |
|---|---|---|
| `handeye_calib/transforms.py` | SE(3) helpers: `T_from_vec`, `vec_from_T`, `invert`, `se3_average`, `rotation_angle_deg` | no |
| `handeye_calib/handeye_model.py` | `board_corners`, `Sample` dataclass, `project_corners` | no |
| `handeye_calib/synthetic.py` | Generate synthetic calibration scenarios; CLI sanity check | no |
| `handeye_calib/handeye_solve.py` | `seed_handeye`, `bundle_adjust`, `solve`, `split_train_test`, `evaluate`, `gate`, `SolveResult` | no |
| `handeye_calib/gates.py` | `StabilityTracker`, `is_diverse`, `quality_ok` (settle/diversity/quality gates) | no |
| `handeye_calib/apply_handeye.py` | `compose_eef_to_mount`, `handeye_yaml_dict`, `patch_urdf_origin` + writer | no (file IO) |
| `handeye_calib/handeye_collect.py` | ROS node: drive arm, settle-gate, capture, detect (reuses `pan_tilt`), accumulate session | yes |
| `handeye_calib/handeye_web.py` | calib_web-style FastAPI + ROS node: author/run/verify/promote | yes |
| `test/test_*.py` | pytest unit tests (core is fully covered) | no |
| `README.md` | user guide + append-only Changelog | — |

**Reuse (do not duplicate):** `pan_tilt.calibration.aruco_detect` (detection/consensus), `pan_tilt.calibration.safety` (envelope), `pan_tilt.calibration.charuco_generate` (board), referenced from the ROS nodes only.

**Test invocation (all tasks):** use the vision venv python with the package on the path. From repo root:
```bash
VPY=src/tk26_vision/.venv-vision-main/bin/python
cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/ -v
```
The math-core tests (Tasks 1–8) need only numpy/scipy/opencv (all in the venv) — **no colcon build required**. The ROS-node tasks (9–10) note their own build/run steps.

---

## Task 0: Package scaffold + reuse wiring

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/package.xml`
- Create: `src/tk26_vision/src/handeye_calib/setup.py`
- Create: `src/tk26_vision/src/handeye_calib/setup.cfg`
- Create: `src/tk26_vision/src/handeye_calib/resource/handeye_calib`
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/__init__.py`
- Create: `src/tk26_vision/src/handeye_calib/README.md`
- Test: `src/tk26_vision/src/handeye_calib/test/test_import.py`

- [ ] **Step 1: Create the package files**

`package.xml`:
```xml
<?xml version="1.0"?>
<package format="3">
  <name>handeye_calib</name>
  <version>0.1.0</version>
  <description>Eye-in-hand calibration for the wrist-mounted RealSense on the xArm.</description>
  <maintainer email="cindy.w0135@gmail.com">tinker</maintainer>
  <license>MIT</license>
  <exec_depend>rclpy</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>tf2_ros</exec_depend>
  <exec_depend>cv_bridge</exec_depend>
  <exec_depend>tinker_arm_msgs</exec_depend>
  <exec_depend>pan_tilt</exec_depend>
  <exec_depend>tinker_robot_config</exec_depend>
  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>python3-pytest</test_depend>
  <export><build_type>ament_python</build_type></export>
</package>
```

`setup.py`:
```python
from setuptools import setup

package_name = 'handeye_calib'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tinker',
    maintainer_email='cindy.w0135@gmail.com',
    description='Eye-in-hand calibration for the wrist-mounted RealSense on the xArm.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'handeye_synthetic_check = handeye_calib.synthetic:main',
            'handeye_collect = handeye_calib.handeye_collect:main',
            'handeye_web = handeye_calib.handeye_web:main',
            'apply_handeye = handeye_calib.apply_handeye:main',
        ],
    },
)
```

`setup.cfg`:
```ini
[develop]
script_dir=$base/lib/handeye_calib
[install]
install_scripts=$base/lib/handeye_calib
```

`resource/handeye_calib`: empty file.

`handeye_calib/__init__.py`: empty file.

`README.md`:
```markdown
# handeye_calib

Eye-in-hand calibration for the wrist-mounted RealSense on the xArm flange (`link_eef`).
Solves `T_eef→color_optical`, verifies it, and writes it to `tinker_robot_config`.

See `../../docs/specs/2026-06-15-xarm-handeye-calibration-design.md` for the design.

## Changelog
- 0.1.0 (2026-06-15): package scaffold.
```

- [ ] **Step 2: Write the reuse-wiring test**

`test/test_import.py`:
```python
def test_pan_tilt_calibration_core_importable():
    # The ROS nodes reuse the pan_tilt detection stack; fail loudly if the
    # dependency wiring is wrong once the workspace is built + sourced.
    import importlib
    for mod in ("pan_tilt.calibration.aruco_detect",
                "pan_tilt.calibration.safety"):
        assert importlib.util.find_spec(mod) is not None, mod
```

- [ ] **Step 3: Build the package with the vision wrapper**

Run: `./src/tk26_vision/scripts/build.sh --packages-select handeye_calib`
Expected: build finishes `Summary: 1 package finished`.

- [ ] **Step 4: Run the import test (sourced)**

Run:
```bash
source src/tk26_vision/install/setup.bash
src/tk26_vision/.venv-vision-main/bin/python -m pytest src/tk26_vision/src/handeye_calib/test/test_import.py -v
```
Expected: PASS (pan_tilt is on the path once sourced).

- [ ] **Step 5: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib
git commit -m "feat(handeye_calib): package scaffold + pan_tilt reuse wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: SE(3) transform helpers

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/transforms.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_transforms.py`

- [ ] **Step 1: Write the failing tests**

`test/test_transforms.py`:
```python
import numpy as np
from handeye_calib import transforms as tf


def test_vec_roundtrip():
    v = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    T = tf.T_from_vec(v)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(tf.vec_from_T(T), v, atol=1e-9)


def test_invert_is_inverse():
    v = np.array([0.3, 0.1, -0.2, 1.0, 2.0, -3.0])
    T = tf.T_from_vec(v)
    np.testing.assert_allclose(tf.invert(T) @ T, np.eye(4), atol=1e-9)


def test_se3_average_of_identical_is_identity_member():
    T = tf.T_from_vec(np.array([0.2, 0.0, 0.1, 0.5, 0.5, 0.5]))
    avg = tf.se3_average([T, T, T])
    np.testing.assert_allclose(avg, T, atol=1e-9)


def test_rotation_angle_deg():
    from scipy.spatial.transform import Rotation as R
    R1 = np.eye(3)
    R2 = R.from_euler('z', 30, degrees=True).as_matrix()
    assert abs(tf.rotation_angle_deg(R1, R2) - 30.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_transforms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'handeye_calib.transforms'`.

- [ ] **Step 3: Implement transforms.py**

```python
"""SE(3) helpers (numpy + scipy only). Parameterization is [rotvec(3), trans(3)]."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def T_from_vec(v6):
    v6 = np.asarray(v6, float)
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(v6[:3]).as_matrix()
    T[:3, 3] = v6[3:]
    return T


def vec_from_T(T):
    rotvec = R.from_matrix(np.asarray(T)[:3, :3]).as_rotvec()
    return np.concatenate([rotvec, np.asarray(T)[:3, 3]])


def T_from_Rt(Rm, t):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert(T):
    Rm = np.asarray(T)[:3, :3]
    t = np.asarray(T)[:3, 3]
    out = np.eye(4)
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def se3_average(Ts):
    """Chordal SE(3) mean: quaternion mean of rotations + arithmetic mean of translations."""
    Ts = [np.asarray(T) for T in Ts]
    quats = R.from_matrix([T[:3, :3] for T in Ts]).as_quat()
    # sign-align quaternions to the first to avoid cancellation
    ref = quats[0]
    quats = np.array([q if np.dot(q, ref) >= 0 else -q for q in quats])
    mean_q = quats.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q)
    out = np.eye(4)
    out[:3, :3] = R.from_quat(mean_q).as_matrix()
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def rotation_angle_deg(R1, R2):
    Rrel = np.asarray(R1).T @ np.asarray(R2)
    return float(np.degrees(R.from_matrix(Rrel).magnitude()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_transforms.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/transforms.py src/handeye_calib/test/test_transforms.py
git commit -m "feat(handeye_calib): SE(3) transform helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Board geometry + reprojection model

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_model.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_model.py`

- [ ] **Step 1: Write the failing tests**

`test/test_model.py`:
```python
import numpy as np
from handeye_calib import transforms as tf
from handeye_calib import handeye_model as hm


def test_board_corners_count_and_centered():
    pts = hm.board_corners(squares_x=5, squares_y=5, square_len=0.04)
    assert pts.shape == (16, 3)              # (5-1)*(5-1) inner corners
    assert np.allclose(pts[:, 2], 0.0)       # planar board
    np.testing.assert_allclose(pts[:, :2].mean(axis=0), [0, 0], atol=1e-9)


def test_project_known_point_on_axis():
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    # board 1 m in front of camera, axes aligned -> corner at board origin maps to principal point
    T_cam_board = tf.T_from_Rt(np.eye(3), [0, 0, 1.0])
    px = hm.project_corners(np.array([[0., 0., 0.]]), T_cam_board, K, dist=None)
    np.testing.assert_allclose(px[0], [320., 240.], atol=1e-9)


def test_project_offset_point():
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    T_cam_board = tf.T_from_Rt(np.eye(3), [0, 0, 2.0])
    px = hm.project_corners(np.array([[0.1, 0.0, 0.0]]), T_cam_board, K, dist=None)
    np.testing.assert_allclose(px[0], [320. + 600 * 0.1 / 2.0, 240.], atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_model.py -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Implement handeye_model.py**

```python
"""Board geometry, the calibration Sample, and the pinhole reprojection used by the solver."""
from dataclasses import dataclass
import numpy as np
import cv2


def board_corners(squares_x=5, squares_y=5, square_len=0.04):
    """Inner ChArUco corner positions (meters), board-centered, z=0 plane.

    Order is row-major over the (squares_x-1) x (squares_y-1) inner grid, matching
    cv2.aruco CharucoBoard chessboard-corner ordering.
    """
    nx, ny = squares_x - 1, squares_y - 1
    xs = (np.arange(1, squares_x)) * square_len
    ys = (np.arange(1, squares_y)) * square_len
    pts = np.array([[xs[i], ys[j], 0.0] for j in range(ny) for i in range(nx)], float)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts


def project_corners(board_pts, T_cam_board, K, dist=None):
    """Project board points (N,3, board frame) into pixels via T_cam_board and K."""
    board_pts = np.asarray(board_pts, float).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(np.asarray(T_cam_board)[:3, :3])
    tvec = np.asarray(T_cam_board)[:3, 3]
    if dist is None:
        dist = np.zeros(5)
    px, _ = cv2.projectPoints(board_pts, rvec, tvec, np.asarray(K, float), np.asarray(dist, float))
    return px.reshape(-1, 2)


@dataclass
class Sample:
    """One accepted calibration pose."""
    T_base_eef: np.ndarray        # 4x4, A_i (from TF/FK)
    T_cam_board: np.ndarray       # 4x4, B_i (from PnP) — seed input
    obs_px: np.ndarray            # (M,2) observed corner pixels
    corner_idx: np.ndarray        # (M,) indices into board_corners() for obs_px
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_model.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/handeye_model.py src/handeye_calib/test/test_model.py
git commit -m "feat(handeye_calib): board geometry + reprojection model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Synthetic scenario generator

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/synthetic.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_synthetic.py`

- [ ] **Step 1: Write the failing tests**

`test/test_synthetic.py`:
```python
import numpy as np
from handeye_calib import synthetic as syn
from handeye_calib import handeye_model as hm


def test_scenario_shapes_and_consistency():
    sc = syn.make_scenario(n_poses=12, pixel_noise=0.0, seed=0)
    assert len(sc.samples) == 12
    # With zero noise, observed pixels must equal reprojection through ground-truth X, Tbb.
    from handeye_calib import transforms as tf
    for s in sc.samples:
        T_cam_board = tf.invert(s.T_base_eef @ sc.X_true) @ sc.Tbb_true
        px = hm.project_corners(sc.board_pts[s.corner_idx], T_cam_board, sc.K)
        np.testing.assert_allclose(px, s.obs_px, atol=1e-6)


def test_pnp_pose_matches_truth_noiseless():
    sc = syn.make_scenario(n_poses=8, pixel_noise=0.0, seed=1)
    from handeye_calib import transforms as tf
    for s in sc.samples:
        T_cam_board_true = tf.invert(s.T_base_eef @ sc.X_true) @ sc.Tbb_true
        np.testing.assert_allclose(s.T_cam_board, T_cam_board_true, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_synthetic.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement synthetic.py**

```python
"""Synthetic eye-in-hand scenarios for testing the solver against ground truth.

Also a CLI sanity check (`handeye_synthetic_check`) that runs the full solve on
synthetic data and prints recovered-vs-true error.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from handeye_calib import transforms as tf
from handeye_calib import handeye_model as hm


@dataclass
class Scenario:
    samples: list          # list[hm.Sample]
    X_true: np.ndarray     # T_eef_cam
    Tbb_true: np.ndarray   # T_base_board
    K: np.ndarray
    board_pts: np.ndarray


def _pnp(board_pts, px, K):
    ok, rvec, tvec = cv2.solvePnP(board_pts.astype(np.float64), px.astype(np.float64),
                                  K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    Rm, _ = cv2.Rodrigues(rvec)
    return tf.T_from_Rt(Rm, tvec.reshape(3))


def make_scenario(n_poses=15, pixel_noise=0.3, seed=0,
                  squares_x=5, squares_y=5, square_len=0.04):
    rng = np.random.default_rng(seed)
    K = np.array([[615., 0, 320.], [0, 615., 240.], [0, 0, 1.]])
    board_pts = hm.board_corners(squares_x, squares_y, square_len)

    # Ground-truth unknowns (plausible wrist mount + a board ~0.5 m in front of base).
    X_true = tf.T_from_vec(np.array([np.pi, -np.pi / 2, 0.0, 0.07, -0.018, 0.024]))
    Tbb_true = tf.T_from_Rt(R.from_euler('xyz', [-90, 0, 5], degrees=True).as_matrix(),
                            [0.5, 0.0, 0.3])

    samples = []
    tries = 0
    while len(samples) < n_poses and tries < n_poses * 20:
        tries += 1
        # Random flange pose that keeps the camera looking at the board with diversity.
        rot = R.from_euler('xyz', rng.uniform(-0.6, 0.6, 3)).as_matrix()
        trans = np.array([0.45, 0.0, 0.35]) + rng.uniform(-0.12, 0.12, 3)
        A = tf.T_from_Rt(rot, trans)
        T_cam_board = tf.invert(A @ X_true) @ Tbb_true
        if T_cam_board[2, 3] < 0.25 or T_cam_board[2, 3] > 0.8:
            continue  # board must be in front, sane standoff
        px = hm.project_corners(board_pts, T_cam_board, K)
        if (px[:, 0].min() < 0 or px[:, 0].max() > 640 or
                px[:, 1].min() < 0 or px[:, 1].max() > 480):
            continue  # board must be fully in frame
        obs = px + rng.normal(0, pixel_noise, px.shape) if pixel_noise else px
        idx = np.arange(len(board_pts))
        samples.append(hm.Sample(T_base_eef=A, T_cam_board=_pnp(board_pts, obs, K),
                                 obs_px=obs, corner_idx=idx))
    if len(samples) < n_poses:
        raise RuntimeError(f"only generated {len(samples)}/{n_poses} poses")
    return Scenario(samples, X_true, Tbb_true, K, board_pts)


def main():
    from handeye_calib import handeye_solve as hs
    sc = make_scenario(n_poses=20, pixel_noise=0.3, seed=3)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts)
    dt = np.linalg.norm(res.X[:3, 3] - sc.X_true[:3, 3]) * 1000
    dr = tf.rotation_angle_deg(res.X[:3, :3], sc.X_true[:3, :3])
    print(f"recovered X error: {dt:.3f} mm, {dr:.4f} deg; status={res.status}")
    print(f"held-out: {res.heldout_metrics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_synthetic.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/synthetic.py src/handeye_calib/test/test_synthetic.py
git commit -m "feat(handeye_calib): synthetic scenario generator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Multi-method hand-eye seed

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_solve.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_seed.py`

- [ ] **Step 1: Write the failing tests**

`test/test_seed.py`:
```python
import numpy as np
from handeye_calib import synthetic as syn, transforms as tf, handeye_solve as hs


def test_seed_recovers_truth_noiseless():
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=2)
    X, Tbb, per_method = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    dt = np.linalg.norm(X[:3, 3] - sc.X_true[:3, 3]) * 1000
    dr = tf.rotation_angle_deg(X[:3, :3], sc.X_true[:3, :3])
    assert dt < 2.0, f"{dt} mm"       # linear seed: a couple mm even noiseless
    assert dr < 0.5, f"{dr} deg"
    assert len(per_method) >= 3        # several OpenCV methods tried


def test_seed_picks_lowest_reproj():
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.3, seed=4)
    X, Tbb, per_method = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    best = min(per_method, key=lambda m: m["reproj_px"])
    np.testing.assert_allclose(X, best["X"], atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_seed.py -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Implement the seed in handeye_solve.py**

```python
"""Eye-in-hand solver: multi-method seed -> bundle-adjust refine -> held-out evaluation."""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import least_squares

from handeye_calib import transforms as tf
from handeye_calib import handeye_model as hm

_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _reproj_rms(X, Tbb, samples, K, dist, board_pts):
    sq = []
    for s in samples:
        T_cam_board = tf.invert(s.T_base_eef @ X) @ Tbb
        pred = hm.project_corners(board_pts[s.corner_idx], T_cam_board, K, dist)
        sq.append(np.sum((pred - s.obs_px) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(sq))))


def _estimate_board_in_base(X, samples):
    return tf.se3_average([s.T_base_eef @ X @ s.T_cam_board for s in samples])


def seed_handeye(samples, K, dist, board_pts):
    """Run all OpenCV hand-eye methods, return the X with lowest reprojection RMS."""
    R_g2b = [np.asarray(s.T_base_eef)[:3, :3] for s in samples]
    t_g2b = [np.asarray(s.T_base_eef)[:3, 3] for s in samples]
    R_t2c = [np.asarray(s.T_cam_board)[:3, :3] for s in samples]
    t_t2c = [np.asarray(s.T_cam_board)[:3, 3] for s in samples]
    per_method = []
    for name, flag in _METHODS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
        except cv2.error:
            continue
        X = tf.T_from_Rt(R_c2g, t_c2g.reshape(3))
        Tbb = _estimate_board_in_base(X, samples)
        per_method.append({"name": name, "X": X, "Tbb": Tbb,
                           "reproj_px": _reproj_rms(X, Tbb, samples, K, dist, board_pts)})
    if not per_method:
        raise RuntimeError("all calibrateHandEye methods failed")
    best = min(per_method, key=lambda m: m["reproj_px"])
    return best["X"], best["Tbb"], per_method
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_seed.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/handeye_solve.py src/handeye_calib/test/test_seed.py
git commit -m "feat(handeye_calib): multi-method hand-eye seed

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Bundle-adjust refine

**Files:**
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_solve.py` (append `bundle_adjust`)
- Test: `src/tk26_vision/src/handeye_calib/test/test_bundle_adjust.py`

- [ ] **Step 1: Write the failing tests**

`test/test_bundle_adjust.py`:
```python
import numpy as np
from handeye_calib import synthetic as syn, transforms as tf, handeye_solve as hs


def test_ba_beats_seed_under_noise():
    sc = syn.make_scenario(n_poses=18, pixel_noise=0.4, seed=7)
    Xs, Tbbs, _ = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    Xb, Tbbb, info = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs)
    seed_err = np.linalg.norm(Xs[:3, 3] - sc.X_true[:3, 3]) * 1000
    ba_err = np.linalg.norm(Xb[:3, 3] - sc.X_true[:3, 3]) * 1000
    assert ba_err <= seed_err + 1e-6
    assert ba_err < 1.0, f"{ba_err} mm"                       # sub-mm under 0.4px noise
    assert tf.rotation_angle_deg(Xb[:3, :3], sc.X_true[:3, :3]) < 0.2
    assert info["final_reproj_px"] < 0.6


def test_ba_exact_when_noiseless():
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=8)
    Xs, Tbbs, _ = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    Xb, _, info = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs)
    assert np.linalg.norm(Xb[:3, 3] - sc.X_true[:3, 3]) * 1000 < 0.05
    assert info["final_reproj_px"] < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_bundle_adjust.py -v`
Expected: FAIL — `bundle_adjust` not defined.

- [ ] **Step 3: Append `bundle_adjust` to handeye_solve.py**

```python
def _residuals(params, samples, K, dist, board_pts):
    X = tf.T_from_vec(params[:6])
    Tbb = tf.T_from_vec(params[6:])
    res = []
    for s in samples:
        T_cam_board = tf.invert(s.T_base_eef @ X) @ Tbb
        pred = hm.project_corners(board_pts[s.corner_idx], T_cam_board, K, dist)
        res.append((pred - s.obs_px).ravel())
    return np.concatenate(res)


def bundle_adjust(samples, K, dist, board_pts, X0, Tbb0):
    """Jointly refine X (T_eef_cam) and Tbb (T_base_board) minimizing corner reprojection."""
    p0 = np.concatenate([tf.vec_from_T(X0), tf.vec_from_T(Tbb0)])
    sol = least_squares(_residuals, p0, loss="soft_l1", method="trf",
                        args=(samples, K, dist, board_pts))
    X = tf.T_from_vec(sol.x[:6])
    Tbb = tf.T_from_vec(sol.x[6:])
    info = {"final_reproj_px": _reproj_rms(X, Tbb, samples, K, dist, board_pts),
            "success": bool(sol.success), "cost": float(sol.cost)}
    return X, Tbb, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_bundle_adjust.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/handeye_solve.py src/handeye_calib/test/test_bundle_adjust.py
git commit -m "feat(handeye_calib): bundle-adjust refine of X and board-in-base

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Held-out split, evaluation, acceptance gate, `solve`

**Files:**
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_solve.py` (append split/evaluate/gate/solve + `SolveResult`)
- Test: `src/tk26_vision/src/handeye_calib/test/test_solve_gate.py`

- [ ] **Step 1: Write the failing tests**

`test/test_solve_gate.py`:
```python
import numpy as np
from handeye_calib import synthetic as syn, handeye_solve as hs


def test_solve_passes_gate_on_clean_data():
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25, rng_seed=0)
    assert res.status == "PASS"
    assert res.heldout_metrics["trans_rmse_m"] < 0.003
    assert res.heldout_metrics["rot_rmse_rad"] < 0.00873
    assert res.heldout_metrics["reproj_px"] < 1.5


def test_gate_thresholds():
    assert hs.gate({"trans_rmse_m": 0.002, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "PASS"
    assert hs.gate({"trans_rmse_m": 0.005, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "WARN"
    assert hs.gate({"trans_rmse_m": 0.02, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "FAIL"


def test_split_is_deterministic_and_disjoint():
    sc = syn.make_scenario(n_poses=10, pixel_noise=0.0, seed=0)
    tr1, te1 = hs.split_train_test(sc.samples, 0.3, rng_seed=5)
    tr2, te2 = hs.split_train_test(sc.samples, 0.3, rng_seed=5)
    assert [id(s) for s in te1] == [id(s) for s in te2]      # deterministic
    assert len(tr1) + len(te1) == 10 and len(te1) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_solve_gate.py -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Append split/evaluate/gate/solve to handeye_solve.py**

```python
@dataclass
class SolveResult:
    X: np.ndarray
    Tbb: np.ndarray
    train_metrics: dict
    heldout_metrics: dict
    status: str
    per_method: list


# pan-tilt parity thresholds
_PASS = {"trans_rmse_m": 0.003, "rot_rmse_rad": 0.00873, "reproj_px": 1.5}
_WARN = {"trans_rmse_m": 0.006, "rot_rmse_rad": 0.01745, "reproj_px": 3.0}


def split_train_test(samples, heldout_frac, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_te = max(1, int(round(len(samples) * heldout_frac)))
    te = sorted(idx[:n_te].tolist())
    tr = sorted(idx[n_te:].tolist())
    return [samples[i] for i in tr], [samples[i] for i in te]


def evaluate(X, Tbb, samples, K, dist, board_pts):
    trans_e, rot_e = [], []
    for s in samples:
        T_pred = tf.invert(s.T_base_eef @ X) @ Tbb     # predicted board-in-cam
        T_obs = s.T_cam_board                           # observed (PnP)
        trans_e.append(np.linalg.norm(T_pred[:3, 3] - T_obs[:3, 3]))
        rot_e.append(np.radians(tf.rotation_angle_deg(T_pred[:3, :3], T_obs[:3, :3])))
    return {"trans_rmse_m": float(np.sqrt(np.mean(np.square(trans_e)))),
            "rot_rmse_rad": float(np.sqrt(np.mean(np.square(rot_e)))),
            "reproj_px": _reproj_rms(X, Tbb, samples, K, dist, board_pts)}


def gate(metrics):
    def ok(th):
        return all(metrics[k] <= th[k] for k in th)
    if ok(_PASS):
        return "PASS"
    if ok(_WARN):
        return "WARN"
    return "FAIL"


def solve(samples, K, dist, board_pts, heldout_frac=0.2, rng_seed=0):
    train, test = split_train_test(samples, heldout_frac, rng_seed)
    X0, Tbb0, per_method = seed_handeye(train, K, dist, board_pts)
    X, Tbb, _ = bundle_adjust(train, K, dist, board_pts, X0, Tbb0)
    train_m = evaluate(X, Tbb, train, K, dist, board_pts)
    held_m = evaluate(X, Tbb, test, K, dist, board_pts)
    return SolveResult(X, Tbb, train_m, held_m, gate(held_m), per_method)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_solve_gate.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the synthetic CLI end-to-end**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m handeye_calib.synthetic`
Expected: prints `recovered X error: <…> mm, <…> deg; status=PASS` with error < 1 mm.

- [ ] **Step 6: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/handeye_solve.py src/handeye_calib/test/test_solve_gate.py
git commit -m "feat(handeye_calib): held-out evaluation + acceptance gate + solve()

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Collection gates (settle / diversity / quality)

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/gates.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_gates.py`

- [ ] **Step 1: Write the failing tests**

`test/test_gates.py`:
```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from handeye_calib import transforms as tf
from handeye_calib import gates


def test_stability_tracker_flags_when_steady():
    trk = gates.StabilityTracker(window=3, rot_tol_deg=0.1, trans_tol_m=0.0003)
    T = tf.T_from_Rt(np.eye(3), [0, 0, 0.5])
    assert trk.update(T) is False        # need a full window
    assert trk.update(T) is False
    assert trk.update(T) is True         # 3 steady frames -> stable


def test_stability_tracker_rejects_jitter():
    trk = gates.StabilityTracker(window=3, rot_tol_deg=0.1, trans_tol_m=0.0003)
    for k in range(5):
        T = tf.T_from_Rt(np.eye(3), [0, 0, 0.5 + 0.01 * k])   # moving 1 cm/frame
        assert trk.update(T) is False


def test_is_diverse():
    accepted = [tf.T_from_Rt(np.eye(3), [0, 0, 0.5])]
    near = tf.T_from_Rt(R.from_euler('z', 10, degrees=True).as_matrix(), [0, 0, 0.5])
    far = tf.T_from_Rt(R.from_euler('z', 40, degrees=True).as_matrix(), [0, 0, 0.5])
    assert gates.is_diverse(near, accepted, min_deg=30) is False
    assert gates.is_diverse(far, accepted, min_deg=30) is True
    assert gates.is_diverse(near, [], min_deg=30) is True     # first pose always ok


def test_quality_ok_reasons():
    ok, reason = gates.quality_ok(n_corners=16, reproj_px=0.8, area_frac=0.2)
    assert ok and reason == "ok"
    ok, reason = gates.quality_ok(n_corners=4, reproj_px=0.8, area_frac=0.2)
    assert not ok and "corners" in reason
    ok, reason = gates.quality_ok(n_corners=16, reproj_px=3.0, area_frac=0.2)
    assert not ok and "reproj" in reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_gates.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement gates.py**

```python
"""Pure-logic collection gates: settle/stability, pose diversity, per-frame quality."""
import numpy as np
from handeye_calib import transforms as tf


class StabilityTracker:
    """Returns True once the last `window` board poses agree within tolerance.

    Absorbs the 1-2 s mount ring: feed live PnP poses; capture only when it
    returns True (or treat repeated False past a timeout as 'did not settle').
    """
    def __init__(self, window=5, rot_tol_deg=0.1, trans_tol_m=0.0003):
        self.window = window
        self.rot_tol_deg = rot_tol_deg
        self.trans_tol_m = trans_tol_m
        self._buf = []

    def reset(self):
        self._buf = []

    def update(self, T_cam_board):
        self._buf.append(np.asarray(T_cam_board))
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if len(self._buf) < self.window:
            return False
        ref = self._buf[-1]
        for T in self._buf[:-1]:
            if tf.rotation_angle_deg(T[:3, :3], ref[:3, :3]) > self.rot_tol_deg:
                return False
            if np.linalg.norm(T[:3, 3] - ref[:3, 3]) > self.trans_tol_m:
                return False
        return True


def is_diverse(T_base_eef_new, accepted, min_deg=30.0):
    """True if the new flange orientation differs from every accepted pose by >= min_deg."""
    if not accepted:
        return True
    Rn = np.asarray(T_base_eef_new)[:3, :3]
    return all(tf.rotation_angle_deg(np.asarray(T)[:3, :3], Rn) >= min_deg for T in accepted)


def quality_ok(n_corners, reproj_px, area_frac,
               min_corners=10, max_reproj_px=1.5, min_area_frac=0.05):
    if n_corners < min_corners:
        return False, f"too few corners ({n_corners}<{min_corners})"
    if reproj_px > max_reproj_px:
        return False, f"reproj too high ({reproj_px:.2f}>{max_reproj_px})"
    if area_frac < min_area_frac:
        return False, f"board too small ({area_frac:.2f}<{min_area_frac})"
    return True, "ok"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_gates.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/gates.py src/handeye_calib/test/test_gates.py
git commit -m "feat(handeye_calib): settle/diversity/quality collection gates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Result composition + storage (`apply_handeye`)

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/apply_handeye.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_apply.py`

- [ ] **Step 1: Write the failing tests**

`test/test_apply.py`:
```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from handeye_calib import transforms as tf
from handeye_calib import apply_handeye as ah


def test_compose_eef_to_mount_roundtrip():
    # Known internal chain mount->color_optical; recover eef->mount from eef->color_optical.
    T_mount_color = tf.T_from_Rt(R.from_euler('xyz', [-90, 0, -90], degrees=True).as_matrix(),
                                 [0.0, 0.015, 0.0])
    T_eef_mount_true = tf.T_from_vec(np.array([0.1, -0.2, 0.05, 0.07, -0.018, 0.024]))
    T_eef_color = T_eef_mount_true @ T_mount_color
    T_eef_mount = ah.compose_eef_to_mount(T_eef_color, T_mount_color)
    np.testing.assert_allclose(T_eef_mount, T_eef_mount_true, atol=1e-9)


def test_yaml_dict_has_required_fields():
    T = tf.T_from_vec(np.array([0.0, 0.0, 0.0, 0.06, -0.01, 0.02]))
    d = ah.handeye_yaml_dict(T_eef_mount=T, T_eef_color=T, num_poses=18,
                             metrics={"trans_rmse_m": 0.002, "rot_rmse_rad": 0.005,
                                      "reproj_px": 0.9}, date="2026-06-15")
    he = d["hand_eye"]
    assert he["reference_frame"] == "link_eef"
    assert he["camera_frame"] == "xarm_camera_link"
    assert len(he["arm_to_camera_xyz"].split()) == 3
    assert len(he["arm_to_camera_rpy"].split()) == 3
    assert he["num_poses"] == 18


def test_patch_urdf_origin(tmp_path):
    xacro = (
        '<robot>\n'
        '  <joint name="xarm_camera_joint" type="fixed">\n'
        '    <origin xyz="0.06746 -0.0175 0.0237" rpy="3.14159 -1.5708 0"/>\n'
        '    <parent link="link_eef"/>\n'
        '    <child link="xarm_camera_link"/>\n'
        '  </joint>\n'
        '</robot>\n'
    )
    new = ah.patch_urdf_origin(xacro, "xarm_camera_joint",
                               xyz=(0.1, 0.2, 0.3), rpy=(0.0, 0.0, 0.0))
    assert 'xyz="0.1 0.2 0.3"' in new
    assert 'rpy="0.0 0.0 0.0"' in new
    assert '0.06746' not in new           # old value replaced
    assert new.count("<joint") == 1       # only the targeted joint touched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_apply.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement apply_handeye.py**

```python
"""Compose the solved transform into the URDF mount frame and persist it.

Solver outputs T_eef_color (link_eef -> xarm_camera_color_optical_frame). The URDF
attaches xarm_camera_link to link_eef; color_optical is a fixed child of camera_link.
So the URDF mount-joint origin we must write is:
    T_eef_mount = T_eef_color @ inv(T_mount_color)
where T_mount_color is the (factory, unchanged) camera_link->color_optical chain.
"""
import re
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from handeye_calib import transforms as tf


def compose_eef_to_mount(T_eef_color, T_mount_color):
    return np.asarray(T_eef_color) @ tf.invert(T_mount_color)


def _xyz_rpy(T):
    xyz = np.asarray(T)[:3, 3]
    rpy = R.from_matrix(np.asarray(T)[:3, :3]).as_euler('xyz')
    return " ".join(f"{v:.9g}" for v in xyz), " ".join(f"{v:.9g}" for v in rpy)


def handeye_yaml_dict(T_eef_mount, T_eef_color, num_poses, metrics, date,
                      square_len_m=0.04):
    mount_xyz, mount_rpy = _xyz_rpy(T_eef_mount)
    color_xyz, color_rpy = _xyz_rpy(T_eef_color)
    return {"hand_eye": {
        "reference_frame": "link_eef",
        "camera_frame": "xarm_camera_link",
        "arm_to_camera_xyz": mount_xyz,
        "arm_to_camera_rpy": mount_rpy,
        "color_optical_xyz": color_xyz,
        "color_optical_rpy": color_rpy,
        "calibration_date": date,
        "calibration_method": "calibrateHandEye+BA",
        "board": {"type": "charuco", "squares": "5x5", "square_len_m": square_len_m},
        "num_poses": int(num_poses),
        "heldout_trans_rmse_m": round(float(metrics["trans_rmse_m"]), 6),
        "heldout_rot_rmse_rad": round(float(metrics["rot_rmse_rad"]), 6),
        "heldout_reproj_px": round(float(metrics["reproj_px"]), 4),
    }}


def patch_urdf_origin(xacro_text, joint_name, xyz, rpy):
    """Replace the <origin .../> inside the named <joint>, leaving everything else intact."""
    xyz_s = " ".join(str(v) for v in xyz)
    rpy_s = " ".join(str(v) for v in rpy)
    pat = re.compile(
        r'(<joint\s+name="' + re.escape(joint_name) + r'".*?<origin\b)[^>]*?(/?>)',
        re.DOTALL)
    if not pat.search(xacro_text):
        raise ValueError(f"origin for joint {joint_name} not found")
    return pat.sub(rf'\1 xyz="{xyz_s}" rpy="{rpy_s}"\2', xacro_text, count=1)


def write_with_backup(path, text):
    import os
    if os.path.exists(path):
        backup = f"{path}.old-{time.strftime('%Y%m%dT%H%M%S')}"
        os.replace(path, backup)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def main():
    raise SystemExit("apply_handeye is used as a library by handeye_web; see README.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_apply.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/apply_handeye.py src/handeye_calib/test/test_apply.py
git commit -m "feat(handeye_calib): result composition + yaml/URDF persistence

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Collection ROS node (`handeye_collect`)

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_collect.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_session.py`

This node is integration-tested on hardware (tier T2+). The testable part — the
session accumulator that turns detections into `Sample`s and applies the gates — is
extracted into a pure `CaptureSession` class and unit-tested here. The rclpy plumbing
(action clients, TF, image sub) is thin and verified by the T2 dry-run in Step 5.

- [ ] **Step 1: Write the failing test for `CaptureSession`**

`test/test_session.py`:
```python
import numpy as np
from handeye_calib import synthetic as syn
from handeye_calib.handeye_collect import CaptureSession


def test_session_accepts_diverse_rejects_redundant():
    sc = syn.make_scenario(n_poses=6, pixel_noise=0.0, seed=0)
    sess = CaptureSession(min_diversity_deg=30.0)
    # First sample always accepted.
    s0 = sc.samples[0]
    assert sess.try_add(s0.T_base_eef, s0.T_cam_board, s0.obs_px, s0.corner_idx,
                        n_corners=16, reproj_px=0.5, area_frac=0.3)[0] is True
    # The same flange pose again -> not diverse -> rejected.
    ok, reason = sess.try_add(s0.T_base_eef, s0.T_cam_board, s0.obs_px, s0.corner_idx,
                              n_corners=16, reproj_px=0.5, area_frac=0.3)
    assert ok is False and "diver" in reason.lower()
    assert len(sess.samples) == 1


def test_session_rejects_low_quality():
    sc = syn.make_scenario(n_poses=3, pixel_noise=0.0, seed=1)
    sess = CaptureSession(min_diversity_deg=30.0)
    s = sc.samples[0]
    ok, reason = sess.try_add(s.T_base_eef, s.T_cam_board, s.obs_px, s.corner_idx,
                              n_corners=4, reproj_px=0.5, area_frac=0.3)
    assert ok is False and "corner" in reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_session.py -v`
Expected: FAIL — cannot import `CaptureSession`.

- [ ] **Step 3: Implement handeye_collect.py**

```python
"""ROS node: drive the xArm to authored poses, settle-gate, detect the ChArUco board
(reusing pan_tilt's detector), accumulate diverse high-quality Samples, save a session.

The pure accumulator (CaptureSession) is unit-tested; the rclpy wiring is exercised by
the hardware dry-run.
"""
import json
import numpy as np

from handeye_calib import handeye_model as hm
from handeye_calib import gates


class CaptureSession:
    def __init__(self, min_diversity_deg=30.0,
                 min_corners=10, max_reproj_px=1.5, min_area_frac=0.05):
        self.min_diversity_deg = min_diversity_deg
        self.q = dict(min_corners=min_corners, max_reproj_px=max_reproj_px,
                      min_area_frac=min_area_frac)
        self.samples = []

    def try_add(self, T_base_eef, T_cam_board, obs_px, corner_idx,
                n_corners, reproj_px, area_frac):
        ok, reason = gates.quality_ok(n_corners, reproj_px, area_frac, **self.q)
        if not ok:
            return False, reason
        accepted_eef = [s.T_base_eef for s in self.samples]
        if not gates.is_diverse(T_base_eef, accepted_eef, self.min_diversity_deg):
            return False, "not diverse (<%g deg)" % self.min_diversity_deg
        self.samples.append(hm.Sample(np.asarray(T_base_eef), np.asarray(T_cam_board),
                                      np.asarray(obs_px), np.asarray(corner_idx)))
        return True, "accepted"

    def to_json(self):
        return json.dumps([{
            "T_base_eef": s.T_base_eef.tolist(),
            "T_cam_board": s.T_cam_board.tolist(),
            "obs_px": s.obs_px.tolist(),
            "corner_idx": s.corner_idx.tolist(),
        } for s in self.samples])


# ---- rclpy node (exercised on hardware; imports guarded so unit tests stay ROS-free) ----
def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from tf2_ros import Buffer, TransformListener
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CameraInfo
    from tinker_arm_msgs.action import JointMove
    from pan_tilt.calibration import aruco_detect
    from handeye_calib import transforms as tf

    class HandeyeCollect(Node):
        def __init__(self):
            super().__init__("handeye_collect")
            self.session = CaptureSession()
            self.bridge = CvBridge()
            self.tf_buffer = Buffer()
            TransformListener(self.tf_buffer, self)
            self.jm = ActionClient(self, JointMove, "/xarm/joint_move")
            self.sub = self.create_subscription(
                Image, "/xarm_camera/color/image_raw", self._on_image, 1)
            self.info_sub = self.create_subscription(
                CameraInfo, "/xarm_camera/color/camera_info", self._on_info, 1)
            self.stability = gates.StabilityTracker()
            self.K = None
            self.get_logger().info("handeye_collect ready")
        # _on_info caches K; _on_image runs aruco_detect + StabilityTracker;
        # a run() coroutine sends JointMove goals, waits settle, then captures.
        # See README 'Collection node' for the full loop; this is hardware-tier code.

    rclpy.init()
    node = HandeyeCollect()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_session.py -v`
Expected: 2 passed.

- [ ] **Step 5: Build + hardware dry-run (T2, operator-in-the-loop)**

Run:
```bash
./src/tk26_vision/scripts/build.sh --packages-select handeye_calib
source src/tk26_vision/install/setup.bash
export ROBOT_NAME=tinker2
ros2 run handeye_calib handeye_collect
```
Expected: node logs `handeye_collect ready`, subscribes to `/xarm_camera/color/*`, and on a manual pose the log shows accept/reject with a reason. (Full automated loop is wired in Task 10's web tool; this step confirms detection + gating against the live camera.) Record results in `src/tk26_vision/DEV_NOTES.md`.

- [ ] **Step 6: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/handeye_collect.py src/handeye_calib/test/test_session.py
git commit -m "feat(handeye_calib): collection node + CaptureSession accumulator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Web tool (`handeye_web`) — authoring, run, verify, promote

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_web_helpers.py`

calib_web (`src/tk26_vision/src/pan_tilt/pan_tilt/calib_web.py`) is the reference
pattern: rclpy node on the main thread, uvicorn/FastAPI on a worker, `node.lock`
for shared state, live MJPEG overlay, waypoint authoring validated against
`pan_tilt.calibration.safety.SafetyEnvelope`, subprocess runner with WebSocket log
fan-out, and diff-preview + atomic promote. Reuse that structure. Unit-test the pure
helpers: pose-set validation and the URDF/yaml diff payload.

- [ ] **Step 1: Write the failing tests for pure helpers**

`test/test_web_helpers.py`:
```python
from handeye_calib.handeye_web import validate_pose_set, diff_payload


def test_validate_pose_set_flags_short_sets():
    ok, msg = validate_pose_set([{"joints": [0] * 7} for _ in range(5)])
    assert ok is False and "at least" in msg


def test_validate_pose_set_accepts_enough():
    ok, msg = validate_pose_set([{"joints": [0] * 7} for _ in range(15)])
    assert ok is True


def test_diff_payload_shows_before_after():
    d = diff_payload(old_xyz="0.06746 -0.0175 0.0237",
                     new_xyz="0.1 0.2 0.3",
                     old_rpy="3.14159 -1.5708 0", new_rpy="0 0 0")
    assert d["xyz"]["old"] == "0.06746 -0.0175 0.0237"
    assert d["xyz"]["new"] == "0.1 0.2 0.3"
    assert d["changed"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_web_helpers.py -v`
Expected: FAIL — cannot import helpers.

- [ ] **Step 3: Implement handeye_web.py (pure helpers + node skeleton)**

```python
"""calib_web-style browser tool for eye-in-hand calibration.

Pure helpers (validate_pose_set, diff_payload) are unit-tested. The FastAPI + rclpy
server mirrors pan_tilt/calib_web.py: live overlay, pose authoring validated against
SafetyEnvelope, subprocess solve with streamed logs, verification overlay, and
diff-preview + atomic promote via handeye_calib.apply_handeye.
"""
MIN_POSES = 12


def validate_pose_set(poses):
    if len(poses) < MIN_POSES:
        return False, f"need at least {MIN_POSES} poses, got {len(poses)}"
    for i, p in enumerate(poses):
        if "joints" not in p or len(p["joints"]) != 7:
            return False, f"pose {i}: expected 7 joint values"
    return True, "ok"


def diff_payload(old_xyz, new_xyz, old_rpy, new_rpy):
    return {
        "xyz": {"old": old_xyz, "new": new_xyz},
        "rpy": {"old": old_rpy, "new": new_rpy},
        "changed": (old_xyz != new_xyz) or (old_rpy != new_rpy),
    }


def main():
    # Mirrors pan_tilt/calib_web.py main(): build rclpy node, start uvicorn worker,
    # serve the authoring/run/verify/promote UI. Hardware-tier; see README.
    import rclpy
    rclpy.init()
    # ... node + uvicorn wiring (reuse calib_web structure) ...
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/test_web_helpers.py -v`
Expected: 3 passed.

- [ ] **Step 5: Build + hardware bring-up (T3, operator-in-the-loop)**

Run:
```bash
./src/tk26_vision/scripts/build.sh --packages-select handeye_calib
source src/tk26_vision/install/setup.bash
export ROBOT_NAME=tinker2
ros2 run handeye_calib handeye_web --ros-args -p bind:=127.0.0.1 -p port:=8766
```
Expected: browser at `http://127.0.0.1:8766` shows the live overlay; authoring a pose set, running collect+solve, and previewing the diff all work; PASS banner on a good run. Record in `DEV_NOTES.md`.

- [ ] **Step 6: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/handeye_web.py src/handeye_calib/test/test_web_helpers.py
git commit -m "feat(handeye_calib): calib_web-style authoring/run/verify/promote tool

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: Full-suite gate, README, changelog

**Files:**
- Modify: `src/tk26_vision/src/handeye_calib/README.md`

- [ ] **Step 1: Run the entire unit suite**

Run: `cd src/tk26_vision/src/handeye_calib && PYTHONPATH=. ../../../.venv-vision-main/bin/python -m pytest test/ -v`
Expected: all tests pass (Tasks 1–10 unit tests).

- [ ] **Step 2: Flesh out the README (usage + Changelog)**

Append to `README.md` the operator workflow (print/mount the 5×5/40 mm board rigidly and fixed; `ros2 run handeye_calib handeye_web ...`; author poses; run; read PASS/WARN/FAIL + live overlay; promote to `hand_eye.yaml`/URDF), a "Verifying without hardware" section (`handeye_synthetic_check`), and:
```markdown
## Changelog
- 0.2.0 (2026-06-15): math core (transforms/model/solver/gates), synthetic harness,
  collection node, calib_web-style web tool, yaml/URDF persistence.
- 0.1.0 (2026-06-15): package scaffold.
```

- [ ] **Step 3: Commit**

```bash
cd src/tk26_vision
git add src/handeye_calib/README.md
git commit -m "docs(handeye_calib): operator + verification guide, changelog 0.2.0

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes (filled during writing)

- **Spec coverage:** §1 problem → Tasks 4–6/8; §2 decisions → board (Tasks 2–3), calib_web (Task 10), color anchor (Task 8 compose), gate (Task 6), Method B (Tasks 4–5). §3 prior-free → seed needs no init (Task 4). §4 settle constraint → `StabilityTracker` (Task 7), used in collect (Task 9). §6 reuse → Task 0 wiring + Tasks 9–10 imports. §7 collection flow → Tasks 7+9. §8 solver → Tasks 4–6. §9 verification → Task 6 (held-out) + Task 10 (live overlay). §10 storage → Task 8 + Task 10 promote. §11 gates → Task 6 thresholds + Task 7. §12 risks → settle (T7), diversity (T7), stale mount (T4). All covered.
- **Open spec items (§13)** are bench-time confirmations (board square length, color topic name, K/N tuning) surfaced to the operator at run time, not code gaps. Board square length flows in as a parameter everywhere (`square_len`).
- **Type consistency:** `Sample(T_base_eef, T_cam_board, obs_px, corner_idx)` used identically in model/synthetic/solve/collect; `project_corners(board_pts, T_cam_board, K, dist)` signature consistent; metrics dict keys `trans_rmse_m`/`rot_rmse_rad`/`reproj_px` consistent across `evaluate`/`gate`/`handeye_yaml_dict`.
- **No placeholders** in the unit-tested core (Tasks 1–8). Tasks 9–10 intentionally keep the rclpy/FastAPI bodies as structured skeletons (hardware-tier integration), with all pure logic extracted and fully TDD'd — the honest boundary for code that needs a live robot + camera to exercise.
