# FoundationStereo Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `foundation_stereo` ROS2 package in tk26_vision with two modes — service+action on-request inference and a streaming depth publisher — backed by the upstream FoundationStereo + Fast-FoundationStereo models. D435 default with optional color-frame alignment, D405 supported via a profile preset.

**Architecture:** Single node holds one shared `StereoRunner` (ported from `dualrRGB-foundationStereo/webapp/stereo_runner.py`, slimmed per spec §10). Three consumers — service handler, action handler, optional streaming worker — serialize through the runner's internal lock. Topic-driven input via `ApproximateTimeSynchronizer`; latched extrinsics topic feeds the color-alignment pipeline. Upstream model code vendored under `thirdparty/foundation_stereo/`; weights stay at a configurable `weights_root`. Lives in its own venv (`.venv-fs/`) like `monocular_depth/.venv-da3/` because torch 2.8+cu128 / TensorRT 10.16 conflict with the shared `.venv-vision-main`.

**Tech Stack:** Python 3.10, ROS2 Humble (`rclpy`, `rclpy_action`, `message_filters`, `cv_bridge`), `torch==2.8.0+cu128`, `tensorrt==10.16.1.11`, vendored `FoundationStereo` + `Fast-FoundationStereo`. Synthetic-data pytest for `color_align`; bag-driven integration smoke for the node.

**Spec:** `docs/superpowers/specs/2026-05-24-foundation-stereo-design.md` (commits `b8d7459` + `22cd3b4`).

---

## File map

**Create:**
- `thirdparty/foundation_stereo/FoundationStereo/` (vendored upstream, source only)
- `thirdparty/foundation_stereo/Fast-FoundationStereo/` (vendored upstream, source only)
- `thirdparty/foundation_stereo/README.md` (vendor notes — what was stripped, version pin)
- `src/tinker_vision_msgs_26/srv/FoundationStereoDepth.srv`
- `src/tinker_vision_msgs_26/action/FoundationStereoDepth.action`
- `src/foundation_stereo/package.xml`
- `src/foundation_stereo/setup.py`
- `src/foundation_stereo/setup.cfg`
- `src/foundation_stereo/requirements.txt`
- `src/foundation_stereo/resource/foundation_stereo`
- `src/foundation_stereo/foundation_stereo/__init__.py`
- `src/foundation_stereo/foundation_stereo/stereo_runner.py`
- `src/foundation_stereo/foundation_stereo/color_align.py`
- `src/foundation_stereo/foundation_stereo/_logging.py`
- `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py`
- `src/foundation_stereo/launch/foundation_stereo.launch.py`
- `src/foundation_stereo/config/foundation_stereo.yaml`
- `src/foundation_stereo/test/test_color_align.py`
- `src/foundation_stereo/test/test_logging.py`
- `src/foundation_stereo/test/test_stereo_runner_imports.py`
- `src/foundation_stereo/README.md`
- `scripts/build_foundation_stereo.sh`

**Modify:**
- `src/tinker_vision_msgs_26/CMakeLists.txt` (register new srv + action)
- `scripts/tests/t0_static.sh` (add foundation_stereo rows)
- `scripts/tests/t1_startup.sh` (add foundation_stereo rows)
- `CLAUDE.md` (architecture overview + build wrapper note)

`.venv-fs/` is provisioned manually (not in git — `.gitignore` already covers `.venv**/`).

---

## Task 1: Vendor upstream source trees under `thirdparty/`

**Files:**
- Create: `thirdparty/foundation_stereo/FoundationStereo/{core,scripts,Utils.py,depth_anything,dinov2,LICENSE,readme.md}`
- Create: `thirdparty/foundation_stereo/Fast-FoundationStereo/{core,scripts,Utils.py,LICENSE.txt,readme.md,requirements.txt,model_card.md}`
- Create: `thirdparty/foundation_stereo/README.md`

- [ ] **Step 1: Copy FoundationStereo source-only tree**

Excludes weights, .venv, .git, captures, __pycache__. The source itself is ~200 KB.

```bash
SRC=/home/tinker/projects/vision_tests/dualrRGB-foundationStereo
DST=src/tk26_vision/thirdparty/foundation_stereo
mkdir -p "$DST"
rsync -a --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='pretrained_models/' \
    --exclude='captures/' \
    --exclude='*.pth' \
    --exclude='*.engine' \
    --exclude='*.onnx' \
    "$SRC/FoundationStereo/" "$DST/FoundationStereo/"
```

- [ ] **Step 2: Copy Fast-FoundationStereo source-only tree**

```bash
SRC=/home/tinker/projects/vision_tests/dualrRGB-foundationStereo
DST=src/tk26_vision/thirdparty/foundation_stereo
rsync -a --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='weights/' \
    --exclude='output*' \
    --exclude='captures/' \
    --exclude='demo_data/' \
    --exclude='*.pth' \
    --exclude='*.engine' \
    --exclude='*.onnx' \
    "$SRC/Fast-FoundationStereo/" "$DST/Fast-FoundationStereo/"
```

- [ ] **Step 3: Sanity-check what was copied**

Run: `du -sh src/tk26_vision/thirdparty/foundation_stereo/{FoundationStereo,Fast-FoundationStereo}`
Expected: each under ~5 MB. Both `core/` directories present. No `.pth`, `.engine`, `.onnx`, or `.git` files anywhere.

```bash
find src/tk26_vision/thirdparty/foundation_stereo -name '*.pth' -o -name '*.engine' -o -name '*.onnx' -o -name '.git' | head
# expect: empty output
```

- [ ] **Step 4: Write vendor README**

Create `src/tk26_vision/thirdparty/foundation_stereo/README.md`:

```markdown
# Vendored FoundationStereo + Fast-FoundationStereo

Source-only mirror of NVIDIA's FoundationStereo (CVPR 2025) and
Fast-FoundationStereo. Stripped to just what the ROS node imports:

- `core/`, `Utils.py`, `scripts/`
- LICENSE files, readme.md, model_card.md, requirements.txt

**Excluded (kept out of the workspace):**
- `pretrained_models/` (FoundationStereo, ~3 GB)
- `weights/`, `output*` (Fast-FoundationStereo TRT engines)
- `.git/`, `.venv/`, `__pycache__/`, `captures/`, `demo_data/`

Weights / TRT engines live at the `weights_root` ROS param of the
`foundation_stereo` node (default: the original reference directory at
`/home/tinker/projects/vision_tests/dualrRGB-foundationStereo`).

To refresh this vendor copy from a newer upstream, re-run the `rsync` lines
in `docs/superpowers/plans/2026-05-24-foundation-stereo.md` Task 1.

Upstream commits at vendor time: see git log for the directory.
```

- [ ] **Step 5: Commit**

```bash
git add src/tk26_vision/thirdparty/foundation_stereo
git commit -m "feat(foundation_stereo): vendor upstream source trees

Source-only mirror of FoundationStereo + Fast-FoundationStereo from the
reference dualrRGB-foundationStereo setup. Weights, TRT engines, and the
upstream .venv stay out of the workspace; they remain at the configurable
weights_root path that the ROS node will accept."
```

---

## Task 2: Add `FoundationStereoDepth.srv` + `.action` interfaces

**Files:**
- Create: `src/tinker_vision_msgs_26/srv/FoundationStereoDepth.srv`
- Create: `src/tinker_vision_msgs_26/action/FoundationStereoDepth.action`
- Modify: `src/tinker_vision_msgs_26/CMakeLists.txt`

- [ ] **Step 1: Create the srv file**

Path: `src/tk26_vision/src/tinker_vision_msgs_26/srv/FoundationStereoDepth.srv`

```
# Request — per-call overrides (empty string / 0 means use node default)
string model_kind
string trt_variant
float32 scale
int32 iters
float32 z_far
bool want_pointcloud
bool want_debug_jpeg
bool align_to_color
---
# Response
int32 status
string error_msg
sensor_msgs/Image depth_image
sensor_msgs/CameraInfo camera_info
sensor_msgs/PointCloud2 pointcloud
sensor_msgs/CompressedImage debug_jpeg
float32 forward_ms
float32 load_s
float32 end_to_end_s
string model_used
string trt_variant_used
```

- [ ] **Step 2: Create the action file**

Path: `src/tk26_vision/src/tinker_vision_msgs_26/action/FoundationStereoDepth.action`

```
# Goal — identical fields to FoundationStereoDepth srv request
string model_kind
string trt_variant
float32 scale
int32 iters
float32 z_far
bool want_pointcloud
bool want_debug_jpeg
bool align_to_color
---
# Result — identical fields to FoundationStereoDepth srv response
int32 status
string error_msg
sensor_msgs/Image depth_image
sensor_msgs/CameraInfo camera_info
sensor_msgs/PointCloud2 pointcloud
sensor_msgs/CompressedImage debug_jpeg
float32 forward_ms
float32 load_s
float32 end_to_end_s
string model_used
string trt_variant_used
---
# Feedback
string current_stage
float32 elapsed_s
```

- [ ] **Step 3: Register both files in CMakeLists.txt**

Edit `src/tk26_vision/src/tinker_vision_msgs_26/CMakeLists.txt`. Add `"srv/FoundationStereoDepth.srv"` to the existing srv list (alphabetical order, after `"srv/FaceRegister.srv"` or wherever it fits the existing ordering). Add `"action/FoundationStereoDepth.action"` to the action list.

The relevant block (post-edit) looks like:

```cmake
  "srv/FaceRegister.srv"
  "srv/FeatureExtraction.srv"
  "srv/FeatureMatching.srv"
  "srv/FollowHead.srv"
  "srv/FoundationStereoDepth.srv"
  "srv/GetImage.srv"
  ...
  "action/FollowHeadAction.action"
  "action/FoundationStereoDepth.action"
  "action/HumanFollowing.action"
  ...
```

- [ ] **Step 4: Build the msgs package and verify both interfaces resolve**

```bash
cd /home/tinker/tk25_ws
./src/tk26_vision/scripts/build.sh --packages-select tinker_vision_msgs_26
source install/setup.bash
ros2 interface show tinker_vision_msgs_26/srv/FoundationStereoDepth | head -5
ros2 interface show tinker_vision_msgs_26/action/FoundationStereoDepth | head -5
```

Expected: both commands print the file contents without `Could not find the interface` errors.

- [ ] **Step 5: Commit**

```bash
git add src/tk26_vision/src/tinker_vision_msgs_26/srv/FoundationStereoDepth.srv \
        src/tk26_vision/src/tinker_vision_msgs_26/action/FoundationStereoDepth.action \
        src/tk26_vision/src/tinker_vision_msgs_26/CMakeLists.txt
git commit -m "feat(tinker_vision_msgs_26): add FoundationStereoDepth srv + action

Identical field schemas for the on-request service (sync) and action
(slow-backend feedback + cancel) of the upcoming foundation_stereo
package. Per spec: response carries depth_image (32FC1 m), optional
pointcloud and debug_jpeg, plus forward_ms / load_s / end_to_end_s
timing fields."
```

---

## Task 3: Package skeleton + entry-point wiring

**Files:**
- Create: `src/foundation_stereo/package.xml`
- Create: `src/foundation_stereo/setup.py`
- Create: `src/foundation_stereo/setup.cfg`
- Create: `src/foundation_stereo/resource/foundation_stereo` (empty marker file)
- Create: `src/foundation_stereo/foundation_stereo/__init__.py`
- Create: `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py` (stub with `main()`)

- [ ] **Step 1: Create package.xml**

Path: `src/tk26_vision/src/foundation_stereo/package.xml`

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>foundation_stereo</name>
  <version>0.0.1</version>
  <description>FoundationStereo + Fast-FoundationStereo ROS2 node: service + action on-request inference and an optional streaming depth publisher. Lives in its own venv (.venv-fs) because torch 2.8+cu128 / TensorRT 10.16 conflict with the shared .venv-vision-main.</description>
  <maintainer email="cindy.w0135@gmail.com">cindy</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>rclpy_action</depend>
  <depend>sensor_msgs</depend>
  <depend>tinker_vision_msgs_26</depend>
  <depend>realsense2_camera_msgs</depend>
  <depend>message_filters</depend>
  <depend>cv_bridge</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 2: Create setup.py**

Path: `src/tk26_vision/src/foundation_stereo/setup.py`

```python
from setuptools import find_packages, setup

package_name = 'foundation_stereo'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/foundation_stereo.launch.py']),
        ('share/' + package_name + '/config',
            ['config/foundation_stereo.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cindy',
    maintainer_email='cindy.w0135@gmail.com',
    description='FoundationStereo + Fast-FoundationStereo ROS2 service/action + streaming depth node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'foundation_stereo_node = foundation_stereo.foundation_stereo_node:main',
        ],
    },
)
```

- [ ] **Step 3: Create setup.cfg + resource marker**

Path: `src/tk26_vision/src/foundation_stereo/setup.cfg`

```ini
[develop]
script_dir=$base/lib/foundation_stereo
[install]
install_scripts=$base/lib/foundation_stereo
```

Then:

```bash
mkdir -p src/tk26_vision/src/foundation_stereo/resource \
         src/tk26_vision/src/foundation_stereo/foundation_stereo \
         src/tk26_vision/src/foundation_stereo/launch \
         src/tk26_vision/src/foundation_stereo/config \
         src/tk26_vision/src/foundation_stereo/test
touch src/tk26_vision/src/foundation_stereo/resource/foundation_stereo
```

- [ ] **Step 4: Create empty `__init__.py` and a `main()` stub**

Path: `src/tk26_vision/src/foundation_stereo/foundation_stereo/__init__.py`

```python
```

Path: `src/tk26_vision/src/foundation_stereo/foundation_stereo/foundation_stereo_node.py`

```python
"""FoundationStereo ROS2 node — service + action + optional streaming publisher.

Skeleton: filled out in Task 8. Existing `main()` lets `ros2 run` resolve the
entry point so packaging tests can pass before the full implementation lands.
"""

from __future__ import annotations


def main(args=None):
    raise NotImplementedError(
        "foundation_stereo_node.main is implemented in Task 8 of the plan."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Build the skeleton and verify the entry point is reachable**

```bash
cd /home/tinker/tk25_ws
./src/tk26_vision/scripts/build.sh --packages-select foundation_stereo
source install/setup.bash
ros2 pkg executables foundation_stereo
```

Expected output:

```
foundation_stereo foundation_stereo_node
```

(Running the executable will raise `NotImplementedError` — that's fine; we just need the entry point registered.)

- [ ] **Step 6: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/{package.xml,setup.py,setup.cfg,resource,foundation_stereo,launch,config,test}
git commit -m "feat(foundation_stereo): package skeleton + entry-point stub

ament_python package with a foundation_stereo_node console script that
raises NotImplementedError until Task 8 fills it in. Lets ros2 pkg
executables resolve the entry so downstream tasks can iterate without
re-doing the packaging boilerplate."
```

---

## Task 4: TDD `color_align.py` (synthetic data, no GPU)

**Files:**
- Create: `src/foundation_stereo/test/test_color_align.py`
- Create: `src/foundation_stereo/foundation_stereo/color_align.py`

- [ ] **Step 1: Write the failing tests**

Path: `src/tk26_vision/src/foundation_stereo/test/test_color_align.py`

```python
"""Synthetic-data tests for color_align.reproject_ir_to_color.

These tests exercise the reprojection math without any GPU / model
dependency, so they run in plain pytest under any environment with
numpy installed.
"""

import numpy as np
import pytest

from foundation_stereo.color_align import reproject_ir_to_color


def _intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)


def test_identity_extrinsics_preserves_depth_at_same_pixels():
    """With identity rotation, zero translation, and matching intrinsics,
    every IR1 pixel should map to its own coordinate in the color grid
    and carry the same depth value."""
    H, W = 60, 80
    depth_ir = np.full((H, W), 2.0, dtype=np.float32)  # uniform 2 m
    K_ir = _intrinsics()
    K_color = _intrinsics()
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)

    depth_color = reproject_ir_to_color(
        depth_ir, K_ir, K_color, R, T,
        out_hw=(H, W),
    )

    assert depth_color.shape == (H, W)
    assert depth_color.dtype == np.float32
    # All valid IR pixels project to themselves; allow zero holes only at
    # the borders (forward projection can leave one-pixel-wide gaps).
    interior = depth_color[2:-2, 2:-2]
    assert np.all(interior > 0), "interior should be fully filled"
    np.testing.assert_allclose(interior, 2.0, atol=1e-3)


def test_translation_shifts_projected_pixels():
    """With a +5 cm translation in X (in the IR1 optical frame), a planar
    surface at known depth should land at predictable shifted color pixels.

    For a point (X_ir, Y_ir, Z_ir) in IR1 coordinates, after applying
    P_c = R·P_ir + T with R=I and T=(0.05, 0, 0)^T:
      u_color = fx · (X_ir + 0.05) / Z_ir + cx
    With Z_ir = 1.0 m, fx = 500, a 5 cm X-shift translates to a 25-pixel
    column shift in the color grid.
    """
    H, W = 60, 80
    Z = 1.0
    depth_ir = np.full((H, W), Z, dtype=np.float32)
    K_ir = _intrinsics()
    K_color = _intrinsics()
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.05, 0.0, 0.0], dtype=np.float32)  # 5 cm in X

    depth_color = reproject_ir_to_color(
        depth_ir, K_ir, K_color, R, T,
        out_hw=(H, W),
    )

    # Pixel u=10 in IR1 (X = (10 - 320) / 500 = -0.62 m) should project to
    # u_color = 500 * (-0.62 + 0.05) / 1.0 + 320 = 285.
    # Use the centre row to pick a representative column-shift signal.
    row = depth_color[H // 2, :]
    filled = np.where(row > 0)[0]
    assert filled.size > 0
    # Left edge of the filled band should be ~25 columns to the right of
    # IR1's left edge (col 0 → col 25 ± a couple pixels of rounding).
    assert 22 <= filled[0] <= 27, f"left edge shifted to {filled[0]} (expected ~25)"


def test_zero_depth_pixels_produce_holes():
    """Invalid (zero) depth in IR1 should not contribute; output cells
    not hit by any valid projection stay zero."""
    H, W = 40, 60
    depth_ir = np.zeros((H, W), dtype=np.float32)
    K_ir = _intrinsics()
    K_color = _intrinsics()
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)

    depth_color = reproject_ir_to_color(
        depth_ir, K_ir, K_color, R, T,
        out_hw=(H, W),
    )

    assert np.all(depth_color == 0.0)


def test_collision_keeps_nearer_z():
    """When two IR pixels project to the same color pixel, the nearer
    Z wins. Construct a 2-row scene where row 0 has Z=1 m and row 1
    has Z=2 m. K_color is given a near-zero fy so both rows' projected
    v_c rounds to 0 — both land at color (u, 0) and collide."""
    H, W = 2, 4
    depth_ir = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ], dtype=np.float32)
    K_ir = _intrinsics(fx=10.0, fy=10.0, cx=2.0, cy=0.0)
    # Tiny fy_color collapses both rows to v_c≈0 → forced collision.
    K_color = _intrinsics(fx=10.0, fy=0.01, cx=2.0, cy=0.0)
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)

    depth_color = reproject_ir_to_color(
        depth_ir, K_ir, K_color, R, T,
        out_hw=(1, W),
    )

    # All cells that received any projection should equal 1.0 (the nearer
    # value), never 2.0.
    filled = depth_color[depth_color > 0]
    assert filled.size > 0
    assert np.all(filled <= 1.0 + 1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail with the import error**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
python -m pytest src/foundation_stereo/test/test_color_align.py -v
```

Expected: `ImportError: cannot import name 'reproject_ir_to_color' from 'foundation_stereo.color_align'` (or module-not-found).

- [ ] **Step 3: Implement `color_align.py`**

Path: `src/tk26_vision/src/foundation_stereo/foundation_stereo/color_align.py`

```python
"""Reproject FoundationStereo depth from the IR1 grid into the color grid.

Single pure-numpy entry point: `reproject_ir_to_color`. Used by both the
streaming worker (`stream_align_to_color=true`) and per-call service /
action requests (`align_to_color=true`). No GPU, no extra deps beyond
numpy.

Algorithm (spec §5):
  1. Backproject every valid IR1 pixel to a 3-D point in IR1 frame.
  2. Transform to color frame: P_c = R · P_ir + T.
  3. Project through K_color: (u_c, v_c) = (fx X_c / Z_c + cx, fy Y_c / Z_c + cy).
  4. Round to color pixel grid; np.minimum.at handles occlusion
     (nearer Z wins on collision).
  5. Pixels not hit by any valid projection stay zero (holes).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def reproject_ir_to_color(
    depth_ir: np.ndarray,
    K_ir: np.ndarray,
    K_color: np.ndarray,
    R_ir_to_color: np.ndarray,
    T_ir_to_color: np.ndarray,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    """Reproject `depth_ir` (m, IR1 grid) into the color camera grid.

    Args:
      depth_ir: (H_ir, W_ir) float32, metres. Zero = invalid.
      K_ir: (3, 3) intrinsics for the IR1 image.
      K_color: (3, 3) intrinsics for the color image.
      R_ir_to_color: (3, 3) rotation IR1 -> color, in the ROS optical
        convention (x right, y down, z forward).
      T_ir_to_color: (3,) translation IR1 -> color, in metres.
      out_hw: (H_color, W_color) — output image shape.

    Returns:
      (H_color, W_color) float32 depth, metres. Zero where nothing
      projected.
    """
    H_ir, W_ir = depth_ir.shape
    H_out, W_out = out_hw

    # 1. Backproject valid pixels.
    fx_ir = float(K_ir[0, 0])
    fy_ir = float(K_ir[1, 1])
    cx_ir = float(K_ir[0, 2])
    cy_ir = float(K_ir[1, 2])

    vv, uu = np.indices((H_ir, W_ir), dtype=np.float32)
    Z = depth_ir
    valid = Z > 0.0
    if not np.any(valid):
        return np.zeros(out_hw, dtype=np.float32)

    X = (uu - cx_ir) * Z / fx_ir
    Y = (vv - cy_ir) * Z / fy_ir
    pts_ir = np.stack([X, Y, Z], axis=-1)  # (H_ir, W_ir, 3)

    # 2. Transform to color frame.
    R = R_ir_to_color.astype(np.float32)
    T = T_ir_to_color.astype(np.float32).reshape(3)
    pts_c = pts_ir @ R.T + T  # (H_ir, W_ir, 3)

    Xc = pts_c[..., 0]
    Yc = pts_c[..., 1]
    Zc = pts_c[..., 2]

    # 3. Project through K_color.
    fx_c = float(K_color[0, 0])
    fy_c = float(K_color[1, 1])
    cx_c = float(K_color[0, 2])
    cy_c = float(K_color[1, 2])

    good = valid & (Zc > 1e-6)
    if not np.any(good):
        return np.zeros(out_hw, dtype=np.float32)

    u_c = fx_c * Xc / np.where(good, Zc, 1.0) + cx_c
    v_c = fy_c * Yc / np.where(good, Zc, 1.0) + cy_c

    ui = np.round(u_c).astype(np.int32)
    vi = np.round(v_c).astype(np.int32)

    in_bounds = good & (ui >= 0) & (ui < W_out) & (vi >= 0) & (vi < H_out)
    if not np.any(in_bounds):
        return np.zeros(out_hw, dtype=np.float32)

    flat_idx = (vi[in_bounds] * W_out + ui[in_bounds]).astype(np.intp)
    z_values = Zc[in_bounds].astype(np.float32)

    # 4. Occlusion: nearer Z wins on collision. Seed with +inf and take min.
    depth_out = np.full(H_out * W_out, np.inf, dtype=np.float32)
    np.minimum.at(depth_out, flat_idx, z_values)

    # 5. Holes left as 0 (rather than inf).
    depth_out[np.isinf(depth_out)] = 0.0
    return depth_out.reshape(H_out, W_out)
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
python -m pytest src/foundation_stereo/test/test_color_align.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/foundation_stereo/color_align.py \
        src/tk26_vision/src/foundation_stereo/test/test_color_align.py
git commit -m "feat(foundation_stereo): color_align — IR1 depth to color grid

Pure-numpy reprojection used by the streaming worker (when
stream_align_to_color=true) and per-call service/action requests
(align_to_color=true). np.minimum.at handles occlusion; pixels not hit
by any valid projection stay zero. Synthetic-data tests cover identity,
known X-translation shift, all-zero input, and collision (nearer Z wins)."
```

---

## Task 5: TDD `_logging.py` (vision_log session resolver)

**Files:**
- Create: `src/foundation_stereo/test/test_logging.py`
- Create: `src/foundation_stereo/foundation_stereo/_logging.py`

- [ ] **Step 1: Write the failing tests**

Path: `src/tk26_vision/src/foundation_stereo/test/test_logging.py`

```python
"""Tests for the vision_log session-directory resolver.

Resolution order per tk26_vision convention (see top-level CLAUDE.md):
  1. $TINKER_VISION_SESSION_TS (must match YYYYmmdd_HHMMSS).
  2. Newest existing `<base>/<YYYYmmdd_HHMMSS>/` subdir by mtime.
  3. Fresh `strftime` cold-start.
"""

import os
import time

import pytest

from foundation_stereo._logging import resolve_session_dir


def test_env_var_takes_priority(tmp_path, monkeypatch):
    base = tmp_path / "vision_log"
    base.mkdir()
    monkeypatch.setenv("TINKER_VISION_SESSION_TS", "20260101_120000")
    out = resolve_session_dir(str(base))
    assert out == str(base / "20260101_120000")
    assert os.path.isdir(out)


def test_env_var_rejected_if_malformed(tmp_path, monkeypatch):
    base = tmp_path / "vision_log"
    base.mkdir()
    monkeypatch.setenv("TINKER_VISION_SESSION_TS", "not-a-timestamp")
    out = resolve_session_dir(str(base))
    # Falls through to fresh-strftime cold-start; basename matches YYYYmmdd_HHMMSS.
    assert os.path.basename(out).count("_") == 1
    assert len(os.path.basename(out)) == 15


def test_newest_subdir_wins_when_env_missing(tmp_path, monkeypatch):
    base = tmp_path / "vision_log"
    base.mkdir()
    older = base / "20250101_000000"
    newer = base / "20260101_000000"
    older.mkdir()
    time.sleep(0.05)
    newer.mkdir()
    monkeypatch.delenv("TINKER_VISION_SESSION_TS", raising=False)

    out = resolve_session_dir(str(base))
    assert out == str(newer)


def test_cold_start_when_no_subdirs(tmp_path, monkeypatch):
    base = tmp_path / "vision_log"
    base.mkdir()
    monkeypatch.delenv("TINKER_VISION_SESSION_TS", raising=False)

    out = resolve_session_dir(str(base))
    assert os.path.dirname(out) == str(base)
    assert len(os.path.basename(out)) == 15
    assert os.path.isdir(out)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python -m pytest src/tk26_vision/src/foundation_stereo/test/test_logging.py -v
```

Expected: `ImportError: cannot import name 'resolve_session_dir' from 'foundation_stereo._logging'`.

- [ ] **Step 3: Implement `_logging.py`**

Path: `src/tk26_vision/src/foundation_stereo/foundation_stereo/_logging.py`

```python
"""Vision-log session-directory resolver — shared with the rest of tk26.

Matches the resolution order documented in src/tk26_vision/CLAUDE.md:
  1. $TINKER_VISION_SESSION_TS (must match YYYYmmdd_HHMMSS).
  2. Newest existing <base>/<YYYYmmdd_HHMMSS>/ subdir by mtime — lets
     late-spawned standalone nodes join the active session.
  3. Fresh strftime cold-start.
"""

from __future__ import annotations

import os
import re
import time

_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def resolve_session_dir(base: str) -> str:
    """Return the active session directory, creating it if necessary."""
    os.makedirs(base, exist_ok=True)

    env_ts = os.environ.get("TINKER_VISION_SESSION_TS", "")
    if _TS_RE.match(env_ts):
        out = os.path.join(base, env_ts)
        os.makedirs(out, exist_ok=True)
        return out

    candidates = []
    for entry in os.listdir(base):
        path = os.path.join(base, entry)
        if _TS_RE.match(entry) and os.path.isdir(path):
            candidates.append((os.path.getmtime(path), path))
    if candidates:
        candidates.sort()
        return candidates[-1][1]

    fresh = os.path.join(base, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(fresh, exist_ok=True)
    return fresh
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
python -m pytest src/tk26_vision/src/foundation_stereo/test/test_logging.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/foundation_stereo/_logging.py \
        src/tk26_vision/src/foundation_stereo/test/test_logging.py
git commit -m "feat(foundation_stereo): vision_log session resolver

Re-implements tk26_vision's shared session-dir convention (env var ->
newest sibling by mtime -> fresh strftime) inside the foundation_stereo
package so it joins the rest of the vision_log/<ts>/ tree when other
nodes are running."
```

---

## Task 6: Port `stereo_runner.py` with the overhead cuts

**Files:**
- Create: `src/foundation_stereo/foundation_stereo/stereo_runner.py`
- Create: `src/foundation_stereo/test/test_stereo_runner_imports.py`

- [ ] **Step 1: Write the import-shape test**

This test runs only when torch + the vendored trees are importable (i.e. under `.venv-fs`). It validates the namespace-swap survives both backends.

Path: `src/tk26_vision/src/foundation_stereo/test/test_stereo_runner_imports.py`

```python
"""Import-shape tests for stereo_runner.

These exercise the namespace-swap logic (FoundationStereo vs
Fast-FoundationStereo both ship a top-level `core/` package with
overlapping module names). The tests require torch and the vendored
thirdparty trees — they're skipped if either is missing, so the rest
of the foundation_stereo suite still runs in a vanilla venv.
"""

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

_VENDOR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "thirdparty",
                 "foundation_stereo")
)
_FS = os.path.join(_VENDOR_ROOT, "FoundationStereo")
_FAST = os.path.join(_VENDOR_ROOT, "Fast-FoundationStereo")

if not (os.path.isdir(_FS) and os.path.isdir(_FAST)):
    pytest.skip("vendored foundation_stereo trees not present",
                allow_module_level=True)


def test_runner_module_imports():
    """The runner module itself must import without instantiating any model."""
    from foundation_stereo import stereo_runner
    assert hasattr(stereo_runner, "StereoRunner")
    assert hasattr(stereo_runner, "InferResult")
    assert hasattr(stereo_runner, "TRT_VARIANTS")


def test_namespace_swap_to_upstream_then_fast():
    """After swapping into upstream then Fast, the right `core.foundation_stereo`
    is on sys.path each time and the cached version doesn't leak across."""
    from foundation_stereo import stereo_runner
    stereo_runner._swap_namespace(_FS)
    import core.foundation_stereo as upstream_core   # noqa: F401
    assert hasattr(upstream_core, "FoundationStereo")

    stereo_runner._swap_namespace(_FAST)
    import core.foundation_stereo as fast_core
    assert hasattr(fast_core, "TrtRunner")


def test_default_iters_table_complete():
    """Every PyTorch backend kind must have a default-iters entry."""
    from foundation_stereo import stereo_runner
    for kind in ("vitl", "vits", "fast_fp32", "fast_fp16"):
        assert kind in stereo_runner._DEFAULT_ITERS
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
# In the FS venv if you have one already; otherwise pytest skips the file.
python -m pytest src/tk26_vision/src/foundation_stereo/test/test_stereo_runner_imports.py -v
```

Expected (without torch): one `skipped` line. Expected (with torch + vendor trees): `ModuleNotFoundError: foundation_stereo.stereo_runner`.

- [ ] **Step 3: Create `stereo_runner.py` (lean port from the reference)**

Path: `src/tk26_vision/src/foundation_stereo/foundation_stereo/stereo_runner.py`

This is a direct adaptation of `dualrRGB-foundationStereo/webapp/stereo_runner.py` with the cuts from spec §10:
- Remove `_gpu_rss_mib()` entirely.
- Remove `reset_peak_memory_stats()`, `max_memory_allocated()`, `max_memory_reserved()` calls and the `peak_*` fields on `InferResult`.
- Make CUDA-event timing opt-in via the `measure_forward_ms` argument to `infer()`.
- Resolve the vendored thirdparty path automatically.

```python
"""FoundationStereo + Fast-FoundationStereo runner — ROS2-stripped port.

Lifted from dualrRGB-foundationStereo/webapp/stereo_runner.py with the
overhead cuts from docs/superpowers/specs/2026-05-24-foundation-stereo-design.md §10:
- no nvidia-smi subprocess
- no PyTorch peak-memory counters
- CUDA-event timing is opt-in via measure_forward_ms

Single-slot model cache, evicted on backend or TRT-variant switch. One
internal threading.Lock serializes GPU access across all callers (service,
action, streaming worker).
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


# Resolve the vendored thirdparty tree. This file lives at
# src/tk26_vision/src/foundation_stereo/foundation_stereo/stereo_runner.py;
# the vendor root is ../../../thirdparty/foundation_stereo/.
_THIS = os.path.dirname(os.path.realpath(__file__))
_VENDOR_ROOT = os.path.realpath(
    os.path.join(_THIS, "..", "..", "..", "thirdparty", "foundation_stereo")
)
_FS_DIR = os.path.join(_VENDOR_ROOT, "FoundationStereo")
_FAST_DIR = os.path.join(_VENDOR_ROOT, "Fast-FoundationStereo")


def _swap_namespace(target_dir: str) -> None:
    """Evict cached `core.*` / `Utils` modules and put `target_dir` at sys.path[0].

    Both FoundationStereo and Fast-FoundationStereo ship a top-level `core/`
    package + `Utils.py` with the same module names but different classes.
    Without this swap, importing one after the other leaks the cached
    classes.
    """
    for name in list(sys.modules):
        if name == "Utils" or name == "core" or name.startswith("core."):
            del sys.modules[name]
    for d in (_FS_DIR, _FAST_DIR):
        while d in sys.path:
            sys.path.remove(d)
    sys.path.insert(0, target_dir)


def _discover_trt_variants(weights_root: str) -> dict:
    """Find any directory under `<weights_root>/Fast-FoundationStereo/` that
    contains a complete two-stage TRT engine set."""
    fast_root = os.path.join(weights_root, "Fast-FoundationStereo")
    out = {}
    if not os.path.isdir(fast_root):
        return out
    for entry in sorted(os.listdir(fast_root)):
        d = os.path.join(fast_root, entry)
        if not os.path.isdir(d):
            continue
        needed = ("feature_runner.engine", "post_runner.engine", "onnx.yaml")
        if all(os.path.exists(os.path.join(d, f)) for f in needed):
            out[entry] = d
    return out


_DEFAULT_ITERS = {"vitl": 32, "vits": 32, "fast_fp32": 8, "fast_fp16": 8}

# Filled in lazily by `StereoRunner.__init__` when `weights_root` is known.
TRT_VARIANTS: dict = {}


@dataclass
class InferResult:
    disp: np.ndarray
    depth: np.ndarray
    vis_jpg: bytes                  # JPEG-encoded disparity vis; empty if not requested
    scale_used: float
    load_s: float = 0.0
    forward_ms: float = 0.0         # 0.0 if measure_forward_ms=False
    forward_s: float = 0.0          # always populated (wall clock)
    post_s: float = 0.0


class StereoRunner:
    def __init__(self, weights_root: str):
        self._weights_root = weights_root
        self._fs_pretrained = os.path.join(
            weights_root, "FoundationStereo", "pretrained_models"
        )
        self._fast_pickle = os.path.join(
            weights_root, "Fast-FoundationStereo", "weights",
            "23-36-37", "model_best_bp2_serialize.pth",
        )
        global TRT_VARIANTS
        TRT_VARIANTS = _discover_trt_variants(weights_root)
        self._default_trt_variant = (
            "output_two_stage" if "output_two_stage" in TRT_VARIANTS
            else next(iter(TRT_VARIANTS), None)
        )
        self._ckpt_map = {
            "vitl":      os.path.join(self._fs_pretrained, "23-51-11", "model_best_bp2.pth"),
            "vits":      os.path.join(self._fs_pretrained, "11-33-40", "model_best_bp2.pth"),
            "fast_fp32": self._fast_pickle,
            "fast_fp16": self._fast_pickle,
            "fast_trt":  TRT_VARIANTS.get(self._default_trt_variant)
                         if self._default_trt_variant else None,
        }

        self._model = None
        self._model_kind: Optional[str] = None
        self._trt_variant: Optional[str] = None
        self._trt_input_hw: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    @property
    def current_model(self) -> Optional[str]:
        return self._model_kind

    @property
    def current_trt_variant(self) -> Optional[str]:
        return self._trt_variant

    def _resolve_variant(self, kind: str, variant: Optional[str]) -> Optional[str]:
        if kind != "fast_trt":
            return None
        if variant in TRT_VARIANTS:
            return variant
        return self._default_trt_variant

    def _ensure_model(self, kind: str, variant: Optional[str] = None) -> None:
        assert kind in self._ckpt_map, f"unknown model kind {kind}"
        resolved = self._resolve_variant(kind, variant)
        cache_key = (kind, resolved)
        current_key = (self._model_kind, self._trt_variant)
        if cache_key == current_key and self._model is not None:
            return

        if self._model is not None:
            logging.info(f"[stereo_runner] freeing {self._model_kind} model")
            del self._model
            self._model = None
            self._model_kind = None
            self._trt_variant = None
            self._trt_input_hw = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.autograd.set_grad_enabled(False)

        if kind in ("fast_fp32", "fast_fp16"):
            pickle = self._fast_pickle
            if not os.path.isfile(pickle):
                raise FileNotFoundError(f"weights missing: {pickle}")
            logging.info(f"[stereo_runner] loading {kind} from {pickle}")
            _swap_namespace(_FAST_DIR)
            model = torch.load(pickle, map_location="cpu", weights_only=False)
            model.cuda().eval()

        elif kind == "fast_trt":
            if resolved is None:
                raise RuntimeError(
                    f"no two-stage TRT engines found under "
                    f"{self._weights_root}/Fast-FoundationStereo/"
                )
            trt_dir = TRT_VARIANTS[resolved]
            logging.info(f"[stereo_runner] loading {kind} variant={resolved} from {trt_dir}")
            _swap_namespace(_FAST_DIR)
            cfg = OmegaConf.load(os.path.join(trt_dir, "onnx.yaml"))
            from core.foundation_stereo import TrtRunner  # noqa: WPS433
            feat_eng = os.path.join(trt_dir, "feature_runner.engine")
            post_eng = os.path.join(trt_dir, "post_runner.engine")
            model = TrtRunner(cfg, feat_eng, post_eng).cuda().eval()
            self._trt_input_hw = (int(cfg.image_size[0]), int(cfg.image_size[1]))
            self._trt_variant = resolved

        elif kind in ("vitl", "vits"):
            ckpt_path = self._ckpt_map[kind]
            cfg_yaml = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"weights missing: {ckpt_path}")
            logging.info(f"[stereo_runner] loading {kind} from {ckpt_path}")
            _swap_namespace(_FS_DIR)
            cfg = OmegaConf.load(cfg_yaml)
            if "vit_size" not in cfg:
                cfg["vit_size"] = "vitl"
            cfg["mixed_precision"] = True
            cfg["valid_iters"] = 32
            cfg["hiera"] = 0
            cfg["low_memory"] = 0
            cfg["corr_implementation"] = cfg.get("corr_implementation", "reg")
            args = OmegaConf.create(cfg)

            from core.foundation_stereo import FoundationStereo  # noqa: WPS433
            model = FoundationStereo(args)
            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model.cuda().eval()

        else:
            raise ValueError(f"unknown kind: {kind}")

        self._model = model
        self._model_kind = kind

    def infer(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        K: np.ndarray,
        baseline: float,
        kind: str = "fast_trt",
        scale: float = 0.5,
        valid_iters: Optional[int] = None,
        z_far: float = 10.0,
        remove_invisible: bool = True,
        trt_variant: Optional[str] = None,
        live: bool = False,
        measure_forward_ms: bool = True,
        want_debug_jpeg: bool = False,
    ) -> InferResult:
        """Run one inference.

        Args:
          left_rgb, right_rgb: (H, W, 3) uint8 stereo pair (RGB order).
          K: (3, 3) intrinsics of the *left* camera, at original
             resolution. Used to derive depth from disparity after scaling.
          baseline: stereo baseline in metres (positive).
          kind: model kind; see _DEFAULT_ITERS keys + 'fast_trt'.
          scale: image-resize factor before inference. Ignored for fast_trt
             (engine input shape is baked).
          valid_iters: per-backend iteration count override.
          z_far: depth clamp in metres.
          remove_invisible: drop pixels whose match would lie outside the
             right image (the reference flag from the upstream demo).
          trt_variant: directory basename inside Fast-FoundationStereo/.
          live: when True, skip depth math / point-cloud build entirely.
             Returns only disparity + JPEG vis.
          measure_forward_ms: when True, record CUDA events around
             model.forward to populate `forward_ms`. ~100 µs sync cost.
          want_debug_jpeg: when True, JPEG-encode the disparity vis into
             InferResult.vis_jpg.
        """
        assert left_rgb.ndim == 3 and left_rgb.shape[2] == 3
        assert right_rgb.shape == left_rgb.shape

        with self._lock:
            resolved = self._resolve_variant(kind, trt_variant)
            cache_key = (kind, resolved)
            current_key = (self._model_kind, self._trt_variant)
            need_load = (cache_key != current_key) or (self._model is None)
            t_load = time.time()
            self._ensure_model(kind, variant=trt_variant)
            load_s = (time.time() - t_load) if need_load else 0.0

            return self._run(
                left_rgb, right_rgb, K, baseline, scale, valid_iters, z_far,
                remove_invisible, live=live,
                measure_forward_ms=measure_forward_ms,
                want_debug_jpeg=want_debug_jpeg, load_s=load_s,
            )

    def _run(self, left_rgb, right_rgb, K, baseline, scale, valid_iters, z_far,
             remove_invisible, *, live, measure_forward_ms, want_debug_jpeg,
             load_s):
        K = K.astype(np.float32).copy()
        scale = float(min(max(scale, 0.05), 1.0))

        img0 = cv2.resize(left_rgb, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(right_rgb, fx=scale, fy=scale, dsize=None)
        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        forward_ms = 0.0
        forward_s = 0.0
        padder = None

        try:
            with torch.inference_mode():
                t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
                t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

                use_events = measure_forward_ms and torch.cuda.is_available()
                if use_events:
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                wall_t0 = time.time()

                if self._model_kind == "fast_trt":
                    Heng, Weng = self._trt_input_hw
                    t0e = torch.nn.functional.interpolate(
                        t0, size=(Heng, Weng), mode="bilinear", align_corners=False)
                    t1e = torch.nn.functional.interpolate(
                        t1, size=(Heng, Weng), mode="bilinear", align_corners=False)
                    if use_events: start_evt.record()
                    disp_e = self._model.forward(t0e, t1e)
                    if use_events: end_evt.record()
                    disp_up = torch.nn.functional.interpolate(
                        disp_e.float(), size=(H, W), mode="bilinear", align_corners=False)
                    disp = (disp_up * (float(W) / float(Weng))).clamp_min(0).data.cpu().numpy().reshape(H, W)
                else:
                    from core.utils.utils import InputPadder  # noqa: WPS433
                    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
                    t0, t1 = padder.pad(t0, t1)

                    iters = (valid_iters if valid_iters else
                             _DEFAULT_ITERS.get(self._model_kind, 32))
                    if self._model_kind == "fast_fp32":
                        with torch.amp.autocast("cuda", enabled=False):
                            if use_events: start_evt.record()
                            disp = self._model.forward(t0, t1, iters=iters, test_mode=True,
                                                       optimize_build_volume="pytorch1")
                            if use_events: end_evt.record()
                    elif self._model_kind == "fast_fp16":
                        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                            if use_events: start_evt.record()
                            disp = self._model.forward(t0, t1, iters=iters, test_mode=True,
                                                       optimize_build_volume="pytorch1")
                            if use_events: end_evt.record()
                    else:  # vitl / vits
                        with torch.cuda.amp.autocast(True):
                            if use_events: start_evt.record()
                            disp = self._model.forward(t0, t1, iters=iters, test_mode=True)
                            if use_events: end_evt.record()
                    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W)

                if use_events:
                    torch.cuda.synchronize()
                    forward_ms = float(start_evt.elapsed_time(end_evt))
                forward_s = time.time() - wall_t0
        finally:
            try:
                del t0, t1, padder
            except NameError:
                pass
            try:
                del t0e, t1e, disp_e, disp_up
            except NameError:
                pass

        post_t0 = time.time()

        # Disparity vis only when explicitly asked.
        vis_jpg = b""
        if want_debug_jpeg:
            from Utils import vis_disparity  # noqa: WPS433
            vis = vis_disparity(disp)
            vis_stacked = np.concatenate([img0_ori, vis], axis=1)
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(vis_stacked, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            vis_jpg = buf.tobytes() if ok else b""

        if live:
            return InferResult(
                disp=disp,
                depth=np.empty(0, dtype=np.float32),
                vis_jpg=vis_jpg,
                scale_used=scale,
                forward_ms=forward_ms,
                forward_s=forward_s,
                post_s=time.time() - post_t0,
                load_s=load_s,
            )

        # Depth at the resized scale; intrinsics scaled accordingly.
        K_scaled = K.copy()
        K_scaled[:2] *= scale

        disp_for_depth = disp.copy()
        if remove_invisible:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            us_right = xx - disp_for_depth
            disp_for_depth[us_right < 0] = np.inf

        depth = K_scaled[0, 0] * baseline / np.where(
            disp_for_depth > 0, disp_for_depth, np.inf)
        depth = np.where((depth > 0) & (depth <= z_far), depth, 0.0).astype(np.float32)
        post_s = time.time() - post_t0

        return InferResult(
            disp=disp,
            depth=depth,
            vis_jpg=vis_jpg,
            scale_used=scale,
            forward_ms=forward_ms,
            forward_s=forward_s,
            post_s=post_s,
            load_s=load_s,
        )
```

- [ ] **Step 4: Provision `.venv-fs/` and run the import-shape test**

If `.venv-fs/` does not yet exist on this workstation, provision it now (one-time, ~5 minutes):

```bash
cd src/tk26_vision
python3.10 -m venv .venv-fs --system-site-packages --symlinks
source .venv-fs/bin/activate
pip install --upgrade pip wheel
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 omegaconf opencv-python-headless imageio einops safetensors huggingface_hub pillow addict
pip install tensorrt==10.16.1.11
pip freeze > .venv-fs/freeze.lock.txt
```

Then run the import-shape test under that venv:

```bash
.venv-fs/bin/python -m pytest src/foundation_stereo/test/test_stereo_runner_imports.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/foundation_stereo/stereo_runner.py \
        src/tk26_vision/src/foundation_stereo/test/test_stereo_runner_imports.py
git commit -m "feat(foundation_stereo): port stereo_runner (lean, no nvidia overhead)

Adapts dualrRGB-foundationStereo/webapp/stereo_runner.py with the spec §10
cuts: no _gpu_rss_mib subprocess, no PyTorch peak-memory counters, CUDA
event timing opt-in via measure_forward_ms. Vendored thirdparty path is
auto-resolved; weights_root is constructor-injected. Import-shape tests
cover the namespace-swap between upstream and Fast variants."
```

---

## Task 7: Build wrapper script + `.venv-fs` documentation

**Files:**
- Create: `scripts/build_foundation_stereo.sh`
- Create: `src/foundation_stereo/requirements.txt`

- [ ] **Step 1: Write `requirements.txt`**

Path: `src/tk26_vision/src/foundation_stereo/requirements.txt`

```
# .venv-fs/ pins (provisioned manually per src/foundation_stereo/README.md).
# torch / torchvision / tensorrt are not listed here because they require
# the --index-url install dance — see the README for the canonical recipe.
numpy==1.26.4
omegaconf
opencv-python-headless
imageio
einops
safetensors
huggingface_hub
pillow
addict
```

- [ ] **Step 2: Write the build wrapper**

Path: `src/tk26_vision/scripts/build_foundation_stereo.sh`

```bash
#!/usr/bin/env bash
# Build the foundation_stereo ROS2 package under .venv-fs.
#
# Mirrors src/tk26_vision/scripts/build_monocular_depth.sh. Defaults to a
# single-package build to avoid cross-venv accidents; pass any extra
# --packages-select / --packages-up-to args to override.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
WS_ROOT="${WS_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"
VENV="${VENV:-$REPO_ROOT/.venv-fs}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "error: .venv-fs activate not found: $VENV/bin/activate" >&2
    echo "       provision it first — see src/foundation_stereo/README.md" >&2
    exit 1
fi
if [ ! -f "$ROS_SETUP" ]; then
    echo "error: ROS setup not found: $ROS_SETUP" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# shellcheck disable=SC1091
source "$ROS_SETUP"
set -u
export ROS2_PTH_WARNED=1

cd "$WS_ROOT"

if [ "$#" -eq 0 ]; then
    set -- --packages-select foundation_stereo
fi

colcon build --symlink-install "$@"

ENTRY_DIR="$WS_ROOT/install/foundation_stereo/lib/foundation_stereo"
TARGET_PY="$VENV/bin/python3"
if [ -d "$ENTRY_DIR" ]; then
    for script in "$ENTRY_DIR"/*; do
        [ -f "$script" ] || continue
        first_line="$(head -n 1 -- "$script" 2>/dev/null || true)"
        case "$first_line" in
            "#!"*python*)
                if [ "$first_line" != "#!$TARGET_PY" ]; then
                    sed -i "1c#!$TARGET_PY" "$script"
                    echo "patched: $script -> $TARGET_PY"
                fi
                ;;
        esac
    done
fi

echo "foundation_stereo build complete (venv: $VENV)"
```

- [ ] **Step 3: Make it executable and run it**

```bash
chmod +x src/tk26_vision/scripts/build_foundation_stereo.sh
./src/tk26_vision/scripts/build_foundation_stereo.sh
```

Expected: `Summary: 1 package finished` for `foundation_stereo`. The entry script `install/foundation_stereo/lib/foundation_stereo/foundation_stereo_node` should have shebang `#!<workspace>/src/tk26_vision/.venv-fs/bin/python3` after the patch step.

Verify:

```bash
head -1 install/foundation_stereo/lib/foundation_stereo/foundation_stereo_node
```

- [ ] **Step 4: Commit**

```bash
git add src/tk26_vision/scripts/build_foundation_stereo.sh \
        src/tk26_vision/src/foundation_stereo/requirements.txt
git commit -m "feat(foundation_stereo): build wrapper + requirements pins

scripts/build_foundation_stereo.sh mirrors build_monocular_depth.sh:
source .venv-fs + ROS, default to --packages-select foundation_stereo,
re-shebang the entry script so ros2 run resolves torch / tensorrt /
the vendored core modules from the right venv."
```

---

## Task 8: Implement the ROS2 node — service path first

**Files:**
- Modify: `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py`

This is the largest task. We build it in three commits: service path (this task), action path (Task 9), streaming worker (Task 10). Each step below ends with the node passing a smoke test before moving on.

- [ ] **Step 1: Replace the stub with the full node skeleton — parameters + sync subscriber + service**

Path: `src/tk26_vision/src/foundation_stereo/foundation_stereo/foundation_stereo_node.py`

```python
"""FoundationStereo ROS2 node — service + action + optional streaming worker.

Spec: docs/superpowers/specs/2026-05-24-foundation-stereo-design.md.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import time

from tinker_vision_msgs_26.srv import FoundationStereoDepth as FSSrv

from foundation_stereo.color_align import reproject_ir_to_color
from foundation_stereo.stereo_runner import StereoRunner, TRT_VARIANTS


# Per-camera-profile defaults. Picked by the `camera_profile` ROS param.
_PROFILES = {
    "d435": dict(
        left_topic="/camera/xarm_camera/infra1/image_rect_raw",
        right_topic="/camera/xarm_camera/infra2/image_rect_raw",
        left_info_topic="/camera/xarm_camera/infra1/camera_info",
        color_info_topic="/camera/xarm_camera/color/camera_info",
        extrinsics_topic="/camera/xarm_camera/extrinsics/depth_to_color",
        baseline_m=0.050,
    ),
    "d405": dict(
        left_topic="/camera/camera/infra1/image_rect_raw",
        right_topic="/camera/camera/infra2/image_rect_raw",
        left_info_topic="/camera/camera/infra1/camera_info",
        color_info_topic="/camera/camera/color/camera_info",
        extrinsics_topic="/camera/camera/extrinsics/depth_to_color",
        baseline_m=0.018,
    ),
}


def _info_to_K(info: CameraInfo) -> np.ndarray:
    """Convert a sensor_msgs/CameraInfo into a (3, 3) intrinsics matrix.
    Prefers the rectified-projection K-block of P over plain K when both
    are populated — matches how realsense2_camera publishes infra1's
    rect intrinsics through P."""
    P = np.asarray(info.p, dtype=np.float32).reshape(3, 4)
    if np.any(P[:3, :3] != 0):
        return P[:3, :3].copy()
    return np.asarray(info.k, dtype=np.float32).reshape(3, 3).copy()


class FoundationStereoNode(Node):

    def __init__(self):
        super().__init__("foundation_stereo")
        self._declare_parameters()
        self._bridge = CvBridge()

        self._runner = StereoRunner(weights_root=self._p("weights_root"))

        # Latest synced stereo triple (left, right, info), under a lock.
        self._latest_lock = threading.Lock()
        self._latest: Optional[Tuple[Image, Image, CameraInfo]] = None

        # Latched-style holders for color CameraInfo + IR1→Color extrinsics.
        self._color_info: Optional[CameraInfo] = None
        self._extrinsics: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self._setup_subscribers()
        self._setup_service()

        self.get_logger().info(
            f"foundation_stereo ready: profile={self._p('camera_profile')}, "
            f"default_model={self._p('default_model_kind')}, "
            f"weights_root={self._p('weights_root')}, "
            f"stream_enabled={self._p('stream_enabled')}, "
            f"trt_variants={list(TRT_VARIANTS.keys())}"
        )

    # ---------- parameters ----------

    def _declare_parameters(self) -> None:
        self.declare_parameter("weights_root",
                               "/home/tinker/projects/vision_tests/dualrRGB-foundationStereo")
        self.declare_parameter("camera_profile", "d435")
        self.declare_parameter("default_model_kind", "fast_trt")
        self.declare_parameter("default_trt_variant", "output_two_stage")
        self.declare_parameter("default_scale", 0.5)
        self.declare_parameter("default_iters", 0)
        self.declare_parameter("default_z_far", 10.0)

        # Topic params with profile-derived defaults applied at runtime
        # (declared empty so the profile fills them in).
        self.declare_parameter("left_topic", "")
        self.declare_parameter("right_topic", "")
        self.declare_parameter("left_info_topic", "")
        self.declare_parameter("color_info_topic", "")
        self.declare_parameter("extrinsics_topic", "")
        self.declare_parameter("baseline_m", 0.0)

        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("sync_queue_size", 5)
        self.declare_parameter("measure_forward_ms", True)

        # Streaming-mode params — declared even when stream_enabled=false
        # so the launch file can preset them uniformly.
        self.declare_parameter("stream_enabled", False)
        self.declare_parameter("stream_align_to_color", True)
        self.declare_parameter("stream_depth_topic", "")
        self.declare_parameter("stream_info_topic", "")
        self.declare_parameter("stream_dtype", "16UC1_mm")
        self.declare_parameter("output_frame_id", "")
        self.declare_parameter("stream_publish_vis", False)
        self.declare_parameter("stream_max_fps", 0.0)
        self.declare_parameter("extrinsics_warmup_timeout_sec", 5.0)
        self.declare_parameter("stream_measure_forward_ms", False)

        self.declare_parameter("vision_logging_enabled", False)
        self.declare_parameter("vision_log_folder", "vision_log")

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _topic_for(self, key: str) -> str:
        explicit = self._p(key)
        if explicit:
            return explicit
        profile = self._p("camera_profile")
        if profile not in _PROFILES:
            raise ValueError(f"unknown camera_profile: {profile!r}")
        return _PROFILES[profile][key]

    def _baseline(self) -> float:
        explicit = float(self._p("baseline_m"))
        if explicit > 0:
            return explicit
        profile = self._p("camera_profile")
        return float(_PROFILES[profile]["baseline_m"])

    # ---------- subscribers ----------

    def _setup_subscribers(self) -> None:
        sub_left = Subscriber(self, Image, self._topic_for("left_topic"),
                              qos_profile=qos_profile_sensor_data)
        sub_right = Subscriber(self, Image, self._topic_for("right_topic"),
                               qos_profile=qos_profile_sensor_data)
        sub_info = Subscriber(self, CameraInfo, self._topic_for("left_info_topic"),
                              qos_profile=qos_profile_sensor_data)
        self._sync = ApproximateTimeSynchronizer(
            [sub_left, sub_right, sub_info],
            queue_size=int(self._p("sync_queue_size")),
            slop=float(self._p("sync_slop_sec")),
        )
        self._sync.registerCallback(self._on_synced)

        # Color CameraInfo (one-shot latest cache).
        self.create_subscription(
            CameraInfo, self._topic_for("color_info_topic"),
            self._on_color_info, qos_profile_sensor_data,
        )
        # IR1->Color extrinsics (latched; small dance to avoid hard dep at import time).
        try:
            from realsense2_camera_msgs.msg import Extrinsics  # type: ignore
            self.create_subscription(
                Extrinsics, self._topic_for("extrinsics_topic"),
                self._on_extrinsics, qos_profile_sensor_data,
            )
        except ImportError:
            self.get_logger().warn(
                "realsense2_camera_msgs not available; color alignment disabled."
            )

    def _on_synced(self, left: Image, right: Image, info: CameraInfo) -> None:
        with self._latest_lock:
            self._latest = (left, right, info)

    def _on_color_info(self, info: CameraInfo) -> None:
        self._color_info = info

    def _on_extrinsics(self, msg) -> None:
        # realsense2_camera_msgs/Extrinsics: rotation (row-major 3x3),
        # translation (3,). They sit in the librealsense *optical* CS;
        # ROS optical CS is identical (x right, y down, z forward), so
        # we can use them directly.
        R = np.asarray(msg.rotation, dtype=np.float32).reshape(3, 3)
        T = np.asarray(msg.translation, dtype=np.float32).reshape(3)
        self._extrinsics = (R, T)

    # ---------- service ----------

    def _setup_service(self) -> None:
        self.create_service(FSSrv, "~/get_depth", self._on_get_depth)

    def _on_get_depth(self, req: FSSrv.Request, resp: FSSrv.Response) -> FSSrv.Response:
        wall_t0 = time.time()

        with self._latest_lock:
            cached = self._latest
        if cached is None:
            resp.status = 1
            resp.error_msg = "no synced stereo frame"
            return resp

        left_msg, right_msg, info_msg = cached
        try:
            left = self._bridge.imgmsg_to_cv2(left_msg, desired_encoding="rgb8")
            right = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding="rgb8")
        except Exception as exc:
            resp.status = 3
            resp.error_msg = f"cv_bridge: {exc}"
            return resp

        K_ir = _info_to_K(info_msg)
        baseline = self._baseline()

        kind = req.model_kind or self._p("default_model_kind")
        trt_variant = req.trt_variant or self._p("default_trt_variant")
        scale = float(req.scale) if req.scale > 0 else float(self._p("default_scale"))
        iters = int(req.iters) if req.iters > 0 else int(self._p("default_iters"))
        z_far = float(req.z_far) if req.z_far > 0 else float(self._p("default_z_far"))
        measure_fwd = bool(self._p("measure_forward_ms"))

        try:
            result = self._runner.infer(
                left_rgb=left, right_rgb=right, K=K_ir, baseline=baseline,
                kind=kind, scale=scale,
                valid_iters=(iters or None), z_far=z_far,
                trt_variant=trt_variant,
                live=False,
                measure_forward_ms=measure_fwd,
                want_debug_jpeg=bool(req.want_debug_jpeg),
            )
        except FileNotFoundError as exc:
            resp.status = 2
            resp.error_msg = str(exc)
            return resp
        except Exception as exc:  # noqa: BLE001
            resp.status = 3
            resp.error_msg = f"{type(exc).__name__}: {exc}"
            return resp

        depth = result.depth  # float32 m at the scaled grid

        # Optionally align into color frame.
        if req.align_to_color:
            if self._color_info is None or self._extrinsics is None:
                resp.status = 3
                resp.error_msg = "extrinsics not available"
                return resp
            K_color = _info_to_K(self._color_info)
            K_ir_scaled = K_ir.copy()
            K_ir_scaled[:2] *= result.scale_used  # cx, cy, fx, fy scale with resize; K[2,2] stays 1
            R, T = self._extrinsics
            depth = reproject_ir_to_color(
                depth, K_ir_scaled, K_color, R, T,
                out_hw=(self._color_info.height, self._color_info.width),
            )
            out_info = self._color_info
        else:
            out_info = info_msg

        # 32FC1 m for srv/action (the streaming worker handles 16UC1 conversion).
        depth_msg = self._bridge.cv2_to_imgmsg(depth.astype(np.float32),
                                               encoding="32FC1")
        depth_msg.header = out_info.header
        resp.depth_image = depth_msg
        resp.camera_info = out_info

        if req.want_debug_jpeg and result.vis_jpg:
            from sensor_msgs.msg import CompressedImage
            cmp = CompressedImage()
            cmp.header = depth_msg.header
            cmp.format = "jpeg"
            cmp.data = list(result.vis_jpg)
            resp.debug_jpeg = cmp

        resp.status = 0
        resp.error_msg = ""
        resp.forward_ms = float(result.forward_ms)
        resp.load_s = float(result.load_s)
        resp.end_to_end_s = float(time.time() - wall_t0)
        resp.model_used = self._runner.current_model or kind
        resp.trt_variant_used = self._runner.current_trt_variant or ""
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = FoundationStereoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build the node and smoke-test that it starts**

```bash
./src/tk26_vision/scripts/build_foundation_stereo.sh
source install/setup.bash
timeout 5 ros2 run foundation_stereo foundation_stereo_node
```

Expected: an `INFO` line like `foundation_stereo ready: profile=d435, ...` followed by the 5 s timeout. No tracebacks. No "cannot import" errors.

- [ ] **Step 3: Verify the service advertises**

In a second terminal (with the node still running, so this time without `timeout`):

```bash
ros2 run foundation_stereo foundation_stereo_node &
NODE_PID=$!
sleep 2
ros2 service list | grep foundation_stereo
ros2 service type /foundation_stereo/get_depth
kill $NODE_PID
```

Expected:

```
/foundation_stereo/get_depth
tinker_vision_msgs_26/srv/FoundationStereoDepth
```

- [ ] **Step 4: Verify the "no synced frame" path returns status=1**

With the node running and no cameras publishing:

```bash
ros2 run foundation_stereo foundation_stereo_node &
NODE_PID=$!
sleep 2
ros2 service call /foundation_stereo/get_depth \
    tinker_vision_msgs_26/srv/FoundationStereoDepth '{}'
kill $NODE_PID
```

Expected response includes `status: 1` and `error_msg: 'no synced stereo frame'`.

- [ ] **Step 5: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/foundation_stereo/foundation_stereo_node.py
git commit -m "feat(foundation_stereo): ROS2 node — service path

Sync triple subscriber (left + right + left CameraInfo) caches the
latest frame; /foundation_stereo/get_depth runs one inference over it.
align_to_color path uses the color CameraInfo + the latched
realsense2_camera_msgs/Extrinsics topic. Action and streaming worker
follow in Tasks 9 and 10."
```

---

## Task 9: Add the action handler

**Files:**
- Modify: `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py`

- [ ] **Step 1: Add the action server**

Edit `foundation_stereo_node.py`. Add these imports at the top:

```python
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from tinker_vision_msgs_26.action import FoundationStereoDepth as FSAction
```

Inside `__init__`, after the `_setup_service()` call, add:

```python
        self._setup_action()
```

Add the new method on the class:

```python
    def _setup_action(self) -> None:
        self._action = ActionServer(
            self,
            FSAction,
            "~/infer_depth",
            execute_callback=self._on_infer_depth,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
        )

    def _on_infer_depth(self, goal_handle):
        req = goal_handle.request
        resp = FSAction.Result()
        feedback = FSAction.Feedback()

        wall_t0 = time.time()

        def fb(stage: str) -> None:
            feedback.current_stage = stage
            feedback.elapsed_s = float(time.time() - wall_t0)
            goal_handle.publish_feedback(feedback)

        with self._latest_lock:
            cached = self._latest
        if cached is None:
            resp.status = 1
            resp.error_msg = "no synced stereo frame"
            goal_handle.succeed()
            return resp

        left_msg, right_msg, info_msg = cached
        try:
            left = self._bridge.imgmsg_to_cv2(left_msg, desired_encoding="rgb8")
            right = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding="rgb8")
        except Exception as exc:
            resp.status = 3
            resp.error_msg = f"cv_bridge: {exc}"
            goal_handle.succeed()
            return resp

        K_ir = _info_to_K(info_msg)
        baseline = self._baseline()

        kind = req.model_kind or self._p("default_model_kind")
        trt_variant = req.trt_variant or self._p("default_trt_variant")
        scale = float(req.scale) if req.scale > 0 else float(self._p("default_scale"))
        iters = int(req.iters) if req.iters > 0 else int(self._p("default_iters"))
        z_far = float(req.z_far) if req.z_far > 0 else float(self._p("default_z_far"))
        measure_fwd = bool(self._p("measure_forward_ms"))

        fb("loading_model")
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            resp.status = 3
            resp.error_msg = "cancelled before inference"
            return resp

        fb("running_forward")
        try:
            result = self._runner.infer(
                left_rgb=left, right_rgb=right, K=K_ir, baseline=baseline,
                kind=kind, scale=scale,
                valid_iters=(iters or None), z_far=z_far,
                trt_variant=trt_variant,
                live=False,
                measure_forward_ms=measure_fwd,
                want_debug_jpeg=bool(req.want_debug_jpeg),
            )
        except FileNotFoundError as exc:
            resp.status = 2
            resp.error_msg = str(exc)
            goal_handle.succeed()
            return resp
        except Exception as exc:  # noqa: BLE001
            resp.status = 3
            resp.error_msg = f"{type(exc).__name__}: {exc}"
            goal_handle.succeed()
            return resp

        depth = result.depth
        if req.align_to_color:
            fb("aligning_to_color")
            if self._color_info is None or self._extrinsics is None:
                resp.status = 3
                resp.error_msg = "extrinsics not available"
                goal_handle.succeed()
                return resp
            K_color = _info_to_K(self._color_info)
            K_ir_scaled = K_ir.copy()
            K_ir_scaled[:2] *= result.scale_used  # cx, cy, fx, fy scale with resize; K[2,2] stays 1
            R, T = self._extrinsics
            depth = reproject_ir_to_color(
                depth, K_ir_scaled, K_color, R, T,
                out_hw=(self._color_info.height, self._color_info.width),
            )
            out_info = self._color_info
        else:
            out_info = info_msg

        fb("encoding_debug" if req.want_debug_jpeg else "running_forward")
        depth_msg = self._bridge.cv2_to_imgmsg(depth.astype(np.float32),
                                               encoding="32FC1")
        depth_msg.header = out_info.header
        resp.depth_image = depth_msg
        resp.camera_info = out_info

        if req.want_debug_jpeg and result.vis_jpg:
            from sensor_msgs.msg import CompressedImage
            cmp = CompressedImage()
            cmp.header = depth_msg.header
            cmp.format = "jpeg"
            cmp.data = list(result.vis_jpg)
            resp.debug_jpeg = cmp

        resp.status = 0
        resp.error_msg = ""
        resp.forward_ms = float(result.forward_ms)
        resp.load_s = float(result.load_s)
        resp.end_to_end_s = float(time.time() - wall_t0)
        resp.model_used = self._runner.current_model or kind
        resp.trt_variant_used = self._runner.current_trt_variant or ""
        goal_handle.succeed()
        return resp
```

- [ ] **Step 2: Build and verify both interfaces advertise**

```bash
./src/tk26_vision/scripts/build_foundation_stereo.sh
source install/setup.bash
ros2 run foundation_stereo foundation_stereo_node &
NODE_PID=$!
sleep 2
ros2 service list | grep foundation_stereo
ros2 action list | grep foundation_stereo
kill $NODE_PID
```

Expected:

```
/foundation_stereo/get_depth
/foundation_stereo/infer_depth
```

- [ ] **Step 3: Test "no synced frame" path through the action**

```bash
ros2 run foundation_stereo foundation_stereo_node &
NODE_PID=$!
sleep 2
ros2 action send_goal /foundation_stereo/infer_depth \
    tinker_vision_msgs_26/action/FoundationStereoDepth '{}'
kill $NODE_PID
```

Expected: result section contains `status: 1` and `error_msg: 'no synced stereo frame'`.

- [ ] **Step 4: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/foundation_stereo/foundation_stereo_node.py
git commit -m "feat(foundation_stereo): add action handler with per-stage feedback

ActionServer at /foundation_stereo/infer_depth shares the StereoRunner
with the service (one model load). Feedback strings step through
loading_model / running_forward / aligning_to_color / encoding_debug
so slow PyTorch backends report progress."
```

---

## Task 10: Add the streaming worker

**Files:**
- Modify: `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py`

- [ ] **Step 1: Add the worker class + plumbing**

Edit `foundation_stereo_node.py`. Add this import at the top:

```python
from sensor_msgs.msg import CompressedImage
```

Add the new helper functions just below `_info_to_K`:

```python
def _depth_to_msg(depth_m: np.ndarray, dtype: str, bridge: CvBridge,
                  header) -> Image:
    if dtype == "16UC1_mm":
        mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        msg = bridge.cv2_to_imgmsg(mm, encoding="16UC1")
    else:  # 32FC1_m
        msg = bridge.cv2_to_imgmsg(depth_m.astype(np.float32), encoding="32FC1")
    msg.header = header
    return msg


def _resolve_stream_topics(node, align: bool) -> Tuple[str, str]:
    depth_topic = node._p("stream_depth_topic")
    info_topic = node._p("stream_info_topic")
    if depth_topic and info_topic:
        return depth_topic, info_topic
    if align:
        return ("~/aligned_depth_to_color/image_rect_raw",
                "~/aligned_depth_to_color/camera_info")
    return "~/depth/image_rect_raw", "~/depth/camera_info"
```

Add a method on `FoundationStereoNode`:

```python
    def _setup_stream(self) -> None:
        if not self._p("stream_enabled"):
            return
        align = bool(self._p("stream_align_to_color"))

        # IMPORTANT: do NOT wait for extrinsics here — __init__ runs before
        # rclpy.spin(), so subscription callbacks can't fire and the loop
        # would always time out. The worker thread (below) does the warmup
        # wait once the executor is alive.

        depth_topic, info_topic = _resolve_stream_topics(self, align)
        self._stream_depth_pub = self.create_publisher(
            Image, depth_topic, qos_profile_sensor_data
        )
        self._stream_info_pub = self.create_publisher(
            CameraInfo, info_topic, qos_profile_sensor_data
        )
        self._stream_vis_pub = (
            self.create_publisher(CompressedImage, "~/debug/disparity/compressed",
                                  qos_profile_sensor_data)
            if self._p("stream_publish_vis") else None
        )

        self._stream_stop = threading.Event()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="fs-stream", daemon=True,
        )
        self._stream_thread.start()
        self.get_logger().info(
            f"streaming publisher started: depth={depth_topic}, "
            f"info={info_topic}, dtype={self._p('stream_dtype')}, align={align}"
        )

    def _stream_loop(self) -> None:
        align = bool(self._p("stream_align_to_color"))
        dtype = str(self._p("stream_dtype"))
        out_frame = str(self._p("output_frame_id"))
        max_fps = float(self._p("stream_max_fps"))
        min_period = (1.0 / max_fps) if max_fps > 0 else 0.0
        measure_fwd = bool(self._p("stream_measure_forward_ms"))

        # Extrinsics warm-up runs here so rclpy.spin()'s executor is
        # already running in the main thread and the latched extrinsics +
        # color_info callbacks can fire.
        if align:
            warmup = float(self._p("extrinsics_warmup_timeout_sec"))
            deadline = time.time() + warmup
            while time.time() < deadline and not self._stream_stop.is_set():
                if self._extrinsics is not None and self._color_info is not None:
                    break
                time.sleep(0.1)
            if self._extrinsics is None or self._color_info is None:
                self.get_logger().error(
                    "stream_align_to_color=true but extrinsics or "
                    f"color_info not received within {warmup} s; "
                    "publisher not emitting."
                )
                return

        last_seq = None
        last_emit = 0.0

        while not self._stream_stop.is_set():
            with self._latest_lock:
                cached = self._latest
            if cached is None:
                time.sleep(0.01)
                continue

            left_msg, right_msg, info_msg = cached
            seq = (left_msg.header.stamp.sec, left_msg.header.stamp.nanosec)
            if seq == last_seq:
                time.sleep(0.001)
                continue
            if min_period > 0 and (time.time() - last_emit) < min_period:
                time.sleep(0.001)
                continue
            last_seq = seq

            try:
                left = self._bridge.imgmsg_to_cv2(left_msg, desired_encoding="rgb8")
                right = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding="rgb8")
            except Exception as exc:
                self.get_logger().warn(f"cv_bridge: {exc}", throttle_duration_sec=5.0)
                continue

            K_ir = _info_to_K(info_msg)
            try:
                result = self._runner.infer(
                    left_rgb=left, right_rgb=right, K=K_ir,
                    baseline=self._baseline(),
                    kind=self._p("default_model_kind"),
                    scale=float(self._p("default_scale")),
                    valid_iters=(int(self._p("default_iters")) or None),
                    z_far=float(self._p("default_z_far")),
                    trt_variant=self._p("default_trt_variant"),
                    live=False,
                    measure_forward_ms=measure_fwd,
                    want_debug_jpeg=bool(self._stream_vis_pub),
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().exception("stream inference failed")
                time.sleep(0.05)
                continue

            depth = result.depth
            if align:
                K_color = _info_to_K(self._color_info)
                R, T = self._extrinsics
                depth = reproject_ir_to_color(
                    depth, K_ir * result.scale_used, K_color, R, T,
                    out_hw=(self._color_info.height, self._color_info.width),
                )
                out_info = self._color_info
            else:
                out_info = info_msg

            header = out_info.header
            if out_frame:
                header.frame_id = out_frame

            depth_msg = _depth_to_msg(depth, dtype, self._bridge, header)
            info_out = CameraInfo()
            info_out.header = header
            info_out.height = out_info.height
            info_out.width = out_info.width
            info_out.distortion_model = out_info.distortion_model
            info_out.d = out_info.d
            info_out.k = out_info.k
            info_out.r = out_info.r
            info_out.p = out_info.p

            self._stream_depth_pub.publish(depth_msg)
            self._stream_info_pub.publish(info_out)

            if self._stream_vis_pub is not None and result.vis_jpg:
                cmp = CompressedImage()
                cmp.header = header
                cmp.format = "jpeg"
                cmp.data = list(result.vis_jpg)
                self._stream_vis_pub.publish(cmp)

            last_emit = time.time()

    def destroy_node(self):
        if getattr(self, "_stream_stop", None) is not None:
            self._stream_stop.set()
        if getattr(self, "_stream_thread", None) is not None:
            self._stream_thread.join(timeout=2.0)
        super().destroy_node()
```

Then call `self._setup_stream()` at the end of `__init__`:

```python
        self._setup_action()
        self._setup_stream()
```

- [ ] **Step 2: Build and verify stream is dormant by default**

```bash
./src/tk26_vision/scripts/build_foundation_stereo.sh
source install/setup.bash
ros2 run foundation_stereo foundation_stereo_node &
NODE_PID=$!
sleep 2
ros2 topic list | grep -E "depth|disparity" || echo "(no streaming topics — correct)"
kill $NODE_PID
```

Expected: no `~/depth/...` or `~/aligned_depth_to_color/...` topics appear when `stream_enabled=false` (the default).

- [ ] **Step 3: Verify stream starts when enabled (against a recorded bag or live cameras)**

Skip if no bag/cameras available — the smoke test runs as part of Task 12 against a known scene.

```bash
# Optional: with a stereo bag playing in another terminal.
ros2 run foundation_stereo foundation_stereo_node \
    --ros-args -p stream_enabled:=true -p stream_align_to_color:=false &
NODE_PID=$!
sleep 7  # allow for extrinsics_warmup (5s) + spin-up
ros2 topic list | grep -E "depth/image_rect_raw"
kill $NODE_PID
```

Expected: `/foundation_stereo/depth/image_rect_raw` and `/foundation_stereo/depth/camera_info` listed.

- [ ] **Step 4: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/foundation_stereo/foundation_stereo_node.py
git commit -m "feat(foundation_stereo): streaming depth publisher

Daemon-thread worker. Off by default; opt-in via stream_enabled. When
stream_align_to_color=true, waits up to extrinsics_warmup_timeout_sec
for the latched extrinsics + color_info before starting — refuses to
publish identity-aligned depth on D435 (15 mm-scale wrong). Honors
output_frame_id, stream_dtype (16UC1 mm default), stream_max_fps, and
optional debug_jpeg vis."
```

---

## Task 11: Launch file + config yaml

**Files:**
- Create: `src/foundation_stereo/launch/foundation_stereo.launch.py`
- Create: `src/foundation_stereo/config/foundation_stereo.yaml`

- [ ] **Step 1: Write the config yaml**

Path: `src/tk26_vision/src/foundation_stereo/config/foundation_stereo.yaml`

```yaml
foundation_stereo:
  ros__parameters:
    # Paths & model defaults
    weights_root: "/home/tinker/projects/vision_tests/dualrRGB-foundationStereo"
    camera_profile: "d435"
    default_model_kind: "fast_trt"
    default_trt_variant: "output_two_stage"
    default_scale: 0.5
    default_iters: 0
    default_z_far: 10.0

    # Topic overrides — leave blank to use the profile's defaults.
    left_topic: ""
    right_topic: ""
    left_info_topic: ""
    color_info_topic: ""
    extrinsics_topic: ""
    baseline_m: 0.0

    # Sync + measurement
    sync_slop_sec: 0.05
    sync_queue_size: 5
    measure_forward_ms: true

    # Streaming
    stream_enabled: false
    stream_align_to_color: true
    stream_depth_topic: ""
    stream_info_topic: ""
    stream_dtype: "16UC1_mm"
    output_frame_id: ""
    stream_publish_vis: false
    stream_max_fps: 0.0
    extrinsics_warmup_timeout_sec: 5.0
    stream_measure_forward_ms: false

    # Logging
    vision_logging_enabled: false
    vision_log_folder: "vision_log"
```

- [ ] **Step 2: Write the launch file**

Path: `src/tk26_vision/src/foundation_stereo/launch/foundation_stereo.launch.py`

```python
"""Launch the foundation_stereo node with the canonical config yaml.

Override individual params via `ros2 launch foundation_stereo
foundation_stereo.launch.py stream_enabled:=true camera_profile:=d405`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = []
    for name, default in [
        ("camera_profile", "d435"),
        ("stream_enabled", "false"),
        ("stream_align_to_color", "true"),
        ("default_model_kind", "fast_trt"),
        ("default_trt_variant", "output_two_stage"),
    ]:
        args.append(DeclareLaunchArgument(name, default_value=default))

    pkg_share = FindPackageShare("foundation_stereo")
    config_path = [pkg_share, "/config/foundation_stereo.yaml"]

    node = Node(
        package="foundation_stereo",
        executable="foundation_stereo_node",
        name="foundation_stereo",
        output="screen",
        parameters=[
            config_path,
            {
                "camera_profile": LaunchConfiguration("camera_profile"),
                "stream_enabled": LaunchConfiguration("stream_enabled"),
                "stream_align_to_color": LaunchConfiguration("stream_align_to_color"),
                "default_model_kind": LaunchConfiguration("default_model_kind"),
                "default_trt_variant": LaunchConfiguration("default_trt_variant"),
            },
        ],
    )

    return LaunchDescription(args + [node])
```

- [ ] **Step 3: Rebuild and test the launch file**

```bash
./src/tk26_vision/scripts/build_foundation_stereo.sh
source install/setup.bash
timeout 5 ros2 launch foundation_stereo foundation_stereo.launch.py
```

Expected: the node boots with the yaml-loaded defaults, prints its `ready:` line, exits cleanly on timeout.

Then with a flag override:

```bash
timeout 5 ros2 launch foundation_stereo foundation_stereo.launch.py \
    camera_profile:=d405
```

Expected: `ready: profile=d405, ...` in the output.

- [ ] **Step 4: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/launch/foundation_stereo.launch.py \
        src/tk26_vision/src/foundation_stereo/config/foundation_stereo.yaml
git commit -m "feat(foundation_stereo): launch file + canonical config yaml

Reads config/foundation_stereo.yaml and exposes the most-tweaked params
(camera_profile, stream_enabled, stream_align_to_color, model_kind,
trt_variant) as launch arguments. yaml-first design matches the
follow_head / monocular_depth pattern."
```

---

## Task 12: Wire into t0_static.sh + t1_startup.sh smoke suite

**Files:**
- Modify: `scripts/tests/t0_static.sh`
- Modify: `scripts/tests/t1_startup.sh`

The existing scripts use `section / pass / fail` helpers from `lib.sh`, plus `t1_check` and `t1_check_multi` in T1. Note that the T0.1 shebang find list is hard-coded and intentionally excludes packages with separate venvs (e.g. `monocular_depth` under `.venv-da3`). We do the same for `foundation_stereo` — its `.venv-fs` shebang is verified by a dedicated T0 row instead.

- [ ] **Step 1: Add the foundation_stereo iface lines to T0.4 in t0_static.sh**

Edit `src/tk26_vision/scripts/tests/t0_static.sh`. Locate the `ifaces=(` array inside `section "T0.4 — ROS interfaces built"` and append two entries:

```bash
    tinker_vision_msgs_26/srv/FoundationStereoDepth
    tinker_vision_msgs_26/action/FoundationStereoDepth
```

- [ ] **Step 2: Add new T0 sections for the foundation_stereo-specific checks**

Append (or insert before the final `T0.<N>` section) to `t0_static.sh`:

```bash
section "T0.fs — foundation_stereo vendored trees present"
if [ -d "$WS_ROOT/src/tk26_vision/thirdparty/foundation_stereo/FoundationStereo/core" ] \
   && [ -d "$WS_ROOT/src/tk26_vision/thirdparty/foundation_stereo/Fast-FoundationStereo/core" ]; then
    pass "T0.fs.vendor"
else
    fail "T0.fs.vendor" "vendored trees missing under src/tk26_vision/thirdparty/foundation_stereo/"
fi

section "T0.fs — foundation_stereo .venv-fs shebang"
FS_ENTRY="$WS_ROOT/install/foundation_stereo/lib/foundation_stereo/foundation_stereo_node"
EXPECTED_FS_SHEBANG="#!$WS_ROOT/src/tk26_vision/.venv-fs/bin/python3"
if [ ! -f "$FS_ENTRY" ]; then
    fail "T0.fs.shebang" "entry script not found: $FS_ENTRY (build first via scripts/build_foundation_stereo.sh)"
else
    first=$(head -1 "$FS_ENTRY" 2>/dev/null || true)
    if [ "$first" = "$EXPECTED_FS_SHEBANG" ]; then
        pass "T0.fs.shebang"
    else
        fail "T0.fs.shebang" "got: $first (want: $EXPECTED_FS_SHEBANG)"
    fi
fi

section "T0.fs — color_align + logging pytest"
if "$WS_ROOT/src/tk26_vision/.venv-fs/bin/python" -m pytest \
        "$WS_ROOT/src/tk26_vision/src/foundation_stereo/test/test_color_align.py" \
        "$WS_ROOT/src/tk26_vision/src/foundation_stereo/test/test_logging.py" \
        -q 2>"$LOG_DIR/t0.fs.pytest.err" >/dev/null; then
    pass "T0.fs.pytest"
else
    fail "T0.fs.pytest" "$(cat "$LOG_DIR/t0.fs.pytest.err")"
fi
```

- [ ] **Step 3: Add a T1 section for foundation_stereo**

Append to `src/tk26_vision/scripts/tests/t1_startup.sh`, after the existing `T1.x` sections, mirroring the `t1_check_multi` pattern:

```bash
section "T1.fs — foundation_stereo advertises srv + action"
t1_check_multi T1.fs foundation_stereo foundation_stereo_node \
    s:/foundation_stereo/get_depth \
    a:/foundation_stereo/infer_depth \
    --
```

The trailing `--` is the sentinel that splits expected interfaces from `ros2 run` extra args (none here).

- [ ] **Step 4: Run both tiers and verify all foundation_stereo rows pass**

```bash
./src/tk26_vision/scripts/tests/t0_static.sh 2>&1 | grep -E "T0.fs|FoundationStereoDepth"
./src/tk26_vision/scripts/tests/t1_startup.sh 2>&1 | grep "T1.fs"
```

Expected: every `T0.fs*` and `T1.fs*` row marked PASS. (T0.4 will show the two new interfaces in its `(N interfaces)` count.)

- [ ] **Step 5: Commit**

```bash
git add src/tk26_vision/scripts/tests/t0_static.sh \
        src/tk26_vision/scripts/tests/t1_startup.sh
git commit -m "test(foundation_stereo): wire T0 + T1 smoke checks

T0: vendored trees, srv+action build, entry point registered,
shebang points at .venv-fs, pytest color_align + logging green.
T1: node starts, advertises both interfaces, clean SIGTERM."
```

---

## Task 13: Package README + CLAUDE.md updates

**Files:**
- Create: `src/foundation_stereo/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the package README**

Path: `src/tk26_vision/src/foundation_stereo/README.md`

```markdown
# foundation_stereo

ROS2 node serving FoundationStereo + Fast-FoundationStereo over two modes:

- **On-request**: `/foundation_stereo/get_depth` service and
  `/foundation_stereo/infer_depth` action (per-call overrides for model,
  scale, iters, TRT variant; the action surfaces per-stage feedback +
  cancellation).
- **Streaming**: optional depth publisher that mimics the realsense driver's
  `aligned_depth_to_color/image_rect_raw` topic shape (`16UC1 mm` by
  default, `SensorDataQoS`).

## Why a separate venv

`torch==2.8.0+cu128` + `tensorrt==10.16.1.11` conflict with the versions in
`.venv-vision-main`. This package builds + runs under
`src/tk26_vision/.venv-fs/`, provisioned manually (see below).

## Provisioning `.venv-fs` (one-time)

```bash
cd src/tk26_vision
python3.10 -m venv .venv-fs --system-site-packages --symlinks
source .venv-fs/bin/activate
pip install --upgrade pip wheel
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r src/foundation_stereo/requirements.txt
pip install tensorrt==10.16.1.11
pip freeze > .venv-fs/freeze.lock.txt
```

## Build + run

```bash
./src/tk26_vision/scripts/build_foundation_stereo.sh
source install/setup.bash

# Default: srv + action only, no streaming.
ros2 launch foundation_stereo foundation_stereo.launch.py

# Streaming, aligned to color (recommended for D435).
ros2 launch foundation_stereo foundation_stereo.launch.py \
    stream_enabled:=true stream_align_to_color:=true
```

## D435 frame-name caveat

`realsense2_camera` publishes the left IR optical frame as
`xarm_camera_infra1_optical_frame`. The xarm URDF declares it as
`xarm_camera_left_ir_optical_frame`. No static_transform_publisher
bridges them. Two clean options when consuming this node's raw-mode
depth:

1. Stream aligned-to-color (the default). Output frame becomes
   `xarm_camera_color_optical_frame`, which agrees between driver and URDF.
2. Set `output_frame_id:=xarm_camera_left_ir_optical_frame` so the published
   depth carries the URDF name. Geometrically equivalent — same sensor.
3. Or run a bridging static TF once at bringup:
   ```bash
   ros2 run tf2_ros static_transform_publisher \
       0 0 0  0 0 0 \
       xarm_camera_left_ir_optical_frame \
       xarm_camera_infra1_optical_frame
   ```

## Spec + plan

- Design: [`docs/superpowers/specs/2026-05-24-foundation-stereo-design.md`](../../docs/superpowers/specs/2026-05-24-foundation-stereo-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-05-24-foundation-stereo.md`](../../docs/superpowers/plans/2026-05-24-foundation-stereo.md)
```

- [ ] **Step 2: Update tk26_vision CLAUDE.md**

Edit `src/tk26_vision/CLAUDE.md`. Locate the existing architecture overview block (`## Architecture` section) and add a row for `foundation_stereo`:

Find the `src/tk26_vision/src/` listing in that section. After the `monocular_depth/` entry, add:

```
└── foundation_stereo/             # FoundationStereo + Fast-FoundationStereo service/action + streaming depth publisher; lives in its own venv `.venv-fs` (torch 2.8 + cu128 + tensorrt 10.16) because those versions conflict with the shared `.venv-vision-main`
```

Locate the `## Build` section (the one mentioning `build_monocular_depth.sh`) and add a paragraph after it:

```markdown
**`foundation_stereo` builds under a third venv.** `torch==2.8.0+cu128` +
`tensorrt==10.16.1.11` conflict with both the shared `.venv-vision-main`
and `.venv-da3`. Use the dedicated wrapper:

```bash
./src/tk26_vision/scripts/build_foundation_stereo.sh [colcon args...]
```

The wrapper sources `.venv-fs/`, runs `colcon build --packages-select
foundation_stereo` (or the args you pass), then re-shebangs the entry-point
script. Provisioning the venv once: see `src/foundation_stereo/README.md`.
```

In the `## Running Nodes` section, add under `# Object detection` or in a new sub-block:

```bash
# FoundationStereo (separate venv .venv-fs)
ros2 launch foundation_stereo foundation_stereo.launch.py
ros2 launch foundation_stereo foundation_stereo.launch.py \
    stream_enabled:=true stream_align_to_color:=true
```

- [ ] **Step 3: Commit**

```bash
git add src/tk26_vision/src/foundation_stereo/README.md \
        src/tk26_vision/CLAUDE.md
git commit -m "docs(foundation_stereo): package README + CLAUDE.md entries

README documents the venv provisioning recipe, the launch invocations,
and the D435 frame-name caveat with the three resolution options.
CLAUDE.md architecture overview + build-wrapper note added for
foundation_stereo alongside the existing monocular_depth entries."
```

---

## Task 14: Live-camera T2 verification (manual, operator-in-the-loop)

**Files:** none — verification only.

- [ ] **Step 1: Bring up the cameras per CAMERA_BRINGUP.md**

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/tinker/tk25_ws/src/tk26_vision/config/fastdds_shm.xml
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=xarm_camera \
    align_depth.enable:=true \
    config_file:=/home/tinker/tk25_ws/src/tk26_vision/config/realsense_qos.yaml
```

In another terminal, verify the IR rate:

```bash
ros2 topic hz /camera/xarm_camera/infra1/image_rect_raw --window 50
```

Expected: ~30 Hz ± 5 ms.

- [ ] **Step 2: Service smoke test — IR-frame depth**

```bash
ros2 launch foundation_stereo foundation_stereo.launch.py &
sleep 2
ros2 service call /foundation_stereo/get_depth \
    tinker_vision_msgs_26/srv/FoundationStereoDepth \
    '{align_to_color: false}'
```

Expected response: `status: 0`, non-empty `depth_image` (height/width matching `infra1` resolution), `forward_ms` between 9–25 ms (fast_trt), `end_to_end_s` ≤ 200 ms.

- [ ] **Step 3: Service smoke test — color-aligned depth**

```bash
ros2 service call /foundation_stereo/get_depth \
    tinker_vision_msgs_26/srv/FoundationStereoDepth \
    '{align_to_color: true}'
```

Expected response: `status: 0`, `depth_image` height/width matching `color/camera_info`, header.frame_id = `xarm_camera_color_optical_frame`.

- [ ] **Step 4: Streaming smoke test**

Kill the previous launch and restart with streaming on:

```bash
ros2 launch foundation_stereo foundation_stereo.launch.py \
    stream_enabled:=true stream_align_to_color:=true &
sleep 7
ros2 topic hz /foundation_stereo/aligned_depth_to_color/image_rect_raw --window 50
```

Expected: ≥ 10 Hz steady.

- [ ] **Step 5: Side-by-side comparison vs the driver's native aligned depth**

```bash
ros2 topic echo --once /camera/xarm_camera/aligned_depth_to_color/image_raw | head -20
ros2 topic echo --once /foundation_stereo/aligned_depth_to_color/image_rect_raw | head -20
```

Expected: both publish on the same grid (same height/width). Per-pixel
agreement in the centre of the frame should be within a few cm — they're
two stereo algorithms, not bit-identical.

If all 5 steps pass: the integration is operationally verified. Record
findings in `src/tk26_vision/DEV_NOTES.md` under a new "FoundationStereo T2"
entry.

---

## Self-review checklist (for the author)

Run this after implementing the plan, before declaring done:

1. **Spec coverage.** Every §1–§11 in the spec has a corresponding task:
   - §1 Goals → Tasks 1, 2, 3 (skeleton); 8/9/10 (modes); 11 (config)
   - §2 Architecture → Tasks 1 (vendor), 3 (package), 8/9/10 (node)
   - §3 Interfaces → Task 2 (srv+action), 8/9/10 (handlers + streaming)
   - §4 Configuration → Tasks 8/11 (params + yaml)
   - §5 Color alignment → Task 4 (TDD), 8/9/10 (wiring)
   - §6 Frame IDs / URDF → Task 13 (README + CLAUDE.md doc)
   - §7 Error handling → Tasks 8/9/10 (status codes), Task 14 (verification)
   - §8 Vendor / venv → Tasks 1 (vendor), 6/7 (venv + build wrapper)
   - §9 Testing → Tasks 4/5 (unit), 6 (import-shape), 12 (T0/T1), 14 (T2)
   - §10 Measurement overhead → Task 6 (lean port)
   - §11 Out-of-scope → not implemented, by design.

2. **Placeholder scan.** No "TBD", "implement later", or step-without-code.
3. **Type consistency.** `StereoRunner.infer(..., measure_forward_ms=...,
   want_debug_jpeg=...)` signature matches every call site in Tasks 8/9/10.
   `InferResult` fields `disp / depth / vis_jpg / scale_used / load_s /
   forward_ms / forward_s / post_s` match the runner's return path and the
   node's reads. `reproject_ir_to_color(depth_ir, K_ir, K_color, R, T,
   out_hw)` signature matches Task 4 (test + impl) and Tasks 8/9/10
   (callers). srv/action field names are identical between schemas.

---

## Execution

Plan complete and saved to `docs/superpowers/plans/2026-05-24-foundation-stereo.md`.

When ready to execute, pick **subagent-driven** (recommended — fresh
subagent per task with two-stage review) or **inline** (executing-plans
skill with checkpoints).
