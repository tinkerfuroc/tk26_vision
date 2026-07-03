# Orbbec HRI Fixed Higher Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch the Orbbec Femto Bolt at a fixed 1920×1080 color resolution for the whole HRI task (enrollment + follow), instead of the vendored 1280×720 default, without breaking any resolution-hardcoded consumer.

**Architecture:** Two independent resolution-hardcode bugs (`object_detection_new/object_seg_yolo.py`, `vision_util/door_detection.py`) get fixed by switching their Orbbec depth path from the ordered colored `PointCloud2` to the registered depth `Image` topic + live `CameraInfo`, reprojected via a new shared, resolution-agnostic helper in `vision_util`. Separately, the launch stack gets a `color_width`/`color_height` passthrough (`vision_driver.launch.py`) that only the HRI launch script (`tmux_hri_vision.sh`) overrides to 1920×1080, plus a larger FastDDS SHM segment to carry the bigger frames. No runtime switching, no BT changes, no vendored-SDK changes.

**Tech Stack:** ROS2 Humble, Python 3.10 (`tk26_vision/.venv-vision-main`), numpy, `cv_bridge`, `pytest`/`unittest`, ROS2 launch (Python), bash, XML (FastDDS profile).

**Spec:** `src/tk26_vision/docs/superpowers/specs/2026-07-03-orbbec-hri-resolution-bump-design.md`

## Global Constraints

- Target color resolution: **1920×1080 MJPG**, confirmed available at full 30fps on the live Femto Bolt (`list_camera_profile_mode_node`). Depth stays **640×576@30fps** — do not touch depth-stream launch args.
- No dynamic/runtime resolution switching — launch-time only. Do not add a BT node, orchestrator service, or mid-task relaunch logic.
- Do **not** touch `pan_tilt/follow_head.py`'s `CameraInfo` latch (`follow_head.py:427-432`) — explicitly out of scope per the design spec.
- Do **not** touch `enable_colored_point_cloud` (stays `true`) — the vendored driver's zero-subscriber early-return already avoids the cost once the two bugs below are fixed.
- Do **not** touch `src/tk25_basic/src/scripts/master_hri2.sh` — confirmed not the live script; only `master_hri.sh` → `tmux_hri_vision.sh` → `vision_driver.launch.py` matters.
- Resolution-safe reprojection must read H/W from the live depth image's own shape every call — never hardcode a resolution.
- This work spans two separate git repos: `src/tk26_vision` (Tasks 1–4, 6) and `src/tk25_basic` (Task 5). Commit within the correct repo root for each task.

---

### Task 1: Shared resolution-agnostic depth reprojection helper

**Files:**
- Create: `src/tk26_vision/src/vision_util/vision_util/depth_reproject.py`
- Create: `src/tk26_vision/src/vision_util/test/test_depth_reproject.py`

**Interfaces:**
- Produces: `vision_util.depth_reproject.decode_depth_metres(depth_arr: np.ndarray) -> np.ndarray` — coerces a decoded depth image (uint16 mm or float32 m) to float32 metres; raises `ValueError` on any other dtype.
- Produces: `vision_util.depth_reproject.depth_image_to_points(depth_m: np.ndarray, k) -> np.ndarray` — pinhole back-projection to an `(H, W, 3)` float32 array (`x`, `y`, `z=depth_m`), where `H, W` always equal `depth_m.shape`. `k` is a 9-element row-major `CameraInfo.k`-style intrinsic matrix (or any indexable with `k[0]=fx, k[2]=cx, k[4]=fy, k[5]=cy`).

This is the DRY fix point: both Task 2 and Task 3 replace an independent hardcoded-720×1280 reprojection with a call into this module.

- [ ] **Step 1: Write the failing tests**

Create `src/tk26_vision/src/vision_util/test/test_depth_reproject.py`:

```python
import unittest

import numpy as np

from vision_util.depth_reproject import decode_depth_metres, depth_image_to_points


class TestDecodeDepthMetres(unittest.TestCase):
    def test_uint16_millimetres_converts_to_metres(self):
        arr = np.array([[2000, 500], [0, 10000]], dtype=np.uint16)
        result = decode_depth_metres(arr)
        np.testing.assert_allclose(result, [[2.0, 0.5], [0.0, 10.0]])
        self.assertEqual(result.dtype, np.float32)

    def test_float32_metres_passes_through_unchanged(self):
        arr = np.array([[1.5, 2.5]], dtype=np.float32)
        result = decode_depth_metres(arr)
        np.testing.assert_allclose(result, arr)

    def test_unsupported_dtype_raises(self):
        arr = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            decode_depth_metres(arr)


class TestDepthImageToPoints(unittest.TestCase):
    def test_center_pixel_back_projects_to_zero_xy(self):
        fx = fy = 500.0
        cx, cy = 320.0, 240.0
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        points = depth_image_to_points(depth, k)

        self.assertAlmostEqual(float(points[240, 320, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(points[240, 320, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(points[240, 320, 2]), 2.0, places=4)

    def test_off_center_pixel_matches_pinhole_formula(self):
        fx = fy = 500.0
        cx, cy = 320.0, 240.0
        depth = np.full((480, 640), 4.0, dtype=np.float32)
        k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        points = depth_image_to_points(depth, k)

        # Pixel (row=240, col=420): x = (420-320)*4/500 = 0.8
        self.assertAlmostEqual(float(points[240, 420, 0]), 0.8, places=4)
        # Pixel (row=340, col=320): y = (340-240)*4/500 = 0.8
        self.assertAlmostEqual(float(points[340, 320, 1]), 0.8, places=4)

    def test_output_shape_matches_input_regardless_of_resolution(self):
        # Regression target: object_seg_yolo.py._pointcloud_to_array used to
        # hardcode (720, 1280) and silently clip/crash at any other size.
        fx = fy = 900.0
        cx, cy = 960.0, 540.0
        depth = np.full((1080, 1920), 3.0, dtype=np.float32)
        k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        points = depth_image_to_points(depth, k)

        self.assertEqual(points.shape, (1080, 1920, 3))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
source .venv-vision-main/bin/activate
python3 -m pytest src/vision_util/test/test_depth_reproject.py -v
```

Expected: `ModuleNotFoundError: No module named 'vision_util.depth_reproject'`.

- [ ] **Step 3: Write the implementation**

Create `src/tk26_vision/src/vision_util/vision_util/depth_reproject.py`:

```python
"""Depth-image -> 3D-points reprojection for the Orbbec, resolution-agnostic.

Standard pinhole back-projection in the camera's own optical frame
(x=right, y=down, z=forward -- the convention the intrinsics describe, and
the same one `vision_util/_pc_utils.py:build_xy_table_cuda` uses on GPU).
Dimensions are always read from the depth image's own shape, never assumed,
so this works at whatever resolution the camera driver is launched with.

Shared by object_detection_new/object_seg_yolo.py and
vision_util/door_detection.py, which used to carry independent copies of
this reprojection hardcoded to 720x1280 (the Orbbec's old default color
resolution) -- see
docs/superpowers/specs/2026-07-03-orbbec-hri-resolution-bump-design.md.
"""
from __future__ import annotations

import numpy as np


def decode_depth_metres(depth_arr: np.ndarray) -> np.ndarray:
    """Coerce a decoded depth image to float32 metres.

    Orbbec Y16 depth decodes to uint16 millimetres via cv_bridge
    ``passthrough``; FoundationStereo-style depth is already float32 metres.
    """
    if depth_arr.dtype == np.uint16:
        return depth_arr.astype(np.float32) * 0.001
    if depth_arr.dtype == np.float32:
        return depth_arr
    raise ValueError(
        f'Unsupported depth dtype {depth_arr.dtype}; expected uint16 mm or '
        'float32 m.'
    )


def depth_image_to_points(depth_m: np.ndarray, k) -> np.ndarray:
    """Back-project a metres depth image to an (H, W, 3) points array.

    Args:
        depth_m: (H, W) depth in metres.
        k: 9-element row-major camera intrinsic matrix (CameraInfo.k):
            fx = k[0], fy = k[4], cx = k[2], cy = k[5].

    Returns:
        (H, W, 3) float32 array; [..., 0]=x, [..., 1]=y, [..., 2]=z(=depth_m),
        in the optical frame the intrinsics describe. H, W always match
        depth_m's own shape -- never hardcoded.
    """
    h, w = depth_m.shape
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    us, vs = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    z = depth_m.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest src/vision_util/test/test_depth_reproject.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Run the package's existing lint tests (new module must not break them)**

```bash
python3 -m pytest src/vision_util/test/test_flake8.py src/vision_util/test/test_pep257.py -v
```

Expected: PASS (the new module has a module docstring and no lint violations).

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/vision_util/vision_util/depth_reproject.py src/vision_util/test/test_depth_reproject.py
git commit -m "$(cat <<'EOF'
feat(vision_util): add resolution-agnostic Orbbec depth reprojection helper

Shared pinhole back-projection (depth Image + CameraInfo -> HxWx3 points)
that reads dimensions from the live depth image every call. Prerequisite
for fixing the two independent 720x1280-hardcoded reprojections in
object_seg_yolo.py and door_detection.py.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 2: Fix `object_seg_yolo.py`'s hardcoded Orbbec depth grid

**Files:**
- Modify: `src/tk26_vision/src/object_detection_new/object_detection_new/object_seg_yolo.py:17,124-129,282-310,355,399-431,575-581`
- Modify: `src/tk26_vision/src/object_detection_new/config/default.yaml:13`

**Interfaces:**
- Consumes: `vision_util.depth_reproject.decode_depth_metres`, `vision_util.depth_reproject.depth_image_to_points` (Task 1).
- Produces: `YOLOSegmentationNode._orbbec_depth_to_array(self, depth_msg: Image, intrinsic: CameraInfo) -> tuple[np.ndarray, np.ndarray]` — replaces `_pointcloud_to_array`. Same `(points, valid_mask)` return contract as before (points: `(H, W, 3)` float array, valid_mask: `(H, W)` bool array), so `_calculate_centroid` and every other downstream caller needs zero changes. `object_detection_generalist/generalist_node.py:320` and `tk_vision_specialized/{placing_location_server,object_match_server}.py` all call the *unchanged* `_process_orbbec_data` signature and inherit this fix automatically — do not touch those three files.

This task has no new automated test of its own: the reprojection math it now delegates to is already fully covered by Task 1's tests, and this codebase's existing convention (`test_yolo_segmentation.py`) deliberately avoids instantiating `YOLOSegmentationNode` in tests because `_init_model()` loads real YOLO weights. Correctness of this integration is verified live in Task 7.

- [ ] **Step 1: Change the Orbbec depth topic default**

In `src/tk26_vision/src/object_detection_new/object_detection_new/object_seg_yolo.py`, in `_declare_parameters` (around line 124-129), change:

```python
        # Orbbec topics
        self.declare_parameter(
            'orbbec_image_topic', '/camera/color/image_raw')
        self.declare_parameter(
            'orbbec_depth_topic', '/camera/depth_registered/points')
        self.declare_parameter(
            'orbbec_camera_info_topic', '/camera/color/camera_info')        
```

to:

```python
        # Orbbec topics
        self.declare_parameter(
            'orbbec_image_topic', '/camera/color/image_raw')
        self.declare_parameter(
            'orbbec_depth_topic', '/camera/depth/image_raw')
        self.declare_parameter(
            'orbbec_camera_info_topic', '/camera/color/camera_info')        
```

- [ ] **Step 2: Change the import line**

Change (line 17):

```python
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
```

to:

```python
from sensor_msgs.msg import Image, CameraInfo
```

Add, alongside the other `vision_util` imports (around line 35-37):

```python
from vision_util.vision_logging import VisionLogger
from vision_util.mask_utils import largest_connected_component_in_bbox
from vision_util.weights_cache import resolve_weights
from vision_util.depth_reproject import decode_depth_metres, depth_image_to_points
```

- [ ] **Step 3: Change the Orbbec depth subscriber from PointCloud2 to Image**

In `_init_subscribers` (around line 282-310), change:

```python
            depth_sub_orbbec = Subscriber(
                self, PointCloud2, orbbec_depth_topic, qos_profile=qos_profile
            )
```

to:

```python
            depth_sub_orbbec = Subscriber(
                self, Image, orbbec_depth_topic, qos_profile=qos_profile
            )
```

- [ ] **Step 4: Update `_orbbec_callback`'s type hint**

Change (line 355):

```python
    def _orbbec_callback(self, rgb_msg: Image, depth_msg: PointCloud2):
```

to:

```python
    def _orbbec_callback(self, rgb_msg: Image, depth_msg: Image):
```

- [ ] **Step 5: Replace `_pointcloud_to_array` with `_orbbec_depth_to_array`**

Delete the whole `_pointcloud_to_array` method (lines 399-431):

```python
    def _pointcloud_to_array(self, pc_msg: PointCloud2, intrinsic: CameraInfo) -> tuple:
        """
        Convert PointCloud2 to point array (Orbbec format).

        Orbbec outputs unordered point cloud, need to reproject to image grid.
        """
        h, w = 720, 1280
        K = np.array(intrinsic.k).reshape((3, 3))

        # Parse point cloud. Derive floats/point from point_step so both the 4-float
        # xyz layout (Femto Bolt default) and the 5-float xyzrgb layout
        # (enable_colored_point_cloud:=true) work — point_step is bytes/point, /4 = floats/point.
        floats_per_point = pc_msg.point_step // 4
        arr = np.frombuffer(pc_msg.data, dtype='<f4')
        N = len(arr) // floats_per_point
        points = arr.reshape((N, floats_per_point))[:, [0, 1, 2]]

        # Project to image coordinates
        points_homo = points / np.repeat(points[:, 2:3], 3, axis=1)
        coor_homo = (K @ points_homo.T).T
        coor = np.rint(coor_homo[:, :2]).astype(int)

        # Create depth image
        depth_img = np.zeros((h, w, 3))
        valid_coords = (coor[:, 0] >= 0) & (coor[:, 0] < w) & \
                       (coor[:, 1] >= 0) & (coor[:, 1] < h)
        depth_img[coor[valid_coords, 1], coor[valid_coords, 0], :] = points[valid_coords]

        # Valid mask
        valid_mask = (depth_img[:, :, 2] > self.min_depth) & \
                     (depth_img[:, :, 2] < self.max_depth)

        return depth_img, valid_mask
```

Replace it with:

```python
    def _orbbec_depth_to_array(self, depth_msg: Image, intrinsic: CameraInfo) -> tuple:
        """Reproject the Orbbec's registered depth Image to a points array.

        Depth is registered to color (depth_registration:=true), so its
        shape and frame always match the live color stream -- at whatever
        resolution the driver is launched with, never a fixed size.
        """
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_m = decode_depth_metres(depth_raw)
        points = depth_image_to_points(depth_m, intrinsic.k)

        valid_mask = (points[:, :, 2] > self.min_depth) & \
                     (points[:, :, 2] < self.max_depth)

        return points, valid_mask
```

- [ ] **Step 6: Update `_process_orbbec_data` to call the new method**

Change (lines 575-581):

```python
    def _process_orbbec_data(self, rgb_msg: Image, depth_msg: PointCloud2,
                             intrinsic: CameraInfo) -> tuple:
        """Process orbbec RGB-D data into usable format."""
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        points, valid_mask = self._pointcloud_to_array(depth_msg, intrinsic)

        return rgb_img, points, valid_mask, depth_msg.header
```

to:

```python
    def _process_orbbec_data(self, rgb_msg: Image, depth_msg: Image,
                             intrinsic: CameraInfo) -> tuple:
        """Process orbbec RGB-D data into usable format."""
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        points, valid_mask = self._orbbec_depth_to_array(depth_msg, intrinsic)

        return rgb_img, points, valid_mask, depth_msg.header
```

- [ ] **Step 7: Update the config yaml default**

In `src/tk26_vision/src/object_detection_new/config/default.yaml`, change:

```yaml
    orbbec_depth_topic: '/camera/depth_registered/points'
```

to:

```yaml
    orbbec_depth_topic: '/camera/depth/image_raw'
```

- [ ] **Step 8: Verify the module imports cleanly and existing tests + lint still pass**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
source .venv-vision-main/bin/activate
python3 -c "import ast; ast.parse(open('src/object_detection_new/object_detection_new/object_seg_yolo.py').read())"
python3 -m pytest src/object_detection_new/test/ -v
```

Expected: AST parse succeeds (no syntax errors); `test_yolo_segmentation.py`, `test_flake8.py`, `test_copyright.py`, `test_pep257.py` all PASS. `test_flake8.py` passing specifically confirms the `PointCloud2` import removal left no unused-import lint failure.

- [ ] **Step 9: Commit**

```bash
git add src/object_detection_new/object_detection_new/object_seg_yolo.py src/object_detection_new/config/default.yaml
git commit -m "$(cat <<'EOF'
fix(object_detection_new): stop hardcoding Orbbec depth grid to 720x1280

_pointcloud_to_array reprojected the Orbbec's colored PointCloud2 into a
buffer hardcoded to (720, 1280), silently dropping or crashing on any
detection outside that window at any other camera resolution -- inherited
unmodified by object_detection_generalist, so this blocked
BtNode_FeatureExtraction/BtNode_FeatureMatching at non-720p. Switches to
the registered depth Image + CameraInfo (vision_util.depth_reproject),
matching the resolution-safe pattern already used in
seat_recommend_bbox.py/get_orbbec_pc.py/person_track_node.py. Side effect:
decouples from the ordered colored point cloud, avoiding the
color-frame-thread CPU reprojection cost documented in orbbec_diagnosis.md.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 3: Fix `door_detection.py`'s hardcoded Orbbec depth grid

**Files:**
- Modify: `src/tk26_vision/src/vision_util/vision_util/door_detection.py`
- Create: `src/tk26_vision/src/vision_util/test/test_door_detection.py`

**Interfaces:**
- Consumes: `vision_util.depth_reproject.decode_depth_metres`, `vision_util.depth_reproject.depth_image_to_points` (Task 1).
- Produces: `DoorDetectionService.img_orbbec_process(self, color_msg, depth_msg: Image, intrinsic_msg: CameraInfo) -> tuple` — same 3-tuple `(color_img, depth_img, validmask)` contract as before; `depth_msg` type changes from `PointCloud2` to `Image`.

Unlike Task 2, `DoorDetectionService.__init__` does no model loading (just subscriptions/service), so it's cheap to instantiate directly in a test — this task gets a real, executable regression test.

- [ ] **Step 1: Write the failing tests**

Create `src/tk26_vision/src/vision_util/test/test_door_detection.py`:

```python
import unittest

import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

from vision_util.door_detection import DoorDetectionService


class TestDoorDetectionOrbbecDepth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = DoorDetectionService()
        self.bridge = CvBridge()

    def tearDown(self):
        self.node.destroy_node()

    def _camera_info(self, w, h, fx=500.0, fy=500.0):
        info = CameraInfo()
        info.width = w
        info.height = h
        cx, cy = w / 2.0, h / 2.0
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        return info

    def test_depth_array_matches_input_resolution_not_hardcoded_720x1280(self):
        # Regression: img_orbbec_process used to hardcode h, w = 720, 1280
        # and silently misalign at any other resolution.
        w, h = 1920, 1080
        depth = np.full((h, w), 2000, dtype=np.uint16)  # 2.0 m in mm
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        intrinsic = self._camera_info(w, h)

        _, depth_img, validmask = self.node.img_orbbec_process(
            None, depth_msg, intrinsic
        )

        self.assertEqual(depth_img.shape, (h, w, 3))
        self.assertEqual(validmask.shape, (h, w))

    def test_center_window_reads_live_depth_at_any_resolution(self):
        w, h = 1920, 1080
        depth = np.full((h, w), 1000, dtype=np.uint16)  # 1.0 m
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        intrinsic = self._camera_info(w, h)

        _, depth_img, validmask = self.node.img_orbbec_process(
            None, depth_msg, intrinsic
        )

        center_h, center_w = depth_img.shape[0] // 2, depth_img.shape[1] // 2
        crop = depth_img[center_h - 10:center_h + 10,
                          center_w - 10:center_w + 10, 2]
        valid_crop = validmask[center_h - 10:center_h + 10,
                                center_w - 10:center_w + 10]

        self.assertGreater(int(valid_crop.sum()), 5)
        self.assertAlmostEqual(
            float((crop * valid_crop).sum() / valid_crop.sum()), 1.0,
            places=2,
        )


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
source .venv-vision-main/bin/activate
python3 -m pytest src/vision_util/test/test_door_detection.py -v
```

Expected: FAIL — `img_orbbec_process` still expects a `PointCloud2`-shaped `depth_msg.data`/`.point_step` (an `Image` message has neither), so this raises an `AttributeError`.

- [ ] **Step 3: Rewrite `door_detection.py`'s Orbbec depth path**

In `src/tk26_vision/src/vision_util/vision_util/door_detection.py`:

Remove the now-unused numpy import (line 14):

```python
import numpy as np
```

Change the import line (line 20):

```python
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
```

to:

```python
from sensor_msgs.msg import CameraInfo, Image
```

Add:

```python
from vision_util.depth_reproject import decode_depth_metres, depth_image_to_points
```

Change the subscription (lines 30-36):

```python
        self.ptcloud_sub_orbbec = self.create_subscription(
            PointCloud2,
            '/camera/depth_registered/points',
            self.points_orbbec_callback,
            qos_profile=10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
```

to:

```python
        self.depth_sub_orbbec = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.points_orbbec_callback,
            qos_profile=10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
```

Replace `img_orbbec_process` (lines 68-83):

```python
    def img_orbbec_process(self, color_msg, depth_msg, intrinsic_msg):
        color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8') if color_msg is not None else None
        K = np.array(intrinsic_msg.k).reshape((3, 3))

        h, w = 720, 1280
        arr = np.frombuffer(depth_msg.data, dtype='<f4')
        N = len(arr) // 5
        points = arr.reshape((N, 5))[:, [0, 1, 2]]
        points_homo = points / np.repeat(points[:, 2:3], 3, axis=1)
        coor_homo = (K @ points_homo.T).T
        coor = np.rint(coor_homo[:, :2]).astype(int)
        depth_img = np.zeros((h, w, 3))
        depth_img[coor[:, 1], coor[:, 0], :] = points
        validmask = (depth_img[:, :, 2] > 1e-3).astype(int)

        return color_img, depth_img, validmask
```

with:

```python
    def img_orbbec_process(self, color_msg, depth_msg, intrinsic_msg):
        color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8') if color_msg is not None else None

        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_m = decode_depth_metres(depth_raw)
        depth_img = depth_image_to_points(depth_m, intrinsic_msg.k)
        validmask = (depth_img[:, :, 2] > 1e-3).astype(int)

        return color_img, depth_img, validmask
```

Change the center-window computation in `door_detection_srv_callback` (lines 106-114):

```python
        _, depth_img, validmask = self.img_orbbec_process(None, depth_msg, intrinsic_msg)

        W, H, L = 1280, 720, 10
        x1, x2, y1, y2 = H // 2 - L, H // 2 + L, W // 2 - L, W // 2 + L

        depth_crop = depth_img[x1:x2, y1:y2, 2]
        validmask_crop = validmask[x1:x2, y1:y2]
```

to:

```python
        _, depth_img, validmask = self.img_orbbec_process(None, depth_msg, intrinsic_msg)

        H, W = depth_img.shape[:2]
        L = 10
        x1, x2, y1, y2 = H // 2 - L, H // 2 + L, W // 2 - L, W // 2 + L

        depth_crop = depth_img[x1:x2, y1:y2, 2]
        validmask_crop = validmask[x1:x2, y1:y2]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest src/vision_util/test/test_door_detection.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Run the package's full test suite (lint + existing tests)**

```bash
python3 -m pytest src/vision_util/test/ -v
```

Expected: all PASS, including `test_flake8.py` (confirms the removed `numpy`/`PointCloud2` imports are genuinely unused and no lint violation was introduced).

- [ ] **Step 6: Commit**

```bash
git add src/vision_util/vision_util/door_detection.py src/vision_util/test/test_door_detection.py
git commit -m "$(cat <<'EOF'
fix(vision_util): stop hardcoding door_detection's Orbbec depth grid

img_orbbec_process hardcoded h, w = 720, 1280 (and a separate 5-floats/point
PointCloud2 layout assumption) when reprojecting the Orbbec depth data --
dormant on the live HRI path today (BtNode_DoorDetection's call site is
commented out) but part of HRI's always-on core launch group, so it would
misbehave at any other resolution the instant it's re-enabled. Switches to
the registered depth Image + CameraInfo (vision_util.depth_reproject),
same fix pattern as object_seg_yolo.py.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 4: Expose `color_width`/`color_height` on `vision_driver.launch.py`

**Files:**
- Modify: `src/tk26_vision/src/vision_bringup/launch/vision_driver.launch.py`

**Interfaces:**
- Produces: two new launch arguments, `color_width` (default `'1280'`) and `color_height` (default `'720'`), threaded into the Orbbec `IncludeLaunchDescription`. Task 5 overrides these to `1920`/`1080` for HRI specifically; every other caller of this launch file is unaffected (same defaults as today).

No automated pytest for launch files in this codebase's existing convention; verified via `ros2 launch --show-args`, which parses the launch description without touching hardware.

- [ ] **Step 1: Add the two launch arguments**

In `src/tk26_vision/src/vision_bringup/launch/vision_driver.launch.py`, in the `args` list, change:

```python
        DeclareLaunchArgument(
            'ffs_stream_align_to_color', default_value='false',
            description=(
                'false = non-aligned /foundation_stereo/depth/* (the form the '
                'cuMotion nvblox collision path consumes). Do not enable unless '
                'a consumer specifically needs aligned-to-color depth.'
            ),
        ),
    ]
```

to:

```python
        DeclareLaunchArgument(
            'ffs_stream_align_to_color', default_value='false',
            description=(
                'false = non-aligned /foundation_stereo/depth/* (the form the '
                'cuMotion nvblox collision path consumes). Do not enable unless '
                'a consumer specifically needs aligned-to-color depth.'
            ),
        ),
        DeclareLaunchArgument(
            'color_width', default_value='1280',
            description=(
                'Orbbec color stream width. Task launch scripts override this '
                '(e.g. HRI raises it for face/feature enrollment quality); '
                'default matches the vendored femto_bolt.launch.py default. '
                'Depth stream resolution is untouched -- SW alignment handles '
                'any color/depth size mismatch.'
            ),
        ),
        DeclareLaunchArgument(
            'color_height', default_value='720',
            description='Orbbec color stream height; see color_width.',
        ),
    ]
```

- [ ] **Step 2: Thread the arguments into the Orbbec include**

Change:

```python
    orbbec = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('orbbec_camera'), '/launch/femto_bolt.launch.py',
        ]),
        launch_arguments={
            'depth_registration': 'true',
            'enable_colored_point_cloud': 'true',
            'enable_ir': 'false',
            'enable_frame_sync': 'false',
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_orbbec')),
    )
```

to:

```python
    orbbec = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('orbbec_camera'), '/launch/femto_bolt.launch.py',
        ]),
        launch_arguments={
            'depth_registration': 'true',
            'enable_colored_point_cloud': 'true',
            'enable_ir': 'false',
            'enable_frame_sync': 'false',
            'color_width': LaunchConfiguration('color_width'),
            'color_height': LaunchConfiguration('color_height'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_orbbec')),
    )
```

- [ ] **Step 3: Verify the launch description parses and lists the new arguments**

```bash
cd /home/tinker/tk25_ws
source install/setup.zsh
ros2 launch vision_bringup vision_driver.launch.py --show-args
```

Expected: exits 0, output includes `'color_width'` and `'color_height'` with their default values, alongside the existing `enable_pan_tilt`/`enable_orbbec`/etc. No hardware/camera required for `--show-args`.

- [ ] **Step 4: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/vision_bringup/launch/vision_driver.launch.py
git commit -m "$(cat <<'EOF'
feat(vision_bringup): expose Orbbec color_width/color_height as launch args

Passthrough only -- defaults (1280/720) match today's behavior for every
existing caller. Lets a task-specific launch script (tmux_hri_vision.sh)
override resolution without touching this generic driver-layer launch.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 5: Override to 1920×1080 in the HRI launch script

**Files:**
- Modify: `src/tk25_basic/src/scripts/tmux_hri_vision.sh`

**Interfaces:**
- Consumes: `color_width`/`color_height` launch args from Task 4.

**Note before editing:** `git status` in `src/tk25_basic` currently shows this file with an unrelated pending local modification (a stray backslash in the *pane 1* `vision_bringup.launch.py` line, unrelated to pane 0 which this task edits). Re-read the file's current content before editing rather than assuming the checked-in baseline — do not touch or revert that unrelated pane-1 change.

- [ ] **Step 1: Add the resolution override to both pane-0 branches**

In `src/tk25_basic/src/scripts/tmux_hri_vision.sh`, change:

```bash
if [ -n "$DEV" ]; then
    tmux send-keys -t $SESSION:$WINDOW.0 \
        "source ~/tk25_ws/install/setup.zsh && ros2 launch vision_bringup vision_driver.launch.py device:=\"$DEV\" launch_robot_state_publisher:=false; exec zsh" C-m
else
    tmux send-keys -t $SESSION:$WINDOW.0 \
        "source ~/tk25_ws/install/setup.zsh && ros2 launch vision_bringup vision_driver.launch.py enable_pan_tilt:=false launch_robot_state_publisher:=false; exec zsh" C-m
fi
```

to:

```bash
if [ -n "$DEV" ]; then
    tmux send-keys -t $SESSION:$WINDOW.0 \
        "source ~/tk25_ws/install/setup.zsh && ros2 launch vision_bringup vision_driver.launch.py device:=\"$DEV\" launch_robot_state_publisher:=false color_width:=1920 color_height:=1080; exec zsh" C-m
else
    tmux send-keys -t $SESSION:$WINDOW.0 \
        "source ~/tk25_ws/install/setup.zsh && ros2 launch vision_bringup vision_driver.launch.py enable_pan_tilt:=false launch_robot_state_publisher:=false color_width:=1920 color_height:=1080; exec zsh" C-m
fi
```

- [ ] **Step 2: Syntax-check the script and confirm both branches got the override**

```bash
cd /home/tinker/tk25_ws/src/tk25_basic
bash -n src/scripts/tmux_hri_vision.sh
grep -c "color_width:=1920 color_height:=1080" src/scripts/tmux_hri_vision.sh
```

Expected: `bash -n` prints nothing (exit 0); `grep -c` prints `2`.

- [ ] **Step 3: Commit**

```bash
git add src/scripts/tmux_hri_vision.sh
git commit -m "$(cat <<'EOF'
feat(scripts): launch HRI's Orbbec driver at 1920x1080 instead of 720p

Only the HRI launch script overrides vision_driver.launch.py's new
color_width/color_height args -- every other task launching that file
keeps the vendored 1280x720 default.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 6: Raise the FastDDS SHM segment size

**Files:**
- Modify: `src/tk26_vision/config/fastdds_shm.xml`

**Interfaces:** None (leaf config file; consumed via the `FASTRTPS_DEFAULT_PROFILES_FILE` env var already set by `vision_driver.launch.py`/`vision_bringup.launch.py`).

- [ ] **Step 1: Bump the segment size and note why**

In `src/tk26_vision/config/fastdds_shm.xml`, change the header comment's closing reference line and the `segment_size` value. Change:

```xml
See src/tk26_vision/DEV_NOTES.md §"2026-04-22 — Camera bringup performance fix"
for the full diagnosis.
-->
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>shm_transport</transport_id>
            <type>SHM</type>
            <segment_size>20971520</segment_size>
```

to:

```xml
See src/tk26_vision/DEV_NOTES.md §"2026-04-22 — Camera bringup performance fix"
for the full diagnosis.

segment_size raised 20MB -> 64MB on 2026-07-03 to carry the HRI task's
1920x1080 Orbbec color frames (~6.2MB raw uncompressed, vs ~1.2MB at
720p) with headroom for several in-flight frames across multiple
subscribers. See
src/tk26_vision/docs/superpowers/specs/2026-07-03-orbbec-hri-resolution-bump-design.md.
-->
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>shm_transport</transport_id>
            <type>SHM</type>
            <segment_size>67108864</segment_size>
```

- [ ] **Step 2: Verify the XML is well-formed and the new value is present**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
python3 -c "import xml.etree.ElementTree as ET; ET.parse('config/fastdds_shm.xml'); print('OK')"
grep -n "segment_size" config/fastdds_shm.xml
```

Expected: prints `OK`; the `segment_size` line shows `67108864`.

- [ ] **Step 3: Commit**

```bash
git add config/fastdds_shm.xml
git commit -m "$(cat <<'EOF'
chore(vision_bringup): raise FastDDS SHM segment 20MB -> 64MB for HRI 1080p

1920x1080 raw color frames (~6.2MB) need headroom this profile's current
20MB segment doesn't have. This profile is shared by pan_tilt+Orbbec+FFS
in the driver-layer launch -- enlarging it is additive/safe, consistent
with this file's prior verified-safe 512KB->20MB raise for the RealSense
IR pair.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Ei1BHsidgbudrkDS3yhuJ
EOF
)"
```

---

### Task 7: Live verification (operator-in-the-loop, requires hardware)

**Files:** None — no code changes. Gate before declaring this feature done.

No automated test can exercise the Orbbec hardware path. Run these in order; each depends on the previous succeeding.

- [ ] **Step 1: Rebuild the touched packages**

```bash
cd /home/tinker/tk25_ws
./src/tk26_vision/scripts/build.sh --packages-select vision_util object_detection_new object_detection_generalist vision_bringup
```

Expected: build succeeds with no errors.

- [ ] **Step 2: Launch the HRI driver layer and confirm 1080p + sustained rate**

```bash
source ~/tk25_ws/src/tk25_basic/src/scripts/tmux_hri_vision.sh
# or run master_hri.sh per the normal operator flow
```

In a separate shell:

```bash
source ~/tk25_ws/install/setup.zsh
ros2 topic echo /camera/color/camera_info --once | grep -E "width|height"
ros2 topic hz /camera/color/image_raw
```

Expected: `width: 1920`, `height: 1080`; `ros2 topic hz` reports ~30Hz sustained (matching the pre-existing 720p baseline in `CAMERA_BRINGUP.md`), no `RcvbufErrors`/drop warnings, and no SHM segment-overflow / negotiation-failure warnings, in the Orbbec node's console output.

- [ ] **Step 3: Confirm the colored point cloud is silent (zero-subscriber theory)**

```bash
timeout 5 ros2 topic hz /camera/depth_registered/points
```

Expected: no messages received within the 5s window (or a near-zero rate) — confirms `object_detection_generalist`/`door_detection` no longer subscribe to it and the vendored driver's zero-subscriber early-return is skipping the expensive reprojection. If this topic IS actively publishing, something still subscribes to it — stop and investigate before proceeding (the SHM sizing in Task 6 assumed this topic would be idle).

- [ ] **Step 4: Run a live HRI session through enrollment and follow**

Start the perception layer (`ros2 launch vision_bringup vision_bringup.launch.py enable_hri:=true`) and run the full HRI behavior tree. Confirm:
- Host/guest feature extraction succeeds (`BtNode_FeatureExtraction`, phases 2-3) — no service hangs, no `ValueError` in `object_detection_generalist`'s log.
- Two-way introduction's feature matching succeeds (`BtNode_FeatureMatching`, phase 5) — centroids returned are sane (not all-zero/dropped).
- Seat recommendation succeeds (`BtNode_SeatRecommendBbox`, phase 4).
- Person tracking/following (phase 6) behaves normally through to task completion.

- [ ] **Step 5: Manually re-verify `door_detection` at 1080p**

```bash
ros2 service call /door_detection_srv tinker_vision_msgs_26/srv/DoorDetection "{camera: 'orbbec'}"
```

Expected: returns `status: 0` with a sane `is_open` value (not an exception/timeout) — confirms the fix works live even though its BT hook stays disabled.

- [ ] **Step 6: Record the outcome**

If all steps pass, this feature is complete — no further code changes needed. If `ros2 topic hz /camera/depth_registered/points` in Step 3 shows nonzero traffic, or Step 2's frame rate is degraded, stop and re-open the design spec's §5 (Bandwidth/SHM sizing) rather than guessing at a larger segment size.
