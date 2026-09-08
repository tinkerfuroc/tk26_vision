# Door Detection via Depth Image — Implementation Plan

> **STATUS: DEFERRED / NOT ADOPTED (2026-07-01).** This plan was implemented then **reverted** — the existing point-cloud door detector was confirmed working live (`/camera/depth_registered/points` is `point_step=20` = 5 floats/point, so the old `reshape((N,5))` is correct). Kept as a ready-to-run blueprint if a future switch to depth-image detection is wanted. See the spec's Status note for details.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken point-cloud parse in the `door_detection` service with a depth-image center-patch check: closed iff the center of the (level, forward) Orbbec depth image shows enough valid pixels nearer than a threshold; else open.

**Architecture:** Split the pure decision math (numpy-only) into `_door_logic.py` for hardware-free unit testing, and make `door_detection.py` a thin ROS node that subscribes to `/camera/depth/image_raw`, decodes it, and delegates to the pure functions. The `DoorDetection.srv`, node name, and service name are unchanged, so `tk25_decision`'s `BtNode_DoorDetection` and the Inspection tree need no edits.

**Tech Stack:** Python 3.10, numpy, rclpy, cv_bridge, ROS2 Humble; `tk26_vision` `.venv-vision-main`.

**Related spec:** `docs/superpowers/specs/2026-07-01-door-detection-depth-image-design.md`

## Global Constraints

- **Repo/pkg:** `tk26_vision` / `vision_util`, branch `dev`. Commit new only; never `--amend`/rebase/force. Stage only the task's exact files by path (the tree is clean now, but keep commits precise).
- **Interfaces frozen:** `DoorDetection.srv`, node name `door_detection_service`, service name `door_detection_srv` unchanged. No `tk25_decision` edits.
- **Detection rule (exact):** `is_open = 0` (closed) iff `valid_count >= min_valid_px` AND `median_m < open_threshold_m`; otherwise `is_open = 1`. Valid pixel = finite AND `> 1e-3 m`.
- **Params + defaults:** `open_threshold_m=1.5` (double), `center_patch_px=30` (int), `min_valid_px=50` (int), `depth_topic='/camera/depth/image_raw'` (string). The three numeric params are re-read per service call; `depth_topic` is read once in `__init__`.
- **Depth encoding:** `16UC1`/`mono16` → millimetres (÷1000); `32FC1` → metres; anything else → `ValueError` → service `status=1`.
- **QoS:** subscribe with `rclpy.qos.qos_profile_sensor_data` (best-effort).
- **Test command (runs against SOURCE, no rebuild needed):**
  ```
  cd /home/tinker/tk25_ws/src/tk26_vision/src/vision_util
  PYTHONPATH=/home/tinker/tk25_ws/src/tk26_vision/src/vision_util:$PYTHONPATH \
    /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_door_detection.py -v
  ```
  (`install/vision_util` is a plain copy, so `ros2 run` / live testing needs a rebuild: `./src/tk26_vision/scripts/build.sh --packages-select vision_util` — the **user drives builds**; Claude runs pytest + read-only checks.)

---

### Task 1: Pure depth-image door logic (`_door_logic.py`)

**Files:**
- Create: `src/vision_util/vision_util/_door_logic.py`
- Test: `src/vision_util/test/test_door_detection.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DoorResult` dataclass — fields `is_open: int`, `valid_count: int`, `median_m: float`.
  - `depth_to_meters(arr: np.ndarray, encoding: str) -> np.ndarray` — float32 metres; raises `ValueError` on unsupported encoding.
  - `evaluate_door(depth_m: np.ndarray, *, open_threshold_m: float, center_patch_px: int, min_valid_px: int) -> DoorResult`.
  Task 2's node imports all three from `vision_util._door_logic`.

- [ ] **Step 1: Write the failing test**

Create `src/vision_util/test/test_door_detection.py`:

```python
import numpy as np
import pytest

from vision_util._door_logic import DoorResult, depth_to_meters, evaluate_door

PARAMS = dict(open_threshold_m=1.5, center_patch_px=30, min_valid_px=50)


def _img(fill_m, h=576, w=640):
    return np.full((h, w), fill_m, dtype=np.float32)


def test_closed_near_surface():
    r = evaluate_door(_img(1.0), **PARAMS)
    assert r.is_open == 0
    assert r.valid_count == 900          # 30x30 center patch
    assert r.median_m == pytest.approx(1.0)


def test_open_far_surface():
    r = evaluate_door(_img(3.0), **PARAMS)
    assert r.is_open == 1
    assert r.median_m == pytest.approx(3.0)


def test_open_when_center_all_invalid():
    # open door beyond range -> center returns 0 (invalid)
    r = evaluate_door(_img(0.0), **PARAMS)
    assert r.is_open == 1
    assert r.valid_count == 0
    assert r.median_m == 0.0


def test_boundary_valid_count():
    img = _img(0.0)
    cy, cx = 576 // 2, 640 // 2
    img[cy:cy + 5, cx:cx + 10] = 1.0     # 50 near pixels inside the 30x30 patch
    r = evaluate_door(img, **PARAMS)
    assert r.valid_count == 50
    assert r.is_open == 0                 # 50 >= 50 and median 1.0 < 1.5
    img[cy, cx] = 0.0                     # one fewer valid -> open
    r2 = evaluate_door(img, **PARAMS)
    assert r2.valid_count == 49
    assert r2.is_open == 1


def test_boundary_threshold():
    assert evaluate_door(_img(1.49), **PARAMS).is_open == 0
    assert evaluate_door(_img(1.51), **PARAMS).is_open == 1


def test_non_finite_excluded():
    img = _img(np.nan)
    cy, cx = 576 // 2, 640 // 2
    img[cy:cy + 8, cx:cx + 8] = 1.0      # 64 finite near pixels
    r = evaluate_door(img, **PARAMS)
    assert r.valid_count == 64
    assert r.is_open == 0
    assert np.isfinite(r.median_m)


def test_small_image_clamps():
    r = evaluate_door(_img(1.0, h=10, w=10), **PARAMS)
    assert r.valid_count == 100           # whole 10x10 clamped patch, valid & near
    assert r.is_open == 0


def test_depth_to_meters_16uc1():
    arr = np.array([[1500, 0], [800, 2000]], dtype=np.uint16)
    out = depth_to_meters(arr, '16UC1')
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[1.5, 0.0], [0.8, 2.0]])


def test_depth_to_meters_32fc1_passthrough():
    out = depth_to_meters(np.array([[1.5, 0.0]], dtype=np.float32), '32FC1')
    np.testing.assert_allclose(out, [[1.5, 0.0]])


def test_depth_to_meters_unsupported_raises():
    with pytest.raises(ValueError):
        depth_to_meters(np.zeros((2, 2), np.uint8), 'rgb8')
```

- [ ] **Step 2: Run the test to verify it fails**

Run the Global-Constraints test command.
Expected: collection/import error — `ModuleNotFoundError: No module named 'vision_util._door_logic'` (the module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Create `src/vision_util/vision_util/_door_logic.py`:

```python
"""Pure (numpy-only) door open/closed decision from a depth image.

No ROS / cv_bridge imports, so it can be unit-tested with synthetic arrays.
door_detection.py (the ROS node) decodes the depth Image into a metres array
and calls these functions.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class DoorResult:
    is_open: int          # 1 = open, 0 = closed
    valid_count: int      # valid pixels in the center patch
    median_m: float       # median depth of valid pixels (0.0 if none)


def depth_to_meters(arr: np.ndarray, encoding: str) -> np.ndarray:
    """Convert a decoded depth image to a float32 metres array.

    16UC1 / mono16 are millimetres; 32FC1 is already metres. Raises
    ValueError on any other encoding so the caller can report it.
    """
    enc = encoding.lower()
    if enc in ('16uc1', 'mono16'):
        return arr.astype(np.float32) / 1000.0
    if enc == '32fc1':
        return arr.astype(np.float32)
    raise ValueError(f'Unsupported depth encoding: {encoding}')


def evaluate_door(depth_m: np.ndarray, *, open_threshold_m: float,
                  center_patch_px: int, min_valid_px: int) -> DoorResult:
    """Decide door open/closed from a metres depth array.

    Extract the centered center_patch_px x center_patch_px patch (clamped to
    the image bounds), keep finite pixels > 1e-3 m as valid, and take their
    median. Closed (is_open=0) iff valid_count >= min_valid_px AND
    median < open_threshold_m; otherwise open (is_open=1).
    """
    h, w = depth_m.shape[:2]
    half_lo = center_patch_px // 2
    half_hi = center_patch_px - half_lo
    r0 = max(0, h // 2 - half_lo)
    r1 = min(h, h // 2 + half_hi)
    c0 = max(0, w // 2 - half_lo)
    c1 = min(w, w // 2 + half_hi)
    patch = depth_m[r0:r1, c0:c1]

    mask = np.isfinite(patch) & (patch > 1e-3)
    valid_count = int(mask.sum())

    if valid_count < min_valid_px:
        return DoorResult(is_open=1, valid_count=valid_count, median_m=0.0)

    median_m = float(np.median(patch[mask]))
    is_open = 0 if median_m < open_threshold_m else 1
    return DoorResult(is_open=is_open, valid_count=valid_count, median_m=median_m)
```

- [ ] **Step 4: Run the test to verify it passes**

Run the Global-Constraints test command.
Expected: 9 passed.

- [ ] **Step 5: Commit** (precise add — stage only these two files)

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/vision_util/vision_util/_door_logic.py src/vision_util/test/test_door_detection.py
git commit -m "feat(vision_util): pure depth-image door-detection logic (_door_logic) + tests"
```

---

### Task 2: Rewrite the `door_detection` node to use the depth image

**Files:**
- Modify (full rewrite): `src/vision_util/vision_util/door_detection.py`

**Interfaces:**
- Consumes: `depth_to_meters`, `evaluate_door` from `vision_util._door_logic` (Task 1).
- Produces: unchanged ROS surface — node `door_detection_service`, service `door_detection_srv` (`DoorDetection`), request field `camera`, response `status`/`error_msg`/`is_open`. No downstream code changes.

- [ ] **Step 1: Replace the entire file**

Overwrite `src/vision_util/vision_util/door_detection.py` with:

```python
"""Door-state detection service (depth-image based).

Reads the center of the Orbbec depth image (assuming the camera is level and
pointing forward): the door is open (is_open=1) when the center sees far / no
depth return, closed (is_open=0) when it sees a near surface within
open_threshold_m. Only the Orbbec camera is supported. The pure decision math
lives in _door_logic.py.
"""
import threading

import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tinker_vision_msgs_26.srv import DoorDetection

from vision_util._door_logic import depth_to_meters, evaluate_door


class DoorDetectionService(Node):
    def __init__(self):
        super().__init__('door_detection_service')

        self.declare_parameter('open_threshold_m', 1.5)
        self.declare_parameter('center_patch_px', 30)
        self.declare_parameter('min_valid_px', 50)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.recent_depth = None

        depth_topic = self.get_parameter('depth_topic').value
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_profile=qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.srv = self.create_service(
            DoorDetection,
            'door_detection_srv',
            self.door_detection_srv_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            f'Door detection service initialized (depth topic: {depth_topic}).'
        )

    def depth_callback(self, msg: Image):
        with self.lock:
            self.recent_depth = msg

    def door_detection_srv_callback(
        self,
        request: DoorDetection.Request,
        response: DoorDetection.Response,
    ):
        if 'realsense' in request.camera:
            response.status = 1
            response.error_msg = 'Only orbbec camera is supported.'
            return response

        with self.lock:
            msg = self.recent_depth

        if msg is None:
            self.get_logger().warn('No depth image received yet.')
            response.status = 1
            response.error_msg = 'No depth image received yet.'
            return response

        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_m = depth_to_meters(depth_raw, msg.encoding)
        except ValueError as exc:
            response.status = 1
            response.error_msg = str(exc)
            return response

        result = evaluate_door(
            depth_m,
            open_threshold_m=self.get_parameter('open_threshold_m').value,
            center_patch_px=self.get_parameter('center_patch_px').value,
            min_valid_px=self.get_parameter('min_valid_px').value,
        )
        self.get_logger().info(
            f'valid={result.valid_count} median={result.median_m:.3f} m '
            f'-> is_open={result.is_open}'
        )
        response.is_open = result.is_open
        response.status = 0
        response.error_msg = ''
        return response


def main():
    rclpy.init()
    node = DoorDetectionService()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify it loads (import check against source)**

There is no new unit test for this task: the decision logic is fully covered by Task 1's `_door_logic` tests, and the rest is ROS glue (subscription, params, cv_bridge decode) that can only be exercised live. The automated gate is that the module imports cleanly (syntax + every dependency resolves):

Run:
```
cd /home/tinker/tk25_ws/src/tk26_vision/src/vision_util
PYTHONPATH=/home/tinker/tk25_ws/src/tk26_vision/src/vision_util:$PYTHONPATH \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "import vision_util.door_detection as d; print('loads OK:', d.DoorDetectionService.__name__)"
```
Expected: `loads OK: DoorDetectionService`

Then re-run Task 1's test command to confirm the pure logic still passes (9 passed) — the node delegates to it unchanged.

- [ ] **Step 3: Commit** (precise, pathspec — `door_detection.py` is tracked)

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git commit -m "feat(vision_util): door_detection uses depth-image center patch (replaces broken point-cloud parse)" \
  src/vision_util/vision_util/door_detection.py
```

- [ ] **Step 4: User-driven live smoke test** (operator, after a rebuild)

The user rebuilds and runs (install is a plain copy, so a rebuild is required for `ros2 run`):
```bash
./src/tk26_vision/scripts/build.sh --packages-select vision_util
# with vision_driver.launch.py (or the Orbbec) up:
ros2 topic echo --field encoding /camera/depth/image_raw     # expect 16UC1
ros2 run vision_util door_detection
ros2 service call /door_detection_srv tinker_vision_msgs_26/srv/DoorDetection "{camera: 'orbbec'}"
```
Expected: facing a near wall/closed door → `is_open: 0`; facing an open doorway → `is_open: 1`; node logs `valid=… median=… -> is_open=…`.

---

## Self-Review

**1. Spec coverage** (against `2026-07-01-door-detection-depth-image-design.md`):
- §3.1 `_door_logic` (DoorResult, evaluate_door, patch clamp, valid mask, median rule) → Task 1 impl + `test_closed/open/boundary/non_finite/small_image`. ✓
- §3.3 encoding guard (`depth_to_meters`) → moved into `_door_logic`; Task 1 `test_depth_to_meters_*`. ✓
- §3.2 node (depth sub, SensorDataQoS, params declared, realsense reject, no-frame status=1, decode→evaluate, log) → Task 2 file. ✓
- §3.2 `main()` shutdown guard (`if rclpy.ok()`) → Task 2 `main()`. ✓
- §4 params + per-call re-read of the three numerics; `depth_topic` read once → Task 2 `__init__`/callback. ✓
- §1.2 srv/node/service names unchanged, no tk25_decision edits → only files touched are `_door_logic.py`, `door_detection.py`, the test. ✓
- §6 unit cases + live smoke → Task 1 tests + Task 2 Step 4. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step is complete; every run step has an exact command + expected output. Task 2's "no new unit test" is stated with its justification (delegated logic + ROS glue), not a vacuous test. ✓

**3. Type consistency:** `DoorResult(is_open,valid_count,median_m)`, `depth_to_meters(arr,encoding)`, `evaluate_door(depth_m,*,open_threshold_m,center_patch_px,min_valid_px)` are defined in Task 1 and imported/called with those exact names/kwargs in Task 2. Param names match the ROS `declare_parameter` keys and the Global Constraints table. ✓
