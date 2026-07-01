# Door Detection via Depth Image — Design Spec

- **Date:** 2026-07-01
- **Status:** DEFERRED (2026-07-01) — **NOT adopted; documented as a future option.** The existing point-cloud detector was confirmed **working live**: `/camera/depth_registered/points` publishes `point_step=20` (5 float32/point — x, y, z, rgb, + 1 padding float), so the old `np.frombuffer(...).reshape((N,5))[:, [0,1,2]]` correctly extracts xyz and never raises (`len == width*5` is always ÷5). The earlier "broken parse" review conclusion was **wrong** (it read the vendored OrbbecSDK source, which builds 16-byte points, instead of confirming the running 20-byte topic). The implementation for this design was built and then **reverted** (feature commits `3e1d3e8`, `c22ad0a` → reverts `5bfdd7a`, `efaff8c`). This document + the plan below are retained as a clean, tested blueprint if a switch to depth-image detection is ever wanted (e.g. to drop the point-cloud dependency or for easier live tuning).
- **Repo touched:** `tk26_vision` (`vision_util`), branch `dev`
- **Author:** Claude (Opus 4.8), with cindy
- **Related:** `vision_util/vision_util/door_detection.py`, `tinker_vision_msgs_26/srv/DoorDetection.srv`; consumed by `tk25_decision`'s `BtNode_DoorDetection` (`TemplateNodes/Vision.py:1271`) in the Inspection tree.

---

## 1. Goal

Rewrite the `door_detection` **service implementation** to decide door open/closed from the Orbbec **depth image** (`/camera/depth/image_raw`, `16UC1` millimetres) instead of the colored point cloud. The current point-cloud parse assumes 5 float32/point (20 B); the tk26 OrbbecSDK v2 driver publishes `depth_registered/points` at `point_step=16` (4 floats: xyz + packed rgb) with `ordered_pc=false`, so `np.frombuffer(...).reshape((N,5))` raises `ValueError` on most calls (and yields garbage otherwise) — the door step never functions on the real robot. The depth-image approach needs no point cloud, no intrinsics, and no projection: read the center of the (level, forward-pointing) camera and check whether it sees a near surface.

### 1.1 Detection logic
Assuming the camera is level and pointing forward, the image center is the optical axis. A **closed** door is a near surface returning dense valid depth under a threshold; an **open** door looks through to far space, which the Femto Bolt returns as either large depth or **0 (invalid, beyond range)**.

`is_open = 0` (closed) **iff** `valid_count ≥ min_valid_px` **AND** `median_depth_m < open_threshold_m`; otherwise `is_open = 1` (open).

### 1.2 In scope
- Rewrite `vision_util/vision_util/door_detection.py` to subscribe to the depth image and evaluate the center patch.
- Extract the pure detection math into a numpy-only module `_door_logic.py` for hardware-free unit testing.
- Expose ROS params (§4). Keep the `DoorDetection.srv`, node name (`door_detection_service`), and service name (`/door_detection_srv`) **unchanged** so `BtNode_DoorDetection` and the Inspection tree need no edits.

### 1.3 Out of scope (YAGNI)
- No change to `DoorDetection.srv`, `BtNode_DoorDetection`, or any tk25_decision code.
- No TF / leveling compensation — the "camera level, pointing forward" assumption is accepted by design.
- No RealSense support (rejected as today).
- No multi-region / horizon-scan logic — single center patch only.

### 1.4 Success criteria
- `_door_logic.evaluate_door(...)` returns the correct `(is_open, valid_count, median_m)` for synthetic depth arrays covering closed / open-far / open-holes / boundary cases, under pytest with no ROS or camera.
- The node subscribes to `/camera/depth/image_raw`, and a `door_detection_srv` call returns `status=0` + a definite `is_open` whenever a depth frame has arrived; `status=1` before the first frame or for a `realsense` request.
- All four params are runtime-overridable; defaults per §4.
- Live: with `vision_driver.launch.py` up, calling the service reads `is_open=0` facing a near wall/closed door and `is_open=1` facing an open doorway.

---

## 2. Background — current state (source-verified)

- **Consumer contract:** `BtNode_DoorDetection` sends `DoorDetection.Request(camera="orbbec")`, and treats `result.is_open == 1` as SUCCESS, anything else as FAILURE (`Vision.py:1326`). In the Inspection tree it's wrapped in `Retry(999)`, so FAILURE just re-polls — the wait-for-open loop. This contract is preserved exactly.
- **Srv (`DoorDetection.srv`):** request `string camera`; response `int32 status, string error_msg, int32 is_open`. Unchanged.
- **Depth source (verified):** `vision_driver.launch.py` includes `femto_bolt.launch.py` with `depth_registration:=true`, `enable_colored_point_cloud:=true`. The depth image is published on `/camera/depth/image_raw`, encoding `TYPE_16UC1` (uint16 millimetres) — `ob_camera_node.cpp:1931-1932`; depth default 640×576. The image center is the optical axis regardless of resolution or registration.
- **Why not the point cloud (the bug being fixed):** `door_detection.py:73-75` does `arr = np.frombuffer(data,'<f4'); N=len(arr)//5; arr.reshape((N,5))`. The cloud is `point_step=16` (4 floats/point) and filtered (`ordered_pc=false`) → dynamic size rarely divisible by 5 → `ValueError` in the callback → the client future never completes → `BtNode_DoorDetection` hangs in `RUNNING`. Both test logs only ever hit "No camera data", so the parse never ran live.

---

## 3. Architecture

Two units with a clean boundary:

### 3.1 `vision_util/vision_util/_door_logic.py` (pure, numpy-only)
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class DoorResult:
    is_open: int          # 1 open, 0 closed
    valid_count: int
    median_m: float       # 0.0 when valid_count == 0

def evaluate_door(depth_m: np.ndarray, *, open_threshold_m: float,
                  center_patch_px: int, min_valid_px: int) -> DoorResult:
    """depth_m: 2D float array in metres; invalid pixels are 0.0 or non-finite.
    Extract the centered center_patch_px × center_patch_px patch, keep finite
    pixels > 1e-3 m as valid, take their median. Closed iff
    valid_count >= min_valid_px and median < open_threshold_m; else open."""
```
- Center patch: rows `[h//2 - p//2 : h//2 + (p - p//2)]`, cols likewise, clamped to the image bounds (handles images smaller than the patch).
- Valid mask: `np.isfinite(patch) & (patch > 1e-3)`.
- `valid_count = int(mask.sum())`. If `valid_count < min_valid_px` → `is_open=1`, `median_m=0.0` (skip median). Else `median_m = float(np.median(patch[mask]))`; `is_open = 0 if median_m < open_threshold_m else 1`.
- No ROS, no cv_bridge, no rclpy imports → importable and testable standalone.

### 3.2 `vision_util/vision_util/door_detection.py` (thin ROS node)
- Subscribe to `depth_topic` (default `/camera/depth/image_raw`) with **`SensorDataQoS`** (best-effort — compatible with reliable *and* best-effort camera publishers; removes the latent QoS-mismatch drop the old `qos=10` could suffer). Store the latest `Image` msg under a lock. Sync callback (drop the old `async def`).
- Declare params (§4) in `__init__` (so they're overridable at launch and via `ros2 param set`).
- Service callback `door_detection_srv_callback`:
  1. If `'realsense' in request.camera`: `status=1, error_msg='Only orbbec camera is supported.'`, return.
  2. Snapshot latest depth msg under lock. If `None`: `status=1, error_msg='No depth image received yet.'`, return.
  3. Read the current param values (`self.get_parameter(...).value`), decode to a metres float array (§3.3), call `evaluate_door(...)`, set `is_open`, `status=0`, `error_msg=''`. Log `valid_count` + `median_m`.

  `depth_topic` is the exception — it's read once in `__init__` to create the subscription (changing the topic needs a restart); the three numeric params are re-read per call for live retuning.
- `main()`: guard shutdown — `if rclpy.ok(): rclpy.shutdown()` — fixing the `rcl_shutdown already called` traceback both logs show.

### 3.3 Depth decode (encoding guard)
`cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')` →
- `16UC1` / `mono16` → `arr.astype(np.float32) / 1000.0` (mm→m).
- `32FC1` → `arr.astype(np.float32)` (already metres) — covers config drift.
- other encoding → `status=1, error_msg='Unsupported depth encoding: <enc>'`.

---

## 4. ROS parameters

| param | type | default | meaning |
|---|---|---|---|
| `open_threshold_m` | double | `1.5` | center-depth below this (with enough valid px) ⇒ closed |
| `center_patch_px` | int | `30` | side length of the centered square patch (30×30) |
| `min_valid_px` | int | `50` | fewer valid pixels in the patch ⇒ treated as open |
| `depth_topic` | string | `/camera/depth/image_raw` | depth image subscription |

The three numeric params (`open_threshold_m`, `center_patch_px`, `min_valid_px`) are re-read per service call, so `ros2 param set` retunes them live without a restart. `depth_topic` is read once in `__init__` to build the subscription (changing it needs a restart).

---

## 5. Error handling
- RealSense request → `status=1` (unchanged behavior).
- No depth frame yet → `status=1` → `BtNode_DoorDetection` FAILURE → `Retry` re-polls (same as today's "No camera data").
- Unsupported encoding → `status=1` with a clear message (won't happen at defaults; guards config drift).
- A valid frame always yields `status=0` and a definite `is_open`. Invalid/holey center (open door beyond range) → `valid_count < min_valid_px` → `is_open=1`, never a crash.

## 6. Testing

**Unit (`src/vision_util/test/test_door_detection.py`, pure `_door_logic`, no ROS/hardware):**
- closed: 30×30 patch filled `1.0 m`, rest far → `is_open=0`, `valid_count=900`.
- open-far: center patch `3.0 m` → `is_open=1`.
- open-holes: center patch all `0.0` (invalid) → `valid_count=0 < 50` → `is_open=1`, `median_m=0.0`.
- boundary valid-count: exactly `min_valid_px` valid near pixels (rest zero) with median `< threshold` → `is_open=0`; one fewer → `is_open=1`.
- boundary threshold: median just below `1.5` → closed; just above → open.
- non-finite handling: patch with NaN/inf mixed with a few near pixels → NaNs excluded, decision from finite valids.
- small image: array smaller than 30×30 → patch clamps, no crash.

**Live (operator, not automated):** `vision_driver.launch.py` up (or bare `ros2 run vision_util door_detection` with the Orbbec running) → `ros2 service call /door_detection_srv tinker_vision_msgs_26/srv/DoorDetection "{camera: 'orbbec'}"` facing a near wall (expect `is_open: 0`) vs an open doorway (expect `is_open: 1`). Confirm `/camera/depth/image_raw` is `16UC1` via `ros2 topic echo --field encoding /camera/depth/image_raw`.

## 7. Files touched
| file | change |
|---|---|
| `src/vision_util/vision_util/_door_logic.py` | **create** — pure numpy detection math |
| `src/vision_util/vision_util/door_detection.py` | **rewrite** — depth-image node using `_door_logic`; params; QoS; encoding guard; shutdown fix |
| `src/vision_util/test/test_door_detection.py` | **create** — unit tests for `_door_logic` |
| `DoorDetection.srv`, `BtNode_DoorDetection`, tk25_decision | **no change** |

Build with `./src/tk26_vision/scripts/build.sh --packages-select vision_util` (or `tkbuild tk26_vision --packages-select vision_util`); `--symlink-install` makes the Python edits live, but the entry-point shebang must point at `.venv-vision-main` (the build wrapper handles it).

## 8. Risks
- **Camera not level / not forward** — accepted by design (§1.3); if the head is tilted (pan-tilt), the "center = forward" assumption breaks. The inspection tree tucks to a fixed arm/nav pose before door detection, so the Orbbec head pose must be its level-forward home when the service is called (operational note).
- **Registered depth holes** — if `depth_registration` reprojection thins the center, `min_valid_px=50` may need lowering; it's a param.
- **Threshold vs park distance** — `open_threshold_m=1.5` assumes the robot parks < 1.5 m from the closed door; retune live via the param.
