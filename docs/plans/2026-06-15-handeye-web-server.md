# handeye_web Server (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the no-op `handeye_web.main()` stub with a real FastAPI + rclpy server that starts reliably (even with no camera/arm), serves an inline single-page UI, streams the live camera with a ChArUco overlay, and drives the eye-in-hand capture → solve → gate → promote loop.

**Architecture:** Three layers. (1) `web_support.py` — pure, ROS-free helpers + the inline HTML (fully unit-tested). (2) `HandeyeWebNode` in `handeye_web.py` — the rclpy data plane (camera/info subs, TF, ChArUco detector, `CaptureSession`, JointMove client) exposing thread-safe accessor methods. (3) `make_app(node)` + `main()` — the FastAPI HTTP plane (uvicorn worker thread + `rclpy.spin` on main), mirroring `pan_tilt/calib_web.py`'s proven threading.

**Tech Stack:** Python 3.10, rclpy, tf2_ros, cv_bridge, FastAPI/uvicorn (venv-only), OpenCV (`cv2.aruco`), numpy/scipy, reused `pan_tilt.calibration` (aruco_detect, safety) + `handeye_calib` solver/gates/apply.

**Spec:** `src/tk26_vision/docs/specs/2026-06-15-xarm-handeye-calibration-design.md` (§5 collection, §8 solver, §9 verification, §10 storage).

## Global Constraints

- Package: `src/tk26_vision/src/handeye_calib/`. Git repo is `src/tk26_vision` (branch `dev`). The workspace root is NOT a git repo — run git inside `src/tk26_vision`.
- **Concurrent committer present** (foundation_stereo, kimi_api, object_detection_generalist, vision_track have uncommitted changes). Commit ONLY the files each task names. Never `git add -A`/`.`, never `--amend`, never rebase.
- `import handeye_calib.handeye_web` and `import handeye_calib.web_support` MUST be ROS-free: rclpy/fastapi imports live INSIDE functions (`main`, `make_app`) or methods, never at module top.
- Do NOT modify `validate_pose_set` / `diff_payload` in `handeye_web.py` — their 3 tests in `test/test_web_helpers.py` must stay green.
- Do NOT add a webui directory or touch `setup.py`/`data_files` — the HTML is an inline string in `web_support.py`.
- Venv python for tests: `src/tk26_vision/.venv-vision-main/bin/python`. Pure tests (Task 1) need no sourcing. Node/app tests (Tasks 2–3) need `source src/tk26_vision/install/setup.bash` first (for rclpy/pan_tilt/tinker_arm_msgs) and `PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH"`.
- Frames contract: `T_base_eef = A_i` = TF `link_base`→`link_eef` (4×4). `T_cam_board = B_i` = ChArUco board pose in the color **optical** frame from PnP (cam←board). The solver returns `X = T_eef→color_optical`. A captured `Sample(T_base_eef, T_cam_board, obs_px, corner_idx)` where `obs_px`=charuco corner pixels and `corner_idx`=charuco corner IDs (which index directly into `handeye_model.board_corners()` — ordering was verified to match `cv2.aruco.CharucoBoard.getChessboardCorners()`).

---

## File structure

| File | Responsibility |
|---|---|
| `handeye_calib/web_support.py` (NEW) | Pure helpers: `tf_to_matrix`, `matrix_to_xyz_rpy`, `charuco_to_sample_arrays`, `gate_color`, `state_payload`, `solve_payload`, `placeholder_jpeg`, `encode_jpeg`, `draw_charuco_overlay`, `INDEX_HTML`. ROS-free. |
| `handeye_calib/handeye_web.py` (MODIFY) | Keep `validate_pose_set`/`diff_payload`. Add `HandeyeWebNode` (rclpy data plane) + `make_app(node)` + `main()`. |
| `test/test_web_support.py` (NEW) | Task 1 pure unit tests. |
| `test/test_web_node.py` (NEW) | Task 2 sourced node construction/method tests. |
| `test/test_web_app.py` (NEW) | Task 3 sourced TestClient endpoint tests. |

---

## Task 1: Pure web-support helpers

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/web_support.py`
- Test: `src/tk26_vision/src/handeye_calib/test/test_web_support.py`

**Interfaces:**
- Consumes: `handeye_calib.transforms` (none directly needed), `handeye_calib.handeye_solve.SolveResult` (for `solve_payload` typing only — built in tests via the dataclass).
- Produces (later tasks rely on these exact signatures):
  - `tf_to_matrix(translation_xyz: seq3, quaternion_xyzw: seq4) -> np.ndarray (4,4)`
  - `matrix_to_xyz_rpy(T) -> (list[float] xyz, list[float] rpy)` — URDF fixed-axis rpy via scipy `as_euler('xyz')`.
  - `charuco_to_sample_arrays(charuco_corners, charuco_ids) -> (np.ndarray (M,2) float, np.ndarray (M,) int)`
  - `gate_color(status: str) -> str`
  - `state_payload(camera_connected: bool, intrinsics_ok: bool, num_samples: int, last_detection: dict|None, status_msg: str) -> dict`
  - `solve_payload(res) -> dict` where `res` is a `SolveResult`
  - `placeholder_jpeg(text: str = "no camera") -> bytes`
  - `encode_jpeg(bgr: np.ndarray) -> bytes`
  - `draw_charuco_overlay(bgr: np.ndarray, corners_xy) -> np.ndarray`
  - `INDEX_HTML: str`

- [ ] **Step 1: Write the failing tests** at `test/test_web_support.py`:
```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from handeye_calib import web_support as ws


def test_tf_to_matrix_identity():
    T = ws.tf_to_matrix([0, 0, 0], [0, 0, 0, 1])
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_tf_to_matrix_known():
    q = R.from_euler('z', 90, degrees=True).as_quat()  # xyzw
    T = ws.tf_to_matrix([1, 2, 3], q)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], R.from_quat(q).as_matrix(), atol=1e-12)


def test_matrix_to_xyz_rpy_urdf_convention():
    # rpy must be the URDF fixed-axis convention: Rz(yaw)Ry(pitch)Rx(roll)
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [0.3, -0.7, 1.1]).as_matrix()
    T[:3, 3] = [0.06, -0.01, 0.02]
    xyz, rpy = ws.matrix_to_xyz_rpy(T)
    assert xyz == [0.06, -0.01, 0.02] or np.allclose(xyz, [0.06, -0.01, 0.02])
    Rr = (R.from_euler('z', rpy[2]).as_matrix() @ R.from_euler('y', rpy[1]).as_matrix()
          @ R.from_euler('x', rpy[0]).as_matrix())
    np.testing.assert_allclose(Rr, T[:3, :3], atol=1e-9)


def test_charuco_to_sample_arrays():
    corners = np.array([[[10., 20.]], [[30., 40.]], [[50., 60.]]])  # (3,1,2) cv2 shape
    ids = np.array([[5], [2], [9]])                                  # (3,1)
    px, idx = ws.charuco_to_sample_arrays(corners, ids)
    assert px.shape == (3, 2) and idx.shape == (3,)
    np.testing.assert_allclose(px, [[10, 20], [30, 40], [50, 60]])
    assert idx.tolist() == [5, 2, 9] and idx.dtype.kind == 'i'


def test_gate_color():
    assert ws.gate_color("PASS") == "#1a9850"
    assert ws.gate_color("WARN") == "#f59e0b"
    assert ws.gate_color("FAIL") == "#d73027"
    assert ws.gate_color("???") == "#888888"


def test_state_payload_keys():
    d = ws.state_payload(False, False, 0, None, "idle")
    assert set(d) == {"camera_connected", "intrinsics_ok", "num_samples",
                      "last_detection", "status_msg"}
    assert d["camera_connected"] is False and d["last_detection"] is None


def test_solve_payload_keys():
    from handeye_calib.handeye_solve import SolveResult
    res = SolveResult(X=np.eye(4), Tbb=np.eye(4),
                      train_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.002, "reproj_px": 0.3},
                      heldout_metrics={"trans_rmse_m": 0.002, "rot_rmse_rad": 0.004, "reproj_px": 0.5},
                      status="PASS", per_method=[])
    p = ws.solve_payload(res)
    assert p["status"] == "PASS"
    assert len(p["X_xyz"]) == 3 and len(p["X_rpy"]) == 3
    assert p["heldout_metrics"]["reproj_px"] == 0.5


def test_placeholder_jpeg_is_jpeg():
    b = ws.placeholder_jpeg("no camera")
    assert isinstance(b, (bytes, bytearray)) and bytes(b[:2]) == b"\xff\xd8"


def test_encode_jpeg_roundtrips_shape():
    import cv2
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :, 1] = 200
    b = ws.encode_jpeg(img)
    assert bytes(b[:2]) == b"\xff\xd8"
    dec = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    assert dec.shape == (48, 64, 3)


def test_draw_overlay_preserves_shape_and_handles_empty():
    img = np.zeros((48, 64, 3), np.uint8)
    out = ws.draw_charuco_overlay(img, np.array([[10.0, 20.0], [30.0, 40.0]]))
    assert out.shape == img.shape
    out2 = ws.draw_charuco_overlay(img, np.empty((0, 2)))
    assert out2.shape == img.shape


def test_index_html_is_nonempty_html():
    assert isinstance(ws.INDEX_HTML, str) and "<html" in ws.INDEX_HTML.lower()
    assert "/api/frame.jpg" in ws.INDEX_HTML  # the UI references the live frame
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /home/tinker/tk25_ws
PYTHONPATH=src/tk26_vision/src/handeye_calib src/tk26_vision/.venv-vision-main/bin/python -m pytest src/tk26_vision/src/handeye_calib/test/test_web_support.py -v
```
Expected: FAIL — `No module named 'handeye_calib.web_support'`.

- [ ] **Step 3: Implement `web_support.py`**
```python
"""Pure, ROS-free helpers + inline UI for handeye_web. No rclpy/fastapi here."""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def tf_to_matrix(translation_xyz, quaternion_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quaternion_xyzw, float)).as_matrix()
    T[:3, 3] = np.asarray(translation_xyz, float)
    return T


def matrix_to_xyz_rpy(T):
    T = np.asarray(T, float)
    xyz = T[:3, 3].tolist()
    rpy = R.from_matrix(T[:3, :3]).as_euler('xyz').tolist()  # URDF fixed-axis convention
    return xyz, rpy


def charuco_to_sample_arrays(charuco_corners, charuco_ids):
    px = np.asarray(charuco_corners, float).reshape(-1, 2)
    idx = np.asarray(charuco_ids).reshape(-1).astype(int)
    return px, idx


_GATE_COLORS = {"PASS": "#1a9850", "WARN": "#f59e0b", "FAIL": "#d73027"}


def gate_color(status):
    return _GATE_COLORS.get(status, "#888888")


def state_payload(camera_connected, intrinsics_ok, num_samples, last_detection, status_msg):
    return {
        "camera_connected": bool(camera_connected),
        "intrinsics_ok": bool(intrinsics_ok),
        "num_samples": int(num_samples),
        "last_detection": last_detection,
        "status_msg": status_msg,
    }


def solve_payload(res):
    xyz, rpy = matrix_to_xyz_rpy(res.X)
    return {
        "status": res.status,
        "X_xyz": xyz,
        "X_rpy": rpy,
        "heldout_metrics": res.heldout_metrics,
        "train_metrics": res.train_metrics,
    }


def encode_jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def placeholder_jpeg(text="no camera", size=(480, 640)):
    img = np.full((size[0], size[1], 3), 40, np.uint8)
    cv2.putText(img, text, (20, size[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (200, 200, 200), 2, cv2.LINE_AA)
    return encode_jpeg(img)


def draw_charuco_overlay(bgr, corners_xy):
    out = bgr.copy()
    for (x, y) in np.asarray(corners_xy, float).reshape(-1, 2):
        cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1, cv2.LINE_AA)
    return out


INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>handeye_web</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee;display:flex;gap:16px;padding:16px}
#left{flex:0 0 660px}#right{flex:1}img{width:640px;background:#000;border:1px solid #333}
button{background:#2563eb;color:#fff;border:0;padding:8px 12px;border-radius:6px;margin:4px 0;cursor:pointer}
textarea{width:100%;height:60px;background:#1b1b1b;color:#eee;border:1px solid #333}
pre{background:#1b1b1b;padding:8px;border-radius:6px;white-space:pre-wrap;max-height:40vh;overflow:auto}
#banner{font-size:20px;font-weight:700;padding:8px;border-radius:6px;text-align:center}
.row{margin:8px 0}</style></head><body>
<div id="left"><img id="cam" src="/api/frame.jpg"><div class="row" id="status">…</div>
<div class="row"><textarea id="joints" placeholder="7 joint values, comma-separated"></textarea>
<button onclick="move()">Move arm</button></div>
<div class="row"><button onclick="post('/api/capture')">Capture pose</button>
<button onclick="post('/api/solve')">Solve</button>
<button onclick="post('/api/promote')">Promote</button></div>
<div id="banner"></div></div>
<div id="right"><h3>Result</h3><pre id="out">—</pre></div>
<script>
const out=document.getElementById('out'),banner=document.getElementById('banner');
function show(o){out.textContent=JSON.stringify(o,null,2);
  if(o.status){banner.textContent=o.status;
    banner.style.background=o.status==='PASS'?'#1a9850':o.status==='WARN'?'#f59e0b':o.status==='FAIL'?'#d73027':'#444';}}
async function post(u){try{const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});show(await r.json());}catch(e){show({error:String(e)});}}
async function move(){const j=document.getElementById('joints').value.split(',').map(Number);
  const r=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({joints:j})});show(await r.json());}
async function poll(){try{const r=await fetch('/api/state');const s=await r.json();
  document.getElementById('status').textContent=`camera:${s.camera_connected} K:${s.intrinsics_ok} samples:${s.num_samples} — ${s.status_msg}`;}catch(e){}}
setInterval(()=>{document.getElementById('cam').src='/api/frame.jpg?t='+Date.now();},200);
setInterval(poll,1000);poll();
</script></body></html>"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/tinker/tk25_ws
PYTHONPATH=src/tk26_vision/src/handeye_calib src/tk26_vision/.venv-vision-main/bin/python -m pytest src/tk26_vision/src/handeye_calib/test/test_web_support.py -v
```
Expected: 11 passed.

- [ ] **Step 5: Commit**
```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/web_support.py src/handeye_calib/test/test_web_support.py
git commit -m "feat(handeye_calib): pure web-support helpers + inline UI for handeye_web

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: HandeyeWebNode (rclpy data plane)

**Files:**
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py` (add `HandeyeWebNode`; keep `validate_pose_set`/`diff_payload`; leave `main` as-is for now)
- Test: `src/tk26_vision/src/handeye_calib/test/test_web_node.py`

**Interfaces:**
- Consumes: `web_support` (Task 1) helpers; `pan_tilt.calibration.aruco_detect` (board/detector/pose — READ that file for exact APIs: `BoardSpec`, `build_board`, `build_detector`, `detect_pose`, `Detection`); `pan_tilt.calibration.safety.SafetyEnvelope` (READ for its constructor + validate method); `handeye_calib.gates` (`StabilityTracker`, plus `CaptureSession` from `handeye_calib.handeye_collect`); `handeye_calib.handeye_model.board_corners`; `handeye_calib.handeye_solve.solve`; `handeye_calib.apply_handeye`.
- Produces (Task 3 relies on these node methods, all returning plain JSON-able values, all thread-safe via `self.lock`):
  - `HandeyeWebNode()` — a `rclpy.node.Node`, attributes `bind_host: str`, `bind_port: int`, `lock: threading.Lock`.
  - `.get_state_dict() -> dict` (via `web_support.state_payload`)
  - `.latest_jpeg() -> bytes` (overlayed frame, or `placeholder_jpeg` when no frame)
  - `.do_move(joints: list[float]) -> dict` `{ok, reason}`
  - `.do_capture() -> dict` `{ok, reason, num_samples}`
  - `.do_solve() -> dict` (`web_support.solve_payload` fields + `ok`, or `{ok:False, reason}`)
  - `.do_promote() -> dict` `{ok, written_path|preview, diff}`

- [ ] **Step 1: READ the reference APIs first** (do not guess):
  - `src/tk26_vision/src/pan_tilt/pan_tilt/calib_web.py` — camera subscription + QoS, `cv_bridge` usage, `_downscale`, and how it builds the board/detector and runs detection per frame.
  - `src/tk26_vision/src/pan_tilt/pan_tilt/calibration/aruco_detect.py` — `build_board`, `build_detector`, `detect_pose`, and what `Detection` exposes (pose, reprojection, and whether it surfaces charuco corner pixels + IDs). If `Detection` does NOT surface per-corner pixels+IDs, call `cv2.aruco.CharucoDetector(board).detectBoard(gray)` directly to get `charucoCorners, charucoIds` and use `board.matchImagePoints(...)` + `cv2.solvePnP` for the pose (cam←board).
  - `src/tk26_vision/src/pan_tilt/pan_tilt/calibration/safety.py` — `SafetyEnvelope` constructor + the method that validates a joint pose.

- [ ] **Step 2: Write the failing test** at `test/test_web_node.py`:
```python
import rclpy
from handeye_calib.handeye_web import HandeyeWebNode


def setup_module(_):
    rclpy.init()


def teardown_module(_):
    if rclpy.ok():
        rclpy.shutdown()


def test_node_constructs_and_safe_defaults():
    node = HandeyeWebNode()
    try:
        st = node.get_state_dict()
        assert st["camera_connected"] is False        # no camera in this env
        assert st["num_samples"] == 0
        jpg = node.latest_jpeg()
        assert isinstance(jpg, (bytes, bytearray)) and bytes(jpg[:2]) == b"\xff\xd8"
        cap = node.do_capture()
        assert cap["ok"] is False                      # nothing to capture
        sol = node.do_solve()
        assert sol["ok"] is False and "sample" in sol["reason"].lower()
    finally:
        node.destroy_node()


def test_do_move_validates_joint_count():
    node = HandeyeWebNode()
    try:
        bad = node.do_move([0.0, 0.0, 0.0])            # wrong arity
        assert bad["ok"] is False
    finally:
        node.destroy_node()
```

- [ ] **Step 3: Run the test to verify it fails**

Run (sourced):
```bash
cd /home/tinker/tk25_ws
source src/tk26_vision/install/setup.bash
PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH" src/tk26_vision/.venv-vision-main/bin/python \
  -m pytest src/tk26_vision/src/handeye_calib/test/test_web_node.py -v
```
Expected: FAIL — `cannot import name 'HandeyeWebNode'`.

- [ ] **Step 4: Implement `HandeyeWebNode`** in `handeye_web.py` (append below the existing helpers). Structure (fill bodies using the APIs read in Step 1):
```python
# at module top: keep MIN_POSES, validate_pose_set, diff_payload unchanged.

def _make_node_class():
    """Lazy import so `import handeye_calib.handeye_web` stays ROS-free."""
    import threading
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.action import ActionClient
    from tf2_ros import Buffer, TransformListener
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CameraInfo
    from tinker_arm_msgs.action import JointMove
    from handeye_calib import web_support as ws
    from handeye_calib import handeye_model as hm
    from handeye_calib import handeye_solve as hs
    from handeye_calib import apply_handeye as ah
    from handeye_calib import gates
    from handeye_calib.handeye_collect import CaptureSession

    class HandeyeWebNode(Node):
        def __init__(self):
            super().__init__("handeye_web")
            # declare params: bind, port, robot_name, color_image_topic,
            # camera_info_topic, base_frame, eef_frame, squares_x, squares_y,
            # square_len_m, marker_len_m, aruco_dict, min_diversity_deg,
            # jointmove_action, mount_to_color_xyz, mount_to_color_rpy.
            self.lock = threading.Lock()
            self.bind_host = self.get_parameter_or_default("bind", "127.0.0.1")
            self.bind_port = int(self.get_parameter_or_default("port", 8766))
            self.bridge = CvBridge()
            self._frame = None      # latest BGR
            self._K = None          # 3x3
            self._last_det = None    # dict for state payload
            self.session = CaptureSession(min_diversity_deg=...)
            self.last_solve = None
            self.tf_buffer = Buffer(); TransformListener(self.tf_buffer, self)
            # build board+detector (aruco_detect), SafetyEnvelope, board_pts=hm.board_corners(...)
            # subscribe Image (qos_profile_sensor_data) -> self._on_image
            # subscribe CameraInfo -> self._on_info (cache K)
            self._jm = ActionClient(self, JointMove, self.get_parameter_or_default("jointmove_action", "/xarm/joint_move"))
            self.get_logger().info("handeye_web node ready")

        # _on_image: cv_bridge -> BGR; run detection; under lock store frame,
        #   last detection dict {corners:int, reproj_px:float}, and the raw
        #   charuco corners/ids/pose for capture.
        # _on_info: cache 3x3 K under lock.

        def get_state_dict(self):
            with self.lock:
                return ws.state_payload(self._frame is not None, self._K is not None,
                                        len(self.session.samples), self._last_det, "ok")

        def latest_jpeg(self):
            with self.lock:
                if self._frame is None:
                    return ws.placeholder_jpeg("no camera")
                frame, corners = self._frame, self._last_corners_xy  # store from _on_image
            return ws.encode_jpeg(ws.draw_charuco_overlay(frame, corners if corners is not None else []))

        def do_move(self, joints):
            if not isinstance(joints, (list, tuple)) or len(joints) != 7:
                return {"ok": False, "reason": "expected 7 joint values"}
            # SafetyEnvelope.validate(...); if invalid -> {ok:False, reason}
            # if not self._jm.server_is_ready(): wait_for_server(timeout 0.5); if absent -> {ok:False,"arm action server unavailable"}
            # send_goal_async(JointMove goal) -> {ok:True,"sent"}
            ...

        def do_capture(self):
            # under lock read frame, K, raw charuco corners/ids/pose
            # if no frame/K/detection -> {ok:False, reason}
            # lookup TF base_frame->eef_frame -> ws.tf_to_matrix -> T_base_eef
            # obs_px, corner_idx = ws.charuco_to_sample_arrays(corners, ids)
            # ok, reason = self.session.try_add(T_base_eef, T_cam_board, obs_px, corner_idx,
            #     n_corners=len(corner_idx), reproj_px=<det reproj>, area_frac=<bbox area / image area>)
            # NOTE settle gate (StabilityTracker) is a v1 TODO — single-frame capture for now.
            # return {ok, reason, num_samples: len(self.session.samples)}
            ...

        def do_solve(self):
            with self.lock:
                samples, K = list(self.session.samples), self._K
            if len(samples) < 6:
                return {"ok": False, "reason": f"need >=6 samples, have {len(samples)}"}
            if K is None:
                return {"ok": False, "reason": "no camera intrinsics"}
            res = hs.solve(samples, K, None, hm.board_corners(self._sx, self._sy, self._sq))
            self.last_solve = res
            return {"ok": True, **ws.solve_payload(res)}

        def do_promote(self):
            # if self.last_solve is None -> {ok:False,"run solve first"}
            # T_mount_color from mount_to_color params (default identity);
            # T_eef_mount = ah.compose_eef_to_mount(self.last_solve.X, T_mount_color)
            # d = ah.handeye_yaml_dict(T_eef_mount, self.last_solve.X, len(samples),
            #     self.last_solve.heldout_metrics, <date param/'unset'>, self._sq)
            # path = robots/<robot_name>/hand_eye.yaml; if robot_name+dir exist:
            #   ah.write_with_backup(path, yaml.safe_dump(d)); return {ok:True, written_path, diff}
            # else return {ok:True, preview:d, diff}
            ...

        def get_parameter_or_default(self, name, default):
            from rclpy.parameter import Parameter
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            return self.get_parameter(name).value

    return HandeyeWebNode


def __getattr__(name):
    # module-level lazy attribute: `from handeye_calib.handeye_web import HandeyeWebNode`
    if name == "HandeyeWebNode":
        return _make_node_class()
    raise AttributeError(name)
```
Implementation notes: `area_frac` = (charuco bbox area) / (image area); compute from `obs_px` extents. The `__getattr__` module hook keeps `import handeye_calib.handeye_web` ROS-free while letting `from handeye_calib.handeye_web import HandeyeWebNode` trigger the lazy class build. Verify ROS-free import is preserved:
```bash
PYTHONPATH=src/tk26_vision/src/handeye_calib src/tk26_vision/.venv-vision-main/bin/python -c "import handeye_calib.handeye_web, sys; print('rclpy' in sys.modules)"
```
Expected: `False`.

- [ ] **Step 5: Run the test to verify it passes**

Run (sourced, as Step 3). Expected: 2 passed. Also confirm the ROS-free import prints `False` and the existing helper tests still pass:
```bash
PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH" src/tk26_vision/.venv-vision-main/bin/python \
  -m pytest src/tk26_vision/src/handeye_calib/test/test_web_helpers.py src/tk26_vision/src/handeye_calib/test/test_web_node.py -v
```
Expected: 5 passed (3 helper + 2 node).

- [ ] **Step 6: Commit**
```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/handeye_web.py src/handeye_calib/test/test_web_node.py
git commit -m "feat(handeye_calib): HandeyeWebNode data plane (camera/TF/detect/capture/solve/promote)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: FastAPI app + main() + live startup smoke

**Files:**
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py` (add `make_app`; implement `main`)
- Test: `src/tk26_vision/src/handeye_calib/test/test_web_app.py`

**Interfaces:**
- Consumes: `HandeyeWebNode` (Task 2) methods; `web_support.INDEX_HTML`.
- Produces: `make_app(node) -> fastapi.FastAPI`; a working `main()` that launches uvicorn (worker thread) + `rclpy.spin(node)`.

- [ ] **Step 1: Write the failing test** at `test/test_web_app.py` (uses Starlette `TestClient` — no network/uvicorn):
```python
import rclpy
from fastapi.testclient import TestClient
from handeye_calib.handeye_web import HandeyeWebNode, make_app


def setup_module(_):
    rclpy.init()


def teardown_module(_):
    if rclpy.ok():
        rclpy.shutdown()


def _client():
    node = HandeyeWebNode()
    return node, TestClient(make_app(node))


def test_index_served():
    node, c = _client()
    try:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "<html" in r.text.lower()
    finally:
        node.destroy_node()


def test_state_endpoint_no_hardware():
    node, c = _client()
    try:
        r = c.get("/api/state")
        assert r.status_code == 200
        assert r.json()["camera_connected"] is False
    finally:
        node.destroy_node()


def test_frame_endpoint_returns_jpeg_placeholder():
    node, c = _client()
    try:
        r = c.get("/api/frame.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"
    finally:
        node.destroy_node()


def test_action_endpoints_degrade_gracefully():
    node, c = _client()
    try:
        assert c.post("/api/capture", json={}).json()["ok"] is False
        assert c.post("/api/solve", json={}).json()["ok"] is False
        assert c.post("/api/move", json={"joints": [0, 0, 0]}).json()["ok"] is False
    finally:
        node.destroy_node()
```

- [ ] **Step 2: Run the test to verify it fails**

Run (sourced):
```bash
cd /home/tinker/tk25_ws
source src/tk26_vision/install/setup.bash
PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH" src/tk26_vision/.venv-vision-main/bin/python \
  -m pytest src/tk26_vision/src/handeye_calib/test/test_web_app.py -v
```
Expected: FAIL — `cannot import name 'make_app'`.

- [ ] **Step 3: Implement `make_app` + `main`** in `handeye_web.py`:
```python
def make_app(node):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, Response, JSONResponse
    from handeye_calib import web_support as ws

    app = FastAPI(title="handeye_web", docs_url="/api/docs")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return ws.INDEX_HTML

    @app.get("/api/state")
    def state():
        return JSONResponse(node.get_state_dict())

    @app.get("/api/frame.jpg")
    def frame():
        return Response(content=node.latest_jpeg(), media_type="image/jpeg")

    @app.post("/api/move")
    async def move(request: Request):
        body = await request.json()
        return JSONResponse(node.do_move(body.get("joints")))

    @app.post("/api/capture")
    def capture():
        return JSONResponse(node.do_capture())

    @app.post("/api/solve")
    def solve():
        return JSONResponse(node.do_solve())

    @app.post("/api/promote")
    def promote():
        return JSONResponse(node.do_promote())

    return app


def main():
    import os
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    import rclpy
    rclpy.init()
    node = HandeyeWebNode()
    app = make_app(node)
    import uvicorn, threading, asyncio
    config = uvicorn.Config(app, host=node.bind_host, port=node.bind_port,
                            log_level="info", access_log=False, loop="asyncio")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve())

    t = threading.Thread(target=_serve, daemon=True, name="uvicorn")
    t.start()
    node.get_logger().info(f"handeye_web listening on http://{node.bind_host}:{node.bind_port}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        t.join(timeout=2.0)
```
Replace the old stub `main()` entirely. `HandeyeWebNode` must be importable at module scope for the test's `from handeye_calib.handeye_web import HandeyeWebNode, make_app` — keep the `__getattr__` lazy hook from Task 2 (it resolves `HandeyeWebNode`); `make_app`/`main` are real module-level defs (their ROS/fastapi imports are inside the bodies).

- [ ] **Step 4: Run the test to verify it passes**

Run (sourced, as Step 2). Expected: 4 passed.

- [ ] **Step 5: Live startup smoke (the user's core acceptance — server starts WITHOUT hardware)**

Run:
```bash
cd /home/tinker/tk25_ws
source src/tk26_vision/install/setup.bash
ros2 run handeye_calib handeye_web --ros-args -p bind:=127.0.0.1 -p port:=8791 &
SRV=$!
sleep 6
curl -fsS -o /dev/null -w "GET / -> HTTP %{http_code}\n" http://127.0.0.1:8791/
curl -fsS http://127.0.0.1:8791/api/state; echo
curl -fsS -o /dev/null -w "GET /api/frame.jpg -> HTTP %{http_code}\n" http://127.0.0.1:8791/api/frame.jpg
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
```
Expected: `GET / -> HTTP 200`, a JSON state line with `"camera_connected": false`, and `GET /api/frame.jpg -> HTTP 200`. (If `ros2 run` doesn't pick up the new code, rebuild: `tkbuild tk26_vision --packages-select handeye_calib`.)

- [ ] **Step 6: Full handeye_calib suite still green**

Run (sourced):
```bash
PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH" src/tk26_vision/.venv-vision-main/bin/python \
  -m pytest src/tk26_vision/src/handeye_calib/test/ -q
```
Expected: all pass (prior 38 + web_support 11 + web_node 2 + web_app 4 = 55).

- [ ] **Step 7: Commit**
```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/handeye_web.py src/handeye_calib/test/test_web_app.py
git commit -m "feat(handeye_calib): handeye_web FastAPI app + main() — server starts + serves UI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** §5 collection (camera + detect + capture gates) → Task 2 `do_capture` (reuses CaptureSession quality+diversity gates; settle gate explicitly deferred as a v1 TODO). §8 solver → Task 2 `do_solve` (reuses `handeye_solve.solve`). §9 verification → gate banner in UI (Task 1 HTML) + `solve_payload` status; live predicted-corner overlay is a follow-up. §10 storage → Task 2 `do_promote` (yaml write + diff preview; URDF patch deferred — `compose_eef_to_mount` is wired but the URDF mount-joint apply is a follow-up, noted). The "it starts" acceptance → Task 3 Step 5 live smoke.
- **Deferred (v1 scope, called out):** settle/StabilityTracker gate in capture, live predicted-corner verification overlay, subprocess solve runner + WS log streaming, on-disk sessions, URDF mount-joint promote, pose-set authoring beyond a raw joints box. These are additive; none block "the server starts and runs the core loop."
- **Placeholder scan:** Task 1 is fully concrete. Tasks 2–3 give complete tests + the app/main code; the node-method BODIES are structured skeletons that require reading 3 named reference files for existing-codebase APIs (aruco_detect/safety/calib_web) — this is reuse of established patterns, not inventable from the plan alone, and is explicitly instructed in Task 2 Step 1.
- **Type consistency:** `Sample(T_base_eef, T_cam_board, obs_px, corner_idx)` used in `do_capture`; `state_payload`/`solve_payload` keys match the Task 1 tests and the UI's `s.camera_connected`/`o.status` reads; `SolveResult` fields (`X`, `status`, `heldout_metrics`, `train_metrics`) consistent across `do_solve`/`solve_payload`/tests; node methods `get_state_dict`/`latest_jpeg`/`do_move`/`do_capture`/`do_solve`/`do_promote` are the same names consumed by `make_app`.
