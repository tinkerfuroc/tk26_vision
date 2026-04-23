"""ROS 2 node + FastAPI app for interactive calibration waypoint authoring.

Run:
    ros2 run pan_tilt calibrate_web --ros-args \\
        -p config:=$(ros2 pkg prefix pan_tilt)/share/pan_tilt/config/calibration.yaml \\
        -p bind:=127.0.0.1 -p port:=8765

Then open http://127.0.0.1:8765 in a browser.

The tool provides:
  - Live camera view with ChArUco detection overlay (tab 1).
  - xArm waypoint authoring: joint-angle input, safety envelope check against
    the current TF, "send to robot" (via xarm set_servo_angle), and draft
    waypoint lists (tab 2).
  - Pan-tilt jog controls for manual visibility checks from the current xArm
    pose (tab 2, same screen).

The full pan-tilt visibility grid sweep and the calibration runner (tabs 3-4
in the plan) are not in this pass — shipped in follow-up PRs.

Threading model: rclpy.spin runs in the main thread; uvicorn runs in a worker
thread. All FastAPI handlers touch the node through `node.lock`-protected
accessors to avoid races with the ROS callbacks.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState

from .calibration.aruco_detect import (
    BoardSpec,
    build_board,
    build_detector,
    detect_pose,
)
from .calibration.safety import SafetyEnvelope
from .calibration.utils import matrix_to_pose, pose_to_matrix


log = logging.getLogger("calib_web")


def _sanitize_for_json(obj):
    """Recursively replace non-finite floats with None so json.dumps succeeds."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ---- shared state -----------------------------------------------------------

@dataclass
class SharedState:
    have_camera: bool = False
    have_pt_state: bool = False
    have_tf: bool = False
    have_xarm_joints: bool = False

    pan_rad: float = 0.0
    tilt_rad: float = 0.0
    pt_feedback_ok: bool = False
    pt_connected: bool = False

    xarm_joint_names: list = field(default_factory=list)
    xarm_joint_positions: list = field(default_factory=list)

    t_base_ee: Optional[list] = None  # 4x4 as nested list

    last_detection_n_corners: int = 0
    last_detection_rms: float = float("inf")
    last_detection_ok: bool = False

    safety: dict = field(default_factory=lambda: SafetyEnvelope().to_dict())


# ---- node -------------------------------------------------------------------

class CalibWebNode(Node):
    def __init__(self):
        super().__init__("calib_web")

        self.declare_parameter("config", "")
        self.declare_parameter("bind", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("draft_yaml_out", "")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("pantilt_cmd_topic", "/pan_tilt_controller/cmd")
        self.declare_parameter("pantilt_state_topic", "/pan_tilt_controller/state")
        self.declare_parameter("xarm_service", "/xarm/set_servo_angle")
        self.declare_parameter("xarm_joint_state_topic", "/xarm/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "link_eef")
        self.declare_parameter("xarm_speed", 0.3)
        self.declare_parameter("xarm_acc", 0.3)
        self.declare_parameter("xarm_service_timeout_sec", 20.0)
        self.declare_parameter("pantilt_speed_raw", 120)
        self.declare_parameter("pantilt_accel_raw", 20)

        self.config_path: str = self.get_parameter("config").value or ""
        self.bind_host: str = self.get_parameter("bind").value
        self.bind_port: int = int(self.get_parameter("port").value)

        default_draft = ""
        if self.config_path:
            p = Path(self.config_path)
            default_draft = str(p.with_name(p.stem + ".draft.yaml"))
        self.draft_yaml_out = Path(self.get_parameter("draft_yaml_out").value or default_draft or "calibration.draft.yaml")

        self._board_spec, self._safety_env, self._loaded_cfg = _load_yaml_config(self.config_path)
        self._board = build_board(self._board_spec)
        self._detector = build_detector(self._board)

        self.bridge = CvBridge()
        self.state = SharedState(safety=self._safety_env.to_dict())
        self.lock = threading.Lock()

        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_stamp_ns: int = 0
        self._latest_K: Optional[np.ndarray] = None
        self._latest_D: Optional[np.ndarray] = None
        self._overlay_jpeg: Optional[bytes] = None
        self._overlay_lock = threading.Lock()

        # Draft waypoint state — authored in-browser, not persisted to disk
        # until the user explicitly hits Save.
        self._waypoints: dict = {
            "phase1_waypoints": list(self._loaded_cfg.get("phase1_waypoints", []) or []),
            "phase2_waypoints": list(self._loaded_cfg.get("phase2_waypoints", []) or []),
            "sanity_xarm_angles_rad": list(self._loaded_cfg.get("sanity_xarm_angles_rad", []) or []),
        }

        qos_sensor = QoSProfile(
            depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            Image, self.get_parameter("image_topic").value,
            self._on_image, qos_sensor,
        )
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._on_camera_info, qos_sensor,
        )
        self.create_subscription(
            PanTiltState, self.get_parameter("pantilt_state_topic").value,
            self._on_pt_state, 10,
        )
        self.create_subscription(
            JointState, self.get_parameter("xarm_joint_state_topic").value,
            self._on_xarm_joints, 10,
        )

        self._pt_pub = self.create_publisher(
            PanTiltCommand, self.get_parameter("pantilt_cmd_topic").value, 10,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._xarm_client = None
        try:
            from xarm_msgs.srv import MoveJoint  # type: ignore
            self._xarm_srv_type = MoveJoint
            self._xarm_client = self.create_client(
                MoveJoint, self.get_parameter("xarm_service").value,
            )
        except ImportError:
            self._xarm_srv_type = None
            self.get_logger().warn(
                "xarm_msgs not found; /api/xarm/move will return 503 until installed."
            )

        # Refresh overlay + TF at 10 Hz.
        self.create_timer(0.1, self._refresh_tick)

    # ---- subs ----------------------------------------------------------------

    def _on_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self.lock:
            self._latest_bgr = img
            self._latest_stamp_ns = stamp_ns
            self.state.have_camera = True

    def _on_camera_info(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        D = np.array(msg.d, dtype=float).flatten()
        with self.lock:
            self._latest_K = K
            self._latest_D = D

    def _on_pt_state(self, msg: PanTiltState):
        with self.lock:
            self.state.pan_rad = float(msg.pan_rad)
            self.state.tilt_rad = float(msg.tilt_rad)
            self.state.pt_feedback_ok = bool(msg.feedback_ok)
            self.state.pt_connected = bool(msg.connected)
            self.state.have_pt_state = True

    def _on_xarm_joints(self, msg: JointState):
        with self.lock:
            self.state.xarm_joint_names = list(msg.name)
            self.state.xarm_joint_positions = list(msg.position)
            self.state.have_xarm_joints = True

    # ---- periodic refresh ----------------------------------------------------

    def _refresh_tick(self):
        # TF lookup
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self.get_parameter("base_frame").value,
                self.get_parameter("ee_frame").value,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
            T = pose_to_matrix(
                [tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z],
                [tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w],
            )
            with self.lock:
                self.state.have_tf = True
                self.state.t_base_ee = T.tolist()
        except Exception:
            pass

        # Detection + overlay
        with self.lock:
            bgr = None if self._latest_bgr is None else self._latest_bgr.copy()
            K = None if self._latest_K is None else self._latest_K.copy()
            D = None if self._latest_D is None else self._latest_D.copy()

        if bgr is None:
            return

        if K is not None and D is not None:
            det = detect_pose(bgr, K, D, board=self._board, detector=self._detector)
            with self.lock:
                self.state.last_detection_n_corners = det.n_corners
                self.state.last_detection_rms = det.reprojection_rms_px if det.success else float("inf")
                self.state.last_detection_ok = det.success
        else:
            det = None

        overlay = self._draw_overlay(bgr, det)
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._overlay_lock:
                self._overlay_jpeg = buf.tobytes()

    def _draw_overlay(self, bgr, det) -> np.ndarray:
        # Downscale for bandwidth.
        h, w = bgr.shape[:2]
        target_w = 960
        if w > target_w:
            scale = target_w / w
            bgr = cv2.resize(bgr, (target_w, int(h * scale)))
            scale_used = scale
        else:
            scale_used = 1.0

        if det is not None and det.success:
            color = (0, 200, 0)
            label = f"corners={det.n_corners}  rms={det.reprojection_rms_px:.2f}px  OK"
        else:
            color = (40, 40, 220)
            corners = det.n_corners if det else 0
            label = f"corners={corners}  NO DETECTION"

        cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, 28), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # Project board outline if detection succeeded.
        if det is not None and det.success and self._latest_K is not None:
            w_m = self._board_spec.squares_x * self._board_spec.square_len_m
            h_m = self._board_spec.squares_y * self._board_spec.square_len_m
            obj = np.array([[0, 0, 0], [w_m, 0, 0], [w_m, h_m, 0], [0, h_m, 0]],
                           dtype=np.float32)
            rvec = cv2.Rodrigues(det.pose_optical[:3, :3])[0]
            tvec = det.pose_optical[:3, 3]
            pts, _ = cv2.projectPoints(obj, rvec, tvec, self._latest_K, self._latest_D)
            pts = (pts.reshape(-1, 2) * scale_used).astype(int)
            cv2.polylines(bgr, [pts], True, color, 2)
        return bgr

    # ---- public accessors (called from FastAPI threads) ----------------------

    def snapshot_state(self) -> dict:
        with self.lock:
            d = asdict(self.state)
        # JSON can't encode inf/nan; sanitize recursively before returning.
        return _sanitize_for_json(d)

    def get_jpeg(self) -> Optional[bytes]:
        with self._overlay_lock:
            return self._overlay_jpeg

    def validate_t_base_ee(self, T: np.ndarray) -> Optional[str]:
        return self._safety_env.validate(T)

    # ---- commands ------------------------------------------------------------

    def publish_pantilt(self, pan_rad: float, tilt_rad: float) -> None:
        msg = PanTiltCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = 0  # ABSOLUTE
        msg.pan_rad = float(pan_rad)
        msg.tilt_rad = float(tilt_rad)
        msg.speed_raw = int(self.get_parameter("pantilt_speed_raw").value)
        msg.accel_raw = int(self.get_parameter("pantilt_accel_raw").value)
        self._pt_pub.publish(msg)

    def call_xarm(self, angles_rad) -> tuple[bool, str]:
        if self._xarm_client is None:
            return False, "xarm_msgs not installed in this venv"
        if not self._xarm_client.wait_for_service(timeout_sec=1.5):
            return False, f"xarm service {self.get_parameter('xarm_service').value} unavailable"
        req = self._xarm_srv_type.Request()
        req.angles = [float(a) for a in angles_rad]
        req.speed = float(self.get_parameter("xarm_speed").value)
        req.acc = float(self.get_parameter("xarm_acc").value)
        req.mvtime = 0.0
        req.wait = True
        req.timeout = float(self.get_parameter("xarm_service_timeout_sec").value)

        fut = self._xarm_client.call_async(req)
        timeout = float(self.get_parameter("xarm_service_timeout_sec").value)
        t0 = time.monotonic()
        while not fut.done() and (time.monotonic() - t0) < timeout:
            time.sleep(0.05)
        if not fut.done():
            return False, "xarm service timed out"
        resp = fut.result()
        if resp.ret != 0:
            return False, f"xarm returned ret={resp.ret}: {resp.message}"
        return True, "ok"

    # ---- waypoint store ------------------------------------------------------

    def list_waypoints(self, phase: str) -> list:
        with self.lock:
            return list(self._waypoints.get(phase, []))

    def set_waypoints(self, phase: str, wps: list) -> None:
        with self.lock:
            self._waypoints[phase] = list(wps)

    def save_waypoints(self) -> Path:
        """Atomic rewrite of the draft YAML, preserving all non-waypoint fields.

        Returns the path written. Output structure mirrors the original
        calibration.yaml so the user can promote with a simple cp.
        """
        base = {k: v for k, v in self._loaded_cfg.items() if k != "__passthrough__"}
        with self.lock:
            for k, v in self._waypoints.items():
                # Convert nested tuples from YAML loader to plain lists.
                base[k] = [list(x) if isinstance(x, (list, tuple)) else x
                           for x in v] if isinstance(v, list) else v

        out = {"collector": base}
        passthrough = self._loaded_cfg.get("__passthrough__", {})
        if "safety_section" in passthrough:
            out["safety"] = passthrough["safety_section"]
        if "board_section" in passthrough:
            out["board"] = passthrough["board_section"]

        tmp = self.draft_yaml_out.with_suffix(self.draft_yaml_out.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(out, sort_keys=False))
        tmp.replace(self.draft_yaml_out)
        return self.draft_yaml_out


# ---- yaml loader ------------------------------------------------------------

def _load_yaml_config(path: str) -> tuple[BoardSpec, SafetyEnvelope, dict]:
    """Read calibration.yaml if provided. Returns (board_spec, safety_env, collector_cfg).

    The returned `collector_cfg` dict stashes the untouched `safety` and `board`
    YAML sections under `__passthrough__` so `save_waypoints` can preserve them.
    """
    board = BoardSpec()
    safety = SafetyEnvelope()
    coll_cfg: dict = {"__passthrough__": {}}

    if not path:
        return board, safety, coll_cfg

    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except FileNotFoundError:
        return board, safety, coll_cfg

    if "collector" in data:
        coll_cfg.update(data["collector"])

    if "safety" in data:
        coll_cfg["__passthrough__"]["safety_section"] = data["safety"]
        for k, v in data["safety"].items():
            if hasattr(safety, k):
                setattr(safety, k, v)

    if "board" in data:
        coll_cfg["__passthrough__"]["board_section"] = data["board"]
        for k, v in data["board"].items():
            if k == "dict":
                try:
                    board.dict_id = getattr(cv2.aruco, v)
                except AttributeError:
                    pass
            elif hasattr(board, k):
                setattr(board, k, v)

    return board, safety, coll_cfg


# ---- FastAPI app ------------------------------------------------------------

def make_app(node: CalibWebNode, webui_dir: Path) -> FastAPI:
    app = FastAPI(title="pan_tilt calibrate_web", docs_url="/api/docs")

    # --- static UI ---------------------------------------------------------
    if webui_dir.exists():
        app.mount("/static", StaticFiles(directory=str(webui_dir)), name="static")

        @app.get("/")
        def root():
            return FileResponse(webui_dir / "index.html")
    else:
        @app.get("/")
        def root_missing():
            return JSONResponse(
                {"error": f"webui not found at {webui_dir}"}, status_code=500,
            )

    # --- state + frame ------------------------------------------------------
    @app.get("/api/state")
    def api_state():
        return node.snapshot_state()

    @app.get("/api/frame.jpg")
    def api_frame():
        buf = node.get_jpeg()
        if buf is None:
            raise HTTPException(404, "no camera frame yet")
        return Response(buf, media_type="image/jpeg", headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
        })

    # --- motion -------------------------------------------------------------
    @app.post("/api/xarm/preview")
    async def api_xarm_preview(req: dict):
        """Evaluate safety for a proposed EE pose (provided as 4x4 matrix).

        We do not compute FK server-side (no xarm kinematic library in this
        venv); preview is useful for hand-measured poses or a TF query.
        Clients that want joint-to-EE preview can jog the robot and re-read.
        """
        if "t_base_ee" not in req:
            raise HTTPException(400, "payload must include 't_base_ee' (4x4 list)")
        T = np.asarray(req["t_base_ee"], dtype=float)
        if T.shape != (4, 4):
            raise HTTPException(400, "t_base_ee must be 4x4")
        reason = node.validate_t_base_ee(T)
        return {"ok": reason is None, "reason": reason or "safe"}

    @app.post("/api/xarm/move")
    async def api_xarm_move(req: dict):
        angles = req.get("angles_rad")
        if not isinstance(angles, list) or not angles:
            raise HTTPException(400, "'angles_rad' must be a non-empty list of floats")
        loop = asyncio.get_event_loop()
        ok, msg = await loop.run_in_executor(None, node.call_xarm, angles)
        return {"ok": ok, "message": msg}

    @app.post("/api/pantilt/move")
    async def api_pantilt_move(req: dict):
        pan_deg = float(req.get("pan_deg", 0.0))
        tilt_deg = float(req.get("tilt_deg", 0.0))
        node.publish_pantilt(math.radians(pan_deg), math.radians(tilt_deg))
        return {"ok": True, "message": f"pan={pan_deg:+.1f} tilt={tilt_deg:+.1f} published"}

    # --- waypoints ----------------------------------------------------------
    VALID_PHASES = {"phase1_waypoints", "phase2_waypoints", "sanity_xarm_angles_rad"}

    @app.get("/api/waypoints")
    def api_waypoints_all():
        return {k: node.list_waypoints(k) for k in VALID_PHASES}

    # /save must be declared BEFORE /{phase} so FastAPI's first-match routing
    # doesn't funnel it into the phase handler (which requires a body).
    @app.post("/api/waypoints/save")
    async def api_waypoints_save():
        try:
            path = node.save_waypoints()
        except Exception as exc:
            raise HTTPException(500, f"save failed: {exc}")
        return {"ok": True, "path": str(path)}

    @app.get("/api/waypoints/{phase}")
    def api_waypoints_get(phase: str):
        if phase not in VALID_PHASES:
            raise HTTPException(404, f"unknown phase: {phase}")
        return {"phase": phase, "waypoints": node.list_waypoints(phase)}

    @app.post("/api/waypoints/{phase}")
    async def api_waypoints_set(phase: str, req: dict):
        if phase not in VALID_PHASES:
            raise HTTPException(404, f"unknown phase: {phase}")
        wps = req.get("waypoints")
        if not isinstance(wps, list):
            raise HTTPException(400, "'waypoints' must be a list")
        # Basic shape validation
        for i, wp in enumerate(wps):
            if phase == "sanity_xarm_angles_rad":
                # sanity is a single list of angles (not a list-of-lists).
                # For POST convenience accept either: a single list of floats,
                # OR a list-of-lists; last entry wins.
                if isinstance(wp, (int, float)):
                    break  # it's a flat list already
                if not isinstance(wp, list):
                    raise HTTPException(400, f"waypoint {i} must be a list of floats")
            else:
                if not isinstance(wp, list) or not all(isinstance(x, (int, float)) for x in wp):
                    raise HTTPException(400, f"waypoint {i} must be a list of floats (radians)")
        if phase == "sanity_xarm_angles_rad":
            # Flatten if the payload is a single-element list-of-lists.
            if wps and isinstance(wps[0], list):
                wps = wps[-1]  # take the last configured pose
        node.set_waypoints(phase, wps)
        return {"phase": phase, "waypoints": node.list_waypoints(phase)}

    # --- WebSocket (10 Hz state push) ---------------------------------------
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                payload = node.snapshot_state()
                await ws.send_text(json.dumps(payload))
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            log.warning("websocket error: %s", exc)

    return app


# ---- main -------------------------------------------------------------------

def _resolve_webui_dir() -> Path:
    """Locate the webui static assets: prefer the installed share/, fall back to src."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("pan_tilt"))
        candidate = share / "webui"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    # Source-tree fallback (running uninstalled).
    src = Path(__file__).resolve().parent.parent / "webui"
    return src


def main():
    rclpy.init()
    node = CalibWebNode()
    webui_dir = _resolve_webui_dir()
    node.get_logger().info(f"web UI static dir: {webui_dir}")
    app = make_app(node, webui_dir)

    import uvicorn  # local import so rclpy init fails fast on ROS issues
    config = uvicorn.Config(
        app, host=node.bind_host, port=node.bind_port,
        log_level="info", access_log=False, loop="asyncio",
    )
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve())

    server_thread = threading.Thread(target=_serve, daemon=True, name="uvicorn")
    server_thread.start()
    node.get_logger().info(
        f"calibrate_web listening on http://{node.bind_host}:{node.bind_port}"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
