"""ROS 2 node + FastAPI app for interactive calibration waypoint authoring.

Run:
    # `config` defaults to the per-robot calibration.yaml resolved via
    # tinker_robot_config (requires $ROBOT_NAME). Override with -p config:=…
    # to point at a custom file.
    ROBOT_NAME=tinker2 ros2 run pan_tilt calibrate_web --ros-args \\
        -p bind:=127.0.0.1 -p port:=8765

Then open http://127.0.0.1:8765 in a browser.

The tool provides:
  - Live camera view with ChArUco detection overlay (tab 1).
  - xArm waypoint authoring: joint-angle input, safety envelope check against
    the current TF, "send to robot" (via tinker_arm_msgs JointMove action),
    and draft waypoint lists (tab 2). A secondary panel accepts a Cartesian
    target pose and relays it via tinker_arm_msgs CartesianMove.
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
import difflib
import io
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
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
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState
from tinker_vision_msgs_26.srv import SetZero

try:
    # Optional at import time so unit tests / tooling can load this module
    # without the workspace fully installed. Resolution failures fall back
    # to an empty default — operators must then pass `-p config:=…`.
    from tinker_robot_config import resolver as _trc_resolver
except ImportError:  # pragma: no cover
    _trc_resolver = None

from .calibration.aruco_detect import (
    BoardSpec,
    build_board,
    build_detector,
    detect_pose,
)
from .calibration.run_calibration import GATES as _RC_GATES
from .calibration.safety import SafetyEnvelope
from .calibration.urdf_targets import list_targets as list_urdf_targets
from .calibration.yaml_targets import list_yaml_targets
from .calibration import apply_to_urdf as _apply_to_urdf_mod
from .calibration.utils import matrix_to_pose, pose_to_matrix
from .calibration.waypoint_predict import (
    chain_predictors,
    pantilt_grid_predictor,
    replay_predictor,
)
from .calibration.waypoint_prune import Predicted, prune_waypoints


log = logging.getLogger("calib_web")


def _default_calib_config_path() -> str:
    """Resolve calibration.yaml via tinker_robot_config; fallback to legacy.

    Returns the absolute install-share path for the per-robot
    ``pan_tilt/calibration.yaml`` (the install symlink chain lands at
    ``tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/calibration.yaml``).
    Returns ``''`` if the resolver isn't available or fails — operators must
    then pass ``-p config:=…`` explicitly. Operator-supplied non-empty values
    are always respected over this default.
    """
    if _trc_resolver is None:
        return ''
    try:
        return str(_trc_resolver.load().path('pan_tilt/calibration.yaml'))
    except Exception:  # pragma: no cover - resolver error path
        # ResolverError, missing ROBOT_NAME, or missing file — operator
        # can still pass -p config:= to override.
        return ''


# Gate thresholds: single source of truth lives in run_calibration.GATES; here
# we just adapt the (filename, key, threshold, label, unit) tuple into the
# (filename, key, threshold, unit, label) shape the web UI table consumes,
# turning the validate-CLI's terse "< 3 mm trans" into a longer human label.
_RC_LABELS = {
    "rms_px":            "intrinsic reprojection RMS",
    "trans_rmse_m":      "hand-eye translation RMSE",
    "rot_rmse_rad":      "hand-eye rotation RMSE",
    "val_trans_rmse_m":  "chain held-out translation RMSE",
    "val_rot_rmse_rad":  "chain held-out rotation RMSE",
}
CALIB_GATES = [
    (fname, key, thresh, unit, _RC_LABELS.get(key, key))
    for fname, key, thresh, _label, unit in _RC_GATES
]


class CalibrateRunner:
    """Spawns post-collection calibration subprocesses and fans out their
    stdout to WebSocket subscribers.

    Only run_calibration subcommands and apply_to_urdf are intended to flow
    through here; the runner is deliberately file-I/O-only so it can never
    kick the physical robot. Collection (which owns pan-tilt + xArm) stays
    a terminal invocation.
    """

    def __init__(self, calib_sessions_dir: Path):
        self.sessions_dir = Path(calib_sessions_dir).expanduser().resolve()
        # run_id -> {"proc": asyncio.subprocess.Process, "session": str, "cmd": list[str], "started": float}
        self._active: dict[str, dict] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    # ---- pub/sub -----------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _broadcast(self, event: dict) -> None:
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow subscriber — drop this frame silently rather than stall
                # the subprocess. The log pane will have a visible gap but
                # the terminal exit frame is small and usually gets through.
                pass
            except RuntimeError:
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    # ---- session dir -------------------------------------------------------

    def session_path(self, name: str) -> Path:
        # Reject absolute paths / parent-traversal; sessions must live inside
        # sessions_dir. `.` and `..` segments get rejected so an attacker-ish
        # HTTP body can't pivot off the sandbox.
        if not name or "/" in name or name in (".", "..") or name.startswith("."):
            raise ValueError(f"invalid session name: {name!r}")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        return (self.sessions_dir / name).resolve()

    def list_sessions(self) -> list[dict]:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for entry in sorted(self.sessions_dir.iterdir()):
            if not entry.is_dir():
                continue
            files = sorted(p.name for p in entry.iterdir() if p.is_file())
            out.append({
                "name": entry.name,
                "path": str(entry),
                "files": files,
                "mtime": entry.stat().st_mtime,
            })
        return out

    def create_session(self, name: str) -> Path:
        path = self.session_path(name)
        path.mkdir(parents=True, exist_ok=False)
        return path

    # ---- subprocess spawn --------------------------------------------------

    async def spawn(self, session: str, argv: list[str], *, label: str) -> dict:
        """Run an arbitrary argv in the session directory with stdout fanned
        out to subscribers. Caller builds argv (so we can host both
        `run_calibration` and `calibrate_collect` without special-casing).
        """
        session_path = self.session_path(session)
        if not session_path.exists():
            raise FileNotFoundError(f"session {session!r} does not exist")
        run_id = uuid.uuid4().hex[:12]
        async with self._lock:
            if any(h["session"] == session for h in self._active.values()):
                raise RuntimeError(f"another run is active on session {session!r}")
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(session_path),
                # New process group so cancel() can signal ros2-run's
                # grandchildren (calibrate_collect) along with the wrapper.
                start_new_session=True,
            )
            self._active[run_id] = {"proc": proc, "session": session}
        asyncio.create_task(self._pump(run_id, proc))
        self._broadcast({
            "type": "start", "run_id": run_id, "session": session,
            "label": label, "argv": argv, "pid": proc.pid,
        })
        return {"run_id": run_id, "pid": proc.pid}

    async def _pump(self, run_id: str, proc: asyncio.subprocess.Process) -> None:
        try:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                self._broadcast({
                    "type": "log", "run_id": run_id,
                    "line": line.decode("utf-8", errors="replace").rstrip("\n"),
                })
            code = await proc.wait()
            self._broadcast({"type": "exit", "run_id": run_id, "code": int(code)})
        except Exception as exc:
            self._broadcast({"type": "exit", "run_id": run_id, "code": -1, "error": str(exc)})
        finally:
            self._active.pop(run_id, None)

    async def cancel(self, run_id: str) -> bool:
        handle = self._active.get(run_id)
        if handle is None:
            return False
        proc = handle["proc"]
        # `ros2 run` spawns calibrate_collect as a grandchild; SIGTERM to the
        # wrapper alone leaks the grandchild. Signal the whole process group
        # (start_new_session=True at spawn means pgid == proc.pid).
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        return True

    # ---- apply_to_urdf diff (synchronous, short-running) -------------------
    #
    # Renders unified diffs for BOTH the xacro and pan_tilt.yaml targets in
    # one shot — runtime offsets must be applied in lockstep with the URDF,
    # and the operator should see both changes before clicking Apply. We
    # call the patcher functions in-process (no subprocess) for deterministic
    # error handling and to avoid stdout parsing.

    async def urdf_diff(self, session: str, results_file: str, xacro_path: str) -> dict:
        session_path = self.session_path(session)
        results_path = session_path / results_file
        if not results_path.is_file():
            raise FileNotFoundError(
                f"results file {results_file!r} not found in session {session!r}"
            )

        params = _apply_to_urdf_mod._load_params(results_path)
        import numpy as _np
        t_a = _np.asarray(params["t_a"], dtype=float)
        t_b_trans = _np.asarray(params["t_b_trans"], dtype=float)
        t_b_rotvec = _np.asarray(params.get("t_b_rotvec", [0, 0, 0]), dtype=float)
        pan_offset_rad = float(params.get("theta_p_offset_rad", 0.0))
        tilt_offset_rad = float(params.get("theta_t_offset_rad", 0.0))

        xacro = Path(xacro_path)
        if not xacro.is_file():
            raise FileNotFoundError(f"xacro {xacro_path!r} not present")
        original_xacro = xacro.read_text()
        try:
            patched_xacro = _apply_to_urdf_mod._patched_xacro(
                original_xacro, t_a, t_b_trans, t_b_rotvec,
                allow_flipped_camera=True,  # diff mode — show what would land
            )
        except _apply_to_urdf_mod.CalibrationApplyError as exc:
            return {"diff": "", "yaml_diff": "", "error": str(exc)}

        urdf_diff_text = "".join(difflib.unified_diff(
            original_xacro.splitlines(keepends=True),
            patched_xacro.splitlines(keepends=True),
            fromfile=str(xacro),
            tofile=str(xacro) + " (calibrated)",
        ))

        yaml_diff_text = ""
        yaml_targets = [t for t in list_yaml_targets() if t.exists]
        if yaml_targets:
            yaml_path = Path(yaml_targets[0].path)
            original_yaml = yaml_path.read_text()
            try:
                patched_yaml = _apply_to_urdf_mod._patch_yaml_offsets(
                    original_yaml, pan_offset_rad, tilt_offset_rad,
                )
            except _apply_to_urdf_mod.CalibrationApplyError as exc:
                yaml_diff_text = f"# YAML patch error: {exc}\n"
            else:
                yaml_diff_text = "".join(difflib.unified_diff(
                    original_yaml.splitlines(keepends=True),
                    patched_yaml.splitlines(keepends=True),
                    fromfile=str(yaml_path),
                    tofile=str(yaml_path) + " (calibrated)",
                ))

        return {
            "diff": urdf_diff_text,
            "yaml_diff": yaml_diff_text,
            "yaml_path": str(yaml_targets[0].path) if yaml_targets else None,
        }

    # ---- apply_to_urdf write (atomic in-place patch with backup) -----------

    async def urdf_apply(self, session: str, results_file: str, xacro_path: str) -> dict:
        """Patch ``xacro_path`` AND `pan_tilt.yaml` from ``<session>/<results_file>``,
        replace in place, leave timestamped backups for both.

        ``xacro_path`` must match one of the entries in ``list_urdf_targets()``.
        That allowlist is the only thing standing between an HTTP body and
        an arbitrary file write, so the check is mandatory here. The YAML
        path is server-side discovered via ``list_yaml_targets()`` and is
        never accepted from the request body.

        Idempotent: when either file already matches the calibration, no
        backup for that file is written.
        """
        target = next(
            (t for t in list_urdf_targets() if t.path == xacro_path),
            None,
        )
        if target is None:
            raise ValueError(
                f"xacro_path {xacro_path!r} is not on the URDF target allowlist"
            )
        if not target.exists:
            raise FileNotFoundError(
                f"target xacro {xacro_path!r} not present in this overlay "
                f"(package {target.build_package} not built?)"
            )

        session_path = self.session_path(session)
        results_path = session_path / results_file
        if not results_path.is_file():
            raise FileNotFoundError(
                f"results file {results_file!r} not found in session {session!r}"
            )

        params = _apply_to_urdf_mod._load_params(results_path)
        import numpy as _np
        t_a = _np.asarray(params["t_a"], dtype=float)
        t_b_trans = _np.asarray(params["t_b_trans"], dtype=float)
        t_b_rotvec = _np.asarray(params.get("t_b_rotvec", [0, 0, 0]), dtype=float)
        pan_offset_rad = float(params.get("theta_p_offset_rad", 0.0))
        tilt_offset_rad = float(params.get("theta_t_offset_rad", 0.0))

        xacro = Path(xacro_path)
        original_xacro = xacro.read_text()
        try:
            patched_xacro = _apply_to_urdf_mod._patched_xacro(
                original_xacro, t_a, t_b_trans, t_b_rotvec,
                allow_flipped_camera=True,
            )
        except _apply_to_urdf_mod.CalibrationApplyError as exc:
            raise RuntimeError(str(exc)) from exc

        yaml_targets = [t for t in list_yaml_targets() if t.exists]
        yaml_path = Path(yaml_targets[0].path) if yaml_targets else None

        try:
            atomic = _apply_to_urdf_mod._atomic_write_pair(
                xacro, patched_xacro,
                yaml_path, pan_offset_rad, tilt_offset_rad,
            )
        except _apply_to_urdf_mod.CalibrationApplyError as exc:
            raise RuntimeError(str(exc)) from exc

        # Diff previews for the UI's success card. Compare current file
        # contents (post-replace) against the .old-<ts> backups.
        urdf_diff_preview = ""
        if atomic["xacro_applied"] and atomic["xacro_backup_path"]:
            urdf_diff_preview = "".join(list(difflib.unified_diff(
                Path(atomic["xacro_backup_path"]).read_text().splitlines(keepends=True),
                xacro.read_text().splitlines(keepends=True),
                fromfile=str(xacro),
                tofile=str(xacro) + " (calibrated)",
            ))[:24])

        yaml_diff_preview = ""
        if atomic["yaml_applied"] and atomic["yaml_backup_path"] and yaml_path is not None:
            yaml_diff_preview = "".join(list(difflib.unified_diff(
                Path(atomic["yaml_backup_path"]).read_text().splitlines(keepends=True),
                yaml_path.read_text().splitlines(keepends=True),
                fromfile=str(yaml_path),
                tofile=str(yaml_path) + " (calibrated)",
            ))[:12])

        applied_anything = atomic["xacro_applied"] or atomic["yaml_applied"]
        return {
            "applied": applied_anything,
            "reason": (
                None if applied_anything else
                "no change — URDF and YAML already match calibration"
            ),
            "build_package": target.build_package,
            "build_command": target.build_command,
            "workspace_hint": target.workspace_hint,
            # URDF surface (back-compat with old client code).
            "backup_path": atomic["xacro_backup_path"],
            "diff_preview": urdf_diff_preview,
            # YAML surface (new).
            "yaml_path": atomic["yaml_path"],
            "yaml_applied": atomic["yaml_applied"],
            "yaml_backup_path": atomic["yaml_backup_path"],
            "yaml_diff_preview": yaml_diff_preview,
            "pan_offset_rad": pan_offset_rad,
            "tilt_offset_rad": tilt_offset_rad,
        }


def _empty_pointcloud(frame_id: str = "base_link"):
    """Zero-point but well-formed PointCloud2 used when no live depth cloud is available.

    The tinker_arm_msgs action server (pick_and_place/pc_proc) unconditionally
    runs `pcl::fromROSMsg` and (if frame_id != "base_link") a tf2
    lookupTransform to base_link on `env_points`. So a default-constructed
    PointCloud2 fails two ways: missing x/y/z fields (pcl warns / can't read)
    and empty frame_id (tf2 aborts with "source_frame cannot be empty"). We
    fix both by declaring x/y/z float32 fields with zero points and pinning
    frame_id to base_link so the server short-circuits the transform step.
    """
    from sensor_msgs.msg import PointCloud2, PointField
    pc = PointCloud2()
    pc.header.frame_id = frame_id
    pc.height = 1
    pc.width = 0
    pc.is_bigendian = False
    pc.is_dense = True
    pc.point_step = 12  # 3 × float32
    pc.row_step = 0
    pc.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    pc.data = b""
    return pc


def _sanitize_for_json(obj):
    """Recursively replace non-finite floats with None so json.dumps succeeds."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _downscale(bgr: np.ndarray, target_w: int = 960) -> np.ndarray:
    """Downsize a BGR image to target_w on the long edge (for bandwidth)."""
    h, w = bgr.shape[:2]
    if w <= target_w:
        return bgr
    scale = target_w / w
    return cv2.resize(bgr, (target_w, int(h * scale)))


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

    # Diagnostics for the browser UI
    image_topic: str = ""
    camera_info_topic: str = ""
    ros_domain_id: str = ""
    frame_count: int = 0
    frame_age_sec: float = float("inf")   # age of the most recent frame
    frame_hz: float = 0.0                 # smoothed over recent 10 frames
    available_image_topics: list = field(default_factory=list)

    safety: dict = field(default_factory=lambda: SafetyEnvelope().to_dict())
    # Pan-tilt calibration grid (firmware degrees). Exposed to the UI so the
    # Pan-Tilt tab can render the same corners the collector will sweep, and
    # operators don't have to cross-reference the yaml to jog into a grid cell.
    grid: dict = field(default_factory=lambda: {"pan_deg": [], "tilt_deg": []})


# ---- node -------------------------------------------------------------------

class CalibWebNode(Node):
    def __init__(self):
        super().__init__("calib_web")

        # Default resolves the per-robot calibration.yaml via
        # tinker_robot_config — install-share path (symlink chain) lands at
        # tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/.
        # Operators may still pass `-p config:=…` to point at a custom file.
        self.declare_parameter("config", _default_calib_config_path())
        self.declare_parameter("bind", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("draft_yaml_out", "")
        # Promote target. The runtime `config` param typically points at the
        # install-tree copy of calibration.yaml, which colcon overwrites on
        # every build -- promoting there silently loses operator edits at the
        # next rebuild. Default the promote target to the source-tree yaml so
        # it's actually persistent. Operators can override per-launch.
        self.declare_parameter("promote_yaml_out", "")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("pantilt_cmd_topic", "/pan_tilt_controller/cmd")
        self.declare_parameter("pantilt_state_topic", "/pan_tilt_controller/state")
        self.declare_parameter(
            "pantilt_set_zero_service", "/pan_tilt_controller/set_zero",
        )
        self.declare_parameter(
            "pantilt_remap_service", "/pan_tilt_controller/remap_servo_ids",
        )
        # Arm motion uses tinker_arm_msgs actions (JointMove / CartesianMove).
        # Both actions are served by the pick_and_place GraspNode and drive
        # MoveIt under the hood, so motion is collision-checked using
        # `env_points` (optional PointCloud2, left empty for calibration).
        self.declare_parameter("joint_move_action", "joint_move_action")
        self.declare_parameter("cartesian_move_action", "cartesian_move_action")
        self.declare_parameter("xarm_joint_state_topic", "/xarm/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "link_eef")
        self.declare_parameter("arm_action_timeout_sec", 30.0)
        self.declare_parameter("pantilt_speed_raw", 120)
        self.declare_parameter("pantilt_accel_raw", 20)
        # Calibrate tab: parent dir for session subdirs. Matches calibrate_collect's
        # default `out_dir` so both halves point at the same sessions by default.
        self.declare_parameter("calib_sessions_dir", "calibration_data")

        self.config_path: str = self.get_parameter("config").value or ""
        self.bind_host: str = self.get_parameter("bind").value
        self.bind_port: int = int(self.get_parameter("port").value)
        self.calib_sessions_dir = Path(
            self.get_parameter("calib_sessions_dir").value or "calibration_data"
        ).expanduser().resolve()
        self.calib_runner = CalibrateRunner(self.calib_sessions_dir)

        default_draft = ""
        if self.config_path:
            p = Path(self.config_path)
            default_draft = str(p.with_name(p.stem + ".draft.yaml"))
        self.draft_yaml_out = Path(self.get_parameter("draft_yaml_out").value or default_draft or "calibration.draft.yaml")

        # Resolve the promote target. If the operator passed an explicit
        # promote_yaml_out, honor it. Otherwise try to walk the runtime
        # config_path back to its source-tree origin -- a typical install path
        # looks like ".../install/pan_tilt/share/pan_tilt/config/calibration.yaml"
        # and the matching source is ".../src/pan_tilt/config/calibration.yaml".
        # Fall back to config_path (with a warning at promote time) only if no
        # source-tree counterpart can be located.
        explicit_promote = self.get_parameter("promote_yaml_out").value
        if explicit_promote:
            self.promote_yaml_out: Optional[Path] = Path(explicit_promote).expanduser().resolve()
        else:
            self.promote_yaml_out = _resolve_source_tree_yaml(self.config_path)

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
        self._raw_jpeg: Optional[bytes] = None
        self._overlay_lock = threading.Lock()
        # Rolling buffer of frame arrival times (monotonic seconds) for Hz estimate.
        self._frame_times: list = []
        self._last_frame_monotonic: float = 0.0

        self.state.image_topic = self.get_parameter("image_topic").value
        self.state.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.state.ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0 (default)")
        self._image_sub = None
        self._camera_info_sub = None
        self._sensor_qos = QoSProfile(
            depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Draft waypoint state — authored in-browser, persisted to disk only
        # when the user clicks Save. On startup we prefer whatever's in the
        # draft YAML (if it exists) over the base config, so a reloaded or
        # crashed session recovers the operator's in-progress lists instead
        # of reverting to whatever was checked into git.
        self._waypoints: dict = {
            "phase1_waypoints": list(self._loaded_cfg.get("phase1_waypoints", []) or []),
            "phase1_waypoints_custom": list(self._loaded_cfg.get("phase1_waypoints_custom", []) or []),
            "phase2_waypoints": list(self._loaded_cfg.get("phase2_waypoints", []) or []),
            "sanity_xarm_angles_rad": list(self._loaded_cfg.get("sanity_xarm_angles_rad", []) or []),
        }
        self._resume_from_draft()

        # Expose the configured pan/tilt grid so the UI can render grid-matched
        # jog presets and a cheat-sheet. Refreshed on each yaml reload so the
        # Pan-Tilt Jog tab tracks operator yaml edits + prune-overwrite output.
        with self.lock:
            self._refresh_state_grid_locked()

        self._subscribe_camera(
            self.state.image_topic, self.state.camera_info_topic,
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
        self._set_zero_client = self.create_client(
            SetZero, self.get_parameter("pantilt_set_zero_service").value,
        )
        self._remap_client = self.create_client(
            Trigger, self.get_parameter("pantilt_remap_service").value,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._joint_move_client = None
        self._cartesian_move_client = None
        self._joint_move_type = None
        self._cartesian_move_type = None
        try:
            from rclpy.action import ActionClient
            from tinker_arm_msgs.action import CartesianMove, JointMove  # type: ignore
            self._joint_move_type = JointMove
            self._cartesian_move_type = CartesianMove
            self._joint_move_client = ActionClient(
                self, JointMove, self.get_parameter("joint_move_action").value,
            )
            self._cartesian_move_client = ActionClient(
                self, CartesianMove,
                self.get_parameter("cartesian_move_action").value,
            )
        except ImportError:
            self.get_logger().warn(
                "tinker_arm_msgs not found; /api/xarm/move* will return 503 "
                "until the tinker_arm_msgs package is built and sourced."
            )

        # Refresh overlay + TF at 10 Hz.
        self.create_timer(0.1, self._refresh_tick)
        # Rescan available topics every 2 s so the UI dropdown stays current.
        self.create_timer(2.0, self._scan_topics)

    # ---- subs ----------------------------------------------------------------

    def _on_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(
                f"cv_bridge conversion failed ({exc}); is the topic publishing bgr8?",
                throttle_duration_sec=5.0,
            )
            return
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        now = time.monotonic()
        with self.lock:
            self._latest_bgr = img
            self._latest_stamp_ns = stamp_ns
            self.state.have_camera = True
            self.state.frame_count += 1
            # Keep the last N=10 arrival times for a rolling Hz estimate.
            self._frame_times.append(now)
            if len(self._frame_times) > 10:
                self._frame_times.pop(0)
            if len(self._frame_times) >= 2:
                dt = self._frame_times[-1] - self._frame_times[0]
                self.state.frame_hz = (len(self._frame_times) - 1) / dt if dt > 0 else 0.0
        self._last_frame_monotonic = now

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

    # ---- camera retarget + topic discovery ---------------------------------

    def _subscribe_camera(self, image_topic: str, camera_info_topic: str):
        """Destroy any existing camera subscriptions and resubscribe."""
        if self._image_sub is not None:
            self.destroy_subscription(self._image_sub)
        if self._camera_info_sub is not None:
            self.destroy_subscription(self._camera_info_sub)

        self._image_sub = self.create_subscription(
            Image, image_topic, self._on_image, self._sensor_qos,
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._on_camera_info, self._sensor_qos,
        )

        # Reset the frame counters so the UI reflects the new subscription.
        with self.lock:
            self.state.image_topic = image_topic
            self.state.camera_info_topic = camera_info_topic
            self.state.frame_count = 0
            self.state.frame_hz = 0.0
            self.state.frame_age_sec = float("inf")
            self.state.have_camera = False
            self._latest_bgr = None
            self._latest_K = None
            self._latest_D = None
            self._frame_times = []
        self._last_frame_monotonic = 0.0
        with self._overlay_lock:
            self._raw_jpeg = None
            self._overlay_jpeg = None

        self.get_logger().info(
            f"subscribed to {image_topic} + {camera_info_topic}",
        )

    def _scan_topics(self):
        """Populate state.available_image_topics with every sensor_msgs/Image topic."""
        try:
            pairs = self.get_topic_names_and_types()
        except Exception:
            return
        image_topics = sorted(
            name for name, types in pairs if "sensor_msgs/msg/Image" in types
        )
        with self.lock:
            self.state.available_image_topics = image_topics

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
            # Update frame-age diagnostic every tick so the UI sees it go stale.
            if self._last_frame_monotonic > 0:
                self.state.frame_age_sec = time.monotonic() - self._last_frame_monotonic

        if bgr is None:
            return

        # Always encode a raw JPEG (downscaled) for the debug toggle, even if
        # we have no intrinsics yet — this is the fastest way to tell whether
        # the camera topic is really producing frames.
        raw_small = _downscale(bgr, 960)
        ok_r, buf_r = cv2.imencode(".jpg", raw_small, [cv2.IMWRITE_JPEG_QUALITY, 72])

        det = None
        if K is not None and D is not None:
            det = detect_pose(bgr, K, D, board=self._board, detector=self._detector)
            with self.lock:
                self.state.last_detection_n_corners = det.n_corners
                self.state.last_detection_rms = det.reprojection_rms_px if det.success else float("inf")
                self.state.last_detection_ok = det.success

        overlay = self._draw_overlay(bgr, det)
        ok_o, buf_o = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])

        with self._overlay_lock:
            if ok_r:
                self._raw_jpeg = buf_r.tobytes()
            if ok_o:
                self._overlay_jpeg = buf_o.tobytes()

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
            pts2d = pts.reshape(-1, 2) * scale_used
            # projectPoints returns NaN/inf when the pose places corners
            # behind the camera or at degenerate locations -- casting those
            # to int triggers RuntimeWarning and draws garbage.
            if np.all(np.isfinite(pts2d)):
                cv2.polylines(bgr, [pts2d.astype(int)], True, color, 2)
        return bgr

    # ---- public accessors (called from FastAPI threads) ----------------------

    def snapshot_state(self) -> dict:
        with self.lock:
            d = asdict(self.state)
        # JSON can't encode inf/nan; sanitize recursively before returning.
        return _sanitize_for_json(d)

    def get_jpeg(self, raw: bool = False) -> Optional[bytes]:
        with self._overlay_lock:
            return self._raw_jpeg if raw else self._overlay_jpeg

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

    def call_set_zero(self, axis: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
        """Trigger the controller's SetZero service. ``axis`` is ``"both"``,
        ``"pan"``, or ``"tilt"``. The controller forwards ``T:502`` to firmware
        for each selected motor; the **current physical pose becomes the new
        servo zero** until the next call.

        Called from FastAPI worker threads; rclpy.spin runs in the main thread
        and drives the future, mirroring the polling pattern in `_run_action`.
        """
        logger = self.get_logger()
        axis_map = {
            "both": SetZero.Request.BOTH,
            "pan":  SetZero.Request.PAN,
            "tilt": SetZero.Request.TILT,
        }
        if axis not in axis_map:
            logger.error(f"[set_zero] unsupported axis {axis!r}")
            return False, f"unsupported axis {axis!r} (expected 'both', 'pan', or 'tilt')"

        service_name = self.get_parameter("pantilt_set_zero_service").value
        logger.info(f"[set_zero] axis={axis!r} (enum={axis_map[axis]}) "
                    f"service={service_name!r}; checking availability…")

        # Spell out *why* the client thinks the service isn't there, so we can
        # distinguish "controller not running" from "wrong topic/namespace".
        if not self._set_zero_client.service_is_ready():
            logger.warn(f"[set_zero] client not ready; waiting up to 1.5 s "
                        f"for {service_name!r}")
        if not self._set_zero_client.wait_for_service(timeout_sec=1.5):
            try:
                discovered = [
                    f"{n} [{','.join(t)}]"
                    for n, t in self.get_service_names_and_types()
                    if any("SetZero" in tt for tt in t)
                ]
            except Exception:
                discovered = ["(service discovery failed)"]
            logger.error(
                f"[set_zero] service {service_name!r} NOT FOUND after 1.5 s. "
                f"SetZero services visible on the network: {discovered or '(none)'}"
            )
            return False, (
                f"service {service_name!r} unavailable. "
                f"Visible SetZero services: {discovered or '(none)'}. "
                f"Is pan_tilt_controller running? Check `ros2 service list | grep set_zero`."
            )

        req = SetZero.Request()
        req.axis = axis_map[axis]
        logger.info(f"[set_zero] sending request axis={req.axis} → {service_name}")
        fut = self._set_zero_client.call_async(req)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout_sec:
            time.sleep(0.05)
        elapsed = time.monotonic() - t0
        if not fut.done():
            logger.error(
                f"[set_zero] no response from {service_name!r} after "
                f"{elapsed:.2f}s (timeout={timeout_sec}s). The controller "
                f"accepted the request but never replied — check the "
                f"controller's stdout for serial errors."
            )
            return False, f"{service_name}: timed out after {timeout_sec:.0f}s"

        resp = fut.result()
        logger.info(
            f"[set_zero] response in {elapsed:.2f}s: "
            f"success={resp.success}, message={resp.message!r}"
        )
        return bool(resp.success), resp.message or ("ok" if resp.success else "failed")

    def call_remap_servo_ids(self, timeout_sec: float = 5.0) -> tuple[bool, str]:
        """Trigger the controller's `~/remap_servo_ids` service. Fires the
        firmware command `{'T':501,'raw':1,'new':2}` which renumbers the
        still-attached servo (id=1) to id=2 — the middle step of the
        zero-state wizard. The operator must have physically disconnected
        the motor that currently holds id=2 before this is called.
        """
        logger = self.get_logger()
        service_name = self.get_parameter("pantilt_remap_service").value
        logger.info(f"[remap] service={service_name!r}; checking availability…")

        if not self._remap_client.service_is_ready():
            logger.warn(f"[remap] client not ready; waiting up to 1.5 s for {service_name!r}")
        if not self._remap_client.wait_for_service(timeout_sec=1.5):
            try:
                discovered = [
                    f"{n} [{','.join(t)}]"
                    for n, t in self.get_service_names_and_types()
                    if any("Trigger" in tt for tt in t)
                ]
            except Exception:
                discovered = ["(service discovery failed)"]
            logger.error(
                f"[remap] service {service_name!r} NOT FOUND after 1.5 s. "
                f"Trigger services visible on the network: {discovered or '(none)'}"
            )
            return False, (
                f"service {service_name!r} unavailable. "
                f"Visible Trigger services: {discovered or '(none)'}. "
                f"Is pan_tilt_controller running with the new build? "
                f"Check `ros2 service list | grep remap_servo_ids`."
            )

        req = Trigger.Request()
        logger.info(f"[remap] sending Trigger request → {service_name}")
        fut = self._remap_client.call_async(req)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout_sec:
            time.sleep(0.05)
        elapsed = time.monotonic() - t0
        if not fut.done():
            logger.error(
                f"[remap] no response from {service_name!r} after "
                f"{elapsed:.2f}s (timeout={timeout_sec}s)."
            )
            return False, f"{service_name}: timed out after {timeout_sec:.0f}s"

        resp = fut.result()
        logger.info(
            f"[remap] response in {elapsed:.2f}s: "
            f"success={resp.success}, message={resp.message!r}"
        )
        return bool(resp.success), resp.message or ("ok" if resp.success else "failed")

    def call_joint_move(self, angles_rad, add_octomap: bool = False) -> tuple[bool, str]:
        """Send a JointMove action goal. angles_rad is padded/truncated to 7 floats
        (joint0..joint6) to match the tinker_arm_msgs action definition.

        `add_octomap` (default False) controls the server-side dynamic scene
        layer. For calibration we want a clean planner state — the EE is in
        known free space and the marker board / fixture should NOT be added
        as obstacles, since they'd make the planner reject the move.
        """
        if self._joint_move_client is None:
            return False, "tinker_arm_msgs not available on the Python path"

        action_name = self.get_parameter("joint_move_action").value
        if not self._joint_move_client.wait_for_server(timeout_sec=1.5):
            return False, f"action '{action_name}' unavailable"

        goal = self._joint_move_type.Goal()
        a = list(angles_rad) + [0.0] * max(0, 7 - len(angles_rad))
        goal.joint0 = float(a[0])
        goal.joint1 = float(a[1])
        goal.joint2 = float(a[2])
        goal.joint3 = float(a[3])
        goal.joint4 = float(a[4])
        goal.joint5 = float(a[5])
        goal.joint6 = float(a[6])
        goal.add_octomap = bool(add_octomap)

        return self._run_action(self._joint_move_client, goal, action_name)

    def call_cartesian_move(self, pose_dict: dict, env_points=None) -> tuple[bool, str]:
        """Send a CartesianMove action goal. pose_dict: {"translation": [x,y,z],
        "rotation": [qx,qy,qz,qw]} in base_link coordinates.
        """
        if self._cartesian_move_client is None:
            return False, "tinker_arm_msgs not available on the Python path"

        action_name = self.get_parameter("cartesian_move_action").value
        if not self._cartesian_move_client.wait_for_server(timeout_sec=1.5):
            return False, f"action '{action_name}' unavailable"

        from geometry_msgs.msg import Pose, Point, Quaternion
        try:
            t = pose_dict["translation"]
            r = pose_dict["rotation"]
            pose = Pose(
                position=Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                orientation=Quaternion(
                    x=float(r[0]), y=float(r[1]), z=float(r[2]), w=float(r[3]),
                ),
            )
        except (KeyError, IndexError, TypeError) as e:
            return False, f"bad pose payload: {e}"

        goal = self._cartesian_move_type.Goal()
        goal.target_pose = pose
        goal.env_points = env_points if env_points is not None else _empty_pointcloud()

        return self._run_action(self._cartesian_move_client, goal, action_name)

    def _run_action(self, client, goal, action_name: str) -> tuple[bool, str]:
        """Send a goal and block-wait for the result. rclpy.spin runs in the
        main thread so futures are driven independently of this helper."""
        timeout = float(self.get_parameter("arm_action_timeout_sec").value)
        send_fut = client.send_goal_async(goal)
        t0 = time.monotonic()
        while not send_fut.done() and time.monotonic() - t0 < 5.0:
            time.sleep(0.05)
        if not send_fut.done():
            return False, f"{action_name}: send_goal timed out"

        goal_handle = send_fut.result()
        if not goal_handle.accepted:
            return False, f"{action_name}: goal rejected"

        result_fut = goal_handle.get_result_async()
        while not result_fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.05)
        if not result_fut.done():
            # Attempt a cancel so the server doesn't keep executing.
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
            return False, f"{action_name}: result timed out after {timeout:.0f}s"

        wrapped = result_fut.result()
        result = getattr(wrapped, "result", wrapped)
        success = bool(getattr(result, "success", False))
        return success, "ok" if success else f"{action_name} returned success=False"

    # ---- waypoint store ------------------------------------------------------

    def list_waypoints(self, phase: str) -> list:
        with self.lock:
            return list(self._waypoints.get(phase, []))

    def set_waypoints(self, phase: str, wps: list) -> None:
        with self.lock:
            self._waypoints[phase] = list(wps)

    def dedupe_waypoints(self, eps_rad: float = 1e-3) -> dict:
        """Drop near-duplicate waypoints in place across all waypoint lists.

        Two waypoints are duplicates if every joint angle agrees within
        `eps_rad` (~0.06 deg, well below the xArm's joint repeatability).
        Order-preserving: first occurrence wins. List-of-floats phases
        (sanity_xarm_angles_rad) are skipped. Returns a per-phase dict of
        how many entries were removed.

        Operator-triggered only -- save no longer dedupes implicitly so
        intentional duplicates used as a self-consistency probe survive.
        """
        removed: dict = {}
        with self.lock:
            for phase, wps in self._waypoints.items():
                if not isinstance(wps, list) or not wps:
                    continue
                if not isinstance(wps[0], (list, tuple)):
                    continue
                kept: list = []
                n_dropped = 0
                for w in wps:
                    arr = np.asarray(w, dtype=float)
                    is_dup = any(
                        np.asarray(k).shape == arr.shape and
                        np.allclose(np.asarray(k), arr, atol=eps_rad)
                        for k in kept
                    )
                    if is_dup:
                        n_dropped += 1
                    else:
                        kept.append(list(w))
                if n_dropped:
                    self._waypoints[phase] = kept
                    removed[phase] = n_dropped
        return removed

    def _serialize_waypoints_yaml(self) -> str:
        """Build the full YAML string for the current waypoint state.

        Duplicates are preserved on save: operators intentionally repeat
        waypoints as a self-consistency probe, and stripping them on save
        silently undoes that. Use the explicit dedupe action instead.
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
        return yaml.safe_dump(out, sort_keys=False)

    def _atomic_write(self, target: Path, text: str) -> None:
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text)
        tmp.replace(target)

    def save_waypoints(self) -> Path:
        """Atomic rewrite of the draft YAML, preserving all non-waypoint fields."""
        self._atomic_write(self.draft_yaml_out, self._serialize_waypoints_yaml())
        return self.draft_yaml_out

    def save_waypoints_to_config(self) -> tuple[Path, Optional[Path]]:
        """Promote the current waypoints to the persistent calibration.yaml.

        Writes to ``self.promote_yaml_out`` (resolved at startup -- see
        ``promote_yaml_out`` param), NOT to the runtime ``config_path`` which
        typically lives under ``install/`` and gets clobbered on every colcon
        build. If the target file exists, rename it to
        ``<stem>.yaml.old-YYYYmmdd_HHMMSS`` alongside before overwriting, so
        the previous version is always recoverable. Returns (written_path,
        backup_path or None).
        Raises RuntimeError if no source-tree promote target could be resolved
        and none was passed explicitly.
        """
        if not self.promote_yaml_out:
            raise RuntimeError(
                "no promote target available -- pass -p promote_yaml_out:=<path> "
                "(or launch with -p config:= pointing under <ws>/install/... so "
                "the source-tree counterpart can be auto-resolved). Refusing to "
                "write to the runtime config_path because colcon would overwrite "
                "it on the next build."
            )
        target = self.promote_yaml_out
        backup: Optional[Path] = None
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = target.with_name(f"{target.stem}{target.suffix}.old-{stamp}")
            target.replace(backup)
        self._atomic_write(target, self._serialize_waypoints_yaml())
        return target, backup

    def _resume_from_draft(self) -> None:
        """Auto-resume from draft yaml at startup. Errors are non-fatal."""
        try:
            self.reload_waypoints_from_yaml(self.draft_yaml_out, log_prefix="auto-resume")
        except FileNotFoundError:
            return
        except (OSError, yaml.YAMLError) as e:
            self.get_logger().warning(
                f"draft YAML exists at {self.draft_yaml_out} but failed to parse: {e}; "
                "falling back to base config"
            )

    def reload_waypoints_from_yaml(self, path: Path,
                                    log_prefix: str = "reload") -> dict:
        """Replace in-memory waypoint lists with whatever the named yaml has.

        Operator-facing reload: lets the UI swap between draft (auto-saved)
        and the source-tree calibration.yaml (manually pruned/edited)
        without restarting the server. Also refreshes the pan/tilt scan
        grid (`pan_grid_deg`/`tilt_grid_deg`/`phase2_grid_pairs`) so the
        Pan-Tilt Jog tab tracks the live yaml — without this the tab shows
        whatever was on disk at server startup, even after operator edits.

        Returns a `{phase: count}` dict of what was loaded. Raises
        FileNotFoundError / OSError / yaml.YAMLError on parse failure --
        callers decide how loud to be.
        """
        data = yaml.safe_load(path.read_text()) or {}
        coll = data.get("collector", {}) or {}
        recovered: dict = {}
        for k in ("phase1_waypoints", "phase1_waypoints_custom",
                  "phase2_waypoints", "sanity_xarm_angles_rad"):
            if k in coll:
                v = coll[k]
                recovered[k] = [list(x) if isinstance(x, (list, tuple)) else x
                                for x in v] if isinstance(v, list) else v
        counts = {k: (len(v) if isinstance(v, list) else 0)
                  for k, v in recovered.items()}

        # Phase-2 sweep config: rectangular grid + optional pruned-pair
        # override. Pulled through into `_loaded_cfg` so the Jog tab + the
        # next `_serialize_waypoints_yaml` see the operator's edits.
        grid_updates: dict = {}
        for k in ("pan_grid_deg", "tilt_grid_deg", "phase2_grid_pairs"):
            if k in coll:
                v = coll[k]
                grid_updates[k] = list(v) if isinstance(v, list) else v

        if recovered or grid_updates:
            with self.lock:
                if recovered:
                    self._waypoints.update(recovered)
                if grid_updates:
                    self._loaded_cfg.update(grid_updates)
                    self._refresh_state_grid_locked()
            self.get_logger().info(f"{log_prefix} from {path}: {counts}")
        else:
            self.get_logger().info(f"{log_prefix} from {path}: no waypoint sections found")
        return counts

    def _refresh_state_grid_locked(self) -> None:
        """Republish the pan/tilt scan grid from `_loaded_cfg` into
        `state.grid` (the Pan-Tilt Jog tab's data source).
        Caller must hold `self.lock`.
        """
        self.state.grid = {
            "pan_deg": list(self._loaded_cfg.get("pan_grid_deg", []) or []),
            "tilt_deg": list(self._loaded_cfg.get("tilt_grid_deg", []) or []),
        }


# ---- yaml loader ------------------------------------------------------------

def _resolve_source_tree_yaml(config_path: str) -> Optional[Path]:
    """Walk `config_path` from the install tree back to the colcon source tree.

    Two install patterns are recognised:

    1. Legacy in-tree layout (pre-P5a):
       ``<ws>/install/pan_tilt/share/pan_tilt/config/calibration.yaml``
       → ``<ws>/src/<...>/pan_tilt/config/calibration.yaml``.

    2. tinker_robot_config per-robot layout (P5a and later):
       ``<ws>/install/tinker_robot_config/share/tinker_robot_config/robots/<robot>/pan_tilt/calibration.yaml``
       → ``<ws>/src/tk25_basic/src/tinker_robot_config/robots/<robot>/pan_tilt/calibration.yaml``.

    Returns the source-tree path if a match is found; otherwise None.

    Important: do NOT call ``Path.resolve()`` on `config_path` in pattern 1.
    Under ``colcon build --symlink-install`` the install file is a symlink
    chain into ``src/``, so resolving collapses the path and erases the
    ``install`` segment we depend on. We instead absolutise *without*
    following symlinks, then -- if the install path doesn't show up that
    way either -- fall back to ``Path.resolve()`` for source-tree paths.

    For pattern 2 we trust the install symlink chain directly because the
    per-robot file is unambiguously canonical: ``Path.resolve()`` on the
    install share lands exactly at the source-tree file we want to write.
    """
    if not config_path:
        return None

    raw = Path(config_path)
    # Make absolute without following symlinks: preserves ``install/`` if
    # the operator launched with the canonical share-dir path.
    p = raw if raw.is_absolute() else (Path.cwd() / raw)

    parts = p.parts

    # Pattern 2: tinker_robot_config per-robot layout. Detect by the
    # ``tinker_robot_config/share/tinker_robot_config/robots/`` signature in
    # the install path, and follow the symlink chain to the source. The
    # install share symlinks straight at the source under tk25_basic, so
    # ``Path.resolve()`` is the cleanest way to land there.
    if (
        "install" in parts
        and "tinker_robot_config" in parts
        and "robots" in parts
    ):
        resolved = p.resolve()
        if resolved.is_file() and "src" in resolved.parts:
            return resolved

    if "install" in parts:
        ws = Path(*parts[: parts.index("install")])
    else:
        # Maybe the operator passed a source-tree path directly. Re-resolve
        # symlinks (this handles the symlink-install case too: the resolved
        # path lands inside src/ and we just return that file when present).
        resolved = p.resolve()
        rparts = resolved.parts
        if "src" not in rparts:
            return None
        # If the resolved source path is under the per-robot
        # tinker_robot_config tree, return it directly — no rglob needed.
        if (
            "tinker_robot_config" in rparts
            and "robots" in rparts
            and resolved.is_file()
        ):
            return resolved
        ws = Path(*rparts[: rparts.index("src")])

    src_root = ws / "src"
    if not src_root.is_dir():
        return None
    rel_tail = Path("pan_tilt") / "config" / p.name
    candidates = [c for c in src_root.rglob(str(rel_tail)) if c.is_file()]
    # Prefer a unique match; if multiple (e.g. multiple worktrees), pick the
    # shortest path which is typically the canonical source location.
    if not candidates:
        return None
    return min(candidates, key=lambda c: len(c.parts))


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
    # We serve index/style/app via explicit routes rather than StaticFiles.
    # StaticFiles with a symlinked install-tree `share/pan_tilt/webui` dir
    # silently 404'd in testing (likely due to how colcon --symlink-install
    # symlinks individual files); explicit FileResponse routes are more
    # predictable and the asset count is tiny.
    if webui_dir.exists():
        @app.get("/")
        def root():
            return FileResponse(webui_dir / "index.html", media_type="text/html")

        @app.get("/static/style.css")
        def static_css():
            return FileResponse(webui_dir / "style.css", media_type="text/css")

        @app.get("/static/app.js")
        def static_js():
            return FileResponse(webui_dir / "app.js",
                                media_type="application/javascript")
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

    # Build the dict_id -> DICT_* name lookup once at app-init time;
    # dir(cv2.aruco) is hundreds of attributes and we'd otherwise scan it
    # per-request.
    _ARUCO_DICT_NAMES = {
        getattr(cv2.aruco, n): n
        for n in dir(cv2.aruco) if n.startswith("DICT_")
    }

    @app.get("/api/board")
    def api_board():
        """ChArUco board spec the detector is currently running with.
        Surfaced in the UI so the operator can verify the active spec
        matches the physical print at a glance.
        """
        b = node._board_spec
        out = asdict(b)
        out["dict"] = _ARUCO_DICT_NAMES.get(b.dict_id, f"id={b.dict_id}")
        out["inner_corners"] = b.n_inner_corners
        out["board_size_m"] = [b.squares_x * b.square_len_m,
                               b.squares_y * b.square_len_m]
        out.pop("dict_id", None)
        return out

    @app.get("/api/topics/image")
    def api_image_topics():
        """Discovered sensor_msgs/Image topics on the current ROS graph."""
        with node.lock:
            return {"topics": list(node.state.available_image_topics)}

    @app.post("/api/camera/resubscribe")
    async def api_camera_resubscribe(req: dict):
        image_topic = req.get("image_topic")
        camera_info_topic = req.get("camera_info_topic") or ""
        if not image_topic:
            raise HTTPException(400, "'image_topic' required")
        # If the caller didn't supply camera_info, derive by convention
        # (.../image_raw -> .../camera_info) which matches the orbbec/realsense
        # drivers' default graph.
        if not camera_info_topic:
            if image_topic.endswith("/image_raw"):
                camera_info_topic = image_topic.rsplit("/", 1)[0] + "/camera_info"
            else:
                camera_info_topic = image_topic.rsplit("/", 1)[0] + "/camera_info"

        def _do_resub():
            node._subscribe_camera(image_topic, camera_info_topic)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_resub)
        return {
            "ok": True,
            "image_topic": image_topic,
            "camera_info_topic": camera_info_topic,
        }

    @app.get("/api/frame.jpg")
    def api_frame(raw: int = 0):
        """Latest camera frame as JPEG.

        Query params:
          raw=1  -> raw BGR, no detection overlay. Useful for debugging when
                   the overlay path fails (e.g. missing camera_info) or just
                   to confirm that frames are arriving at all.
        """
        buf = node.get_jpeg(raw=bool(raw))
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
        """Send a JointMove goal to the tinker_arm_msgs action server."""
        angles = req.get("angles_rad")
        if not isinstance(angles, list) or not angles:
            raise HTTPException(400, "'angles_rad' must be a non-empty list of floats")
        loop = asyncio.get_event_loop()
        ok, msg = await loop.run_in_executor(None, node.call_joint_move, angles)
        return {"ok": ok, "message": msg}

    @app.post("/api/xarm/move_cartesian")
    async def api_xarm_move_cartesian(req: dict):
        """Send a CartesianMove goal (target_pose in base_link).

        Payload: {"translation": [x,y,z], "rotation": [qx,qy,qz,qw]}.
        """
        pose = req.get("target_pose", req)
        if "translation" not in pose or "rotation" not in pose:
            raise HTTPException(400, "payload must have 'translation' and 'rotation'")
        loop = asyncio.get_event_loop()
        ok, msg = await loop.run_in_executor(None, node.call_cartesian_move, pose)
        return {"ok": ok, "message": msg}

    @app.post("/api/pantilt/move")
    async def api_pantilt_move(req: dict):
        pan_deg = float(req.get("pan_deg", 0.0))
        tilt_deg = float(req.get("tilt_deg", 0.0))
        node.publish_pantilt(math.radians(pan_deg), math.radians(tilt_deg))
        return {"ok": True, "message": f"pan={pan_deg:+.1f} tilt={tilt_deg:+.1f} published"}

    # Pan-tilt zero-state wizard — split across two endpoints because the
    # firmware procedure requires the operator to physically unplug and then
    # reconnect motor 2 between steps. The browser walks the user through:
    #
    #   1. (Browser-only) operator jogs the pan-tilt to the desired zero pose
    #   2. (Browser-only) operator unplugs motor 2, clicks "continue"
    #   3. POST /api/pantilt/zero_wizard/remap     → firmware T:501 raw=1 new=2
    #   4. (Browser-only) operator reconnects motor 2, clicks "continue"
    #   5. POST /api/pantilt/zero_wizard/finalize  → firmware T:502 id=1, then id=2
    #
    # Once step 3 has been issued the firmware servo IDs are mutated; aborting
    # the wizard at that point leaves the chain in a half-configured state and
    # the operator must complete step 5 manually (or restart the wizard).
    @app.post("/api/pantilt/zero_wizard/remap")
    async def api_pantilt_zero_wizard_remap(req: dict):
        """Step 3 of the zero-state wizard. Sends `T:501 raw=1 new=2`.
        The operator must have unplugged motor 2 BEFORE calling this.
        """
        node.get_logger().info(
            f"[zero_wizard] HTTP POST /zero_wizard/remap received: body={req!r}"
        )
        loop = asyncio.get_event_loop()
        try:
            ok, msg = await loop.run_in_executor(None, node.call_remap_servo_ids)
        except Exception as exc:
            node.get_logger().error(
                f"[zero_wizard] uncaught exception in call_remap_servo_ids: {exc!r}"
            )
            raise HTTPException(500, f"remap crashed: {exc}")
        node.get_logger().info(
            f"[zero_wizard] /remap HTTP response: ok={ok}, message={msg!r}"
        )
        return {"ok": ok, "message": msg, "step": "remap"}

    @app.post("/api/pantilt/zero_wizard/finalize")
    async def api_pantilt_zero_wizard_finalize(req: dict):
        """Step 5 of the zero-state wizard. Sends T:502 for both motor IDs
        (the controller's SetZero(BOTH) handler does id=1 then id=2 in order).
        The operator must have reconnected motor 2 BEFORE calling this.
        """
        node.get_logger().info(
            f"[zero_wizard] HTTP POST /zero_wizard/finalize received: body={req!r}"
        )
        loop = asyncio.get_event_loop()
        try:
            ok, msg = await loop.run_in_executor(None, node.call_set_zero, "both")
        except Exception as exc:
            node.get_logger().error(
                f"[zero_wizard] uncaught exception in call_set_zero('both'): {exc!r}"
            )
            raise HTTPException(500, f"finalize crashed: {exc}")
        node.get_logger().info(
            f"[zero_wizard] /finalize HTTP response: ok={ok}, message={msg!r}"
        )
        return {"ok": ok, "message": msg, "step": "finalize"}

    # --- waypoints ----------------------------------------------------------
    VALID_PHASES = {"phase1_waypoints", "phase1_waypoints_custom",
                    "phase2_waypoints", "sanity_xarm_angles_rad"}

    @app.get("/api/waypoints")
    def api_waypoints_all():
        return {k: node.list_waypoints(k) for k in VALID_PHASES}

    # /save and /promote must be declared BEFORE /{phase} so FastAPI's
    # first-match routing doesn't funnel them into the phase handler.
    @app.post("/api/waypoints/save")
    async def api_waypoints_save():
        try:
            path = node.save_waypoints()
        except Exception as exc:
            raise HTTPException(500, f"save failed: {exc}")
        return {"ok": True, "path": str(path)}

    @app.post("/api/waypoints/dedupe")
    async def api_waypoints_dedupe():
        """Drop near-duplicate waypoints from the in-memory lists.

        Operator-triggered: save/promote no longer dedupe implicitly.
        Caller is expected to follow up with /save or /promote to persist.
        """
        try:
            removed = node.dedupe_waypoints()
        except Exception as exc:
            raise HTTPException(500, f"dedupe failed: {exc}")
        return {"ok": True, "removed": removed}

    @app.post("/api/waypoints/promote")
    async def api_waypoints_promote():
        """Overwrite the persistent calibration.yaml (source-tree, NOT the
        install copy) with the current waypoints, renaming the old file to
        <stem>.yaml.old-<timestamp> alongside. The target path is resolved
        once at startup -- see /api/waypoints/paths.
        """
        try:
            written, backup = node.save_waypoints_to_config()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(500, f"promote failed: {exc}")
        return {
            "ok": True,
            "path": str(written),
            "backup": str(backup) if backup else None,
        }

    @app.post("/api/waypoints/reload")
    def api_waypoints_reload(req: dict):
        """Reload waypoint lists from a chosen yaml source.

        Body: `{"source": "draft" | "promote"}`. Default: "draft".
        - "draft":   reload from `draft_yaml_out` (the auto-saved working copy)
        - "promote": reload from `promote_yaml_out` (the source-tree
                     calibration.yaml the operator manually prunes/edits)

        Returns `{ok, source, path, counts}`. Raises 404 if the chosen
        source isn't configured (no `--config` at startup → no promote
        target) or doesn't exist on disk; 400 on parse failure.
        """
        source = (req or {}).get("source", "draft")
        path_map = {
            "draft":   node.draft_yaml_out,
            "promote": node.promote_yaml_out,
        }
        if source not in path_map:
            raise HTTPException(400, f"unknown source: {source!r} (use 'draft' or 'promote')")
        path = path_map[source]
        if path is None:
            raise HTTPException(404, f"no {source} target configured -- "
                                     "did you launch with -p config:=...?")
        try:
            counts = node.reload_waypoints_from_yaml(path,
                                                     log_prefix=f"reload from {source}")
        except FileNotFoundError:
            raise HTTPException(404, f"{source} yaml not found at {path}")
        except (OSError, yaml.YAMLError) as e:
            raise HTTPException(400, f"parse failed: {e}")
        return {"ok": True, "source": source, "path": str(path), "counts": counts}

    @app.get("/api/waypoints/paths")
    def api_waypoints_paths():
        """Expose the three yaml paths the UI cares about:
        - `config`: runtime yaml that calibrate_web loaded at startup
        - `draft`:  where unsaved edits land (via /save)
        - `promote`: persistent target for /promote (source-tree, never install/)
        """
        return {
            "config": node.config_path or None,
            "draft": str(node.draft_yaml_out),
            "promote": str(node.promote_yaml_out) if node.promote_yaml_out else None,
        }

    @app.get("/api/calib/phase1_custom_park")
    def api_phase1_custom_park_get():
        """Operator-chosen pan/tilt for the Phase-1 custom-park dataset.
        Lives in the loaded yaml's collector section so it round-trips
        through save/promote like the rest of the calibration config.
        """
        cfg = node._loaded_cfg
        return {
            "pan_deg": float(cfg.get("phase1_custom_park_pan_deg", 0.0)),
            "tilt_deg": float(cfg.get("phase1_custom_park_tilt_deg", 0.0)),
        }

    @app.post("/api/calib/phase1_custom_park")
    def api_phase1_custom_park_set(body: dict):
        try:
            pan = float(body["pan_deg"])
            tilt = float(body["tilt_deg"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "body must be {pan_deg: float, tilt_deg: float}")
        # Soft envelope check matching the operator-declared limits.
        if not (-30.0 <= pan <= 30.0):
            raise HTTPException(400, f"pan_deg out of envelope (±30): {pan}")
        if not (0.0 <= tilt <= 30.0):
            raise HTTPException(400, f"tilt_deg out of envelope (0..+30): {tilt}")
        with node.lock:
            node._loaded_cfg["phase1_custom_park_pan_deg"] = pan
            node._loaded_cfg["phase1_custom_park_tilt_deg"] = tilt
        return {"ok": True, "pan_deg": pan, "tilt_deg": tilt}

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

    # --- calibrate tab ------------------------------------------------------
    #
    # All endpoints here are file-I/O only -- they never move the robot. The
    # collection step (which does move the robot) stays a terminal invocation
    # per the design decision documented in shimmying-fluttering-koala.md.

    def _parse_session_file(sess_path: Path, name: str) -> dict:
        """Read one of the session's JSONs and surface a status summary.

        Returns {exists, mtime, n_samples?, trans_rmse_m?, rot_rmse_rad?,
        val_trans_rmse_m?, val_rot_rmse_rad?}. Missing keys are simply absent.
        Never raises on bad JSON -- returns {exists, error} instead.
        """
        p = sess_path / name
        try:
            text = p.read_text()
            mtime = p.stat().st_mtime
        except (FileNotFoundError, IsADirectoryError):
            return {"exists": False}
        info: dict = {"exists": True, "mtime": mtime, "path": str(p)}
        try:
            data = json.loads(text)
        except Exception as exc:
            info["error"] = f"parse: {exc}"
            return info
        # Collector-side (raw samples list).
        if isinstance(data, dict) and "samples" in data:
            info["n_samples"] = len(data["samples"])
        # Analyser-side (aggregate RMSEs).
        for k in (
            "trans_rmse_m", "rot_rmse_rad",
            "val_trans_rmse_m", "val_rot_rmse_rad",
            "train_trans_rmse_m", "train_rot_rmse_rad",
            "n_train", "n_val", "rms_px",
            "phase1_park_pan_rad", "phase1_park_tilt_rad",
        ):
            if isinstance(data, dict) and k in data:
                info[k] = data[k]
        # Phase-4 validation.json: surface verdict + self-consistency rmse so
        # the file-status table can show the verdict pill at-a-glance.
        if isinstance(data, dict) and data.get("phase") == "validation":
            info["verdict"] = data.get("verdict")
            sc = data.get("self_consistency") or {}
            if "trans_rmse_m" in sc:
                info["trans_rmse_m"] = sc["trans_rmse_m"]
            if "rot_rmse_rad" in sc:
                info["rot_rmse_rad"] = sc["rot_rmse_rad"]
            if "n_samples_used" in data:
                info["n_samples"] = data["n_samples_used"]
        return info

    def _gates_from_files(files: dict[str, dict]) -> list[dict]:
        """Compute gate pass/fail rows from an already-parsed file inventory
        so we don't re-open + re-parse the same JSONs we just read for the
        files-table response."""
        out = []
        for fname, key, thresh, unit, label in CALIB_GATES:
            info = files.get(fname) or {}
            common = {"file": fname, "key": key, "label": label,
                      "unit": unit, "threshold": thresh}
            if not info.get("exists") or key not in info:
                out.append({**common, "status": "missing"})
                continue
            value = float(info[key])
            status = "pass" if value <= thresh else "fail"
            out.append({
                **common, "value": value, "status": status,
            })
        return out

    @app.get("/api/calib/sessions")
    def api_calib_sessions():
        return {
            "sessions_dir": str(node.calib_runner.sessions_dir),
            "sessions": node.calib_runner.list_sessions(),
        }

    @app.post("/api/calib/session")
    async def api_calib_create_session(req: dict):
        name = (req or {}).get("name", "")
        try:
            path = node.calib_runner.create_session(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except FileExistsError:
            raise HTTPException(409, f"session {name!r} already exists")
        return {"name": name, "path": str(path)}

    @app.get("/api/calib/session/{name}")
    def api_calib_session_detail(name: str):
        try:
            sess_path = node.calib_runner.session_path(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if not sess_path.is_dir():
            raise HTTPException(404, f"unknown session: {name}")
        tracked = [
            "phase1_handeye.json", "phase1_handeye_custom.json",
            "phase2_chain.json", "sanity.json", "phase4_validation.json",
            "intrinsic.json", "handeye.json", "handeye_custom.json",
            "chain.json", "polish.json", "validation.json", "dry_run.json",
        ]
        files = {f: _parse_session_file(sess_path, f) for f in tracked}
        return {
            "name": name,
            "path": str(sess_path),
            "files": files,
            "gates": _gates_from_files(files),
        }

    @app.get("/api/calib/session/{name}/coverage")
    def api_calib_coverage(name: str):
        """Per-sample marker positions across the camera FoV.

        Returns angular positions (degrees off the optical axis,
        horizontal / vertical) computed directly from the body-frame
        translation of the marker. Lets the operator visually spot under-
        sampled regions of the camera FoV without needing to know K (we
        just compute atan2(y,x) and atan2(z,x) in body coords).

        Body-frame convention is X forward (depth), Y left, Z up. We flip
        Y and Z signs so the canvas reads like the camera image: +X axis
        points right (image-X right), +Y axis points down (image-Y down).
        """
        try:
            sess_path = node.calib_runner.session_path(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if not sess_path.is_dir():
            raise HTTPException(404, f"unknown session: {name}")

        # Snapshot live K + frame dims once so per-sample projection and the
        # FoV box use the same numbers.
        with node.lock:
            K = node._latest_K.copy() if getattr(node, "_latest_K", None) is not None else None
            bgr = node._latest_bgr
            img_shape = bgr.shape[:2] if bgr is not None else None
        fov_h_deg, fov_v_deg = 80.0, 51.0
        image_w, image_h = (None, None)
        if K is not None and img_shape:
            image_h, image_w = int(img_shape[0]), int(img_shape[1])
            fov_h_deg = math.degrees(2 * math.atan(image_w / (2 * float(K[0, 0]))))
            fov_v_deg = math.degrees(2 * math.atan(image_h / (2 * float(K[1, 1]))))

        # Body (X fwd, Y left, Z up) -> optical (X right, Y down, Z fwd).
        # Inverse of utils.R_BODY_FROM_OPTICAL; used to project samples into
        # normalized image coordinates so the client can draw them on a flat
        # canvas with the same K/dims the camera would use.
        R_OPT_FROM_BODY = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ])

        out: list[dict] = []
        for fname in ("phase1_handeye.json", "phase1_handeye_custom.json",
                      "phase2_chain.json"):
            try:
                data = json.loads((sess_path / fname).read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            for idx, s in enumerate(data.get("samples", [])):
                t = s.get("t_cam_marker_body", {}).get("translation")
                if not t or len(t) != 3:
                    continue
                x, y, z = float(t[0]), float(t[1]), float(t[2])
                if x <= 0:
                    continue  # behind the camera, skip
                # Flip Y / Z so canvas X+ is right, Y+ is down (matches image)
                horiz_deg = math.degrees(math.atan2(-y, x))
                vert_deg = math.degrees(math.atan2(-z, x))
                depth_m = math.sqrt(x * x + y * y + z * z)
                u_norm: Optional[float] = None
                v_norm: Optional[float] = None
                if K is not None and image_w and image_h:
                    t_opt = R_OPT_FROM_BODY @ np.array([x, y, z])
                    z_o = float(t_opt[2])
                    if z_o > 0:
                        u = float(K[0, 0]) * float(t_opt[0]) / z_o + float(K[0, 2])
                        v = float(K[1, 1]) * float(t_opt[1]) / z_o + float(K[1, 2])
                        if math.isfinite(u) and math.isfinite(v):
                            u_norm = u / image_w
                            v_norm = v / image_h
                out.append({
                    "phase": fname.replace(".json", ""),
                    "index": idx,
                    "label": s.get("label", ""),
                    "horiz_deg": horiz_deg,
                    "vert_deg": vert_deg,
                    "depth_m": depth_m,
                    "u_norm": u_norm,
                    "v_norm": v_norm,
                })
        return {
            "samples": out,
            "fov_h_deg": fov_h_deg,
            "fov_v_deg": fov_v_deg,
            "image_w": image_w,
            "image_h": image_h,
            "have_intrinsics": K is not None and image_w is not None,
        }

    @app.get("/api/calib/session/{name}/file/{filename}")
    def api_calib_session_file(name: str, filename: str):
        try:
            sess_path = node.calib_runner.session_path(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        # Only let through specific analyser files so we don't hand back
        # arbitrary blobs (the runner's sandboxing already covers ../ etc.).
        allowed = {"handeye.json", "handeye_custom.json",
                   "chain.json", "polish.json", "validation.json",
                   "phase1_handeye.json", "phase1_handeye_custom.json",
                   "phase2_chain.json", "sanity.json",
                   "phase4_validation.json",
                   "intrinsic.json", "dry_run.json"}
        if filename not in allowed:
            raise HTTPException(404, f"unknown file: {filename}")
        try:
            return json.loads((sess_path / filename).read_text())
        except FileNotFoundError:
            raise HTTPException(404, f"{filename} not found in session")
        except Exception as exc:
            raise HTTPException(500, f"parse: {exc}")

    # Commands accepted by the run endpoint, with the session-relative files
    # each one REQUIRES before it can run. Keep the allowlist tight so a
    # malformed POST can't ask us to run arbitrary argv. `collect` subcommands
    # have empty prereq lists -- they generate files rather than consume them.
    # Which session-relative basenames each command will accept as the
    # operator-chosen input file. Keep these tight — `api_calib_run` pipes
    # whatever the front-end sends straight into a subprocess argv, so any
    # client-supplied filename has to be checked against these allowlists.
    _CHAIN_HANDEYE_ALLOWLIST = {"handeye.json", "handeye_custom.json"}
    _POLISH_PHASE1_ALLOWLIST = {"phase1_handeye.json", "phase1_handeye_custom.json"}
    _POLISH_SEED_ALLOWLIST = {"chain.json"}
    # Validate (Phase 4) accepts polish.json or chain.json as the params
    # under test. Phase 4 is xArm-independent (board is fixed in base_link),
    # so no handeye file is involved.
    _VALIDATE_PARAMS_ALLOWLIST = {"polish.json", "chain.json"}

    _CALIB_PREREQS = {
        # analysis subcommands -> run_calibration.py
        "handeye":         ["phase1_handeye.json"],
        "handeye_custom":  ["phase1_handeye_custom.json"],
        # Chain accepts a per-request handeye file (default handeye.json), so the
        # static prereq carries only what's always required; the chosen handeye
        # file is checked per-request below.
        "chain":           ["phase2_chain.json"],
        # Polish accepts per-request phase1 list + seed, validated below.
        "polish":          ["phase2_chain.json"],
        # Validate (Phase 4) needs the collected validation samples; the
        # params choice is checked per-request below.
        "validate":        ["phase4_validation.json"],
        # collection subcommands -> calibrate_collect.py (moves the robot)
        "collect_phase1":         [],     # canonical level park (pan=0, tilt=+30)
        "collect_phase1_custom":  [],     # operator-chosen park, see /api/calib/phase1_custom_park
        "collect_dry_run":        [],     # preflight: validate motion only, no image capture
        "collect_phase2": ["phase1_handeye.json"],  # not technically required,
                                                     # but Phase 2 without
                                                     # Phase 1 won't produce a
                                                     # usable calibration. Gate
                                                     # so the operator doesn't
                                                     # waste a 20-min sweep.
        "collect_sanity": [],
        "collect_both":   [],
        # Phase 4 collection: drives pan-tilt across a sweep with arm held
        # static. No file prereqs (it generates phase4_validation.json).
        "collect_phase4_validation": [],
    }
    _ANALYSIS_CMDS = {"handeye", "handeye_custom", "chain", "polish", "validate"}
    # Endpoint-facing collect commands map onto `phase:=<phase>` for
    # calibrate_collect by stripping the "collect_" prefix.

    @app.post("/api/calib/run")
    async def api_calib_run(req: dict):
        session = (req or {}).get("session", "")
        cmd = (req or {}).get("cmd", "")
        extra_flags = list((req or {}).get("flags", []))
        if cmd not in _CALIB_PREREQS:
            raise HTTPException(400, f"unknown command: {cmd}")
        try:
            sess_path = node.calib_runner.session_path(session)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if not sess_path.is_dir():
            raise HTTPException(404, f"unknown session: {session}")

        # Prerequisite check -- fail clean with a 400 BEFORE we spawn. Without
        # this, the subprocess emits a Python traceback into the log pane on
        # an out-of-order click (e.g. hitting "chain" before Phase 1 is run).
        missing = [f for f in _CALIB_PREREQS[cmd] if not (sess_path / f).is_file()]
        if missing:
            raise HTTPException(
                400,
                f"{cmd} needs {', '.join(missing)} in the session -- "
                "run the upstream step first"
            )

        if cmd in _ANALYSIS_CMDS:
            # Analysis path: python -m pan_tilt.calibration.run_calibration <sub>
            # `handeye_custom` is not a real subcommand -- it's the same handeye
            # solver pointed at the custom-park dataset, with --out-name set so
            # it doesn't clobber the level handeye.json.
            sub_cmd = "handeye" if cmd == "handeye_custom" else cmd
            cmd_args: list[str] = [sub_cmd]
            if cmd == "handeye":
                cmd_args.append(str(sess_path / "phase1_handeye.json"))
                cmd_args += ["--out", str(sess_path)]
            elif cmd == "handeye_custom":
                cmd_args.append(str(sess_path / "phase1_handeye_custom.json"))
                cmd_args += ["--out", str(sess_path),
                             "--out-name", "handeye_custom.json"]
            elif cmd == "chain":
                handeye_choice = (req or {}).get("handeye") or "handeye.json"
                if handeye_choice not in _CHAIN_HANDEYE_ALLOWLIST:
                    raise HTTPException(
                        400,
                        f"chain: handeye must be one of "
                        f"{sorted(_CHAIN_HANDEYE_ALLOWLIST)}, got {handeye_choice!r}"
                    )
                if not (sess_path / handeye_choice).is_file():
                    raise HTTPException(
                        400,
                        f"chain: {handeye_choice} not found in session — "
                        f"run handeye first"
                    )
                cmd_args.append(str(sess_path / "phase2_chain.json"))
                cmd_args += ["--handeye", str(sess_path / handeye_choice)]
                cmd_args += ["--out", str(sess_path)]
            elif cmd == "polish":
                phase1_choice = (req or {}).get("phase1") or ["phase1_handeye.json"]
                if not isinstance(phase1_choice, list) or not phase1_choice:
                    raise HTTPException(
                        400, "polish: phase1 must be a non-empty list of basenames"
                    )
                bad_phase1 = [f for f in phase1_choice if f not in _POLISH_PHASE1_ALLOWLIST]
                if bad_phase1:
                    raise HTTPException(
                        400,
                        f"polish: phase1 entries must be in "
                        f"{sorted(_POLISH_PHASE1_ALLOWLIST)}, got {bad_phase1}"
                    )
                missing_phase1 = [f for f in phase1_choice if not (sess_path / f).is_file()]
                if missing_phase1:
                    raise HTTPException(
                        400,
                        f"polish: phase1 file(s) not found in session: {missing_phase1}"
                    )
                seed_choice = (req or {}).get("seed") or "chain.json"
                if seed_choice not in _POLISH_SEED_ALLOWLIST:
                    raise HTTPException(
                        400,
                        f"polish: seed must be one of "
                        f"{sorted(_POLISH_SEED_ALLOWLIST)}, got {seed_choice!r}"
                    )
                if not (sess_path / seed_choice).is_file():
                    raise HTTPException(
                        400,
                        f"polish: {seed_choice} not found in session — "
                        f"run chain first"
                    )
                cmd_args += ["--phase1",
                             *[str(sess_path / f) for f in phase1_choice]]
                cmd_args += ["--phase2", str(sess_path / "phase2_chain.json")]
                cmd_args += ["--seed", str(sess_path / seed_choice)]
                cmd_args += ["--out", str(sess_path)]
            elif cmd == "validate":
                params_choice = (req or {}).get("params") or "polish.json"
                if params_choice not in _VALIDATE_PARAMS_ALLOWLIST:
                    raise HTTPException(
                        400,
                        f"validate: params must be one of "
                        f"{sorted(_VALIDATE_PARAMS_ALLOWLIST)}, got {params_choice!r}"
                    )
                if not (sess_path / params_choice).is_file():
                    raise HTTPException(
                        400,
                        f"validate: {params_choice} not found in session — "
                        f"run polish (or chain) first"
                    )
                cmd_args += ["--phase4", str(sess_path / "phase4_validation.json"),
                             "--params", str(sess_path / params_choice),
                             "--out", str(sess_path)]
            # Allowlist of client flags passed through to run_calibration.
            analysis_flags = {
                "--fit-pan-offset", "--lock-tb-rotation", "--unlock-tb-rotation",
                "--verbose",
            }
            for f in extra_flags:
                if f not in analysis_flags:
                    raise HTTPException(400, f"disallowed flag: {f}")
                cmd_args.append(f)
            argv = [sys.executable, "-u", "-m", "pan_tilt.calibration.run_calibration", *cmd_args]
            label = f"run_calibration {cmd}"

        else:
            # Collection path: ros2 run pan_tilt calibrate_collect -- moves the
            # robot. Requires a config yaml (the same one calib_web loaded);
            # reject if the server wasn't launched with -p config:=...
            if not node.config_path:
                raise HTTPException(
                    400,
                    "calibrate_web was launched without -p config:=... "
                    "calibrate_collect needs the board spec + waypoint lists "
                    "from calibration.yaml; relaunch with the config param."
                )
            if extra_flags:
                raise HTTPException(400, "collect commands accept no extra flags")
            # Snapshot current in-memory waypoints to the draft yaml and feed
            # *that* to calibrate_collect, so collect runs always reflect what
            # the operator sees in the xArm Waypoints tab without forcing them
            # to "Promote to calibration.yaml" first. The draft preserves
            # board+safety sections via __passthrough__, so it's a complete
            # config from calibrate_collect's POV.
            collect_config = str(node.save_waypoints())
            phase_arg = cmd.removeprefix("collect_")
            argv = [
                "ros2", "run", "pan_tilt", "calibrate_collect", "--ros-args",
                "-p", f"config:={collect_config}",
                "-p", f"out_dir:={sess_path}",
                "-p", f"phase:={phase_arg}",
            ]
            label = f"calibrate_collect --phase {phase_arg}"

        try:
            result = await node.calib_runner.spawn(session, argv, label=label)
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(409, str(exc))
        return result

    @app.post("/api/calib/runs/{run_id}/cancel")
    async def api_calib_cancel(run_id: str):
        ok = await node.calib_runner.cancel(run_id)
        if not ok:
            raise HTTPException(404, f"no active run with id {run_id}")
        return {"ok": True, "run_id": run_id}

    @app.get("/api/calib/urdf_targets")
    def api_calib_urdf_targets():
        return {"targets": [t.to_dict() for t in list_urdf_targets()]}

    @app.post("/api/calib/urdf_diff")
    async def api_calib_urdf_diff(req: dict):
        session = (req or {}).get("session", "")
        xacro_path = (req or {}).get("xacro_path", "")
        # Default to polish.json: the joint refinement is consistently a few
        # mm tighter than chain alone on real datasets.
        results_file = (req or {}).get("results_file", "polish.json")
        if not xacro_path:
            raise HTTPException(400, "xacro_path required")
        try:
            result = await node.calib_runner.urdf_diff(session, results_file, xacro_path)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:
            raise HTTPException(500, str(exc))
        return result

    @app.post("/api/calib/urdf_apply")
    async def api_calib_urdf_apply(req: dict):
        session = (req or {}).get("session", "")
        xacro_path = (req or {}).get("xacro_path", "")
        results_file = (req or {}).get("results_file", "polish.json")
        if not xacro_path:
            raise HTTPException(400, "xacro_path required")
        try:
            result = await node.calib_runner.urdf_apply(session, results_file, xacro_path)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except PermissionError as exc:
            raise HTTPException(500, f"cannot write xacro: {exc}")
        except RuntimeError as exc:
            raise HTTPException(500, str(exc))
        return result

    @app.get("/api/calib/gates")
    def api_calib_gates():
        """Expose the gate thresholds so the UI stays in sync with Python."""
        return {"gates": [
            {"file": f, "key": k, "threshold": t, "unit": u, "label": lbl}
            for f, k, t, u, lbl in CALIB_GATES
        ]}

    @app.get("/api/calib/commands")
    def api_calib_commands():
        """Expose the per-command prereq file list + whether collect is
        available (i.e. a config yaml was passed at startup). The UI uses
        this to disable run buttons that can't execute yet."""
        return {
            "prereqs": _CALIB_PREREQS,
            "collect_enabled": bool(node.config_path),
            "config_path": node.config_path or None,
        }

    # --- Prune (preview-then-apply) -----------------------------------------
    #
    # Operator-driven waypoint pruning by end-point pose similarity. The flow
    # is intentionally two-step: Preview returns kept/dropped counts and a
    # per-row breakdown without writing anything; Apply re-runs the same
    # deterministic prune and writes a timestamped sidecar yaml + report.
    # The original calibration.yaml is never modified.

    PRUNE_PHASE_LABEL_PREFIX = {
        "phase1_waypoints":         "phase1",
        "phase1_waypoints_custom":  "phase1_custom",
        "phase2_grid":              "phase2_grid",
    }
    PRUNE_DEFAULT_FACTORS = {
        "phase1_waypoints": {
            "trans_tol_m": 0.05, "rot_tol_deg": 8.0,
            "min_count": 8, "min_rot_diversity_pairs": 6,
            "min_rot_diversity_deg": 28.0, "seed_index": 0,
        },
        "phase1_waypoints_custom": {
            "trans_tol_m": 0.05, "rot_tol_deg": 8.0,
            "min_count": 8, "min_rot_diversity_pairs": 6,
            "min_rot_diversity_deg": 28.0, "seed_index": 0,
        },
        "phase2_grid": {
            "trans_tol_m": 0.04, "rot_tol_deg": 6.0,
            "min_count": 6, "min_rot_diversity_pairs": 0,
            "min_rot_diversity_deg": 28.0, "seed_index": 0,
        },
    }

    def _list_prior_runs() -> list[dict]:
        out: list[dict] = []
        sessions = node.calib_runner.sessions_dir
        if not sessions.exists():
            return out
        for run_dir in sessions.iterdir():
            if not run_dir.is_dir():
                continue
            for fname in ("phase1_handeye.json", "phase1_handeye_custom.json"):
                p = run_dir / fname
                if not p.is_file():
                    continue
                try:
                    raw = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
                n = len(samples) if isinstance(samples, list) else None
                out.append({
                    "name": f"{run_dir.name}/{fname}",
                    "path": str(p),
                    "mtime": p.stat().st_mtime,
                    "n_samples": n,
                })
        out.sort(key=lambda r: r["mtime"], reverse=True)
        return out

    def _build_payloads(phase: str) -> tuple[list[dict], dict]:
        prefix = PRUNE_PHASE_LABEL_PREFIX[phase]
        if phase == "phase2_grid":
            with node.lock:
                pan_grid = list(node._loaded_cfg.get("pan_grid_deg", []) or [])
                tilt_grid = list(node._loaded_cfg.get("tilt_grid_deg", []) or [])
            payloads: list[dict] = []
            for pi, p_deg in enumerate(pan_grid):
                for ti, t_deg in enumerate(tilt_grid):
                    payloads.append({
                        "label": f"{prefix}/p{p_deg:+.1f}t{t_deg:+.1f}",
                        "pan_deg": float(p_deg),
                        "tilt_deg": float(t_deg),
                        "pan_idx": pi,
                        "tilt_idx": ti,
                    })
            meta = {"kind": "grid", "pan_grid_deg": pan_grid,
                    "tilt_grid_deg": tilt_grid}
            return payloads, meta

        joints_lists = node.list_waypoints(phase)
        payloads = [
            {
                "label": f"{prefix}/{i}",
                "joints": list(angles) if isinstance(angles, (list, tuple)) else angles,
                "yaml_index": i,
            }
            for i, angles in enumerate(joints_lists)
        ]
        return payloads, {"kind": "joint_list"}

    def _build_predictor(phase: str, predictor_choice: str, prior_run_path: Optional[str]):
        info: dict = {"requested": predictor_choice, "prior_run_path": prior_run_path}
        predictors = []
        if phase in ("phase1_waypoints", "phase1_waypoints_custom"):
            if predictor_choice in ("auto", "replay_only") and prior_run_path:
                try:
                    predictors.append(replay_predictor(prior_run_path))
                    info["replay"] = f"loaded {prior_run_path}"
                except (OSError, ValueError) as exc:
                    info["replay_error"] = str(exc)
            if predictor_choice == "replay_only" and not predictors:
                info["fallback"] = "replay_only requested but no prior run loaded"
            # Phase-1 FK is intentionally absent: yourdfpy isn't in the venv
            # and the calibration workflow always produces a prior run.
            # Per-row failures show up in the UI as "no prediction".
        elif phase == "phase2_grid":
            # Phase-2 cell similarity is determined by camera pose, which is
            # FK-only (it doesn't depend on the xArm anchor). Prior-run replay
            # would mix anchor-dependent marker poses, so it's not used here.
            predictors.append(pantilt_grid_predictor())
            info["fk"] = "pantilt_grid_predictor(default_params)"
        return chain_predictors(predictors), info

    def _normalize_factors(phase: str, raw: dict) -> dict:
        defaults = PRUNE_DEFAULT_FACTORS[phase]
        out = dict(defaults)
        if isinstance(raw, dict):
            for k in defaults:
                if k in raw and raw[k] is not None:
                    out[k] = raw[k]
        try:
            return {
                "trans_tol_m": float(out["trans_tol_m"]),
                "rot_tol_deg": float(out["rot_tol_deg"]),
                "min_count": int(out["min_count"]),
                "min_rot_diversity_pairs": int(out["min_rot_diversity_pairs"]),
                "min_rot_diversity_deg": float(out["min_rot_diversity_deg"]),
                "seed_index": int(out["seed_index"]),
            }
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"bad factor value: {exc}")

    def _normalize_overrides(raw) -> dict[int, str]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise HTTPException(400, "overrides must be a {index: 'keep'|'drop'} mapping")
        out: dict[int, str] = {}
        for k, v in raw.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                raise HTTPException(400, f"override key {k!r} is not an integer index")
            if v not in ("keep", "drop"):
                raise HTTPException(400, f"override value {v!r} must be 'keep' or 'drop'")
            out[idx] = v
        return out

    def _run_prune(req: dict) -> tuple[dict, dict, list[dict], dict]:
        phase = (req or {}).get("phase")
        if phase not in PRUNE_PHASE_LABEL_PREFIX:
            raise HTTPException(
                404,
                f"unknown phase: {phase!r}. valid: {sorted(PRUNE_PHASE_LABEL_PREFIX)}",
            )
        factors = _normalize_factors(phase, (req or {}).get("factors", {}))
        overrides = _normalize_overrides((req or {}).get("overrides"))
        predictor_choice = str((req or {}).get("predictor_choice", "auto"))
        if predictor_choice not in ("auto", "replay_only", "fk_only"):
            raise HTTPException(
                400,
                f"predictor_choice must be one of auto/replay_only/fk_only; "
                f"got {predictor_choice!r}",
            )
        prior_run_path = (req or {}).get("prior_run_path") or None

        payloads, meta = _build_payloads(phase)
        if not payloads:
            raise HTTPException(
                400,
                f"phase {phase!r} has no waypoints loaded — author them on the "
                "Waypoints tab or pass a non-empty calibration.yaml at startup",
            )
        predict_fn, predictor_info = _build_predictor(
            phase, predictor_choice, prior_run_path,
        )
        result = prune_waypoints(
            payloads, predict_fn,
            trans_tol_m=factors["trans_tol_m"],
            rot_tol_deg=factors["rot_tol_deg"],
            min_count=factors["min_count"],
            min_rot_diversity_pairs=factors["min_rot_diversity_pairs"],
            min_rot_diversity_rad=math.radians(factors["min_rot_diversity_deg"]),
            seed_index=factors["seed_index"],
            overrides=overrides,
        )
        diagnostics = dict(result.diagnostics)
        if "replay" in predictor_info:
            n_failed = diagnostics.get("n_predict_failed", 0)
            if n_failed > 0 and n_failed / max(1, len(payloads)) > 0.20:
                diagnostics["warning"] = (
                    f"prior run looks stale: {n_failed} of {len(payloads)} "
                    "labels not found — is this the right phase1_handeye.json?"
                )
        response = result.to_dict()
        response["diagnostics"] = diagnostics
        response["phase"] = phase
        response["predictor_info"] = predictor_info
        response["meta"] = meta
        return response, factors, payloads, meta

    @app.get("/api/calib/prune_inputs")
    def api_calib_prune_inputs(phase: str):
        if phase not in PRUNE_PHASE_LABEL_PREFIX:
            raise HTTPException(
                404,
                f"unknown phase: {phase!r}. valid: {sorted(PRUNE_PHASE_LABEL_PREFIX)}",
            )
        if phase == "phase2_grid":
            with node.lock:
                pan_grid = list(node._loaded_cfg.get("pan_grid_deg", []) or [])
                tilt_grid = list(node._loaded_cfg.get("tilt_grid_deg", []) or [])
            n_items = len(pan_grid) * len(tilt_grid)
        else:
            n_items = len(node.list_waypoints(phase))
        return {
            "phase": phase,
            "n_items": n_items,
            "default_factors": PRUNE_DEFAULT_FACTORS[phase],
            "label_prefix": PRUNE_PHASE_LABEL_PREFIX[phase],
            "prior_runs": _list_prior_runs(),
        }

    @app.post("/api/calib/prune_preview")
    def api_calib_prune_preview(req: dict):
        response, _factors, _payloads, _meta = _run_prune(req)
        response["wrote"] = None
        return response

    @app.post("/api/calib/prune_apply")
    def api_calib_prune_apply(req: dict):
        if not bool((req or {}).get("confirm")):
            raise HTTPException(
                400,
                "apply requires confirm=true; preview first, then re-issue "
                "with confirm=true",
            )
        response, factors, payloads, meta = _run_prune(req)
        try:
            written_paths = _write_prune_sidecar(
                phase=response["phase"],
                factors=factors,
                payloads=payloads,
                meta=meta,
                preview=response,
            )
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        response["wrote"] = written_paths
        return response

    @app.post("/api/calib/prune_overwrite")
    def api_calib_prune_overwrite(req: dict):
        """Overwrite the source-tree calibration.yaml with the pruned set,
        renaming the current file to ``<stem>.yaml.old-<YYYYmmdd_HHMMSS>``
        first. Mirrors the existing waypoint-promote backup convention.
        """
        if not bool((req or {}).get("confirm")):
            raise HTTPException(
                400,
                "overwrite requires confirm=true; preview first, then re-issue "
                "with confirm=true",
            )
        response, factors, payloads, meta = _run_prune(req)
        try:
            written_paths = _overwrite_source_with_prune(
                phase=response["phase"],
                factors=factors,
                payloads=payloads,
                meta=meta,
                preview=response,
            )
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        # The source yaml on disk is now the new truth; pull it back into
        # `_loaded_cfg` + `state.grid` so the Pan-Tilt Jog tab reflects the
        # pruned grid without an extra "Reload from calibration.yaml" click.
        # Best-effort -- the file write succeeded so don't fail the request
        # if the reload trips on a transient parse issue.
        reload_fn = getattr(node, "reload_waypoints_from_yaml", None)
        if callable(reload_fn) and node.promote_yaml_out is not None:
            try:
                reload_fn(node.promote_yaml_out, log_prefix="prune-overwrite")
            except (FileNotFoundError, OSError, yaml.YAMLError) as exc:
                log.warning("prune-overwrite reload failed: %s", exc)
        response["wrote"] = written_paths
        return response

    def _build_pruned_collector(
        *, phase: str, payloads: list[dict], preview: dict, src_collector: dict,
    ) -> dict:
        """Apply the pruned-set surgery to a copy of the source collector
        dict. Returns the modified dict — caller is responsible for writing
        it. Shared by the sidecar Apply and the in-place Overwrite paths.
        """
        if phase == "phase2_grid":
            kept_pairs = [
                [float(payloads[i]["pan_deg"]), float(payloads[i]["tilt_deg"])]
                for i in preview["kept_indices"]
            ]
            return {**src_collector, "phase2_grid_pairs": kept_pairs}
        existing = list(src_collector.get(phase, []) or [])
        kept_yaml = [
            list(existing[i])
            for i in sorted(preview["kept_indices"])
            if 0 <= i < len(existing)
        ]
        return {**src_collector, phase: kept_yaml}

    def _read_source_yaml() -> tuple[dict, dict]:
        """Load + sanity-check the source-tree calibration.yaml.

        Returns ``(full_data, collector_dict)``. Raises RuntimeError on any
        failure, with the same message style the existing flows use."""
        if not node.promote_yaml_out:
            raise RuntimeError(
                "no source-tree calibration.yaml could be resolved — pass "
                "-p promote_yaml_out:=<path> at startup so calib_web knows "
                "where the canonical yaml lives."
            )
        try:
            src_text = node.promote_yaml_out.read_text()
            src_data = yaml.safe_load(src_text) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(
                f"could not read source yaml {node.promote_yaml_out}: {exc}"
            )
        coll = src_data.get("collector")
        if not isinstance(coll, dict):
            raise RuntimeError(
                f"source yaml {node.promote_yaml_out} has no 'collector' section"
            )
        return src_data, coll

    def _write_prune_sidecar(
        *, phase: str, factors: dict, payloads: list[dict],
        meta: dict, preview: dict,
    ) -> dict:
        """Write ``calibration.pruned.<phase>.<ts>.yaml`` plus a
        ``prune_report.<phase>.<ts>.json`` next to the source-tree
        calibration.yaml. Sidecar = copy of the source with only the pruned
        section replaced. Phase-1 phases get their joint-list filtered;
        Phase-2 grid pruning emits ``phase2_grid_pairs`` that ``run_phase2``
        consumes in preference to the rectangular grid.
        """
        src_data, src_coll = _read_source_yaml()
        target_dir = node.promote_yaml_out.parent
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        sidecar = target_dir / f"calibration.pruned.{phase}.{ts}.yaml"
        report = target_dir / f"prune_report.{phase}.{ts}.json"

        n = 1
        while sidecar.exists() or report.exists():
            sidecar = target_dir / f"calibration.pruned.{phase}.{ts}.{n}.yaml"
            report = target_dir / f"prune_report.{phase}.{ts}.{n}.json"
            n += 1

        coll = _build_pruned_collector(
            phase=phase, payloads=payloads, preview=preview, src_collector=src_coll,
        )
        out = {"collector": coll}
        if "safety" in src_data:
            out["safety"] = src_data["safety"]
        if "board" in src_data:
            out["board"] = src_data["board"]

        header = (
            f"# Generated by calib_web prune-apply on {ts}.\n"
            f"# Source: {node.promote_yaml_out}\n"
            f"# Phase: {phase}\n"
            f"# Factors: {json.dumps(factors)}\n"
            f"# Result: {preview['headline']}\n"
            f"# Predictor: {json.dumps(preview.get('predictor_info', {}))}\n"
        )
        sidecar_text = header + yaml.safe_dump(out, sort_keys=False)
        node._atomic_write(sidecar, sidecar_text)

        report_payload = _build_prune_report_payload(
            phase=phase, ts=ts, factors=factors, preview=preview,
            extra={"sidecar_yaml": str(sidecar)},
        )
        node._atomic_write(report, json.dumps(report_payload, indent=2))
        return {"sidecar_yaml": str(sidecar), "report_json": str(report)}

    def _overwrite_source_with_prune(
        *, phase: str, factors: dict, payloads: list[dict],
        meta: dict, preview: dict,
    ) -> dict:
        """Overwrite ``node.promote_yaml_out`` with the pruned set.

        Renames the existing file to ``<stem>.yaml.old-<ts>`` first so the
        previous version is always recoverable — same convention as
        ``CalibWebNode.save_waypoints_to_config``. Also drops a
        ``prune_report.<phase>.<ts>.json`` next to it for audit.
        """
        src_data, src_coll = _read_source_yaml()
        target = node.promote_yaml_out
        target_dir = target.parent
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        report = target_dir / f"prune_report.{phase}.{ts}.json"

        # Disambiguate the report filename if a sub-second collision lands
        # on an existing file (matches the sidecar convention).
        n = 1
        while report.exists():
            report = target_dir / f"prune_report.{phase}.{ts}.{n}.json"
            n += 1

        coll = _build_pruned_collector(
            phase=phase, payloads=payloads, preview=preview, src_collector=src_coll,
        )
        out = {"collector": coll}
        if "safety" in src_data:
            out["safety"] = src_data["safety"]
        if "board" in src_data:
            out["board"] = src_data["board"]

        header = (
            f"# Overwritten by calib_web prune-overwrite on {ts}.\n"
            f"# Phase: {phase}\n"
            f"# Factors: {json.dumps(factors)}\n"
            f"# Result: {preview['headline']}\n"
            f"# Predictor: {json.dumps(preview.get('predictor_info', {}))}\n"
            f"# Previous version backed up to "
            f"{target.stem}{target.suffix}.old-<ts> in this directory.\n"
        )
        new_text = header + yaml.safe_dump(out, sort_keys=False)

        backup: Optional[Path] = None
        if target.exists():
            backup = target.with_name(f"{target.stem}{target.suffix}.old-{ts}")
            # If two overwrites land in the same second, fall back to a
            # numbered suffix so we never clobber a backup.
            n = 1
            while backup.exists():
                backup = target.with_name(
                    f"{target.stem}{target.suffix}.old-{ts}.{n}"
                )
                n += 1
            target.replace(backup)
        node._atomic_write(target, new_text)

        report_payload = _build_prune_report_payload(
            phase=phase, ts=ts, factors=factors, preview=preview,
            extra={
                "overwrote_yaml": str(target),
                "backup_yaml": str(backup) if backup else None,
            },
        )
        node._atomic_write(report, json.dumps(report_payload, indent=2))
        return {
            "wrote_yaml": str(target),
            "backup_yaml": str(backup) if backup else None,
            "report_json": str(report),
        }

    def _build_prune_report_payload(
        *, phase: str, ts: str, factors: dict, preview: dict, extra: dict,
    ) -> dict:
        return {
            "phase": phase,
            "ts": ts,
            "factors": factors,
            "headline": preview["headline"],
            "predictor_info": preview.get("predictor_info", {}),
            "diagnostics": preview.get("diagnostics", {}),
            "kept_indices": preview["kept_indices"],
            "dropped_indices": preview["dropped_indices"],
            "items": preview["items"],
            "source_yaml": str(node.promote_yaml_out),
            **extra,
        }

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

    # --- WebSocket (calibration subprocess log fanout) ----------------------
    @app.websocket("/ws/calib-log")
    async def ws_calib_log(ws: WebSocket):
        await ws.accept()
        q = node.calib_runner.subscribe()
        try:
            while True:
                event = await q.get()
                await ws.send_text(json.dumps(event))
        except WebSocketDisconnect:
            return
        except Exception as exc:
            log.warning("calib-log websocket error: %s", exc)
        finally:
            node.calib_runner.unsubscribe(q)

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
    # Force UDPv4-only transport before rclpy.init() so Node() doesn't stall
    # doing SHM discovery against the 150+ segments typically owned by
    # camera + manipulation stacks on a live robot. The UI is not on a
    # throughput-critical path — a downscaled preview at modest Hz is fine.
    # Users who really need SHM (e.g. to cohost with a high-rate consumer)
    # can pre-set FASTDDS_BUILTIN_TRANSPORTS themselves.
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    # The workspace shm.xml profile pins SHM as the preferred transport and
    # overrides FASTDDS_BUILTIN_TRANSPORTS; ignore it for this node so we
    # don't re-enter the SHM stall path.
    os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)

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
