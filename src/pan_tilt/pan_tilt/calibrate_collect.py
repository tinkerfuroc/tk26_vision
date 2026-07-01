"""ROS 2 node: pan-tilt / xArm calibration dataset collector.

Workflow
--------
1. Load YAML config (ChArUco spec, xArm waypoints for Phase 1 and Phase 2,
   pan-tilt grid, safety envelope, timing constants).
2. Subscribe to RGB + camera_info. Wait until we have a first camera_info and
   a first pan-tilt state.
3. Run Phase 1: at servo-zero, iterate xArm waypoints, capture a sample per
   waypoint. Write Phase-1 JSON.
4. Run Phase 2: iterate (xArm waypoint x pan-tilt grid cell), capture a sample
   per cell with backlash mitigation. Write Phase-2 JSON.
5. Re-capture a sanity pose at the end and compare to the start (drift gate).

Safety
------
Every xArm target is validated against:
  - a software Z-floor (configurable, default 0.25 m)
  - a cylindrical exclusion around the pan-tilt mast
before the JointMove action goal is sent. A violation aborts the session
cleanly; we do not partially succeed.

The node does not depend on MoveIt and does not do collision checking against
other robot links beyond the coarse envelope; operators must pre-validate the
waypoint list in RViz.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState

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
    cluster_consensus,
    detect_pose,
)
from .calibration.custom_naming import (
    custom_dataset_filenames,
    migrate_custom_datasets,
    sanitize_custom_name,
)
from .calibration.safety import SafetyEnvelope
from .calibration.utils import (
    matrix_to_pose_dict,
    optical_to_body,
    pose_error_scalars,
    pose_to_matrix,
)


# tinker_arm_msgs is imported lazily at node construction so this module stays
# importable on hosts without the tinker_manipulation stack installed.


def _default_calib_config_path() -> str:
    """Resolve calibration.yaml via tinker_robot_config; fallback to legacy.

    Returns the absolute install-share path for the per-robot
    ``pan_tilt/calibration.yaml``. Returns ``''`` if the resolver isn't
    available or fails — operators must then pass ``-p config:=…``.
    Operator-supplied non-empty values are always respected over this default.
    """
    if _trc_resolver is None:
        return ''
    try:
        return str(_trc_resolver.load().path('pan_tilt/calibration.yaml'))
    except Exception:  # pragma: no cover - resolver error path
        return ''


# ---- config -----------------------------------------------------------------

@dataclass
class CollectConfig:
    board: BoardSpec = field(default_factory=BoardSpec)
    safety: SafetyEnvelope = field(default_factory=SafetyEnvelope)

    phase1_waypoints: list = field(default_factory=list)   # list of joint-angle lists (rad), used at firmware (pan=0, tilt=level_tilt_deg) — the canonical "level" park
    # Named CUSTOM hand-eye datasets — each {name, park_pan_deg, park_tilt_deg,
    # waypoints}. Populated by `migrate_custom_datasets` after load so a legacy
    # single-custom yaml (the flat keys below) still works.
    phase1_custom_datasets: list = field(default_factory=list)
    # LEGACY single-custom fields. Still read from old YAMLs by `_load_config`'s
    # flat override, then folded into `phase1_custom_datasets`. Not serialized.
    phase1_waypoints_custom: list = field(default_factory=list)  # used at the operator-chosen custom park
    # Custom Phase-1 park pose (firmware degrees). Defaults to (0, 0) so old
    # behavior (camera looking 30° down) holds for unconfigured installs.
    phase1_custom_park_pan_deg: float = 0.0
    phase1_custom_park_tilt_deg: float = 0.0
    # Canonical "level" park tilt (firmware deg) for the Phase-1 hand-eye — the
    # head pose where the camera optical axis is horizontal. HARDWARE-SPECIFIC:
    # tinker2's 2026 head remount makes level = +30; tinker1's older mount makes
    # level = +45. Override per-robot via the collector: section of
    # calibration.yaml. Default 30 preserves tinker2 behavior when unset.
    level_tilt_deg: float = 30.0
    phase2_waypoints: list = field(default_factory=list)
    pan_grid_deg: list = field(default_factory=lambda: [-30.0, -15.0, 0.0, 15.0, 30.0])
    # Firmware degrees. Grid spans from physical level (firmware +30) DOWN
    # by 30 deg -- it does NOT go above level because the board sits on the
    # xArm EE below the camera, and anything tilted above level points the
    # camera at the ceiling. Per pan_tilt_model.py, "+firmware = tilt up",
    # so smaller firmware values = looking further down. Servo-zero
    # (firmware 0) = 30 deg below level.
    tilt_grid_deg: list = field(default_factory=lambda: [6.0, 12.0, 18.0, 24.0, 30.0])
    # Optional pruned override of the (pan, tilt) cross-product. When set,
    # `run_phase2` iterates these (pan_deg, tilt_deg) pairs instead of the
    # rectangular `pan_grid_deg × tilt_grid_deg`. Produced by the calib_web
    # prune-apply endpoint; absent from default yamls.
    phase2_grid_pairs: list = field(default_factory=list)
    sanity_xarm_angles_rad: list = field(default_factory=list)

    # Timing + convergence. Tolerance is generous (1.0 deg) because the
    # serial-bus servo's natural feedback jitter is ~0.1-0.5 deg even when
    # mechanically stopped; a tighter tolerance kept resetting the hold
    # counter and timing out the settle wait at boundary positions.
    servo_settle_tol_deg: float = 1.0
    # 0.5 s of in-tolerance feedback before declaring settled. Don't bump
    # higher: the serial servo's natural feedback jitter (~0.1-0.3 deg) at
    # 0.3 deg tolerance can repeatedly reset the hold counter and we run
    # out the 6 s overall settle timeout. Post-settle steadiness needed for
    # blur-free capture is delivered by `xarm_settle_sec` and
    # `pre_capture_quiet_sec` instead.
    servo_settle_hold_sec: float = 0.5
    # Backlash-collapse overshoot is DISABLED by default: the servo just
    # parks at whatever (pan, tilt) is demanded. Set > 0 to re-enable a
    # per-axis overshoot-then-return pass; the soft envelope below clamps
    # the intermediate so it never asks for an unreachable angle. Defaults
    # match the operator-declared envelope: pan ±30, tilt ≤+30 (the
    # "physical level" ceiling -- anything above points the camera at the
    # ceiling). The controller's hard clamp at tilt_max_deg=30 will reject
    # any over-bound intermediate as well.
    servo_backlash_overshoot_deg: float = 0.0
    servo_backlash_pause_sec: float = 0.2
    pan_overshoot_max_deg: float = 30.0
    pan_overshoot_min_deg: float = -30.0
    tilt_overshoot_max_deg: float = 30.0
    tilt_overshoot_min_deg: float = 0.0
    # xArm joint feedback says "done" before the EE finishes oscillating.
    # 1.5 s leaves the board mechanically steady so 1080p frames aren't blurred.
    # Used as a fallback wait when joint_state-based convergence isn't available
    # (no /joint_states publisher); the new image-stability gate is the primary
    # mechanism for catching mechanical ring-down.
    xarm_settle_sec: float = 1.5
    arm_action_timeout_sec: float = 30.0
    # Joint convergence gate (A1). After JointMove returns success, we wait for
    # /joint_states to actually report all controlled joints within tolerance
    # of the commanded target, held for hold_sec. This catches the failure
    # mode where the action server returns success before the trajectory
    # has finished executing.
    joint_state_topic: str = "/joint_states"
    # Tolerance is generous (~0.86°) to ride above the deceleration-phase
    # transient noise on the xArm's /joint_states stream. Final steady-state
    # accuracy is ~0.05° but in-flight ticks during deceleration can read
    # 0.3-0.6° momentarily, which would keep resetting the hold counter.
    # 0.86° max joint error at typical reach (~0.5 m) ≈ 0.5 mm EE-position
    # noise post-settle — well under the calibration's PnP noise floor.
    joint_settle_tol_rad: float = 0.015   # ~0.86°
    joint_settle_hold_sec: float = 0.2
    # 10 s cap accommodates slow trajectories needed to keep the marker in
    # view during phase1_custom; fast moves still settle in <1 s.
    joint_settle_timeout_sec: float = 10.0
    # Image-stability gate (A2). Soft early-exit: when corner positions
    # settle within `image_stable_tol_px` for `image_stable_hold_sec`, capture
    # immediately. On timeout (or detector misses), fall back to the
    # xarm_settle_sec fixed wait and proceed -- DO NOT skip the cell. The
    # downstream cell-level robust averaging handles residual jitter; gating
    # the cell to oblivion just throws away usable data.
    #
    # Tolerance widened from 0.3 px to 1.0 px because 1080p sensor noise +
    # sub-pixel corner refinement variance produces ~0.3-0.5 px frame-to-frame
    # jitter even on a perfectly stationary board, which kept the original
    # 0.3 px gate from ever converging on noisy scenes.
    image_stable_tol_px: float = 1.0
    image_stable_hold_sec: float = 0.4
    image_stable_window_frames: int = 5
    image_stable_timeout_sec: float = 5.0
    image_stable_min_frames: int = 3
    # Post-capture TF consistency check (A3). Re-lookup base→ee after the
    # capture loop and reject the sample if the EE pose drifted during the
    # window — the arm was still moving while we were collecting frames.
    capture_drift_max_trans_m: float = 0.001   # 1 mm
    capture_drift_max_rot_deg: float = 0.1
    # Duplicate-EE guard (A4). Reject any sample whose recorded T_base_ee
    # matches an already-accepted sample in the same phase within these
    # tolerances. By design no two waypoints in the operator's yaml should
    # produce the same EE pose; a duplicate signals a motion-completion
    # race or a yaml bug.
    duplicate_ee_reject_trans_m: float = 0.001   # 1 mm
    duplicate_ee_reject_rot_deg: float = 0.1
    # Skew gate is defensive against in-motion captures; the settle wait
    # already confirmed the mechanism is parked, so the recorded angles do
    # not change between consecutive state messages and a generous threshold
    # is fine. 20 ms was below the achievable minimum (camera at 30 Hz +
    # state publisher at ~10-20 Hz means the closest-pair skew floor is
    # ~25-50 ms, plus another ~30-100 ms for detect_pose at 1080p).
    sample_stamp_skew_max_ms: float = 200.0
    # Cell-capture loop budget. `frames_per_cell` is the TARGET; the loop
    # exits successfully once `min_valid_detections` are collected by
    # `capture_timeout_sec`. `detection_lookahead_frames` lets short CMOS
    # flicker streaks pass without aborting the cell. The min floor is set
    # to what `robust_average` actually needs (3) so cells with genuinely
    # flickery detection still produce data.
    frames_per_cell: int = 10
    frame_min_interval_ms: float = 40.0
    capture_timeout_sec: float = 8.0
    detection_lookahead_frames: int = 5
    min_valid_detections: int = 3
    # Fixed quiet period right before each cell's image capture, after
    # mechanical settles, to let exposure/AGC and camera buffers stabilise.
    pre_capture_quiet_sec: float = 0.3
    # Extra quiet period AFTER pan-tilt motion in Phase 2, before capture.
    # The pan-tilt mast resonates noticeably for a beat after the servo
    # feedback says "settled" -- 1 s extra here is much cheaper than a
    # blurred-detection cell (which downstream just ends up rejected by
    # the per-frame minimum-corner gate or gives an outlier IPPE flip).
    phase2_pantilt_settle_sec: float = 1.0
    # Inter-duplicate check: when two phase-1 waypoints have the same EE
    # pose (operator-defined repeats serving as a self-consistency probe),
    # their captured marker poses MUST match within these tolerances. If
    # they don't, per-frame PnP is unstable for this scene+geometry and
    # downstream hand-eye RMSE will be garbage even with more samples.
    # `duplicate_ee_match_*` controls when two waypoints count as duplicates;
    # `duplicate_marker_max_*` is the failure threshold for the marker delta.
    duplicate_ee_match_trans_m: float = 0.005   # 5 mm
    duplicate_ee_match_rot_deg: float = 1.0
    duplicate_marker_max_trans_m: float = 0.005   # 5 mm
    duplicate_marker_max_rot_deg: float = 1.0

    # Topics / actions (override via yaml).
    image_topic: str = "/camera/color/image_raw"
    camera_info_topic: str = "/camera/color/camera_info"
    pantilt_cmd_topic: str = "/pan_tilt_controller/cmd"
    pantilt_state_topic: str = "/pan_tilt_controller/state"
    # tinker_arm_msgs actions on the pick_and_place GraspNode.
    joint_move_action: str = "joint_move_action"
    cartesian_move_action: str = "cartesian_move_action"
    base_frame: str = "base_link"
    ee_frame: str = "link_eef"

    # Speed/accel raw values for pan-tilt servo.
    pantilt_speed_raw: int = 120
    pantilt_accel_raw: int = 20

    # ---- Phase 4: end-to-end validation -----------------------------------
    # Phase 4 expects a ChArUco board fixed in `base_link` (e.g. mounted on a
    # tripod, taped to a fixture, sitting on a table — anywhere stationary
    # within the camera's reachable FoV). The xArm is irrelevant: this phase
    # neither commands JointMove nor reads link_eef TF. The pan-tilt sweeps
    # across N (pan, tilt) poses; downstream `validate` then checks that the
    # FK chain projects the marker to a consistent base_link pose across
    # views.
    #
    # 5 corners + center are always included for coverage; remaining samples
    # are uniform-random within (pan_range, tilt_range). Defaults to the
    # convex hull of pan_grid_deg/tilt_grid_deg if unset.
    n_validation_samples: int = 20
    validation_seed: int = 0
    validation_pan_range_deg: list = field(default_factory=list)   # [min, max]
    validation_tilt_range_deg: list = field(default_factory=list)  # [min, max]


def _load_config(path: Optional[str]) -> CollectConfig:
    cfg = CollectConfig()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Flat override; only known keys.
    for k, v in data.get("collector", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if "safety" in data:
        for k, v in data["safety"].items():
            if hasattr(cfg.safety, k):
                setattr(cfg.safety, k, v)
    if "board" in data:
        for k, v in data["board"].items():
            if k == "dict":
                cfg.board.dict_id = getattr(cv2.aruco, v)
            elif hasattr(cfg.board, k):
                setattr(cfg.board, k, v)
    # Normalize custom datasets: prefer the named-list form, else migrate the
    # legacy flat keys (phase1_waypoints_custom + phase1_custom_park_*) into a
    # single entry named "custom". Operates on the raw collector dict so both
    # shapes round-trip without data loss.
    cfg.phase1_custom_datasets = migrate_custom_datasets(data.get("collector", {}))
    return cfg


# ---- node -------------------------------------------------------------------

class CalibrateCollectNode(Node):
    def __init__(self):
        super().__init__("calibrate_collect")

        # Default resolves the per-robot calibration.yaml via
        # tinker_robot_config (uses $ROBOT_NAME). Override with -p config:=…
        # to point at a custom file (e.g. a pruned sidecar produced by
        # calib_web).
        self.declare_parameter("config", _default_calib_config_path())
        self.declare_parameter("out_dir", "calibration_data")
        self.declare_parameter("phase", "both")  # both | phase1 | phase1_custom | phase2 | sanity | phase4_validation | dry_run
        # Which named custom dataset to collect when phase==phase1_custom. Empty
        # selects the sole dataset (error if there are several).
        self.declare_parameter("custom_name", "")

        config_path = self.get_parameter("config").value or None
        self._cfg = _load_config(config_path)
        self._out_dir = Path(self.get_parameter("out_dir").value)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._phase_select = self.get_parameter("phase").value
        self._custom_name = (self.get_parameter("custom_name").value or "").strip()

        # Echo waypoint counts so an unexpectedly-pruned sidecar config is
        # obvious in the terminal -- if the operator points -p config:= at a
        # `calibration.pruned.*.yaml` produced by calib_web's prune-apply
        # endpoint, the small numbers here are the kept set.
        n_p2_cells = (len(self._cfg.phase2_grid_pairs)
                      if self._cfg.phase2_grid_pairs
                      else len(self._cfg.pan_grid_deg) * len(self._cfg.tilt_grid_deg))
        custom_summary = ", ".join(
            f"{d['name']}({len(d.get('waypoints', []))})"
            for d in self._cfg.phase1_custom_datasets
        ) or "(none)"
        self.get_logger().info(
            f"loaded config from {config_path}: "
            f"phase1={len(self._cfg.phase1_waypoints)} "
            f"phase1_custom_datasets=[{custom_summary}] "
            f"phase2_anchors={len(self._cfg.phase2_waypoints)} "
            f"phase2_cells={n_p2_cells}"
            + (" (pruned grid pairs)" if self._cfg.phase2_grid_pairs else "")
        )

        self._bridge = CvBridge()
        self._board = build_board(self._cfg.board)
        self._detector = build_detector(self._board)
        self.get_logger().info(
            f"ChArUco board: {self._cfg.board.squares_x}x{self._cfg.board.squares_y} squares, "
            f"square={self._cfg.board.square_len_m*1000:.1f}mm, "
            f"marker={self._cfg.board.marker_len_m*1000:.1f}mm, "
            f"dict_id={self._cfg.board.dict_id}, "
            f"inner_corners={self._cfg.board.n_inner_corners}"
        )

        # ---- camera subs --------------------------------------------------------
        qos_sensor = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._latest_image: Optional[tuple] = None     # (stamp_ns, np.ndarray bgr)
        self._latest_K: Optional[np.ndarray] = None
        self._latest_D: Optional[np.ndarray] = None
        self._image_lock = threading.Lock()

        self.create_subscription(Image, self._cfg.image_topic, self._on_image, qos_sensor)
        self.create_subscription(
            CameraInfo, self._cfg.camera_info_topic, self._on_camera_info, qos_sensor
        )

        # ---- pan-tilt cmd/state ----------------------------------------------
        self._pt_pub = self.create_publisher(PanTiltCommand, self._cfg.pantilt_cmd_topic, 10)
        self._pt_state: Optional[PanTiltState] = None
        self._pt_state_lock = threading.Lock()
        self.create_subscription(
            PanTiltState, self._cfg.pantilt_state_topic, self._on_pt_state, 10
        )

        # ---- /joint_states for xArm convergence gate -------------------------
        self._joint_state: Optional[dict[str, float]] = None
        self._joint_state_lock = threading.Lock()
        self.create_subscription(
            JointState, self._cfg.joint_state_topic, self._on_joint_state, 10
        )

        # ---- tf2 for base -> ee ----------------------------------------------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ---- tinker_arm_msgs JointMove action client ---------------------------
        try:
            from rclpy.action import ActionClient
            from tinker_arm_msgs.action import JointMove  # type: ignore
        except ImportError:
            self.get_logger().error(
                "tinker_arm_msgs not available; build/source the tinker_manipulation stack."
            )
            raise
        self._joint_move_type = JointMove
        self._joint_move_client = ActionClient(
            self, JointMove, self._cfg.joint_move_action,
        )

    # ---- subs --------------------------------------------------------------

    def _on_image(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # cv_bridge errors
            self.get_logger().warn(f"cv_bridge error: {exc}")
            return
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self._image_lock:
            self._latest_image = (stamp_ns, img)

    def _on_camera_info(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        D = np.array(msg.d, dtype=float).flatten()
        with self._image_lock:
            self._latest_K = K
            self._latest_D = D

    def _on_pt_state(self, msg: PanTiltState):
        with self._pt_state_lock:
            self._pt_state = msg

    def _on_joint_state(self, msg: JointState):
        with self._joint_state_lock:
            self._joint_state = dict(zip(msg.name, msg.position))

    # ---- small helpers -----------------------------------------------------

    def _get_image(self) -> Optional[tuple]:
        with self._image_lock:
            if self._latest_image is None or self._latest_K is None:
                return None
            return (*self._latest_image, self._latest_K.copy(), self._latest_D.copy())

    def _get_pt_state(self) -> Optional[PanTiltState]:
        with self._pt_state_lock:
            return self._pt_state

    def _get_joint_state(self) -> Optional[dict[str, float]]:
        with self._joint_state_lock:
            return None if self._joint_state is None else dict(self._joint_state)

    def _wait_for_streams(self, timeout_sec: float = 10.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_sec:
            if self._get_image() is not None and self._get_pt_state() is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    # ---- convergence gates -------------------------------------------------

    def _wait_joint_convergence(self, target_rad: list[float],
                                 joint_names: Optional[list[str]] = None) -> bool:
        """Block until /joint_states reports the controlled joints within
        `joint_settle_tol_rad` of the target, held for `joint_settle_hold_sec`.

        `target_rad` is the goal angle list in the same order JointMove uses
        (joint0..joint6 for xArm7). When `joint_names` is None we infer them
        as 'joint1'..'joint{N}' (matches xArm convention) and rely on substring
        matches against /joint_states names.

        Returns False on timeout. In that case the caller MUST treat the
        sample as invalid -- the action server reported success but the arm
        hasn't actually arrived.
        """
        tol = float(self._cfg.joint_settle_tol_rad)
        hold = float(self._cfg.joint_settle_hold_sec)
        timeout = float(self._cfg.joint_settle_timeout_sec)
        names = joint_names or [f"joint{i+1}" for i in range(len(target_rad))]

        t0 = time.monotonic()
        ok_since: Optional[float] = None
        no_state_logged = False
        last_max_err: Optional[float] = None
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.02)
            js = self._get_joint_state()
            if js is None:
                if not no_state_logged and time.monotonic() - t0 > 1.0:
                    self.get_logger().warn(
                        f"joint convergence: no {self._cfg.joint_state_topic} "
                        "messages yet -- is the arm publishing /joint_states?"
                    )
                    no_state_logged = True
                ok_since = None
                continue
            errs = []
            for name, tgt in zip(names, target_rad):
                if name in js:
                    errs.append(abs(js[name] - tgt))
            if len(errs) < len(target_rad):
                # Some joint name didn't match. Don't silently accept.
                ok_since = None
                continue
            last_max_err = max(errs)
            if last_max_err < tol:
                ok_since = ok_since or time.monotonic()
                if time.monotonic() - ok_since >= hold:
                    return True
            else:
                ok_since = None

        self.get_logger().error(
            f"joint convergence TIMEOUT after {timeout:.1f}s; "
            f"max joint error = {math.degrees(last_max_err):.3f} deg "
            f"vs tol = {math.degrees(tol):.3f} deg"
            if last_max_err is not None else
            f"joint convergence TIMEOUT after {timeout:.1f}s with no /joint_states received"
        )
        return False

    def _wait_image_stable(self, timeout_sec: Optional[float] = None) -> bool:
        """Wait until the ChArUco corner geometry stops moving in the image.

        Runs the detector at the camera framerate (best-effort) and tracks
        per-corner pixel positions across a sliding window. The scene is
        considered "stable" when, for `image_stable_hold_sec` continuously,
        every detected corner stays within `image_stable_tol_px` of its
        window mean AND the per-corner stddev across the window is below
        2x the tolerance.

        Falls back to a fixed `xarm_settle_sec` wait if the detector misses
        for >1 s (e.g. motion blur during the early window).

        Returns True when stable, False on timeout.
        """
        timeout = timeout_sec if timeout_sec is not None else self._cfg.image_stable_timeout_sec
        tol_px = float(self._cfg.image_stable_tol_px)
        hold = float(self._cfg.image_stable_hold_sec)
        win = int(self._cfg.image_stable_window_frames)
        min_frames = int(self._cfg.image_stable_min_frames)

        t0 = time.monotonic()
        last_seen_stamp_ns = 0
        last_detect_at: Optional[float] = None
        # Each history entry: (t, {corner_id: (x,y)}). We key by id so we
        # can intersect across frames cleanly.
        history: list[tuple[float, dict[int, np.ndarray]]] = []
        stable_since: Optional[float] = None
        miss_since: Optional[float] = None

        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.02)
            img_tup = self._get_image()
            if img_tup is None:
                continue
            stamp_ns, img, _, _ = img_tup
            if stamp_ns == last_seen_stamp_ns:
                continue
            last_seen_stamp_ns = stamp_ns
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ch_corners, ch_ids, _, _ = self._detector.detectBoard(gray)
            now = time.monotonic()

            if ch_ids is None or len(ch_ids) < 4:
                miss_since = miss_since or now
                stable_since = None
                history.clear()
                if now - miss_since > 1.0:
                    self.get_logger().warn(
                        f"image-stability gate: detector missed for "
                        f"{now - miss_since:.1f}s; falling back to fixed "
                        f"{self._cfg.xarm_settle_sec:.1f}s wait"
                    )
                    time.sleep(self._cfg.xarm_settle_sec)
                    return True
                continue
            miss_since = None
            last_detect_at = now

            corners = ch_corners.reshape(-1, 2)
            id_list = [int(x) for x in ch_ids.flatten()]
            id_to_pos = {cid: corners[k].copy() for k, cid in enumerate(id_list)}
            history.append((now, id_to_pos))
            history = history[-win:]
            if len(history) < min_frames:
                stable_since = None
                continue

            # Compare over the corner IDs visible in EVERY window frame --
            # missing/recovered corners would otherwise read as motion.
            common_ids = set.intersection(*(set(h[1]) for h in history))
            if len(common_ids) < 4:
                stable_since = None
                continue
            sorted_ids = sorted(common_ids)
            arr = np.stack([
                np.array([h[1][cid] for cid in sorted_ids]) for h in history
            ])                                            # (W, n_common, 2)
            mean_pos = arr.mean(axis=0)
            dev = np.linalg.norm(arr - mean_pos, axis=2)  # (W, n_common)
            std = dev.std(axis=0)
            max_dev_now = float(dev[-1].max())
            max_std_window = float(std.max())

            if max_dev_now < tol_px and max_std_window < 2.0 * tol_px:
                stable_since = stable_since or now
                if now - stable_since >= hold:
                    return True
            else:
                stable_since = None

        if last_detect_at is None:
            # Detector never saw the board. The cell will produce no valid
            # captures anyway; let the cell-capture loop's min-detection gate
            # surface that and skip there (so the operator sees the more
            # informative "only N valid detections" message).
            self.get_logger().error(
                f"image-stability gate TIMEOUT after {timeout:.1f}s with no "
                "successful detections"
            )
            return False
        # Detector worked but corners never quieted within tolerance --
        # likely persistent low-amplitude wobble. Sleep the fixed fallback
        # and proceed; the cell-level robust_average + post-capture TF
        # drift check will catch any cell that's actually unusable.
        self.get_logger().warn(
            f"image-stability gate TIMEOUT after {timeout:.1f}s; "
            f"falling back to {self._cfg.xarm_settle_sec:.1f}s fixed wait and proceeding"
        )
        time.sleep(self._cfg.xarm_settle_sec)
        return True

    # ---- safety ------------------------------------------------------------

    def _validate_ee_pose(self, T_base_ee: np.ndarray) -> Optional[str]:
        """Return None if safe, or a human-readable reason for rejection."""
        return self._cfg.safety.validate(T_base_ee)

    # ---- motion primitives -------------------------------------------------

    def _send_pt_goal(self, pan_deg: float, tilt_deg: float):
        msg = PanTiltCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = 0  # ABSOLUTE
        msg.pan_rad = math.radians(pan_deg)
        msg.tilt_rad = math.radians(tilt_deg)
        msg.speed_raw = self._cfg.pantilt_speed_raw
        msg.accel_raw = self._cfg.pantilt_accel_raw
        self._pt_pub.publish(msg)

    def _wait_pt_settle(self, pan_deg: float, tilt_deg: float, timeout_sec: float = 10.0) -> bool:
        tol = math.radians(self._cfg.servo_settle_tol_deg)
        hold = self._cfg.servo_settle_hold_sec
        target = (math.radians(pan_deg), math.radians(tilt_deg))

        t0 = time.monotonic()
        ok_since: Optional[float] = None
        last_st = None
        no_state_logged = False
        while time.monotonic() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            st = self._get_pt_state()
            if st is None:
                if not no_state_logged and time.monotonic() - t0 > 1.0:
                    self.get_logger().warn(
                        "settle wait: no PanTiltState received yet -- is "
                        "pan_tilt_state_publisher running and on the right topic?"
                    )
                    no_state_logged = True
                ok_since = None
                continue
            last_st = st
            if not st.feedback_ok:
                ok_since = None
                continue
            if (abs(st.pan_rad - target[0]) < tol
                    and abs(st.tilt_rad - target[1]) < tol):
                ok_since = ok_since or time.monotonic()
                if time.monotonic() - ok_since >= hold:
                    return True
            else:
                ok_since = None

        # Timeout. Dump diagnostic so the operator can see what the gate was
        # rejecting -- without this, "Failed to park" is opaque.
        if last_st is None:
            self.get_logger().error(
                f"settle wait TIMEOUT after {timeout_sec:.1f}s with NO state msgs received"
            )
        else:
            self.get_logger().error(
                f"settle wait TIMEOUT after {timeout_sec:.1f}s: "
                f"target=(pan={pan_deg:+.2f},tilt={tilt_deg:+.2f}) deg, "
                f"observed=(pan={math.degrees(last_st.pan_rad):+.2f},"
                f"tilt={math.degrees(last_st.tilt_rad):+.2f}) deg, "
                f"feedback_ok={last_st.feedback_ok}, "
                f"|err|=(pan={math.degrees(abs(last_st.pan_rad-target[0])):.2f},"
                f"tilt={math.degrees(abs(last_st.tilt_rad-target[1])):.2f}) "
                f"vs tol={self._cfg.servo_settle_tol_deg:.2f} deg"
            )
        return False

    @staticmethod
    def _signed_overshoot(target: float, mag: float, lo: float, hi: float) -> float:
        """Pick a backlash-collapse offset that keeps target+offset within [lo, hi].

        Prefers the positive direction (consistent with the historical
        "overshoot up then return down" approach); flips to negative if that
        would exceed `hi`. Returns 0 if neither direction has room (caller
        should skip the overshoot pass entirely).
        """
        if target + mag <= hi:
            return mag
        if target - mag >= lo:
            return -mag
        return 0.0

    def _send_pt_with_backlash(self, pan_deg: float, tilt_deg: float) -> bool:
        """Move to (pan, tilt). Optionally collapses backlash via an
        overshoot-then-return pass when ``servo_backlash_overshoot_deg > 0``;
        defaults to a direct goal with no overshoot.
        """
        mag = self._cfg.servo_backlash_overshoot_deg
        if mag <= 0.0:
            self._send_pt_goal(pan_deg, tilt_deg)
            return self._wait_pt_settle(pan_deg, tilt_deg, timeout_sec=10.0)

        pan_off = self._signed_overshoot(
            pan_deg, mag,
            self._cfg.pan_overshoot_min_deg, self._cfg.pan_overshoot_max_deg,
        )
        tilt_off = self._signed_overshoot(
            tilt_deg, mag,
            self._cfg.tilt_overshoot_min_deg, self._cfg.tilt_overshoot_max_deg,
        )
        if pan_off == 0.0 and tilt_off == 0.0:
            self._send_pt_goal(pan_deg, tilt_deg)
            return self._wait_pt_settle(pan_deg, tilt_deg, timeout_sec=10.0)

        pan_int = pan_deg + pan_off
        tilt_int = tilt_deg + tilt_off
        self._send_pt_goal(pan_int, tilt_int)
        if not self._wait_pt_settle(pan_int, tilt_int, timeout_sec=10.0):
            self.get_logger().warn(
                f"pan-tilt didn't reach overshoot target "
                f"(pan={pan_int:+.1f}, tilt={tilt_int:+.1f})"
            )
            return False
        time.sleep(self._cfg.servo_backlash_pause_sec)

        self._send_pt_goal(pan_deg, tilt_deg)
        return self._wait_pt_settle(pan_deg, tilt_deg, timeout_sec=10.0)

    def _send_xarm_joint(self, angles_rad) -> bool:
        """Send a JointMove action goal. Pads to 7 joints for xarm7."""
        if not self._joint_move_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"JointMove action '{self._cfg.joint_move_action}' not available"
            )
            return False

        goal = self._joint_move_type.Goal()
        a = list(angles_rad) + [0.0] * max(0, 7 - len(angles_rad))
        goal.joint0 = float(a[0])
        goal.joint1 = float(a[1])
        goal.joint2 = float(a[2])
        goal.joint3 = float(a[3])
        goal.joint4 = float(a[4])
        goal.joint5 = float(a[5])
        goal.joint6 = float(a[6])
        # JointMove's env_points field was replaced by add_octomap (bool).
        # Stay False: during calibration the planner should not pull a fresh
        # octomap from sensors -- the EE / marker board / fixture would all
        # become obstacles and reject otherwise-feasible waypoints.
        goal.add_octomap = False

        timeout = float(self._cfg.arm_action_timeout_sec)
        send_fut = self._joint_move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=5.0)
        if not send_fut.done():
            self.get_logger().error("JointMove send_goal timed out")
            return False
        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().error("JointMove goal rejected")
            return False

        result_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=timeout)
        if not result_fut.done():
            self.get_logger().error(f"JointMove result timed out after {timeout:.0f}s")
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
            return False

        result = result_fut.result().result
        if not getattr(result, "success", False):
            self.get_logger().error("JointMove reported success=False")
            return False
        # Don't trust the action result -- some xArm drivers return success
        # when the trajectory has been DISPATCHED, not when the joints have
        # actually arrived. Gate on /joint_states.
        n_joints = min(len(angles_rad), 7)
        target = [float(angles_rad[i]) for i in range(n_joints)]
        if self._get_joint_state() is not None:
            if not self._wait_joint_convergence(target):
                return False
        else:
            # No /joint_states publisher -- fall back to the legacy fixed
            # sleep so we don't fail closed in environments without it.
            self.get_logger().warn(
                f"no {self._cfg.joint_state_topic} available; "
                f"falling back to fixed {self._cfg.xarm_settle_sec:.1f}s wait"
            )
            time.sleep(self._cfg.xarm_settle_sec)
        return True

    # ---- sample capture -----------------------------------------------------

    def _lookup_base_ee(self, log_label: str) -> Optional[np.ndarray]:
        try:
            tf_msg: TransformStamped = self._tf_buffer.lookup_transform(
                self._cfg.base_frame,
                self._cfg.ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except Exception as exc:
            self.get_logger().warn(f"[{log_label}] tf lookup failed: {exc}")
            return None
        return pose_to_matrix(
            [tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z],
            [tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w],
        )

    def _capture_cell(self, log_label: str, *, require_ee: bool = True) -> Optional[dict]:
        """Capture N frames at the current pan-tilt state and assemble a sample dict.

        Order of operations:
          1. Confirm pt feedback is healthy.
          2. Image-stability gate: wait for the marker geometry in the image
             to stop moving (replaces the old fixed pre_capture_quiet_sec).
             This is what catches mechanical ring-down + post-settle exposure
             drift; it's the load-bearing gate against duplicate-EE samples.
          3. (require_ee=True only) Lookup base→ee TF for the canonical
             T_base_ee; phase-1/phase-2/sanity all need this because the
             marker is on the EE.
          4. Capture frames; cluster_consensus picks the dominant pose.
          5. (require_ee=True only) Re-lookup base→ee; reject if the EE
             drifted during capture.

        Phase 4 uses require_ee=False — the marker is fixed in base_link
        (e.g. mounted on a tripod or fixture), so there's no EE pose to
        record and no EE drift to guard against. The xArm need not even be
        running. Output sample has no `t_base_ee` field in that mode.
        """
        pt_state = self._get_pt_state()
        if pt_state is None or not pt_state.feedback_ok:
            self.get_logger().warn(f"[{log_label}] no pt state, skipping")
            return None

        if not self._wait_image_stable():
            self.get_logger().warn(
                f"[{log_label}] image-stability gate did not converge; "
                "skipping cell"
            )
            return None

        if require_ee:
            T_base_ee = self._lookup_base_ee(log_label)
            if T_base_ee is None:
                return None
        else:
            T_base_ee = None

        detections = []
        per_image_stamps_ns: list[tuple[int, int]] = []
        last_seen_stamp = 0
        miss_streak = 0
        # Tally rejection reasons across the cell so the warn message on
        # skip can tell the operator WHY (low corner count vs high reproj
        # vs no detection at all) rather than just "0 valid detections".
        reject_tally: dict[str, int] = {}
        deadline = time.monotonic() + self._cfg.capture_timeout_sec
        while time.monotonic() < deadline and len(detections) < self._cfg.frames_per_cell:
            rclpy.spin_once(self, timeout_sec=0.05)
            img_tup = self._get_image()
            if img_tup is None:
                continue
            stamp_ns, img, K, D = img_tup
            if stamp_ns == last_seen_stamp:
                continue
            last_seen_stamp = stamp_ns
            det = detect_pose(img, K, D, board=self._board, detector=self._detector)
            if not det.valid():
                reason = det.reject_reason()
                reject_tally[reason] = reject_tally.get(reason, 0) + 1
                miss_streak += 1
                if (miss_streak > self._cfg.detection_lookahead_frames
                        and len(detections) >= self._cfg.min_valid_detections):
                    break
                continue
            miss_streak = 0
            cur_pt = self._get_pt_state()
            if cur_pt is None:
                continue
            cur_pt_ns = cur_pt.header.stamp.sec * 1_000_000_000 + cur_pt.header.stamp.nanosec
            detections.append(det)
            per_image_stamps_ns.append((stamp_ns, cur_pt_ns))

        if len(detections) < self._cfg.min_valid_detections:
            tally = ", ".join(f"{n}× '{r}'" for r, n in
                              sorted(reject_tally.items(), key=lambda kv: -kv[1]))
            self.get_logger().warn(
                f"[{log_label}] only {len(detections)} valid detections in "
                f"{self._cfg.capture_timeout_sec:.1f}s "
                f"(need >={self._cfg.min_valid_detections}); skipping cell. "
                f"Rejected: {tally or '(none)'}"
            )
            return None

        best_idx, best_skew_ns = min(
            enumerate(abs(i - p) for i, p in per_image_stamps_ns),
            key=lambda x: x[1],
        )
        skew_ms = best_skew_ns / 1e6
        if skew_ms > self._cfg.sample_stamp_skew_max_ms:
            self.get_logger().warn(
                f"[{log_label}] image-vs-state skew {skew_ms:.1f} ms exceeds "
                f"{self._cfg.sample_stamp_skew_max_ms} ms; skipping"
            )
            return None
        canonical_image_stamp_ns, canonical_pt_state_stamp_ns = per_image_stamps_ns[best_idx]

        consensus = cluster_consensus(detections)
        if consensus is None:
            self.get_logger().warn(
                f"[{log_label}] cluster_consensus could not reach quorum on "
                f"{len(detections)} frames -- per-cell pose unreliable; "
                "skipping cell"
            )
            return None

        # Post-capture TF consistency check (A3). If the EE drifted during
        # capture, the recorded T_base_ee no longer matches the marker pose
        # we'd see now -- discard the sample rather than emit a mismatched
        # pair. Skipped when require_ee=False (phase 4: no EE involved).
        if require_ee:
            T_base_ee_post = self._lookup_base_ee(log_label)
            if T_base_ee_post is None:
                return None
            drift_t, drift_r = pose_error_scalars(T_base_ee, T_base_ee_post)
            if (drift_t > self._cfg.capture_drift_max_trans_m
                    or drift_r > math.radians(self._cfg.capture_drift_max_rot_deg)):
                self.get_logger().warn(
                    f"[{log_label}] EE drifted during capture: "
                    f"{drift_t*1000:.2f} mm / {math.degrees(drift_r):.3f} deg "
                    f"(thresh {self._cfg.capture_drift_max_trans_m*1000:.1f} mm / "
                    f"{self._cfg.capture_drift_max_rot_deg:.2f} deg); skipping"
                )
                return None

        t_cam_marker_body = optical_to_body(consensus.pose_optical)

        sample = {
            "theta_pan_rad": float(pt_state.pan_rad),
            "theta_tilt_rad": float(pt_state.tilt_rad),
            "t_cam_marker_body": matrix_to_pose_dict(t_cam_marker_body),
            "image_stamp_ns": int(canonical_image_stamp_ns),
            "state_stamp_ns": int(canonical_pt_state_stamp_ns),
            "detection_quality": int(consensus.n_corners),
            "reprojection_rms_px": float(consensus.reprojection_rms_px),
            "label": log_label,
        }
        if T_base_ee is not None:
            sample["t_base_ee"] = matrix_to_pose_dict(T_base_ee)
        return sample

    def _is_duplicate_ee(self, T_new: np.ndarray, prior_samples: list[dict]) -> Optional[str]:
        """Return the label of any prior sample whose T_base_ee duplicates T_new.

        Tolerances come from `duplicate_ee_reject_*`. Distinct waypoints in
        the operator's yaml should never produce the same EE pose; a hit
        signals either a motion-completion race (the JointMove returned
        success without the arm having moved) or an accidental yaml
        duplication. Either way the new sample is invalid.
        """
        tol_t = float(self._cfg.duplicate_ee_reject_trans_m)
        tol_r = math.radians(float(self._cfg.duplicate_ee_reject_rot_deg))
        for prev in prior_samples:
            T_prev = pose_to_matrix(
                prev["t_base_ee"]["translation"], prev["t_base_ee"]["rotation"]
            )
            t, r = pose_error_scalars(T_new, T_prev)
            if t < tol_t and r < tol_r:
                return str(prev.get("label", "?"))
        return None

    # ---- phases ------------------------------------------------------------

    def _check_duplicate_consistency(self, samples: list, log_prefix: str) -> int:
        """Verify that any two samples with matching t_base_ee also have
        matching t_cam_marker_body. Operator-defined duplicate waypoints in
        the yaml serve as a self-consistency probe -- a large per-pair
        marker delta means per-frame PnP is unstable on this geometry and
        the downstream hand-eye solve will be garbage regardless of how
        many more samples are added.

        Returns the number of failing pairs. Logs a per-pair table and a
        summary verdict; does not abort (operator decides whether to
        re-collect).
        """
        ee_match_trans = self._cfg.duplicate_ee_match_trans_m
        ee_match_rot = math.radians(self._cfg.duplicate_ee_match_rot_deg)
        max_marker_trans = self._cfg.duplicate_marker_max_trans_m
        max_marker_rot = math.radians(self._cfg.duplicate_marker_max_rot_deg)

        # Pre-build SE(3) matrices once -- inner double loop hit each sample N times.
        ee_mats = [pose_to_matrix(s["t_base_ee"]["translation"], s["t_base_ee"]["rotation"])
                   for s in samples]
        cm_mats = [pose_to_matrix(s["t_cam_marker_body"]["translation"], s["t_cam_marker_body"]["rotation"])
                   for s in samples]

        n_dup_pairs = 0
        n_fail = 0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                ee_trans, ee_rot = pose_error_scalars(ee_mats[i], ee_mats[j])
                if ee_trans > ee_match_trans or ee_rot > ee_match_rot:
                    continue
                n_dup_pairs += 1
                d_trans, d_rot = pose_error_scalars(cm_mats[i], cm_mats[j])
                pass_ = d_trans <= max_marker_trans and d_rot <= max_marker_rot
                msg = (f"{log_prefix} duplicate-EE pair ({samples[i]['label']}, "
                       f"{samples[j]['label']}): marker delta = "
                       f"{d_trans*1000:.1f} mm / {math.degrees(d_rot):.2f} deg")
                if pass_:
                    self.get_logger().info(f"{msg} [pass]")
                else:
                    n_fail += 1
                    self.get_logger().warn(f"{msg} [FAIL]")

        if n_dup_pairs == 0:
            self.get_logger().info(
                f"{log_prefix} duplicate-EE consistency check: no duplicates in waypoints"
            )
        elif n_fail == 0:
            self.get_logger().info(
                f"{log_prefix} duplicate-EE consistency check: {n_dup_pairs}/{n_dup_pairs} pairs PASS"
            )
        else:
            self.get_logger().error(
                f"{log_prefix} duplicate-EE consistency check: {n_fail}/{n_dup_pairs} pairs FAIL "
                f"-- per-frame PnP is unstable on this geometry; recollecting with the "
                f"same waypoints will not improve hand-eye RMSE. Consider: closer board, "
                f"bigger board, intrinsic recalibration, or rigid (non-tape) marker mount."
            )
        return n_fail

    def run_phase1(self, *, park_pan_deg: float = 0.0,
                    park_tilt_deg: float = 30.0,
                    waypoints: Optional[list] = None,
                    label_prefix: str = "phase1") -> list:
        """Park pan-tilt at the chosen (pan, tilt), then iterate xArm waypoints
        and capture one averaged sample per waypoint.

        Two operating modes today:
          - level (default):  pan=0, tilt=+30 (camera horizontal). Canonical
            hand-eye reference; not operator-mutable.
          - custom:           operator-chosen pan/tilt for a robot whose
            geometry needs a different head pose to keep the marker in view.
            Both modes solve the SAME hand-eye geometry independently;
            running both gives an inter-park self-consistency check on the
            recovered T_ee_marker (it must agree across parks).

        Per pan_tilt_model.py "+firmware = tilt up", so firmware +30 = level
        and firmware 0 = servo zero = looking 30° down.
        """
        if waypoints is None:
            waypoints = self._cfg.phase1_waypoints
        self.get_logger().info(
            f"=== Phase 1 ({label_prefix}): hand-eye at firmware "
            f"(pan={park_pan_deg:+.1f}, tilt={park_tilt_deg:+.1f}) ==="
        )
        samples = []

        if not self._send_pt_with_backlash(park_pan_deg, park_tilt_deg):
            self.get_logger().error(
                f"Failed to park pan-tilt at ({park_pan_deg:+.1f}, {park_tilt_deg:+.1f})"
            )
            return samples

        for i, angles in enumerate(waypoints):
            self.get_logger().info(f"{label_prefix} waypoint {i+1}/{len(waypoints)}")
            if not self._send_xarm_joint(angles):
                # Real motion failure (kinematic infeasibility, server reject,
                # joint convergence timeout from a far-off final pose).
                # Skip this waypoint and try the next -- one bad waypoint
                # shouldn't waste the rest of the run.
                self.get_logger().error(
                    f"{label_prefix} waypoint {i+1}/{len(waypoints)}: xArm move failed; skipping"
                )
                continue
            # Pre-move safety check via forward TF (now that move succeeded).
            pt_state = self._get_pt_state()
            if pt_state is None:
                continue
            sample = self._capture_cell(f"{label_prefix}/{i}")
            if sample is None:
                continue

            # Post-hoc safety check.
            T_base_ee = pose_to_matrix(
                sample["t_base_ee"]["translation"], sample["t_base_ee"]["rotation"]
            )
            reason = self._validate_ee_pose(T_base_ee)
            if reason:
                self.get_logger().error(f"{label_prefix} sample at {i} violates envelope: {reason}")
                continue
            dup_label = self._is_duplicate_ee(T_base_ee, samples)
            if dup_label is not None:
                self.get_logger().error(
                    f"{label_prefix}/{i} EE pose duplicates {dup_label} within "
                    f"{self._cfg.duplicate_ee_reject_trans_m*1000:.1f} mm / "
                    f"{self._cfg.duplicate_ee_reject_rot_deg:.2f} deg "
                    "-- likely motion-completion failure or yaml duplicate; "
                    "rejecting sample"
                )
                continue
            samples.append(sample)

        self.get_logger().info(f"{label_prefix} collected {len(samples)} samples")
        self._check_duplicate_consistency(samples, label_prefix)
        return samples

    def run_phase2(self) -> list:
        """xArm at each waypoint x full pan-tilt grid."""
        self.get_logger().info("=== Phase 2: pan-tilt chain fit ===")
        samples = []

        for wi, angles in enumerate(self._cfg.phase2_waypoints):
            self.get_logger().info(
                f"Phase2 xArm pose {wi+1}/{len(self._cfg.phase2_waypoints)}"
            )
            if not self._send_xarm_joint(angles):
                self.get_logger().error("xArm move failed; skipping this pose")
                continue

            # Honour a pruned grid (calib_web prune-apply) if present;
            # otherwise iterate the full pan x tilt cross-product.
            if self._cfg.phase2_grid_pairs:
                cell_iter = [
                    (float(p), float(t))
                    for p, t in self._cfg.phase2_grid_pairs
                ]
            else:
                cell_iter = [
                    (float(p), float(t))
                    for p in self._cfg.pan_grid_deg
                    for t in self._cfg.tilt_grid_deg
                ]
            for pan_deg, tilt_deg in cell_iter:
                self.get_logger().info(f"  cell pan={pan_deg:+.0f} tilt={tilt_deg:+.0f}")
                if not self._send_pt_with_backlash(pan_deg, tilt_deg):
                    continue
                # Wait for the pan-tilt mast to stop resonating before
                # capturing -- the servo settle is in tolerance, but
                # mechanical ring-down keeps producing motion blur for
                # ~1 s on the dual-bracket mount.
                time.sleep(self._cfg.phase2_pantilt_settle_sec)
                label = f"phase2/w{wi}/p{pan_deg:+.0f}t{tilt_deg:+.0f}"
                sample = self._capture_cell(label)
                if sample is None:
                    continue
                T_base_ee = pose_to_matrix(
                    sample["t_base_ee"]["translation"],
                    sample["t_base_ee"]["rotation"],
                )
                reason = self._validate_ee_pose(T_base_ee)
                if reason:
                    self.get_logger().error(f"Envelope violation: {reason}")
                    continue
                # Phase-2 keeps the same xArm waypoint across all pan/tilt
                # cells of one sweep, so duplicates are EXPECTED within a
                # waypoint group. Only flag duplicates against samples
                # captured at a DIFFERENT waypoint group.
                other_wp_samples = [
                    s for s in samples
                    if not s["label"].startswith(f"phase2/w{wi}/")
                ]
                dup_label = self._is_duplicate_ee(T_base_ee, other_wp_samples)
                if dup_label is not None:
                    self.get_logger().error(
                        f"{label} EE pose duplicates {dup_label} (different "
                        f"waypoint group) -- yaml waypoint collision; "
                        "rejecting sample"
                    )
                    continue
                samples.append(sample)

        self.get_logger().info(f"Phase 2 collected {len(samples)} samples")
        return samples

    def run_sanity(self) -> Optional[dict]:
        if not self._cfg.sanity_xarm_angles_rad:
            return None
        self.get_logger().info("=== Sanity pose ===")
        if not self._send_xarm_joint(self._cfg.sanity_xarm_angles_rad):
            return None
        # Park at physical level (firmware tilt=+30); same reason as Phase 1.
        if not self._send_pt_with_backlash(0.0, 30.0):
            return None
        return self._capture_cell("sanity")

    def run_phase4_validation(self) -> dict:
        """Phase 4: end-to-end calibration check, xArm-free.

        Operator places a ChArUco board anywhere stationary in `base_link`
        (tripod, taped to a wall, on a table) and visible across the
        configured pan-tilt sweep. This routine drives the pan-tilt to a
        deterministic 5-corner grid + N random samples within the sweep
        range and snapshots ChArUco at each view. The xArm is not commanded
        and `link_eef` is not read.

        Downstream `cmd_validate` composes T_base_marker through the FK
        chain and reports the spread across views.
        """
        self.get_logger().info("=== Phase 4: validation sweep (xArm-independent) ===")

        # Resolve sweep bounds. Explicit ranges win; otherwise fall back to
        # the convex hull of the phase-2 grids so phase 4 doesn't ask the
        # head to move beyond what was characterised.
        if (len(self._cfg.validation_pan_range_deg) == 2
                and len(self._cfg.validation_tilt_range_deg) == 2):
            pan_min, pan_max = sorted(map(float, self._cfg.validation_pan_range_deg))
            tilt_min, tilt_max = sorted(map(float, self._cfg.validation_tilt_range_deg))
        else:
            if not self._cfg.pan_grid_deg or not self._cfg.tilt_grid_deg:
                self.get_logger().error(
                    "Phase 4: no validation_*_range_deg and no pan_grid_deg/"
                    "tilt_grid_deg to fall back on; cannot pick sweep bounds."
                )
                return {"samples": [], "skipped": []}
            pan_min = float(min(self._cfg.pan_grid_deg))
            pan_max = float(max(self._cfg.pan_grid_deg))
            tilt_min = float(min(self._cfg.tilt_grid_deg))
            tilt_max = float(max(self._cfg.tilt_grid_deg))

        # Deterministic 5-point corner grid + uniform-random fill. Covering
        # the corners surfaces edge-of-sweep failures (where a wrong T_B
        # rotation or a wrong theta_t_offset accumulates the most), and the
        # random fill distinguishes systematic bias from a coincidental
        # corner-only fit.
        n_total = max(5, int(self._cfg.n_validation_samples))
        rng = np.random.default_rng(int(self._cfg.validation_seed))
        pan_mid = 0.5 * (pan_min + pan_max)
        tilt_mid = 0.5 * (tilt_min + tilt_max)
        corner_set = [
            (pan_min, tilt_min),
            (pan_max, tilt_min),
            (pan_min, tilt_max),
            (pan_max, tilt_max),
            (pan_mid, tilt_mid),
        ]
        n_random = n_total - len(corner_set)
        random_set = [
            (float(rng.uniform(pan_min, pan_max)),
             float(rng.uniform(tilt_min, tilt_max)))
            for _ in range(n_random)
        ]
        sweep = corner_set + random_set
        self.get_logger().info(
            f"Phase 4: pan ∈ [{pan_min:+.1f}, {pan_max:+.1f}], "
            f"tilt ∈ [{tilt_min:+.1f}, {tilt_max:+.1f}], "
            f"{len(corner_set)} corner + {len(random_set)} random "
            f"(seed={self._cfg.validation_seed})"
        )

        samples: list[dict] = []
        skipped: list[dict] = []
        for i, (pan_deg, tilt_deg) in enumerate(sweep):
            label = f"phase4/{i:02d}_p{pan_deg:+.1f}t{tilt_deg:+.1f}"
            self.get_logger().info(f"  cell {label}")
            if not self._send_pt_with_backlash(pan_deg, tilt_deg):
                skipped.append({"i": i, "pan_deg": pan_deg, "tilt_deg": tilt_deg,
                                "reason": "pt_move_failed"})
                continue
            time.sleep(self._cfg.phase2_pantilt_settle_sec)
            sample = self._capture_cell(label, require_ee=False)
            if sample is None:
                skipped.append({"i": i, "pan_deg": pan_deg, "tilt_deg": tilt_deg,
                                "reason": "capture_failed"})
                continue
            samples.append(sample)

        self.get_logger().info(
            f"Phase 4: kept {len(samples)}/{len(sweep)} samples "
            f"({len(skipped)} skipped)"
        )
        return {
            "samples": samples,
            "skipped": skipped,
            "rng_seed": int(self._cfg.validation_seed),
            "pan_range_deg": [pan_min, pan_max],
            "tilt_range_deg": [tilt_min, tilt_max],
        }

    # ---- driver ------------------------------------------------------------

    def run_dry(self) -> dict:
        """Preflight: send each waypoint via JointMove with no image capture
        and no pan-tilt motion. Reports per-waypoint success/fail + per-list
        totals so the operator can fix bad yaml entries without committing
        to a 30-min collect.
        """
        results: dict = {}
        lists = [("phase1_waypoints", self._cfg.phase1_waypoints)]
        for d in self._cfg.phase1_custom_datasets:
            lists.append((f"phase1_custom[{d['name']}]", d.get("waypoints", [])))
        lists.append(("phase2_waypoints", self._cfg.phase2_waypoints))
        for list_name, waypoints in lists:
            if not waypoints:
                continue
            self.get_logger().info(
                f"=== Dry-run: validating {list_name} ({len(waypoints)} waypoints) ==="
            )
            ok, fail = [], []
            for i, angles in enumerate(waypoints):
                self.get_logger().info(
                    f"{list_name}[{i+1}/{len(waypoints)}]: sending JointMove…"
                )
                if self._send_xarm_joint(angles):
                    ok.append(i)
                else:
                    fail.append(i)
                    self.get_logger().warn(f"  -> {list_name}[{i}] FAILED")
            results[list_name] = {"ok": ok, "fail": fail,
                                   "total": len(waypoints)}
            self.get_logger().info(
                f"=== {list_name}: {len(ok)}/{len(waypoints)} OK, "
                f"{len(fail)} fail (indices: {fail}) ==="
            )
        return results

    def _resolve_custom_dataset(self) -> Optional[dict]:
        """Pick the custom dataset to collect for phase==phase1_custom.

        Empty `custom_name` + exactly one dataset -> that dataset. Empty +
        several -> error (ambiguous). A given name must exist. Returns None on
        any error after logging, so the caller can abort cleanly.
        """
        datasets = self._cfg.phase1_custom_datasets
        if not datasets:
            self.get_logger().error(
                "phase1_custom requested but no phase1_custom_datasets are "
                "configured -- author one in the calib web Waypoints tab"
            )
            return None
        names = [d["name"] for d in datasets]
        if not self._custom_name:
            if len(datasets) == 1:
                return datasets[0]
            self.get_logger().error(
                f"phase1_custom is ambiguous: {len(datasets)} datasets "
                f"({names}); pass -p custom_name:=<one of them>"
            )
            return None
        try:
            want = sanitize_custom_name(self._custom_name)
        except ValueError as exc:
            self.get_logger().error(f"bad custom_name: {exc}")
            return None
        for d in datasets:
            if d["name"] == want:
                return d
        self.get_logger().error(
            f"unknown custom dataset {want!r}; configured: {names}"
        )
        return None

    def run(self):
        if not self._wait_for_streams(timeout_sec=15.0):
            self.get_logger().error("Timed out waiting for camera/pan-tilt streams")
            return 1

        if self._phase_select == "dry_run":
            results = self.run_dry()
            self._out_dir.mkdir(parents=True, exist_ok=True)
            (self._out_dir / "dry_run.json").write_text(json.dumps(results, indent=2))
            self.get_logger().info(f"Wrote dry-run report to {self._out_dir}/dry_run.json")
            return 0

        if self._phase_select == "phase4_validation":
            phase4 = self.run_phase4_validation()
            self._out_dir.mkdir(parents=True, exist_ok=True)
            (self._out_dir / "phase4_validation.json").write_text(
                json.dumps(phase4, indent=2)
            )
            self.get_logger().info(
                f"Wrote phase4_validation.json to {self._out_dir}"
            )
            return 0

        sanity_start = self.run_sanity()
        phase1 = (self.run_phase1(park_pan_deg=0.0,
                                   park_tilt_deg=self._cfg.level_tilt_deg,
                                   waypoints=self._cfg.phase1_waypoints,
                                   label_prefix="phase1")
                  if self._phase_select in ("both", "phase1") else [])
        phase1_custom: list = []
        custom_dataset: Optional[dict] = None
        if self._phase_select == "phase1_custom":
            custom_dataset = self._resolve_custom_dataset()
            if custom_dataset is None:
                return 1
            phase1_custom = self.run_phase1(
                park_pan_deg=float(custom_dataset["park_pan_deg"]),
                park_tilt_deg=float(custom_dataset["park_tilt_deg"]),
                waypoints=custom_dataset.get("waypoints", []),
                label_prefix=f"phase1_custom_{custom_dataset['name']}")
        phase2 = self.run_phase2() if self._phase_select in ("both", "phase2") else []
        sanity_end = self.run_sanity()

        self._out_dir.mkdir(parents=True, exist_ok=True)
        if phase1:
            (self._out_dir / "phase1_handeye.json").write_text(
                json.dumps({"samples": phase1}, indent=2)
            )
        if phase1_custom and custom_dataset is not None:
            collect_fname, _ = custom_dataset_filenames(custom_dataset["name"])
            (self._out_dir / collect_fname).write_text(
                json.dumps({"samples": phase1_custom,
                            "custom_name": custom_dataset["name"]}, indent=2)
            )
            self.get_logger().info(
                f"wrote {len(phase1_custom)} custom samples to {collect_fname}"
            )
        if phase2:
            (self._out_dir / "phase2_chain.json").write_text(
                json.dumps({"samples": phase2}, indent=2)
            )
        if sanity_start or sanity_end:
            (self._out_dir / "sanity.json").write_text(json.dumps({
                "start": sanity_start, "end": sanity_end
            }, indent=2))

        self.get_logger().info(f"Wrote samples to {self._out_dir}")
        return 0


def main():
    rclpy.init()
    node = CalibrateCollectNode()
    try:
        rc = node.run()
    except KeyboardInterrupt:
        rc = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    import sys
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
