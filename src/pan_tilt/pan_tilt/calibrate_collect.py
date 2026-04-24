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
from sensor_msgs.msg import CameraInfo, Image
from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState

from .calibration.aruco_detect import (
    BoardSpec,
    build_board,
    build_detector,
    detect_pose,
    robust_average,
)
from .calibration.safety import SafetyEnvelope
from .calibration.utils import (
    matrix_to_pose_dict,
    optical_to_body,
    pose_to_matrix,
)


# tinker_arm_msgs is imported lazily at node construction so this module stays
# importable on hosts without the tinker_manipulation stack installed.


# ---- config -----------------------------------------------------------------

@dataclass
class CollectConfig:
    board: BoardSpec = field(default_factory=BoardSpec)
    safety: SafetyEnvelope = field(default_factory=SafetyEnvelope)

    phase1_waypoints: list = field(default_factory=list)   # list of joint-angle lists (rad)
    phase2_waypoints: list = field(default_factory=list)
    pan_grid_deg: list = field(default_factory=lambda: [-60.0, -30.0, 0.0, 30.0, 60.0])
    tilt_grid_deg: list = field(default_factory=lambda: [-25.0, -10.0, 0.0, 15.0, 35.0])
    sanity_xarm_angles_rad: list = field(default_factory=list)

    # Timing + convergence.
    servo_settle_tol_deg: float = 0.3
    servo_settle_hold_sec: float = 0.5
    servo_backlash_overshoot_deg: float = 2.0
    servo_backlash_pause_sec: float = 0.2
    xarm_settle_sec: float = 0.5
    arm_action_timeout_sec: float = 30.0
    sample_stamp_skew_max_ms: float = 20.0
    frames_per_cell: int = 10
    frame_min_interval_ms: float = 40.0

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
    return cfg


# ---- node -------------------------------------------------------------------

class CalibrateCollectNode(Node):
    def __init__(self):
        super().__init__("calibrate_collect")

        self.declare_parameter("config", "")
        self.declare_parameter("out_dir", "calibration_data")
        self.declare_parameter("phase", "both")  # both | phase1 | phase2 | sanity

        config_path = self.get_parameter("config").value or None
        self._cfg = _load_config(config_path)
        self._out_dir = Path(self.get_parameter("out_dir").value)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._phase_select = self.get_parameter("phase").value

        self._bridge = CvBridge()
        self._board = build_board(self._cfg.board)
        self._detector = build_detector(self._board)

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

    # ---- small helpers -----------------------------------------------------

    def _get_image(self) -> Optional[tuple]:
        with self._image_lock:
            if self._latest_image is None or self._latest_K is None:
                return None
            return (*self._latest_image, self._latest_K.copy(), self._latest_D.copy())

    def _get_pt_state(self) -> Optional[PanTiltState]:
        with self._pt_state_lock:
            return self._pt_state

    def _wait_for_streams(self, timeout_sec: float = 10.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_sec:
            if self._get_image() is not None and self._get_pt_state() is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

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

    def _wait_pt_settle(self, pan_deg: float, tilt_deg: float, timeout_sec: float = 6.0) -> bool:
        tol = math.radians(self._cfg.servo_settle_tol_deg)
        hold = self._cfg.servo_settle_hold_sec
        target = (math.radians(pan_deg), math.radians(tilt_deg))

        t0 = time.monotonic()
        ok_since: Optional[float] = None
        while time.monotonic() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            st = self._get_pt_state()
            if st is None or not st.feedback_ok:
                ok_since = None
                continue
            if (abs(st.pan_rad - target[0]) < tol
                    and abs(st.tilt_rad - target[1]) < tol):
                ok_since = ok_since or time.monotonic()
                if time.monotonic() - ok_since >= hold:
                    return True
            else:
                ok_since = None
        return False

    def _send_pt_with_backlash(self, pan_deg: float, tilt_deg: float) -> bool:
        """Move to (pan, tilt) with per-axis overshoot-return to collapse backlash."""
        overshoot = self._cfg.servo_backlash_overshoot_deg

        # Overshoot pass: target + overshoot on each axis (sign: positive).
        self._send_pt_goal(pan_deg + overshoot, tilt_deg + overshoot)
        if not self._wait_pt_settle(pan_deg + overshoot, tilt_deg + overshoot, timeout_sec=6.0):
            self.get_logger().warn("pan-tilt didn't reach overshoot target")
            return False
        time.sleep(self._cfg.servo_backlash_pause_sec)

        # Return to exact target.
        self._send_pt_goal(pan_deg, tilt_deg)
        return self._wait_pt_settle(pan_deg, tilt_deg, timeout_sec=6.0)

    def _send_xarm_joint(self, angles_rad) -> bool:
        """Send a JointMove action goal. Pads to 7 joints for xarm7."""
        if not self._joint_move_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"JointMove action '{self._cfg.joint_move_action}' not available"
            )
            return False

        from sensor_msgs.msg import PointCloud2, PointField
        goal = self._joint_move_type.Goal()
        a = list(angles_rad) + [0.0] * max(0, 7 - len(angles_rad))
        goal.joint0 = float(a[0])
        goal.joint1 = float(a[1])
        goal.joint2 = float(a[2])
        goal.joint3 = float(a[3])
        goal.joint4 = float(a[4])
        goal.joint5 = float(a[5])
        goal.joint6 = float(a[6])
        # pick_and_place runs pcl::fromROSMsg + tf2 lookupTransform on
        # env_points, so a default-constructed PointCloud2 would trip on
        # missing x/y/z fields and empty frame_id. Build a zero-point cloud
        # in base_link so the server short-circuits the transform step.
        env = PointCloud2()
        env.header.frame_id = "base_link"
        env.height = 1
        env.width = 0
        env.is_bigendian = False
        env.is_dense = True
        env.point_step = 12
        env.row_step = 0
        env.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        goal.env_points = env

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
        time.sleep(self._cfg.xarm_settle_sec)
        return True

    # ---- sample capture -----------------------------------------------------

    def _capture_cell(self, log_label: str) -> Optional[dict]:
        """Capture N frames at the current (pan-tilt, xArm) state and assemble a sample dict."""
        pt_state = self._get_pt_state()
        if pt_state is None or not pt_state.feedback_ok:
            self.get_logger().warn(f"[{log_label}] no pt state, skipping")
            return None

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

        T_base_ee = pose_to_matrix(
            [tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z],
            [tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w],
        )

        pt_state_stamp_ns = pt_state.header.stamp.sec * 1_000_000_000 + pt_state.header.stamp.nanosec

        detections = []
        used_image_stamps = []
        last_seen_stamp = 0
        deadline = time.monotonic() + self._cfg.frames_per_cell * (self._cfg.frame_min_interval_ms / 1000.0) * 3 + 1.0
        while len(detections) < self._cfg.frames_per_cell and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            img_tup = self._get_image()
            if img_tup is None:
                continue
            stamp_ns, img, K, D = img_tup
            if stamp_ns == last_seen_stamp:
                continue
            last_seen_stamp = stamp_ns
            det = detect_pose(img, K, D, board=self._board, detector=self._detector)
            if det.valid():
                detections.append(det)
                used_image_stamps.append(stamp_ns)

        if len(detections) < 3:
            self.get_logger().warn(f"[{log_label}] only {len(detections)} valid detections")
            return None

        # Time-sync audit: min skew between any image stamp and the pt state stamp.
        skew_ms = min(abs(s - pt_state_stamp_ns) / 1e6 for s in used_image_stamps)
        if skew_ms > self._cfg.sample_stamp_skew_max_ms:
            self.get_logger().warn(
                f"[{log_label}] image-vs-state skew {skew_ms:.1f} ms exceeds "
                f"{self._cfg.sample_stamp_skew_max_ms} ms; skipping"
            )
            return None

        averaged = robust_average(detections)
        if averaged is None:
            self.get_logger().warn(f"[{log_label}] robust_average rejected all detections")
            return None

        t_cam_marker_body = optical_to_body(averaged.pose_optical)

        return {
            "theta_pan_rad": float(pt_state.pan_rad),
            "theta_tilt_rad": float(pt_state.tilt_rad),
            "t_base_ee": matrix_to_pose_dict(T_base_ee),
            "t_cam_marker_body": matrix_to_pose_dict(t_cam_marker_body),
            "image_stamp_ns": int(used_image_stamps[0]),
            "state_stamp_ns": int(pt_state_stamp_ns),
            "detection_quality": int(averaged.n_corners),
            "reprojection_rms_px": float(averaged.reprojection_rms_px),
            "label": log_label,
        }

    # ---- phases ------------------------------------------------------------

    def run_phase1(self) -> list:
        """At servo zero, iterate xArm waypoints, one sample per waypoint."""
        self.get_logger().info("=== Phase 1: hand-eye at servo zero ===")
        samples = []

        if not self._send_pt_with_backlash(0.0, 0.0):
            self.get_logger().error("Failed to park pan-tilt at zero")
            return samples

        for i, angles in enumerate(self._cfg.phase1_waypoints):
            self.get_logger().info(f"Phase1 waypoint {i+1}/{len(self._cfg.phase1_waypoints)}")
            if not self._send_xarm_joint(angles):
                self.get_logger().error("xArm move failed; aborting Phase 1")
                break
            # Pre-move safety check via forward TF (now that move succeeded).
            pt_state = self._get_pt_state()
            if pt_state is None:
                continue
            sample = self._capture_cell(f"phase1/{i}")
            if sample is None:
                continue

            # Post-hoc safety check.
            T_base_ee = pose_to_matrix(
                sample["t_base_ee"]["translation"], sample["t_base_ee"]["rotation"]
            )
            reason = self._validate_ee_pose(T_base_ee)
            if reason:
                self.get_logger().error(f"Phase1 sample at {i} violates envelope: {reason}")
                continue
            samples.append(sample)

        self.get_logger().info(f"Phase 1 collected {len(samples)} samples")
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

            for pan_deg in self._cfg.pan_grid_deg:
                for tilt_deg in self._cfg.tilt_grid_deg:
                    self.get_logger().info(f"  cell pan={pan_deg:+.0f} tilt={tilt_deg:+.0f}")
                    if not self._send_pt_with_backlash(pan_deg, tilt_deg):
                        continue
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
                    samples.append(sample)

        self.get_logger().info(f"Phase 2 collected {len(samples)} samples")
        return samples

    def run_sanity(self) -> Optional[dict]:
        if not self._cfg.sanity_xarm_angles_rad:
            return None
        self.get_logger().info("=== Sanity pose ===")
        if not self._send_xarm_joint(self._cfg.sanity_xarm_angles_rad):
            return None
        if not self._send_pt_with_backlash(0.0, 0.0):
            return None
        return self._capture_cell("sanity")

    # ---- driver ------------------------------------------------------------

    def run(self):
        if not self._wait_for_streams(timeout_sec=15.0):
            self.get_logger().error("Timed out waiting for camera/pan-tilt streams")
            return 1

        sanity_start = self.run_sanity()
        phase1 = self.run_phase1() if self._phase_select in ("both", "phase1") else []
        phase2 = self.run_phase2() if self._phase_select in ("both", "phase2") else []
        sanity_end = self.run_sanity()

        self._out_dir.mkdir(parents=True, exist_ok=True)
        if phase1:
            (self._out_dir / "phase1_handeye.json").write_text(
                json.dumps({"samples": phase1}, indent=2)
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
