"""Joint-state publisher for the pan-tilt assembly."""

import math
import time
from typing import Optional

from .calibration.utils import wrap_to_pi

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from tinker_vision_msgs_26.msg import PanTiltState
except ImportError:  # pragma: no cover — only absent outside a sourced ROS env
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    JointState = None  # type: ignore[assignment]
    PanTiltState = None  # type: ignore[assignment]


def _load_profile():
    """Load the ROBOT_NAME-keyed profile, or None when unavailable.

    Isolated for test monkeypatching. Any failure (package absent, ROBOT_NAME
    unset, unknown robot) degrades to None — the caller falls back to the
    package-yaml params so dev machines keep working.
    """
    try:
        from tinker_robot_config import resolver
    except ImportError:
        return None
    try:
        return resolver.load()
    except Exception:
        return None


def _load_per_robot_offsets(logger):
    cfg = _load_profile()
    if cfg is None:
        logger.warning(
            'tinker_robot_config profile unavailable (ROBOT_NAME unset or '
            'package missing) — falling back to package-yaml pan/tilt offsets. '
            'These are NOT per-robot; calibrated robots must set '
            'robots/<robot>/pan_tilt/offsets.yaml.')
        return None
    pan = cfg.get('pan_tilt.offsets.pan_offset_rad')
    tilt = cfg.get('pan_tilt.offsets.tilt_offset_rad')
    if pan is None or tilt is None:
        logger.warning(
            'profile has no pan_tilt.offsets.{pan,tilt}_offset_rad — '
            'falling back to package-yaml offsets (NOT per-robot).')
        return None
    return float(pan), float(tilt)


def _resolve_offset(raw_rad: float, name: str, logger=None) -> float:
    """Wrap a calibration joint offset to (-pi, pi]; warn if it arrived
    out of range. An out-of-range offset (|x| > pi) means whoever wrote the
    config skipped normalization, and is a smell that the offset and the URDF
    T_b may be from different solves (the 2026-06-30 camera-tilt bug)."""
    wrapped = wrap_to_pi(raw_rad)
    if logger is not None and abs(float(raw_rad)) > math.pi + 1e-6:
        logger.warning(
            f"{name}={float(raw_rad):+.4f} rad ({math.degrees(float(raw_rad)):+.1f} deg) "
            f"was out of [-pi, pi]; wrapped to {wrapped:+.4f} rad. Verify the URDF "
            f"camera_mount T_b came from the SAME solve as this offset."
        )
    return wrapped


class PanTiltStatePublisherNode(Node):
    def __init__(self):
        super().__init__('pan_tilt_state_publisher')

        self.declare_parameter('state_topic', '/pan_tilt_controller/state')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('pan_joint_name', 'pan_joint')
        self.declare_parameter('tilt_joint_name', 'tilt_joint')
        self.declare_parameter('stale_timeout_sec', 0.5)
        # Calibration-derived offsets (rad) added to firmware feedback before
        # publishing to /joint_states. The URDF chain has no notion of the
        # firmware-zero-vs-URDF-zero offset; without these the TF chain
        # mis-represents the camera's pose by up to ~45° (the parked-tilt
        # angle), which is what produced "points project below ground" in
        # the seat-recommend pipeline. Source values from polish.json's
        # theta_p_offset_rad / theta_t_offset_rad after each calibration.
        self.declare_parameter('pan_offset_rad', 0.0)
        self.declare_parameter('tilt_offset_rad', 0.0)

        self._state_topic = self.get_parameter('state_topic').value
        self._joint_state_topic = self.get_parameter('joint_state_topic').value
        self._pan_joint_name = self.get_parameter('pan_joint_name').value
        self._tilt_joint_name = self.get_parameter('tilt_joint_name').value
        self._stale_timeout_sec = float(self.get_parameter('stale_timeout_sec').value)
        per_robot = _load_per_robot_offsets(self.get_logger())
        if per_robot is not None:
            raw_pan, raw_tilt = per_robot
            self.get_logger().info('pan/tilt offsets: per-robot profile (ROBOT_NAME)')
        else:
            raw_pan = float(self.get_parameter('pan_offset_rad').value)
            raw_tilt = float(self.get_parameter('tilt_offset_rad').value)
        self._pan_offset_rad = _resolve_offset(raw_pan, 'pan_offset_rad', self.get_logger())
        self._tilt_offset_rad = _resolve_offset(raw_tilt, 'tilt_offset_rad', self.get_logger())
        self.get_logger().info(
            f"Calibration offsets: pan={self._pan_offset_rad:+.4f} rad "
            f"({math.degrees(self._pan_offset_rad):+.2f}°), "
            f"tilt={self._tilt_offset_rad:+.4f} rad "
            f"({math.degrees(self._tilt_offset_rad):+.2f}°). "
            f"Source these from polish.json's theta_p_offset_rad / "
            f"theta_t_offset_rad after each calibration; both zero means "
            f"the URDF will mis-represent the camera pose at any non-zero "
            f"firmware tilt."
        )

        self._joint_state_pub = self.create_publisher(
            JointState,
            self._joint_state_topic,
            10,
        )
        self.create_subscription(
            PanTiltState,
            self._state_topic,
            self._handle_state,
            10,
        )

        self._last_feedback_time: Optional[float] = None
        self._warned_stale = False
        self.create_timer(0.5, self._check_staleness)

    def _handle_state(self, msg: PanTiltState):
        joint_state = JointState()
        joint_state.header = msg.header
        joint_state.name = [self._pan_joint_name, self._tilt_joint_name]
        # Conventions match the calibration FK in pan_tilt_model:
        #   forward_kinematics does R_z(-(pan + p_off)) @ ... @ R_y(tilt + t_off)
        #   URDF pan_joint axis="0 0 -1" → R_axis(joint) = R_z(-joint)
        #   URDF tilt_joint axis="0 1 0" → R_axis(joint) = R_y(+joint)
        # so the URDF computes the right TF iff joint_value = firmware + offset.
        joint_state.position = [
            msg.pan_rad + self._pan_offset_rad,
            msg.tilt_rad + self._tilt_offset_rad,
        ]
        self._joint_state_pub.publish(joint_state)

        if msg.feedback_ok:
            self._last_feedback_time = time.monotonic()
            self._warned_stale = False

    def _check_staleness(self):
        if self._last_feedback_time is None:
            return
        if (time.monotonic() - self._last_feedback_time) <= self._stale_timeout_sec:
            return
        if self._warned_stale:
            return
        self._warned_stale = True
        self.get_logger().warn(
            'Pan-tilt feedback is stale; joint states are no longer fresh.',
        )


def main():
    rclpy.init()
    node = PanTiltStatePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
