"""Joint-state publisher for the pan-tilt assembly."""

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tinker_vision_msgs_26.msg import PanTiltState


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
        self._pan_offset_rad = float(self.get_parameter('pan_offset_rad').value)
        self._tilt_offset_rad = float(self.get_parameter('tilt_offset_rad').value)
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
