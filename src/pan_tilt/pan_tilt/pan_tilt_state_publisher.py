"""Joint-state publisher for the pan-tilt assembly."""

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

        self._state_topic = self.get_parameter('state_topic').value
        self._joint_state_topic = self.get_parameter('joint_state_topic').value
        self._pan_joint_name = self.get_parameter('pan_joint_name').value
        self._tilt_joint_name = self.get_parameter('tilt_joint_name').value
        self._stale_timeout_sec = float(self.get_parameter('stale_timeout_sec').value)

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
        joint_state.position = [msg.pan_rad, msg.tilt_rad]
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
