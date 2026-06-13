# Copyright 2026 Tinker Team
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Sim stand-in for person_track_server: a topic->action adapter.

Subscribes dummy_person's /target_points (NaN sentinel = lost) + reacq_state and
serves the /track_person action, translating that one topic stream into
TrackPerson feedback. dummy_person's topics stay the single source of truth, so
the feedback the BT sees can never diverge from what follow_server consumes.
The BT does not route person->nav through its blackboard, so only liveness +
reacquisition_state must be faithful (see the F4 design spec)."""
import math
import threading
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import UInt8
from tinker_vision_msgs_26.action import TrackPerson


class TrackBuffer:
    """Pure (ROS-free) mirror of the latest tracker topic state."""

    def __init__(self, stale_timeout_sec=1.0):
        self._stale = float(stale_timeout_sec)
        self._pos = None          # (x, y, frame) of the last FINITE point
        self._lost = True
        self._last_point_t = None
        self._reacq = 0

    def on_point(self, x, y, frame, t):
        if math.isfinite(x) and math.isfinite(y):
            self._pos = (x, y, frame)
            self._lost = False
            self._last_point_t = t
        else:
            self._lost = True     # NaN sentinel: lost, but HOLD the last position

    def on_reacq(self, v):
        self._reacq = int(v)

    def lost(self, now):
        if self._lost or self._last_point_t is None:
            return True
        return (now - self._last_point_t) > self._stale

    def position(self):
        return self._pos

    def reacq(self):
        return self._reacq


class SimTrackServer(Node):
    def __init__(self):
        super().__init__("sim_track_server")
        self.declare_parameter("target_points_topic", "/target_points")
        self.declare_parameter("reacq_topic", "/person_track_node/reacq_state")
        self.declare_parameter("feedback_rate", 10.0)
        self.declare_parameter("stale_timeout_sec", 1.0)
        self._rate = float(self.get_parameter("feedback_rate").value)
        self._buf = TrackBuffer(self.get_parameter("stale_timeout_sec").value)
        self._lock = threading.Lock()
        self.create_subscription(
            PointStamped, self.get_parameter("target_points_topic").value,
            self._on_point, 10)
        self.create_subscription(
            UInt8, self.get_parameter("reacq_topic").value, self._on_reacq, 10)
        self._action = ActionServer(
            self, TrackPerson, "track_person",
            execute_callback=self._execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup())
        self.get_logger().info("sim_track_server ready (mirrors /target_points)")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_point(self, msg):
        with self._lock:
            self._buf.on_point(msg.point.x, msg.point.y,
                               msg.header.frame_id or "map", self._now())

    def _on_reacq(self, msg):
        with self._lock:
            self._buf.on_reacq(msg.data)

    def _execute(self, goal_handle):
        # Continuous action: stream feedback until cancelled (the BT cancels on
        # teardown). Never self-completes; BtNode_TrackPersonAction maps
        # CANCELED -> SUCCESS, abort -> FAILURE.
        period = 1.0 / self._rate if self._rate > 0 else 0.1
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = TrackPerson.Result()
                result.status = 0
                return result
            with self._lock:
                lost = self._buf.lost(self._now())
                pos = self._buf.position()
                reacq = self._buf.reacq()
            fb = TrackPerson.Feedback()
            fb.target_lost = bool(lost)
            fb.target_track_id = 1
            fb.is_transformation_successful = True
            fb.reacquisition_state = int(reacq)
            p = PointStamped()
            if pos is not None:
                p.point.x, p.point.y, p.header.frame_id = pos[0], pos[1], pos[2]
            fb.target_position = p
            goal_handle.publish_feedback(fb)
            time.sleep(period)
        result = TrackPerson.Result()
        result.status = 0
        return result


def main(args=None):
    rclpy.init(args=args)
    node = SimTrackServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
